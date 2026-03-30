//! Eager loading for ServerlessLLM format (owned buffers).
//!
//! Partition-native loading: one read per partition, shared backing buffers.

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::TensorMetadata;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::index::{Index, PartitionPlan};
use super::tensor::Tensor;

/// Load plan: one read per partition, tensors assembled from shared backing.
#[derive(Debug)]
struct LoadPlan {
    partitions: Vec<PartitionRead>,
    index: Arc<Index>,
}

#[derive(Debug)]
struct PartitionRead {
    partition_id: usize,
    path: PathBuf,
    size: u64,
}

impl LoadPlan {
    fn compile(index: &Index, base_path: &Path) -> Self {
        let base_path_str = base_path.to_string_lossy();

        let partitions: Vec<PartitionRead> = index
            .partition_ids()
            .iter()
            .filter_map(|partition_id| index.partition(*partition_id))
            .map(|plan: &PartitionPlan| PartitionRead {
                partition_id: plan.partition_id,
                path: PathBuf::from(format!("{}_{}", base_path_str, plan.partition_id)),
                size: plan.max_required_size,
            })
            .collect();

        LoadPlan {
            partitions,
            index: Arc::new(index.clone()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LoadStats {
    partition_count: usize,
    tensor_count: usize,
    total_bytes: u64,
    max_partition_bytes: u64,
}

const SYNC_BASE_COST_NS: f64 = 220_000.0;
const ASYNC_BASE_COST_NS: f64 = 700_000.0;
const SYNC_PER_PARTITION_COST_NS: f64 = 45_000.0;
const ASYNC_PER_PARTITION_COST_NS: f64 = 105_000.0;
const SYNC_PER_TENSOR_COST_NS: f64 = 2_000.0;
const ASYNC_PER_TENSOR_COST_NS: f64 = 3_500.0;
const ASYNC_OVERHEAD_PER_BYTE: f64 = 0.15;
const THROUGHPUT_BPS: f64 = 7.5 * 1024.0 * 1024.0 * 1024.0;
const PARALLELISM_TARGET_BYTES: f64 = 128.0 * 1024.0 * 1024.0;

fn bytes_to_ns(bytes: u64, throughput_bps: f64) -> f64 {
    (bytes as f64 / throughput_bps) * 1_000_000_000.0
}

fn load_stats(index: &Index) -> LoadStats {
    let mut total_bytes = 0u64;
    let mut max_partition_bytes = 0u64;

    for plan in index.partitions().values() {
        total_bytes = total_bytes.saturating_add(plan.max_required_size);
        max_partition_bytes = max_partition_bytes.max(plan.max_required_size);
    }

    LoadStats {
        partition_count: index.partition_ids().len(),
        tensor_count: index.len(),
        total_bytes,
        max_partition_bytes,
    }
}

fn effective_async_parallelism(stats: &LoadStats) -> f64 {
    if stats.partition_count <= 1 {
        return 1.0;
    }

    let avg_partition_bytes = stats.total_bytes.max(1) as f64 / stats.partition_count as f64;
    let max_partition_bytes = stats.max_partition_bytes.max(1) as f64;
    let count_factor = (stats.partition_count as f64).ln_1p();
    let size_factor = (PARALLELISM_TARGET_BYTES / avg_partition_bytes)
        .sqrt()
        .clamp(0.5, 4.0);
    let skew_factor = (max_partition_bytes / avg_partition_bytes)
        .sqrt()
        .clamp(1.0, 4.0);

    (1.0 + count_factor * size_factor / skew_factor).clamp(1.0, stats.partition_count as f64)
}

fn estimate_sync_cost(stats: &LoadStats) -> f64 {
    SYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS)
        + SYNC_PER_PARTITION_COST_NS * stats.partition_count as f64
        + SYNC_PER_TENSOR_COST_NS * stats.tensor_count as f64
}

fn estimate_async_cost(stats: &LoadStats) -> f64 {
    let parallelism = effective_async_parallelism(stats);

    ASYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS) / parallelism
        + ASYNC_PER_PARTITION_COST_NS * stats.partition_count as f64
        + ASYNC_PER_TENSOR_COST_NS * stats.tensor_count as f64
        + ASYNC_OVERHEAD_PER_BYTE * stats.max_partition_bytes as f64
}

fn choose_load_backend(stats: &LoadStats) -> bool {
    if stats.partition_count <= 1 {
        return false;
    }
    estimate_async_cost(stats) < estimate_sync_cost(stats)
}

type Tensors = HashMap<Arc<str>, Tensor>;

/// Execute load plan synchronously - partition-native with shared backing.
fn execute_load_plan_sync(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }

    let index = &plan.index;

    // Validate all partitions first
    for read in &plan.partitions {
        let metadata =
            std::fs::metadata(&read.path).map_err(|_| ReaderError::PartitionNotFound {
                partition_id: read.partition_id,
                path: read.path.to_string_lossy().to_string(),
            })?;
        if metadata.len() < read.size {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: metadata.len(),
                required: read.size,
            });
        }
    }

    // Fast path: single partition - use direct load, avoid batch overhead
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let mut reader = backends::SyncReader::new();
        let data = reader.load(&read.path).map_err(ReaderError::from)?;

        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }

        let backing: Arc<[u8]> = data.into();

        let mut tensors = HashMap::with_capacity(index.len());
        for name in index.tensor_names().iter() {
            let desc = index.get(name.as_ref()).unwrap();
            tensors.insert(
                name.clone(),
                Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)),
            );
        }
        return Ok(tensors);
    }

    // Multi-partition: parallel batch reads
    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();

    let mut reader = backends::SyncReader::new();
    let results = reader.load_range_batch(&requests).map_err(ReaderError::from)?;

    // Build partition buffers in indexed slots to avoid hashing hot-path lookups.
    let max_partition_id = plan
        .partitions
        .iter()
        .map(|read| read.partition_id)
        .max()
        .unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (read, result) in plan.partitions.iter().zip(results) {
        let (buf, _, _) = result;
        if buf.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: buf.len() as u64,
                required: read.size,
            });
        }
        partition_buffers[read.partition_id] = Some(buf);
    }

    // Assemble tensors from shared partition buffers
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
    }

    Ok(tensors)
}

/// Execute load plan asynchronously - partition-native with shared backing.
async fn execute_load_plan_async(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }

    let index = &plan.index;

    // Validate all partitions in parallel to avoid serial metadata round-trips.
    let validations: Vec<_> = plan
        .partitions
        .iter()
        .map(|read| async {
            let metadata = tokio::fs::metadata(&read.path).await.map_err(|_| {
                ReaderError::PartitionNotFound {
                    partition_id: read.partition_id,
                    path: read.path.to_string_lossy().to_string(),
                }
            })?;

            if metadata.len() < read.size {
                return Err(ReaderError::PartitionTooSmall {
                    path: read.path.to_string_lossy().to_string(),
                    actual: metadata.len(),
                    required: read.size,
                });
            }

            Ok(())
        })
        .collect();

    futures::future::try_join_all(validations).await?;

    // Fast path: single partition - use direct load, avoid batch overhead
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let mut reader = backends::AsyncReader::new();
        let data = reader.load(&read.path).await.map_err(ReaderError::from)?;

        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }

        let backing: Arc<[u8]> = data.into();

        let mut tensors = HashMap::with_capacity(index.len());
        for name in index.tensor_names().iter() {
            let desc = index.get(name.as_ref()).unwrap();
            tensors.insert(
                name.clone(),
                Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)),
            );
        }
        return Ok(tensors);
    }

    // Multi-partition: parallel reads using load() per partition.
    // Each load() internally chunks large files to saturate the async I/O queue.
    // load_range_batch() would only issue one concurrent read per partition file,
    // which underutilizes the device compared to load()'s backend-local internal chunking.
    let load_futures: Vec<_> = plan
        .partitions
        .iter()
        .map(|read| async move {
            let mut reader = backends::AsyncReader::new();
            let data = reader.load(&read.path).await.map_err(ReaderError::from)?;
            if data.len() < read.size as usize {
                return Err(ReaderError::PartitionTooSmall {
                    path: read.path.to_string_lossy().to_string(),
                    actual: data.len() as u64,
                    required: read.size,
                });
            }
            let owned: Arc<[u8]> = data.into();
            Ok((read.partition_id, owned))
        })
        .collect();

    let results: Vec<(usize, Arc<[u8]>)> = futures::future::join_all(load_futures)
        .await
        .into_iter()
        .collect::<ReaderResult<Vec<_>>>()?;
    let all_results = results;

    // Build partition buffers in indexed slots to avoid hashing hot-path lookups.
    let max_partition_id = plan
        .partitions
        .iter()
        .map(|read| read.partition_id)
        .max()
        .unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (partition_id, buf) in all_results {
        partition_buffers[partition_id] = Some(buf);
    }

    // Assemble tensors from shared partition buffers
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
    }

    Ok(tensors)
}

/// ServerlessLLM model with all tensors loaded into memory (eager loading).
#[derive(Debug, Clone)]
pub struct Model {
    tensors: HashMap<Arc<str>, Tensor>,
    tensor_names: Arc<[Arc<str>]>,
}

impl Model {
    fn compile_load_plan(directory: impl AsRef<Path>) -> ReaderResult<(Index, LoadPlan)> {
        let dir_path = directory.as_ref();
        let index = Index::load_sync(dir_path.join("tensor_index.json"))?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        Ok((index, plan))
    }

    /// Loads a ServerlessLLM model from directory using the core heuristic.
    pub async fn load(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        if choose_load_backend(&load_stats(&index)) {
            let tensors = execute_load_plan_async(&plan).await?;
            return Ok(Self {
                tensors,
                tensor_names: index.tensor_names().to_vec().into(),
            });
        }

        let tensors = execute_load_plan_sync(&plan)?;
        let tensor_names = index.tensor_names().to_vec().into();
        Ok(Self {
            tensors,
            tensor_names,
        })
    }

    /// Loads a ServerlessLLM model from directory asynchronously with eager loading.
    pub async fn load_async(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load(dir_path.join("tensor_index.json")).await?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        let tensors = execute_load_plan_async(&plan).await?;
        let tensor_names = index.tensor_names().to_vec().into();
        Ok(Self {
            tensors,
            tensor_names,
        })
    }

    /// Loads a ServerlessLLM model from directory synchronously with eager loading.
    pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        let tensors = execute_load_plan_sync(&plan)?;
        let tensor_names = index.tensor_names().to_vec().into();
        Ok(Self {
            tensors,
            tensor_names,
        })
    }

    /// Returns a reference to the tensor with the given name.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Returns tensor names (cached, sorted).
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }

    /// Returns the number of tensors in the loaded model.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true when no tensors are loaded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl<'a> IntoIterator for &'a Model {
    type Item = (&'a Arc<str>, &'a Tensor);
    type IntoIter = std::collections::hash_map::Iter<'a, Arc<str>, Tensor>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for Model {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> &[Arc<str>] {
        Model::tensor_names(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_len_and_is_empty() {
        let model = Model {
            tensors: HashMap::new(),
            tensor_names: Arc::new([]),
        };
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn choose_sync_for_single_large_partition() {
        let stats = LoadStats {
            partition_count: 1,
            tensor_count: 1,
            total_bytes: 2 * 1024 * 1024 * 1024,
            max_partition_bytes: 2 * 1024 * 1024 * 1024,
        };

        assert!(!choose_load_backend(&stats));
    }

    #[test]
    fn choose_async_for_many_small_partitions() {
        let stats = LoadStats {
            partition_count: 16,
            tensor_count: 1024,
            total_bytes: 256 * 1024 * 1024,
            max_partition_bytes: 32 * 1024 * 1024,
        };

        assert!(choose_load_backend(&stats));
    }

    #[test]
    fn choose_sync_for_low_fanout_partitions() {
        let stats = LoadStats {
            partition_count: 2,
            tensor_count: 160,
            total_bytes: 524 * 1024 * 1024,
            max_partition_bytes: 334 * 1024 * 1024,
        };

        assert!(!choose_load_backend(&stats));
    }
}
