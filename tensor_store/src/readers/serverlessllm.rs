//! `ServerlessLLM` format reader.
//!
//! This module provides functionality to parse `ServerlessLLM` tensor index files,
//! load tensor data from partition files, and validate partition integrity.
//!
//! # Format Structure
//!
//! ```text
//! tensor_index.json:
//! {
//!   "tensor_name": [offset, size, [shape...], [stride...], "dtype"],
//!   ...
//! }
//!
//! tensor.data_0: Binary tensor data (partition 0)
//! tensor.data_1: Binary tensor data (partition 1)
//! ...
//! ```
//!
//! # Usage Examples
//!
//! ## Parse Index and Load Tensors
//!
//! ```rust,ignore
//! use tensor_store::readers::serverlessllm;
//!
//! // Parse index file
//! let index = serverlessllm::parse_index("model/tensor_index.json").await?;
//! println!("Found {} tensors across {} partitions", index.len(), index.partition_count());
//!
//! // Load tensor data (validation is automatic)
//! let weight = index.load_tensor("model/tensor.data", "layer.0.weight").await?;
//! let bias = index.load_tensor("model/tensor.data", "layer.0.bias").await?;
//! ```
//!
//! ## Batch Validation (Optional)
//!
//! Validation happens automatically when loading tensors, but you can validate
//! all partitions upfront if desired:
//!
//! ```rust,ignore
//! // Validate all partition files before loading (optional)
//! index.validate_partitions("model/tensor.data")?;
//! index.validate_partition_sizes("model/tensor.data")?;
//!
//! // Now load tensors (will skip redundant validation)
//! for name in index.tensor_names() {
//!     let data = index.load_tensor("model/tensor.data", name).await?;
//! }
//! ```
//!
use crate::backends;
use crate::readers::error::{ReaderError, ReaderResult};
use crate::readers::traits::{AsyncReader, SyncReader, TensorMetadata};
use std::collections::HashMap;
use std::path::Path;

// Re-export shared TensorEntry type for backwards compatibility
pub use crate::types::serverlessllm::TensorEntry;

/// Parsed `ServerlessLLM` index
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ServerlessLLMIndex {
    /// All tensors in the index
    tensors: HashMap<String, TensorEntry>,
}

impl ServerlessLLMIndex {
    /// Creates a new empty index.
    #[inline]
    #[must_use] 
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the tensors map.
    #[inline]
    #[must_use] 
    pub const fn tensors(&self) -> &HashMap<String, TensorEntry> {
        &self.tensors
    }

    /// Gets a tensor entry by name.
    #[inline]
    #[must_use] 
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// Returns an iterator over tensor names and entries.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorEntry)> {
        self.tensors.iter()
    }

    /// Loads tensor data from a partition file asynchronously.
    ///
    /// Automatically validates that the partition file exists and has sufficient size
    /// before attempting to load the data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_name` - Name of the tensor to load
    ///
    /// # Returns
    ///
    /// The raw tensor data as bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor name is not found in the index
    /// - The partition file doesn't exist
    /// - The partition file is too small
    /// - I/O errors occur during reading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = ServerlessLLMIndex::load("model/tensor_index.json").await?;
    /// let data = index.load_tensor("model/tensor.data", "layer.0.weight").await?;
    /// ```
    #[inline]
    pub async fn load_tensor(
        &self,
        base_path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> ReaderResult<Vec<u8>> {
        let entry = self.tensors.get(tensor_name).ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor '{tensor_name}' not found in index"))
        })?;

        let partition_path = format!("{}_{}", base_path.as_ref().display(), entry.partition_id);

        // Validate partition file before loading
        Self::validate_single_partition(&partition_path, entry)?;

        backends::load_range(&partition_path, entry.offset, usize::try_from(entry.size).map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?)
            .await
            .map_err(ReaderError::from)
    }

    /// Loads tensor data from a partition file synchronously.
    ///
    /// Automatically validates that the partition file exists and has sufficient size
    /// before attempting to load the data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_name` - Name of the tensor to load
    ///
    /// # Returns
    ///
    /// The raw tensor data as bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor name is not found in the index
    /// - The partition file doesn't exist
    /// - The partition file is too small
    /// - I/O errors occur during reading
    #[inline]
    pub fn load_tensor_sync(
        &self,
        base_path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> ReaderResult<Vec<u8>> {
        let entry = self.tensors.get(tensor_name).ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor '{tensor_name}' not found in index"))
        })?;

        let partition_path = format!("{}_{}", base_path.as_ref().display(), entry.partition_id);

        // Validate partition file before loading
        Self::validate_single_partition(&partition_path, entry)?;

        backends::sync::load_range(&partition_path, entry.offset, usize::try_from(entry.size).map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?)
            .map_err(ReaderError::from)
    }

    /// Validates a single partition file for a specific tensor entry.
    fn validate_single_partition(
        partition_path: &str,
        entry: &TensorEntry,
    ) -> ReaderResult<()> {
        let metadata = std::fs::metadata(partition_path).map_err(|e| {
            ReaderError::ServerlessLlm(format!(
                "partition file '{partition_path}' not found or inaccessible: {e}"
            ))
        })?;

        let required_size = entry.offset.checked_add(entry.size).ok_or_else(|| ReaderError::ServerlessLlm("offset + size overflow".to_owned()))?;
        let actual_size = metadata.len();

        if actual_size < required_size {
            return Err(ReaderError::ServerlessLlm(format!(
                "partition file '{partition_path}' is too small: has {actual_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {}",
                entry.offset, entry.size
            )));
        }

        Ok(())
    }

    /// Validates that all partition files exist and are readable.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// Ok if all partition files exist, Err with details of missing files.
    #[inline]
    pub fn validate_partitions(&self, base_path: impl AsRef<Path>) -> ReaderResult<()> {
        use std::collections::HashSet;

        // Collect unique partition IDs
        let partition_ids: HashSet<usize> = self
            .tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect();

        let mut missing_partitions = Vec::new();

        for partition_id in partition_ids {
            let partition_path = format!("{}_{}", base_path.as_ref().display(), partition_id);
            if !std::path::Path::new(&partition_path).exists() {
                missing_partitions.push(partition_id);
            }
        }

        if !missing_partitions.is_empty() {
            return Err(ReaderError::ServerlessLlm(format!(
                "missing partition files: {missing_partitions:?}"
            )));
        }

        Ok(())
    }

    /// Validates partition file sizes match expected tensor data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// Ok if all partition files have sufficient size, Err with details.
    #[inline]
    pub fn validate_partition_sizes(&self, base_path: impl AsRef<Path>) -> ReaderResult<()> {
        use std::collections::HashMap;

        // Calculate expected size for each partition
        let mut partition_sizes: HashMap<usize, u64> = HashMap::new();
        for entry in self.tensors.values() {
            let max_offset = entry.offset + entry.size;
            partition_sizes
                .entry(entry.partition_id)
                .and_modify(|size| *size = (*size).max(max_offset))
                .or_insert(max_offset);
        }

        let mut errors = Vec::new();

        for (partition_id, expected_size) in partition_sizes {
            let partition_path = format!("{}_{}", base_path.as_ref().display(), partition_id);
            match std::fs::metadata(&partition_path) {
                Ok(metadata) => {
                    let actual_size = metadata.len();
                    if actual_size < expected_size {
                        errors.push(format!(
                            "partition {partition_id}: expected at least {expected_size} bytes, found {actual_size} bytes"
                        ));
                    }
                }
                Err(e) => {
                    errors.push(format!("partition {partition_id}: {e}"));
                }
            }
        }

        if !errors.is_empty() {
            let errors_str = errors.join("; ");
            return Err(ReaderError::ServerlessLlm(format!(
                "partition validation failed: {errors_str}"
            )));
        }

        Ok(())
    }

    /// Returns the total number of unique partition files.
    #[inline]
    #[must_use] 
    pub fn partition_count(&self) -> usize {
        use std::collections::HashSet;
        self.tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Returns the partition IDs used by this index.
    #[inline]
    #[must_use] 
    pub fn partition_ids(&self) -> Vec<usize> {
        use std::collections::HashSet;
        let mut ids: Vec<_> = self
            .tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort_unstable();
        ids
    }
}

impl<'a> IntoIterator for &'a ServerlessLLMIndex {
    type Item = (&'a String, &'a TensorEntry);
    type IntoIter = std::collections::hash_map::Iter<'a, String, TensorEntry>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for ServerlessLLMIndex {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(std::string::String::as_str).collect()
    }
}

impl AsyncReader for ServerlessLLMIndex {
    type Output = Self;

    #[inline]
    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let data = backends::load(path.as_ref().to_str().ok_or_else(|| ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned()))?).await?;
        parse_index_impl(&data)
    }
}

impl SyncReader for ServerlessLLMIndex {
    type Output = Self;

    #[inline]
    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let data = backends::sync::load(path.as_ref().to_str().ok_or_else(|| ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned()))?)?;
        parse_index_impl(&data)
    }
}

/// Parse `ServerlessLLM` tensor index file.
#[inline]
pub async fn parse_index(path: impl AsRef<Path>) -> ReaderResult<ServerlessLLMIndex> {
    ServerlessLLMIndex::load(path).await
}

/// Parse `ServerlessLLM` tensor index synchronously (mmap on Linux).
#[inline]
pub fn parse_index_sync(path: impl AsRef<Path>) -> ReaderResult<ServerlessLLMIndex> {
    ServerlessLLMIndex::load_sync(path)
}

/// Core parsing implementation shared by async and sync versions.
#[allow(clippy::get_first)]
fn parse_index_impl(data: &[u8]) -> ReaderResult<ServerlessLLMIndex> {
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(data)
        .map_err(|err| ReaderError::ServerlessLlm(format!("JSON parse error: {err}")))?;

    let mut tensors = HashMap::with_capacity(raw.len());

    for (name, value) in raw {
        let arr = value.as_array().ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor entry '{name}' must be an array"))
        })?;

        // Support both 5-field (offset, size, shape, stride, dtype) and
        // 6-field entries where partition_id is present.
        let (offset, size, shape, stride, dtype, partition_id) = match arr.len() {
            5 => {
                let offset = to_u64(arr.get(0).expect("array has 5 elements"), &name)?;
                let size = to_u64(arr.get(1).expect("array has 5 elements"), &name)?;
                let shape = to_i64_vec(arr.get(2).expect("array has 5 elements"), &name)?;
                let stride = to_i64_vec(arr.get(3).expect("array has 5 elements"), &name)?;
                let dtype = to_string(arr.get(4).expect("array has 5 elements"), &name)?;
                (offset, size, shape, stride, dtype, 0usize)
            }
            6 => {
                let offset = to_u64(arr.get(0).expect("array has 6 elements"), &name)?;
                let size = to_u64(arr.get(1).expect("array has 6 elements"), &name)?;
                let shape = to_i64_vec(arr.get(2).expect("array has 6 elements"), &name)?;
                let stride = to_i64_vec(arr.get(3).expect("array has 6 elements"), &name)?;
                let dtype = to_string(arr.get(4).expect("array has 6 elements"), &name)?;
                let partition_id = to_usize(arr.get(5).expect("array has 6 elements"), &name)?;
                (offset, size, shape, stride, dtype, partition_id)
            }
            _ => {
                return Err(ReaderError::ServerlessLlm(format!(
                    "tensor entry '{}' must have 5 or 6 elements, got {}",
                    name,
                    arr.len()
                )));
            }
        };

        tensors.insert(
            name,
            TensorEntry {
                offset,
                size,
                shape,
                stride,
                dtype,
                partition_id,
            },
        );
    }

    Ok(ServerlessLLMIndex { tensors })
}

fn to_u64(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<u64> {
    value.as_u64().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected u64 in tensor '{tensor_name}', got {value:?}"
        ))
    })
}

fn to_usize(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected usize in tensor '{tensor_name}', got {value:?}"
            ))
        })
}

fn to_string(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<String> {
    value.as_str().map(std::string::ToString::to_string).ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected string in tensor '{tensor_name}', got {value:?}"
        ))
    })
}

fn to_i64_vec(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<Vec<i64>> {
    let arr = value.as_array().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected array in tensor '{tensor_name}', got {value:?}"
        ))
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(v.as_i64().ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected integer in tensor '{tensor_name}', got {v:?}"
            ))
        })?);
    }
    Ok(out)
}
