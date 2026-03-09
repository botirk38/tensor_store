//! Shared helpers for ServerlessLLM loaders.

use crate::formats::error::{ReaderError, ReaderResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use super::index::Index;
use super::tensor::{IndexEntry, Tensor};

/// Build the full partition file path from base path and partition ID.
pub fn build_partition_path(base_path: &str, partition_id: usize) -> String {
    format!("{}_{}", base_path, partition_id)
}

/// Compute the required byte size: offset + size.
pub fn required_size(entry: &IndexEntry, name: &str) -> ReaderResult<u64> {
    entry
        .offset
        .checked_add(entry.size)
        .ok_or_else(|| ReaderError::OffsetOverflow {
            name: name.to_string(),
        })
}

/// Validate that partition exists and is large enough, using cache when possible.
pub fn validate_partition_size(
    partition_path: &Path,
    partition_id: usize,
     required_size: u64,
     cache: &Mutex<HashMap<usize, u64>>,
 ) -> ReaderResult<u64> {
     // Check cache first
     if let Ok(guard) = cache.lock()
         && let Some(&cached_size) = guard.get(&partition_id) {
             if cached_size < required_size {
                 return Err(ReaderError::PartitionTooSmall {
                     path: partition_path.display().to_string(),
                     actual: cached_size,
                     required: required_size,
                 });
             }
             return Ok(cached_size);
         }

    // Cache miss - stat file
    let metadata =
        std::fs::metadata(partition_path).map_err(|_| ReaderError::PartitionNotFound {
            partition_id,
            path: partition_path.display().to_string(),
        })?;

    let actual_size = metadata.len();

    // Update cache
    let _ = cache.lock().map(|mut guard| {
        guard.insert(partition_id, actual_size);
    });

    if actual_size < required_size {
        return Err(ReaderError::PartitionTooSmall {
            path: partition_path.display().to_string(),
            actual: actual_size,
            required: required_size,
        });
    }

    Ok(actual_size)
}

/// Extract a tensor from partition buffer at the given offset and length.
pub fn slice_tensor_from_partition(
    name: String,
    partition_buf: &[u8],
    entry: &IndexEntry,
) -> ReaderResult<Tensor> {
    let start = usize::try_from(entry.offset).map_err(|_| ReaderError::SizeTooLarge {
        name: name.clone(),
        size: entry.size,
    })?;

    let len = usize::try_from(entry.size).map_err(|_| ReaderError::SizeTooLarge {
        name: name.clone(),
        size: entry.size,
    })?;

    let end = start
        .checked_add(len)
        .ok_or_else(|| ReaderError::SizeTooLarge {
            name: name.clone(),
            size: entry.size,
        })?;

    if end > partition_buf.len() {
        return Err(ReaderError::PartitionTooSmall {
            path: format!("partition_{}", entry.partition_id),
            actual: u64::try_from(partition_buf.len()).unwrap_or(u64::MAX),
            required: entry.offset + entry.size,
        });
    }

    let data = partition_buf[start..end].to_vec();
    Ok(Tensor::from_owned(data, entry.clone()))
}

/// Build a request list for batching, grouped by partition.
pub fn group_requests_by_partition<'a>(
    index: &'a Index,
    tensor_names: &[&str],
) -> ReaderResult<HashMap<usize, Vec<(String, &'a IndexEntry)>>> {
    let mut partition_requests: HashMap<usize, Vec<(String, &IndexEntry)>> = HashMap::new();

    for &name in tensor_names {
        let entry = index.get(name).ok_or_else(|| ReaderError::TensorNotFound {
            name: name.to_string(),
        })?;

         partition_requests
             .entry(entry.partition_id)
             .or_default()
             .push((name.to_string(), entry));
    }

    Ok(partition_requests)
}
