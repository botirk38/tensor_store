//! Lazy loading for ServerlessLLM format (memory-mapped).

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::TensorMetadata;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use super::index::Index;
use super::tensor::TensorMmap;

/// ServerlessLLM model with memory-mapped partition files (lazy loading).
#[derive(Debug)]
pub struct MmapModel {
    index: Index,
    partitions: HashMap<usize, backends::mmap::Mmap>,
}

impl MmapModel {
    /// Loads a ServerlessLLM model from directory with mmap-based lazy loading.
    pub fn load(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = Index::load_sync(&index_path)?;
        let partition_ids = index.partition_ids();

        // Parallel mmap of all partition files
        let partitions: Result<HashMap<usize, backends::mmap::Mmap>, ReaderError> = partition_ids
            .par_iter()
            .map(|&partition_id| {
                let partition_path = format!("{}_{}", data_path.display(), partition_id);
                let mmap = backends::mmap::map(&partition_path)?;
                Ok((partition_id, mmap))
            })
            .collect();

        Ok(Self {
            index,
            partitions: partitions?,
        })
    }

    /// Returns a lazy view of the tensor with the given name.
    ///
    /// The tensor data is not copied - it's a zero-copy view into the memory-mapped file.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<TensorMmap> {
        let entry = self.index.get(name)?;
        let mmap = self.partitions.get(&entry.partition_id)?;

        // Create a range view of the mmap for this tensor
        let start = usize::try_from(entry.offset).ok()?;
        let len = usize::try_from(entry.size).ok()?;
        let end = start.checked_add(len)?;

        if end > mmap.len() {
            return None;
        }

        // Create a sub-slice mmap (this is zero-copy)
        let tensor_mmap = backends::mmap::Mmap {
            inner: std::sync::Arc::clone(&mmap.inner),
            start: mmap.start + start,
            len,
        };

        Some(TensorMmap::new(tensor_mmap, entry.clone()))
    }

    /// Returns tensor names.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.tensor_names()
    }

    /// Returns the number of tensors in the model.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns true when no tensors are mapped.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Returns an iterator over all tensor names.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.index.tensor_names().into_iter().filter_map(|s| {
            // This is a bit awkward, but we return references to the names from the index
            self.index
                .iter()
                .find_map(|(name, _)| if name.as_str() == s { Some(name) } else { None })
        })
    }
}

impl TensorMetadata for MmapModel {
    #[inline]
    fn len(&self) -> usize {
        self.index.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.index.contains(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.index.tensor_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_model_empty() {
        let model = MmapModel {
            index: Index::new(),
            partitions: HashMap::new(),
        };
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }
}
