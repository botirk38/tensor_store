//! TensorStore format reader.
//!
//! This module provides functionality to parse TensorStore index files
//! and extract shard information for parallel loading.
//!
//! # Format Structure
//!
//! ```text
//! [Header: 8 bytes]
//!   magic: 4 bytes ("TSML")
//!   tensor_count: 4 bytes
//! [Index Entries: 64 bytes each]
//!   shard_id: 1 byte
//!   offset: 7 bytes (56-bit)
//!   size: 4 bytes
//!   dtype: 1 byte
//!   rank: 1 byte
//!   name_len: 2 bytes
//!   shape: 32 bytes (8 × u32)
//!   name_inline: 16 bytes
//! ```
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::readers::tensorstore;
//!
//! // Parse TensorStore index
//! let index = tensorstore::parse_index("model.index").await?;
//!
//! // Access tensor information
//! for entry in &index {
//!     println!("Tensor in shard {} at offset {}",
//!              entry.shard_id, entry.offset);
//! }
//! ```

use crate::readers::error::{ReaderError, ReaderResult};
use crate::readers::traits::{AsyncReader, SyncReader, TensorMetadata};
use std::path::Path;

/// Parsed TensorStore index
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct TensorStoreIndex {
    /// All index entries
    entries: Vec<IndexEntry>,
}

impl TensorStoreIndex {
    /// Creates a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the entries.
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }

    /// Gets an entry by index.
    pub fn get(&self, index: usize) -> Option<&IndexEntry> {
        self.entries.get(index)
    }

    /// Returns an iterator over the entries.
    pub fn iter(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.iter()
    }

    /// Gets the tensor name from an entry.
    pub fn tensor_name<'a>(&self, entry: &'a IndexEntry) -> &'a str {
        let name_bytes = &entry.name_inline[..entry.name_len as usize];
        std::str::from_utf8(name_bytes).unwrap_or("<invalid utf8>")
    }
}

impl<'a> IntoIterator for &'a TensorStoreIndex {
    type Item = &'a IndexEntry;
    type IntoIter = std::slice::Iter<'a, IndexEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl TensorMetadata for TensorStoreIndex {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|entry| {
            let entry_name_bytes = &entry.name_inline[..entry.name_len as usize];
            if let Ok(entry_name) = std::str::from_utf8(entry_name_bytes) {
                entry_name == name
            } else {
                false
            }
        })
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter_map(|entry| {
                let name_bytes = &entry.name_inline[..entry.name_len as usize];
                std::str::from_utf8(name_bytes).ok()
            })
            .collect()
    }
}

impl AsyncReader for TensorStoreIndex {
    type Output = Self;

    async fn load(_path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        // TODO: Implement binary parsing of TensorStore index format
        Err(ReaderError::TensorStore(
            "TensorStore parsing not yet implemented".to_string(),
        ))
    }
}

impl SyncReader for TensorStoreIndex {
    type Output = Self;

    fn load_sync(_path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        // TODO: Implement binary parsing of TensorStore index format
        Err(ReaderError::TensorStore(
            "TensorStore parsing not yet implemented".to_string(),
        ))
    }
}

/// Single index entry
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IndexEntry {
    /// Which shard file (0-255)
    pub shard_id: u8,
    /// Byte offset within shard
    pub offset: u64,
    /// Tensor data size in bytes
    pub size: u32,
    /// Data type
    pub dtype: u8,
    /// Number of dimensions
    pub rank: u8,
    /// Length of tensor name
    pub name_len: u16,
    /// Tensor shape (up to 8 dimensions)
    pub shape: [u32; 8],
    /// Inline tensor name (up to 16 bytes)
    pub name_inline: [u8; 16],
}

/// Parse TensorStore index file.
///
/// # Note
///
/// This function is not yet implemented and will return an error.
pub async fn parse_index(path: impl AsRef<Path>) -> ReaderResult<TensorStoreIndex> {
    TensorStoreIndex::load(path).await
}

/// Parse TensorStore index file synchronously.
///
/// # Note
///
/// This function is not yet implemented and will return an error.
pub fn parse_index_sync(path: impl AsRef<Path>) -> ReaderResult<TensorStoreIndex> {
    TensorStoreIndex::load_sync(path)
}
