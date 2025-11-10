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
//! use tensor_store::readers::formats::tensorstore;
//!
//! // Parse TensorStore index
//! let index = tensorstore::parse_index("model.index").await?;
//!
//! // Access tensor information
//! for entry in &index.entries {
//!     println!("Tensor in shard {} at offset {}",
//!              entry.shard_id, entry.offset);
//! }
//! ```

/// Parsed TensorStore index
#[derive(Debug)]
pub struct TensorStoreIndex {
    /// All index entries
    pub entries: Vec<IndexEntry>,
}

/// Single index entry
#[derive(Debug)]
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

/// Parse TensorStore index file (Future implementation)
pub async fn parse_index(_path: &str) -> crate::IoResult<TensorStoreIndex> {
    todo!("Implement TensorStore index parsing")
}
