//! TensorStore format writer.
//!
//! This module exposes a lightweight [`TensorStoreWriter`] that mirrors the
//! layout produced by the (future) TensorStore reader. All methods are async
//! placeholders so the DX stays consistent while format details are
//! implemented.
//!
//! ```rust,ignore
//! use tensor_store::writers::tensorstore::{TensorStoreWriter, IndexEntry};
//!
//! let writer = TensorStoreWriter::new();
//! let entries = vec![IndexEntry::default()];
//! writer.write_index("model.index", &entries).await?;
//! writer.write_shard("shard_0.bin", 0, &[0u8; 1024]).await?;
//! ```

use crate::writers::error::WriterResult;

// Re-export shared IndexEntry type
pub use crate::types::tensorstore::IndexEntry;

/// High-level writer for TensorStore checkpoint artifacts.
#[derive(Debug, Default, Clone, Copy)]
pub struct TensorStoreWriter;

impl TensorStoreWriter {
    /// Create a new writer instance.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Write the TensorStore index file.
    pub async fn write_index(&self, output_path: &str, entries: &[IndexEntry]) -> WriterResult<()> {
        write_index(output_path, entries).await
    }

    /// Write a binary shard containing tensor data.
    pub async fn write_shard(
        &self,
        output_path: &str,
        shard_id: u8,
        data: &[u8],
    ) -> WriterResult<()> {
        write_shard(output_path, shard_id, data).await
    }
}

/// Write a TensorStore index file.
pub async fn write_index(_output_path: &str, _entries: &[IndexEntry]) -> WriterResult<()> {
    todo!("Implement TensorStore index writing")
}

/// Write a TensorStore shard file.
pub async fn write_shard(_output_path: &str, _shard_id: u8, _data: &[u8]) -> WriterResult<()> {
    todo!("Implement TensorStore shard writing")
}
