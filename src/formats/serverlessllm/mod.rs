//! ServerlessLLM format implementation.
//!
//! This module provides reading and writing support for the ServerlessLLM format.

// Internal modules
pub mod helpers;
pub mod index;
pub mod mmap;
pub mod owned;
pub mod tensor;
pub mod writer;

// Re-export data types
pub use index::Index;
pub use mmap::MmapModel;
pub use owned::Model;
pub use tensor::{Tensor, TensorMmap};

// Re-export writer functions
pub use writer::{write_index, write_index_sync, write_partition, write_partition_sync};

pub use helpers::{RECOMMENDED_PARTITION_TARGET_BYTES, recommended_partition_count};
