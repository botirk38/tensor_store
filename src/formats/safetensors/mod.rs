//! SafeTensors format implementation.
//!
//! This module provides reading and writing support for the SafeTensors format.

pub mod reader;
pub mod writer;

pub use safetensors::serialize;

// Re-export reader types
pub use reader::{
    Dtype, SafeTensorError, MmapModel, Model, Tensor,
};

// Re-export writer types
pub use writer::{MetadataMap, Writer, TensorView, View};
