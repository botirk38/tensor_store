//! Checkpoint writers module for serializing tensor data to various formats.
//!
//! This module provides functionality to write model checkpoints in various formats.
//! It focuses on format-specific serialization logic and delegates I/O operations
//! to the top-level `backends` module.
//!
//! # Architecture
//!
//! ```text
//! writers/
//! ├── serverlessllm.rs   Write ServerlessLLM format
//! ├── tensorstore.rs     Write TensorStore format
//! └── mod.rs
//! ```
//!
//! # Design Philosophy
//!
//! - **Format-specific**: Each writer handles only its target format
//! - **No conversion logic**: Writers expect pre-converted data
//! - **Async-first**: All operations are async for consistency
//! - **Separation of concerns**: Writing vs conversion vs I/O
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::writers::serverlessllm;
//!
//! // Write ServerlessLLM index
//! serverlessllm::write_index("tensor_index.json", &tensors).await?;
//!
//! // Write partition data
//! serverlessllm::write_partition("tensor.data_0", 0, &data).await?;
//! ```

pub mod serverlessllm;
pub mod tensorstore;

pub use std::io::Result as IoResult;
