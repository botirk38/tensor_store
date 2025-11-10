//! Checkpoint format readers.
//!
//! This module provides functionality to parse and deserialize different
//! checkpoint formats. Each format reader extracts metadata and provides
//! structured access to tensor information.
//!
//! # Architecture
//!
//! ```text
//! readers/
//! ├── safetensors.rs     Parse SafeTensors metadata
//! ├── tensorstore.rs     Parse TensorStore index
//! ├── serverlessllm.rs   Parse ServerlessLLM index
//! └── mod.rs
//! ```
//!
//! # Design Philosophy
//!
//! - **Streaming**: Parse metadata without loading all data into memory
//! - **Structured**: Return typed data structures, not raw bytes
//! - **Validation**: Check format integrity during parsing
//! - **Extensible**: Easy to add new format readers
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::readers::safetensors;
//!
//! // Parse SafeTensors file
//! let tensors = safetensors::load("model.safetensors").await?;
//!
//! // Access tensor information
//! for name in tensors.names() {
//!     let view = tensors.tensor(name)?;
//!     println!("Tensor: {} ({:?})", name, view.shape());
//! }
//! ```

pub mod safetensors;
pub mod serverlessllm;
pub mod tensorstore;
