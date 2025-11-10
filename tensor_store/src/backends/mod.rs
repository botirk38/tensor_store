//! Format-agnostic async I/O operations for writing.
//!
//! This module provides platform-specific primitives for writing data to disk.
//! Operations are designed to be zero-copy where possible and async-first.
//!
//! # Platform Support
//!
//! - **Linux**: io_uring backend for maximum performance
//! - **Cross-platform**: tokio-based async I/O fallback
//! - **Universal**: mmap backend for fast random access reads
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::writers::backends;
//!
//! // Write complete buffer
//! backends::write_all("output.bin", &data).await?;
//!
//! // Write at specific offset
//! backends::write_range("output.bin", 1024, &data).await?;
//!
//! // Create and preallocate file
//! backends::create_file("output.bin").await?;
//! backends::truncate_file("output.bin", 1_000_000).await?;
//! ```

#[cfg(target_os = "linux")]
pub mod io_uring;

pub mod async_io;
pub mod mmap;

pub use std::io::Result as IoResult;
