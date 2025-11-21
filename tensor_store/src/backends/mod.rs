//! High-performance I/O backends for tensor storage.
//!
//! This module provides zero-copy I/O operations optimized for large tensor files.
//! The API is async-first with explicit sync alternatives for blocking contexts.
//!
//! # Platform Support
//!
//! - **Linux**: io_uring backend for async operations (maximum performance)
//! - **Cross-platform**: tokio async I/O fallback
//! - **Sync operations**: memory-mapped I/O on Linux, standard file I/O elsewhere
//!
//! # Usage
//!
//! ```rust,ignore
//! use tensor_store::backends;
//! use std::path::Path;
//!
//! // Async operations (default, recommended)
//! let data = backends::load("model.safetensors").await?;
//! let data = backends::load_parallel("model.safetensors", 4).await?;
//! backends::write_all("output.bin", &data).await?;
//!
//! // Sync operations (for blocking contexts)
//! let data = backends::sync::load("model.safetensors")?;
//! let chunk = backends::sync::load_range("model.safetensors", 1024, 512)?;
//! ```

#[cfg(target_os = "linux")]
pub mod io_uring;

pub mod async_io;
pub mod mmap;

use std::path::Path;
pub use std::io::Result as IoResult;

// ============================================================================
// Async operations (default, platform-optimized)
// ============================================================================

/// Load entire file contents asynchronously.
///
/// Uses io_uring on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load;

#[cfg(not(target_os = "linux"))]
pub use async_io::load;

/// Load file in parallel chunks asynchronously.
///
/// Uses io_uring on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load_parallel;

#[cfg(not(target_os = "linux"))]
pub use async_io::load_parallel;

/// Load a specific byte range from file asynchronously.
///
/// Uses io_uring on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::load_range;

#[cfg(not(target_os = "linux"))]
pub use async_io::load_range;

/// Write entire buffer to file asynchronously.
///
/// Uses io_uring on Linux, tokio async I/O elsewhere.
#[cfg(target_os = "linux")]
pub use io_uring::write_all;

#[cfg(not(target_os = "linux"))]
pub use async_io::write_all;

// ============================================================================
// Sync operations
// ============================================================================

/// Synchronous I/O operations for blocking contexts.
pub mod sync {
    use super::*;

    /// Load entire file contents synchronously.
    ///
    /// Uses memory-mapped I/O on Linux, standard file I/O elsewhere.
    #[inline]
    pub fn load(path: impl AsRef<Path>) -> IoResult<Vec<u8>> {
        #[cfg(target_os = "linux")]
        {
            super::mmap::load_blocking(path)
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::fs::read(path)
        }
    }

    /// Load a specific byte range synchronously.
    ///
    /// Uses memory-mapped I/O on Linux, standard file I/O elsewhere.
    #[inline]
    pub fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
        #[cfg(target_os = "linux")]
        {
            super::mmap::load_range_blocking(path, offset, len)
        }
        #[cfg(not(target_os = "linux"))]
        {
            use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(path)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)
                .map_err(|e| Error::new(ErrorKind::UnexpectedEof, e))?;
            Ok(buf)
        }
    }
}
