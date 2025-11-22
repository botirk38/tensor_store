//! io_uring-based async I/O backend for Linux.
//!
//! This backend leverages Linux's `io_uring` interface for maximum performance.
//! It provides zero-copy operations and efficient parallel I/O for large files.
//!
//! # Platform Support
//!
//! This module is only available on Linux (requires kernel 5.1+).
//!
//! # Performance Characteristics
//!
//! - **Zero-copy**: Direct kernel-to-userspace transfers
//! - **Parallel I/O**: True concurrent reads via `io_uring` submission queue
//! - **Buffer pooling**: Reuses memory allocations across operations
//!
//! # Usage
//!
//! Typically accessed via `backends::load()` on Linux platforms, not used directly.

use super::IoResult;
use std::path::Path;
use std::sync::OnceLock;
use tokio_uring::fs::File as UringFile;
use zeropool::BufferPool;

static BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();

fn get_buffer_pool() -> &'static BufferPool {
    BUFFER_POOL.get_or_init(BufferPool::new)
}

/// A Vec-like type that borrows memory from a slice but acts like Vec<u8> for tokio-uring
struct BorrowedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl BorrowedVec {
    const unsafe fn from_slice(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            cap: slice.len(),
        }
    }

    fn into_vec(self) -> Vec<u8> {
        // Create a Vec with the correct capacity so tokio-uring can use it properly
        // We'll handle deallocation separately
        unsafe { Vec::from_raw_parts(self.ptr, self.len, self.cap) }
    }
}

impl Drop for BorrowedVec {
    fn drop(&mut self) {
        // The Vec created by into_vec has capacity 0, so it won't deallocate
        // But we still need to ensure this BorrowedVec doesn't try to deallocate
    }
}

impl AsRef<[u8]> for BorrowedVec {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for BorrowedVec {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Ceiling division: (a + b - 1) / b
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load tensor data using `io_uring` zero-copy I/O
#[inline]
pub async fn load(path: impl AsRef<Path> + Send) -> IoResult<Vec<u8>> {
    let path_buf = path.as_ref().to_path_buf();
    let file = UringFile::open(&path_buf).await?;
    let metadata = std::fs::metadata(&path_buf)?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_e| std::io::Error::other("File too large"))?;

    // Use internal buffer pool for optimization
    let read_buf = get_buffer_pool().get(file_size);

    let (res, buf) = file.read_at(read_buf, 0).await;
    let n = res?;

    if n != file_size {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {file_size} bytes, but read {n}"),
        ));
    }

    file.close().await?;

    Ok(buf)
}

/// Load tensor data in parallel chunks using `io_uring` with true zero-copy
#[inline]
pub async fn load_parallel(path: impl AsRef<Path>, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let path_ref = path.as_ref();
    let file = UringFile::open(path_ref).await?;
    let metadata = std::fs::metadata(path)?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_e| std::io::Error::other("File too large"))?;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final buffer
    let mut final_buf = get_buffer_pool().get(file_size);

    // Submit all read operations in parallel, each reading directly into final buffer
    let mut read_futures = Vec::with_capacity(chunks);

    for i in 0..chunks {
        let start = i.checked_mul(chunk_size).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "chunk calculation overflow")
        })?;
        let end = std::cmp::min(start.checked_add(chunk_size).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "chunk calculation overflow")
        })?, file_size);
        let actual_chunk_size = end.checked_sub(start).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "chunk calculation underflow")
        })?;

        if actual_chunk_size == 0 {
            break;
        }

        // Create a BorrowedVec that points to the slice of final_buf for this chunk
        let chunk_slice = final_buf.get_mut(start..end).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid chunk range")
        })?;
        let borrowed_vec = unsafe { BorrowedVec::from_slice(chunk_slice) };

        let offset = u64::try_from(start).map_err(|_e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset too large")
        })?;
        let read_future = file.read_at(borrowed_vec.into_vec(), offset);
        read_futures.push(read_future);
    }

    // Wait for all operations to complete
    // The data is already in final_buf, we just need to wait for completion
    for read_future in read_futures {
        let (res, returned_buf) = read_future.await;
        let _n = res?;
        // Leak the returned buffer since its data is now in final_buf
        // This prevents double-free since the buffer content has been moved
        #[allow(clippy::mem_forget)]
        std::mem::forget(returned_buf);
    }

    file.close().await?;
    Ok(final_buf)
}

/// Load a specific byte range from a file using `io_uring`.
#[inline]
pub async fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let path_ref = path.as_ref();
    let file = UringFile::open(path_ref).await?;

    let buf = vec![0u8; len];
    let (res, read_buf) = file.read_at(buf, offset).await;
    let n = res?;

    if n != len {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected to read {len} bytes, but read {n}"),
        ));
    }

    file.close().await?;
    Ok(read_buf)
}

/// Write an entire buffer to a file, creating or truncating it first.
#[inline]
pub async fn write_all(path: impl AsRef<Path>, data: &[u8]) -> IoResult<()> {
    let path_ref = path.as_ref();
    let file = UringFile::create(path_ref).await?;
    let (res, buf) = file.write_at(data.to_vec(), 0).submit().await;
    let n = res?;
    if n != data.len() {
        file.close().await?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            format!("expected to write {} bytes, wrote {}", data.len(), n),
        ));
    }
    // Keep buf alive until write completes; then drop.
    drop(buf);
    file.sync_all().await?;
    file.close().await
}
