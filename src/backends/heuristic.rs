//! Shared load heuristics used across all backends.
//!
//! This module centralizes the decision logic for when to use single-read
//! vs chunked parallel reads, ensuring consistent behavior across sync,
//! tokio async, and io_uring backends.

/// Maximum size for a single read operation.
/// Files larger than this automatically use parallel chunked reading.
pub const MAX_SINGLE_READ: usize = 512 * 1024 * 1024; // 512MB

/// Target queue depth for parallel operations.
/// This saturates NVMe devices without overwhelming the kernel.
pub const DEFAULT_QUEUE_DEPTH: usize = 64;

/// Minimum chunk size in bytes (16MB).
/// Prevents oversplitting small files into tiny requests.
pub const MIN_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Maximum chunk size in bytes (64MB).
/// Prevents undersplitting large files.
pub const MAX_CHUNK_SIZE: usize = 64 * 1024 * 1024;

/// Calculate the optimal number of chunks for parallel reading.
///
/// Returns 1 if the file should be read in a single operation.
#[inline]
pub fn calculate_chunks(file_size: usize) -> usize {
    if file_size <= MAX_SINGLE_READ {
        return 1;
    }
    let min_chunks = file_size.div_ceil(MAX_CHUNK_SIZE);
    let max_chunks = file_size.div_ceil(MIN_CHUNK_SIZE);
    min_chunks
        .max(1)
        .clamp(1, max_chunks.min(DEFAULT_QUEUE_DEPTH))
}

/// Calculate optimal chunk size for parallel reading.
#[inline]
pub fn calculate_chunk_size(file_size: usize, chunks: usize) -> usize {
    if chunks <= 1 {
        return file_size;
    }
    file_size.div_ceil(chunks)
}

/// Determines whether a file should use single-read or chunked parallel loading.
#[inline]
pub fn should_use_chunked(file_size: usize) -> bool {
    file_size > MAX_SINGLE_READ
}

/// Calculate optimal chunks for async I/O to saturate the io_uring queue.
///
/// Unlike calculate_chunks which only chunks files > 512MB, this function
/// always returns enough chunks to saturate the target queue depth, even
/// for small files. This maximizes async I/O throughput.
#[inline]
pub fn calculate_chunks_async(file_size: usize) -> usize {
    if file_size == 0 {
        return 1;
    }
    let size_based_chunks = calculate_chunks(file_size);
    let min_chunks = DEFAULT_QUEUE_DEPTH.min(file_size.div_ceil(MIN_CHUNK_SIZE));
    size_based_chunks
        .max(min_chunks)
        .clamp(1, DEFAULT_QUEUE_DEPTH)
}
