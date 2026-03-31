//! io_uring-based explicit I/O backend for Linux.
//!
//! This backend provides specialized Linux I/O using the `io_uring` interface directly,
//! as opposed to the default `AsyncReader`/`AsyncWriter` which are Tokio-backed on all
//! platforms.
//!
//! Use this directly when you want explicit io_uring behavior on Linux. On other
//! platforms this module is not available.
//!
//! # Design
//!
//! - Single ring, no user-space threading
//! - Batch reads/writes, keep ring full continuously
//! - Two explicit paths: buffered/page-cache and direct-I/O
//! - Direct-I/O only when layout is prevalidated as aligned
//! - Ring built with SINGLE_ISSUER + COOP_TASKRUN + SUBMIT_ALL for single-threaded saturation

use super::odirect::{is_block_aligned, open_direct_read_sync, alloc_aligned, can_use_direct_read};
use super::{BatchRequest, IoResult, batch::FlattenedResult, byte::OwnedBytes, get_buffer_pool, calculate_chunks, build_chunk_plan, validate_read_count, MAX_CHUNK_SIZE};
use io_uring::{opcode, types, IoUring};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;

const RING_DEPTH: usize = 32;

fn build_ring() -> IoResult<IoUring> {
    IoUring::builder()
        .setup_single_issuer()
        .setup_coop_taskrun()
        .setup_submit_all()
        .build(RING_DEPTH as u32)
}

pub struct Reader;

impl Reader {
    pub const fn new() -> Self {
        Self
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> IoResult<OwnedBytes> {
        let file = File::open(path.as_ref())?;
        let file_size = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("file too large"))?;
        if file_size == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        let chunk_size = file_size.div_ceil(calculate_chunks(file_size));
        if can_use_direct_read(file_size, chunk_size) {
            return self.load_direct(path.as_ref(), file_size);
        }
        self.load_buffered(file, file_size)
    }

    fn load_buffered(&mut self, file: File, file_size: usize) -> IoResult<OwnedBytes> {
        let mut buffer = get_buffer_pool().get(file_size);
        let base_ptr = buffer.as_mut_ptr();
        let plan = build_chunk_plan(file_size);
        
        if plan.is_empty() {
            return Ok(OwnedBytes::Pooled(buffer));
        }

        let mut ring = build_ring()?;
        let fd = file.as_raw_fd();
        
        let mut pending = plan.len();
        let mut next_idx = 0;
        
        while pending > 0 {
            while next_idx < plan.len() && ring.submission().capacity() > 0 {
                let chunk = &plan[next_idx];
                let ptr = unsafe { base_ptr.add(chunk.offset as usize) };
                let sqe = opcode::Read::new(types::Fd(fd), ptr, chunk.len as u32)
                    .offset(chunk.offset)
                    .build()
                    .user_data(next_idx as u64);
                
                unsafe {
                    if ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                next_idx += 1;
            }
            
            ring.submit_and_wait(1)?;
            
            let cq = ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                if idx >= plan.len() {
                    continue;
                }
                let chunk = &plan[idx];
                let result = cqe.result();
                if result < 0 {
                    return Err(std::io::Error::other(format!(
                        "read error at chunk {} (offset {}): {}",
                        idx, chunk.offset, result
                    )));
                }
                let bytes_read = result as usize;
                if bytes_read < chunk.len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read at chunk {}: expected {} bytes, got {}",
                            idx, chunk.len, bytes_read
                        ),
                    ));
                }
                pending -= 1;
            }
        }

        drop(file);
        Ok(OwnedBytes::Pooled(buffer))
    }

    fn load_direct(&mut self, path: &Path, file_size: usize) -> IoResult<OwnedBytes> {
        let file = open_direct_read_sync(path)?;
        let mut buffer = alloc_aligned(file_size)?;
        buffer.set_len(file_size);
        let base_ptr = buffer.as_mut_ptr();
        let plan = build_chunk_plan(file_size);
        
        if plan.is_empty() {
            return Ok(OwnedBytes::Aligned(buffer));
        }

        let mut ring = build_ring()?;
        let fd = file.as_raw_fd();
        
        let mut pending = plan.len();
        let mut next_idx = 0;
        
        while pending > 0 {
            while next_idx < plan.len() && ring.submission().capacity() > 0 {
                let chunk = &plan[next_idx];
                let ptr = unsafe { base_ptr.add(chunk.offset as usize) };
                let sqe = opcode::Read::new(types::Fd(fd), ptr, chunk.len as u32)
                    .offset(chunk.offset)
                    .build()
                    .user_data(next_idx as u64);
                
                unsafe {
                    if ring.submission().push(&sqe).is_err() {
                        break;
                    }
                }
                next_idx += 1;
            }
            
            ring.submit_and_wait(1)?;
            
            let cq = ring.completion();
            for cqe in cq {
                let idx = cqe.user_data() as usize;
                if idx >= plan.len() {
                    continue;
                }
                let chunk = &plan[idx];
                let result = cqe.result();
                if result < 0 {
                    return Err(std::io::Error::other(format!(
                        "direct read error at chunk {} (offset {}): {}",
                        idx, chunk.offset, result
                    )));
                }
                let bytes_read = result as usize;
                if bytes_read < chunk.len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short direct read at chunk {}: expected {} bytes, got {}",
                            idx, chunk.len, bytes_read
                        ),
                    ));
                }
                pending -= 1;
            }
        }

        Ok(OwnedBytes::Aligned(buffer))
    }

    pub fn load_range(&mut self, path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        if len == 0 {
            return Ok(OwnedBytes::Shared(Arc::new([])));
        }

        let path_ref = path.as_ref();
        
        if is_block_aligned(offset, len) {
            let file = open_direct_read_sync(path_ref)?;
            return self.load_range_direct(file, offset, len);
        }
        
        let file = File::open(path_ref)?;
        self.load_range_buffered(file, offset, len)
    }

    fn load_range_buffered(&mut self, file: File, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut buffer = get_buffer_pool().get(len);
        let ptr = buffer.as_mut_ptr();

        let mut ring = build_ring()?;
        let fd = file.as_raw_fd();
        
        let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            ring.submission().push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;

        let mut cq: io_uring::CompletionQueue<'_> = ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;
        
        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!("read error: {}", cqe.result())));
        }
        
        let bytes_read = cqe.result() as usize;
        validate_read_count(bytes_read, len)?;
        
        drop(file);
        Ok(OwnedBytes::Pooled(buffer))
    }

    fn load_range_direct(&mut self, file: File, offset: u64, len: usize) -> IoResult<OwnedBytes> {
        let mut buffer = alloc_aligned(len)?;
        let ptr = buffer.as_mut_ptr();

        let mut ring = build_ring()?;
        let fd = file.as_raw_fd();
        
        let sqe = opcode::Read::new(types::Fd(fd), ptr, len as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            ring.submission().push(&sqe)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;

        let mut cq: io_uring::CompletionQueue<'_> = ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;
        
        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!("direct read error: {}", cqe.result())));
        }
        
        let bytes_read = cqe.result() as usize;
        validate_read_count(bytes_read, len)?;
        
        buffer.set_len(bytes_read);
        drop(file);
        Ok(OwnedBytes::Aligned(buffer))
    }

    pub fn load_range_batch(&mut self, requests: &[BatchRequest]) -> IoResult<Vec<FlattenedResult>> {
        use super::batch::group_requests_by_file;
        use std::collections::HashMap;

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let grouped = group_requests_by_file(requests);
        let mut all_results: Vec<(usize, std::sync::Arc<[u8]>, usize, usize)> =
            Vec::with_capacity(requests.len());

        for (path, reqs) in grouped {
            if reqs.is_empty() {
                continue;
            }

            let file = File::open(&path)?;
            let fd = file.as_raw_fd();
            
            let ring_depth = RING_DEPTH.max(reqs.len());
            let mut ring = IoUring::builder()
                .setup_single_issuer()
                .setup_coop_taskrun()
                .setup_submit_all()
                .build(ring_depth as u32)?;

            struct PendingBuffer {
                data: zeropool::PooledBuffer,
                expected_len: usize,
            }
            
            let mut pending_buffers: HashMap<usize, PendingBuffer> = HashMap::new();
            let mut pending_count = 0;
            
            let mut next_idx = 0;
            let total = reqs.len();
            
            while pending_count > 0 || next_idx < total {
                while next_idx < total && ring.submission().capacity() > 0 {
                    let req = &reqs[next_idx];
                    
                    if req.len == 0 {
                        all_results.push((req.idx, std::sync::Arc::new([]), 0, 0));
                        next_idx += 1;
                        continue;
                    }
                    
                    let mut buffer = get_buffer_pool().get(req.len);
                    let ptr = buffer.as_mut_ptr();
                    
                    let sqe = opcode::Read::new(types::Fd(fd), ptr, req.len as u32)
                        .offset(req.offset)
                        .build()
                        .user_data(req.idx as u64);
                    
                    unsafe {
                        if ring.submission().push(&sqe).is_err() {
                            break;
                        }
                    }
                    
                    pending_buffers.insert(req.idx, PendingBuffer {
                        data: buffer,
                        expected_len: req.len,
                    });
                    pending_count += 1;
                    next_idx += 1;
                }
                
                if pending_count > 0 && ring.submission().capacity() == 0 {
                    ring.submit()?;
                } else if pending_count > 0 {
                    ring.submit_and_wait(1)?;
                } else {
                    break;
                }
                
                let cq: io_uring::CompletionQueue<'_> = ring.completion();
                for cqe in cq {
                    let idx = cqe.user_data() as usize;
                    let result = cqe.result();
                    
                    if result < 0 {
                        return Err(std::io::Error::other(format!(
                            "batch read error for idx {}: {}",
                            idx, result
                        )));
                    }
                    
                    let bytes_read = result as usize;
                    
                    if let Some(pending) = pending_buffers.remove(&idx) {
                        validate_read_count(bytes_read, pending.expected_len)?;
                        let data: std::sync::Arc<[u8]> = pending.data.into_inner().into();
                        all_results.push((idx, data, 0, bytes_read));
                        pending_count -= 1;
                    }
                }
            }

            drop(file);
        }

        all_results.sort_by_key(|(idx, _, _, _)| *idx);
        Ok(all_results
            .into_iter()
            .map(|(_, data, offset, len)| (data, offset, len))
            .collect())
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Writer {
    file: Option<File>,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer").finish_non_exhaustive()
    }
}

impl Writer {
    pub fn create(path: &Path) -> IoResult<Self> {
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self {
            file: Some(file),
            path: path.to_path_buf(),
        })
    }

    pub fn write_all(&mut self, data: &[u8]) -> IoResult<()> {
        let file = self.file.take().ok_or_else(|| std::io::Error::other("writer closed"))?;
        drop(file);

        {
            let file = std::fs::File::create(&self.path)?;
            file.set_len(0)?;
            drop(file);
        }

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.path)?;

        let fd = file.as_raw_fd();
        self.file = Some(file);

        let mut ring = IoUring::builder()
            .setup_single_issuer()
            .setup_coop_taskrun()
            .setup_submit_all()
            .build(RING_DEPTH as u32)?;

        write_chunks(&mut ring, fd, data, 0)?;

        let fsync_e = opcode::Fsync::new(types::Fd(fd))
            .build()
            .user_data(u64::MAX);

        unsafe {
            ring.submission().push(&fsync_e)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;

        let cq: io_uring::CompletionQueue<'_> = ring.completion();
        for cqe in cq {
            if cqe.user_data() == u64::MAX && cqe.result() < 0 {
                return Err(std::io::Error::other(format!("fsync error: {}", cqe.result())));
            }
        }

        Ok(())
    }

    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> IoResult<()> {
        let file = self.file.as_mut().ok_or_else(|| std::io::Error::other("writer closed"))?;
        let fd = file.as_raw_fd();

        let mut ring = IoUring::builder()
            .setup_single_issuer()
            .setup_coop_taskrun()
            .setup_submit_all()
            .build(RING_DEPTH as u32)?;

        write_chunks(&mut ring, fd, data, offset)?;

        Ok(())
    }

    pub fn sync_all(&mut self) -> IoResult<()> {
        let file = self.file.as_mut().ok_or_else(|| std::io::Error::other("writer closed"))?;
        let fd = file.as_raw_fd();

        let mut ring = build_ring()?;
        let fsync_e = opcode::Fsync::new(types::Fd(fd)).build().user_data(0);

        unsafe {
            ring.submission().push(&fsync_e)
                .map_err(|_| std::io::Error::other("submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;

        let mut cq: io_uring::CompletionQueue<'_> = ring.completion();
        let cqe = cq
            .next()
            .ok_or_else(|| std::io::Error::other("completion queue empty"))?;
        if cqe.result() < 0 {
            return Err(std::io::Error::other(format!("fsync error: {}", cqe.result())));
        }

        Ok(())
    }
}

/// Internal write loop that handles partial writes by resubmitting the unwritten tail.
fn write_chunks(ring: &mut IoUring, fd: i32, data: &[u8], base_offset: u64) -> IoResult<()> {
    let mut offset = 0;
    let mut pending: std::collections::HashMap<u64, (u64, u32)> = std::collections::HashMap::new();
    let mut completions: Vec<(u64, i32)> = Vec::with_capacity(RING_DEPTH);

    while offset < data.len() || !pending.is_empty() {
        while offset < data.len() && ring.submission().capacity() > 0 {
            let chunk_len = (data.len() - offset).min(MAX_CHUNK_SIZE) as u32;
            let chunk_ptr = data[offset..offset + chunk_len as usize].as_ptr();
            let file_offset = base_offset + offset as u64;

            let write_e = opcode::Write::new(types::Fd(fd), chunk_ptr, chunk_len)
                .offset(file_offset)
                .build()
                .user_data(file_offset);

            unsafe {
                if ring.submission().push(&write_e).is_err() {
                    break;
                }
            }

            pending.insert(file_offset, (offset as u64, chunk_len));
            offset += chunk_len as usize;
        }

        if !pending.is_empty() && (offset >= data.len() || ring.submission().capacity() == 0) {
            ring.submit()?;
        } else if !pending.is_empty() {
            ring.submit_and_wait(1)?;
        } else {
            break;
        }

        completions.clear();
        let cq = ring.completion();
        for cqe in cq {
            completions.push((cqe.user_data(), cqe.result()));
        }

        for (file_offset, result) in &completions {
            if *result < 0 {
                return Err(std::io::Error::other(format!(
                    "write error at offset {}: {}",
                    file_offset, result
                )));
            }

            let bytes_written = *result as u32;
            if let Some(&(start, requested)) = pending.get(file_offset) {
                if bytes_written < requested {
                    let unwritten_start = start + bytes_written as u64;
                    let mut unwritten_len = (requested - bytes_written) as usize;
                    let mut sub_offset = 0;
                    while sub_offset < unwritten_len && ring.submission().capacity() > 0 {
                        let sub_len = unwritten_len.min(MAX_CHUNK_SIZE) as u32;
                        let data_start = unwritten_start as usize + sub_offset;
                        let sub_ptr = data[data_start..data_start + sub_len as usize].as_ptr();
                        let sub_file_offset = base_offset + unwritten_start + sub_offset as u64;
                        let sub_e = opcode::Write::new(types::Fd(fd), sub_ptr, sub_len)
                            .offset(sub_file_offset)
                            .build()
                            .user_data(sub_file_offset);
                        unsafe {
                            if ring.submission().push(&sub_e).is_err() {
                                break;
                            }
                        }
                        pending.insert(sub_file_offset, (unwritten_start + sub_offset as u64, sub_len));
                        sub_offset += sub_len as usize;
                        unwritten_len -= sub_len as usize;
                    }
                    if unwritten_len == 0 {
                        pending.remove(file_offset);
                    }
                } else {
                    pending.remove(file_offset);
                }
            }
        }
    }

    Ok(())
}