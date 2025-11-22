use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;

use memmap2::MmapOptions;
use region::page;

use super::{IoResult, pooled_buffer::PooledBuffer};

#[inline]
fn ensure_range(file_len: u64, offset: u64, len: usize) -> IoResult<()> {
    let end = offset
        .checked_add(
            u64::try_from(len)
                .map_err(|_e| Error::new(ErrorKind::InvalidInput, "length too large for u64"))?,
        )
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "range overflows u64"))?;

    if end > file_len {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            format!("range {offset}..{end} exceeds file length {file_len} bytes",),
        ));
    }
    Ok(())
}

/// Load tensor data using a read-only memory map.
///
/// This backend maps the full file and copies the requested bytes into
/// a `PooledBuffer`. The mapping ensures the kernel performs the heavy lifting
/// which can be faster than standard buffered reads for large files.
/// Load tensor data using memory mapping (blocking)
#[inline]
pub fn load_blocking(path: impl AsRef<Path>) -> IoResult<Vec<u8>> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let metadata = file.metadata()?;
    let file_len = usize::try_from(metadata.len())
        .map_err(|_e| Error::new(ErrorKind::InvalidInput, "file too large"))?;

    if file_len == 0 {
        return Ok(PooledBuffer::with_capacity(0).into_vec());
    }

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mut buf = PooledBuffer::with_capacity(file_len);
    buf.as_mut_slice().copy_from_slice(&mmap[..]);
    buf.truncate(file_len);
    Ok(buf.into_vec())
}

/// Load a specific range from a tensor file using a partial memory map.
#[inline]
pub fn load_range_blocking(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    if len == 0 {
        return Ok(PooledBuffer::with_capacity(0).into_vec());
    }

    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let metadata = file.metadata()?;
    ensure_range(metadata.len(), offset, len)?;

    let page_size = u64::try_from(page::size()).unwrap_or(4096);
    let aligned_offset = offset
        .checked_div(page_size)
        .unwrap_or(0)
        .checked_mul(page_size)
        .unwrap_or(0);
    let offset_delta = usize::try_from(offset.saturating_sub(aligned_offset))
        .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?;
    let map_len = len
        .checked_add(offset_delta)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "range length overflow"))?;

    let mmap = unsafe {
        MmapOptions::new()
            .offset(aligned_offset)
            .len(map_len)
            .map(&file)?
    };

    let start = offset_delta;
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "range end overflow"))?;
    let slice = mmap
        .get(start..end)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "invalid slice range"))?;

    let mut buf = PooledBuffer::with_capacity(len);
    buf.as_mut_slice().copy_from_slice(slice);
    buf.truncate(len);
    Ok(buf.into_vec())
}
