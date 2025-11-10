use super::IoResult;
use memmap2::MmapOptions;
use region::page;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;

#[inline]
fn ensure_range(file_len: u64, offset: u64, len: usize) -> IoResult<()> {
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "range overflows u64"))?;

    if end > file_len {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "range {}..{} exceeds file length {} bytes",
                offset, end, file_len
            ),
        ));
    }
    Ok(())
}

#[inline]
fn open_file(path: &str) -> IoResult<File> {
    let path = Path::new(path);
    File::open(path)
}

/// Load tensor data using a read-only memory map.
///
/// This backend maps the full file and copies the requested bytes into
/// a `Vec<u8>`. The mapping ensures the kernel performs the heavy lifting
/// which can be faster than standard buffered reads for large files.
pub async fn load(path: &str) -> IoResult<Vec<u8>> {
    let file = open_file(path)?;
    let metadata = file.metadata()?;
    let file_len = metadata.len() as usize;

    if file_len == 0 {
        return Ok(Vec::new());
    }

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap[..].to_vec())
}

/// Load tensor data in chunks via memory mapping.
///
/// Since memory mapping already provides random access, this simply
/// maps the full file and returns its contents. The `chunks` parameter
/// is kept for API compatibility.
pub async fn load_parallel(path: &str, _chunks: usize) -> IoResult<Vec<u8>> {
    load(path).await
}

/// Load a specific range from a tensor file using a partial memory map.
pub async fn load_range(path: &str, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }

    let file = open_file(path)?;
    let metadata = file.metadata()?;
    ensure_range(metadata.len(), offset, len)?;

    let page_size = page::size() as u64;
    let aligned_offset = (offset / page_size) * page_size;
    let offset_delta = (offset - aligned_offset) as usize;
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
    let end = start + len;
    Ok(mmap[start..end].to_vec())
}
