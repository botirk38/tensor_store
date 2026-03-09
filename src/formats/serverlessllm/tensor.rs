//! Tensor data containers for ServerlessLLM format.

use crate::backends;
use crate::formats::traits::TensorView;

/// Metadata for a single tensor in the ServerlessLLM index.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    /// Byte offset of the tensor data within the partition file
    pub offset: u64,

    /// Size of the tensor data in bytes
    pub size: u64,

    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,

    /// Stride information for the tensor
    pub stride: Vec<usize>,

    /// Data type of the tensor (e.g., "float32", "int64")
    pub dtype: String,

    /// Partition ID where this tensor is stored
    pub partition_id: usize,
}

/// View into a memory-mapped tensor with metadata access (lazy loading).
#[derive(Debug)]
pub struct TensorMmap {
    mmap: backends::mmap::Mmap,
    entry: IndexEntry,
}

impl TensorMmap {
    /// Creates a new TensorMmap from memory-mapped data.
    #[inline]
    #[must_use]
    pub const fn new(mmap: backends::mmap::Mmap, entry: IndexEntry) -> Self {
        Self { mmap, entry }
    }

    /// Returns the memory-mapped tensor data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.entry.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.entry.size
    }
}

impl TensorView for TensorMmap {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    #[inline]
    fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    #[inline]
    fn data(&self) -> &[u8] {
        self.data()
    }
}

/// Owned tensor with data loaded into memory.
///
/// Supports zero-copy access by holding a reference to shared buffers
/// with offset/length metadata.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<u8>,
    offset: usize,
    len: usize,
    entry: IndexEntry,
}

impl Tensor {
    /// Creates a new Tensor from buffer data and metadata.
    #[inline]
    #[must_use]
    pub const fn new(data: Vec<u8>, offset: usize, len: usize, entry: IndexEntry) -> Self {
        Self {
            data,
            offset,
            len,
            entry,
        }
    }

    /// Creates a new Tensor from owned data.
    #[inline]
    #[must_use]
    pub const fn from_owned(data: Vec<u8>, entry: IndexEntry) -> Self {
        let len = data.len();
        Self {
            data,
            offset: 0,
            len,
            entry,
        }
    }

    /// Returns the raw tensor data as a slice (zero-copy).
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Consumes the tensor and returns the raw tensor data.
    ///
    /// If this tensor covers the entire buffer, returns the owned `Vec<u8>`.
    /// Otherwise, returns a copy of the slice.
    #[inline]
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        if self.offset == 0 && self.len == self.data.len() {
            self.data
        } else {
            self.data[self.offset..self.offset + self.len].to_vec()
        }
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.entry.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.entry.size
    }

    /// Returns the partition id containing this tensor.
    #[inline]
    #[must_use]
    pub const fn partition_id(&self) -> usize {
        self.entry.partition_id
    }
}

impl TensorView for Tensor {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.entry.shape
    }

    #[inline]
    fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    #[inline]
    fn data(&self) -> &[u8] {
        self.data()
    }
}
