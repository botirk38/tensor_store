//! Owned byte buffer types for backend I/O.
//!
//! This module provides `OwnedBytes`, a zero-copy owned byte container
//! that can be backed by different storage types (pooled, aligned, shared, mmap).

use std::sync::Arc;

pub use zeropool::PooledBuffer;

use super::mmap::Mmap;
#[cfg(target_os = "linux")]
use super::odirect::AlignedBuffer;

/// Owned byte buffer that can be backed by different storage types.
/// Supports zero-copy access patterns while preserving ownership semantics.
pub enum OwnedBytes {
    Pooled(PooledBuffer),
    #[cfg(target_os = "linux")]
    Aligned(AlignedBuffer),
    Shared(Arc<[u8]>),
    Mmap(Mmap),
}

impl OwnedBytes {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Pooled(b) => b.len(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.len(),
            Self::Shared(b) => b.len(),
            Self::Mmap(b) => b.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to `Vec<u8>`. Copies for aligned and mmap-backed storage
    /// to avoid UB from mismatched allocator layouts.
    pub fn into_vec(self) -> Vec<u8> {
        match self {
            Self::Pooled(b) => b.into_inner(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice().to_vec(),
            Self::Shared(b) => b.to_vec(),
            Self::Mmap(b) => b.as_slice().to_vec(),
        }
    }

    /// Convert to `Arc<[u8]>`. Copies for aligned and mmap-backed storage
    /// to avoid UB from mismatched allocator layouts.
    pub fn into_shared(self) -> Arc<[u8]> {
        match self {
            Self::Pooled(b) => b.into_inner().into(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice().into(),
            Self::Shared(b) => b,
            Self::Mmap(b) => Arc::from(b.as_slice()),
        }
    }
}

impl AsRef<[u8]> for OwnedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Pooled(b) => b.as_ref(),
            #[cfg(target_os = "linux")]
            Self::Aligned(b) => b.as_slice(),
            Self::Shared(b) => b.as_ref(),
            Self::Mmap(b) => b.as_slice(),
        }
    }
}

impl std::ops::Deref for OwnedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl std::fmt::Debug for OwnedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedBytes")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
