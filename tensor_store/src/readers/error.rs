//! Error types for tensor readers.

use std::fmt;
use std::io;

/// Unified error type for all tensor readers.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReaderError {
    /// I/O error occurred during reading.
    Io(io::Error),

    /// SafeTensors format error.
    SafeTensors(safetensors::SafeTensorError),

    /// ServerlessLLM format error (invalid JSON or structure).
    ServerlessLlm(String),

    /// TensorStore format error.
    TensorStore(String),

    /// Invalid tensor metadata.
    InvalidMetadata(String),
}

impl fmt::Display for ReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReaderError::Io(err) => write!(f, "I/O error: {}", err),
            ReaderError::SafeTensors(err) => write!(f, "SafeTensors error: {}", err),
            ReaderError::ServerlessLlm(msg) => write!(f, "ServerlessLLM format error: {}", msg),
            ReaderError::TensorStore(msg) => write!(f, "TensorStore format error: {}", msg),
            ReaderError::InvalidMetadata(msg) => write!(f, "Invalid tensor metadata: {}", msg),
        }
    }
}

impl std::error::Error for ReaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReaderError::Io(err) => Some(err),
            ReaderError::SafeTensors(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for ReaderError {
    fn from(err: io::Error) -> Self {
        ReaderError::Io(err)
    }
}

impl From<safetensors::SafeTensorError> for ReaderError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        ReaderError::SafeTensors(err)
    }
}

/// Result type alias for reader operations.
pub type ReaderResult<T> = Result<T, ReaderError>;
