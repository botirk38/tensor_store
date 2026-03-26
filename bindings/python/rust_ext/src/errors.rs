//! Python exception mapping for tensor_store errors.

use pyo3::prelude::*;
use tensor_store::ReaderError;

pub fn map_reader_error(e: ReaderError) -> PyErr {
    let msg = format!("tensor_store error: {e}");
    pyo3::exceptions::PyValueError::new_err(msg)
}

pub fn tensor_not_found(name: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("tensor not found: {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_map_reader_error() {
        let io_err = ReaderError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        let py_err = map_reader_error(io_err);
        let msg = format!("{py_err}");
        assert!(msg.contains("tensor_store error"));
        assert!(msg.contains("file not found"));
    }
}
