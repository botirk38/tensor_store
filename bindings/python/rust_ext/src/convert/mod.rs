//! Framework conversion dispatch.

mod tensorflow;
mod torch;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use tensorflow::{raw_to_tensorflow_tensor, tf_tensor_to_raw};
pub use torch::{raw_to_torch_tensor, torch_tensor_to_raw};

pub struct TensorData<'a> {
    pub shape: &'a [usize],
    pub dtype: &'a str,
    pub data: &'a [u8],
}

pub fn convert_tensor<'a>(
    py: Python<'_>,
    framework: &str,
    tensor_data: TensorData<'a>,
    device: &str,
) -> PyResult<PyObject> {
    match framework {
        "torch" => raw_to_torch_tensor(
            py,
            tensor_data.shape,
            tensor_data.dtype,
            tensor_data.data,
            device,
        ),
        "tensorflow" | "tf" => raw_to_tensorflow_tensor(
            py,
            tensor_data.shape,
            tensor_data.dtype,
            tensor_data.data,
            device,
        ),
        _ => Err(PyValueError::new_err(format!(
            "unsupported framework: {}. Supported: torch, tensorflow",
            framework
        ))),
    }
}

pub fn extract_tensor_raw(
    py: Python<'_>,
    framework: &str,
    tensor: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, String, Vec<u8>)> {
    match framework {
        "torch" => torch_tensor_to_raw(py, tensor),
        "tensorflow" | "tf" => tf_tensor_to_raw(py, tensor),
        _ => Err(PyValueError::new_err(format!(
            "unsupported framework: {}. Supported: torch, tensorflow",
            framework
        ))),
    }
}
