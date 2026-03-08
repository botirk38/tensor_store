//! Framework conversion dispatch.

mod torch;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use torch::raw_to_torch_tensor;

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
        _ => Err(PyValueError::new_err(format!(
            "unsupported framework: {}. Supported: torch",
            framework
        ))),
    }
}
