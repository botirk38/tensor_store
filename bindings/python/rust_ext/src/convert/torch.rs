//! PyTorch tensor conversion.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Maps tensor_store/safetensors dtypes to PyTorch dtype names.
fn safetensors_dtype_to_torch_name(dtype: &str) -> Option<&'static str> {
    match dtype {
        "F64" | "float64" | "torch.float64" => Some("float64"),
        "F32" | "float32" | "torch.float32" => Some("float32"),
        "F16" | "float16" | "torch.float16" => Some("float16"),
        "BF16" | "bfloat16" | "torch.bfloat16" => Some("bfloat16"),
        "I64" | "int64" | "torch.int64" => Some("int64"),
        "I32" | "int32" | "torch.int32" => Some("int32"),
        "I16" | "int16" | "torch.int16" => Some("int16"),
        "I8" | "int8" | "torch.int8" => Some("int8"),
        "U64" | "uint64" | "torch.uint64" => Some("uint64"),
        "U32" | "uint32" | "torch.uint32" => Some("uint32"),
        "U16" | "uint16" | "torch.uint16" => Some("uint16"),
        "U8" | "uint8" | "torch.uint8" => Some("uint8"),
        "BOOL" | "bool" | "torch.bool" => Some("bool"),
        "F8_E4M3" | "float8_e4m3fn" | "torch.float8_e4m3fn" => Some("float8_e4m3fn"),
        "F8_E5M2" | "float8_e5m2" | "torch.float8_e5m2" => Some("float8_e5m2"),
        "C64" | "complex64" | "torch.complex64" => Some("complex64"),
        _ => None,
    }
}

/// Builds a torch.Tensor from raw tensor data.
pub fn raw_to_torch_tensor(
    py: Python<'_>,
    shape: &[usize],
    dtype: &str,
    data: &[u8],
    device: &str,
) -> PyResult<PyObject> {
    let torch = py.import("torch")?;

    let dtype_name = safetensors_dtype_to_torch_name(dtype).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype for torch: {}", dtype))
    })?;

    let torch_dtype = torch.getattr(dtype_name)?;
    let torch_uint8 = torch.getattr("uint8")?;

    let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let numel: i64 = shape_i64.iter().product();

    let tensor = if numel == 0 {
        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", torch_dtype)?;
        torch.call_method("zeros", (shape_py,), Some(&kwargs))?
    } else {
        let data_bytes = PyBytes::new(py, data);
        let buf = data_bytes.as_any();

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("buffer", buf)?;
        kwargs.set_item("dtype", torch_uint8)?;

        let arr = torch.call_method("frombuffer", (), Some(&kwargs))?;
        let view_kwargs = pyo3::types::PyDict::new(py);
        view_kwargs.set_item("dtype", torch_dtype)?;

        let mut tensor = arr.call_method("view", (), Some(&view_kwargs))?;

        let sys = py.import("sys")?;
        let byteorder: String = sys.getattr("byteorder")?.extract()?;
        if byteorder == "big" {
            let numpy_arr = tensor.call_method0("numpy")?;
            let swap_kwargs = pyo3::types::PyDict::new(py);
            swap_kwargs.set_item("inplace", false)?;
            let swapped = numpy_arr.call_method("byteswap", (), Some(&swap_kwargs))?;
            tensor = torch.call_method1("from_numpy", (swapped,))?;
            let view_kwargs = pyo3::types::PyDict::new(py);
            view_kwargs.set_item("dtype", torch.getattr(dtype_name)?)?;
            tensor = tensor.call_method("view", (), Some(&view_kwargs))?;
        }

        let shape_py: PyObject = shape_i64.into_pyobject(py)?.unbind();
        tensor.call_method1("reshape", (shape_py,))?
    };

    let result = if device != "cpu" && device != "cpu:0" {
        let kwargs = pyo3::types::PyDict::new(py);
        tensor.call_method("to", (device,), Some(&kwargs))?
    } else {
        tensor
    };

    Ok(result.into())
}
