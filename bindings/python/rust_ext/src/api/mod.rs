//! Python API surface: handles and functions, organized by format.

mod convert;
mod safetensors;
mod serverlessllm;

use std::future::Future;

pub use convert::convert_safetensors_to_serverlessllm;
pub use safetensors::{
    load_safetensors, load_safetensors_async, load_safetensors_sync, open_safetensors,
    save_safetensors, save_safetensors_bytes, SafeTensorsHandlePy,
};
pub use serverlessllm::{
    load_serverlessllm, load_serverlessllm_async, load_serverlessllm_sync, open_serverlessllm,
    ServerlessLLMHandlePy,
};

fn run_async<T>(
    future: impl Future<Output = tensor_store::ReaderResult<T>>,
) -> tensor_store::ReaderResult<T> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            tensor_store::ReaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?
        .block_on(future)
}
