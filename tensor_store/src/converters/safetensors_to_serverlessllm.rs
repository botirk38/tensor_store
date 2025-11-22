//! SafeTensors to ServerlessLLM conversion.
//!
//! This module provides functionality to convert SafeTensors format
//! checkpoints to ServerlessLLM format with configurable partitioning.
//!
//! # Conversion Process
//!
//! 1. Parse SafeTensors metadata using readers
//! 2. Convert dtypes and extract tensor data
//! 3. Partition tensors across multiple files
//! 4. Write ServerlessLLM format using writers
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::converters::safetensors_to_serverlessllm;
//!
//! // Convert with 8 partitions
//! convert_safetensors_to_serverlessllm(
//!     "model.safetensors",
//!     "output_dir/",
//!     8,
//! ).await?;
//! ```

use crate::readers::safetensors::Dtype;
use crate::writers::error::{WriterError, WriterResult};
use crate::writers::serverlessllm::{ServerlessLlmWriter, TensorEntry};
use std::collections::HashMap;
use std::path::Path;

/// Convert SafeTensors to ServerlessLLM format
pub async fn convert_safetensors_to_serverlessllm(
    input_path: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_string(),
        ));
    }

    // Use parallel loading for better I/O performance
    let owned = crate::readers::safetensors::load_parallel(input_path, 4)
        .await
        .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;
    let tensors = owned.tensors();

    let mut blobs = Vec::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).map_err(WriterError::SafeTensors)?;
        let data = view.data().to_vec();
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
        let stride = calculate_contiguous_stride(&shape);
        let dtype = dtype_to_serverlessllm(view.dtype())?.to_string();
        blobs.push(TensorBlob {
            name: name.to_string(),
            data,
            shape,
            stride,
            dtype,
        });
    }

    // Largest-first round-robin for better balance.
    blobs.sort_by(|a, b| b.data.len().cmp(&a.data.len()));

    let mut partitions: Vec<Vec<u8>> = vec![Vec::new(); partition_count];
    let mut index: HashMap<String, TensorEntry> = HashMap::with_capacity(blobs.len());

    for (i, blob) in blobs.into_iter().enumerate() {
        let partition_id = i % partition_count;
        let partition = &mut partitions[partition_id];
        let offset = partition.len() as u64;
        let size = blob.data.len() as u64;

        partition.extend_from_slice(&blob.data);

        index.insert(
            blob.name,
            TensorEntry {
                offset,
                size,
                shape: blob.shape,
                stride: blob.stride,
                dtype: blob.dtype,
                partition_id,
            },
        );
    }

    let writer = ServerlessLlmWriter::new();

    let out_dir = Path::new(output_dir);
    tokio::fs::create_dir_all(out_dir).await?;

    let index_path = out_dir.join("tensor_index.json");
    writer.write_index(&index_path, &index).await?;

    // Parallelize partition writing using concurrent futures
    // Uses platform-appropriate backend (io_uring on Linux, async_io elsewhere)
    let write_futures: Vec<_> = partitions
        .into_iter()
        .enumerate()
        .map(|(id, data)| {
            let part_path = out_dir.join(format!("tensor.data_{}", id));
            async move { writer.write_partition(&part_path, &data).await }
        })
        .collect();

    // Execute all writes concurrently
    futures::future::try_join_all(write_futures).await?;

    Ok(())
}

fn dtype_to_serverlessllm(dtype: Dtype) -> WriterResult<&'static str> {
    let mapped = match dtype {
        Dtype::F32 => "torch.float32",
        Dtype::F16 => "torch.float16",
        Dtype::BF16 => "torch.bfloat16",
        Dtype::F64 => "torch.float64",
        Dtype::I32 => "torch.int32",
        Dtype::I16 => "torch.int16",
        Dtype::I8 => "torch.int8",
        Dtype::I64 => "torch.int64",
        Dtype::U32 => "torch.uint32",
        Dtype::U16 => "torch.uint16",
        Dtype::U8 => "torch.uint8",
        Dtype::U64 => "torch.uint64",
        Dtype::BOOL => "torch.bool",
        other => {
            return Err(WriterError::InvalidInput(format!(
                "unsupported dtype: {:?}",
                other
            )));
        }
    };

    Ok(mapped)
}

fn calculate_contiguous_stride(shape: &[i64]) -> Vec<i64> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut stride = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

#[derive(Debug)]
struct TensorBlob {
    name: String,
    data: Vec<u8>,
    shape: Vec<i64>,
    stride: Vec<i64>,
    dtype: String,
}
