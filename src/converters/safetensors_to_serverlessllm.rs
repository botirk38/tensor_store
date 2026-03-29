//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! This module converts a directory of `*.safetensors` shards into a single ServerlessLLM
//! artifact. Input shards are discovered lexicographically, tensor names are de-duplicated across
//! shards, and tensors are assigned to partitions round-robin in discovery order.

use crate::backends;
use crate::formats::error::{WriterError, WriterResult};
use crate::formats::safetensors::{Dtype, Model};
use crate::formats::serverlessllm::{write_index, write_partition, writer::TensorWriteEntry};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Convert a directory of `SafeTensors` shards to `ServerlessLLM` format.
///
/// The directory must contain one or more `*.safetensors` files. If a safetensors index JSON is
/// present, it is validated against the discovered shard set. The conversion is deterministic:
/// shards are processed lexicographically and tensors are assigned round-robin across partitions.
#[inline]
pub async fn convert_safetensors_to_serverlessllm(
    input_dir: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }

    let input_dir = Path::new(input_dir);
    let output_dir = Path::new(output_dir);

    let shard_paths = discover_safetensors_shards(input_dir)?;
    validate_index_manifest(input_dir, &shard_paths)?;

    let mut blobs = Vec::new();
    let mut seen_names = BTreeSet::new();

    for shard_path in &shard_paths {
        let shard = backends::async_backend()
            .load(shard_path)
            .await
            .map_err(WriterError::from)?;
        let model = Model::from_bytes(shard)
            .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;
        let tensors = model.tensors();
        for name in model.tensor_names() {
            let tensor_name = name.as_ref().to_owned();
            if !seen_names.insert(tensor_name.clone()) {
                return Err(WriterError::InvalidInput(format!(
                    "duplicate tensor name across shards: {tensor_name}"
                )));
            }

            let view = tensors
                .tensor(name.as_ref())
                .map_err(WriterError::SafeTensors)?;
            let data = view.data().to_vec();
            let shape: Vec<usize> = view.shape().to_vec();
            let stride = calculate_contiguous_stride(&shape);
            let dtype = dtype_to_serverlessllm(view.dtype())?.to_owned();
            blobs.push(TensorBlob {
                name: tensor_name,
                data,
                shape,
                stride,
                dtype,
            });
        }
    }

    if blobs.is_empty() {
        return Err(WriterError::InvalidInput(
            "no tensors found in input directory".to_owned(),
        ));
    }

    let mut partitions: Vec<Vec<u8>> = vec![Vec::new(); partition_count];
    let mut index: HashMap<String, TensorWriteEntry> = HashMap::with_capacity(blobs.len());

    for (i, blob) in blobs.into_iter().enumerate() {
        let partition_id = i % partition_count;
        let partition = partitions
            .get_mut(partition_id)
            .ok_or_else(|| WriterError::InvalidInput("partition index out of bounds".to_owned()))?;
        let offset = partition.len() as u64;
        let size = blob.data.len() as u64;

        partition.extend_from_slice(&blob.data);

        index.insert(
            blob.name,
            TensorWriteEntry {
                offset,
                size,
                shape: blob.shape,
                stride: blob.stride,
                dtype: blob.dtype,
                partition_id,
            },
        );
    }

    tokio::fs::create_dir_all(output_dir).await?;

    let index_path = output_dir.join("tensor_index.json");
    write_index(&index_path, &index).await?;

    let write_futures: Vec<_> = partitions
        .into_iter()
        .enumerate()
        .map(|(id, data)| {
            let part_path = output_dir.join(format!("tensor.data_{id}"));
            async move { write_partition(&part_path, data).await }
        })
        .collect();

    futures::future::try_join_all(write_futures).await?;

    Ok(())
}

fn discover_safetensors_shards(input_dir: &Path) -> WriterResult<Vec<PathBuf>> {
    if !input_dir.is_dir() {
        return Err(WriterError::InvalidInput(format!(
            "input path is not a directory: {}",
            input_dir.display()
        )));
    }

    let mut shards = Vec::new();
    for entry in std::fs::read_dir(input_dir).map_err(WriterError::from)? {
        let entry = entry.map_err(WriterError::from)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.ends_with(".safetensors") {
            shards.push(path);
        }
    }

    shards.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if shards.is_empty() {
        return Err(WriterError::InvalidInput(format!(
            "no .safetensors files found in {}",
            input_dir.display()
        )));
    }

    Ok(shards)
}

fn validate_index_manifest(input_dir: &Path, shard_paths: &[PathBuf]) -> WriterResult<()> {
    let mut index_files = Vec::new();
    for entry in std::fs::read_dir(input_dir).map_err(WriterError::from)? {
        let entry = entry.map_err(WriterError::from)?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".safetensors.index.json"))
        {
            index_files.push(path);
        }
    }

    if index_files.is_empty() {
        return Ok(());
    }

    let shard_names: BTreeSet<String> = shard_paths
        .iter()
        .filter_map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|s| s.to_owned())
        })
        .collect();

    for index_path in index_files {
        let bytes = std::fs::read(&index_path).map_err(WriterError::from)?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            WriterError::InvalidInput(format!(
                "failed to parse index manifest {}: {e}",
                index_path.display()
            ))
        })?;
        let weight_map = json
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                WriterError::InvalidInput(format!(
                    "index manifest {} is missing weight_map",
                    index_path.display()
                ))
            })?;

        let referenced: BTreeSet<String> = weight_map
            .values()
            .filter_map(|value| value.as_str().map(|s| s.to_owned()))
            .collect();

        if referenced != shard_names {
            return Err(WriterError::InvalidInput(format!(
                "index manifest {} does not match discovered shard set",
                index_path.display()
            )));
        }
    }

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
        Dtype::F4
        | Dtype::F6_E2M3
        | Dtype::F6_E3M2
        | Dtype::F8_E5M2
        | Dtype::F8_E4M3
        | Dtype::F8_E8M0
        | Dtype::C64
        | _ => {
            return Err(WriterError::InvalidInput(format!(
                "unsupported dtype: {dtype:?}",
            )));
        }
    };

    Ok(mapped)
}

fn calculate_contiguous_stride(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut stride = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        let next_i = i + 1;
        let next_stride = stride.get(next_i).copied().unwrap_or(1);
        let next_shape = shape.get(next_i).copied().unwrap_or(1);
        if let Some(s) = stride.get_mut(i) {
            *s = next_stride.saturating_mul(next_shape);
        }
    }
    stride
}

#[derive(Debug)]
struct TensorBlob {
    name: String,
    data: Vec<u8>,
    shape: Vec<usize>,
    stride: Vec<usize>,
    dtype: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::safetensors::{Dtype, TensorView};
    use crate::formats::serverlessllm::Index;
    use safetensors::serialize;
    use std::fs;
    use tempfile::TempDir;

    fn write_shard(dir: &Path, name: &str, tensors: Vec<(&str, TensorView<'_>)>) -> PathBuf {
        let path = dir.join(name);
        let bytes = serialize(tensors, None).expect("serialize tensors");
        fs::write(&path, bytes).expect("write safetensors file");
        path
    }

    #[test]
    fn test_dtype_to_serverlessllm_supported() {
        assert_eq!(dtype_to_serverlessllm(Dtype::F32).unwrap(), "torch.float32");
        assert_eq!(dtype_to_serverlessllm(Dtype::F16).unwrap(), "torch.float16");
        assert_eq!(
            dtype_to_serverlessllm(Dtype::BF16).unwrap(),
            "torch.bfloat16"
        );
        assert_eq!(dtype_to_serverlessllm(Dtype::I32).unwrap(), "torch.int32");
        assert_eq!(dtype_to_serverlessllm(Dtype::I8).unwrap(), "torch.int8");
        assert_eq!(dtype_to_serverlessllm(Dtype::U8).unwrap(), "torch.uint8");
        assert_eq!(dtype_to_serverlessllm(Dtype::BOOL).unwrap(), "torch.bool");
    }

    #[test]
    fn test_calculate_contiguous_stride_2d() {
        assert_eq!(calculate_contiguous_stride(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_discover_shards_empty_dir_fails() {
        let dir = TempDir::new().unwrap();
        let err = discover_safetensors_shards(dir.path()).unwrap_err();
        assert!(matches!(err, WriterError::InvalidInput(_)));
    }

    #[test]
    fn test_convert_multi_shard_roundtrip() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("output");

        let t1 = TensorView::new(Dtype::U8, vec![4], &[1u8, 2, 3, 4]).unwrap();
        let t2 = TensorView::new(Dtype::U8, vec![6], &[5u8, 6, 7, 8, 9, 10]).unwrap();
        let t3 = TensorView::new(Dtype::U8, vec![2], &[11u8, 12]).unwrap();

        write_shard(
            dir.path(),
            "model-00002-of-00002.safetensors",
            vec![("bias", t2)],
        );
        write_shard(
            dir.path(),
            "model-00001-of-00002.safetensors",
            vec![("weight", t1), ("ln", t3)],
        );

        crate::test_utils::run_async(async {
            convert_safetensors_to_serverlessllm(
                dir.path().to_str().unwrap(),
                output.to_str().unwrap(),
                2,
            )
            .await
            .expect("conversion failed");
        });

        assert!(output.join("tensor_index.json").exists());
        assert!(output.join("tensor.data_0").exists());
        assert!(output.join("tensor.data_1").exists());

        let index = Index::load_sync(output.join("tensor_index.json")).expect("parse index");
        assert_eq!(index.len(), 3);
        assert_eq!(index.get("ln").unwrap().partition_id, 0);
        assert_eq!(index.get("weight").unwrap().partition_id, 1);
        assert_eq!(index.get("bias").unwrap().partition_id, 0);
    }

    #[test]
    fn test_convert_duplicate_tensor_names_fail() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("output");

        let t1 = TensorView::new(Dtype::U8, vec![4], &[1u8, 2, 3, 4]).unwrap();
        let t2 = TensorView::new(Dtype::U8, vec![6], &[5u8, 6, 7, 8, 9, 10]).unwrap();

        write_shard(dir.path(), "a.safetensors", vec![("dup", t1)]);
        write_shard(dir.path(), "b.safetensors", vec![("dup", t2)]);

        crate::test_utils::run_async(async {
            let err = convert_safetensors_to_serverlessllm(
                dir.path().to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await
            .unwrap_err();
            assert!(matches!(err, WriterError::InvalidInput(_)));
        });
    }
}
