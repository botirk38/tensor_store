# convert

Converts SafeTensors model files to ServerlessLLM format with partitioned storage.

## Overview

This binary converts a directory of SafeTensors shards into the ServerlessLLM format, which partitions tensor data across multiple files for efficient loading and parallel access.

## Usage

```bash
cargo run --bin convert -- <input_dir> <output_dir> [partition_count]
```

### Arguments

- `<input_dir>` - Directory containing one or more `.safetensors` shards
- `<output_dir>` - Directory where converted files will be written
- `[partition_count]` - (Optional) Number of partitions to create. Defaults to `tensor_store::recommended_partition_count` (`ceil(model_bytes / 512 MiB)`, minimum 1)

### Output

The conversion produces:
- `tensor_index.json` - Index file mapping tensor names to partition locations
- `tensor.data_0` through `tensor.data_{N-1}` - Partitioned tensor data files

## Examples

Convert with automatic partition count:
```bash
cargo run --bin convert -- ./model_dir ./output
```

Convert with specific partition count:
```bash
cargo run --bin convert -- ./model_dir ./output 8
```

## Performance Considerations

### Default partition count

Automatic conversion uses the shared ServerlessLLM helper:

`max(1, ceil(model_bytes / 512 MiB))`

There is no fixed upper cap. For custom layouts, pass an explicit `[partition_count]`.

### Conversion Speed

Typical conversion speeds (on NVMe SSD):
- Small models (<1GB): ~2-5 seconds
- Medium models (1-10GB): ~10-30 seconds
- Large models (>10GB): ~1-2 minutes per 10GB

## Troubleshooting

### "Failed to create output directory"
Ensure the parent directory exists and you have write permissions:
```bash
mkdir -p output
 cargo run --bin convert -- ./model_dir ./output 8
```

### "Out of memory"
For very large models, ensure sufficient RAM. The converter loads shard data in chunks but still needs memory for partition buffers.

### "Partition files are uneven"
This is normal - tensors are distributed round-robin to partitions. Some variation in partition sizes is expected.

## Output Structure

After conversion, `output_dir` contains:
```
output_dir/
├── tensor_index.json          # Metadata (dtype, shape, partition info)
├── tensor.data_0              # Partition 0 (first 1/N of tensors)
├── tensor.data_1              # Partition 1
├── ...
└── tensor.data_{N-1}          # Last partition
```

## Platform Support

- **Linux**: Uses io_uring for high-performance async I/O
- **Other platforms**: Uses tokio runtime for async operations

## See Also

- [SafeTensors Module](../../safetensors/README.md)
- [ServerlessLLM Module](../../serverlessllm/README.md)
- [Converters Architecture](../../converters/README.md)
