# convert

Converts SafeTensors model files to ServerlessLLM format with partitioned storage.

## Usage

```bash
cargo run --bin convert -- <input_dir> <output_dir> [OPTIONS]
```

### Arguments

- `<input_dir>` - Directory containing one or more `.safetensors` shards
- `<output_dir>` - Directory where converted files will be written

### Options

- `-p, --partitions <COUNT>` - Number of partitions (default: auto from model size)
- `-b, --backend <BACKEND>` - Backend to use for conversion (default, sync, async, io-uring)

### Output

The conversion produces:
- `metadata.json` - Metadata and tensor index
- `tensor.data_0` through `tensor.data_{N-1}` - Partitioned tensor data files

## Examples

Convert with automatic partition count:
```bash
cargo run --bin convert -- ./model_dir ./output
```

Convert with specific partition count:
```bash
cargo run --bin convert -- ./model_dir ./output --partitions 8
```

Convert with explicit backend:
```bash
cargo run --bin convert -- ./model_dir ./output --backend sync
```

## Performance Considerations

### Default partition count

Automatic conversion uses: `max(1, ceil(model_bytes / 512 MiB))`

There is no fixed upper cap. For custom layouts, pass `--partitions` explicitly.

## Output Structure

After conversion, `output_dir` contains:
```
output_dir/
├── metadata.json          # Metadata (dtype, shape, partition info)
├── tensor.data_0          # Partition 0
├── tensor.data_1          # Partition 1
├── ...
└── tensor.data_{N-1}      # Last partition
```

## Platform Support

- **Linux**: sync, async, and io_uring backends available
- **Other platforms**: sync and async backends available

## See Also

- [SafeTensors Format](../../formats/safetensors/README.md)
- [ServerlessLLM Format](../../formats/serverlessllm/README.md)
