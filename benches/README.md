# Benchmarks

Performance benchmark suite for tensor_store using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Overview

The benchmark suite compares different I/O backends and loading strategies for SafeTensors and ServerlessLLM formats:

- **SafeTensors benchmarks** (`benches/safetensors.rs`)
- **ServerlessLLM benchmarks** (`benches/serverlessllm.rs`)

## Running Benchmarks

### Run all benchmarks

```bash
cargo bench
```

### Run specific format benchmarks

```bash
# SafeTensors only
cargo bench --bench safetensors

# ServerlessLLM only
cargo bench --bench serverlessllm
```

### Run specific backend benchmarks

```bash
# Run only sync benchmarks
cargo bench -- sync

# Run only mmap benchmarks
cargo bench -- mmap

# Run only async benchmarks (Linux: io_uring, non-Linux: tokio)
cargo bench -- async
```

## Benchmark Suite

Each format has benchmarks per backend (`safetensors_*`, `serverlessllm_*`):

### SafeTensors (`benches/safetensors.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `safetensors_sync` | `load/{model}` | All | Synchronous loading |
| `safetensors_mmap` | `load/{model}` | All | Memory-mapped loading |
| `safetensors_async` | `load/{model}` | All | Async loading (io_uring on Linux, tokio elsewhere) |
| `safetensors_async_parallel` | `load/{model}` | Linux only | Parallel async loading |

### ServerlessLLM (`benches/serverlessllm.rs`)

| Group | Benchmark | Platform | Description |
|-------|-----------|----------|-------------|
| `serverlessllm_sync` | `load/{model}` | All | Synchronous loading |
| `serverlessllm_mmap` | `load/{model}` | All | Memory-mapped loading |
| `serverlessllm_async` | `load/{model}` | All | Async loading (io_uring on Linux, tokio elsewhere) |

## Test Fixtures

Benchmarks automatically discover model fixtures in the `fixtures/` directory:

### SafeTensors Format

```
fixtures/
└── {model-name}/
    ├── model.safetensors      # SafeTensors file
    └── README.md              # Model metadata
```

### ServerlessLLM Format

```
fixtures/
└── {model-name}/
    ├── model_serverlessllm/
    │   ├── tensor_index.json  # Metadata and tensor index
    │   ├── tensor.data_0      # Partitioned data files
    │   ├── tensor.data_1
    │   └── ...
    └── README.md
```

## Downloading Test Fixtures

Use the provided Python script to download models from HuggingFace:

```bash
cd scripts
uv run python download_models.py Qwen/Qwen2-0.5B --convert --verify
```

See [scripts/README.md](../scripts/README.md) for details.

## Understanding Results

### Criterion Output

Criterion provides statistical analysis including:
- Mean execution time
- Standard deviation
- Comparison with previous runs
- Outlier detection

Example output:
```
safetensors_sync/load/qwen-qwen2-0.5b
                        time:   [52.341 ms 52.523 ms 52.728 ms]
                        change: [-3.2415% -2.5741% -1.9251%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### Interpreting Performance

**Cold vs Warm Cache**: First run may be slower due to cold file system cache. Criterion runs multiple iterations with warmup to measure warm cache performance.

**mmap Benchmarks**: The `touch_pages()` function ensures mmap benchmarks trigger page faults for realistic measurement. Without this, mmap would appear artificially fast by only mapping memory without actually reading data.

**Parallel Benchmarks**: Parallel loading with N chunks typically shows benefits for large files (>100MB) on systems with multiple cores and fast storage.

## Platform-Specific Benchmarks

### Linux (io_uring)

On Linux with kernel 5.1+, benchmarks use the io_uring backend:

```bash
# Run io_uring specific benchmarks
cargo bench -- io_uring
```

Features tested:
- Zero-copy I/O with io_uring
- Parallel batched operations
- Fixed buffer pools

### Non-Linux (Tokio)

On macOS and Windows, benchmarks use Tokio's async I/O:

```bash
# Run tokio specific benchmarks
cargo bench -- tokio
```

## Profiling Benchmarks

### Flamegraph Integration

Generate flamegraphs for specific benchmarks:

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile SafeTensors sync loading
cargo flamegraph --bench safetensors -- safetensors_sync

# Profile ServerlessLLM async loading (Linux)
cargo flamegraph --bench serverlessllm -- serverlessllm_async

# Profile with sudo for perf access (Linux)
sudo cargo flamegraph --bench safetensors -- safetensors_async
```

See [profiling/README.md](../profiling/README.md) for detailed profiling guide.

### Using the profiling binaries

For more control, use the dedicated profiling binaries:

```bash
# SafeTensors profiling
cargo run --bin safetensors_reader --release -- fixtures/model/model.safetensors

# ServerlessLLM profiling
cargo run --bin serverlessllm_reader --release -- fixtures/model/model_serverlessllm
```

## Benchmark Methodology

### File Discovery

Benchmarks automatically discover fixtures in `fixtures/` directory:
- Scans all subdirectories
- For SafeTensors: looks for `model.safetensors`
- For ServerlessLLM: looks for `model_serverlessllm/` directory
- Sorts by name for consistent ordering

### Measurement Strategy

Each benchmark:
1. Uses `criterion::black_box()` to prevent compiler optimizations
2. Measures tensor count and data size
3. For mmap: touches pages to trigger real I/O
4. Runs multiple iterations for statistical significance

### Performance Considerations

- **warmup**: Criterion performs warmup runs to populate file system cache
- **Outliers**: Criterion detects and reports statistical outliers
- **Noise**: System noise is measured and factored into confidence intervals

## Reference Results

Example results from development machine (Linux 6.17.0, 16 cores):

**SafeTensors (523MB file)**:
- `safetensors_async_parallel`: ~52ms
- `safetensors_async`: ~55ms
- `safetensors_sync`: ~58ms
- `safetensors_mmap`: ~65ms (with page touches)

**Key findings**:
- Parallel async: ~5% faster than sync baseline
- Zero-copy optimizations: 50% faster than naive parallel
- Buffer pooling: 70% faster than non-pooled

## Troubleshooting

### "No benchmarks found"

Make sure test fixtures exist:
```bash
ls fixtures/*/model.safetensors
ls fixtures/*/model_serverlessllm/
```

### io_uring benchmarks not running

Check kernel version (need 5.1+):
```bash
uname -r
```

### Inconsistent results

- Close other applications to reduce system noise
- Run with `--sample-size 50` for more samples
- Check CPU frequency scaling settings
- Disable CPU turbo boost for consistency

### Permission errors with flamegraph

```bash
# Add perf_event_paranoid setting
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Or run with sudo
sudo -E cargo flamegraph --bench safetensors
```

## Contributing Performance Data

When reporting performance results:
1. Include system specs (CPU, RAM, storage type)
2. Include kernel version (Linux) or OS version
3. Specify fixture size and model name
4. Run with `--save-baseline` to track over time

```bash
# Save baseline for tracking
cargo bench --save-baseline my-machine

# Compare against baseline
cargo bench --baseline my-machine
```
