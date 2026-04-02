# tensor_store

Adaptive checkpoint loading for large language models, with sync, async, and io_uring backends.

Tensor formats: SafeTensors and ServerlessLLM.

## Quick Start

Prerequisites: Rust toolchain (stable), [uv](https://github.com/astral-sh/uv) for Python scripts.

Set up one fixture model:

```bash
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen3-0.6B
cd ..
```

For a broader benchmark ladder, also download `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, and `Qwen/Qwen3-32B`.

Run demos:

```bash
cargo run --release --bin demo -- safetensors all
cargo run --release --bin demo -- serverlessllm all
```

If no fixture exists, demo commands fail with a "No fixtures found under `fixtures/`" error.

## Minimal Usage

```rust
use tensor_store::safetensors;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = safetensors::Model::load("model_dir").await?;
    println!("Loaded {} tensors", model.tensor_names().len());
    Ok(())
}
```

## Backend Selection

The library exposes an adaptive `default` policy that chooses between:

- **sync**: blocking POSIX reads with dynamic chunking and parallelism
- **async**: Tokio-based async loading with dynamic batching
- **io_uring**: Linux io_uring with multi-worker submission (Linux only)

The heuristic is workload-aware: smaller eager checkpoints favour `sync`, larger multi-shard checkpoints favour `io_uring`, and partitioned ServerlessLLM layouts switch between `async` and `io_uring` depending on size and structure.

## Docs

- [I/O Backends](src/backends/README.md)
- [SafeTensors Support](src/formats/safetensors/README.md)
- [ServerlessLLM Support](src/formats/serverlessllm/README.md)
- [Converter Binary](src/bin/convert/README.md)
- [Demo Binary](src/bin/demo/README.md)
- [Profile Binary](src/bin/profile/README.md)
- [Benchmarks](benches/README.md)
- [Profiling](profiling/README.md)
- [Scripts](scripts/README.md)
- [Python Bindings](bindings/python/README.md)
- [Results](results/README.md)
- [Report](report/README.md)

## Tests

```bash
cargo test --lib --locked
cargo clippy --lib --locked -- -D warnings
```

Python binding tests:

```bash
cd bindings/python
uv sync --group dev --group bench
uv run pytest tests/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
