# tensor_store

High-performance Rust tensor I/O for LLM model files, using `io_uring` on Linux and Tokio elsewhere.

Tensor formats: SafeTensors and ServerlessLLM.

## Quick Start

Set up one fixture model:

```bash
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen2-0.5B
cd ..
```

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
    let tensors = safetensors::load("model.safetensors").await?;
    println!("Loaded {} tensors", tensors.names().len());
    Ok(())
}
```

## Docs

- [I/O Backends](src/backends/README.md)
- [SafeTensors Support](src/safetensors/README.md)
- [ServerlessLLM Support](src/serverlessllm/README.md)
- [Converter Binary](src/bin/convert/README.md)
- [Demo Binary](src/bin/demo/README.md)
- [Profile Binary](src/bin/profile/README.md)
- [Benchmarks](benches/README.md)
- [Profiling](profiling/README.md)
- [Scripts](scripts/README.md)
- [Python Bindings](bindings/python/README.md)
- [Changelog](CHANGELOG.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
