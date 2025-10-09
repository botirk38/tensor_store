# tensor_store

High-performance tensor loading library optimized for io_uring on Linux, with fallback support for other platforms.

## Features

- **Zero-copy I/O** with io_uring on Linux (kernel 5.1+)
- **Parallel loading** with batched io_uring operations
- **Fixed buffer support** for kernel-registered buffers (eliminates copy overhead)
- **Jemalloc allocator** for improved memory management
- **Cross-platform** with Tokio fallback for non-Linux systems

## Loading Strategies

### Basic Loading (`load_safetensors`)

Single read operation using io_uring's `read_at`:

```rust
#[tokio_uring::main]
async fn main() -> std::io::Result<()> {
    let data = tensor_store::load_safetensors("model.safetensors").await?;
    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

### Parallel Loading (`load_safetensors_parallel`)

Batches multiple read operations for parallel execution:

```rust
#[tokio_uring::main]
async fn main() -> std::io::Result<()> {
    let data = tensor_store::load_safetensors_parallel("model.safetensors").await?;
    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

Default: 4 chunks. Customize with:

```rust
let data = tensor_store::load_safetensors_parallel_with_chunks("model.safetensors", 8).await?;
```

### Fixed Buffer Loading (`load_safetensors_fixed`)

Uses kernel-registered buffers for zero-copy I/O:

```rust
use tensor_store::FixedBufPool;

#[tokio_uring::main]
async fn main() -> std::io::Result<()> {
    let bufs: Vec<Vec<u8>> = (0..4)
        .map(|_| vec![0u8; 64 * 1024 * 1024])
        .collect();
    let buf_pool = FixedBufPool::new(bufs);
    buf_pool.register()?;
    
    let data = tensor_store::load_safetensors_fixed("model.safetensors", &buf_pool).await?;
    println!("Loaded {} bytes", data.len());
    Ok(())
}
```

## Architecture

```
tensor_store/
├── src/
│   ├── lib.rs              # Public API
│   └── loaders/
│       ├── mod.rs          # Loader module definitions
│       ├── uring.rs        # io_uring implementations (Linux)
│       └── tokio.rs        # Tokio fallback (non-Linux)
├── benches/
│   └── tensor_loading.rs   # Performance benchmarks
└── examples/
    ├── basic_load.rs
    ├── parallel_load.rs
    └── fixed_buffer_load.rs
```

## Performance Optimizations

Based on research from [io_uring ecosystem analysis](../research/02_iouring_ecosystem.md):

- ✅ **File handle reuse**: Single open for parallel reads
- ✅ **Batched operations**: Submit all reads before awaiting
- ✅ **Explicit cleanup**: Proper async file closing
- ✅ **Fixed buffers**: Kernel-registered buffers via `FixedBufPool`
- ⚠️ **IOPOLL**: Requires O_DIRECT (not exposed by tokio-uring) - see [IOPOLL.md](./IOPOLL.md)

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Compares:
- Synchronous `std::fs` loading
- io_uring basic loading
- io_uring parallel loading
- io_uring fixed buffer loading

## Examples

```bash
cargo run --example basic_load
cargo run --example parallel_load
cargo run --example fixed_buffer_load
```

## Requirements

- **Linux**: Kernel 5.1+ for io_uring support
- **Other platforms**: Tokio runtime

## License

See [LICENSE](../LICENSE)
