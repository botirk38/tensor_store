# profile

Profiling harness for measuring tensor_store loader performance.

## Usage

```bash
cargo run --bin profile -- <COMMAND> <CASE> [OPTIONS]
```

## Commands

### SafeTensors Profiling

```bash
cargo run --bin profile -- safetensors <CASE> [--fixture <NAME>] [--iterations <N>] [--cold-cache]
```

**Available Cases:**
- `default` - Heuristic default (picks best backend based on cost model)
- `sync` - Synchronous blocking I/O with thread-per-chunk parallelism
- `async` - Tokio async I/O
- `mmap` - Memory-mapped file access
- `io-uring` - Explicit io_uring backend (Linux only)

### ServerlessLLM Profiling

```bash
cargo run --bin profile -- serverlessllm <CASE> [--fixture <NAME>] [--iterations <N>] [--cold-cache]
```

**Available Cases:**
- `default` - Heuristic default
- `sync` - Synchronous load
- `async` - Tokio async load
- `mmap` - Memory-mapped access
- `io-uring` - Explicit io_uring backend (Linux only)

## Options

- `-f, --fixture <NAME>` - Specify a fixture name (e.g., gpt2, qwen-qwen2-0.5b, eleutherai-pythia-1.4b-deduped)
- `-i, --iterations <N>` - Number of iterations to run (default: 1)
- `--cold-cache` - Drop page cache before first iteration via `posix_fadvise(DONTNEED)`

## Examples

Profile with heuristic default:
```bash
cargo run --bin profile -- safetensors default --fixture gpt2
```

Profile sync backend:
```bash
cargo run --bin profile -- safetensors sync --fixture gpt2
```

Profile io_uring backend:
```bash
cargo run --bin profile -- safetensors io-uring --fixture gpt2
```

Profile with perf stat:
```bash
cargo build --release --bin profile
perf stat -e cycles,instructions,cache-references,cache-misses ./target/release/profile safetensors sync --fixture gpt2
```

Profile all fixtures:
```bash
cargo run --bin profile -- safetensors default
```

## Performance Notes

- `sync` is typically fastest for local file access due to intra-shard thread parallelism
- `async` (Tokio) is useful for concurrent I/O scenarios
- `io-uring` is available as an explicit backend but sync is usually faster due to better parallelism
- `mmap` is fastest for repeated access to large files that fit in memory

## Fixture Setup

Test fixtures should be placed in the `fixtures/` directory:
- `fixtures/gpt2/` - GPT-2 model (548MB)
- `fixtures/qwen-qwen2-0.5b/` - Qwen2-0.5B model (988MB)
- `fixtures/eleutherai-pythia-1.4b-deduped/` - Pythia-1.4B (5.6GB, 2 shards)
