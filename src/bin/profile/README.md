# profile

Profiling harness for measuring tensor_store loader performance.

## Usage

```bash
cargo run --bin profile -- <COMMAND> <CASE> [OPTIONS]
```

## Commands

### SafeTensors Profiling

```bash
cargo run --bin profile -- safetensors <CASE> [--fixture <NAME>] [--iterations <N>]
```

**Available Cases:**
- `default` - Measured default policy for the current workload regime
- `sync` - Synchronous blocking I/O with thread-per-chunk parallelism
- `async` - Tokio async I/O
- `mmap` - Memory-mapped file access
- `io-uring` - Explicit io_uring backend (Linux only)

### ServerlessLLM Profiling

```bash
cargo run --bin profile -- serverlessllm <CASE> [--fixture <NAME>] [--iterations <N>]
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

## Cold-Cache Runs

This harness does not perform cache eviction. For cold-cache measurements, prepare the environment manually before running:

```bash
sync
echo 3 > /proc/sys/vm/drop_caches
./target/release/profile safetensors sync --fixture gpt2
```

## Performance Notes

- `sync` remains a strong baseline for smaller whole-file eager loads
- `async` is often competitive for smaller range-heavy `ServerlessLLM` loads
- multi-worker `io-uring` can win on medium and large eager loads; validate with profiling on your target machine
- `mmap` is fastest for repeated access to large files that fit in memory

## Fixture Setup

Test fixtures should be placed in the `fixtures/` directory:
- `fixtures/gpt2/` - GPT-2 model (548MB)
- `fixtures/qwen-qwen2-0.5b/` - Qwen2-0.5B model (988MB)
- `fixtures/eleutherai-pythia-1.4b-deduped/` - Pythia-1.4B (5.6GB, 2 shards)
