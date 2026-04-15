# Scripts

## Entry point

- **`run_benchmarks.sh`** — from the **repository root**, runs all pytest benchmarks under `bindings/python/benchmarks/` via `uv`, builds the extension with `maturin`, and writes pytest-benchmark JSON (default `results/benchmarks/pytest_benchmark.json`). Requires **`TENSOR_STORE_BENCH_MODEL`** (Hugging Face model id).

## Typical usage

Run Python benchmarks:

```bash
export TENSOR_STORE_BENCH_MODEL=Qwen/Qwen3-8B
./scripts/run_benchmarks.sh
```

Python dependencies: `cd bindings/python && uv sync` (see [`bindings/python/README.md`](../bindings/python/README.md)).

## Notes

- Rust-layer timings use the `profile` binary with `--model-id` or `--fixture`; see the repository root [`README.md`](../README.md).
- Archived TSV/JSON under `results/h100/` were produced on a dedicated experiment host; replicate **ordering and regime behaviour**, not byte-identical paths.
