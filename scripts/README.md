# Scripts

## Entry points

- **`download_models.py`** — downloads SafeTensors fixtures from Hugging Face (optional ServerlessLLM conversion).
- **`run_benchmarks.sh`** — from the **repository root**, runs all pytest benchmarks under `bindings/python/benchmarks/` via `uv`, builds the extension with `maturin`, and writes pytest-benchmark JSON (default `results/benchmarks/pytest_benchmark.json`). Requires **`TENSOR_STORE_BENCH_MODEL`** (Hugging Face model id).

## Setup

```bash
cd scripts
uv sync
```

## Typical usage

Download a fixture:

```bash
uv run python download_models.py Qwen/Qwen3-8B --verify
```

Run Python benchmarks (from repo root):

```bash
export TENSOR_STORE_BENCH_MODEL=Qwen/Qwen3-8B
./scripts/run_benchmarks.sh
```

## Notes

- Rust-layer timings use the `profile` binary (`cargo build --release --bin profile`); see the repository root `README.md` for invocation and cold-cache procedure.
- Archived TSV/JSON under `results/h100/` were produced on a dedicated experiment host; replicate **ordering and regime behaviour**, not byte-identical paths.
