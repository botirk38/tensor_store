# TensorStore

**Product:** Adaptive checkpoint loading for LLMs (Rust `tensor_store`, Python `tensor_store_py`). Backends: synchronous POSIX I/O, Tokio async, Linux `io_uring`, plus an adaptive default. Formats: SafeTensors, ServerlessLLM-style layouts.

**Paper:** *Load by Design: Adaptive Heuristics for LLM Checkpoint Loading* — sources in [`paper/`](paper/), built from [`paper/main.tex`](paper/main.tex).

This README is the **single entry point** for developers and for **reproducing paper measurements**. Methodological detail (cold cache, measurement layers, variance) is in the PDF appendix on reproducibility; below are **commands and paths** only.

---

## Repository

| | |
|--|--|
| **Public clone** | `https://github.com/botirk38/tensor_store` |
| **Licence** | Apache License, Version 2.0 ([`LICENSE`](LICENSE)) |

Pin software next to any exported numbers:

```bash
git rev-parse HEAD
git status -sb
```

Replication should match **backend ordering and regime behaviour** (SafeTensors crossover; ServerlessLLM without sync leading), not byte-identical timings on different hardware.

---

## Build the paper PDF

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**arXiv packaging** (`00README.json`, tarball layout, plain abstract): [`paper/README.md`](paper/README.md).

---

## Reproduce experiments (paper tables)

**Needs:** Linux (`io_uring` is Linux-only), Rust (see `rust-version` in [`Cargo.toml`](Cargo.toml)), [`uv`](https://docs.astral.sh/uv/) for Python scripts. **GPU** only for vLLM / GPU-backed Python paths. Cold-cache replication assumes you can run `sync` and `echo 3 > /proc/sys/vm/drop_caches` (usually root); use a dedicated machine so cache drops do not affect other workloads.

**Fixtures:** download checkpoints with `scripts/download_models.py` (see [`scripts/README.md`](scripts/README.md) for setup: `cd scripts && uv sync`).

**Rust profiling binary** (single run example):

```bash
cargo build --release --bin profile
cargo run --release --bin profile -- safetensors sync --fixture qwen-qwen3-8b --iterations 1
```

Formats: `safetensors` or `serverlessllm`. Backends: `sync`, `async`, `io-uring`, `default`. Full cold-cache procedure (including cache drops) matches the paper’s Experimental Setup; invoke `profile` after each drop as in that section.

**Benchmark entry point** (from repository root, after fixtures exist):

| Script | Typical output |
|--------|----------------|
| [`scripts/run_benchmarks.sh`](scripts/run_benchmarks.sh) | pytest-benchmark JSON under `results/benchmarks/` (set `TENSOR_STORE_BENCH_MODEL` to a Hugging Face model id; includes SafeTensors, ServerlessLLM, and vLLM pytest suites unless `TENSOR_STORE_BENCH_NO_VLLM=1`) |

Script setup and notes: [`scripts/README.md`](scripts/README.md).

**What counts as replicated:** same **ranking** of backends for the same fixture/format within jitter; SafeTensors crossover between sync-favoured and `io_uring`-favoured regimes at comparable scales; ServerlessLLM without sync at the front of the explicit ranking when methodology matches. Python/vLLM: expect **directional** effects, not identical milliseconds.

---

## Record environment beside results

Paste something like the following next to tables or in a log (no extra tooling required):

```bash
date -u
uname -a
rustc --version
python3 --version
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
( cd bindings/python && uv pip show vllm 2>/dev/null | head -5 )
```

Kernel and CUDA versions affect `io_uring` behaviour and GPU initialisation.

---

## Quick start (development)

```bash
cd scripts
uv sync
uv run python download_models.py Qwen/Qwen3-0.6B
cd ..
```

Demos:

```bash
cargo run --release --bin demo -- safetensors all
cargo run --release --bin demo -- serverlessllm all
```

## Python

```bash
cd bindings/python
uv sync --group dev --group torch
uv run python examples/pytorch.py gpt2 --prompt "Hello"
```

## Tests

```bash
cargo test --lib --locked
cargo clippy --lib --locked -- -D warnings
```

## Python benchmarks

From the repo root, set `TENSOR_STORE_BENCH_MODEL` to a Hugging Face model id, then:

```bash
./scripts/run_benchmarks.sh
```

Default JSON: `results/benchmarks/pytest_benchmark.json`. Details: [`bindings/python/benchmarks/README.md`](bindings/python/benchmarks/README.md).

## Component docs

- [I/O Backends](src/backends/README.md)
- [SafeTensors](src/formats/safetensors/README.md)
- [ServerlessLLM](src/formats/serverlessllm/README.md)
- [Python Bindings](bindings/python/README.md)

## FAQ

- **Name collision:** *TensorStore* in this repository is the checkpoint-loading system described in the paper (Rust crate `tensor_store`, Python `tensor_store_py`). It is **not** Google’s unrelated TensorStore library for N-dimensional arrays.
- **Platform:** The `io_uring` backend is **Linux-only**. Other platforms are out of scope for this codebase.
- **What “replication” means:** Match **backend ordering and regime behaviour** (e.g. SafeTensors crossover; ServerlessLLM without synchronous loading leading) rather than identical millisecond timings on different hardware.
- **Cold cache:** Published cold-cache numbers assume `sync` and `drop_caches` as in the paper; warm-cache behaviour is different by design.
- **vLLM / GPU:** Integration numbers depend on engine settings and GPU; compare against the stock loader when interpreting tables.

## Known-good toolchain (indicative)

Values below mirror the crate and Python package; newer toolchains often work but are not guaranteed without CI coverage.

| Component | Notes |
|-----------|--------|
| **Rust** | `rust-version` in [`Cargo.toml`](Cargo.toml) is **1.92** (see `rustc --version`). |
| **Python** | [`bindings/python/pyproject.toml`](bindings/python/pyproject.toml) requires **Python ≥ 3.12** for the bindings package. |
| **uv** | Used for scripts under `scripts/` and Python workflows (see [uv](https://docs.astral.sh/uv/)). |
| **OS** | Linux for full feature parity (`io_uring`). |

## How to cite

- **CFF (GitHub “Cite this repository”):** [`CITATION.cff`](CITATION.cff)
- **BibTeX (copy-paste):** [`CITATION.bib`](CITATION.bib)

After an arXiv announcement, add the identifier to your `.bib` entry and to any citation metadata you maintain.

## Release tags (paper checkpoints)

To tie a paper revision to a **fixed** tree without new experiments, tag the repository after the manuscript freeze, for example:

```bash
git tag -a v0.1.0-paper -m "Paper revision snapshot"
git push origin v0.1.0-paper
```

Mention that tag (or `git rev-parse HEAD`) next to exported tables.

## Licence

Apache 2.0
