# Results

Experimental results from the H100 benchmark suite.

## Structure

- `analysis/` — Profiling notes and bottleneck analysis from early investigation
- `profile/` — Rust-layer benchmark outputs
  - `full_cold_matrix.tsv` — Full cold-cache matrix across formats, fixtures, and backends
  - `anchor_reps.tsv` — Five-repetition cold-cache anchor measurements
  - `safetensors_cold.tsv` — Earlier SafeTensors cold runs
- `python/` — Python benchmark JSON artifacts (pytest-benchmark output)
- `vllm/` — vLLM benchmark results
  - `full_matrix.tsv` — Complete vLLM matrix across all fixtures, loaders, and benchmark kinds
  - `raw/` — Raw vLLM benchmark logs (ignored in git, present on experiment host)

## Reproducing

Rust profiling results were generated on a dedicated H100 Linux server. See `scripts/README.md` for the runner scripts used to produce these artifacts.
