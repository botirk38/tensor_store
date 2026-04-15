#!/usr/bin/env bash
# Run pytest benchmarks under bindings/python/benchmarks via uv.
# Writes pytest-benchmark JSON (--benchmark-json).
#
# Required:
#   TENSOR_STORE_BENCH_MODEL   HuggingFace model id (pytest --model-id)
#
# Optional:
#   TENSOR_STORE_BENCH_JSON         Output JSON path (default: <repo>/results/benchmarks/pytest_benchmark.json)
#   TENSOR_STORE_SKIP_MAURIN=1      Skip maturin develop --release
#   TENSOR_STORE_BENCH_NO_VLLM=1    Omit bench_vllm.py; uv sync uses --group dev --group torch (no vLLM stack)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if git -C "${repo_root}" rev-parse --show-toplevel &>/dev/null; then
  repo_root="$(git -C "${repo_root}" rev-parse --show-toplevel)"
fi

if [[ -z "${TENSOR_STORE_BENCH_MODEL:-}" ]]; then
  echo "error: set TENSOR_STORE_BENCH_MODEL to a Hugging Face model id (example: gpt2)" >&2
  exit 1
fi

py_dir="${repo_root}/bindings/python"
bench_json="${TENSOR_STORE_BENCH_JSON:-${repo_root}/results/benchmarks/pytest_benchmark.json}"
mkdir -p "$(dirname "${bench_json}")"

if [[ "${TENSOR_STORE_BENCH_NO_VLLM:-}" == "1" ]]; then
  echo "==> uv sync (dev + torch; skipping vLLM dependency group)"
  uv --directory "${py_dir}" sync --group dev --group torch
else
  echo "==> uv sync (dev + vllm)"
  uv --directory "${py_dir}" sync --group dev --group vllm
fi

if [[ "${TENSOR_STORE_SKIP_MAURIN:-}" != "1" ]]; then
  echo "==> maturin develop --release"
  uv --directory "${py_dir}" run maturin develop --release
else
  echo "==> skipping maturin (TENSOR_STORE_SKIP_MAURIN=1)"
fi

tests=(benchmarks/bench_safetensors.py benchmarks/bench_serverlessllm.py)
if [[ "${TENSOR_STORE_BENCH_NO_VLLM:-}" != "1" ]]; then
  tests+=(benchmarks/bench_vllm.py)
else
  echo "==> skipping bench_vllm.py (TENSOR_STORE_BENCH_NO_VLLM=1)"
fi

echo "==> pytest -> ${bench_json}"
uv --directory "${py_dir}" run pytest "${tests[@]}" -v \
  --model-id "${TENSOR_STORE_BENCH_MODEL}" \
  --benchmark-json="${bench_json}"

echo "Done: ${bench_json}"
