#!/usr/bin/env bash
# Exit 0 iff all six paper-ladder pytest-benchmark JSON files exist and are non-empty.
# Usage: verify_paper_pytest_outputs.sh [results_dir]  (default: <repo>/results/benchmarks)
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
if git -C "${repo_root}" rev-parse --show-toplevel &>/dev/null; then
  repo_root="$(git -C "${repo_root}" rev-parse --show-toplevel)"
fi
dir="${1:-${repo_root}/results/benchmarks}"
# Slugs match benchmarks/hub_model.repo_dir_name for the paper ladder models.
expected=(
  pytest_benchmark_qwen-qwen3-0.6b.json
  pytest_benchmark_huggingfacetb-smollm3-3b.json
  pytest_benchmark_qwen-qwen3-4b.json
  pytest_benchmark_qwen-qwen3-8b.json
  pytest_benchmark_qwen-qwen3-14b.json
  pytest_benchmark_qwen-qwen3-32b.json
)
ok=0
for f in "${expected[@]}"; do
  path="${dir}/${f}"
  if [[ ! -s "${path}" ]]; then
    echo "missing or empty: ${path}" >&2
    ok=1
  fi
done
if [[ "${ok}" -eq 0 ]]; then
  echo "ok: all ${#expected[@]} paper ladder JSON files present under ${dir}"
fi
exit "${ok}"
