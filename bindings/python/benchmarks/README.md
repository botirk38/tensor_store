# Tensor Store Benchmarks

Benchmarks live under `bindings/python/benchmarks` and use `pytest-benchmark`.

Run from `bindings/python` with `uv run`:

```bash
uv run pytest benchmarks/bench_safetensors.py -v
uv run pytest benchmarks/bench_serverlessllm.py -v
uv run pytest benchmarks/bench_vllm.py -v --model-id gpt2
```

Files:

- `bench_safetensors.py` - native `safetensors` vs tensor_store backends
- `bench_serverlessllm.py` - tensor_store ServerlessLLM microbenchmarks
- `bench_vllm.py` - vLLM integration benchmarks

vLLM benchmark matrix:

- native
- tensor_store SafeTensors: `sync`, `mmap`
- tensor_store ServerlessLLM: `sync`, `mmap`

The vLLM benchmark uses a benchmark-only custom loader named `tensor_store` and
passes loader behavior through `model_loader_extra_config`.

ServerlessLLM artifacts are generated on demand from resolved local safetensors
and cached under `bindings/python/benchmarks/.cache/`.
