# Tensor Store Benchmarks

Real-model benchmarks using `pytest-benchmark`.

## Quick Start

```bash
cd bindings/python

# SafeTensors benchmarks (supports multi-shard models)
uv run pytest benchmarks/bench_safetensors.py -v --model-id gpt2

# ServerlessLLM benchmarks (multi-shard supported)
uv run pytest benchmarks/bench_serverlessllm.py -v --model-id gpt2

# vLLM integration benchmarks
uv run pytest benchmarks/bench_vllm.py -v --model-id gpt2
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--model-id` | **Required.** HuggingFace model ID |
| `--fixtures-dir` | Directory for downloaded models and cached artifacts |

## Benchmark Suites

| File | Description |
|------|-------------|
| `bench_safetensors.py` | SafeTensors loading: native vs tensor_store (sync, mmap, default) |
| `bench_serverlessllm.py` | ServerlessLLM loading (sync, mmap, default) |
| `bench_vllm.py` | vLLM integration: init, TTFT, steady-state decode |

## SafeTensors Benchmarks

**Backends:**
- `native` - `safetensors.torch.load_file`
- `tensor_store sync` - `tensor_store.load_safetensors_sync`
- `tensor_store mmap` - `tensor_store.load_safetensors_mmap`
- `tensor_store default` - `tensor_store.load_safetensors`

**Cache modes:** `warm`, `cold`

## ServerlessLLM Benchmarks

Uses the shared size-based heuristic for partition count: `max(1, ceil(total_bytes / 512 MiB))`.

**Backends:**
- `sync`
- `mmap`
- `default`

**Cache modes:** `warm`, `cold`

## vLLM Benchmarks

**Loaders:**
- `native` - default vLLM loader
- `ts_safetensors_sync` - tensor_store safetensors, sync backend
- `ts_safetensors_mmap` - tensor_store safetensors, mmap backend
- `ts_serverlessllm_sync` - tensor_store ServerlessLLM, sync backend
- `ts_serverlessllm_mmap` - tensor_store ServerlessLLM, mmap backend

**Benchmark kinds:**
- `load_only` - model initialization time
- `ttft` - time to first token
- `steady_state_decode` - average decode time after warmup

## Partition heuristic

Default ServerlessLLM partition counts follow the Rust helper:

`max(1, ceil(total_bytes / 512 MiB))`

There is no artificial upper cap (beyond practical `usize` limits). Override with explicit conversion
arguments if you need a different layout.

## Default backend policy

- `open_*` defaults to `mmap`.
- `load_*` defaults to eager loading and chooses between async and sync backends.
- `load_*` does not auto-select `mmap`.

## Recommended Models

For this machine, use the fixture ladder:
- Small: `gpt2`
- Medium-small: `Qwen/Qwen2-0.5B`
- Small-medium: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Medium: `Qwen/Qwen2-1.5B`
- Medium-large: `EleutherAI/pythia-1.4b-deduped`
- Large: `EleutherAI/pythia-2.8b-deduped`

Avoid 7B+ fixtures unless you have enough RAM for full eager loads.
