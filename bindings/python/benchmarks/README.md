# Tensor Store Benchmarks

Real-model benchmarks using `pytest-benchmark`.

## Quick Start

```bash
cd bindings/python

# SafeTensors benchmarks (supports multi-shard models)
uv run pytest benchmarks/bench_safetensors.py -v --model-id gpt2

# ServerlessLLM benchmarks (single-shard models only)
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

Uses size-based heuristic for partition count. Only works with single-shard models.

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

## Partition Heuristic

| Model Size | Partitions |
|------------|------------|
| < 512 MiB  | 1          |
| < 2 GiB    | 2          |
| < 8 GiB    | 4          |
| < 24 GiB   | 8          |
| < 64 GiB   | 16         |
| >= 64 GiB  | 32         |

Capped at `min(32, cpu_count * 2)`.

## Recommended Models

For H100 experiments:
- Small: `meta-llama/Llama-3.1-8B`
- Medium: `Qwen/Qwen2.5-14B`
- Large: `Qwen/Qwen2.5-32B`
