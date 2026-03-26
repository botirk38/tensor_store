# Benchmark Suite - AI Agent Context

## Overview

This benchmark suite compares tensor_store performance against native libraries (safetensors) for model weight loading, and measures vLLM time-to-first-token (TTFT) with and without tensor_store integration.

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures: model paths, synthetic fixtures, cache control, model downloading |
| `fixtures.py` | Model creation: synthetic GPT-2, ServerlessLLM format conversion |
| `bench_safetensors.py` | SafeTensors loading: native (safetensors) vs tensor_store comparison |
| `bench_serverlessllm.py` | ServerlessLLM loading (tensor_store only - no native reference) |
| `bench_vllm_ttft.py` | vLLM time-to-first-token with/without tensor_store integration |
| `vllm_integration.py` | Monkey patch helper to inject tensor_store into vLLM |

## Data Flow

```
Model Source (HuggingFace or synthetic fixture)
         ↓
conftest.py fixture (downloads model if needed, provides path)
         ↓
bench_*.py (runs benchmark with model path)
         ↓
pytest-benchmark (collects timing metrics)
```

## Model Loading Paths

### Native SafeTensors (Reference)

```python
from safetensors.torch import load_file
weights = load_file(path)  # Uses safetensors library
```

### Tensor Store SafeTensors

```python
from tensor_store_py.torch import load_file, load_file_mmap
weights = load_file(path)      # Sync with io_uring (Linux) or fallback
weights = load_file_mmap(path)  # Memory-mapped lazy loading
```

### Tensor Store ServerlessLLM

```python
from tensor_store_py._tensor_store_rust import load_serverlessllm_sync
weights = load_serverlessllm_sync(path)  # Custom format with partitioning
```

### vLLM Integration (Monkey Patch)

```python
# BEFORE importing vLLM - this is critical
from benchmarks.vllm_integration import patch_vllm
patch_vllm()  # Replaces safetensors.torch.load_file with tensor_store version

# Now vLLM will use tensor_store for weight loading
from vllm import LLM
model = LLM(model="gpt2")
```

## Patterns

### Cold Cache Benchmark

```python
from benchmarks.fixtures import drop_page_cache

def test_load_cold(benchmark, safetensors_path):
    def run():
        drop_page_cache(safetensors_path)  # Clear OS page cache
        return load_file(safetensors_path)
    benchmark(run)
```

### Warm Cache Benchmark

```python
def test_load_warm(benchmark, safetensors_path):
    # First call loads into page cache
    load_file(safetensors_path)
    
    # Subsequent calls hit warm cache
    def run():
        return load_file(safetensors_path)
    benchmark(run)
```

### Parametrized Model Fixture

```python
import pytest
from benchmarks.fixtures import download_model_if_missing

@pytest.fixture(params=["gpt2", "Qwen/Qwen2-0.5B"])
def model_path(request, tmp_path):
    return download_model_if_missing(request.param, tmp_path)

def test_load(model_path):
    load_file(model_path)
```

### vLLM TTFT Benchmark

```python
import pytest
from vllm import LLM
from benchmarks.vllm_integration import patch_vllm, unpatch_vllm

@pytest.mark.parametrize("use_tensor_store", [True, False])
def test_vllm_ttft(benchmark, model_name, use_tensor_store):
    if use_tensor_store:
        patch_vllm()
    
    def run():
        model = LLM(model=model_name)
        # Generate one token to measure TTFT
        output = model.generate("Hello", max_tokens=1)
        return output
    
    result = benchmark(run)
    
    if use_tensor_store:
        unpatch_vllm()
```

## Configuration

### Environment Variables

- `MODEL_NAME`: HuggingFace model ID to download (e.g., `gpt2`, `Qwen/Qwen2-0.5B`)
- `FIXTURES_DIR`: Directory to store downloaded models (default: `fixtures/`)

### Pytest Options

```bash
# Run specific benchmarks
pytest benchmarks/bench_safetensors.py -v

# Filter by cache condition
pytest benchmarks/ -k "cold" -v
pytest benchmarks/ -k "warm" -v

# Filter by backend
pytest benchmarks/ -k "native" -v
pytest benchmarks/ -k "tensor_store" -v
```

## Dependencies

- `vllm>=0.6.0` - vLLM inference engine for TTFT benchmarks
- `safetensors>=0.7.0` - Reference implementation for comparison
- `torch>=2.1` - PyTorch for tensor operations
- `pytest-benchmark>=5.2.3` - Benchmark collection and reporting
- `huggingface_hub>=0.20.0` - Model downloads from HuggingFace

## Implementation Notes

### vLLM Monkey Patch Timing

The monkey patch MUST be applied BEFORE importing vLLM:

```python
# WRONG - vLLM already imported, weights loaded before patch
import vllm
from benchmarks.vllm_integration import patch_vllm
patch_vllm()  # Too late

# CORRECT - patch before import
from benchmarks.vllm_integration import patch_vllm
patch_vllm()  # Replaces safetensors.load_file
import vllm  # Now uses patched version
```

### ServerlessLLM Format

ServerlessLLM is a custom format with partitioning support. There is no native Python library - tensor_store is the reference implementation. This is why benchmarks only test tensor_store for ServerlessLLM.

### Cache Dropping

The `drop_page_cache()` function uses `os.posix_fadvise()` on Linux. On non-Linux systems, it's a no-op. This means cold cache tests only work on Linux.

### Memory-mapped Loading

`load_file_mmap()` returns lazy tensors - actual data is loaded on access. The benchmark fixture includes `touch_tensor(t)` calls to force page faults for fair comparison with synchronous loading.

## Testing Locally

```bash
# Quick smoke test with synthetic fixture
pytest benchmarks/bench_safetensors.py::test_load_sync_warm -v

# Test with real model download
MODEL_NAME=gpt2 pytest benchmarks/bench_safetensors.py -v

# Test vLLM integration (requires GPU or CPU fallback)
MODEL_NAME=gpt2 pytest benchmarks/bench_vllm_ttft.py -v
```