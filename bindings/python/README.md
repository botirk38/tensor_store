# tensor_store Python Bindings

PyTorch-first Python bindings for the [tensor_store](../README.md) Rust library.

## Installation

Requires Rust 1.92+ (`rustup update`) and Python 3.9–3.12.

### Core bindings

```bash
cd bindings/python
uv sync --group dev
uv run maturin develop --release
```

### Benchmark setup (non-vLLM)

```bash
uv sync --group dev --group bench
uv run maturin develop --release
```

### Benchmark setup (with vLLM)

```bash
uv sync --group dev --group bench-vllm
uv run maturin develop --release
```

## Import layout

- **`tensor_store_py`** — package metadata only (`__version__` via `__all__`).
- **`tensor_store_py._tensor_store_rust`** — native I/O and handles (`load_*`, `open_*`, `save_*`, handle classes, `TensorStoreError`).
- **`tensor_store_py.torch`** — PyTorch-oriented helpers (`load_file`, `save_file`, …).
- **`tensor_store_py.tensorflow`** — TensorFlow-oriented helpers (`load_file`, `save_file`, …).

## Quick start (PyTorch)

```python
import asyncio

from tensor_store_py.torch import load_file as pytorch_load_file, open_file as pytorch_open_file

# Lazy: mmap-backed handle (see rust helpers for async/sync/mmap variants)
f = pytorch_open_file("model.safetensors")
for name in f.keys():
    tensor = f.get_tensor(name)
    print(name, tensor.shape, tensor.dtype)

# Eager: full dict load (sync path)
state_dict = pytorch_load_file("model.safetensors", device="cpu")
```

## Quick start (low-level extension)

```python
import asyncio

from tensor_store_py._tensor_store_rust import (
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_sync,
)

# Blocking eager load (defaults to PyTorch tensors)
weights = load_safetensors_sync("model.safetensors")
```

Async entrypoints return awaitables (no `_sync` / `_mmap` suffix):

```python
async def main():
    from tensor_store_py._tensor_store_rust import load_safetensors, open_safetensors

    handle = await open_safetensors("model.safetensors")
    state = await load_safetensors("model.safetensors")

asyncio.run(main())
```

## TensorFlow

```python
from tensor_store_py.tensorflow import load_file as tf_load_file

weights = tf_load_file("model.safetensors", device="/CPU:0")
```

## Requirements

- Python 3.9+
- PyTorch 2.1+
- TensorFlow 2.11+
- Linux or macOS (ServerlessLLM mmap is Linux-only; owned fallback elsewhere)

## Backends

- **async** (no suffix): native async I/O (Tokio; io_uring on Linux), returns an awaitable
- **sync** (`_sync` suffix): blocking read into memory
- **mmap** (`_mmap` suffix): memory-mapped, zero-copy on CPU

The GIL is released during I/O and parsing where applicable.
