# TensorStore Examples

This directory contains runnable examples demonstrating how to use TensorStore to load model checkpoints and run inference with PyTorch and vLLM.

## Setup

Install dependencies:

```bash
cd bindings/python
uv sync --group dev --group torch       # For PyTorch examples
uv sync --group dev --group vllm       # For vLLM examples
uv run maturin develop --release
```

## Usage

All examples only need a HuggingFace model ID - they automatically:
1. Download the model in SafeTensors format from HuggingFace
2. Convert to ServerlessLLM format if needed

### PyTorch Example

```bash
cd bindings/python
uv run python examples/pytorch.py gpt2 --prompt "Hello, world!"
```

Arguments:
- `model`: HuggingFace model ID (e.g., gpt2, Qwen/Qwen2-0.5B)
- `--prompt`: Prompt for text generation
- `--backend`: TensorStore I/O backend (default, sync, io-uring)
- `--format`: Checkpoint format (safetensors, serverlessllm) - defaults to safetensors
- `--device`: Device to load tensors to (default: cuda)
- `--max_tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 1.0)

### vLLM Example

```bash
cd bindings/python
uv run python examples/vllm_infer.py --model gpt2 --prompt "Hello, world!"
```

Arguments:
- `--model`: HuggingFace model ID
- `--prompt`: Prompt for text generation
- `--backend`: TensorStore I/O backend (default, sync, io-uring)
- `--max_tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 1.0)

## Formats

### SafeTensors (default for PyTorch)
SafeTensors format stores all tensor data contiguously in one or more files with a metadata header. This is the default format for the PyTorch example.

### ServerlessLLM (default for vLLM)
ServerlessLLM format partitions tensor data across multiple files, increasing request count and file-switching overhead. This is the default format for vLLM examples.

To use ServerlessLLM format with PyTorch, add `--format serverlessllm`:

```bash
uv run python examples/pytorch.py gpt2 --prompt "Hello" --format serverlessllm
```

Note: vLLM example uses serverlessllm format by default (see `vllm_infer.py`).

The first run will automatically convert the model to ServerlessLLM format. Subsequent runs use the cached conversion.

## Backends

- **default**: Automatic backend selection (uses io_uring on Linux if available)
- **sync**: Synchronous POSIX I/O
- **io-uring**: Linux io_uring for asynchronous I/O (Linux only)

## Manual Download and Conversion

If you want to download and convert manually:

```python
from huggingface_hub import snapshot_download

# Download SafeTensors
model_dir = snapshot_download("gpt2", allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])
print(f"Model: {model_dir}")

# Convert to ServerlessLLM
from tensor_store_py._tensor_store_rust import convert_safetensors_to_serverlessllm

output_dir = model_dir + "_serverlessllm"
convert_safetensors_to_serverlessllm(model_dir, output_dir, partitions=1)
```
