# Scripts

Utility scripts for the tensor_store project.

## Overview

This directory contains Python scripts for downloading model fixtures and preparing benchmark data.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Dependencies

Managed via `pyproject.toml`:
- `huggingface-hub` - Downloading models from HuggingFace
- `safetensors` - Verifying SafeTensors file integrity
- `numpy` - Numerical operations
- `requests` - HTTP requests

## Setup

```bash
cd scripts
uv sync
```

## Scripts

### download_models.py

Downloads SafeTensors format models from HuggingFace and **automatically converts them to ServerlessLLM format**.

#### Usage

```bash
# Download a model (automatically converts to ServerlessLLM)
uv run python download_models.py Qwen/Qwen2-0.5B

# Download with integrity verification
uv run python download_models.py Qwen/Qwen2-0.5B --verify

# Download SafeTensors only (skip ServerlessLLM conversion)
uv run python download_models.py Qwen/Qwen2-0.5B --no-convert

# Download to a custom directory
uv run python download_models.py Qwen/Qwen2-0.5B --output-dir ./my_fixtures

# Download a private model with token
uv run python download_models.py meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN

# Override auto-calculated partition count
uv run python download_models.py Qwen/Qwen2-0.5B --partitions 16
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `repo` | HuggingFace repository ID (required) | - |
| `--output-dir` | Output directory for fixtures | `../fixtures` |
| `--token` | HuggingFace API token for private repos | None |
| `--no-convert` | Skip ServerlessLLM conversion (SafeTensors only) | False |
| `--partitions` | Number of partitions for conversion | Auto (based on size) |
| `--verify` | Verify SafeTensors file integrity after download | False |

#### Partition count heuristic

When `--partitions` is omitted, the script uses the same default as the Rust library:

`max(1, ceil(model_bytes / 512 MiB))`

There is no fixed upper cap (for example 16 or 32). Pass `--partitions` explicitly to override.

#### Output Structure

After downloading a model, the script creates:

```
fixtures/
└── qwen-qwen2-0.5b/
    ├── README.md              # Model metadata and file info
    ├── *.safetensors           # All downloaded SafeTensors shards
    └── model_serverlessllm/   # (if --convert flag used)
        ├── metadata.json
        └── *.bin              # Partitioned tensor data
```

#### Features

- **Automatic retry**: Retries downloads up to 3 times on failure
- **Sharded model support**: Keeps all downloaded shards and converts the full directory
- **Integrity verification**: Optional SHA256 hash verification
- **Model metadata**: Creates README with model info from HuggingFace API
- **Format conversion**: Optional conversion to ServerlessLLM format

## Examples

### Download a small model for testing

```bash
# ~500MiB-class models typically auto-convert to 1 ServerlessLLM partition (512 MiB target size)
uv run python download_models.py Qwen/Qwen2-0.5B --verify
```

### Download a larger model

```bash
# Larger models pick higher partition counts automatically (see ceil(bytes / 512 MiB))
uv run python download_models.py Qwen/Qwen2-1.5B --verify
```

### Download SafeTensors only

```bash
# Skip ServerlessLLM conversion
uv run python download_models.py Qwen/Qwen2-0.5B --no-convert
```

### Batch download multiple models

```bash
# Download multiple models
for model in "gpt2" "Qwen/Qwen2-0.5B" "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "Qwen/Qwen2-1.5B" "EleutherAI/pythia-2.8b-deduped"; do
    uv run python download_models.py "$model" --verify
done
```

## Environment Variables

- `HF_TOKEN` - HuggingFace API token (alternative to `--token` flag)
- `HF_HOME` - Custom HuggingFace cache directory

## Troubleshooting

### "No .safetensors files found"
The model may not have SafeTensors format files. Check the model's HuggingFace page to verify SafeTensors availability.

### "convert binary not found"
Build the converter first:
```bash
cargo build --release --bin convert
```

### Authentication errors
For gated models (like Llama 2), you need to:
1. Accept the model license on HuggingFace
2. Provide your token via `--token` or `HF_TOKEN` environment variable
