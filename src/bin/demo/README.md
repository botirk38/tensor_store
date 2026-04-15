# demo

Interactive demonstration tool showcasing SafeTensors and ServerlessLLM loader capabilities. Pass a **Hugging Face model id**; checkpoints come from the Hub cache (and ServerlessLLM is converted under the OS cache when needed).

## Usage

```bash
cargo run [--release] --bin demo -- <COMMAND> <SCENARIO> --model-id <ORG/NAME>
```

**Important**: Put `--release` (recommended) before `--bin`, and use `--` before program arguments.

## Commands

### SafeTensors

```bash
cargo run --release --bin demo -- safetensors <SCENARIO> --model-id <HF/MODEL>
```

**Scenarios:** `async`, `sync`, `mmap`, `metadata`, `all`. **`parallel-sync`** is not wired for Hub-backed models (returns a clear error).

### ServerlessLLM

```bash
cargo run --release --bin demo -- serverlessllm <SCENARIO> --model-id <HF/MODEL>
```

Same scenario list; **`parallel-sync`** is not wired for Hub-backed models.

## Examples

```bash
cargo run --release --bin demo -- safetensors async --model-id Qwen/Qwen3-0.6B
cargo run --release --bin demo -- serverlessllm all --model-id Qwen/Qwen3-0.6B
```

## Purpose

- Compare loading strategies on a real checkpoint
- Inspect tensor metadata and ServerlessLLM partitions
