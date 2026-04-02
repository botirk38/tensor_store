# demo

Interactive demonstration tool showcasing SafeTensors and ServerlessLLM loader capabilities.

## Usage

```bash
cargo run [--release] --bin demo -- <COMMAND> <SCENARIO> [OPTIONS]
```

**Important**: The `--release` flag (optional but recommended) goes before `--bin`, and `--` separates cargo flags from program arguments.

## Commands

### SafeTensors Demonstrations

```bash
cargo run --bin demo -- safetensors <SCENARIO> [--fixture <NAME>]
```

**Available Scenarios:**
- `async` - Async sequential loading
- `sync` - Synchronous loading
- `mmap` - Memory-mapped lazy loading
- `parallel-sync` - Sync parallel multi-core loading (blocking I/O, multiple threads)
- `metadata` - Detailed tensor metadata exploration
- `all` - Run all scenarios sequentially

### ServerlessLLM Demonstrations

```bash
cargo run --bin demo -- serverlessllm <SCENARIO> [--fixture <NAME>]
```

**Available Scenarios:**
- `async` - Async sequential loading
- `sync` - Synchronous loading
- `mmap` - Memory-mapped lazy loading
- `parallel-sync` - Sync parallel multi-core loading (blocking I/O, multiple threads)
- `metadata` - Index structure and partition statistics
- `all` - Run all scenarios sequentially

## Options

- `-f, --fixture <NAME>` - Specify a fixture name

## Examples

Run async SafeTensors demo:
```bash
cargo run --bin demo -- safetensors async
```

Run all ServerlessLLM scenarios:
```bash
cargo run --bin demo -- serverlessllm all
```

Demo with specific fixture:
```bash
cargo run --bin demo -- safetensors async --fixture qwen-qwen3-0.6b
```

## Purpose

Use this tool to:
- Understand different loading strategies
- Compare performance characteristics
- Explore tensor metadata structures
- Learn about partition-based loading in ServerlessLLM
