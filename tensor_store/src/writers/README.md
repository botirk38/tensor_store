# Writers Module

This module provides functionality to **serialize and write** checkpoint formats to disk.

## Purpose

Writers are responsible for:
- **Serializing** checkpoint data to specific formats
- **Writing** format-specific file structures
- **Creating** output files and directories
- **Format-specific logic** for writing different checkpoint types

## Architecture Principle

**Writers should ONLY write their own format.** They should not:
- Convert between formats
- Know about other checkpoint formats
- Perform any transformation logic

## Module Structure

```
writers/
├── serverlessllm.rs   # ServerlessLLM format writer
├── tensorstore.rs     # TensorStore format writer
└── mod.rs            # Module exports
```

## Key Interfaces

### ServerlessLLM Writer
```rust
use tensor_store::writers::serverlessllm;

// Write tensor index (JSON)
serverlessllm::write_index("tensor_index.json", &tensors).await?;

// Write partition data (binary)
serverlessllm::write_partition("tensor.data_0", 0, &data).await?;
```

### Backend I/O Operations
```rust
use tensor_store::backends;

// Async file operations
backends::async_io::write_all(path, data).await?;
backends::async_io::write_range(path, offset, data).await?;

// io_uring operations (Linux only)
backends::io_uring::write_all(path, data).await?;
```

## Usage Pattern

```rust
// 1. Converters prepare data in target format
let serverlessllm_tensors = converters::prepare_serverlessllm_data(safetensors_data);

// 2. Use appropriate writer for output format
writers::serverlessllm::write_index(output_path, &serverlessllm_tensors).await?;

// 3. Write partition files as needed
for (partition_id, data) in partitions {
    let partition_path = format!("tensor.data_{}", partition_id);
    writers::serverlessllm::write_partition(&partition_path, partition_id, &data).await?;
}
```

## Backend Selection

The module provides multiple I/O backends optimized for different scenarios:

- **async_io**: General-purpose async file operations using Tokio
- **io_uring**: High-performance zero-copy operations on Linux systems

Backends are automatically selected based on platform and availability.