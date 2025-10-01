# TensorStore Format Specification

## Design Philosophy

TensorStore format is specifically designed to exploit the full capabilities of io_uring, including IORING_OP_READV vectored operations, submission queue batching, completion queue polling optimizations, and memory alignment strategies that maximize asynchronous I/O throughput while minimizing kernel-userspace transitions.

The format serves as a critical component in demonstrating how co-designed storage formats and I/O architectures can achieve superior performance compared to traditional multi-threaded approaches.

## Design Goals

### Primary Objectives
1. **io_uring Optimization**: Format designed for vectored I/O and batch operations
2. **Memory Alignment**: 64-byte alignment for optimal DMA transfers and GPU kernel access
3. **Zero-Copy Loading**: Direct memory mapping with minimal overhead
4. **NUMA Awareness**: Layout optimized for multi-GPU NUMA topologies
5. **Performance**: Target 5-10x improvement over existing formats

### Secondary Objectives
1. **Security**: No arbitrary code execution like pickle
2. **Compatibility**: Easy conversion from existing formats
3. **Extensibility**: Forward-compatible versioning scheme
4. **Debugging**: Human-readable metadata for inspection

## Format Specification

### File Structure Overview

```
TensorStore File (.tensorstore):
┌─────────────────────────────────────────────────────────────┐
│ File Header (64 bytes)                                      │
├─────────────────────────────────────────────────────────────┤
│ Metadata Section (Variable length, aligned)                │
├─────────────────────────────────────────────────────────────┤
│ Tensor Data Section (64-byte aligned)                      │
└─────────────────────────────────────────────────────────────┘
```

### File Header (64 bytes)

The header contains:
- Magic number: "TNSRSTR\0" (8 bytes)
- Format version: 0x00000001 (4 bytes)
- Header size: 64 (4 bytes)
- Metadata size in bytes (8 bytes)
- Data section offset (8 bytes)
- Total file size (8 bytes)
- Number of tensors (4 bytes)
- CRC32 checksum of metadata (4 bytes)
- Reserved space for future use (24 bytes, zero-filled)

### Metadata Section

The metadata is stored as JSON for human readability and debugging, followed by padding to ensure 64-byte alignment for the data section.

#### Metadata Schema

The JSON metadata contains:
- **Format version and timestamps**
- **Source format information** (e.g., converted from safetensors)
- **Model metadata**: name, type, parameter count, precision
- **Per-tensor information**:
  - Data type and shape
  - Data offset and size within file
  - Alignment requirements (64 bytes)
  - Chunk information for batched I/O
  - NUMA hints for memory allocation
  - I/O access patterns and priorities
- **I/O optimization hints**:
  - Vectored I/O groups for batch operations
  - Recommended prefetch order
  - Optimal batch sizes for different storage types

### Tensor Data Section

#### Alignment Strategy
- **64-byte alignment** for all tensor data start positions
- **Padding inserted** between tensors to maintain alignment
- **Contiguous layout** within each tensor for optimal I/O

#### Data Layout
```
Tensor Data Layout:
┌─────────────────────────────────────────┐
│ Tensor 1 Data (64-byte aligned)        │
├─────────────────────────────────────────┤
│ Padding (if needed for next alignment) │
├─────────────────────────────────────────┤
│ Tensor 2 Data (64-byte aligned)        │
├─────────────────────────────────────────┤
│ ...                                     │
└─────────────────────────────────────────┘
```

## Addressing Safetensors Limitations

### Memory Alignment Issues

**Sources: DeepWiki analysis of huggingface/safetensors and 2024 web research on CUDA alignment issues**

From DeepWiki research, safetensors has documented alignment considerations:

**Source: DeepWiki on safetensors**: "SafeTensors addresses alignment by sorting tensors during serialization by descending dtype alignment and then by name. The JSON header is also padded to ensure 8-byte alignment for the subsequent binary data."

However, 2024 web search reveals ongoing alignment issues in practice:
- **Source: GitHub Issue #36961**: "Gemma3: Cuda error: misaligned address" with safetensors version 0.5.3
- **Source: Safetensors documentation**: "Sub 1 bytes dtypes: Dtypes can now have lower than 1 byte size, this makes alignment&addressing tricky. For now, the library will simply error out whenever an operation triggers a non aligned read."

#### TensorStore Solution
1. **Fixed 64-byte alignment** for all tensor data
2. **Padded metadata section** to ensure data section starts at aligned boundary
3. **Alignment validation** during format creation and loading

### Sub-byte Datatypes

**Source: Safetensors documentation (DeepWiki analysis)**
Safetensors documented limitation: "Sub 1 bytes dtypes: Dtypes can now have lower than 1 byte size, this makes alignment&addressing tricky. For now, the library will simply error out whenever an operation triggers a non aligned read."

#### TensorStore Approach
1. **Minimum 8-bit alignment** for all tensor elements
2. **Packed representation** for boolean/bit tensors with proper padding
3. **Clear error messages** for unsupported datatype configurations

## io_uring Optimization Features

### Core io_uring Capabilities

#### 1. Vectored I/O Operations
- **IORING_OP_READV**: Read multiple tensor chunks in single operation
- **Batch Operations**: Group related reads for efficiency
- **Reduced Syscall Overhead**: Fewer kernel transitions

#### 2. Submission Queue Batching
- **Batch Submission**: Submit multiple operations at once
- **Efficient Completion**: Poll completion queue for results
- **Async Processing**: Non-blocking I/O operations

### Vectored I/O Support

#### Pre-computed iovec Structures
```json
"vectored_io_metadata": {
  "iovec_groups": [
    {
      "group_id": 0,
      "total_size": 134217728,
      "iovecs": [
        {"offset": 4096, "length": 32768},
        {"offset": 36864, "length": 32768},
        {"offset": 69632, "length": 32768}
      ]
    }
  ]
}
```

#### Batch Loading Instructions
```json
"batch_loading": {
  "optimal_batch_sizes": {
    "nvme_ssd": 16,
    "sata_ssd": 8,
    "network_storage": 4
  },
  "concurrency_hints": {
    "max_concurrent_tensors": 4,
    "max_io_depth": 32
  }
}
```

### Read-ahead Optimization

#### Basic Prefetching Strategy
```json
"prefetch_strategy": {
  "access_patterns": {
    "embedding.weight": {
      "pattern": "random_access",
      "prefetch_size": 65536,
      "locality": "temporal"
    },
    "transformer.layers.*.weight": {
      "pattern": "sequential",
      "prefetch_size": 1048576,
      "locality": "spatial"
    }
  }
}
```

## NUMA and Multi-GPU Support

### NUMA Topology Awareness

#### Memory Allocation Hints
```json
"numa_topology": {
  "nodes": [
    {
      "node_id": 0,
      "gpus": [0, 1],
      "tensors": ["embedding.weight", "lm_head.weight"]
    },
    {
      "node_id": 1,
      "gpus": [2, 3],
      "tensors": ["transformer.layers.0-15.*"]
    }
  ]
}
```

#### GPU Affinity Mapping
```json
"gpu_placement": {
  "strategy": "minimize_pcie_hops",
  "mappings": [
    {
      "tensor_pattern": "transformer.layers.0-7.*",
      "preferred_gpus": [0, 1],
      "numa_node": 0
    },
    {
      "tensor_pattern": "transformer.layers.8-15.*",
      "preferred_gpus": [2, 3],
      "numa_node": 1
    }
  ]
}
```

## Conversion Pipeline

### From Safetensors

The conversion process involves:
1. Parse safetensors header and extract tensor metadata
2. Analyze tensor access patterns for optimal io_uring layout
3. Compute optimal layout with 64-byte alignment requirements
4. Generate TensorStore metadata with vectored I/O hints
5. Write aligned tensor data with padding for optimal async access

### From PyTorch

The PyTorch conversion process:
1. Load PyTorch checkpoint and deserialize tensor data
2. Extract tensor metadata including shapes, dtypes, and sizes
3. Optimize tensor ordering for sequential io_uring access patterns
4. Serialize with 64-byte alignment and vectored I/O optimization hints

## Loading Implementation

### Async Tensor Loader

The loader implementation leverages io_uring's capabilities:

**Single Tensor Loading:**
1. Lookup tensor metadata from the file header
2. Allocate aligned memory buffer matching tensor requirements
3. Compute optimal io_uring chunk layout for vectored reads
4. Submit IORING_OP_READV operation with computed iovecs
5. Create tensor from aligned buffer with zero-copy semantics

**Batch Tensor Loading:**
1. Analyze batch access patterns and compute optimal layout
2. Prepare multiple vectored operations for concurrent execution
3. Submit all operations to io_uring submission queue
4. Process completion queue results and construct tensors
5. Return batch of loaded tensors with minimal memory copying

### Memory Management

**Aligned Memory Pool Strategy:**
- Maintain pools of pre-allocated aligned buffers for different sizes
- Round allocation sizes up to alignment boundaries (64-byte minimum)
- Reuse existing buffers when possible to minimize allocation overhead
- Support NUMA-aware allocation for multi-GPU topologies
- Handle alignment requirements for optimal DMA transfer performance

## Performance Validation

### Benchmarking Strategy

**Comprehensive Testing Framework:**
- Test multiple tensor formats (TensorStore, safetensors, PyTorch)
- Vary tensor sizes from small (1MB) to large (10GB)
- Test different access patterns (sequential, random, batch)
- Measure loading duration, throughput, CPU usage, and memory consumption
- Compare io_uring performance against traditional multi-threaded approaches

### Target Performance Metrics

#### Loading Speed
- **>20% faster** than safetensors with tokio::fs baseline
- **Comparable to or better than** ServerlessLLM through io_uring optimizations
- **Linear scaling** with storage bandwidth

#### CPU Efficiency
- **Measurable reduction** in CPU overhead vs multi-threaded approaches
- **Reduced context switching** through async event loop
- **Fewer syscalls** through vectored I/O operations

#### Memory Efficiency
- **Zero-copy loading** with memory mapping
- **Minimal memory overhead** beyond tensor data
- **NUMA-aware allocation** for multi-GPU systems

## Error Handling and Validation

### Format Validation

**File Integrity Checks:**
- Verify magic number and format version compatibility
- Validate JSON metadata structure and required fields
- Check tensor data alignment meets 64-byte requirements
- Verify CRC32 checksums for metadata and tensor data integrity

### Runtime Error Recovery

**Error Categories:**
- **I/O Errors:** Handle file access, permission, and storage issues
- **Alignment Errors:** Detect and report misaligned tensor data
- **Corrupted Data:** Identify checksum mismatches and data corruption
- **Memory Errors:** Handle insufficient memory and allocation failures

This specification provides the foundation for implementing TensorStore as a high-performance, io_uring-optimized tensor storage format that addresses the limitations of existing approaches while providing significant performance improvements.