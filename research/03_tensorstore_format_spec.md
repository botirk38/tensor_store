# TensorStore Format Specification

## Design Philosophy

**Note: This custom format is primarily for educational purposes** to understand binary format design, memory alignment, and I/O optimization. The real performance gains in this project come from the io_uring loading approach, which works effectively with existing formats like safetensors.

TensorStore format is designed as a learning exercise inspired by safetensors and ServerlessLLM checkpoint formats, exploring optimizations for io_uring-based asynchronous I/O and high-performance tensor loading scenarios.

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

#### Intelligent Prefetching
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

```rust
pub struct SafetensorsConverter {
    alignment: usize,
    chunk_size: usize,
    numa_topology: Option<NumaTopology>,
}

impl SafetensorsConverter {
    pub async fn convert(
        &self,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<ConversionReport, ConversionError> {
        // 1. Parse safetensors header
        let header = self.parse_safetensors_header(input_path).await?;

        // 2. Analyze tensor access patterns
        let access_patterns = self.analyze_access_patterns(&header)?;

        // 3. Compute optimal layout with alignment
        let layout = self.compute_optimal_layout(&header, &access_patterns)?;

        // 4. Generate TensorStore metadata
        let metadata = self.generate_metadata(&header, &layout)?;

        // 5. Write aligned tensor data
        self.write_aligned_data(input_path, output_path, &layout).await?;

        Ok(ConversionReport {
            original_size: input_path.metadata()?.len(),
            converted_size: output_path.metadata()?.len(),
            alignment_overhead: layout.total_padding,
            optimization_level: self.optimization_level(),
        })
    }
}
```

### From PyTorch

```rust
impl PyTorchConverter {
    pub async fn convert_checkpoint(
        &self,
        checkpoint_path: &Path,
        output_path: &Path,
    ) -> Result<ConversionReport, ConversionError> {
        // 1. Load PyTorch checkpoint
        let checkpoint = self.load_pytorch_checkpoint(checkpoint_path)?;

        // 2. Extract tensor metadata
        let tensors = self.extract_tensor_info(&checkpoint)?;

        // 3. Optimize tensor ordering
        let ordered_tensors = self.optimize_tensor_order(&tensors)?;

        // 4. Serialize with alignment
        self.serialize_aligned(&ordered_tensors, output_path).await?;

        Ok(ConversionReport::new())
    }
}
```

## Loading Implementation

### Async Tensor Loader

```rust
pub struct TensorStoreLoader {
    file: tokio_uring::fs::File,
    metadata: TensorStoreMetadata,
    memory_pool: AlignedMemoryPool,
}

impl TensorStoreLoader {
    pub async fn load_tensor(&self, name: &str) -> Result<Tensor, LoadError> {
        let tensor_info = self.metadata.tensors.get(name)
            .ok_or(LoadError::TensorNotFound)?;

        // Allocate aligned memory
        let buffer = self.memory_pool.allocate_aligned(
            tensor_info.data_size,
            tensor_info.alignment,
        )?;

        // Perform vectored read
        let chunks = self.compute_io_chunks(tensor_info)?;
        let (result, buffer) = self.file.read_vectored_at(buffer, &chunks).await;
        result?;

        // Create tensor from buffer
        Ok(Tensor::from_aligned_buffer(
            buffer,
            &tensor_info.shape,
            tensor_info.dtype,
        ))
    }

    pub async fn load_tensor_batch(
        &self,
        names: &[&str],
    ) -> Result<Vec<Tensor>, LoadError> {
        let batch_info = self.compute_batch_layout(names)?;
        let operations = self.prepare_batch_operations(&batch_info)?;

        // Submit all operations concurrently
        let results = join_all(operations).await;

        // Process results and create tensors
        self.process_batch_results(results, names)
    }
}
```

### Memory Management

```rust
pub struct AlignedMemoryPool {
    pools: HashMap<usize, Vec<AlignedBuffer>>,
    numa_node: Option<usize>,
}

impl AlignedMemoryPool {
    pub fn allocate_aligned(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<AlignedBuffer, AllocationError> {
        // Round up to alignment boundary
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        // Try to reuse existing buffer
        if let Some(buffer) = self.pools.get_mut(&aligned_size)
            .and_then(|pool| pool.pop()) {
            return Ok(buffer);
        }

        // Allocate new aligned buffer
        let layout = Layout::from_size_align(aligned_size, alignment)?;
        let ptr = unsafe { alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(AllocationError::OutOfMemory);
        }

        Ok(AlignedBuffer::new(ptr, aligned_size, alignment))
    }
}
```

## Performance Validation

### Benchmarking Strategy

```rust
pub struct PerformanceBenchmark {
    formats: Vec<FormatType>,
    tensor_sizes: Vec<usize>,
    access_patterns: Vec<AccessPattern>,
}

impl PerformanceBenchmark {
    pub async fn run_comprehensive_benchmark(
        &self,
    ) -> Result<BenchmarkReport, BenchmarkError> {
        let mut results = BenchmarkReport::new();

        for format in &self.formats {
            for &size in &self.tensor_sizes {
                for pattern in &self.access_patterns {
                    let result = self.benchmark_configuration(
                        format,
                        size,
                        pattern,
                    ).await?;

                    results.add_result(result);
                }
            }
        }

        Ok(results)
    }

    async fn benchmark_configuration(
        &self,
        format: &FormatType,
        size: usize,
        pattern: &AccessPattern,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let loader = self.create_loader(format)?;
        let tensors = self.generate_test_tensors(size, pattern)?;

        let start = Instant::now();
        let loaded_tensors = loader.load_tensor_batch(&tensors).await?;
        let duration = start.elapsed();

        Ok(BenchmarkResult {
            format: format.clone(),
            size,
            pattern: pattern.clone(),
            duration,
            throughput: self.calculate_throughput(size, duration),
            cpu_usage: self.measure_cpu_usage(),
            memory_usage: self.measure_memory_usage(),
        })
    }
}
```

### Target Performance Metrics

#### Loading Speed
- **5-10x faster** than safetensors baseline
- **Similar to ServerlessLLM** performance gains
- **Linear scaling** with storage bandwidth

#### CPU Efficiency
- **30-50% reduction** in CPU overhead vs multi-threaded approaches
- **Minimal context switching** due to async event loop
- **Reduced syscall overhead** through vectored I/O

#### Memory Efficiency
- **Zero-copy loading** with memory mapping
- **Minimal memory overhead** beyond tensor data
- **NUMA-aware allocation** for multi-GPU systems

## Error Handling and Validation

### Format Validation

```rust
pub struct FormatValidator {
    strict_mode: bool,
    alignment_checks: bool,
}

impl FormatValidator {
    pub fn validate_file(&self, path: &Path) -> Result<ValidationReport, ValidationError> {
        let mut report = ValidationReport::new();

        // Check magic number and version
        self.validate_header(path, &mut report)?;

        // Validate metadata JSON
        self.validate_metadata(path, &mut report)?;

        // Check tensor data alignment
        if self.alignment_checks {
            self.validate_alignment(path, &mut report)?;
        }

        // Verify checksums
        self.validate_checksums(path, &mut report)?;

        Ok(report)
    }
}
```

### Runtime Error Recovery

```rust
pub enum LoadError {
    IoError(std::io::Error),
    AlignmentError { expected: usize, actual: usize },
    CorruptedData { tensor: String, checksum_mismatch: bool },
    InsufficientMemory { requested: usize, available: usize },
}

impl From<std::io::Error> for LoadError {
    fn from(err: std::io::Error) -> Self {
        LoadError::IoError(err)
    }
}
```

This specification provides the foundation for implementing TensorStore as a high-performance, io_uring-optimized tensor storage format that addresses the limitations of existing approaches while providing significant performance improvements.