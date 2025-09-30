# MVP Implementation Strategy

## Overview

This document outlines the **Minimum Viable Product (MVP)** implementation strategy for TensorStore, focusing on core functionality needed to validate the io_uring approach and demonstrate basic performance improvements.

## MVP Goals

### Primary Objective
**Validate the core hypothesis**: Can io_uring provide measurable performance improvements for tensor loading compared to traditional async I/O approaches?

**Note**: The real innovation is in the io_uring loading approach, not the file format. We include a custom TensorStore format for learning purposes, but the main comparison should be:
- **Baseline**: safetensors with tokio::fs
- **Test**: safetensors with tokio-uring
- **Bonus**: TensorStore format with tokio-uring (educational)

### MVP Success Criteria
1. **Primary**: Demonstrate io_uring performance improvement loading safetensors vs tokio::fs
2. **Secondary**: Load tensors from custom TensorStore format using io_uring (learning)
3. **Proof of concept**: Show that io_uring approach is technically viable and beneficial

### Explicitly OUT OF SCOPE for MVP
- NUMA awareness
- Multi-GPU support
- Complex prefetching
- Production-grade error handling
- Comprehensive testing framework
- Advanced memory management
- Conversion from multiple formats

## MVP Technology Stack

### Minimal Dependencies
```toml
[package]
name = "tensorstore-mvp"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core async I/O
tokio-uring = "0.5"
tokio = { version = "1.0", features = ["rt", "macros"] }

# Basic serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Memory alignment
bytemuck = { version = "1.0", features = ["derive"] }

# For testing/comparison
safetensors = "0.4"

# Error handling
thiserror = "1.0"

# Testing
criterion = { version = "0.5", features = ["html_reports"] }
```

## MVP Project Structure

```
tensorstore-mvp/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI for testing
│   ├── lib.rs                   # Public API
│   ├── format.rs                # Simple TensorStore format
│   ├── loader.rs                # Basic async tensor loader
│   ├── converter.rs             # Safetensors → TensorStore
│   └── error.rs                 # Basic error types
├── benches/
│   └── comparison.rs            # Performance comparison
└── tests/
    └── basic.rs                 # Basic functionality tests
```

## MVP Core Components

### 1. Simple TensorStore Format

```rust
// format.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStoreHeader {
    pub magic: [u8; 8],           // "TNSRSTR\0"
    pub version: u32,             // 1
    pub metadata_size: u64,       // Size of JSON metadata
    pub data_offset: u64,         // Where tensor data starts
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,            // "float32", "float16", etc.
    pub shape: Vec<usize>,
    pub data_offset: u64,         // Offset from data_offset in header
    pub size_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStoreMetadata {
    pub tensors: Vec<TensorInfo>,
}

// Ensure 64-byte alignment for all tensor data
pub const TENSOR_ALIGNMENT: usize = 64;
```

### 2. Basic io_uring Loader

The loader implementation provides:
- **File opening** using tokio-uring File API
- **Header reading** to parse format metadata
- **Metadata parsing** from JSON section
- **Tensor loading** with async read operations at specific offsets
- **Buffer management** with proper alignment
- **Error handling** for I/O operations and format validation

### 3. Simple Converter

The converter implementation handles:
- **Safetensors parsing** to extract tensor metadata and data
- **Alignment calculations** to ensure 64-byte boundaries
- **Metadata generation** with tensor information and I/O hints
- **Binary layout** with proper padding between sections
- **File writing** using standard async I/O for the conversion process

### 4. Basic Error Types

The error handling covers:
- **I/O errors** from file operations
- **JSON parsing errors** from metadata deserialization
- **Safetensors errors** from format conversion
- **Tensor lookup errors** for missing tensors
- **Format validation errors** for corrupted files

## MVP Testing Strategy

### Basic Functionality Test

Test workflow:
1. **Create test safetensors file** with dummy tensor data
2. **Convert to TensorStore format** using the converter
3. **Load tensors using io_uring** with the async loader
4. **Verify functionality** by checking tensor list and data integrity

### Performance Comparison Benchmark

```rust
// benches/comparison.rs
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn benchmark_loading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Primary comparison: io_uring vs standard async I/O
    c.bench_function("safetensors_tokio_fs", |b| {
        b.to_async(&rt).iter(|| async {
            // Baseline: Load safetensors using tokio::fs
            load_safetensors_with_tokio_fs("test.safetensors").await
        })
    });

    c.bench_function("safetensors_tokio_uring", |b| {
        b.to_async(&rt).iter(|| async {
            // Test: Load safetensors using tokio-uring
            load_safetensors_with_uring("test.safetensors").await
        })
    });

    // Educational comparison: custom format
    c.bench_function("tensorstore_tokio_uring", |b| {
        b.to_async(&rt).iter(|| async {
            // Learning: Load custom format using tokio-uring
            load_tensorstore_with_uring("test.tensorstore").await
        })
    });
}

criterion_group!(benches, benchmark_loading);
criterion_main!(benches);
```

## MVP Development Plan

### Week 1: Core Implementation
1. **Day 1-2**: Set up project structure and dependencies
2. **Day 3-4**: Implement basic TensorStore format
3. **Day 5-7**: Implement io_uring loader

### Week 2: Testing and Validation
1. **Day 1-2**: Implement safetensors converter
2. **Day 3-4**: Create basic tests and benchmarks
3. **Day 5-7**: Performance testing and validation

### Success Metrics for MVP
- **Primary Goal**: Show measurable improvement of tokio-uring vs tokio::fs loading safetensors (target: >20% faster)
- **Secondary Goal**: Successfully implement custom TensorStore format (learning exercise)
- **Reliability**: Basic tests pass consistently

## Decision Points

### Go/No-Go Criteria
1. **Technical feasibility**: MVP works on target hardware
2. **Performance improvement**: Shows meaningful gains over baseline
3. **Development velocity**: Can implement core features within timeframe

### If MVP Succeeds
- Proceed with full TensorStore implementation
- Add advanced features (NUMA, multi-GPU, etc.)
- Implement production-grade error handling

### If MVP Fails
- Analyze bottlenecks and performance characteristics
- Consider alternative approaches (different I/O strategies)
- Document findings for future reference

This MVP strategy focuses on validating the core concept with minimal complexity, allowing for quick iteration and clear go/no-go decisions.

## Post-MVP: Next Steps if Successful

### Phase 3: Full Implementation (Weeks 4-12)

If the MVP demonstrates >20% performance improvement and validates the core approach, proceed with full implementation:

#### 3.1 Production TensorStore Format
- Enhanced format with compression integration
- Chunk-based storage for large tensors
- Format versioning and backward compatibility
- Advanced validation and error recovery

#### 3.2 Advanced io_uring Engine
- Vectored I/O operations for batch loading
- Intelligent prefetching based on access patterns
- NUMA-aware memory allocation
- Comprehensive error handling and recovery mechanisms

#### 3.3 Multi-GPU and Scaling Support
- Concurrent model loading across multiple GPUs
- Priority-based loading scheduling
- Resource isolation between models
- Memory sharing for common layers

### Phase 4: Production Readiness (Weeks 13-16)

#### 4.1 Framework Integration
- PyTorch tensor integration
- HuggingFace Transformers adapter
- Python bindings using PyO3
- C FFI for broader compatibility

#### 4.2 Production Testing
- Comprehensive performance benchmarking
- Comparison against ServerlessLLM baseline
- Production workload validation
- Memory and CPU efficiency analysis

### Phase 5: Advanced Features (Weeks 17-24)

#### 5.1 Cross-Platform Support
- Windows IOCP backend implementation
- macOS kqueue support
- Runtime I/O backend selection

#### 5.2 Enterprise Features
- Network storage support (NFS, S3)
- Distributed loading across nodes
- Monitoring and observability integration
- Security and access control

## Post-MVP: Alternative Paths if Unsuccessful

### If Performance Gains < 20%
1. **Analyze bottlenecks**: Profile to identify where gains are lost
2. **Optimize implementation**: Focus on critical performance paths
3. **Adjust targets**: Consider lower but still meaningful improvements
4. **Hybrid approach**: Combine io_uring with threading for optimal performance

### If Technical Issues Block Progress
1. **Fallback strategy**: Implement tokio::fs fallback for compatibility
2. **Alternative runtimes**: Evaluate monoio or glommio as alternatives
3. **Scope reduction**: Focus on specific use cases where io_uring excels
4. **Documentation**: Document findings for future io_uring adoption

### If Ecosystem Issues Arise
1. **Fork tokio-uring**: Take ownership of maintenance if needed
2. **Contribute upstream**: Fix issues and contribute back to ecosystem
3. **Build coalition**: Work with other users to share maintenance burden
4. **Timing consideration**: Wait for ecosystem maturity if needed

This staged approach ensures that resources are invested incrementally based on validated success at each phase.