# IOPOLL Configuration for Direct I/O

## Overview
To enable O_DIRECT and IOPOLL for low-latency tensor loading, the tokio-uring runtime must be configured with custom builder settings.

## Implementation

### Runtime Setup with IOPOLL
```rust
use tokio_uring::{builder, uring_builder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    builder()
        .uring_builder(
            &uring_builder()
                .setup_iopoll()  // Enable polling for low latency
        )
        .start(async {
            // Your tensor loading code here
            Ok(())
        })
}
```

### Requirements
1. **Kernel**: Linux 5.11+ (for stable IOPOLL)
2. **File System**: Must support polling (e.g., ext4, xfs with proper flags)
3. **Files**: Must be opened with `O_DIRECT` flag
4. **Hardware**: NVMe SSDs recommended for maximum benefit

### Performance Impact
- **Latency**: Reduced by polling instead of IRQ
- **CPU**: Higher CPU usage due to busy-waiting
- **Throughput**: Better for small, frequent I/O operations

### When to Use
- ✅ Low-latency model serving with frequent small reads
- ✅ NVMe storage with high IOPS
- ✅ Dedicated CPU cores available for polling
- ❌ Batch loading where latency isn't critical
- ❌ Limited CPU resources
- ❌ Network storage (not supported)

### Integration with TensorStore
The `load_safetensors_fixed()` function can benefit from IOPOLL when:
1. Runtime is configured with `.setup_iopoll()`
2. Files are opened with O_DIRECT (requires custom file opening, not yet supported in tokio-uring)
3. FixedBufPool is used for pre-registered buffers

### Future Work
- Add O_DIRECT file opening support (requires forking tokio-uring or using raw io_uring)
- Benchmark IOPOLL vs standard mode for different workloads
- Implement adaptive polling based on workload characteristics
