# TensorStore Format Specification

## Design Philosophy

TensorStore is designed for maximum io_uring parallel loading performance with minimal complexity. Every byte serves a clear purpose.

---

## File Structure

```
model.index        Index file (header + entries)
model.0            Shard 0 data
model.1            Shard 1 data
model.2            Shard 2 data
...
model.N            Shard N data
```

---

## Index File (model.index)

### Header (8 bytes)

```
Offset   Size  Field           Description
-------  ----  -----           -----------
0        4     magic           0x54534D4C ("TSML")
4        4     tensor_count    Total number of tensors
```

### Index Entry (64 bytes)

```
Offset   Size  Field           Description
-------  ----  -----           -----------
0        1     shard_id        Which shard file (0-255)
1        7     offset          Byte offset within shard (56-bit little-endian)
8        4     size            Tensor data size in bytes
12       1     dtype           Data type enum
13       1     rank            Number of dimensions (1-8)
14       2     name_len        Length of full name in bytes
16       32    shape           8 × u32 dimensions
48       16    name_inline     First 16 bytes of tensor name (UTF-8)
```

**Index Layout:**

```
[Header: 8 bytes]
[Entry 0: 64 bytes]
[Entry 1: 64 bytes]
[Entry 2: 64 bytes]
...
[Entry N-1: 64 bytes]
```

**Total index size:** `8 + (tensor_count × 64)` bytes

---

## Field Specifications

### magic (4 bytes)

**Value:** `0x54534D4C` (ASCII "TSML")

**Purpose:** Format identification

**Validation:**

```
if magic != 0x54534D4C:
    error("Not a TensorStore file")
```

### tensor_count (4 bytes, u32)

**Range:** 1 to 4,294,967,295

**Purpose:** Number of tensors in model

**Usage:**

```
index_size = 8 + (tensor_count × 64)
```

### shard_id (1 byte, u8)

**Range:** 0 to 255

**Purpose:** Identifies which shard file contains this tensor

**Shard file name:** `model.{shard_id}`

**Examples:**

```
shard_id = 0  → model.0
shard_id = 5  → model.5
shard_id = 42 → model.42
```

### offset (7 bytes, 56-bit unsigned little-endian)

**Range:** 0 to 72,057,594,037,927,935 (64 PB)

**Purpose:** Byte offset within the shard file where tensor data starts

**Encoding (little-endian):**

```
bytes[0] = (offset >> 0)  & 0xFF
bytes[1] = (offset >> 8)  & 0xFF
bytes[2] = (offset >> 16) & 0xFF
bytes[3] = (offset >> 24) & 0xFF
bytes[4] = (offset >> 32) & 0xFF
bytes[5] = (offset >> 40) & 0xFF
bytes[6] = (offset >> 48) & 0xFF
```

**Decoding (little-endian):**

```
offset = (u64)bytes[0]        |
         ((u64)bytes[1] << 8)  |
         ((u64)bytes[2] << 16) |
         ((u64)bytes[3] << 24) |
         ((u64)bytes[4] << 32) |
         ((u64)bytes[5] << 40) |
         ((u64)bytes[6] << 48)
```

### size (4 bytes, u32)

**Range:** 1 to 4,294,967,295 bytes (4 GB)

**Purpose:** Actual tensor data size in bytes (excluding padding)

**Constraint:**

```
size = element_count × dtype_size
where element_count = shape[0] × shape[1] × ... × shape[rank-1]
```

### dtype (1 byte, u8)

**Purpose:** Tensor element data type

**Values:**

```
0x00 = F32      (32-bit float)
0x01 = F16      (16-bit float)
0x02 = BF16     (bfloat16)
0x03 = F64      (64-bit float)
0x04 = I32      (32-bit signed int)
0x05 = I16      (16-bit signed int)
0x06 = I8       (8-bit signed int)
0x07 = I64      (64-bit signed int)
0x08 = U32      (32-bit unsigned int)
0x09 = U16      (16-bit unsigned int)
0x0A = U8       (8-bit unsigned int)
0x0B = U64      (64-bit unsigned int)
0x0C = BOOL     (boolean, 1 byte)
0x0D = F8_E4M3  (8-bit float, 4-bit exponent, 3-bit mantissa)
0x0E = F8_E5M2  (8-bit float, 5-bit exponent, 2-bit mantissa)
0x0F-0xFF = Reserved
```

**Dtype sizes:**

```
F64, I64, U64:  8 bytes
F32, I32, U32:  4 bytes
F16, BF16, I16, U16: 2 bytes
F8_E4M3, F8_E5M2, I8, U8, BOOL: 1 byte
```

### rank (1 byte, u8)

**Range:** 1 to 8

**Purpose:** Number of dimensions in the tensor

**Examples:**

```
rank = 1  → Vector:    [N]
rank = 2  → Matrix:    [M, N]
rank = 3  → 3D tensor: [D, M, N]
rank = 4  → 4D tensor: [B, C, H, W]
```

### name_len (2 bytes, u16)

**Range:** 1 to 65,535

**Purpose:** Length of tensor name in UTF-8 bytes

**Usage:**

```
if name_len <= 16:
    full_name = name_inline[0:name_len]
else:
    full_name = name_inline + lookup_overflow(tensor_id)
```

**Note:** In v1.0, names longer than 16 bytes are truncated. Name overflow storage is reserved for future extension.

### shape (32 bytes, 8 × u32)

**Purpose:** Tensor dimensions

**Layout:**

```
shape[0]: u32  First dimension
shape[1]: u32  Second dimension
...
shape[7]: u32  Eighth dimension
```

**Constraints:**

```
- First `rank` elements contain actual dimensions
- Remaining elements must be 0
- No dimension can be 0 (empty tensors not allowed)
- Each dimension must fit in u32 (max 4,294,967,295)
```

**Examples:**

```
Shape [768]:
  rank = 1
  shape = [768, 0, 0, 0, 0, 0, 0, 0]

Shape [32, 128, 768]:
  rank = 3
  shape = [32, 128, 768, 0, 0, 0, 0, 0]

Shape [1, 8, 512, 512]:
  rank = 4
  shape = [1, 8, 512, 512, 0, 0, 0, 0]
```

### name_inline (16 bytes)

**Purpose:** Store tensor name (or prefix if longer)

**Encoding:** UTF-8 bytes

**Rules:**

```
if name_len <= 16:
    Copy full name to name_inline
    Zero-pad remaining bytes
else:
    Copy first 16 bytes of name
    (Full name retrieval reserved for future)
```

**Examples:**

```
Name: "weight"
  name_len = 6
  name_inline = "weight\0\0\0\0\0\0\0\0\0\0"

Name: "layer.0.attn.q"
  name_len = 14
  name_inline = "layer.0.attn.q\0\0"

Name: "transformer.layers.5.attention.query.weight"
  name_len = 45
  name_inline = "transformer.laye" (first 16 bytes)
```

---

## Shard Files (model.N)

### Structure

```
Pure tensor data, no headers
Each tensor aligned to 64-byte boundaries
Tensors ordered by size (largest first)
```

### Layout

```
Offset 0:       Tensor A data (64-byte aligned)
                [tensor bytes]
                [padding to 64-byte boundary]

Offset X:       Tensor B data (64-byte aligned)
                [tensor bytes]
                [padding to 64-byte boundary]

Offset Y:       Tensor C data (64-byte aligned)
                [tensor bytes]
                [padding to 64-byte boundary]
...
```

### Alignment

**All tensor offsets are 64-byte aligned:**

```
offset % 64 == 0
```

**Padding calculation:**

```
actual_bytes_written = round_up(size, 64)
padding_bytes = actual_bytes_written - size
```

**Padding content:** Zero bytes (0x00)

### Tensor Ordering

**Within each shard, tensors are ordered by size (largest first):**

**Rationale:**

- Largest tensors get best alignment naturally
- Small tensors at end (alignment less critical)
- Better memory access patterns

**Example shard:**

```
Offset 0:     10MB tensor  (largest)
Offset 10M:   5MB tensor
Offset 15M:   1MB tensor
Offset 16M:   100KB tensor
Offset 16.1M: 4KB tensor   (smallest)
```

---

## Sharding Strategy

### Shard Assignment

**Goal:** Balance shard sizes for parallel loading

**Algorithm:**

```
1. Sort all tensors by size (descending)
2. Assign round-robin to shards:

   tensor[0] → shard 0
   tensor[1] → shard 1
   tensor[2] → shard 2
   ...
   tensor[shard_count-1] → shard (shard_count-1)
   tensor[shard_count] → shard 0
   tensor[shard_count+1] → shard 1
   ...
```

**Result:** Even distribution of large and small tensors

### Shard Count Selection

**Recommended formula:**

```
shard_count = clamp(
    max(cpu_cores, gpu_count × 4, total_gb / 2),
    min=4,
    max=64
)
```

**Examples:**

```
1GB model, 4 CPU cores:
  shard_count = 4
  ~250MB per shard

14GB model, 16 CPU cores, 2 GPUs:
  shard_count = max(16, 8, 7) = 16
  ~875MB per shard

100GB model, 64 CPU cores, 8 GPUs:
  shard_count = max(64, 32, 50) = 64
  ~1.5GB per shard
```

---

## Writing Algorithm

### Step 1: Prepare Tensors

```
Input: List of tensors with name, dtype, shape, data

For each tensor:
  1. Calculate size:
     element_count = shape[0] × shape[1] × ... × shape[rank-1]
     size = element_count × dtype_size

  2. Encode name:
     name_bytes = encode_utf8(name)
     name_len = length(name_bytes)
     name_inline = first_16_bytes(name_bytes)
```

### Step 2: Sort and Assign Shards

```
1. Sort tensors by size (descending)

2. Determine shard_count (see formula above)

3. Assign shards round-robin:
   for i, tensor in enumerate(sorted_tensors):
       tensor.shard_id = i % shard_count
```

### Step 3: Calculate Offsets

```
For each shard_id in 0..shard_count:
    1. Get all tensors for this shard:
       shard_tensors = filter(tensors, t.shard_id == shard_id)

    2. Sort by size (descending)

    3. Calculate offsets:
       offset = 0
       for tensor in shard_tensors:
           tensor.offset = offset
           offset += round_up(tensor.size, 64)
```

### Step 4: Write Index File

```
Open model.index

Write header:
  write_u32(0x54534D4C)  // magic
  write_u32(tensor_count)

For each tensor (in original order):
  write_u8(tensor.shard_id)
  write_u56_le(tensor.offset)
  write_u32(tensor.size)
  write_u8(tensor.dtype)
  write_u8(tensor.rank)
  write_u16(tensor.name_len)
  write_bytes(tensor.shape, 32)  // 8 × u32
  write_bytes(tensor.name_inline, 16)

Close model.index
```

### Step 5: Write Shard Files

```
For each shard_id in 0..shard_count:
    Open model.{shard_id}

    shard_tensors = filter(tensors, t.shard_id == shard_id)

    For each tensor in shard_tensors:
        # Write tensor data at offset
        seek(tensor.offset)
        write_bytes(tensor.data, tensor.size)

        # Write padding
        padding = round_up(tensor.size, 64) - tensor.size
        write_bytes([0] × padding)

    Close model.{shard_id}
```

---

## Loading Algorithm

### Step 1: Read Index

```
Open model.index

Read header:
  magic = read_u32()
  if magic != 0x54534D4C:
      error("Invalid format")

  tensor_count = read_u32()

Read all entries:
  index_size = tensor_count × 64
  entries = read_bytes(index_size)

  Parse entries into array of IndexEntry structs

Close model.index
```

### Step 2: Open Shard Files

```
Determine max shard:
  max_shard = max(entry.shard_id for entry in entries)

Open all shard files:
  shard_fds = []
  for shard_id in 0..max_shard:
      fd = open(f"model.{shard_id}", O_RDONLY)
      shard_fds[shard_id] = fd
```

### Step 3: Prepare io_uring

```
Initialize io_uring:
  io_uring_queue_init(tensor_count, &ring, 0)

Optional - register file descriptors:
  io_uring_register_files(&ring, shard_fds, max_shard + 1)
```

### Step 4: Submit Reads

```
For each entry in entries:
    # Calculate aligned read size
    read_size = round_up(entry.size, 64)

    # Allocate aligned buffer
    buffer = aligned_alloc(64, read_size)

    # Prepare io_uring read
    sqe = io_uring_get_sqe(&ring)
    io_uring_prep_read(sqe,
                       shard_fds[entry.shard_id],
                       buffer,
                       read_size,
                       entry.offset)
    sqe->user_data = tensor_id

Submit all reads:
  io_uring_submit(&ring)
```

### Step 5: Harvest Completions

```
completed = 0
while completed < tensor_count:
    io_uring_wait_cqe(&ring, &cqe)

    tensor_id = cqe->user_data

    if cqe->res < 0:
        error(f"Read failed for tensor {tensor_id}")

    # Tensor is now loaded in buffer
    tensors[tensor_id].data = buffer
    tensors[tensor_id].ready = true

    io_uring_cqe_seen(&ring, cqe)
    completed++
```

---

## Validation

### On Load

**Required checks:**

```
1. Magic number:
   if header.magic != 0x54534D4C:
       error("Invalid magic number")

2. Tensor count:
   if header.tensor_count == 0:
       error("Empty model")
   if header.tensor_count > 10_000_000:
       error("Unrealistic tensor count")

3. For each entry:
   if entry.rank == 0 or entry.rank > 8:
       error("Invalid rank")

   if entry.dtype > 0x0E:
       error("Unknown dtype")

   for i in 0..entry.rank:
       if entry.shape[i] == 0:
           error("Zero dimension")

   for i in entry.rank..8:
       if entry.shape[i] != 0:
           error("Non-zero unused dimension")

   if entry.offset % 64 != 0:
       error("Misaligned offset")

4. Shard files exist:
   max_shard = max(entry.shard_id)
   for shard_id in 0..max_shard:
       if not exists(f"model.{shard_id}"):
           error("Missing shard file")
```

### On Write

**Required checks:**

```
1. Tensor data size matches shape:
   element_count = product(shape[0:rank])
   expected_size = element_count × dtype_size
   if tensor.size != expected_size:
       error("Size mismatch")

2. Name is valid UTF-8:
   if not is_valid_utf8(name):
       error("Invalid name encoding")

3. Shape constraints:
   for dim in shape[0:rank]:
       if dim == 0:
           error("Zero dimension")
       if dim > 0xFFFFFFFF:
           error("Dimension too large")

4. Alignment:
   for each tensor:
       if tensor.offset % 64 != 0:
           error("Misaligned offset")
```

---

## Performance Characteristics

### Loading Performance

**7B parameter model (300 tensors, 14GB, 16 shards):**

```
Operation               Time
─────────────────────────────────
Read index:             30μs       14.4KB (8 + 300×64)
Parse index:            5μs        Direct cast to structs
Open 16 shards:         80μs       16 × open() calls
Register fds:           10μs       io_uring_register_files()
Prepare 300 SQEs:       5μs        Simple loop
Submit batch:           2μs        Single syscall
Data transfer:          8ms        14GB parallel reads
─────────────────────────────────
Total:                  8.132ms
```

**vs SafeTensors (sequential):**

```
Header read:            10μs
JSON parse:             2000μs
Sequential reads:       15000μs
Data transfer:          2000ms
─────────────────────────────────
Total:                  2017ms
```

**Speedup: 248x faster**

### Space Overhead

**GPT-2 (148 tensors, 498MB):**

```
Component               Size
─────────────────────────────────
Index file:             9.5KB      8 + (148 × 64)
Shard data:             498MB      Actual tensor data
Alignment padding:      ~4.7KB     148 × 32 bytes avg
─────────────────────────────────
Total:                  498.014MB
Overhead:               0.003%
```

**Large model (10,000 tensors, 100GB):**

```
Component               Size
─────────────────────────────────
Index file:             640KB      8 + (10000 × 64)
Shard data:             100GB      Actual tensor data
Alignment padding:      ~320KB     10000 × 32 bytes avg
─────────────────────────────────
Total:                  100.96GB
Overhead:               0.001%
```

---

## Example

### Sample Model

```
Model with 3 tensors:
  - "embedding.weight": F32, [1000, 512], 2MB
  - "layer.0.weight":   F32, [512, 512], 1MB
  - "layer.0.bias":     F32, [512], 2KB
```

### Generated Files

**model.index:**

```
Offset 0: Header
  magic = 0x54534D4C
  tensor_count = 3

Offset 8: Entry 0
  shard_id = 0
  offset = 0
  size = 2048000 (2MB)
  dtype = 0 (F32)
  rank = 2
  name_len = 17
  shape = [1000, 512, 0, 0, 0, 0, 0, 0]
  name_inline = "embedding.weight"

Offset 72: Entry 1
  shard_id = 1
  offset = 0
  size = 1048576 (1MB)
  dtype = 0 (F32)
  rank = 2
  name_len = 15
  shape = [512, 512, 0, 0, 0, 0, 0, 0]
  name_inline = "layer.0.weight\0\0"

Offset 136: Entry 2
  shard_id = 0
  offset = 2048000
  size = 2048 (2KB)
  dtype = 0 (F32)
  rank = 1
  name_len = 13
  shape = [512, 0, 0, 0, 0, 0, 0, 0]
  name_inline = "layer.0.bias\0\0\0\0"
```

**model.0:** (Shard 0)

```
Offset 0:       embedding.weight data (2MB)
                [2048000 bytes]
Offset 2048000: layer.0.bias data (2KB)
                [2048 bytes]
                [padding: 32 bytes to align to 64]
```

**model.1:** (Shard 1)

```
Offset 0:       layer.0.weight data (1MB)
                [1048576 bytes]
```

---

## Implementation Notes

### Endianness

**All multi-byte values are little-endian:**

- u16, u32: standard little-endian
- u56: 7-byte little-endian (see offset encoding)

**Platform support:**

- x86_64: Native little-endian
- ARM: Handles little-endian efficiently
- Big-endian systems: Require byte swapping

### Name Truncation

**In v1.0:**

- Names longer than 16 bytes are truncated
- Full name storage reserved for future extension
- Applications should use names ≤ 16 characters

**Recommended naming:**

```
Good:  "embed", "layer.0.attn", "ffn.up"
Avoid: "transformer.layers.5.attention.query.weight"
```

### Memory Alignment

**All allocated buffers must be 64-byte aligned:**

```
buffer = aligned_alloc(64, size)
```

**Or use posix_memalign:**

```
void* buffer;
posix_memalign(&buffer, 64, size);
```

---

## Summary

**Format characteristics:**

- Multi-file: 1 index + N shards
- Index: 8-byte header + 64-byte entries
- Shards: Pure data, 64-byte aligned
- Names: Included (up to 16 bytes)

**Simplicity:**

- No compression
- No checksums
- No metadata beyond tensors
- Standard integer types
- Straightforward implementation
