"""SafeTensors benchmarks: load_file across all shards for real models.

Benchmarks native safetensors.torch.load_file vs tensor_store implementations.
Supports multi-shard models by loading all shards and aggregating full-model cost.
"""

from safetensors.torch import load_file as safetensors_load_file

from benchmarks.fixtures import drop_page_cache_for_shards, touch_tensor
from tensor_store_py._tensor_store_rust import (
    load_safetensors,
    load_safetensors_mmap,
    load_safetensors_sync,
    open_safetensors,
    open_safetensors_mmap,
    open_safetensors_sync,
)


def _load_all_shards_load_file(files, loader_fn):
    """Load all shards and touch every tensor. Returns total tensor count."""
    total_count = 0
    for path in files:
        result = loader_fn(path)
        total_count += sum(touch_tensor(t) for t in result.values())
    return total_count


def _open_get_all_shards(files, open_fn):
    """Open all shards and get every tensor. Returns total tensor count."""
    total_count = 0
    for path in files:
        handle = open_fn(path)
        for k in handle.keys():
            total_count += touch_tensor(handle.get_tensor(k))
    return total_count


def test_load_native_warm(benchmark, safetensors_files):
    """Benchmark native safetensors.torch.load_file across all shards (warm cache)."""

    def run():
        return _load_all_shards_load_file(safetensors_files, safetensors_load_file)

    result = benchmark(run)
    assert result > 0


def test_load_native_cold(benchmark, safetensors_files):
    """Benchmark native safetensors.torch.load_file across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _load_all_shards_load_file(safetensors_files, safetensors_load_file)

    result = benchmark(run)
    assert result > 0


def test_load_tensorstore_sync_warm(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors_sync across all shards (warm cache)."""

    def run():
        return _load_all_shards_load_file(safetensors_files, load_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_load_tensorstore_sync_cold(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors_sync across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _load_all_shards_load_file(safetensors_files, load_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_load_mmap_warm(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors_mmap across all shards (warm cache)."""

    def run():
        return _load_all_shards_load_file(safetensors_files, load_safetensors_mmap)

    result = benchmark(run)
    assert result > 0


def test_load_mmap_cold(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors_mmap across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _load_all_shards_load_file(safetensors_files, load_safetensors_mmap)

    result = benchmark(run)
    assert result > 0


def test_load_default_warm(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors default backend across all shards (warm cache)."""

    def run():
        return _load_all_shards_load_file(safetensors_files, load_safetensors)

    result = benchmark(run)
    assert result > 0


def test_load_default_cold(benchmark, safetensors_files):
    """Benchmark tensor_store load_safetensors default backend across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _load_all_shards_load_file(safetensors_files, load_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_sync_warm(benchmark, safetensors_files):
    """Benchmark open_safetensors_sync + get_tensor across all shards (warm)."""

    def run():
        return _open_get_all_shards(safetensors_files, open_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_sync_cold(benchmark, safetensors_files):
    """Benchmark open_safetensors_sync + get_tensor across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _open_get_all_shards(safetensors_files, open_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_mmap_warm(benchmark, safetensors_files):
    """Benchmark open_safetensors_mmap + get_tensor across all shards (warm)."""

    def run():
        return _open_get_all_shards(safetensors_files, open_safetensors_mmap)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_mmap_cold(benchmark, safetensors_files):
    """Benchmark open_safetensors_mmap + get_tensor across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _open_get_all_shards(safetensors_files, open_safetensors_mmap)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_default_warm(benchmark, safetensors_files):
    """Benchmark open_safetensors (default) + get_tensor across all shards (warm)."""

    def run():
        return _open_get_all_shards(safetensors_files, open_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_default_cold(benchmark, safetensors_files):
    """Benchmark open_safetensors (default) + get_tensor across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _open_get_all_shards(safetensors_files, open_safetensors)

    result = benchmark(run)
    assert result > 0
