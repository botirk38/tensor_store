"""SafeTensors benchmarks: native file baselines vs tensor_store directory loads."""

from safetensors.torch import load_file as safetensors_load_file

from benchmarks.fixtures import (
    drop_page_cache,
    drop_page_cache_for_shards,
    touch_tensor,
)
from tensor_store_py._tensor_store_rust import (
    load_safetensors,
    load_safetensors_async,
    load_safetensors_sync,
    open_safetensors,
)


def _load_all_files(files, loader_fn):
    total_count = 0
    for path in files:
        result = loader_fn(path)
        total_count += sum(touch_tensor(t) for t in result.values())
    return total_count


def _load_dir(path, loader_fn):
    result = loader_fn(path)
    return sum(touch_tensor(t) for t in result.values())


def _open_dir(path, open_fn):
    handle = open_fn(path)
    total_count = 0
    for k in handle.keys():
        total_count += touch_tensor(handle.get_tensor(k))
    return total_count


def test_load_native_warm(benchmark, safetensors_files):
    """Benchmark native safetensors.torch.load_file across all shards (warm cache)."""

    def run():
        return _load_all_files(safetensors_files, safetensors_load_file)

    result = benchmark(run)
    assert result > 0


def test_load_native_cold(benchmark, safetensors_files):
    """Benchmark native safetensors.torch.load_file across all shards (cold cache)."""

    def run():
        drop_page_cache_for_shards(safetensors_files)
        return _load_all_files(safetensors_files, safetensors_load_file)

    result = benchmark(run)
    assert result > 0


def test_load_tensorstore_sync_warm(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors_sync for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_load_tensorstore_sync_cold(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors_sync for the full model directory."""

    def run():
        drop_page_cache(safetensors_dir)
        return _load_dir(safetensors_dir, load_safetensors_sync)

    result = benchmark(run)
    assert result > 0


def test_load_async_warm(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors_async for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors_async)

    result = benchmark(run)
    assert result > 0


def test_load_async_cold(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors_async for the full model directory."""

    def run():
        drop_page_cache(safetensors_dir)
        return _load_dir(safetensors_dir, load_safetensors_async)

    result = benchmark(run)
    assert result > 0


def test_load_default_warm(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors default backend for the full model directory."""

    def run():
        return _load_dir(safetensors_dir, load_safetensors)

    result = benchmark(run)
    assert result > 0


def test_load_default_cold(benchmark, safetensors_dir):
    """Benchmark tensor_store load_safetensors default backend for the full model directory."""

    def run():
        drop_page_cache(safetensors_dir)
        return _load_dir(safetensors_dir, load_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_warm(benchmark, safetensors_dir):
    """Benchmark open_safetensors + get_tensor for the full model directory."""

    def run():
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_cold(benchmark, safetensors_dir):
    """Benchmark open_safetensors + get_tensor for the full model directory."""

    def run():
        drop_page_cache(safetensors_dir)
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_default_warm(benchmark, safetensors_dir):
    """Benchmark open_safetensors (default) + get_tensor for the full model directory."""

    def run():
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0


def test_open_get_tensor_default_cold(benchmark, safetensors_dir):
    """Benchmark open_safetensors (default) + get_tensor for the full model directory."""

    def run():
        drop_page_cache(safetensors_dir)
        return _open_dir(safetensors_dir, open_safetensors)

    result = benchmark(run)
    assert result > 0
