"""ServerlessLLM benchmarks for real models.

Benchmarks tensor_store ServerlessLLM loading with different backends (default, sync, mmap).
Partition count uses the shared size-based heuristic.
"""

from benchmarks.fixtures import drop_page_cache, touch_tensor
from tensor_store_py._tensor_store_rust import (
    load_serverlessllm,
    load_serverlessllm_sync,
    open_serverlessllm,
    open_serverlessllm_mmap,
    open_serverlessllm_sync,
)


def test_load_sync_warm(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_sync (warm cache)."""

    def run():
        result = load_serverlessllm_sync(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_sync_cold(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm_sync (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        result = load_serverlessllm_sync(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_warm(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm (default backend, warm cache)."""

    def run():
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_load_default_cold(benchmark, serverlessllm_dir):
    """Benchmark load_serverlessllm (default backend, cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        result = load_serverlessllm(serverlessllm_dir)
        sum(touch_tensor(t) for t in result.values())
        return result

    result = benchmark(run)
    assert len(result) > 0


def test_open_get_tensor_sync_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_sync + get_tensor (warm)."""

    def run():
        handle = open_serverlessllm_sync(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_sync_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_sync + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm_sync(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_mmap + get_tensor (warm)."""

    def run():
        handle = open_serverlessllm_mmap(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_mmap_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm_mmap + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm_mmap(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default_warm(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm (default) + get_tensor (warm)."""

    def run():
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)


def test_open_get_tensor_default_cold(benchmark, serverlessllm_dir):
    """Benchmark open_serverlessllm (default) + get_tensor (cold cache)."""

    def run():
        drop_page_cache(serverlessllm_dir)
        handle = open_serverlessllm(serverlessllm_dir)
        for k in handle.keys():
            touch_tensor(handle.get_tensor(k))

    benchmark(run)
