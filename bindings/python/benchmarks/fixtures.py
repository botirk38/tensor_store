"""Benchmark fixtures: real-model loading, conversion, and utilities."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tensor_store_py._tensor_store_rust import convert_safetensors_to_serverlessllm


def partition_count(total_bytes: int) -> int:
    """Choose partition count based on total model size.

    This heuristic balances parallelism with overhead:
    - < 512 MiB  -> 1 partition
    - < 2 GiB    -> 2 partitions
    - < 8 GiB    -> 4 partitions
    - < 24 GiB   -> 8 partitions
    - < 64 GiB   -> 16 partitions
    - otherwise  -> 32 partitions

    Capped at min(32, cpu_count * 2).
    """
    cpu_count = os.cpu_count() or 4
    max_parts = min(32, cpu_count * 2)

    if total_bytes < 512 * 1024**2:
        parts = 1
    elif total_bytes < 2 * 1024**3:
        parts = 2
    elif total_bytes < 8 * 1024**3:
        parts = 4
    elif total_bytes < 24 * 1024**3:
        parts = 8
    elif total_bytes < 64 * 1024**3:
        parts = 16
    else:
        parts = 32

    return min(parts, max_parts)


def _cache_root() -> Path:
    return Path(__file__).parent / ".cache"


def repo_dir_name(repo_id: str) -> str:
    """Convert a HuggingFace repo ID into a filesystem-friendly directory name."""
    return repo_id.replace("/", "-").lower()


@dataclass
class ModelDescriptor:
    """Metadata about a real HuggingFace model for benchmarking."""
    model_id: str
    safetensors_files: list[Path]
    shard_count: int
    total_bytes: int
    partition_count: int


def get_model_safetensors(model_name: str, fixtures_dir: Path) -> list[Path]:
    """Download model from HuggingFace and return all safetensors shard paths.

    Args:
        model_name: HuggingFace model ID (e.g., 'gpt2', 'Qwen/Qwen2-0.5B')
        fixtures_dir: Directory to store downloaded models

    Returns:
        List of paths to all .safetensors shard files, sorted by name for determinism
    """
    from huggingface_hub import snapshot_download

    fixtures_dir = Path(fixtures_dir)
    model_dir = fixtures_dir / repo_dir_name(model_name)

    if not model_dir.exists():
        print(f"Downloading {model_name} to {model_dir}...")
        download_dir = snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            allow_patterns=["*.safetensors"],
            ignore_patterns=[
                "*.bin",
                "*.msgpack",
                "*.h5",
                "*.pb",
                "*.onnx",
                "*.tflite",
            ],
        )
    else:
        download_dir = str(model_dir)

    safetensors_files = sorted(
        Path(download_dir).glob("*.safetensors"),
        key=lambda f: f.name,
    )

    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {download_dir} for model {model_name}"
        )

    return safetensors_files


def get_model_descriptor(model_name: str, fixtures_dir: Path) -> ModelDescriptor:
    """Get full metadata for a real model.

    Returns a ModelDescriptor with file list, shard count, total size, and
    recommended partition count based on the size heuristic.
    """
    files = get_model_safetensors(model_name, fixtures_dir)
    total_bytes = sum(f.stat().st_size for f in files)
    parts = partition_count(total_bytes)

    return ModelDescriptor(
        model_id=model_name,
        safetensors_files=files,
        shard_count=len(files),
        total_bytes=total_bytes,
        partition_count=parts,
    )


def ensure_serverlessllm_artifact(
    model_name: str,
    safetensors_files: list[Path],
    revision: str | None = None,
    fixtures_dir: Path | None = None,
) -> Path:
    """Ensure a ServerlessLLM artifact exists for the model, converting if needed.

    Uses the source file path + revision + partition count as the cache key,
    so repeated runs with the same inputs reuse cached artifacts.

    Args:
        model_name: HuggingFace model ID
        safetensors_files: List of source safetensors shard paths
        revision: Optional revision (defaults to 'main')
        fixtures_dir: Optional fixtures directory for caching

    Returns:
        Path to the ServerlessLLM artifact directory
    """
    if fixtures_dir is None:
        fixtures_dir = _cache_root() / "serverlessllm"
    else:
        fixtures_dir = Path(fixtures_dir) / "serverlessllm"

    total_bytes = sum(f.stat().st_size for f in safetensors_files)
    partition_count_override = partition_count(total_bytes)

    if len(safetensors_files) != 1:
        raise ValueError(
            f"ServerlessLLM conversion requires a single safetensors shard, "
            f"got {len(safetensors_files)} shards for {model_name}. "
            f"Shard-aware ServerlessLLM conversion not yet supported."
        )

    source_path = safetensors_files[0]
    revision_str = revision or "main"

    key_input = f"{source_path}:{revision_str}:{partition_count_override}:v1"
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()[:16]
    out_dir = fixtures_dir / repo_dir_name(model_name) / cache_key

    index_path = out_dir / "tensor_index.json"
    if index_path.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Converting {model_name} to ServerlessLLM "
        f"(partition_count={partition_count_override})..."
    )

    convert_safetensors_to_serverlessllm(
        str(source_path),
        str(out_dir),
        partition_count_override,
    )

    return out_dir


def get_or_build_serverlessllm(
    model_name: str,
    fixtures_dir: Path,
) -> tuple[Path, ModelDescriptor]:
    """Convenience helper to get both the ServerlessLLM artifact and model metadata.

    Returns:
        Tuple of (serverlessllm_dir, model_descriptor)
    """
    desc = get_model_descriptor(model_name, fixtures_dir)

    if desc.shard_count > 1:
        raise ValueError(
            f"Model {model_name} has {desc.shard_count} shards. "
            f"ServerlessLLM requires single-shard models. "
            f"Consider using bench_safetensors.py instead for multi-shard models."
        )

    sllm_dir = ensure_serverlessllm_artifact(
        model_name,
        desc.safetensors_files,
        fixtures_dir=fixtures_dir,
    )

    return sllm_dir, desc


def touch_tensor(t: torch.Tensor) -> float:
    """Force read of all pages (e.g. for mmap-backed tensors). Returns sum for sanity."""
    return t.sum().item()


def drop_page_cache(path: str | Path) -> None:
    """Hint kernel to drop page cache for path. Unix-only; no-op on Windows."""
    path = Path(path)
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        if path.is_file():
            with open(path, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        else:
            for f in path.glob("*.data_*"):
                with open(f, "rb") as fp:
                    os.posix_fadvise(fp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def drop_page_cache_for_shards(files: list[Path]) -> None:
    """Drop page cache for multiple shard files."""
    for f in files:
        drop_page_cache(f)
