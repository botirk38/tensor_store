"""Pytest configuration for benchmarks: CLI options and fixtures."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--model-id",
        action="store",
        required=True,
        help="HuggingFace model ID (e.g., gpt2, Qwen/Qwen2-0.5B)",
    )
    parser.addoption(
        "--fixtures-dir",
        action="store",
        default=None,
        help="Directory to store downloaded models and artifacts",
    )


@pytest.fixture(scope="session")
def model_id(request):
    """Model ID from command line."""
    return request.config.getoption("--model-id")


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory, request):
    """Base directory for model downloads and cached artifacts."""
    user_dir = request.config.getoption("--fixtures-dir")
    if user_dir:
        return Path(user_dir)
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def model_descriptor(model_id, fixtures_dir):
    """Full metadata for the benchmark model."""
    from benchmarks.fixtures import get_model_descriptor

    return get_model_descriptor(model_id, fixtures_dir)


@pytest.fixture(scope="session")
def safetensors_files(model_descriptor):
    """List of all safetensors shard paths for the model."""
    return model_descriptor.safetensors_files


@pytest.fixture(scope="session")
def safetensors_path(model_descriptor):
    """Path to the first (or only) safetensors file."""
    return str(model_descriptor.safetensors_files[0])


@pytest.fixture(scope="session")
def serverlessllm_dir(model_id, fixtures_dir):
    """Path to a ServerlessLLM artifact for the model.

    Uses the size-based heuristic for partition count.
    Only works for single-shard models. Skips for multi-shard models.
    """
    from benchmarks.fixtures import get_or_build_serverlessllm

    try:
        sllm_dir, _ = get_or_build_serverlessllm(model_id, fixtures_dir)
        return str(sllm_dir)
    except ValueError as e:
        pytest.skip(str(e))
