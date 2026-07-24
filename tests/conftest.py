from pathlib import Path
import pytest

from tests._support.environment import (
    CompilerToolchain,
    ensure_cuda_bin_on_path,
    is_wsl,
    python_subprocess_environment,
)
from tests._support.policy import MARKERS


def pytest_configure(config):
    ensure_cuda_bin_on_path()
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture(scope="session")
def compiler_toolchain() -> CompilerToolchain:
    return CompilerToolchain.discover()


@pytest.fixture
def python_subprocess_env() -> dict[str, str]:
    return python_subprocess_environment()


@pytest.fixture
def apple_gpu_jit_runtime() -> None:
    """Gate JIT integration tests on the shared Apple runtime ABI."""
    from tests._support.apple import require_apple_gpu_jit_runtime

    require_apple_gpu_jit_runtime()


@pytest.fixture
def apple_accelerate() -> None:
    from tests._support.apple import require_apple_accelerate

    require_apple_accelerate()


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    return "archive" in path.parts


def pytest_runtest_setup(item):
    """Apply centralized host and Apple-device execution boundaries.

    ``hardware_apple_gpu`` means that a test requires an actual Metal device;
    it is not a synonym for an Apple-flavoured reference test.  Centralising
    this boundary gives every marked test the same explicit skip reason and
    prevents individual tests from quietly choosing a NumPy fallback.
    """
    if item.get_closest_marker("native_host") is not None and is_wsl():
        pytest.skip(
            "native-host test skipped under WSL; this test deliberately aborts "
            "a compiler child process"
        )
    if item.get_closest_marker("metal4") is not None:
        from tests._support.apple import require_apple_metal4

        require_apple_metal4()
        return
    if item.get_closest_marker("hardware_apple_gpu") is not None:
        from tests._support.apple import require_apple_metal

        require_apple_metal()
