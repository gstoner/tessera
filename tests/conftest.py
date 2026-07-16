from pathlib import Path
import sys

import pytest

from tests._support.environment import (
    CompilerToolchain,
    ensure_cuda_bin_on_path,
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


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    return "archive" in path.parts


def pytest_runtest_setup(item):
    """Keep the Apple device lane honest and out of portable test runs.

    ``hardware_apple_gpu`` means that a test requires an actual Metal device;
    it is not a synonym for an Apple-flavoured reference test.  Centralising
    this boundary gives every marked test the same explicit skip reason and
    prevents individual tests from quietly choosing a NumPy fallback.
    """
    if item.get_closest_marker("hardware_apple_gpu") is None:
        return
    if sys.platform != "darwin":
        pytest.skip("hardware_apple_gpu requires a Darwin host")

    from tessera.runtime import DeviceTensor

    if not DeviceTensor.is_metal():
        pytest.skip(
            "hardware_apple_gpu requires an available Metal device "
            "(run the exact-device lane outside the sandbox with a fresh runtime)"
        )
