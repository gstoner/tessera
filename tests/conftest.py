from pathlib import Path

import pytest

from tests._support.environment import (
    CompilerToolchain,
    python_subprocess_environment,
)
from tests._support.policy import MARKERS


def pytest_configure(config):
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
