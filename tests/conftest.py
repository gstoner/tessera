from collections import Counter
from pathlib import Path
import pytest

from tests._support.environment import (
    CompilerToolchain,
    ensure_cuda_bin_on_path,
    is_wsl,
    python_subprocess_environment,
)
from tests._support.policy import MARKERS
from tests._support.compiler_ownership import (
    compiler_platform_skip_reason,
    compiler_test_required_platform,
    selected_compiler_test_platform,
)


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


def pytest_collection_modifyitems(config, items):
    """Skip compiler proofs whose declared owner is a different system."""

    try:
        selected = selected_compiler_test_platform()
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if selected is None:
        return
    for item in items:
        if item.get_closest_marker("compiler_tool") is None:
            continue
        required = compiler_test_required_platform(item)
        if required is None or required[0] == selected:
            continue
        item.add_marker(
            pytest.mark.skip(reason=compiler_platform_skip_reason(required[1]))
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Make cross-platform compiler skips visible without treating them as passes."""

    counts: Counter[str] = Counter()
    prefix = "compiler-test platform mismatch: requires "
    for report in terminalreporter.stats.get("skipped", ()):
        longrepr = getattr(report, "longrepr", None)
        reason = (
            longrepr[2]
            if isinstance(longrepr, tuple) and len(longrepr) == 3
            else str(longrepr)
        )
        reason = reason.removeprefix("Skipped: ")
        if not reason.startswith(prefix):
            continue
        required = reason.removeprefix(prefix).split(";", maxsplit=1)[0]
        counts[required] += 1
    if not counts:
        return
    terminalreporter.write_sep("-", "compiler tests skipped for other systems")
    for platform, count in sorted(counts.items()):
        noun = "test" if count == 1 else "tests"
        terminalreporter.write_line(
            f"{count} {noun} skipped: requires {platform}; run on a {platform} system"
        )


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
