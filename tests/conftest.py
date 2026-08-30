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
from tests._support.device_accounting import (
    DeviceLedger,
    families_for_keywords,
    format_hollow_lane_report,
)

_DEVICE_LEDGER_KEY = "tessera_device_ledger"

#: Set at configure time so ``pytest_runtest_logreport`` -- which receives a
#: report and no config -- can reach the session's ledger.
_ACTIVE_LEDGER: DeviceLedger | None = None


def pytest_configure(config):
    global _ACTIVE_LEDGER
    ensure_cuda_bin_on_path()
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    _ACTIVE_LEDGER = DeviceLedger()
    setattr(config, _DEVICE_LEDGER_KEY, _ACTIVE_LEDGER)


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


def pytest_runtest_logreport(report):
    """Tally device-lane execution as it happens.

    A test counts as *executed* once its call phase runs, whether it passed or
    failed -- the question this ledger answers is whether the lane was
    reachable, not whether it was correct. Skips are recorded from setup so a
    device gate that declines before the body counts as the skip it is.
    """
    if _ACTIVE_LEDGER is None:
        return
    ledger = _ACTIVE_LEDGER
    families = families_for_keywords(report.keywords)
    if not families:
        return
    if report.skipped and report.when == "setup":
        ledger.record(families, executed=False)
    elif report.when == "call":
        ledger.record(families, executed=not report.skipped)


def pytest_sessionfinish(session, exitstatus):
    """Fail a session whose device lanes skipped on a host that has the device.

    This is deliberately fatal rather than a warning. The incident it exists
    for produced a warning-shaped situation -- 395 skips printed in plain
    sight -- and the run was still read as evidence, because exit 0 is what
    gets believed. A signal that does not change the exit code does not change
    what anyone concludes.
    """
    # Under xdist every worker also runs this hook, holding only the shard of
    # tests it happened to receive -- a worker handed nothing but skipped
    # device tests would report a hollow lane for the whole run. The controller
    # receives all workers' reports, so it is the only process that can judge
    # this; workers identify themselves by carrying `workerinput`.
    if hasattr(session.config, "workerinput"):
        return
    ledger = getattr(session.config, _DEVICE_LEDGER_KEY, None)
    if ledger is None:
        return
    hollow = ledger.hollow_lanes()
    if not hollow:
        return
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "hollow device lane", red=True)
        for line in format_hollow_lane_report(hollow):
            reporter.write_line(line)
    if exitstatus == 0:
        session.exitstatus = 1


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
    if item.get_closest_marker("hardware_nvidia") is not None:
        from tests._support.nvidia import nvidia_cuda_host_ready

        if not nvidia_cuda_host_ready():
            pytest.skip("requires an NVIDIA GPU with the CUDA toolkit")
        return
    if item.get_closest_marker("hardware_avx512") is not None:
        from tests._support.environment import avx512_is_plausibly_present

        if not avx512_is_plausibly_present():
            pytest.skip("requires an x86 host with AVX-512")
        return
    if item.get_closest_marker("hardware_amx") is not None:
        # AMX is retired (superseded by ACE) and no fleet host has it, so this
        # always skips today. It is a skip rather than a failure because the
        # marker states a hardware requirement the host cannot meet -- without
        # this, `test_amx_int8_gemm.py` fails on arm64 at the compile step,
        # which reads as a code defect rather than absent hardware.
        from tests._support.environment import amx_is_plausibly_present

        if not amx_is_plausibly_present():
            pytest.skip("requires Intel AMX hardware (retired target; no fleet host has it)")
        return
    if item.get_closest_marker("metal4") is not None:
        from tests._support.apple import require_apple_metal4

        require_apple_metal4()
        return
    if item.get_closest_marker("hardware_apple_gpu") is not None:
        from tests._support.apple import require_apple_metal

        require_apple_metal()
