from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_x86_timing_provider_is_wired_into_tprof() -> None:
    header = ROOT / "tools/profiler/include/tprof/x86_timing_provider.h"
    source = ROOT / "tools/profiler/src/runtime/x86_timing_provider.cpp"
    cmake = (ROOT / "tools/profiler/CMakeLists.txt").read_text()
    cli = (ROOT / "tools/profiler/cli/tprof.cpp").read_text()

    for phrase in (
        "CLOCK_MONOTONIC_RAW",
        "__rdtscp",
        "PERF_FORMAT_TOTAL_TIME_ENABLED",
        "PERF_FORMAT_TOTAL_TIME_RUNNING",
        "PERF_COUNT_SW_TASK_CLOCK",
        "PERF_COUNT_HW_INSTRUCTIONS",
        "PERF_IOC_FLAG_GROUP",
    ):
        assert phrase in source.read_text()
    assert "x86_timing_provider.cpp" in cmake
    assert "x86 timing-status" in cli
    assert "x86_perf_group_t" in header.read_text()


def test_x86_timing_provider_clock_harness(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")
    harness = tmp_path / "x86_timing_harness.cpp"
    executable = tmp_path / "x86_timing_harness"
    harness.write_text(
        r"""
#include "tprof/x86_timing_provider.h"

int main() {
  const auto caps = tprof::x86_timing_capabilities();
  const auto first = tprof::x86_clock_snapshot();
  const auto second = tprof::x86_clock_snapshot();
  if (first.steady_ns == 0 || second.steady_ns < first.steady_ns) return 1;
  if (caps.monotonic_raw &&
      (!first.monotonic_raw_valid || second.monotonic_raw_ns < first.monotonic_raw_ns)) {
    return 2;
  }
  if (caps.rdtscp && (!first.tsc_valid || second.tsc_cycles < first.tsc_cycles)) return 3;

  tprof::x86_perf_group_t perf;
  if (perf.open()) {
    if (!perf.start()) return 4;
    volatile unsigned long value = 0;
    for (unsigned i = 0; i < 10000; ++i) value += i;
    const auto sample = perf.stop();
    if (!sample.valid || sample.task_clock_ns == 0 || sample.time_running_ns == 0 ||
        !(sample.available_events_mask & tprof::X86_PERF_TASK_CLOCK)) {
      return 5;
    }
  }
  return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "tools/profiler/include"),
            str(harness),
            str(ROOT / "tools/profiler/src/runtime/x86_timing_provider.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
