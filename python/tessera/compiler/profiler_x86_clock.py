"""x86 TSC timing witness measured around the real benchmark region.

Sync WSL-TIMING-ADMISSION-2026-09-26, x86 follow-up. The earlier probe
(``tools/profiler/cli/tprof.cpp``) timed its own ``sleep(100ms)`` in a separate
process after the benchmark and derived the TSC frequency from that same
interval, so "TSC agrees with CLOCK_MONOTONIC_RAW" held by construction and
said nothing about the benchmark. This module instead:

1. calibrates the TSC frequency over **separate** calibration intervals (A),
   pinned to one CPU, before anything is measured;
2. reads (CLOCK_MONOTONIC_RAW, rdtscp) immediately around **each measured
   region** (B) on that same CPU;
3. converts B's TSC delta with A's frequency and compares it with B's raw
   delta. Agreement within the providers' 5% band is then a real statement:
   the frequency was not fitted to the interval it is checked against.

WSL2 caveat, stated rather than hidden: under Hyper-V the raw clock is itself
derived from the TSC through a hypervisor-supplied scale, so this proves the
TSC-to-time scale is stable across intervals, not that an independent
oscillator agrees. The same limit applies to the device-clock witnesses on the
GPU lanes, whose event clocks also ride the host path.

The clock reads come from a tiny C helper compiled on first use with the host
C compiler (``rdtscp`` and ``clock_gettime`` have no Python binding); it is
measurement tooling, not a lowering route.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
import shutil
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HELPER_C = r"""
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
/* out[0]=raw ns before, out[1]=tsc, out[2]=aux (cpu), out[3]=raw ns after */
void tessera_x86_clock_snapshot(uint64_t *out) {
  struct timespec a, b; unsigned aux = 0;
  clock_gettime(CLOCK_MONOTONIC_RAW, &a);
  _mm_lfence();
  uint64_t tsc = __rdtscp(&aux);
  _mm_lfence();
  clock_gettime(CLOCK_MONOTONIC_RAW, &b);
  out[0] = (uint64_t)a.tv_sec * 1000000000ull + (uint64_t)a.tv_nsec;
  out[1] = tsc;
  out[2] = aux;
  out[3] = (uint64_t)b.tv_sec * 1000000000ull + (uint64_t)b.tv_nsec;
}
"""

CALIBRATION_INTERVALS = 5
CALIBRATION_SECONDS = 0.2
#: Maximum relative spread of the calibration intervals' frequencies; a TSC
#: whose rate moves more than this between intervals is not a usable clock.
CALIBRATION_SPREAD_LIMIT = 0.001


class X86ClockError(RuntimeError):
    """The host cannot produce a TSC witness (not x86 Linux, no invariant TSC, ...)."""


@dataclass(frozen=True)
class Snapshot:
    raw_ns: int      # midpoint of the raw reads bracketing rdtscp
    tsc: int
    cpu: int         # logical CPU from the rdtscp aux value (low 12 bits)


_LIB: ct.CDLL | None = None


def _library() -> ct.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise X86ClockError("the TSC witness needs x86_64 Linux")
    compiler = shutil.which("cc")
    if compiler is None:
        raise X86ClockError("the x86 clock helper needs a C compiler on PATH as `cc`")
    directory = Path(tempfile.mkdtemp(prefix="tessera-x86-clock-"))
    source, shared = directory / "clock.c", directory / "libclock.so"
    try:
        source.write_text(_HELPER_C)
        try:
            subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(shared)],
                           check=True, capture_output=True, text=True, timeout=60)
        except subprocess.CalledProcessError as exc:
            raise X86ClockError(
                f"could not build the x86 clock helper: {exc.stderr.strip()[:400]}") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise X86ClockError(f"could not build the x86 clock helper: {exc}") from exc
        try:
            lib = ct.CDLL(str(shared))
        except OSError as exc:
            raise X86ClockError(f"could not load the x86 clock helper: {exc}") from exc
    finally:
        # The mapping survives unlinking on Linux; leave no per-process debris.
        shutil.rmtree(directory, ignore_errors=True)
    lib.tessera_x86_clock_snapshot.argtypes = [ct.POINTER(ct.c_uint64)]
    lib.tessera_x86_clock_snapshot.restype = None
    _LIB = lib
    return lib


def snapshot() -> Snapshot:
    buf = (ct.c_uint64 * 4)()
    _library().tessera_x86_clock_snapshot(buf)
    return Snapshot(raw_ns=(buf[0] + buf[3]) // 2, tsc=buf[1], cpu=buf[2] & 0xFFF)


def invariant_tsc() -> bool:
    """constant_tsc and nonstop_tsc: the TSC rate does not follow P- or C-states."""
    try:
        flags = next(line for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("flags")).split()
    except (OSError, StopIteration):
        return False
    return "constant_tsc" in flags and "nonstop_tsc" in flags


def calibrate(*, intervals: int = CALIBRATION_INTERVALS,
              seconds: float = CALIBRATION_SECONDS) -> dict[str, Any]:
    """TSC Hz from separate busy intervals on the current (pinned) CPU."""
    rates, windows = [], []
    for _ in range(intervals):
        start = snapshot()
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            pass
        end = snapshot()
        if start.cpu != end.cpu:
            raise X86ClockError("calibration migrated between CPUs; pin the process first")
        rates.append((end.tsc - start.tsc) * 1e9 / (end.raw_ns - start.raw_ns))
        windows.append({"raw_start_ns": start.raw_ns, "raw_end_ns": end.raw_ns,
                        "tsc_start": start.tsc, "tsc_end": end.tsc, "cpu": start.cpu})
    hz = statistics.median(rates)
    spread = (max(rates) - min(rates)) / hz
    if spread > CALIBRATION_SPREAD_LIMIT:
        raise X86ClockError(f"TSC rate moved {spread:.3%} across calibration intervals")
    return {"frequency_hz": hz, "spread": spread, "rates_hz": rates, "windows": windows}


def measure(region: Callable[[], Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Time one region with the TSC and the raw clock, on the calibrated CPU."""
    host_start = time.perf_counter_ns()
    start = snapshot()
    region()
    end = snapshot()
    host_end = time.perf_counter_ns()
    last = calibration["windows"][-1]
    if start.raw_ns <= last["raw_end_ns"]:
        raise X86ClockError("measurement window overlaps the calibration window")
    return {"raw_start_ns": start.raw_ns, "raw_end_ns": end.raw_ns,
            "tsc_start": start.tsc, "tsc_end": end.tsc,
            "logical_cpu_start": start.cpu, "logical_cpu_end": end.cpu,
            "host_wall_ns": host_end - host_start}


def witness_clocks(calibration: dict[str, Any], window: dict[str, Any], *,
                   refused: str | None = None) -> dict[str, Any]:
    """Build the ``tsc_cycles`` / ``monotonic_raw_ns`` records for one window.

    ``profiler_timing`` then re-derives the TSC nanoseconds from these raw
    integers and A's frequency and refuses promotion unless raw agrees within
    5%. Nothing here is a self-declared pass/fail flag.
    """
    from .profiler_timing import measured_clock
    raw = window["raw_end_ns"] - window["raw_start_ns"]
    cycles = window["tsc_end"] - window["tsc_start"]
    if raw <= 0 or cycles <= 0:
        raise X86ClockError("measured window is empty or went backwards")
    return {
        "monotonic_raw_ns": measured_clock("monotonic_raw_ns", source="clock_monotonic_raw", value=raw),
        "tsc_cycles": measured_clock(
            "tsc_cycles", source="rdtscp", value=cycles,
            provenance={"invariant_tsc": invariant_tsc(),
                        "logical_cpu_start": window["logical_cpu_start"],
                        "logical_cpu_end": window["logical_cpu_end"],
                        "calibrated_frequency_hz": calibration["frequency_hz"],
                        "frequency_source": "independent_calibration_interval",
                        "calibration_spread": calibration["spread"],
                        "calibration_sha256": calibration_digest(calibration),
                        # Stored whole so a validator can re-derive the
                        # frequency, spread, CPU and ordering instead of
                        # trusting the fields above.
                        "calibration": calibration,
                        "measurement_window": window,
                        **({"promotion_refused": refused} if refused else {})},
            calibrated_against=("monotonic_raw_ns",), eligible_for_promotion=refused is None),
    }


def witness_sample(calibration: dict[str, Any], window: dict[str, Any],
                   digests: dict[str, str], *,
                   execution_environment: str | None = None) -> dict[str, Any]:
    """A ``tessera.profiler_timing.v1`` sample for one measured window: the
    TSC (converted with the separate calibration interval's frequency) against
    CLOCK_MONOTONIC_RAW over the same region, bound to ``digests`` (the row's
    image first). The packet's ``tsc_witness`` route re-validates it."""
    from .profiler_timing import (
        ProfilerTimingError, build_timing_sample, measured_clock, unavailable_clock,
        wsl_promotion_refusals)
    if not isinstance(digests.get("image"), str) or not digests["image"]:
        raise X86ClockError("a TSC witness must name the measured image digest as `image`")
    if execution_environment is None:
        execution_environment = ("wsl2" if "microsoft" in platform.release().lower()
                                 else "bare_metal")

    def build(refused: str | None) -> dict[str, Any]:
        clocks = witness_clocks(calibration, window, refused=refused)
        clocks["host_wall_ns"] = measured_clock(
            "host_wall_ns", source="perf_counter", value=window["host_wall_ns"])
        clocks["perf_task_clock_ns"] = unavailable_clock(
            "perf_task_clock_ns", source="perf_event_task_clock",
            reason="NOT_REQUESTED_FOR_TSC_WITNESS")
        return build_timing_sample(
            sample_id=f"x86-tsc-{digests['image'][:12]}-{window['raw_start_ns']}",
            target="x86", clocks=clocks, artifact_digests=digests, batch_size=1,
            warm_state="warm", synchronization="serial runtime.launch",
            execution_environment=execution_environment,
            environment={"kernel_release": platform.release()})

    # A window whose TSC and raw clock disagree is a measured fact about this
    # row, not a reason to abort the benchmark: record the sample with the
    # TSC ineligible and the refusal named, and the packet route stays
    # profiler_correlated. Any other invalidity (a host without an invariant
    # TSC, a migrated CPU) propagates: it is a fact about the host, and a run
    # that cannot produce a valid witness must say so rather than record one.
    try:
        sample = build(None)
    except ProfilerTimingError:
        sample = build("unvalidated")
        refusals = wsl_promotion_refusals("x86", {
            **sample["clocks"],
            "tsc_cycles": {**sample["clocks"]["tsc_cycles"], "eligible_for_promotion": True}})
        if not refusals:
            raise
        sample = build("; ".join(refusals))
    else:
        refusals = wsl_promotion_refusals("x86", sample["clocks"])
        if refusals:
            sample = build("; ".join(refusals))
    return sample


def calibration_digest(calibration: dict[str, Any]) -> str:
    """Canonical digest of a calibration record (JSON, sorted keys)."""
    return hashlib.sha256(
        json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_witness_sample(sample: Any) -> str | None:
    """Re-derive a TSC witness from its stored integers; None when consistent.

    Nothing the sample states is trusted: the clock values must equal their
    measurement window's deltas, the frequency must be the median of the
    stored calibration intervals' own rates (each recomputed from its raw
    integers) with the stored spread within the limit, every calibration
    interval and both ends of the measurement must be on one logical CPU, the
    measurement must start after the last calibration interval ends, and the
    stored digest must match the stored calibration. Returns the first
    inconsistency found, as a reason string.
    """
    try:
        clocks = sample["clocks"]
        tsc, raw = clocks["tsc_cycles"], clocks["monotonic_raw_ns"]
        prov = tsc["provenance"]
        window, calibration = prov["measurement_window"], prov["calibration"]
        windows, rates = calibration["windows"], calibration["rates_hz"]
    except (KeyError, TypeError):
        return "witness lacks its measurement window or stored calibration"
    if prov.get("frequency_source") != "independent_calibration_interval":
        return "frequency was not taken from separate calibration intervals"
    if prov.get("calibration_sha256") != calibration_digest(calibration):
        return "calibration digest does not match the stored calibration"
    if raw.get("value") != window["raw_end_ns"] - window["raw_start_ns"]:
        return "monotonic_raw_ns is not its measurement window's delta"
    if tsc.get("value") != window["tsc_end"] - window["tsc_start"]:
        return "tsc_cycles is not its measurement window's delta"
    if not windows or len(windows) != len(rates):
        return "calibration intervals and rates disagree in number"
    for interval, rate in zip(windows, rates):
        elapsed = interval["raw_end_ns"] - interval["raw_start_ns"]
        if elapsed <= 0 or not math.isclose(
                (interval["tsc_end"] - interval["tsc_start"]) * 1e9 / elapsed, rate, rel_tol=1e-9):
            return "a calibration rate is not its interval's own measurement"
    hz = statistics.median(rates)
    if not math.isclose(prov.get("calibrated_frequency_hz", 0.0), hz, rel_tol=1e-12):
        return "calibrated frequency is not the median of the calibration rates"
    if (max(rates) - min(rates)) / hz > CALIBRATION_SPREAD_LIMIT:
        return "calibration intervals disagree beyond the spread limit"
    cpus = {interval["cpu"] for interval in windows} | {
        window["logical_cpu_start"], window["logical_cpu_end"]}
    if len(cpus) != 1:
        return f"calibration and measurement ran on different CPUs {sorted(cpus)}"
    if window["raw_start_ns"] <= max(interval["raw_end_ns"] for interval in windows):
        return "measurement window does not follow the calibration intervals"
    return None


def pin_current_cpu() -> int:
    """Pin to the CPU we are on now, so calibration and measurement share it."""
    cpu = snapshot().cpu
    setaffinity = getattr(os, "sched_setaffinity", None)
    if setaffinity is None:
        raise X86ClockError("this host cannot pin a process to one CPU")
    setaffinity(0, {cpu})
    return cpu


__all__ = ["X86ClockError", "calibrate", "calibration_digest", "invariant_tsc", "measure",
           "pin_current_cpu", "snapshot", "verify_witness_sample", "witness_clocks",
           "witness_sample"]
