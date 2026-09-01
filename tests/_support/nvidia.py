"""Shared NVIDIA exact-device test probes and provenance assertions."""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from typing import Any

import pytest
from tests._support.environment import (
    ensure_cuda_bin_on_path,
    ensure_nvidia_driver_on_path,
    nvidia_gpu_is_plausibly_present,
)


def nvidia_cuda_toolchain_available() -> bool:
    """Whether the host exposes an NVIDIA CUDA compiler."""
    ensure_cuda_bin_on_path()
    return bool(shutil.which("nvcc") or os.path.isfile("/usr/local/cuda/bin/nvcc"))


def nvidia_mma_runtime_available() -> bool:
    """Whether the shipped NVIDIA MMA runtime can execute on this host."""
    if not nvidia_cuda_toolchain_available():
        return False
    try:
        from tessera import runtime as rt
        return rt._nvidia_mma_runtime_available()
    except Exception:
        return False


def nvidia_mma_ptx_launch_available() -> bool:
    """Whether the MMA runtime and the shipped PTX launch bridge are usable."""
    if not nvidia_mma_runtime_available():
        return False
    try:
        from tessera import runtime as rt
        return rt._load_nvidia_ptx_launch() is not None
    except Exception:
        return False


def require_nvidia_mma_runtime() -> Any:
    """Return the runtime or skip with a stable, capability-specific reason."""
    if not nvidia_cuda_toolchain_available():
        pytest.skip("nvcc not installed")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no usable NVIDIA CUDA device")
    return rt


def nvidia_cuda_host_ready() -> bool:
    """Whether CUDA tooling and the NVIDIA driver are reachable from this host.

    Repairs PATH first, then WARNS if a GPU is evidently present but still
    unreachable. A silent skip is the correct answer for a host with no NVIDIA
    hardware; it is the WRONG answer for a host that has one and cannot see it,
    because the run then reports success having executed nothing. That is not
    hypothetical -- it hid 80 real failures on The-Super-Bear until 2026-08-30,
    two of which were compiler defects.
    """
    ensure_nvidia_driver_on_path()
    ready = nvidia_cuda_toolchain_available() and shutil.which("nvidia-smi") is not None
    if ready:
        try:
            ready = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=5, check=False,
            ).returncode == 0
        except OSError:
            ready = False
    if not ready and nvidia_gpu_is_plausibly_present():
        warnings.warn(
            "NVIDIA device lanes are being SKIPPED on a host that appears to "
            "have an NVIDIA GPU. This is an environment problem, not an absent "
            "device, and the skips will look like a clean run. Source "
            "scripts/_nvidia_env.sh (or put the driver shim -- /usr/lib/wsl/lib "
            "under WSL2 -- and the CUDA toolkit on PATH) before treating any "
            "result from this session as device evidence.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ready


def assert_native_gpu(result: dict[str, Any]) -> None:
    """Require a successful result with actual NVIDIA device provenance."""
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu", result.get("reason")


_device_model_probe: Any = False


def nvidia_device_model() -> str | None:
    """The GPU's marketing model string, e.g. ``"NVIDIA GeForce RTX 5070"``.

    Distinct from ``runtime._nvidia_device_name()``, which returns a compute
    capability tag (``"sm_120"``). That tag is the right key for *code*
    generation and for the autotune cache -- kernels are compiled per
    capability -- but it is the wrong key for a *performance ranking*.
    Compute capability 12.0 spans the whole consumer Blackwell line, and an
    RTX 5070 and a 5090 differ in SM count, cache and memory bandwidth by
    enough to reorder two kernels that sit 2-16% apart. Gating a measured
    ranking on ``sm_120`` would therefore assert on hardware it was never
    measured on, which is what this exists to prevent.

    Returns ``None`` (never raises) when there is no usable CUDA driver, so a
    caller can skip rather than fail on a host with no GPU.
    """
    global _device_model_probe
    if _device_model_probe is not False:
        return _device_model_probe
    _device_model_probe = None
    try:
        import ctypes

        ensure_nvidia_driver_on_path()
        cu = None
        for candidate in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
            try:
                cu = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if cu is None or cu.cuInit(0) != 0:
            return None
        device = ctypes.c_int(0)
        if cu.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None
        buffer = ctypes.create_string_buffer(256)
        if cu.cuDeviceGetName(buffer, ctypes.c_int(256), device) != 0:
            return None
        name = buffer.value.decode(errors="replace").strip()
        _device_model_probe = name or None
    except Exception:
        _device_model_probe = None
    return _device_model_probe


# --- low-precision route promotions: re-derived, not trusted --------------------


def lowp_near_winner_set(run: "dict[str, float]", noise: float) -> "set[str]":
    """Candidates indistinguishable from the fastest in one run.

    Mirrors `finalize_low_precision_native_routes._near`: everything within
    `noise` of the floor, NOT "the winner beat the field by more than noise".
    The distinction is the whole rule and is easy to invert -- a margin-based
    reading of the same file reports seven promotions as violations, all of
    them correct under the rule the recorder actually applies.
    """
    values = {name: float(value) for name, value in run.items()}
    floor = min(values.values())
    return {name for name, value in values.items()
            if value <= floor * (1.0 + noise)}


def lowp_route_promotion_violations(
    row: "dict[str, Any]", noise: float,
) -> "list[str]":
    """Re-derive one row's promotion from the timings it retained.

    `noise_fraction` is declared per row and in `method`, the recorder applies
    it, and until now no consumer re-derived it from the committed artifact --
    so a hand-edited or regressed file passed the ratchet as long as its counts
    still added up. This closes that: every recorded flag is recomputed from
    `timings`, which is the only field that is raw measurement rather than
    conclusion.

    Returns the field names that disagree with their own evidence.
    """
    violations: list[str] = []
    timings = row.get("timings")
    if not isinstance(timings, dict) or not timings:
        return ["missing_timings"]

    per_domain: dict[str, set[str]] = {}
    for domain, block in timings.items():
        runs = block.get("runs")
        if not isinstance(runs, list) or not runs:
            violations.append(f"{domain}:missing_runs")
            continue
        derived = set.intersection(
            *(lowp_near_winner_set(run, noise) for run in runs))
        per_domain[domain] = derived
        if set(block.get("near_winner_consensus", [])) != derived:
            violations.append(f"{domain}:near_winner_consensus")
        # The per-run winner must be the per-run floor, or `run_winners` is
        # describing a different measurement than `runs` holds.
        for index, run in enumerate(runs):
            fastest = min(run, key=lambda name: float(run[name]))
            recorded = (block.get("run_winners") or [None] * len(runs))[index]
            if recorded != fastest:
                violations.append(f"{domain}:run_winner[{index}]")
    if not per_domain:
        return violations or ["no_domains"]

    cross = set.intersection(*per_domain.values())
    if bool(cross) != bool(row.get("timing_domain_consensus")):
        violations.append("timing_domain_consensus")
    derived_winner = min(
        cross,
        key=lambda name: (sum(float(run[name]) for block in timings.values()
                              for run in block["runs"]), name),
    ) if cross else None
    if derived_winner != row.get("winner"):
        violations.append("winner")
    # Promotion needs a winner AND retained resource evidence for it: a route
    # nobody can point at a cubin for is not a proven route.
    if bool(derived_winner and row.get("resources")) != bool(
            row.get("selector_promoted")):
        violations.append("selector_promoted")
    if derived_winner is not None:
        for domain, derived in per_domain.items():
            if derived_winner not in derived:
                violations.append(f"{domain}:winner_not_near_best")
    return violations


def retune_stability_violations(
    row: "dict[str, Any]", noise: float,
) -> "list[str]":
    """Re-derive one legacy-retune row's stability flags from its own runs.

    `record_legacy_retune` computes, per timing domain,
    ``|run0 - run1| / max(run0, run1) <= NOISE`` -- and `noise_policy` was
    carried into the artifact where the ratchet asserted it **equals 0.03** and
    nothing compared it to a measurement. A regressed recording whose runs
    disagree by 40% keeps `stable: true` and passes, because the only thing
    checked was that the policy constant had not changed.
    """
    violations: list[str] = []
    runs = row.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        return ["missing_paired_runs"]
    for domain, field in (("device_event_ms", "device_stable"),
                          ("end_to_end_ms", "end_to_end_stable")):
        try:
            first, second = float(runs[0][domain]), float(runs[1][domain])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{domain}:unreadable")
            continue
        widest = max(first, second)
        if widest <= 0.0:
            violations.append(f"{domain}:non_positive")
            continue
        if (abs(first - second) / widest <= noise) != bool(row.get(field)):
            violations.append(field)
    if bool(row.get("stable")) != (bool(row.get("device_stable"))
                                   and bool(row.get("end_to_end_stable"))):
        violations.append("stable")
    return violations


def retune_winner_consensus_violations(
    case_rows: "list[dict[str, Any]]",
) -> "list[str]":
    """Re-derive `*_winner_consensus` across the candidates of one case.

    A candidate claims consensus when it is the fastest in **both** runs of
    that domain. This needs the whole case, not one row, which is why it is
    separate: a per-row check cannot see the field it is asserting about.
    """
    violations: list[str] = []
    for domain, field in (("device_event_ms", "device_winner_consensus"),
                          ("end_to_end_ms", "end_to_end_winner_consensus")):
        try:
            per_run = [min(case_rows,
                           key=lambda row: float(row["runs"][index][domain])
                           )["candidate"] for index in (0, 1)]
        except (KeyError, IndexError, TypeError, ValueError):
            violations.append(f"{domain}:unreadable")
            continue
        for row in case_rows:
            derived = all(name == row["candidate"] for name in per_run)
            if derived != bool(row.get(field)):
                violations.append(f"{row['candidate']}:{field}")
    return violations
