"""Phase E3 / Pillar 2 — the autotuning flywheel: deterministic (candidate,
outcome) records (docs/audit/compiler/EVALUATOR_PLAN.md §6–§7).

This is the *data layer* the autotuner and a future learned cost model consume —
distinct from ``autotune_v2`` (the Bayesian search). Each record pairs a schedule
candidate with its **measured** latency on a real backend AND an **analytical
roofline prediction**, on the same row, so every record is simultaneously a
training sample and a model-residual audit.

Disciplines from the plan (§7):
  * **Never aggregate across ``device_id``** — schedules don't transfer across
    chips; ``device_id`` is required and is the partition key.
  * Keep ``roofline_predicted_ms`` (and later ``model_predicted_ms``) on the same
    row as the measurement, so residuals are auditable without re-instrumenting.
  * Latency is host-specific and non-deterministic: tests assert record
    *structure* + that the measurement was native, never a fixed number (the
    perf-ratchet discipline).

Honest scope: on Apple GPU the MPS matmul is a black box (no ``tile_q``/stage
knobs), so the real candidate space here is dtype / fusion choice. The richer
tile/stage schedule space arrives with the NVIDIA executable lane. And for small
shapes the measured latency is **launch-overhead-dominated**, far above the
compute/BW roofline — the residual makes that visible rather than hiding it.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DevicePeak:
    """Analytical roofline anchors for one device. Nominal until calibrated
    per ``device_id`` — the record's residual is what validates it."""

    name: str
    peak_tflops: float        # TFLOP/s (dtype-specific in reality; nominal here)
    peak_bw_gb_s: float       # GB/s


# Nominal Apple-GPU anchors — fallback when the chip can't be identified.
NOMINAL_APPLE_GPU_PEAK = DevicePeak("apple_gpu_nominal", peak_tflops=8.0, peak_bw_gb_s=200.0)

# Per-family calibrated anchors (approximate published f32 GPU peak + memory BW;
# core counts vary within a family, so these are family-level, not exact —
# calibratable per device_id). Checked most-specific first (ultra > max > pro).
_APPLE_GPU_PEAKS: tuple[tuple[str, DevicePeak], ...] = (
    ("m1-ultra", DevicePeak("apple-m1-ultra", 21.0, 800.0)),
    ("m1-max", DevicePeak("apple-m1-max", 10.4, 400.0)),
    ("m1-pro", DevicePeak("apple-m1-pro", 5.2, 200.0)),
    ("m1", DevicePeak("apple-m1", 2.6, 68.0)),
    ("m2-ultra", DevicePeak("apple-m2-ultra", 27.2, 800.0)),
    ("m2-max", DevicePeak("apple-m2-max", 13.6, 400.0)),
    ("m2-pro", DevicePeak("apple-m2-pro", 6.8, 200.0)),
    ("m2", DevicePeak("apple-m2", 3.6, 100.0)),
    ("m3-max", DevicePeak("apple-m3-max", 14.0, 300.0)),
    ("m3-pro", DevicePeak("apple-m3-pro", 7.0, 150.0)),
    ("m3", DevicePeak("apple-m3", 3.5, 100.0)),
    ("m4-max", DevicePeak("apple-m4-max", 17.0, 410.0)),
    ("m4", DevicePeak("apple-m4", 4.3, 120.0)),
)


def _apple_chip_brand() -> str | None:
    """The Apple Silicon chip brand (e.g. ``Apple M1 Max``) via sysctl, or None
    off-Darwin / on failure. The SoC GPU's identity == the chip identity."""
    import subprocess

    # Use platform.system() (not sys.platform): mypy platform-narrows
    # sys.platform to the type-check host, which would mark the sysctl path
    # unreachable under --warn-unreachable on a non-Darwin CI.
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def default_device_id(target: str) -> str:
    """Coarse fallback device identity (``arch-system``)."""
    u = platform.uname()
    return f"{target}:{u.machine}-{u.system}".lower()


def detect_device_id(target: str) -> str:
    """Calibrated device identity. For Apple targets on Darwin this is the real
    chip (``apple_gpu:apple-m1-max``) so records are keyed by the actual GPU;
    otherwise the coarse fallback. This is the partition key — never aggregate
    across it."""
    brand = _apple_chip_brand()
    if brand is not None and target.startswith("apple"):
        slug = brand.lower().replace(" ", "-")
        return f"{target}:{slug}"
    return default_device_id(target)


def peak_for_device(device_id: str) -> DevicePeak:
    """The calibrated roofline anchors for a device_id, or the nominal fallback."""
    d = device_id.lower()
    for frag, peak in _APPLE_GPU_PEAKS:
        if frag in d:
            return peak
    return NOMINAL_APPLE_GPU_PEAK


@dataclass(frozen=True)
class LatencyStats:
    median_ms: float
    p10_ms: float
    p90_ms: float
    reps: int


@dataclass(frozen=True)
class AutotuneRecord:
    """One deterministic (candidate, outcome) row (plan §7)."""

    schema_version: int
    op_chain: str
    problem_shape: dict[str, int]
    dtype: str
    target: str
    device_id: str                       # NEVER aggregate across this
    schedule: dict[str, Any]             # the candidate (dtype/fusion on Apple; tile/stage later)
    legal: bool
    violation_reason: str
    latency: LatencyStats | None         # None when the candidate did not run natively
    achieved_tflops: float | None
    roofline_predicted_ms: float
    model_predicted_ms: float | None     # filled once a learned scorer exists
    search_method: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency"] = asdict(self.latency) if self.latency is not None else None
        return d

    @property
    def roofline_residual_ms(self) -> float | None:
        """measured − predicted; positive = overhead/inefficiency above roofline."""
        if self.latency is None:
            return None
        return self.latency.median_ms - self.roofline_predicted_ms


def matmul_flops_bytes(m: int, n: int, k: int, *, dtype_bytes: int = 4) -> tuple[float, float]:
    """FLOPs (2·M·N·K) and bytes moved ((MK+KN+MN)·dtype) for one matmul."""
    flops = 2.0 * m * n * k
    bytes_moved = float((m * k + k * n + m * n) * dtype_bytes)
    return flops, bytes_moved


def roofline_ms(flops: float, bytes_moved: float, peak: DevicePeak) -> float:
    """Analytical roofline lower bound (ROFT-style): max(compute, memory) time.
    The asymptotic floor — measured latency includes launch/dispatch overhead the
    roofline omits (the residual quantifies it)."""
    compute_s = flops / (peak.peak_tflops * 1e12)
    memory_s = bytes_moved / (peak.peak_bw_gb_s * 1e9)
    return max(compute_s, memory_s) * 1e3


def measure_latency(
    target: str,
    fn: Any,
    args: tuple[Any, ...],
    *,
    reps: int = 20,
    warmup: int = 3,
) -> LatencyStats | None:
    """Offline microbench (Triton-anatomy hygiene): warm up to amortize
    compile/cache, then time ``reps`` native runs. Returns ``None`` if the
    program does not execute natively (no perf signal for a fallback)."""
    from tessera.compiler.evaluator import run_native

    for _ in range(max(0, warmup)):
        _, native = run_native(target, fn, args)
        if not native:
            return None

    samples: list[float] = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        _, native = run_native(target, fn, args)
        dt = (time.perf_counter() - t0) * 1e3
        if not native:
            return None
        samples.append(dt)

    samples.sort()
    n = len(samples)
    return LatencyStats(
        median_ms=samples[n // 2],
        p10_ms=samples[max(0, int(0.1 * n))],
        p90_ms=samples[min(n - 1, int(0.9 * n))],
        reps=n,
    )


def record_matmul(
    target: str,
    fn: Any,
    args: tuple[Any, ...],
    *,
    m: int,
    n: int,
    k: int,
    dtype: str = "f32",
    device_id: str | None = None,
    peak: DevicePeak | None = None,
    schedule: dict[str, Any] | None = None,
    search_method: str = "manual",
    reps: int = 20,
) -> AutotuneRecord:
    """Measure one matmul candidate and build its deterministic record (measured
    latency + analytical roofline on the same row). ``device_id`` and ``peak``
    default to the calibrated chip + its family anchors."""
    did = device_id or detect_device_id(target)
    use_peak = peak or peak_for_device(did)
    dtype_bytes = 2 if dtype in ("f16", "bf16") else 4
    flops, bytes_moved = matmul_flops_bytes(m, n, k, dtype_bytes=dtype_bytes)
    predicted = roofline_ms(flops, bytes_moved, use_peak)
    stats = measure_latency(target, fn, args, reps=reps)
    achieved = (
        flops / (stats.median_ms * 1e-3) / 1e12 if stats is not None else None
    )
    return AutotuneRecord(
        schema_version=SCHEMA_VERSION,
        op_chain="matmul",
        problem_shape={"M": m, "N": n, "K": k},
        dtype=dtype,
        target=target,
        device_id=did,
        schedule=schedule or {"dtype": dtype, "fusion_choice": "none"},
        legal=True,
        violation_reason="",
        latency=stats,
        achieved_tflops=achieved,
        roofline_predicted_ms=predicted,
        model_predicted_ms=None,
        search_method=search_method,
    )


def sweep_matmul(
    target: str,
    fn: Any,
    sizes: tuple[int, ...],
    rng: Any,
    *,
    dtype: str = "f32",
    peak: DevicePeak | None = None,
    reps: int = 10,
) -> list[AutotuneRecord]:
    """Record a square-matmul candidate corpus across ``sizes`` for one dtype.
    Turns the flywheel from one row into a corpus — the data a cost model trains
    on. Inputs are dtype-typed so the native kernel path is exercised."""
    import numpy as np

    np_dtype = np.float16 if dtype == "f16" else np.float32
    records: list[AutotuneRecord] = []
    for s in sizes:
        a = rng.standard_normal((s, s)).astype(np_dtype)
        b = rng.standard_normal((s, s)).astype(np_dtype)
        records.append(
            record_matmul(
                target, fn, (a, b), m=s, n=s, k=s, dtype=dtype, peak=peak,
                schedule={"dtype": dtype, "size": s}, search_method="sweep", reps=reps,
            )
        )
    return records


def efficiency_trend(records: list[AutotuneRecord]) -> list[tuple[int, float]]:
    """``(problem_size, achieved_tflops)`` sorted by size, for records that ran
    natively. The launch-overhead fraction shrinks as size grows, so achieved
    efficiency climbs — this exposes that trend (and makes the roofline residual
    a real efficiency signal at the large end, not just overhead at the small)."""
    return sorted(
        (r.problem_shape["M"], r.achieved_tflops)
        for r in records
        if r.achieved_tflops is not None
    )


# ── persist + distill (Triton-anatomy: corpus → O(1) decision-tree dispatch) ──


def save_corpus(records: list[AutotuneRecord], path: str) -> None:
    """Write a record corpus to JSON. Host-specific (latencies) — write to an
    output/cache path, not a checked-in baseline (the perf-ratchet discipline)."""
    import json

    with open(path, "w") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2)


def load_corpus(path: str) -> list[AutotuneRecord]:
    """Read a record corpus written by :func:`save_corpus`."""
    import json

    with open(path) as fh:
        rows = json.load(fh)
    out: list[AutotuneRecord] = []
    for d in rows:
        lat = d.get("latency")
        d = {**d, "latency": LatencyStats(**lat) if lat is not None else None}
        out.append(AutotuneRecord(**d))
    return out


def size_bucket(n: int) -> str:
    """Coarse problem-size bucket for the decision-tree key (keeps the table
    small and generalizes across nearby shapes)."""
    if n < 256:
        return "xs"
    if n < 1024:
        return "s"
    if n < 4096:
        return "m"
    return "l"


def _dispatch_key(op_chain: str, dtype: str, size: int) -> str:
    return f"{op_chain}|{dtype}|{size_bucket(size)}"


def distill_dispatch(records: list[AutotuneRecord]) -> dict[str, dict[str, Any]]:
    """Distill a corpus into an O(1) decision-tree: for each
    ``(op_chain, dtype, size-bucket)`` class, the schedule with the lowest median
    latency. This is the cheap production dispatch the autotuning results bake
    down to (Triton-anatomy)."""
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for r in records:
        if r.latency is None:
            continue
        key = _dispatch_key(r.op_chain, r.dtype, r.problem_shape["M"])
        ms = r.latency.median_ms
        if key not in best or ms < best[key][0]:
            best[key] = (ms, {
                "schedule": r.schedule,
                "median_ms": ms,
                "device_id": r.device_id,
            })
    return {k: v for k, (_, v) in best.items()}


def lookup_dispatch(
    table: dict[str, dict[str, Any]], op_chain: str, dtype: str, size: int
) -> dict[str, Any] | None:
    """O(1) lookup into a distilled dispatch table; ``None`` if the class is
    uncovered (caller falls back to a live search / default)."""
    return table.get(_dispatch_key(op_chain, dtype, size))
