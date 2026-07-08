#!/usr/bin/env python3
"""Roofline attainment for the E2 hot-path ratchets (Workstream J / W7).

The latency ratchet (`perf_gate.evaluate_ratchet`) answers "did a hot path get
slower?" — a *relative* bar. J adds the *absolute* bar the plan asks for: **% of
peak**. Each row's wall-clock median is turned into achieved TFLOP/s and divided
by the device's grounded peak, so a hot path is judged by how close it runs to the
silicon's ceiling — and a regression *below* an attainment floor fails the gate,
symmetric with the latency cap.

**Honest scope.** The ratchet `median_ms` is end-to-end wall-clock (H2D / launch /
D2H, and for the compiled lanes a `tessera-opt` shell-out), NOT an isolated kernel
time. So `pct_peak` here is an **end-to-end attainment** — a stringent lower bound
on kernel efficiency, and the most honest "what does a caller actually get" bar. It
is expected to be well under a hand-tuned kernel's isolated attainment; the point
is to make the number visible and ratchet it upward.

**Peak is grounded, not asserted** (Decision #27): each device's peak carries a
`source` string deriving it from `rocminfo` (CU/SIMD/clock) + documented RDNA3
rates, so the constant is auditable and correctable in one place.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

#: Per-device compute/bandwidth peaks. Keyed by the autotune device tag
#: (`rocm:gfx1151`, `nvidia:sm_120`, …). `peak_tflops` is per dtype.
DEVICE_PEAK: dict[str, dict[str, Any]] = {
    "rocm:gfx1151": {
        "peak_tflops": {"f16": 59.4, "bf16": 59.4, "f32": 29.7},
        "peak_bw_gb_s": 256.0,
        "source": (
            "rocminfo: 40 CU x 2 SIMD32 x 32 lanes = 2560 ALU; x2 FLOP (FMA) "
            "x2 (RDNA3 dual-issue VOPD) x 2.9 GHz = 29.7 TF fp32; fp16/bf16 WMMA "
            "packed 2x = 59.4 TF. BW: Strix Halo LPDDR5X-8000 256-bit unified "
            "= 256 GB/s. Theoretical peak (dual-issue); end-to-end attainment "
            "against it is a lower bound."
        ),
    },
}


def _bytes_of(dtype: str) -> int:
    return {"f16": 2, "bf16": 2, "f32": 4, "float16": 2, "bfloat16": 2,
            "float32": 4}.get(dtype, 2)


def op_flops(op: str, shape: str) -> Optional[int]:
    """FLOPs for one invocation of ``op`` at ``shape`` (the ratchet shape string),
    or ``None`` for an op with no FLOP model. matmul ``MxNxK`` = 2·M·N·K;
    flash_attn ``BxHxSxD`` = 4·B·H·S²·D (QKᵀ + PV, softmax negligible)."""
    dims = [int(x) for x in shape.split("x")]
    if op == "matmul" and len(dims) == 3:
        m, n, k = dims
        return 2 * m * n * k
    if op == "flash_attn" and len(dims) == 4:
        b, h, s, d = dims
        return 4 * b * h * s * s * d
    return None


def op_bytes(op: str, shape: str, dtype: str) -> Optional[int]:
    """Minimum DRAM traffic (bytes) for ``op`` — operands + result, no reuse."""
    w = _bytes_of(dtype)
    dims = [int(x) for x in shape.split("x")]
    if op == "matmul" and len(dims) == 3:
        m, n, k = dims
        return (m * k + k * n + m * n) * w
    if op == "flash_attn" and len(dims) == 4:
        b, h, s, d = dims
        return 4 * b * h * s * d * w          # Q, K, V, O
    return None


def achieved_tflops(op: str, shape: str, median_ms: float) -> Optional[float]:
    flops = op_flops(op, shape)
    if flops is None or median_ms <= 0:
        return None
    return flops / (median_ms * 1e-3) / 1e12


def pct_peak(op: str, shape: str, dtype: str, median_ms: float,
             device: str) -> Optional[float]:
    """End-to-end compute attainment: achieved TFLOP/s ÷ the device's peak for
    ``dtype``. ``None`` when the op has no FLOP model or the device/dtype peak is
    unknown (never guessed)."""
    ach = achieved_tflops(op, shape, median_ms)
    dev = DEVICE_PEAK.get(device)
    if ach is None or dev is None:
        return None
    peak = dev["peak_tflops"].get(dtype)
    if not peak:
        return None
    return ach / peak


def annotate_rows(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    """Add ``achieved_tflops`` + ``pct_peak`` to each row that has a FLOP model
    (computed from the row's existing ``median_ms`` — no re-timing). Rows without
    a model are returned unchanged."""
    for r in rows:
        ach = achieved_tflops(r["op"], r["shape"], float(r["median_ms"]))
        if ach is None:
            continue
        pk = pct_peak(r["op"], r["shape"], r["dtype"],
                      float(r["median_ms"]), device)
        r["achieved_tflops"] = round(ach, 4)
        if pk is not None:
            r["pct_peak"] = round(pk, 5)
    return rows


def evaluate_attainment(rows: list[Mapping[str, Any]],
                        baseline: Mapping[str, Any],
                        device: str) -> list[str]:
    """Gate measured rows against each baseline row's ``attainment_floor`` (% of
    peak). A row whose measured ``pct_peak`` falls below its floor fails — the
    absolute-attainment analog of the latency ratchet. Rows with no floor are not
    gated (opt-in). A baseline row with a floor but no matching measurement fails
    on coverage."""
    failures: list[str] = []

    def key(r: Mapping[str, Any]) -> tuple:
        return (r.get("op"), r.get("shape"), r.get("dtype"), r.get("mode"))

    floors = {key(r): float(r["attainment_floor"])
              for r in baseline.get("rows", []) if "attainment_floor" in r}
    seen: set[tuple] = set()
    for row in rows:
        k = key(row)
        floor = floors.get(k)
        if floor is None:
            continue
        seen.add(k)
        # Measured ratchet-report rows carry `latency_ms` (like evaluate_ratchet
        # reads); baseline/self-check rows carry `median_ms`. Accept either.
        t = row.get("latency_ms", row.get("median_ms", 0.0))
        pk = pct_peak(row.get("op", ""), row.get("shape", ""),
                      row.get("dtype", ""), float(t), device)
        if pk is None:
            failures.append(f"{k[0]} {k[1]} {k[2]} {k[3]}: no attainment "
                            f"(missing FLOP model or device peak for {device!r})")
        elif pk < floor:
            failures.append(
                f"{k[0]} {k[1]} {k[2]} {k[3]}: pct_peak={pk:.4f} below floor "
                f"{floor:.4f}")
    for k in sorted(floors.keys() - seen, key=str):
        failures.append(f"{k[0]} {k[1]} {k[2]} {k[3]}: no measurement "
                        f"(attainment coverage)")
    return failures
