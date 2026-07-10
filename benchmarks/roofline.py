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


#: flash_attn backward ÷ forward FLOP ratio. FlashAttention-2 (Dao 2023, §2.2)
#: states the backward pass is ~2.5× the forward's matmul FLOPs (dQ = dP·K,
#: dK = dPᵀ·Q, dV = Pᵀ·dO, dP = dO·Vᵀ, plus the dS scaling and the on-chip
#: S = Q·Kᵀ recompute). Tessera's kernel ADDITIONALLY recomputes the forward O
#: (the fa_pre pass), so its achieved attainment against this algorithmic model
#: is a lower bound — consistent with the end-to-end honesty note above.
_FLASH_BWD_FWD_RATIO = 2.5


def op_flops(op: str, shape: str, causal: bool = True) -> Optional[int]:
    """FLOPs for one invocation of ``op`` at ``shape`` (the ratchet shape string),
    or ``None`` for an op with no FLOP model. matmul / gemm_f32 ``MxNxK`` = 2·M·N·K;
    flash_attn ``BxHxSxD`` = 4·B·H·P·D and flash_attn_bwd = 2.5× that
    (``_FLASH_BWD_FWD_RATIO``), where P is the number of query·key score pairs.

    ``causal`` (default True — the recorded ROCm flash hot paths run causal) makes
    P the **triangular** count S·(S+1)/2 rather than the dense S², because the
    generated causal kernels skip the masked upper tiles in *both* the forward and
    the dK/dV + dQ backward loops (GenerateWMMAFlashAttnBwdKernel.cpp). Charging
    the full dense S² for a causal row would ~2× the achieved TFLOP/s and inflate
    the MFU sign-off; the model must match the work actually timed."""
    dims = [int(x) for x in shape.split("x")]
    # gemm_f32 is a plain GEMM (the register-blocked f32 VALU kernel) — same
    # 2·M·N·K work as the WMMA matmul, just f32 storage (gated vs the 29.7 TF peak).
    if op in ("matmul", "gemm_f32") and len(dims) == 3:
        m, n, k = dims
        return 2 * m * n * k
    if op in ("flash_attn", "flash_attn_bwd") and len(dims) == 4:
        b, h, s, d = dims
        pairs = s * (s + 1) // 2 if causal else s * s   # triangular vs dense
        fwd = 4 * b * h * pairs * d
        return fwd if op == "flash_attn" else int(_FLASH_BWD_FWD_RATIO * fwd)
    return None


def op_bytes(op: str, shape: str, dtype: str) -> Optional[int]:
    """Minimum DRAM traffic (bytes) for ``op`` — operands + result, no reuse."""
    w = _bytes_of(dtype)
    dims = [int(x) for x in shape.split("x")]
    if op in ("matmul", "gemm_f32") and len(dims) == 3:
        m, n, k = dims
        return (m * k + k * n + m * n) * w
    if op == "flash_attn" and len(dims) == 4:
        b, h, s, d = dims
        return 4 * b * h * s * d * w          # Q, K, V, O
    if op == "flash_attn_bwd" and len(dims) == 4:
        b, h, s, d = dims
        return 7 * b * h * s * d * w          # Q, K, V, dO in; dQ, dK, dV out
    # Pure-movement lanes (no FLOP model → bandwidth-bound, see pct_peak_bw). Each
    # moves rows of data with no reuse: read the touched rows + write them once.
    # Shape ``RxW`` = rows moved × row width (elements). `_MOVEMENT_OPS` gates the
    # bandwidth branch in annotate_rows so these never enter the compute path.
    if op in _MOVEMENT_OPS and len(dims) == 2:
        rows_moved, width = dims
        return 2 * rows_moved * width * w      # read rows + write rows
    return None


#: Ops characterized by bandwidth attainment (achieved GB/s ÷ peak BW), not
#: compute attainment — the gfx1151 gather/scatter movement kernels. They have an
#: op_bytes model but deliberately NO op_flops (near-zero arithmetic), so
#: annotate_rows routes them to the bandwidth branch.
_MOVEMENT_OPS: frozenset[str] = frozenset({
    "kv_cache_append", "kv_cache_read", "moe_dispatch", "moe_combine",
})


def achieved_tflops(op: str, shape: str, median_ms: float,
                    causal: bool = True) -> Optional[float]:
    flops = op_flops(op, shape, causal=causal)
    if flops is None or median_ms <= 0:
        return None
    return flops / (median_ms * 1e-3) / 1e12


def pct_peak(op: str, shape: str, dtype: str, median_ms: float,
             device: str, causal: bool = True) -> Optional[float]:
    """End-to-end compute attainment: achieved TFLOP/s ÷ the device's peak for
    ``dtype``. ``None`` when the op has no FLOP model or the device/dtype peak is
    unknown (never guessed). ``causal`` flows to :func:`op_flops` (the recorded
    flash rows are causal), so a row's floor and its live re-time use the same
    triangular FLOP basis — keep the default so the gate stays self-consistent."""
    ach = achieved_tflops(op, shape, median_ms, causal=causal)
    dev = DEVICE_PEAK.get(device)
    if ach is None or dev is None:
        return None
    peak = dev["peak_tflops"].get(dtype)
    if not peak:
        return None
    return ach / peak


def achieved_gbps(op: str, shape: str, dtype: str,
                  median_ms: float) -> Optional[float]:
    """Achieved DRAM bandwidth (GB/s) for a movement op — its min-traffic bytes
    ÷ wall-clock. ``None`` when the op has no byte model."""
    nbytes = op_bytes(op, shape, dtype)
    if nbytes is None or median_ms <= 0:
        return None
    return nbytes / (median_ms * 1e-3) / 1e9


def pct_peak_bw(op: str, shape: str, dtype: str, median_ms: float,
                device: str) -> Optional[float]:
    """End-to-end bandwidth attainment: achieved GB/s ÷ the device's peak DRAM
    bandwidth. The right absolute bar for the memory-bound movement lanes
    (gather/scatter), where FLOP % is meaningless (near-zero arithmetic). ``None``
    when the op has no byte model or the device's peak BW is unknown."""
    ach = achieved_gbps(op, shape, dtype, median_ms)
    dev = DEVICE_PEAK.get(device)
    if ach is None or dev is None:
        return None
    peak = dev.get("peak_bw_gb_s")
    if not peak:
        return None
    return ach / peak


def annotate_rows(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    """Annotate each row with its attainment, computed from the row's existing
    ``median_ms`` (no re-timing). A FLOP-modeled row gets ``achieved_tflops`` +
    ``pct_peak`` (compute attainment); a pure-movement row (``_MOVEMENT_OPS``,
    byte model but no FLOP model) gets ``achieved_gbps`` + ``pct_peak_bw``
    (bandwidth attainment). Rows with neither model are returned unchanged."""
    for r in rows:
        ach = achieved_tflops(r["op"], r["shape"], float(r["median_ms"]))
        if ach is not None:                    # compute-bound lane
            pk = pct_peak(r["op"], r["shape"], r["dtype"],
                          float(r["median_ms"]), device)
            r["achieved_tflops"] = round(ach, 4)
            if pk is not None:
                r["pct_peak"] = round(pk, 5)
            continue
        gbps = achieved_gbps(r["op"], r["shape"], r["dtype"],
                             float(r["median_ms"]))
        if gbps is not None:                   # memory-bound movement lane
            pkb = pct_peak_bw(r["op"], r["shape"], r["dtype"],
                              float(r["median_ms"]), device)
            r["achieved_gbps"] = round(gbps, 4)
            if pkb is not None:
                r["pct_peak_bw"] = round(pkb, 5)
    return rows


def evaluate_attainment(rows: list[Mapping[str, Any]],
                        baseline: Mapping[str, Any],
                        device: str) -> list[str]:
    """Gate measured rows against each baseline row's attainment floor — the
    absolute-attainment analog of the latency ratchet. A compute row carries
    ``attainment_floor`` (gated on ``pct_peak``); a memory-bound movement row
    carries ``bw_attainment_floor`` (gated on ``pct_peak_bw``). A row whose
    measured attainment falls below its floor fails; rows with no floor are not
    gated (opt-in); a floor with no matching measurement fails on coverage."""
    failures: list[str] = []

    def key(r: Mapping[str, Any]) -> tuple:
        return (r.get("op"), r.get("shape"), r.get("dtype"), r.get("mode"))

    # Two floor kinds, keyed the same way. `compute` reads pct_peak; `bw` reads
    # pct_peak_bw. A baseline row has at most one (an op is compute XOR movement).
    compute_floors = {key(r): float(r["attainment_floor"])
                      for r in baseline.get("rows", []) if "attainment_floor" in r}
    bw_floors = {key(r): float(r["bw_attainment_floor"])
                 for r in baseline.get("rows", []) if "bw_attainment_floor" in r}
    seen: set[tuple] = set()
    for row in rows:
        k = key(row)
        # Measured ratchet-report rows carry `latency_ms` (like evaluate_ratchet
        # reads); baseline/self-check rows carry `median_ms`. Accept either.
        t = float(row.get("latency_ms", row.get("median_ms", 0.0)))
        if k in compute_floors:
            seen.add(k)
            pk = pct_peak(row.get("op", ""), row.get("shape", ""),
                          row.get("dtype", ""), t, device)
            if pk is None:
                failures.append(f"{k[0]} {k[1]} {k[2]} {k[3]}: no attainment "
                                f"(missing FLOP model or peak for {device!r})")
            elif pk < compute_floors[k]:
                failures.append(
                    f"{k[0]} {k[1]} {k[2]} {k[3]}: pct_peak={pk:.4f} below floor "
                    f"{compute_floors[k]:.4f}")
        if k in bw_floors:
            seen.add(k)
            pkb = pct_peak_bw(row.get("op", ""), row.get("shape", ""),
                              row.get("dtype", ""), t, device)
            if pkb is None:
                failures.append(f"{k[0]} {k[1]} {k[2]} {k[3]}: no bw attainment "
                                f"(missing byte model or peak BW for {device!r})")
            elif pkb < bw_floors[k]:
                failures.append(
                    f"{k[0]} {k[1]} {k[2]} {k[3]}: pct_peak_bw={pkb:.5f} below "
                    f"floor {bw_floors[k]:.5f}")
    for k in sorted((compute_floors.keys() | bw_floors.keys()) - seen, key=str):
        failures.append(f"{k[0]} {k[1]} {k[2]} {k[3]}: no measurement "
                        f"(attainment coverage)")
    return failures
