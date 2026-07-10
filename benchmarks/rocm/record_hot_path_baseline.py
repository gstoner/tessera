"""E2 (2026-07-06) — record the ROCm gfx1151 hot-path perf-ratchet baseline.

Times the WMMA matmul hot path through the SAME production C-ABI symbol
(`tessera_rocm_wmma_gemm_f16`, RDNA WMMA + f32 accumulate) that
`runtime._execute_rocm_wmma_artifact` dispatches to, then writes
`benchmarks/baselines/rocm_gfx1151_hot_paths.json` with per-row
thresholds = median * margin. The CI ratchet (`perf_gate.evaluate_ratchet`,
locked by `tests/unit/test_rocm_perf_ratchet.py`) re-times and fails on
regressions past the margin.

Host-gated exactly like the Apple lane: with no AMD GPU / GEMM lib the
runtime probe (`_rocm_wmma_runtime_available`) returns False, so this
skip-cleans (prints a note, writes nothing, exit 0) — it NEVER fabricates
GPU numbers (repo Decision #26; benchmarks/rocm/*.py honesty rule).

Run on the machine that hosts CI, once the gfx1151 GPU is live
(`hipGetDeviceCount` sees the device):

    python benchmarks/rocm/record_hot_path_baseline.py [--margin 2.0]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "benchmarks" / "baselines" / "rocm_gfx1151_hot_paths.json"

# WMMA matmul ladder, f16 storage / f32 accumulate — the executable ROCm hot
# path (compiler-generated matmul, repo Decision #26). Kept small and square so
# a ratchet regression pins to the primitive, not to shape edge cases.
HOT_PATH_SIZES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

# Compiled FA-2 flash_attn ladder, (batch, heads, seq, head_dim), f16 storage /
# f32 accumulate — the SECOND executable ROCm hot path (compiler-generated WMMA
# flash attention via ``rt._rocm_flash_attn``, which shells to ``tessera-opt``).
# head_dim 64/128 covers the common attention regimes; recorded only when the
# compiled lane is live (else the row skip-cleans — never fabricated).
FLASH_ATTN_SHAPES = [(1, 8, 512, 64), (1, 8, 1024, 64), (1, 16, 1024, 128)]

# Compiled FA-2 BACKWARD ladder (dQ/dK/dV), (batch, heads, seq, head_dim), f16
# storage / f32 accumulate — the reverse-mode hot path
# (rocm_flash_attn_bwd_compiled: fa_pre/fa_dkdv/fa_dq + the forward-O recompute).
# Correctness-first (no LDS/perf tuning), so pct_peak is expectedly small; it is
# now FLOP-modeled (roofline: 2.5× the forward — the FA-2 ratio), so these rows
# carry a real attainment floor, not just a latency cap (repo Decision #26).
FLASH_ATTN_BWD_SHAPES = [(1, 8, 512, 64), (1, 16, 1024, 128)]

# Register-blocked f32 GEMM ladder (M, N, K) — the plain-VALU f32 kernel
# (generate-rocm-gemm-f32-kernel) grouped-SwiGLU rides. TM×TN=4×4 output-tile
# register blocking is a ~1.6x win over one-thread-per-output at 1024³
# (STRIX_HALO Stage F: register-budget tiling is the lever). FLOP-modeled
# (roofline: 2·M·N·K, gated vs the 29.7 TF f32 peak), so its rows now carry an
# attainment floor alongside the latency cap.
GEMM_F32_SIZES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

# Memory-bound movement lanes — the gfx1151 gather/scatter kernels underneath the
# KV-cache and MoE-transport paths. These are BANDWIDTH-bound (near-zero FLOPs),
# so roofline gates them on achieved GB/s ÷ the 256 GB/s peak, not FLOP %. The
# ratchet shape is `RxW` (rows moved × row width in elements); op_bytes models
# read-rows + write-rows (2·R·W·4). Recorded only when the compiled lane is live.
#
# KV-cache append/read: (n_rows_moved, row_width). append writes N new rows into a
# (max_seq, row_width) buffer; read gathers M rows. `KV_ROW_WIDTH` = heads·head_dim.
KV_ROW_WIDTH = 8 * 64
KV_APPEND_ROWS = [128, 256]
KV_READ_ROWS = [256, 512]
KV_MAX_SEQ = 1024
# MoE dispatch/combine: (packed_slots, hidden). The plan routes `tokens` tokens
# top_k ways into `experts` experts at `capacity` each; the packed row count S is
# derived from the plan, so these shapes drive the plan, not the byte model.
MOE_HIDDEN = 512
MOE_CASES = [  # (tokens, experts, top_k, capacity)
    (512, 8, 2, 160),
    (1024, 8, 2, 320),
]


def _median_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def hot_path_cases(rt):
    """(op, shape, dtype, mode, thunk) per named hot path — module-level so the
    ratchet test re-times the IDENTICAL work through the production symbol.

    The matmul thunks call the shipped `tessera_rocm_wmma_gemm_f16` C-ABI symbol
    (host pointers; the lib does H2D/compute/D2H), the same symbol
    `_execute_rocm_wmma_artifact` dispatches to. Raises if the lib/symbol is
    absent — callers gate on `_rocm_wmma_runtime_available()` first.

    The flash_attn thunks call `rt._rocm_flash_attn` (the compiler-generated
    WMMA FA-2 forward). They are appended only when
    `_rocm_compiled_flash_attn_available()` is True, so a box with the GEMM lane
    but no compiled flash lane records matmul rows only (never fabricated)."""
    lib = rt._load_rocm_gemm_runtime()
    if lib is None:
        raise RuntimeError("libtessera_rocm_gemm.so not loadable — no ROCm lane")
    sym = rt._ROCM_GEMM_SYMBOLS["float16"]
    fn = getattr(lib, sym, None)
    if fn is None:
        raise RuntimeError(f"libtessera_rocm_gemm.so lacks {sym}")

    rng = np.random.default_rng(0)

    def _make(m, n, k):
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        d = np.zeros((m, n), np.float32)

        def _run():
            rc = fn(a.ctypes.data_as(ctypes.c_void_p),
                    b.ctypes.data_as(ctypes.c_void_p),
                    d.ctypes.data_as(ctypes.c_void_p), m, n, k)
            if rc != 0:
                raise RuntimeError(f"{sym} rc={rc} at {m}x{n}x{k}")
        return _run

    def _make_fa(b, h, s, d):
        q = rng.standard_normal((b, h, s, d)).astype(np.float16)
        k = rng.standard_normal((b, h, s, d)).astype(np.float16)
        v = rng.standard_normal((b, h, s, d)).astype(np.float16)

        def _run():
            rt._rocm_flash_attn(q, k, v, causal=True)
        return _run

    def _make_fa_bwd(b, h, s, d):
        q = rng.standard_normal((b, h, s, d)).astype(np.float16)
        k = rng.standard_normal((b, h, s, d)).astype(np.float16)
        v = rng.standard_normal((b, h, s, d)).astype(np.float16)
        do = rng.standard_normal((b, h, s, d)).astype(np.float16)
        art = rt.RuntimeArtifact(metadata={
            "target": "rocm", "compiler_path": "rocm_flash_attn_bwd_compiled",
            "executable": True, "execution_kind": "native_gpu",
            "arg_names": ["do", "q", "k", "v"], "output_name": "g",
            "ops": [{"op_name": "tessera.flash_attn_bwd", "result": "g",
                     "operands": ["do", "q", "k", "v"],
                     "kwargs": {"causal": True}}]})

        def _run():
            res = rt.launch(art, (do, q, k, v))
            if not res["ok"]:
                raise RuntimeError(res.get("reason"))
        return _run

    def _make_gemm_f32(m, n, k):
        a = rng.standard_normal((m, k)).astype(np.float32)
        bb = rng.standard_normal((k, n)).astype(np.float32)

        def _run():
            rt._rocm_f32_gemm(a, bb, np)
        return _run

    # ── memory-bound movement thunks (gather/scatter) ────────────────────────
    def _kv_art(op_name, arg_names, kwargs):
        return rt.RuntimeArtifact(metadata={
            "target": "rocm", "compiler_path": "rocm_kv_cache_compiled",
            "executable": True, "arg_names": list(arg_names),
            "ops": [{"op_name": op_name, "operands": list(arg_names),
                     "kwargs": kwargs}]})

    def _make_kv_append(n_rows):
        buf = rng.standard_normal((KV_MAX_SEQ, KV_ROW_WIDTH)).astype(np.float32)
        new = rng.standard_normal((n_rows, KV_ROW_WIDTH)).astype(np.float32)
        art = _kv_art("tessera.kv_cache.append", ["buf", "new"], {"start": 0})

        def _run():
            res = rt.launch(art, (buf, new))
            if not res["ok"]:
                raise RuntimeError(res.get("reason"))
        return _run

    def _make_kv_read(m_rows):
        buf = rng.standard_normal((KV_MAX_SEQ, KV_ROW_WIDTH)).astype(np.float32)
        art = _kv_art("tessera.kv_cache.read", ["buf"], {"start": 0, "end": m_rows})

        def _run():
            res = rt.launch(art, (buf,))
            if not res["ok"]:
                raise RuntimeError(res.get("reason"))
        return _run

    def _moe_art(op_name, arg_names):
        return rt.RuntimeArtifact(metadata={
            "target": "rocm", "compiler_path": "rocm_moe_transport_compiled",
            "executable": True, "arg_names": list(arg_names),
            "ops": [{"op_name": op_name}]})

    def _moe_plan(tokens, experts, top_k, capacity):
        from tessera.stdlib import moe
        eids = rng.integers(0, experts, size=(tokens, top_k), dtype=np.int64)
        w = rng.random((tokens, top_k), dtype=np.float32)
        w /= w.sum(axis=1, keepdims=True)
        return moe.plan_dispatch(eids, w, experts, capacity=capacity)

    def _make_moe_dispatch(plan, packed_rows):
        x = rng.standard_normal((int(plan.num_tokens), MOE_HIDDEN)).astype(np.float32)
        art = _moe_art("tessera.moe_dispatch", ["x", "plan"])

        def _run():
            res = rt.launch(art, {"x": x, "plan": plan})
            if not res["ok"]:
                raise RuntimeError(res.get("reason"))
        return _run

    def _make_moe_combine(plan, packed_rows):
        partials = rng.standard_normal((packed_rows, MOE_HIDDEN)).astype(np.float32)
        art = _moe_art("tessera.moe_combine", ["partials", "plan"])

        def _run():
            res = rt.launch(art, {"partials": partials, "plan": plan})
            if not res["ok"]:
                raise RuntimeError(res.get("reason"))
        return _run

    cases = []
    for (m, n, k) in HOT_PATH_SIZES:
        cases.append(("matmul", f"{m}x{n}x{k}", "f16", "wmma", _make(m, n, k)))
    if rt._rocm_compiled_flash_attn_available():
        for (b, h, s, d) in FLASH_ATTN_SHAPES:
            cases.append(("flash_attn", f"{b}x{h}x{s}x{d}", "f16",
                          "flash_attn", _make_fa(b, h, s, d)))
        for (b, h, s, d) in FLASH_ATTN_BWD_SHAPES:
            cases.append(("flash_attn_bwd", f"{b}x{h}x{s}x{d}", "f16",
                          "flash_attn_bwd", _make_fa_bwd(b, h, s, d)))
    # f32 GEMM gates on its OWN probe (generate-rocm-gemm-f32-kernel is a
    # DIFFERENT pass from the flash lane), so a host where the f32 GEMM works but
    # the flash lane is missing/broken still records its gemm_f32 ratchet rows.
    if rt._rocm_compiled_gemm_f32_available():
        for (m, n, k) in GEMM_F32_SIZES:
            cases.append(("gemm_f32", f"{m}x{n}x{k}", "f32", "gemm_f32",
                          _make_gemm_f32(m, n, k)))
    # Movement lanes (gather/scatter). Gate on tessera-opt + a usable GPU (the
    # kv/moe executors build a scatter/gather hsaco and launch it) — same probe
    # the moe-transport test uses. Bandwidth-modeled: shape `RxW` = rows × width.
    if rt._tessera_opt_path() is not None and rt._rocm_wmma_runtime_available():
        from tessera.stdlib import moe
        for n in KV_APPEND_ROWS:
            cases.append(("kv_cache_append", f"{n}x{KV_ROW_WIDTH}", "f32",
                          "kv_cache_append", _make_kv_append(n)))
        for m in KV_READ_ROWS:
            cases.append(("kv_cache_read", f"{m}x{KV_ROW_WIDTH}", "f32",
                          "kv_cache_read", _make_kv_read(m)))
        for (tokens, experts, top_k, capacity) in MOE_CASES:
            plan = _moe_plan(tokens, experts, top_k, capacity)
            # S = packed slot count (kept, expert-sorted) — derive from the plan's
            # oracle dispatch so the ratchet shape matches the bytes actually moved.
            x0 = rng.standard_normal((tokens, MOE_HIDDEN)).astype(np.float32)
            packed_rows = int(moe.dispatch(x0, plan).shape[0])
            shape = f"{packed_rows}x{MOE_HIDDEN}"
            cases.append(("moe_dispatch", shape, "f32", "moe_dispatch",
                          _make_moe_dispatch(plan, packed_rows)))
            cases.append(("moe_combine", shape, "f32", "moe_combine",
                          _make_moe_combine(plan, packed_rows)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=2.0,
                        help="threshold = median * margin (CI noise headroom)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    from tessera import runtime as rt

    if not rt._rocm_wmma_runtime_available():
        print("rocm WMMA runtime unavailable (no AMD GPU / GEMM lib); "
              "skipping baseline record (no numbers fabricated)")
        return 0

    rows = []
    for op, shape, dtype, mode, thunk in hot_path_cases(rt):
        med = _median_ms(thunk, reps=args.reps)
        rows.append({
            "op": op, "shape": shape, "dtype": dtype, "mode": mode,
            "median_ms": round(med, 4),
            "max_latency_ms": round(med * args.margin, 4),
        })
        print(f"{op:12s} {shape:16s} median {med:8.3f} ms  "
              f"cap {med * args.margin:8.3f} ms")
    # Workstream J: annotate each FLOP-modeled row with roofline attainment
    # (achieved TFLOP/s + pct_peak) + an attainment_floor = pct_peak / margin,
    # symmetric with the latency cap so `perf_gate --attainment` can ratchet it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import math

    import roofline as _rl
    _rl.annotate_rows(rows, f"rocm:{rt._rocm_chip()}")
    for r in rows:
        # Round every floor DOWN (not to-nearest): a floor rounded *up* can
        # exceed attainment/margin, so a run inside the latency cap (margin×
        # median) would still trip the attainment gate. Flooring keeps the
        # absolute gate no stricter than the relative latency margin. Compute
        # rows carry pct_peak → attainment_floor; movement rows carry pct_peak_bw
        # → bw_attainment_floor (one or the other, never both).
        if "pct_peak" in r:
            r["attainment_floor"] = math.floor(
                r["pct_peak"] / args.margin * 1e5) / 1e5
        elif "pct_peak_bw" in r:
            r["bw_attainment_floor"] = math.floor(
                r["pct_peak_bw"] / args.margin * 1e5) / 1e5
    OUT.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1",
        "margin": args.margin,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
