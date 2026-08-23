#!/usr/bin/env python3
"""check_consistency.py — the anti-Table-V gate (plan §C1, §4 Phase 4).

Reads the single source of truth ``results.jsonl`` (and optionally
``counters.jsonl``) and FAILS (exit 1) if any internal inconsistency exists. This
is the mechanized form of the hand-audit that found the paper's Table V
contradiction: TOPS must equal 2N^3/latency, reconstructed speedups must match,
AI ratios must track byte ratios, and no (kernel, dtype, N) may appear with two
different durations. Report generation is blocked on a red gate.

Usage:  python3 check_consistency.py results.jsonl [counters.jsonl]
Self-test:  python3 check_consistency.py --selftest
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict

TOPS_TOL = 0.005      # 0.5% — TOPS vs 2N^3/latency
SPEEDUP_TOL = 0.01    # 1%
AI_TOL = 0.02         # 2% — cross-precision AI doubling
DUR_XCHECK_TOL = 0.15 # 15% — ncu duration vs event median (profiling overhead)

BYTES_PER_ELEM = {"fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5,
                  "fp8": 1, "nvfp4": 0.5, "fp4_e2m1": 0.5, "fp32": 4}


def _fail(errs: list[str], msg: str) -> None:
    errs.append(msg)


#: statuses that mean the row was disqualified before/at timing — it carries no
#: latency/tflops on purpose and must be EXCLUDED from the numeric checks (else a
#: correctly-rejected kernel or the supported NO_INT4=1 fallback turns Phase 4
#: red and blocks the report for the valid candidates). Their metadata is still
#: validated.
DISQUALIFIED = {"WRONG", "EXEC_FAIL", "COMPILED_OUT", "SKIPPED_BY_PROBE"}


def _measured(r: dict) -> bool:
    """A row that carries a real measurement: non-null latency and not flagged
    as disqualified. Only these rows feed the quantitative gate."""
    return r.get("latency_ms") is not None and r.get("status", "OK") not in DISQUALIFIED


def check(rows: list[dict], counters: list[dict] | None = None) -> list[str]:
    errs: list[str] = []
    if not rows:
        return ["results.jsonl is empty"]

    measured = [r for r in rows if _measured(r)]

    # A disqualified row must not smuggle in a latency/tflops (that would be a
    # contradiction, not a clean rejection).
    for r in rows:
        if r.get("status") in DISQUALIFIED and (
                r.get("latency_ms") is not None or r.get("tflops") is not None):
            _fail(errs, f"{r.get('status')} row carries latency/tflops "
                        f"(should be null): {r.get('kernel')}")

    # 1) TOPS == 2 N^3 / latency  (square GEMM assumed via shape MxNxK == N^3)
    for r in measured:
        n = r.get("n") or r.get("shape", [None])[0]
        lat = r.get("latency_ms")
        tops = r.get("tflops")
        if n is None or lat in (None, 0) or tops is None:
            _fail(errs, f"measured row missing n/latency/tflops: {r.get('kernel')}@{n}")
            continue
        expect = (2 * n ** 3) / (lat / 1e3) / 1e12
        if abs(expect - tops) / max(expect, 1e-12) > TOPS_TOL:
            _fail(errs, f"TOPS mismatch {r['kernel']}@{n}: "
                        f"reported {tops:.4f} vs 2N^3/lat {expect:.4f}")

    # 2) No (kernel, dtype, n) appears with two different latencies — the exact
    #    failure mode of Table V (fp16_wmma at two contradictory values).
    seen: dict[tuple, float] = {}
    for r in measured:
        n = r.get("n") or r.get("shape", [None])[0]
        key = (r["kernel"], r.get("dtype"), n)
        lat = r.get("latency_ms")
        if key in seen and abs(seen[key] - lat) / max(seen[key], 1e-12) > 0.02:
            _fail(errs, f"duplicate {key} with conflicting latency "
                        f"{seen[key]} vs {lat}")
        seen.setdefault(key, lat)

    # 3) reported same-precision speedups reconstruct from latencies
    #    (a row may carry speedup_vs_wmma; verify against the baseline latency)
    by_dt_n_kernel = {(r.get("dtype"), r.get("n") or r.get("shape", [None])[0],
                       r["kernel"]): r for r in measured}
    for r in measured:
        sp = r.get("speedup_vs_wmma")
        if sp is None:
            continue
        n = r.get("n") or r.get("shape", [None])[0]
        base = by_dt_n_kernel.get((r.get("dtype"), n, f"{r.get('dtype')}_wmma"))
        if base is None:
            continue
        expect = base["latency_ms"] / r["latency_ms"]
        if abs(expect - sp) / max(expect, 1e-12) > SPEEDUP_TOL:
            _fail(errs, f"speedup mismatch {r['kernel']}@{n}: "
                        f"reported {sp:.3f} vs latency ratio {expect:.3f}")

    # 4) AI scaling: operand bytes shrink by precision, but the output store is
    #    always four bytes here (f32/s32).  Therefore it is *not* a literal 2x
    #    doubling: AI(bpe) = 2N / (2*bpe + 4).  This catches a real accounting
    #    drift without falsely rejecting the correct output-inclusive model.
    # Representation is part of the memory contract.  In particular, the
    # pre-expanded FP16 INT4 baseline must not be compared as packed INT4.
    ai_by = defaultdict(dict)  # (role,representation,n) -> {dtype: ai_theoretical}
    for r in measured:
        ai = r.get("ai_theoretical")
        if ai is None:
            continue
        n = r.get("n") or r.get("shape", [None])[0]
        ai_by[(r.get("role", "opt"), r.get("representation", "native"), n)][r.get("dtype")] = ai
    for (_role, _representation, _n), d in ai_by.items():
        if "fp16" in d and "int8" in d:
            expect = (2 * BYTES_PER_ELEM["fp16"] + 4) / (2 * BYTES_PER_ELEM["int8"] + 4)
            if abs(d["int8"] / d["fp16"] - expect) > AI_TOL:
                _fail(errs, f"AI ratio int8/fp16 != {expect:.3f} at n={_n}: {d['int8']/d['fp16']:.3f}")
        if "int8" in d and "int4" in d:
            expect = (2 * BYTES_PER_ELEM["int8"] + 4) / (2 * BYTES_PER_ELEM["int4"] + 4)
            if abs(d["int4"] / d["int8"] - expect) > AI_TOL:
                _fail(errs, f"AI ratio int4/int8 != {expect:.3f} at n={_n}: {d['int4']/d['int8']:.3f}")

    # 5) cross-check ncu durations against event medians (C2) — mismatch reported,
    #    never silently used.
    if counters:
        med = {(r["kernel"], r.get("dtype"), r.get("n") or r.get("shape", [None])[0]):
               r["latency_ms"] for r in measured}
        for c in counters:
            n = c.get("n") or c.get("shape", [None])[0]
            k = (c.get("kernel"), c.get("dtype"), n)
            ncu_ms = c.get("ncu_duration_ms")
            timing_scope = next((r.get("timing_scope", "kernel") for r in measured
                                 if (r["kernel"], r.get("dtype"), r.get("n") or r.get("shape", [None])[0]) == k), None)
            # NCU reports a library's selected internal kernel; CUDA events
            # measure its public call.  The two scopes are recorded together
            # but intentionally not compared as if they were one duration.
            if k in med and ncu_ms and timing_scope == "kernel":
                rel = abs(ncu_ms - med[k]) / max(med[k], 1e-12)
                if rel > DUR_XCHECK_TOL:
                    _fail(errs, f"ncu vs event duration divergence {k}: "
                                f"ncu {ncu_ms} vs event {med[k]} ({rel*100:.0f}%)")

    # 6) CoV honesty (C7): every row must carry cov and clocks_locked
    for r in rows:
        if "cov" not in r or "clocks_locked" not in r:
            _fail(errs, f"row missing cov/clocks_locked (C7): {r.get('kernel')}")
    return errs


def _selftest() -> int:
    good = [
        {"kernel": "fp16_wmma", "dtype": "fp16", "n": 2048, "latency_ms": 11.85,
         "tflops": (2*2048**3)/(11.85/1e3)/1e12, "ai_theoretical": 512.0,
         "cov": 0.01, "clocks_locked": False, "role": "opt"},
        {"kernel": "int8_wmma", "dtype": "int8", "n": 2048, "latency_ms": 8.55,
         "tflops": (2*2048**3)/(8.55/1e3)/1e12, "ai_theoretical": 682.6666667,
         "cov": 0.01, "clocks_locked": False, "role": "opt"},
        {"kernel": "int8_ptx_mma_k32", "dtype": "int8", "n": 2048, "latency_ms": 5.41,
         "tflops": (2*2048**3)/(5.41/1e3)/1e12, "speedup_vs_wmma": 8.55/5.41,
         "cov": 0.01, "clocks_locked": False, "role": "opt"},
    ]
    assert check(good) == [], check(good)
    # A pre-expanded FP16 baseline is logically INT4 but has FP16 operand
    # traffic.  It must remain outside packed-INT4 AI comparisons.
    representation_split = good + [
        {"kernel": "int4_ptx_mma_k64", "dtype": "int4", "n": 2048, "latency_ms": 6.4,
         "tflops": (2*2048**3)/(6.4/1e3)/1e12, "ai_theoretical": 819.2,
         "cov": 0.01, "clocks_locked": False, "role": "opt", "representation": "native"},
        {"kernel": "int4_wmma_preexpanded_fp16", "dtype": "int4", "n": 2048, "latency_ms": 11.0,
         "tflops": (2*2048**3)/(11.0/1e3)/1e12, "ai_theoretical": 512.0,
         "cov": 0.01, "clocks_locked": False, "role": "opt", "representation": "preexpanded_fp16"},
    ]
    assert check(representation_split) == [], check(representation_split)
    # disqualified rows (null latency) must NOT turn the gate red
    with_skips = good + [
        {"kernel": "int4_ptx_mma_k64", "dtype": "int4", "n": 2048,
         "latency_ms": None, "tflops": None, "cov": None, "clocks_locked": False,
         "status": "SKIPPED_BY_PROBE", "ai_theoretical": 10.92},
        {"kernel": "int4_wmma", "dtype": "int4", "n": 2048, "latency_ms": None,
         "tflops": None, "cov": None, "clocks_locked": False, "status": "COMPILED_OUT"},
    ]
    assert check(with_skips) == [], check(with_skips)
    # but a disqualified row that smuggles in a latency IS an error
    bad_skip = good + [{"kernel": "x", "dtype": "int4", "n": 512, "latency_ms": 1.0,
                        "tflops": 1.0, "cov": 0.0, "clocks_locked": False,
                        "status": "EXEC_FAIL"}]
    assert any("carries latency" in e for e in check(bad_skip)), check(bad_skip)
    # inject the Table-V bug: same key, contradictory latency
    bad = good + [{"kernel": "fp16_wmma", "dtype": "fp16", "n": 2048,
                   "latency_ms": 680.1, "tflops": (2*2048**3)/(680.1/1e3)/1e12,
                   "cov": 0.01, "clocks_locked": False, "role": "opt"}]
    errs = check(bad)
    assert any("conflicting latency" in e for e in errs), errs
    # inject a TOPS lie
    bad2 = [{"kernel": "x", "dtype": "fp16", "n": 1024, "latency_ms": 1.0,
             "tflops": 999.0, "cov": 0.0, "clocks_locked": True}]
    assert any("TOPS mismatch" in e for e in check(bad2))
    print("selftest OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    if len(sys.argv) < 2:
        print(__doc__); return 2
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    counters = None
    if len(sys.argv) > 2:
        counters = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
    errs = check(rows, counters)
    if errs:
        print("CONSISTENCY GATE: RED", file=sys.stderr)
        for e in errs:
            print("  - " + e, file=sys.stderr)
        return 1
    print("CONSISTENCY GATE: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
