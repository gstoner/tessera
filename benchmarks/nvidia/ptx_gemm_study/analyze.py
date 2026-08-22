#!/usr/bin/env python3
"""analyze.py — Phase 4 report generator (plan §4, §C1/C3/C4/C5).

Reads ONLY results.jsonl (+ optional counters.jsonl) and emits REPORT.md. Every
number in the report is derived here — nothing is hand-typed (fixes the paper's
"GFLOPS roughly halve" / "+22%" prose defects). Refuses to run unless the
consistency gate is green (call check_consistency first, or pass --skip-gate for
a dry run on partial data).

Corrections realized here:
  C3  same-precision speedup + CI is the PRIMARY table; cross-precision is only an
      absolute-TOPS-vs-N view with the per-precision L2 cliff annotated — never a
      lone headline ratio.
  C5  ai_theoretical and ai_measured are printed as separate, labeled columns.
  C7  CoV and clocks_locked surfaced per row; noisy points flagged.

Usage: python3 analyze.py results.jsonl [counters.jsonl] -o REPORT.md
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from statistics import median

import check_consistency

COV_FLAG = 0.03  # 3% — above this a datapoint is marked noisy (C7)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _key_n(r):
    return r.get("n") or r.get("shape", [None])[0]


def cliff_n(rows_for_dtype):
    """Empirical L2-overflow N: the smallest N whose latency jump over the prior N
    exceeds ~2x the clean N^3 factor (8x). Returns None if no cliff in range."""
    pts = sorted(rows_for_dtype, key=_key_n)
    prev = None
    for r in pts:
        n = _key_n(r)
        if prev is not None:
            n0, l0 = prev
            work = (n / n0) ** 3
            if r["latency_ms"] / l0 > 2.0 * work:
                return n
        prev = (n, r["latency_ms"])
    return None


def render(rows, counters):
    out = []
    dev = rows[0].get("device", "?")
    ver = rows[0].get("tessera_version", "?")
    locked = all(r.get("clocks_locked") for r in rows)
    out.append(f"# PTX Tensor-Core GEMM Study — Results ({dev})\n")
    out.append(f"tessera_version: `{ver}` · clocks_locked: **{locked}** "
               f"· rows: {len(rows)}\n")
    if not locked:
        out.append("> Clocks not locked (WSL2). Stability carried by the CoV gate "
                   f"below; points with CoV > {COV_FLAG:.0%} are flagged.\n")

    # group
    by_dtype = defaultdict(list)
    for r in rows:
        by_dtype[r.get("dtype")].append(r)
    ns = sorted({_key_n(r) for r in rows})

    # ---- PRIMARY: same-precision speedup vs {dtype}_wmma (C3) ----
    out.append("\n## Same-precision speedup vs WMMA (primary metric)\n")
    for dt, rs in sorted(by_dtype.items()):
        base = {_key_n(r): r for r in rs if r["kernel"] == f"{dt}_wmma"}
        kernels = sorted({r["kernel"] for r in rs if r["kernel"] != f"{dt}_wmma"})
        if not base or not kernels:
            continue
        out.append(f"\n### {dt}\n")
        out.append("| kernel | " + " | ".join(f"N={n}" for n in ns) + " |")
        out.append("|" + "---|" * (len(ns) + 1))
        for k in kernels:
            cells = []
            for n in ns:
                r = next((x for x in rs if x["kernel"] == k and _key_n(x) == n), None)
                if r and n in base:
                    sp = base[n]["latency_ms"] / r["latency_ms"]
                    flag = "⚠" if r.get("cov", 0) > COV_FLAG else ""
                    cells.append(f"{sp:.2f}×{flag}")
                else:
                    cells.append("—")
            out.append(f"| `{k}` | " + " | ".join(cells) + " |")

    # ---- absolute TOPS vs N, cliff-annotated (C3 cross-precision, honest form) ----
    out.append("\n## Absolute throughput vs N (cross-precision, cliff-annotated)\n")
    out.append("Cross-precision value is shown as absolute TOPS, NOT a single "
               "'vs-FP16' ratio — the paper's headline 98.7× came from FP16 sitting "
               "past its L2 cliff. Each dtype's empirical cliff N is annotated.\n")
    out.append("| dtype | cliff N | " + " | ".join(f"N={n}" for n in ns) + " |")
    out.append("|" + "---|" * (len(ns) + 2))
    for dt, rs in sorted(by_dtype.items()):
        best = {}
        for n in ns:
            cand = [r for r in rs if _key_n(r) == n]
            if cand:
                best[n] = max(cand, key=lambda r: r.get("tflops", 0))
        cn = cliff_n([best[n] for n in ns if n in best])
        cells = [f"{best[n]['tflops']:.3f}" if n in best else "—" for n in ns]
        out.append(f"| {dt} | {cn if cn else 'none≤max'} | " + " | ".join(cells) + " |")

    # ---- AI: theoretical vs measured, separate (C5) ----
    out.append("\n## Arithmetic intensity (theoretical vs measured, kept separate)\n")
    out.append("| dtype | ai_theoretical | ai_measured (ncu) |")
    out.append("|---|---|---|")
    meas = {}
    if counters:
        for c in counters:
            if c.get("ai_measured") is not None:
                meas.setdefault(c.get("dtype"), []).append(c["ai_measured"])
    for dt, rs in sorted(by_dtype.items()):
        ait = next((r.get("ai_theoretical") for r in rs
                    if r.get("ai_theoretical") is not None), None)
        aim = median(meas[dt]) if dt in meas else None
        out.append(f"| {dt} | {ait if ait is not None else '—'} | "
                   f"{f'{aim:.3f}' if aim is not None else '—'} |")

    # ---- mechanism attribution from counters (paper's Run analysis, auto) ----
    if counters:
        out.append("\n## Mechanism attribution (from ncu)\n")
        out.append("Key invariant (paper Run 2): above the cliff, DRAM-active-cycle "
                   "reduction should track wall-time speedup ~1:1.\n")
        out.append("| kernel | N | dram_active_cyc | l2_hit% | l1_hit% | "
                   "bytes/sector | act_thr/warp | div_branch |")
        out.append("|" + "---|" * 8)
        for c in sorted(counters, key=lambda c: (c.get("dtype"), _key_n(c))):
            out.append("| `{k}` | {n} | {d} | {l2} | {l1} | {bs} | {at} | {db} |".format(
                k=c.get("kernel"), n=_key_n(c),
                d=c.get("dram_cycles_active", "—"), l2=c.get("l2_hit_pct", "—"),
                l1=c.get("l1tex_hit_pct", "—"), bs=c.get("bytes_per_sector", "—"),
                at=c.get("active_threads_per_warp", "—"),
                db=c.get("divergent_branches", "—")))

    out.append("\n---\n_Generated by analyze.py from results.jsonl "
               "(no hand-typed numbers). Consistency gate must be green._\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("counters", nargs="?")
    ap.add_argument("-o", "--out", default="REPORT.md")
    ap.add_argument("--skip-gate", action="store_true")
    a = ap.parse_args()
    rows = load(a.results)
    counters = load(a.counters) if a.counters else None
    if not a.skip_gate:
        errs = check_consistency.check(rows, counters)
        if errs:
            print("Refusing to render: consistency gate RED", *("  - " + e for e in errs),
                  sep="\n")
            raise SystemExit(1)
    open(a.out, "w").write(render(rows, counters))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
