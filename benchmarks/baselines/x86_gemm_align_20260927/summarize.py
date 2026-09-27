#!/usr/bin/env python3
"""Summarize probe_gemm_align.py JSONL: per (shape, B%64), the range of the
per-process medians before and after, the after/before ratio range, and the
numerics flags. Prints a Markdown table; `--check` exits non-zero if any row
lacks bitwise equality or a shape's M > 1 alignment effect after the fix
exceeds `--max-spread` (default 5%). Shapes whose after-fix call is under
10 µs are not checked: there the ~4.5 µs ctypes call itself is most of the
number (32^3 reads 4.5-4.9 µs before AND after).

The alignment effect is the spread of the per-offset BEST process
(max over B%64 of the fastest process / min over B%64 of the fastest
process): a process-level lottery that does not follow B%64 (present before
and after the fix, see README) moves medians of three processes but not the
best-of-N per offset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-spread", type=float, default=0.05)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.jsonl) if line.strip()]
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["M"], r["N"], r["K"], r["b_mod64"])].append(r)
    host = {r["host"] for r in rows}
    print(f"host(s): {', '.join(sorted(host))}; processes: {len(rows)}; "
          f"timing_source: {sorted({r['timing_source'] for r in rows})}; "
          f"max TSC/raw disagreement: {max(r['max_tsc_raw_disagreement'] for r in rows):.2e}")
    print()
    print("| M×N×K | B%64 | before µs (min–max) | after µs (min–max) | after/before | bitwise | launch=direct |")
    print("|---|---|---|---|---|---|---|")
    failures = []
    by_shape: dict = defaultdict(dict)
    for (m, n, k, off), rs in sorted(groups.items()):
        b = [r["before_us"] for r in rs]
        a = [r["after_us"] for r in rs]
        ratio = [r["after_over_before"] for r in rs]
        bit = all(r["bitwise_before_eq_after"] for r in rs)
        launch = {r["launch_eq_direct_after"] for r in rs}
        by_shape[(m, n, k)][off] = (b, a)
        fmt = (lambda v: f"{min(v):.1f}–{max(v):.1f}") if max(b) >= 10 else (lambda v: f"{min(v):.2f}–{max(v):.2f}")
        print(f"| {m}×{n}×{k} | {off} | {fmt(b)} | {fmt(a)} | {min(ratio):.2f}–{max(ratio):.2f} | "
              f"{'yes' if bit else 'NO'} | {'/'.join(str(x) for x in sorted(launch, key=str))} |")
        if not bit or False in launch:
            failures.append(f"{m}x{n}x{k} B%64={off}: numerics mismatch")
    print()
    print("| M×N×K | before: alignment effect | after: alignment effect | after: per-process spread (all offsets) |")
    print("|---|---|---|---|")
    for shape, per in sorted(by_shape.items()):
        bm = [min(b) for b, _ in per.values()]
        am = [min(a) for _, a in per.values()]
        spread_b, spread_a = max(bm) / min(bm), max(am) / min(am)
        every = [x for _, a in per.values() for x in a]
        print(f"| {'×'.join(map(str, shape))} | {spread_b:.2f} | {spread_a:.2f} | "
              f"{max(every) / min(every):.2f} |")
        if shape[0] > 1 and min(every) >= 10.0 and spread_a - 1 > args.max_spread:
            failures.append(f"{shape}: after-fix alignment spread {spread_a:.2f}")
    if args.check and failures:
        print("\nFAIL:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
