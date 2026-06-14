#!/usr/bin/env python3
"""Compare two per-step loss logs and emit a PASS/FAIL verdict.

Usage:
    compare.py BASE.log FEAT.log [--rtol R] [--atol A]

Log format: one loss per line as a float, OR "step<whitespace>loss" per line.
Blank lines and lines starting with '#' are ignored.

Exit 0 + "PASS" iff the logs have equal length and every step agrees within
``abs(b - f) <= atol + rtol * abs(b)``. Exit 1 + "FAIL: ..." at the first
divergence (or on length mismatch / parse error).
"""

from __future__ import annotations

import argparse


def _read_losses(path: str) -> list[float]:
    losses: list[float] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tok = parts[-1]  # last column is the loss ("step<TAB>loss" or "loss")
            try:
                losses.append(float(tok))
            except ValueError:
                print(f"FAIL: {path}:{lineno}: cannot parse loss from {line!r}")
                raise SystemExit(1)
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("feat")
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    base = _read_losses(args.base)
    feat = _read_losses(args.feat)

    if not base or not feat:
        print(f"FAIL: empty log (base={len(base)} steps, feat={len(feat)} steps)")
        raise SystemExit(1)
    if len(base) != len(feat):
        print(f"FAIL: length mismatch (base={len(base)} steps, feat={len(feat)} steps)")
        raise SystemExit(1)

    worst = 0.0
    for i, (b, f) in enumerate(zip(base, feat)):
        tol = args.atol + args.rtol * abs(b)
        diff = abs(b - f)
        worst = max(worst, diff)
        if diff > tol:
            print(
                f"FAIL: step {i}: base={b:.6g} feat={f:.6g} "
                f"|Δ|={diff:.3g} > tol={tol:.3g} (rtol={args.rtol}, atol={args.atol})"
            )
            raise SystemExit(1)

    print(
        f"PASS: {len(base)} steps within tol "
        f"(rtol={args.rtol}, atol={args.atol}); worst |Δ|={worst:.3g}"
    )


if __name__ == "__main__":
    main()
