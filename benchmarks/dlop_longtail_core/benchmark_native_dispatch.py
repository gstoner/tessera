#!/usr/bin/env python3
"""Observed native ANN dispatch lane, separate from the static DLOP catalog."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.native_ann_adapter import measure  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repeat', default=5, type=int)
    args = p.parse_args()
    rows = [measure(args.backend, args.compiler, activation=a, repeat=args.repeat)
            for a in ('relu', 'abs', 'square')]
    args.output.write_text(json.dumps(dict(schema=1, rows=rows, promotion_eligible=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
