#!/usr/bin/env python3
"""Native CUDA/HIP two-affine ANN adapter; no reference fallback."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from benchmarks.native_ann_adapter import measure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['nvidia', 'rocm'])
    parser.add_argument('--compiler', default=os.environ.get('TESSERA_OPT'), type=Path)
    parser.add_argument('--rows', default=16, type=int)
    parser.add_argument('--width', default=8, type=int)
    parser.add_argument('--activation', default='relu', choices=['relu', 'abs', 'square'])
    parser.add_argument('--repeat', default=5, type=int)
    args = parser.parse_args()
    if args.compiler is None:
        parser.error('--compiler or TESSERA_OPT is required')
    print(json.dumps(measure(**vars(args))))


if __name__ == '__main__':
    main()
