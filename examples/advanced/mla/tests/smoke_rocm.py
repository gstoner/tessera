#!/usr/bin/env python3
"""Run the native ROCm MLA decode-step slice on the selected HIP device."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
for path in (ROOT, REPO / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mla.rocm_decode import run_rocm_decode_smoke


if __name__ == "__main__":
    print(run_rocm_decode_smoke())
