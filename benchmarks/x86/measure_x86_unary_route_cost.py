"""Compile-cost probe for the x86 unary route (E2E-REAL-6, x86 unary family).

Times packaging (not execution) of x86 softmax / reduction through the
retired Graph-owned constructor (``tests/_support/x86_unary_baseline.py``) and
the compiled Graph -> Schedule -> Tile route (``x86_native.package_*``) for a
sequence of distinct shapes, interleaved so both routes see the same host state.

Neither route keeps an image cache: the x86 image payload is the prebuilt,
shape-free shared object, and every package call re-runs its lowering
subprocesses. This probe measures that per-call cost; it is a compile-cost
claim for the host that runs it, not a runtime claim.

Run on an x86 host with the shared image built:
    PYTHONPATH=python:. python benchmarks/x86/measure_x86_unary_route_cost.py
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time

from tessera.compiler import x86_native
from tests._support import x86_unary_baseline as baseline
from tests.unit.test_x86_unary_differential import _reduction, _softmax

SHAPES = [(3, 17), (4, 33), (2, 3, 5), (8, 64), (16, 128), (2, 7, 9), (32, 257), (5, 1)]


def _time(call) -> float:
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1e3


def main() -> int:
    if not x86_native.tools_available():
        print("x86 native image toolchain unavailable on this host", file=sys.stderr)
        return 2
    rows = []
    for family, build, old, new in (
        ("softmax", _softmax, baseline.package_softmax, x86_native.package_softmax),
        ("reduction", _reduction, baseline.package_reduction, x86_native.package_reduction),
    ):
        old_ms, new_ms = [], []
        for shape in SHAPES:
            module = build(shape=shape)
            old_ms.append(_time(lambda: old(module, pipeline_name="tessera-lower-to-x86")))
            new_ms.append(_time(lambda: new(module, pipeline_name="tessera-lower-to-x86")))
        rows.append({
            "family": family, "shapes": len(SHAPES),
            "retired_median_ms": round(statistics.median(old_ms), 2),
            "compiled_median_ms": round(statistics.median(new_ms), 2),
            "retired_ms": [round(v, 2) for v in old_ms],
            "compiled_ms": [round(v, 2) for v in new_ms],
        })
    print(json.dumps({"host": platform.node(), "claim": "compile_cost", "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
