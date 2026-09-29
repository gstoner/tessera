"""Compile-cost probe for the x86 package compile cache (E2E-REAL-6, 2026-09-28).

Times one Graph-input package call through the compiled Graph -> Schedule ->
Tile route with cold package and compiler caches (cleared before each sample: every
``tessera-opt`` run, the ``--version`` probe and the shared-object read happen)
and then warm (exact repeat: every compiler boundary is a memo hit, the
descriptor and replay comparisons are rebuilt). The retired Graph-owned
constructor is timed alongside, interleaved and also from a cleared cache,
as the reference point. The compiler binary's
SHA-256 is memoized per process on its stat signature and is paid once before
the first row (``process_first_call_ms`` reports that one-off).

This is a compile-cost claim for the host that runs it, nothing more: it says
nothing about kernel speed. Run on an x86 host with the AVX-512 shared object,
serialized with other timing work:
    flock /tmp/tessera-timing.lock env PYTHONPATH=python:. \\
        python benchmarks/x86/measure_x86_package_cache.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

from tessera.compiler import (scheduled_absolute, scheduled_kernel, x86_breadth,
                              x86_compile_cache, x86_native)
from tests._support import x86_kernel_baseline as kernel_baseline
from tests._support import x86_unary_baseline as unary_baseline
from tests.unit.test_x86_kernel_differential import _module
from tests.unit.test_x86_unary_differential import _reduction, _softmax

PIPELINE = "tessera-lower-to-x86"
WARM_REPEATS = 7
COLD_REPEATS = 5


def _cases():
    f32 = "fp32"
    yield ("softmax", _softmax((16, 128)), unary_baseline.package_softmax, x86_native.package_softmax)
    yield ("reduction", _reduction("tessera.sum", (16, 128)), unary_baseline.package_reduction,
           x86_native.package_reduction)
    yield ("elementwise.sub", _module("tessera.sub", [("a", (16, 128), f32), ("b", (16, 128), f32)],
                                      ((16, 128), f32)),
           kernel_baseline.package_elementwise, x86_native.package_elementwise)
    yield ("elementwise.exp", _module("tessera.exp", [("x", (16, 128), f32)], ((16, 128), f32)),
           kernel_baseline.package_elementwise, x86_native.package_elementwise)
    yield ("elementwise.abs", _module("tessera.absolute", [("x", (16, 128), f32)], ((16, 128), f32)),
           kernel_baseline.package_elementwise, x86_native.package_elementwise)
    yield ("cohort2.argmax", _module("tessera.argmax", [("x", (16, 128), f32)], ((16,), "int32"),
                                     {"axis": -1}),
           kernel_baseline.package_cohort2, x86_native.package_cohort2)
    yield ("cohort2.rmsnorm", _module("tessera.rmsnorm", [("x", (16, 128), f32)], ((16, 128), f32)),
           kernel_baseline.package_cohort2, x86_native.package_cohort2)
    yield ("cohort2.rope", _module("tessera.rope", [("x", (16, 128), f32), ("t", (16, 128), f32)],
                                   ((16, 128), f32)),
           kernel_baseline.package_cohort2, x86_native.package_cohort2)
    yield ("breadth.gather", _module("tessera.gather", [("s", (4096,), f32), ("i", (512,), "int64")],
                                     ((512,), f32)),
           kernel_baseline.package_graph_breadth, x86_breadth.package_graph_breadth)
    yield ("breadth.mse_loss", _module("tessera.mse_loss", [("p", (4096,), f32), ("t", (4096,), f32)],
                                       ((4096,), f32), {"reduction": "none"}),
           kernel_baseline.package_graph_breadth, x86_breadth.package_graph_breadth)
    yield ("breadth.cholesky", _module("tessera.cholesky", [("m", (32, 32), f32)], ((32, 32), f32)),
           kernel_baseline.package_graph_breadth, x86_breadth.package_graph_breadth)

    yield ("breadth.cholesky_batched", _module("tessera.cholesky", [("m", (2, 32, 32), f32)],
                                               ((2, 32, 32), f32)),
           kernel_baseline.package_graph_breadth, x86_breadth.package_graph_breadth)
    yield ("breadth.tri_solve_batched", _module("tessera.tri_solve",
              [("m", (2, 32, 32), f32), ("rhs", (2, 32, 5), f32)], ((2, 32, 5), f32)),
           kernel_baseline.package_graph_breadth, x86_breadth.package_graph_breadth)


def _ms(call) -> float:
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1e3


def _clear_package_caches() -> None:
    # #875 added verified-artifact caches above the compiler-run cache. A cold
    # sample must clear every layer or it silently times a package cache hit.
    x86_compile_cache.clear()
    with scheduled_kernel._X86_GRAPH_CACHE_LOCK:
        scheduled_kernel._X86_GRAPH_CACHE.clear()
    with scheduled_absolute._LOWER_CACHE_LOCK:
        scheduled_absolute._LOWER_CACHE.clear()
    with x86_native._UNARY_PACKAGE_CACHE_LOCK:
        x86_native._SCHEDULED_UNARY_PACKAGE_CACHE.clear()


def main() -> int:
    if not x86_native.tools_available():
        print("x86 native image toolchain unavailable on this host", file=sys.stderr)
        return 2
    _clear_package_caches()
    first_module = next(_cases())[1]
    process_first = _ms(lambda: x86_native.package_softmax(first_module, pipeline_name=PIPELINE))
    rows = []
    def cold_ms(packager, module) -> tuple[float, int]:
        _clear_package_caches()
        ms = _ms(lambda: packager(module, pipeline_name=PIPELINE))
        return ms, x86_compile_cache.stats()["misses"]

    for name, module, retired, compiled in _cases():
        # Interleaved, each sample from an empty cache: the retired constructor
        # also reaches `tessera-opt` through `x86_native._lower`, which is now
        # memoized, so an uncleared repeat would time a hit, not the route.
        retired_cold, compiled_cold, compiled_runs = [], [], []
        for _ in range(COLD_REPEATS):
            retired_cold.append(cold_ms(retired, module)[0])
            ms, runs = cold_ms(compiled, module)
            compiled_cold.append(ms)
            compiled_runs.append(runs)
        before = x86_compile_cache.stats()["misses"]
        warm = [_ms(lambda: compiled(module, pipeline_name=PIPELINE)) for _ in range(WARM_REPEATS)]
        rows.append({
            "case": name,
            "retired_cold_median_ms": round(statistics.median(retired_cold), 2),
            "compiled_cold_median_ms": round(statistics.median(compiled_cold), 2),
            "compiled_warm_median_ms": round(statistics.median(warm), 2),
            "retired_cold_ms": [round(v, 2) for v in retired_cold],
            "compiled_cold_ms": [round(v, 2) for v in compiled_cold],
            "compiled_warm_ms": [round(v, 2) for v in warm],
            "cold_compiler_runs": max(compiled_runs),
            "warm_compiler_runs": x86_compile_cache.stats()["misses"] - before,
        })
    print(json.dumps({
        "host": platform.node(), "claim": "compile_cost",
        "warm_repeats": WARM_REPEATS, "cold_repeats": COLD_REPEATS,
        "load_average": list(os.getloadavg()),
        "process_first_call_ms": round(process_first, 2),
        "rows": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
