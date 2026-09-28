"""Compile- and load-cost probe for the x86 unary route (E2E-REAL-6, x86 unary family).

Times packaging and first/warm launch of x86 softmax / reduction through the
retired Graph-owned constructor (``tests/_support/x86_unary_baseline.py``) and
the compiled Graph -> Schedule -> Tile route (``x86_native.package_*``) for a
sequence of distinct shapes, interleaved so both routes see the same host state.

The compiled route now memoizes an exact completed package; this probe records
both the first call for each shape and an immediate repeated call. A different
shape still runs the compiler. The runtime loads one copy of the
image per distinct ``image_digest`` (``runtime._load_x86_native_image``), and
``image_digest`` binds the Target IR digest, so the probe also counts distinct
digests / loaded images and times the first (loading) and a warm launch per
shape (runtime rows). Both are claims for the host that runs it only.

Run on an x86 host with the shared image built:
    PYTHONPATH=python:. python benchmarks/x86/measure_x86_unary_route_cost.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

from tessera import runtime as rt
from tessera.compiler import x86_native
from tests._support import x86_unary_baseline as baseline
from tests.unit.test_x86_unary_differential import _reduction, _softmax

SHAPES = [(3, 17), (4, 33), (2, 3, 5), (8, 64), (16, 128), (2, 7, 9), (32, 257), (5, 1)]


def _time(call) -> float:
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1e3


def _launch_ms(package, shape) -> tuple[float, float]:
    from tests.unit.test_x86_unary_differential import _args, _inputs

    x = _inputs(shape, 0)
    artifact = rt.RuntimeArtifact(metadata={"target": "x86"}, native_image=package.image,
                                  launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    times = []
    for _ in range(2):
        args, _ = _args(package, x)
        start = time.perf_counter()
        result = rt.launch(artifact, args)
        times.append((time.perf_counter() - start) * 1e3)
        if not result["ok"]:
            raise RuntimeError(result)
    return times[0], times[1]


def _rss_kib() -> int:
    try:
        with open(f"/proc/{os.getpid()}/status") as handle:
            return next(int(line.split()[1]) for line in handle if line.startswith("VmRSS:"))
    except (OSError, StopIteration):
        return -1


def main() -> int:
    if not x86_native.tools_available():
        print("x86 native image toolchain unavailable on this host", file=sys.stderr)
        return 2
    x86_native._UNARY_PACKAGE_CACHE.clear()
    x86_native._SCHEDULED_UNARY_PACKAGE_CACHE.clear()
    rows = []
    for family, build, old, new in (
        ("softmax", _softmax, baseline.package_softmax, x86_native.package_softmax),
        ("reduction", _reduction, baseline.package_reduction, x86_native.package_reduction),
    ):
        record: dict[str, dict[str, list]] = {
            route: {"package_ms": [], "repeat_package_ms": [],
                    "first_launch_ms": [], "warm_launch_ms": [], "digests": []}
            for route in ("retired", "compiled")
        }
        for shape in SHAPES:
            module = build(shape=shape)
            for route, packager in (("retired", old), ("compiled", new)):
                holder = []
                record[route]["package_ms"].append(_time(
                    lambda: holder.append(packager(module, pipeline_name="tessera-lower-to-x86"))))
                repeated = []
                record[route]["repeat_package_ms"].append(_time(
                    lambda: repeated.append(packager(module, pipeline_name="tessera-lower-to-x86"))))
                if repeated[0].image.image_digest != holder[0].image.image_digest:
                    raise RuntimeError("repeated package changed native image identity")
                first, warm = _launch_ms(holder[0], shape)
                record[route]["first_launch_ms"].append(first)
                record[route]["warm_launch_ms"].append(warm)
                record[route]["digests"].append(holder[0].image.image_digest)
        row: dict[str, object] = {"family": family, "shapes": len(SHAPES)}
        for route, values in record.items():
            row[route] = {
                "package_median_ms": round(statistics.median(values["package_ms"]), 2),
                "repeat_package_median_ms": round(statistics.median(values["repeat_package_ms"]), 2),
                "first_launch_median_ms": round(statistics.median(values["first_launch_ms"]), 3),
                "warm_launch_median_ms": round(statistics.median(values["warm_launch_ms"]), 3),
                "distinct_image_digests": len(set(values["digests"])),
                "package_ms": [round(v, 2) for v in values["package_ms"]],
                "repeat_package_ms": [round(v, 2) for v in values["repeat_package_ms"]],
                "first_launch_ms": [round(v, 3) for v in values["first_launch_ms"]],
            }
        rows.append(row)
    print(json.dumps({
        "host": platform.node(), "claims": ["compile_cost", "runtime_load"],
        "image_digests_seen": len(rt._x86_native_image_libraries),
        "loaded_x86_objects": len({lib._handle for lib in rt._x86_native_image_libraries.values()}),
        "vm_rss_kib": _rss_kib(),
        "payload_bytes": len(x86_native._library_path().read_bytes()),
        "rows": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
