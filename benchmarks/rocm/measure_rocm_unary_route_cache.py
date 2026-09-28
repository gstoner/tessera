"""E2E-REAL-6 (ROCm unary family): compile-cache granularity, retired vs compiled route.

The retired Graph-owned constructors authored shape-free Tile text, so one
HSACO served every shape of a (storage, kind, axis) key. The compiled route's
Tile IR is the replay of a Schedule record whose digest binds the static shape,
so until FOUNDATION-BATCH-2-2026-09-27 each new shape was a cold compile. The
route now keys the image on its shape-free kernel identity
(``rocm_native._compile_shape_free_tile_ir``), so a new shape -- and a new
Graph symbol, the last row of each compiled route -- is a warm hit on one
image. This script measures the per-shape cost on the owning host and prints
each row's image digest and entry symbol (it prints, it does not promote
anything). Run on Princess-Luna (or Tajasarus with TESSERA_ROCM_CHIP=gfx1201
semantics -- the retired baseline is gfx1151-only) with the ROCm env sourced.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler import rocm_native  # noqa: E402
from tests._support import rocm_unary_baseline as baseline  # noqa: E402
from tests.unit.test_rocm_unary_migration import _reduction, _softmax  # noqa: E402

PIPELINE = "tessera-lower-to-rocm"


def _run(label, packager, modules):
    rocm_native._cache.clear()
    rocm_native._shape_free_targets.clear()
    rows = []
    for module in modules:
        start = time.perf_counter()
        package = packager(module, pipeline_name=PIPELINE)
        rows.append({
            "shape": [g.value for g in package.descriptor.shape_guards
                      if g.binding == package.descriptor.buffers[0].name],
            "compile_state": package.image.compile_state,
            "ms": round((time.perf_counter() - start) * 1000.0, 1),
            "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest()[:16],
            "image_digest": package.image.image_digest[:16],
            "entry": package.descriptor.entry_symbol,
            "graph_symbol": module.functions[0].name,
        })
    return {"route": label, "rows": rows}


def main() -> None:
    shapes = [(3, 17), (4, 256), (2, 257), (8, 64)]
    softmax = [_softmax("fp16", shape) for shape in shapes]
    reduce_ = [_reduction("bf16", "mean", (2,) + shape, 1) for shape in shapes]
    renamed_softmax, renamed_reduce = _softmax("fp16", (6, 33)), _reduction("bf16", "mean", (2, 6, 33), 1)
    renamed_softmax.functions[0].name = renamed_reduce.functions[0].name = "another_graph_symbol"
    report = [
        _run("retired_softmax_f16", baseline.baseline_softmax, softmax),
        _run("compiled_softmax_f16", rocm_native.package_softmax, softmax + [renamed_softmax]),
        _run("retired_reduce_bf16", baseline.baseline_reduction, reduce_),
        _run("compiled_reduce_bf16", rocm_native.package_reduction, reduce_ + [renamed_reduce]),
    ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
