"""Exact gfx1151 bounded paged-read package/launch host timing.

Run from the repository root in Princess-Luna WSL with a built tessera-opt.
These are host wall times, including Python and HIP runtime work; they are
not GPU kernel timings.
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_native import package_paged_kv_read
from tests.unit.test_rocm_e2e_spine import _paged_kv_module


def median_ms(call, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    return statistics.median(samples)


def main() -> None:
    rng = np.random.default_rng(2203)
    pages = np.ascontiguousarray(rng.standard_normal((4, 4, 3, 8)), dtype=np.float32)
    table = np.array([2, 0, 3, 1], dtype=np.int32)
    logical = pages[table].reshape(16, 3, 8)
    rows = []
    for start, end in ((0, 1), (3, 10), (0, 16)):
        module = _paged_kv_module(start=start, end=end)
        package = package_paged_kv_read(module, pipeline_name="tessera-lower-to-rocm")
        output = np.empty((end - start, 3, 8), dtype=np.float32)
        artifact = rt.RuntimeArtifact(
            graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
            metadata={"target": "rocm_gfx1151"}, native_image=package.image,
            launch_descriptor=package.descriptor,
        )
        bindings = {
            "pages": pages, "page_table": table, "slice": output,
            "P": 4, "LP": 4, "PageSize": 4, "H": 3, "D": 8,
            "Start": start, "Tokens": end - start,
        }

        def launch() -> None:
            result = rt.launch(artifact, bindings)
            if result.get("ok") is not True:
                raise RuntimeError(result.get("reason", "launch failed"))
            np.testing.assert_array_equal(output, logical[start:end])

        launch()
        rows.append({
            "start": start, "end": end, "image_digest": package.image.image_digest,
            "package_median_ms": median_ms(
                lambda: package_paged_kv_read(module, pipeline_name="tessera-lower-to-rocm"), 7
            ),
            "launch_median_ms": median_ms(launch, 21),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
