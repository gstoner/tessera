"""Diagnostic: the capacity-selected gate-product tile explains the small-M cost.

GFX1201-BOUNDED-QUANTIZED-H-20261029. Compiles the v8 program at h_bound=512
with row capacity 16 and 256 and times only the gate product (HIP events,
200-invocation windows, 7 alternating rounds) against the static program at
M = 16. Writes tile_check.json. Diagnostic only; promotes nothing.
"""
import json
import pathlib
import statistics

import numpy as np
import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source

K, N = 4096, 256


def main():
    lib = rt._load_rocm_native_movement_runtime()
    rows = []
    for m_bound in (16, 256):
        fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(16, 128, K, N))
        native = fn.compile_native_quantized_program(*inputs(16, 128, K, N), m_bound=m_bound, h_bound=512).native
        tile = native.native_stages[2].descriptor.provenance["macro_tile"]
        for h in (128, 512):
            roots = inputs(16, h, K, N, seed=7)
            bounded = native.prepare(roots, runtime_library=lib._name)
            static = package_quantized_product_graph(source(16, h, K, N)).prepare(roots, runtime_library=lib._name)
            bounded.invoke()
            static.invoke()
            times = {"bounded": [], "static": []}
            for round_ in range(7):
                arms = (("bounded", bounded), ("static", static))
                for arm, owner in arms if round_ % 2 == 0 else arms[::-1]:
                    times[arm].append(1000 * owner.invoke_range(2, 1, 200, timed=True)[1])
            g, _ = bounded.invoke()
            x = bounded.read(g)[0]
            g, _ = static.invoke()
            y = static.read(g)[0]
            b, s = statistics.median(times["bounded"]), statistics.median(times["static"])
            rows.append({"m_bound": m_bound, "bounded_tile": tile,
                         "static_tile": static.package.native_stages[2].descriptor.provenance["macro_tile"],
                         "active": [16, h], "bounded_us": round(b, 2), "static_us": round(s, 2),
                         "ratio": round(b / s, 2),
                         "bitwise_equal": bool(np.array_equal(x.view(np.uint16), y.view(np.uint16)))})
            print(json.dumps(rows[-1]), flush=True)
            bounded.close()
            static.close()
    pathlib.Path(__file__).with_name("tile_check.json").write_text(json.dumps({"unit": "median us per gate-product invocation", "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
