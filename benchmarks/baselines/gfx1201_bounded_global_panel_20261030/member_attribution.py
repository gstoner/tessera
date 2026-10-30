"""Diagnostic: per-member device windows, H-bounded owner vs static owners.

GFX1201-BOUNDED-GLOBAL-PANEL-20261030. One v8 program (m_bound=256,
h_bound=2048 at K/N 4096/256) in one prepared owner, updated to each active
(M, H); the control is the static v4 program at that shape. Each member is
timed alone (HIP events, windows of 200 invocations after one full
invocation), 7 rounds with the arm order alternating. Writes members.json.
Diagnostic only; promotes nothing.
"""
import json
import pathlib
import statistics

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source

K, N = 4096, 256
SHAPES = [(16, 2048), (256, 2048)]
MEMBERS = ["gate quantizer", "gate/up expansion", "gate product", "SiLU", "down quantizer",
           "down expansion", "down product"]


def main():
    lib = rt._load_rocm_native_movement_runtime()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(16, 512, K, N))
    native = fn.compile_native_quantized_program(*inputs(16, 512, K, N), m_bound=256, h_bound=2048).native
    roots = {s: inputs(s[0], s[1], K, N, seed=7) for s in SHAPES}
    bounded = native.prepare(roots[SHAPES[0]], runtime_library=lib._name)
    record = {"members": MEMBERS, "unit": "median us per member invocation",
              "gate_product_macro_tile": native.native_stages[2].descriptor.provenance["macro_tile"],
              "rows": {}}
    for shape in SHAPES:
        static = package_quantized_product_graph(source(shape[0], shape[1], K, N)).prepare(
            roots[shape], runtime_library=lib._name)
        bounded.update(roots[shape])
        bounded.invoke()
        static.invoke()
        samples = {"bounded": [[] for _ in MEMBERS], "static": [[] for _ in MEMBERS]}
        for round_ in range(7):
            arms = (("bounded", bounded), ("static", static))
            for arm, owner in arms if round_ % 2 == 0 else arms[::-1]:
                for i in range(len(MEMBERS)):
                    _, ms = owner.invoke_range(i, 1, 200, timed=True)
                    samples[arm][i].append(1000 * ms)
        record["rows"]["x".join(map(str, shape))] = {
            "static_gate_product_macro_tile": static.package.native_stages[2].descriptor.provenance["macro_tile"],
            **{arm: [round(statistics.median(v), 2) for v in values] for arm, values in samples.items()}}
        static.close()
        print(shape, json.dumps(record["rows"]["x".join(map(str, shape))]), flush=True)
    bounded.close()
    pathlib.Path(__file__).with_name("members.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
