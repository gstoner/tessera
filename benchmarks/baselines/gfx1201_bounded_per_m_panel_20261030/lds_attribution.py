"""Diagnostic: which projection flag costs the shape-free LDS gate image (GFX1201-BOUNDED-PER-M-PANEL-20261030).

Swaps only the gate-product image of the static MLP at (M, H) for images built from
the same recipe with different projection flags, behind a kept warm-up owner, and
times that member alone (HIP events, 100-invocation windows, 7 alternating rounds).
Usage: lds_attribution.py M H. Diagnostic only; promotes nothing.
"""
import dataclasses, json, statistics, sys
import numpy as np
from tessera import runtime as rt
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs
import tessera.compiler.rocm_quantized_product_program as qp
import tessera.compiler.rocm_quantized_mlp_program as mp
from tessera.compiler.rocm_fp8_blockscale import package_blockscale
from tests.unit.test_native_quantized_mlp_program import source
M, H, K, N = int(sys.argv[1]), int(sys.argv[2]), 4096, 256
lib = rt._load_rocm_native_movement_runtime()
static = qp.package_quantized_product_graph(source(M, H, K, N))
recipe = static.gate.product_recipe
arms = {
    "A static checked": dict(project_image_identity=False),
    "B runtime_mn whole": dict(project_image_identity=True, lds_runtime_k=False),
    "C +runtime_k": dict(project_image_identity=True, lds_runtime_k=True),
    "D +arbitrary_m": dict(project_image_identity=True, lds_runtime_k=True, arbitrary_m=True),
    "E +arbitrary_n": dict(project_image_identity=True, lds_runtime_k=True, arbitrary_m=True, arbitrary_n=True),
    "F arbitrary_m only": dict(project_image_identity=True, lds_runtime_k=False, arbitrary_m=True),
    "A2 static checked, prepared last": dict(project_image_identity=False),
}
# Validation re-derives the static image; this experiment swaps it on purpose.
qp.NativeQuantizedProductProgram.validate = lambda self: None
mp.NativeQuantizedMLPProgram.validate = lambda self: None
roots = inputs(M, H, K, N, seed=7)
# A sacrificial first owner, kept alive and never timed.
warm = static.prepare(roots, runtime_library=lib._name); warm.invoke()
owners, ref = {}, None
for name, flags in arms.items():
    product = package_blockscale(recipe, pipeline_name="tessera-rocm-executable", **flags)
    gate = dataclasses.replace(static.gate, product=product)
    program = dataclasses.replace(static, gate=gate)
    owners[name] = program.prepare(roots, runtime_library=lib._name)
    owners[name].invoke()
    out = owners[name].read(owners[name].invoke()[0])[0]
    if ref is None: ref = out
    assert np.array_equal(out.view(np.uint16), ref.view(np.uint16)), name
t = {n: [] for n in owners}
names = list(owners)
for r in range(7):
    for n in (names if r % 2 == 0 else names[::-1]):
        t[n].append(1000 * owners[n].invoke_range(2, 1, 100, timed=True)[1])
base = statistics.median(t[names[0]])
for n in names:
    print(f"{M}x{H} {n}: {statistics.median(t[n]):.1f} us ({statistics.median(t[n]) / base:.3f}x)", flush=True)
