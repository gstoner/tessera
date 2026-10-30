"""Diagnostic: per-active-M product images vs static programs on gfx1201.

GFX1201-BOUNDED-PER-M-PANEL-20261030. One M-only bounded program
(m_bound=256 at H/K/N 2048/4096/256) in one prepared owner, updated to each
active M; the control is the static v4 program at that M. Both the whole
program (50-invocation windows) and the gate product alone (100-invocation
windows) are timed with HIP events, 7 rounds with the arm order alternating.
Records the gate image the owner selected. Writes per_m.json. Diagnostic
only; promotes nothing.
"""
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

import numpy as np
import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source

H, K, N = 2048, 4096, 256
ACTIVE = (16, 64, 100, 128, 200, 256)


def main():
    lib = rt._load_rocm_native_movement_runtime()
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(256, H, K, N))
    native = fn.compile_native_quantized_program(*inputs(256, H, K, N), m_bound=256).native
    # On gfx1201 the first owner a process prepares ran ~17% slower for its
    # whole life at large shapes (GFX1201-BOUNDED-PER-M-PANEL-20261030); a kept,
    # never-timed warm-up owner takes that slot so neither arm is penalized.
    warm = package_quantized_product_graph(source(16, H, K, N)).prepare(
        inputs(16, H, K, N), runtime_library=lib._name)
    warm.invoke()
    bounded = native.prepare(inputs(256, H, K, N), runtime_library=lib._name)
    rows = {}
    for m in ACTIVE:
        roots = inputs(m, H, K, N, seed=7)
        static = package_quantized_product_graph(source(m, H, K, N))
        control = static.prepare(roots, runtime_library=lib._name)
        bounded.update(roots)
        bounded.invoke()
        control.invoke()
        whole = {"bounded": [], "static": []}
        gate = {"bounded": [], "static": []}
        for round_ in range(7):
            arms = (("bounded", bounded), ("static", control))
            for arm, owner in arms if round_ % 2 == 0 else arms[::-1]:
                whole[arm].append(owner.invoke(repeats=50, timed=True)[1])
                gate[arm].append(1000 * owner.invoke_range(2, 1, 100, timed=True)[1])
        x = bounded.read(bounded.invoke()[0])[0]
        y = control.read(control.invoke()[0])[0]
        med = {k: {a: statistics.median(v) for a, v in d.items()} for k, d in (("whole", whole), ("gate", gate))}
        rows[str(m)] = {
            "gate_image": bounded.variant_info()[2],
            "static_gate_tile": static.native_stages[2].descriptor.provenance["macro_tile"],
            "whole_ms": {a: round(v, 4) for a, v in med["whole"].items()},
            "whole_ratio": round(med["whole"]["bounded"] / med["whole"]["static"], 3),
            "gate_us": {a: round(v, 1) for a, v in med["gate"].items()},
            "gate_ratio": round(med["gate"]["bounded"] / med["gate"]["static"], 3),
            "bitwise_equal": bool(np.array_equal(x.view(np.uint16), y.view(np.uint16)))}
        control.close()
        print(m, json.dumps(rows[str(m)]), flush=True)
    bounded.close()
    warm.close()
    record = {
        "sync_key": "GFX1201-BOUNDED-PER-M-PANEL-20261030",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest(),
        "device": "gfx1201 (RX 9070 XT, Tajasarus, WSL2)", "hkn": [H, K, N], "m_bound": 256,
        "warm_up_owner": True,
        "gate_variants": [[v.min_m, v.max_m, v.native.descriptor.provenance["macro_tile"]] for v in native.gate.variants],
        "rows": rows}
    pathlib.Path(__file__).with_name("per_m.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
