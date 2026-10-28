"""Diagnostic: public bounded-row quantized MLP call vs public static calls at each M.

GFX1201-PUBLIC-BOUNDED-CALL-TIMING-20261028. One public ``@jit`` function is
compiled once with ``compile_native_quantized_program(..., m_bound=B)``; each
static arm is a separate public ``@jit`` function traced and compiled at that M.
A third arm is the same bounded program held in one prepared owner and driven by
an activation-only row update, invoke and readback -- what a public call would
cost if it retained its owner across calls instead of preparing one per call.

All three arms are host wall time per call, measured in windows of ``--calls``
calls, with the arm order rotated every round. Outputs are compared bitwise
before timing. Host wall on WSL2 is diagnostic only; it promotes nothing.
"""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time

import numpy as np
import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt

ARMS = ("public_bounded", "public_static", "prepared_bounded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 12288, 256), metavar=("H", "K", "N"))
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--actives", type=int, nargs="+", default=(1, 16, 64, 256))
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()
    h, k, n = args.shape
    bound = args.capacity
    if any(type(v) is not int or v <= 0 for v in (h, k, n, bound, args.rounds, args.calls, *args.actives)):
        parser.error("positive shapes and counts required")
    if any(m > bound for m in args.actives):
        parser.error("every active M must fit the capacity")
    if rt._rocm_live_arch() != "gfx1201":
        raise SystemExit("requires owning gfx1201")
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise SystemExit("native ROCm movement runtime unavailable")

    roots = {m: inputs(m, h, k, n, seed=500 + m) for m in args.actives}
    for m in args.actives:
        roots[m][1:] = roots[args.actives[0]][1:]  # one weight set; activations vary

    bounded_fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(args.actives[0], h, k, n))
    start = time.perf_counter()
    program = bounded_fn.compile_native_quantized_program(*roots[args.actives[0]], m_bound=bound)
    bounded_compile_s = time.perf_counter() - start
    if not program.bounded:
        raise SystemExit("explicit bound did not produce a bounded program")

    static_fn, static_cold_s = {}, {}
    for m in args.actives:
        fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(m, h, k, n))
        start = time.perf_counter()
        fn(*roots[m])
        static_cold_s[m] = time.perf_counter() - start
        static_fn[m] = fn

    owner = program.native.prepare(roots[args.actives[0]], runtime_library=lib._name)

    def prepared_call(m):
        owner.update_activations(roots[m][0])
        generation, _ = owner.invoke()
        return owner.read(generation)[0]

    calls = {"public_bounded": lambda m: bounded_fn(*roots[m]),
             "public_static": lambda m: static_fn[m](*roots[m]),
             "prepared_bounded": prepared_call}

    # Correctness precedes timing and is excluded from every window.
    equal = {}
    for m in args.actives:
        outputs = {arm: calls[arm](m) for arm in ARMS}
        if bounded_fn.execution_kind != "native_gpu" or bounded_fn._rocm_quantized_product_last_program is not program:
            raise SystemExit("public bounded call did not bind the bounded program")
        if static_fn[m].execution_kind != "native_gpu":
            raise SystemExit(f"public static call at M={m} did not execute natively")
        if outputs["public_bounded"].shape != (m, n):
            raise SystemExit(f"public bounded output shape {outputs['public_bounded'].shape} at M={m}")
        bits = {arm: out.view(np.uint16) for arm, out in outputs.items()}
        if not np.array_equal(bits["public_bounded"], bits["prepared_bounded"]):
            raise SystemExit(f"public and prepared bounded outputs differ at M={m}")
        # Different M schedules may differ in accumulation; record, do not assume.
        equal[m] = bool(np.array_equal(bits["public_bounded"], bits["public_static"]))

    samples = {m: {arm: [] for arm in ARMS} for m in args.actives}
    for round_ in range(args.rounds):
        order = list(args.actives) if round_ % 2 == 0 else list(args.actives)[::-1]
        arms = ARMS[round_ % 3:] + ARMS[:round_ % 3]
        if round_ % 2:
            arms = arms[::-1]
        for m in order:
            for arm in arms:
                call = calls[arm]
                start = time.perf_counter()
                for _ in range(args.calls):
                    call(m)
                samples[m][arm].append((time.perf_counter() - start) * 1000 / args.calls)
    allocation_bytes, allocation_count = owner.rows_info()[2:4]
    owner.close()

    rows = {}
    for m in args.actives:
        med = {arm: statistics.median(samples[m][arm]) for arm in ARMS}
        rows[str(m)] = {**{f"{arm}_ms": round(med[arm], 3) for arm in ARMS},
                        "public_bounded_over_static": round(med["public_bounded"] / med["public_static"], 3),
                        "prepared_over_public_bounded": round(med["prepared_bounded"] / med["public_bounded"], 3),
                        "bitwise_equal_static_at_m": equal[m],
                        "static_cold_first_call_s": round(static_cold_s[m], 2),
                        "samples_ms": {arm: [round(v, 3) for v in samples[m][arm]] for arm in ARMS}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    record = {
        "sync_key": "GFX1201-PUBLIC-BOUNDED-CALL-TIMING-20261028", "commit": commit, "worktree_dirty": dirty,
        "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest(),
        "device": "gfx1201 (RX 9070 XT, Tajasarus, WSL2)", "shape_hkn": [h, k, n], "capacity_m": bound,
        "bounded_compile_s": round(bounded_compile_s, 2),
        "bounded_allocation_bytes": allocation_bytes, "bounded_allocation_count": allocation_count,
        "timing": (f"host wall per call, windows of {args.calls} calls, {args.rounds} rounds with the arm "
                   "order rotated and reversed each round; median ms. public_bounded = public @jit call "
                   "through the bounded program (prepares its owner every call); public_static = public "
                   "@jit call compiled at that M; prepared_bounded = one held owner, activation-only update "
                   "+ invoke + readback. Diagnostic, not a promotion."),
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({m: {key: value for key, value in row.items() if key != "samples_ms"}
                      for m, row in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
