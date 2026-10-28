"""Diagnostic: K-bounded quantized MLP owner vs a static owner at each (M, K).

GFX1201-BOUNDED-QUANTIZED-K-20261028. One program is compiled with
``compile_native_quantized_program(..., m_bound=MB, k_bound=KB)`` and held in
one prepared owner; each active (M, K) takes that owner's full update (new
weights at that K). The control at each (M, K) is the static v4 program
compiled at exactly that shape, in its own owner.

Both arms are HIP-event device time per program invocation, in windows of
``--repeats`` invocations, interleaved and order-alternated every round.
Outputs are compared bitwise before timing. HIP events on WSL2 are not an
admitted kernel-side witness; this packet is diagnostic and promotes nothing.
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
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tests.unit.test_native_quantized_mlp_program import source


def _pairs(text):
    m, k = text.split("x")
    return int(m), int(k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hn", type=int, nargs=2, default=(128, 256), metavar=("H", "N"))
    parser.add_argument("--bounds", type=int, nargs=2, default=(256, 12288), metavar=("MB", "KB"))
    parser.add_argument("--actives", type=_pairs, nargs="+",
                        default=[(16, 4096), (16, 12288), (64, 8192), (256, 4096), (256, 12288)])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    h, n = args.hn
    mb, kb = args.bounds
    if any(m > mb or k > kb or k % 16 for m, k in args.actives):
        parser.error("every active (M, K) must fit the bounds with K a multiple of 16")
    if rt._rocm_live_arch() != "gfx1201":
        raise SystemExit("requires owning gfx1201")
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise SystemExit("native ROCm movement runtime unavailable")

    m0, k0 = args.actives[0]
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(m0, h, k0, n))
    start = time.perf_counter()
    program = fn.compile_native_quantized_program(*inputs(m0, h, k0, n), m_bound=mb, k_bound=kb)
    compile_s = time.perf_counter() - start
    if not program.native.k_bounded:
        raise SystemExit("explicit bounds did not produce a K-bounded program")
    roots = {mk: inputs(*mk[:1], h, mk[1], n, seed=1500 + i) for i, mk in enumerate(args.actives)}
    bounded = program.native.prepare(roots[args.actives[0]], runtime_library=lib._name)
    static, static_compile_s = {}, {}
    for mk in args.actives:
        start = time.perf_counter()
        package = package_quantized_product_graph(source(mk[0], h, mk[1], n))
        static_compile_s[mk] = time.perf_counter() - start
        static[mk] = package.prepare(roots[mk], runtime_library=lib._name)

    def run_bounded(mk, repeats):
        bounded.update(roots[mk])
        generation, elapsed = bounded.invoke(repeats=repeats, timed=True)
        return generation, elapsed

    equal = {}
    for mk in args.actives:
        generation, _ = run_bounded(mk, 1)
        got = bounded.read(generation)[0]
        generation, _ = static[mk].invoke()
        want = static[mk].read(generation)[0]
        equal[mk] = bool(np.array_equal(got.view(np.uint16), want.view(np.uint16)))
        if not equal[mk]:
            raise SystemExit(f"bounded and static outputs differ at (M, K) = {mk}")

    samples = {mk: {"bounded": [], "static": []} for mk in args.actives}
    for round_ in range(args.rounds):
        order = list(args.actives) if round_ % 2 == 0 else list(args.actives)[::-1]
        for mk in order:
            arms = ("bounded", "static") if round_ % 2 == 0 else ("static", "bounded")
            for arm in arms:
                if arm == "bounded":
                    # The update precedes the timed window; only invocations are timed.
                    _, elapsed = run_bounded(mk, args.repeats)
                else:
                    _, elapsed = static[mk].invoke(repeats=args.repeats, timed=True)
                samples[mk][arm].append(elapsed)
    info = bounded.shape_info()
    bounded.close()
    for owner in static.values():
        owner.close()

    rows = {}
    for mk in args.actives:
        b = statistics.median(samples[mk]["bounded"])
        s = statistics.median(samples[mk]["static"])
        rows[f"{mk[0]}x{mk[1]}"] = {
            "bounded_ms": round(b, 4), "static_ms": round(s, 4), "bounded_over_static": round(b / s, 3),
            "bounded_range_ms": [round(min(samples[mk]["bounded"]), 4), round(max(samples[mk]["bounded"]), 4)],
            "static_range_ms": [round(min(samples[mk]["static"]), 4), round(max(samples[mk]["static"]), 4)],
            "bitwise_equal": equal[mk], "static_compile_s": round(static_compile_s[mk], 2),
            "samples_ms": {arm: [round(v, 4) for v in values] for arm, values in samples[mk].items()}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    record = {
        "sync_key": "GFX1201-BOUNDED-QUANTIZED-K-20261028", "commit": commit, "worktree_dirty": dirty,
        "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest(),
        "device": "gfx1201 (RX 9070 XT, Tajasarus, WSL2)", "hn": [h, n], "bounds_mk": [mb, kb],
        "bounded_compile_s": round(compile_s, 2),
        "bounded_allocation_bytes": info[4], "bounded_allocation_count": info[5],
        "timing": (f"HIP-event device ms per program invocation, windows of {args.repeats} invocations, "
                   f"{args.rounds} rounds, arm and (M, K) order alternated each round; median. bounded = "
                   "one K-bounded owner updated to each (M, K); static = the v4 program compiled at that "
                   "(M, K). Diagnostic, not a promotion."),
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: {k: v for k, v in row.items() if k != "samples_ms"} for key, row in rows.items()},
                     indent=1))


if __name__ == "__main__":
    main()
