"""Diagnostic: K/N/H-bounded quantized MLP owner vs a static owner at each shape.

GFX1201-BOUNDED-QUANTIZED-K-20261028 / -N-20261029 / -H-20261029.
One program is compiled with ``compile_native_quantized_program(...,
m_bound=MB, k_bound=KB[, n_bound=NB])`` and held in one prepared owner; each
active (M, K[, N]) takes that owner's full update (new weights at that shape).
The control at each shape is the static v4 program compiled at exactly that
shape, in its own owner. ``--n-bound`` selects the v7 program; actives are
then ``MxKxN``. ``--h-bound HB`` selects the v8 program instead
(``m_bound=MB, h_bound=HB`` with ``--bounds MB HB``); ``--kn`` fixes K and N
and actives are ``MxH``.

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
    return tuple(int(v) for v in text.split("x"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hn", type=int, nargs=2, default=(128, 256), metavar=("H", "N"))
    parser.add_argument("--bounds", type=int, nargs=2, default=(256, 12288), metavar=("MB", "KB"))
    parser.add_argument("--actives", type=_pairs, nargs="+",
                        default=[(16, 4096), (16, 12288), (64, 8192), (256, 4096), (256, 12288)])
    parser.add_argument("--n-bound", type=int, default=None)
    parser.add_argument("--h-bound", action="store_true",
                        help="bound H instead of K: --bounds is (MB, HB), actives MxH at --kn")
    parser.add_argument("--kn", type=int, nargs=2, default=(4096, 256), metavar=("K", "N"))
    parser.add_argument("--sync-key", default="GFX1201-BOUNDED-QUANTIZED-K-20261028")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    h, n = args.hn
    mb, kb = args.bounds
    nb = args.n_bound
    hidden = args.h_bound
    if hidden and nb is not None:
        parser.error("--h-bound does not compose with --n-bound")
    width = 2 if nb is None else 3
    if any(len(a) != width for a in args.actives):
        parser.error(f"actives are {'MxH' if hidden else 'MxK' if nb is None else 'MxKxN'}")
    if any(a[0] > mb or a[1] > kb or a[1] % 16 or (nb is not None and a[2] > nb) for a in args.actives):
        parser.error("every active shape must fit the bounds with K (or H) a multiple of 16")
    # Every shape is (M, H, K, N).
    if hidden:
        shapes = [(a[0], a[1], *args.kn) for a in args.actives]
    else:
        shapes = [(a[0], h, a[1], a[2] if nb is not None else n) for a in args.actives]
    if rt._rocm_live_arch() != "gfx1201":
        raise SystemExit("requires owning gfx1201")
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise SystemExit("native ROCm movement runtime unavailable")

    m0, h0, k0, n0 = shapes[0]
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(m0, h0, k0, n0))
    if hidden:
        bounds = {"m_bound": mb, "h_bound": kb}
    else:
        bounds = {"m_bound": mb, "k_bound": kb, **({} if nb is None else {"n_bound": nb})}
    start = time.perf_counter()
    program = fn.compile_native_quantized_program(*inputs(m0, h0, k0, n0), **bounds)
    compile_s = time.perf_counter() - start
    if (program.native.h_bounded != hidden or program.native.k_bounded == hidden
            or program.native.n_bounded != (nb is not None)):
        raise SystemExit("explicit bounds did not produce the requested bounded program")
    roots = {mk: inputs(*mk, seed=1500 + i) for i, mk in enumerate(shapes)}
    bounded = program.native.prepare(roots[shapes[0]], runtime_library=lib._name)
    static, static_compile_s = {}, {}
    for mk in shapes:
        start = time.perf_counter()
        package = package_quantized_product_graph(source(*mk))
        static_compile_s[mk] = time.perf_counter() - start
        static[mk] = package.prepare(roots[mk], runtime_library=lib._name)

    def run_bounded(mk, repeats):
        bounded.update(roots[mk])
        generation, elapsed = bounded.invoke(repeats=repeats, timed=True)
        return generation, elapsed

    equal = {}
    for mk in shapes:
        generation, _ = run_bounded(mk, 1)
        got = bounded.read(generation)[0]
        generation, _ = static[mk].invoke()
        want = static[mk].read(generation)[0]
        equal[mk] = bool(np.array_equal(got.view(np.uint16), want.view(np.uint16)))
        if not equal[mk]:
            raise SystemExit(f"bounded and static outputs differ at (M, H, K, N) = {mk}")

    samples = {mk: {"bounded": [], "static": []} for mk in shapes}
    for round_ in range(args.rounds):
        order = list(shapes) if round_ % 2 == 0 else list(shapes)[::-1]
        for mk in order:
            arms = ("bounded", "static") if round_ % 2 == 0 else ("static", "bounded")
            for arm in arms:
                if arm == "bounded":
                    # The update precedes the timed window; only invocations are timed.
                    _, elapsed = run_bounded(mk, args.repeats)
                else:
                    _, elapsed = static[mk].invoke(repeats=args.repeats, timed=True)
                samples[mk][arm].append(elapsed)
    info = bounded.axes_info()[2:] if nb is not None or hidden else bounded.shape_info()[4:]
    bounded.close()
    for owner in static.values():
        owner.close()

    rows = {}
    for mk in shapes:
        b = statistics.median(samples[mk]["bounded"])
        s = statistics.median(samples[mk]["static"])
        key = (mk[0], mk[1]) if hidden else (mk[0], mk[2], mk[3]) if nb is not None else (mk[0], mk[2])
        rows["x".join(str(v) for v in key)] = {
            "bounded_ms": round(b, 4), "static_ms": round(s, 4), "bounded_over_static": round(b / s, 3),
            "bounded_range_ms": [round(min(samples[mk]["bounded"]), 4), round(max(samples[mk]["bounded"]), 4)],
            "static_range_ms": [round(min(samples[mk]["static"]), 4), round(max(samples[mk]["static"]), 4)],
            "bitwise_equal": equal[mk], "static_compile_s": round(static_compile_s[mk], 2),
            "samples_ms": {arm: [round(v, 4) for v in values] for arm, values in samples[mk].items()}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    record = {
        "sync_key": args.sync_key, "commit": commit, "worktree_dirty": dirty,
        "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest(),
        "device": "gfx1201 (RX 9070 XT, Tajasarus, WSL2)",
        **({"kn": list(args.kn), "bounds_mh": [mb, kb]} if hidden else {"hn": [h, n]}),
        **({} if hidden else {"bounds_mk": [mb, kb]} if nb is None else {"bounds_mkn": [mb, kb, nb]}),
        "bounded_compile_s": round(compile_s, 2),
        "bounded_allocation_bytes": info[0], "bounded_allocation_count": info[1],
        "timing": (f"HIP-event device ms per program invocation, windows of {args.repeats} invocations, "
                   f"{args.rounds} rounds, arm and (M, K) order alternated each round; median. bounded = "
                   "one bounded owner updated to each shape; static = the v4 program compiled at that "
                   "shape. Diagnostic, not a promotion."),
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: {k: v for k, v in row.items() if k != "samples_ms"} for key, row in rows.items()},
                     indent=1))


if __name__ == "__main__":
    main()
