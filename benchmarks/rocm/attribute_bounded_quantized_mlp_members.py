"""Diagnostic: per-member device windows, bounded owner vs static owners.

GFX1151-QUANTIZED-MLP-TIMING-20261101. Generalizes the gfx1201 H packet's
packet-local ``member_attribution.py``
(``benchmarks/baselines/gfx1201_h_bounded_quantized_mlp_20261029/``) to the
M-, K-, N- and H-bounded programs and to the live owning architecture
(``quantized_mlp_arch``).

One bounded program in one prepared owner is updated to each active shape; the
control is the static v4 program at that shape. Each member is timed alone
(HIP events, windows of ``--repeats`` invocations after one full invocation),
``--rounds`` rounds with the arm order alternating. A kept, never-timed warm-up
owner is prepared first. Diagnostic only; promotes nothing.
"""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

import numpy as np
import tessera as ts
from benchmarks.rocm import quantized_mlp_arch as owning
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

MEMBERS = ["gate quantizer", "gate/up expansion", "gate product", "SiLU", "down quantizer",
           "down expansion", "down product"]


def _shape(text):
    values = tuple(int(v) for v in text.split("x"))
    if len(values) != 4 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("shapes are MxHxKxN")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bounds", required=True,
                        help="comma list of bounds, e.g. m=256,k=12288 or m=256,h=512")
    parser.add_argument("--shapes", type=_shape, nargs="+", required=True, metavar="MxHxKxN")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=200)
    owning.add_arch_arguments(parser, None)
    args = parser.parse_args()
    if not args.sync_key:
        parser.error("--sync-key is required")
    bounds = {}
    for item in args.bounds.split(","):
        name, _, value = item.partition("=")
        if name not in ("m", "k", "n", "h") or not value.isdigit():
            parser.error(f"bad bound {item}")
        bounds[f"{name}_bound"] = int(value)
    if "m_bound" not in bounds:
        parser.error("an m bound is required")
    arch = owning.owning_arch()
    owning.check_sync_key(args.sync_key, arch)
    lock = owning.device_lock(args.device_lock)
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise SystemExit("native ROCm movement runtime unavailable")

    first = args.shapes[0]
    roots = {s: inputs(*s, seed=7 + i) for i, s in enumerate(args.shapes)}
    warm = package_quantized_product_graph(owning.source(*first, arch)).prepare(
        roots[first], runtime_library=lib._name)
    warm.invoke()
    fn = ts.jit(target=owning.target(arch), native_required=True)(make_mlp(*first))
    native = fn.compile_native_quantized_program(*roots[first], **bounds).native
    bounded = native.prepare(roots[first], runtime_library=lib._name)
    record = {"sync_key": args.sync_key, "architecture": arch, "device": owning.device(arch),
              "device_lock": lock, "warm_up_owner": True, "bounds": bounds, "members": MEMBERS,
              "route": owning.ROUTE, "latency_source": "hip_event_member_range_window",
              "timing_source": "hip_event_member_range_window",
              "unit": "median us per member invocation",
              "timing": (f"each member alone through invoke_range, HIP-event windows of {args.repeats} "
                         f"invocations, {args.rounds} rounds with arm order alternating; median. "
                         "Diagnostic, not a promotion."),
              "gate_product_macro_tile": native.native_stages[2].descriptor.provenance.get("macro_tile"),
              "rows": {}}
    for shape in args.shapes:
        static = package_quantized_product_graph(owning.source(*shape, arch)).prepare(
            roots[shape], runtime_library=lib._name)
        bounded.update(roots[shape])
        generation, _ = bounded.invoke()
        got = bounded.read(generation)[0]
        generation, _ = static.invoke()
        want = static.read(generation)[0]
        equal = bool(np.array_equal(got.view(np.uint16), want.view(np.uint16)))
        if not equal:
            raise SystemExit(f"bounded and static outputs differ at {shape}")
        samples = {"bounded": [[] for _ in MEMBERS], "static": [[] for _ in MEMBERS]}
        for round_ in range(args.rounds):
            arms = (("bounded", bounded), ("static", static))
            for arm, owner in arms if round_ % 2 == 0 else arms[::-1]:
                for i in range(len(MEMBERS)):
                    _, ms = owner.invoke_range(i, 1, args.repeats, timed=True)
                    samples[arm][i].append(1000 * ms)
        record["rows"]["x".join(map(str, shape))] = {
            "bitwise_equal": equal,
            "static_gate_product_macro_tile":
                static.package.native_stages[2].descriptor.provenance.get("macro_tile"),
            **{arm: [round(statistics.median(v), 2) for v in values] for arm, values in samples.items()}}
        static.close()
        print(shape, json.dumps(record["rows"]["x".join(map(str, shape))]), flush=True)
    bounded.close()
    warm.close()
    record.update({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
