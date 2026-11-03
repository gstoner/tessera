"""Diagnostic probes of the gfx1151 bounded quantized MLP (2026-11-02).

GFX1151-QUANTIZED-MLP-PROBES-20261102. Follows the timing packets of
GFX1151-QUANTIZED-MLP-TIMING-20261101 and probes their three open findings:

``panel``  The H-bounded owner (m_bound=256, h_bound=512, K/N 4096/256) against
           the static program, timing the gate product alone and the whole
           program. ``--force-min-m 256`` compiles the bounded image with its
           register panel chosen at capacity (the production rule chooses it at
           M = 1). The override is a packet-local monkeypatch of the module key
           ``tessera.rocm.blockscale_min_m`` for this process only; it changes
           no production code, and the recorder refuses unless the bounded
           gate product actually took the expected panel.
``rows``   The M-bounded owner (m_bound=256, H/K/N 128/12288/256) against the
           static program at each M, every member timed alone, every window
           kept, so the M = 1 and M = 256 medians get ranges and attribution.
``isa``    Writes the gate-product HSACO of the bounded and static images at
           the same-panel shapes, for disassembly on the box.

Every timed mode prepares a never-timed warm-up owner first, compares outputs
bitwise before timing, alternates arm order each round and checks the shared
device lock. Diagnostic only; promotes nothing.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

import numpy as np
import tessera as ts
from benchmarks.rocm import quantized_mlp_arch as owning
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
from tessera import runtime as rt
from tessera.compiler import rocm_quantized_product_program as qpp
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

MEMBERS = ["gate quantizer", "gate/up expansion", "gate product", "SiLU", "down quantizer",
           "down expansion", "down product"]
HERE = pathlib.Path(__file__).resolve().parent


def force_min_m(value):
    """Choose every arbitrary-M product panel at ``value`` rows in this process."""
    admitted, verify = qpp.admitted_rows_graph, qpp.verify_blockscale_schedule

    def admitted_rows_graph(graph_ir, min_m=1, **kwargs):
        return admitted(graph_ir, min_m=value if min_m == 1 else min_m, **kwargs)

    def verify_blockscale_schedule(shape, schedule_ir, *, min_m=None, **kwargs):
        return verify(shape, schedule_ir, min_m=value if min_m == 1 else min_m, **kwargs)

    qpp.admitted_rows_graph = admitted_rows_graph
    qpp.verify_blockscale_schedule = verify_blockscale_schedule


def tile(owner_or_package):
    package = getattr(owner_or_package, "package", owner_or_package)
    return package.native_stages[2].descriptor.provenance.get("macro_tile")


def bitwise(a, b):
    ga, _ = a.invoke()
    gb, _ = b.invoke()
    return bool(np.array_equal(a.read(ga)[0].view(np.uint16), b.read(gb)[0].view(np.uint16)))


def summary(values):
    ordered = sorted(values)
    return {"median": round(ordered[len(ordered) // 2], 2), "min": round(ordered[0], 2),
            "max": round(ordered[-1], 2), "samples": [round(v, 3) for v in values]}


def window(owner, members, repeats):
    """Median-ready µs per invocation: one member alone, or the whole program."""
    if members == "program":
        return 1000 * owner.invoke(repeats=repeats, timed=True)[1]
    return 1000 * owner.invoke_range(members, 1, repeats, timed=True)[1]


def compare(bounded, static, scopes, rounds, repeats):
    samples = {arm: {str(s): [] for s in scopes} for arm in ("bounded", "static")}
    for round_ in range(rounds):
        arms = (("bounded", bounded), ("static", static))
        for arm, owner in arms if round_ % 2 == 0 else arms[::-1]:
            for scope in scopes:
                samples[arm][str(scope)].append(window(owner, scope, repeats))
    return {arm: {scope: summary(v) for scope, v in values.items()} for arm, values in samples.items()}


def provenance(lib):
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(pathlib.Path(lib._name).read_bytes()).hexdigest()}


def compile_bounded(arch, shape, roots, **bounds):
    fn = ts.jit(target=owning.target(arch), native_required=True)(make_mlp(*shape))
    return fn.compile_native_quantized_program(*roots, **bounds).native


def run_panel(args, arch, lib, record):
    k, n = 4096, 256
    actives = [(16, 512), (64, 512), (256, 512), (256, 256), (256, 128)]
    if args.force_min_m:
        force_min_m(args.force_min_m)
    first = (16, 512, k, n)
    roots = {a: inputs(a[0], a[1], k, n, seed=7 + i) for i, a in enumerate(actives)}
    warm = package_quantized_product_graph(owning.source(*first, arch)).prepare(
        roots[actives[0]], runtime_library=lib._name)
    warm.invoke()
    native = compile_bounded(arch, first, roots[actives[0]], m_bound=256, h_bound=512)
    record["bounded_gate_tile"] = tile(native)
    expected = [32, 32] if args.force_min_m == 256 else [16, 32]
    if record["bounded_gate_tile"] != expected:
        raise SystemExit(f"bounded gate product took {record['bounded_gate_tile']}, expected {expected}")
    bounded = native.prepare(roots[actives[0]], runtime_library=lib._name)
    for m, h in actives:
        static = package_quantized_product_graph(owning.source(m, h, k, n, arch)).prepare(
            roots[(m, h)], runtime_library=lib._name)
        bounded.update(roots[(m, h)])
        equal = bitwise(bounded, static)
        if not equal:
            raise SystemExit(f"bounded and static outputs differ at {m}x{h}")
        row = {"static_gate_tile": tile(static), "bitwise_equal": equal,
               **compare(bounded, static, (2, "program"), args.rounds, args.repeats)}
        record["rows"][f"{m}x{h}x{k}x{n}"] = row
        print(m, h, row["static_gate_tile"], {a: {s: row[a][s]["median"] for s in row[a]}
                                              for a in ("bounded", "static")}, flush=True)
        static.close()
    bounded.close()
    warm.close()


def run_rows(args, arch, lib, record):
    h, k, n = 128, 12288, 256
    ms = [1, 2, 8, 16, 32, 64, 128, 256]
    roots = {m: inputs(m, h, k, n, seed=7 + i) for i, m in enumerate(ms)}
    warm = package_quantized_product_graph(owning.source(16, h, k, n, arch)).prepare(
        roots[16], runtime_library=lib._name)
    warm.invoke()
    native = compile_bounded(arch, (16, h, k, n), roots[16], m_bound=256)
    record["bounded_gate_tile"] = tile(native)
    bounded = native.prepare(roots[16], runtime_library=lib._name)
    for m in ms:
        static = package_quantized_product_graph(owning.source(m, h, k, n, arch)).prepare(
            roots[m], runtime_library=lib._name)
        bounded.update(roots[m])
        equal = bitwise(bounded, static)
        if not equal:
            raise SystemExit(f"bounded and static outputs differ at M = {m}")
        row = {"static_gate_tile": tile(static), "bitwise_equal": equal,
               **compare(bounded, static, (*range(7), "program"), args.rounds, args.repeats)}
        record["rows"][str(m)] = row
        print(m, {a: [row[a][str(i)]["median"] for i in range(7)] + [row[a]["program"]["median"]]
                  for a in ("bounded", "static")}, flush=True)
        static.close()
    bounded.close()
    warm.close()


def run_isa(args, arch, lib, record):
    out = args.output.parent / "isa"
    out.mkdir(parents=True, exist_ok=True)
    cases = {"rows_256x128x12288x256": ((256, 128, 12288, 256), {"m_bound": 256}),
             "n_256x128x12288x128": ((256, 128, 12288, 128), {"m_bound": 256, "k_bound": 12288,
                                                               "n_bound": 512})}
    for name, (shape, bounds) in cases.items():
        roots = inputs(*shape, seed=7)
        images = {"bounded": compile_bounded(arch, shape, roots, **bounds),
                  "static": package_quantized_product_graph(owning.source(*shape, arch))}
        record["rows"][name] = {}
        for arm, package in images.items():
            stage = package.native_stages[2]
            path = out / f"{name}_{arm}_gate_product.hsaco"
            path.write_bytes(stage.image.payload)
            (out / f"{name}_{arm}_gate_product.backend.ll").write_text(stage.backend_ir)
            record["rows"][name][arm] = {
                "macro_tile": stage.descriptor.provenance.get("macro_tile"),
                "entry_symbol": stage.descriptor.entry_symbol,
                "hsaco_sha256": hashlib.sha256(stage.image.payload).hexdigest(), "hsaco": path.name}
        print(name, record["rows"][name], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["panel", "rows", "isa"])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--force-min-m", type=int, choices=[256], default=None)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=200)
    owning.add_arch_arguments(parser, "GFX1151-QUANTIZED-MLP-PROBES-20261102")
    args = parser.parse_args()
    if args.force_min_m and args.mode != "panel":
        parser.error("--force-min-m applies to the panel probe only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arch = owning.owning_arch()
    owning.check_sync_key(args.sync_key, arch)
    lock = owning.device_lock(args.device_lock)
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise SystemExit("native ROCm movement runtime unavailable")
    record = {"sync_key": args.sync_key, "mode": args.mode, "architecture": arch,
              "device": owning.device(arch), "device_lock": lock, "warm_up_owner": args.mode != "isa",
              "force_min_m": args.force_min_m, "members": MEMBERS, "route": owning.ROUTE,
              "latency_source": "hip_event_member_range_window",
              "timing_source": {"member": "hip_event_member_range_window",
                                "program": "hip_event_owner_window"},
              "unit": "us per invocation; median, min, max and every window",
              "timing": (f"HIP-event windows of {args.repeats} invocations, {args.rounds} rounds, "
                         "arm order alternating each round. Diagnostic, not a promotion."),
              "rows": {}}
    {"panel": run_panel, "rows": run_rows, "isa": run_isa}[args.mode](args, arch, lib, record)
    record.update(provenance(lib))
    args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
