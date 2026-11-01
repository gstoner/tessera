"""Diagnostic: bounded-row MLP owner vs static owners compiled at each M.

GFX1201-NVFP4-BOUNDED-MLP-20261027. One bounded owner prepared at capacity
serves every active M (activation-only update, weights resident); each static
arm is a separately compiled and prepared seven-member program at that M. Both
are checked bitwise against each other before timing, then timed in alternating
ABAB/BABA rounds through the owner's native HIP event window. HIP events on WSL2
are diagnostic only; they promote nothing.

The recorder runs on the live owning architecture (gfx1201 or gfx1151,
``quantized_mlp_arch``) and labels the packet with it. A kept, never-timed
warm-up owner is prepared first, so neither arm takes the slower-for-life first
owner slot (GFX1201-BOUNDED-PER-M-PANEL-20261030); ``--no-warm-up-owner``
reproduces the original gfx1201 recording.
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
from benchmarks.rocm import quantized_mlp_arch as owning
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--shape", type=int, nargs=3, default=(128, 12288, 256), metavar=("H", "K", "N"))
parser.add_argument("--capacity", type=int, default=256)
parser.add_argument("--actives", type=int, nargs="+", default=(1, 16, 64, 256))
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--repeats", type=int, default=50)
parser.add_argument("--no-warm-up-owner", dest="warm_up_owner", action="store_false")
owning.add_arch_arguments(parser, "GFX1201-NVFP4-BOUNDED-MLP-20261027")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
arch = owning.owning_arch()
owning.check_sync_key(args.sync_key, arch)
lock = owning.device_lock(args.device_lock)
runtime = rt._load_rocm_native_movement_runtime()
if runtime is None:
    raise SystemExit("native ROCm movement runtime unavailable")
lib = runtime._name


def source(m, h, k, n):
    return owning.source(m, h, k, n, arch)


h, k, n = args.shape
bound = args.capacity
text = source(args.actives[0], h, k, n).replace(
    "tessera.native.quantized_product}",
    f"tessera.native.quantized_product, tessera.native.quantized_m_bound = {bound} : i64}}")
start = time.perf_counter()
bounded_package = package_quantized_product_graph(text)
bounded_compile_s = time.perf_counter() - start
roots = {m: inputs(m, h, k, n, seed=500 + m) for m in args.actives}
for m in args.actives:
    roots[m][1:] = roots[args.actives[0]][1:]  # one weight set; activations vary
warm = None
if args.warm_up_owner:
    warm = package_quantized_product_graph(source(args.actives[0], h, k, n)).prepare(
        roots[args.actives[0]], runtime_library=lib)
    warm.invoke()
bounded = bounded_package.prepare(roots[args.actives[0]], runtime_library=lib)
static, compile_s = {}, {}
for m in args.actives:
    start = time.perf_counter()
    package = package_quantized_product_graph(source(m, h, k, n))
    compile_s[m] = time.perf_counter() - start
    static[m] = package.prepare(roots[m], runtime_library=lib)

update_ms = {m: [] for m in args.actives}
samples = {m: {"bounded": [], "static": []} for m in args.actives}
for round_ in range(args.rounds):
    order = list(args.actives) if round_ % 2 == 0 else list(args.actives)[::-1]
    for m in order:
        start = time.perf_counter()
        bounded.update_activations(roots[m][0])
        update_ms[m].append((time.perf_counter() - start) * 1000)
        arms = ("bounded", "static") if round_ % 2 == 0 else ("static", "bounded")
        outputs = {}
        for arm in arms:
            owner = bounded if arm == "bounded" else static[m]
            generation, elapsed = owner.invoke(repeats=args.repeats, timed=True)
            samples[m][arm].append(elapsed * 1000)
            outputs[arm] = owner.read(generation)[0]
        if round_ == 0:
            # Different M schedules may differ in accumulation; record, do not assume.
            outputs["equal"] = bool(np.array_equal(outputs["bounded"].view(np.uint16),
                                                   outputs["static"].view(np.uint16)))
            samples[m]["bitwise_equal_static_at_m"] = outputs["equal"]
capacity_info = bounded.rows_info()
bounded.close()
for owner in static.values():
    owner.close()
if warm is not None:
    warm.close()
rows = {}
for m in args.actives:
    b, s = statistics.median(samples[m]["bounded"]), statistics.median(samples[m]["static"])
    rows[str(m)] = {"bounded_us": round(b, 2), "static_us": round(s, 2),
                    "bounded_over_static": round(b / s, 3),
                    "bitwise_equal_static_at_m": samples[m]["bitwise_equal_static_at_m"],
                    "activation_update_ms": round(statistics.median(update_ms[m]), 3),
                    "static_compile_s": round(compile_s[m], 2)}
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
args.output.write_text(json.dumps({
    "sync_key": args.sync_key, "commit": commit, "worktree_dirty": dirty,
    "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    "compiler_sha256": hashlib.sha256(pathlib.Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
    "provider_sha256": hashlib.sha256(pathlib.Path(lib).read_bytes()).hexdigest(),
    "architecture": arch, "device": owning.device(arch), "device_lock": lock,
    "warm_up_owner": args.warm_up_owner, "route": owning.ROUTE,
    "latency_source": owning.HIP_EVENT_TIMING, "timing_source": owning.HIP_EVENT_TIMING,
    "activation_update_timing_source": owning.HOST_WALL_TIMING, "shape_hkn": [h, k, n], "capacity_m": bound,
    "bounded_compile_s": round(bounded_compile_s, 2),
    "bounded_allocation_bytes": capacity_info[2], "bounded_allocation_count": capacity_info[3],
    "timing": (f"native owner HIP event window per program invocation averaged over {args.repeats} "
               f"repeats; {args.rounds} alternating rounds; median us. Activation update is host wall "
               "time of the bounded owner's activation-only row update. Diagnostic, not a promotion."),
    "rows": rows}, indent=1) + "\n")
print(json.dumps(rows, indent=1))
