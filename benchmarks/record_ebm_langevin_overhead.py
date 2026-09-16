#!/usr/bin/env python3
"""What the cooperative EBM Langevin kernel costs against the Python-emitted lane.

The GA/EBM review asks for the dispatch / host-transfer / wall-time comparison
before the native route may displace `rocm_ebm_langevin_compiled`. The two
routes differ structurally, and that difference is the point:

  native (row-program)   the gradient is derived by the compiler and evaluated
                         INSIDE the kernel, the K steps run in registers, so a
                         K-step loop is ONE launch and one host round trip.
  Python-emitted         the kernel takes `(y, grad)`: the caller computes the
                         gradient on the host and launches once PER step, so a
                         K-step loop is K launches and K round trips.

Measured at temperature 0, where both routes compute exactly the same
iterated descent `y - eta*(y - x)`, so the comparison is like-for-like and the
agreement is checked here rather than assumed. The noise policies differ
between the routes (each has its own Philox counter scheme), so the T > 0 rows
are recorded for the native route alone and are NOT a comparison.

Both routes are dispatched at the SAME wrapper depth — `runtime.launch` on an
artifact — so this measures the routes, not one route plus a wrapper.

Wall clock only (`latency_source = "host_wall_clock"`): neither WSL2 ROCm box
exposes `/dev/kfd`, so `rocprofv3` returns no dispatch or counter records and a
kernel-time attribution is unavailable here. Nothing in this packet promotes a
lane; `promotion_eligible` is false and a bare-metal calibration is still owed
on any NVIDIA timing.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from tessera import runtime as rt  # noqa: E402
from tessera.compiler.llvm_tools import llvm_bin_dir  # noqa: E402
from tessera.compiler.scheduled_matmul import find_tessera_opt  # noqa: E402
from tessera.ebm import native_langevin as nl  # noqa: E402

CASES = [((4, 8), 1), ((4, 8), 8), ((16, 64), 8), ((16, 64), 32), ((64, 256), 32)]
REPS = 9
ETA = 0.05


def _median_ms(fn, reps=REPS):
    fn()                      # warm up: compile, cache, first touch
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), min(samples)


def _python_step_artifact(target, kwargs):
    path = "rocm_ebm_langevin_compiled" if target == "rocm" else "x86_ebm_langevin_compiled"
    return rt.RuntimeArtifact(metadata={
        "target": target, "compiler_path": path, "executable": True,
        "arg_names": ["y", "g"], "output_name": "o",
        "ops": [{"op_name": "tessera.ebm.langevin_step", "result": "o",
                 "operands": ["y", "g"], "kwargs": kwargs}]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nvidia", "rocm"], required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llvm = llvm_bin_dir()
    if llvm is None:
        raise SystemExit("matched LLVM tools are required")
    if args.backend == "rocm":
        native_target, python_target = "rocm", "rocm"
    else:
        # sm_120 has no Python-emitted EBM Langevin lane; the x86 one is the
        # only other compiled implementation and runs on this host's CPU, so
        # the comparison there is native-GPU vs native-CPU and is labelled so.
        native_target, python_target = "nvidia_sm120", "x86"
    rng = np.random.default_rng(20260916)
    rows = []
    for shape, steps in CASES:
        y0 = rng.standard_normal(shape).astype(np.float32)
        x = rng.standard_normal(shape).astype(np.float32)
        key = [17, 4]
        native_art = nl.package_ebm_langevin_native(shape, eta=ETA, temperature=0.0, steps=steps,
                                                    target=native_target)
        step_art = _python_step_artifact(python_target, {"eta": ETA, "temperature": 0.0})

        def native():
            out = rt.launch(native_art, (y0, x, key))
            if not out["ok"]:
                raise SystemExit(f"native route refused: {out.get('reason')}")
            return np.asarray(out["output"][0])

        def python_loop():
            y = y0
            for _ in range(steps):
                # The host owns the gradient for this route — that IS the route.
                grad = (y - x).astype(np.float32)
                res = rt.launch(step_art, (y, grad))
                if not res["ok"]:
                    raise SystemExit(f"python-emitted route refused: {res.get('reason')}")
                y = np.asarray(res["output"], np.float32)
            return y

        got, want = native(), python_loop()
        agree = float(np.max(np.abs(got - want)))
        if not np.allclose(got, want, rtol=1e-5, atol=1e-5):
            raise SystemExit(f"{shape} K={steps}: the two routes disagree at T=0 (max abs {agree}); "
                             "the timing comparison would not be like-for-like")
        native_ms, native_min = _median_ms(native)
        python_ms, python_min = _median_ms(python_loop)
        element_bytes = int(np.prod(shape)) * 4
        rows.append(dict(
            shape=list(shape), steps=steps, temperature=0.0, max_abs_difference=agree,
            native=dict(route="row_program_cooperative_kernel", target=native_target,
                        launches_per_loop=1, host_round_trips_per_loop=1,
                        host_bytes_per_loop=3 * element_bytes + 16,
                        median_ms=native_ms, min_ms=native_min,
                        gradient="compiler-derived, evaluated in the kernel"),
            python_emitted=dict(route="python_emitted_step_kernel", target=python_target,
                                launches_per_loop=steps, host_round_trips_per_loop=steps,
                                host_bytes_per_loop=steps * 3 * element_bytes,
                                median_ms=python_ms, min_ms=python_min,
                                gradient="host, per step"),
            speedup_median=python_ms / native_ms if native_ms else None,
            latency_source="host_wall_clock"))
        print(json.dumps(rows[-1]["native"] | {"shape": list(shape), "steps": steps,
                                               "python_ms": python_ms}), flush=True)

    # The native route also runs the cases the other cannot: noise and the
    # gradient stay on the device, so T > 0 costs no extra launches.
    hot = []
    for shape, steps in CASES[:3]:
        y0 = rng.standard_normal(shape).astype(np.float32)
        x = rng.standard_normal(shape).astype(np.float32)
        art = nl.package_ebm_langevin_native(shape, eta=ETA, temperature=0.6, steps=steps, target=native_target)
        ms, lo = _median_ms(lambda: rt.launch(art, (y0, x, [3, 3])))
        hot.append(dict(shape=list(shape), steps=steps, temperature=0.6, launches_per_loop=1,
                        median_ms=ms, min_ms=lo, latency_source="host_wall_clock"))

    packet = dict(schema=1, backend=args.backend, chip=args.chip, host=platform.node(),
                  compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                  recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  llvm_bin=str(llvm), eta=ETA, reps=REPS, rows=rows, native_with_noise=hot,
                  comparison="temperature 0, where both routes compute the same iterated descent; "
                             "the agreement is checked per row before the timing is kept",
                  claims=["the native route is ONE launch and one host round trip per K-step loop; "
                          "the Python-emitted route is K of each, because its kernel takes the "
                          "gradient from the caller",
                          "both routes are dispatched through runtime.launch on an artifact, so this "
                          "measures the routes and not a wrapper difference"],
                  not_claimed=["no promotion: this packet does not move any lane",
                               "wall clock only — neither WSL2 ROCm box exposes /dev/kfd, so rocprofv3 "
                               "returns no dispatch or counter records and kernel time is unavailable",
                               "WSL2 timings do not promote; bare-metal calibration is owed",
                               "the T > 0 rows are the native route alone: the two routes' Philox "
                               "counter policies differ, so their samples are not comparable"],
                  promotion_eligible=False, measured_performance=True,
                  rocm_chip_env=os.environ.get("TESSERA_ROCM_CHIP"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), rows=len(rows),
                          speedups=[round(r["speedup_median"], 2) for r in rows])))


if __name__ == "__main__":
    main()
