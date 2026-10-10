"""Correctness-gated public mapped-output timing on owning gfx1201."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import numpy as np

from tessera import runtime
from tessera.autodiff import vmap
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tests.unit.test_native_typed_scaled_vmap import case


def record(fmt, axis, mode="primal"):
    scalar, leading, values, oracle = case("independent_rhs", fmt, False, (2, 17, 19, 256))
    owner = vmap(scalar, in_axes=leading._frontend_batch_axes, out_axes=axis)
    expected = (np.moveaxis(oracle, 0, axis),)
    prepared_inputs = values
    if mode == "reverse":
        import tessera as ts
        from tests.device.rocm.test_public_mapped_scale_vjp import scale_oracle
        reverse = ts.jit(target="rocm_gfx1201", autodiff="reverse",
                         wrt=("sa", "sb"))(scalar._fn)
        owner = vmap(reverse, in_axes=leading._frontend_batch_axes, out_axes=axis)
        seed = np.random.default_rng(908).uniform(-.5, .5, oracle.shape).astype(np.float32)
        dy = np.ascontiguousarray(np.moveaxis(seed, 0, axis))
        expected = scale_oracle(values, leading._frontend_batch_axes, seed)
        prepared_inputs = (*values, dy)
        def invoke():
            return owner.native_backward(*values, out_cotangents=dy)
    elif mode == "jvp":
        import tessera as ts
        forward = ts.jit(target="rocm_gfx1201", autodiff="forward",
                         wrt=("sa", "sb"))(scalar._fn)
        owner = vmap(forward, in_axes=leading._frontend_batch_axes, out_axes=axis)
        seeds = (values[2] * np.float32(.1), values[3] * np.float32(.2))
        expected = (expected[0], expected[0] * .3)
        prepared_inputs = (*values, *seeds)
        def invoke():
            return owner.native_jvp(*values, tangents=seeds)
    else:
        def invoke():
            return (owner(*values),)
    def compare(outputs):
        for output, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(output, reference, rtol=4e-5, atol=3e-5)
    start = time.perf_counter()
    actual = invoke()
    cold_ms = (time.perf_counter() - start) * 1e3
    compare(actual)
    if mode == "reverse":
        package = owner._native_backward_artifact
    elif mode == "jvp":
        contract = next(iter(owner._native_jvp_packages.values())).contract
        package = NativeScaledProgram.from_manifest(
            contract["steps"][0]["child_metadata"]["native_scaled_program"])
    else:
        package = owner._native_composed_scaled_last_program
    program = json.loads(package.program_json)
    library = os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]
    with PreparedScaledProgram(package, prepared_inputs, runtime_library=library) as prepared:
        repetitions = 1024
        while True:
            generation, member_ms = prepared.profile_members(repeats=repetitions)
            compare(prepared.read(generation))
            if min(member_ms) * repetitions >= 20 or repetitions == 65536:
                break
            repetitions *= 2
        captured = []
        for _ in range(5):
            generation, member_ms = prepared.profile_members(repeats=repetitions)
            compare(prepared.read(generation))
            captured.append(list(member_ms))
    warm = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            actual = invoke()
        warm.append((time.perf_counter() - start) * 1e3 / 20)
    compare(actual)
    return {
        "mode": mode, "format": fmt, "shape_bmnk": [2, 17, 19, 256], "out_axes": axis,
        "output_shapes": [list(output.shape) for output in actual],
        "execution_receipt": (owner.last_backward_execution if mode == "reverse" else
                              owner.last_jvp_execution if mode == "jvp" else
                              owner._native_descriptor_last_receipt)["execution_kind"],
        "members": [step["operation"] for step in program["steps"]],
        "correctness": "passed_before_and_after_timing",
        "max_abs_error": max(float(np.max(np.abs(output - reference))) for output, reference in zip(actual, expected, strict=True)),
        "cold_public_compile_execute_ms": cold_ms,
        "captured_repetitions": repetitions,
        "captured_member_samples_ms": captured,
        "captured_member_median_ms": [median(row[i] for row in captured) for i in range(len(captured[0]))],
        "captured_min_window_ms": min(min(row) * repetitions for row in captured),
        "warm_public_samples_ms": warm, "warm_public_median_ms": median(warm),
        "graph_sha256": hashlib.sha256(package.graph_ir.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("all", "reverse"), default="all")
    args = parser.parse_args()
    architecture = runtime._rocm_live_arch()
    if architecture != "gfx1201":
        raise RuntimeError(f"owning gfx1201 required, observed {architecture}")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(os.environ["TESSERA_OPT"]), Path(os.environ["TESSERA_ROCM_OPT"]),
             Path(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
             root / "python/tessera/compiler/native_vmap.py",
             root / "python/tessera/compiler/jit.py",
             root / "python/tessera/compiler/rocm_typed_scaled_native.py",
             root / "python/tessera/compiler/native_scaled_program.py",
             root / "src/transforms/lib/NativeScaledMatmulProgram.h",
             root / "src/compiler/programming_model/lib/NativeScaleTranspose.h",
             Path(__file__).resolve()]
    packet = {
        "architecture": architecture,
        "device_inventory": subprocess.run(["rocminfo"], check=True, capture_output=True,
                                           text=True, timeout=30).stdout,
        "identity_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases": ([record("fp32", axis, "reverse") for axis in (1, -1)] if args.mode == "reverse" else
                  [record(fmt, axis) for fmt in ("fp32", "e8m0") for axis in (1, -1)] +
                  [record("fp32", axis, "jvp") for axis in (1, -1)]),
        "timing_scope": {
            "captured_members": "grouped pure-SSA device graph windows including dispatch; capture/instantiation/copies excluded",
            "warm_public": "ordinary vmap JIT call including input preparation, cache admission, HIP execution and copied return",
            "cold_public": "first compiler-owned public call including frontend certificate, compilation and execution",
        },
        "claim": "bounded public static mapped execution and cost attribution; no speedup or general AD closure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
