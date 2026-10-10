"""Correctness-gated native four-f32 adjoint attribution on gfx1201."""
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
from tessera.compiler.native_scaled_program import package_native_scaled_vjp, PreparedScaledProgram
from tests.device.rocm.test_floating_scaled_adjoint import inputs, oracle, batch_inputs, batch_oracle
from tests.unit.test_native_floating_scaled_adjoint import floating_source, floating_batch_source


def record(ta, tb, placed, repetitions, policy=None):
    values = inputs(ta, tb) if policy is None else batch_inputs(policy, ta, tb)
    seed = np.ascontiguousarray(values[-1].T) if placed else values[-1]
    frame = (*values[:4], seed)
    expected = oracle(values, ta, tb) if policy is None else batch_oracle(values, ta, tb)
    start = time.perf_counter()
    source = (floating_source(ta, tb, permute=placed) if policy is None
              else floating_batch_source(policy, ta, tb))
    package = package_native_scaled_vjp(source)
    compile_ms = (time.perf_counter() - start) * 1e3
    program = json.loads(package.program_json)
    def compare(outputs):
        for output, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(output, reference, rtol=4e-5, atol=3e-6)
    captured, native, host = [], [], []
    with PreparedScaledProgram(package, frame,
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as owner:
        generation, _ = owner.invoke()
        compare(owner.read(generation))
        for _ in range(5):
            generation, members = owner.profile_members(repeats=repetitions)
            compare(owner.read(generation))
            captured.append(list(members))
            generation, elapsed = owner.invoke(repeats=128, timed=True)
            compare(owner.read(generation))
            native.append(elapsed)
            start = time.perf_counter()
            for _ in range(20):
                owner.update(frame)
                generation, _ = owner.invoke()
                actual = owner.read(generation)
            host.append((time.perf_counter() - start) * 1e3 / 20)
        compare(actual)
    return {
        "transpose_a": ta, "transpose_b": tb, "output_permuted": placed,
        "logical_mnk": [2, 5, 9], "block_nk": [4, 4],
        "batching": policy, "operand_shapes": [list(value.shape) for value in values],
        "gradient_roles": program["gradient_roles"],
        "members": [row.get("gradient_role", row["operation"]) for row in program["steps"]],
        "correctness": "independent_float64_passed_before_and_after_timing",
        "max_abs_error": max(float(np.max(np.abs(output - reference)))
                             for output, reference in zip(actual, expected, strict=True)),
        "compile_package_ms": compile_ms, "captured_repetitions": repetitions,
        "captured_member_samples_ms": captured,
        "captured_member_median_ms": [median(row[i] for row in captured)
                                      for i in range(len(captured[0]))],
        "captured_min_window_ms": min(min(row) * repetitions for row in captured),
        "interleaved_native_program_samples_ms": native,
        "interleaved_native_program_median_ms": median(native),
        "checked_update_invoke_read_samples_ms": host,
        "checked_update_invoke_read_median_ms": median(host),
        "graph_sha256": hashlib.sha256(package.graph_ir.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 65536:
        raise ValueError("capture repetitions must be in 1..65536")
    architecture = runtime._rocm_live_arch()
    if architecture != "gfx1201":
        raise RuntimeError(f"owning gfx1201 required, observed {architecture}")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(os.environ[name]) for name in
             ("TESSERA_OPT", "TESSERA_ROCM_OPT", "TESSERA_ROCM_NATIVE_MOVEMENT_LIB")]
    paths += [root / name for name in (
        "src/compiler/ir/LinearTransposeInterface.cpp",
        "src/compiler/ir/TesseraOps.cpp",
        "src/compiler/ir/include/Tessera/IR/StructuredReductionContract.h",
        "src/compiler/programming_model/lib/NativeScaleTranspose.h",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "python/tessera/compiler/native_scaled_program.py",
        "tests/device/rocm/test_floating_scaled_adjoint.py",
        "tests/unit/test_native_floating_scaled_adjoint.py")]
    paths.append(Path(__file__).resolve())
    packet = {
        "architecture": architecture,
        "device_inventory": subprocess.run(["rocminfo"], check=True, capture_output=True,
                                           text=True, timeout=30).stdout,
        "identity_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases": [record(ta, tb, placed, args.repetitions, policy)
                  for ta, tb, placed, policy in ((False, False, False, None),
                      (True, True, True, None), (True, False, False, "shared_lhs"),
                      (True, True, False, "broadcast"))],
        "timing_scope": {
            "captured_members": "grouped pure-SSA device graph windows including dispatch; capture/instantiation/copies excluded",
            "interleaved_native_program": "ordinary 128-repeat native launch window including host dispatch between kernels",
            "checked_update_invoke_read": "prepared ABI update, native execution and copied outputs; excludes frontend and compilation",
            "compile": "textual Graph AD/Schedule/Tile/Target native package compilation",
        },
        "claim": "static continuous f32 native adjoint execution and attribution; no speedup, public Python JIT or generic transformation closure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
