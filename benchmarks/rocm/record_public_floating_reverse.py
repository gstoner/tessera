"""Correctness-gated public f32 reverse and native ABI timings on gfx1201."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch

import numpy as np

from tessera import runtime
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.device.rocm.test_floating_scaled_adjoint import inputs, oracle, batch_inputs, batch_oracle
from tests.unit.test_public_floating_scaled_reverse import floating_owner, floating_batch_owner


def record(policy, ta, tb, repetitions):
    values = inputs(ta, tb) if policy is None else batch_inputs(policy, ta, tb)
    expected = oracle(values, ta, tb) if policy is None else batch_oracle(values, ta, tb)

    def compare(actual):
        for output, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(output, reference, rtol=4e-5, atol=3e-6)

    start = time.perf_counter()
    owner = (floating_owner(ta, tb) if policy is None else
             floating_batch_owner(policy, ta, tb))
    decoration_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    actual = owner.native_backward(*values[:4], out_cotangents=values[4])
    first_call_ms = (time.perf_counter() - start) * 1e3
    compare(actual)
    receipt = owner.last_backward_execution
    if (receipt["execution_kind"] != "native_gpu" or
            receipt["evidence_target"] != "rocm_gfx1201" or
            receipt["frontend_authority"] != "tracer"):
        raise RuntimeError("public reverse receipt does not prove the owning native route")
    package = owner._native_backward_artifact
    public, native, captured = [], [], []

    def forbid_compile(*args, **kwargs):
        raise AssertionError("warm public reverse invoked a compiler")

    with patch("subprocess.run", side_effect=forbid_compile):
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                actual = owner.native_backward(*values[:4], out_cotangents=values[4])
            public.append((time.perf_counter() - start) * 1e3 / 20)
            compare(actual)
    with PreparedScaledProgram(package, values,
            runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        generation, _ = prepared.invoke()
        compare(prepared.read(generation))
        for _ in range(5):
            generation, members = prepared.profile_members(repeats=repetitions)
            captured.append(list(members))
            compare(prepared.read(generation))
            generation, elapsed = prepared.invoke(repeats=128, timed=True)
            native.append(elapsed)
            compare(prepared.read(generation))
    program = json.loads(package.program_json)
    return {
        "batching": policy, "transpose_a": ta, "transpose_b": tb,
        "operand_shapes": [list(value.shape) for value in values],
        "gradient_roles": program["gradient_roles"],
        "members": [row.get("gradient_role", row["operation"]) for row in program["steps"]],
        "correctness": "independent_float64_passed_before_and_after_timing",
        "max_abs_error": max(float(np.max(np.abs(output-reference)))
                             for output, reference in zip(actual, expected, strict=True)),
        "frontend_decoration_ms": decoration_ms, "first_public_reverse_ms": first_call_ms,
        "warm_public_reverse_samples_ms": public, "warm_public_reverse_median_ms": median(public),
        "captured_repetitions": repetitions, "captured_member_samples_ms": captured,
        "captured_member_median_ms": [median(row[i] for row in captured)
                                      for i in range(len(captured[0]))],
        "captured_min_window_ms": min(min(row)*repetitions for row in captured),
        "interleaved_native_program_samples_ms": native,
        "interleaved_native_program_median_ms": median(native),
        "source_graph_sha256": receipt["source_graph_ir_digest"],
        "native_package_sha256": receipt["artifact_hash"],
        "physical_attestation": receipt["physical_attestation"],
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
    paths = [Path(os.environ[name]) for name in (
        "TESSERA_OPT", "TESSERA_ROCM_OPT", "TESSERA_ROCM_NATIVE_MOVEMENT_LIB")]
    paths += [root/name for name in (
        "python/tessera/compiler/capabilities.py",
        "python/tessera/compiler/graph_ir.py",
        "python/tessera/compiler/dtype_flow_audit.py",
        "python/tessera/compiler/reference_typed_scaled_matmul.py",
        "python/tessera/compiler/native_scaled_program.py",
        "python/tessera/compiler/native_vjp_plugins.py",
        "src/compiler/ir/LinearTransposeInterface.cpp",
        "src/compiler/ir/TesseraOps.cpp",
        "tests/unit/test_public_floating_scaled_reverse.py",
        "tests/device/rocm/test_public_floating_scaled_reverse.py",
        "tests/device/rocm/test_floating_scaled_adjoint.py")]
    paths.append(Path(__file__).resolve())
    packet = {
        "architecture": architecture,
        "device_inventory": subprocess.run(["rocminfo"], check=True, capture_output=True,
                                           text=True, timeout=30).stdout,
        "identity_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cases": [record(policy, ta, tb, args.repetitions) for policy, ta, tb in (
            (None, False, False), (None, True, True),
            ("shared_lhs", True, False), ("broadcast", True, True))],
        "timing_scope": {
            "frontend_decoration": "source parsing and JIT owner setup before concrete inputs",
            "first_public_reverse": "concrete frontend certificate/trace, native package compilation, preparation, execution and copied gradients",
            "warm_public_reverse": "public native_backward including concrete frontend work and new native owner preparation, execution and copied gradients; compiler calls forbidden",
            "captured_members": "grouped pure-SSA device graph windows including dispatch; capture/instantiation/copies excluded",
            "interleaved_native_program": "ordinary 128-repeat prepared native launch window including host dispatch between kernels",
        },
        "claim": "owning public Python reverse through typed Graph/native AD/Schedule/Tile/Target/LLVM/HSACO/checked ABI; no primal f32 kernel or generic AD closure claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
