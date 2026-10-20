"""Correctness-gated public/device latency controls for native tensor JIT migration."""
from contextlib import ExitStack
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
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_native_tensor_program_package_execution import (
    CASES, FUNCTIONS, oracle, roots, values,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity():
    tool = Path(os.environ["TESSERA_OPT"])
    provider = Path(os.environ["TESSERA_NATIVE_FRAME_TEST_LIB"])
    files = (
        "python/tessera/compiler/jit.py", "python/tessera/compiler/execution_matrix.py",
        "python/tessera/compiler/native_tensor_program.py",
        "python/tessera/compiler/native_tensor_jit.py",
        "python/tessera/compiler/prepared_native_tensor_frame.py", "python/tessera/runtime.py",
        "tests/device/nvidia/test_native_tensor_program_package_execution.py",
    )
    compiler_source = tool.parent / "source-identity.json"
    compiler = json.loads(compiler_source.read_text())
    if compiler["compiler_sha256"] != digest(tool.read_bytes()):
        raise RuntimeError("compiler source receipt differs from the live tool")
    gpu = subprocess.check_output([
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader",
        ], text=True).strip()
    if len(gpu.splitlines()) != 1 or gpu.split(",")[-1].strip() != "12.0":
        raise RuntimeError("recorder requires one visible owning SM120 GPU")
    return {
        "gpu": gpu,
        "compiler_sha256": digest(tool.read_bytes()), "provider_sha256": digest(provider.read_bytes()),
        "compiler_version": subprocess.check_output([str(tool), "--version"], text=True),
        "compiler_source": compiler,
        "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: digest(Path(name).read_bytes()) for name in files},
        "recorder_sha256": digest(Path(__file__).read_bytes()),
    }


def case(kind, storage, route, rounds, repeats):
    a, b = values(storage, 71)
    snapshots = a.copy(), b.copy()
    started = time.perf_counter_ns()
    fn = ts.jit(target="nvidia_sm120")(FUNCTIONS[kind])
    decoration_ms = (time.perf_counter_ns() - started) / 1e6
    try:
        started = time.perf_counter_ns()
        try:
            result = fn(*roots(kind, a, b))
        except ValueError as error:
            if (route == "legacy" and kind == "norm"
                    and str(error) == "LHS normalization needs a separate affine/axis contract"):
                return {"kind": kind, "storage": storage, "status": "unsupported",
                        "reason": str(error), "route": route}
            raise
        cold_ms = (time.perf_counter_ns() - started) / 1e6
        expected = oracle(kind, a, b)
        np.testing.assert_allclose(result, expected, rtol=.004, atol=2e-5)
        artifact = fn.runtime_artifact()
        if route == "native":
            program = fn._native_tensor_last_program
            owner = fn._native_tensor_last_owner
            assert program is not None and owner is not None
            assert artifact.metadata["compiler_path"] == "canonical_native_tensor_program"
            images = program.images
            before = owner.stats()
        else:
            assert getattr(fn, "_native_tensor_last_program", None) is None
            assert fn._nvidia_lhs_last_program is not None
            assert artifact.metadata["compiler_path"] == "canonical_nvidia_lhs_program"
            images = tuple(package.image.payload for package in fn.native_lhs_packages())
            calls = tuple(fn._nvidia_lhs_prepared_calls.values())
            assert len(calls) == 1
            owner = calls[0]
            before = None

        def forbidden(*args, **kwargs):
            raise RuntimeError("timed public call invoked external tools or native replay retraced")

        samples = []
        with ExitStack() as guarded:
            guarded.enter_context(patch.object(subprocess, "run", side_effect=forbidden))
            guarded.enter_context(patch.object(subprocess, "Popen", side_effect=forbidden))
            if route == "native":
                guarded.enter_context(patch.object(fn, "_trace_frontend_capture", side_effect=forbidden))
            for _ in range(rounds):
                started = time.perf_counter_ns()
                for _ in range(repeats):
                    result = fn(*roots(kind, a, b))
                samples.append((time.perf_counter_ns() - started) / 1e6 / repeats)
        np.testing.assert_allclose(result, expected, rtol=.004, atol=2e-5)
        assert fn.execution_kind == "native_gpu"
        saved = result.copy()
        windows = {}
        for captured in (False, True):
            raw = owner.profile(64, captured=captured)
            if "output" in raw:
                np.testing.assert_allclose(raw["output"], expected, rtol=.004, atol=2e-5)
            windows["captured" if captured else "enqueued"] = {
                "program_ms": raw["program_ms"],
                "stage_windows_ms": raw.get("stage_ms", raw.get("grouped_stage_ms")),
                "domain": raw.get("measurement", raw.get("timing_domain")),
            }
        np.testing.assert_array_equal(result, saved)
        np.testing.assert_array_equal(a, snapshots[0])
        np.testing.assert_array_equal(b, snapshots[1])
        if route == "native":
            after = owner.stats()
            assert before["capacity"] == after["capacity"]
            assert before["allocations"] == after["allocations"] == 1
        else:
            after = None
        return {
            "kind": kind, "storage": storage, "status": "passed", "route": route,
            "shape_mnk": [17, 11, 32], "correctness": "passed_before_after_and_profile",
            "decoration_ms": decoration_ms, "cold_call_ms": cold_ms,
            "samples_completed_public_ms": samples, "median_completed_public_ms": median(samples),
            "rounds": rounds, "calls_per_sample": repeats, "device_windows": windows,
            "artifact_hash": artifact.artifact_hash,
            "ir_sha256": {name: digest(getattr(artifact, name).encode()) if getattr(artifact, name) else None
                          for name in ("graph_ir", "schedule_ir", "tile_ir", "target_ir")},
            "image_sha256": [digest(image) for image in images],
            "before": before, "after": after,
        }
    finally:
        fn.close_native_storage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("native", "legacy"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 1000 or not 1 <= args.repeats <= 10000:
        raise ValueError("benchmark counts exceed the bounded recording envelope")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("recorder requires owning SM120 execution")
    source = identity()
    rows = []
    for kind, storage in CASES:
        row = case(kind, storage, args.route, args.rounds, args.repeats)
        rows.append(row)
        print(kind, storage, row["status"], row.get("median_completed_public_ms"), flush=True)
    packet = {
        "schema": "tessera.native_tensor_jit_measurement.v1", "label": args.label,
        "architecture": "sm_120", "route": args.route, "identity": source, "rows": rows,
        "scope": "M17/N11/K32 completed public latency; cold compile/bind separated; "
                 "CUDA program/stage windows exclude host copies and are independent/nonadditive; "
                 "no pure-kernel, throughput or physical schedule promotion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(packet, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
