"""Capture exact native packages matching an existing SM120 timing packet.

Run on the owning host, from the frozen repository root:
  PYTHONPATH=python:. python capture_programs.py --timings timings.json --output programs
No timings are recorded here. This is compiler-free replay and numerical proof.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product, product_reversed

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--timings", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
timings = json.loads(options.timings.read_text())
expected_source = timings["source_before"]
assert expected_source == timings["source_after"] and not expected_source["status"]

def snapshot():
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
        "files": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for path in expected_source["files"]},
        "compiler_sha256": hashlib.sha256(Path(rt._tessera_opt_path()).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(
            Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
    }

def check(outputs, references, budgets):
    result = []
    assert len(outputs) == 2 and not np.shares_memory(*outputs)
    for output, reference, budget in zip(outputs, references, budgets, strict=True):
        assert output.dtype == np.float32 and np.isfinite(output).all()
        error = np.abs(output.astype(np.float64) - reference)
        assert np.all(error <= budget)
        result.append({"max_abs_error": float(error.max()),
                       "max_error_over_budget": float((error / budget).max())})
    return result

source_before = snapshot()
assert source_before == expected_source, "capture must use the exact measured source/tools"
assert rt._nvidia_device_name() == "sm_120"
probe = subprocess.check_output([
    "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=uuid,compute_cap", "--format=csv,noheader"
], text=True).strip().split(",")
assert len(probe) == 2 and probe[0].strip() == timings["device_uuid"] and probe[1].strip() == "12.0"
options.output.mkdir(parents=True, exist_ok=True)
rows = []
for measured in timings["rows"]:
    storage, shape, permuted = measured["storage"], measured["shape_mnk"], measured["permuted"]
    a, b, da = inputs({"storage": storage, "shape_mnk": shape})
    host_roots = (b, a, da) if permuted else (a, b, da)
    references, budgets = oracle(a, b, da)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        host = fn.native_jvp(*host_roots[:2], tangents=(da,))
        host_errors = check(host, references, budgets)
        artifact = fn.native_jvp_runtime_artifact()
        geometry = [member["abi"]["block_size"]
                    for member in artifact.metadata["native_program"]["members"]]
        assert geometry == [128, 32, 32], f"portable/native launch geometry mismatch: {geometry}"
        assert artifact.artifact_hash == measured["artifact_hash"], "artifact differs from measured package"
        images = [member["image_digest"] for member in artifact.metadata["native_program"]["members"]]
        assert images == measured["image_digests"], "native images differ from measured images"
        serialized = artifact.to_json()
        restored = rt.RuntimeArtifact.from_json(serialized)
        fn.close_native_storage()  # Force standalone owner preparation during replay.
        with ExitStack() as stack:
            sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in host_roots]
            resident = tuple(session.upload(value) for session, value in
                             zip(sessions, host_roots, strict=True))
            with patch("subprocess.run", side_effect=AssertionError("replay invoked a compiler/subprocess")):
                result = rt.launch(restored, resident)
            assert result["ok"], result
            assert result["execution_kind"] == "native_gpu"
            assert all(receipt["input_binding"] == "cuda_resident_roots" and receipt["root_uploads"] == 0
                       for receipt in result["component_receipts"])
            replay_errors = check(result["output"], references, budgets)
            for expected, actual in zip(host, result["output"], strict=True):
                np.testing.assert_array_equal(expected, actual)
        m, n, k = shape
        name = f"{storage}-m{m}-n{n}-k{k}.json.gz"
        (options.output / name).write_bytes(gzip.compress(serialized.encode(), mtime=0))
        rows.append({
            "storage": storage, "shape_mnk": shape, "permuted": permuted, "artifact": name,
            "artifact_hash": artifact.artifact_hash, "image_digests": images,
            "host_errors": host_errors, "standalone_resident_errors": replay_errors,
            "bitwise_host_resident_equal": True, "compiler_subprocesses_during_replay": 0,
            "result_ownership": "two independent completed host f32 arrays",
        })
    finally:
        fn.close_native_storage()
source_after = snapshot()
assert source_before == source_after
(options.output / "index.json").write_text(json.dumps({
    "device_uuid": timings["device_uuid"], "compute_capability": "12.0",
    "source_before": source_before, "source_after": source_after,
    "scope": "same-package independent numerical and standalone resident replay; no timing claim",
    "rows": rows,
}, indent=2) + "\n")
print(json.dumps({"captured_programs": len(rows), "source": source_before["commit"]}))
