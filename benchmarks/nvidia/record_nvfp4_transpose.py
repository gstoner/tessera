#!/usr/bin/env python3
"""Matched-value native NVFP4 orientation characterization on SM120."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tests.device.nvidia.test_nvfp4_transpose_jit import CASES, oriented_product, oriented_inputs

ROOT = Path(__file__).resolve().parents[2]


def record(mode, ta, tb, rows, n, k, samples):
    values, oracle = oriented_inputs(mode, ta, tb, rows, n, k)
    call = ts.jit(oriented_product(ta, tb, None if mode == "rank_two" else mode), target="nvidia_sm120")
    start = time.perf_counter()
    result = call(*values)
    cold = (time.perf_counter() - start) * 1e3
    np.testing.assert_allclose(result, oracle, rtol=0, atol=2e-3)
    assert call._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    wall = []
    for _ in range(samples):
        start = time.perf_counter()
        result = call(*values)
        wall.append((time.perf_counter() - start) * 1e3)
        np.testing.assert_allclose(result, oracle, rtol=0, atol=2e-3)
    artifact = call._cached_artifact
    descriptor, image = artifact.launch_descriptor, artifact.native_image
    inputs = sorted((item for item in descriptor.buffers if item.direction == "input"), key=lambda item: item.ordinal)
    output = next(item for item in descriptor.buffers if item.direction == "output")
    arrays = (values[0].storage, values[1].storage, values[2], values[3])
    buffers = dict(zip((item.name for item in inputs), arrays, strict=True))
    buffers[output.name] = np.empty_like(result)
    dims = list(descriptor.provenance["shape"])
    if len(descriptor.scalars) == 5:
        dims.extend((rows, 3))
    scalars = dict(zip((item.name for item in sorted(descriptor.scalars, key=lambda item: item.ordinal)), dims, strict=True))
    arguments = {"buffers": buffers, "scalars": scalars}
    events = [rt._nvidia_native_descriptor_device_latency(image, descriptor, arguments, reps=100, warmup=20) for _ in range(samples)]
    receipt = rt.launch(artifact, arguments)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu"
    np.testing.assert_allclose(buffers[output.name], oracle, rtol=0, atol=2e-3)
    return {"mode": mode, "transposeA": ta, "transposeB": tb,
        "shape_bmnk": [1 if mode == "rank_two" else 3, rows, n, k],
        "logical_a_shape": values[0].shape, "logical_b_shape": values[1].shape,
        "physical_shapes": [value.shape for value in arrays],
        "oracle_sha256": hashlib.sha256(oracle.tobytes()).hexdigest(),
        "maximum_absolute_error": float(np.max(np.abs(result.astype(np.float64) - oracle))),
        "cold_public_ms": cold, "warm_public_ms": wall,
        "warm_public_median_ms": statistics.median(wall),
        "resident_device_event_ms": events, "resident_device_event_median_ms": statistics.median(events),
        "entry_symbol": descriptor.entry_symbol, "abi_id": descriptor.abi_id,
        "schedule_digest": descriptor.provenance["schedule_digest"],
        "tile_ir_digest": descriptor.provenance["tile_ir_digest"],
        "image_sha256": hashlib.sha256(image.payload).hexdigest(),
        "resources": rt._nvidia_native_descriptor_resources(image, descriptor, block_size=32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("at least three samples required")
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or not device.endswith("12.0"):
        raise RuntimeError("recorder requires one visible SM120 device")
    sources = ("python/tessera/compiler/graph_ir.py", "python/tessera/compiler/scheduled_matmul.py",
        "python/tessera/compiler/nvidia_native.py", "python/tessera/compiler/jit.py", "python/tessera/compiler/native_vmap.py",
        "python/tessera/runtime.py", "src/compiler/ir/TesseraOps.cpp", "src/compiler/ir/TileOps.cpp", "src/compiler/programming_model/ir/ScheduleDialect.cpp", "src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "tests/device/nvidia/test_nvfp4_transpose_jit.py", "benchmarks/nvidia/record_nvfp4_transpose.py")
    tools = (os.environ["TESSERA_OPT"], os.environ["TESSERA_NVIDIA_OPT"], os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"])
    packet = {"schema": "tessera.nvfp4.native_orientation.v1", "work_item": "W1.1",
        "sync_key": "NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06", "architecture": "sm_120a", "device": device,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT, text=True).strip()),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "tool_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in tools},
        "route": "Python frontend -> Graph MLIR -> Schedule -> Tile -> NVIDIA Target -> PTX -> checked ABI",
        "selector_changed": False, "strategy_promotion": False,
        "correctness": "independent decoded fp64 oracle before timing, each public sample and final portable launch",
        "timing_domains": {"cold_public": "trace/compile/package/synchronized host-buffer launch wall",
            "warm_public": "cached package validation, output allocation, copies and synchronized launch wall",
            "device_event": "resident native 100-launch CUDA-event window; upload/readback excluded; dispatch gaps included"},
        "rows": [record(mode, ta, tb, m, n, k, args.samples) for m, n, k in ((17, 19, 129), (128, 128, 256)) for mode, ta, tb in CASES]}
    groups = {}
    for row in packet["rows"]:
        key = (row["mode"], tuple(row["shape_bmnk"]))
        previous = groups.setdefault(key, row["oracle_sha256"])
        assert previous == row["oracle_sha256"], "orientation arms must use identical quantized values"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
