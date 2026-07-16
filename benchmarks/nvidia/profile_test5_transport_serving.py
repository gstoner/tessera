"""Targeted Nsight launcher for TEST-5 MoE transport and paged-KV routes."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    from tessera.compiler.emit import nvidia_cuda as nv
    if artifact_file := os.environ.get("TESSERA_TEST5_PROFILE_ARTIFACTS"):
        artifacts = json.loads(Path(artifact_file).read_text())
        nv._moe_artifact = artifacts.get("moe")
        nv._resident_ops_artifact = artifacts.get("resident_ops")
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((257, 193)) * .1).astype(np.float32)
    tok = np.arange(257, dtype=np.int32)[::-1]
    packed = nv.run_moe_dispatch_f32(x, tok)
    nv.measure_moe_dispatch_device(x, tok, reps=1)
    weights = np.ones(257, np.float32)
    nv.measure_moe_combine_device(packed, tok, weights, 257, reps=1)
    groups = np.array([51, 0, 73, 61, 72], np.int64)
    ew = (rng.standard_normal((5, 193, 127)) * .1).astype(np.float32)
    nv.measure_grouped_gemm_device(x, ew, groups, reps=1)

    spec = importlib.util.spec_from_file_location(
        "serving_targeted", ROOT / "benchmarks/nvidia/benchmark_serving.py")
    assert spec and spec.loader
    serving = importlib.util.module_from_spec(spec); spec.loader.exec_module(serving)
    serving.run_benchmark([], tokens=16, chunk=4, slots=4,
                          kv_tokens=[512], heads=8, dim=64,
                          page_size=16, reps=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
