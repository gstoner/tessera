#!/usr/bin/env python3
"""Diagnostic source-matched gfx1151 matmul image reuse and numerical packet."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_matmul
from tests.unit.test_scheduled_matmul_consumers import _module


SHAPES = ((16, 16, 16), (32, 16, 16), (48, 16, 16))
ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def record(samples: int = 31) -> dict[str, object]:
    if samples < 3:
        raise ValueError("at least three launch samples are required")
    live = rt._rocm_live_arch()
    configured = rt._rocm_chip()
    if live != configured or live != "gfx1151":
        raise RuntimeError(f"gfx1151 source-matched packet requires live/configured gfx1151, got {live}/{configured}")
    tool = rocm_native._tessera_opt()
    if tool is None:
        raise RuntimeError("a rebuilt ROCm tessera-opt is required")
    tool_path = Path(tool).resolve()
    source_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    rng = np.random.default_rng(885)
    rows = []
    scheduled_artifacts = []
    digests = []
    for shape in SHAPES:
        m, k, n = shape
        scheduled = scheduled_matmul.lower_scheduled_matmul(
            _module(target="rocm", shape=shape), target="rocm_gfx1151"
        )
        scheduled_artifacts.append(scheduled)
        start = time.perf_counter_ns()
        package = rocm_native.package_scheduled_matmul(
            scheduled, pipeline_name="tessera-lower-to-rocm"
        )
        package_ms = (time.perf_counter_ns() - start) / 1e6
        a = (rng.normal(size=(m, k)) * 0.2).astype(np.float16)
        b = (rng.normal(size=(k, n)) * 0.2).astype(np.float16)
        output = np.zeros((m, n), np.float32)
        artifact = rt.RuntimeArtifact(
            metadata={"target": package.image.target},
            native_image=package.image,
            launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir,
            target_ir=package.target_ir,
        )
        arguments = {"buffers": {"a": a, "b": b, "o": output},
                     "scalars": {"M": m, "N": n, "K": k}}
        host_samples_ns = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            result = rt.launch(artifact, arguments)
            host_samples_ns.append(time.perf_counter_ns() - start)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"native gfx1151 launch failed: {result}")
        reference = a.astype(np.float32) @ b.astype(np.float32)
        max_error = float(np.max(np.abs(output - reference)))
        if max_error > 0.02:
            raise RuntimeError(f"matmul {shape} max absolute error {max_error} exceeds 0.02")
        if package.descriptor.entry_symbol != package.image.entry_points[0].symbol:
            raise RuntimeError("image symbol and checked descriptor differ")
        if "tessera.schedule_hash" in package.target_ir:
            raise RuntimeError("per-shape Schedule hash reached the binary Target directive")
        digests.append(package.image.image_digest)
        rows.append({
            "shape": list(shape),
            "schedule_digest": scheduled.schedule_digest,
            "tile_ir_digest": scheduled.tile_digest,
            "image_digest": package.image.image_digest,
            "entry_symbol": package.descriptor.entry_symbol,
            "compile_state": package.image.compile_state,
            "package_ms": package_ms,
            "host_launch_samples_ns": host_samples_ns,
            "host_launch_median_ms": statistics.median(host_samples_ns) / 1e6,
            "max_abs_error": max_error,
        })
    if len(set(digests)) != 1 or [row["compile_state"] for row in rows] != ["cold", "warm_cache", "warm_cache"]:
        raise RuntimeError("three shapes did not share exactly one compiled image")
    # Historical Tile-text key as a compile-only control. It is run from this
    # same source and compiler, but is not launched or used as a speedup claim.
    control = []
    for shape, scheduled in zip(SHAPES, scheduled_artifacts):
        start = time.perf_counter_ns()
        compiled = rocm_native._compile_scheduled_matmul_tile_ir(scheduled.tile_ir)
        control.append({
            "shape": list(shape),
            "compile_state": compiled[-1],
            "compile_ms": (time.perf_counter_ns() - start) / 1e6,
            "payload_sha256": hashlib.sha256(compiled[2]).hexdigest(),
        })
    if [row["compile_state"] for row in control] != ["cold"] * len(SHAPES):
        raise RuntimeError("historical Tile-text key did not compile each shape")
    return {
        "schema": "tessera.rocm_matmul_shape_key.v1",
        "architecture": live,
        "configured_architecture": configured,
        "source_commit": source_commit,
        "source_dirty": dirty,
        "compiler_path": str(tool_path),
        "compiler_sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
        "timing_domain": "synchronized_wsl_host_wall",
        "promotion_eligible": False,
        "samples_per_shape": samples,
        "rows": rows,
        "tile_text_control_compile_only": control,
    }


if __name__ == "__main__":
    print(json.dumps(record(), indent=2, sort_keys=True))
