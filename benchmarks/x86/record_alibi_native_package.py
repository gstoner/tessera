#!/usr/bin/env python3
"""Numerical and host-wall package/launch record for the native x86 ALiBi route."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from tessera import runtime as rt
from tessera.compiler import x86_compile_cache, x86_native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType


def _module(heads: int, seq: int) -> GraphIRModule:
    slopes = IRType(f"tensor<{heads}xf32>", (str(heads),), "fp32")
    bias = IRType(f"tensor<{heads}x{seq}x{seq}xf32>",
                  (str(heads), str(seq), str(seq)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="physical_alibi", args=[IRArg("slopes", slopes)], result_types=[bias],
        body=[IROp(result="bias", op_name="tessera.alibi", operands=["%slopes"],
                   operand_types=[str(slopes)], result_type=str(bias),
                   kwargs={"num_heads": heads, "seq_len": seq})],
        return_values=["%bias"],
    )])


def _median_ms(samples: list[int]) -> float:
    return statistics.median(samples) / 1e6


def record(samples: int) -> dict:
    if samples < 3:
        raise ValueError("at least three paired samples required")
    library = x86_native._library_path()
    compiler = x86_native._tessera_opt()
    if library is None or compiler is None:
        raise RuntimeError("native x86 compiler and AVX-512 shared image required")
    cpu = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    cpu_fields = {
        key.strip(): value.strip()
        for line in cpu.splitlines() if ":" in line
        for key, value in (line.split(":", 1),)
    }
    flags = set(cpu_fields.get("flags", "").split())
    if not {"avx512f", "avx512bw"}.issubset(flags):
        raise RuntimeError("the selected host does not expose required AVX-512 features")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    rng = np.random.default_rng(20260929)
    rows = []
    for heads, seq in ((1, 1), (4, 7), (8, 33), (32, 128)):
        module = _module(heads, seq)
        x86_compile_cache.clear()
        start = time.perf_counter_ns()
        package = x86_native.package_cohort2(module, pipeline_name="tessera-lower-to-x86")
        cold_ns = time.perf_counter_ns() - start
        packages = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            repeat = x86_native.package_cohort2(module, pipeline_name="tessera-lower-to-x86")
            packages.append(time.perf_counter_ns() - start)
            if repeat.image.image_digest != package.image.image_digest:
                raise RuntimeError("warm package image identity changed")
        slopes = np.ascontiguousarray(rng.normal(size=heads), dtype=np.float32)
        output = np.empty((heads, seq, seq), dtype=np.float32)
        artifact = rt.RuntimeArtifact(
            metadata={"target": "x86"}, native_image=package.image,
            launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
            target_ir=package.target_ir,
        )
        args = {"slopes": slopes, "bias": output, "H": heads, "S": seq}
        launches = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            result = rt.launch(artifact, args)
            launches.append(time.perf_counter_ns() - start)
            if not result.get("ok") or result.get("execution_kind") != "native_cpu":
                raise RuntimeError(f"native ALiBi launch refused: {result}")
            if result.get("image_digest") != package.image.image_digest:
                raise RuntimeError("ALiBi launch image receipt differs from package")
        pos = np.arange(seq, dtype=np.float32)
        expected = slopes[:, None, None] * (pos[None, None, :] - pos[None, :, None])
        error = np.max(np.abs(output - expected))
        if error > 1e-5:
            raise RuntimeError(f"ALiBi numerical error {error}")
        rows.append({
            "heads": heads, "seq": seq,
            "image_digest": package.image.image_digest,
            "schedule_digest": package.descriptor.provenance["schedule_digest"],
            "cold_package_ms": cold_ns / 1e6,
            "warm_package_median_ms": _median_ms(packages),
            "launch_host_wall_median_ms": _median_ms(launches),
            "max_abs_error": float(error),
        })
    return {
        "schema": "tessera.x86_alibi_native_package.v1",
        "source_commit": source,
        "source_dirty": dirty,
        "host": platform.platform(),
        "cpu_model": cpu_fields.get("model name", "").strip(),
        "architecture": "zen5-avx512",
        "timing_domain": "synchronized_host_wall",
        "promotion_eligible": False,
        "samples_per_row": samples,
        "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = record(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
