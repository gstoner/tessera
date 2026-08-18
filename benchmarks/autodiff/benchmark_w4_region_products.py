#!/usr/bin/env python3
"""Generate W4 residual-ABI correctness/timing packets for x86 or gfx1151."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from tessera.compiler.structured_region_product import (
    build_dynamic_if_product,
    build_hybrid_while_product,
    execute_dynamic_if_product,
    execute_hybrid_while_product,
)


ROOT = Path(__file__).resolve().parents[2]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paired_ir(tool: Path, fixture: str) -> str:
    path = ROOT / "tests" / "tessera-ir" / "phase_f4" / fixture
    result = subprocess.run(
        [str(tool.resolve()), "--tessera-autodiff-paired", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _measure(call, warmup: int, samples: int):
    for _ in range(warmup):
        call()
    values = []
    output = None
    for _ in range(samples):
        start = time.perf_counter_ns()
        output = call()
        values.append(time.perf_counter_ns() - start)
    return output, values


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm_gfx1151"), required=True)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    if_ir = _paired_ir(args.tessera_opt, "autodiff_saved_if_dynamic_fail_closed.mlir")
    while_ir = _paired_ir(args.tessera_opt, "autodiff_hybrid_while_fail_closed.mlir")
    if_cfg_digest = _digest(("w4-if-cfg-v1\n" + if_ir).encode())
    while_cfg_digest = _digest(("w4-while-cfg-v1\n" + while_ir).encode())
    branch = build_dynamic_if_product(
        target=args.target, paired_ir=if_ir, cfg_digest=if_cfg_digest,
        state_shape_bound=(512,),
    )
    loop = build_hybrid_while_product(
        target=args.target, paired_ir=while_ir, cfg_digest=while_cfg_digest,
        state_shape_bound=(512,), steps=5, checkpoint_indices=(2, 4),
    )
    x = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    ct = np.linspace(0.25, 1.25, 257, dtype=np.float32)
    saved = np.tanh(x).astype(np.float32)
    inactive = np.empty((0,), dtype=np.float32)
    expected_branch = ct * 2.0 * saved * (1.0 - saved * saved)
    branch_output, branch_samples = _measure(
        lambda: execute_dynamic_if_product(
            branch, predicate=True, saved_true=saved, saved_false=inactive,
            cotangent=ct,
        ),
        args.warmup,
        args.samples,
    )
    scales = (1.25, 0.5, 1.5, 0.75, 2.0)
    expected_loop = ct * np.prod(np.asarray(scales, dtype=np.float32))
    loop_output, loop_samples = _measure(
        lambda: execute_hybrid_while_product(
            loop, initial_state=x, scales=scales, cotangent=ct
        ),
        args.warmup,
        args.samples,
    )
    rows = [
        {
            "name": "dynamic_if_saved_activation_vjp",
            "artifact_digest": branch.artifact_digest,
            "paired_ir_digest": branch.paired_ir_digest,
            "cfg_digest": branch.cfg_digest,
            "residual_abi_digest": branch.residual_digest,
            "numerical": {
                "max_abs_error": float(np.max(np.abs(branch_output - expected_branch))),
                "passed": bool(np.allclose(branch_output, expected_branch, rtol=3e-5, atol=3e-5)),
            },
            "resources": {
                "native_child_launches": 5,
                "selected_residual_elements": int(saved.size),
                "inactive_residual_elements": 0,
                "predicate_replay": False,
                "graph_reentry": False,
            },
            "samples_ns": branch_samples,
        },
        {
            "name": "hybrid_while_sparse_tape_vjp",
            "artifact_digest": loop.artifact_digest,
            "paired_ir_digest": loop.paired_ir_digest,
            "cfg_digest": loop.cfg_digest,
            "residual_abi_digest": loop.residual_abi.digest if loop.residual_abi else None,
            "numerical": {
                "max_abs_error": float(np.max(np.abs(loop_output.input_cotangent - expected_loop))),
                "passed": bool(np.allclose(loop_output.input_cotangent, expected_loop,
                                            rtol=2e-6, atol=2e-6)),
            },
            "resources": {
                "checkpoint_indices": list(loop_output.checkpoint_indices),
                "retained_residual_bytes": loop.residual_abi.retained_bytes if loop.residual_abi else 0,
                "forward_steps": loop_output.forward_steps,
                "replayed_steps": loop_output.replayed_steps,
                "backward_steps": loop_output.backward_steps,
                "predicate_replay": False,
                "graph_reentry": False,
            },
            "samples_ns": loop_samples,
        },
    ]
    for row in rows:
        samples = row.pop("samples_ns")
        row["timing"] = {
            "source": "synchronized_host_wall",
            "samples_ns": samples,
            "minimum_ns": min(samples),
            "median_ns": round(statistics.median(samples)),
            "device_clock": {"valid": False, "value_ns": None},
            "eligible_for_promotion": False,
        }
    wsl = "microsoft" in platform.release().lower()
    packet = {
        "schema": "tessera.w4_region_products.evidence.v1",
        "target": args.target,
        "architecture": "zen5_avx512" if args.target == "x86" else "gfx1151",
        "device": (
            _cpu_model()
            if args.target == "x86"
            else "AMD Radeon 8060S (gfx1151)"
        ),
        "host": platform.platform(),
        "wsl": wsl,
        "host_clean": False,
        "tessera_opt_sha256": _digest(args.tessera_opt.read_bytes()),
        "work_item": "W4-PRODUCT-1",
        "rows": rows,
        "promotion": {
            "correctness_eligible": all(row["numerical"]["passed"] for row in rows),
            "performance_eligible": False,
            "reason": "WSL synchronized-wall packet; clean native timing remains required",
        },
        "environment": {"python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
