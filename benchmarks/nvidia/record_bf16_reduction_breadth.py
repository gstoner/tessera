"""Record exact-SM120 evidence for compiler-owned BF16 reduction breadth.

The packet measures serial and cooperative-128 Tile-image schedules in two
disjoint repeated-median cohorts.  Every sample discards its first launch, then
amortizes resident device-event work and allocation/copy-inclusive end-to-end
work independently.  It records numerical, resource, image, and cold/warm
cache evidence but never changes the production selector.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import ml_dtypes
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_bf16_reduction_breadth.json"
POLICY = 0.04


WORKLOADS = (
    ("sum", (1, 257), 1, False),
    ("mean", (3, 17, 5), 1, True),
    ("max", (2, 5, 19), 2, False),
    ("min", (5, 7, 3), 0, True),
    ("amax", (2, 9, 17), 1, True),
    ("amin", (2, 9, 17), 1, False),
)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _delta(a: float, b: float) -> float:
    return abs(a - b) / min(a, b) if min(a, b) else 0.0


def _module(kind: str, shape: tuple[int, ...], axis: int, keepdims: bool):
    from tessera.compiler.graph_ir import (
        GraphIRFunction,
        GraphIRModule,
        IRArg,
        IROp,
        IRType,
    )

    input_dims = "x".join(map(str, shape))
    output_shape = list(shape)
    if keepdims:
        output_shape[axis] = 1
    else:
        output_shape.pop(axis)
    output_dims = "x".join(map(str, output_shape))
    x_type = IRType(
        f"tensor<{input_dims}xbf16>", tuple(map(str, shape)), "bf16"
    )
    output_type = IRType(
        f"tensor<{output_dims}xf32>", tuple(map(str, output_shape)), "fp32"
    )
    op_name = "tessera.reduce" if kind == "sum" else f"tessera.{kind}"
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"bf16_{kind}_{axis}_{int(keepdims)}",
                args=[IRArg("x", x_type)],
                result_types=[output_type],
                body=[
                    IROp(
                        result="o",
                        op_name=op_name,
                        operands=["%x"],
                        operand_types=[str(x_type)],
                        result_type=str(output_type),
                        kwargs={"axis": axis, "keepdims": keepdims},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _normalized_kind(kind: str) -> str:
    return {"amax": "max", "amin": "min"}.get(kind, kind)


def _reference(x: np.ndarray, kind: str, axis: int, keepdims: bool) -> np.ndarray:
    x32 = x.astype(np.float32)
    fn = {
        "sum": np.sum,
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "amax": np.amax,
        "amin": np.amin,
    }[kind]
    return np.asarray(fn(x32, axis=axis, keepdims=keepdims), dtype=np.float32)


def _paired_runs(
    candidates: dict[str, dict[str, object]],
    *,
    samples: int,
    device_reps: int,
    end_to_end_reps: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[str]], list[float]]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    names = tuple(candidates)
    device = [{name: [] for name in names} for _ in range(2)]
    end_to_end = [{name: [] for name in names} for _ in range(2)]
    conditioning = []
    for sample in range(samples):
        conditioning.append(condition_sm120())
        cohort_order = (0, 1) if sample % 2 == 0 else (1, 0)
        candidate_order = names if sample % 2 == 0 else names[::-1]
        for cohort in cohort_order:
            for name in candidate_order:
                device_fn = candidates[name]["device"]
                end_to_end_fn = candidates[name]["end_to_end"]
                device_fn(1)  # Deliberately discard first resident launch.
                device[cohort][name].append(device_fn(device_reps))
                end_to_end_fn()  # Deliberately discard first complete call.
                start = time.perf_counter()
                for _ in range(end_to_end_reps):
                    end_to_end_fn()
                end_to_end[cohort][name].append(
                    (time.perf_counter() - start) * 1e3 / end_to_end_reps
                )
    runs = {
        name: [
            {
                "run": cohort + 1,
                "device_event_ms": statistics.median(device[cohort][name]),
                "end_to_end_ms": statistics.median(end_to_end[cohort][name]),
                "device_samples_ms": device[cohort][name],
                "end_to_end_samples_ms": end_to_end[cohort][name],
            }
            for cohort in range(2)
        ]
        for name in names
    }
    winners = {
        domain: [
            min(names, key=lambda name: runs[name][cohort][domain])
            for cohort in range(2)
        ]
        for domain in ("device_event_ms", "end_to_end_ms")
    }
    return runs, winners, conditioning


def record(
    samples: int, device_reps: int, end_to_end_reps: int, warmup: int
) -> dict[str, object]:
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module

    device = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bridge = rt._load_nvidia_ptx_launch()
    rows: list[dict[str, object]] = []
    for workload_index, (kind, shape, axis, keepdims) in enumerate(WORKLOADS):
        module = _module(kind, shape, axis, keepdims)
        rng = np.random.default_rng(120_160 + workload_index)
        x = np.ascontiguousarray(
            (rng.standard_normal(shape) * 0.4).astype(ml_dtypes.bfloat16)
        )
        expected = _reference(x, kind, axis, keepdims)
        outer = int(np.prod(shape[:axis], dtype=np.int64))
        extent = shape[axis]
        inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        candidates: dict[str, dict[str, object]] = {}
        for schedule in ("serial", "cooperative_128"):
            nvidia_native._cache.clear()
            started = time.perf_counter()
            bundle = compile_graph_module(
                module,
                source_origin="NVIDIA-BF16-CANONICAL-BREADTH",
                target="nvidia_sm120",
                options={
                    "package_native": True,
                    "nvidia_reduction_schedule": schedule,
                },
                enable_tool_validation=False,
            )
            cold_ms = (time.perf_counter() - started) * 1e3
            started = time.perf_counter()
            warm_bundle = compile_graph_module(
                module,
                source_origin="NVIDIA-BF16-CANONICAL-BREADTH",
                target="nvidia_sm120",
                options={
                    "package_native": True,
                    "nvidia_reduction_schedule": schedule,
                },
                enable_tool_validation=False,
            )
            warm_ms = (time.perf_counter() - started) * 1e3
            assert bundle.native_image and bundle.launch_descriptor
            assert warm_bundle.native_image and warm_bundle.launch_descriptor
            artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
            output = np.zeros(expected.shape, dtype=np.float32)
            bindings = {
                "x": x,
                "o": output,
                "Outer": outer,
                "AxisExtent": extent,
                "Inner": inner,
            }
            launch = rt.launch(artifact, bindings)
            if not launch["ok"]:
                raise RuntimeError(launch.get("reason"))
            np.testing.assert_allclose(
                output, expected, rtol=2e-2, atol=2e-2, equal_nan=True
            )
            raw = (ctypes.c_void_p * 2)(int(x.ctypes.data), int(output.ctypes.data))
            dims = (ctypes.c_int64 * 3)(outer, extent, inner)

            def device_event(
                repetitions: int,
                *,
                compiled=bundle,
                buffers=raw,
                launch_dims=dims,
            ) -> float:
                latency = ctypes.c_float()
                rc = bridge.tessera_nvidia_ptx_benchmark(
                    compiled.launch_descriptor.entry_symbol.encode(),
                    buffers,
                    2,
                    launch_dims,
                    3,
                    warmup,
                    repetitions,
                    ctypes.byref(latency),
                )
                if rc:
                    raise RuntimeError(f"BF16 reduction benchmark failed: rc={rc}")
                return float(latency.value)

            def end_to_end(
                *, runtime_artifact=artifact, runtime_bindings=bindings
            ) -> None:
                result = rt.launch(runtime_artifact, runtime_bindings)
                if not result["ok"]:
                    raise RuntimeError(result.get("reason"))

            assembly_resources = (
                bundle.native_image.resource_record.to_dict()
                if bundle.native_image.resource_record
                else None
            )
            resources = {
                "assembly": assembly_resources,
                "live_driver": rt._nvidia_native_descriptor_resources(
                    bundle.native_image,
                    bundle.launch_descriptor,
                    block_size=128,
                ),
            }
            candidates[f"canonical_{schedule}"] = {
                "device": device_event,
                "end_to_end": end_to_end,
                "resources": resources,
                "resource_fingerprint": _fingerprint(resources),
                "image_digest": bundle.native_image.image_digest,
                "compile": {
                    "cold_ms": cold_ms,
                    "warm_ms": warm_ms,
                    "cold_state": bundle.native_image.compile_state,
                    "warm_state": warm_bundle.native_image.compile_state,
                    "image_digest_reproducible": (
                        bundle.native_image.image_digest
                        == warm_bundle.native_image.image_digest
                    ),
                    "entry_symbol_reproducible": (
                        bundle.launch_descriptor.entry_symbol
                        == warm_bundle.launch_descriptor.entry_symbol
                    ),
                },
                "numerical": {
                    "max_abs_error": float(
                        np.nanmax(np.abs(output.astype(np.float64) - expected))
                    ),
                    "rtol": 2e-2,
                    "atol": 2e-2,
                    "passed": True,
                },
            }
        measured, winners, conditioning = _paired_runs(
            candidates,
            samples=samples,
            device_reps=device_reps,
            end_to_end_reps=end_to_end_reps,
        )
        for name, candidate in candidates.items():
            stability = {
                domain: _delta(measured[name][0][domain], measured[name][1][domain])
                for domain in ("device_event_ms", "end_to_end_ms")
            }
            consensus = all(winners[domain] == [name, name] for domain in winners)
            stable = all(value <= POLICY for value in stability.values())
            rows.append(
                {
                    "op": "reduction",
                    "storage": "bf16",
                    "accumulation": "fp32",
                    "output": "fp32",
                    "kind": kind,
                    "normalized_kind": _normalized_kind(kind),
                    "shape": list(shape),
                    "axis": axis,
                    "keepdims": keepdims,
                    "candidate": name,
                    "runs": measured[name],
                    "stability_fraction": stability,
                    "stable": stable,
                    "winner_consensus": consensus,
                    "selector_eligible": stable and consensus,
                    "resources": candidate["resources"],
                    "resource_fingerprint": candidate["resource_fingerprint"],
                    "image_digest": candidate["image_digest"],
                    "compile": candidate["compile"],
                    "numerical": candidate["numerical"],
                    "clock_conditioning_ms": conditioning,
                    "selector_changed": False,
                }
            )
    return {
        "schema": "tessera.nvidia.bf16-reduction-breadth.v1",
        "work_item": "NVIDIA-BF16-CANONICAL-BREADTH",
        "device": device,
        "stability_policy": {
            "relative_fraction": POLICY,
            "runs": 2,
            "required_domains": ["device_event_ms", "end_to_end_ms"],
        },
        "method": {
            "samples_per_run": samples,
            "device_repetitions_per_sample": device_reps,
            "end_to_end_repetitions_per_sample": end_to_end_reps,
            "discarded_launches_per_candidate_sample": 1,
            "sampling": "two_disjoint_time_interleaved_cohorts",
            "candidate_order": "rotated_interleaved",
            "clock_conditioning": "sm120 resident 1024^3 TF32 GEMM before every paired cohort sample",
            "timing_domains": {
                "device_event_ms": "resident CUDA-event interval",
                "end_to_end_ms": "runtime.launch including allocation/copies",
            },
        },
        "contract": {
            "storage": "bf16",
            "accumulation": "fp32",
            "output": "fp32",
            "schedules": ["serial", "cooperative_128"],
            "kinds": ["sum", "mean", "max", "min", "amax", "amin"],
            "arbitrary_axes": True,
            "keepdims": [False, True],
            "ragged_boundaries": True,
            "nonfinite_proof": "tests/device/nvidia/test_e2e_spine_native.py",
        },
        "rows": rows,
        "selector_promotions": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--end-to-end-reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = record(
        args.samples, args.device_reps, args.end_to_end_reps, args.warmup
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(payload['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
