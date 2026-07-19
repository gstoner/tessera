"""Record canonical NVIDIA-E2E-2 reduction timing and resource evidence."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_reduction.json"
SHAPES = ((256, 1024), (127, 259))


def _module(shape: tuple[int, int], storage: str, kind: str):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    rows, columns = shape
    ir = "f16" if storage == "fp16" else "f32"
    x = IRType(f"tensor<{rows}x{columns}x{ir}>", (str(rows), str(columns)), storage)
    out = IRType(f"tensor<{rows}xf32>", (str(rows),), "fp32")
    op_name = "tessera.reduce" if kind == "sum" else f"tessera.{kind}"
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"e2e_reduce_{kind}_{ir}",
                args=[IRArg("x", x)],
                result_types=[out],
                body=[
                    IROp(
                        result="o",
                        op_name=op_name,
                        operands=["%x"],
                        operand_types=[str(x)],
                        result_type=str(out),
                        kwargs={"axis": -1, "keepdims": False},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _delta(a: float, b: float) -> float:
    return abs(a - b) / min(a, b) if min(a, b) else 0.0


def record(samples: int, device_reps: int, e2e_reps: int, warmup: int) -> dict:
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module

    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bridge = rt._load_nvidia_ptx_launch()
    if bridge is None:
        raise RuntimeError("libtessera_nvidia_ptx_launch.so is unavailable")
    rows = []
    for storage in ("fp16", "fp32"):
        for kind in ("sum", "mean", "max"):
            for shape in SHAPES:
                module = _module(shape, storage, kind)
                nvidia_native._cache.clear()
                started = time.perf_counter()
                bundle = compile_graph_module(
                    module,
                    source_origin="NVIDIA-E2E-2",
                    target="nvidia_sm120",
                    options={"package_native": True},
                    enable_tool_validation=False,
                )
                cold_ms = (time.perf_counter() - started) * 1e3
                started = time.perf_counter()
                warm = compile_graph_module(
                    module,
                    source_origin="NVIDIA-E2E-2",
                    target="nvidia_sm120",
                    options={"package_native": True},
                    enable_tool_validation=False,
                )
                warm_ms = (time.perf_counter() - started) * 1e3
                assert bundle.native_image and bundle.launch_descriptor and warm.native_image
                assert bundle.native_image.image_digest == warm.native_image.image_digest
                artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
                rng = np.random.default_rng(121_000 + sum(shape))
                dtype = np.float16 if storage == "fp16" else np.float32
                x = np.ascontiguousarray((rng.standard_normal(shape) * 1.25).astype(dtype))
                output = np.zeros((shape[0],), np.float32)
                bindings = {
                    "x": x,
                    "o": output,
                    "Outer": shape[0],
                    "AxisExtent": shape[1],
                    "Inner": 1,
                }
                smoke = rt.launch(artifact, bindings)
                if not smoke["ok"]:
                    raise RuntimeError(str(smoke.get("reason")))
                reference = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind](
                    x.astype(np.float32),
                    axis=-1,
                )
                np.testing.assert_allclose(output, reference, rtol=0, atol=3e-3)
                entry = bundle.launch_descriptor.entry_symbol.encode()
                raw = (ctypes.c_void_p * 2)(int(x.ctypes.data), int(output.ctypes.data))
                dims = (ctypes.c_int64 * 3)(shape[0], shape[1], 1)
                cohorts = [{"device": [], "e2e": []}, {"device": [], "e2e": []}]
                for sample in range(samples):
                    for cohort in (0, 1) if sample % 2 == 0 else (1, 0):
                        latency = ctypes.c_float()
                        rc = bridge.tessera_nvidia_ptx_benchmark(
                            entry,
                            raw,
                            2,
                            dims,
                            3,
                            warmup,
                            device_reps,
                            ctypes.byref(latency),
                        )
                        if rc:
                            raise RuntimeError(f"device benchmark returned rc={rc}")
                        cohorts[cohort]["device"].append(float(latency.value))
                        started = time.perf_counter()
                        for _ in range(e2e_reps):
                            result = rt.launch(artifact, bindings)
                            if not result["ok"]:
                                raise RuntimeError(str(result.get("reason")))
                        cohorts[cohort]["e2e"].append((time.perf_counter() - started) * 1e3 / e2e_reps)
                runs = [
                    {
                        "device_event_ms": statistics.median(item["device"]),
                        "end_to_end_ms": statistics.median(item["e2e"]),
                        "device_samples_ms": item["device"],
                        "end_to_end_samples_ms": item["e2e"],
                    }
                    for item in cohorts
                ]
                resources = bundle.native_image.resource_record
                rows.append(
                    {
                        "storage": storage,
                        "kind": kind,
                        "shape": list(shape),
                        "entry": bundle.launch_descriptor.entry_symbol,
                        "abi_id": bundle.launch_descriptor.abi_id,
                        "image_digest": bundle.native_image.image_digest,
                        "compile": {
                            "cold_ms": cold_ms,
                            "warm_ms": warm_ms,
                            "cold_state": bundle.native_image.compile_state,
                            "warm_state": warm.native_image.compile_state,
                        },
                        "resources": resources.to_dict() if resources else None,
                        "runs": runs,
                        "stability": {
                            "device_fraction": _delta(runs[0]["device_event_ms"], runs[1]["device_event_ms"]),
                            "end_to_end_fraction": _delta(runs[0]["end_to_end_ms"], runs[1]["end_to_end_ms"]),
                            "policy_fraction": 0.04,
                        },
                        "selector_changed": False,
                    }
                )
    return {
        "schema": "tessera.nvidia.e2e-spine-reduction.v1",
        "work_item": "NVIDIA-E2E-2",
        "device": device,
        "method": {
            "runs": 2,
            "samples_per_run": samples,
            "device_repetitions_per_sample": device_reps,
            "end_to_end_repetitions_per_sample": e2e_reps,
            "warmup": warmup,
            "sampling": "sample_interleaved_disjoint_cohorts",
            "timing_domains": ["device_event", "end_to_end"],
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--e2e-reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = record(args.samples, args.device_reps, args.e2e_reps, args.warmup)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
