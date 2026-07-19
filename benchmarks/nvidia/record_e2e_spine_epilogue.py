"""Record canonical SM120 fused-epilogue descriptor timing/resources."""

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
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_epilogue.json"
SHAPES = ((256, 256, 256), (127, 259, 63))


def _module(storage: str, activation: str, shape: tuple[int, int, int]):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    m, n, k = shape
    ir = "f16" if storage == "fp16" else "bf16"
    a = IRType(f"tensor<{m}x{k}x{ir}>", (str(m), str(k)), storage)
    b = IRType(f"tensor<{k}x{n}x{ir}>", (str(k), str(n)), storage)
    bias = IRType(f"tensor<{n}xf32>", (str(n),), "fp32")
    residual = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"e2e_epilogue_{ir}_{activation}",
                args=[IRArg("a", a), IRArg("b", b), IRArg("bias", bias), IRArg("residual", residual)],
                result_types=[out],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(out),
                        kwargs={"bias": "%bias", "activation": activation, "residual": "%residual"},
                    )
                ],
                return_values=["%c"],
            )
        ]
    )


def _delta(a: float, b: float) -> float:
    return abs(a - b) / min(a, b) if min(a, b) else 0.0


def record(samples: int, device_reps: int, e2e_reps: int, warmup: int) -> dict:
    import ml_dtypes
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
        raise RuntimeError("NVIDIA PTX bridge unavailable")
    rows = []
    for storage in ("fp16", "bf16"):
        for activation in ("none", "relu", "gelu", "silu"):
            for shape in SHAPES:
                module = _module(storage, activation, shape)
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
                m, n, k = shape
                rng = np.random.default_rng(121_200 + sum(shape))
                dtype = np.float16 if storage == "fp16" else ml_dtypes.bfloat16
                arrays = {
                    "a": np.ascontiguousarray((rng.standard_normal((m, k)) * 0.15).astype(dtype)),
                    "b": np.asfortranarray((rng.standard_normal((k, n)) * 0.15).astype(dtype)),
                    "bias": np.ascontiguousarray((rng.standard_normal(n) * 0.05).astype(np.float32)),
                    "residual": np.ascontiguousarray((rng.standard_normal((m, n)) * 0.05).astype(np.float32)),
                    "c": np.zeros((m, n), np.float32),
                }
                artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
                bindings = {**arrays, "M": m, "N": n, "K": k}
                if not rt.launch(artifact, bindings)["ok"]:
                    raise RuntimeError("descriptor smoke failed")
                ordered = sorted(bundle.launch_descriptor.buffers, key=lambda item: item.ordinal)
                raw = (ctypes.c_void_p * len(ordered))(*(int(arrays[item.name].ctypes.data) for item in ordered))
                dims = (ctypes.c_int64 * 3)(m, n, k)
                cohorts = [{"device": [], "e2e": []}, {"device": [], "e2e": []}]
                for sample in range(samples):
                    for cohort in (0, 1) if sample % 2 == 0 else (1, 0):
                        latency = ctypes.c_float()
                        rc = bridge.tessera_nvidia_ptx_benchmark(
                            bundle.launch_descriptor.entry_symbol.encode(),
                            raw,
                            len(ordered),
                            dims,
                            3,
                            warmup,
                            device_reps,
                            ctypes.byref(latency),
                        )
                        if rc:
                            raise RuntimeError(f"device benchmark rc={rc}")
                        cohorts[cohort]["device"].append(float(latency.value))
                        started = time.perf_counter()
                        for _ in range(e2e_reps):
                            if not rt.launch(artifact, bindings)["ok"]:
                                raise RuntimeError("descriptor launch failed")
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
                resource = bundle.native_image.resource_record
                rows.append(
                    {
                        "storage": storage,
                        "activation": activation,
                        "bias": True,
                        "residual": True,
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
                        "resources": resource.to_dict() if resource else None,
                        "runs": runs,
                        "stability": {
                            "device_fraction": _delta(runs[0]["device_event_ms"], runs[1]["device_event_ms"]),
                            "end_to_end_fraction": _delta(runs[0]["end_to_end_ms"], runs[1]["end_to_end_ms"]),
                            "policy_fraction": 0.03,
                        },
                        "selector_changed": False,
                    }
                )
    return {
        "schema": "tessera.nvidia.e2e-spine-epilogue.v1",
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
    parser.add_argument("--device-reps", type=int, default=50)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(record(args.samples, args.device_reps, args.e2e_reps, args.warmup), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
