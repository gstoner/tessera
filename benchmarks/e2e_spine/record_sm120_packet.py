#!/usr/bin/env python3
"""Record the shared softmax/reduction E2E-SPINE-3 packet on exact SM120."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def _module(family: str):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    shape = (2, 2) if family == "softmax" else (2, 3)
    source = IRType(
        f"tensor<{shape[0]}x{shape[1]}xf32>", tuple(map(str, shape)), "fp32",
    )
    result = source if family == "softmax" else IRType("tensor<2xf32>", ("2",), "fp32")
    operation = "tessera.softmax" if family == "softmax" else "tessera.sum"
    kwargs = {"axis": -1} if family == "softmax" else {"axis": -1, "keepdims": False}
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"fleet_{family}", args=[IRArg("x", source)], result_types=[result],
        body=[IROp(
            result="o", op_name=operation, operands=["%x"], operand_types=[str(source)],
            result_type=str(result), kwargs=kwargs,
        )], return_values=["%o"],
    )])


def _stability(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values) * 100.0


def _e2e_medians(call, *, samples: int, repetitions: int) -> list[float]:
    call()  # discard first-use registration and module-load effects
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            for _ in range(repetitions):
                result = call()
                if result.get("ok") is not True:
                    raise RuntimeError(str(result.get("reason")))
            cohorts[cohort].append((time.perf_counter_ns() - started) / repetitions)
    return [float(statistics.median(values)) for values in cohorts]


def _device_medians(
    bridge, *, entry: str, buffers, dims, samples: int, warmups: int, repetitions: int,
) -> list[float]:
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            latency = ctypes.c_float()
            rc = bridge.tessera_nvidia_ptx_benchmark(
                entry.encode(), buffers, 2, dims, len(dims), warmups,
                repetitions, ctypes.byref(latency),
            )
            if rc:
                raise RuntimeError(f"SM120 device benchmark returned {rc} for {entry}")
            cohorts[cohort].append(float(latency.value) * 1_000_000.0)
    return [float(statistics.median(values)) for values in cohorts]


def record(
    *, samples: int, device_repetitions: int, e2e_repetitions: int,
    warmups: int, stability_limit: float,
) -> tuple[dict, dict]:
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.e2e_fleet import load_fixture_corpus, validate_backend_report

    identity = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not identity.endswith("12.0"):
        raise RuntimeError(f"assigned E2E packet requires compute capability 12.0, found {identity}")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    bridge = rt._load_nvidia_ptx_launch()
    if bridge is None:
        raise RuntimeError("SM120 PTX launch/benchmark bridge is unavailable")
    corpus = load_fixture_corpus()
    definitions = (
        ("softmax", "softmax-f32-2x2-extreme-v1"),
        ("reduction", "reduce-sum-f32-axis1-v1"),
    )
    fixture_rows, cache_rows, benchmark_rows, resource_rows = [], [], [], []
    toolchains: set[str] = set()
    for family, fixture_id in definitions:
        module = _module(family)
        nvidia_native._cache.clear()
        cold = compile_graph_module(
            module, source_origin="E2E-SPINE-3", target="nvidia_sm120",
            options={"package_native": True}, enable_tool_validation=False,
        )
        warm = compile_graph_module(
            module, source_origin="E2E-SPINE-3", target="nvidia_sm120",
            options={"package_native": True}, enable_tool_validation=False,
        )
        if not cold.native_image or not cold.launch_descriptor or not warm.native_image or not warm.launch_descriptor:
            raise RuntimeError(f"{family} did not produce a native image and descriptor")
        image, descriptor = cold.native_image, cold.launch_descriptor
        if (
            image.cache_key != warm.native_image.cache_key
            or image.image_digest != warm.native_image.image_digest
            or descriptor.descriptor_digest != warm.launch_descriptor.descriptor_digest
        ):
            raise RuntimeError(f"{family} cold/warm SM120 compilation is not reproducible")
        toolchains.add(image.toolchain_fingerprint)
        artifact = compile_result_from_bundle(cold, module=module).to_runtime_artifact()
        x = np.array(corpus[fixture_id]["inputs"]["x"], dtype=np.float32)
        output = np.zeros_like(x) if family == "softmax" else np.zeros((2,), np.float32)
        bindings = (
            {"x": x, "o": output, "Rows": 2, "K": 2}
            if family == "softmax"
            else {"x": x, "o": output, "Outer": 2, "AxisExtent": 3, "Inner": 1}
        )
        smoke = rt.launch(artifact, bindings)
        if smoke.get("ok") is not True:
            raise RuntimeError(str(smoke.get("reason")))
        expected = np.asarray(corpus[fixture_id]["oracle"], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)
        raw = (ctypes.c_void_p * 2)(int(x.ctypes.data), int(output.ctypes.data))
        dims_values = (2, 2) if family == "softmax" else (2, 3, 1)
        dims = (ctypes.c_int64 * len(dims_values))(*dims_values)
        medians = {
            "device_event": _device_medians(
                bridge, entry=descriptor.entry_symbol, buffers=raw, dims=dims,
                samples=samples, warmups=warmups, repetitions=device_repetitions,
            ),
            "end_to_end": _e2e_medians(
                lambda: rt.launch(artifact, bindings), samples=samples,
                repetitions=e2e_repetitions,
            ),
        }
        resource = image.resource_record.to_dict() if image.resource_record else {}
        resource_fingerprint = hashlib.sha256(json.dumps({
            "entry": descriptor.entry_symbol, "image": image.image_digest,
            "resource": resource,
        }, sort_keys=True).encode()).hexdigest()
        for domain, values in medians.items():
            benchmark_rows.append({
                "family": family, "route": str(descriptor.provenance.get("schedule", "canonical_descriptor")),
                "timing_domain": domain, "median_ns": float(statistics.median(values)),
                "run_medians_ns": values, "stability_limit_pct": stability_limit,
                "stable": _stability(values) <= stability_limit, "selected": True,
                "repetitions": (
                    samples * device_repetitions if domain == "device_event"
                    else samples * e2e_repetitions
                ),
                "warmups": warmups, "discard_first": True,
                "resource_fingerprint": resource_fingerprint,
            })
        fixture_rows.append({
            "fixture_id": fixture_id,
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": output.tolist(), "image_digest": image.image_digest,
            "descriptor_digest": descriptor.descriptor_digest,
        })
        cache_rows.append({
            "fixture_id": fixture_id,
            "cold": {
                "compile_state": image.compile_state, "cache_key": image.cache_key,
                "image_digest": image.image_digest, "descriptor_digest": descriptor.descriptor_digest,
            },
            "warm": {
                "compile_state": warm.native_image.compile_state,
                "cache_key": warm.native_image.cache_key,
                "image_digest": warm.native_image.image_digest,
                "descriptor_digest": warm.launch_descriptor.descriptor_digest,
            },
        })
        resource_rows.append({
            "family": family, "entry": descriptor.entry_symbol,
            "selected_route": descriptor.provenance.get("schedule", "canonical_descriptor"),
            "resource_fingerprint": resource_fingerprint, "record": resource,
        })

    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": "nvidia_sm120", "architecture": "sm_120a",
        "device": {"exact": True, "identity": identity},
        "source_commit": source_commit,
        "toolchain_fingerprint": hashlib.sha256("\n".join(sorted(toolchains)).encode()).hexdigest(),
        "scope": ["softmax", "reduction"],
        "required_timing_domains": ["device_event", "end_to_end"],
        "fixtures": fixture_rows, "cache_proofs": cache_rows, "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)
    return report, {
        "schema": "tessera.e2e-sm120-resource-record.v1", "device": identity,
        "rows": resource_rows,
    }


def main() -> int:
    from tessera.compiler.e2e_fleet import seal_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-repetitions", type=int, default=100)
    parser.add_argument("--e2e-repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--stability-limit", type=float, default=4.0)
    parser.add_argument("--packet-dir", type=Path)
    args = parser.parse_args()
    report, resources = record(
        samples=args.samples, device_repetitions=args.device_repetitions,
        e2e_repetitions=args.e2e_repetitions, warmups=args.warmups,
        stability_limit=args.stability_limit,
    )
    if args.packet_dir:
        args.packet_dir.mkdir(parents=True, exist_ok=True)
        (args.packet_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (args.packet_dir / "resources.json").write_text(
            json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        seal_packet(args.packet_dir)
        print(f"sealed {args.packet_dir}")
    else:
        print(json.dumps({"report": report, "resources": resources}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
