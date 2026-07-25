#!/usr/bin/env python3
"""Record the complete E2E-SPINE-3 packet on an exact SM120 device.

The packet replays one small shared differential fixture through every
compiler-owned family descriptor in the closed SM120 scope.  ``selected`` means
that the canonical descriptor supplied the release evidence; it does not change
any production selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
PIPELINE = "tessera-nvidia-pipeline-sm120"


def _tensor(shape: tuple[int, ...], ir_dtype: str, storage: str):
    from tessera.compiler.graph_ir import IRType

    dims = "x".join(map(str, shape))
    return IRType(f"tensor<{dims}x{ir_dtype}>", tuple(map(str, shape)), storage)


def _module(family: str):
    from tessera.compiler.graph_ir import (
        GraphIRFunction,
        GraphIRModule,
        IRArg,
        IROp,
    )
    from tessera.compiler.primitive_coverage import NumericPolicy

    if family == "matmul":
        a, b, o = (
            _tensor((2, 3), "f32", "fp32"),
            _tensor((3, 2), "f32", "fp32"),
            _tensor((2, 2), "f32", "fp32"),
        )
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_matmul",
                    args=[IRArg("a", a), IRArg("b", b)],
                    result_types=[o],
                    body=[
                        IROp(
                            result="c",
                            op_name="tessera.matmul",
                            operands=["%a", "%b"],
                            operand_types=[str(a), str(b)],
                            result_type=str(o),
                            numeric_policy=NumericPolicy(
                                storage="fp32",
                                accum="fp32",
                                math_mode="tf32",
                            ),
                        )
                    ],
                    return_values=["%c"],
                )
            ]
        )
    if family == "softmax":
        x = _tensor((2, 2), "f32", "fp32")
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_softmax",
                    args=[IRArg("x", x)],
                    result_types=[x],
                    body=[
                        IROp(
                            result="o",
                            op_name="tessera.softmax",
                            operands=["%x"],
                            operand_types=[str(x)],
                            result_type=str(x),
                            kwargs={"axis": -1},
                        )
                    ],
                    return_values=["%o"],
                )
            ]
        )
    if family == "reduction":
        x, o = _tensor((2, 3), "f32", "fp32"), _tensor((2,), "f32", "fp32")
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_reduction",
                    args=[IRArg("x", x)],
                    result_types=[o],
                    body=[
                        IROp(
                            result="o",
                            op_name="tessera.sum",
                            operands=["%x"],
                            operand_types=[str(x)],
                            result_type=str(o),
                            kwargs={"axis": -1, "keepdims": False},
                        )
                    ],
                    return_values=["%o"],
                )
            ]
        )
    if family == "epilogue":
        a = _tensor((2, 2), "f16", "fp16")
        b = _tensor((2, 2), "f16", "fp16")
        bias = _tensor((2,), "f32", "fp32")
        residual = _tensor((2, 2), "f32", "fp32")
        o = _tensor((2, 2), "f32", "fp32")
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_epilogue",
                    args=[
                        IRArg("a", a),
                        IRArg("b", b),
                        IRArg("bias", bias),
                        IRArg("residual", residual),
                    ],
                    result_types=[o],
                    body=[
                        IROp(
                            result="c",
                            op_name="tessera.matmul",
                            operands=["%a", "%b"],
                            operand_types=[str(a), str(b)],
                            result_type=str(o),
                            kwargs={
                                "bias": "%bias",
                                "activation": "relu",
                                "residual": "%residual",
                            },
                        )
                    ],
                    return_values=["%c"],
                )
            ]
        )
    if family == "attention":
        q = _tensor((1, 1, 2, 2), "f32", "fp32")
        k = _tensor((1, 1, 2, 2), "f32", "fp32")
        v = _tensor((1, 1, 2, 2), "f32", "fp32")
        o = _tensor((1, 1, 2, 2), "f32", "fp32")
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_attention",
                    args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
                    result_types=[o],
                    body=[
                        IROp(
                            result="o",
                            op_name="tessera.flash_attn",
                            operands=["%q", "%k", "%v"],
                            operand_types=[str(q), str(k), str(v)],
                            result_type=str(o),
                            kwargs={"scale": 2.0**-0.5, "causal": False},
                        )
                    ],
                    return_values=["%o"],
                )
            ]
        )
    if family == "paged_kv":
        pages = _tensor((2, 2, 1, 2), "f32", "fp32")
        table = _tensor((2,), "i32", "int32")
        result = _tensor((2, 1, 2), "f32", "fp32")
        return GraphIRModule(
            functions=[
                GraphIRFunction(
                    name="fleet_paged_kv",
                    args=[IRArg("pages", pages), IRArg("page_table", table)],
                    result_types=[result],
                    body=[
                        IROp(
                            result="slice",
                            op_name="tessera.kv_cache.read",
                            operands=["%pages", "%page_table"],
                            operand_types=[str(pages), str(table)],
                            result_type=str(result),
                            kwargs={"start": 1, "end": 3},
                        )
                    ],
                    return_values=["%slice"],
                )
            ]
        )
    raise ValueError(f"family {family!r} does not use a Graph module")


def _stability(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values) * 100.0


def _two_run_medians(
    call: Callable[[], float],
    *,
    samples: int,
) -> list[float]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in (0, 1) if sample % 2 == 0 else (1, 0):
            condition_sm120(reps=20)
            cohorts[cohort].append(float(call()))
    return [float(statistics.median(values)) for values in cohorts]


def _end_to_end_medians(
    call: Callable[[], dict[str, Any]],
    *,
    samples: int,
    repetitions: int,
) -> list[float]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    call()  # discard registration, module-load, and first-use cache effects
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in (0, 1) if sample % 2 == 0 else (1, 0):
            condition_sm120(reps=20)
            started = time.perf_counter_ns()
            for _ in range(repetitions):
                result = call()
                if result.get("ok") is not True:
                    raise RuntimeError(str(result.get("reason")))
            cohorts[cohort].append((time.perf_counter_ns() - started) / repetitions)
    return [float(statistics.median(values)) for values in cohorts]


def _graph_packages(family: str):
    from tessera.compiler import nvidia_native
    from tessera.compiler.driver import compile_graph_module

    module = _module(family)
    nvidia_native._cache.clear()
    options: dict[str, Any] = {"package_native": True}
    if family == "matmul":
        options["nvidia_schedule"] = "direct"
    cold = compile_graph_module(
        module,
        source_origin="E2E-SPINE-3",
        target="nvidia_sm120",
        options=options,
        enable_tool_validation=False,
    )
    warm = compile_graph_module(
        module,
        source_origin="E2E-SPINE-3",
        target="nvidia_sm120",
        options=options,
        enable_tool_validation=False,
    )
    if not cold.native_image or not cold.launch_descriptor:
        raise RuntimeError(f"{family} did not produce a launchable native package")
    if not warm.native_image or not warm.launch_descriptor:
        raise RuntimeError(f"{family} warm compilation lost its native package")
    return cold.native_image, cold.launch_descriptor, warm.native_image, warm.launch_descriptor


def _direct_packages(family: str):
    from tessera.compiler import nvidia_native

    nvidia_native._cache.clear()
    if family == "replay_ssm":
        cold = nvidia_native.package_replay_ssm_kernels(
            batch=1,
            channels=1,
            state_dim=1,
            capacity=1,
            async_slots=2,
            pipeline_name=PIPELINE,
        )[0]
        warm = nvidia_native.package_replay_ssm_kernels(
            batch=1,
            channels=1,
            state_dim=1,
            capacity=1,
            async_slots=2,
            pipeline_name=PIPELINE,
        )[0]
    elif family == "moe":
        arguments = {
            "num_tokens": 3,
            "num_slots": 2,
            "hidden": 2,
            "expert_count": 1,
            "expert_k": 2,
            "expert_n": 2,
            "group_offsets": (0, 2),
            "pipeline_name": PIPELINE,
            "storage": "fp32",
        }
        cold = nvidia_native.package_moe_kernels(**arguments)[0]
        warm = nvidia_native.package_moe_kernels(**arguments)[0]
    else:
        raise ValueError(f"family {family!r} does not use a direct package")
    return cold.image, cold.descriptor, warm.image, warm.descriptor


def _case_data(
    family: str,
    fixture: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    inputs = fixture["inputs"]
    if family == "matmul":
        arrays = {
            "a": np.ascontiguousarray(inputs["a"], dtype=np.float32),
            "b": np.asfortranarray(inputs["b"], dtype=np.float32),
            "c": np.zeros((2, 2), np.float32),
        }
        return arrays, {"M": 2, "N": 2, "K": 3}, arrays["c"]
    if family == "softmax":
        arrays = {
            "x": np.ascontiguousarray(inputs["x"], dtype=np.float32),
            "o": np.zeros((2, 2), np.float32),
        }
        return arrays, {"Rows": 2, "K": 2}, arrays["o"]
    if family == "reduction":
        arrays = {
            "x": np.ascontiguousarray(inputs["x"], dtype=np.float32),
            "o": np.zeros((2,), np.float32),
        }
        return arrays, {"Outer": 2, "AxisExtent": 3, "Inner": 1}, arrays["o"]
    if family == "epilogue":
        arrays = {
            "a": np.ascontiguousarray(inputs["a"], dtype=np.float16),
            "b": np.asfortranarray(inputs["b"], dtype=np.float16),
            "bias": np.ascontiguousarray(inputs["bias"], dtype=np.float32),
            "residual": np.ascontiguousarray(inputs["residual"], dtype=np.float32),
            "c": np.zeros((2, 2), np.float32),
        }
        return arrays, {"M": 2, "N": 2, "K": 2}, arrays["c"]
    if family == "attention":
        arrays = {
            "q": np.ascontiguousarray(inputs["q"], dtype=np.float32),
            "k": np.ascontiguousarray(inputs["k"], dtype=np.float32),
            "v": np.ascontiguousarray(inputs["v"], dtype=np.float32),
            "o": np.zeros((1, 1, 2, 2), np.float32),
        }
        scalars = {"B": 1, "Hq": 1, "Hkv": 1, "Sq": 2, "Sk": 2, "D": 2, "Dv": 2}
        return arrays, scalars, arrays["o"]
    if family == "paged_kv":
        arrays = {
            "pages": np.ascontiguousarray(inputs["pages"], dtype=np.float32),
            "page_table": np.ascontiguousarray(inputs["page_table"], dtype=np.int32),
            "slice": np.zeros((2, 1, 2), np.float32),
        }
        scalars = {
            "P": 2,
            "LP": 2,
            "PageSize": 2,
            "H": 1,
            "D": 2,
            "Start": 1,
            "Tokens": 2,
        }
        return arrays, scalars, arrays["slice"]
    if family == "replay_ssm":
        arrays = {
            "delta": np.ascontiguousarray(inputs["delta"], dtype=np.float32),
            "x": np.ascontiguousarray(inputs["x"], dtype=np.float32),
            "B": np.ascontiguousarray(inputs["bcoef"], dtype=np.float32),
            "S0": np.ascontiguousarray(inputs["s0"], dtype=np.float32),
            "C": np.ascontiguousarray(inputs["c"], dtype=np.float32),
            "A": np.ascontiguousarray(inputs["a"], dtype=np.float32),
            "Y": np.zeros((1, 1, 1), np.float32),
        }
        return arrays, {"Batch": 1, "Channels": 1, "State": 1, "Tokens": 1}, arrays["Y"]
    if family == "moe":
        arrays = {
            "X": np.ascontiguousarray(inputs["x"], dtype=np.float32),
            "token_of_slot": np.ascontiguousarray(inputs["token"], dtype=np.int32),
            "dispatched": np.zeros((2, 2), np.float32),
        }
        return arrays, {"Tokens": 3, "Slots": 2, "Hidden": 2}, arrays["dispatched"]
    raise ValueError(f"unsupported packet family {family!r}")


def record(
    *,
    samples: int,
    device_repetitions: int,
    e2e_repetitions: int,
    warmups: int,
    stability_limit: float,
    families: tuple[str, ...] | None = None,
    base_report: dict[str, Any] | None = None,
    base_resources: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tessera import runtime as rt
    from tessera.compiler.e2e_fleet import (
        load_fixture_corpus,
        validate_backend_report,
    )

    identity = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not identity.endswith("12.0"):
        raise RuntimeError(f"assigned E2E packet requires compute capability 12.0, found {identity}")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if rt._load_nvidia_ptx_launch() is None:
        raise RuntimeError("SM120 PTX launch/benchmark bridge is unavailable")

    corpus = load_fixture_corpus()
    definitions = (
        ("matmul", "matmul-f32-2x3x2-v1"),
        ("softmax", "softmax-f32-2x2-extreme-v1"),
        ("reduction", "reduce-sum-f32-axis1-v1"),
        ("epilogue", "epilogue-f16-bias-relu-residual-2x2-v1"),
        ("attention", "attention-f32-uniform-1x1x2x2-v1"),
        ("paged_kv", "paged-kv-f32-permuted-2x2-v1"),
        ("replay_ssm", "replay-ssm-f32-decode-1x1x1x1-v1"),
        ("moe", "moe-dispatch-f32-permuted-3x2-v1"),
    )
    known_families = {family for family, _ in definitions}
    requested = known_families if families is None else set(families)
    unknown = requested - known_families
    if unknown:
        raise ValueError("unknown SM120 packet families: " + ", ".join(sorted(unknown)))

    retained_fixture_ids: set[str] = set()
    if base_report is not None:
        if base_report.get("target") != "nvidia_sm120" or base_report.get("architecture") != "sm_120a":
            raise ValueError("base report is not the SM120 fleet packet")
        retained_fixture_ids = {
            str(row["fixture_id"])
            for row in base_report.get("fixtures", [])
            if str(corpus[str(row["fixture_id"])]["family"]) not in requested
        }
    fixture_rows: list[dict[str, Any]] = [
        dict(row) for row in (base_report or {}).get("fixtures", []) if str(row["fixture_id"]) in retained_fixture_ids
    ]
    cache_rows: list[dict[str, Any]] = [
        dict(row)
        for row in (base_report or {}).get("cache_proofs", [])
        if str(row["fixture_id"]) in retained_fixture_ids
    ]
    retained_families = {str(corpus[fixture_id]["family"]) for fixture_id in retained_fixture_ids}
    benchmark_rows: list[dict[str, Any]] = [
        dict(row) for row in (base_report or {}).get("benchmarks", []) if str(row["family"]) in retained_families
    ]
    resource_rows: list[dict[str, Any]] = [
        dict(row) for row in (base_resources or {}).get("rows", []) if str(row["family"]) in retained_families
    ]
    toolchains: set[str] = set()
    if base_report and base_report.get("toolchain_fingerprint"):
        toolchains.add(str(base_report["toolchain_fingerprint"]))

    for family, fixture_id in definitions:
        if family not in requested:
            continue
        if family in {"replay_ssm", "moe"}:
            image, descriptor, warm_image, warm_descriptor = _direct_packages(family)
        else:
            image, descriptor, warm_image, warm_descriptor = _graph_packages(family)
        if (
            image.cache_key != warm_image.cache_key
            or image.image_digest != warm_image.image_digest
            or descriptor.descriptor_digest != warm_descriptor.descriptor_digest
        ):
            raise RuntimeError(f"{family} cold/warm compilation is not reproducible")

        arrays, scalars, output = _case_data(family, corpus[fixture_id])
        bindings: dict[str, Any] = {**arrays, **scalars}

        def launch() -> dict[str, Any]:
            if family == "replay_ssm":
                from tessera.compiler.emit.nvidia_cuda import NvidiaReplayDeviceState

                state = NvidiaReplayDeviceState(
                    arrays["S0"],
                    arrays["A"],
                    capacity=1,
                    async_slots=2,
                )
                try:
                    state.append(arrays["delta"][0], arrays["x"][0], arrays["B"][0], 0)
                    arrays["Y"][0] = state.decode(arrays["C"], 1)
                finally:
                    state.close()
                return {"ok": True, "output": arrays["Y"]}
            result = rt._submit_nvidia_sm120_native(
                image,
                descriptor,
                arrays,
                scalars,
                None,
            )
            return (
                result
                if isinstance(result, dict)
                else {
                    "ok": True,
                    "output": result,
                }
            )

        smoke = launch()
        if smoke.get("ok") is not True:
            raise RuntimeError(f"{family}: {smoke.get('reason')}")
        expected = np.asarray(corpus[fixture_id]["oracle"], dtype=np.float32)
        policy = corpus[fixture_id]["tolerance"]
        np.testing.assert_allclose(
            output,
            expected,
            rtol=float(policy["rtol"]),
            atol=float(policy["atol"]),
            equal_nan=bool(policy.get("equal_nan", False)),
        )
        actual = output.copy().tolist()

        retained_device_repetitions = device_repetitions * (10 if family == "moe" else 1)
        if family == "replay_ssm":
            from tessera.compiler.emit.nvidia_cuda import NvidiaReplayDeviceState

            replay_state = NvidiaReplayDeviceState(
                arrays["S0"],
                arrays["A"],
                capacity=1,
                async_slots=2,
            )
            replay_state.append(
                arrays["delta"][0],
                arrays["x"][0],
                arrays["B"][0],
                0,
            )

            def replay_device_event() -> float:
                values: list[float] = []
                for _ in range(retained_device_repetitions):
                    pending = replay_state.submit_block_async(
                        arrays["delta"],
                        arrays["x"],
                        arrays["B"],
                        arrays["C"],
                        0,
                    )
                    values.append(float(pending.event.elapsed_ms()))
                    pending.wait()
                return float(statistics.median(values)) * 1_000_000.0

            try:
                for _ in range(warmups):
                    replay_device_event()
                device_values = _two_run_medians(
                    replay_device_event,
                    samples=samples,
                )
            finally:
                replay_state.close()
        else:
            device_values = _two_run_medians(
                lambda: (
                    rt._nvidia_native_descriptor_device_latency(
                        image,
                        descriptor,
                        bindings,
                        reps=retained_device_repetitions,
                        warmup=warmups,
                    )
                    * 1_000_000.0
                ),
                samples=samples,
            )
        retained_e2e_repetitions = e2e_repetitions * (
            1 if family == "replay_ssm" else 100 if family == "matmul" else 10
        )
        e2e_values = _end_to_end_medians(
            launch,
            samples=samples,
            repetitions=retained_e2e_repetitions,
        )
        resource = image.resource_record.to_dict() if image.resource_record else {}
        resource_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "entry": descriptor.entry_symbol,
                    "image": image.image_digest,
                    "resource": resource,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for domain, values in (
            ("device_event", device_values),
            ("end_to_end", e2e_values),
        ):
            benchmark_rows.append(
                {
                    "family": family,
                    "route": str(
                        descriptor.provenance.get(
                            "schedule",
                            descriptor.provenance.get("route", "canonical_descriptor"),
                        )
                    ),
                    "timing_domain": domain,
                    "median_ns": float(statistics.median(values)),
                    "run_medians_ns": values,
                    "stability_limit_pct": stability_limit,
                    "stable": _stability(values) <= stability_limit,
                    "selected": True,
                    "repetitions": (
                        samples * retained_device_repetitions
                        if domain == "device_event"
                        else samples * retained_e2e_repetitions
                    ),
                    "warmups": warmups,
                    "discard_first": True,
                    "resource_fingerprint": resource_fingerprint,
                }
            )
        print(
            f"{family}: device={_stability(device_values):.3f}% end_to_end={_stability(e2e_values):.3f}%",
            flush=True,
        )
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "levels": {"a": "proven", "b": "proven", "c": "proven"},
                "actual": actual,
                "image_digest": image.image_digest,
                "descriptor_digest": descriptor.descriptor_digest,
            }
        )
        cache_rows.append(
            {
                "fixture_id": fixture_id,
                "cold": {
                    "compile_state": image.compile_state,
                    "cache_key": image.cache_key,
                    "image_digest": image.image_digest,
                    "descriptor_digest": descriptor.descriptor_digest,
                },
                "warm": {
                    "compile_state": warm_image.compile_state,
                    "cache_key": warm_image.cache_key,
                    "image_digest": warm_image.image_digest,
                    "descriptor_digest": warm_descriptor.descriptor_digest,
                },
            }
        )
        resource_rows.append(
            {
                "family": family,
                "entry": descriptor.entry_symbol,
                "selected_route": benchmark_rows[-1]["route"],
                "resource_fingerprint": resource_fingerprint,
                "record": resource,
            }
        )
        toolchains.add(image.toolchain_fingerprint)

    present_families = {str(corpus[str(row["fixture_id"])]["family"]) for row in fixture_rows}
    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": "nvidia_sm120",
        "architecture": "sm_120a",
        "device": {"exact": True, "identity": identity},
        "source_commit": source_commit,
        "toolchain_fingerprint": hashlib.sha256("\n".join(sorted(toolchains)).encode()).hexdigest(),
        "scope": [family for family, _ in definitions if family in present_families],
        "required_timing_domains": ["device_event", "end_to_end"],
        "fixtures": fixture_rows,
        "cache_proofs": cache_rows,
        "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)
    return report, {
        "schema": "tessera.e2e-sm120-resource-record.v1",
        "device": identity,
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
    parser.add_argument(
        "--families",
        nargs="+",
        choices=(
            "matmul",
            "softmax",
            "reduction",
            "epilogue",
            "attention",
            "paged_kv",
            "replay_ssm",
            "moe",
        ),
        help="Replace only these family rows; retain other rows from --base-packet.",
    )
    parser.add_argument(
        "--base-packet",
        type=Path,
        help="Validated packet whose unselected family rows are retained.",
    )
    parser.add_argument("--packet-dir", type=Path)
    args = parser.parse_args()
    base_report = base_resources = None
    if args.base_packet:
        from tessera.compiler.e2e_fleet import validate_packet

        validate_packet(args.base_packet)
        base_report = json.loads((args.base_packet / "report.json").read_text(encoding="utf-8"))
        resources_path = args.base_packet / "resources.json"
        if resources_path.is_file():
            base_resources = json.loads(resources_path.read_text(encoding="utf-8"))
    report, resources = record(
        samples=args.samples,
        device_repetitions=args.device_repetitions,
        e2e_repetitions=args.e2e_repetitions,
        warmups=args.warmups,
        stability_limit=args.stability_limit,
        families=tuple(args.families) if args.families else None,
        base_report=base_report,
        base_resources=base_resources,
    )
    if args.packet_dir:
        args.packet_dir.mkdir(parents=True, exist_ok=True)
        (args.packet_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.packet_dir / "resources.json").write_text(
            json.dumps(resources, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seal_packet(args.packet_dir)
        print(f"sealed {args.packet_dir}")
    else:
        print(
            json.dumps(
                {"report": report, "resources": resources},
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
