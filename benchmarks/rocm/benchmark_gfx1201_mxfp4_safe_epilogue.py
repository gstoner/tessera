#!/usr/bin/env python3
"""Correctness-gated, matched safe-epilogue versus ordinary folded prefill."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import package_mxfp4_folded_prefill
from tessera.compiler.rocm_mxfp4_prefill_memory import (
    MXFP4LayerShape, assess_prefill_weight_residency,
)
from tessera.compiler.rocm_mxfp4_tn4_experiment import package_folded_tn4_experiment
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_packed_folded as packed_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.inspect_gfx1201_folded_prefill import selected_symbol_isa_evidence


def _parse_model_layer(value: str) -> MXFP4LayerShape:
    try:
        n, k, count = (int(part) for part in value.split(","))
        return MXFP4LayerShape(n, k, count)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("model layer must be N,K,count") from error


def _candidate_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    copies: int, folded: Any, *, tn4: bool = False,
) -> base._Engine:
    package = (
        package_folded_tn4_experiment(case.m, case.n, case.k, folded)
        if tn4 else package_mxfp4_folded_prefill(
            case.m, case.n, case.k, folded, allow_approximate=True,
            entry="tessera_mxfp4_folded_safe_epilogue",
            safe_epilogue_scales=inputs["a_scale"],
        )
    )
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), package.image.payload) != 0:
        raise RuntimeError("safe folded MXFP4 module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, package.descriptor.entry_symbol.encode()
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("safe folded MXFP4 entry missing")
    arrays = (
        inputs["a"], folded.weight_bytes, inputs["a_scale"],
        folded.row_reference, inputs["output"],
    )
    device_copies = base._copies(hip, arrays, copies)

    def launch(bundle: base._DeviceArrays) -> None:
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in bundle.device),
            ctypes.c_int64(case.m), ctypes.c_int64(case.n), ctypes.c_int64(case.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        grid = package.descriptor.geometry.grid
        workgroup = package.descriptor.geometry.workgroup
        assert grid is not None and workgroup is not None
        rc = hip.hipModuleLaunchKernel(function, *grid, *workgroup, 0, None, arguments, None)
        if rc != 0:
            raise RuntimeError(f"safe folded MXFP4 launch failed rc={rc}")

    engine = base._Engine(
        "tessera_folded_tn4" if tn4 else "tessera_folded_safe_epilogue",
        hip, device_copies, launch, 4,
        {
            "abi": package.descriptor.abi_id,
            "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            "selected_isa": selected_symbol_isa_evidence(
                package.image.payload, package.descriptor.entry_symbol,
            ),
            "route": package.descriptor.provenance,
            **base._code_object_evidence(package.image.payload),
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine


def benchmark(
    cases: tuple[base.Case, ...], radiance_module: Path, *,
    radiance_revision: str, warmup: int = 6, trials: int = 11,
    iterations: int = 12,
    model_layers: tuple[MXFP4LayerShape, ...] = (),
    available_extra_bytes: int = 0,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("safe epilogue benchmark requires selected gfx1201")
    if os.environ.get("RADIANCE_MXFP4_WPERM") != "1":
        raise ValueError("matched Radiance requires fragment-order WPERM=1")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("safe epilogue benchmark requires HIP")
    radiance = base._load_radiance(radiance_module)
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.workload != "prefill" or case.k % 64:
            raise ValueError("safe epilogue comparison requires K64 prefill")
        inputs = base._logical_inputs(case)
        exact = base._tessera_engine(hip, case, inputs, 3, None, None)
        ordinary, folded = folded_bench.folded_engine(hip, case, inputs, 3)
        safe = _candidate_engine(hip, case, inputs, 3, folded)
        tn4 = _candidate_engine(hip, case, inputs, 3, folded, tn4=True)
        packed, packed_payload = packed_bench.packed_folded_engine(
            hip, case, inputs, 3, integer_decode=False,
            batched_loads=True, permute_decode=True,
        )
        independent = base._radiance_engine(hip, radiance, case, inputs, 3)
        engines = [exact, ordinary, safe, tn4, packed, independent]
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sampled_rows, sampled_cols, exact_reference = base._sampled_exact_reference(
                case, inputs,
            )
            np.testing.assert_array_equal(
                outputs["tessera"][np.ix_(sampled_rows, sampled_cols)], exact_reference,
            )
            if not folded.lossless or not packed_payload.lossless:
                raise RuntimeError("safe epilogue packet requires lossless fold")
            for engine in engines[1:]:
                np.testing.assert_array_equal(
                    outputs[engine.name].view(np.uint16),
                    outputs["tessera"].view(np.uint16),
                    err_msg=f"{case.label}: {engine.name} differs from exact K32",
                )
            samples = base._measure_interleaved(
                hip, engines, warmup=warmup, trials=trials, iterations=iterations,
            )
            for engine in engines:
                rows.append({
                    "case": case.label, "engine": engine.name,
                    "median_ms": statistics.median(samples[engine]),
                    "samples_ms": samples[engine],
                    "output_sha256": hashlib.sha256(
                        outputs[engine.name].view(np.uint8)
                    ).hexdigest(),
                    "metadata": engine.metadata,
                })
        finally:
            for engine in reversed(engines):
                engine.close()
    root = Path(__file__).resolve().parents[2]
    return {
        "schema": "tessera.rocm.gfx1201_mxfp4_safe_epilogue_benchmark.v1",
        "sync_key": "GFX1201-MXFP4-SAFE-EPILOGUE-2026-09-23",
        "device": base._selected_device_name(hip),
        "architecture": rt._rocm_live_arch(),
        "source_revision": base._git_revision(),
        "radiance": {
            "revision": radiance_revision,
            "binary_sha256": base._sha256(radiance_module),
            "weight_layout": "fragment_order", "wperm": 1,
        },
        "generator_sha256": base._sha256(
            root / "python/tessera/compiler/rocm_mxfp4_folded.py"
        ),
        "tn4_generator_sha256": base._sha256(
            root / "python/tessera/compiler/rocm_mxfp4_tn4_experiment.py"
        ),
        "packed_generator_sha256": base._sha256(
            root / "python/tessera/compiler/rocm_mxfp4_packed_folded.py"
        ),
        "packed_benchmark_sha256": base._sha256(
            root / "benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py"
        ),
        "benchmark_sha256": base._sha256(Path(__file__)),
        "model_layers": [
            {"n": layer.n, "k": layer.k, "count": layer.count}
            for layer in model_layers
        ],
        "model_weight_residency": (
            assess_prefill_weight_residency(
                model_layers, available_extra_bytes=available_extra_bytes,
            ) if model_layers else None
        ),
        "timing_order": "alternating_interleaved_per_shape",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=base._parse_case)
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--model-layer", action="append", type=_parse_model_layer)
    parser.add_argument("--available-extra-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = benchmark(
        tuple(args.case or (
            base.Case("prefill", 256, 5120, 8704),
            base.Case("prefill", 1024, 17408, 5120),
        )),
        args.radiance_module, radiance_revision=args.radiance_revision,
        model_layers=tuple(args.model_layer or ()),
        available_extra_bytes=args.available_extra_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
