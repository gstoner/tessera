#!/usr/bin/env python3
"""Matched exact-vs-folded-vs-Radiance gfx1201 prefill timing."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import (
    package_mxfp4_folded_prefill, prepare_folded_weights,
)
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base


def folded_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    copies: int,
) -> tuple[base._Engine, mx.FoldedRowReference]:
    folded = prepare_folded_weights(
        inputs["packed_row_major"], inputs["b_scale"],
        allow_approximate=True,
    )
    package = package_mxfp4_folded_prefill(
        case.m, case.n, case.k, folded, allow_approximate=True,
    )
    isa = base._code_object_evidence(package.image.payload)
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), package.image.payload) != 0:
        raise RuntimeError("folded MXFP4 module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, package.descriptor.entry_symbol.encode()
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("folded MXFP4 entry missing")
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
        rc = hip.hipModuleLaunchKernel(
            function, *grid, *workgroup, 0, None, arguments, None
        )
        if rc != 0:
            raise RuntimeError(f"folded MXFP4 launch failed rc={rc}")

    engine = base._Engine(
        "tessera_folded", hip, device_copies, launch, 4,
        {
            "abi": package.descriptor.abi_id,
            "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            "route": package.descriptor.provenance,
            **isa,
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine, folded


def benchmark(
    cases: tuple[base.Case, ...], radiance_module: Path | None, *,
    radiance_revision: str | None = None,
    warmup: int = 6, trials: int = 11, iterations: int = 12,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("folded MXFP4 benchmark requires selected gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("folded MXFP4 benchmark requires HIP")
    if radiance_module is not None and not radiance_revision:
        raise ValueError("matched Radiance timing requires its pinned source revision")
    radiance = base._load_radiance(radiance_module) if radiance_module else None
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.workload != "prefill":
            raise ValueError("folded BM256/TM4 benchmark requires prefill")
        inputs = base._logical_inputs(case)
        exact = base._tessera_engine(hip, case, inputs, 3, None, None)
        folded_engine_obj, folded = folded_engine(hip, case, inputs, 3)
        engines = [exact, folded_engine_obj]
        if radiance is not None:
            engines.append(base._radiance_engine(hip, radiance, case, inputs, 3))
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sampled_rows, sampled_cols, exact_reference = (
                base._sampled_exact_reference(case, inputs)
            )
            np.testing.assert_array_equal(
                outputs["tessera"][np.ix_(sampled_rows, sampled_cols)],
                exact_reference,
                err_msg=f"{case.label}: exact K32 route fails independent oracle",
            )
            if folded.lossless:
                for name, output in outputs.items():
                    np.testing.assert_array_equal(
                        output.view(np.uint16), outputs["tessera"].view(np.uint16),
                        err_msg=f"{case.label}: {name} diverged from exact K32 oracle",
                    )
            else:
                raise RuntimeError(
                    "inexact benchmark payload requires a separate quantified output gate"
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
    return {
        "schema": "tessera.rocm.gfx1201_mxfp4_folded_benchmark.v1",
        "device": base._selected_device_name(hip),
        "architecture": rt._rocm_live_arch(),
        "source_revision": base._git_revision(),
        "radiance": None if radiance_module is None else {
            "revision": radiance_revision,
            "binary_sha256": base._sha256(radiance_module),
        },
        "folded_generator_sha256": base._sha256(
            Path(__file__).resolve().parents[2]
            / "python/tessera/compiler/rocm_mxfp4_folded.py"
        ),
        "timing_order": "alternating_interleaved_per_shape",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=base._parse_case)
    parser.add_argument("--radiance-module", type=Path)
    parser.add_argument("--radiance-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = benchmark(
        tuple(args.case or (
            base.Case("prefill", 256, 5120, 8704),
            base.Case("prefill", 1024, 17408, 5120),
        )),
        args.radiance_module, radiance_revision=args.radiance_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
