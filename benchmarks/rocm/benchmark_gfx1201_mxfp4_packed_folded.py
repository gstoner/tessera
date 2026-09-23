#!/usr/bin/env python3
"""Matched packed-vs-expanded folded-vs-Radiance gfx1201 prefill timing."""
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
from tessera.compiler.rocm_mxfp4_packed_folded import (
    PackedFoldedPayload,
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.inspect_gfx1201_folded_prefill import (
    selected_symbol_isa_evidence,
)


def packed_folded_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray], copies: int,
    *, integer_decode: bool, batched_loads: bool = False,
    batched_a_loads: bool = False, reuse_pair_scales: bool = False,
) -> tuple[base._Engine, PackedFoldedPayload]:
    payload = prepare_packed_folded_payload(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    package = package_mxfp4_packed_folded_prefill(
        case.m, payload, integer_decode=integer_decode,
        batched_loads=batched_loads, batched_a_loads=batched_a_loads,
        reuse_pair_scales=reuse_pair_scales,
    )
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), package.image.payload) != 0:
        raise RuntimeError("packed folded MXFP4 module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, package.descriptor.entry_symbol.encode()
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("packed folded MXFP4 entry missing")
    arrays = (
        inputs["a"], payload.weight_bytes, inputs["a_scale"],
        payload.scale_plane, inputs["output"],
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
            raise RuntimeError(f"packed folded MXFP4 launch failed rc={rc}")

    engine = base._Engine(
        (
            "tessera_packed_pair_scale_integer" if reuse_pair_scales else
            "tessera_packed_batched_ab_integer" if batched_a_loads and batched_loads else
            "tessera_packed_batched_a_integer" if batched_a_loads else
            "tessera_packed_batched_b_integer" if batched_loads else
            "tessera_packed_integer" if integer_decode else "tessera_packed_table"
        ),
        hip, device_copies, launch, 4,
        {
            "abi": package.descriptor.abi_id,
            "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            "selected_isa": selected_symbol_isa_evidence(
                package.image.payload, package.descriptor.entry_symbol,
            ),
            "compiler_fingerprint": package.image.compiler_fingerprint,
            "toolchain_fingerprint": package.image.toolchain_fingerprint,
            "route": package.descriptor.provenance,
            **base._code_object_evidence(package.image.payload),
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine, payload


def benchmark(
    cases: tuple[base.Case, ...], radiance_module: Path, *,
    radiance_revision: str, tessera_opt: Path,
    include_batched: bool = False,
    warmup: int = 6, trials: int = 11, iterations: int = 12,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("packed folded benchmark requires selected gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("packed folded benchmark requires HIP")
    if os.environ.get("RADIANCE_MXFP4_WPERM") != "1":
        raise ValueError("matched fragment-order Radiance requires WPERM=1")
    radiance = base._load_radiance(radiance_module)
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.workload != "prefill":
            raise ValueError("packed folded benchmark requires prefill")
        inputs = base._logical_inputs(case)
        exact = base._tessera_engine(hip, case, inputs, 3, None, None)
        expanded, folded = folded_bench.folded_engine(
            hip, case, inputs, 3, tessera_opt=tessera_opt,
        )
        packed_table, payload = packed_folded_engine(
            hip, case, inputs, 3, integer_decode=False,
        )
        packed_integer, integer_payload = packed_folded_engine(
            hip, case, inputs, 3, integer_decode=True,
        )
        batched_engines: list[base._Engine] = []
        batched_payloads: list[PackedFoldedPayload] = []
        if include_batched:
            for batched_b, batched_a, pair_scale in (
                (True, False, False), (False, True, False),
                (True, True, False), (True, False, True),
            ):
                candidate, candidate_payload = packed_folded_engine(
                    hip, case, inputs, 3, integer_decode=True,
                    batched_loads=batched_b, batched_a_loads=batched_a,
                    reuse_pair_scales=pair_scale,
                )
                batched_engines.append(candidate)
                batched_payloads.append(candidate_payload)
        independent = base._radiance_engine(hip, radiance, case, inputs, 3)
        engines = [exact, expanded, packed_table, packed_integer]
        engines.extend(batched_engines)
        engines.append(independent)
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sampled_rows, sampled_cols, exact_reference = (
                base._sampled_exact_reference(case, inputs)
            )
            np.testing.assert_array_equal(
                outputs["tessera"][np.ix_(sampled_rows, sampled_cols)],
                exact_reference,
            )
            if (not folded.lossless or not payload.lossless or
                    not integer_payload.lossless or
                    any(not candidate.lossless for candidate in batched_payloads)):
                raise RuntimeError("matched timing requires lossless folded inputs")
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
        "schema": (
            "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v2"
            if include_batched else
            "tessera.rocm.gfx1201_mxfp4_packed_folded_benchmark.v1"
        ),
        "sync_key": (
            "GFX1201-PACKED-STAGING-ABLATION-2026-09-23"
            if include_batched else "GFX1201-PACKED-FOLDED-DECODE-2026-09-23"
        ),
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
        "packed_abi_sha256": base._sha256(
            root / "python/tessera/compiler/rocm_mxfp4_packed_folded.py"
        ),
        "benchmark_sha256": base._sha256(Path(__file__)),
        "timing_order": "alternating_interleaved_per_shape",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=base._parse_case)
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-batched", action="store_true")
    args = parser.parse_args()
    packet = benchmark(
        tuple(args.case or (
            base.Case("prefill", 256, 5120, 8704),
            base.Case("prefill", 1024, 17408, 5120),
        )),
        args.radiance_module, radiance_revision=args.radiance_revision,
        tessera_opt=args.tessera_opt, include_batched=args.include_batched,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
