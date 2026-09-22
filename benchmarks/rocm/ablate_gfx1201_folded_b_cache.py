#!/usr/bin/env python3
"""One-lever B-load cache ablation for folded gfx1201 prefill.

Changes one B vector load to a non-temporal load. Never selects the variant
for production; output and selected ISA must be checked before timing.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import (
    emit_mxfp4_folded_prefill_hip, prepare_folded_weights,
    package_mxfp4_folded_prefill,
)
from tessera.compiler.rocm_mxfp4_native import _extract_gfx1201_hsaco, _rocm_hipcc
from tessera.compiler.rocm_native import _rocm_path
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base


ROOT = Path(__file__).resolve().parents[2]
OLD_LOAD = (
    "value = *reinterpret_cast<const copy_u32x4 *>"
    "(B + safe * K + kb + off);"
)
NEW_LOAD = (
    "value = __builtin_nontemporal_load("
    "reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off));"
)


def variant_source() -> str:
    source = emit_mxfp4_folded_prefill_hip()
    if source.count(OLD_LOAD) != 1:
        raise RuntimeError("B cache ablation requires exactly one baseline load site")
    return source.replace(OLD_LOAD, NEW_LOAD)


def _compile_variant(source: str) -> bytes:
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("B cache ablation requires hipcc")
    with tempfile.TemporaryDirectory(prefix="tessera-folded-bcache-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        result = subprocess.run(
            [
                str(compiler), "-x", "hip", "-O3", "--genco",
                "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
                str(source_path), "-o", str(bundle_path),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "B cache ablation compilation failed: " + result.stderr[-800:]
            )
        return _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)


def _variant_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    folded: Any, payload: bytes, source_sha: str,
) -> base._Engine:
    package = package_mxfp4_folded_prefill(
        case.m, case.n, case.k, folded, allow_approximate=True,
    )
    image = replace(
        package.image, payload=payload, target_ir_digest=source_sha,
    )
    descriptor = replace(
        package.descriptor,
        image_digest=image.image_digest,
        provenance={
            **package.descriptor.provenance,
            "ablation": "B_global_load_non_temporal_only",
            "variant_source_sha256": source_sha,
        },
    )
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), image.payload) != 0:
        raise RuntimeError("B cache ablation module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, descriptor.entry_symbol.encode(),
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("B cache ablation entry missing")
    arrays = (
        inputs["a"], folded.weight_bytes, inputs["a_scale"],
        folded.row_reference, inputs["output"],
    )
    device_copies = base._copies(hip, arrays, 3)

    def launch(bundle: base._DeviceArrays) -> None:
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in bundle.device),
            ctypes.c_int64(case.m), ctypes.c_int64(case.n), ctypes.c_int64(case.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        grid = descriptor.geometry.grid
        workgroup = descriptor.geometry.workgroup
        assert grid is not None and workgroup is not None
        rc = hip.hipModuleLaunchKernel(
            function, *grid, *workgroup, 0, None, arguments, None,
        )
        if rc != 0:
            raise RuntimeError(f"B cache ablation launch failed rc={rc}")

    engine = base._Engine(
        "tessera_b_nontemporal", hip, device_copies, launch, 4,
        {
            "source_sha256": source_sha,
            "image_sha256": hashlib.sha256(image.payload).hexdigest(),
            "abi": descriptor.abi_id,
            **base._code_object_evidence(image.payload),
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine


def benchmark(
    radiance_module: Path, radiance_revision: str, tessera_opt: Path,
    *, warmup: int = 6, trials: int = 11, iterations: int = 12,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("B cache ablation requires selected gfx1201")
    if os.environ.get("RADIANCE_MXFP4_WPERM") != "1":
        raise ValueError("matched Radiance requires fragment-order WPERM=1")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("B cache ablation requires HIP")
    radiance = base._load_radiance(radiance_module)
    source = variant_source()
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    payload = _compile_variant(source)
    rows: list[dict[str, object]] = []
    for case in (
        base.Case("prefill", 256, 5120, 8704),
        base.Case("prefill", 1024, 17408, 5120),
    ):
        inputs = base._logical_inputs(case)
        exact = base._tessera_engine(hip, case, inputs, 3, None, None)
        baseline, folded = folded_bench.folded_engine(
            hip, case, inputs, 3, tessera_opt=tessera_opt,
        )
        variant = _variant_engine(hip, case, inputs, folded, payload, source_sha)
        independent = base._radiance_engine(hip, radiance, case, inputs, 3)
        engines = (exact, baseline, variant, independent)
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sampled_rows, sampled_cols, reference = (
                base._sampled_exact_reference(case, inputs)
            )
            np.testing.assert_array_equal(
                outputs["tessera"][np.ix_(sampled_rows, sampled_cols)],
                reference,
            )
            if not folded.lossless:
                raise RuntimeError("B cache ablation inputs must fold losslessly")
            for name, output in outputs.items():
                np.testing.assert_array_equal(
                    output.view(np.uint16), outputs["tessera"].view(np.uint16),
                    err_msg=f"{case.label}: {name} changed BF16 output",
                )
            samples = base._measure_interleaved(
                hip, list(engines), warmup=warmup, trials=trials,
                iterations=iterations,
            )
            for engine in engines:
                rows.append({
                    "case": case.label, "engine": engine.name,
                    "median_ms": statistics.median(samples[engine]),
                    "samples_ms": samples[engine],
                    "output_sha256": hashlib.sha256(
                        outputs[engine.name].view(np.uint8),
                    ).hexdigest(),
                    "metadata": engine.metadata,
                })
        finally:
            for engine in reversed(engines):
                engine.close()
    return {
        "schema": "tessera.rocm.gfx1201_folded_b_cache_ablation.v1",
        "source_revision": base._git_revision(),
        "benchmark_sha256": base._sha256(Path(__file__)),
        "generator_sha256": base._sha256(
            ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py"
        ),
        "variant_source_sha256": source_sha,
        "radiance_revision": radiance_revision,
        "radiance_binary_sha256": base._sha256(radiance_module),
        "radiance_wperm": 1,
        "selected": "tessera_folded",
        "method": "alternating_interleaved_hip_events",
        "phase_attribution_admissible": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = benchmark(
        args.radiance_module, args.radiance_revision, args.tessera_opt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
