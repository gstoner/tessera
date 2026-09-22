#!/usr/bin/env python3
"""Diagnostic, non-selecting per-CTA phase attribution for folded MXFP4 prefill.

The trace uses same-CTA wall-clock deltas only. It does not claim cross-CU
clock validation, stall attribution, or IKF-P0/P3 closure.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import (
    emit_mxfp4_folded_prefill_hip, prepare_folded_weights,
)
from tessera.compiler.rocm_mxfp4_native import _extract_gfx1201_hsaco, _rocm_hipcc
from tessera.compiler.rocm_native import _rocm_path
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base


def _trace_image() -> bytes:
    rocm = _rocm_path()
    compiler = _rocm_hipcc(rocm)
    if compiler is None:
        raise RuntimeError("gfx1201 phase trace requires hipcc")
    with tempfile.TemporaryDirectory(prefix="tessera-folded-phase-") as temporary:
        root = Path(temporary)
        source = root / "phase.hip"
        bundle = root / "phase.hipfb"
        hsaco = root / "phase.hsaco"
        source.write_text(emit_mxfp4_folded_prefill_hip())
        command = [
            str(compiler), "-x", "hip", "-O3", "--genco",
            "--offload-arch=gfx1201", f"--rocm-path={rocm}",
            "-DTESSERA_FOLDED_PHASE_TRACE=1", str(source), "-o", str(bundle),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle.is_file():
            raise RuntimeError(result.stderr.strip() or "phase HIP compilation failed")
        return _extract_gfx1201_hsaco(bundle, hsaco, rocm)


def _trace_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    image: bytes, copies: int,
) -> base._Engine:
    folded = prepare_folded_weights(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    if not folded.lossless:
        raise RuntimeError("phase comparison requires lossless-fold inputs")
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), image) != 0:
        raise RuntimeError("phase HSACO load failed")
    entry = b"tessera_mxfp4_folded_prefill"
    if hip.hipModuleGetFunction(ctypes.byref(function), module, entry) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("phase kernel entry missing")
    slots = ((case.m + 255) // 256) * ((case.n + 63) // 64)
    arrays = (
        inputs["a"], folded.weight_bytes, inputs["a_scale"],
        folded.row_reference, inputs["output"], np.zeros((slots, 4), dtype=np.uint64),
    )
    device_copies = base._copies(hip, arrays, copies)

    def launch(bundle: base._DeviceArrays) -> None:
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in bundle.device[:5]),
            ctypes.c_int64(case.m), ctypes.c_int64(case.n), ctypes.c_int64(case.k),
            ctypes.c_void_p(bundle.device[5].value),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        rc = hip.hipModuleLaunchKernel(
            function, (case.n + 63) // 64, (case.m + 255) // 256, 1,
            256, 1, 1, 0, None, arguments, None,
        )
        if rc != 0:
            raise RuntimeError(f"phase launch failed rc={rc}")

    engine = base._Engine(
        "folded_phase_l2", hip, device_copies, launch, 4,
        {"instr_level": 2, "image_sha256": hashlib.sha256(image).hexdigest()},
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine


def _phase_summary(slots: np.ndarray) -> dict[str, object]:
    if slots.ndim != 2 or slots.shape[1] != 4 or not np.all(slots[:, 0]):
        raise RuntimeError("phase trace has missing or malformed CTA slots")
    if np.any(slots[:, 1] < slots[:, 0]):
        raise RuntimeError("same-CTA wall clock moved backwards")
    if np.any(slots[:, 2] == 0) or np.any(slots[:, 3] == 0):
        raise RuntimeError("phase trace contains an empty stage")
    elapsed = slots[:, 1] - slots[:, 0]
    copy = slots[:, 2]
    compute = slots[:, 3]
    if np.any(copy + compute > elapsed):
        raise RuntimeError("phase sums exceed their same-CTA elapsed interval")
    return {
        "cta_slots": int(len(slots)),
        "copy_ticks_median": float(np.median(copy)),
        "compute_ticks_median": float(np.median(compute)),
        "elapsed_ticks_median": float(np.median(elapsed)),
        "copy_fraction_median": float(np.median(copy / elapsed)),
        "compute_fraction_median": float(np.median(compute / elapsed)),
        "clock_scope": "same_cta_delta_only_cross_cu_unvalidated",
    }


def measure(case: base.Case, *, trials: int = 11, iterations: int = 12) -> dict[str, object]:
    if case.workload != "prefill" or rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("phase attribution requires gfx1201 prefill")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("phase attribution requires HIP")
    inputs = base._logical_inputs(case)
    image = _trace_image()
    production, _ = folded_bench.folded_engine(hip, case, inputs, 3)
    traced = _trace_engine(hip, case, inputs, image, 3)
    try:
        expected = production.output()
        np.testing.assert_array_equal(traced.output().view(np.uint16), expected.view(np.uint16))
        slots = traced.copies[0].download(5)
        phases = _phase_summary(slots)
        samples = base._measure_interleaved(
            hip, [production, traced], warmup=6, trials=trials,
            iterations=iterations,
        )
        plain = samples[production]
        instrumented = samples[traced]
        overhead = statistics.median(instrumented) / statistics.median(plain) - 1.0
        return {
            "schema": "tessera.rocm.gfx1201_folded_phase_diagnostic.v1",
            "case": case.label,
            "device": base._selected_device_name(hip),
            "source_revision": base._git_revision(),
            "generator_sha256": base._sha256(
                Path(__file__).resolve().parents[2]
                / "python/tessera/compiler/rocm_mxfp4_folded.py"
            ),
            "recorder_sha256": base._sha256(Path(__file__)),
            "production_image_sha256": production.metadata["image_sha256"],
            "trace_image_sha256": hashlib.sha256(image).hexdigest(),
            "trace_isa_resources": base._code_object_evidence(image),
            "instrumentation_level": 2,
            "promotion_eligible": False,
            "ikf_p0_complete": False,
            "timing_order": "alternating_interleaved",
            "production_ms": plain,
            "instrumented_ms": instrumented,
            "median_overhead_fraction": overhead,
            "phases": phases,
        }
    finally:
        traced.close()
        production.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=base._parse_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = measure(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
