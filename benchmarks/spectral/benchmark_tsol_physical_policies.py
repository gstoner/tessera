#!/usr/bin/env python3
"""Exact-host TSOL physical-policy packet for Zen 5 or gfx1151.

The v4 public package ABI synchronizes before returning, so the reported
interval is synchronized host wall time.  It is selector-eligible for x86.
Under WSL/DXG it is correctness and comparative host-overhead evidence only,
not a replacement for a bare-metal HIP device-event interval.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyCase:
    name: str
    op_name: str
    operands: tuple[np.ndarray, ...]
    axis: int = -1
    hop: int | None = None
    storage: str = "f32"
    normalization: str = "backward"
    seed_operands: tuple[np.ndarray, ...] | None = None
    signature: tuple[tuple[int | None, ...], ...] | None = None
    bounds: tuple[tuple[int, ...], ...] | None = None


def _dtype(storage: str):
    if storage == "f32":
        return np.float32
    if storage == "f16":
        return np.float16
    import ml_dtypes

    return ml_dtypes.bfloat16


def _artifact(target: str, case: PolicyCase):
    from tessera import runtime as rt
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    seed = case.seed_operands or case.operands
    contract = lower_scheduled_spectral(
        target=target,
        op_name=case.op_name,
        input_shapes=tuple(tuple(value.shape) for value in seed),
        axis=case.axis,
        hop=case.hop,
        input_signature=case.signature,
        shape_bounds=case.bounds,
        storage=case.storage,
        normalization=case.normalization,
    ).to_metadata()
    return rt.RuntimeArtifact(
        metadata={
            "target": target,
            "compiler_path": (
                "x86_spectral_compiled"
                if target == "x86"
                else "rocm_spectral_compiled"
            ),
            "executable": True,
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "arg_names": [f"a{index}" for index in range(len(case.operands))],
            "output_name": "o",
            "scheduled_spectral": contract,
        }
    )


def _stft_reference(x: np.ndarray, window: np.ndarray, hop: int, axis: int):
    moved = np.moveaxis(x.astype(np.float32), axis, -1)
    win = window.astype(np.float32)
    frames = np.stack(
        [
            moved[..., start : start + win.size] * win
            for start in range(0, moved.shape[-1] - win.size + 1, hop)
        ],
        axis=-2,
    )
    result = np.fft.rfft(frames.astype(np.float32), axis=-1).astype(np.complex64)
    return np.moveaxis(result, (-2, -1), (axis, axis + 1))


def _istft_reference(
    spectra: np.ndarray, window: np.ndarray, hop: int, axis: int
) -> np.ndarray:
    frame_axis = axis - 1
    moved = np.moveaxis(spectra.astype(np.complex64), (frame_axis, axis), (-2, -1))
    win = window.astype(np.float32)
    frames = np.fft.irfft(moved, n=win.size, axis=-1).astype(np.float32)
    samples = (frames.shape[-2] - 1) * hop + win.size
    output = np.zeros(frames.shape[:-2] + (samples,), dtype=np.float32)
    weight = np.zeros_like(output)
    for index in range(frames.shape[-2]):
        start = index * hop
        output[..., start : start + win.size] += frames[..., index, :] * win
        weight[..., start : start + win.size] += win * win
    output /= np.maximum(weight, np.float32(1.0e-12))
    return np.moveaxis(output, -1, frame_axis)


def _reference(case: PolicyCase) -> np.ndarray:
    operands = case.operands
    axis = case.axis if case.axis >= 0 else operands[0].ndim + case.axis
    if case.op_name == "tessera.spectral_filter":
        return (operands[0] * operands[1]).astype(np.complex64)
    if case.op_name == "tessera.dct":
        source = operands[0].astype(np.float32)
        n = source.shape[axis]
        mirrored = np.concatenate([source, np.flip(source, axis)], axis).astype(
            np.complex64
        )
        output = np.take(np.fft.fft(mirrored, axis=axis).real, np.arange(n), axis)
        transform_length = 2 * n
    elif case.op_name == "tessera.spectral_conv":
        left = np.moveaxis(operands[0].astype(np.float32), axis, -1)
        right = np.moveaxis(operands[1].astype(np.float32), axis, -1)
        rows = [
            np.convolve(a, b)
            for a, b in zip(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1]))
        ]
        output = np.moveaxis(
            np.stack(rows).reshape(left.shape[:-1] + (len(rows[0]),)), -1, axis
        )
        transform_length = 1
    elif case.op_name == "tessera.stft":
        output = _stft_reference(operands[0], operands[1], int(case.hop), axis)
        transform_length = operands[1].size
    else:
        output = _istft_reference(operands[0], operands[1], int(case.hop), axis)
        transform_length = operands[1].size

    if case.normalization != "backward" and case.op_name != "tessera.spectral_conv":
        root = np.sqrt(float(transform_length))
        if case.op_name == "tessera.istft":
            output *= transform_length if case.normalization == "forward" else root
        else:
            output *= 1.0 / transform_length if case.normalization == "forward" else 1.0 / root
    return np.asarray(
        output,
        dtype=np.complex64 if case.op_name == "tessera.stft" else np.float32,
    )


def _cast_operands(values: tuple[np.ndarray, ...], storage: str, op_name: str):
    if op_name == "tessera.spectral_filter":
        return tuple(np.asarray(value, dtype=np.complex64) for value in values)
    dtype = _dtype(storage)
    return tuple(
        np.asarray(value, dtype=np.complex64 if op_name == "tessera.istft" and index == 0 else dtype)
        for index, value in enumerate(values)
    )


def _cases() -> list[PolicyCase]:
    rng = np.random.default_rng(20260806)

    def real(shape):
        return rng.standard_normal(shape).astype(np.float32)

    def spectrum(batch=8, samples=128, window_length=32, hop=16):
        window = np.hanning(window_length).astype(np.float32)
        signal = real((batch, samples))
        return _stft_reference(signal, window, hop, 1), window

    baseline = [
        PolicyCase("baseline_filter", "tessera.spectral_filter", (
            (real((8, 17)) + 1j * real((8, 17))).astype(np.complex64),
            (real((8, 17)) + 1j * real((8, 17))).astype(np.complex64),
        )),
        PolicyCase("baseline_dct", "tessera.dct", (real((8, 64)),)),
        PolicyCase("baseline_conv", "tessera.spectral_conv", (real((8, 64)), real((8, 17)))),
        PolicyCase("baseline_stft", "tessera.stft", (real((8, 128)), np.hanning(32).astype(np.float32)), hop=16),
    ]
    spectra, window = spectrum()
    baseline.append(PolicyCase("baseline_istft", "tessera.istft", (spectra, window), axis=2, hop=16))

    dynamic: list[PolicyCase] = []
    for case in baseline:
        actual = case.operands
        seed = tuple(value[:4] if value.ndim > 1 else value for value in actual)
        signatures = tuple(
            ((None,) + value.shape[1:]) if value.ndim > 1 else value.shape
            for value in actual
        )
        bounds = tuple(
            ((8,) + value.shape[1:]) if value.ndim > 1 else value.shape
            for value in actual
        )
        dynamic.append(PolicyCase(
            case.name.replace("baseline", "bounded_dynamic"), case.op_name,
            actual, axis=case.axis, hop=case.hop, seed_operands=seed,
            signature=signatures, bounds=bounds,
        ))

    axis_cases = [
        PolicyCase("axis0_dct", "tessera.dct", (real((64, 8)),), axis=0),
        PolicyCase("axis0_conv", "tessera.spectral_conv", (real((64, 8)), real((17, 8))), axis=0),
        PolicyCase("axis0_stft", "tessera.stft", (real((128, 8)), np.hanning(32).astype(np.float32)), axis=0, hop=16),
    ]
    axis_spectra = _stft_reference(real((128, 8)), np.hanning(32).astype(np.float32), 16, 0)
    axis_cases.append(PolicyCase(
        "axis1_istft", "tessera.istft",
        (axis_spectra, np.hanning(32).astype(np.float32)), axis=1, hop=16,
    ))

    storage_cases: list[PolicyCase] = []
    for storage in ("f16", "bf16"):
        for case in baseline[1:]:
            storage_cases.append(PolicyCase(
                f"storage_{storage}_{case.op_name.removeprefix('tessera.')}",
                case.op_name, _cast_operands(case.operands, storage, case.op_name),
                axis=case.axis, hop=case.hop, storage=storage,
            ))

    normalization_cases: list[PolicyCase] = []
    for normalization in ("forward", "ortho"):
        for case in (baseline[1], baseline[3], baseline[4]):
            normalization_cases.append(PolicyCase(
                f"normalization_{normalization}_{case.op_name.removeprefix('tessera.')}",
                case.op_name, case.operands, axis=case.axis, hop=case.hop,
                normalization=normalization,
            ))

    combined = []
    for storage in ("f16", "bf16"):
        actual = _cast_operands((real((64, 6)),), storage, "tessera.dct")
        seed = tuple(value[:, :4] for value in actual)
        combined.append(PolicyCase(
            f"combined_dynamic_axis_ortho_{storage}_dct", "tessera.dct", actual,
            axis=0, storage=storage, normalization="ortho", seed_operands=seed,
            signature=((64, None),), bounds=((64, 8),),
        ))
    return baseline + dynamic + axis_cases + storage_cases + normalization_cases + combined


def _run(target: str, iterations: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.scheduled_spectral import validate_scheduled_spectral_metadata

    rows = []
    for case in _cases():
        artifact = _artifact(target, case)
        initial = artifact.metadata["scheduled_spectral"]
        executed = validate_scheduled_spectral_metadata(
            initial, input_shapes=[value.shape for value in case.operands]
        )
        respecialized = initial["schedule_digest"] != executed["schedule_digest"]
        if executed["shape_policy"] == "bounded_runtime_specialization_v1" and not respecialized:
            raise RuntimeError(
                f"{case.name} did not exercise a distinct bounded specialization"
            )
        start = time.perf_counter_ns()
        first = rt.launch(artifact, case.operands)
        cold_ns = time.perf_counter_ns() - start
        if not first.get("ok"):
            raise RuntimeError(first.get("reason", "TSOL launch failed"))
        samples = []
        output = None
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = rt.launch(artifact, case.operands)
            samples.append(time.perf_counter_ns() - start)
            if not result.get("ok"):
                raise RuntimeError(result.get("reason", "TSOL launch failed"))
            output = np.asarray(result["output"])
        reference = _reference(case)
        error = np.abs(output.astype(np.complex64) - reference.astype(np.complex64))
        max_error = float(error.max(initial=0.0))
        rms_error = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
        limit = {"f32": 2.0e-2, "f16": 1.5e-1, "bf16": 7.5e-1}[case.storage]
        if max_error > limit:
            raise RuntimeError(
                f"{case.name} max_abs_error={max_error} exceeds {limit}"
            )
        median_ns = statistics.median(samples)
        rows.append({
            "case": case.name,
            "op_name": case.op_name,
            "input_shapes": [list(value.shape) for value in case.operands],
            "output_shape": list(output.shape),
            "axis": case.axis,
            "storage": case.storage,
            "normalization": case.normalization,
            "shape_policy": executed["shape_policy"],
            "template_digest": executed["template_digest"],
            "initial_schedule_digest": initial["schedule_digest"],
            "executed_schedule_digest": executed["schedule_digest"],
            "runtime_respecialized": respecialized,
            "tile_digest": executed["tile_digest"],
            "cold_ms": cold_ns / 1.0e6,
            "warm_median_ms": median_ns / 1.0e6,
            "warm_mad_ms": statistics.median(abs(value - median_ns) for value in samples) / 1.0e6,
            "output_melems_per_s": output.size / (median_ns / 1.0e3),
            "max_abs_error": max_error,
            "rms_error": rms_error,
            "error_limit": limit,
            "abi_storage": executed["abi_storage"],
            "storage_boundary": executed["storage_conversion"],
            "axis_packing": executed["axis_packing"],
            "native_entry": executed["native_entry"],
            "fusion_topology": executed["fusion_topology"],
            "child_fft_digests": executed["child_fft_digests"],
            "child_physical_lengths": [
                child["physical_length"] for child in executed["child_ffts"]
            ],
            "workspace_bytes": executed["workspace_bytes"],
        })
    return {
        "schema": "tessera.tsol_physical_policy_evidence.v2",
        "target": target,
        "architecture": "zen5-avx512" if target == "x86" else "gfx1151",
        "host": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "package_abi": (
            "tessera.x86.spectral_composite.v5"
            if target == "x86"
            else "tessera.rocm.spectral_composite.v5"
        ),
        "timing_domain": "synchronized_host_wall",
        "selector_eligible": target == "x86",
        "device_event_follow_up": None if target == "x86" else "bare_metal_required",
        "iterations": iterations,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm"), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(
        _run(args.target, args.iterations), indent=2, sort_keys=True
    ) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
