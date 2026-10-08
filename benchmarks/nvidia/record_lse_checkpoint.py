"""Record the exact-SM120 saved-LSE versus recompute checkpoint packet.

The candidates are deliberately separate forward and backward native packages:
the saved lane owns f32[B,Hq,Sq] as an output/input pointer, while recompute
has no such buffer.  This recorder never promotes a selector; it records both
CUDA-event and end-to-end domains for the policy decision.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import _nvidia_native_descriptor_device_latency, _nvidia_native_descriptor_resources, launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_checkpoint_native import (
    _backward_module, _forward_module, _reference,
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray,
                  *, rtol: float = 4e-5, atol: float = 4e-5) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        error = float(np.max(np.abs(actual - expected)))
        raise RuntimeError(
            f"{name} failed the fp64 oracle: max_abs_error={error:.8g}, "
            f"rtol={rtol}, atol={atol}"
        )
    return float(np.max(np.abs(actual - expected)))


def _shape(text: str) -> tuple[int, int, int, int, int, int, int]:
    values = tuple(int(value) for value in text.split("x"))
    if len(values) != 7 or min(values) < 1:
        raise argparse.ArgumentTypeError("shapes must be positive BxHqxHkvxSqxSkxDxDv")
    b, hq, hkv, _, _, _, _ = values
    if hq % hkv:
        raise argparse.ArgumentTypeError("Hq must be divisible by Hkv")
    return values


def _compile(module):
    return compile_graph_module(
        module, source_origin="NVIDIA-LSE-1", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )


def _window_repetitions(cap: int, pilot_ms: float, window_ms: float) -> int:
    """Choose a bounded count; event and wall domains calibrate independently."""
    if type(cap) is not int or cap <= 0:
        raise ValueError("window repetition cap must be a positive integer")
    if not math.isfinite(pilot_ms) or pilot_ms < 0:
        raise ValueError("pilot latency must be finite and nonnegative")
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("adaptive window must be finite and positive")
    return cap if pilot_ms == 0 else min(cap, max(1, math.ceil(window_ms / pilot_ms)))


def _sample(bundle, module, args: dict[str, object], *, samples: int, reps: int, warmup: int, expected_outputs: dict[str, np.ndarray], window_ms: float | None = None) -> dict[str, object]:
    descriptor, image = bundle.launch_descriptor, bundle.native_image
    assert descriptor is not None and image is not None
    provenance = descriptor.provenance
    provenance_keys = (
        "graph_ir_digest", "schedule_digest", "schedule_ir_digest",
        "tile_ir_digest", "target_ir_digest",
    )
    complete_provenance = all(provenance.get(key) for key in provenance_keys)
    if not complete_provenance:
        missing = [key for key in provenance_keys if not provenance.get(key)]
        raise RuntimeError(
            f"{descriptor.entry_symbol} checkpoint route lacks compiler ancestry: {missing}"
        )
    if provenance["target_ir_digest"] != image.target_ir_digest:
        raise RuntimeError("checkpoint Target digest disagrees with packaged native image")
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    if not launch(artifact, args)["ok"]:
        raise RuntimeError(f"{descriptor.entry_symbol} smoke launch failed")
    def validate_outputs():
        return {name: _assert_close(f"{descriptor.entry_symbol} timed {name}", args[name], expected)
                for name, expected in expected_outputs.items()}
    validate_outputs()
    event_reps, wall_reps, event_warmup = reps, reps, warmup
    calibration: dict[str, float] = {}
    if window_ms is not None:
        print(json.dumps({"entry": descriptor.entry_symbol, "phase": "calibrate"}),
              file=sys.stderr, flush=True)
        for name in expected_outputs:
            cast(np.ndarray, args[name]).fill(np.nan)
        pilot_event = _nvidia_native_descriptor_device_latency(
            image, descriptor, args, reps=1, warmup=0)
        validate_outputs()
        started = time.perf_counter()
        if not launch(artifact, args)["ok"]:
            raise RuntimeError("adaptive wall calibration launch failed")
        pilot_wall = (time.perf_counter() - started) * 1e3
        validate_outputs()
        event_reps = _window_repetitions(reps, pilot_event, window_ms)
        wall_reps = _window_repetitions(reps, pilot_wall, window_ms)
        event_warmup = _window_repetitions(warmup, pilot_event, window_ms) if warmup else 0
        calibration = {"device_event_ms": pilot_event, "end_to_end_ms": pilot_wall}
    device = []
    event_errors = []
    for sample in range(samples):
        print(json.dumps({"entry": descriptor.entry_symbol,
                          "phase": "device_event", "sample": sample}),
              file=sys.stderr, flush=True)
        # Poison only output buffers so a missing profiler readback cannot
        # accidentally validate values from the earlier correctness launch.
        for name in expected_outputs:
            cast(np.ndarray, args[name]).fill(np.nan)
        device.append(_nvidia_native_descriptor_device_latency(
            image, descriptor, args, reps=event_reps, warmup=event_warmup))
        event_errors.append(validate_outputs())
    e2e: list[float] = []
    wall_errors = []
    for sample in range(samples):
        print(json.dumps({"entry": descriptor.entry_symbol,
                          "phase": "end_to_end", "sample": sample}),
              file=sys.stderr, flush=True)
        started = time.perf_counter()
        for _ in range(wall_reps):
            if not launch(artifact, args)["ok"]:
                raise RuntimeError(f"{descriptor.entry_symbol} end-to-end launch failed")
        e2e.append((time.perf_counter() - started) * 1e3 / wall_reps)
        wall_errors.append(validate_outputs())
    return {
        "entry": descriptor.entry_symbol,
        "abi_id": descriptor.abi_id,
        "device_event_repetitions": event_reps,
        "device_event_warmup": event_warmup,
        "end_to_end_repetitions": wall_reps,
        "adaptive_window_ms": window_ms,
        "calibration": calibration,
        "device_event_samples_ms": device,
        "device_event_median_ms": _median(device),
        "end_to_end_samples_ms": e2e,
        "end_to_end_median_ms": _median(e2e),
        "resources": _nvidia_native_descriptor_resources(image, descriptor, block_size=128),
        "workspace_bytes": descriptor.workspace.bytes,
        "timed_event_output_errors": event_errors,
        "timed_wall_output_errors": wall_errors,
        "event_readback": "after_stop_event_excluded_from_latency",
        "pipeline": (
            "GraphIR->ScheduleIR->TileIR->NVIDIA Target IR->native image"
            if complete_provenance else "recompute_comparator_package"
        ),
        "compiler_provenance_complete": complete_provenance,
        "compiler_provenance": {
            key: provenance[key] for key in provenance_keys if key in provenance
        },
    }


def _paired_sample(bundles, modules, forward_args, backward_args, *, samples,
                   reps, expected_o, expected_lse, expected_grads, window_ms=None):
    """Matched diagnostic two-package wall calls; not a production launch loop."""
    artifacts = {stage: {kind: compile_result_from_bundle(bundle, module=modules[stage][kind]).to_runtime_artifact()
                         for kind, bundle in variants.items()}
                 for stage, variants in bundles.items()}
    ancestry: dict[str, dict[str, dict[str, object]]] = {}
    for stage, variants in bundles.items():
        ancestry[stage] = {}
        for kind, bundle in variants.items():
            descriptor, image = bundle.launch_descriptor, bundle.native_image
            if descriptor is None or image is None:
                raise RuntimeError("paired checkpoint lacks a native package")
            hashes = {key: descriptor.provenance.get(key) for key in
                      ("graph_ir_digest", "schedule_digest", "schedule_ir_digest", "tile_ir_digest", "target_ir_digest")}
            if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
                raise RuntimeError("paired checkpoint lacks complete compiler ancestry")
            if hashes["target_ir_digest"] != image.target_ir_digest:
                raise RuntimeError("paired checkpoint Target digest disagrees with image")
            ancestry[stage][kind] = hashes
    counts = {kind: reps for kind in ("saved", "recompute")}
    timings: dict[str, list[float]] = {kind: [] for kind in counts}
    errors: dict[str, list[dict[str, float]]] = {kind: [] for kind in counts}
    pilots = {}

    def call(kind):
        for stage, arguments in (("forward", forward_args[kind]), ("backward", backward_args[kind])):
            result = launch(artifacts[stage][kind], arguments)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"{kind} paired {stage} failed: {result}")

    def validate(kind):
        values = {"o": _assert_close(f"{kind} paired O", forward_args[kind]["o"], expected_o)}
        if kind == "saved":
            values["row_lse"] = _assert_close("paired LSE", forward_args[kind]["row_lse"], expected_lse)
        for name, expected in zip(("dq", "dk", "dv"), expected_grads, strict=True):
            values[name] = _assert_close(f"{kind} paired {name}", backward_args[kind][name], expected)
        return values

    def poison(kind):
        cast(np.ndarray, forward_args[kind]["o"]).fill(np.nan)
        if kind == "saved":
            cast(np.ndarray, forward_args[kind]["row_lse"]).fill(np.nan)
        for name in ("dq", "dk", "dv"):
            cast(np.ndarray, backward_args[kind][name]).fill(np.nan)

    for kind in counts:
        poison(kind)
        started = time.perf_counter()
        call(kind)
        pilots[kind] = (time.perf_counter() - started) * 1e3
        validate(kind)
        if window_ms is not None:
            counts[kind] = _window_repetitions(reps, pilots[kind], window_ms)
    for sample in range(samples):
        order = ("saved", "recompute") if sample % 2 == 0 else ("recompute", "saved")
        for kind in order:
            print(json.dumps({"phase": "paired_wall", "kind": kind, "sample": sample}),
                  file=sys.stderr, flush=True)
            poison(kind)
            started = time.perf_counter()
            for _ in range(counts[kind]):
                call(kind)
            timings[kind].append((time.perf_counter() - started) * 1e3 / counts[kind])
            errors[kind].append(validate(kind))
    return {"timing_scope": "two checked host-buffer native package calls; upload/download and synchronization included; host allocations, poisoning and oracle excluded; not resident-frame timing",
            "compiler_provenance": ancestry,
            "order": "alternating_saved_recompute", "samples_ms": timings,
            "medians_ms": {kind: _median(values) for kind, values in timings.items()},
            "repetitions": counts, "pilot_ms": pilots, "output_errors": errors,
            "saved_residual_bytes": cast(np.ndarray, forward_args["saved"]["o"]).nbytes + cast(np.ndarray, forward_args["saved"]["row_lse"]).nbytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--adaptive-window-ms", type=float,
                        help="calibrate event/wall counts separately; reps and warmup become caps")
    parser.add_argument("--shapes", type=_shape, nargs="+", default=(
        (1, 2, 1, 3, 4, 4, 3),
        (1, 2, 1, 8, 8, 8, 8),
        (1, 4, 2, 15, 17, 16, 12),
        (1, 4, 2, 127, 131, 64, 64),
        (1, 4, 2, 256, 256, 64, 64),
    ))
    parser.add_argument("--paired-only", action="store_true", help="measure matched complete host-buffer pairs instead of individual kernels")
    parser.add_argument("--bias", action="store_true", help="Exact [B,Hq,Sq,Sk] f32 bias operand")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.reps <= 0 or args.warmup < 0:
        raise ValueError("samples/reps must be positive and warmup nonnegative")
    if args.adaptive_window_ms is not None and (
            not math.isfinite(args.adaptive_window_ms) or args.adaptive_window_ms <= 0):
        raise ValueError("adaptive window must be finite and positive")
    if not nvidia_cuda_host_ready():
        print(json.dumps({"status": "skipped", "reason": "host WSL CUDA SM120 unavailable"}))
        return 0
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if len(device.splitlines()) != 1 or device.split(",")[-1].strip() != "12.0":
        raise RuntimeError("one selected exact SM120 device is required")
    rng = np.random.default_rng(120_101)
    packets: list[dict[str, object]] = []
    for shape in args.shapes:
        print(json.dumps({"shape": shape, "phase": "compile"}),
              file=sys.stderr, flush=True)
        b, hq, hkv, sq, sk, d, dv = shape
        modules = {
            "forward": {kind: _forward_module(saved=kind == "saved", shape=shape, bias=args.bias) for kind in ("recompute", "saved")},
            "backward": {kind: _backward_module(saved=kind == "saved", shape=shape, bias=args.bias) for kind in ("recompute", "saved")},
        }
        bundles = {stage: {kind: _compile(module) for kind, module in variants.items()}
                   for stage, variants in modules.items()}
        q = np.ascontiguousarray((rng.normal(size=(b, hq, sq, d)) * .2).astype(np.float32))
        k = np.ascontiguousarray((rng.normal(size=(b, hkv, sk, d)) * .2).astype(np.float32))
        v = np.ascontiguousarray((rng.normal(size=(b, hkv, sk, dv)) * .2).astype(np.float32))
        do = np.ascontiguousarray((rng.normal(size=(b, hq, sq, dv)) * .2).astype(np.float32))
        bias = np.ascontiguousarray((rng.normal(size=(b, hq, sq, sk)) * .3).astype(np.float32)) if args.bias else None
        scalars = {**({"bias": bias} if args.bias else {}), "B": b, "Hq": hq, "Hkv": hkv, "Sq": sq, "Sk": sk, "D": d, "Dv": dv}
        row_lse = np.empty((b, hq, sq), dtype=np.float32)
        forward_args = {
            "saved": {"q": q, "k": k, "v": v, "o": np.empty((b, hq, sq, dv), np.float32), "row_lse": row_lse, **scalars},
            "recompute": {"q": q, "k": k, "v": v, "o": np.empty((b, hq, sq, dv), np.float32), **scalars},
        }
        saved_artifact = compile_result_from_bundle(bundles["forward"]["saved"], module=modules["forward"]["saved"]).to_runtime_artifact()
        if not launch(saved_artifact, forward_args["saved"])["ok"]:
            raise RuntimeError("saved-LSE producer setup failed")
        backward_args = {
            "saved": {"do": do, "q": q, "k": k, "v": v, "output": forward_args["saved"]["o"], "row_lse": row_lse,
                      "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v), **scalars},
            "recompute": {"do": do, "q": q, "k": k, "v": v,
                          "dq": np.empty_like(q), "dk": np.empty_like(k), "dv": np.empty_like(v), **scalars},
        }
        # Gate every shape and both checkpoint modes on an independent
        # numerical oracle before collecting any timing samples.
        expected_o, expected_lse, expected_grads = _reference(q, k, v, do, bias)
        correctness: dict[str, object] = {"oracle": "independent_fp64_attention_and_gradients"}
        for kind in ("saved", "recompute"):
            forward_result = launch(
                compile_result_from_bundle(
                    bundles["forward"][kind], module=modules["forward"][kind]
                ).to_runtime_artifact(),
                forward_args[kind],
            )
            if not forward_result["ok"]:
                raise RuntimeError(
                    f"{kind}-LSE forward correctness launch failed: "
                    f"{forward_result.get('reason')}"
                )
            forward_errors = {
                "output_max_abs": _assert_close(
                    f"{kind}-LSE forward output",
                    forward_args[kind]["o"], expected_o,
                ),
            }
            if kind == "saved":
                forward_errors["row_lse_max_abs"] = _assert_close(
                    "saved-LSE forward row_lse",
                    forward_args[kind]["row_lse"], expected_lse,
                )
            correctness[f"forward_{kind}"] = forward_errors

        for kind in ("saved", "recompute"):
            backward_result = launch(
                compile_result_from_bundle(
                    bundles["backward"][kind], module=modules["backward"][kind]
                ).to_runtime_artifact(),
                backward_args[kind],
            )
            if not backward_result["ok"]:
                raise RuntimeError(
                    f"{kind}-LSE backward correctness launch failed: "
                    f"{backward_result.get('reason')}"
                )
            grad_errors = [
                _assert_close(f"{kind}-LSE {name}", actual, expected)
                for name, actual, expected in zip(
                    ("dq", "dk", "dv"),
                    (backward_args[kind]["dq"], backward_args[kind]["dk"],
                     backward_args[kind]["dv"]),
                    expected_grads, strict=True,
                )
            ]
            correctness[f"backward_{kind}_gradient_max_abs"] = max(grad_errors)

        rows = {} if args.paired_only else {
            stage: {kind: _sample(bundles[stage][kind], modules[stage][kind], stage_args,
                                  samples=args.samples, reps=args.reps, warmup=args.warmup, window_ms=args.adaptive_window_ms,
                                  expected_outputs=({"o": expected_o, **({"row_lse": expected_lse} if kind == "saved" else {})}
                                                    if stage == "forward" else dict(zip(("dq", "dk", "dv"), expected_grads, strict=True))))
                    for kind, stage_args in variants.items()}
            for stage, variants in (("forward", forward_args), ("backward", backward_args))
        }
        for variants in rows.values():
            for result in variants.values():
                result["correctness"] = "passed_before_and_after_every_event_and_wall_window"
        paired = _paired_sample(bundles, modules, forward_args, backward_args,
            samples=args.samples, reps=args.reps, window_ms=args.adaptive_window_ms,
            expected_o=expected_o, expected_lse=expected_lse, expected_grads=expected_grads)
        packets.append({"shape": list(shape), "bias": args.bias, "correctness": correctness, "rows": rows, "paired": paired})
    packet = {
        "schema": "tessera.nvidia.lse_checkpoint.benchmark.v6",
        "work_item": "NVIDIA-LSE-1",
        "device": device,
        "target": "nvidia_sm120",
        "method": {
                   "saved_route_pipeline": "GraphIR->ScheduleIR->TileIR->NVIDIA Target IR->native image",
                   "recompute_route_note": "all comparator and saved arms require verified Graph/Schedule/Tile ancestry",
                   "timing_domains": ["cuda_event", "end_to_end"], "samples": args.samples,
                   "repetitions": args.reps, "warmup": args.warmup, "adaptive_window_ms": args.adaptive_window_ms,
                   "repetition_policy": "per_arm_independent_domains" if args.adaptive_window_ms else "fixed",
                   "native_paired_ad_policy": "saved_O_LSE", "standalone_comparators": ["saved", "recompute"], "paired_only": args.paired_only, "ncu_required": True},
        "packets": packets,
        "selector_changed": False,
        "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "nvidia_compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        "native_launch_library_sha256": hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                "src/compiler/programming_model/lib/NativeCheckpoint.h",
                "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
                "src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td",
                "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
                "python/tessera/compiler/scheduled_checkpoint.py",
                "python/tessera/compiler/driver.py",
                "python/tessera/compiler/nvidia_native.py",
                "python/tessera/compiler/resident_attention.py",
                "python/tessera/runtime.py",
                "benchmarks/nvidia/record_lse_checkpoint.py",
                "tests/device/nvidia/test_lse_checkpoint_native.py",
            )
        },
    }
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
