#!/usr/bin/env python3
"""Single-image A/B row-pitch attribution with independently checked native math.

Device windows include HIP graph dispatch. Public wall time includes package
launch overhead and transfers; neither is a profiler phase measurement.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.ablate_gfx1201_folded_load_schedule import package_engine
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_folded_native_package import compile_for_tool, SOURCES
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock

ROOT = Path(__file__).resolve().parents[2]


def source_state(tool):
    paths = (*SOURCES, "src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/programming_model/ir/ScheduleDialect.cpp",
             "src/compiler/ir/TesseraOps.cpp", "src/compiler/ir/TileDialect.cpp",
             "src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td",
             "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
             str(Path(__file__).relative_to(ROOT)))
    return {"commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "compiler_sha256": base._sha256(tool),
            "sources": {path: base._sha256(ROOT / path) for path in paths}}


def public_launch(package, arrays, shape):
    m, n, k = shape
    lda, ldb = arrays[0].shape[1], arrays[1].shape[1]
    artifact = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    scalars = {"M": m, "N": n, "K": k}
    if "a_row_stride" in package.descriptor.provenance:
        scalars.update(LDA=lda, LDB=ldb)
    start = time.perf_counter_ns()
    result = rt.launch(artifact, {"buffers": dict(zip(
        ("a", "b_folded", "a_scale", "row_reference", "output"), arrays)),
        "scalars": scalars})
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if not result.get("ok") or result.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"checked package launch failed: {result}")
    return elapsed


def run_case(hip, clock, tool, reference_tool, n, k, trials, reverse):
    case = base.Case("prefill", 256, n, k)
    inputs = base._logical_inputs(case)
    folded = prepare_folded_weights(inputs["packed_row_major"], inputs["b_scale"],
                                    allow_approximate=True)
    if not folded.lossless:
        raise RuntimeError("input-pitch packet requires lossless matched operands")
    control = compile_for_tool(tool, inputs["a"], inputs["a_scale"], folded)
    frozen = compile_for_tool(reference_tool, inputs["a"], inputs["a_scale"], folded)
    if control.package.image.payload != frozen.package.image.payload:
        raise RuntimeError("existing contiguous native image changed")
    variants = [("dense", k, k), ("a_only", k + 16, k),
                ("b_only", k, k + 16), ("both", k + 16, k + 16)]
    if reverse:
        variants.reverse()
    programs = {"contiguous_control": control}
    for name, lda, ldb in variants:
        programs[name] = compile_for_tool(tool, inputs["a"], inputs["a_scale"],
            folded, input_row_strides=(lda, ldb))
    images = {programs[name].package.image.payload_digest for name, _, _ in variants}
    if len(images) != 1:
        raise RuntimeError("row pitches changed the native kernel image")
    arrays_by_name, engines, outputs = {}, [], {}
    with ExitStack() as cleanup:
        for name, program in programs.items():
            package = program.package
            lda = package.descriptor.provenance.get("a_row_stride", k)
            ldb = package.descriptor.provenance.get("b_row_stride", k)
            a = np.full((256, lda), 0x7f, np.uint8)
            b = np.full((n, ldb), 0xff, np.uint8)
            a[:, :k], b[:, :k] = inputs["a"], folded.weight_bytes
            arrays = (a, b, inputs["a_scale"], folded.row_reference,
                      np.empty_like(inputs["output"]))
            arrays_by_name[name] = arrays
            engine = package_engine(hip, case, {**inputs, "a": a, "output": arrays[-1]},
                SimpleNamespace(weight_bytes=b, row_reference=folded.row_reference),
                package, name=name, metadata={"receipt": program.route_receipt}, copies=3)
            cleanup.callback(engine.close)
            engines.append(engine)
            outputs[name] = engine.output().copy()
        rows, cols, expected = base._sampled_exact_reference(case, inputs)
        control_output = outputs["contiguous_control"]
        for output in outputs.values():
            np.testing.assert_array_equal(output.view(np.uint16), control_output.view(np.uint16))
            np.testing.assert_array_equal(output[np.ix_(rows, cols)], expected)
        graphs = FoldedGraphWindows(clock)
        cleanup.callback(graphs.close)
        # Capture both clock-bracketed and plain forms; poison/check all three
        # resident outputs before any timings are admitted.
        for engine in engines:
            for bracketed in (True, False):
                for bundle in engine.copies:
                    poison = np.empty_like(control_output)
                    poison.view(np.uint8).fill(255)
                    clock._check(hip.hipMemcpy(bundle.device[4],
                        poison.ctypes.data_as(ctypes.c_void_p), poison.nbytes, 1), "output poison")
                graphs.window(engine, 12, bracketed=bracketed)
                for bundle in engine.copies:
                    np.testing.assert_array_equal(bundle.download(4).view(np.uint16),
                                                  control_output.view(np.uint16))
        samples = {name: [] for name in programs}
        wall = {name: [] for name in programs}
        launches = 192
        for trial in range(trials):
            order = engines if trial % 2 == 0 else list(reversed(engines))
            for engine in order:
                sample = graphs.window(engine, launches, bracketed=True)
                if sample["device_window_ms"] < 5 or sample["device_event_disagreement"] > 0.05:
                    raise RuntimeError(f"device window witness failed: {sample}")
                plain = graphs.window(engine, launches, bracketed=False)
                samples[engine.name].append({"clock": sample, "plain": plain})
        for trial in range(3):
            for engine in (engines if trial % 2 == 0 else list(reversed(engines))):
                name = engine.name
                wall[name].append(public_launch(programs[name].package, arrays_by_name[name], (256, n, k)))
                np.testing.assert_array_equal(arrays_by_name[name][-1].view(np.uint16),
                                              control_output.view(np.uint16))
        for engine in engines:
            for bundle in engine.copies:
                np.testing.assert_array_equal(bundle.download(4).view(np.uint16),
                                              control_output.view(np.uint16))
        return {"shape_mnk": [256, n, k], "same_pitched_image_sha256": next(iter(images)),
            "contiguous_image_sha256": control.package.image.payload_digest,
            "frozen_contiguous_image_identical": True,
            "correctness": "full_bitwise_control_plus_sampled_independent_oracle_and_poisoned_graph_outputs",
            "rows": [{"name": engine.name, "image_evidence": engine.metadata,
                "physical_shapes": [list(a.shape) for a in arrays_by_name[engine.name]],
                "physical_bytes": [a.nbytes for a in arrays_by_name[engine.name]],
                "receipt": programs[engine.name].route_receipt,
                "captured_device_per_launch_median_ms": statistics.median(
                    s["clock"]["device_window_ms"] / launches for s in samples[engine.name]),
                "plain_event_per_launch_median_ms": statistics.median(
                    s["plain"]["event_window_ms"] / launches for s in samples[engine.name]),
                "windows": samples[engine.name], "public_wall_samples_ms": wall[engine.name],
                "public_wall_median_ms": statistics.median(wall[engine.name]),
                "output_sha256": hashlib.sha256(outputs[engine.name].tobytes()).hexdigest()}
                for engine in engines]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--reference-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--n", action="append", type=int)
    parser.add_argument("--k", action="append", type=int)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 3 or not args.reference_opt.is_file():
        raise ValueError("three trials and a frozen reference compiler are required")
    ns, ks = args.n or [8192, 16384], args.k or [4032, 4096, 4160]
    if any(n <= 0 or n % 64 for n in ns) or any(k <= 0 or k % 64 for k in ks):
        raise ValueError("positive N/K must be divisible by 64")
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("launch interception must be absent during timings")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    before = source_state(args.tessera_opt)
    reference_before = base._sha256(args.reference_opt)
    pairs = [(n, k) for n in ns for k in ks]
    if args.reverse:
        pairs.reverse()
    clock = DeviceClock(hip, args.tessera_opt, args.llvm_bin)
    try:
        rows = [run_case(hip, clock, args.tessera_opt, args.reference_opt,
                        n, k, args.trials, args.reverse) for n, k in pairs]
    finally:
        clock.close()
    after = source_state(args.tessera_opt)
    if before != after or reference_before != base._sha256(args.reference_opt):
        raise RuntimeError("compiler or source changed during measurement")
    packet = {"schema": "tessera.rocm.folded_input_pitches.v1",
        "architecture": rt._rocm_live_arch(), "device": base._selected_device_name(hip),
        "source_before": before, "source_after": after,
        "reference_compiler_sha256": reference_before,
        "reverse": args.reverse, "trials": args.trials,
        "timing_scope": "resident HIP graph device window including graph dispatch; separate checked public launch wall time",
        "attribution_limit": "row pitches and physical span sizes vary; this is not a cache-counter or per-phase measurement",
        "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(rows)}))


if __name__ == "__main__":
    main()
