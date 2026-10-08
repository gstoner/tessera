#!/usr/bin/env python3
"""Interleaved resident candidate/reference FP8 and MXFP8 compiler attribution.

Diagnostic only: packages retain their native Graph/Schedule/Tile construction.
Each compiler pair shares exact quantized operands and is verified before and
after every graph window. Resident graph time includes device dispatch.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from benchmarks.rocm.benchmark_gfx1201_three_formats import quantize_fp8, quantize_folded, verify, errors
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from tessera.compiler.rocm_mxfp4_folded import FoldedPrefillSchedule, folded_prefill_grid
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock


class SharedImageResident:
    """Borrow one resident allocation/ABI while owning a second compiler image."""

    def __init__(self, hip, package, owner):
        self.hip, self.owner, self.copies = hip, owner, [0]
        self.grid, self.block, self.argv = owner.grid, owner.block, owner.argv
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.blob = C.create_string_buffer(package.image.payload)
        Resident.check(hip.hipModuleLoadData(C.byref(self.module), self.blob))
        try:
            Resident.check(hip.hipModuleGetFunction(
                C.byref(self.function), self.module,
                package.descriptor.entry_symbol.encode()))
        except BaseException:
            self.close()
            raise

    def launch_on_stream(self, stream):
        Resident.check(self.hip.hipModuleLaunchKernel(
            self.function, *self.grid, *self.block, 0, stream, self.argv, None))

    def result(self, like):
        return self.owner.result(like)

    def close(self):
        self.hip.hipDeviceSynchronize()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module = C.c_void_p()


def loaded_resources(hip, engine):
    """Query the loaded module function at its actual launch block size."""
    get_attribute = hip.hipFuncGetAttribute
    get_attribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
    get_attribute.restype = C.c_int
    # Installed HIP driver_types.h enum: shared=1, local=3, registers=4.
    result = {}
    for name, attribute in (("static_shared_bytes", 1),
                            ("local_bytes_per_thread", 3),
                            ("registers_per_thread", 4)):
        value = C.c_int()
        Resident.check(get_attribute(C.byref(value), attribute, engine.function))
        result[name] = value.value
    occupancy = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
    occupancy.restype = C.c_int
    threads = int(np.prod(engine.block))
    blocks = C.c_int()
    Resident.check(occupancy(C.byref(blocks), engine.function, threads, 0))
    result.update(block_threads=threads, dynamic_shared_bytes=0,
                  active_blocks_per_multiprocessor=blocks.value,
                  provenance="live HIP module attributes and occupancy at actual block")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def collect_interleaved_windows(graphs, engines, *, windows, min_window_ms, verify_output):
    """Admit complete paired series only after both arms meet the duration floor.

    Short series are retained explicitly, then rerun at a common larger launch
    count. Never select individual fast/slow observations from an admitted pair.
    """
    warm=[]
    for label,engine in engines.items():
        for _ in range(3):
            sample=graphs.window(engine,32,bracketed=True)
            verify_output(engine)
            warm.append(sample["device_window_ms"]/32)
    fastest=min(warm)
    if not np.isfinite(fastest) or fastest<=0:
        raise RuntimeError("graph calibration requires positive finite device durations")
    launches=max(32,int(np.ceil(1.5*min_window_ms/fastest)))
    rejected=[]
    while launches<=4096:
        samples={label:[] for label in engines};orders=[]
        for index in range(windows):
            order=["candidate","reference"] if index%2==0 else ["reference","candidate"]
            orders.append(order)
            for label in order:
                sample=graphs.window(engines[label],launches,bracketed=True)
                verify_output(engines[label]);samples[label].append(sample)
        durations=[sample["device_window_ms"] for series in samples.values() for sample in series]
        if all(np.isfinite(value) and value>=min_window_ms for value in durations):
            return samples,orders,dict(minimum_device_window_ms=min_window_ms,
                observed_minimum_ms=min(durations),launches=launches,
                calibration_per_launch_ms=warm,rejected_series=rejected,
                status="duration_admitted")
        rejected.append(dict(launches=launches,orders=orders,samples=samples,
                             reason="at_least_one_window_below_requested_duration"))
        launches*=2
    raise RuntimeError("graph duration admission exceeded 4096 launches; no timing series admitted")


def run_shape(args, hip, clock, shape):
    m, n, k = shape
    rng = np.random.default_rng(args.seed)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    rows = []
    formats = [
        ("fp8_k128_n128", 128, 128, False),
        ("fp8_k32_n1", 32, 1, False),
        ("mxfp8_k32_n1", 32, 1, True),
    ]
    if args.include_mxfp4:
        formats.append(("mxfp4_folded", 32, 1, False))
    for name, gk, gn, e8m0 in formats:
        folded_arm = name == "mxfp4_folded"
        if folded_arm:
            qa, sa, folded, da, exact_b, db = quantize_folded(a, b)
            buffers = dict(a=qa, b_folded=folded.weight_bytes, a_scale=sa,
                           row_reference=folded.row_reference)
        else:
            qa, qb, scales, da, db = quantize_fp8(a, b, gk, gn, e8m0=e8m0)
        ideal, absolute = da @ db, np.abs(da) @ np.abs(db)
        output = np.full((m, n), -101, ml_dtypes.bfloat16)
        if folded_arm:
            buffers["output"] = output
        else:
            buffers = dict(a=qa, b=np.ascontiguousarray(qb.T),
                           a_scale=scales[0], b_scale=scales[1], o=output)
        profile = BlockScaleShape(m, n, k, gk, gn, "nk", "bf16")
        engines, metadata, samples = {}, {}, {"candidate": [], "reference": []}
        graphs = FoldedGraphWindows(clock)
        try:
            for label, compiler in [("candidate", args.compiler), ("reference", args.reference)]:
                os.environ["TESSERA_OPT"] = str(compiler.resolve())
                geometry = {}
                if folded_arm:
                    package = compile_folded_scaled_matmul(
                        qa, sa, folded, tessera_opt=compiler, allow_approximate=True).package
                    prov = package.descriptor.provenance
                    schedule = FoldedPrefillSchedule(
                        raster_group_m=prov["raster_group_m"], workgroup_mode=prov["workgroup_mode"],
                        staging_prefetch=prov["staging_prefetch"], epilogue=prov["epilogue_schedule"],
                        row_guard=prov["row_guard"])
                    geometry = dict(grid=folded_prefill_grid(m, n, schedule), block=(256, 1, 1))
                else:
                    package = compile_mxfp8(profile) if e8m0 else compile_blockscale(profile)
                artifact = rt.RuntimeArtifact(
                    metadata={"target": package.image.target},
                    native_image=package.image, launch_descriptor=package.descriptor,
                    tile_ir=package.tile_ir, target_ir=package.target_ir)
                receipt = rt.launch(artifact, dict(buffers=buffers, scalars=dict(M=m, N=n, K=k)))
                if not receipt.get("ok") or receipt.get("execution_kind") != "native_gpu":
                    raise RuntimeError(receipt)
                verify(output, ideal, absolute, k)
                engine = (Resident(hip, package, buffers, profile, **geometry) if label == "candidate"
                          else SharedImageResident(hip, package, engines["candidate"]))
                engines[label] = engine
                engine.launch_on_stream(None)
                correctness = verify(engine.result(output), ideal, absolute, k)
                metadata[label] = dict(
                    image_sha256=hashlib.sha256(package.image.payload).hexdigest(),
                    tile_sha256=hashlib.sha256(package.tile_ir.encode()).hexdigest(),
                    target_sha256=hashlib.sha256(package.target_ir.encode()).hexdigest(),
                    provenance=dict(package.descriptor.provenance),
                    entry=package.descriptor.entry_symbol,
                    grid=engine.grid, block=engine.block, correctness=correctness,
                    loaded_resources=loaded_resources(hip, engine))
            samples,orders,admission=collect_interleaved_windows(
                graphs,engines,windows=args.windows,min_window_ms=args.min_window_ms,
                verify_output=lambda engine:verify(engine.result(output),ideal,absolute,k))
            medians = {label: statistics.median(
                item["device_window_ms"] / item["launches"] for item in values)
                for label, values in samples.items()}
            paired = [samples["candidate"][i]["device_window_ms"] /
                      samples["reference"][i]["device_window_ms"] for i in range(args.windows)]
            rows.append(dict(shape_mnk=list(shape), format=name, metadata=metadata,
                physical_weight_storage="expanded_e4m3_bytes" if folded_arm else "e4m3_bytes",
                approximate_policy="explicit_allow" if folded_arm else "none",
                folding_output_error=errors(ideal, da @ exact_b) if folded_arm else None,
                input_sha256={key: hashlib.sha256(value.tobytes()).hexdigest()
                              for key, value in buffers.items() if key not in {"o", "output"}},
                samples=samples, orders=orders, timing_admission=admission, median_device_ms=medians,
                paired_candidate_over_reference=paired,
                median_paired_ratio=statistics.median(paired)))
            print(shape, name, medians, "paired ratio", statistics.median(paired), flush=True)
        finally:
            graphs.close()
            for engine in reversed(list(engines.values())):
                engine.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler-source", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--include-mxfp4", action="store_true")
    parser.add_argument("--windows", type=int, default=11)
    parser.add_argument("--min-window-ms", type=float, default=6)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--shapes", default="200,1024,1536;256,1024,1024;197,1056,1536")
    args = parser.parse_args()
    shapes = [tuple(map(int, value.split(","))) for value in args.shapes.split(";")]
    if args.windows < 3 or args.min_window_ms <= 0 or any(
        len(s) != 3 or min(s) <= 0 or s[2] % 128 for s in shapes
    ):
        parser.error("positive shapes with K128, positive windows, at least three trials required")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires live gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, device = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(device, len(device), ordinal.value))
    os.environ["TESSERA_OPT"] = str(args.compiler.resolve())
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    try:
        packet = dict(schema="tessera.gfx1201.interleaved_compiler_formats.v1",
            architecture=rt._rocm_live_arch(), device=device.value.decode(),
            device_ordinal=ordinal.value, seed=args.seed,
            timing_scope="resident HIP graph device execution plus dispatch; shared allocation addresses; no host transfers",
            compiler_sha256=sha(args.compiler), reference_compiler_sha256=sha(args.reference),
            source_sha256={str(path.relative_to(ROOT)): sha(path) for path in [
                Path(__file__), ROOT / "python/tessera/runtime.py",
                ROOT / "python/tessera/compiler/rocm_fp8_blockscale.py",
                ROOT / "python/tessera/compiler/rocm_mxfp8_blockscale.py",
                ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp"]},
            compiler_source_sha256=sha(args.compiler_source) if args.compiler_source else None,
            reference_source_sha256=sha(args.reference_source) if args.reference_source else None,
            selector_promotion=False, rows=[])
        for shape in shapes:
            packet["rows"].extend(run_shape(args, hip, clock, shape))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    finally:
        clock.close()


if __name__ == "__main__":
    main()
