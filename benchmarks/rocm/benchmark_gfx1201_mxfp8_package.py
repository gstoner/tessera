#!/usr/bin/env python3
"""Matched FP8-fp32-scale / MXFP8-E8M0 package timing on gfx1201.

Same quantized operands, K32 groups, per-column power-of-two scales and BF16
outputs. This is a numerical-contract-matched comparison, not a comparison
against the production FP8 K128/N128 schedule or MXFP4 folded math.
Graph replay separates GPU execution/dispatch from checked runtime.launch
host staging, transfers and synchronization. It does not isolate ISA phases.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import re
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from tests.device.rocm.test_mxfp8_scheduled_scale import _reference
from tests._support import rocm_isa


class Resident:
    """Own buffers and module through graph replay; borrowing graph closes first."""
    def __init__(self, hip, package, buffers, shape, *, grid=None, block=None):
        self.hip, self.copies, self._next = hip, [0], 0
        self.pointers, self.values = [], []
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.blob = C.create_string_buffer(package.image.payload)
        self.check(hip.hipModuleLoadData(C.byref(self.module), self.blob))
        try:
            self.check(hip.hipModuleGetFunction(C.byref(self.function), self.module,
                                               package.descriptor.entry_symbol.encode()))
            for binding in sorted(package.descriptor.buffers, key=lambda b: b.ordinal):
                host = buffers[binding.name]
                pointer = C.c_void_p()
                self.check(hip.hipMalloc(C.byref(pointer), host.nbytes))
                self.pointers.append(pointer)
                self.check(hip.hipMemcpy(pointer, C.c_void_p(host.ctypes.data), host.nbytes, 1))
                self.values.extend([C.c_void_p(pointer.value), C.c_void_p(pointer.value),
                                    C.c_int64(0), C.c_int64(host.size), C.c_int64(1)])
            self.values.extend([C.c_int64(shape.m), C.c_int64(shape.n), C.c_int64(shape.k)])
            self.argv = (C.c_void_p * len(self.values))(
                *[C.cast(C.byref(value), C.c_void_p) for value in self.values])
            if (grid is None) != (block is None):
                raise ValueError("resident timing requires both grid and block overrides")
            if grid is None:
                bm, bn = package.descriptor.provenance["macro_tile"]
                grid = ((shape.n + bn - 1)//bn, (shape.m + bm - 1)//bm, 1)
                block = package.descriptor.provenance["workgroup"]
            self.grid, self.block = grid, block
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"HIP resident package rc={status}")

    def launch_on_stream(self, stream):
        self.check(self.hip.hipModuleLaunchKernel(
            self.function, *self.grid, *self.block, 0, stream, self.argv, None))

    def result(self, like):
        out = np.empty_like(like)
        self.check(self.hip.hipDeviceSynchronize())
        self.check(self.hip.hipMemcpy(C.c_void_p(out.ctypes.data), self.pointers[-1], out.nbytes, 2))
        return out

    def close(self):
        self.hip.hipDeviceSynchronize()
        for ptr in self.pointers:
            self.hip.hipFree(ptr)
        self.pointers.clear()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module.value = None


def run(mnk, layout, args, hip, clock):
    shape = BlockScaleShape(*mnk, 32, 1, layout, "bf16")
    rng = np.random.default_rng(271)
    a = rng.integers(-3, 4, (shape.m, shape.k)).astype(ml_dtypes.float8_e4m3fn)
    b = rng.integers(-3, 4, (shape.k, shape.n)).astype(ml_dtypes.float8_e4m3fn)
    sa = rng.integers(125, 130, (shape.m, shape.groups), dtype=np.uint8)
    sb = rng.integers(125, 130, (shape.groups, shape.n), dtype=np.uint8)
    expected = _reference(a, b, sa, sb).astype(ml_dtypes.bfloat16)
    weight = np.ascontiguousarray(b.T) if layout == "nk" else b
    records, engines = {}, {}
    graphs = FoldedGraphWindows(clock)
    try:
        for name in ("fp8_fp32_scales", "mxfp8_e8m0_scales"):
            start = time.perf_counter_ns()
            package = compile_blockscale(shape) if name.startswith("fp8_") else compile_mxfp8(shape)
            compile_ms = (time.perf_counter_ns() - start)/1e6
            isa = rocm_isa.disassemble(package.image.payload, chip="gfx1201")
            if "v_wmma_f32_16x16x16_fp8_fp8" not in isa:
                raise RuntimeError("native package has no RDNA4 FP8 WMMA")
            isa_name = f"{args.output.stem}_{shape.m}_{shape.n}_{shape.k}_{layout}_{name}.s"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent/isa_name).write_text(isa)
            isa_counts = dict(rocm_isa.mnemonics(isa, r"v_(?:wmma|mul|cvt|add|ldexp)_\w+"))
            scales = (sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float32),
                      sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)) if name.startswith("fp8_") else (sa, sb)
            output = np.empty_like(expected)
            buffers = dict(a=a, b=weight, a_scale=scales[0], b_scale=scales[1], o=output)
            artifact = rt.RuntimeArtifact(metadata={"target": package.image.target},
                native_image=package.image, launch_descriptor=package.descriptor,
                tile_ir=package.tile_ir, target_ir=package.target_ir)
            binding = dict(buffers=buffers, scalars=dict(M=shape.m, N=shape.n, K=shape.k))
            receipt = rt.launch(artifact, binding)
            if not receipt["ok"] or receipt["execution_kind"] != "native_gpu":
                raise RuntimeError(receipt)
            np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))
            engine = Resident(hip, package, buffers, shape)
            engines[name] = engine
            engine.launch_on_stream(None)
            np.testing.assert_array_equal(engine.result(output).view(np.uint16), expected.view(np.uint16))
            e2e = []
            for _ in range(args.windows):
                start = time.perf_counter_ns()
                for _ in range(3):
                    receipt = rt.launch(artifact, binding)
                    if not receipt["ok"]:
                        raise RuntimeError(receipt)
                e2e.append((time.perf_counter_ns()-start)/1e6/3)
            records[name] = dict(abi=package.descriptor.abi_id, image_digest=package.image.image_digest,
                descriptor_digest=package.descriptor.descriptor_digest, compile_ms=compile_ms,
                payload_sha256=hashlib.sha256(package.image.payload).hexdigest(),
                isa_file=isa_name, isa_sha256=hashlib.sha256(isa.encode()).hexdigest(),
                static_isa_counts=isa_counts,
                correctness="bitwise_bf16_before_timing", workgroup=engine.block, grid=engine.grid,
                end_to_end_ms=e2e, end_to_end_median_ms=statistics.median(e2e),
                device_windows=[], ir_digests=dict(
                    tile=hashlib.sha256(package.tile_ir.encode()).hexdigest(),
                    target=hashlib.sha256(package.target_ir.encode()).hexdigest()))
        counts = {}
        for name, engine in engines.items():
            count = 32
            while True:
                sample = graphs.window(engine, count, bracketed=True)
                if sample["device_window_ms"] >= 20:
                    break
                count *= 2
            counts[name] = count
        for trial in range(args.windows):
            order = list(engines)
            if trial % 2:
                order.reverse()
            for name in order:
                sample = graphs.window(engines[name], counts[name], bracketed=True)
                if sample["device_event_disagreement"] > .05:
                    raise RuntimeError(f"device clock and HIP event disagree by more than 5%: {sample}")
                records[name]["device_windows"].append(sample)
        for name, engine in engines.items():
            record = records[name]
            record["device_execution_dispatch_median_ms"] = statistics.median(
                s["device_window_ms"]/s["launches"] for s in record["device_windows"])
            # Numerical proof after every replay, not only the pre-timing launch.
            np.testing.assert_array_equal(engine.result(expected).view(np.uint16), expected.view(np.uint16))
        return dict(shape_mnk=list(mnk), layout=layout, output="bf16", arms=records,
            mxfp8_over_fp8_device_ratio=records["mxfp8_e8m0_scales"]["device_execution_dispatch_median_ms"] /
                records["fp8_fp32_scales"]["device_execution_dispatch_median_ms"])
    finally:
        graphs.close()
        for engine in engines.values():
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path,
        help="saved TileToROCM.cpp corresponding to a preserved reference compiler")
    parser.add_argument("--windows", type=int, default=5)
    args = parser.parse_args()
    if args.windows < 3:
        parser.error("at least three alternating windows required")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("benchmark requires the live gfx1201 device")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, device = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(device, len(device), ordinal.value))
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    try:
        rows = [run(shape, layout, args, hip, clock)
                for shape in [(17, 19, 64), (200, 256, 128), (256, 512, 1024)]
                for layout in ("kn", "nk")]
    finally:
        clock.close()
    packet = dict(schema="tessera.gfx1201.mxfp8_checked_package.v1", architecture="gfx1201",
        device=device.value.decode(), device_ordinal=ordinal.value, rows=rows,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        timing_scope="resident graph device execution plus GPU dispatch; checked runtime end-to-end separately",
        selector_promotion=False, mxfp4_comparison_pending=True,
        source_sha256={str(p): hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in map(Path, [
            "benchmarks/rocm/benchmark_gfx1201_mxfp8_package.py", "python/tessera/runtime.py",
            "python/tessera/compiler/rocm_mxfp8_blockscale.py",
            "python/tessera/compiler/rocm_fp8_blockscale.py",
            "src/compiler/programming_model/lib/PMPasses.cpp",
            "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
            "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
            "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp"])})
    if args.reference_source:
        packet["source_sha256"]["src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp"] = hashlib.sha256(args.reference_source.read_bytes()).hexdigest()
        packet["reference_source_override"] = str(args.reference_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2)+"\n")
    print(json.dumps({str((r["shape_mnk"], r["layout"])): r["mxfp8_over_fp8_device_ratio"] for r in rows}))


if __name__ == "__main__":
    main()
