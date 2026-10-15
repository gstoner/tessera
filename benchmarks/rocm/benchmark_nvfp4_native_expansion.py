"""Gfx1201 native expansion characterization, not a matmul/model benchmark."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import time

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import reference_nvfp4_expand, nvfp4_expansion_policy
from tessera.compiler.rocm_nvfp4_expand_native import (
    build_nvfp4_expansion_graph, package_nvfp4_expansion, execute_nvfp4_expansion,
)
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from tests._support import rocm_isa



class ResidentExpansion:
    """Own all five allocations and the module until borrowed graphs close."""
    def __init__(self, hip, package, arrays):
        self.hip, self.copies, self._next = hip, [0], 0
        self.pointers, self.values = [], []
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.blob = C.create_string_buffer(package.image.payload)
        self.grid, self.block = package.descriptor.geometry.grid, package.descriptor.geometry.workgroup
        try:
            self.check(hip.hipModuleLoadData(C.byref(self.module), self.blob))
            self.check(hip.hipModuleGetFunction(C.byref(self.function), self.module,
                                               package.descriptor.entry_symbol.encode()))
            for i, host in enumerate(arrays):
                ptr = C.c_void_p()
                self.check(hip.hipMalloc(C.byref(ptr), host.nbytes))
                self.pointers.append(ptr)
                if i < 3:
                    self.check(hip.hipMemcpy(ptr, C.c_void_p(host.ctypes.data), host.nbytes, 1))
                self.values.extend([C.c_void_p(ptr.value), C.c_void_p(ptr.value),
                                    C.c_int64(0), C.c_int64(host.size), C.c_int64(1)])
            self.argv = (C.c_void_p * len(self.values))(
                *[C.cast(C.byref(value), C.c_void_p) for value in self.values])
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"HIP expansion status {status}")

    def launch_on_stream(self, stream):
        self.check(self.hip.hipModuleLaunchKernel(
            self.function, *self.grid, *self.block, 0, stream, self.argv, None))

    def outputs(self, arrays):
        self.check(self.hip.hipDeviceSynchronize())
        outputs = [np.empty_like(a) for a in arrays[3:]]
        for ptr, host in zip(self.pointers[3:], outputs):
            self.check(self.hip.hipMemcpy(C.c_void_p(host.ctypes.data), ptr, host.nbytes, 2))
        return outputs

    def close(self):
        self.hip.hipDeviceSynchronize()
        for ptr in self.pointers:
            self.hip.hipFree(ptr)
        self.pointers.clear()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module.value = None


def run(args):
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 device required")
    hip = rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("HIP unavailable")
    ResidentExpansion.check(hip.hipInit(0))
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    ResidentExpansion.check(hip.hipGetDevice(C.byref(ordinal)))
    ResidentExpansion.check(hip.hipDeviceGetName(name, len(name), ordinal.value))
    if not name.value:
        raise RuntimeError("missing device identity")
    clock = DeviceClock(hip, args.compiler, args.llvm_bin)
    records = []
    try:
        for n, k in ((7, 48), (128, 4096), (24576, 4096)):
            rng = np.random.default_rng(314)
            codes = rng.integers(0, 256, (n, k//2), dtype=np.uint8)
            scales = rng.integers(0, 127, (n, k//16), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
            scales[0, 0] = -0.
            globals_ = np.array([0.000123456789, 0.007654321], np.float64)
            expected = reference_nvfp4_expand(codes, scales, globals_, row_offsets=[0, n//2, n],
                                              numeric_policy=nvfp4_expansion_policy())
            start = time.perf_counter_ns()
            package = package_nvfp4_expansion(build_nvfp4_expansion_graph(
                n, k, [0, n//2, n], numeric_policy=nvfp4_expansion_policy()))
            compile_ms = (time.perf_counter_ns()-start)/1e6
            native = package.native
            arrays = [codes, scales, globals_] + [np.empty_like(x) for x in expected]

            def check(outputs):
                for got, want, dtype in zip(outputs, expected, (np.uint8, np.uint32)):
                    np.testing.assert_array_equal(got.view(dtype), want.view(dtype))

            check(execute_nvfp4_expansion(package, codes, scales, globals_))
            engine = ResidentExpansion(hip, native, arrays)
            graphs = FoldedGraphWindows(clock)
            try:
                engine.launch_on_stream(None)
                check(engine.outputs(arrays))
                count = 32
                while graphs.window(engine, count, bracketed=True)["device_window_ms"] < 20:
                    count *= 2
                    if count > 65536:
                        raise RuntimeError("unable to produce a stable timing window")
                windows = [graphs.window(engine, count, bracketed=True) for _ in range(3)]
                if any(w["device_event_disagreement"] > .05 for w in windows):
                    raise RuntimeError("device/event timing disagrees by more than 5%")
                check(engine.outputs(arrays))
                host = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    got = execute_nvfp4_expansion(package, codes, scales, globals_)
                    host.append((time.perf_counter_ns()-start)/1e6)
                    check(got)
                isa = rocm_isa.disassemble(native.image.payload, chip="gfx1201")
                if "v_cvt_f32_f64" not in isa:
                    raise RuntimeError("native image has no f64-to-f32 rounding")
                stem = native.image.payload_digest
                (args.output.parent / (stem + ".s")).write_text(isa)
                records.append(dict(
                    shape_nk=[n, k], input_kind="seeded_synthetic_not_checkpoint",
                    packed_input_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
                    scales_sha256=hashlib.sha256(scales.tobytes()).hexdigest(),
                    projection_globals=globals_.tolist(),
                    image_sha256=stem, descriptor_sha256=native.descriptor.descriptor_digest,
                    correctness="bitwise_both_outputs_before_and_after_timing",
                    compile_wall_ms=compile_ms, windows=windows,
                    device_graph_median_ms=statistics.median(w["device_window_ms"]/count for w in windows),
                    completed_host_samples_ms=host, completed_host_median_ms=statistics.median(host),
                    scope="isolated native producer; allocations/transfers excluded from device graph; warm repeated inputs"))
                print(json.dumps(records[-1]), flush=True)
            finally:
                graphs.close()
                engine.close()
    finally:
        clock.close()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "python/tessera/compiler/nvfp4_expansion.py",
             root / "python/tessera/compiler/rocm_nvfp4_expand_native.py",
             root / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/NativeNVFP4Expand.h"]
    return dict(architecture="gfx1201", device=name.value.decode(), records=records,
                route="GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                scope="native producer characterization; no resident consumer composition or model quality claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args), indent=2) + "\n")
