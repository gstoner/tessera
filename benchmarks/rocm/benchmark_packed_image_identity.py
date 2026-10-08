"""Matched exact-gfx1201 static versus runtime-M/N packed image experiment."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4_packed_folded as packed
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape
from tests.device.rocm.test_packed_native_folded import inputs
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows


def run(shape, hip, clock):
    a, sa, payload, expected = inputs(*shape)
    records, engines, bindings = {}, {}, {}
    graphs = FoldedGraphWindows(clock)
    try:
        for name, projected in (("static", False), ("runtime_mn", True)):
            start = time.perf_counter_ns()
            package = packed.compile_packed_folded_scaled_matmul(
                a, sa, payload, tessera_opt=Path(os.environ["TESSERA_OPT"]),
                runtime_mn=projected).package
            compile_ms = (time.perf_counter_ns() - start) / 1e6
            output = np.full(expected.shape, np.nan, ml_dtypes.bfloat16)
            buffers = dict(a=a, b_packed=payload.weight_bytes, a_scale=sa,
                           scale_plane=payload.scale_plane, output=output)
            artifact = rt.RuntimeArtifact(
                metadata={"target": "rocm_gfx1201"}, native_image=package.image,
                launch_descriptor=package.descriptor,
                tile_ir=package.tile_ir, target_ir=package.target_ir)
            binding = dict(buffers=buffers, scalars=dict(zip(("M", "N", "K"), shape, strict=True)))
            result = rt.launch(artifact, binding)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(result)
            np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))
            engine = Resident(hip, package, buffers, BlockScaleShape(*shape, 32, 1, "nk", "bf16"),
                              grid=package.descriptor.geometry.grid, block=(256, 1, 1))
            engines[name] = engine
            bindings[name] = artifact, binding
            engine.launch_on_stream(C.c_void_p())
            np.testing.assert_array_equal(engine.result(output).view(np.uint16), expected.view(np.uint16))
            records[name] = dict(
                image_digest=package.image.image_digest,
                payload_sha256=package.image.payload_digest,
                target_ir_sha256=hashlib.sha256(package.target_ir.encode()).hexdigest(),
                authored_target_ir_sha256=package.descriptor.provenance["authored_target_ir_sha256"],
                entry=package.descriptor.entry_symbol, compile_state=package.image.compile_state,
                packaging_wall_ms=compile_ms, correctness="independent_oracle_bitwise_bf16",
                geometry=package.descriptor.geometry.to_dict(),
                device_windows=[], end_to_end_ms=[])
        for trial in range(7):
            names = list(engines)
            if trial % 2:
                names.reverse()
            for name in names:
                records[name]["device_windows"].append(graphs.window(engines[name], 1024, bracketed=True))
                artifact, binding = bindings[name]
                start = time.perf_counter_ns()
                for _ in range(3):
                    result = rt.launch(artifact, binding)
                    if not result.get("ok"):
                        raise RuntimeError(result)
                records[name]["end_to_end_ms"].append((time.perf_counter_ns() - start) / 1e6 / 3)
        for name, record in records.items():
            record["device_median_ms"] = statistics.median(
                x["device_window_ms"] / x["launches"] for x in record["device_windows"])
            record["end_to_end_median_ms"] = statistics.median(record["end_to_end_ms"])
            np.testing.assert_array_equal(engines[name].result(expected).view(np.uint16), expected.view(np.uint16))
        return dict(shape_mnk=shape, arms=records,
                    projected_over_static_device=records["runtime_mn"]["device_median_ms"]/records["static"]["device_median_ms"],
                    projected_over_static_e2e=records["runtime_mn"]["end_to_end_median_ms"]/records["static"]["end_to_end_median_ms"])
    finally:
        graphs.close()
        for engine in engines.values():
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    args = parser.parse_args()
    arch = rt._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError(f"exact gfx1201 required, observed {arch}")
    info = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True).stdout
    names = [line.strip() for line in info.splitlines() if "Marketing Name:" in line]
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal, device_name = C.c_int(), C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(device_name, len(device_name), ordinal.value))
    clock = DeviceClock(hip, Path(os.environ["TESSERA_OPT"]), args.llvm_bin)
    try:
        cases = [run(shape, hip, clock) for shape in (
            (128,32,256),(200,80,256),(256,64,256),(512,128,256),
            (200,80,2048),(256,80,2048))]
        packet = dict(architecture=arch, device=device_name.value.decode(),
                      device_ordinal=ordinal.value, device_marketing_names=names,
                      toolchain=subprocess.run([os.environ["TESSERA_OPT"], "--version"],
                          capture_output=True, text=True, check=True).stdout,
                      route="Graph->Schedule->Tile->ROCm Target->LLVM->HSACO",
                      scope="packed consumer only; not dynamic semantic Graph or native ingest",
                      timing="resident graph event windows versus checked allocating launch wall",
                      cases=cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2) + "\n")
        print(json.dumps([{k:v for k,v in row.items() if k!="arms"} for row in cases],indent=2))
    finally:
        clock.close()


if __name__ == "__main__":
    main()
