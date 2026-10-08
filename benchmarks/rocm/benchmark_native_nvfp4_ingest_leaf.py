"""Diagnostic physical converter evidence, before Graph/Schedule/Tile integration."""
import argparse
import ctypes as C
import hashlib
import json
import os
import socket
from pathlib import Path
import statistics

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx, rocm_nvfp4_ingest as ingest
from tests.device.rocm.test_native_nvfp4_ingest_leaf import run_leaf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if rt._rocm_live_arch() != "gfx1201":
    raise RuntimeError("exact gfx1201 required")
hip = rt._load_hip_for_launch()
if hip is None or hip.hipInit(0):
    raise RuntimeError("HIP unavailable")
device, ordinal = C.create_string_buffer(256), C.c_int()
if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(device,len(device),ordinal.value):
    raise RuntimeError("device identity unavailable")
rows = []
for n,k in ((33,256),(256,1024),(2048,4096)):
    rng = np.random.default_rng(n+k)
    projections = []
    for name, count, global_scale in (("gate",n,.5),("up",n+1,2.)):
        codes = rng.integers(0,16,(count,k),dtype=np.uint8)
        scales = rng.integers(0,127,(count,k//16),dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        projections.append(ingest.NVFP4Projection(
            name,mx.pack_e2m1_codes(codes),scales,global_scale))
    row = run_leaf(projections,timing=True)
    row["resident_event_median_ms"] = statistics.median(row["resident_event_ms"])
    row["source_sha256"] = {
        p.name: hashlib.sha256(p.packed_codes.tobytes()+p.e4m3_scales.tobytes()).hexdigest()
        for p in projections}
    rows.append(row)
    root = Path(__file__).resolve().parents[2]
    compiler = Path(os.environ["TESSERA_OPT"])
    files = [Path(__file__), root/"tests/device/rocm/test_native_nvfp4_ingest_leaf.py",
        root/"python/tessera/compiler/rocm_nvfp4_ingest.py",
        root/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/NativeNVFP4Ingest.h",
        root/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMFpQuantKernel.cpp",
        root/"src/compiler/codegen/Tessera_ROCM_Backend/lib/IR/TesseraROCMOps.cpp",
        root/"src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td"]
    packet = dict(schema="tessera.gfx1201.native_ingest_leaf.v1",
        architecture="gfx1201",device=device.value.decode(),
        device_ordinal=ordinal.value,host=socket.gethostname(),rows=rows,
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        boundary="physical leaf; checked ABI and Graph/Schedule/Tile integration pending")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(shape=row["shape_nk"],device_ms=row["resident_event_median_ms"],
        host_oracle_ms=row["host_oracle_ms"])),flush=True)
