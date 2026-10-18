"""Correctness-gated native source VJP timing, with distinct timing domains."""
from __future__ import annotations

import argparse
import csv
import ctypes as ct
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts

from tessera import runtime as rt
from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _snapshot():
    names = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    names += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    production = {name: _sha(Path(name).read_bytes()) for name in names
                  if name.startswith(("src/", "python/", "benchmarks/nvidia/"))}
    provider = Path(rt._load_nvidia_ptx_launch()._name).resolve()
    tools = {name: _sha(Path(os.environ[name]).read_bytes())
             for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT")}
    return dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                modified_source_sha256=production, compiler_binary_sha256=tools,
                provider=str(provider), provider_sha256=_sha(provider.read_bytes()))


def _device():
    # Query the context actually selected by the prepared owner, then resolve
    # that CUDA device's PCI ID in nvidia-smi instead of assuming GPU ordinal 0.
    lib = ct.CDLL("libcuda.so.1")
    lib.cuCtxGetDevice.argtypes = [ct.POINTER(ct.c_int)]
    lib.cuCtxGetDevice.restype = ct.c_int
    lib.cuDeviceGetPCIBusId.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
    lib.cuDeviceGetPCIBusId.restype = ct.c_int
    device, pci = ct.c_int(), ct.create_string_buffer(64)
    if lib.cuCtxGetDevice(ct.byref(device)) or lib.cuDeviceGetPCIBusId(pci, len(pci), device):
        raise RuntimeError("source reverse benchmark requires a live owning CUDA context")
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=pci.bus_id,name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], text=True)
    def identity(value):
        return value.strip().lower().lstrip("0")
    matches = [row for row in csv.reader(io.StringIO(text))
               if identity(row[0]) == identity(pci.value.decode())]
    if len(matches) != 1 or matches[0][3].strip() != "12.0":
        raise RuntimeError("active source reverse CUDA device does not resolve to exact SM120")
    row = [value.strip() for value in matches[0]]
    return dict(pci_bus_id=row[0], name=row[1], uuid=row[2],
                compute_capability=row[3], driver_version=row[4], cuda_ordinal=device.value)


def inputs(shape, storage, permuted, seed):
    m, n, k = shape
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .75, (m, k)).astype(dtype)
    b = rng.normal(0, .2, (k, n)).astype(dtype)
    cotangent = rng.normal(0, .2, (m, n)).astype(np.float32)
    return (b, x, cotangent) if permuted else (x, b, cotangent)


def reference(roots, permuted):
    x, b, cotangent = (roots[1], roots[0], roots[2]) if permuted else roots
    xx, bb, dy = (value.astype(np.float64) for value in (x, b, cotangent))
    wide = dy @ bb.T
    edge = wide.astype(x.dtype).astype(np.float64)
    g = np.where(xx > 0, edge, 0)
    inverse = 1 / np.sqrt(np.mean(xx * xx, axis=1, keepdims=True) + 1e-5)
    expected = g * inverse - xx * inverse**3 * np.mean(g * xx, axis=1, keepdims=True)
    def ulp(value):
        stored = value.astype(x.dtype)
        return np.abs(np.nextafter(stored, x.dtype.type(np.inf)).astype(np.float64) -
                      stored.astype(np.float64))
    n, k = b.shape[1], x.shape[1]
    eps = np.finfo(np.float32).eps
    if n * eps >= 1:
        raise ValueError("shape exceeds the benchmark's finite FP32 error-bound envelope")
    edge_error = n * eps / (1 - n * eps) * (np.abs(dy) @ np.abs(bb.T)) + ulp(wide)
    propagated = inverse * edge_error + np.abs(xx) * inverse**3 * np.mean(
        np.abs(xx) * edge_error, axis=1, keepdims=True)
    budget = propagated + k * 32 * eps * (np.abs(expected) + .01) + ulp(expected)
    return expected, budget


def _check(outputs, expected, budget, dtype):
    if (len(outputs) != 1 or outputs[0].dtype != dtype or not np.isfinite(outputs[0]).all()
            or not np.all(np.abs(outputs[0].astype(np.float64) - expected) <= budget)):
        raise RuntimeError("source reverse failed its independent pullback oracle")


def run_shape(shape, storage, permuted, *, samples, repetitions, native_repetitions, artifact_dir):
    roots = inputs(shape, storage, permuted, seed=18_281 + sum(shape))
    expected, budget = reference(roots, permuted)
    dtype = roots[int(permuted)].dtype
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        first = time.perf_counter()
        actual = fn.native_backward(*roots[:2], out_cotangents=roots[2])
        first_ms = (time.perf_counter() - first) * 1000
        _check(actual, expected, budget, dtype)
        certificate = fn.last_backward_execution["execution_certificate"]
        if certificate["evidence_scope"] != "exact_device":
            raise RuntimeError("source reverse benchmark lacks an exact-device execution certificate")
        owner = next(iter(fn._native_prepared_vjp_calls.values()))
        _check(owner(roots), expected, budget, dtype)
        for _ in range(5):
            fn.native_backward(*roots[:2], out_cotangents=roots[2])
            owner(roots)
        wall = {"public_native_backward": [], "direct_prepared_owner": []}
        enqueue, captured = [], []
        def public():
            return fn.native_backward(*roots[:2], out_cotangents=roots[2])
        for sample in range(samples):
            order = ("public_native_backward", "direct_prepared_owner")
            if sample % 2:
                order = order[::-1]
            for mode in order:
                call = public if mode == "public_native_backward" else lambda: owner(roots)
                start = time.perf_counter()
                for _ in range(repetitions):
                    call()
                wall[mode].append((time.perf_counter() - start) * 1000 / repetitions)
            for capture, destination in ((False, enqueue), (True, captured)):
                output, timing = owner.profile(roots, repeats=native_repetitions, captured=capture)
                _check(output, expected, budget, dtype)
                destination.append(timing)
        _check(public(), expected, budget, dtype)
        artifact = fn.native_backward_runtime_artifact()
        artifact_record = None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            name = "program_" + "x".join(map(str, shape)) + "_" + storage + ("_ba" if permuted else "_ab") + ".json.gz"
            raw = artifact.to_json().encode()
            compressed = gzip.compress(raw, mtime=0)
            (artifact_dir / name).write_bytes(compressed)
            artifact_record = dict(file=name, raw_bytes=len(raw), compressed_bytes=len(compressed),
                                   raw_sha256=_sha(raw), compressed_sha256=_sha(compressed))
        return dict(shape_mnk=list(shape), storage=storage, permuted=permuted,
                    correctness="passed_before_and_after_timing",
                    max_abs_error=float(np.max(np.abs(actual[0].astype(np.float64) - expected))),
                    first_call_ms=first_ms, completed_host_samples_ms=wall,
                    completed_host_median_ms={mode: statistics.median(values) for mode, values in wall.items()},
                    native_enqueue=enqueue, native_captured=captured,
                    image_digests=owner.receipt["image_digests"], artifact=artifact_record,
                    plan_digest=hashlib.sha256(owner.native.plan_json.encode()).hexdigest(),
                    execution_certificate=certificate)
    finally:
        fn.close_native_storage()


def _shape(value):
    result = tuple(int(part) for part in value.lower().split("x"))
    if len(result) != 3 or any(part <= 0 for part in result):
        raise argparse.ArgumentTypeError("expected positive MxNxK")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=_shape, default=[(17, 19, 35), (64, 64, 128), (128, 128, 512)])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--native-repetitions", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 3 or args.repetitions <= 0 or not 0 < args.native_repetitions <= 4096:
        parser.error("require samples >= 3, repetitions > 0 and native repetitions in [1,4096]")
    before = _snapshot()
    rows = [run_shape(shape, storage, permuted, samples=args.samples,
                      repetitions=args.repetitions, native_repetitions=args.native_repetitions,
                      artifact_dir=args.artifact_dir)
            for shape in args.shapes for storage in ("f16", "bf16") for permuted in (False, True)]
    after = _snapshot()
    if before != after:
        raise RuntimeError("source/compiler/provider changed during benchmark")
    packet = dict(schema="tessera.nvidia.source_reverse_benchmark.v1",
                  work_item="W11-MIXED-ADJOINT-CONTRACTION-20261018",
                  device=_device(), source_before=before, source_after=after,
                  method=dict(samples=args.samples, repetitions=args.repetitions,
                              native_repetitions=args.native_repetitions,
                              timing_domains="completed host wall, CUDA stage/program enqueue/capture kept separate",
                              host_order="counterbalanced public/direct native owner",
                              first_call="includes certification/packaging; no cold compiler-cache claim",
                              scope="static source-only pullback; no speedup or general reverse claim"),
                  rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(f"source reverse: {len(rows)} correctness-gated timing rows written to {args.output}")


if __name__ == "__main__":
    main()
