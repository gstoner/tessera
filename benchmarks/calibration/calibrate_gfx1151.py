#!/usr/bin/env python3
"""Selector-gated calibration sweep for gfx1151 (Strix Halo / Radeon 8060S).

Measures the two fields `target_perf.py`'s radeon_8060s profile deliberately
left absent ("calibrate rather than pick a convention"):

  * ``peak_tflops.bf16:matrix`` / ``peak_tflops.fp16:matrix`` — a
    register-resident WMMA microbenchmark: 8 independent accumulator chains per
    wave (ILP to cover WMMA latency), zero memory traffic in the timed loop.
    This is the *achievable device ceiling* in the TileSight Table-3 sense
    (measured-vs-spec), not a kernel-quality number.
  * ``dram_bw_gbps`` — a grid-stride vector<4xf32> copy over 256 MiB,
    counting read+write bytes.

Bare-metal measurements use HIP device events and retain independent host-wall
samples.  The raw measurement must be collected as a fresh child of
``tprof_rocm_native_capture.py`` and finalized with that ROCprofiler activity
artifact. WSL host-wall measurements are accepted only with
``--allow-provisional`` and can never become selector authority.

Output: a `calibration_corpus` JSON (target_perf.apply_corpus schema v1)
written to benchmarks/baselines/, then verified by loading it and asking
SchedulePlanner.for_target("rocm_gfx1151") — the call that raised before this
sweep existed.

Compile/launch machinery mirrors tests/unit/test_rocm_gemm_staged_async_copy.py
(tessera-rocm-opt → mlir-opt gpu-module-to-binary → ctypes HIP).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))

from tests._support.rocm_build import rocm_opt_path  # noqa: E402

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
WS = "#gpu.address_space<workgroup>"
_REQUIRED_METRICS = (
    "dram_bw_gbps",
    "peak_tflops.fp16:matrix",
    "peak_tflops.bf16:matrix",
)
_EXPECTED_KERNEL = {
    "dram_bw_gbps": "copy_bw",
    "peak_tflops.fp16:matrix": "wmma_peak",
    "peak_tflops.bf16:matrix": "wmma_peak",
}


# ── compile/launch machinery (pattern-proven in the staged-WMMA device test) ──


def find_mlir_opt():
    for c in ("/usr/lib/llvm-23/bin/mlir-opt", "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    import shutil

    return os.environ.get("TESSERA_MLIR_OPT") or shutil.which("mlir-opt")


def extract_hsaco(text: str) -> bytes:
    i = text.index('bin = "') + len('bin = "')
    out = bytearray()
    j = i
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(text):
        c = text[j]
        if c == '"':
            break
        if c == "\\":
            nx = text[j + 1 : j + 3]
            if len(nx) == 2 and nx[0] in hexd and nx[1] in hexd:
                out.append(int(nx, 16))
                j += 3
                continue
            if text[j + 1] in simple:
                out.append(simple[text[j + 1]])
                j += 2
                continue
        out.append(ord(c))
        j += 1
    return bytes(out)


def hip_lib():
    rocm_lib = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(rocm_lib, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
            except OSError:
                pass
    return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_LOCAL)


def _active_device_identity(hip) -> dict[str, object]:
    """Query the active HIP device instead of trusting the requested target."""

    ordinal = ctypes.c_int()
    if hip.hipGetDevice(ctypes.byref(ordinal)) != 0:
        raise RuntimeError("hipGetDevice failed")
    properties = ctypes.create_string_buffer(16 * 1024)
    getter = getattr(hip, "hipGetDevicePropertiesR0600", None)
    query = "hipGetDevicePropertiesR0600"
    if getter is None:
        getter = hip.hipGetDeviceProperties
        query = "hipGetDeviceProperties"
    getter.argtypes = [ctypes.c_void_p, ctypes.c_int]
    getter.restype = ctypes.c_int
    if getter(ctypes.byref(properties), ordinal.value) != 0:
        raise RuntimeError(f"{query} failed")
    matches = sorted(
        {match.decode("ascii") for match in re.findall(rb"gfx[0-9]+", properties.raw)}
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{query} did not yield one unambiguous AMD architecture: {matches}"
        )
    name = properties.raw[:256].split(b"\0", 1)[0].decode("utf-8", "replace")
    return {
        "query": query,
        "ordinal": ordinal.value,
        "architecture": matches[0],
        "name": name,
    }


def compile_kernel(mlir: str, extra_rocm_passes: list[str] | None = None) -> bytes:
    rocm_opt = rocm_opt_path()
    mlir_opt = find_mlir_opt()
    if rocm_opt is None or mlir_opt is None:
        raise SystemExit("need tessera-rocm-opt + mlir-opt built")
    passes = (extra_rocm_passes or []) + ["--lower-tessera-target-to-rocdl"]
    lowered = subprocess.run([str(rocm_opt), "-", *passes], input=mlir, capture_output=True, text=True)
    if lowered.returncode != 0:
        raise RuntimeError(f"rocm-opt failed:\n{lowered.stderr[:800]}")
    pl = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pl}"], input=lowered.stdout, capture_output=True, text=True)
    if ser.returncode != 0 or "gpu.binary" not in ser.stdout:
        raise RuntimeError(f"serialize failed:\n{ser.stderr[:800]}")
    hsaco = extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"
    return hsaco


def memref_args(ptr, n):
    return [
        ctypes.c_void_p(ptr.value),
        ctypes.c_void_p(ptr.value),
        ctypes.c_int64(0),
        ctypes.c_int64(n),
        ctypes.c_int64(1),
    ]


class Launcher:
    def __init__(self, hip, hsaco: bytes, name: bytes):
        self.hip = hip
        self.mod = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.mod), hsaco) != 0:
            raise SystemExit("no usable AMD GPU (hipModuleLoadData failed)")
        self.fn = ctypes.c_void_p()
        assert hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, name) == 0
        self.launch = hip.hipModuleLaunchKernel
        self.launch.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )

    def run_timed(self, grid, block, args, reps=5) -> dict[str, object]:
        arr = (ctypes.c_void_p * len(args))()
        for i, x in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(x), ctypes.c_void_p)
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        if self.hip.hipEventCreate(ctypes.byref(start)) != 0:
            raise RuntimeError("hipEventCreate(start) failed")
        if self.hip.hipEventCreate(ctypes.byref(stop)) != 0:
            self.hip.hipEventDestroy(start)
            raise RuntimeError("hipEventCreate(stop) failed")
        host_samples: list[float] = []
        event_samples: list[float] = []
        for rep in range(reps + 1):  # first is warmup
            assert self.hip.hipEventRecord(start, None) == 0
            t0 = time.perf_counter()
            assert self.launch(self.fn, grid, 1, 1, block, 1, 1, 0, None, arr, None) == 0
            assert self.hip.hipEventRecord(stop, None) == 0
            assert self.hip.hipEventSynchronize(stop) == 0
            t1 = time.perf_counter()
            elapsed_ms = ctypes.c_float()
            assert self.hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start, stop) == 0
            if rep > 0:
                host_samples.append(t1 - t0)
                event_samples.append(float(elapsed_ms.value) / 1_000.0)
        self.hip.hipEventDestroy(start)
        self.hip.hipEventDestroy(stop)
        host_median = statistics.median(host_samples)
        event_median = statistics.median(event_samples)
        event_valid = math.isfinite(event_median) and event_median > 0.0
        agreement = abs(event_median - host_median) / event_median if event_valid else math.inf
        return {
            "host_wall_seconds": host_samples,
            "hip_event_seconds": event_samples,
            "host_wall_median_seconds": host_median,
            "hip_event_median_seconds": event_median,
            "hip_event_valid": event_valid,
            "host_event_relative_difference": agreement,
        }


# ── kernels ──────────────────────────────────────────────────────────────────


def wmma_peak_mlir(elem: str) -> str:
    """8 independent WMMA accumulator chains per wave, register-resident."""
    chains = 8
    inits = "\n".join(f"      %z{k} = arith.constant dense<0.0> : vector<8xf32>" for k in range(chains))
    iter_args = ", ".join(f"%acc{k} = %z{k}" for k in range(chains))
    types = ", ".join(["vector<8xf32>"] * chains)
    body = "\n".join(
        f"        %n{k} = tessera_rocm.wmma %a, %b, %acc{k}"
        f" : vector<16x{elem}>, vector<16x{elem}>, vector<8xf32>"
        f" -> vector<8xf32>"
        for k in range(chains)
    )
    yields = ", ".join(f"%n{k}" for k in range(chains))
    results = ", ".join(f"%r{k}" for k in range(chains))
    adds = "\n".join(f"      %s{k} = arith.addf %s{k - 1}, %r{k} : vector<8xf32>" for k in range(1, chains))
    return f"""
module {{
  gpu.module @m {{
    gpu.func @wmma_peak(%out: memref<?xf32>, %iters: i64) kernel {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %a = arith.constant dense<1.0> : vector<16x{elem}>
      %b = arith.constant dense<1.0> : vector<16x{elem}>
{inits}
      %n = arith.index_cast %iters : i64 to index
      {results} = scf.for %i = %c0 to %n step %c1
          iter_args({iter_args}) -> ({types}) {{
{body}
        scf.yield {yields} : {types}
      }}
      %s0 = arith.addf %r0, %r0 : vector<8xf32>
{adds}
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %base = arith.muli %bid, %bdim : index
      %gid = arith.addi %base, %tid : index
      %v = vector.extract %s{chains - 1}[0] : f32 from vector<8xf32>
      memref.store %v, %out[%gid] : memref<?xf32>
      gpu.return
    }}
  }}
}}
"""


def copy_bw_mlir() -> str:
    """Grid-stride vector<4xf32> copy: dst[i] = src[i]."""
    return """
module {
  gpu.module @m {
    gpu.func @copy_bw(%src: memref<?xf32>, %dst: memref<?xf32>, %n4: i64) kernel {
      %c4 = arith.constant 4 : index
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %gdim = gpu.grid_dim x
      %base = arith.muli %bid, %bdim : index
      %gid = arith.addi %base, %tid : index
      %stride = arith.muli %gdim, %bdim : index
      %n = arith.index_cast %n4 : i64 to index
      scf.for %i = %gid to %n step %stride {
        %addr = arith.muli %i, %c4 : index
        %v = vector.load %src[%addr] : memref<?xf32>, vector<4xf32>
        vector.store %v, %dst[%addr] : memref<?xf32>, vector<4xf32>
      }
      gpu.return
    }
  }
}
"""


# ── measurements ─────────────────────────────────────────────────────────────


def _timing_seconds(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"timing record is missing numeric {field}")
    return float(value)


def measure_wmma_peak(hip, elem: str, *, wsl: bool) -> tuple[float, dict[str, object]]:
    hsaco = compile_kernel(wmma_peak_mlir(elem))
    lch = Launcher(hip, hsaco, b"wmma_peak")
    blocks, threads = 320, 256  # 8 waves/block -> 2560 waves total
    waves = blocks * (threads // 32)
    chains, flop_per_wmma = 8, 2 * 16 * 16 * 16
    nfloat = blocks * threads
    out = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(out), 4 * nfloat) == 0
    # Auto-scale iters to ~0.3 s using a probe run.
    iters = 2000
    args = memref_args(out, nfloat) + [ctypes.c_int64(iters)]
    probe = lch.run_timed(blocks, threads, args, reps=2)
    probe_seconds = _timing_seconds(probe, "host_wall_median_seconds")
    iters = max(2000, int(iters * 0.3 / max(probe_seconds, 1e-6)))
    args = memref_args(out, nfloat) + [ctypes.c_int64(iters)]
    timing = lch.run_timed(blocks, threads, args, reps=5)
    hip.hipFree(out)
    timing["kernel_name"] = "wmma_peak"
    timing["image_sha256"] = hashlib.sha256(hsaco).hexdigest()
    timing["iterations"] = iters
    timing["waves"] = waves
    timing["chains"] = chains
    timing["flop_per_wmma"] = flop_per_wmma
    timing["timing_source"] = "host_wall" if wsl else "hip_event"
    t = _timing_seconds(timing, "host_wall_median_seconds" if wsl else "hip_event_median_seconds")
    tflops = waves * iters * chains * flop_per_wmma / t / 1e12
    print(f"  wmma {elem}: {tflops:.2f} TFLOP/s (iters={iters}, waves={waves}, t={t * 1e3:.1f} ms)")
    return tflops, timing


def measure_dram_bw(hip, *, wsl: bool) -> tuple[float, dict[str, object]]:
    hsaco = compile_kernel(copy_bw_mlir())
    lch = Launcher(hip, hsaco, b"copy_bw")
    nfloat = 64 * 1024 * 1024  # 256 MiB per buffer
    nbytes = 4 * nfloat
    src, dst = ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(src), nbytes) == 0
    assert hip.hipMalloc(ctypes.byref(dst), nbytes) == 0
    hip.hipMemset(src, 1, nbytes)
    blocks, threads = 2048, 256
    args = memref_args(src, nfloat) + memref_args(dst, nfloat) + [ctypes.c_int64(nfloat // 4)]
    timing = lch.run_timed(blocks, threads, args, reps=7)
    for d in (src, dst):
        hip.hipFree(d)
    timing["kernel_name"] = "copy_bw"
    timing["image_sha256"] = hashlib.sha256(hsaco).hexdigest()
    timing["bytes_read_write"] = 2 * nbytes
    timing["timing_source"] = "host_wall" if wsl else "hip_event"
    t = _timing_seconds(timing, "host_wall_median_seconds" if wsl else "hip_event_median_seconds")
    gbps = 2 * nbytes / t / 1e9
    print(f"  dram copy: {gbps:.1f} GB/s (read+write, 256 MiB, t={t * 1e3:.2f} ms)")
    return gbps, timing


def _source_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"source_commit": commit, "worktree_dirty": dirty}


def _is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or "wsl" in platform.release().lower()


def _measure(*, allow_provisional: bool) -> dict[str, object]:
    wsl = _is_wsl()
    if wsl and not allow_provisional:
        raise SystemExit(
            "selector calibration requires bare-metal gfx1151; this host is WSL. "
            "Use --allow-provisional only for pruning diagnostics."
        )
    hip = hip_lib()
    if hip.hipInit(0) != 0:
        raise SystemExit("hipInit failed — no ROCm host")
    identity = _active_device_identity(hip)
    actual_arch = str(identity["architecture"])
    if actual_arch != CHIP:
        raise SystemExit(
            f"requested calibration target {CHIP}, but active HIP device is {actual_arch}"
        )
    print(f"calibrating {actual_arch} ({'provisional WSL wall' if wsl else 'HIP device event'})")

    results: dict[str, float] = {}
    timings: dict[str, dict[str, object]] = {}
    bandwidth, timings["dram_bw_gbps"] = measure_dram_bw(hip, wsl=wsl)
    results["dram_bw_gbps"] = round(bandwidth, 1)
    for elem, key in (("f16", "peak_tflops.fp16:matrix"), ("bf16", "peak_tflops.bf16:matrix")):
        try:
            value, timings[key] = measure_wmma_peak(hip, elem, wsl=wsl)
            results[key] = round(value, 2)
        except RuntimeError as e:
            print(f"  wmma {elem}: NOT MEASURED ({str(e)[:120]}...)")
    return {
        "schema": "tessera.gfx1151_calibration_measurement.v1",
        "architecture": actual_arch,
        "requested_architecture": CHIP,
        "device_identity": identity,
        "device": "radeon_8060s",
        "execution_environment": "wsl2" if wsl else "bare_metal",
        "measured_on": date.today().isoformat(),
        "host": platform.node(),
        "source": _source_state(),
        "results": results,
        "timings": timings,
    }


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _positive_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"calibration {field} must be finite and positive")
    return float(value)


def _nonnegative_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"calibration {field} must be finite and nonnegative")
    return float(value)


def _validate_metric_binding(
    name: str, result: object, timing: object, *, bare_metal: bool
) -> None:
    if not isinstance(timing, dict):
        raise ValueError(f"calibration timing row {name} must be a mapping")
    if timing.get("kernel_name") != _EXPECTED_KERNEL[name]:
        raise ValueError(f"calibration timing row {name} names the wrong kernel")
    image = timing.get("image_sha256")
    if not isinstance(image, str) or re.fullmatch(r"[0-9a-f]{64}", image) is None:
        raise ValueError(f"calibration timing row {name} lacks a valid image digest")
    host_samples = timing.get("host_wall_seconds")
    event_samples = timing.get("hip_event_seconds")
    if (
        not isinstance(host_samples, list)
        or not isinstance(event_samples, list)
        or len(host_samples) < 2
        or len(host_samples) != len(event_samples)
    ):
        raise ValueError(f"calibration timing row {name} requires paired samples")
    for index, sample in enumerate(host_samples):
        _positive_number(sample, f"{name}.host_sample[{index}]")
    for index, sample in enumerate(event_samples):
        (_positive_number if bare_metal else _nonnegative_number)(
            sample, f"{name}.event_sample[{index}]"
        )
    host_median = _positive_number(
        timing.get("host_wall_median_seconds"), f"{name}.host median"
    )
    event_median = (_positive_number if bare_metal else _nonnegative_number)(
        timing.get("hip_event_median_seconds"), f"{name}.event median"
    )
    if bare_metal and timing.get("timing_source") != "hip_event":
        raise ValueError(f"bare-metal calibration row {name} must use HIP events")
    if not bare_metal and timing.get("timing_source") != "host_wall":
        raise ValueError(f"provisional calibration row {name} must use host wall time")
    authoritative_median = event_median if bare_metal else host_median
    measured = _positive_number(result, name)
    if name == "dram_bw_gbps":
        work = _positive_number(timing.get("bytes_read_write"), "bytes_read_write")
        recomputed = work / authoritative_median / 1e9
        tolerance = 0.051
    else:
        work = 1.0
        for field in ("waves", "iterations", "chains", "flop_per_wmma"):
            work *= _positive_number(timing.get(field), f"{name}.{field}")
        recomputed = work / authoritative_median / 1e12
        tolerance = 0.006
    if not math.isclose(measured, recomputed, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"calibration result {name}={measured} disagrees with its timing/work metadata ({recomputed})"
        )
    if not math.isclose(host_median, statistics.median(host_samples), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"calibration timing row {name} has a stale host median")
    if not math.isclose(event_median, statistics.median(event_samples), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"calibration timing row {name} has a stale event median")


def _finalize(
    raw: dict[str, object],
    capture: dict[str, object] | None,
    *,
    allow_provisional: bool,
    measurement_path: Path | None = None,
) -> dict[str, object]:
    if raw.get("schema") != "tessera.gfx1151_calibration_measurement.v1":
        raise ValueError("unsupported gfx1151 calibration measurement schema")
    if raw.get("architecture") != "gfx1151":
        raise ValueError("calibration measurement did not prove exact gfx1151")
    if raw.get("requested_architecture") != "gfx1151":
        raise ValueError("calibration measurement was not requested for gfx1151")
    identity = raw.get("device_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("architecture") != "gfx1151"
        or identity.get("query")
        not in {"hipGetDeviceProperties", "hipGetDevicePropertiesR0600"}
        or not isinstance(identity.get("ordinal"), int)
    ):
        raise ValueError("calibration measurement lacks queried gfx1151 identity")
    results = raw.get("results")
    timings = raw.get("timings")
    required = set(_REQUIRED_METRICS)
    if not isinstance(results, dict) or set(results) != required:
        raise ValueError("calibration results require exact bandwidth plus fp16/bf16 rows")
    if not isinstance(timings, dict) or set(timings) != required:
        raise ValueError("calibration timings require exact bandwidth plus fp16/bf16 rows")
    bare_metal = raw.get("execution_environment") == "bare_metal"
    for name in _REQUIRED_METRICS:
        _validate_metric_binding(
            name, results[name], timings[name], bare_metal=bare_metal
        )

    reasons: list[str] = []
    if raw.get("execution_environment") != "bare_metal":
        reasons.append("BARE_METAL_REQUIRED")
    for name, timing in timings.items():
        if not isinstance(timing, dict) or timing.get("hip_event_valid") is not True:
            reasons.append(f"HIP_DEVICE_EVENT_INVALID:{name}")
        elif float(timing.get("host_event_relative_difference", math.inf)) > 0.10:
            reasons.append(f"HOST_DEVICE_CLOCK_DISAGREEMENT:{name}")
    capture_digest = None
    capture_summary: dict[str, object] | None = None
    if capture is None:
        reasons.append("ROCPROFILER_ACTIVITY_NOT_COLLECTED")
    else:
        from tessera.compiler.profiler_rocm_native import validate_rocm_native_capture

        validate_rocm_native_capture(capture)
        proof = capture.get("proof", {})
        if capture.get("provider") != "rocprofiler" or capture.get("status") != "collected":
            reasons.append("ROCPROFILER_CAPTURE_MISSING")
        if not isinstance(proof, dict) or not proof.get("dispatch_activity_seen"):
            reasons.append("ROCPROFILER_DISPATCH_MISSING")
        if not isinstance(proof, dict) or not (proof.get("hip_callback_seen") or proof.get("hsa_callback_seen")):
            reasons.append("ROCPROFILER_RUNTIME_CALLBACK_MISSING")
        application = capture.get("application", [])
        application_arguments = application if isinstance(application, list) else []
        measurement_argument = str(measurement_path.resolve()) if measurement_path is not None else None
        resolved_arguments = {
            str(Path(argument).resolve())
            for argument in application_arguments
            if isinstance(argument, str) and argument.endswith(".json")
        }
        if (
            not isinstance(application, list)
            or "--measurement-only" not in application_arguments
            or measurement_argument not in resolved_arguments
        ):
            reasons.append("ROCPROFILER_MEASUREMENT_LINEAGE_MISSING")
        trace = capture.get("provider_trace", {})
        kernel_records = (
            [
                record
                for record in trace.get("records", [])
                if isinstance(record, dict) and record.get("kind") == "device_activity"
            ]
            if isinstance(trace, dict)
            else []
        )
        kernel_names = {str(record.get("name")) for record in kernel_records}
        missing_kernels = {"copy_bw", "wmma_peak"} - kernel_names
        if missing_kernels:
            reasons.append("ROCPROFILER_KERNEL_CORRELATION_MISSING:" + ",".join(sorted(missing_kernels)))
        kernel_counts = {
            name: sum(str(record.get("name")) == name for record in kernel_records)
            for name in ("copy_bw", "wmma_peak")
        }
        if kernel_counts["copy_bw"] < 1 or kernel_counts["wmma_peak"] < 2:
            reasons.append("ROCPROFILER_DISPATCH_CARDINALITY_MISSING")
        capture_digest = _sha256_json(capture)
        capture_summary = {
            "provider": capture.get("provider"),
            "status": capture.get("status"),
            "proof": proof,
            "correlated_kernel_names": sorted(kernel_names),
            "correlated_kernel_counts": kernel_counts,
            "capture_sha256": capture_digest,
            "output_documents": capture.get("output_documents", []),
        }
    source = raw.get("source", {})
    if not isinstance(source, dict) or source.get("worktree_dirty") is not False:
        reasons.append("SOURCE_WORKTREE_DIRTY")
    selector_eligible = not reasons
    if reasons and not allow_provisional:
        raise ValueError("calibration is not selector-eligible: " + ", ".join(reasons))

    corpus = {
        "kind": "calibration_corpus",
        "version": 1,
        "measured_on": raw["measured_on"],
        "host": raw["host"],
        "evidence_tier": ("selector_grade_bare_metal" if selector_eligible else "provisional_pruning_only"),
        "selector_eligible": selector_eligible,
        "ineligibility_reasons": reasons,
        "method": (
            "HIP-event median with independent host-wall samples; fresh-process "
            "ROCprofiler dispatch/activity correlation; register-resident "
            "8-chain WMMA and 256 MiB vector copy"
        ),
        "source": source,
        "measurement_sha256": _sha256_json(raw),
        "profiler_capture": capture_summary,
        "timing_records": timings,
        "device_identity": identity,
        "devices": {"radeon_8060s": results},
    }
    return corpus


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-only", type=Path)
    parser.add_argument("--finalize-measurement", type=Path)
    parser.add_argument("--native-capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args(argv)

    if args.finalize_measurement:
        raw = json.loads(args.finalize_measurement.read_text(encoding="utf-8"))
        capture = json.loads(args.native_capture.read_text(encoding="utf-8")) if args.native_capture else None
    else:
        raw = _measure(allow_provisional=args.allow_provisional)
        if args.measurement_only:
            args.measurement_only.parent.mkdir(parents=True, exist_ok=True)
            args.measurement_only.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"measurement written: {args.measurement_only}")
            return
        capture = None

    corpus = _finalize(
        raw,
        capture,
        allow_provisional=args.allow_provisional,
        measurement_path=args.finalize_measurement,
    )
    out_path = args.output or (
        REPO
        / "benchmarks"
        / "baselines"
        / f"rocm_gfx1151_calibration_{date.today().isoformat().replace('-', '_')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"corpus written: {out_path}")

    if corpus["selector_eligible"]:
        # Verification: only selector-grade evidence may mutate measured state.
        from tessera.compiler import target_perf

        updated = target_perf.load_corpus(out_path)
        print(f"apply_corpus updated: {updated}")
        from tessera.compiler.schedule_planner import SchedulePlanner

        planner = SchedulePlanner.for_target("rocm_gfx1151")
        print(f"SchedulePlanner.for_target('rocm_gfx1151') -> OK ({planner!r})")
    else:
        print("provisional packet written; selector registry was not mutated")


if __name__ == "__main__":
    main()
