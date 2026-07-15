#!/usr/bin/env python3
"""Repeated-median, correctness-gated gfx1151 generated-GEMM schedule sweep.

This is the promotion-grade successor to the one-shot GEMM ladders.  It keeps
inputs resident, records one HIP-event mean per trial, reports the median and
MAD across trials, validates every tile against a common device result and a
NumPy oracle, and extracts assembler code-object resource metadata.  The full
matrix covers square, rectangular, ragged, dtype, and fused-epilogue cases.

The JSON is evidence, not a selector update.  A separate promotion step may use
only rows whose correctness and stability gates pass.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
TILES = ((1, 1), (2, 2), (2, 4), (3, 4), (4, 4))

BASE_SHAPES = (
    ("square", 512, 512, 512),
    ("square", 1024, 1024, 1024),
    ("square", 1536, 1536, 1536),
    ("square", 2048, 2048, 2048),
    ("square", 3072, 3072, 3072),
    ("square", 4096, 4096, 4096),
    ("rectangular", 128, 4096, 4096),
    ("rectangular", 256, 11008, 4096),
    ("rectangular", 512, 4096, 1024),
    ("rectangular", 1024, 4096, 1024),
    ("rectangular", 2048, 8192, 2048),
    ("rectangular", 4096, 11008, 4096),
    ("ragged", 513, 769, 257),
    ("ragged", 1009, 1537, 1025),
    ("ragged", 2049, 4093, 2051),
)
DTYPE_SHAPES = ((1024, 1024, 1024), (2048, 2048, 2048),
                (4096, 4096, 4096))
DTYPES = ("f16", "bf16", "int8", "int4")
EPILOGUE_SHAPES = ((1024, 1024, 1024), (2048, 2048, 2048))
EPILOGUES = (
    (True, "none"),
    (False, "relu"),
    (True, "relu"),
    (True, "gelu"),
    (True, "silu"),
)


@dataclass(frozen=True)
class Case:
    family: str
    m: int
    n: int
    k: int
    dtype: str = "f16"
    bias: bool = False
    activation: str = "none"

    @property
    def key(self) -> str:
        epi = ("bias+" if self.bias else "") + self.activation
        return f"{self.family}:{self.m}x{self.n}x{self.k}:{self.dtype}:{epi}"


def matrix_cases(quick: bool = False) -> list[Case]:
    if quick:
        return [
            Case("square", 256, 256, 256),
            Case("rectangular", 129, 257, 193),
            Case("ragged", 131, 127, 65),
            Case("dtype", 256, 256, 256, "int8"),
            Case("epilogue", 256, 256, 256, bias=True, activation="gelu"),
        ]
    cases = [Case(f, m, n, k) for f, m, n, k in BASE_SHAPES]
    cases += [Case("dtype", m, n, k, dtype=dtype)
              for m, n, k in DTYPE_SHAPES for dtype in DTYPES]
    cases += [Case("epilogue", m, n, k, bias=bias, activation=activation)
              for m, n, k in EPILOGUE_SHAPES
              for bias, activation in EPILOGUES]
    # Plain f16 duplicates are intentionally removed: the base square row is
    # the same executable contract and already carries all five tiles.
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for case in cases:
        identity = (case.family, case.m, case.n, case.k, case.dtype,
                    case.bias, case.activation)
        if identity not in seen:
            seen.add(identity)
            unique.append(case)
    return unique


def _input_dtype(dtype: str) -> Any:
    if dtype == "f16":
        return np.float16
    if dtype == "bf16":
        bf16 = rt._bfloat16_dtype()
        if bf16 is None:
            raise RuntimeError("NumPy/ml_dtypes bfloat16 support is unavailable")
        return bf16
    return np.int8


def _inputs(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    seed = sum((i + 1) * ord(c) for i, c in enumerate(case.key)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    if case.dtype == "int8":
        a = rng.integers(-4, 5, (case.m, case.k), dtype=np.int8)
        b = rng.integers(-4, 5, (case.k, case.n), dtype=np.int8)
    elif case.dtype == "int4":
        a = rng.integers(-8, 8, (case.m, case.k), dtype=np.int8)
        b = rng.integers(-8, 8, (case.k, case.n), dtype=np.int8)
    else:
        dtype = _input_dtype(case.dtype)
        a = np.asarray(rng.standard_normal((case.m, case.k), dtype=np.float32)
                       * 0.125, dtype=dtype)
        b = np.asarray(rng.standard_normal((case.k, case.n), dtype=np.float32)
                       * 0.125, dtype=dtype)
    bias = (rng.standard_normal(case.n, dtype=np.float32) * 0.125
            if case.bias else None)
    return np.ascontiguousarray(a), np.ascontiguousarray(b), bias


def _reference(case: Case, a: np.ndarray, b: np.ndarray,
               bias: np.ndarray | None) -> np.ndarray | None:
    if case.dtype in ("int8", "int4"):
        # NumPy's int32 matmul is not BLAS-accelerated.  A full 4096^3 oracle
        # takes minutes and measures the CPU rather than the kernel.  Large
        # integer rows use deterministic exact output samples below, while every
        # tile is still compared bit-for-bit against the full device oracle.
        if case.m * case.n * case.k > 10_000_000_000:
            return None
        return a.astype(np.int32) @ b.astype(np.int32)
    out = a.astype(np.float32) @ b.astype(np.float32)
    if bias is not None:
        out += bias
    if case.activation == "relu":
        np.maximum(out, 0.0, out=out)
    elif case.activation == "gelu":
        x3 = out * out * out
        out *= 0.5 * (1.0 + np.tanh(
            np.float32(0.7978845608028654) *
            (out + np.float32(0.044715) * x3)))
    elif case.activation == "silu":
        out /= 1.0 + np.exp(-out)
    return out


def _sampled_integer_error(actual: np.ndarray, a: np.ndarray,
                           b: np.ndarray) -> dict[str, Any]:
    rows = np.unique(np.linspace(0, actual.shape[0] - 1, 8, dtype=np.int64))
    cols = np.unique(np.linspace(0, actual.shape[1] - 1, 8, dtype=np.int64))
    mismatch = 0
    max_abs = 0
    for row in rows:
        av = a[row].astype(np.int64)
        for col in cols:
            expected = int(np.dot(av, b[:, col].astype(np.int64)))
            delta = abs(int(actual[row, col]) - expected)
            mismatch += int(delta != 0)
            max_abs = max(max_abs, delta)
    return {"max_abs_error": max_abs,
            "normalized_error": 0.0 if mismatch == 0 else math.inf,
            "mismatch_count": mismatch,
            "sample_count": int(len(rows) * len(cols))}


def _mr(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


class DeviceCase:
    def __init__(self, hip: ctypes.CDLL, hsaco: bytes, case: Case,
                 tile: tuple[int, int], a: np.ndarray, b: np.ndarray,
                 bias: np.ndarray | None):
        self.hip = hip
        self.case = case
        self.tile = tile
        self.mod = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.mod), hsaco) != 0:
            raise RuntimeError("hipModuleLoadData failed")
        self.fn = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, b"gemm") != 0:
            raise RuntimeError("kernel symbol gemm was not found")
        self.out_dtype = np.int32 if case.dtype in ("int8", "int4") else np.float32
        self.output = np.empty((case.m, case.n), dtype=self.out_dtype)
        self.devs: list[ctypes.c_void_p] = []
        da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        for host, dev in ((a, da), (b, db), (self.output, dd)):
            if hip.hipMalloc(ctypes.byref(dev), host.nbytes) != 0:
                self.close()
                raise RuntimeError("hipMalloc failed")
            self.devs.append(dev)
        hip.hipMemcpy(da, a.ctypes.data_as(ctypes.c_void_p), a.nbytes, 1)
        hip.hipMemcpy(db, b.ctypes.data_as(ctypes.c_void_p), b.nbytes, 1)
        args = (_mr(da, a.size) + _mr(db, b.size) + _mr(dd, self.output.size)
                + [ctypes.c_int64(case.m), ctypes.c_int64(case.n),
                   ctypes.c_int64(case.k)])
        if bias is not None:
            dbias = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(dbias), bias.nbytes) != 0:
                self.close()
                raise RuntimeError("hipMalloc bias failed")
            self.devs.append(dbias)
            hip.hipMemcpy(dbias, bias.ctypes.data_as(ctypes.c_void_p),
                          bias.nbytes, 1)
            args += _mr(dbias, bias.size)
        self.dd = dd
        self.arg_values = args
        self.arg_array = (ctypes.c_void_p * len(args))()
        for i, value in enumerate(args):
            self.arg_array[i] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        mt, nt = tile
        self.grid = ((case.n + 16 * nt - 1) // (16 * nt),
                     (case.m + 16 * mt - 1) // (16 * mt))

    def launch(self) -> int:
        gx, gy = self.grid
        return self.hip.hipModuleLaunchKernel(
            self.fn, gx, gy, 1, 32, 1, 1, 0, None, self.arg_array, None)

    def download(self) -> np.ndarray:
        if self.launch() != 0 or self.hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("GEMM correctness launch failed")
        self.hip.hipMemcpy(self.output.ctypes.data_as(ctypes.c_void_p), self.dd,
                           self.output.nbytes, 2)
        return self.output.copy()

    def measure(self, *, trials: int, iterations: int,
                warmup: int) -> list[float]:
        for _ in range(warmup):
            if self.launch() != 0:
                raise RuntimeError("GEMM warmup launch failed")
        self.hip.hipDeviceSynchronize()
        samples = []
        for _ in range(trials):
            start, stop = ctypes.c_void_p(), ctypes.c_void_p()
            self.hip.hipEventCreate(ctypes.byref(start))
            self.hip.hipEventCreate(ctypes.byref(stop))
            self.hip.hipEventRecord(start, None)
            for _ in range(iterations):
                if self.launch() != 0:
                    raise RuntimeError("GEMM timed launch failed")
            self.hip.hipEventRecord(stop, None)
            self.hip.hipEventSynchronize(stop)
            elapsed = ctypes.c_float()
            self.hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop)
            self.hip.hipEventDestroy(start)
            self.hip.hipEventDestroy(stop)
            samples.append(float(elapsed.value) / iterations)
        return samples

    def close(self) -> None:
        for dev in getattr(self, "devs", []):
            self.hip.hipFree(dev)
        self.devs = []
        mod = getattr(self, "mod", None)
        if mod and mod.value and hasattr(self.hip, "hipModuleUnload"):
            self.hip.hipModuleUnload(mod)


def _tool(name: str) -> str | None:
    candidates = (Path("/opt/rocm/llvm/bin") / name,
                  Path("/usr/lib/llvm-22/bin") / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def code_object_resources(hsaco: bytes) -> dict[str, Any]:
    readobj = _tool("llvm-readobj")
    objdump = _tool("llvm-objdump")
    resources: dict[str, Any] = {
        "vgpr_count": None, "sgpr_count": None, "agpr_count": None,
        "lds_bytes": None, "scratch_bytes": None, "private_segment_bytes": None,
        "wavefront_size": None, "max_flat_workgroup_size": None,
        "vgpr_spill_count": None, "sgpr_spill_count": None,
        "spill_count": None, "spills": None,
        "vgpr_limited_waves_per_simd": None,
        "occupancy_model": "gfx1151 1536 VGPR/SIMD divided by assembler VGPR count",
    }
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as obj:
        obj.write(hsaco)
        obj.flush()
        texts = []
        if readobj:
            run = subprocess.run([readobj, "--notes", obj.name],
                                 capture_output=True, text=True)
            texts.append(run.stdout + run.stderr)
        if objdump:
            run = subprocess.run([objdump, "-d", obj.name],
                                 capture_output=True, text=True)
            texts.append(run.stdout + run.stderr)
        text = "\n".join(texts)
    aliases = {
        "vgpr_count": (r"\.vgpr_count:\s*(\d+)", r"VGPRCount:\s*(\d+)"),
        "sgpr_count": (r"\.sgpr_count:\s*(\d+)", r"SGPRCount:\s*(\d+)"),
        "agpr_count": (r"\.agpr_count:\s*(\d+)",),
        "lds_bytes": (r"\.group_segment_fixed_size:\s*(\d+)",),
        "private_segment_bytes": (r"\.private_segment_fixed_size:\s*(\d+)",),
        "wavefront_size": (r"\.wavefront_size:\s*(\d+)",),
        "max_flat_workgroup_size": (r"\.max_flat_workgroup_size:\s*(\d+)",),
        "vgpr_spill_count": (r"\.vgpr_spill_count:\s*(\d+)",),
        "sgpr_spill_count": (r"\.sgpr_spill_count:\s*(\d+)",),
    }
    for field, patterns in aliases.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                resources[field] = int(match.group(1))
                break
    resources["scratch_bytes"] = resources["private_segment_bytes"]
    spill_values = [resources[name] for name in
                    ("vgpr_spill_count", "sgpr_spill_count")
                    if resources[name] is not None]
    resources["spill_count"] = sum(spill_values) if spill_values else None
    resources["spills"] = bool(resources["scratch_bytes"] or
                               resources["spill_count"])
    if resources["vgpr_count"]:
        resources["vgpr_limited_waves_per_simd"] = max(
            1, min(16, 1536 // resources["vgpr_count"]))
    return resources


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if np.issubdtype(expected.dtype, np.integer):
        mismatch = int(np.count_nonzero(actual != expected))
        return {"max_abs_error": int(np.max(np.abs(
                    actual.astype(np.int64) - expected.astype(np.int64)))),
                "normalized_error": 0.0 if mismatch == 0 else math.inf,
                "mismatch_count": mismatch}
    delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(np.max(delta))
    scale = max(1.0, float(np.max(np.abs(expected))))
    return {"max_abs_error": max_abs,
            "normalized_error": max_abs / scale,
            "mismatch_count": None}


def _correct(case: Case, metrics: dict[str, Any]) -> bool:
    if case.dtype in ("int8", "int4"):
        return metrics["mismatch_count"] == 0
    # Absolute f32-accumulation drift scales with K/output magnitude.  The
    # normalized bound catches real failures while tolerating WMMA reduction
    # order; tiny outputs retain the established 5e-2 absolute gate.
    return (metrics["max_abs_error"] <= 5e-2
            or metrics["normalized_error"] <= 2e-3)


def summarize_winners(rows: list[dict[str, Any]],
                      cases: list[Case]) -> list[dict[str, Any]]:
    winners = []
    for case in cases:
        candidates = [row for row in rows if row["case"] == case.key
                      and row["correct"]]
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda row: row["median_ms"])
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else winner
        paired = [other / chosen for chosen, other in zip(
            winner["trials_ms"], runner_up["trials_ms"], strict=True)]
        paired_median = statistics.median(paired)
        paired_mad = statistics.median(
            abs(value - paired_median) for value in paired)
        win_rate = sum(chosen < other for chosen, other in zip(
            winner["trials_ms"], runner_up["trials_ms"], strict=True)) / len(paired)
        # Promotion stability is a paired decision property.  Absolute device
        # clocks may move on an APU; a selector is stable when its winner beats
        # the runner-up in >=75% of interleaved rounds and the paired median is
        # at least 3%.  Raw latency MAD remains visible but is not substituted
        # for paired evidence.
        stable = win_rate >= 0.75 and paired_median >= 1.03
        winners.append({
            "case": case.key, "tile": winner["tile"],
            "runner_up_tile": runner_up["tile"],
            "median_ms": winner["median_ms"],
            "tflops_or_tops": winner["tflops_or_tops"],
            "margin_vs_runner_up": (
                runner_up["median_ms"] / winner["median_ms"] - 1.0),
            "paired_median_speedup": paired_median,
            "paired_ratio_mad": paired_mad,
            "paired_win_rate": win_rate,
            "absolute_relative_mad": winner["relative_mad"],
            "stable": stable,
            "resources": winner["resources"],
        })
    return winners


def run_matrix(*, quick: bool, trials: int, iterations: int,
               warmup: int, checkpoint: Path | None = None,
               resume: bool = False,
               cases: list[Case] | None = None) -> dict[str, Any]:
    if CHIP != "gfx1151":
        raise RuntimeError(f"this evidence matrix is exact-target gfx1151, got {CHIP}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("a live gfx1151 HIP device is required")
    selected_cases = cases if cases is not None else matrix_cases(quick)
    rows: list[dict[str, Any]] = []
    if resume and checkpoint is not None and checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        prior_rows = list(prior.get("rows") or [])
        counts: dict[str, int] = {}
        for row in prior_rows:
            counts[row["case"]] = counts.get(row["case"], 0) + 1
        complete = {key for key, count in counts.items() if count == len(TILES)}
        rows = [row for row in prior_rows if row["case"] in complete]
    resource_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case_index, case in enumerate(selected_cases, 1):
        if sum(row["case"] == case.key for row in rows) == len(TILES):
            print(f"[{case_index}/{len(selected_cases)}] {case.key} (resume)",
                  flush=True)
            continue
        print(f"[{case_index}/{len(selected_cases)}] {case.key}", flush=True)
        a, b, bias = _inputs(case)
        reference = _reference(case, a, b, bias)
        device_oracle: np.ndarray | None = None
        tile_info: dict[tuple[int, int], dict[str, Any]] = {}
        # Correctness/resource pass first.  Timing is a separate interleaved
        # pass so thermal or clock drift cannot systematically favor the tile
        # measured first or last.
        for mt, nt in TILES:
            hsaco = rt._build_compiled_gemm_hsaco(
                mt, nt, case.dtype, bias=case.bias,
                activation=case.activation)
            resource_key = (mt, nt, case.dtype, case.bias, case.activation)
            resources = resource_cache.setdefault(
                resource_key, code_object_resources(hsaco))
            device = DeviceCase(hip, hsaco, case, (mt, nt), a, b, bias)
            try:
                actual = device.download()
                oracle_metrics = (_sampled_integer_error(actual, a, b)
                                  if reference is None
                                  else _error_metrics(actual, reference))
                oracle_kind = ("sampled_exact_int64+full_device_equivalence"
                               if reference is None else "full_numpy")
                if device_oracle is None:
                    device_oracle = actual
                schedule_metrics = _error_metrics(actual, device_oracle)
            finally:
                device.close()
            tile_info[(mt, nt)] = {
                "hsaco": hsaco, "resources": resources,
                "oracle": oracle_metrics, "oracle_kind": oracle_kind,
                "device_oracle": schedule_metrics,
            }

        samples_by_tile = {tile: [] for tile in TILES}
        for trial in range(trials):
            # Rotate and alternate direction.  Every tile occupies every part
            # of the round over a seven-trial run, while reversal cancels
            # monotonic temperature/clock movement within adjacent rounds.
            offset = trial % len(TILES)
            order = list(TILES[offset:] + TILES[:offset])
            if trial % 2:
                order.reverse()
            for tile in order:
                device = DeviceCase(hip, tile_info[tile]["hsaco"], case,
                                    tile, a, b, bias)
                try:
                    sample = device.measure(
                        trials=1, iterations=iterations, warmup=warmup)[0]
                finally:
                    device.close()
                samples_by_tile[tile].append(sample)

        for mt, nt in TILES:
            info = tile_info[(mt, nt)]
            samples = samples_by_tile[(mt, nt)]
            median_ms = statistics.median(samples)
            mad_ms = statistics.median(abs(x - median_ms) for x in samples)
            tflops = 2.0 * case.m * case.n * case.k / (median_ms * 1e9)
            passed = (_correct(case, info["oracle"])
                      and info["device_oracle"]["max_abs_error"] == 0)
            row = {
                "case": case.key, "family": case.family,
                "shape": [case.m, case.n, case.k], "dtype": case.dtype,
                "bias": case.bias, "activation": case.activation,
                "tile": [mt, nt], "trials_ms": samples,
                "median_ms": median_ms, "mad_ms": mad_ms,
                "min_ms": min(samples), "max_ms": max(samples),
                "relative_mad": mad_ms / median_ms,
                "tflops_or_tops": tflops,
                "oracle": info["oracle"],
                "oracle_kind": info["oracle_kind"],
                "device_oracle": info["device_oracle"],
                "correct": passed, "resources": info["resources"],
            }
            rows.append(row)
            if checkpoint is not None:
                checkpoint.write_text(json.dumps({
                    "schema": "tessera.rocm.gemm_schedule_matrix.partial.v1",
                    "target": "rocm", "evidence_arch": CHIP,
                    "trials": trials, "iterations_per_trial": iterations,
                    "rows": rows,
                }, indent=2) + "\n")
            print(f"  {mt}x{nt}: {median_ms:.4f} ms {tflops:.2f} T/s "
                  f"MAD={row['relative_mad']:.2%} correct={passed}", flush=True)
    winners = summarize_winners(rows, selected_cases)
    return {
        "schema": "tessera.rocm.gemm_schedule_matrix.v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": "rocm", "evidence_arch": CHIP,
        "timing": ("HIP event, resident buffers, median of interleaved "
                   "rotated per-trial means"),
        "trials": trials, "iterations_per_trial": iterations,
        "warmup_launches": warmup,
        "tiles": [list(tile) for tile in TILES],
        "rows": rows, "winners": winners,
        "all_correct": all(row["correct"] for row in rows),
        "all_stable": all(winner["stable"] for winner in winners),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resummarize", action="store_true",
                        help="recompute paired winner gates from --output")
    parser.add_argument("--case", action="append", default=[],
                        help="run only an exact case key; repeat as needed")
    args = parser.parse_args()
    if min(args.trials, args.iterations, args.warmup) < 1:
        parser.error("trials, iterations, and warmup must be positive")
    output = Path(args.output)
    selected_cases = matrix_cases(args.quick)
    if args.case:
        requested = set(args.case)
        selected_cases = [case for case in selected_cases if case.key in requested]
        missing = requested - {case.key for case in selected_cases}
        if missing:
            parser.error(f"unknown case key(s): {sorted(missing)}")
    if args.resummarize:
        result = json.loads(output.read_text())
        result["winners"] = summarize_winners(
            result["rows"], selected_cases)
        result["all_stable"] = all(row["stable"] for row in result["winners"])
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"resummarized {output}: stable={result['all_stable']}")
        return 0
    result = run_matrix(quick=args.quick, trials=args.trials,
                        iterations=args.iterations, warmup=args.warmup,
                        checkpoint=output, resume=args.resume,
                        cases=selected_cases)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}: {len(result['rows'])} rows, "
          f"correct={result['all_correct']} stable={result['all_stable']}")
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
