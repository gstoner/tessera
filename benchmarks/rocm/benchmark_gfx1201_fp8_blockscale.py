#!/usr/bin/env python3
"""gfx1201 block-scaled FP8 W8A8: Tessera's typed route vs AITER's Triton kernel.

ROCM-FP8-BLOCKSCALE-1 (sync GFX1201-LANES-2026-09-27). One process, one device,
one set of logical inputs, two kernels:

* **Tessera** -- ``tessera.scaled_matmul`` compiled Graph -> Schedule -> Tile ->
  Target -> HSACO by ``tessera.compiler.rocm_fp8_blockscale`` (the production
  route; a ``--panel``/``--k-unroll`` override exists only for the recorded
  sweep and is labelled as such in every row).
* **AITER** -- ``_gemm_a8w8_blockscale_kernel`` from the local AITER checkout,
  compiled ahead of time by Triton for gfx1201 with the tuned gfx1201 JSON
  config for the (N, K) pair and M bucket. The kernel source is imported and
  compiled unmodified; nothing is copied. The harness reproduces only the host
  wrapper's grid and argument specialization (Triton's JIT rules: a stride of
  1 becomes a constexpr, pointers and multiples of 16 get divisibility 16).
  AITER's split-K buckets (``NUM_KSPLIT > 1``) need its reduce kernel, which
  this harness does not drive, so those (shape, M) rows are recorded as not
  measured rather than timed with a different config.

Each kernel runs with its own production layouts: AITER reads the weight as
``w[N, K]`` (``b = w.T``) and writes bf16; Tessera reads ``B[K, N]`` and writes
f32 (its contract's plain f32 store). Both consume the same e4m3 values and
fp32 scales, and both are checked against an fp64 oracle of the block-scaled
math before any timing is taken.

Timing: paired, interleaved windows (ABAB then BABA) on one stream. Each
window is bracketed by the compiler-built device-clock marker
(``tessera-opt --tessera-device-clock-span``; ``llvm.readsteadycounter`` at
``hipDeviceAttributeWallClockRate``), a HIP event pair and host wall clock.
Windows are sized to at least ``--min-window-ms`` so the ROCm device-clock
admission (>= 5 ms windows) applies. The device clock is the reported source;
events and host wall are cross-checks recorded per window.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import ml_dtypes  # noqa: E402

from tessera.compiler.rocm_fp8_blockscale import (  # noqa: E402
    BlockScaleShape,
    blockscale_reference,
    lower_blockscale,
    package_blockscale,
)

P = ct.c_void_p


# --------------------------------------------------------------------------
# HIP
# --------------------------------------------------------------------------
class Hip:
    def __init__(self) -> None:
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        self.lib = ct.CDLL(str(root / "lib" / "libamdhip64.so"))
        self.check(self.lib.hipInit(0))
        self.lib.hipModuleLaunchKernel.argtypes = [
            P, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
            ct.c_uint, P, P, P]
        self.lib.hipEventElapsedTime.argtypes = [ct.POINTER(ct.c_float), P, P]
        self.lib.hipMemcpy.argtypes = [P, P, ct.c_size_t, ct.c_int]
        self.lib.hipMalloc.argtypes = [ct.POINTER(P), ct.c_size_t]
        self.lib.hipDeviceGetAttribute.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]

    @staticmethod
    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"HIP call failed with status {status}")

    def malloc(self, size: int) -> P:
        pointer = P()
        self.check(self.lib.hipMalloc(ct.byref(pointer), max(size, 1)))
        return pointer

    def upload(self, array: np.ndarray) -> P:
        array = np.ascontiguousarray(array)
        pointer = self.malloc(array.nbytes)
        self.check(self.lib.hipMemcpy(pointer, array.ctypes.data_as(P), array.nbytes, 1))
        return pointer

    def download(self, pointer: P, like: np.ndarray) -> np.ndarray:
        out = np.empty_like(like)
        self.check(self.lib.hipDeviceSynchronize())
        self.check(self.lib.hipMemcpy(out.ctypes.data_as(P), pointer, out.nbytes, 2))
        return out

    def module(self, image: bytes, symbol: str) -> tuple[P, P, object]:
        blob = ct.create_string_buffer(image, len(image))
        module, function = P(), P()
        self.check(self.lib.hipModuleLoadData(ct.byref(module), ct.cast(blob, P)))
        self.check(self.lib.hipModuleGetFunction(ct.byref(function), module, symbol.encode()))
        return module, function, blob

    def wall_clock_rate_khz(self) -> int:
        """hipDeviceAttributeWallClockRate, the rate the marker's steady counter
        ticks at. The enum value comes from compiling against THIS host's HIP
        header (the probe record_ssd_gpu.py uses). A header parse was tried
        first and read the wrong enumerator -- a rate of 1 kHz -- because the
        enum's explicit-value ranges are not one line per entry."""
        include = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "include"
        with tempfile.TemporaryDirectory(prefix="hip-enum-") as tmp:
            src, exe = Path(tmp) / "probe.c", Path(tmp) / "probe"
            src.write_text("#include <hip/hip_runtime_api.h>\n#include <stdio.h>\n"
                           'int main(void){printf("%d",(int)hipDeviceAttributeWallClockRate);'
                           "return 0;}\n")
            subprocess.run(["cc", "-D__HIP_PLATFORM_AMD__", "-I", str(include), str(src), "-o",
                            str(exe)], check=True, capture_output=True, timeout=60)
            value = int(subprocess.run([str(exe)], check=True, capture_output=True, text=True,
                                       timeout=30).stdout)
        rate = ct.c_int()
        self.check(self.lib.hipDeviceGetAttribute(ct.byref(rate), value, 0))
        if rate.value <= 0:
            raise RuntimeError("hipDeviceAttributeWallClockRate is not positive")
        return rate.value


class Launch:
    """A kernel bound to its arguments, launchable repeatedly."""

    def __init__(self, hip: Hip, image: bytes, symbol: str, args: list, grid, block, shared=0):
        self.hip = hip
        self.module, self.function, self._blob = hip.module(image, symbol)
        self._args = args
        self.argv = (P * len(args))(*[ct.cast(ct.byref(a), P) for a in args])
        self.grid, self.block, self.shared = grid, block, shared

    def __call__(self) -> None:
        self.hip.check(self.hip.lib.hipModuleLaunchKernel(
            self.function, *self.grid, *self.block, self.shared, None, self.argv, None))


# --------------------------------------------------------------------------
# Inputs and the oracle
# --------------------------------------------------------------------------
def make_inputs(m: int, n: int, k: int, *, seed: int):
    rng = np.random.default_rng(seed)
    groups, n_groups = k // 128, (n + 127) // 128
    a = (rng.standard_normal((m, k)) * 2.0).astype(ml_dtypes.float8_e4m3fn)
    b = (rng.standard_normal((k, n)) * 2.0).astype(ml_dtypes.float8_e4m3fn)
    a_scale = np.exp(rng.uniform(-2.0, 2.0, size=(m, groups))).astype(np.float32)
    b_scale = np.exp(rng.uniform(-2.0, 2.0, size=(groups, n_groups))).astype(np.float32)
    return a, b, a_scale, b_scale


def check_close(got: np.ndarray, want: np.ndarray, magnitude: np.ndarray, *, rel: float) -> float:
    err = np.abs(got.astype(np.float64) - want)
    worst = float((err / (magnitude + 1e-30)).max())
    if worst > rel:
        raise SystemExit(f"result disagrees with the fp64 oracle: {worst:.3e} > {rel:.1e}")
    return worst


# --------------------------------------------------------------------------
# Tessera
# --------------------------------------------------------------------------
def memref(pointer: P, size: int) -> list:
    return [P(pointer.value), P(pointer.value), ct.c_int64(0), ct.c_int64(size), ct.c_int64(1)]


def tessera_launch(hip, device, shape: BlockScaleShape, *, panel=None, k_unroll=1,
                   scale_group_panels=-1, lds=None):
    """``panel`` overrides the carrier's macro tile; ``lds`` =
    (warps, pipeline_depth, stage_k, pad_bytes, prefetch) additionally
    overrides its staging to the LDS-staged multi-wave body. Both are
    sweep-only: such a row is labelled ``panel_override`` and never stands
    for the production route, which is whatever the Schedule chose."""
    program = lower_blockscale(shape)
    overridden = panel is not None or lds is not None
    tile_ir = program.tile_ir
    if panel is not None:
        tile_ir = re.sub(r"tessera\.macro_tile_m = \d+", f"tessera.macro_tile_m = {panel[0]}", tile_ir)
        tile_ir = re.sub(r"tessera\.macro_tile_n = \d+", f"tessera.macro_tile_n = {panel[1]}", tile_ir)
    stage_k = pad = prefetch = -1
    if lds is not None:
        warps, depth, stage_k, pad, prefetch = lds
        staging = "lds" if warps > 1 or depth > 1 else "global"
        tile_ir = re.sub(r'staging = "\w+"', f'staging = "{staging}"', tile_ir)
        tile_ir = re.sub(r"(?<![\w.])warps = \d+", f"warps = {warps}", tile_ir)
        tile_ir = re.sub(r"tessera\.pipeline_depth = \d+", f"tessera.pipeline_depth = {depth}",
                         tile_ir)
    if overridden:
        program = type(program)(program.shape, program.entry, program.graph_ir,
                                program.schedule_ir, tile_ir)
    package = package_blockscale(program, k_unroll=k_unroll, scale_group_panels=scale_group_panels,
                                 blockscale_stage_k=stage_k, blockscale_lds_pad_bytes=pad,
                                 blockscale_prefetch=prefetch)
    prov = package.descriptor.provenance
    block_m, block_n = prov["macro_tile"]
    threads = int(prov["workgroup"][0])
    weight = device["wt"] if shape.weight_layout == "nk" else device["b"]
    args = (memref(device["a"], shape.m * shape.k) + memref(weight, shape.k * shape.n)
            + memref(device["sa"], shape.m * shape.groups)
            + memref(device["sb"], shape.groups * shape.n_groups)
            + memref(device["o32"], shape.m * shape.n)
            + [ct.c_int64(shape.m), ct.c_int64(shape.n), ct.c_int64(shape.k)])
    grid = ((shape.n + block_n - 1) // block_n, (shape.m + block_m - 1) // block_m, 1)
    launch = Launch(hip, package.image.payload, package.descriptor.entry_symbol, args, grid,
                    (threads, 1, 1))
    meta = {
        "route": prov["route"], "physical_route": prov["physical_route"],
        "panel_override": overridden, "macro_tile": [block_m, block_n],
        "staging": prov["staging"], "warps": prov["warps"],
        "pipeline_depth": prov["pipeline_depth"], "workgroup_threads": threads,
        "blockscale_stage_k": stage_k, "blockscale_lds_pad_bytes": pad,
        "blockscale_prefetch": prefetch,
        "macro_k": prov["macro_k"], "k_unroll": k_unroll,
        "scale_group_panels": scale_group_panels, "weight_layout": shape.weight_layout,
        "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "schedule_hash": prov["schedule_hash"], "abi_id": package.descriptor.abi_id,
        "output": "f32",
    }
    return launch, meta, package


# --------------------------------------------------------------------------
# AITER (imported unmodified from the local checkout, compiled AOT by Triton)
# --------------------------------------------------------------------------
def _aiter_kernel(aiter_root: Path):
    def pkg(name: str, path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    base = aiter_root / "aiter"
    for name, rel in [("aiter", ""), ("aiter.ops", "ops"), ("aiter.ops.triton", "ops/triton"),
                      ("aiter.ops.triton._triton_kernels", "ops/triton/_triton_kernels"),
                      ("aiter.ops.triton._triton_kernels.gemm", "ops/triton/_triton_kernels/gemm"),
                      ("aiter.ops.triton._triton_kernels.gemm.basic",
                       "ops/triton/_triton_kernels/gemm/basic"),
                      ("aiter.ops.triton.utils", "ops/triton/utils"),
                      ("aiter.ops.triton.utils._triton", "ops/triton/utils/_triton")]:
        pkg(name, base / rel)
    # The config helper and arch probe import a framework runtime; this
    # harness reads the gfx1201 JSON itself, so neither is needed.
    stub = types.ModuleType("aiter.ops.triton.utils.gemm_config_utils")
    stub.get_gemm_config = None
    sys.modules[stub.__name__] = stub
    module = importlib.import_module("aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale")
    kernel = module._gemm_a8w8_blockscale_kernel
    source = Path(module.__file__).read_bytes()
    return getattr(kernel, "fn", kernel), hashlib.sha256(source).hexdigest()


def aiter_config(aiter_root: Path, n: int, k: int, m: int) -> tuple[dict, str]:
    folder = aiter_root / "aiter/ops/triton/configs/gfx1201/triton/gemm/gemm_a8w8_blockscale"
    path = folder / f"GEMM-A8W8_BLOCKSCALE-N={n}-K={k}.json"
    if not path.is_file():
        path = folder / "DEFAULT.json"
    table = json.loads(path.read_text())
    for key in sorted((key for key in table if key.startswith("M_LEQ_")), key=lambda s: int(s[6:])):
        if m <= int(key[6:]):
            return table[key], f"{path.name}:{key}"
    return table["any"], f"{path.name}:any"


def aiter_launch(hip, device, fn, m: int, n: int, k: int, config: dict):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if config["NUM_KSPLIT"] != 1:
        raise NotImplementedError("AITER split-K bucket (needs its reduce kernel)")
    if config["BLOCK_SIZE_K"] != 128 or k % 128:
        raise NotImplementedError("harness drives GROUP_K == BLOCK_SIZE_K == 128 only")
    values = dict(M=m, N=n, K=k, stride_am=k, stride_ak=1, stride_bk=1, stride_bn=k,
                  stride_ck=m * n, stride_cm=n, stride_cn=1,
                  stride_ascale_m=k // 128, stride_ascale_k=1,
                  stride_bscale_k=1, stride_bscale_n=k // 128)
    pointer_types = {"a_ptr": "*fp8e4nv", "b_ptr": "*fp8e4nv", "c_ptr": "*bf16",
                     "a_scale_ptr": "*fp32", "b_scale_ptr": "*fp32"}
    constexprs = dict(GROUP_K=128, GROUP_N=128, BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
                      BLOCK_SIZE_N=config["BLOCK_SIZE_N"], BLOCK_SIZE_K=128,
                      GROUP_SIZE_M=config["GROUP_SIZE_M"], NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=k,
                      EVEN_K=True, cache_modifier=config.get("cache_modifier"),
                      num_stages=config["num_stages"])
    signature, attrs, runtime = {}, {}, []
    pointers = {"a_ptr": device["a"], "b_ptr": device["wt"], "c_ptr": device["o16"],
                "a_scale_ptr": device["sa"], "b_scale_ptr": device["sbt"]}
    for index, name in enumerate(fn.arg_names):
        if name in pointer_types:
            signature[name] = pointer_types[name]
            attrs[(index,)] = [["tt.divisibility", 16]]
            runtime.append(P(pointers[name].value))
        elif name in values and values[name] == 1:
            signature[name] = "constexpr"
            constexprs[name] = 1
        elif name in values:
            signature[name] = "i32"
            if values[name] % 16 == 0:
                attrs[(index,)] = [["tt.divisibility", 16]]
            runtime.append(ct.c_int32(values[name]))
        else:
            signature[name] = "constexpr"
    source = ASTSource(fn=fn, signature=signature,
                       constexprs={(fn.arg_names.index(k_),): v for k_, v in constexprs.items()},
                       attrs=attrs)
    # A key the tuned JSON omits is left to Triton's default, exactly as the
    # wrapper's `**config` launch would leave it.
    options = {key: config[key] for key in
               ("num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim", "kpack")
               if key in config}
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx1201", 32), options=options)
    meta = compiled.metadata
    if meta.global_scratch_size or meta.profile_scratch_size:
        raise NotImplementedError("AITER kernel requests scratch; harness passes none")
    runtime += [P(0), P(0)]  # global_scratch, profile_scratch (both size 0)
    grid = (math.ceil(m / config["BLOCK_SIZE_M"]) * math.ceil(n / config["BLOCK_SIZE_N"]), 1, 1)
    launch = Launch(hip, compiled.asm["hsaco"], meta.name, runtime, grid,
                    (meta.num_warps * 32, 1, 1), meta.shared)
    wmma = sorted(set(re.findall(r"v_wmma_\w+", compiled.asm["amdgcn"])))
    info = {"triton_version": triton.__version__, "kernel": meta.name,
            "hsaco_sha256": hashlib.sha256(compiled.asm["hsaco"]).hexdigest(),
            "wmma": wmma, "shared_bytes": meta.shared, "num_warps": meta.num_warps,
            "output": "bf16"}
    return launch, info


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
class Clock:
    def __init__(self, hip: Hip, marker_image: bytes, marker_entry: str) -> None:
        self.hip = hip
        self.rate_khz = hip.wall_clock_rate_khz()
        self.module, self.marker, self._blob = hip.module(marker_image, marker_entry)
        self.span = hip.malloc(16)
        self.marker_argv = (P * 1)(ct.cast(ct.byref(self.span), P))
        self.events = [P(), P()]
        for event in self.events:
            hip.check(hip.lib.hipEventCreate(ct.byref(event)))
        self.host_span = (ct.c_uint64 * 2)()

    def window(self, launch: Launch, count: int) -> dict:
        hip = self.hip
        self.host_span[0], self.host_span[1] = (1 << 64) - 1, 0
        hip.check(hip.lib.hipMemcpy(self.span, ct.addressof(self.host_span), 16, 1))
        hip.check(hip.lib.hipDeviceSynchronize())
        start = time.perf_counter_ns()
        hip.check(hip.lib.hipEventRecord(self.events[0], None))
        hip.check(hip.lib.hipModuleLaunchKernel(self.marker, 1, 1, 1, 1, 1, 1, 0, None,
                                                self.marker_argv, None))
        for _ in range(count):
            launch()
        hip.check(hip.lib.hipModuleLaunchKernel(self.marker, 1, 1, 1, 1, 1, 1, 0, None,
                                                self.marker_argv, None))
        hip.check(hip.lib.hipEventRecord(self.events[1], None))
        hip.check(hip.lib.hipEventSynchronize(self.events[1]))
        host_ns = time.perf_counter_ns() - start
        ms = ct.c_float()
        hip.check(hip.lib.hipEventElapsedTime(ct.byref(ms), self.events[0], self.events[1]))
        hip.check(hip.lib.hipMemcpy(ct.addressof(self.host_span), self.span, 16, 2))
        if self.host_span[0] == (1 << 64) - 1 or self.host_span[1] <= self.host_span[0]:
            raise SystemExit(f"device-clock span was not written: {list(self.host_span)}")
        device_ns = (self.host_span[1] - self.host_span[0]) * 1_000_000 / self.rate_khz
        return {"launches": count, "device_ns": device_ns, "event_ns": ms.value * 1e6,
                "host_ns": host_ns}


def build_marker(compiler: Path):
    from tessera.compiler.llvm_tools import llvm_bin_dir
    from tessera.compiler.native_device_clock import build_device_clock_marker

    llvm_bin = llvm_bin_dir()
    if llvm_bin is None:
        raise SystemExit("matched LLVM 23 tools not found (set TESSERA_LLVM_BIN)")
    return build_device_clock_marker(compiler=compiler, llvm_bin=llvm_bin, backend="rocm",
                                     chip="gfx1201")


def paired_windows(clock: Clock, arms: dict, *, windows: int, min_window_ms: float) -> dict:
    """ABAB / BABA interleaved windows, each at least ``min_window_ms`` long."""
    counts = {}
    # Warm every arm together first. Arms calibrated right after a host-side
    # compile (Triton's takes seconds) were sized on a device still ramping
    # its clock, and their measured windows then came in at 1.5-4.6 ms.
    warm_until = time.perf_counter() + 0.5
    while time.perf_counter() < warm_until:
        for launch in arms.values():
            for _ in range(20):
                launch()
        clock.hip.check(clock.hip.lib.hipDeviceSynchronize())
    for name, launch in arms.items():
        for _ in range(3):
            launch()
        # Size each arm's window on a WARM probe and then confirm it: a single
        # cold probe overestimated the per-launch time (the device was still
        # ramping) and left short kernels' windows at 2-4 ms, under the ROCm
        # device-clock admission floor. Grow until a confirmation window
        # clears the floor with 20% headroom.
        count = 50
        for _ in range(6):
            probe = clock.window(launch, count)
            if probe["device_ns"] >= 1.2 * min_window_ms * 1e6:
                break
            per = probe["device_ns"] / count
            count = max(count * 2, math.ceil(1.3 * min_window_ms * 1e6 / per))
        counts[name] = count
    names = list(arms)
    for attempt in range(3):
        rows: dict = {name: [] for name in arms}
        for index in range(windows):
            for name in (names if index % 2 == 0 else list(reversed(names))):
                rows[name].append(clock.window(arms[name], counts[name]))
        # Every window must clear the floor; an arm that fell short is
        # re-sized from its own measured windows and the whole paired set is
        # re-run, so the arms stay interleaved with each other.
        short = {name: min(s["device_ns"] for s in samples) / 1e6
                 for name, samples in rows.items()
                 if min(s["device_ns"] for s in samples) < min_window_ms * 1e6}
        if not short:
            break
        for name, shortest_ms in short.items():
            counts[name] = math.ceil(counts[name] * 1.3 * min_window_ms / shortest_ms)
    summary = {}
    for name, samples in rows.items():
        per_device = [s["device_ns"] / s["launches"] for s in samples]
        per_event = [s["event_ns"] / s["launches"] for s in samples]
        agreement = [abs(d - e) / e for d, e in zip(per_device, per_event)]
        summary[name] = {
            "launches_per_window": counts[name],
            "paired_set_attempts": attempt + 1,
            "window_ms_min": min(s["device_ns"] for s in samples) / 1e6,
            "median_us": float(np.median(per_device)) / 1e3,
            "min_us": float(np.min(per_device)) / 1e3,
            "event_median_us": float(np.median(per_event)) / 1e3,
            "device_event_disagreement_max": max(agreement),
            # The ROCm device-clock witness is admitted for windows of at least
            # 5 ms whose clocks agree within 5% (WSL-TIMING-ADMISSION-2026-09-26).
            "admissible": (min(s["device_ns"] for s in samples) >= 5e6
                           and max(agreement) <= 0.05),
            "windows": samples,
        }
    return summary


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--aiter-root", type=Path, default=Path.home() / "programming/aiter")
    parser.add_argument("--shape", action="append", default=[],
                        help="M,N,K (repeatable); K and N multiples of 128")
    parser.add_argument("--sweep", action="append", default=[],
                        help="Tessera-only sweep variant PMxPN:U:G:L (register panel) or "
                             "lds:MMxMN:W:D:S:P:F:L (LDS body: macro tile, warps, pipeline "
                             "depth, stage K (-1 default), pad bytes (-1 default), prefetch "
                             "(-1 = carrier), layout); no AITER arm unless --with-aiter")
    parser.add_argument("--alt-compiler", action="append", default=[],
                        help="NAME=PATH: a second tessera-opt a sweep variant names with a "
                             "trailing @NAME, so two compiler builds are timed paired and "
                             "interleaved in one process (diagnostic A/B)")
    parser.add_argument("--with-aiter", action="store_true",
                        help="also time AITER alongside --sweep variants")
    parser.add_argument("--no-production", action="store_true",
                        help="omit the production kn arm (the nk production arm is always kept)")
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--min-window-ms", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Assigned, not defaulted: an inherited TESSERA_OPT would otherwise
    # compile every arm with a binary other than the one recorded below.
    os.environ["TESSERA_OPT"] = str(args.compiler.resolve())
    alt_compilers = {}
    for item in args.alt_compiler:
        name, _, path = item.partition("=")
        alt_compilers[name] = Path(path).resolve()

    hip = Hip()
    marker = build_marker(args.compiler)
    clock = Clock(hip, marker.image, marker.entry)
    fn, aiter_source_sha = ((None, None) if args.sweep and not args.with_aiter
                            else _aiter_kernel(args.aiter_root))

    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()

    record = {
        "work_item": "ROCM-FP8-BLOCKSCALE-1", "sync_key": "GFX1201-LANES-2026-09-27",
        "host": platform.node(), "kernel_release": platform.release(),
        "source_commit": git("rev-parse", "HEAD"), "worktree_dirty": bool(git("status", "--porcelain")),
        "compiler": str(args.compiler.resolve()),
        "compiler_sha256": hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        "alt_compilers": {name: {"path": str(path),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for name, path in alt_compilers.items()},
        "timing_source": "device_clock_marker (llvm.readsteadycounter), hip_event + host wall cross-checks",
        "marker_image_sha256": marker.image_sha256, "wall_clock_rate_khz": clock.rate_khz,
        "windows": args.windows, "min_window_ms": args.min_window_ms,
        "aiter_kernel_source_sha256": aiter_source_sha,
        "rows": [],
    }
    for text in args.shape:
        m, n, k = (int(v) for v in text.split(","))
        shape = BlockScaleShape(m, n, k, 128, 128)
        a, b, sa, sb = make_inputs(m, n, k, seed=m * 31 + n * 7 + k)
        want = blockscale_reference(a.astype(np.float32), b.astype(np.float32), sa, sb,
                                    scale_k=128, scale_n=128)
        magnitude = blockscale_reference(np.abs(a.astype(np.float32)), np.abs(b.astype(np.float32)),
                                         sa, sb, scale_k=128, scale_n=128)
        device = {
            "a": hip.upload(a.view(np.uint8)), "b": hip.upload(b.view(np.uint8)),
            "wt": hip.upload(np.ascontiguousarray(b.T).view(np.uint8)),
            "sa": hip.upload(sa), "sb": hip.upload(sb),
            "sbt": hip.upload(np.ascontiguousarray(sb.T)),
            "o32": hip.malloc(m * n * 4), "o16": hip.malloc(m * n * 2),
        }
        row: dict = {"shape": [m, n, k], "scale_block": [128, 128], "flop": 2 * m * n * k}
        arms: dict = {}
        # Production arms: both named weight layouts, generator defaults.
        # Sweep spec: PMxPN:U:G:L -- panel, whole groups per iteration, panels
        # per inner group step (-1 = default), weight layout kn|nk.
        variants: list = []
        if not args.sweep:
            variants = [(None, 1, -1, "kn", None), (None, 1, -1, "nk", None)]
        elif args.with_aiter:
            variants = [(None, 1, -1, "nk", None)]
            if not args.no_production:
                variants.insert(0, (None, 1, -1, "kn", None))
        for spec in args.sweep:
            spec, _, alias = spec.partition("@")
            parts = spec.split(":")
            if parts[0] == "lds":
                macro = tuple(int(v) for v in parts[1].split("x"))
                warps, depth, stage_k, pad, prefetch = (int(v) for v in parts[2:7])
                variants.append((macro, 1, -1, parts[7], (warps, depth, stage_k, pad, prefetch),
                                 alias))
            else:
                variants.append((tuple(int(v) for v in parts[0].split("x")), int(parts[1]),
                                 int(parts[2]), parts[3], None, alias))
        for panel, unroll, group_panels, layout, lds, *rest in variants:
            alias = rest[0] if rest else ""
            os.environ["TESSERA_OPT"] = str(alt_compilers[alias] if alias else
                                            args.compiler.resolve())
            if lds is not None:
                label = (f"tessera_{layout}_lds{panel[0]}x{panel[1]}_w{lds[0]}_d{lds[1]}"
                         f"_s{lds[2]}_p{lds[3]}_f{lds[4]}")
            elif panel is None:
                label = f"tessera_{layout}"
            else:
                label = f"tessera_{layout}_{panel[0]}x{panel[1]}_u{unroll}_g{group_panels}"
            if alias:
                label += f"@{alias}"
            shape = BlockScaleShape(m, n, k, 128, 128, layout)
            try:
                launch, meta, _ = tessera_launch(hip, device, shape, panel=panel, k_unroll=unroll,
                                                 scale_group_panels=group_panels, lds=lds)
            except Exception as error:  # a variant the generator refuses is a result, not a crash
                row[label] = {"refused": str(error)[:400]}
                continue
            launch()
            got = hip.download(device["o32"], np.zeros((m, n), np.float32))
            meta["oracle_max_rel_err"] = check_close(got, want, magnitude, rel=1e-5)
            row[label] = meta
            arms[label] = launch
        if fn is not None:
            config, source = aiter_config(args.aiter_root, n, k, m)
            try:
                launch, info = aiter_launch(hip, device, fn, m, n, k, config)
                launch()
                got = hip.download(device["o16"], np.zeros((m, n), ml_dtypes.bfloat16))
                info["oracle_max_rel_err"] = check_close(got, want, magnitude, rel=2 ** -7)
                info["config"] = config
                info["config_source"] = source
                row["aiter"] = info
                arms["aiter"] = launch
            except NotImplementedError as why:
                row["aiter"] = {"not_measured": str(why), "config": config, "config_source": source}
        timing = paired_windows(clock, arms, windows=args.windows, min_window_ms=args.min_window_ms)
        row["timing"] = timing
        for name, summary in timing.items():
            summary["tflops"] = row["flop"] / (summary["median_us"] * 1e-6) / 1e12
        if "aiter" in timing:
            # > 1 means Tessera is slower. Median per-launch device-clock time.
            row["time_over_aiter"] = {
                name: summary["median_us"] / timing["aiter"]["median_us"]
                for name, summary in timing.items() if name != "aiter"}
        print(json.dumps({"shape": row["shape"], **{k_: round(v["median_us"], 2)
                                                    for k_, v in timing.items()},
                          "time_over_aiter": row.get("time_over_aiter")}), flush=True)
        record["rows"].append(row)
        for pointer in device.values():
            hip.lib.hipFree(pointer)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
