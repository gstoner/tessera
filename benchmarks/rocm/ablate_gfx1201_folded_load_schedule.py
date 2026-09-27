#!/usr/bin/env python3
"""Load-schedule engines and diagnostic ablations for folded gfx1201 MXFP4.

Owner ROCM-MXFP4-W4A8-1; sync GFX1201-LANES-2026-09-27. An engine here is
the production folded kernel under an explicit
:class:`~tessera.compiler.rocm_mxfp4_folded.FoldedPrefillSchedule` (the same
checked source edits the packager applies), optionally with one of the
diagnostic transforms below applied on top. None of them changes the tile
(BM256/BN64/BK64, TM4/TN2), the E4M3 operands, the per-lane WMMA order, or
any element's epilogue arithmetic, so each must produce bitwise-identical
BF16 output; the recorder refuses timing otherwise. Probes are the only
exception and are labelled ``diagnostic_bound``.

This module never registers an ABI or changes what the compiled Graph route
packages; the selected schedule is chosen in Target IR.

Diagnostic transforms (not production schedules):

``uncond``
    The four K16 compute steps of a complete K64 slab run without the
    per-step ``kb + step * 16 < K`` guard.
``lds_pipe``
    ``uncond`` plus one-step look-ahead fragment double buffering: step
    ``s + 1``'s LDS fragments are requested before step ``s``'s WMMAs.
``lds_barrier``
    Both main-loop workgroup barriers fence LDS only (release/acquire on the
    ``local`` address space around ``s_barrier``).
``probe_a_hot`` / ``probe_b_hot``
    Output-changing attribution probes: the operand's global loads read a
    cache-hot 64-byte column (``kb & 0``) instead of the K slab. They bound
    what that stream's traffic costs and are never candidates.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from tessera.compiler.rocm_mxfp4_folded import (
    FOLDED_PREFILL_SCHEDULE_V1, FoldedPrefillSchedule,
    emit_mxfp4_folded_prefill_hip, package_mxfp4_folded_prefill,
)
from tessera.compiler.rocm_mxfp4_native import _extract_gfx1201_hsaco, _rocm_hipcc
from tessera.compiler.rocm_native import _rocm_path
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.inspect_gfx1201_folded_prefill import (
    selected_symbol_isa_evidence,
)


DIAGNOSTICS = ("uncond", "lds_pipe", "lds_barrier")
PROBES = ("probe_a_hot", "probe_b_hot")

_UNCOND_OLD = "for (int step = 0; step < 4 && kb + step * 16 < K; ++step) {"
_UNCOND_NEW = "for (int step = 0; step < 4; ++step) {"

_COMPUTE_OLD = '''    for (int step = 0; step < 4; ++step) {
      fragment_i32x2 af[4], bf[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned char *p = sA + (wm * 64 + i * 16 + col) * 80 + step * 16 + half;
        af[i][0] = *reinterpret_cast<const int *>(p);
        af[i][1] = *reinterpret_cast<const int *>(p + 4);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const unsigned char *p = sB + (wn * 32 + j * 16 + col) * 80 + step * 16 + half;
        bf[j][0] = *reinterpret_cast<const int *>(p);
        bf[j][1] = *reinterpret_cast<const int *>(p + 4);
      }
      __builtin_amdgcn_sched_barrier(6);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
              af[i], bf[j], acc[i][j]);
    }
'''

_COMPUTE_PIPE = '''    {
      // One-step look-ahead: step s + 1's fragments are requested before
      // step s's WMMAs, so LDS latency overlaps matrix work.
      fragment_i32x2 af[2][4], bf[2][2];
      auto load_step = [&](int buf, int step) __attribute__((always_inline)) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned char *p = sA + (wm * 64 + i * 16 + col) * 80 + step * 16 + half;
          af[buf][i][0] = *reinterpret_cast<const int *>(p);
          af[buf][i][1] = *reinterpret_cast<const int *>(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = sB + (wn * 32 + j * 16 + col) * 80 + step * 16 + half;
          bf[buf][j][0] = *reinterpret_cast<const int *>(p);
          bf[buf][j][1] = *reinterpret_cast<const int *>(p + 4);
        }
      };
      load_step(0, 0);
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        if (step + 1 < 4) load_step((step + 1) & 1, step + 1);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                af[step & 1][i], bf[step & 1][j], acc[i][j]);
      }
    }
'''

_LDS_BARRIER_HELPER = '''
// LDS-scoped workgroup barrier: release/acquire on the local address space
// only, so the barrier does not drain outstanding global loads.
static __device__ __forceinline__ void tessera_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
'''
_KERNEL_HEAD = "// Four row fragments and two column fragments per wave"
_COPY_BARRIER_OLD = (
    "    __syncthreads();\n#ifdef TESSERA_FOLDED_PHASE_TRACE\n"
    "    unsigned long long compute_start"
)
_TAIL_BARRIER_OLD = (
    "    __syncthreads();\n#ifdef TESSERA_FOLDED_PHASE_TRACE\n"
    "    __syncthreads();  // every wave finishes WMMA"
)

_A_LOAD = "value = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);"
_B_LOAD = "value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);"


def _once(source: str, old: str, new: str, what: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"folded load-schedule template changed near {what}")
    return source.replace(old, new)


def variant_source(
    schedule: FoldedPrefillSchedule = FOLDED_PREFILL_SCHEDULE_V1,
    diagnostics: tuple[str, ...] = (),
) -> str:
    """Production K64 source under ``schedule`` plus checked diagnostic edits."""
    unknown = sorted(set(diagnostics) - set(DIAGNOSTICS) - set(PROBES))
    if unknown:
        raise ValueError(f"unknown folded load-schedule diagnostics: {unknown}")
    chosen = set(diagnostics)
    if "lds_pipe" in chosen:
        chosen.add("uncond")
    source = emit_mxfp4_folded_prefill_hip(full_k64=True, schedule=schedule)
    for probe, old in (("probe_a_hot", _A_LOAD), ("probe_b_hot", _B_LOAD)):
        if probe in chosen:
            if schedule.staging_prefetch != "none":
                raise ValueError("hot-operand probes apply to the unprefetched copy only")
            # Two textual K64/K32 branches; hipcc keeps only the K64 one.
            if source.count(old) != 2:
                raise RuntimeError(f"folded load-schedule template changed near {probe}")
            source = source.replace(old, old.replace("+ kb +", "+ (kb & 0) +"))
    if "uncond" in chosen:
        source = _once(source, _UNCOND_OLD, _UNCOND_NEW, "K16 step guard")
    if "lds_pipe" in chosen:
        source = _once(source, _COMPUTE_OLD, _COMPUTE_PIPE, "K16 compute block")
    if "lds_barrier" in chosen:
        source = _once(source, _KERNEL_HEAD, _LDS_BARRIER_HELPER + _KERNEL_HEAD, "kernel head")
        source = _once(source, _COPY_BARRIER_OLD,
                       _COPY_BARRIER_OLD.replace("__syncthreads()", "tessera_lds_barrier()", 1),
                       "copy barrier")
        source = _once(source, _TAIL_BARRIER_OLD,
                       _TAIL_BARRIER_OLD.replace("__syncthreads()", "tessera_lds_barrier()", 1),
                       "tail barrier")
    return source


def schedule_label(
    schedule: FoldedPrefillSchedule, diagnostics: tuple[str, ...] = (),
) -> str:
    parts = []
    if schedule.raster_group_m:
        parts.append(f"raster_g{schedule.raster_group_m}")
    if schedule.workgroup_mode != "wgp":
        parts.append(schedule.workgroup_mode + "mode")
    if schedule.staging_prefetch != "none":
        parts.append("prefetch")
    if schedule.epilogue != "predicated_scalar_scales":
        parts.append("vector_epilogue")
    parts.extend(d for d in DIAGNOSTICS + PROBES if d in diagnostics)
    return "+".join(parts) if parts else "v1"


def compile_source(source: str, flags: tuple[str, ...]) -> bytes:
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("folded load-schedule ablation requires hipcc")
    with tempfile.TemporaryDirectory(prefix="tessera-folded-lsched-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        result = subprocess.run(
            [
                str(compiler), "-x", "hip", "-O3", "--genco", *flags,
                "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
                str(source_path), "-o", str(bundle_path),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "folded load-schedule compilation failed: " + result.stderr[-800:]
            )
        return _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)


def package_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    folded: Any, package: Any, *, name: str, metadata: dict[str, object],
) -> base._Engine:
    """Launch ``package`` with its own descriptor geometry on rotating copies."""
    payload = package.image.payload
    descriptor = package.descriptor
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), payload) != 0:
        raise RuntimeError(f"folded engine {name} module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, descriptor.entry_symbol.encode(),
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError(f"folded engine {name} entry missing")
    grid = descriptor.geometry.grid
    workgroup = descriptor.geometry.workgroup
    if grid is None or workgroup is None:
        hip.hipModuleUnload(module)
        raise RuntimeError("folded engine requires fixed launch geometry")
    arrays = (
        inputs["a"], folded.weight_bytes, inputs["a_scale"],
        folded.row_reference, inputs["output"],
    )
    try:
        device_copies = base._copies(hip, arrays, 3)
    except Exception:
        hip.hipModuleUnload(module)
        raise

    def launch(bundle: base._DeviceArrays) -> None:
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in bundle.device),
            ctypes.c_int64(case.m), ctypes.c_int64(case.n), ctypes.c_int64(case.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        rc = hip.hipModuleLaunchKernel(
            function, *grid, *workgroup, 0, None, arguments, None,
        )
        if rc != 0:
            raise RuntimeError(f"folded engine {name} launch failed rc={rc}")

    engine = base._Engine(
        name, hip, device_copies, launch, 4,
        {
            **metadata,
            "grid": list(grid),
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "abi": descriptor.abi_id,
            "selected_isa": selected_symbol_isa_evidence(
                payload, descriptor.entry_symbol,
            ),
            **base._code_object_evidence(payload),
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine


def schedule_engine(
    hip: ctypes.CDLL, case: base.Case, inputs: dict[str, np.ndarray],
    folded: Any, schedule: FoldedPrefillSchedule,
    diagnostics: tuple[str, ...] = (), *, name: str | None = None,
) -> base._Engine:
    """Compile ``schedule`` (plus diagnostics) and bind its own descriptor."""
    package = package_mxfp4_folded_prefill(
        case.m, case.n, case.k, folded, allow_approximate=True, schedule=schedule,
    )
    label = schedule_label(schedule, diagnostics)
    metadata: dict[str, object] = {
        "schedule": schedule.as_dict(), "diagnostics": sorted(diagnostics),
        "diagnostic_bound": bool(set(diagnostics) & set(PROBES)),
        "compile_flags": list(schedule.compile_flags()),
        "route": "direct_package" if not diagnostics else "diagnostic_source_edit",
    }
    if diagnostics:
        source = variant_source(schedule, diagnostics)
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        payload = compile_source(source, schedule.compile_flags())
        image = replace(package.image, payload=payload, target_ir_digest=source_sha)
        package = replace(
            package, image=image,
            descriptor=replace(package.descriptor, image_digest=image.image_digest),
        )
        metadata["source_sha256"] = source_sha
    else:
        metadata["source_sha256"] = package.image.target_ir_digest
    return package_engine(
        hip, case, inputs, folded, package,
        name=name or f"tessera_{label}", metadata=metadata,
    )


__all__ = [
    "DIAGNOSTICS", "PROBES", "compile_source", "package_engine",
    "schedule_engine", "schedule_label", "variant_source",
]
