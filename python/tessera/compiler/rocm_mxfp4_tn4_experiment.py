"""Manual TN4/BN128 folded-prefill ablation, not a selected schedule."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import tempfile

from .native_artifact import LaunchGeometry, NativeEntryPoint
from .rocm_mxfp4 import FoldedRowReference
from .rocm_mxfp4_folded import (
    emit_mxfp4_folded_prefill_hip, package_mxfp4_folded_prefill,
)
from .rocm_mxfp4_native import _extract_gfx1201_hsaco, _rocm_hipcc
from .rocm_native import ROCMNativePackage, _rocm_path


TN4_EXPERIMENT_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k."
    "e4m3_e4m3_e8m0_bf16.approx_bm256_tn4_experiment.v1"
)


def emit_folded_tn4_experiment_hip(
    entry: str = "tessera_mxfp4_folded_tn4",
) -> str:
    """Widen only the expanded-weight N tile while preserving K64 staging."""
    source = emit_mxfp4_folded_prefill_hip(entry, full_k64=True)

    def once(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"TN4 experiment expected one source site: {old[:55]}")
        source = source.replace(old, new)

    once("unsigned char sB[64 * 80]", "unsigned char sB[128 * 80]")
    once("floatx8 acc[4][2]", "floatx8 acc[4][4]")
    once("const long n0 = (long)blockIdx.x * 64;", "const long n0 = (long)blockIdx.x * 128;")
    once("fragment_i32x2 af[4], bf[2]", "fragment_i32x2 af[4], bf[4]")
    if source.count("wn * 32 + j * 16") != 2:
        raise RuntimeError("TN4 experiment expected two wave N-stride sites")
    source = source.replace("wn * 32 + j * 16", "wn * 64 + j * 16")
    old_b = '''    {
      const long row = n0 + tid / 4;
      const int off = (tid & 3) * 16;
      copy_u32x4 value = {};
      if constexpr (true) {
        const long safe = row < N ? row : N - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);
      } else if (kb + off < K) {
        const long safe = row < N ? row : N - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);
      }
      *reinterpret_cast<copy_u32x4 *>(sB + (tid / 4) * 80 + off) = value;
    }'''
    new_b = '''#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int slot = tid + q * 256;
      const long row = n0 + slot / 4;
      const int off = (slot & 3) * 16;
      const long safe = row < N ? row : N - 1;
      const copy_u32x4 value = *reinterpret_cast<const copy_u32x4 *>(
          B + safe * K + kb + off);
      *reinterpret_cast<copy_u32x4 *>(sB + (slot / 4) * 80 + off) = value;
    }'''
    once(old_b, new_b)
    if source.count("for (int j = 0; j < 2; ++j)") != 4:
        raise RuntimeError("TN4 experiment expected four N-fragment loops")
    source = source.replace("for (int j = 0; j < 2; ++j)", "for (int j = 0; j < 4; ++j)")
    return source


def package_folded_tn4_experiment(
    m: int, n: int, k: int, folded: FoldedRowReference,
) -> ROCMNativePackage:
    """Build an isolated manual candidate; never register it as proved ABI."""
    if k % 64 or n < 128:
        raise ValueError("TN4 experiment requires K64 and N >= 128")
    baseline = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=True,
    )
    entry = "tessera_mxfp4_folded_tn4"
    source = emit_folded_tn4_experiment_hip(entry)
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("TN4 experiment requires the HIP compiler driver")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-tn4-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler), "-x", "hip", "-O3", "--genco",
            "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
            str(source_path), "-o", str(bundle_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle_path.is_file():
            raise RuntimeError(
                "TN4 HSACO compilation failed: "
                + (result.stderr.strip() or f"hipcc exited {result.returncode}")
            )
        payload = _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)
    image = replace(
        baseline.image, payload=payload,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        entry_points=(NativeEntryPoint(entry, TN4_EXPERIMENT_ABI),),
    )
    provenance = dict(baseline.descriptor.provenance)
    provenance.update({
        "sync_key": "GFX1201-MXFP4-TN4-ABLATION-2026-09-23",
        "route": "folded_row_reference_bm256_tn4_manual",
        "block_n": 128, "tile_n_per_wave": 4,
        "execution_state": "manual_executable_candidate",
    })
    descriptor = replace(
        baseline.descriptor, image_digest=image.image_digest,
        entry_symbol=entry, abi_id=TN4_EXPERIMENT_ABI,
        geometry=LaunchGeometry(grid=((n + 127) // 128, (m + 255) // 256, 1), workgroup=(256, 1, 1)),
        provenance=provenance,
    )
    return replace(
        baseline, tile_ir=f"rocm.mxfp4 folded TN4 experiment M={m} N={n} K={k}",
        target_ir=source, backend_ir=" ".join(command[:-2]),
        image=image, descriptor=descriptor,
    )


__all__ = ["TN4_EXPERIMENT_ABI", "emit_folded_tn4_experiment_hip", "package_folded_tn4_experiment"]
