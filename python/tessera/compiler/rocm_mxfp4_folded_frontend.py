"""Shape-specialized frontend for the opt-in gfx1201 folded scaled matmul.

This is a physical package author, not a public logical MXFP4 dtype.  It
constructs the Graph op from typed operands rather than replaying a checked-in
MLIR fixture, and never selects the approximate route implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .rocm_mxfp4 import FoldedRowReference
from .rocm_mxfp4_folded import GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI
from .rocm_mxfp4_folded_carrier import package_folded_scaled_wmma_target_ir
from .rocm_native import ROCMNativePackage


@dataclass(frozen=True)
class FoldedScaledMatmulProgram:
    """A compiled Graph-owned folded package and its selection receipt."""

    package: ROCMNativePackage
    graph_ir: str

    @property
    def route_receipt(self) -> dict[str, object]:
        package = self.package
        provenance = package.descriptor.provenance
        return {
            "route": provenance["route"],
            "physical_contract": provenance["physical_contract"],
            "selected_schedule": {
                "block_m": provenance["block_m"],
                "block_n": provenance["block_n"],
                "block_k": provenance["block_k"],
                "tile_m_per_wave": provenance["tile_m_per_wave"],
                "tile_n_per_wave": provenance["tile_n_per_wave"],
            },
            "schedule_hash": provenance["schedule_hash"],
            "abi_id": package.descriptor.abi_id,
            "entry_symbol": package.descriptor.entry_symbol,
            "hsaco_sha256": package.image.payload_digest,
            "artifact_image_digest": package.image.image_digest,
            "graph_ir_sha256": hashlib.sha256(self.graph_ir.encode()).hexdigest(),
            "tile_ir_sha256": provenance["tile_ir_sha256"],
            "target_ir_sha256": provenance["target_ir_sha256"],
            "numeric_policy": provenance["numeric_policy"],
            "fold_lossless": provenance["fold_lossless"],
            "fold_inexact_value_count": provenance["fold_inexact_value_count"],
        }


def author_folded_scaled_matmul_graph(m: int, n: int, k: int) -> str:
    """Author the distinct approximate physical Graph operation."""
    if min(m, n, k) <= 0 or m <= 64 or k % 64:
        raise ValueError("folded scaled_matmul requires M>64, N>0, K divisible by 64")
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @folded_w4a8(%a: tensor<{m}x{k}xui8>,
                         %b: tensor<{n}x{k}xui8>,
                         %sa: tensor<{m}xf32>,
                         %ref: tensor<{n}xui8>) -> tensor<{m}x{n}xbf16> {{
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {{
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {{accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"}},
      scale_layout = {{granularity = "output_column", block = [1, {k}], format = "e8m0_row_reference"}}
    }} : (tensor<{m}x{k}xui8>, tensor<{n}x{k}xui8>, tensor<{m}xf32>,
         tensor<{n}xui8>) -> tensor<{m}x{n}xbf16>
    return %0 : tensor<{m}x{n}xbf16>
  }}
}}
'''


def _lower(tessera_opt: Path, graph_ir: str, *, target: bool) -> str:
    command = [
        str(tessera_opt), "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
    ]
    if target:
        command.append("--lower-tile-to-rocm=arch=gfx1201")
    result = subprocess.run(
        command, input=graph_ir, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "folded scaled_matmul lowering failed: "
            + (result.stderr.strip() or f"tessera-opt exited {result.returncode}")
        )
    return result.stdout


def compile_folded_scaled_matmul(
    a: np.ndarray, a_scale: np.ndarray, folded: FoldedRowReference, *,
    tessera_opt: Path, allow_approximate: bool = False,
) -> FoldedScaledMatmulProgram:
    """Compile a typed frontend call into the exact gfx1201 folded ABI.

    Callers retain the returned package and launch it with matching buffers;
    checkpoint conversion belongs to model loading, not each invocation.
    """
    if not allow_approximate or folded.approximate_policy != "explicit_allow":
        raise ValueError("folded scaled_matmul requires explicit approximate policy")
    if a.ndim != 2 or a.dtype != np.uint8 or not a.flags.c_contiguous:
        raise ValueError("folded scaled_matmul A must be contiguous raw E4M3 [M,K]")
    m, k = a.shape
    if a_scale.shape != (m,) or a_scale.dtype != np.float32:
        raise ValueError("folded scaled_matmul token scales must be fp32 [M]")
    if folded.weight_bytes.ndim != 2 or folded.weight_bytes.shape[1] != k:
        raise ValueError("folded scaled_matmul weight K must match A")
    n = folded.weight_bytes.shape[0]
    if folded.row_reference.shape != (n,):
        raise ValueError("folded scaled_matmul row reference must be [N]")
    if not tessera_opt.is_file():
        raise FileNotFoundError(f"folded scaled_matmul compiler not found: {tessera_opt}")
    graph_ir = author_folded_scaled_matmul_graph(m, n, k)
    tile_ir = _lower(tessera_opt, graph_ir, target=False)
    target_ir = _lower(tessera_opt, graph_ir, target=True)
    package = package_folded_scaled_wmma_target_ir(
        tile_ir, target_ir, folded, allow_approximate=True,
    )
    if package.descriptor.abi_id != GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI:
        raise ValueError("folded scaled_matmul selected a mismatched package ABI")
    return FoldedScaledMatmulProgram(package=package, graph_ir=graph_ir)


__all__ = [
    "FoldedScaledMatmulProgram",
    "author_folded_scaled_matmul_graph",
    "compile_folded_scaled_matmul",
]
