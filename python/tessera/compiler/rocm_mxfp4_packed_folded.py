"""Candidate packed-weight payload for a distinct gfx1201 folded prefill ABI.

This is a model-load representation and Graph→Target artifact contract, not
an executable route. The current folded HSACO consumes expanded E4M3 bytes
and must reject this object. An independent packed-decode kernel still needs
proof before registering execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .rocm_mxfp4 import (
    FoldedRowReference,
    MXFP4_CHECKPOINT_LAYOUT_V1,
    MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    convert_weight_layout,
)
from .rocm_mxfp4_folded import prepare_folded_weights
from .rocm_mxfp4_native import _schedule_hash, _target_string_attr


PACKED_FOLDED_SCALE_PLANE_V1 = "mxfp4.e8m0.k32_plus_row_reference.last_row.v1"
PACKED_FOLDED_WEIGHT_LAYOUT_V1 = MXFP4_GFX12_FRAGMENT_LAYOUT_V1
PACKED_FOLDED_PHYSICAL_V1 = "rocm_mxfp4_w4a8_packed_folded_prefill_v1"
PACKED_FOLDED_TARGET_ABI_V1 = (
    "tessera.rocm.mxfp4_w4a8.a_bpacked_sa_scaleplane_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.approx_bm256_tm4.v1"
)


@dataclass(frozen=True)
class PackedFoldedPayload:
    """Load-time packed B plus K32 E8M0 scales and one row reference.

    `weight_bytes` is fragment-order `[N,K/2]` with low nibble = even K.
    `scale_plane` is `[K/32+1,N]` in row-major uint8. Its last row is the
    E8M0 row reference used by the approximate folded epilogue; earlier rows
    are the original per-K32 exponents, including reserved zero-block codes.
    """

    weight_bytes: np.ndarray
    scale_plane: np.ndarray
    lossless: bool
    inexact_value_count: int
    max_normalized_abs_error: float
    max_normalized_relative_error: float
    approximate_policy: str = "explicit_allow"
    weight_layout: str = PACKED_FOLDED_WEIGHT_LAYOUT_V1
    scale_layout: str = PACKED_FOLDED_SCALE_PLANE_V1

    def __post_init__(self) -> None:
        weight = self.weight_bytes
        scales = self.scale_plane
        if self.approximate_policy != "explicit_allow":
            raise ValueError("packed folded payload requires explicit approximate policy")
        if self.weight_layout != PACKED_FOLDED_WEIGHT_LAYOUT_V1:
            raise ValueError("packed folded payload requires gfx12 fragment-order weights")
        if self.scale_layout != PACKED_FOLDED_SCALE_PLANE_V1:
            raise ValueError("packed folded payload requires the versioned scale plane")
        if weight.dtype != np.uint8 or weight.ndim != 2 or not weight.flags.c_contiguous:
            raise TypeError("packed folded weights must be contiguous uint8 [N,K/2]")
        n, half_k = weight.shape
        k = half_k * 2
        if n <= 0 or n % 16 or k <= 0 or k % 64:
            raise ValueError("packed folded weights require N%16=0 and K%64=0")
        if (scales.dtype != np.uint8 or scales.shape != (k // 32 + 1, n)
                or not scales.flags.c_contiguous):
            raise TypeError("packed folded scale plane must be contiguous uint8 [K/32+1,N]")
        if np.any(scales[:-1] == 255) or np.any(scales[-1] == 255):
            raise ValueError("packed folded E8M0 code 255 is reserved")
        if not np.array_equal(scales[-1], scales[:-1].max(axis=0)):
            raise ValueError("packed folded row reference must be the block-exponent maximum")
        if self.inexact_value_count < 0 or self.max_normalized_abs_error < 0:
            raise ValueError("packed folded loss metadata must be nonnegative")
        # Package/model loading owns these buffers. Do not retain mutable
        # checkpoint views whose bytes can change after a receipt is issued.
        stable_weight = np.array(weight, copy=True, order="C")
        stable_scales = np.array(scales, copy=True, order="C")
        stable_weight.setflags(write=False)
        stable_scales.setflags(write=False)
        object.__setattr__(self, "weight_bytes", stable_weight)
        object.__setattr__(self, "scale_plane", stable_scales)

    @property
    def shape(self) -> tuple[int, int]:
        return self.weight_bytes.shape[0], self.weight_bytes.shape[1] * 2

    def receipt(self) -> dict[str, object]:
        """Bind the candidate layout, scale plane, numerical loss and bytes."""
        return {
            "weight_layout": self.weight_layout,
            "scale_layout": self.scale_layout,
            "execution_mode": "folded_row_reference_explicit_approximate",
            "approximate_policy": self.approximate_policy,
            "shape": self.shape,
            "weight_sha256": hashlib.sha256(self.weight_bytes.tobytes()).hexdigest(),
            "scale_plane_sha256": hashlib.sha256(self.scale_plane.tobytes()).hexdigest(),
            "fold_lossless": self.lossless,
            "fold_inexact_value_count": self.inexact_value_count,
            "fold_max_normalized_abs_error": self.max_normalized_abs_error,
            "fold_max_normalized_relative_error": self.max_normalized_relative_error,
            "execution_state": "artifact_only",
        }


def prepare_packed_folded_payload(
    packed_checkpoint: np.ndarray, scale_exponents: np.ndarray, *,
    allow_approximate: bool = False,
) -> PackedFoldedPayload:
    """Convert once at model load while preserving the exact K32 scale plane."""
    folded = prepare_folded_weights(
        packed_checkpoint, scale_exponents,
        allow_approximate=allow_approximate,
    )
    fragment = convert_weight_layout(
        packed_checkpoint,
        source=MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    )
    plane = np.ascontiguousarray(
        np.concatenate((scale_exponents, folded.row_reference[None, :]), axis=0)
    )
    return PackedFoldedPayload(
        weight_bytes=fragment, scale_plane=plane,
        lossless=folded.lossless,
        inexact_value_count=folded.inexact_value_count,
        max_normalized_abs_error=folded.max_normalized_abs_error,
        max_normalized_relative_error=folded.max_normalized_relative_error,
    )


def folded_oracle_from_packed(payload: PackedFoldedPayload) -> FoldedRowReference:
    """Reconstruct the declared approximate oracle from the packed payload."""
    checkpoint = convert_weight_layout(
        payload.weight_bytes,
        source=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
        destination=MXFP4_CHECKPOINT_LAYOUT_V1,
    )
    result = prepare_folded_weights(
        checkpoint, payload.scale_plane[:-1], allow_approximate=True,
    )
    if not np.array_equal(result.row_reference, payload.scale_plane[-1]):
        raise ValueError("packed folded reference row disagrees with the oracle")
    return result


def author_packed_folded_graph(m: int, payload: PackedFoldedPayload) -> str:
    """Author the versioned packed physical Graph contract for this payload."""
    n, k = payload.shape
    if m <= 64:
        raise ValueError("packed folded prefill requires M>64")
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @packed_folded_w4a8(%a: tensor<{m}x{k}xui8>,
                                %b: tensor<{n}x{k // 2}xui8>,
                                %sa: tensor<{m}xf32>,
                                %plane: tensor<{k // 32 + 1}x{n}xui8>) -> tensor<{m}x{n}xbf16> {{
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %plane) {{
      physical_contract = "{PACKED_FOLDED_PHYSICAL_V1}",
      numeric_policy = {{accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"}},
      scale_layout = {{granularity = "output_column", block = [1, {k}], format = "e8m0_k32_plus_row_reference"}}
    }} : (tensor<{m}x{k}xui8>, tensor<{n}x{k // 2}xui8>, tensor<{m}xf32>,
         tensor<{k // 32 + 1}x{n}xui8>) -> tensor<{m}x{n}xbf16>
    return %0 : tensor<{m}x{n}xbf16>
  }}
}}
'''


def lower_packed_folded_artifact(
    m: int, payload: PackedFoldedPayload, *, tessera_opt: Path,
) -> dict[str, object]:
    """Lower Graph→Target and issue a hash-bound artifact-only receipt.

    This deliberately does not materialize or launch an HSACO: the packed
    decode kernel is not implemented or proved on gfx1201 yet.
    """
    if not tessera_opt.is_file():
        raise FileNotFoundError(f"packed folded compiler not found: {tessera_opt}")
    graph_ir = author_packed_folded_graph(m, payload)

    def lower(*, target: bool) -> str:
        cmd = [str(tessera_opt), "--tessera-graph-to-schedule", "--tessera-schedule-to-tile"]
        if target:
            cmd.append("--lower-tile-to-rocm=arch=gfx1201")
        result = subprocess.run(cmd, input=graph_ir, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError("packed folded lowering failed: " + result.stderr.strip())
        return result.stdout

    tile_ir = lower(target=False)
    target_ir = lower(target=True)
    tiles = [line.strip() for line in tile_ir.splitlines() if "tile.scaled_matmul_kernel" in line]
    targets = [line.strip() for line in target_ir.splitlines()
               if "tessera_rocm.scaled_wmma_gemm" in line]
    if len(tiles) != 1 or len(targets) != 1:
        raise ValueError("packed folded lowering requires one Tile and Target carrier")
    tile, target = tiles[0], targets[0]
    for operation in (tile, target):
        if _target_string_attr(operation, "physical_contract") != PACKED_FOLDED_PHYSICAL_V1:
            raise ValueError("packed folded physical contract was lost during lowering")
    for key, expected in {
        "abi": "a_bpacked_sa_scaleplane_d_m_n_k",
        "package_abi": PACKED_FOLDED_TARGET_ABI_V1,
        "scale_format": "e8m0_k32_plus_row_reference",
    }.items():
        if _target_string_attr(target, key) != expected:
            raise ValueError(f"packed folded Target requires {key}={expected!r}")
    if _schedule_hash(tile, carrier="packed folded Tile IR") != _schedule_hash(
        target, carrier="packed folded Target IR",
    ):
        raise ValueError("packed folded Tile/Target schedule hashes disagree")
    return {
        **payload.receipt(),
        "physical_contract": PACKED_FOLDED_PHYSICAL_V1,
        "target_abi": PACKED_FOLDED_TARGET_ABI_V1,
        "schedule_hash": _schedule_hash(target, carrier="packed folded Target IR"),
        "graph_ir_sha256": hashlib.sha256(graph_ir.encode()).hexdigest(),
        "tile_ir_sha256": hashlib.sha256(tile_ir.encode()).hexdigest(),
        "target_ir_sha256": hashlib.sha256(target_ir.encode()).hexdigest(),
        "hsaco_sha256": None,
    }


__all__ = [
    "PACKED_FOLDED_SCALE_PLANE_V1", "PACKED_FOLDED_WEIGHT_LAYOUT_V1",
    "PACKED_FOLDED_PHYSICAL_V1", "PACKED_FOLDED_TARGET_ABI_V1",
    "PackedFoldedPayload", "prepare_packed_folded_payload",
    "folded_oracle_from_packed", "author_packed_folded_graph",
    "lower_packed_folded_artifact",
]
