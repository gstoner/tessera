"""
TPUTargetProfile — compile-time description of a TPU device target.

Captures the hardware constraints that the compiler needs to validate
matmul shapes, select tile sizes, and annotate Shardy mesh regions.

Key facts about TPU MXU:
  - Systolic array processes 128×128 tiles natively
  - All matmul M/N/K dims must be multiples of 128
  - bf16 inputs, f32 accumulators (bfloat16 → float32 accumulation)
  - int8 quantized dot is supported (int8 × int8 → int32)
  - HBM bandwidth: ~1.2 TB/s per chip (v4/v5)

When @jit(target=TPUTargetProfile(...)) is used, the JIT automatically
injects Divisible("M", 128) + Divisible("N", 128) + Divisible("K", 128)
constraint checks before IR emission.

Reference: docs/spec/COMPILER_REFERENCE.md §TPU
           src/compiler/codegen/Tessera_TPU_Backend/
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# TPU generations
# ─────────────────────────────────────────────────────────────────────────────

class TPUGeneration:
    """Known TPU hardware generations."""
    V3   = "v3"    # 32 GB HBM, 123 TFLOPS bf16
    V4   = "v4"    # 32 GB HBM, 275 TFLOPS bf16
    V5E  = "v5e"   # 16 GB HBM, 197 TFLOPS bf16 (edge/inference)
    V5P  = "v5p"   # 95 GB HBM, 459 TFLOPS bf16 (pod-scale training)


# MXU tile size by generation (rows × cols of the systolic array)
_MXU_TILE: Dict[str, int] = {
    TPUGeneration.V3:  128,
    TPUGeneration.V4:  128,
    TPUGeneration.V5E: 128,
    TPUGeneration.V5P: 128,
}

# HBM capacity in GB
_HBM_GB: Dict[str, int] = {
    TPUGeneration.V3:  32,
    TPUGeneration.V4:  32,
    TPUGeneration.V5E: 16,
    TPUGeneration.V5P: 95,
}

# Peak bf16 TFLOPS per chip
_PEAK_TFLOPS: Dict[str, float] = {
    TPUGeneration.V3:  123.0,
    TPUGeneration.V4:  275.0,
    TPUGeneration.V5E: 197.0,
    TPUGeneration.V5P: 459.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# TPUTargetProfile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TPUTargetProfile:
    """
    Compile-time description of a TPU target for Tessera's JIT.

    Attributes:
        generation  : TPU generation string (use TPUGeneration.* constants)
        mesh_axes   : logical mesh axis sizes, e.g. {"data": 4, "model": 2}
        num_chips   : total chips in the pod slice (must equal product of mesh_axes)
        dtype       : default compute dtype ("bf16", "int8")
        mxu_tile    : MXU systolic array size (auto-set from generation)

    Example:
        tpu = TPUTargetProfile(generation=TPUGeneration.V4,
                               mesh_axes={"data": 4, "model": 2})
        tpu.validate_matmul_dims(M=512, N=1024, K=4096)   # passes
        tpu.validate_matmul_dims(M=500, N=1024, K=4096)   # raises ValueError
    """
    generation: str = TPUGeneration.V4
    mesh_axes: Dict[str, int] = field(default_factory=lambda: {"data": 1})
    dtype: str = "bf16"
    # mxu_tile is derived from generation; override only for testing
    _mxu_tile_override: Optional[int] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        valid_gens = {TPUGeneration.V3, TPUGeneration.V4, TPUGeneration.V5E, TPUGeneration.V5P}
        if self.generation not in valid_gens:
            raise ValueError(
                f"Unknown TPU generation {self.generation!r}. "
                f"Valid: {sorted(valid_gens)}"
            )
        if self.dtype not in ("bf16", "int8", "fp32"):
            raise ValueError(
                f"Unsupported TPU dtype {self.dtype!r}. Use 'bf16', 'int8', or 'fp32'."
            )
        for axis, size in self.mesh_axes.items():
            if not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"mesh_axes[{axis!r}] must be a positive int, got {size!r}"
                )

    @property
    def mxu_tile(self) -> int:
        """Size of the MXU systolic array (always 128 for all current TPU generations)."""
        if self._mxu_tile_override is not None:
            return self._mxu_tile_override
        return _MXU_TILE.get(self.generation, 128)

    @property
    def num_chips(self) -> int:
        """Total number of chips implied by the mesh axis sizes."""
        result = 1
        for s in self.mesh_axes.values():
            result *= s
        return result

    @property
    def hbm_gb(self) -> int:
        return _HBM_GB.get(self.generation, 32)

    @property
    def peak_tflops(self) -> float:
        return _PEAK_TFLOPS.get(self.generation, 275.0)

    def validate_matmul_dims(self, M: int, N: int, K: int) -> None:
        """
        Raise ValueError if M, N, or K are not multiples of the MXU tile size.

        This is called automatically by @jit(target=tpu_profile) before IR
        emission. It corresponds to injecting:
            tessera.require(tessera.constraint.Divisible("M", 128))
            tessera.require(tessera.constraint.Divisible("N", 128))
            tessera.require(tessera.constraint.Divisible("K", 128))

        Args:
            M, N, K: matmul dimensions to validate

        Raises:
            ValueError with a precise message identifying the failing dimension.
        """
        t = self.mxu_tile
        for name, val in (("M", M), ("N", N), ("K", K)):
            if val % t != 0:
                raise ValueError(
                    f"TPU {self.generation} MXU requires {name}={val} to be "
                    f"divisible by {t} (MXU tile size). "
                    f"Nearest valid value: {((val + t - 1) // t) * t}"
                )

    def to_mlir_attrs(self) -> str:
        """Serialize as a MLIR attribute string for tessera-tpu-opt."""
        axes = ", ".join(f'"{k}" = {v}' for k, v in self.mesh_axes.items())
        return (
            f'{{tessera.tpu_target = {{generation = "{self.generation}", '
            f'mxu_tile = {self.mxu_tile}, dtype = "{self.dtype}", '
            f'mesh = {{{axes}}}}}}}'
        )

    def auto_constraints(self) -> Dict[str, int]:
        """
        Return the set of Divisible constraints injected automatically by @jit.
        Useful for testing: assert tpu.auto_constraints() == {"M": 128, "N": 128, "K": 128}
        """
        return {"M": self.mxu_tile, "N": self.mxu_tile, "K": self.mxu_tile}

    def __repr__(self) -> str:
        mesh_str = ", ".join(f"{k}={v}" for k, v in self.mesh_axes.items())
        return (
            f"TPUTargetProfile(generation={self.generation!r}, "
            f"mesh={{{mesh_str}}}, dtype={self.dtype!r}, "
            f"mxu_tile={self.mxu_tile})"
        )
