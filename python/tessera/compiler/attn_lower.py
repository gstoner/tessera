"""
tessera.compiler.attn_lower — FlashAttention lowering configuration.

Phase 3: holds tile sizes and pipeline decisions for FA-2 / FA-4 lowering.
Consumed by TileIRLoweringPass (C++) via MLIR attributes, and by @jit when
the function body contains tessera.ops.flash_attn.

Usage:
    from tessera.compiler.attn_lower import FlashAttnLoweringConfig

    cfg = FlashAttnLoweringConfig(tile_q=64, tile_kv=64, pipeline_stages=2,
                                   causal=True, dropout_p=0.0)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


class TesseraAttnConfigError(Exception):
    pass


@dataclass
class FlashAttnLoweringConfig:
    """
    Controls how tessera.flash_attn is lowered to FA-4 Tile IR.

    Attributes:
        tile_q         : Q tile size (rows processed per outer loop step).
        tile_kv        : KV tile size (cols processed per inner loop step).
        pipeline_stages: software double-buffer stages (≥1).
        causal         : emit CausalMaskOp in the inner loop.
        dropout_p      : dropout probability; 0.0 → no DropoutMaskOp emitted.
        seed           : RNG seed for dropout (required if dropout_p > 0).

    Defaults are tuned for SM_90 (Hopper) with BF16 operands.
    Phase 5 autotuner sweeps tile_q and tile_kv; store them as attributes
    tessera.tile_q / tessera.tile_kv on the emitted tessera.flash_attn op so
    the autotuner can modify them without re-emitting Graph IR.
    """

    tile_q: int = 64
    tile_kv: int = 64
    pipeline_stages: int = 2
    causal: bool = False
    dropout_p: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tile_q <= 0 or (self.tile_q & (self.tile_q - 1)) != 0:
            raise TesseraAttnConfigError(
                f"tile_q must be a positive power of 2, got {self.tile_q}"
            )
        if self.tile_kv <= 0 or (self.tile_kv & (self.tile_kv - 1)) != 0:
            raise TesseraAttnConfigError(
                f"tile_kv must be a positive power of 2, got {self.tile_kv}"
            )
        if not (0.0 <= self.dropout_p < 1.0):
            raise TesseraAttnConfigError(
                f"dropout_p must be in [0, 1), got {self.dropout_p}"
            )
        if self.dropout_p > 0.0 and self.seed is None:
            raise TesseraAttnConfigError(
                "dropout_p > 0 requires a seed for reproducibility. "
                "Pass seed= to FlashAttnLoweringConfig or use @jit(seed=...)."
            )
        if self.pipeline_stages < 1:
            raise TesseraAttnConfigError(
                f"pipeline_stages must be >= 1, got {self.pipeline_stages}"
            )

    @property
    def has_dropout(self) -> bool:
        return self.dropout_p > 0.0

    def to_mlir_attrs(self) -> str:
        """Return inline attr dict for the tessera.flash_attn op."""
        parts = [
            f"tessera.tile_q = {self.tile_q} : i32",
            f"tessera.tile_kv = {self.tile_kv} : i32",
            f"tessera.pipeline_stages = {self.pipeline_stages} : i32",
            f'causal = {"true" if self.causal else "false"}',
        ]
        if self.has_dropout:
            parts.append(f"dropout_p = {self.dropout_p:.6f} : f32")
        return "{" + ", ".join(parts) + "}"

    def __repr__(self) -> str:
        return (
            f"FlashAttnLoweringConfig(tile_q={self.tile_q}, "
            f"tile_kv={self.tile_kv}, stages={self.pipeline_stages}, "
            f"causal={self.causal}, dropout_p={self.dropout_p})"
        )


# Default config for SM_90 BF16 FlashAttention (used by @jit when no explicit
# config is provided and target.isa >= ISA.SM_90).
SM90_DEFAULT = FlashAttnLoweringConfig(
    tile_q=64,
    tile_kv=64,
    pipeline_stages=2,
    causal=False,
)
