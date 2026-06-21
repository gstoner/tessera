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
from typing import Any, Iterable, List, Optional


class TesseraAttnConfigError(Exception):
    pass


def _resolve_lds_budget(budget: "int | Any") -> int:
    """Coerce an LDS budget argument to bytes.

    Accepts a plain ``int`` (bytes) or any object exposing ``lds_capacity_bytes``
    (e.g. an ``ROCmTargetProfile``) — duck-typed so ``attn_lower`` stays
    import-light and free of a backend dependency.  Raises a stable diagnostic
    for anything else.
    """
    if isinstance(budget, bool):  # bool is an int subclass — reject explicitly
        raise TesseraAttnConfigError(
            f"LDS budget must be an int (bytes) or have .lds_capacity_bytes; "
            f"got bool {budget!r}")
    if isinstance(budget, int):
        if budget <= 0:
            raise TesseraAttnConfigError(
                f"LDS budget must be positive, got {budget}")
        return budget
    cap = getattr(budget, "lds_capacity_bytes", None)
    if isinstance(cap, int) and cap > 0:
        return cap
    raise TesseraAttnConfigError(
        "LDS budget must be a positive int (bytes) or an object exposing a "
        f"positive .lds_capacity_bytes; got {budget!r}")


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

    def lds_bytes(
        self,
        *,
        head_dim: int,
        dtype_bytes: int = 2,
        stage_v: bool = True,
        stage_q: bool = False,
    ) -> int:
        """Estimate the LDS (shared-memory) footprint of this tiling, in bytes.

        FlashAttention stages the current KV block in LDS: a ``tile_kv ×
        head_dim`` K tile (and, by default, an equal-size V tile), each
        replicated across the ``pipeline_stages`` software-pipeline buffers so
        the next block's load overlaps the current block's MFMA.  The output /
        score accumulators stay in registers (the moonmath writeup keeps them in
        VGPRs), so they are not counted here.

        Two flags model the writeup's advanced layout:

        * ``stage_v=False`` — V has been moved to L1 (the writeup's deliberate
          "keep V hot in L1" choice), so only K occupies LDS.
        * ``stage_q=True`` — additionally count a ``tile_q × head_dim`` Q tile in
          LDS (the "3Q tiling" variant that streams a third Q tile through a
          ping-pong LDS buffer once V is out of the way).

        ``dtype_bytes`` is the storage element width (2 for bf16/fp16, 1 for
        fp8, 4 for fp32).
        """
        if head_dim <= 0:
            raise TesseraAttnConfigError(
                f"head_dim must be positive, got {head_dim}")
        if dtype_bytes <= 0:
            raise TesseraAttnConfigError(
                f"dtype_bytes must be positive, got {dtype_bytes}")
        kv_tile = self.tile_kv * head_dim * dtype_bytes
        total = self.pipeline_stages * kv_tile  # K, N-buffered
        if stage_v:
            total += self.pipeline_stages * kv_tile  # V also in LDS
        if stage_q:
            total += self.tile_q * head_dim * dtype_bytes
        return total

    def fits_lds(
        self,
        budget: "int | Any",
        *,
        head_dim: int,
        dtype_bytes: int = 2,
        stage_v: bool = True,
        stage_q: bool = False,
    ) -> bool:
        """True iff this tiling's :meth:`lds_bytes` fits within ``budget``.

        ``budget`` is bytes (int) or an object with ``lds_capacity_bytes`` (an
        ``ROCmTargetProfile``).  This is the feasibility predicate the autotuner
        uses to prune configs *before* sweeping — turning a blind tile sweep into
        a budget-constrained one (the LDS budget doubles on CDNA 4, so more
        configs become feasible there)."""
        cap = _resolve_lds_budget(budget)
        return self.lds_bytes(
            head_dim=head_dim, dtype_bytes=dtype_bytes,
            stage_v=stage_v, stage_q=stage_q,
        ) <= cap

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


#: Default power-of-two tile sizes the autotuner sweeps for FlashAttention.
_DEFAULT_TILE_Q = (32, 64, 128, 256)
_DEFAULT_TILE_KV = (32, 64, 128, 256)
_DEFAULT_STAGES = (2, 3)


def candidate_configs(
    *,
    tile_q: Iterable[int] = _DEFAULT_TILE_Q,
    tile_kv: Iterable[int] = _DEFAULT_TILE_KV,
    pipeline_stages: Iterable[int] = _DEFAULT_STAGES,
    causal: bool = False,
) -> List[FlashAttnLoweringConfig]:
    """Enumerate the FlashAttention tiling search space (the raw, unpruned grid).

    The cross product of ``tile_q × tile_kv × pipeline_stages`` — the configs the
    autotuner would sweep before any feasibility filter.  Ordering is
    deterministic (tile_q, then tile_kv, then stages)."""
    out: List[FlashAttnLoweringConfig] = []
    for q in tile_q:
        for kv in tile_kv:
            for stages in pipeline_stages:
                out.append(
                    FlashAttnLoweringConfig(
                        tile_q=q,
                        tile_kv=kv,
                        pipeline_stages=stages,
                        causal=causal,
                    )
                )
    return out


def feasible_configs(
    budget: "int | Any",
    *,
    head_dim: int,
    dtype_bytes: int = 2,
    stage_v: bool = True,
    stage_q: bool = False,
    candidates: Optional[Iterable[FlashAttnLoweringConfig]] = None,
) -> List[FlashAttnLoweringConfig]:
    """Prune the tiling search space to configs that fit the LDS budget.

    ``budget`` is bytes (int) or an object exposing ``lds_capacity_bytes`` (an
    ``ROCmTargetProfile``).  This is the LDS-budget-aware step the moonmath
    writeup motivates: a blind tile sweep wastes trials on tilings that would
    spill / fail to launch, so the autotuner should only consider tilings whose
    K/V staging fits in shared memory.  CDNA 4's doubled LDS budget admits
    strictly more configs — feed it the bigger ``lds_capacity_bytes`` and this
    returns the larger feasible set automatically.

    Returns the feasible subset in candidate order.  Raises a stable diagnostic
    only on a malformed budget; an empty result (nothing fits) is returned as an
    empty list, not an error.
    """
    cap = _resolve_lds_budget(budget)
    pool = candidate_configs() if candidates is None else candidates
    return [
        c for c in pool
        if c.lds_bytes(
            head_dim=head_dim, dtype_bytes=dtype_bytes,
            stage_v=stage_v, stage_q=stage_q,
        ) <= cap
    ]
