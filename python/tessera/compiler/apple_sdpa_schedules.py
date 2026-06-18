"""Apple SDPA (attention) schedule candidates — seeds mined from MLX (production).

The attention sibling of :mod:`tessera.compiler.apple_gemm_schedules`. Grounded by
reading ``mlx/backend/metal/scaled_dot_product_attention.cpp`` (2026-06-17): MLX
routes SDPA by sequence length, head dim, mask/causal/sinks, GQA factor, and **NAX
availability**, with steel-attention block configs + function-constant specializations.
Encoded here as Tessera **schedule candidates** (seeds + axes + routing), NOT copied
kernels (Decision #23 — MLX is a reference, never a runtime dep).

Grounded routing (matmul.cpp siblings cited inline):
  * **full** attention (prefill) when ``q_len > 8`` and head_dim ∈ {64,80,128};
    block ``bq=64``, ``bk = 32`` (NAX) / ``head_dim<128 ? 32 : 16`` (Metal),
    ``bd=head_dim``, ``wm=4``, ``wn=1``;
  * NAX path when ``is_nax_available()`` and ``head_dim != 80`` (else the Metal path);
  * **vector** (decode) when ``q_len <= 8`` and head_dim ∈ {64,96,128,…};
  * **vector-2pass** split-K for long context: ``blocks = 128`` (N≤8192) / 256 / 1024;
  * specialization function constants ``has_mask`` (300), ``do_causal`` (301),
    ``has_sinks`` (302); ``gqa_factor = q_heads / kv_heads``; align_Q/K.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# MLX-grounded supported head dims (scaled_dot_product_attention.cpp ~line 621-626).
FULL_SUPPORTED_HEAD_DIMS: frozenset[int] = frozenset({64, 80, 128})
VECTOR_SUPPORTED_HEAD_DIMS: frozenset[int] = frozenset({64, 96, 128, 256})
# MLX routes full (prefill) when query_sequence_length > this; else vector (decode).
FULL_MIN_QUERY_LEN = 8
# NAX SDPA path is disabled for head_dim 80 (scaled_dot_product_attention.cpp:177).
NAX_UNSUPPORTED_HEAD_DIM = 80


class AttnPath(Enum):
    FULL_NAX = "full_nax"          # steel attention on the NAX matrix unit
    FULL_METAL = "full_metal"      # steel attention, standard Metal simdgroup path
    VECTOR = "vector"              # decode (short query)
    VECTOR_2PASS = "vector_2pass"  # decode, split over K for long context


@dataclass(frozen=True)
class AttnTile:
    """Full-attention block: ``bq×bk`` over ``bd``(head dim), ``wm×wn`` simdgroups."""

    bq: int
    bk: int
    bd: int
    wm: int
    wn: int


@dataclass(frozen=True)
class AttnSpec:
    """SDPA specialization — MLX's function constants 300/301/302 + the GQA factor."""

    has_mask: bool        # fc 300
    do_causal: bool       # fc 301
    has_sinks: bool       # fc 302
    gqa_factor: int       # q_heads / kv_heads


@dataclass(frozen=True)
class AttnSchedule:
    path: AttnPath
    spec: AttnSpec
    tile: AttnTile | None = None       # set for the full paths
    vector_blocks: int | None = None   # set for VECTOR_2PASS


def full_attn_tile(head_dim: int, *, nax: bool) -> AttnTile:
    """The MLX steel-attention block for full attention. NAX path uses ``bk=32``;
    the Metal path uses ``bk = head_dim < 128 ? 32 : 16``. ``bq=64, wm=4, wn=1``."""
    bk = 32 if nax else (32 if head_dim < 128 else 16)
    return AttnTile(bq=64, bk=bk, bd=head_dim, wm=4, wn=1)


def vector_2pass_blocks(kv_len: int) -> int:
    """Split-K block count for the decode 2-pass
    (scaled_dot_product_attention.cpp ~line 449): 128 (N≤8192) / 256 / 1024.

    Standalone block-count lookup over the **full** ``kv_len`` range. Note the
    ``≤8192 → 128`` row is only reached when 2-pass is *forced* (a direct call):
    :func:`select_attn_schedule` auto-selects 2-pass only for ``kv_len > 8192`` (so
    via the router the count is always ≥256). The two thresholds are independent by
    design — this is a pure lookup, the router owns the routing policy."""
    if kv_len <= 8192:
        return 128
    if kv_len <= 32768:
        return 256
    return 1024


def select_attn_schedule(
    q_len: int,
    kv_len: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    *,
    nax_available: bool = False,
    has_mask: bool = False,
    do_causal: bool = False,
    has_sinks: bool = False,
    value_head_dim: int | None = None,
) -> AttnSchedule:
    """MLX's SDPA routing decision → a Tessera attention schedule candidate.

    ``q_len <= 8`` → decode (vector, 2-pass for long ``kv_len``); else prefill
    (full, NAX when ``nax_available`` and ``head_dim != 80``, else Metal). ``gqa_factor
    = q_heads // kv_heads``."""
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(f"q_heads {q_heads} not a multiple of kv_heads {kv_heads} (GQA)")
    spec = AttnSpec(has_mask=has_mask, do_causal=do_causal, has_sinks=has_sinks,
                    gqa_factor=q_heads // kv_heads)
    vhd = value_head_dim if value_head_dim is not None else head_dim

    # Decode (short query) → vector; long context → 2-pass.
    if q_len <= FULL_MIN_QUERY_LEN:
        if head_dim not in VECTOR_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"vector (decode) head_dim {head_dim} unsupported — MLX vector SDPA "
                f"supports {sorted(VECTOR_SUPPORTED_HEAD_DIMS)}")
        if kv_len > 8192:
            return AttnSchedule(AttnPath.VECTOR_2PASS, spec,
                                vector_blocks=vector_2pass_blocks(kv_len))
        return AttnSchedule(AttnPath.VECTOR, spec)

    # Prefill → full attention (head_dim must match q==v and be supported).
    if head_dim not in FULL_SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"full (prefill) head_dim {head_dim} unsupported — MLX full SDPA "
            f"supports {sorted(FULL_SUPPORTED_HEAD_DIMS)}")
    use_nax = nax_available and head_dim != NAX_UNSUPPORTED_HEAD_DIM and head_dim == vhd
    path = AttnPath.FULL_NAX if use_nax else AttnPath.FULL_METAL
    return AttnSchedule(path, spec, tile=full_attn_tile(head_dim, nax=use_nax))


__all__ = [
    "AttnPath",
    "AttnTile",
    "AttnSpec",
    "AttnSchedule",
    "FULL_SUPPORTED_HEAD_DIMS",
    "VECTOR_SUPPORTED_HEAD_DIMS",
    "FULL_MIN_QUERY_LEN",
    "NAX_UNSUPPORTED_HEAD_DIM",
    "full_attn_tile",
    "vector_2pass_blocks",
    "select_attn_schedule",
]
