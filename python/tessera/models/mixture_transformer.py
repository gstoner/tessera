"""Mixture-of-Transformers (MoT) — Cosmos-3-style dual-tower decoder layer.

Reference (numpy / Apple-GPU ``metal_runtime``) transcription of the
Mixture-of-Transformers architecture in NVIDIA **Cosmos 3: Omnimodal World
Models for Physical AI** (2026-06, §2.3 "Mixture-of-Transformers Architecture",
Fig. 5 & 14). Distinct from a classic MoE: routing is **deterministic by token
role**, not learned per-token. Each decoder layer carries *two complete
parameter sets* — a Reasoner tower (autoregressive) and a Generator tower
(diffusion) — that are independent everywhere except a **shared self-attention
operator** (Cosmos §2.3.1: "the two pathways ... meet only at a shared
self-attention operator").

Per-tower parameters (Cosmos §2.3.1):
  * LayerNorms (here RMSNorm)
  * attention projection matrices  Q / K / V / O
  * feed-forward network (MLP)

The join rule (Cosmos §2.3.1, Fig. 5):
  * ``Q_AR`` attends **causally** over ``K_AR, V_AR`` only — reasoning stays
    autoregressively self-contained.
  * ``Q_DM`` attends **bidirectionally** over the concatenation ``[K_AR; K_DM]``,
    ``[V_AR; V_DM]`` — generation is conditioned on the reasoning context.

Two numerically-identical formulations are provided (the metamorphic oracle in
``tests/unit/test_varlen_sdpa.py`` proves they agree):
  * :func:`dual_stream_attention_dense` — one masked ``ops.flash_attn`` over the
    packed ``[AR, DM]`` sequence with :func:`cosmos_join_bias`. This is the
    FlexAttention-equivalent reference path.
  * :func:`dual_stream_attention_varlen` — the Cosmos "two-way flat attention":
    a causal varlen launch for the Reasoner and a bidirectional varlen launch
    for the Generator (Q packed separately from the interleaved ``[R_i, G_i]``
    key/value stream), via :func:`tessera.nn.varlen.varlen_sdpa`.

Scope honesty: this is the *correctness / contract* layer. It runs today on
numpy and on Apple GPU (the rank-3 ``flash_attn`` lane reports
``execution_mode="metal_runtime"``). The FA-3 / NATTEN performance kernels that
carry Cosmos's measured 22% are NVIDIA-frontier and remain on Tessera's
hardware-gated ``backend_kernel`` wall (Phase G/H).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import ops
from ..nn import functional as F
from ..nn.varlen import (
    cu_seqlens_from_lengths,
    varlen_sdpa,
)

__all__ = [
    "REASONER",
    "GENERATOR",
    "MixtureTransformerConfig",
    "MixtureTransformerDimError",
    "TowerWeights",
    "cosmos_join_bias",
    "split_roles",
    "dual_stream_attention_dense",
    "dual_stream_attention_varlen",
    "synthetic_tower_weights",
    "verify_config",
]

# Role ids for the deterministic role-routing (Cosmos: AR front, DM back).
REASONER = 0
GENERATOR = 1

_NEG_INF = -1e30


class MixtureTransformerDimError(ValueError):
    """Raised when a MoT config or call violates the dual-tower dim contract."""


@dataclass(frozen=True)
class MixtureTransformerConfig:
    """Dual-tower MoT layer config.

    The two towers share ``hidden_size``, ``num_heads`` and ``head_dim`` (they
    must — they meet at one attention operator over a common sequence) but carry
    independent projection / norm / FFN weights and may size their FFNs
    differently.
    """

    hidden_size: int = 256
    num_heads: int = 4
    reasoner_intermediate: int = 512
    generator_intermediate: int = 512
    rms_eps: float = 1e-6
    dtype: str = "fp32"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise MixtureTransformerDimError(
                f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}"
            )
        return self.hidden_size // self.num_heads


@dataclass
class TowerWeights:
    """One tower's parameters: RMSNorm gain + Q/K/V/O projections + SwiGLU FFN.

    Projection weights are ``(hidden, hidden)``; FFN is gate/up ``(hidden, inter)``
    and down ``(inter, hidden)`` — a standard SwiGLU MLP.
    """

    norm_w: np.ndarray          # (hidden,)
    wq: np.ndarray              # (hidden, hidden)
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    ffn_norm_w: np.ndarray      # (hidden,)
    w_gate: np.ndarray          # (hidden, inter)
    w_up: np.ndarray            # (hidden, inter)
    w_down: np.ndarray          # (inter, hidden)


def verify_config(cfg: MixtureTransformerConfig) -> None:
    """Decoration-time dim contract check (raises MixtureTransformerDimError)."""
    if cfg.hidden_size <= 0 or cfg.num_heads <= 0:
        raise MixtureTransformerDimError("hidden_size and num_heads must be positive")
    _ = cfg.head_dim  # triggers divisibility check
    if cfg.reasoner_intermediate <= 0 or cfg.generator_intermediate <= 0:
        raise MixtureTransformerDimError("FFN intermediate sizes must be positive")


# ---------------------------------------------------------------------------
# Role layout + the join mask
# ---------------------------------------------------------------------------

def split_roles(role_ids) -> tuple[np.ndarray, np.ndarray]:
    """Return the (reasoner_positions, generator_positions) index arrays.

    Cosmos packs the AR subsequence at the front and the DM subsequence at the
    back of one sequence; this does not *require* contiguity, but the role id per
    position is what drives tower routing and the join mask.
    """
    rid = np.asarray(role_ids)
    if rid.ndim != 1:
        raise MixtureTransformerDimError("role_ids must be a 1-D per-position vector")
    uniq = set(np.unique(rid).tolist())
    if not uniq <= {REASONER, GENERATOR}:
        raise MixtureTransformerDimError(
            f"role_ids must be in {{REASONER={REASONER}, GENERATOR={GENERATOR}}}; got {sorted(uniq)}"
        )
    ar = np.where(rid == REASONER)[0]
    dm = np.where(rid == GENERATOR)[0]
    return ar, dm


def cosmos_join_bias(role_ids) -> np.ndarray:
    """Additive ``(S, S)`` mask implementing the Cosmos-3 joint-attention rule.

    For query position ``q`` and key position ``k`` (Cosmos §2.3.1, Fig. 5):

      * ``q`` is a **Reasoner** token → allowed iff ``k`` is also Reasoner and
        ``k <= q`` (causal self-attention over AR only).
      * ``q`` is a **Generator** token → allowed for *any* ``k`` (bidirectional
        over the union ``[AR; DM]``).

    Allowed entries are ``0``, forbidden are ``-inf``; this is fed to a single
    dense ``ops.flash_attn(attn_bias=...)`` over the packed sequence. The crucial
    asymmetry — AR never attends to DM — is what keeps reasoning self-contained.
    """
    rid = np.asarray(role_ids)
    if rid.ndim != 1:
        raise MixtureTransformerDimError("role_ids must be a 1-D per-position vector")
    S = rid.shape[0]
    q_is_ar = (rid == REASONER)[:, None]          # (S, 1)
    k_is_ar = (rid == REASONER)[None, :]          # (1, S)
    qpos = np.arange(S)[:, None]
    kpos = np.arange(S)[None, :]

    # Reasoner queries: causal, AR keys only.
    ar_allow = q_is_ar & k_is_ar & (kpos <= qpos)
    # Generator queries: everything (bidirectional over [AR; DM]).
    dm_allow = (~q_is_ar)
    allow = ar_allow | dm_allow
    return np.where(allow, 0.0, _NEG_INF).astype(np.float32)


# ---------------------------------------------------------------------------
# Tower sublayers
# ---------------------------------------------------------------------------

def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * np.asarray(w, dtype=np.float32)


def _swiglu_ffn(x: np.ndarray, tw: TowerWeights, eps: float) -> np.ndarray:
    h = _rmsnorm(x, tw.ffn_norm_w, eps)
    gate = np.asarray(F.linear_general(h, tw.w_gate))
    up = np.asarray(F.linear_general(h, tw.w_up))
    silu = gate / (1.0 + np.exp(-gate))
    return np.asarray(F.linear_general(silu * up, tw.w_down))


def _project_heads(x: np.ndarray, W: np.ndarray, H: int, Dh: int) -> np.ndarray:
    """(B, S, hidden) @ (hidden, hidden) -> (B, H, S, Dh)."""
    y = np.asarray(F.linear_general(x, W))
    B, S, _ = y.shape
    return y.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)


def _qkv_per_role(
    x: np.ndarray,
    role_ids: np.ndarray,
    reasoner: TowerWeights,
    generator: TowerWeights,
    cfg: MixtureTransformerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tower normed Q/K/V over the packed sequence.

    Each position is normed + projected by its *own* tower's weights, then the
    rows are scattered back into one packed ``(B, H, S, Dh)`` tensor per Q/K/V so
    the shared attention operator sees a single sequence. This is the dual-tower
    parameter split (Cosmos §2.3.1) made explicit.
    """
    H, Dh, eps = cfg.num_heads, cfg.head_dim, cfg.rms_eps
    B, S, _ = x.shape
    ar_idx, dm_idx = split_roles(role_ids)

    Q = np.zeros((B, H, S, Dh), dtype=np.float32)
    K = np.zeros((B, H, S, Dh), dtype=np.float32)
    Vt = np.zeros((B, H, S, Dh), dtype=np.float32)

    for idx, tw in ((ar_idx, reasoner), (dm_idx, generator)):
        if idx.size == 0:
            continue
        xs = _rmsnorm(x[:, idx, :], tw.norm_w, eps)          # (B, n, hidden)
        Q[:, :, idx, :] = _project_heads(xs, tw.wq, H, Dh)
        K[:, :, idx, :] = _project_heads(xs, tw.wk, H, Dh)
        Vt[:, :, idx, :] = _project_heads(xs, tw.wv, H, Dh)
    return Q, K, Vt


def _output_proj_per_role(
    attn: np.ndarray,
    role_ids: np.ndarray,
    reasoner: TowerWeights,
    generator: TowerWeights,
) -> np.ndarray:
    """Per-tower output projection ``O`` over packed attention output (B, S, hidden)."""
    ar_idx, dm_idx = split_roles(role_ids)
    B, S, hidden = attn.shape
    out = np.zeros((B, S, hidden), dtype=np.float32)
    for idx, tw in ((ar_idx, reasoner), (dm_idx, generator)):
        if idx.size == 0:
            continue
        out[:, idx, :] = np.asarray(F.linear_general(attn[:, idx, :], tw.wo))
    return out


# ---------------------------------------------------------------------------
# Dual-stream attention — dense form (the reference / oracle)
# ---------------------------------------------------------------------------

def dual_stream_attention_dense(
    x,
    role_ids,
    reasoner: TowerWeights,
    generator: TowerWeights,
    cfg: MixtureTransformerConfig,
    *,
    scale: Optional[float] = None,
    attention_fn=None,
):
    """Cosmos-3 dual-stream attention via a single masked ``ops.flash_attn``.

    Per-tower Q/K/V/O projections + the shared attention operator, with the join
    rule expressed as :func:`cosmos_join_bias`. Heads fold into the batch axis so
    the attention core rides the rank-3 ``flash_attn`` lane (Apple GPU
    ``metal_runtime``). Returns ``(B, S, hidden)``.
    """
    verify_config(cfg)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise MixtureTransformerDimError("x must be rank-3 (B, S, hidden)")
    B, S, hidden = x.shape
    if hidden != cfg.hidden_size:
        raise MixtureTransformerDimError(f"x hidden {hidden} != cfg.hidden_size {cfg.hidden_size}")
    rid = np.asarray(role_ids)
    if rid.shape[0] != S:
        raise MixtureTransformerDimError(f"role_ids length {rid.shape[0]} != seq len {S}")
    H, Dh = cfg.num_heads, cfg.head_dim
    if scale is None:
        scale = float(Dh) ** -0.5

    Q, K, Vt = _qkv_per_role(x, rid, reasoner, generator, cfg)
    bias2 = cosmos_join_bias(rid)                                  # (S, S)
    bias = np.broadcast_to(bias2, (B * H, S, S)).astype(np.float32)

    core = attention_fn if attention_fn is not None else ops.flash_attn
    q3 = Q.reshape(B * H, S, Dh)
    k3 = K.reshape(B * H, S, Dh)
    v3 = Vt.reshape(B * H, S, Dh)
    o3 = core(q3, k3, v3, scale=scale, causal=False, attn_bias=bias)
    attn = np.asarray(o3).reshape(B, H, S, Dh).transpose(0, 2, 1, 3).reshape(B, S, hidden)
    return _output_proj_per_role(attn, rid, reasoner, generator)


# ---------------------------------------------------------------------------
# Dual-stream attention — varlen form (Cosmos "two-way flat attention")
# ---------------------------------------------------------------------------

def dual_stream_attention_varlen(
    x,
    role_ids,
    reasoner: TowerWeights,
    generator: TowerWeights,
    cfg: MixtureTransformerConfig,
    *,
    scale: Optional[float] = None,
    attention_fn=None,
):
    """Cosmos-3 "two-way flat attention" (§5.2.2, Fig. 14) over a single sample.

    Two varlen launches:
      (a) Reasoner pathway — causal varlen SDPA over the AR tokens only.
      (b) Generator pathway — bidirectional varlen SDPA where the Generator
          queries attend over the interleaved ``[AR; DM]`` key/value block (Q
          packed separately from KV, ``cu_seqlens_q != cu_seqlens_k``).

    This single-sample form keeps the role layout explicit and is what the
    metamorphic oracle checks against :func:`dual_stream_attention_dense`. Batch
    ``B == 1`` (heads fold into the varlen "head" axis); multi-sample packing is
    the natural extension (concat the per-sample blocks and their cu_seqlens).
    Returns ``(1, S, hidden)``.
    """
    verify_config(cfg)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 1:
        raise MixtureTransformerDimError(
            "dual_stream_attention_varlen takes a single sample, x shape (1, S, hidden)"
        )
    B, S, hidden = x.shape
    rid = np.asarray(role_ids)
    if rid.shape[0] != S:
        raise MixtureTransformerDimError(f"role_ids length {rid.shape[0]} != seq len {S}")
    H, Dh = cfg.num_heads, cfg.head_dim
    if scale is None:
        scale = float(Dh) ** -0.5

    Q, K, Vt = _qkv_per_role(x, rid, reasoner, generator, cfg)     # (1, H, S, Dh)
    Q, K, Vt = Q[0], K[0], Vt[0]                                   # (H, S, Dh)
    ar_idx, dm_idx = split_roles(rid)
    n_ar, n_dm = int(ar_idx.size), int(dm_idx.size)

    attn = np.zeros((H, S, Dh), dtype=np.float32)

    # (a) Reasoner pathway — causal varlen over AR keys only.
    if n_ar > 0:
        cu = cu_seqlens_from_lengths([n_ar])
        o_ar = varlen_sdpa(
            Q[:, ar_idx, :], K[:, ar_idx, :], Vt[:, ar_idx, :],
            cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True,
            scale=scale, attention_fn=attention_fn,
        )
        attn[:, ar_idx, :] = np.asarray(o_ar)

    # (b) Generator pathway — bidirectional varlen over [AR; DM] keys/values.
    if n_dm > 0:
        kv_idx = np.concatenate([ar_idx, dm_idx])                 # interleaved [R_i, G_i] block
        cu_q = cu_seqlens_from_lengths([n_dm])
        cu_k = cu_seqlens_from_lengths([n_ar + n_dm])
        o_dm = varlen_sdpa(
            Q[:, dm_idx, :], K[:, kv_idx, :], Vt[:, kv_idx, :],
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, causal=False,
            scale=scale, attention_fn=attention_fn,
        )
        attn[:, dm_idx, :] = np.asarray(o_dm)

    attn_bshd = attn.transpose(1, 0, 2).reshape(1, S, hidden)
    return _output_proj_per_role(attn_bshd, rid, reasoner, generator)


# ---------------------------------------------------------------------------
# Synthetic weights for tests / smoke
# ---------------------------------------------------------------------------

def synthetic_tower_weights(cfg: MixtureTransformerConfig, inter: int, seed: int) -> TowerWeights:
    """Deterministic small random weights for one tower (test/smoke fixture)."""
    rng = np.random.default_rng(seed)
    h = cfg.hidden_size
    s = 1.0 / np.sqrt(h)

    def proj():
        return (rng.standard_normal((h, h)) * s).astype(np.float32)

    return TowerWeights(
        norm_w=np.ones(h, dtype=np.float32),
        wq=proj(), wk=proj(), wv=proj(), wo=proj(),
        ffn_norm_w=np.ones(h, dtype=np.float32),
        w_gate=(rng.standard_normal((h, inter)) * s).astype(np.float32),
        w_up=(rng.standard_normal((h, inter)) * s).astype(np.float32),
        w_down=(rng.standard_normal((inter, h)) * (1.0 / np.sqrt(inter))).astype(np.float32),
    )
