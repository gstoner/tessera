"""``tessera.stdlib.dspark`` -- DSpark draft-block reference contracts.

DSpark is the draft side of the DeepSeek/speculative-decode roadmap: choose a
small set of anchors, run a block draft around each anchor, score confidence,
then hand the proposal to the existing target-verify/spec-accept path.  This
module is deliberately a NumPy reference surface.  Fused CUDA/ROCm draft-block
kernels should match these shapes and values before they claim native proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _arr(x: Any, *, dtype=None) -> np.ndarray:
    a = np.asarray(x._data if hasattr(x, "_data") else x)
    return a.astype(dtype, copy=False) if dtype is not None else a


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class DSparkConfig:
    """Static draft-block shape contract.

    ``num_anchors`` and ``block_size`` are static in the first fused-kernel
    target.  The reference keeps them explicit so tests can pin the exact
    ``[B, A, D, V]`` logits surface that DS2 will lower for ROCm.
    """

    num_anchors: int
    block_size: int
    vocab_size: int
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


@dataclass(frozen=True)
class DSparkWeights:
    """Minimal vanilla DSpark draft head weights.

    The draft state starts from the target hidden state at each anchor, mixes in
    the previous token embedding, then emits draft logits and confidence logits.
    ``markov`` is optional vanilla Markov bias ``[V, H]`` for the previously
    emitted draft token.  Gated/RNN variants can extend this contract without
    changing the outer ``[B, A, D, V]`` surface.
    """

    token_embedding: np.ndarray  # (V, H)
    hidden_proj: np.ndarray      # (H, H)
    token_proj: np.ndarray       # (H, H)
    out_proj: np.ndarray         # (H, V)
    confidence_proj: np.ndarray  # (H,)
    markov: np.ndarray | None = None  # (V, H)

    def __post_init__(self) -> None:
        emb = _arr(self.token_embedding, dtype=np.float32)
        hp = _arr(self.hidden_proj, dtype=np.float32)
        tp = _arr(self.token_proj, dtype=np.float32)
        op = _arr(self.out_proj, dtype=np.float32)
        cp = _arr(self.confidence_proj, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError("token_embedding must be (V, H)")
        V, H = emb.shape
        if hp.shape != (H, H):
            raise ValueError("hidden_proj must be (H, H)")
        if tp.shape != (H, H):
            raise ValueError("token_proj must be (H, H)")
        if op.shape != (H, V):
            raise ValueError("out_proj must be (H, V)")
        if cp.shape != (H,):
            raise ValueError("confidence_proj must be (H,)")
        if self.markov is not None and _arr(self.markov).shape != (V, H):
            raise ValueError("markov must be (V, H)")


@dataclass(frozen=True)
class DSparkDraftOutput:
    logits: np.ndarray            # (B, A, D, V)
    confidence_logits: np.ndarray # (B, A, D)
    tokens: np.ndarray            # (B, A, D) greedy draft tokens
    hidden: np.ndarray            # (B, A, D, H)


@dataclass(frozen=True)
class DSparkProposal:
    anchor_index: np.ndarray      # (B,)
    anchor_position: np.ndarray   # (B,)
    prefix_length: np.ndarray     # (B,)
    tokens: np.ndarray            # (B, D), trailing entries are still present
    confidence: np.ndarray        # (B, D)


def sample_anchors(
    scores_or_length: Any,
    num_anchors: int,
    *,
    block_size: int = 1,
    mode: str = "topk",
) -> np.ndarray:
    """Select deterministic anchor positions.

    ``mode="topk"`` expects a 1-D score vector and returns the highest scoring
    valid positions in ascending order.  ``mode="uniform"`` accepts either an
    integer sequence length or a score vector and spreads anchors over the valid
    range.  Valid anchors leave room for ``block_size`` drafted tokens.
    """

    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if np.isscalar(scores_or_length):
        length = int(np.asarray(scores_or_length).item())
        scores = None
    else:
        scores = _arr(scores_or_length, dtype=np.float64).reshape(-1)
        length = int(scores.shape[0])
    max_start = length - block_size
    if max_start < 0:
        raise ValueError("sequence length must be at least block_size")
    valid = np.arange(max_start + 1, dtype=np.int64)
    if num_anchors > valid.shape[0]:
        raise ValueError("num_anchors exceeds number of valid anchor positions")
    if mode == "uniform":
        if num_anchors == 1:
            return np.array([valid[0]], dtype=np.int64)
        idx = np.rint(np.linspace(0, valid.shape[0] - 1, num_anchors)).astype(np.int64)
        return valid[idx]
    if mode == "topk":
        if scores is None:
            raise ValueError("topk anchor sampling requires scores")
        order = np.argsort(-scores[valid], kind="stable")[:num_anchors]
        return np.sort(valid[order]).astype(np.int64)
    raise ValueError(f"unknown anchor sampling mode {mode!r}")


def anchor_candidate_mask(
    anchors: Any,
    seq_len: int,
    block_size: int,
) -> np.ndarray:
    """Return ``(A, S)`` bool mask for positions covered by each anchor block."""

    a = _arr(anchors, dtype=np.int64).reshape(-1)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    mask = np.zeros((a.shape[0], seq_len), dtype=bool)
    for i, start in enumerate(a):
        if start < 0 or start + block_size > seq_len:
            raise ValueError("anchor block is out of range")
        mask[i, start:start + block_size] = True
    return mask


def anchor_block_attention_mask(
    anchors: Any,
    context_len: int,
    block_size: int,
    *,
    causal: bool = True,
) -> np.ndarray:
    """Return DSpark block attention mask ``(A, D, context_len + D)``.

    Each draft step can attend to the target context up to its absolute anchor
    position and, when ``causal=True``, earlier draft positions in the same
    block.  This is the block-mask contract DS2 will lower.
    """

    a = _arr(anchors, dtype=np.int64).reshape(-1)
    if context_len <= 0 or block_size <= 0:
        raise ValueError("context_len and block_size must be positive")
    mask = np.zeros((a.shape[0], block_size, context_len + block_size), dtype=bool)
    for ai, start in enumerate(a):
        if start < 0 or start >= context_len:
            raise ValueError("anchor must be inside the context")
        for d in range(block_size):
            upto = min(context_len, int(start) + d + 1)
            mask[ai, d, :upto] = True
            if causal:
                mask[ai, d, context_len:context_len + d] = True
            else:
                mask[ai, d, context_len:context_len + block_size] = True
    return mask


def draft_block_forward(
    target_hidden: Any,
    prev_tokens: Any,
    anchors: Any,
    weights: DSparkWeights,
    cfg: DSparkConfig,
) -> DSparkDraftOutput:
    """Reference DSpark vanilla draft block.

    Inputs:
    * ``target_hidden``: ``(B, S, H)`` target-cache hidden states.
    * ``prev_tokens``: ``(B,)`` current committed token for the first draft step.
    * ``anchors``: ``(A,)`` anchor positions into ``target_hidden``.

    Outputs match the roadmap contract: logits ``[B, anchors, block_size, vocab]``
    plus confidence logits over the same ``[B, anchors, block_size]`` grid.
    """

    th = _arr(target_hidden, dtype=np.float32)
    pt = _arr(prev_tokens, dtype=np.int64).reshape(-1)
    anc = _arr(anchors, dtype=np.int64).reshape(-1)
    if th.ndim != 3:
        raise ValueError("target_hidden must be (B, S, H)")
    B, S, H = th.shape
    if pt.shape != (B,):
        raise ValueError("prev_tokens must be (B,)")
    if anc.shape != (cfg.num_anchors,):
        raise ValueError("anchors must be (num_anchors,)")
    if anc.size and (anc.min() < 0 or anc.max() >= S):
        raise ValueError("anchors out of target_hidden range")
    if _arr(weights.token_embedding).shape != (cfg.vocab_size, H):
        raise ValueError("weights do not match cfg vocab/hidden size")
    if pt.size and (pt.min() < 0 or pt.max() >= cfg.vocab_size):
        raise ValueError("prev_tokens out of vocab range")

    emb = _arr(weights.token_embedding, dtype=np.float32)
    hp = _arr(weights.hidden_proj, dtype=np.float32)
    tp = _arr(weights.token_proj, dtype=np.float32)
    op = _arr(weights.out_proj, dtype=np.float32)
    cp = _arr(weights.confidence_proj, dtype=np.float32)
    mk = None if weights.markov is None else _arr(weights.markov, dtype=np.float32)

    logits = np.empty((B, cfg.num_anchors, cfg.block_size, cfg.vocab_size), dtype=np.float32)
    conf = np.empty((B, cfg.num_anchors, cfg.block_size), dtype=np.float32)
    hidden = np.empty((B, cfg.num_anchors, cfg.block_size, H), dtype=np.float32)
    tokens = np.empty((B, cfg.num_anchors, cfg.block_size), dtype=np.int64)

    for b in range(B):
        for ai, start in enumerate(anc):
            state = th[b, int(start)]
            prev = int(pt[b])
            for d in range(cfg.block_size):
                token_term = emb[prev] @ tp
                markov_term = 0.0 if mk is None else mk[prev]
                state = np.tanh(state @ hp + token_term + markov_term).astype(np.float32)
                row = state @ op
                logits[b, ai, d] = row
                conf[b, ai, d] = float(state @ cp)
                tok = int(np.argmax(row))
                tokens[b, ai, d] = tok
                prev = tok
                hidden[b, ai, d] = state
    return DSparkDraftOutput(logits=logits, confidence_logits=conf,
                             tokens=tokens, hidden=hidden)


def confident_prefix(confidence_logits: Any, threshold: float) -> np.ndarray:
    """Leading confident prefix length for every ``(B, A)`` draft block."""

    c = _arr(confidence_logits, dtype=np.float32)
    if c.ndim != 3:
        raise ValueError("confidence_logits must be (B, A, D)")
    probs = _sigmoid(c)
    ok = probs >= float(threshold)
    out = np.zeros(c.shape[:2], dtype=np.int64)
    for idx in np.ndindex(c.shape[0], c.shape[1]):
        n = 0
        for flag in ok[idx]:
            if not bool(flag):
                break
            n += 1
        out[idx] = n
    return out


def select_proposal(
    draft: DSparkDraftOutput,
    anchors: Any,
    *,
    threshold: float,
) -> DSparkProposal:
    """Pick one anchor per batch row by longest confident prefix, first tie wins."""

    anc = _arr(anchors, dtype=np.int64).reshape(-1)
    lengths = confident_prefix(draft.confidence_logits, threshold)
    B, A = lengths.shape
    if anc.shape != (A,):
        raise ValueError("anchors must match draft anchor dimension")
    best = np.argmax(lengths, axis=1).astype(np.int64)
    rows = np.arange(B)
    return DSparkProposal(
        anchor_index=best,
        anchor_position=anc[best],
        prefix_length=lengths[rows, best].astype(np.int64),
        tokens=draft.tokens[rows, best].astype(np.int64),
        confidence=_sigmoid(draft.confidence_logits[rows, best]).astype(np.float32),
    )


def dspark_losses(
    logits: Any,
    target_ids: Any,
    *,
    mask: Any | None = None,
    target_logits: Any | None = None,
    confidence_logits: Any | None = None,
    confidence_targets: Any | None = None,
    l1_weight: float = 0.0,
    prob_weight: float = 0.0,
) -> dict[str, float]:
    """CE plus optional L1/probability-matching losses over ``[B,A,D,V]`` logits."""

    lg = _arr(logits, dtype=np.float64)
    ids = _arr(target_ids, dtype=np.int64)
    if lg.ndim != 4:
        raise ValueError("logits must be (B, A, D, V)")
    if ids.shape != lg.shape[:-1]:
        raise ValueError("target_ids must be (B, A, D)")
    if ids.size and (ids.min() < 0 or ids.max() >= lg.shape[-1]):
        raise ValueError("target_ids out of vocab range")
    m = np.ones(ids.shape, dtype=bool) if mask is None else _arr(mask).astype(bool)
    if m.shape != ids.shape:
        raise ValueError("mask must be (B, A, D)")
    denom = max(1, int(np.count_nonzero(m)))
    logp = np.log(_softmax(lg, axis=-1) + 1e-30)
    ce_vals = -np.take_along_axis(logp, ids[..., None], axis=-1)[..., 0]
    ce = float(np.sum(np.where(m, ce_vals, 0.0)) / denom)
    total = ce
    result: dict[str, float] = {"ce": ce}

    if target_logits is not None:
        tl = _arr(target_logits, dtype=np.float64)
        if tl.shape != lg.shape:
            raise ValueError("target_logits must match logits")
        if l1_weight:
            l1 = float(np.sum(np.where(m[..., None], np.abs(lg - tl), 0.0)) /
                       (denom * lg.shape[-1]))
            result["l1"] = l1
            total += float(l1_weight) * l1
        if prob_weight:
            prob = float(np.sum(np.where(
                m[..., None],
                np.abs(_softmax(lg, axis=-1) - _softmax(tl, axis=-1)),
                0.0,
            )) / (denom * lg.shape[-1]))
            result["prob"] = prob
            total += float(prob_weight) * prob

    if confidence_logits is not None or confidence_targets is not None:
        if confidence_logits is None or confidence_targets is None:
            raise ValueError("confidence logits and targets must be provided together")
        cl = _arr(confidence_logits, dtype=np.float64)
        ct = _arr(confidence_targets, dtype=np.float64)
        if cl.shape != ids.shape or ct.shape != ids.shape:
            raise ValueError("confidence tensors must be (B, A, D)")
        p = np.clip(_sigmoid(cl), 1e-7, 1.0 - 1e-7)
        bce_vals = -(ct * np.log(p) + (1.0 - ct) * np.log(1.0 - p))
        bce = float(np.sum(np.where(m, bce_vals, 0.0)) / denom)
        result["confidence_bce"] = bce
        total += bce

    result["total"] = float(total)
    return result


__all__ = [
    "DSparkConfig",
    "DSparkWeights",
    "DSparkDraftOutput",
    "DSparkProposal",
    "sample_anchors",
    "anchor_candidate_mask",
    "anchor_block_attention_mask",
    "draft_block_forward",
    "confident_prefix",
    "select_proposal",
    "dspark_losses",
]
