"""Speculative decoding scheduler primitives (Theme 6).

Speculative decoding pairs a small **draft** model with a larger
**target** model: the draft generates a tree of candidate continuations,
the target verifies them in a single batched forward, and only the
prefix the target endorses is committed to the KV cache. The win is
end-to-end throughput — the target runs once for many draft tokens.

This module ships the *scheduler* primitives — the Python control flow
that builds the tree, runs the verification probability check, computes
the acceptance mask, and advances a KV cache by the accepted prefix
length. Per-backend Graph IR control flow ops (so a `@tessera.jit` of
the whole loop can lower into a single dispatched kernel) are deferred
to Phase G — the Python surface unblocks
``examples/advanced/speculative_decoding/`` end-to-end on the CPU
reference path today.

API contract:

    tree = tessera.speculative.expand_tree(prefix, branching, depth)
    result = tessera.speculative.batch_verify(
        target_log_probs, draft_log_probs, draft_tokens, *, rng=None,
    )
    cache = tessera.speculative.advance_kv(cache, result.accepted_prefix)

Design references:
    Leviathan et al. 2023 — "Fast Inference from Transformers via
    Speculative Decoding" (the canonical acceptance rule, eq. 1).
    Cai et al. 2024 — "Medusa: Simple LLM Inference Acceleration".
    Yggdrasil — multi-branch tree variant (used by `examples/advanced/
    speculative_decoding/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tree expansion
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DraftTree:
    """Flat representation of a speculative draft tree.

    Each node stores its parent index (``parents[i]``) and the token
    id chosen at that step (``tokens[i]``). Index 0 is the implicit root
    holding the prompt's last token; nodes 1..N are draft proposals.

    ``branching`` and ``depth`` are recorded for convenience — the
    scheduler doesn't actually require them at verify time, but they
    help with reshaping batched verifier inputs.
    """

    tokens: np.ndarray         # shape (num_nodes,), int64 token ids
    parents: np.ndarray        # shape (num_nodes,), int64 parent indices
    paths: np.ndarray          # shape (num_paths, depth + 1), int64 — flat root→leaf paths
    branching: int
    depth: int

    @property
    def num_nodes(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_paths(self) -> int:
        return int(self.paths.shape[0])


def expand_tree(
    root_token: int,
    *,
    draft_tokens: Sequence[Sequence[int]],
    branching: int,
    depth: int,
) -> DraftTree:
    """Build a balanced draft tree of width ``branching`` and ``depth``.

    ``draft_tokens`` is the per-level draft proposal: ``draft_tokens[d][b]``
    is the token chosen at level ``d`` for branch index ``b`` of every
    node at the parent level. The simplest case is a fixed branching
    factor (``branching ** depth`` total leaves), which matches Yggdrasil
    and Medusa-style decoders.

    Returns a :class:`DraftTree` whose ``paths`` is a flat
    ``(branching ** depth, depth + 1)`` int64 matrix of root-to-leaf
    sequences — the form `batch_verify` consumes.
    """
    if branching <= 0 or depth < 0:
        raise ValueError("branching > 0 and depth >= 0 required")
    if len(draft_tokens) != depth:
        raise ValueError(
            f"draft_tokens must have one row per level (got "
            f"{len(draft_tokens)} rows, expected {depth})"
        )
    for d, level in enumerate(draft_tokens):
        if len(level) != branching:
            raise ValueError(
                f"draft_tokens[{d}] has {len(level)} entries; expected "
                f"branching={branching}"
            )

    # Build BFS level-order: level 0 is just the root, level d has
    # branching**d nodes — total = (branching**(depth+1) - 1) /
    # (branching - 1). Stack the per-level slabs.
    tokens_list: list[int] = [int(root_token)]
    parents_list: list[int] = [-1]
    level_starts: list[int] = [0]  # node index where each level begins
    for d in range(depth):
        prev_start = level_starts[-1]
        prev_count = (1 if d == 0 else branching ** d)
        level_starts.append(len(tokens_list))
        for parent_offset in range(prev_count):
            parent_idx = prev_start + parent_offset
            for b in range(branching):
                tokens_list.append(int(draft_tokens[d][b]))
                parents_list.append(parent_idx)

    tokens = np.asarray(tokens_list, dtype=np.int64)
    parents = np.asarray(parents_list, dtype=np.int64)

    # Build the flat paths matrix. Each leaf at level `depth` has a
    # unique walk back through `parents`. With balanced branching =
    # branching**depth leaves.
    leaves_start = level_starts[depth] if depth > 0 else 0
    leaves_end = len(tokens_list)
    num_paths = leaves_end - leaves_start if depth > 0 else 1

    paths = np.zeros((num_paths, depth + 1), dtype=np.int64)
    for i in range(num_paths):
        leaf = leaves_start + i if depth > 0 else 0
        cursor = leaf
        for d in range(depth, -1, -1):
            paths[i, d] = tokens[cursor]
            cursor = int(parents[cursor]) if d > 0 else cursor

    return DraftTree(
        tokens=tokens, parents=parents, paths=paths,
        branching=branching, depth=depth,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch verification — Leviathan et al. acceptance rule
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of running the target model on a draft tree.

    Attributes
    ----------
    acceptance_mask
        Boolean array, shape ``(num_paths, depth)`` — True means the
        target endorsed the draft token at that path/step under the
        Leviathan acceptance rule.
    accepted_path_idx
        Index of the longest-accepted-prefix path through the tree (the
        one we'll commit to the KV cache). Selection rule is "first
        path with the longest accepted prefix" — deterministic and
        matches Yggdrasil's tie-break.
    accepted_prefix_length
        Number of accepted draft tokens (0..depth). Equal to the prefix
        length that gets committed to the KV cache.
    accepted_prefix
        The accepted token ids, length ``accepted_prefix_length``.
    """

    acceptance_mask: np.ndarray
    accepted_path_idx: int
    accepted_prefix_length: int
    accepted_prefix: np.ndarray


@dataclass(frozen=True)
class SpeculativeSamplingConfig:
    """Sampling policy used for draft and target probability accounting."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    eps: float = 1e-12


@dataclass(frozen=True)
class DraftSample:
    """A sampled MTP draft chain with cached draft distributions."""

    tokens: np.ndarray
    probs: np.ndarray
    token_probs: np.ndarray
    log_probs: np.ndarray


@dataclass(frozen=True)
class RejectionSamplingResult:
    """Distribution-preserving chain verification result."""

    accepted: int
    new_tokens: np.ndarray
    acceptance_mask: np.ndarray
    acceptance_probs: np.ndarray
    rejected_at: int | None
    residual_probs: np.ndarray
    emitted_from: str


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("sampling logits must be finite")
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _normalize_probs(x: np.ndarray, *, eps: float = 1e-12, name: str = "probs") -> np.ndarray:
    p = np.asarray(x, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = p.sum(axis=-1, keepdims=True)
    if np.any(total <= eps):
        raise ValueError(f"{name} rows must have positive mass")
    return p / total


def _sampling_probs(logits: np.ndarray, config: SpeculativeSamplingConfig) -> np.ndarray:
    lg = np.asarray(logits, dtype=np.float64)
    temp = max(float(config.temperature), config.eps)
    flat = (lg / temp).reshape(-1, lg.shape[-1])
    V = flat.shape[-1]
    out = np.empty_like(flat)
    for i, row0 in enumerate(flat):
        row = row0.copy()
        if config.top_k and 0 < config.top_k < V:
            kth = np.partition(row, -config.top_k)[-config.top_k]
            row = np.where(row < kth, -np.inf, row)
        p = _softmax(row)
        if config.top_p and 0.0 < config.top_p < 1.0:
            order = np.argsort(-p, kind="stable")
            csum = np.cumsum(p[order])
            cut = int(np.searchsorted(csum, config.top_p, side="left")) + 1
            keep = order[:cut]
            masked = np.zeros_like(p)
            masked[keep] = p[keep]
            p = masked / max(masked.sum(), config.eps)
        out[i] = p
    return out.reshape(lg.shape)


def sample_draft_chain(
    draft_logits: np.ndarray,
    *,
    config: SpeculativeSamplingConfig | None = None,
    rng: Optional[np.random.Generator] = None,
) -> DraftSample:
    """Sample a linear MTP draft chain from ``q`` and cache draft probabilities."""
    cfg = config or SpeculativeSamplingConfig()
    probs = _sampling_probs(np.asarray(draft_logits), cfg)
    flat = probs.reshape(-1, probs.shape[-1])
    gen = rng if rng is not None else np.random.default_rng()
    tokens = np.empty(flat.shape[0], dtype=np.int64)
    for i, p in enumerate(flat):
        tokens[i] = int(gen.choice(p.shape[0], p=p))
    token_probs = flat[np.arange(flat.shape[0]), tokens]
    log_probs = np.log(np.maximum(token_probs, cfg.eps))
    return DraftSample(
        tokens=tokens.reshape(probs.shape[:-1]),
        probs=probs,
        token_probs=token_probs.reshape(probs.shape[:-1]),
        log_probs=log_probs.reshape(probs.shape[:-1]),
    )


def residual_distribution(target_probs: np.ndarray, draft_probs: np.ndarray, *,
                          eps: float = 1e-12) -> np.ndarray:
    """Return ``normalize(max(0, p - q))`` for a rejected speculative step."""
    p = np.asarray(target_probs, dtype=np.float64)
    q = np.asarray(draft_probs, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"target/draft prob shapes must match; got {p.shape} vs {q.shape}")
    p = _normalize_probs(p, eps=eps, name="target_probs")
    q = _normalize_probs(q, eps=eps, name="draft_probs")
    resid = np.maximum(p - q, 0.0)
    total = resid.sum(axis=-1, keepdims=True)
    fallback = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    return np.where(total > eps, resid / np.maximum(total, eps), fallback)


def rejection_verify_chain(
    draft_tokens: np.ndarray,
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
) -> RejectionSamplingResult:
    """Verify a linear MTP draft with distribution-preserving rejection sampling.

    ``target_probs`` may have one extra row for the all-accepted bonus token.
    """
    d = np.asarray(draft_tokens, dtype=np.int64).reshape(-1)
    q = np.asarray(draft_probs, dtype=np.float64)
    p = np.asarray(target_probs, dtype=np.float64)
    if q.ndim != 2 or p.ndim != 2:
        raise ValueError("draft_probs and target_probs must be rank-2")
    if q.shape[0] != d.shape[0] or p.shape[0] < d.shape[0] or q.shape[1] != p.shape[1]:
        raise ValueError(
            "draft tokens/probs and target probs must have compatible "
            f"chain/vocab shapes; got tokens={d.shape}, q={q.shape}, p={p.shape}")
    if np.any(d < 0) or np.any(d >= q.shape[1]):
        raise ValueError("draft_tokens contain ids outside the vocabulary")
    q = _normalize_probs(q, eps=eps, name="draft_probs")
    p = _normalize_probs(p, eps=eps, name="target_probs")
    gen = rng if rng is not None else np.random.default_rng()
    V = q.shape[1]
    acceptance_probs = np.zeros(d.shape[0], dtype=np.float64)
    acceptance_mask = np.zeros(d.shape[0], dtype=bool)
    new_tokens: list[int] = []
    for i, tok in enumerate(d):
        q_tok = q[i, tok]
        p_tok = p[i, tok]
        accept_prob = 1.0 if q_tok <= eps and p_tok > eps else min(1.0, p_tok / max(q_tok, eps))
        acceptance_probs[i] = accept_prob
        if gen.random() <= accept_prob:
            acceptance_mask[i] = True
            new_tokens.append(int(tok))
            continue
        resid = residual_distribution(p[i], q[i], eps=eps)
        new_tokens.append(int(gen.choice(V, p=resid)))
        return RejectionSamplingResult(
            accepted=i,
            new_tokens=np.asarray(new_tokens, dtype=np.int64),
            acceptance_mask=acceptance_mask,
            acceptance_probs=acceptance_probs,
            rejected_at=i,
            residual_probs=resid,
            emitted_from="residual",
        )
    bonus_row = p[d.shape[0]] if p.shape[0] > d.shape[0] else p[-1]
    bonus = bonus_row / np.maximum(bonus_row.sum(), eps)
    new_tokens.append(int(gen.choice(V, p=bonus)))
    return RejectionSamplingResult(
        accepted=d.shape[0],
        new_tokens=np.asarray(new_tokens, dtype=np.int64),
        acceptance_mask=acceptance_mask,
        acceptance_probs=acceptance_probs,
        rejected_at=None,
        residual_probs=bonus,
        emitted_from="target_bonus",
    )


def acceptance_probabilities(
    target_log_probs: np.ndarray, draft_log_probs: np.ndarray,
) -> np.ndarray:
    """Per-token acceptance probability ``min(1, p_target / p_draft)``
    from the Leviathan acceptance rule.

    Both inputs are log-probabilities of the *same* draft tokens; the
    return is ``exp(target_log_probs - draft_log_probs)`` clipped to
    ``[0, 1]``. Shape passes through.
    """
    target_log_probs = np.asarray(target_log_probs, dtype=np.float64)
    draft_log_probs = np.asarray(draft_log_probs, dtype=np.float64)
    if target_log_probs.shape != draft_log_probs.shape:
        raise ValueError(
            f"target/draft log-prob shapes must match; got "
            f"{target_log_probs.shape} vs {draft_log_probs.shape}"
        )
    log_ratio = target_log_probs - draft_log_probs
    return np.clip(np.exp(log_ratio), 0.0, 1.0)


def batch_verify(
    *,
    target_log_probs: np.ndarray,   # shape (num_paths, depth)
    draft_log_probs: np.ndarray,    # shape (num_paths, depth)
    paths: np.ndarray,              # shape (num_paths, depth + 1) — root + draft tokens
    rng: Optional[np.random.Generator] = None,
) -> VerificationResult:
    """Run the Leviathan acceptance rule across every path of a draft tree.

    For each ``(path, step)`` we draw ``u ~ Uniform(0, 1)`` and accept the
    draft token iff ``u <= p_target / p_draft``. We then walk each path
    left-to-right, accepting tokens until the first rejection — the
    accepted-prefix length per path is the run-length of leading
    ``True`` values in ``acceptance_mask``.

    The returned ``accepted_path_idx`` is the first path achieving the
    longest accepted prefix (deterministic tie-break). Use the returned
    ``accepted_prefix`` to advance the KV cache via :func:`advance_kv`.
    """
    target_log_probs = np.asarray(target_log_probs)
    draft_log_probs = np.asarray(draft_log_probs)
    paths = np.asarray(paths, dtype=np.int64)
    if target_log_probs.shape != draft_log_probs.shape:
        raise ValueError(
            f"target/draft log-prob shapes must match; got "
            f"{target_log_probs.shape} vs {draft_log_probs.shape}"
        )
    if target_log_probs.ndim != 2:
        raise ValueError(
            f"target_log_probs must be 2D (num_paths, depth); "
            f"got shape {target_log_probs.shape}"
        )
    num_paths, depth = target_log_probs.shape
    if paths.shape != (num_paths, depth + 1):
        raise ValueError(
            f"paths must have shape (num_paths={num_paths}, depth+1="
            f"{depth + 1}); got {paths.shape}"
        )

    if rng is None:
        rng = np.random.default_rng()
    accept_probs = acceptance_probabilities(target_log_probs, draft_log_probs)
    u = rng.uniform(size=accept_probs.shape)
    acceptance_mask = u <= accept_probs

    # Per-path accepted-prefix length: leading run of True values.
    cumulative = np.cumprod(acceptance_mask, axis=1)
    accepted_lengths = cumulative.sum(axis=1)

    # Pick the path with the longest accepted prefix (first wins on ties).
    best_idx = int(np.argmax(accepted_lengths))
    best_len = int(accepted_lengths[best_idx])
    # paths[i, 0] is the root token; the draft tokens are paths[i, 1:].
    accepted_prefix = paths[best_idx, 1:1 + best_len].copy()

    return VerificationResult(
        acceptance_mask=acceptance_mask,
        accepted_path_idx=best_idx,
        accepted_prefix_length=best_len,
        accepted_prefix=accepted_prefix,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache advance — commit only the accepted prefix
# ─────────────────────────────────────────────────────────────────────────────


def advance_kv(cache, accepted_prefix_length: int):
    """Advance ``cache`` by exactly ``accepted_prefix_length`` tokens.

    Speculative decoding ran the target on a tree of ``num_paths``
    candidates, but only the accepted prefix becomes part of the
    sequence. This op trims the cache so its ``current_seq`` equals the
    pre-speculation seq plus the accepted-prefix length — discarding
    the rejected draft tokens.

    The cache is mutated in place and returned for chaining. Today this
    is a metadata op (the underlying numpy buffer is fixed-size); on a
    paged backend it'll free the rejected pages.

    Works on both :class:`tessera.cache.KVCacheHandle` and
    :class:`tessera.cache.LatentKVCacheHandle`.
    """
    if accepted_prefix_length < 0:
        raise ValueError("accepted_prefix_length must be non-negative")
    if not hasattr(cache, "current_seq"):
        raise TypeError(
            f"advance_kv expects a tessera.cache.* handle; got "
            f"{type(cache).__name__}"
        )
    # The caller pre-appended the entire draft tree to the cache; we
    # need to trim back to (pre_seq + accepted_prefix_length).
    # The convention: caller passes the *new* expected current_seq
    # explicitly via accepted_prefix_length being interpreted as "this
    # many tokens from the front are kept, the rest discarded".
    if accepted_prefix_length > cache.current_seq:
        raise ValueError(
            f"accepted_prefix_length={accepted_prefix_length} exceeds "
            f"cache.current_seq={cache.current_seq}"
        )
    # Zero out the trailing slots that would otherwise leak rejected
    # draft state into the next step.
    if hasattr(cache, "keys"):
        cache.keys[accepted_prefix_length:cache.current_seq] = 0
        cache.values[accepted_prefix_length:cache.current_seq] = 0
        if getattr(cache, "_scales", None) is not None:
            cache._scales[:, accepted_prefix_length:cache.current_seq] = 0
    elif hasattr(cache, "latents"):
        cache.latents[accepted_prefix_length:cache.current_seq] = 0
    cache.current_seq = accepted_prefix_length
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# SSM decode-state advance — the ReplaySSM sibling of advance_kv (Track-R)
# ─────────────────────────────────────────────────────────────────────────────


def advance_ssm(handle, num_accepted: int, *, num_drafts: int):
    """Commit the accepted speculative prefix on an SSM decode-state handle.

    The SSM analogue of :func:`advance_kv`.  Attention models trim a KV cache;
    selective-SSM models (Mamba-2 / Gated DeltaNet) instead carry a recurrent
    state, which ReplaySSM keeps as a checkpoint plus a ring buffer of recent
    replay inputs (:class:`tessera.cache.SSMStateHandle`).

    The caller appended ``num_drafts`` speculative tokens to the handle's ring
    buffer (one ``append``/``step`` per draft).  The target accepted the first
    ``num_accepted`` of them; this op rejects the rest by **rewinding the
    cursor** — ``handle.rollback(num_drafts - num_accepted)`` — with no
    per-position state snapshot.  Because ReplaySSM's flush rule reserves
    ``2*spec_window`` slots, the draft burst never forced a flush, so the
    rejected tokens are still live in the buffer and rollback is exact.

    Mutates the handle in place and returns it for chaining.
    """
    if num_drafts < 0:
        raise ValueError("num_drafts must be non-negative")
    if not (0 <= num_accepted <= num_drafts):
        raise ValueError(
            f"num_accepted must be in [0, num_drafts={num_drafts}]; "
            f"got {num_accepted}"
        )
    if not (hasattr(handle, "rollback") and hasattr(handle, "count")):
        raise TypeError(
            f"advance_ssm expects a tessera.cache.SSMStateHandle-like handle "
            f"(with rollback()/count); got {type(handle).__name__}"
        )
    if num_drafts > handle.count:
        raise ValueError(
            f"num_drafts={num_drafts} exceeds live replay tokens "
            f"handle.count={handle.count} (a flush dropped draft tokens — "
            f"raise spec_window so the burst fits the ring buffer)"
        )
    handle.rollback(num_drafts - num_accepted)
    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: end-to-end speculative-step orchestration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SpeculativeStep:
    """One step of speculative decoding orchestration.

    Holds the configuration and provides a single :meth:`run` method that
    drives the four primitives in order: ``expand_tree → batch_verify →
    advance_kv → return accepted prefix``.
    """

    branching: int
    depth: int

    def run(
        self,
        *,
        root_token: int,
        draft_tokens: Sequence[Sequence[int]],
        target_log_probs: np.ndarray,
        draft_log_probs: np.ndarray,
        cache=None,
        cache_pre_seq: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> VerificationResult:
        """Run one speculative-decoding step.

        ``cache`` and ``cache_pre_seq`` are optional — if provided, the
        cache is advanced by the accepted-prefix length. The caller is
        expected to have appended the full draft to the cache before
        invocation; ``cache_pre_seq`` records the pre-speculation
        length so we know where the accepted slice ends.
        """
        tree = expand_tree(
            root_token,
            draft_tokens=draft_tokens,
            branching=self.branching,
            depth=self.depth,
        )
        result = batch_verify(
            target_log_probs=target_log_probs,
            draft_log_probs=draft_log_probs,
            paths=tree.paths,
            rng=rng,
        )
        if cache is not None:
            advance_kv(cache, cache_pre_seq + result.accepted_prefix_length)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Gumiho serial-draft loop — the SD1 primitives composed as ONE decode loop
# ─────────────────────────────────────────────────────────────────────────────
def autoregressive_decode(*, prompt, target_next, max_new: int) -> list[int]:
    """Plain greedy autoregressive decode — the reference that the speculative
    loop must reproduce token-for-token. ``target_next(context) -> int`` is the
    target model's greedy next token given the context so far."""
    seq = list(prompt)
    for _ in range(int(max_new)):
        seq.append(int(target_next(seq)))
    return seq


def gumiho_serial_draft(*, prompt, draft_next, target_next, max_new: int,
                        draft_len: int) -> list[int]:
    """Gumiho-style serial-draft speculative decode as ONE composed loop.

    Each iteration runs the SD1 primitive chain over a serial draft:

      1. **draft** ``draft_len`` tokens autoregressively from the small draft model
         (``draft_next(context) -> int``);
      2. **target_verify** — the target model scores the current + draft prefix,
         giving its greedy next token at each of ``draft_len+1`` positions
         (``target_next`` over each prefix; the ``tessera.target_verify`` I/O
         contract);
      3. **spec_accept** — accept the longest matching draft prefix and emit the
         accepted tokens plus one bonus (the target's correction at the first
         divergence — the greedy ``dflash_linear_verify`` rule);
      4. **cache_commit** — the accepted prefix length advances the cursor.

    The loop wrapper is exactly the kind of bounded recurrence that lowers to one
    ``control_scan`` device dispatch (CF4e), so the serial draft executes as one
    backend loop rather than one launch per token.

    Greedy invariant (proven in the tests): for any ``draft_next``, the emitted
    sequence **equals** :func:`autoregressive_decode` with ``target_next`` —
    speculation changes only the number of target calls, never the output.
    """
    if draft_len < 1:
        raise ValueError("draft_len must be >= 1")
    seq = list(prompt)
    base = len(seq)
    while len(seq) - base < int(max_new):
        # 1. serial draft from the draft model.
        draft: list[int] = []
        ctx = list(seq)
        for _ in range(int(draft_len)):
            tok = int(draft_next(ctx))
            draft.append(tok)
            ctx.append(tok)
        # 2. target greedy next-token at each verified position (target_verify).
        target = [int(target_next(seq + draft[:k])) for k in range(draft_len + 1)]
        # 3. greedy accept: longest matching prefix, then the one bonus correction.
        accepted = 0
        for i in range(draft_len):
            if draft[i] == target[i]:
                accepted += 1
            else:
                break
        # accepted draft tokens (== the target's) + 1 bonus (target[accepted]).
        seq.extend(draft[:accepted] + [target[accepted]])
        # 4. cache_commit would advance the cursor by `accepted` here.
    return seq[: base + int(max_new)]


__all__ = [
    "DraftTree",
    "VerificationResult",
    "SpeculativeSamplingConfig",
    "DraftSample",
    "RejectionSamplingResult",
    "SpeculativeStep",
    "expand_tree",
    "sample_draft_chain",
    "rejection_verify_chain",
    "residual_distribution",
    "acceptance_probabilities",
    "batch_verify",
    "advance_kv",
    "advance_ssm",
    "autoregressive_decode",
    "gumiho_serial_draft",
]
