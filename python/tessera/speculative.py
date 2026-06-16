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


__all__ = [
    "DraftTree",
    "VerificationResult",
    "SpeculativeStep",
    "expand_tree",
    "acceptance_probabilities",
    "batch_verify",
    "advance_kv",
    "advance_ssm",
]
