"""Named multimodal graph walks + encoder-free ops (Workstream F).

The audit's #7/#8 lessons, re-scoped per the P3 feedback. The MiniMax-M3 graph
builder already emits image/video preprocess → patch_embed → patch_merge →
media_project → splice nodes. The two real gaps:

  #7 — no **named graph-walk abstraction**: the graph is one flat node list, not
       addressable entry points like ``vision_prefill`` / ``text_decode`` /
       ``image_gen`` that compile and schedule independently.
  #8 — no native **audio / coordinate-token** path: patch/audio projection is
       opaque preprocessing, not first-class low-latency ops.

This module supplies both:

  * :class:`ModelWalk` + :func:`partition_walks` — split a multimodal graph into
    named walks. Oracle: the partition is lossless (walks reconstruct the graph).
  * First-class **encoder-free ops** (`patch_embed`, `coordinate_lookup`,
    `audio_frame_projection`, `splice_embeddings`) + an executable
    :class:`EncoderFreeVLM` whose named walks recompose into the monolithic
    forward — proven numerically by :func:`verify_walk_parity`.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream F).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Named-walk abstraction over a multimodal graph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelWalk:
    """A named, independently-schedulable entry point over a subset of graph nodes.

    ``name`` is the walk (``vision_prefill`` / ``video_prefill`` / ``text_decode``
    / ``splice`` / ``image_gen``). ``nodes`` are the graph nodes it owns;
    ``consumes`` / ``produces`` declare its handoff so walks chain like the
    prefill→decode handoff in Workstream B.
    """

    name: str
    nodes: tuple[Any, ...]
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    @property
    def op_sequence(self) -> tuple[str, ...]:
        return tuple(n.op for n in self.nodes)


# Which ops belong to which walk. media-tower ops form the per-modality prefill
# walks; the splice op is its own join walk.
_VISION_OPS = ("image_preprocess", "patch_embed", "patch_merge", "media_project")
_VIDEO_OPS = ("video_frame_sample", "patch_embed", "patch_merge", "media_project")
_AUDIO_OPS = ("audio_frame_sample", "audio_frame_projection", "media_project")


def partition_walks(graph: Any) -> dict[str, ModelWalk]:
    """Split a multimodal graph into named walks.

    Groups by the ``media_kind`` attr on each node (image→vision_prefill,
    video→video_prefill, audio→audio_prefill); the ``splice_embeddings`` node
    becomes the ``splice`` join walk. Nodes with no media_kind that precede a
    media chain (preprocess / frame_sample) attach to that modality's walk.
    """
    buckets: dict[str, list[Any]] = {}
    splice_nodes: list[Any] = []
    # Pending preprocess nodes wait for the first media_kind node to claim them.
    pending: list[Any] = []

    for n in graph.nodes:
        if n.op == "splice_embeddings":
            splice_nodes.append(n)
            continue
        kind = n.attrs.get("media_kind")
        if kind is None:
            # preprocess / frame-sample node — infer modality from its op.
            if n.op == "image_preprocess":
                buckets.setdefault("vision_prefill", []).append(n)
            elif n.op == "video_frame_sample":
                buckets.setdefault("video_prefill", []).append(n)
            elif n.op == "audio_frame_sample":
                buckets.setdefault("audio_prefill", []).append(n)
            else:
                pending.append(n)
            continue
        walk = {"image": "vision_prefill", "video": "video_prefill",
                "audio": "audio_prefill"}.get(kind, f"{kind}_prefill")
        bucket = buckets.setdefault(walk, [])
        bucket.extend(pending)
        pending.clear()
        bucket.append(n)

    walks: dict[str, ModelWalk] = {}
    for name, nodes in buckets.items():
        walks[name] = ModelWalk(
            name=name, nodes=tuple(nodes),
            consumes=(nodes[0].inputs[0][0] if isinstance(nodes[0].inputs[0][0], str)
                      else "pixels",),
            produces=(f"{name.replace('_prefill', '')}_projected",),
        )
    if splice_nodes:
        walks["splice"] = ModelWalk(
            name="splice", nodes=tuple(splice_nodes),
            consumes=tuple(w.produces[0] for w in walks.values()) + ("text_embeddings",),
            produces=("spliced_embeddings",),
        )
    return walks


def walks_reconstruct_graph(graph: Any, walks: dict[str, ModelWalk]) -> bool:
    """Oracle: the partition is lossless — every graph node lands in exactly one
    walk, and no node is invented."""
    walk_nodes: list[Any] = []
    for w in walks.values():
        walk_nodes.extend(w.nodes)
    return sorted(id(n) for n in walk_nodes) == sorted(id(n) for n in graph.nodes)


# ─────────────────────────────────────────────────────────────────────────────
# First-class encoder-free ops (#8) — raw patch / coordinate / audio projection
# ─────────────────────────────────────────────────────────────────────────────


def patch_embed(pixels: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Raw patch projection (Gemma-style encoder-free): ``(P, patch_dim) @ W``.

    No convolutional encoder — patches are flattened and linearly projected, the
    encoder-free path the audit calls for.
    """
    pixels = np.asarray(pixels, np.float32)
    return pixels @ np.asarray(weight, np.float32)


def coordinate_lookup(positions: np.ndarray, table: np.ndarray) -> np.ndarray:
    """First-class coordinate/position-token lookup: gather rows of ``table``."""
    idx = np.asarray(positions, dtype=np.int64).reshape(-1)
    return np.asarray(table, np.float32)[idx]


def audio_frame_projection(frames: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """First-class audio frame projection: ``(F, frame_dim) @ W`` → ``(F, D)``."""
    frames = np.asarray(frames, np.float32)
    return frames @ np.asarray(weight, np.float32)


def splice_embeddings(*streams: np.ndarray) -> np.ndarray:
    """Token splice: concatenate modality token streams into one sequence."""
    return np.concatenate([np.asarray(s, np.float32) for s in streams], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Executable encoder-free VLM — proves walk decomposition == monolith forward
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EncoderFreeWeights:
    patch_w: np.ndarray        # (patch_dim, D)
    pos_table: np.ndarray      # (max_pos, D)
    audio_w: np.ndarray        # (frame_dim, D)
    text_table: np.ndarray     # (vocab, D)


@dataclass
class EncoderFreeVLM:
    """A tiny encoder-free multimodal model: raw patch + coordinate + audio
    projection spliced with text tokens. Exposes both a monolithic ``forward``
    and named ``walks`` that recompose to it."""

    weights: EncoderFreeWeights

    def vision_prefill(self, pixels: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return patch_embed(pixels, self.weights.patch_w) + \
            coordinate_lookup(positions, self.weights.pos_table)

    def audio_prefill(self, audio: np.ndarray) -> np.ndarray:
        return audio_frame_projection(audio, self.weights.audio_w)

    def text_decode(self, token_ids: np.ndarray) -> np.ndarray:
        return coordinate_lookup(token_ids, self.weights.text_table)

    def forward(self, pixels, positions, audio, token_ids) -> np.ndarray:
        """Monolithic forward over all modalities."""
        v = self.vision_prefill(pixels, positions)
        a = self.audio_prefill(audio)
        t = self.text_decode(token_ids)
        return splice_embeddings(v, a, t)

    def walks(self) -> dict[str, Callable[..., np.ndarray]]:
        """The named entry points — vision_prefill / audio_prefill / text_decode /
        splice — each independently runnable."""
        return {
            "vision_prefill": self.vision_prefill,
            "audio_prefill": self.audio_prefill,
            "text_decode": self.text_decode,
            "splice": splice_embeddings,
        }


@dataclass(frozen=True)
class WalkParityVerdict:
    relation: str            # "equivalent" | "divergent"
    max_abs_err: float
    detail: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.relation == "equivalent"


def verify_walk_parity(
    model: EncoderFreeVLM, pixels, positions, audio, token_ids, *, tol: float = 1e-6,
) -> WalkParityVerdict:
    """Oracle: running the named walks in sequence equals the monolithic forward.

    A miscompiled walk partition — a dropped modality, a mis-ordered splice —
    diverges here. The same semantics-preserving invariant as Workstreams A/B.
    """
    walks = model.walks()
    v = walks["vision_prefill"](pixels, positions)
    a = walks["audio_prefill"](audio)
    t = walks["text_decode"](token_ids)
    walked = walks["splice"](v, a, t)
    mono = model.forward(pixels, positions, audio, token_ids)

    max_abs_err = float(np.max(np.abs(np.asarray(walked) - np.asarray(mono))))
    rel = "equivalent" if max_abs_err <= tol else "divergent"
    detail = (f"walks ≡ monolith (max_abs_err={max_abs_err:.2e})"
              if rel == "equivalent" else
              f"walk decomposition diverges (max_abs_err={max_abs_err:.2e} > {tol:.0e})")
    return WalkParityVerdict(rel, max_abs_err, detail)


__all__ = [
    "ModelWalk",
    "partition_walks",
    "walks_reconstruct_graph",
    "patch_embed",
    "coordinate_lookup",
    "audio_frame_projection",
    "splice_embeddings",
    "EncoderFreeWeights",
    "EncoderFreeVLM",
    "verify_walk_parity",
    "WalkParityVerdict",
]
