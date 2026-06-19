"""Workstream F — named multimodal walks + encoder-free ops.

Proves: (1) the MiniMax-M3 multimodal graph partitions losslessly into named
walks (vision_prefill / video_prefill / splice); (2) the first-class encoder-free
ops (patch/coordinate/audio projection) compose into a forward that equals running
the named walks in sequence.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream F).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.model_walk import (
    ModelWalk, partition_walks, walks_reconstruct_graph,
    patch_embed, coordinate_lookup, audio_frame_projection, splice_embeddings,
    EncoderFreeWeights, EncoderFreeVLM, verify_walk_parity)


# ── named walks over the real MiniMax-M3 graph ───────────────────────────────


def _graph(**kw):
    from tessera.models.minimax_m3 import build_multimodal_graph
    return build_multimodal_graph(**kw)


def test_partition_yields_named_walks():
    walks = partition_walks(_graph(include_image=True, include_video=True))
    assert "vision_prefill" in walks
    assert "video_prefill" in walks
    assert "splice" in walks
    assert all(isinstance(w, ModelWalk) for w in walks.values())


def test_partition_is_lossless():
    g = _graph(include_image=True, include_video=True)
    walks = partition_walks(g)
    assert walks_reconstruct_graph(g, walks)


def test_vision_walk_has_the_media_chain():
    walks = partition_walks(_graph(include_image=True, include_video=False))
    ops = walks["vision_prefill"].op_sequence
    assert "patch_embed" in ops and "patch_merge" in ops and "media_project" in ops


def test_splice_walk_consumes_media_and_text():
    walks = partition_walks(_graph(include_image=True, include_video=False))
    splice = walks["splice"]
    assert "text_embeddings" in splice.consumes
    assert any("projected" in c for c in splice.consumes)


def test_image_only_graph_has_no_video_walk():
    walks = partition_walks(_graph(include_image=True, include_video=False))
    assert "video_prefill" not in walks


# ── first-class encoder-free ops ─────────────────────────────────────────────


def test_patch_embed_is_raw_projection():
    rng = np.random.default_rng(0)
    pixels = rng.standard_normal((9, 12)).astype(np.float32)   # 9 patches × patch_dim
    W = rng.standard_normal((12, 16)).astype(np.float32)
    np.testing.assert_allclose(patch_embed(pixels, W), pixels @ W, rtol=1e-6)


def test_coordinate_lookup_gathers_rows():
    table = np.arange(20).reshape(5, 4).astype(np.float32)
    out = coordinate_lookup([0, 2, 4], table)
    np.testing.assert_array_equal(out, table[[0, 2, 4]])


def test_audio_frame_projection_shape():
    frames = np.zeros((7, 8), np.float32)
    W = np.zeros((8, 16), np.float32)
    assert audio_frame_projection(frames, W).shape == (7, 16)


def test_splice_concatenates_streams():
    a = np.ones((2, 4), np.float32)
    b = np.zeros((3, 4), np.float32)
    out = splice_embeddings(a, b)
    assert out.shape == (5, 4)


# ── walk decomposition ≡ monolithic forward (the F oracle) ───────────────────


def _vlm(D=8, seed=0):
    rng = np.random.default_rng(seed)
    w = EncoderFreeWeights(
        patch_w=rng.standard_normal((12, D)).astype(np.float32),
        pos_table=rng.standard_normal((64, D)).astype(np.float32),
        audio_w=rng.standard_normal((10, D)).astype(np.float32),
        text_table=rng.standard_normal((100, D)).astype(np.float32),
    )
    return EncoderFreeVLM(w)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_walk_parity(seed):
    model = _vlm(seed=seed)
    rng = np.random.default_rng(seed + 10)
    pixels = rng.standard_normal((9, 12)).astype(np.float32)
    positions = np.arange(9)
    audio = rng.standard_normal((5, 10)).astype(np.float32)
    token_ids = rng.integers(0, 100, size=7)

    verdict = verify_walk_parity(model, pixels, positions, audio, token_ids)
    assert verdict.is_equivalent, verdict.detail
    assert verdict.max_abs_err < 1e-6


def test_walks_are_named_entry_points():
    model = _vlm()
    walks = model.walks()
    assert set(walks) == {"vision_prefill", "audio_prefill", "text_decode", "splice"}
    assert all(callable(fn) for fn in walks.values())
