"""ROCm structured-compute runtime lane."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses, memory, ops
from tessera.compiler import diffusion_schedule as D
from tessera.nn import functional as F


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_structured_compute_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
    })


def _launch(rt, op_name, names, args, kwargs=None):
    res = rt.launch(_artifact(rt, op_name, names, kwargs), args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_structured_compute_compiled"
    return res["output"]


def test_rocm_structured_loss_and_vision_layout_match_reference_on_gpu():
    rt = _rocm_or_skip()
    log_probs = np.log(np.array([
        [[0.6, 0.2, 0.2]],
        [[0.1, 0.7, 0.2]],
        [[0.1, 0.2, 0.7]],
    ], np.float32))
    targets = np.array([[1, 2]], np.int64)
    ilen = np.array([3], np.int64)
    tlen = np.array([2], np.int64)
    out = _launch(rt, "tessera.ctc_loss", ("lp", "targets", "ilen", "tlen"),
                  (log_probs, targets, ilen, tlen), {"blank": 0})
    np.testing.assert_allclose(out, losses.ctc_loss(log_probs, targets, ilen, tlen), atol=1e-6)

    x = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    np.testing.assert_allclose(
        _launch(rt, "tessera.center_crop", ("x",), (x,), {"size": (2, 3)}),
        ops.center_crop(x, size=(2, 3)),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.image_resize", ("x",), (x,), {"size": (3, 4)}),
        ops.image_resize(x, size=(3, 4)),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.interpolate", ("x",), (x,), {"scale_factor": 1.5}),
        ops.interpolate(x, scale_factor=1.5),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.pixel_unshuffle", ("x",), (x[:, :, :4, :4],),
                {"downscale_factor": 2}),
        ops.pixel_unshuffle(x[:, :, :4, :4], downscale_factor=2),
    )


def test_rocm_structured_model_recurrent_and_stencil_match_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(73)
    x = rng.standard_normal((2, 3, 7)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.conv1d", ("x", "w", "b"), (x, w, b), {"padding": 1}),
        F.conv1d(x, w, b, padding=1),
        atol=1e-6,
    )

    xt = rng.standard_normal((2, 3)).astype(np.float32)
    h = rng.standard_normal((2, 5)).astype(np.float32)
    Wih3 = rng.standard_normal((3, 15)).astype(np.float32)
    Whh3 = rng.standard_normal((5, 15)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.gru_cell", ("x", "h", "Wih", "Whh"),
                (xt, h, Wih3, Whh3)),
        F.gru_cell(xt, h, Wih3, Whh3),
        atol=1e-6,
    )

    # LSTM cell (gate order i,f,g,o): W_ih/W_hh are (in,4H)/(H,4H); returns
    # (h_new, c_new). Runs through the same structured-compute lane.
    cprev = rng.standard_normal((2, 5)).astype(np.float32)
    Wih4 = rng.standard_normal((3, 20)).astype(np.float32)
    Whh4 = rng.standard_normal((5, 20)).astype(np.float32)
    lh, lc = _launch(rt, "tessera.lstm_cell", ("x", "h", "c", "Wih", "Whh"),
                     (xt, h, cprev, Wih4, Whh4))
    rh, rc = F.lstm_cell(xt, h, cprev, Wih4, Whh4)
    np.testing.assert_allclose(lh, rh, atol=1e-6)
    np.testing.assert_allclose(lc, rc, atol=1e-6)

    a = rng.standard_normal((2, 4)).astype(np.float32)
    weight = rng.standard_normal((4, 6)).astype(np.float32)
    la = rng.standard_normal((4, 2)).astype(np.float32)
    lb = rng.standard_normal((2, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.lora_linear", ("x", "w", "a", "b"),
                (a, weight, la, lb), {"alpha": 2.0}),
        F.lora_linear(a, weight, la, lb, alpha=2.0),
        atol=1e-6,
    )

    dw = rng.standard_normal((3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.depthwise_conv1d", ("x", "w"), (x, dw),
                {"kernel_size": 3, "padding": 1}),
        ops.depthwise_conv1d(x, dw, kernel_size=3, padding=1),
        atol=1e-6,
    )


def test_rocm_structured_attention_and_scan_match_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(79)
    q = rng.standard_normal((2, 3, 4)).astype(np.float32)
    k = rng.standard_normal((2, 5, 4)).astype(np.float32)
    v = rng.standard_normal((2, 5, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.cross_attention", ("q", "k", "v"), (q, k, v)),
        ops.cross_attention(q, k, v),
        atol=1e-6,
    )

    latents = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = rng.standard_normal((2, 5, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.perceiver_resampler", ("latents", "x"), (latents, x)),
        ops.perceiver_resampler(latents, x),
        atol=1e-6,
    )

    def step(carry, item):
        return carry + item

    xs = rng.standard_normal((4, 3)).astype(np.float32)
    init = np.zeros((3,), np.float32)
    got_fwd, got_bwd = _launch(
        rt,
        "tessera.bidirectional_scan",
        ("fn", "init_fwd", "init_bwd", "xs"),
        (step, init, init, xs),
    )
    exp_fwd, exp_bwd = F.bidirectional_scan(step, init, init, xs)
    np.testing.assert_allclose(got_fwd, exp_fwd, atol=1e-6)
    np.testing.assert_allclose(got_bwd, exp_bwd, atol=1e-6)

    q4 = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    scores = rng.standard_normal((1, 2, 4, 2)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.attn_compressed_blocks", ("q", "k", "v"),
                (q4, q4, q4)),
        ops.attn_compressed_blocks(q4, q4, q4),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.attn_top_k_blocks", ("q", "k", "v", "scores"),
                (q4, q4, q4, scores), {"top_k": 1, "block_size": 2,
                                        "causal": True}),
        ops.attn_top_k_blocks(q4, q4, q4, scores=scores, top_k=1,
                              block_size=2, causal=True),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.lookahead_sparse_attention", ("q", "k", "v"),
                (q4, q4, q4), {"window_size": 2, "block_size": 2,
                               "causal": True}),
        ops.lookahead_sparse_attention(q4, q4, q4, window_size=2,
                                       block_size=2, causal=True),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.power_attn", ("q", "k", "v"), (q4, q4, q4),
                {"deg": 2, "causal": True}),
        ops.power_attn(q4, q4, q4, deg=2, causal=True),
        atol=1e-6,
    )


def test_rocm_structured_layout_tail_matches_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(83)
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    mask = x > 0.0
    np.testing.assert_array_equal(
        _launch(rt, "tessera.arange", ("start", "stop"), (1, 7), {"dtype": "i32"}),
        ops.arange(1, 7, dtype="i32"),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.masked_fill", ("x", "mask"), (x, mask), {"value": -3.0}),
        ops.masked_fill(x, mask, value=-3.0),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.transpose", ("x",), (x,), {"axes": (0, 2, 1)}),
        ops.transpose(x, axes=(0, 2, 1)),
    )
    np.testing.assert_allclose(
        _launch(rt, "tessera.rearrange", ("x", "layout"), (x, (0, 2, 1))),
        ops.rearrange(x, (0, 2, 1)),
    )
    rope, rest = _launch(rt, "tessera.rope_split", ("x",), (x,), {"rope_dim": 2})
    np.testing.assert_allclose(
        _launch(rt, "tessera.rope_merge", ("rope", "rest"), (rope, rest)),
        ops.rope_merge(*ops.rope_split(x, rope_dim=2)),
    )

    hidden = rng.standard_normal((2, 4, 3)).astype(np.float32)
    w_router = rng.standard_normal((3, 3)).astype(np.float32)
    depth = _launch(rt, "tessera.mor_router", ("x", "w"), (hidden, w_router), {"max_depth": 3})
    part = _launch(rt, "tessera.mor_partition", ("x", "depth"), (hidden, depth), {"step": 2})
    np.testing.assert_allclose(
        _launch(rt, "tessera.mor_scatter", ("full", "updated", "mask"), (hidden, hidden + 1.0, part)),
        ops.mor_scatter(hidden, hidden + 1.0, part),
    )


def test_rocm_structured_helper_tail_matches_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(87)
    sigma = np.array([0.2, 0.5, 1.0], np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.edm_loss_weight", ("sigma",), (sigma,)),
        D.edm_loss_weight(sigma),
    )
    got_scalings = _launch(rt, "tessera.edm_precondition", ("sigma",), (sigma,))
    exp_scalings = D.edm_precondition(sigma)
    for field in ("c_skip", "c_out", "c_in", "c_noise"):
        np.testing.assert_allclose(getattr(got_scalings, field), getattr(exp_scalings, field))

    row = rng.standard_normal((4, 6)).astype(np.float32)
    col = rng.standard_normal((5, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.factorized_pos_emb", ("row", "col"), (row, col),
                {"grid_h": 3, "grid_w": 4}),
        ops.factorized_pos_emb(row, col, grid_h=3, grid_w=4),
    )

    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    mask = np.array([[True, False, True], [False, True, False]])
    source = rng.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.masked_scatter", ("x", "mask", "source"), (x, mask, source)),
        ops.masked_scatter(x, mask, source),
    )

    table = memory.MemoryTable(
        keys=np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], np.float32),
        values=np.array([[10.0, 0.0], [0.0, 20.0], [8.0, 2.0]], np.float32),
    )
    got_mem = _launch(rt, "tessera.memory_read", ("table", "query"),
                      (table, np.array([1.0, 0.0], np.float32)), {"top_k": 2})
    exp_mem = memory.memory_read(table, np.array([1.0, 0.0], np.float32), top_k=2)
    np.testing.assert_allclose(got_mem.values, exp_mem.values)
    np.testing.assert_array_equal(got_mem.indices, exp_mem.indices)

    rope_x = rng.standard_normal((2, 4)).astype(np.float32)
    positions = np.array([[0, 1]], np.float32)
    inv_freq = np.array([1.0, 0.5], np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.mrope_2d", ("x", "positions", "inv_freq"),
                (rope_x, positions, inv_freq), {"sections": (2,)}),
        ops.mrope_2d(rope_x, positions, inv_freq, sections=(2,)),
    )

    logits = rng.standard_normal((2, 4)).astype(np.float32)
    got_state = _launch(rt, "tessera.online_softmax_state", ("x",), (logits,), {"axis": -1})
    exp_state = ops.online_softmax_state(logits, axis=-1)
    np.testing.assert_allclose(got_state[0], exp_state[0])
    np.testing.assert_allclose(got_state[1], exp_state[1])

    weight = rng.standard_normal((3, 2, 2)).astype(np.float32)
    np.testing.assert_allclose(
        _launch(rt, "tessera.spectral_norm", ("weight",), (weight,)),
        F.spectral_norm(weight),
    )
