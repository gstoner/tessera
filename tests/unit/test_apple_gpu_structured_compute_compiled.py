"""Apple GPU structured-compute lane — the convolution family.

Parity with ``test_x86_structured_compute_compiled`` / the ROCm lane: conv1d,
conv_transpose, and depthwise_conv1d reach an executable ``apple_gpu`` path via
``runtime.launch()`` and match the reference primitive. Host-structured
im2col/layout bookkeeping; direct execute/compare evidence, not a bespoke fused
Metal kernel.
"""

from __future__ import annotations

import numpy as np

from tessera import losses, memory, ops
from tessera import runtime as rt
from tessera.compiler import diffusion_schedule as D
from tessera.nn import functional as F


def _artifact(op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_gpu_structured_compute_compiled",
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


def _launch(op_name, names, args, kwargs=None):
    res = rt.launch(_artifact(op_name, names, kwargs), args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_structured_compute_compiled"
    assert res["execution_kind"] == "native_gpu"
    return res["output"]


def test_apple_gpu_conv1d_matches_reference():
    rng = np.random.default_rng(0x9091)
    # conv1d: x [N, C_in, L], weight [C_out, C_in/groups, K]
    x = rng.standard_normal((2, 3, 7)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.conv1d", ("x", "w", "b"), (x, w, b),
                {"padding": 1}),
        F.conv1d(x, w, b, padding=1),
        atol=1e-6,
    )
    # strided + dilated + grouped variant
    xg = rng.standard_normal((1, 4, 9)).astype(np.float32)
    wg = rng.standard_normal((6, 2, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.conv1d", ("x", "w"), (xg, wg),
                {"stride": 2, "padding": 2, "dilation": 2, "groups": 2}),
        F.conv1d(xg, wg, stride=2, padding=2, dilation=2, groups=2),
        atol=1e-6,
    )


def test_apple_gpu_conv_transpose_matches_reference():
    rng = np.random.default_rng(0x9092)
    # conv_transpose: x [N, C_in, L], weight [C_in, C_out/groups, K]
    x = rng.standard_normal((2, 3, 5)).astype(np.float32)
    w = rng.standard_normal((3, 4, 3)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.conv_transpose", ("x", "w", "b"), (x, w, b),
                {"stride": 2, "padding": 1, "output_padding": 1}),
        F.conv_transpose(x, w, b, stride=2, padding=1, output_padding=1),
        atol=1e-6,
    )


def test_apple_gpu_depthwise_conv1d_matches_reference():
    rng = np.random.default_rng(0x9093)
    # depthwise_conv1d: x [N, C, L], weight [C, K]
    x = rng.standard_normal((2, 3, 7)).astype(np.float32)
    dw = rng.standard_normal((3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.depthwise_conv1d", ("x", "w"), (x, dw),
                {"kernel_size": 3, "padding": 1}),
        ops.depthwise_conv1d(x, dw, kernel_size=3, padding=1),
        atol=1e-6,
    )
    # causal variant
    np.testing.assert_allclose(
        _launch("tessera.depthwise_conv1d", ("x", "w"), (x, dw),
                {"kernel_size": 3, "causal": True}),
        ops.depthwise_conv1d(x, dw, kernel_size=3, causal=True),
        atol=1e-6,
    )


def test_apple_gpu_ctc_loss_matches_reference():
    # ctc_loss rides the structured-compute lane (in _SINGLE_GPU_COMPUTE_REFERENCE_OPS)
    log_probs = np.log(np.array([
        [[0.6, 0.2, 0.2]],
        [[0.1, 0.7, 0.2]],
        [[0.1, 0.2, 0.7]],
    ], np.float32))
    targets = np.array([[1, 2]], np.int64)
    ilen = np.array([3], np.int64)
    tlen = np.array([2], np.int64)
    np.testing.assert_allclose(
        _launch("tessera.ctc_loss", ("lp", "t", "il", "tl"),
                (log_probs, targets, ilen, tlen), {"blank": 0}),
        losses.ctc_loss(log_probs, targets, ilen, tlen),
        atol=1e-6,
    )


def test_apple_gpu_edm_loss_weight_matches_reference():
    sigma = np.array([0.2, 0.5, 1.0], np.float32)
    np.testing.assert_allclose(
        _launch("tessera.edm_loss_weight", ("sigma",), (sigma,)),
        D.edm_loss_weight(sigma),
    )


# ── Structured-compute tail (2026-07-09): vision/layout, recurrent, MoR, VLM,
#    RoPE, and other host-structured primitives on the apple_gpu structured lane.

def test_apple_gpu_vision_layout_ops_match_reference():
    rng = np.random.default_rng(0xC01)
    x = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    np.testing.assert_allclose(
        _launch("tessera.center_crop", ("x",), (x,), {"size": (2, 3)}),
        ops.center_crop(x, size=(2, 3)))
    np.testing.assert_allclose(
        _launch("tessera.image_resize", ("x",), (x,), {"size": (3, 4)}),
        ops.image_resize(x, size=(3, 4)))
    np.testing.assert_allclose(
        _launch("tessera.interpolate", ("x",), (x,), {"scale_factor": 1.5}),
        ops.interpolate(x, scale_factor=1.5))
    np.testing.assert_allclose(
        _launch("tessera.patchify", ("x",), (x[:, :, :4, :4],), {"patch_size": 2}),
        ops.patchify(x[:, :, :4, :4], patch_size=2))
    ps = rng.standard_normal((1, 8, 3, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.pixel_shuffle", ("x",), (ps,), {"upscale_factor": 2}),
        ops.pixel_shuffle(ps, upscale_factor=2))
    pu = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.pixel_unshuffle", ("x",), (pu,), {"downscale_factor": 2}),
        ops.pixel_unshuffle(pu, downscale_factor=2))


def test_apple_gpu_recurrent_and_lora_match_reference():
    rng = np.random.default_rng(0xC02)
    xt = rng.standard_normal((2, 3)).astype(np.float32)
    h = rng.standard_normal((2, 5)).astype(np.float32)
    Wih = rng.standard_normal((3, 5)).astype(np.float32)
    Whh = rng.standard_normal((5, 5)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.simple_rnn_cell", ("x", "h", "Wih", "Whh"),
                (xt, h, Wih, Whh)),
        F.simple_rnn_cell(xt, h, Wih, Whh), atol=1e-6)
    Wih3 = rng.standard_normal((3, 15)).astype(np.float32)
    Whh3 = rng.standard_normal((5, 15)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.gru_cell", ("x", "h", "Wih", "Whh"), (xt, h, Wih3, Whh3)),
        F.gru_cell(xt, h, Wih3, Whh3), atol=1e-6)
    a = rng.standard_normal((2, 4)).astype(np.float32)
    weight = rng.standard_normal((4, 6)).astype(np.float32)
    la = rng.standard_normal((4, 2)).astype(np.float32)
    lb = rng.standard_normal((2, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.lora_linear", ("x", "w", "a", "b"),
                (a, weight, la, lb), {"alpha": 2.0}),
        F.lora_linear(a, weight, la, lb, alpha=2.0), atol=1e-6)


def test_apple_gpu_attention_resampler_scan_match_reference():
    rng = np.random.default_rng(0xC03)
    q = rng.standard_normal((2, 3, 4)).astype(np.float32)
    k = rng.standard_normal((2, 5, 4)).astype(np.float32)
    v = rng.standard_normal((2, 5, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.cross_attention", ("q", "k", "v"), (q, k, v)),
        ops.cross_attention(q, k, v), atol=1e-6)
    latents = rng.standard_normal((2, 3, 4)).astype(np.float32)
    xr = rng.standard_normal((2, 5, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.perceiver_resampler", ("latents", "x"), (latents, xr)),
        ops.perceiver_resampler(latents, xr), atol=1e-6)

    def step(carry, item):
        return carry + item

    xs = rng.standard_normal((4, 3)).astype(np.float32)
    init = np.zeros((3,), np.float32)
    got_fwd, got_bwd = _launch(
        "tessera.bidirectional_scan", ("fn", "init_fwd", "init_bwd", "xs"),
        (step, init, init, xs))
    exp_fwd, exp_bwd = F.bidirectional_scan(step, init, init, xs)
    np.testing.assert_allclose(got_fwd, exp_fwd, atol=1e-6)
    np.testing.assert_allclose(got_bwd, exp_bwd, atol=1e-6)


def test_apple_gpu_layout_and_mor_tail_match_reference():
    rng = np.random.default_rng(0xC04)
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    mask = x > 0.0
    np.testing.assert_array_equal(
        _launch("tessera.arange", ("start", "stop"), (1, 7), {"dtype": "i32"}),
        ops.arange(1, 7, dtype="i32"))
    np.testing.assert_allclose(
        _launch("tessera.masked_fill", ("x", "mask"), (x, mask), {"value": -3.0}),
        ops.masked_fill(x, mask, value=-3.0))
    np.testing.assert_allclose(
        _launch("tessera.rearrange", ("x", "layout"), (x, (0, 2, 1))),
        ops.rearrange(x, (0, 2, 1)))
    np.testing.assert_allclose(
        _launch("tessera.pack", ("x", "layout"), (x, (0, 2, 1))),
        ops.pack(x, (0, 2, 1)))
    np.testing.assert_allclose(
        _launch("tessera.unpack", ("x",), (x,)), ops.unpack(x))
    np.testing.assert_allclose(
        _launch("tessera.tile_view", ("x",), (x,), {"BM": 2, "BN": 2}),
        ops.tile_view(x, BM=2, BN=2))
    rope, rest = _launch("tessera.rope_split", ("x",), (x,), {"rope_dim": 2})
    exp_rope, exp_rest = ops.rope_split(x, rope_dim=2)
    np.testing.assert_allclose(rope, exp_rope)
    np.testing.assert_allclose(rest, exp_rest)
    np.testing.assert_allclose(
        _launch("tessera.rope_merge", ("rope", "rest"), (rope, rest)),
        ops.rope_merge(rope, rest))
    hidden = rng.standard_normal((2, 4, 3)).astype(np.float32)
    w_router = rng.standard_normal((3, 3)).astype(np.float32)
    depth = _launch("tessera.mor_router", ("x", "w"), (hidden, w_router),
                    {"max_depth": 3})
    np.testing.assert_array_equal(depth, ops.mor_router(hidden, w_router, max_depth=3))
    part = _launch("tessera.mor_partition", ("x", "depth"), (hidden, depth),
                   {"step": 2})
    np.testing.assert_array_equal(part, ops.mor_partition(hidden, depth, step=2))
    updated = hidden + 1.0
    np.testing.assert_allclose(
        _launch("tessera.mor_scatter", ("full", "updated", "mask"),
                (hidden, updated, part)),
        ops.mor_scatter(hidden, updated, part))


def test_apple_gpu_helper_tail_match_reference():
    rng = np.random.default_rng(0xC05)
    sigma = np.array([0.2, 0.5, 1.0], np.float32)
    got_scalings = _launch("tessera.edm_precondition", ("sigma",), (sigma,))
    exp_scalings = D.edm_precondition(sigma)
    for field in ("c_skip", "c_out", "c_in", "c_noise"):
        np.testing.assert_allclose(getattr(got_scalings, field),
                                   getattr(exp_scalings, field))
    row = rng.standard_normal((4, 6)).astype(np.float32)
    col = rng.standard_normal((5, 6)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.factorized_pos_emb", ("row", "col"), (row, col),
                {"grid_h": 3, "grid_w": 4}),
        ops.factorized_pos_emb(row, col, grid_h=3, grid_w=4))
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    mask = np.array([[True, False, True], [False, True, False]])
    source = rng.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.masked_scatter", ("x", "mask", "source"),
                (x, mask, source)),
        ops.masked_scatter(x, mask, source))
    table = memory.MemoryTable(
        keys=np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], np.float32),
        values=np.array([[10.0, 0.0], [0.0, 20.0], [8.0, 2.0]], np.float32))
    got_mem = _launch("tessera.memory_read", ("table", "query"),
                      (table, np.array([1.0, 0.0], np.float32)), {"top_k": 2})
    exp_mem = memory.memory_read(table, np.array([1.0, 0.0], np.float32), top_k=2)
    np.testing.assert_allclose(got_mem.values, exp_mem.values)
    np.testing.assert_array_equal(got_mem.indices, exp_mem.indices)
    rope_x = rng.standard_normal((2, 4)).astype(np.float32)
    positions = np.array([[0, 1]], np.float32)
    inv_freq = np.array([1.0, 0.5], np.float32)
    np.testing.assert_allclose(
        _launch("tessera.mrope_2d", ("x", "positions", "inv_freq"),
                (rope_x, positions, inv_freq), {"sections": (2,)}),
        ops.mrope_2d(rope_x, positions, inv_freq, sections=(2,)))
    logits = rng.standard_normal((2, 4)).astype(np.float32)
    got_state = _launch("tessera.online_softmax_state", ("x",), (logits,),
                        {"axis": -1})
    exp_state = ops.online_softmax_state(logits, axis=-1)
    np.testing.assert_allclose(got_state[0], exp_state[0])
    np.testing.assert_allclose(got_state[1], exp_state[1])
    weight = rng.standard_normal((3, 2, 2)).astype(np.float32)
    np.testing.assert_allclose(
        _launch("tessera.spectral_norm", ("weight",), (weight,)),
        F.spectral_norm(weight))
