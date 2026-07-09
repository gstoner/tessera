"""Apple GPU structured-compute lane — the convolution family.

Parity with ``test_x86_structured_compute_compiled`` / the ROCm lane: conv1d,
conv_transpose, and depthwise_conv1d reach an executable ``apple_gpu`` path via
``runtime.launch()`` and match the reference primitive. Host-structured
im2col/layout bookkeeping; direct execute/compare evidence, not a bespoke fused
Metal kernel.
"""

from __future__ import annotations

import numpy as np

from tessera import losses, ops
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
