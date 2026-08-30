"""`nn.functional` conv1d / conv_transpose — vectorized against the loop oracle.

Both reference convolutions were six-deep Python scalar loops (~N*C_out*L_out*
C_in/g*K interpreter-level multiply-adds per call). They now build windows with
`sliding_window_view` and contract with `einsum`; `conv_transpose` scatters a
single `einsum` contribution tensor into an uncropped buffer, one vectorized
strided add per kernel tap.

The oracles below are the exact loop nests that were replaced, kept here rather
than in the module: they are what makes the rewrite checkable, and they are far
too slow to ship on the call path (code review 2026-08-29, P3).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.nn import functional as F


def _conv1d_loop_oracle(x, w, bias=None, *, stride=1, padding=0, dilation=1,
                        groups=1):
    n, c_in, length = x.shape
    c_out, _, kernel = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out_len = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    out = np.zeros((n, c_out, out_len), dtype=np.float32)
    opg, ipg = c_out // groups, c_in // groups
    for b in range(n):
        for g in range(groups):
            in_base, out_base = g * ipg, g * opg
            for oc in range(opg):
                for pos in range(out_len):
                    acc, start = 0.0, pos * stride
                    for ic in range(ipg):
                        for k in range(kernel):
                            acc += (padded[b, in_base + ic, start + k * dilation]
                                    * w[out_base + oc, ic, k])
                    out[b, out_base + oc, pos] = acc
    if bias is not None:
        out += np.asarray(bias).reshape(1, c_out, 1)
    return out


def _conv_transpose_loop_oracle(x, w, bias=None, *, stride=1, padding=0,
                                output_padding=0, dilation=1, groups=1):
    n, c_in, length = x.shape
    _, copg, kernel = w.shape
    c_out = copg * groups
    out_len = ((length - 1) * stride - 2 * padding + dilation * (kernel - 1)
               + output_padding + 1)
    out = np.zeros((n, c_out, out_len), dtype=np.float32)
    ipg = c_in // groups
    for b in range(n):
        for g in range(groups):
            in_base, out_base = g * ipg, g * copg
            for ic in range(ipg):
                for pos in range(length):
                    for k in range(kernel):
                        op = pos * stride - padding + k * dilation
                        if 0 <= op < out_len:
                            out[b, out_base:out_base + copg, op] += (
                                x[b, in_base + ic, pos] * w[in_base + ic, :, k])
    if bias is not None:
        out += np.asarray(bias).reshape(1, c_out, 1)
    return out


#: Named corners rather than the full cross product: each entry exercises one
#: index-arithmetic hazard of the window/scatter rewrite. The whole 324/594-case
#: cross product was swept once during the port and agreed to 3.8e-06.
_CONV1D_CASES = [
    ("plain", dict(kernel=3)),
    ("stride", dict(kernel=3, stride=2)),
    ("padded", dict(kernel=3, padding=2)),
    ("dilated", dict(kernel=3, dilation=2)),
    ("grouped", dict(kernel=3, groups=4)),
    ("k1", dict(kernel=1, stride=3)),
    ("stride_gt_kernel", dict(kernel=3, stride=3, padding=1)),
    ("all_axes", dict(kernel=4, stride=2, padding=2, dilation=2, groups=2)),
]
_CONVT_CASES = [
    ("plain", dict(kernel=3)),
    ("upsample", dict(kernel=3, stride=2)),
    ("cropped", dict(kernel=3, stride=2, padding=1)),
    ("output_padding", dict(kernel=3, stride=2, output_padding=1)),
    ("dilated", dict(kernel=3, dilation=2)),
    ("grouped", dict(kernel=3, stride=2, groups=4)),
    ("k1", dict(kernel=1, stride=3)),
    ("all_axes", dict(kernel=4, stride=3, padding=2, dilation=2, groups=2,
                      output_padding=1)),
]


@pytest.mark.parametrize("name,kw", _CONV1D_CASES, ids=[c[0] for c in _CONV1D_CASES])
@pytest.mark.parametrize("use_bias", [False, True])
def test_conv1d_matches_the_scalar_loop_oracle(name, kw, use_bias):
    rng = np.random.default_rng(20260829)
    groups, kernel = kw.get("groups", 1), kw["kernel"]
    n, c_in, c_out, length = 2, 8, 8, 17
    x = rng.standard_normal((n, c_in, length)).astype(np.float32)
    w = rng.standard_normal((c_out, c_in // groups, kernel)).astype(np.float32)
    bias = rng.standard_normal(c_out).astype(np.float32) if use_bias else None
    call = {k: v for k, v in kw.items() if k != "kernel"}
    got = F.conv1d(x, w, bias, **call)
    expect = _conv1d_loop_oracle(x, w, bias, **call)
    assert got.shape == expect.shape and got.dtype == expect.dtype
    # fp32 accumulation ORDER differs (einsum vs left-to-right), so this is a
    # tolerance comparison, not a bit-exactness one.
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,kw", _CONVT_CASES, ids=[c[0] for c in _CONVT_CASES])
@pytest.mark.parametrize("use_bias", [False, True])
def test_conv_transpose_matches_the_scalar_loop_oracle(name, kw, use_bias):
    rng = np.random.default_rng(20260829)
    groups, kernel = kw.get("groups", 1), kw["kernel"]
    n, c_in, length, copg = 2, 8, 11, 3
    x = rng.standard_normal((n, c_in, length)).astype(np.float32)
    w = rng.standard_normal((c_in, copg, kernel)).astype(np.float32)
    bias = (rng.standard_normal(copg * groups).astype(np.float32)
            if use_bias else None)
    call = {k: v for k, v in kw.items() if k != "kernel"}
    got = F.conv_transpose(x, w, bias, **call)
    expect = _conv_transpose_loop_oracle(x, w, bias, **call)
    assert got.shape == expect.shape and got.dtype == expect.dtype
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)


def test_conv1d_still_validates_shapes_and_groups_before_contracting():
    """The validation used to sit above the loop nest; moving to einsum must not
    let a bad group split reach numpy as a reshape error instead."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 8, 16)).astype(np.float32)
    with pytest.raises(ValueError, match="x .N,C,L. and weight"):
        F.conv1d(x[0], rng.standard_normal((4, 8, 3)).astype(np.float32))
    with pytest.raises(ValueError, match="groups must divide"):
        F.conv1d(x, rng.standard_normal((4, 8, 3)).astype(np.float32), groups=3)
    with pytest.raises(ValueError, match="weight input channels must equal"):
        F.conv1d(x, rng.standard_normal((4, 8, 3)).astype(np.float32), groups=2)
    with pytest.raises(ValueError, match="output length must be positive"):
        F.conv1d(x, rng.standard_normal((4, 8, 32)).astype(np.float32))


@pytest.mark.performance
def test_conv1d_is_fast_enough_to_be_a_usable_reference():
    """A shape the loop nest took ~6 s on. The bound is loose on purpose — it
    pins that this is a vectorized contraction, not the interpreter."""
    import time

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 64, 256)).astype(np.float32)
    w = rng.standard_normal((64, 64, 3)).astype(np.float32)
    F.conv1d(x, w)                       # warm numpy/einsum path selection
    t0 = time.perf_counter()
    out = F.conv1d(x, w)
    elapsed = time.perf_counter() - t0
    assert out.shape == (8, 64, 254)
    assert elapsed < 1.0, f"conv1d took {elapsed:.3f}s for a 1.6e8-MAC shape"
