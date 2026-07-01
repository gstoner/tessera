"""VLM connector-pack primitives — forward numerics + autodiff coverage.

The vision-language glue layer (Decision #24 audit truth lives in
`primitive_coverage.py`; gap write-up in
`docs/audit/coverage/COVERAGE_AUDIT.md`).  Each primitive ships a numpy
reference on `tessera.ops.*` plus registered VJP/JVP; these tests pin the
forward semantics and check the analytic (V/J)VP against central finite
differences at fp64.

Landed so far:
  * masked_scatter — modality fusion (P0)
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp


# ── finite-difference helpers ────────────────────────────────────────────────


def _numeric_grad(fn, x, eps=1e-5):
    """Central-difference gradient of ``sum(fn(x))`` w.r.t. x."""
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = float(np.asarray(fn(x)).sum())
        x[idx] = orig - eps
        f_minus = float(np.asarray(fn(x)).sum())
        x[idx] = orig
        g[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def _numeric_jvp(fn, x, dx, eps=1e-6):
    """Directional derivative of fn at x along dx, central difference."""
    plus = np.asarray(fn(x + eps * dx), dtype=np.float64)
    minus = np.asarray(fn(x - eps * dx), dtype=np.float64)
    return (plus - minus) / (2 * eps)


# ════════════════════════════════════════════════════════════════════════════
# masked_scatter — VLM modality fusion (P0)
# ════════════════════════════════════════════════════════════════════════════


def test_masked_scatter_is_registered():
    assert hasattr(tessera.ops, "masked_scatter")
    assert get_vjp("masked_scatter") is not None
    assert get_jvp("masked_scatter") is not None


def test_masked_scatter_vlm_fusion_semantics():
    """The (B,S) token mask over an (B,S,D) embedding sequence: each True
    token slot is overwritten by one D-vector row of source, in order."""
    B, S, D, N = 2, 5, 4, 3
    rng = np.random.default_rng(0)
    token_embd = rng.normal(size=(B, S, D))
    # N image-placeholder slots somewhere in the sequence (flattened order).
    mask = np.zeros((B, S), dtype=bool)
    flat_positions = [(0, 1), (0, 2), (1, 0)]  # 3 True slots → 3 source rows
    for b, s in flat_positions:
        mask[b, s] = True
    image_embd = rng.normal(size=(N, D))

    out = np.asarray(
        tessera.ops.masked_scatter(token_embd, mask, image_embd)
    )

    # Manual reference: broadcast mask over D, splice source rows in order.
    expected = token_embd.copy()
    expected[mask] = image_embd
    np.testing.assert_allclose(out, expected)
    # Unmasked slots are untouched.
    np.testing.assert_allclose(out[~mask], token_embd[~mask])


def test_masked_scatter_matches_torch_flatten_semantics():
    """Same-shape mask, source consumed in flattened row-major order."""
    x = np.zeros((2, 3))
    mask = np.array([[True, False, True], [False, True, False]])
    source = np.array([10.0, 20.0, 30.0, 40.0])  # only first 3 consumed
    out = np.asarray(tessera.ops.masked_scatter(x, mask, source))
    np.testing.assert_allclose(out, [[10.0, 0.0, 20.0], [0.0, 30.0, 0.0]])


def test_masked_scatter_source_too_small_raises():
    x = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)  # 4 True positions
    with pytest.raises(ValueError, match="source"):
        tessera.ops.masked_scatter(x, mask, np.array([1.0, 2.0]))


def test_masked_scatter_vjp_matches_numeric():
    B, S, D, N = 2, 4, 3, 4
    rng = np.random.default_rng(7)
    x = rng.normal(size=(B, S, D))
    mask = np.zeros((B, S), dtype=bool)
    for b, s in [(0, 0), (0, 3), (1, 1), (1, 2)]:
        mask[b, s] = True
    source = rng.normal(size=(N, D))

    def fwd_x(x_in):
        return np.asarray(tessera.ops.masked_scatter(x_in, mask, source))

    def fwd_src(src_in):
        return np.asarray(tessera.ops.masked_scatter(x, mask, src_in))

    dout = np.ones((B, S, D))
    dx, dmask, dsource = get_vjp("masked_scatter")(dout, x, mask, source)
    assert dmask is None
    np.testing.assert_allclose(dx, _numeric_grad(fwd_x, x), atol=1e-6)
    np.testing.assert_allclose(dsource, _numeric_grad(fwd_src, source), atol=1e-6)


def test_masked_scatter_tape_end_to_end():
    """Full reverse-mode through the autodiff tape: gradient flows to the base
    embeddings (unmasked slots) and to the spliced source rows."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 3, 4))
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    mask[1, 2] = True
    src = rng.normal(size=(2, 4))

    x_p = tessera.nn.Parameter(x.copy())
    src_p = tessera.nn.Parameter(src.copy())
    with tessera.autodiff.tape() as t:
        out = tessera.ops.masked_scatter(x_p, mask, src_p)
        loss = tessera.ops.reduce(out, op="sum")
        t.backward(loss)

    gx = x_p.grad.numpy()
    gs = src_p.grad.numpy()
    # d(sum)/dx = 1 on unmasked slots, 0 where overwritten; d(sum)/dsrc = 1.
    assert np.allclose(gx[mask], 0.0)
    assert np.allclose(gx[~mask], 1.0)
    assert np.allclose(gs, 1.0)


def test_masked_scatter_jvp_matches_numeric():
    B, S, D, N = 2, 4, 3, 4
    rng = np.random.default_rng(11)
    x = rng.normal(size=(B, S, D))
    mask = np.zeros((B, S), dtype=bool)
    for b, s in [(0, 0), (0, 3), (1, 1), (1, 2)]:
        mask[b, s] = True
    source = rng.normal(size=(N, D))
    dx = rng.normal(size=x.shape)
    dsource = rng.normal(size=source.shape)

    primal_out, tan_out = get_jvp("masked_scatter")(
        (x, mask, source), (dx, None, dsource)
    )
    np.testing.assert_allclose(
        primal_out, np.asarray(tessera.ops.masked_scatter(x, mask, source))
    )

    # Numeric directional derivative perturbing x and source together.
    def fwd(t):
        return np.asarray(
            tessera.ops.masked_scatter(x + t * dx, mask, source + t * dsource)
        )

    eps = 1e-6
    tan_numeric = (fwd(eps) - fwd(-eps)) / (2 * eps)
    np.testing.assert_allclose(tan_out, tan_numeric, atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# image preprocessing pack (P0): image_resize / interpolate / center_crop /
# image_normalize
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ["image_resize", "interpolate", "center_crop", "image_normalize"])
def test_preprocessing_ops_registered(name):
    assert hasattr(tessera.ops, name)
    assert get_vjp(name) is not None
    assert get_jvp(name) is not None


def test_image_resize_shorter_side_preserves_aspect():
    # CHW image 3×8×16 → shorter side (8) resized to 4 → 3×4×8.
    x = np.random.default_rng(0).normal(size=(3, 8, 16))
    out = np.asarray(tessera.ops.image_resize(x, size=4, layout="chw"))
    assert out.shape == (3, 4, 8)


def test_image_resize_to_explicit_size():
    x = np.random.default_rng(0).normal(size=(2, 3, 8, 8))  # NCHW
    out = np.asarray(tessera.ops.image_resize(x, size=(5, 6), layout="nchw"))
    assert out.shape == (2, 3, 5, 6)


def test_image_resize_identity_when_same_size():
    x = np.random.default_rng(1).normal(size=(1, 3, 6, 6))
    out = np.asarray(tessera.ops.image_resize(x, size=(6, 6), layout="nchw"))
    np.testing.assert_allclose(out, x, atol=1e-9)


def test_image_resize_antialias_raises():
    x = np.zeros((1, 3, 8, 8))
    with pytest.raises(NotImplementedError, match="antialias"):
        tessera.ops.image_resize(x, size=4, antialias=True)


def test_interpolate_requires_exactly_one_of_size_scale():
    x = np.zeros((1, 3, 8, 8))
    with pytest.raises(ValueError, match="exactly one"):
        tessera.ops.interpolate(x)
    with pytest.raises(ValueError, match="exactly one"):
        tessera.ops.interpolate(x, size=(4, 4), scale_factor=2.0)


def test_interpolate_scale_factor():
    x = np.random.default_rng(2).normal(size=(1, 3, 4, 4))
    out = np.asarray(tessera.ops.interpolate(x, scale_factor=2.0, layout="nchw"))
    assert out.shape == (1, 3, 8, 8)


def test_interpolate_nhwc_layout_matches_nchw():
    x = np.random.default_rng(3).normal(size=(1, 4, 4, 3))  # NHWC
    out_nhwc = np.asarray(tessera.ops.interpolate(x, size=(8, 8), layout="nhwc"))
    out_nchw = np.asarray(
        tessera.ops.interpolate(np.transpose(x, (0, 3, 1, 2)), size=(8, 8), layout="nchw")
    )
    assert out_nhwc.shape == (1, 8, 8, 3)
    np.testing.assert_allclose(out_nhwc, np.transpose(out_nchw, (0, 2, 3, 1)), atol=1e-9)


def test_center_crop_central_window():
    x = np.arange(1 * 1 * 4 * 4, dtype=np.float64).reshape(1, 1, 4, 4)
    out = np.asarray(tessera.ops.center_crop(x, size=2, layout="nchw"))
    np.testing.assert_allclose(out[0, 0], [[5, 6], [9, 10]])


def test_center_crop_too_big_raises():
    x = np.zeros((1, 3, 4, 4))
    with pytest.raises(ValueError, match="exceeds"):
        tessera.ops.center_crop(x, size=8, layout="nchw")


def test_image_normalize_per_channel():
    x = np.random.default_rng(4).normal(size=(1, 3, 2, 2))
    mean = [0.5, 0.25, 0.0]
    std = [2.0, 1.0, 0.5]
    out = np.asarray(tessera.ops.image_normalize(x, mean=mean, std=std, layout="nchw"))
    expected = (x - np.reshape(mean, (1, 3, 1, 1))) / np.reshape(std, (1, 3, 1, 1))
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_image_normalize_zero_std_raises():
    x = np.zeros((1, 3, 2, 2))
    with pytest.raises(ValueError, match="nonzero"):
        tessera.ops.image_normalize(x, mean=0.0, std=0.0)


def test_vlm_transform_composition_shapes():
    """resize shorter side → center crop → normalize, the article's pipeline."""
    x = np.random.default_rng(5).normal(size=(3, 50, 80))  # CHW
    r = tessera.ops.image_resize(x, size=32, layout="chw")          # → 3×32×51
    c = tessera.ops.center_crop(r, size=32, layout="chw")           # → 3×32×32
    n = np.asarray(
        tessera.ops.image_normalize(c, mean=0.5, std=0.5, layout="chw")
    )
    assert n.shape == (3, 32, 32)


# ── autodiff: each preprocessing op is linear, VJP/JVP vs finite difference ──

_PREP_CASES = [
    ("image_resize", lambda x: tessera.ops.image_resize(x, size=(5, 6), layout="nchw"),
     dict(size=(5, 6), mode="bilinear", align_corners=False, layout="nchw")),
    ("interpolate", lambda x: tessera.ops.interpolate(x, scale_factor=1.5, layout="nchw"),
     dict(scale_factor=1.5, mode="bilinear", align_corners=False, layout="nchw")),
    ("center_crop", lambda x: tessera.ops.center_crop(x, size=2, layout="nchw"),
     dict(size=2, layout="nchw")),
    ("image_normalize", lambda x: tessera.ops.image_normalize(x, mean=[0.1, 0.2, 0.3], std=[1.5, 0.7, 2.0], layout="nchw"),
     dict(mean=[0.1, 0.2, 0.3], std=[1.5, 0.7, 2.0], layout="nchw")),
]


@pytest.mark.parametrize("name,fwd,kw", _PREP_CASES, ids=[c[0] for c in _PREP_CASES])
def test_preprocessing_vjp_matches_numeric(name, fwd, kw):
    x = np.random.default_rng(9).normal(size=(1, 3, 4, 4))
    out = np.asarray(fwd(x))
    dout = np.random.default_rng(10).normal(size=out.shape)

    (dx,) = get_vjp(name)(dout, x, **kw)

    # Numeric VJP: <dout, J·e_i> per input element = directional via dot.
    dx_numeric = np.zeros_like(x, dtype=np.float64)
    eps = 1e-6
    xf = x.astype(np.float64)
    it = np.nditer(xf, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = xf[idx]
        xf[idx] = orig + eps
        fp = np.asarray(fwd(xf), dtype=np.float64)
        xf[idx] = orig - eps
        fm = np.asarray(fwd(xf), dtype=np.float64)
        xf[idx] = orig
        dx_numeric[idx] = float(np.sum(dout * (fp - fm) / (2 * eps)))
        it.iternext()
    np.testing.assert_allclose(np.asarray(dx), dx_numeric, atol=1e-5)


@pytest.mark.parametrize("name,fwd,kw", _PREP_CASES, ids=[c[0] for c in _PREP_CASES])
def test_preprocessing_jvp_matches_numeric(name, fwd, kw):
    x = np.random.default_rng(12).normal(size=(1, 3, 4, 4))
    dx = np.random.default_rng(13).normal(size=x.shape)
    primal_out, tan_out = get_jvp(name)((x,), (dx,), **kw)
    np.testing.assert_allclose(primal_out, np.asarray(fwd(x)), atol=1e-9)
    tan_numeric = _numeric_jvp(lambda v: fwd(v), x, dx, eps=1e-6)
    np.testing.assert_allclose(tan_out, tan_numeric, atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# patch embedder (P1): patchify / factorized_pos_emb / patch_embed
# ════════════════════════════════════════════════════════════════════════════


def test_patch_embed_ops_registered():
    for n in ("patchify", "factorized_pos_emb"):
        assert hasattr(tessera.ops, n)
        assert get_vjp(n) is not None
        assert get_jvp(n) is not None


def test_patchify_shapes_and_values():
    B, C, H, W, P = 2, 3, 8, 8, 4
    x = np.random.default_rng(0).normal(size=(B, C, H, W))
    out = np.asarray(tessera.ops.patchify(x, patch_size=P, layout="nchw"))
    assert out.shape == (B, (H // P) * (W // P), C * P * P)
    # Reference via explicit reshape/permute (the article's recipe).
    nh, nw = H // P, W // P
    ref = x.reshape(B, C, nh, P, nw, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, nh * nw, C * P * P)
    np.testing.assert_allclose(out, ref)


def test_patchify_indivisible_raises():
    x = np.zeros((1, 3, 7, 8))
    with pytest.raises(ValueError, match="divisible"):
        tessera.ops.patchify(x, patch_size=4)


def test_patchify_vjp_jvp_match_numeric():
    B, C, H, W, P = 1, 2, 4, 4, 2
    rng = np.random.default_rng(1)
    x = rng.normal(size=(B, C, H, W))
    out = np.asarray(tessera.ops.patchify(x, patch_size=P))
    dout = rng.normal(size=out.shape)

    (dx,) = get_vjp("patchify")(dout, x, patch_size=P, layout="nchw")
    # patchify is an orthogonal permutation: <dout, patchify(v)> = <vjp, v>.
    dx_numeric = _numeric_grad(
        lambda v: dout * np.asarray(tessera.ops.patchify(v, patch_size=P)), x
    )
    np.testing.assert_allclose(np.asarray(dx), dx_numeric, atol=1e-6)

    dxt = rng.normal(size=x.shape)
    primal_out, tan_out = get_jvp("patchify")((x,), (dxt,), patch_size=P, layout="nchw")
    np.testing.assert_allclose(primal_out, out)
    tan_numeric = _numeric_jvp(lambda v: tessera.ops.patchify(v, patch_size=P), x, dxt)
    np.testing.assert_allclose(tan_out, tan_numeric, atol=1e-6)


def test_factorized_pos_emb_values():
    gh, gw, D = 3, 4, 5
    rng = np.random.default_rng(2)
    row = rng.normal(size=(gh, D))
    col = rng.normal(size=(gw, D))
    out = np.asarray(tessera.ops.factorized_pos_emb(row, col, grid_h=gh, grid_w=gw))
    assert out.shape == (gh * gw, D)
    for i in range(gh):
        for j in range(gw):
            np.testing.assert_allclose(out[i * gw + j], row[i] + col[j])


def test_factorized_pos_emb_table_too_small_raises():
    with pytest.raises(ValueError, match="too small"):
        tessera.ops.factorized_pos_emb(np.zeros((2, 4)), np.zeros((4, 4)), grid_h=3, grid_w=4)


def test_factorized_pos_emb_vjp_matches_numeric():
    gh, gw, D = 3, 4, 5
    rng = np.random.default_rng(3)
    # Tables larger than the grid — unused rows must get zero gradient.
    row = rng.normal(size=(gh + 2, D))
    col = rng.normal(size=(gw + 1, D))
    out = np.asarray(tessera.ops.factorized_pos_emb(row, col, grid_h=gh, grid_w=gw))
    dout = rng.normal(size=out.shape)

    drow, dcol = get_vjp("factorized_pos_emb")(dout, row, col, grid_h=gh, grid_w=gw)
    drow_num = _numeric_grad(
        lambda r: dout * np.asarray(tessera.ops.factorized_pos_emb(r, col, grid_h=gh, grid_w=gw)), row
    )
    dcol_num = _numeric_grad(
        lambda c: dout * np.asarray(tessera.ops.factorized_pos_emb(row, c, grid_h=gh, grid_w=gw)), col
    )
    np.testing.assert_allclose(np.asarray(drow), drow_num, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dcol), dcol_num, atol=1e-6)
    # Unused table rows carry no gradient.
    assert np.allclose(np.asarray(drow)[gh:], 0.0)
    assert np.allclose(np.asarray(dcol)[gw:], 0.0)


def test_existing_patch_embed_still_works():
    """The pre-existing NHWC patch_embed (patchify + optional projection) is
    untouched; the new atomic ops sit alongside it."""
    rng = np.random.default_rng(4)
    media = rng.normal(size=(2, 8, 8, 3))  # NHWC
    flat = np.asarray(tessera.ops.patch_embed(media, patch_size=4))
    assert flat.shape == (2, 4, 4 * 4 * 3)


def test_gemma_embedder_composition_tape():
    """The Gemma-4 embedder = patchify → linear projection → factorized 2-D
    positional embedding, composed from the new atomic primitives. The whole
    chain is differentiable through the tape: gradient reaches the image, the
    projection weight, and both positional tables."""
    B, C, H, W, P, D = 1, 2, 4, 4, 2, 3
    rng = np.random.default_rng(5)
    nh, nw = H // P, W // P
    img = tessera.nn.Parameter(rng.normal(size=(B, C, H, W)))
    weight = tessera.nn.Parameter(rng.normal(size=(C * P * P, D)))
    pos_row = tessera.nn.Parameter(rng.normal(size=(nh, D)))
    pos_col = tessera.nn.Parameter(rng.normal(size=(nw, D)))

    with tessera.autodiff.tape() as t:
        patches = tessera.ops.patchify(img, patch_size=P, layout="nchw")
        proj = tessera.ops.matmul(patches, weight)            # (B, N, D)
        pos = tessera.ops.factorized_pos_emb(pos_row, pos_col, grid_h=nh, grid_w=nw)
        out = tessera.ops.add(proj, pos)                      # broadcast (N,D) over batch
        loss = tessera.ops.reduce(out, op="sum")
        t.backward(loss)

    for name, prm in [("img", img), ("weight", weight),
                      ("pos_row", pos_row), ("pos_col", pos_col)]:
        g = prm.grad
        assert g is not None, f"no gradient for {name}"
        assert np.isfinite(np.asarray(g.numpy())).all(), f"non-finite grad for {name}"
    # Each positional table entry feeds B output rows summed → grad = B * D-ones
    # over the D feature dim, per used table row.
    np.testing.assert_allclose(pos_row.grad.numpy(), np.full((nh, D), B * nw))
    np.testing.assert_allclose(pos_col.grad.numpy(), np.full((nw, D), B * nh))


# ════════════════════════════════════════════════════════════════════════════
# mrope_2d — multimodal M-RoPE (P1)
# ════════════════════════════════════════════════════════════════════════════


def test_mrope_2d_registered():
    assert hasattr(tessera.ops, "mrope_2d")
    assert get_vjp("mrope_2d") is not None
    assert get_jvp("mrope_2d") is not None


def test_mrope_2d_sum_of_sections_validated():
    x = np.zeros((1, 4, 8))         # Hd=8 → Hd//2=4
    positions = np.zeros((2, 4))
    inv_freq = np.ones(4)
    with pytest.raises(ValueError, match="sections"):
        tessera.ops.mrope_2d(x, positions, inv_freq, sections=(2, 3))  # sums to 5≠4


def test_mrope_2d_preserves_norm():
    rng = np.random.default_rng(0)
    Hd, S = 8, 5
    x = rng.normal(size=(2, S, Hd))
    positions = rng.integers(0, 10, size=(2, S)).astype(np.float64)
    inv_freq = 1.0 / (10000 ** (np.arange(0, Hd // 2) / (Hd // 2)))
    out = np.asarray(tessera.ops.mrope_2d(x, positions, inv_freq, sections=(2, 2)))
    # Rotation is orthogonal per even/odd pair → preserves L2 norm per token.
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-9
    )


def test_mrope_2d_reduces_to_rope_for_single_section():
    """One section with one position axis = standard 1-D rope."""
    rng = np.random.default_rng(1)
    Hd, S = 8, 6
    x = rng.normal(size=(1, S, Hd))
    pos = np.arange(S, dtype=np.float64)
    inv_freq = 1.0 / (10000 ** (np.arange(0, Hd // 2) / (Hd // 2)))
    out = np.asarray(
        tessera.ops.mrope_2d(x, pos[None, :], inv_freq, sections=(Hd // 2,))
    )
    # rope(x, theta) with theta = outer(pos, inv_freq) expanded to Hd.
    theta = np.zeros((S, Hd))
    theta[:, 0::2] = pos[:, None] * inv_freq[None, :]
    ref = np.asarray(tessera.ops.rope(x, theta))
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_mrope_2d_sections_use_distinct_axes():
    """Pairs in section 0 rotate by positions[0]; section 1 by positions[1]."""
    Hd, S = 8, 3
    x = np.ones((1, S, Hd))
    inv_freq = np.ones(Hd // 2)
    # axis 0 = all-zero position (no rotation), axis 1 = nonzero.
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = np.asarray(tessera.ops.mrope_2d(x, positions, inv_freq, sections=(2, 2)))
    # First 2 pairs (dims 0..3) unrotated → unchanged ones.
    np.testing.assert_allclose(out[..., 0:4], x[..., 0:4])
    # Last 2 pairs (dims 4..7) rotated → changed.
    assert not np.allclose(out[..., 4:8], x[..., 4:8])


def test_mrope_2d_vjp_matches_numeric():
    rng = np.random.default_rng(2)
    Hd, S = 8, 4
    x = rng.normal(size=(2, S, Hd))
    positions = rng.integers(0, 5, size=(2, S)).astype(np.float64)
    inv_freq = 1.0 / (10000 ** (np.arange(0, Hd // 2) / (Hd // 2)))
    sec = (2, 2)
    out = np.asarray(tessera.ops.mrope_2d(x, positions, inv_freq, sections=sec))
    dout = rng.normal(size=out.shape)

    dx, dpos, dfreq = get_vjp("mrope_2d")(dout, x, positions, inv_freq, sections=sec)
    assert dpos is None and dfreq is None
    dx_numeric = _numeric_grad(
        lambda v: dout * np.asarray(tessera.ops.mrope_2d(v, positions, inv_freq, sections=sec)), x
    )
    np.testing.assert_allclose(np.asarray(dx), dx_numeric, atol=1e-6)


def test_mrope_2d_jvp_matches_numeric():
    rng = np.random.default_rng(3)
    Hd, S = 8, 4
    x = rng.normal(size=(1, S, Hd))
    positions = rng.integers(0, 5, size=(2, S)).astype(np.float64)
    inv_freq = 1.0 / (10000 ** (np.arange(0, Hd // 2) / (Hd // 2)))
    sec = (1, 3)
    dx = rng.normal(size=x.shape)
    primal_out, tan_out = get_jvp("mrope_2d")(
        (x, positions, inv_freq), (dx, None, None), sections=sec
    )
    np.testing.assert_allclose(
        primal_out, np.asarray(tessera.ops.mrope_2d(x, positions, inv_freq, sections=sec))
    )
    tan_numeric = _numeric_jvp(
        lambda v: tessera.ops.mrope_2d(v, positions, inv_freq, sections=sec), x, dx
    )
    np.testing.assert_allclose(tan_out, tan_numeric, atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# P2: pixel_shuffle / pixel_unshuffle + cross_attention / perceiver_resampler
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ["pixel_unshuffle", "pixel_shuffle", "cross_attention"])
def test_p2_ops_registered(name):
    assert hasattr(tessera.ops, name)
    assert get_vjp(name) is not None
    assert get_jvp(name) is not None


def test_pixel_unshuffle_shape_and_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8, 8))
    down = np.asarray(tessera.ops.pixel_unshuffle(x, downscale_factor=2))
    assert down.shape == (2, 3 * 4, 4, 4)
    # pixel_shuffle inverts pixel_unshuffle exactly.
    up = np.asarray(tessera.ops.pixel_shuffle(down, upscale_factor=2))
    np.testing.assert_allclose(up, x)


def test_pixel_shuffle_nhwc_layout_roundtrip():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 6, 6, 2))
    down = np.asarray(
        tessera.ops.pixel_unshuffle(x, downscale_factor=3, layout="nhwc"))
    assert down.shape == (1, 2, 2, 18)
    up = np.asarray(
        tessera.ops.pixel_shuffle(down, upscale_factor=3, layout="nhwc"))
    np.testing.assert_allclose(up, x)


def test_pixel_unshuffle_indivisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        tessera.ops.pixel_unshuffle(np.zeros((1, 3, 7, 8)), downscale_factor=2)


def test_pixel_ops_vjp_jvp_match_numeric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 2, 4, 4))
    out = np.asarray(tessera.ops.pixel_unshuffle(x, downscale_factor=2))
    dout = rng.normal(size=out.shape)
    (dx,) = get_vjp("pixel_unshuffle")(dout, x, downscale_factor=2, layout="nchw")
    dx_num = _numeric_grad(
        lambda v: dout * np.asarray(tessera.ops.pixel_unshuffle(v, downscale_factor=2)), x
    )
    np.testing.assert_allclose(np.asarray(dx), dx_num, atol=1e-6)
    dxt = rng.normal(size=x.shape)
    primal, tan = get_jvp("pixel_unshuffle")((x,), (dxt,), downscale_factor=2, layout="nchw")
    np.testing.assert_allclose(primal, out)
    np.testing.assert_allclose(
        tan, _numeric_jvp(lambda v: tessera.ops.pixel_unshuffle(v, downscale_factor=2), x, dxt), atol=1e-6
    )


def test_cross_attention_matches_reference_sdpa():
    rng = np.random.default_rng(2)
    Sq, Sk, d, dv = 3, 5, 4, 6
    q = rng.normal(size=(2, Sq, d))
    k = rng.normal(size=(2, Sk, d))
    v = rng.normal(size=(2, Sk, dv))
    out = np.asarray(tessera.ops.cross_attention(q, k, v))
    assert out.shape == (2, Sq, dv)
    # Reference SDPA.
    s = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    np.testing.assert_allclose(out, p @ v, atol=1e-12)


def test_cross_attention_rows_are_convex_combinations():
    # Each output row is a softmax-weighted (convex) mix of value rows, so it
    # lies within the value range.
    rng = np.random.default_rng(3)
    q = rng.normal(size=(1, 2, 4))
    k = rng.normal(size=(1, 5, 4))
    v = rng.uniform(0.0, 1.0, size=(1, 5, 3))
    out = np.asarray(tessera.ops.cross_attention(q, k, v))
    assert out.min() >= v.min() - 1e-9 and out.max() <= v.max() + 1e-9


def test_cross_attention_vjp_matches_numeric():
    rng = np.random.default_rng(4)
    Sq, Sk, d, dv = 2, 3, 4, 3
    q = rng.normal(size=(1, Sq, d))
    k = rng.normal(size=(1, Sk, d))
    v = rng.normal(size=(1, Sk, dv))
    out = np.asarray(tessera.ops.cross_attention(q, k, v))
    dout = rng.normal(size=out.shape)
    dq, dk, dv_ = get_vjp("cross_attention")(dout, q, k, v)
    for analytic, arr, idx in [(dq, q, "q"), (dk, k, "k"), (dv_, v, "v")]:
        num = _numeric_grad(
            lambda a, _i=idx: dout * np.asarray(
                tessera.ops.cross_attention(
                    a if _i == "q" else q, a if _i == "k" else k, a if _i == "v" else v)), arr
        )
        np.testing.assert_allclose(np.asarray(analytic), num, atol=1e-5, err_msg=f"d{idx}")


def test_cross_attention_jvp_matches_numeric():
    rng = np.random.default_rng(5)
    Sq, Sk, d, dv = 2, 3, 4, 3
    q = rng.normal(size=(1, Sq, d))
    k = rng.normal(size=(1, Sk, d))
    v = rng.normal(size=(1, Sk, dv))
    dq = rng.normal(size=q.shape)
    dk = rng.normal(size=k.shape)
    dvt = rng.normal(size=v.shape)
    primal, tan = get_jvp("cross_attention")((q, k, v), (dq, dk, dvt))
    np.testing.assert_allclose(primal, np.asarray(tessera.ops.cross_attention(q, k, v)))
    eps = 1e-6
    fp = np.asarray(tessera.ops.cross_attention(q + eps * dq, k + eps * dk, v + eps * dvt))
    fm = np.asarray(tessera.ops.cross_attention(q - eps * dq, k - eps * dk, v - eps * dvt))
    np.testing.assert_allclose(tan, (fp - fm) / (2 * eps), atol=1e-5)


def test_perceiver_resampler_compresses_and_is_differentiable():
    """Learned latents cross-attend to a variable-length feature sequence,
    compressing it to len(latents) tokens; differentiable through the tape."""
    rng = np.random.default_rng(6)
    n_latents, S, d = 4, 16, 5
    latents = tessera.nn.Parameter(rng.normal(size=(1, n_latents, d)))
    feats = tessera.nn.Parameter(rng.normal(size=(1, S, d)))
    with tessera.autodiff.tape() as t:
        out = tessera.ops.perceiver_resampler(latents, feats)
        loss = tessera.ops.reduce(out, op="sum")
        t.backward(loss)
    assert np.asarray(out).shape == (1, n_latents, d)   # S=16 → 4 tokens
    for nm, prm in [("latents", latents), ("feats", feats)]:
        g = prm.grad
        assert g is not None and np.isfinite(np.asarray(g.numpy())).all(), nm


def test_perceiver_resampler_matches_cross_attention_composite():
    rng = np.random.default_rng(11)
    latents = rng.normal(size=(1, 3, 4))
    features = rng.normal(size=(1, 7, 4))
    out = np.asarray(tessera.ops.perceiver_resampler(latents, features))
    ref = np.asarray(tessera.ops.cross_attention(latents, features, features))
    assert out.shape == (1, 3, 4)
    np.testing.assert_allclose(out, ref, atol=1e-12)
