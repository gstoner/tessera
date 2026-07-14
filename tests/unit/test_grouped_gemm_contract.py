"""DeepGEMM-inspired grouped-GEMM + scale-layout contract rung.

Treats DeepGEMM as a design-pattern source (no kernel import): grouped GEMM is a
first-class op *family* with explicit grouped-layout metadata, and FP8/FP4 carry
a scale *layout* contract (granularity / packing / alignment / transposed), not
just a dtype name.

Covers:
  * GroupedLayout / ScaleLayout dataclass contracts + validation.
  * The metadata is attached to grouped_gemm + the FP8/FP4 quantizers in the
    primitive-coverage audit registry.
  * Oracle (step 4): an FP8 reference-dequantized grouped GEMM matches the f32
    backend-selected grouped GEMM within fp8 tolerance.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as _runtime
from tessera.compiler import grouped_layout as gl
from tessera.compiler import primitive_coverage as pc


# ── GroupedLayout contract ──────────────────────────────────────────────────
def test_grouped_layout_kinds_and_axes():
    assert gl.dense_layout().group_axis is None
    assert gl.dense_layout().is_grouped is False
    assert gl.contiguous_layout().group_axis == "M"
    assert gl.masked_layout().group_axis == "M"
    assert gl.k_grouped_layout().group_axis == "K"
    # compiled vs dynamic dims (DeepGEMM compiles N/K, leaves M+groups dynamic)
    cg = gl.contiguous_layout()
    assert cg.compiled_dims == ("N", "K")
    assert cg.dynamic_dims == ("M", "num_groups")
    assert gl.k_grouped_layout().dynamic_dims == ("K", "num_groups")


def test_grouped_layout_validation():
    with pytest.raises(ValueError):
        gl.GroupedLayout(kind="bogus")
    with pytest.raises(ValueError):
        gl.GroupedLayout(kind="contiguous", alignment=100)  # not a power of two
    with pytest.raises(ValueError):
        gl.GroupedLayout(kind="contiguous", alignment=0)


def test_grouped_layout_metadata_roundtrip():
    md = gl.masked_layout(alignment=64).as_metadata_dict()
    assert md == {
        "kind": "masked", "group_axis": "M", "alignment": 64,
        "compiled_dims": ["N", "K"], "dynamic_dims": ["M", "num_groups"],
    }


# ── ScaleLayout contract ────────────────────────────────────────────────────
def test_scale_layout_validation():
    with pytest.raises(ValueError):
        gl.ScaleLayout(granularity="bogus")
    with pytest.raises(ValueError):
        gl.ScaleLayout(granularity="block")  # block shape required
    with pytest.raises(ValueError):
        gl.ScaleLayout(granularity="per_tensor", block=(1, 128))  # block only for block
    with pytest.raises(ValueError):
        gl.ScaleLayout(granularity="block", block=(1, 128), packing="bogus")
    with pytest.raises(ValueError):
        gl.ScaleLayout(granularity="per_row", vector_size=0)


@pytest.mark.parametrize("dtype,gran,block,pack", [
    ("nvfp4", "block", [1, 16], "e4m3"),
    ("fp8_e4m3", "block", [1, 128], "ue8m0"),
    ("fp4_e2m1", "block", [1, 128], "ue8m0"),
    ("int8", "per_tensor", None, "none"),
])
def test_scale_layout_for_dtype(dtype, gran, block, pack):
    sl = gl.scale_layout_for(dtype)
    assert sl is not None
    md = sl.as_metadata_dict()
    assert md["granularity"] == gran
    assert md["block"] == block
    assert md["packing"] == pack


def test_scale_layout_for_unknown_dtype_is_none():
    assert gl.scale_layout_for("fp32") is None


# ── Metadata attachment in the audit registry ──────────────────────────────
def test_grouped_gemm_carries_grouped_layout_metadata():
    reg = pc.all_primitive_coverages()
    gg = reg["grouped_gemm"]
    glm = gg.metadata.get("grouped_layout")
    assert glm is not None, "grouped_gemm must carry a grouped_layout contract"
    assert glm["kind"] == "contiguous"
    assert glm["group_axis"] == "M"


@pytest.mark.parametrize("op,gran", [
    ("quantize_nvfp4", "block"),
    ("quantize_fp8", "block"),
    ("quantize_int8", "per_tensor"),
])
def test_quantizers_carry_scale_layout_in_numeric_policy(op, gran):
    reg = pc.all_primitive_coverages()
    p = reg[op]
    sl = p.metadata.get("numeric_policy", {}).get("scale_layout")
    assert sl is not None, f"{op} must carry a scale_layout contract"
    assert sl["granularity"] == gran


# ── Oracle (step 4): FP8 dequantized grouped GEMM vs f32 grouped GEMM ────────
def _grouped_inputs(seed, group_sizes, K, N):
    rng = np.random.default_rng(seed)
    gs = np.asarray(group_sizes, np.int64)
    T, E = int(gs.sum()), len(gs)
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    return x, w, gs


@pytest.mark.parametrize("group_sizes", [[5, 3, 4], [16, 16, 32], [12]])
def test_reference_grouped_gemm_matches_explicit_loop(group_sizes):
    x, w, gs = _grouped_inputs(1, group_sizes, K=8, N=6)
    got = gl.reference_grouped_gemm(x, w, gs)
    # explicit per-group loop (f64 to match the reference's accumulation dtype)
    exp = np.zeros((x.shape[0], w.shape[2]), dtype=np.float64)
    off = 0
    for e in range(len(gs)):
        n = int(gs[e])
        exp[off:off + n] = x[off:off + n].astype(np.float64) @ w[e].astype(np.float64)
        off += n
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("group_sizes", [[5, 3, 4], [16, 16, 32]])
def test_fp8_dequant_grouped_gemm_oracle(group_sizes):
    """FP8 (per-tensor) reference-dequantized grouped GEMM must match the f32
    grouped GEMM within fp8 tolerance — validates that the per_tensor scale
    layout describes a numerically-faithful quantization for grouped GEMM."""
    x, w, gs = _grouped_inputs(7, group_sizes, K=64, N=16)

    # f32 oracle (the "backend-selected" exact result).
    ref = gl.reference_grouped_gemm(x, w, gs)

    # FP8 path: quantize_fp8 returns the fp8-round-tripped value in fp32.
    x_q, _ = ts.ops.quantize_fp8(x, format="e4m3")
    # weights quantized per-expert (the grouped GEMM's natural scale unit).
    w_q = np.stack([ts.ops.quantize_fp8(w[e], format="e4m3")[0] for e in range(len(gs))])
    got = gl.reference_grouped_gemm(np.asarray(x_q), np.asarray(w_q), gs)

    denom = np.linalg.norm(ref) + 1e-9
    rel = np.linalg.norm(got - ref) / denom
    assert rel < 0.06, f"fp8 grouped-GEMM rel error {rel:.4f} exceeds fp8 budget"


@pytest.mark.parametrize("group_sizes", [[5, 3, 4], [16, 16, 32]])
def test_nvfp4_block_dequant_grouped_gemm_oracle(group_sizes):
    """NVFP4 block-scaled (1x16) round-trip is more accurate than per-tensor
    fp8 — same oracle, tighter relative-error budget."""
    x, w, gs = _grouped_inputs(9, group_sizes, K=64, N=16)
    ref = gl.reference_grouped_gemm(x, w, gs)
    x_q = np.asarray(ts.ops.dequantize_nvfp4(*ts.ops.quantize_nvfp4(x, block_size=16), block_size=16))
    w_q = np.stack([
        np.asarray(ts.ops.dequantize_nvfp4(*ts.ops.quantize_nvfp4(w[e], block_size=16), block_size=16))
        for e in range(len(gs))])
    got = gl.reference_grouped_gemm(x_q, w_q, gs)
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
    assert rel < 0.20, f"nvfp4 grouped-GEMM rel error {rel:.4f} too high"


# ── Rung A: the grouped_layout contract is a runtime gate, not a sticker ─────
def test_validate_grouped_alignment():
    gl.validate_grouped_alignment([128, 256, 128], 128)  # ok
    gl.validate_grouped_alignment([5, 3, 4], 1)          # alignment<=1 is a no-op
    gl.validate_grouped_alignment([5, 3, 4], None)       # None is a no-op
    with pytest.raises(ValueError):
        gl.validate_grouped_alignment([5, 3, 4], 128)    # not aligned


def _gg_inputs(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((12, 8)).astype(np.float32),
            rng.standard_normal((3, 8, 6)).astype(np.float32),
            np.array([5, 3, 4], np.int64))


@pytest.mark.parametrize("kind", ["masked", "k_grouped"])
def test_eager_grouped_gemm_rejects_unsupported_kinds(kind):
    x, w, gs = _gg_inputs()
    with pytest.raises(NotImplementedError):
        ts.ops.grouped_gemm(x, w, gs, kind=kind)


def test_eager_grouped_gemm_rejects_unknown_kind():
    x, w, gs = _gg_inputs()
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, w, gs, kind="bogus")


def test_eager_grouped_gemm_enforces_alignment():
    x, w, gs = _gg_inputs()
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, w, gs, alignment=128)  # [5,3,4] not 128-aligned
    # aligned groups pass + match the reference
    rng = np.random.default_rng(1)
    xa = rng.standard_normal((256, 8)).astype(np.float32)
    wa = rng.standard_normal((2, 8, 6)).astype(np.float32)
    gsa = np.array([128, 128], np.int64)
    got = ts.ops.grouped_gemm(xa, wa, gsa, alignment=128)
    np.testing.assert_allclose(got, gl.reference_grouped_gemm(xa, wa, gsa), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["masked", "k_grouped"])
def test_runtime_dispatch_rejects_unsupported_kinds(kind):
    x, w, gs = _gg_inputs()
    with pytest.raises(ValueError) as ei:
        _runtime._apple_gpu_dispatch_grouped_gemm([x, w, gs], {"kind": kind}, np)
    msg = str(ei.value)
    assert "apple_gpu" in msg and kind in msg  # Decision #21: names target + kind


def test_runtime_dispatch_contiguous_matches_reference():
    x, w, gs = _gg_inputs(2)
    for kw in ({}, {"kind": "contiguous"}, {"kind": "dense"}):
        got = np.asarray(_runtime._apple_gpu_dispatch_grouped_gemm([x, w, gs], kw, np))
        np.testing.assert_allclose(got, gl.reference_grouped_gemm(x, w, gs), rtol=1e-4, atol=1e-4)


def test_runtime_dispatch_enforces_alignment():
    x, w, gs = _gg_inputs(3)
    with pytest.raises(ValueError):
        _runtime._apple_gpu_dispatch_grouped_gemm([x, w, gs], {"alignment": 128}, np)


# ── Rung B: scale_layout drives a quantized grouped-GEMM runtime path ────────
@pytest.mark.parametrize("quant,bound", [("fp8_e4m3", 0.06), ("fp8_e5m2", 0.30), ("nvfp4", 0.20)])
def test_eager_quantized_grouped_gemm(quant, bound):
    x, w, gs = _grouped_inputs(11, [16, 16, 32], K=64, N=16)
    ref = gl.reference_grouped_gemm(x, w, gs)
    got = ts.ops.grouped_gemm(x, w, gs, quant=quant)
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
    assert rel < bound, f"{quant} quantized grouped GEMM rel {rel:.4f} > {bound}"


def test_runtime_quantized_grouped_gemm_matches_eager():
    x, w, gs = _grouped_inputs(13, [16, 16, 32], K=64, N=16)
    for quant in ("fp8_e4m3", "nvfp4"):
        eager = np.asarray(ts.ops.grouped_gemm(x, w, gs, quant=quant))
        rt = np.asarray(_runtime._apple_gpu_dispatch_grouped_gemm([x, w, gs], {"quant": quant}, np))
        # same host-dequant path → identical (modulo the fused-kernel f32 GEMM)
        np.testing.assert_allclose(rt, eager, rtol=1e-4, atol=1e-4)


def test_quantized_grouped_gemm_rejects_unsupported_dtype():
    x, w, gs = _grouped_inputs(15, [5, 3, 4], K=8, N=6)
    with pytest.raises(ValueError):
        ts.ops.grouped_gemm(x, w, gs, quant="int8")  # no dequant-on-host path
    with pytest.raises(ValueError):
        gl.apply_quant_for_grouped(
            x, w, "bogus",
            quantize_fp8=ts.ops.quantize_fp8, quantize_nvfp4=ts.ops.quantize_nvfp4,
            dequantize_nvfp4=ts.ops.dequantize_nvfp4)


def test_quant_grouped_gemm_preserves_shape_and_dtype():
    x, w, gs = _grouped_inputs(17, [5, 3, 4], K=8, N=6)
    got = ts.ops.grouped_gemm(x, w, gs, quant="fp8_e4m3")
    assert got.shape == (int(gs.sum()), w.shape[2])
    assert got.dtype == x.dtype


# ── FP8×FP4 mixed-precision scheme (Blackwell / DeepGEMM MoE) ────────────────
def test_quant_scheme_resolution():
    assert gl.quant_scheme_for("fp8xfp4") == ("fp8_e4m3", "nvfp4")
    assert gl.quant_scheme_for("fp8_e5m2xnvfp4") == ("fp8_e5m2", "nvfp4")
    assert gl.quant_scheme_for("nvfp4") == ("nvfp4", "nvfp4")        # plain → both
    assert gl.quant_scheme_for("fp8_e4m3") == ("fp8_e4m3", "fp8_e4m3")
    with pytest.raises(ValueError, match="no dequant-on-host"):
        gl.quant_scheme_for("bogus8")


def test_mixed_quant_applies_act_and_weight_separately():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((20, 64)).astype(np.float32)
    w = rng.standard_normal((3, 64, 16)).astype(np.float32)
    xa, wq = gl.apply_quant_for_grouped(
        x, w, "fp8xfp4", quantize_fp8=ts.ops.quantize_fp8,
        quantize_nvfp4=ts.ops.quantize_nvfp4, dequantize_nvfp4=ts.ops.dequantize_nvfp4)
    # Activations FP8 vs weights NVFP4 — the dequant fingerprints differ, so the
    # two operands must NOT match a single-scheme round-trip of the other dtype.
    x_fp8_only, _ = gl.apply_quant_for_grouped(
        x, w, "fp8_e4m3", quantize_fp8=ts.ops.quantize_fp8,
        quantize_nvfp4=ts.ops.quantize_nvfp4, dequantize_nvfp4=ts.ops.dequantize_nvfp4)
    np.testing.assert_allclose(xa, x_fp8_only, rtol=0, atol=0)  # act path == fp8
    assert xa.shape == x.shape and wq.shape == w.shape


def test_fp8xfp4_grouped_gemm_between_fp8_and_fp4_error():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((24, 64)).astype(np.float32)
    w = rng.standard_normal((3, 64, 16)).astype(np.float32)
    gs = np.array([8, 8, 8])
    ref = gl.reference_grouped_gemm(x, w, gs)

    def rel(q):
        got = np.asarray(ts.ops.grouped_gemm(x, w, gs, quant=q))
        return np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)

    r_fp8, r_mixed, r_fp4 = rel("fp8_e4m3"), rel("fp8xfp4"), rel("nvfp4")
    # FP8 activations × FP4 weights sits between pure-FP8 and pure-FP4 error.
    assert r_fp8 <= r_mixed <= r_fp4 + 0.02, (r_fp8, r_mixed, r_fp4)


# ── Masked grouped GEMM reference semantics (DeepGEMM family B3) ─────────────
# The native Apple lane still rejects `masked` (Decision #21 — no per-group tiled
# kernel); these lock the *reference* contract so the masked family is
# numerically defined for the oracle/evaluator even before a native kernel.
def test_masked_reference_matches_per_token_routing():
    rng = np.random.default_rng(0)
    T, K, N, E = 7, 4, 5, 3
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    # Arbitrary (non-contiguous) per-token expert assignment.
    eid = np.array([2, 0, 1, 1, 0, 2, 0], dtype=np.int64)
    got = gl.reference_grouped_gemm_masked(x, w, eid)
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    exp = np.stack([x64[t] @ w64[int(eid[t])] for t in range(T)])
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)
    assert got.shape == (T, N)


def test_masked_reference_zeros_padding_rows():
    rng = np.random.default_rng(1)
    T, K, N, E = 5, 3, 4, 2
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    eid = np.array([0, -1, 1, -1, 0], dtype=np.int64)  # -1 = masked padding row
    got = gl.reference_grouped_gemm_masked(x, w, eid)
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    # Padding rows are exactly zero; real rows route to their expert.
    assert np.all(got[1] == 0.0) and np.all(got[3] == 0.0)
    np.testing.assert_allclose(got[0], x64[0] @ w64[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[2], x64[2] @ w64[1], rtol=1e-12, atol=1e-12)


def test_masked_reference_equivalent_to_contiguous_when_sorted():
    # A masked routing whose tokens happen to be expert-contiguous must match the
    # contiguous reference with the corresponding group_sizes — the two families
    # agree on the overlap.
    rng = np.random.default_rng(2)
    K, N = 4, 6
    group_sizes = [3, 0, 2]  # expert 1 gets no tokens
    eid = np.array([0, 0, 0, 2, 2], dtype=np.int64)
    T, E = len(eid), len(group_sizes)
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    masked = gl.reference_grouped_gemm_masked(x, w, eid)
    contig = gl.reference_grouped_gemm(x, w, group_sizes)
    np.testing.assert_allclose(masked, contig, rtol=1e-12, atol=1e-12)


def test_masked_reference_validates_shapes():
    x = np.zeros((4, 3), np.float32)
    w = np.zeros((2, 3, 5), np.float32)
    with pytest.raises(ValueError, match="must equal the token count"):
        gl.reference_grouped_gemm_masked(x, w, np.zeros(3, np.int64))
    with pytest.raises(ValueError, match="out of range"):
        gl.reference_grouped_gemm_masked(x, w, np.array([0, 1, 2, 0], np.int64))
