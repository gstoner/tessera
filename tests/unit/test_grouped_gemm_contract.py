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
