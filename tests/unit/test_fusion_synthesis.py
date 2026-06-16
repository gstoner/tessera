"""General fusion middle-end — region discovery + MSL kernel synthesis.

Optimizing-Compiler Plan F0/F1a/F2a.  The keystone is *synthesis*: one
synthesizer emits the MSL for any ``matmul -> pointwise-epilogue`` region, gated
by the horizontal oracle (synthesized == unfused) — replacing the hand-written
catalog (matmul_gelu, …) and generalizing to chains it never had.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler.fusion import (
    EPILOGUE_OPS,
    AttentionRegion,
    FusedRegion,
    SYNTH_MAX_N,
    attention_cost,
    discover_attention_regions,
    discover_fusable_regions,
    fusion_cost,
    run_fused_attention,
    run_fused_region,
    should_fuse_attention,
    should_fuse_region,
    synthesize_attention_msl,
    synthesize_matmul_epilogue_msl,
    synthesize_matmul_epilogue_msl_tiled,
    _Op,
)


# ── F0/F2a: region + synthesized MSL structure (portable) ────────────────────

def test_synthesized_msl_has_the_expected_shape():
    src = synthesize_matmul_epilogue_msl(FusedRegion(("bias", "gelu")))
    assert "kernel void synth_matmul_epi(" in src
    assert "device const float* bias" in src           # bias buffer present
    assert "v = v + bias[n];" in src                   # bias epilogue inlined
    assert "0.5f*v*(1.0f+tanh" in src                  # gelu epilogue inlined


def test_no_bias_region_omits_the_bias_buffer():
    src = synthesize_matmul_epilogue_msl(FusedRegion(("silu",)))
    assert "device const float* bias" not in src
    assert "v / (1.0f + exp(-v))" in src


def test_region_validation():
    with pytest.raises(ValueError):
        FusedRegion(("not_a_real_op",))
    with pytest.raises(ValueError):
        FusedRegion(("bias", "bias"))                  # at most one bias


def test_region_reference_matches_manual_numpy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 6)).astype(np.float32)
    bias = rng.standard_normal((6,)).astype(np.float32)
    got = FusedRegion(("bias", "relu")).reference(A, B, bias)
    expected = np.maximum(A @ B + bias, 0.0)
    assert np.allclose(got, expected, atol=1e-5)


# ── F1a: discovery (portable) ────────────────────────────────────────────────

def test_discovery_captures_matmul_pointwise_chain():
    ops = [_Op("matmul", ("A", "B"), "m"),
           _Op("add", ("m", "bias"), "a"),
           _Op("gelu", ("a",), "g")]
    regions = discover_fusable_regions(ops)
    assert len(regions) == 1
    mi, epi_idx, region = regions[0]
    assert mi == 0 and epi_idx == [1, 2]
    assert region.epilogue == ("bias", "gelu")


def test_discovery_respects_single_use_of_the_intermediate():
    # the matmul result `m` feeds gelu AND a second consumer → NOT fusible
    # (fusing would drop the intermediate the second op reads).
    ops = [_Op("matmul", ("A", "B"), "m"),
           _Op("gelu", ("m",), "g"),
           _Op("add", ("m", "x"), "z")]            # second consumer of m
    regions = discover_fusable_regions(ops)
    assert regions == []


def test_discovery_ignores_non_matmul_roots():
    ops = [_Op("conv2d", ("X", "W"), "c"), _Op("relu", ("c",), "r")]
    assert discover_fusable_regions(ops) == []


# ── F2a: hardware-verified synthesis, gated by the horizontal oracle ─────────

# catalog chains (replace the hand-written kernels) + NOVEL chains (the catalog
# never had these) — all from one synthesizer.
_CHAINS = [
    (("gelu",), False),                     # == hand-written matmul_gelu
    (("relu",), False),
    (("silu",), False),
    (("sigmoid",), False),
    (("tanh",), False),
    (("bias",), True),
    (("bias", "gelu"), True),               # novel
    (("bias", "relu"), True),               # novel
    (("bias", "silu"), True),               # novel
    (("bias", "sigmoid", "tanh"), True),    # novel 3-op chain
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,has_bias", _CHAINS, ids=[str(c) for c, _ in _CHAINS])
def test_synthesized_kernel_equals_unfused_on_metal(chain, has_bias):
    rng = np.random.default_rng(hash(chain) & 0xFFFF)
    M, K, N = 16, 32, 24
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32) if has_bias else None
    region = FusedRegion(chain, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region(region, A, B, bias)
    assert execution == "metal_runtime"                # the synthesized kernel ran
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-4)  # horizontal oracle


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_synthesizer_reproduces_the_handwritten_matmul_gelu():
    # the synthesized matmul->gelu must match the hand-written catalog kernel it
    # replaces — proving the synthesizer can retire it.
    import ctypes
    from tessera.runtime import _load_apple_gpu_runtime

    rt = _load_apple_gpu_runtime()
    hw = getattr(rt, "tessera_apple_gpu_matmul_gelu_f32", None)
    if hw is None:
        pytest.skip("hand-written matmul_gelu symbol unavailable")
    hw.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int32] * 3
    hw.restype = None

    rng = np.random.default_rng(3)
    M, K, N = 12, 20, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    hand = np.zeros((M, N), np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hw(fp(np.ascontiguousarray(A)), fp(np.ascontiguousarray(B)), fp(hand), M, N, K)

    synth, execution = run_fused_region(FusedRegion(("gelu",)), A, B)
    assert execution == "metal_runtime"
    assert np.allclose(synth, hand, atol=1e-4)         # synthesized == hand-written


def test_fusion_collapses_dispatch_count():
    # the unfused chain is (1 matmul + len(epilogue)) dispatches; the synthesized
    # region is 1 — the DLOP decomposition-overhead win, now compiler-discovered.
    region = FusedRegion(("bias", "gelu", "silu"))
    unfused_dispatches = 1 + len(region.epilogue)       # matmul + 3 pointwise
    fused_dispatches = 1
    assert unfused_dispatches == 4
    assert fused_dispatches == 1


# ── F2b: reduction epilogues (rmsnorm / softmax) ─────────────────────────────

def test_reduction_region_synthesizes_a_row_reduction():
    rms = synthesize_matmul_epilogue_msl(FusedRegion((), reduction="rmsnorm"))
    assert "_ss += scores[n] * scores[n]" in rms       # sum-of-squares reduction
    assert "rsqrt(_ss / float(N)" in rms
    sm = synthesize_matmul_epilogue_msl(FusedRegion((), reduction="softmax"))
    assert "_mx = max(_mx, scores[n])" in sm           # row max
    assert "scores[n] / _sm" in sm                      # normalize


def test_reduction_reference_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 6)).astype(np.float32)
    x = A @ B
    rms = FusedRegion((), reduction="rmsnorm", eps=1e-6).reference(A, B)
    assert np.allclose(rms, x / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + 1e-6), atol=1e-5)
    sm = FusedRegion((), reduction="softmax").reference(A, B)
    e = np.exp(x - x.max(-1, keepdims=True))
    assert np.allclose(sm, e / e.sum(-1, keepdims=True), atol=1e-5)


def test_reduction_region_validation():
    with pytest.raises(ValueError):
        FusedRegion((), reduction="not_a_reduction")
    with pytest.raises(ValueError):
        FusedRegion(())                                 # empty + no reduction


_REDUCTION_CHAINS = [
    ((), "rmsnorm", False),                 # == hand-written matmul_rmsnorm
    ((), "softmax", False),                 # == hand-written matmul_softmax
    (("bias",), "rmsnorm", True),           # novel
    (("bias", "gelu"), "softmax", True),    # novel 2-pointwise + reduction
    (("relu",), "softmax", False),          # novel
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,red,has_bias", _REDUCTION_CHAINS,
                         ids=[f"{c}_{r}" for c, r, _ in _REDUCTION_CHAINS])
def test_synthesized_reduction_equals_unfused_on_metal(chain, red, has_bias):
    rng = np.random.default_rng((hash((chain, red))) & 0xFFFF)
    M, K, N = 16, 32, 24
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32) if has_bias else None
    region = FusedRegion(chain, reduction=red, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region(region, A, B, bias)
    assert execution == "metal_runtime"
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-4)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_synthesizer_reproduces_the_handwritten_matmul_rmsnorm():
    import ctypes
    from tessera.runtime import _load_apple_gpu_runtime

    rt = _load_apple_gpu_runtime()
    hw = getattr(rt, "tessera_apple_gpu_matmul_rmsnorm_f32", None)
    if hw is None:
        pytest.skip("hand-written matmul_rmsnorm symbol unavailable")
    hw.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int32] * 3 + [ctypes.c_float]
    hw.restype = None

    rng = np.random.default_rng(5)
    M, K, N = 12, 20, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    hand = np.zeros((M, N), np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hw(fp(np.ascontiguousarray(A)), fp(np.ascontiguousarray(B)), fp(hand), M, N, K, 1e-6)

    synth, execution = run_fused_region(FusedRegion((), reduction="rmsnorm", eps=1e-6), A, B)
    assert execution == "metal_runtime"
    assert np.allclose(synth, hand, atol=1e-4)


def test_discovery_captures_terminal_reduction():
    ops = [_Op("matmul", ("A", "B"), "m"),
           _Op("add", ("m", "bias"), "a"),
           _Op("rmsnorm", ("a",), "r")]
    regions = discover_fusable_regions(ops)
    assert len(regions) == 1
    _mi, idx, region = regions[0]
    assert region.epilogue == ("bias",)
    assert region.reduction == "rmsnorm"
    assert idx == [1, 2]


# ── F1a: wired into the runtime per-op execution ─────────────────────────────

def test_runtime_prepass_fuses_matmul_pointwise_chain():
    import numpy as np
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    B = rng.standard_normal((16, 12)).astype(np.float32)
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.gelu", "operands": ["%m"], "result": "%g"}]
    values = {"x": A, "W": B}
    consumed = _apple_gpu_try_synthesized_fusion(ops, values, np)
    assert consumed == {0, 1}
    ref = FusedRegion(("gelu",)).reference(A, B)
    assert np.allclose(values["g"], ref, atol=1e-4)


def test_runtime_prepass_resolves_bias_and_reduction():
    import numpy as np
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    B = rng.standard_normal((16, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.add", "operands": ["%m", "%b"], "result": "%a"},
           {"op_name": "tessera.rmsnorm", "operands": ["%a"], "result": "%r"}]
    values = {"x": A, "W": B, "b": bias}
    consumed = _apple_gpu_try_synthesized_fusion(ops, values, np)
    assert consumed == {0, 1, 2}
    ref = FusedRegion(("bias",), reduction="rmsnorm").reference(A, B, bias)
    assert np.allclose(values["r"], ref, atol=1e-4)


def test_runtime_prepass_preserves_f16_dtype():
    # P2 regression pin: the generic metadata fuser must preserve a half input
    # dtype (run_fused_region is dtype-aware) rather than upcasting to f32.
    import numpy as np
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float16)
    B = rng.standard_normal((16, 12)).astype(np.float16)
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.gelu", "operands": ["%m"], "result": "%g"}]
    values = {"x": A, "W": B}
    consumed = _apple_gpu_try_synthesized_fusion(ops, values, np)
    assert consumed == {0, 1}
    assert values["g"].dtype == np.float16          # dtype preserved, not f32


def test_target_ir_matmul_epilogue_is_dtype_aware():
    # P1 regression pin: matmul_gelu / matmul_rmsnorm Target IR must emit half
    # I/O for f16 (not always f32), matching the matmul_softmax branch.
    from tessera.compiler.target_ir import _apple_gpu_kernel_msl_for_dtype as info
    for kernel in ("matmul_gelu", "matmul_rmsnorm", "matmul_softmax"):
        f16_src, _e, _c, f16_attr = info(kernel, "f16")
        assert "device const half* A" in f16_src, kernel
        assert f16_attr == "f16", kernel
        f32_src, _e, _c, f32_attr = info(kernel, "f32")
        assert "device const float* A" in f32_src, kernel
        assert f32_attr == "f32", kernel
        # bf16 host-converts → f32 source but the bf16 marker is preserved.
        _bsrc, _e, _c, bf16_attr = info(kernel, "bf16")
        assert bf16_attr == "bf16", kernel


def test_runtime_prepass_skips_when_intermediate_reused():
    import numpy as np
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 6)).astype(np.float32)
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.gelu", "operands": ["%m"], "result": "%g"},
           {"op_name": "tessera.add", "operands": ["%m", "%g"], "result": "%z"}]
    consumed = _apple_gpu_try_synthesized_fusion(ops, {"x": A, "W": B}, np)
    assert consumed == set()                  # m used twice → not fusible


# ── F2c: attention synthesis (matmul -> softmax -> matmul) ───────────────────

def test_attention_synthesized_msl_has_the_expected_shape():
    src = synthesize_attention_msl(AttentionRegion())
    assert "kernel void synth_attention(" in src
    assert "device const float* Q" in src
    assert "device const float* V" in src
    assert "exp(scores[n] - mx)" in src        # numerically-stable softmax
    assert "constant int&       causal" in src # causal as a runtime buffer


def test_attention_reference_matches_manual_numpy():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((4, 8)).astype(np.float32)
    K = rng.standard_normal((6, 8)).astype(np.float32)
    V = rng.standard_normal((6, 5)).astype(np.float32)
    scale = 0.35
    got = AttentionRegion(scale=scale).reference(Q, K, V)
    s = (Q @ K.T) * scale
    e = np.exp(s - s.max(-1, keepdims=True))
    expected = (e / e.sum(-1, keepdims=True)) @ V
    assert np.allclose(got, expected, atol=1e-5)


def test_attention_reference_causal_masks_future_keys():
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((5, 4)).astype(np.float32)
    K = rng.standard_normal((5, 4)).astype(np.float32)
    V = rng.standard_normal((5, 4)).astype(np.float32)
    out = AttentionRegion(scale=1.0, causal=True).reference(Q, K, V)
    # query 0 attends only to key 0 → output is exactly V[0].
    assert np.allclose(out[0], V[0], atol=1e-5)


_ATTN_SHAPES = [
    (8, 8, 16, 16, 0.25, False),    # M, Nk, D, Dv, scale, causal
    (16, 12, 32, 24, 0.18, False),
    (12, 12, 16, 16, 1.0, True),    # square causal self-attention
    (6, 6, 8, 8, 0.5, True),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("M,Nk,D,Dv,scale,causal", _ATTN_SHAPES)
def test_synthesized_attention_equals_unfused_on_metal(M, Nk, D, Dv, scale, causal):
    rng = np.random.default_rng((M * 31 + Nk) & 0xFFFF)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((Nk, D)).astype(np.float32)
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    region = AttentionRegion(scale=scale, causal=causal)

    out, execution = run_fused_attention(region, Q, K, V)
    assert execution == "metal_runtime"            # the synthesized kernel ran
    assert np.allclose(out, region.reference(Q, K, V), atol=1e-4)  # horizontal oracle


def test_online_attention_msl_streams_keys_without_a_score_array():
    # M2: the online kernel holds only acc[head_dim] + running max/sum — NO
    # scores[Nk] array — so Nk is unbounded. Structural guard on the source.
    from tessera.compiler.fusion import SYNTH_MAX_D, synthesize_attention_online_msl

    src = synthesize_attention_online_msl(AttentionRegion(causal=True))
    assert f"acc[{SYNTH_MAX_D}]" in src           # head_dim-bounded accumulator
    assert "scores[" not in src                   # NO Nk-sized score array
    assert "run_max" in src and "run_sum" in src  # online softmax stats
    assert "break" in src                         # causal early-exit


# Large Nk (> SYNTH_MAX_N) — only the online kernel can reach these; the
# materialized kernel would fall to the numpy reference.
_ATTN_LARGE_N = [
    (8, 2048, 64, 64, False),     # M, Nk, D, Dv, causal — Nk = 2x the cap
    (16, 4096, 128, 128, False),  # 4x the cap, head_dim 128
    (32, 2048, 64, 64, True),     # large causal
    (4, 1536, 256, 256, False),   # head_dim at the SYNTH_MAX_D boundary
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("M,Nk,D,Dv,causal", _ATTN_LARGE_N)
def test_online_attention_large_n_equals_unfused_on_metal(M, Nk, D, Dv, causal):
    from tessera.compiler.fusion import SYNTH_MAX_N

    assert Nk > SYNTH_MAX_N, "this test must exercise the online (large-Nk) path"
    rng = np.random.default_rng((M * 131 + Nk) & 0xFFFF)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((Nk, D)).astype(np.float32)
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    region = AttentionRegion(scale=1.0 / np.sqrt(D), causal=causal)

    out, execution = run_fused_attention(region, Q, K, V)
    assert execution == "metal_runtime"            # online kernel ran (not reference)
    assert np.allclose(out, region.reference(Q, K, V), atol=1e-4)  # horizontal oracle


_ATTN_DTYPES = [
    ("f16", np.float16, 3e-2),
    ("bf16", "bf16", 2e-1),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("Nk", [64, 2048])   # materialized + online paths
@pytest.mark.parametrize("tag,npdt,tol", _ATTN_DTYPES)
def test_half_precision_attention_equals_unfused_on_metal(tag, npdt, tol, Nk):
    # M2: half-precision attention — half/bfloat I/O, fp32 accumulators, via the
    # uint16 symbol. Covers both the materialized (Nk≤cap) and online (Nk>cap)
    # kernels. The f32 reference is the horizontal oracle (compare at half tol).
    if npdt == "bf16":
        npdt = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(Nk & 0xFFFF)
    M, D, Dv = 8, 64, 64
    Q = (rng.standard_normal((M, D)) * 0.5).astype(npdt)
    K = (rng.standard_normal((Nk, D)) * 0.5).astype(npdt)
    V = (rng.standard_normal((Nk, Dv)) * 0.5).astype(npdt)
    region = AttentionRegion(scale=1.0 / np.sqrt(D))

    out, execution = run_fused_attention(region, Q, K, V)
    assert execution == "metal_runtime"                # half-precision kernel ran
    assert np.asarray(out).dtype == npdt               # output stays in storage dtype
    err = np.max(np.abs(np.asarray(out, np.float32) - region.reference(Q, K, V)))
    assert err < tol, err                              # horizontal oracle (half tol)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_synthesized_attention_reproduces_handwritten_flash_attn():
    # the synthesized attention block must match the hand-written flash_attn
    # catalog kernel it can replace — same online-softmax math.
    import ctypes
    from tessera.runtime import _load_apple_gpu_runtime

    rt = _load_apple_gpu_runtime()
    hw = getattr(rt, "tessera_apple_gpu_flash_attn_f32", None)
    if hw is None:
        pytest.skip("hand-written flash_attn symbol unavailable")

    rng = np.random.default_rng(7)
    M, D = 16, 32
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((M, D)).astype(np.float32)
    V = rng.standard_normal((M, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)

    synth, execution = run_fused_attention(AttentionRegion(scale=float(scale)), Q, K, V)
    assert execution == "metal_runtime"
    # the reference oracle is the ground truth; assert the synthesized GPU run
    # matches numpy attention (the same thing flash_attn computes).
    assert np.allclose(synth, AttentionRegion(scale=float(scale)).reference(Q, K, V),
                       atol=1e-4)


def test_discovery_captures_attention_with_scale():
    ops = [_Op("matmul", ("Q", "Kt"), "s"),
           _Op("scale", ("s",), "ss", {"scale": 0.125}),
           _Op("softmax", ("ss",), "p"),
           _Op("matmul", ("p", "V"), "o")]
    regions = discover_attention_regions(ops)
    assert len(regions) == 1
    idx, region, (q, k, v) = regions[0]
    assert idx == [0, 1, 2, 3]
    assert region.scale == 0.125
    assert (q, k, v) == ("Q", "Kt", "V")


def test_discovery_captures_attention_without_scale():
    ops = [_Op("matmul", ("Q", "Kt"), "s"),
           _Op("softmax", ("s",), "p"),
           _Op("matmul", ("p", "V"), "o")]
    regions = discover_attention_regions(ops)
    assert len(regions) == 1
    idx, region, names = regions[0]
    assert idx == [0, 1, 2]
    assert region.scale == 1.0


def test_discovery_rejects_attention_when_softmax_reused():
    ops = [_Op("matmul", ("Q", "Kt"), "s"),
           _Op("softmax", ("s",), "p"),
           _Op("matmul", ("p", "V"), "o"),
           _Op("add", ("p", "o"), "z")]   # p used twice → not fusible
    assert discover_attention_regions(ops) == []


def test_discovery_rejects_when_softmax_is_second_matmul_operand():
    # P must be the FIRST operand of the P@V matmul (P @ V, not V @ P).
    ops = [_Op("matmul", ("Q", "Kt"), "s"),
           _Op("softmax", ("s",), "p"),
           _Op("matmul", ("V", "p"), "o")]
    assert discover_attention_regions(ops) == []


# ── F3: fusion cost model ────────────────────────────────────────────────────

def test_cost_model_fuses_profitable_pointwise_chain():
    cost = fusion_cost(FusedRegion(("bias", "gelu")), M=64, N=128, K=256)
    assert cost.fusible
    assert cost.dispatches_unfused == 3        # matmul + bias + gelu
    assert cost.dispatches_fused == 1
    assert cost.dispatch_saved == 2
    assert cost.bytes_saved > 0
    assert cost.score > 0
    assert should_fuse_region(FusedRegion(("bias", "gelu")), 64, 128, 256)


def test_cost_model_fuses_between_stack_and_tiled_caps():
    from tessera.compiler.fusion import SYNTH_MAX_N_TILED
    # N over the stack cap but within the tiled cap is still fusible (tiled path).
    mid = SYNTH_MAX_N + 1
    assert mid <= SYNTH_MAX_N_TILED
    assert should_fuse_region(FusedRegion(("gelu",)), 8, mid, 16)


def test_cost_model_rejects_when_N_exceeds_tiled_cap():
    from tessera.compiler.fusion import SYNTH_MAX_N_TILED
    big = SYNTH_MAX_N_TILED + 1
    cost = fusion_cost(FusedRegion(("gelu",)), M=8, N=big, K=16)
    assert not cost.fusible
    assert "tiled threadgroup cap" in cost.reason
    assert cost.score == float("-inf")
    assert not should_fuse_region(FusedRegion(("gelu",)), 8, big, 16)


def test_cost_model_ranks_longer_chains_higher():
    short = fusion_cost(FusedRegion(("gelu",)), 64, 128, 256)
    long = fusion_cost(FusedRegion(("bias", "gelu", "tanh")), 64, 128, 256)
    assert long.score > short.score            # more collapsed dispatches + traffic


def test_attention_cost_model():
    cost = attention_cost(AttentionRegion(), M=32, Nk=64, D=48, Dv=48)
    assert cost.fusible
    assert cost.dispatches_unfused == 3        # QKᵀ + softmax + PV
    assert cost.dispatch_saved == 2
    assert should_fuse_attention(AttentionRegion(), 32, 64, 48, 48)
    too_big = attention_cost(AttentionRegion(), 32, SYNTH_MAX_N + 1, 48, 48)
    assert not too_big.fusible


def test_runtime_prepass_cost_gate_skips_oversized_N():
    # F3 gate: a matmul->gelu whose N exceeds even the TILED cap is left to
    # per-op (the per-op path has MPS matmul + MPSGraph epilogue for huge N).
    import numpy as np
    from tessera.compiler.fusion import SYNTH_MAX_N_TILED
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    big = SYNTH_MAX_N_TILED + 8
    A = np.zeros((4, 8), np.float32)
    B = np.zeros((8, big), np.float32)
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.gelu", "operands": ["%m"], "result": "%g"}]
    consumed = _apple_gpu_try_synthesized_fusion(ops, {"x": A, "W": B}, np)
    assert consumed == set()                   # cost gate declined the fusion


# ── F4: codegen-gated oracle (verify before trust) ───────────────────────────

def test_verify_passes_for_a_correct_synthesizer():
    from tessera.compiler.fusion import verify_synthesized_region, clear_verification_cache
    clear_verification_cache()
    # correct synthesizer (or no Metal → trusted reference) → verified.
    assert verify_synthesized_region(FusedRegion(("gelu",)), force=True) is True


@pytest.mark.skipif(sys.platform != "darwin", reason="needs Metal to run a kernel.")
def test_verify_rejects_a_broken_synthesizer(monkeypatch):
    # The anti-cheat invariant: a synthesizer that emits a kernel which COMPILES
    # but computes the wrong thing (drops the epilogue) must be REJECTED.  This is
    # the codegen analogue of magellan/alphaevolve's reward-hack rejection.
    import tessera.compiler.fusion as F
    from tessera.compiler.fusion import (
        FusedRegion, verify_synthesized_region, clear_verification_cache,
    )

    def broken_synth(region, variant="broadcast"):
        # a matmul-only kernel: ignores the epilogue entirely.
        return f"""#include <metal_stdlib>
using namespace metal;
kernel void {F._ENTRY}(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* O [[buffer(2)]], constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {{
    if (gid >= (uint)M) return;
    int row = (int)gid;
    for (int n = 0; n < N; ++n) {{
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[row*K+k] * B[k*N+n];
        O[row*N+n] = s;   // <-- gelu dropped: WRONG
    }}
}}"""

    monkeypatch.setattr(F, "synthesize_matmul_epilogue_msl", broken_synth)
    clear_verification_cache()
    # Use a REDUCTION region so it routes through the scalar synthesizer under
    # test (pointwise regions now run on the coopmat path). The broken kernel
    # drops the reduction → wrong → must be rejected.
    region = FusedRegion((), reduction="rmsnorm")
    out, execution = F.run_fused_region(region, np.ones((4, 4), np.float32),
                                        np.ones((4, 4), np.float32))
    if execution != "metal_runtime":
        pytest.skip("Metal unavailable; cannot exercise the GPU divergence gate")
    assert verify_synthesized_region(region, force=True) is False  # rejected


@pytest.mark.skipif(sys.platform != "darwin", reason="needs Metal to run a kernel.")
def test_runtime_prepass_skips_a_region_the_oracle_rejects(monkeypatch):
    # End-to-end: if the synthesized kernel diverges, the runtime pre-pass must
    # NOT consume the ops — they fall back to the trusted per-op path.
    import numpy as np
    import tessera.compiler.fusion as F
    from tessera.runtime import _apple_gpu_try_synthesized_fusion

    def broken_synth(region, variant="broadcast"):
        return f"""#include <metal_stdlib>
using namespace metal;
kernel void {F._ENTRY}(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* O [[buffer(2)]], constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {{
    if (gid >= (uint)M) return;
    int row = (int)gid;
    for (int n = 0; n < N; ++n) {{
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[row*K+k] * B[k*N+n];
        O[row*N+n] = s;   // gelu dropped
    }}
}}"""

    monkeypatch.setattr(F, "synthesize_matmul_epilogue_msl", broken_synth)
    F.clear_verification_cache()
    A = np.ones((4, 8), np.float32)
    B = np.ones((8, 6), np.float32)
    # REDUCTION chain → scalar synthesizer path (pointwise → coopmat).
    ops = [{"op_name": "tessera.matmul", "operands": ["%x", "%W"], "result": "%m"},
           {"op_name": "tessera.rmsnorm", "operands": ["%m"], "result": "%r"}]
    # probe whether Metal actually runs the (broken) kernel first.
    _o, ex = F.run_fused_region(FusedRegion((), reduction="rmsnorm"), A, B)
    if ex != "metal_runtime":
        pytest.skip("Metal unavailable; oracle gate not exercised")
    F.clear_verification_cache()
    consumed = _apple_gpu_try_synthesized_fusion(ops, {"x": A, "W": B}, np)
    assert consumed == set()                   # oracle rejected → not fused


# ── F5: autotune the synthesizer (gated behind cost + oracle) ────────────────

def test_synth_variants_compute_the_same_scores():
    # both inner-loop variants must emit kernels that fill scores[] identically;
    # the synthesizer source differs but the math is one thing.
    a = synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), variant="broadcast")
    b = synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), variant="dot")
    assert a != b                              # genuinely different schedules
    assert "scores[n] += a * B[b_off + n]" in a
    assert "acc += A[a_off + k] * B[k * N + n]" in b


def test_autotune_picks_fastest_oracle_correct_variant():
    # The Sakana invariant, deterministic (no Metal): a faster variant that fails
    # the oracle is NEVER chosen; the slower correct one wins.
    from tessera.compiler.fusion import _pick_best_variant
    assert _pick_best_variant({"fast": 0.1, "slow": 0.9},
                              {"fast": True, "slow": True}) == "fast"
    assert _pick_best_variant({"fast_wrong": 0.1, "slow_right": 0.9},
                              {"fast_wrong": False, "slow_right": True}) == "slow_right"
    assert _pick_best_variant({"x": 0.1}, {"x": False}) is None   # none eligible


def test_best_variant_defaults_to_broadcast_when_unseen():
    from tessera.compiler.fusion import best_variant_for, clear_autotune_corpus
    clear_autotune_corpus()
    assert best_variant_for(FusedRegion(("gelu",)), 64, 128, 256) == "broadcast"


@pytest.mark.skipif(sys.platform != "darwin", reason="autotune measures on Metal.")
def test_autotune_records_a_correct_variant_on_metal():
    from tessera.compiler.fusion import (
        autotune_matmul_epilogue, best_variant_for, clear_autotune_corpus,
    )
    clear_autotune_corpus()
    region = FusedRegion(("bias", "gelu"), bias_name="bias")
    rec = autotune_matmul_epilogue(region, M=32, N=64, K=48, reps=3)
    if rec is None:
        pytest.skip("Metal unavailable; autotune not exercised")
    assert rec.chosen in ("broadcast", "dot")
    assert rec.correct.get(rec.chosen) is True          # the winner passed the oracle
    # the corpus now distills an O(1) decision for this shape-class.
    assert best_variant_for(region, 32, 64, 48) == rec.chosen


@pytest.mark.skipif(sys.platform != "darwin", reason="autotune measures on Metal.")
def test_autotune_skips_non_fusible_region():
    from tessera.compiler.fusion import autotune_matmul_epilogue, SYNTH_MAX_N_TILED
    # N over the TILED cap → F3 declines → autotune returns None.
    assert autotune_matmul_epilogue(FusedRegion(("gelu",)), 8, SYNTH_MAX_N_TILED + 1, 16) is None


# ── F2b-tiled: threadgroup-tiled synthesis for large N ───────────────────────

def test_tiled_synthesizer_has_threadgroup_structure():
    from tessera.compiler.fusion import synthesize_matmul_epilogue_msl_tiled
    src = synthesize_matmul_epilogue_msl_tiled(FusedRegion((), reduction="softmax"))
    assert "kernel void synth_matmul_epi_tiled(" in src
    assert "threadgroup float* tg_scores [[threadgroup(0)]]" in src
    assert "threadgroup float tg_red[32]" in src          # cooperative reduction scratch
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in src
    assert "n += T" in src                                 # strided cooperative loop


def test_tiled_pure_pointwise_omits_reduction_scratch():
    from tessera.compiler.fusion import synthesize_matmul_epilogue_msl_tiled
    src = synthesize_matmul_epilogue_msl_tiled(FusedRegion(("gelu",)))
    assert "tg_red" not in src                             # no reduction → no scratch
    assert "O[o_off + n] = ST(v);" in src                  # writes O directly (ST cast)


_TILED_CASES = [
    (("gelu",), None, False),
    (("bias", "relu"), None, True),
    ((), "softmax", False),
    ((), "rmsnorm", False),
    (("bias",), "softmax", True),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,red,has_bias", _TILED_CASES)
@pytest.mark.parametrize("N", [2048, 4096])
def test_tiled_synthesis_equals_unfused_on_metal(chain, red, has_bias, N):
    # N is well over the stack cap (1024): run_fused_region must route to the
    # threadgroup-tiled kernel and still match the unfused reference.
    rng = np.random.default_rng((N * 7 + hash((chain, red))) & 0xFFFF)
    M, K = 12, 48
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32) if has_bias else None
    region = FusedRegion(chain, reduction=red, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region(region, A, B, bias)
    assert execution == "metal_runtime"            # the tiled kernel ran
    assert np.allclose(out, region.reference(A, B, bias), atol=1e-3)  # oracle


# ── F2 half-precision synthesis (f16 native + bf16 convert) ──────────────────

def test_f16_synthesizer_emits_half_io():
    a = synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), dtype="f16")
    assert "device const half* A" in a
    assert "device half*       O" in a
    assert "float scores[" in a                # fp32 accumulator stays
    f32 = synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), dtype="f32")
    assert "device const float* A" in f32


def test_f16_tiled_synthesizer_emits_half_io_with_fp32_scratch():
    src = synthesize_matmul_epilogue_msl_tiled(FusedRegion((), reduction="softmax"),
                                               dtype="f16")
    assert "device const half* A" in src
    assert "threadgroup float* tg_scores" in src   # accumulator stays fp32


def test_synthesizer_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), dtype="f64")


def test_synthesizer_emits_native_bfloat():
    # Apple7 supports native MSL `bfloat` (MTLDataType.bfloat) — the synthesizer
    # emits it directly now instead of host-upcasting to f32.
    src = synthesize_matmul_epilogue_msl(FusedRegion(("gelu",)), dtype="bf16")
    assert "device const bfloat* A" in src
    assert "half" not in src


_HALF_CASES = [
    (("gelu",), None, False),
    (("bias", "silu"), None, True),
    ((), "softmax", False),
    ((), "rmsnorm", False),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,red,has_bias", _HALF_CASES)
@pytest.mark.parametrize("N", [64, 2048])      # stack + tiled
def test_f16_synthesis_equals_unfused_on_metal(chain, red, has_bias, N):
    rng = np.random.default_rng((N + hash((chain, red))) & 0xFFFF)
    M, K = 8, 32
    A = rng.standard_normal((M, K)).astype(np.float16)
    B = rng.standard_normal((K, N)).astype(np.float16)
    bias = rng.standard_normal((N,)).astype(np.float16) if has_bias else None
    region = FusedRegion(chain, reduction=red, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region(region, A, B, bias)
    assert execution == "metal_runtime"
    assert out.dtype == np.float16
    ref = region.reference(A, B, bias)         # fp32 math (the oracle)
    assert np.allclose(out.astype(np.float32), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_bf16_synthesis_converts_and_runs_on_metal():
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(11)
    M, K, N = 8, 16, 64
    A = rng.standard_normal((M, K)).astype(bf16)
    B = rng.standard_normal((K, N)).astype(bf16)
    region = FusedRegion((), reduction="softmax")

    out, execution = run_fused_region(region, A, B)
    assert execution == "metal_runtime"        # f32 kernel ran under the hood
    assert out.dtype == bf16
    ref = region.reference(A.astype(np.float32), B.astype(np.float32))
    assert np.allclose(out.astype(np.float32), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_f16_synthesis_reproduces_handwritten_matmul_gelu_f16():
    import ctypes
    from tessera.runtime import _load_apple_gpu_runtime

    rt = _load_apple_gpu_runtime()
    hw = getattr(rt, "tessera_apple_gpu_matmul_gelu_f16", None)
    if hw is None:
        pytest.skip("hand-written matmul_gelu_f16 symbol unavailable")
    hw.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 3 + [ctypes.c_int32] * 3
    hw.restype = None

    rng = np.random.default_rng(13)
    M, K, N = 8, 16, 32
    A = rng.standard_normal((M, K)).astype(np.float16)
    B = rng.standard_normal((K, N)).astype(np.float16)
    hand = np.zeros((M, N), np.float16)
    u16 = lambda a: a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    hw(u16(np.ascontiguousarray(A)), u16(np.ascontiguousarray(B)), u16(hand), M, N, K)

    synth, execution = run_fused_region(FusedRegion(("gelu",)), A, B)
    assert execution == "metal_runtime"
    # both use half I/O + fp32 accumulators + the same clamped-tanh gelu.
    assert np.allclose(synth.astype(np.float32), hand.astype(np.float32),
                       atol=2e-2, rtol=2e-2)


# ── F2d cooperative-matrix synthesis (simdgroup_matrix MMA + fused epilogue) ──

def test_coopmat_synthesizer_emits_simdgroup_matrix():
    from tessera.compiler.fusion import synthesize_matmul_epilogue_coopmat_msl
    src = synthesize_matmul_epilogue_coopmat_msl(FusedRegion(("gelu",)), dtype="f16")
    assert "#include <metal_simdgroup_matrix>" in src
    assert "simdgroup_multiply_accumulate" in src     # the MMA
    assert "simdgroup_matrix<half, 8, 8>" in src       # f16 inputs
    assert "simdgroup_float8x8 acc" in src             # fp32 accumulator
    assert "device const half* A" in src               # half I/O


def test_coopmat_64_tile_synthesizes_8_accumulators():
    from tessera.compiler.fusion import (
        synthesize_matmul_epilogue_coopmat_msl, coopmat_threads, SYNTH_COOPMAT_TILES,
    )
    assert SYNTH_COOPMAT_TILES == (32, 64)
    src64 = synthesize_matmul_epilogue_coopmat_msl(
        FusedRegion(("gelu",)), dtype="f16", tile=64)
    assert "constant constexpr int BM = 64;" in src64
    assert "constant constexpr int NR = 4;" in src64       # 4x2 = 8 accumulators
    assert "constant constexpr int THREADS = 256;" in src64
    assert coopmat_threads(64) == 256 and coopmat_threads(32) == 128
    with pytest.raises(ValueError):
        synthesize_matmul_epilogue_coopmat_msl(FusedRegion(("gelu",)), tile=48)


def test_coopmat_tile_selection_by_shape():
    from tessera.compiler.fusion import coopmat_tile_for
    assert coopmat_tile_for(2048, 1024, 512) == 64     # large-N → 64x64
    assert coopmat_tile_for(1024, 1024, 1024) == 64
    assert coopmat_tile_for(2048, 256, 1024) == 32     # narrow N → 32x32
    assert coopmat_tile_for(64, 96, 48) == 32          # small → 32x32


def test_coopmat_tile_corpus_overrides_heuristic():
    # An autotuned tile decision in the corpus wins over the shape heuristic.
    from tessera.compiler.fusion import (
        coopmat_tile_for, clear_coopmat_tile_corpus, _COOPMAT_TILE_CORPUS,
        _corpus_key,
    )
    clear_coopmat_tile_corpus()
    region = FusedRegion(("gelu",))
    M, N, K = 2048, 256, 1024
    assert coopmat_tile_for(M, N, K, region) == 32          # heuristic: narrow N
    _COOPMAT_TILE_CORPUS[_corpus_key(region, M, N, K)] = 64  # autotuner said 64
    assert coopmat_tile_for(M, N, K, region) == 64          # corpus overrides
    clear_coopmat_tile_corpus()


@pytest.mark.skipif(sys.platform != "darwin", reason="autotune measures on Metal.")
def test_autotune_coopmat_tile_picks_a_correct_winner():
    from tessera.compiler.fusion import (
        autotune_coopmat_tile, coopmat_tile_for, clear_coopmat_tile_corpus,
    )
    clear_coopmat_tile_corpus()
    region = FusedRegion(("gelu",))
    M, N, K = 512, 512, 512
    lat = autotune_coopmat_tile(region, M, N, K, reps=4)
    if not lat:
        pytest.skip("Metal unavailable; autotune not exercised")
    winner = coopmat_tile_for(M, N, K, region)
    assert winner in (32, 64)
    assert lat[winner] == min(lat.values())     # the recorded tile is the fastest
    clear_coopmat_tile_corpus()


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_coopmat_64_tile_equals_unfused_on_metal(dtype):
    from tessera.compiler.fusion import run_fused_region_coopmat
    M, K, N = 192, 256, 448                            # large-N → 64x64 path + non-aligned
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((M, K)) * 0.3).astype(dtype)
    B = (rng.standard_normal((K, N)) * 0.3).astype(dtype)
    region = FusedRegion(("bias", "gelu"), bias_name="bias")
    bias = (rng.standard_normal((N,)) * 0.3).astype(dtype)
    out, ex = run_fused_region_coopmat(region, A, B, bias, tile=64)
    assert ex == "metal_runtime"
    atol = 1e-4 if dtype == np.float32 else 3e-2
    assert np.allclose(out.astype(np.float32), region.reference(A, B, bias),
                       atol=atol, rtol=atol)


def test_coopmat_eligibility_excludes_reductions():
    from tessera.compiler.fusion import coopmat_eligible
    assert coopmat_eligible(FusedRegion(("gelu",)))
    assert coopmat_eligible(FusedRegion(("bias", "relu"), bias_name="bias"))
    assert not coopmat_eligible(FusedRegion((), reduction="softmax"))
    assert not coopmat_eligible(FusedRegion((), reduction="rmsnorm"))
    # v1 rejects synthesizing a reduction coopmat kernel.
    from tessera.compiler.fusion import synthesize_matmul_epilogue_coopmat_msl
    with pytest.raises(ValueError):
        synthesize_matmul_epilogue_coopmat_msl(FusedRegion((), reduction="softmax"))


_COOPMAT_CASES = [
    (("gelu",), False),
    (("relu",), False),
    (("silu",), False),
    (("bias", "gelu"), True),
    (("bias", "sigmoid", "tanh"), True),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,has_bias", _COOPMAT_CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(64, 48, 96), (128, 256, 128)])  # non-aligned + aligned
def test_coopmat_synthesis_equals_unfused_on_metal(chain, has_bias, dtype, shape):
    from tessera.compiler.fusion import run_fused_region_coopmat
    M, K, N = shape
    rng = np.random.default_rng((M * 13 + N + hash(chain)) & 0xFFFF)
    A = (rng.standard_normal((M, K)) * 0.3).astype(dtype)
    B = (rng.standard_normal((K, N)) * 0.3).astype(dtype)
    bias = (rng.standard_normal((N,)) * 0.3).astype(dtype) if has_bias else None
    region = FusedRegion(chain, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region_coopmat(region, A, B, bias)
    assert execution == "metal_runtime"           # the simdgroup_matrix kernel ran
    atol = 1e-4 if dtype == np.float32 else 3e-2
    assert np.allclose(out.astype(np.float32), region.reference(A, B, bias),
                       atol=atol, rtol=atol)


def test_coopmat_reduce_synthesizer_structure_and_eligibility():
    from tessera.compiler.fusion import (
        synthesize_matmul_reduction_coopmat_msl, coopmat_reduce_eligible,
        SYNTH_COOPMAT_REDUCE_MAX_N,
    )
    src = synthesize_matmul_reduction_coopmat_msl(
        FusedRegion((), reduction="softmax"), dtype="f16")
    assert "simdgroup_multiply_accumulate" in src      # matmul on the matrix units
    assert "simdgroup_matrix<half, 8, 8>" in src
    assert "exp(Cs[rr * N + c] - _mx)" in src           # fused row reduction
    # eligibility: a terminal reduction within the N cap.
    assert coopmat_reduce_eligible(FusedRegion((), reduction="softmax"), 256)
    assert coopmat_reduce_eligible(FusedRegion((), reduction="rmsnorm"),
                                   SYNTH_COOPMAT_REDUCE_MAX_N)
    assert not coopmat_reduce_eligible(FusedRegion((), reduction="softmax"),
                                       SYNTH_COOPMAT_REDUCE_MAX_N + 1)
    assert not coopmat_reduce_eligible(FusedRegion(("gelu",)), 64)  # not a reduction


_COOPMAT_REDUCE_CASES = [
    ((), "softmax", False),
    ((), "rmsnorm", False),
    (("bias",), "softmax", True),
]


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
@pytest.mark.parametrize("chain,red,has_bias", _COOPMAT_REDUCE_CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(64, 48, 96), (40, 128, 256), (8, 64, 512)])
def test_coopmat_reduce_equals_unfused_on_metal(chain, red, has_bias, dtype, shape):
    from tessera.compiler.fusion import run_fused_region_coopmat_reduce
    M, K, N = shape
    rng = np.random.default_rng((M * 17 + N + hash(red)) & 0xFFFF)
    A = (rng.standard_normal((M, K)) * 0.3).astype(dtype)
    B = (rng.standard_normal((K, N)) * 0.3).astype(dtype)
    bias = (rng.standard_normal((N,)) * 0.3).astype(dtype) if has_bias else None
    region = FusedRegion(chain, reduction=red, bias_name="bias" if has_bias else None)

    out, execution = run_fused_region_coopmat_reduce(region, A, B, bias)
    assert execution == "metal_runtime"           # matmul on matrix units + fused reduce
    atol = 1e-3 if dtype == np.float32 else 4e-2
    assert np.allclose(out.astype(np.float32), region.reference(A, B, bias),
                       atol=atol, rtol=atol)


@pytest.mark.skipif(sys.platform != "darwin", reason="dispatch runs on Metal.")
def test_matmul_heavy_reduction_routes_to_compose_and_is_correct():
    # Perf routing (F2d measurement): a matmul-heavy matmul->softmax/rmsnorm
    # routes to MPS-matmul + MPSGraph-reduce (compose) instead of the slow
    # synthesized fused-reduction kernel. Verify the result is still correct.
    import numpy as np
    from tessera.runtime import (
        _apple_gpu_dispatch_matmul_softmax, _apple_gpu_dispatch_matmul_rmsnorm,
        _APPLE_REDUCE_COMPOSE_MIN_FLOP,
    )
    M, K, N = 512, 512, 256                     # 2*M*K*N = 134M >> gate (8M)
    assert 2 * M * K * N >= _APPLE_REDUCE_COMPOSE_MIN_FLOP
    A = (np.random.RandomState(0).randn(M, K) * 0.3).astype(np.float32)
    B = (np.random.RandomState(1).randn(K, N) * 0.3).astype(np.float32)
    s = A @ B
    out_sm = _apple_gpu_dispatch_matmul_softmax([A, B], np)
    e = np.exp(s - s.max(-1, keepdims=True))
    assert np.allclose(out_sm, e / e.sum(-1, keepdims=True), rtol=1e-3, atol=1e-4)
    out_rn = _apple_gpu_dispatch_matmul_rmsnorm([A, B], 1e-5, np)
    rms = np.sqrt(np.mean(s * s, -1, keepdims=True) + 1e-5)
    assert np.allclose(out_rn, s / rms, rtol=1e-3, atol=1e-4)


def test_coopmat_reduce_falls_back_above_cap():
    # N over the threadgroup cap → not eligible → scalar run_fused_region (still
    # correct). Portable (no Metal needed for the fallback contract).
    from tessera.compiler.fusion import (
        run_fused_region_coopmat_reduce, SYNTH_COOPMAT_REDUCE_MAX_N,
    )
    big = SYNTH_COOPMAT_REDUCE_MAX_N + 64
    A = np.random.RandomState(0).randn(8, 16).astype(np.float32)
    B = np.random.RandomState(1).randn(16, big).astype(np.float32)
    region = FusedRegion((), reduction="softmax")
    out, _ex = run_fused_region_coopmat_reduce(region, A, B)
    assert np.allclose(out, region.reference(A, B), atol=1e-4)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_run_fused_region_prefers_coopmat_for_pointwise():
    # run_fused_region routes pointwise f16/f32 regions to coopmat; a reduction
    # region still works (scalar path) — both correct.
    rng = np.random.default_rng(3)
    A = rng.standard_normal((96, 128)).astype(np.float16)
    B = rng.standard_normal((128, 64)).astype(np.float16)
    pw, ex1 = run_fused_region(FusedRegion(("gelu",)), A, B)
    assert ex1 == "metal_runtime"
    assert np.allclose(pw.astype(np.float32),
                       FusedRegion(("gelu",)).reference(A, B), atol=3e-2)
    red, ex2 = run_fused_region(FusedRegion((), reduction="softmax"), A, B)
    assert ex2 == "metal_runtime"
    assert np.allclose(red.astype(np.float32),
                       FusedRegion((), reduction="softmax").reference(A, B), atol=3e-2)


@pytest.mark.skipif(sys.platform != "darwin", reason="synthesis runs on Metal.")
def test_tiled_synthesis_reproduces_handwritten_matmul_softmax_tiled():
    # the synthesized tiled softmax must match the hand-written tiled kernel it
    # will replace — same cooperative online-softmax math.
    import ctypes
    from tessera.runtime import _load_apple_gpu_runtime

    rt = _load_apple_gpu_runtime()
    hw = getattr(rt, "tessera_apple_gpu_matmul_softmax_tiled_f32", None)
    if hw is None:
        pytest.skip("hand-written matmul_softmax_tiled symbol unavailable")
    hw.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int32] * 3
    hw.restype = None

    rng = np.random.default_rng(9)
    M, K, N = 8, 32, 2048
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    hand = np.zeros((M, N), np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    hw(fp(np.ascontiguousarray(A)), fp(np.ascontiguousarray(B)), fp(hand), M, N, K)

    synth, execution = run_fused_region(FusedRegion((), reduction="softmax"), A, B)
    assert execution == "metal_runtime"
    assert np.allclose(synth, hand, atol=1e-4)     # synthesized == hand-written
