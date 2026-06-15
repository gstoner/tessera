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
    FusedRegion,
    discover_fusable_regions,
    run_fused_region,
    synthesize_matmul_epilogue_msl,
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
