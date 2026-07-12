"""Workstream C2 — NVIDIA (sm_120) plugin: generic synth → CUDA.

Two layers:

1. **Registration + emit paths (host-free)** — a full three-seam plugin for
   target "nvidia": the emitter turns fused / attention / gated / pointwise regions
   into CUDA source; unsupported regions/policies/dtypes raise EmitError.
2. **Live gates (needs a live NVIDIA GPU + nvcc)** — the generically-synthesized
   CUDA kernels compile with nvcc, run on-device ("nvidia_cuda"), match numpy, and
   pass the same universal F4 oracle as ROCm/x86. The generic lane covers all four
   fusion_core region kinds (FusedRegion, AttentionRegion — C4, GatedMatmulRegion +
   PointwiseGraphRegion — C5); bare GEMM is served by the B1 matmul candidates
   (shipped + emitted mma.sync).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera.compiler.fusion as F
import tessera.compiler.emit.nvidia_cuda as nvidia  # noqa: F401 — self-registers
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import OP_FUSED_REGION, OP_MATMUL, Tier
from tessera.compiler.emit.kernel_emitter import (
    EmitError, SpecPolicy, get_emitter, get_runner,
)


def _nvidia_cuda_live() -> bool:
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return False
    try:
        from tessera import runtime as rt
        return rt._nvidia_mma_runtime_available()
    except Exception:
        return False


# ── 1. Registration + emit + decline paths (host-free) ────────────────────────

def test_nvidia_runner_registered():
    r = get_runner("nvidia")
    assert r.target == "nvidia"
    assert r.accuracy_atol is None          # f32 lane — exact, no budget widening


def test_nvidia_emitter_registered_produces_cuda():
    from tessera.compiler.emit.kernel_cache import get_compiler
    src = get_emitter("nvidia").emit(F.FusedRegion(epilogue=("bias", "gelu")),
                                     dtype="f32")
    assert src.lang == "cuda"
    assert src.entry == "tessera_nvidia_fused"
    assert "__global__" in src.source and 'extern "C"' in src.source
    assert "<<<" in src.source and "cudaMemcpy" in src.source
    assert callable(get_compiler("nvidia"))


def test_nvidia_emit_is_deterministic():
    # Byte-identical emit for the same region (golden-IR / cache-key discipline).
    e = get_emitter("nvidia")
    r = F.FusedRegion(epilogue=("relu",), reduction="softmax")
    assert e.emit(r).source == e.emit(r).source


def test_nvidia_emitter_rejects_unsupported():
    e = get_emitter("nvidia")
    with pytest.raises(EmitError, match="cannot emit"):
        e.emit(F.MatmulRegion())                   # bare GEMM → the GEMM candidates
    with pytest.raises(EmitError, match="f32"):
        e.emit(F.FusedRegion(epilogue=("relu",)), dtype="f16")
    # DYNAMIC is now supported (W2): the runtime-arg lane is dims-invariant, so it
    # emits the same source as BUCKET (one kernel serves every shape).
    dyn = e.emit(F.FusedRegion(epilogue=("relu",)), spec=SpecPolicy.DYNAMIC)
    assert dyn.spec is SpecPolicy.DYNAMIC


def test_nvidia_c5_candidates_registered_and_emit():
    # C5: the generic lane now covers gated (SwiGLU) + pointwise-DAG region kinds
    # too — Tier-1 candidates + emitter support (attention landed in C4).
    from tessera.compiler.emit.candidate import OP_GATED_MATMUL, OP_POINTWISE
    gated = {c.name for c in C.candidates_for("nvidia", OP_GATED_MATMUL)}
    pw = {c.name for c in C.candidates_for("nvidia", OP_POINTWISE)}
    assert "nvidia_gated" in gated and "nvidia_pointwise" in pw
    e = get_emitter("nvidia")
    gsrc = e.emit(F.GatedMatmulRegion(gate_act="silu")).source
    assert "tessera_nvidia_gated" in gsrc and "<<<" in gsrc
    region = F.PointwiseGraphRegion(
        ops=(("add", ("a", "b"), "s"), ("relu", ("s",), "o")),
        inputs=("a", "b"), output="o")
    psrc = e.emit(region).source
    assert "tessera_nvidia_pointwise" in psrc and "sign" in psrc   # sign shim


def test_nvidia_flash_attn_candidate_registered_and_emits():
    # C4: the synthesized flash-attention lane — a Tier-1 attention candidate, and
    # the emitter produces CUDA for an AttentionRegion.
    from tessera.compiler.emit.candidate import OP_ATTENTION
    cands = {c.name: c for c in C.candidates_for("nvidia", OP_ATTENTION)}
    assert "nvidia_flash_attn" in cands
    assert cands["nvidia_flash_attn"].tier == Tier.SYNTHESIZED
    src = get_emitter("nvidia").emit(F.AttentionRegion(scale=0.25)).source
    assert "tessera_nvidia_attn" in src and "expf" in src and "<<<" in src


def test_nvidia_mma_attn_candidate_registered():
    # Tensor-core attention perf lane: Tier-2 emitted, f16 accuracy budget.
    from tessera.compiler.emit.candidate import OP_ATTENTION
    cands = {c.name: c for c in C.candidates_for("nvidia", OP_ATTENTION)}
    assert "nvidia_mma_attn" in cands
    mf = cands["nvidia_mma_attn"]
    assert mf.tier == Tier.EMITTED and mf.accuracy_atol == 5e-3
    assert mf.op == OP_ATTENTION


def test_nvidia_mma_fused_candidate_registered_and_applies():
    # Tensor-core perf lane: a Tier-2 emitted FusedRegion candidate (mma.sync GEMM
    # + bias/activation epilogue), f16 storage → f16 accuracy budget.
    cands = {c.name: c for c in C.candidates_for("nvidia", OP_FUSED_REGION)}
    assert "nvidia_mma_fused" in cands
    mf = cands["nvidia_mma_fused"]
    assert mf.tier == Tier.EMITTED and mf.accuracy_atol == 5e-3
    # fusable: bias?, one activation, no reduction/residual/prologue.
    assert mf.applies_to(F.FusedRegion(epilogue=("bias", "gelu")))
    assert mf.applies_to(F.FusedRegion(epilogue=("relu",)))
    assert not mf.applies_to(F.FusedRegion(epilogue=("bias",), reduction="softmax"))
    assert not mf.applies_to(F.FusedRegion(epilogue=("relu",), residual=True))
    assert not mf.applies_to(F.FusedRegion(epilogue=("gelu",), prologue=("relu",)))


def test_nvidia_mma_fused_rejects_bad_inputs():
    # PR #301 review: a mismatched contraction dim or a short bias must raise (like
    # FusedRegion.reference), not launch the C ABI on an overreading buffer.
    # Host-free — validated before any GPU work.
    mf = next(c for c in C.candidates_for("nvidia", OP_FUSED_REGION)
              if c.name == "nvidia_mma_fused")
    region = F.FusedRegion(epilogue=("bias", "relu"))
    A = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError):                 # K 24 != A's K 16
        mf.run(region, A, np.zeros((24, 8), np.float32), np.zeros((8,), np.float32))
    with pytest.raises(ValueError):                 # bias len 7 != N 8
        mf.run(region, A, np.zeros((16, 8), np.float32), np.zeros((7,), np.float32))


def test_nvidia_generic_candidate_registered():
    cands = C.candidates_for("nvidia", OP_FUSED_REGION)
    names = [c.name for c in cands]
    assert "nvidia_generic_cuda" in names
    gen = next(c for c in cands if c.name == "nvidia_generic_cuda")
    assert gen.tier == Tier.SYNTHESIZED and gen.target == "nvidia"


def test_nvidia_arbitrated_residual_threads_not_raises():
    # PR #290 review: a residual FusedRegion routed through run_arbitrated passes
    # inputs positionally (A, B, bias, residual). The candidate must thread the
    # residual (not drop it into *a → missing-buffer guard → ValueError). Host-free:
    # off-GPU the candidate declines and the arbiter falls back to the numpy
    # reference — the point is it returns a correct result, never raises.
    region = F.FusedRegion(epilogue=("relu",), residual=True)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    res = rng.standard_normal((8, 16)).astype(np.float32)
    out, tag = C.run_arbitrated(region, OP_FUSED_REGION, "nvidia", A, B, None, res)
    np.testing.assert_allclose(out, region.reference(A, B, None, res), atol=1e-2)


def test_nvidia_gemm_helpers_reject_mismatched_k():
    # PR #294 review: a mismatched contraction dim (A is MxK, B is K2xN, K != K2)
    # must raise before the C ABI overreads B — validated ahead of the lib load,
    # so this is host-free (like MatmulRegion.reference / the JIT path).
    from tessera import runtime as rt
    A = np.zeros((32, 16), np.float32)
    B = np.zeros((24, 8), np.float32)          # K 24 != A's K 16
    with pytest.raises(ValueError, match="matching K"):
        rt._nvidia_mma_gemm_2d(A, B, "bfloat16")
    with pytest.raises(ValueError, match="matching K"):
        rt._nvidia_ptx_gemm_2d(A, B, "bfloat16")


def test_nvidia_matmul_candidates_registered():
    # B1: the bare-matmul op-kind + the two GEMM lanes (shipped Tier-3, emitted
    # Tier-2) registered under (nvidia, matmul).
    cands = {c.name: c for c in C.candidates_for("nvidia", OP_MATMUL)}
    assert set(cands) == {"nvidia_mma_gemm_shipped", "nvidia_mma_gemm_emitted"}
    assert cands["nvidia_mma_gemm_shipped"].tier == Tier.HAND_TUNED
    assert cands["nvidia_mma_gemm_emitted"].tier == Tier.EMITTED
    for c in cands.values():
        assert c.op == OP_MATMUL
        assert c.accuracy_atol == 5e-3                       # 16-bit storage budget
        assert c.applies_to(F.MatmulRegion(dtype="bfloat16"))
        assert c.applies_to(F.MatmulRegion(dtype="float16"))
        assert not c.applies_to(F.MatmulRegion(dtype="float32"))     # not 16-bit
        assert not c.applies_to(F.FusedRegion(epilogue=("relu",)))   # not a matmul


def test_nvidia_matmul_off_gpu_arbitrates_to_reference():
    # Host-free: with no GPU the candidates are unavailable, so the arbiter finds
    # no winner and run_arbitrated returns the numpy reference (never raises).
    if _nvidia_cuda_live():
        pytest.skip("GPU present — covered by the live arbitration test")
    region = F.MatmulRegion(dtype="bfloat16")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 16)).astype(np.float32)
    out, tag = C.run_arbitrated(region, OP_MATMUL, "nvidia", A, B)
    assert tag == "reference"
    np.testing.assert_allclose(out, region.reference(A, B), atol=1e-3)


def test_nvidia_missing_required_buffer_declines_not_segfault():
    # Same NULL-deref guard as x86/ROCm: a residual/bias region without the buffer
    # must not launch the CUDA kernel (which would deref a null). Child process so a
    # regression is a failed assert, not a crashed session.
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(
        """
        import numpy as np
        import tessera.compiler.fusion as F
        import tessera.compiler.emit.nvidia_cuda as nvidia
        r = nvidia.NvidiaCudaRunner()
        A = np.zeros((8, 12), np.float32)
        B = np.zeros((12, 16), np.float32)
        for region in (F.FusedRegion(epilogue=("relu",), residual=True),
                       F.FusedRegion(epilogue=("bias", "relu"))):
            try:
                r.run_fused_region(region, A, B, None)
                raise SystemExit("expected ValueError, got a result")
            except ValueError:
                pass
        print("ok")
        """
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, (
        f"missing-buffer guard failed (rc={p.returncode}, -11=SIGSEGV): "
        f"{p.stderr[-300:]}")
    assert "ok" in p.stdout


# ── 2. Live gate (needs a live NVIDIA GPU + nvcc) ─────────────────────────────

_C2_CHAINS = [
    F.FusedRegion(epilogue=("relu",)),
    F.FusedRegion(epilogue=("bias", "gelu")),
    F.FusedRegion(epilogue=("silu",)),
    F.FusedRegion(epilogue=("bias",), reduction="softmax"),
    F.FusedRegion(epilogue=(), reduction="rmsnorm"),
    F.FusedRegion(epilogue=("relu",), reduction="layer_norm"),
    F.FusedRegion(epilogue=("gelu",), prologue=("relu",)),
]


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("region", _C2_CHAINS,
                         ids=lambda r: f"{r.epilogue}/{r.reduction}/{r.prologue}")
def test_live_nvidia_generic_cuda_gated(region):
    # C2: the generically-synthesized CUDA FusedRegion kernel compiles with nvcc,
    # runs on an NVIDIA GPU ("nvidia_cuda"), matches numpy (f32), passes F4.
    F.clear_verification_cache()
    runner = get_runner("nvidia")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = rng.standard_normal((16,)).astype(np.float32) if region.has_bias else None
    out, execution = runner.run_fused_region(region, A, B, bias)
    assert execution == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(A, B, bias), atol=1e-3)
    assert F.verify_synthesized_region(region, runner=runner, force=True) is True


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_arbitrated_residual_executes():
    # PR #290 review: an arbitrated residual fusion must EXECUTE on-GPU (residual
    # threaded through the positional inputs), not fall back / raise. Verify
    # (default) also exercises the residual probe added to the F4 oracle.
    F.clear_verification_cache()
    region = F.FusedRegion(epilogue=("relu",), residual=True)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    res = rng.standard_normal((8, 16)).astype(np.float32)
    out, tag = C.run_arbitrated(region, OP_FUSED_REGION, "nvidia", A, B, None, res)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(A, B, None, res), atol=1e-3)


def _nvidia_matmul_live() -> bool:
    if not _nvidia_cuda_live():
        return False
    try:
        from tessera import runtime as rt
        return rt._load_nvidia_ptx_launch() is not None
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_matmul_live(),
                    reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("shape", [(16, 8, 16), (32, 16, 32), (64, 64, 64)],
                         ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
def test_live_nvidia_matmul_arbitrated(dtype, shape):
    # B1: the arbiter picks the hand-tuned shipped GEMM (Tier 3) by default and
    # runs it on-GPU; the E3 escape hatch forces the compiler-emitted lane (Tier 2)
    # through the PTX launch bridge. Both match the dtype-rounded reference.
    F.clear_verification_cache()
    M, N, K = shape
    region = F.MatmulRegion(dtype=dtype)
    rng = np.random.default_rng(M + K)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float32)
    ref = region.reference(A, B)
    out, tag = C.run_arbitrated(region, OP_MATMUL, "nvidia", A, B)
    assert tag == "nvidia_mma_shipped"                       # Tier-3 default
    np.testing.assert_allclose(out, ref, atol=5e-3, rtol=0)
    out2, tag2 = C.run_arbitrated(region, OP_MATMUL, "nvidia", A, B,
                                  force="nvidia_mma_gemm_emitted")
    assert tag2 == "nvidia_ptx_gemm"                         # Tier-2, forced (E3)
    np.testing.assert_allclose(out2, ref, atol=5e-3, rtol=0)


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_matmul_live(),
                    reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required")
def test_live_nvidia_matmul_measured_autotune():
    # D2: measured_arbitrate times BOTH GEMM lanes on-device and caches the winner
    # by (device, target, op, bucket, dtype) — measure-at-first-miss.
    from tessera.compiler.emit import autotune as AT
    F.clear_verification_cache()
    region = F.MatmulRegion(dtype="bfloat16")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((64, 64)) * 0.4).astype(np.float32)
    B = (rng.standard_normal((64, 64)) * 0.4).astype(np.float32)
    cache = AT.MeasureCache()
    win = AT.measured_arbitrate(region, OP_MATMUL, "nvidia", A, B,
                                dims=(64, 64, 64), dtype="bfloat16",
                                cache=cache, reps=6, warmup=2)
    assert win is not None and win.name in (
        "nvidia_mma_gemm_shipped", "nvidia_mma_gemm_emitted")
    assert cache.misses == 1 and cache.size == 1
    rec = cache.to_dict()["records"][0]
    assert set(rec["candidates"]) == {"nvidia_mma_gemm_shipped",
                                      "nvidia_mma_gemm_emitted"}
    AT.measured_arbitrate(region, OP_MATMUL, "nvidia", A, B, dims=(64, 64, 64),
                          dtype="bfloat16", cache=cache, reps=6, warmup=2)
    assert cache.hits == 1                                   # re-query hits the cache


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_matmul_live(),
                    reason="live NVIDIA GPU + shipped GEMM + PTX launch bridge required")
def test_live_nvidia_emitted_ragged_degrade_is_logged():
    # D3: force the emitted lane on a RAGGED shape it verifies (on the aligned
    # probe) but cannot run — it declines to the reference at execution time, which
    # the arbiter log records as a silent degrade (selected != reference tag).
    F.clear_verification_cache()
    region = F.MatmulRegion(dtype="bfloat16")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((24, 16)) * 0.4).astype(np.float32)   # M=24 not %16
    B = (rng.standard_normal((16, 8)) * 0.4).astype(np.float32)
    C.reset_arbiter_dispatch_log()
    out, tag = C.run_arbitrated(region, OP_MATMUL, "nvidia", A, B,
                                force="nvidia_mma_gemm_emitted")
    assert tag == "reference"                                # declined (ragged)
    np.testing.assert_allclose(out, region.reference(A, B), atol=5e-3)
    hist = C.arbiter_dispatch_histogram(target="nvidia", op=OP_MATMUL)
    assert hist[("nvidia", OP_MATMUL)]["degraded"] == 1


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("scale,causal", [(1.0, False), (0.25, False), (0.125, True)])
@pytest.mark.parametrize("shape", [(8, 8, 16, 16), (16, 24, 32, 64)],
                         ids=lambda s: "x".join(map(str, s)))
def test_live_nvidia_flash_attention(scale, causal, shape):
    # C4: the synthesized flash-attention kernel (online softmax) runs on-GPU,
    # matches the numpy reference, and passes the same universal F4 oracle — the
    # attention analog of the GEMM execute-compare proof.
    from tessera.compiler.emit.candidate import OP_ATTENTION
    from tessera.compiler.emit.kernel_emitter import get_runner
    F.clear_verification_cache()
    M, Nk, D, Dv = shape
    region = F.AttentionRegion(scale=scale, causal=causal)
    rng = np.random.default_rng(M + Nk + D)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((Nk, D)).astype(np.float32)
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    out, tag = get_runner("nvidia").run_fused_attention(region, Q, K, V)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(Q, K, V), atol=1e-4)
    assert F.verify_synthesized_attention(region, runner=get_runner("nvidia"),
                                          force=True) is True
    # the arbiter serves attention on-GPU — it now prefers the Tier-2 tensor-core
    # lane (f16 budget) over this Tier-1 scalar lane, so compare at the f16 budget.
    aout, atag = C.run_arbitrated(region, OP_ATTENTION, "nvidia", Q, K, V)
    assert atag == "nvidia_cuda"
    np.testing.assert_allclose(aout, region.reference(Q, K, V), atol=5e-3)


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("act", ["silu", "gelu", "relu"])
def test_live_nvidia_gated_swiglu(act):
    # C5: the synthesized SwiGLU-gate lane executes on-GPU and matches.
    F.clear_verification_cache()
    region = F.GatedMatmulRegion(gate_act=act)
    rng = np.random.default_rng(hash(act) % 1000)
    A = rng.standard_normal((16, 32)).astype(np.float32)
    Wg = rng.standard_normal((32, 24)).astype(np.float32)
    Wu = rng.standard_normal((32, 24)).astype(np.float32)
    out, tag = get_runner("nvidia").run_gated_matmul_region(region, A, Wg, Wu)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(A, Wg, Wu), atol=1e-3)
    assert F.verify_synthesized_gated(region, runner=get_runner("nvidia"),
                                      force=True) is True


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_pointwise_dag():
    # C5: the synthesized pointwise-DAG lane — d = relu(a+b) * sigmoid(c) — runs
    # on-GPU (from the POINTWISE_OPS C-expr table) and matches.
    F.clear_verification_cache()
    region = F.PointwiseGraphRegion(
        ops=(("add", ("a", "b"), "s"), ("relu", ("s",), "ra"),
             ("sigmoid", ("c",), "sc"), ("mul", ("ra", "sc"), "o")),
        inputs=("a", "b", "c"), output="o")
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal((6, 7)).astype(np.float32) for _ in range(3)]
    out, tag = get_runner("nvidia").run_pointwise_graph(region, arrs)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(*arrs), atol=1e-5)
    assert F.verify_synthesized_pointwise(region, runner=get_runner("nvidia"),
                                          force=True) is True


def test_nvidia_pointwise_emits_gelu_and_nan_safe_shims():
    # PR #297 review: gelu's POINTWISE_OPS template uses clamp() (no CUDA builtin),
    # and sign must preserve NaN like np.sign. The emitted source defines both
    # shims, NaN-aware.
    dag = F.PointwiseGraphRegion(ops=(("add", ("a", "b"), "s"),
                                      ("gelu", ("s",), "o")),
                                 inputs=("a", "b"), output="o")
    src = get_emitter("nvidia").emit(dag).source
    assert "float clamp(float" in src and "float sign(float" in src
    assert "isnan(x) ? x" in src                     # NaN-preserving


def test_nvidia_pointwise_max_min_nan_shims_and_buffer_free():
    # PR #297 review 2: max/min route through NaN-propagating shims (np.maximum/
    # np.minimum semantics, not CUDA's NaN-suppressing max/min), and a partial
    # cudaMalloc failure frees the buffers already allocated (no device leak).
    e = get_emitter("nvidia")
    mm = e.emit(F.PointwiseGraphRegion(
        ops=(("maximum", ("a", "b"), "m"), ("minimum", ("m", "a"), "o")),
        inputs=("a", "b"), output="o")).source
    assert "tsr_max(" in mm and "tsr_min(" in mm
    assert "(isnan(a)||isnan(b)) ? NAN" in mm
    three = e.emit(F.PointwiseGraphRegion(
        ops=(("add", ("a", "b"), "s"), ("mul", ("s", "c"), "o")),
        inputs=("a", "b", "c"), output="o")).source
    assert "cudaFree(d0); cudaFree(d1); return 3;" in three   # cumulative free


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_pointwise_max_min_preserve_nan():
    # PR #297 review 2, on-GPU: maximum/minimum on NaN data must propagate NaN
    # (np.maximum/np.minimum), not select the numeric operand like CUDA max/min.
    r = get_runner("nvidia")
    dag = F.PointwiseGraphRegion(
        ops=(("maximum", ("a", "b"), "m"), ("relu", ("m",), "o")),
        inputs=("a", "b"), output="o")
    a = np.array([[np.nan, 2.0, -1.0]], np.float32)
    b = np.array([[5.0, np.nan, -3.0]], np.float32)
    out, tag = r.run_pointwise_graph(dag, [a, b])
    assert tag == "nvidia_cuda"
    ref = dag.reference(a, b)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))   # NaN in place
    np.testing.assert_allclose(out[~np.isnan(out)], ref[~np.isnan(ref)], atol=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_pointwise_gelu_and_nan_sign():
    # PR #297 review, on-GPU: a gelu DAG must COMPILE + run (was failing on the
    # undefined clamp), and a sign chain must preserve NaN (np.sign semantics).
    r = get_runner("nvidia")
    gelu = F.PointwiseGraphRegion(ops=(("add", ("a", "b"), "s"),
                                       ("gelu", ("s",), "o")),
                                  inputs=("a", "b"), output="o")
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 6)).astype(np.float32)
    b = rng.standard_normal((5, 6)).astype(np.float32)
    out, tag = r.run_pointwise_graph(gelu, [a, b])
    assert tag == "nvidia_cuda"                       # device_verified_jit + ran (not declined)
    np.testing.assert_allclose(out, gelu.reference(a, b), atol=1e-4)

    sign = F.PointwiseGraphRegion(ops=(("abs", ("a",), "aa"), ("sign", ("aa",), "o")),
                                  inputs=("a",), output="o")
    x = np.array([[1.0, -2.0, np.nan, 0.0]], np.float32)
    sout, stag = r.run_pointwise_graph(sign, [x])
    assert stag == "nvidia_cuda"
    ref = sign.reference(x)
    np.testing.assert_array_equal(np.isnan(sout), np.isnan(ref))   # NaN preserved
    np.testing.assert_allclose(sout[~np.isnan(sout)], ref[~np.isnan(ref)], atol=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("epilogue", [("bias", "gelu"), ("relu",), ("silu",),
                                      ("bias", "relu")])
def test_live_nvidia_mma_fused_tensor_core(epilogue):
    # The tensor-core fused lane (mma.sync GEMM + bias/activation, f16) runs on-GPU,
    # matches the f32 reference within the f16 budget, F4-passes, and the arbiter
    # prefers it (Tier-2) over the scalar generic lane (Tier-1).
    F.clear_verification_cache()
    region = F.FusedRegion(epilogue=epilogue)
    mf = next(c for c in C.candidates_for("nvidia", OP_FUSED_REGION)
              if c.name == "nvidia_mma_fused")
    rng = np.random.default_rng(len(epilogue))
    A = (rng.standard_normal((64, 64)) * 0.3).astype(np.float32)
    B = (rng.standard_normal((64, 64)) * 0.3).astype(np.float32)
    bias = ((rng.standard_normal((64,)) * 0.3).astype(np.float32)
            if region.has_bias else None)
    out, tag = mf.run(region, A, B, bias)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(A, B, bias), atol=5e-3)
    assert C.verify_candidate(mf, region) is True
    win = C.arbitrate(region, OP_FUSED_REGION, "nvidia")
    assert win is not None and win.name == "nvidia_mma_fused"    # Tier-2 > Tier-1


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
@pytest.mark.parametrize("scale,causal", [(0.125, False), (0.25, False), (0.125, True)])
@pytest.mark.parametrize("shape", [(16, 16, 16, 16), (32, 48, 32, 64),
                                   (64, 128, 64, 64)],
                         ids=lambda s: "x".join(map(str, s)))
def test_live_nvidia_mma_attn_tensor_core(scale, causal, shape):
    # The tensor-core flash-attention lane (two mma.sync matmuls, smem-staged
    # softmax, f16) runs on-GPU on well-conditioned inputs, matches the f32
    # reference within the f16 budget, F4-passes, and the arbiter prefers it.
    from tessera.compiler.emit.candidate import OP_ATTENTION
    F.clear_verification_cache()
    M, Nk, D, Dv = shape
    region = F.AttentionRegion(scale=scale, causal=causal)
    mf = next(c for c in C.candidates_for("nvidia", OP_ATTENTION)
              if c.name == "nvidia_mma_attn")
    rng = np.random.default_rng(M + Nk)
    Q = rng.standard_normal((M, D)).astype(np.float32)      # amax ~ a few (safe)
    K = rng.standard_normal((Nk, D)).astype(np.float32)
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    out, tag = mf.run(region, Q, K, V)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(Q, K, V), atol=5e-3)
    assert C.verify_candidate(mf, region) is True
    win = C.arbitrate(region, OP_ATTENTION, "nvidia")
    assert win is not None and win.name == "nvidia_mma_attn"


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_mma_attn_large_nk_delegates_to_scalar():
    # A KV length past the smem cap delegates to the scalar flash lane rather than
    # declining — still a real on-GPU kernel within budget.
    from tessera.compiler.emit.candidate import OP_ATTENTION  # noqa: F401
    region = F.AttentionRegion(scale=0.125)
    mf = next(c for c in C.candidates_for("nvidia", "attention")
              if c.name == "nvidia_mma_attn")
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((16, 64)).astype(np.float32)
    K = rng.standard_normal((2048, 64)).astype(np.float32)   # Nk > smem cap
    V = rng.standard_normal((2048, 64)).astype(np.float32)
    out, tag = mf.run(region, Q, K, V)
    assert tag == "nvidia_cuda"
    np.testing.assert_allclose(out, region.reference(Q, K, V), atol=5e-3)


@pytest.mark.slow
@pytest.mark.skipif(not _nvidia_cuda_live(),
                    reason="live NVIDIA GPU + nvcc required")
def test_live_nvidia_mma_attn_large_magnitude_delegates_not_degrades():
    # PR #302 review: f16 rounding blows the 5e-3 budget on large-magnitude /
    # large-scale f32 attention (softmax sharpens), which the F4 probe misses. The
    # candidate must delegate those to the EXACT scalar lane, never silently degrade.
    region = F.AttentionRegion(scale=1.0)              # sharp softmax
    mf = next(c for c in C.candidates_for("nvidia", "attention")
              if c.name == "nvidia_mma_attn")
    rng = np.random.default_rng(2)
    Q = (rng.standard_normal((32, 64)) * 100.0).astype(np.float32)   # amax ~ 350
    K = (rng.standard_normal((48, 64)) * 100.0).astype(np.float32)
    V = (rng.standard_normal((48, 64)) * 100.0).astype(np.float32)
    out, tag = mf.run(region, Q, K, V)
    assert tag == "nvidia_cuda"                        # delegated to the scalar lane
    # the returned result tracks the f32 reference (not an f16-degraded one).
    ref = region.reference(Q, K, V)
    assert float(np.max(np.abs(out - ref))) / (float(np.max(np.abs(ref))) + 1e-9) < 1e-2
