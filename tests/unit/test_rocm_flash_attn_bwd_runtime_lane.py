"""ROCm flash_attn BACKWARD reachable through runtime.launch() on gfx1151.

`test_rocm_flash_attn_bwd_compiled.py` proves the compiler-generated FA-2
backward kernel directly (build hsaco + hand-launch fa_pre/fa_dkdv/fa_dq). THIS
fixture proves the runtime *lane* — `compiler_path="rocm_flash_attn_bwd_compiled"`
routed through `runtime.launch()` — so the backward is a first-class executing
lane like the forward, not just a standalone kernel. Operands are (dO, Q, K, V);
O is recomputed on-device via the forward lane. dQ/dK/dV are compared to the
numpy attention-backward reference (the same math as autodiff `vjp_flash_attn`).

Core MHA (scale + causal), f16/bf16 storage, f32 accumulate. Skip-clean:
tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _ref_bwd(Q, K, V, dO, scale, causal, bias=None, window=0, softcap=0.0):
    """numpy FA backward (f32 math from the f16/bf16 storage inputs). Handles
    GQA/MQA (query head h reads KV head g = h // (H//G); dK/dV accumulate over the
    group), optional additive bias [B,H,Sq,Sk], sliding window W (q attends keys
    in (q-W, q], implicitly causal), and Gemma-2 soft-cap (S=cap*tanh(S_raw/cap),
    backward scales dS by 1-tanh^2). Q/dO are [B,H,Sq,D]; K/V are [B,G,Sk,D]."""
    B, H, Sq, D = Q.shape
    G, Sk = K.shape[1], K.shape[2]
    ratio = H // G
    Qf, Kf, Vf, dOf = (a.astype(np.float32) for a in (Q, K, V, dO))
    biasf = None if bias is None else np.asarray(bias, np.float32)
    i = np.arange(Sq)[:, None]; j = np.arange(Sk)[None, :]
    dQ = np.zeros((B, H, Sq, D), np.float32)
    dK = np.zeros((B, G, Sk, D), np.float32)
    dV = np.zeros((B, G, Sk, D), np.float32)
    for b in range(B):
        for h in range(H):
            g = h // ratio
            s_raw = scale * (Qf[b, h] @ Kf[b, g].T)
            if softcap:
                t = np.tanh(s_raw / softcap)
                s = softcap * t
                capderiv = 1.0 - t * t
            else:
                s = s_raw
                capderiv = 1.0
            if biasf is not None:
                s = s + biasf[b, h]
            mask = np.zeros((Sq, Sk), bool)
            if causal or window:
                mask |= (j > i)                 # causal upper (future keys)
            if window:
                mask |= ((i - j) >= window)     # too-old keys
            s = np.where(mask, -1e30, s)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s); p = p / p.sum(-1, keepdims=True)
            O = p @ Vf[b, g]
            dp = dOf[b, h] @ Vf[b, g].T
            dq_row = np.sum(O * dOf[b, h], axis=-1, keepdims=True)
            ds = p * (dp - dq_row) * capderiv   # dS_raw = dS_capped * cap'
            dQ[b, h] = scale * (ds @ Kf[b, g])
            dK[b, g] += scale * (ds.T @ Qf[b, h])
            dV[b, g] += p.T @ dOf[b, h]
    return dQ, dK, dV


def _art(rt, causal, scale, bias=False, window=0, softcap=0.0):
    names = ["do", "q", "k", "v"] + (["bias"] if bias else [])
    kw = {"scale": scale, "causal": causal}
    if window:
        kw["window"] = int(window)
    if softcap:
        kw["logit_softcap"] = float(softcap)
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_bwd_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "g",
        "ops": [{"op_name": "tessera.flash_attn_bwd", "result": "g",
                 "operands": names, "kwargs": kw}]})


def _run(rt, dO, Q, K, V, *, causal, scale, bias=None, window=0, softcap=0.0):
    call = (dO, Q, K, V) if bias is None else (dO, Q, K, V, bias)
    res = rt.launch(
        _art(rt, causal, scale, bias is not None, window, softcap), call)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_flash_attn_bwd_compiled"
    dQ, dK, dV = res["output"]
    return np.asarray(dQ), np.asarray(dK), np.asarray(dV)


def _relerr(got, ref):
    return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-6))


@pytest.mark.parametrize("D,B,H,Sq,Sk,causal", [
    (16, 1, 1, 16, 16, False),
    (16, 1, 2, 32, 48, False),
    (64, 2, 2, 48, 48, True),
    (16, 1, 1, 20, 40, False),     # ragged Sq/Sk
    (64, 1, 2, 32, 32, True),      # causal, D=64
])
def test_bwd_runtime_lane_matches_numpy(D, B, H, Sq, Sk, causal):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7 + D + Sq + Sk + int(causal))
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, causal)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=causal, scale=scale)
    assert dQ.shape == Q.shape and dK.shape == K.shape and dV.shape == V.shape
    tol = 5e-3          # f16 storage, f32 accumulate — measured ~2-4e-4
    eQ, eK, eV = _relerr(dQ, dQ_ref), _relerr(dK, dK_ref), _relerr(dV, dV_ref)
    assert eQ < tol and eK < tol and eV < tol, (
        f"rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"@ D={D} {B}x{H}x{Sq}x{Sk} causal={causal}")


def test_bwd_runtime_lane_bf16():
    rt = _rocm_or_skip()
    bf16 = rt._bfloat16_dtype()
    if bf16 is None:
        pytest.skip("no bfloat16 dtype available")
    rng = np.random.default_rng(99)
    B, H, Sq, Sk, D = 1, 2, 32, 32, 16
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(bf16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(bf16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(bf16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(bf16)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, False)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=False, scale=scale)
    tol = 3e-2          # bf16 has ~8 mantissa bits — looser bound
    assert (_relerr(dQ, dQ_ref) < tol and _relerr(dK, dK_ref) < tol
            and _relerr(dV, dV_ref) < tol)


@pytest.mark.parametrize("D,B,H,G,Sq,Sk,causal", [
    (16, 1, 8, 1, 32, 32, False),   # MQA (one shared KV head)
    (16, 2, 8, 2, 32, 48, False),   # GQA, ratio 4
    (64, 1, 8, 4, 48, 48, True),    # GQA, ratio 2, causal
    (16, 1, 4, 4, 32, 32, False),   # MHA-equivalence (G == H)
])
def test_bwd_runtime_lane_gqa_matches_numpy(D, B, H, G, Sq, Sk, causal):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(23 + D + H + G + int(causal))
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, G, Sk, D)) * 0.3).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, causal)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=causal, scale=scale)
    assert dQ.shape == Q.shape and dK.shape == K.shape and dV.shape == V.shape
    tol = 5e-3
    eQ, eK, eV = _relerr(dQ, dQ_ref), _relerr(dK, dK_ref), _relerr(dV, dV_ref)
    assert eQ < tol and eK < tol and eV < tol, (
        f"GQA rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"@ D={D} H={H} G={G} causal={causal}")


@pytest.mark.parametrize("D,B,H,Sq,Sk,causal", [
    (16, 1, 2, 32, 32, False),
    (16, 1, 2, 24, 40, False),     # ragged
    (64, 1, 2, 32, 32, True),      # causal + bias
])
def test_bwd_runtime_lane_attn_bias_matches_numpy(D, B, H, Sq, Sk, causal):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(31 + D + Sq + Sk + int(causal))
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    bias = (rng.standard_normal((B, H, Sq, Sk)) * 0.5).astype(np.float32)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, causal, bias=bias)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=causal, scale=scale, bias=bias)
    tol = 5e-3
    eQ, eK, eV = _relerr(dQ, dQ_ref), _relerr(dK, dK_ref), _relerr(dV, dV_ref)
    assert eQ < tol and eK < tol and eV < tol, (
        f"bias rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"@ D={D} {B}x{H}x{Sq}x{Sk} causal={causal}")


def test_bwd_runtime_lane_bias_broadcast_per_head():
    # Bias broadcast from [H,Sq,Sk] (no batch axis) to B*H — exercises the
    # host broadcast path.
    rt = _rocm_or_skip()
    rng = np.random.default_rng(88)
    B, H, Sq, Sk, D = 2, 2, 16, 16, 16
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    bias_h = (rng.standard_normal((H, Sq, Sk)) * 0.5).astype(np.float32)
    scale = 1.0 / float(np.sqrt(D))
    full = np.broadcast_to(bias_h, (B, H, Sq, Sk))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, False, bias=full)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=False, scale=scale, bias=bias_h)
    tol = 5e-3
    assert (_relerr(dQ, dQ_ref) < tol and _relerr(dK, dK_ref) < tol
            and _relerr(dV, dV_ref) < tol)


@pytest.mark.parametrize("D,B,H,Sq,Sk,W", [
    (16, 1, 2, 32, 32, 8),
    (16, 1, 2, 40, 40, 16),
    (64, 1, 2, 48, 48, 24),
])
def test_bwd_runtime_lane_sliding_window_matches_numpy(D, B, H, Sq, Sk, W):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(51 + D + Sq + W)
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    # window is implicitly causal → reference uses causal=False + window=W.
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, False, window=W)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=False, scale=scale, window=W)
    tol = 5e-3
    eQ, eK, eV = _relerr(dQ, dQ_ref), _relerr(dK, dK_ref), _relerr(dV, dV_ref)
    assert eQ < tol and eK < tol and eV < tol, (
        f"window rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"@ D={D} {B}x{H}x{Sq}x{Sk} W={W}")


@pytest.mark.parametrize("D,B,H,Sq,Sk,cap,causal", [
    (16, 1, 2, 32, 32, 30.0, False),
    (16, 1, 2, 24, 40, 50.0, False),
    (64, 1, 2, 32, 32, 20.0, True),
])
def test_bwd_runtime_lane_softcap_matches_numpy(D, B, H, Sq, Sk, cap, causal):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(61 + D + Sq + int(cap))
    # larger magnitude so the tanh soft-cap is actually exercised.
    Q = (rng.standard_normal((B, H, Sq, D)) * 1.0).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 1.0).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.5).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.5).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, causal, softcap=cap)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=causal, scale=scale, softcap=cap)
    tol = 8e-3
    eQ, eK, eV = _relerr(dQ, dQ_ref), _relerr(dK, dK_ref), _relerr(dV, dV_ref)
    assert eQ < tol and eK < tol and eV < tol, (
        f"softcap rel-err dQ={eQ:.2e} dK={eK:.2e} dV={eV:.2e} "
        f"@ D={D} {B}x{H}x{Sq}x{Sk} cap={cap} causal={causal}")


def test_bwd_runtime_lane_window_softcap_bias_compose():
    # window + softcap + bias together (also exercises the arg-order plumbing).
    rt = _rocm_or_skip()
    rng = np.random.default_rng(77)
    B, H, Sq, Sk, D, W, cap = 1, 2, 32, 32, 16, 12, 30.0
    Q = (rng.standard_normal((B, H, Sq, D)) * 0.6).astype(np.float16)
    K = (rng.standard_normal((B, H, Sk, D)) * 0.6).astype(np.float16)
    V = (rng.standard_normal((B, H, Sk, D)) * 0.5).astype(np.float16)
    dO = (rng.standard_normal((B, H, Sq, D)) * 0.5).astype(np.float16)
    bias = (rng.standard_normal((B, H, Sq, Sk)) * 0.3).astype(np.float32)
    scale = 1.0 / float(np.sqrt(D))
    dQ_ref, dK_ref, dV_ref = _ref_bwd(Q, K, V, dO, scale, False, bias=bias,
                                      window=W, softcap=cap)
    dQ, dK, dV = _run(rt, dO, Q, K, V, causal=False, scale=scale, bias=bias,
                      window=W, softcap=cap)
    tol = 8e-3
    assert (_relerr(dQ, dQ_ref) < tol and _relerr(dK, dK_ref) < tol
            and _relerr(dV, dV_ref) < tol)


def test_bwd_runtime_lane_rejects_bad_group():
    # Query heads not divisible by KV heads → reject clearly (Decision #21).
    rt = _rocm_or_skip()
    B, Hq, Hkv, Sq, Sk, D = 1, 6, 4, 16, 16, 16   # 6 % 4 != 0
    Q = np.zeros((B, Hq, Sq, D), np.float16)
    K = np.zeros((B, Hkv, Sk, D), np.float16)
    V = np.zeros((B, Hkv, Sk, D), np.float16)
    dO = np.zeros((B, Hq, Sq, D), np.float16)
    res = rt.launch(_art(rt, False, 0.25), (dO, Q, K, V))
    assert res["ok"] is False
