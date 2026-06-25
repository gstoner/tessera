"""Compiler-generated linear attention (quadratic-parallel form) on gfx1151.

Linear attention replaces softmax(QKᵀ)V with the feature-map factorization
`O = (φ(Q) φ(K)ᵀ ⊙ causal) @ V` — NO softmax, no normalization (a distinct
algorithm from flash attention). The `tessera_rocm.linear_attn` directive
expands (via `generate-wmma-linear-attn-kernel`) into a WMMA kernel: compute
`A = φ(Q)φ(K)ᵀ` (WMMA over head-dim chunks), mask it **multiplicatively**
(masked → 0), and accumulate `O += A @ V` (WMMA), no final divide. Reaches
`runtime.launch()` via `compiler_path="rocm_linear_attn_compiled"` — feature
map ∈ {identity, relu}, causal + non-causal.

Validated vs the canonical reference (`_apple_gpu_dispatch_linear_attn` math):
`O = (φ(Q)φ(K)ᵀ ⊙ tril) @ V`.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _la_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, causal, feature_map=None, log_decay=None,
              op_name="tessera.linear_attn", extra_kwargs=None):
    kwargs = {"causal": bool(causal)}
    if feature_map is not None:
        kwargs["feature_map"] = feature_map
    if log_decay is not None:
        kwargs["log_decay"] = float(log_decay)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_linear_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": kwargs}],
    })


def _fmap(x, name):
    if name == "identity":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "polynomial_2":
        return x * x
    raise AssertionError(name)


def _linear_attn_ref(q, k, v, causal, feature_map, log_decay=None):
    """Canonical reference: O = (φ(Q)φ(K)ᵀ ⊙ tril [⊙ λ^(i-j)]) @ V, [B,H,S,D].
    With log_decay set, the decay mask λ^(i-j) is applied over the causal band
    (the per-head-constant case of the reference's dc[i]/dc[j] ratio)."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    phiQ = _fmap(q.astype(np.float64), feature_map)
    phiK = _fmap(k.astype(np.float64), feature_map)
    i = np.arange(Sq)[:, None]
    j = np.arange(Sk)[None, :]
    o = np.zeros((B, H, Sq, D), np.float64)
    for b in range(B):
        for h in range(H):
            A = phiQ[b, h] @ phiK[b, h].T               # [Sq, Sk]
            if log_decay is not None:
                A = A * np.exp((i - j) * float(log_decay))   # λ^(i-j)
            if causal:
                A = np.where(j > i, 0.0, A)              # multiplicative tril
            o[b, h] = A @ v[b, h].astype(np.float64)
    return o.astype(np.float32)


@pytest.mark.parametrize("feature_map", ["identity", "relu", "polynomial_2"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("D,B,H,S", [
    (16, 1, 2, 32),
    (16, 1, 1, 48),
    (64, 2, 2, 32),
    (16, 1, 1, 40),    # ragged S (not a multiple of 16)
])
def test_launch_linear_attn_matches_numpy(feature_map, causal, D, B, H, S):
    rt = _la_or_skip()
    rng = np.random.default_rng(2 + D + H + S + int(causal) + len(feature_map))
    # Modest magnitude — linear attention is unnormalized, so keep values small
    # to avoid the f16 score truncation blowing past tolerance.
    q = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)

    res = rt.launch(_artifact(rt, causal, feature_map), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_linear_attn_compiled"
    out = res["output"].reshape(B, H, S, D)

    ref = _linear_attn_ref(q, k, v, causal, feature_map)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 3e-2, (
        f"linear_attn fmap={feature_map} causal={causal} {B}x{H}x{S}x{D} "
        f"maxerr={maxerr}")


@pytest.mark.parametrize("feature_map,lam", [
    ("identity", 0.9),       # lightning_attention
    ("identity", 0.95),
    ("polynomial_2", 0.9),   # (degree-2) retention
])
@pytest.mark.parametrize("D,S", [(16, 32), (64, 48), (16, 40)])
def test_launch_decay_matches_numpy(feature_map, lam, D, S):
    """Decay-masked variants: A[i,j] *= λ^(i-j) over the causal band — matches
    the reference's per-head-constant dc[i]/dc[j] ratio."""
    rt = _la_or_skip()
    B, H = 1, 2
    log_decay = float(np.log(lam))
    rng = np.random.default_rng(31 + D + S + int(lam * 100))
    q = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)

    res = rt.launch(_artifact(rt, True, feature_map, log_decay), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    out = res["output"].reshape(B, H, S, D)

    ref = _linear_attn_ref(q, k, v, True, feature_map, log_decay)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 3e-2, (
        f"decay fmap={feature_map} λ={lam} {S}x{D} maxerr={maxerr}")


# ── named-op dispatch: lightning_attention / retention by OP NAME ─────────────
# The lane must accept these op names (not just tessera.linear_attn) and map
# each to its canonical (feature_map, decay) config, so a caller stamping
# `tessera.lightning_attention` / `tessera.retention` actually executes — this is
# what the normative COMPILER_REFERENCE table advertises as ROCm hardware-runtime.

@pytest.mark.parametrize("op_name,fmap,lam", [
    ("tessera.lightning_attention", "identity", 0.9),
    ("tessera.lightning_attention", "identity", None),   # decay optional
    ("tessera.retention", "polynomial_2", 0.9),          # degree-2 retention
])
@pytest.mark.parametrize("D,S", [(16, 32), (64, 48)])
def test_launch_named_decay_ops_match_numpy(op_name, fmap, lam, D, S):
    rt = _la_or_skip()
    B, H = 1, 2
    log_decay = float(np.log(lam)) if lam is not None else None
    rng = np.random.default_rng(53 + D + S + (int(lam * 100) if lam else 0))
    q = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.25).astype(np.float16)

    # No feature_map kwarg — the op NAME must pin it (identity for lightning,
    # x² for retention).
    res = rt.launch(_artifact(rt, True, feature_map=None, log_decay=log_decay,
                              op_name=op_name), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    out = res["output"].reshape(B, H, S, D)
    ref = _linear_attn_ref(q, k, v, True, fmap, log_decay)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 3e-2, f"{op_name} λ={lam} {S}x{D} maxerr={maxerr}"


def test_retention_degree_not_2_is_rejected():
    """The kernel realizes only degree-2 retention (φ = x²); deg != 2 (which the
    reference handles by pre-powering Q/K) is a named error, not a silent wrong
    answer. Pure validation — needs no device."""
    from tessera import runtime as rt
    q = np.zeros((1, 1, 32, 16), np.float16)
    art = _artifact(rt, True, op_name="tessera.retention",
                    extra_kwargs={"deg": 3})
    with pytest.raises(ValueError, match="degree 2 only"):
        rt._execute_rocm_compiled_linear_attn(art, (q, q, q))


def test_unknown_op_name_is_rejected():
    """A non-family op name routed here is a clean error, never a silent run."""
    from tessera import runtime as rt
    q = np.zeros((1, 1, 16, 16), np.float16)
    art = _artifact(rt, True, op_name="tessera.flash_attn")
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_linear_attn(art, (q, q, q))


def test_causal_differs_from_full():
    """A causal linear-attn result must differ from the non-causal one — guards
    against the causal mask being silently ignored."""
    rt = _la_or_skip()
    D, B, H, S = 16, 1, 1, 48
    rng = np.random.default_rng(77)
    q = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)

    causal = rt.launch(_artifact(rt, True, "identity"), (q, k, v))["output"]
    full = _linear_attn_ref(q, k, v, False, "identity")
    # Compare an EARLY query row (index 0): causal sees only key 0, non-causal
    # sees all S keys, so they must differ. (The LAST row sees all keys either
    # way, so it would be identical — not a useful probe.)
    co = causal.reshape(B, H, S, D)[0, 0, 0]
    diff = float(np.max(np.abs(co - full[0, 0, 0])))
    assert diff > 1e-2, "causal linear-attn output indistinguishable from full"


def test_mismatched_kv_seqlen_is_rejected_not_oob():
    """K and V must share the key length. `sk` (from K) drives the V device
    copy size, so a shorter V would read past the host buffer — the lane must
    reject it with a clean ValueError, never read OOB. Pure shape validation
    (runs before any GPU / tessera-opt call), so it needs no device."""
    from tessera import runtime as rt
    q = np.zeros((1, 1, 32, 16), np.float16)
    k = np.zeros((1, 1, 64, 16), np.float16)   # K: 64 keys
    v = np.zeros((1, 1, 32, 16), np.float16)   # V: only 32 — would OOB at sk=64
    art = _artifact(rt, True, "identity")
    with pytest.raises(ValueError, match="share the sequence length"):
        rt._execute_rocm_compiled_linear_attn(art, (q, k, v))
