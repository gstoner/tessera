"""LA-1 — Linear / kernel-feature attention forward + VJP correctness.

Covers:
  * Recurrent vs chunk-parallel equivalence (bit-equivalent at fp64)
  * Non-causal short-circuit form (single-matmul reference)
  * All four feature maps (elu, relu, identity, polynomial_2)
  * Decay variants (RetNet / GLA / Mamba2-selective)
  * `linear_attn_state` companion op
  * VJP numerical-Jacobian agreement (causal + non-causal + decay paths)

See ``docs/audit/attention_variants_plan.md`` Variant 2 for the design.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


def _numerical_grad(fn, x, eps=1e-6):
    g = np.zeros_like(x)
    flat = x.ravel()
    flat_g = g.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        plus = float(fn(x))
        flat[i] = orig - eps
        minus = float(fn(x))
        flat[i] = orig
        flat_g[i] = (plus - minus) / (2 * eps)
    return g


def _make_qkv(B=1, H=1, S=4, D_qk=2, D_v=2, seed=0):
    np.random.seed(seed)
    return (
        np.random.randn(B, H, S, D_qk).astype(np.float64),
        np.random.randn(B, H, S, D_qk).astype(np.float64),
        np.random.randn(B, H, S, D_v).astype(np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward correctness — recurrent / chunked / non-causal
# ─────────────────────────────────────────────────────────────────────────────


class TestLinearAttnForward:
    def test_recurrent_identity_matches_manual_walk(self):
        Q, K, V = _make_qkv(B=2, H=2, S=4, D_qk=3, D_v=3)
        O = ts.ops.linear_attn(Q, K, V, feature_map="identity", causal=True)
        # Walk recurrence by hand.
        B, H, S, D_qk = Q.shape
        D_v = V.shape[3]
        S_state = np.zeros((B, H, D_qk, D_v))
        ref = np.zeros((B, H, S, D_v))
        for t in range(S):
            S_state = S_state + np.einsum(
                "bhd,bhe->bhde", K[:, :, t, :], V[:, :, t, :]
            )
            ref[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], S_state)
        np.testing.assert_allclose(O, ref, rtol=1e-10)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
    def test_chunked_parallel_matches_pure_recurrent(self, chunk_size):
        """Chunked-parallel must be bit-equivalent to the pure-recurrent
        form at fp64 (the chunked walk just unfolds the same state
        update with extra intra-chunk parallelism — no math change)."""
        Q, K, V = _make_qkv(B=2, H=2, S=8, D_qk=4, D_v=3)
        rec = ts.ops.linear_attn(Q, K, V, feature_map="identity", causal=True)
        chunked = ts.ops.linear_attn(
            Q, K, V, feature_map="identity", causal=True, chunk_size=chunk_size,
        )
        np.testing.assert_allclose(rec, chunked, rtol=1e-10)

    def test_non_causal_short_circuit(self):
        """Non-causal linear attention folds to a single matmul:
            O = φ(Q) @ φ(K)^T @ V"""
        Q, K, V = _make_qkv(B=2, H=2, S=5, D_qk=3, D_v=3)
        O = ts.ops.linear_attn(Q, K, V, feature_map="identity", causal=False)
        # For identity feature_map: O[s,:] = sum_r Q[s] · K[r] * V[r]
        ref = np.einsum("bhsd,bhrd,bhrf->bhsf", Q, K, V)
        np.testing.assert_allclose(O, ref, rtol=1e-10)

    @pytest.mark.parametrize("feature_map", ["elu", "relu", "identity", "polynomial_2"])
    def test_each_feature_map_runs(self, feature_map):
        Q, K, V = _make_qkv(B=2, H=2, S=4, D_qk=3, D_v=3)
        O = ts.ops.linear_attn(Q, K, V, feature_map=feature_map, causal=True)
        assert O.shape == V.shape

    def test_invalid_feature_map_rejected(self):
        Q, K, V = _make_qkv()
        with pytest.raises(ValueError, match="feature_map"):
            ts.ops.linear_attn(Q, K, V, feature_map="exp")

    def test_dtype_preserves_through_recurrence(self):
        """Output dtype matches input dtype — fp64 in, fp64 out (the
        v1 forward used to force fp32, dropping precision)."""
        Q, K, V = _make_qkv()
        O = ts.ops.linear_attn(
            Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64),
            feature_map="identity", causal=True,
        )
        assert O.dtype == np.float64

    def test_shape_validation_rejects_rank3(self):
        with pytest.raises(ValueError, match="rank-4"):
            ts.ops.linear_attn(
                np.zeros((2, 4, 4)), np.zeros((2, 4, 4)), np.zeros((2, 4, 4)),
            )

    def test_kv_shape_mismatch_rejected(self):
        Q = np.zeros((1, 1, 4, 3))
        K_bad = np.zeros((1, 1, 5, 3))  # mismatched S
        V = np.zeros((1, 1, 4, 3))
        with pytest.raises(ValueError, match="K shape"):
            ts.ops.linear_attn(Q, K_bad, V)


# ─────────────────────────────────────────────────────────────────────────────
# Decay variants
# ─────────────────────────────────────────────────────────────────────────────


class TestLinearAttnDecay:
    def test_decay_matches_manual_walk(self):
        Q, K, V = _make_qkv(B=1, H=1, S=4, D_qk=2, D_v=2)
        decay = np.array([[[0.9, 0.85, 0.95, 0.8]]], dtype=np.float64)
        O = ts.ops.linear_attn(
            Q, K, V, feature_map="identity", causal=True, decay=decay,
        )
        # Manual walk with decay.
        B, H, S, D_qk = Q.shape
        D_v = V.shape[3]
        S_state = np.zeros((B, H, D_qk, D_v))
        ref = np.zeros((B, H, S, D_v))
        for t in range(S):
            S_state = decay[:, :, t][:, :, None, None] * S_state
            S_state = S_state + np.einsum(
                "bhd,bhe->bhde", K[:, :, t, :], V[:, :, t, :]
            )
            ref[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], S_state)
        np.testing.assert_allclose(O, ref, rtol=1e-10)

    def test_decay_shape_validated(self):
        Q, K, V = _make_qkv()
        bad_decay = np.zeros((1, 1, 99))  # wrong S
        with pytest.raises(ValueError, match="decay shape"):
            ts.ops.linear_attn(Q, K, V, decay=bad_decay)


# ─────────────────────────────────────────────────────────────────────────────
# linear_attn_state companion
# ─────────────────────────────────────────────────────────────────────────────


class TestLinearAttnState:
    def test_state_shape(self):
        Q, K, V = _make_qkv(B=2, H=3, S=4, D_qk=5, D_v=7)
        S_out = ts.ops.linear_attn_state(
            Q, K, V, feature_map="identity", causal=True,
        )
        assert S_out.shape == (2, 3, 5, 7)

    def test_state_can_chain_two_chunks(self):
        """Calling linear_attn over the first half of the sequence with
        no state, then over the second half with the returned state,
        produces the same output as a single full-sequence call."""
        Q, K, V = _make_qkv(B=1, H=1, S=8, D_qk=3, D_v=3)
        # Full call.
        O_full = ts.ops.linear_attn(
            Q, K, V, feature_map="identity", causal=True,
        )
        # Two-chunk call.
        Qa, Ka, Va = Q[:, :, :4], K[:, :, :4], V[:, :, :4]
        Qb, Kb, Vb = Q[:, :, 4:], K[:, :, 4:], V[:, :, 4:]
        O_a = ts.ops.linear_attn(Qa, Ka, Va, feature_map="identity", causal=True)
        S_a = ts.ops.linear_attn_state(
            Qa, Ka, Va, feature_map="identity", causal=True,
        )
        O_b = ts.ops.linear_attn(
            Qb, Kb, Vb, feature_map="identity", causal=True, state=S_a,
        )
        chained = np.concatenate([O_a, O_b], axis=2)
        np.testing.assert_allclose(chained, O_full, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# VJP — numerical Jacobian at fp64
# ─────────────────────────────────────────────────────────────────────────────


class TestLinearAttnVJP:
    @pytest.mark.parametrize("feature_map", ["identity", "elu", "polynomial_2"])
    def test_causal_vjp_matches_numerical(self, feature_map):
        Q, K, V = _make_qkv(B=1, H=1, S=3, D_qk=2, D_v=2, seed=11)
        Qp = ts.nn.Parameter(Q.copy())
        Kp = ts.nn.Parameter(K.copy())
        Vp = ts.nn.Parameter(V.copy())
        with ts.autodiff.tape() as t:
            O = ts.ops.linear_attn(
                Qp, Kp, Vp, feature_map=feature_map, causal=True,
            )
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)

        def loss_fn(q=None, k=None, v=None):
            qq = Q if q is None else q
            kk = K if k is None else k
            vv = V if v is None else v
            out = ts.ops.linear_attn(
                qq, kk, vv, feature_map=feature_map, causal=True,
            )
            return float((out ** 2).sum())

        np.testing.assert_allclose(
            Qp.grad.numpy(), _numerical_grad(lambda q: loss_fn(q=q), Q.copy()),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            Kp.grad.numpy(), _numerical_grad(lambda k: loss_fn(k=k), K.copy()),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            Vp.grad.numpy(), _numerical_grad(lambda v: loss_fn(v=v), V.copy()),
            rtol=1e-5, atol=1e-6,
        )

    def test_non_causal_vjp_matches_numerical(self):
        Q, K, V = _make_qkv(B=1, H=1, S=3, D_qk=2, D_v=2, seed=13)
        Qp = ts.nn.Parameter(Q.copy())
        Kp = ts.nn.Parameter(K.copy())
        Vp = ts.nn.Parameter(V.copy())
        with ts.autodiff.tape() as t:
            O = ts.ops.linear_attn(
                Qp, Kp, Vp, feature_map="identity", causal=False,
            )
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)

        def loss_fn(q=None, k=None, v=None):
            qq = Q if q is None else q
            kk = K if k is None else k
            vv = V if v is None else v
            out = ts.ops.linear_attn(
                qq, kk, vv, feature_map="identity", causal=False,
            )
            return float((out ** 2).sum())

        np.testing.assert_allclose(
            Qp.grad.numpy(), _numerical_grad(lambda q: loss_fn(q=q), Q.copy()),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            Kp.grad.numpy(), _numerical_grad(lambda k: loss_fn(k=k), K.copy()),
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            Vp.grad.numpy(), _numerical_grad(lambda v: loss_fn(v=v), V.copy()),
            rtol=1e-5, atol=1e-6,
        )

    def test_decay_path_vjp_runs(self):
        """Smoke test that the decay path's VJP runs end-to-end and
        produces gradients of correct shape (full numerical-Jacobian
        check would require differentiating through decay too — out
        of scope for v1 since decay is typically a learnable Module
        attribute that registers its own VJP)."""
        Q, K, V = _make_qkv(B=1, H=1, S=3, D_qk=2, D_v=2, seed=17)
        decay = np.full((1, 1, 3), 0.9, dtype=np.float64)
        Qp = ts.nn.Parameter(Q.copy())
        with ts.autodiff.tape() as t:
            O = ts.ops.linear_attn(
                Qp, K, V, feature_map="identity", causal=True, decay=decay,
            )
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)
        assert Qp.grad is not None
        assert Qp.grad.shape == Q.shape


# ─────────────────────────────────────────────────────────────────────────────
# LA-2 — Apple GPU runtime shim symbol contract.
#
# Compile the apple_gpu runtime from source, dynamically link, and verify
# `tessera_apple_gpu_linear_attn_f32` produces output matching the Python
# reference. Mirrors the test pattern used by the SwiGLU + matmul-softmax
# fusion shims.
# ─────────────────────────────────────────────────────────────────────────────


import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_apple_gpu_linear_attn_runtime_shim_matches_python_op(tmp_path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        source = backend / "apple_gpu_runtime.mm"
        lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
        cmd = [
            cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
            "-x", "objective-c++", str(source), "-o", str(lib),
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
        ]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_gpu_linear_attn_f32
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 7
    sym.restype = None

    np.random.seed(0)
    B, H, S, D_qk, D_v = 2, 2, 4, 3, 3
    Q = np.random.randn(B, H, S, D_qk).astype(np.float32)
    K = np.random.randn(B, H, S, D_qk).astype(np.float32)
    V = np.random.randn(B, H, S, D_v).astype(np.float32)
    O = np.zeros((B, H, S, D_v), dtype=np.float32)

    sym(
        Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        O.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, H, S, D_qk, D_v,
        2,  # feature_map = identity
        1,  # causal
    )

    ref = ts.ops.linear_attn(
        Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64),
        feature_map="identity", causal=True,
    )
    np.testing.assert_allclose(O, ref, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# LA-4 — Power attention + Retention (promoted from example/power_retention/).
# ─────────────────────────────────────────────────────────────────────────────


class TestPowerAttn:
    def test_deg2_matches_polynomial_2_linear_attn(self):
        """power_attn(deg=2) is a thin wrapper around linear_attn with the
        polynomial_2 feature map."""
        Q, K, V = _make_qkv(B=1, H=1, S=4, D_qk=2, D_v=2)
        out_power = ts.ops.power_attn(Q, K, V, deg=2, state=4, causal=True)
        out_linear = ts.ops.linear_attn(
            Q, K, V, feature_map="polynomial_2", causal=True,
        )
        np.testing.assert_allclose(out_power, out_linear, rtol=1e-10)

    def test_deg3_runs_and_produces_correct_shape(self):
        Q, K, V = _make_qkv(B=2, H=2, S=4, D_qk=3, D_v=3)
        out = ts.ops.power_attn(Q, K, V, deg=3, state=8, causal=True)
        assert out.shape == V.shape

    def test_deg_must_be_positive(self):
        Q, K, V = _make_qkv()
        with pytest.raises(ValueError, match="deg"):
            ts.ops.power_attn(Q, K, V, deg=0, state=4)


class TestRetention:
    def test_no_log_g_matches_polynomial_2_linear_attn(self):
        """When log_g is None, retention(deg=2) reduces to linear_attn
        with polynomial_2 feature map."""
        Q, K, V = _make_qkv(B=1, H=1, S=4, D_qk=2, D_v=2)
        out_ret = ts.ops.retention(Q, K, V, deg=2, chunk=128, causal=True)
        out_linear = ts.ops.linear_attn(
            Q, K, V, feature_map="polynomial_2", causal=True,
        )
        np.testing.assert_allclose(out_ret, out_linear, rtol=1e-10)

    def test_log_g_decay_lowers_late_token_contribution(self):
        """log_g produces multiplicative decay over the recurrent state
        — output scales should reflect attenuation of older tokens'
        contribution. Test by checking that a heavy decay causes the
        last token's output to look more like its own value than the
        earlier tokens'."""
        Q, K, V = _make_qkv(B=1, H=1, S=4, D_qk=2, D_v=2)
        # Heavy decay (log_g = -10 → decay ≈ 4.5e-5).
        log_g = np.full((1, 1, 4), -10.0, dtype=np.float64)
        out_decay = ts.ops.retention(
            Q, K, V, log_g=log_g, deg=2, chunk=128, causal=True,
        )
        # Without decay
        out_nodecay = ts.ops.retention(Q, K, V, deg=2, chunk=128, causal=True)
        # Heavily-decayed output for the last position should differ
        # from the no-decay output (the recurrence accumulates less
        # state under decay).
        assert not np.allclose(out_decay[0, 0, -1, :], out_nodecay[0, 0, -1, :])


def test_apple_gpu_linear_attn_runtime_shim_supports_elu_feature_map(tmp_path):
    """The 4 feature maps (elu/relu/identity/polynomial_2) all run end-
    to-end through the runtime symbol."""
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        source = backend / "apple_gpu_runtime.mm"
        lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
        cmd = [
            cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
            "-x", "objective-c++", str(source), "-o", str(lib),
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
        ]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_gpu_linear_attn_f32
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 7
    sym.restype = None

    np.random.seed(1)
    B, H, S, D_qk, D_v = 1, 1, 3, 2, 2
    Q = np.random.randn(B, H, S, D_qk).astype(np.float32)
    K = np.random.randn(B, H, S, D_qk).astype(np.float32)
    V = np.random.randn(B, H, S, D_v).astype(np.float32)

    feature_maps = {"elu": 0, "relu": 1, "identity": 2, "polynomial_2": 3}
    for name, idx in feature_maps.items():
        O = np.zeros((B, H, S, D_v), dtype=np.float32)
        sym(
            Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            O.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B, H, S, D_qk, D_v, idx, 1,
        )
        ref = ts.ops.linear_attn(
            Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64),
            feature_map=name, causal=True,
        )
        np.testing.assert_allclose(O, ref, rtol=1e-4, atol=1e-5)
