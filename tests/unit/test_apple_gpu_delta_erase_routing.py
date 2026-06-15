"""Track L (graph→Metal erase routing) — `gated_deltanet(erase=True)` must run
the *genuine* DeltaNet rule on the runtime path, not the composed linear form.

The runtime dispatcher (`_apple_gpu_dispatch_delta_attn`) routes `erase=True`
(non-modified) to the L1.1 kernel via `_apple_gpu_delta_true_rule`; `erase=False`
(default) keeps the backward-compatible linear form.  Keys are L2-normalized (the
L1.1 conditioning finding) so f32 ≡ f64.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import runtime as R
from tessera.stdlib import delta_rule as dr

_GPU = agb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime unavailable")

_B, _H, _S, _D = 2, 3, 16, 16


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _qkv(seed=0):
    rng = np.random.default_rng(seed)
    Q = _normalize(rng.standard_normal((_B, _H, _S, _D))).astype(np.float32)
    K = _normalize(rng.standard_normal((_B, _H, _S, _D))).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    return Q, K, V


def _bd(seed=1):
    rng = np.random.default_rng(seed)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((_B, _H, _S))))).astype(np.float32)
    decay = (1.0 / (1.0 + np.exp(-(rng.standard_normal((_B, _H, _S)) + 2)))).astype(np.float32)
    return beta, decay


# Module-level @jit fn (source is inspectable, unlike a REPL/heredoc def).
@ts.jit(target="apple_gpu")
def _delta_true(q, k, v, b, d):
    return ts.ops.gated_deltanet(q, k, v, beta=b, decay=d, erase=True)


def test_dispatcher_erase_true_is_genuine_rule():
    Q, K, V = _qkv(2)
    beta, decay = _bd(3)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


def test_dispatcher_erase_false_default_is_linear():
    """Backward compatibility: the default still routes to the linear form."""
    Q, K, V = _qkv(4)
    beta, decay = _bd(5)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True}, np)
    lin = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=False)
    np.testing.assert_allclose(np.asarray(out), lin, rtol=1e-4, atol=1e-4)


def test_dispatcher_erase_with_output_gate():
    Q, K, V = _qkv(6)
    beta, decay = _bd(7)
    gate = np.random.default_rng(8).standard_normal((_B, _H, _S, _D)).astype(np.float32)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "gate": gate, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, gate=gate, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


@gpu
def test_jit_path_threads_erase_end_to_end():
    """Full @jit(target='apple_gpu') path must carry erase=True to the runtime."""
    Q, K, V = _qkv(9)
    beta, decay = _bd(10)
    y = np.asarray(_delta_true(Q, K, V, beta, decay))
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
