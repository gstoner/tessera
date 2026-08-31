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
gpu = pytest.mark.hardware_apple_gpu

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


@ts.jit(target="apple_gpu")
def _delta_all_optionals(q, k, v, g, b, d):
    return ts.ops.gated_deltanet(q, k, v, gate=g, beta=b, decay=d, erase=True)


@ts.jit(target="apple_gpu")
def _delta_decay_only(q, k, v, d):
    return ts.ops.gated_deltanet(q, k, v, decay=d, erase=True)


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


def test_jit_emits_erase_attribute_in_graph_ir():
    """item 1: erase is first-class — the @jit graph IR text carries it (not just
    a Python kwarg).  Paired with the ODS attr (`gated_deltanet_erase.mlir`)."""
    art = _delta_true.runtime_artifact()
    txt = str(getattr(art, "graph_ir", "") or "")
    if not txt:
        txt = str(getattr(_delta_true, "to_mlir", lambda: "")())
    assert "gated_deltanet" in txt and "erase" in txt


# ── item 2: kimi_delta_attention(erase=True) also routes to the genuine kernel ──
def test_kimi_delta_erase_routes_to_genuine_rule():
    """kimi_delta_attention is modified=False, so erase=True routes to the same
    genuine DeltaNet kernel (its numpy reference is the genuine rule too)."""
    Q, K, V = _qkv(11)
    beta, decay = _bd(12)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.kimi_delta_attention", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


def test_kimi_delta_default_is_linear():
    Q, K, V = _qkv(13)
    beta, decay = _bd(14)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.kimi_delta_attention", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True}, np)
    lin = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=False)
    np.testing.assert_allclose(np.asarray(out), lin, rtol=1e-4, atol=1e-4)


# ── item 3: prefill routes to the chunked kernel; both envelopes stay correct ──
@gpu
def test_prefill_chunked_envelope_dv16_is_correct():
    """S>1, D_v≤16 → the faster chunked (L2.2 coop) kernel; chunk ≡ recurrent."""
    Q, K, V = _qkv(15)   # D_v = 16
    beta, decay = _bd(16)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


@gpu
def test_recurrent_envelope_dv32_is_correct():
    """D_v=32 (>16, ≤64) is outside the chunked envelope → recurrent kernel."""
    rng = np.random.default_rng(17)
    B, H, S, Dqk, Dv = 2, 2, 12, 8, 32           # Dqk*Dv = 256 (in recurrent env)
    Q = _normalize(rng.standard_normal((B, H, S, Dqk))).astype(np.float32)
    K = _normalize(rng.standard_normal((B, H, S, Dqk))).astype(np.float32)
    V = rng.standard_normal((B, H, S, Dv)).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, S))))).astype(np.float32)
    decay = (1.0 / (1.0 + np.exp(-(rng.standard_normal((B, H, S)) + 2)))).astype(np.float32)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


# ── the optional-operand ABI ────────────────────────────────────────────────
#
# `gate`/`beta`/`decay` are optional TENSOR operands, so the compiled path
# carries them as trailing SSA values rather than attributes. Two facts have to
# survive that, and neither did.
#
# ORDER. Until `tessera.gated_deltanet` was declared in
# `graph_ir._KEYWORD_OPERANDS`, keyword operands were appended sorted by name,
# so `gated_deltanet(q, k, v, gate=g, beta=b, decay=d)` emitted them as
# (beta, decay, gate) against an ABI of (gate, beta, decay) — every optional
# bound to the wrong slot by a positional reader.
#
# PRESENCE. Order alone is not enough: given `[Q, K, V, %x]` the one optional
# sits at index 3 whichever slot it fills, so the IR must also say which slots
# are filled. That is what `has_gate`/`has_beta`/`has_decay` are for — the same
# flags `_execute_rocm_compiled_deltanet` and the SM120 lane already read.
#
# These are host-free: they assert what the frontend emits, not what a GPU
# computes.


def _emitted_op(fn, *args):
    fn(*args)
    ops = ((fn.runtime_artifact().metadata or {}).get("ops")) or []
    assert len(ops) == 1, f"expected one op, got {[o.get('op_name') for o in ops]}"
    return ops[0]


def _gate():
    return np.random.default_rng(8).standard_normal((_B, _H, _S, _D)).astype(np.float32)


def test_optional_operands_emit_in_abi_order_not_alphabetical():
    """(gate, beta, decay) — not the (beta, decay, gate) that sorting gives."""
    Q, K, V = _qkv(9)
    beta, decay = _bd(10)
    op = _emitted_op(_delta_all_optionals, Q, K, V, _gate(), beta, decay)
    names = [str(n) for n in op.get("operands", [])]
    assert names == ["q", "k", "v", "g", "b", "d"], (
        f"optional operands are out of ABI order: {names}. Sorted-by-name "
        "gives ['q','k','v','b','d','g'], which binds beta->gate, "
        "decay->beta, gate->decay."
    )


def test_presence_flags_identify_which_optionals_are_bound():
    """Position cannot carry this: one optional is at index 3 either way."""
    Q, K, V = _qkv(9)
    beta, decay = _bd(10)

    both = _emitted_op(_delta_true, Q, K, V, beta, decay).get("kwargs") or {}
    assert (both.get("has_gate"), both.get("has_beta"), both.get("has_decay")) \
        == (False, True, True), both

    only_decay = _emitted_op(_delta_decay_only, Q, K, V, decay).get("kwargs") or {}
    assert (only_decay.get("has_gate"), only_decay.get("has_beta"),
            only_decay.get("has_decay")) == (False, False, True), only_decay
    assert len(_emitted_op(_delta_decay_only, Q, K, V, decay)
               .get("operands", [])) == 4


def test_dispatcher_refuses_to_guess_undecodable_operands():
    """Trailing operands with no flags must raise, not silently drop.

    This is the exact shape of the original defect: the dispatcher read the
    optionals from kwargs only, found none, and computed the *unweighted* rule
    while returning it as the requested one. Nothing failed — the extra
    operands were dropped in silence.
    """
    Q, K, V = _qkv(9)
    beta, decay = _bd(10)
    with pytest.raises(ValueError, match="has_gate/has_beta/has_decay"):
        R._apple_gpu_dispatch_delta_attn(
            "tessera.gated_deltanet", [Q, K, V, beta, decay],
            {"causal": True, "erase": True}, np)


def test_dispatcher_rejects_flags_that_disagree_with_the_operands():
    Q, K, V = _qkv(9)
    beta, _ = _bd(10)
    with pytest.raises(ValueError, match="flags declare"):
        R._apple_gpu_dispatch_delta_attn(
            "tessera.gated_deltanet", [Q, K, V, beta],
            {"causal": True, "erase": True,
             "has_beta": True, "has_decay": True}, np)


def test_dispatcher_still_accepts_the_eager_kwargs_convention():
    """The numpy path has no IR and legitimately passes optionals as kwargs."""
    Q, K, V = _qkv(2)
    beta, decay = _bd(3)
    out = R._apple_gpu_dispatch_delta_attn(
        "tessera.gated_deltanet", [Q, K, V],
        {"beta": beta, "decay": decay, "causal": True, "erase": True}, np)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, erase=True)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


@gpu
def test_jit_path_threads_every_optional_end_to_end():
    """All three optionals through @jit — the case the order bug broke."""
    Q, K, V = _qkv(9)
    beta, decay = _bd(10)
    gate = _gate()
    y = np.asarray(_delta_all_optionals(Q, K, V, gate, beta, decay))
    ref = dr.gated_delta_rule_recurrent(
        Q, K, V, beta=beta, decay=decay, gate=gate, erase=True)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_jit_path_with_one_optional_binds_the_right_slot():
    """`decay` alone must bind to decay, not to another optional slot.

    The control is the **beta** slot, not the gate slot. `gate` is
    (B, H, S, D_v) while `beta` and `decay` are both (B, H, S), so misbinding
    decay as gate raises a shape error and would be caught anywhere. Beta is
    the slot that accepts it silently and returns a different recurrence —
    the only confusion that can reach a user as wrong numbers.
    """
    Q, K, V = _qkv(9)
    _, decay = _bd(10)
    y = np.asarray(_delta_decay_only(Q, K, V, decay))
    ref = dr.gated_delta_rule_recurrent(Q, K, V, decay=decay, erase=True)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)
    misbound = dr.gated_delta_rule_recurrent(Q, K, V, beta=decay, erase=True)
    assert not np.allclose(y, misbound, rtol=1e-4, atol=1e-4), (
        "decay bound into the beta slot is numerically indistinguishable for "
        "these inputs, so this test cannot detect the defect it exists for"
    )
