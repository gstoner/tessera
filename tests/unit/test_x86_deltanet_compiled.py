"""The AVX-512 x86 DeltaNet / gated-delta linear-attention causal scan.

The x86 analog of the ROCm rocm_deltanet_compiled lane: an artifact stamped
``compiler_path = "x86_deltanet_compiled"`` routes through ``runtime.launch()`` →
``_execute_x86_compiled_deltanet`` → the hand-written AVX-512 kernel
(avx512_deltanet_f32), matching the numpy reference tessera._delta_attention_impl
(exposed as ops.gated_deltanet / kimi_delta_attention / modified_delta_attention).

Covers the recurrence variants (erase / modified / gate / beta / decay). Genuine
new x86 codegen — the first native x86 deltanet lane. Host-verifiable (no GPU);
skips only if the x86 elementwise lib isn't built.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import execution_matrix as em
from tessera.autodiff.vjp import get_vjp


def test_execution_matrix_has_x86_deltanet_row():
    row = em.lookup("x86", "x86_deltanet_compiled")
    assert row is not None
    assert row.executable and row.execution_kind == "native_cpu"


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


_OPS = {
    "gated_deltanet": ts.ops.gated_deltanet,
    "kimi_delta_attention": ts.ops.kimi_delta_attention,
    "modified_delta_attention": ts.ops.modified_delta_attention,
}


def _artifact(rt, op_name, names, erase, has_gate, has_beta, has_decay):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_deltanet_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": f"tessera.{op_name}", "result": "o",
                 "operands": names,
                 "kwargs": {"causal": True, "erase": erase,
                            "has_gate": has_gate, "has_beta": has_beta,
                            "has_decay": has_decay}}]})


def _backward_artifact(rt, op_name, names, erase):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_deltanet_bwd_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "grads",
        "ops": [{
            "op_name": f"tessera.{op_name}", "result": "grads",
            "operands": names,
            "kwargs": {
                "causal": True, "erase": erase, "has_gate": True,
                "has_beta": True, "has_decay": True,
                "chunk_size": 2, "workers": 2,
            },
        }],
    })


@pytest.mark.parametrize("op_name", list(_OPS))
@pytest.mark.parametrize("erase,gate,beta,decay", [
    (False, False, False, False),   # gated linear attention (base)
    (True, False, True, False),     # delta rule + beta
    (False, True, False, True),     # gate + decay
    (True, True, True, True),       # everything on
])
def test_x86_deltanet_matches_reference(op_name, erase, gate, beta, decay):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash((op_name, erase, gate, beta, decay)) % (2**32))
    B, H, S, Dqk, Dv = 2, 2, 6, 4, 4
    Q = rng.standard_normal((B, H, S, Dqk)).astype(np.float32)
    K = rng.standard_normal((B, H, S, Dqk)).astype(np.float32)
    # L2-normalize keys — DeltaNet is ill-conditioned in f32 otherwise (per the
    # reference's L1.1 note); the compiled kernel and oracle both see normed K.
    K = (K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
    V = rng.standard_normal((B, H, S, Dv)).astype(np.float32)
    g = rng.standard_normal((B, H, S, Dv)).astype(np.float32) if gate else None
    b = np.abs(rng.standard_normal((B, H, S))).astype(np.float32) if beta else None
    d = (0.5 + 0.4 * rng.random((B, H, S))).astype(np.float32) if decay else None

    names, inputs = ["q", "k", "v"], [Q, K, V]
    if gate:
        names.append("gate"); inputs.append(g)
    if beta:
        names.append("beta"); inputs.append(b)
    if decay:
        names.append("decay"); inputs.append(d)

    res = rt.launch(_artifact(rt, op_name, names, erase, gate, beta, decay),
                    tuple(inputs))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_deltanet_compiled"
    got = np.asarray(res["output"], np.float64)

    ref = np.asarray(_OPS[op_name](
        Q, K, V, gate=g, beta=b, decay=d, causal=True, erase=erase), np.float64)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-3,
                               err_msg=f"{op_name} erase={erase} "
                               f"g={gate} b={beta} d={decay}")


def test_x86_deltanet_rejects_non_causal():
    rt = _x86_or_skip()
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_deltanet_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.gated_deltanet", "operands": ["q", "k", "v"],
                 "kwargs": {"causal": False}}]})
    z = np.zeros((1, 1, 2, 2), np.float32)
    with pytest.raises(ValueError):
        rt._execute_x86_compiled_deltanet(art, (z, z, z))


@pytest.mark.parametrize(
    ("op_name", "erase"),
    [
        ("gated_deltanet", False),
        ("gated_deltanet", True),
        ("modified_delta_attention", False),
        ("modified_delta_attention", True),
    ],
)
def test_x86_deltanet_backward_matches_analytic_vjp(op_name, erase):
    rt = _x86_or_skip()
    rng = np.random.default_rng(20260728 + int(erase))
    shape = (2, 2, 5, 3)
    q = (rng.standard_normal(shape) * 0.2).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6
    v = (rng.standard_normal(shape) * 0.2).astype(np.float32)
    gate = rng.standard_normal(shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.8, shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.7, 0.95, shape[:-1]).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    values = (q, k, v, gate, beta, decay, dy)
    names = ["q", "k", "v", "gate", "beta", "decay", "dy"]
    result = rt.launch(
        _backward_artifact(rt, op_name, names, erase), values
    )
    assert result["ok"] is True, result.get("reason")
    expected = get_vjp(op_name)(
        dy, q, k, v, gate, beta, decay, erase=erase
    )
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-3)
