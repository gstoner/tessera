"""Apple GPU reference tail lane — the heterogeneous remainder.

MLA latent-KV (compress / expand_k / expand_v), alibi, lgamma / digamma,
fused_epilogue, asymmetric_bce, normalize_group_advantages, and the
speculative-decode accept ops (spec_accept / spec_accept_sample /
spec_accept_tree_sample). Apple ships no device kernel for these, so each reuses
its public tessera reference. Reachable via
`compiler_path="apple_gpu_tail_compiled"`; execution_kind=reference_cpu.
Validated vs tessera.ops / tessera.losses / tessera.rl.
"""

from __future__ import annotations

import numpy as np

from tessera import losses
from tessera import ops as O
from tessera import rl
from tessera import runtime as rt


def _run(op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_tail_compiled",
        "executable": True, "execution_kind": "reference_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs or {})}]})
    res = rt.launch(art, tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_tail_compiled"
    assert res["execution_kind"] == "reference_cpu"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(0)


def test_lgamma_digamma():
    x = np.abs(_RNG.standard_normal((4, 5)).astype(np.float32)) + 0.5
    np.testing.assert_allclose(_run("tessera.lgamma", [x]), np.asarray(O.lgamma(x)),
                               atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_run("tessera.digamma", [x]),
                               np.asarray(O.digamma(x)), atol=1e-4, rtol=1e-4)


def test_asymmetric_bce_and_group_norm():
    lg = _RNG.standard_normal((4, 5)).astype(np.float32)
    tg = _RNG.integers(0, 2, (4, 5)).astype(np.float32)
    np.testing.assert_allclose(
        _run("tessera.asymmetric_bce", [lg, tg],
             {"pos_weight": 2.0, "neg_weight": 0.5}),
        np.asarray(losses.asymmetric_bce(lg, tg, pos_weight=2.0, neg_weight=0.5)),
        atol=1e-4, rtol=1e-4)
    rew = _RNG.standard_normal((2, 4)).astype(np.float32)
    np.testing.assert_allclose(
        _run("tessera.normalize_group_advantages", [rew], {"group_axis": 1}),
        np.asarray(rl.normalize_group_advantages(rew, group_axis=1)),
        atol=1e-4, rtol=1e-4)


def test_alibi():
    np.testing.assert_allclose(
        _run("tessera.alibi", [], {"num_heads": 4, "seq_len": 6}),
        np.asarray(O.alibi(num_heads=4, seq_len=6)), atol=1e-4, rtol=1e-4)


def test_mla_latent_kv():
    x = _RNG.standard_normal((3, 4)).astype(np.float32)
    w_dkv = _RNG.standard_normal((4, 2)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.latent_kv_compress", [x, w_dkv]),
                               np.asarray(O.latent_kv_compress(x, w_dkv)),
                               atol=1e-4, rtol=1e-4)
    c = _RNG.standard_normal((3, 2)).astype(np.float32)
    w_uk = _RNG.standard_normal((2, 4)).astype(np.float32)
    np.testing.assert_allclose(_run("tessera.latent_kv_expand_k", [c, w_uk]),
                               np.asarray(O.latent_kv_expand_k(c, w_uk)),
                               atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_run("tessera.latent_kv_expand_v", [c, w_uk]),
                               np.asarray(O.latent_kv_expand_v(c, w_uk)),
                               atol=1e-4, rtol=1e-4)


def test_fused_epilogue():
    x = _RNG.standard_normal((3, 4)).astype(np.float32)
    bias = _RNG.standard_normal((4,)).astype(np.float32)
    np.testing.assert_allclose(
        _run("tessera.fused_epilogue", [x, bias], {"activation": "relu"}),
        np.asarray(O.fused_epilogue(x, bias=bias, activation="relu")),
        atol=1e-4, rtol=1e-4)


def test_spec_accept():
    draft = _RNG.integers(0, 5, (2, 3), dtype=np.int32)
    target = _RNG.integers(0, 5, (2, 4), dtype=np.int32)
    np.testing.assert_array_equal(_run("tessera.spec_accept", [draft, target]),
                                  np.asarray(O.spec_accept(draft, target)))


def test_spec_accept_sample():
    D, V = 3, 4
    draft = _RNG.integers(0, V, (D,)).astype(np.int32)
    tp = np.abs(_RNG.standard_normal((D + 1, V))).astype(np.float32)
    dp = np.abs(_RNG.standard_normal((D, V))).astype(np.float32)
    au = _RNG.random((D,)).astype(np.float32)
    ru = _RNG.random((1,)).astype(np.float32)
    np.testing.assert_array_equal(
        _run("tessera.spec_accept_sample", [draft, tp, dp, au, ru]),
        np.asarray(O.spec_accept_sample(draft, tp, dp, au, ru)))


def test_spec_accept_tree_sample():
    P, D = 2, 3
    tlp = _RNG.standard_normal((P, D)).astype(np.float32)
    dlp = _RNG.standard_normal((P, D)).astype(np.float32)
    au = _RNG.random((P, D)).astype(np.float32)
    np.testing.assert_array_equal(
        _run("tessera.spec_accept_tree_sample", [tlp, dlp, au]),
        np.asarray(O.spec_accept_tree_sample(tlp, dlp, au)))
