"""ROCm device_verified_jit-composite metric/contrastive losses."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_metric_loss_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
    })


def test_rocm_metric_losses_match_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(61)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    y = rng.standard_normal((5, 4)).astype(np.float32)
    z = rng.standard_normal((5, 4)).astype(np.float32)
    target = np.array([1, 0, 1, 0, 1], np.float32)
    cases = [
        ("tessera.wasserstein_distance", ("x", "y"), (x, y), {}, losses.wasserstein_distance(x, y)),
        ("tessera.cosine_embedding_loss", ("x", "y", "target"), (x, y, target), {"margin": 0.2}, losses.cosine_embedding_loss(x, y, target, margin=0.2)),
        ("tessera.contrastive_loss", ("x", "y", "target"), (x, y, target), {"margin": 0.7}, losses.contrastive_loss(x, y, target, margin=0.7)),
        ("tessera.triplet_loss", ("x", "y", "z"), (x, y, z), {"margin": 0.5}, losses.triplet_loss(x, y, z, margin=0.5)),
    ]
    for op_name, names, args, kwargs, ref in cases:
        res = rt.launch(_artifact(rt, op_name, names, kwargs), args)
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_metric_loss_compiled"
        np.testing.assert_allclose(res["output"], ref, atol=2e-4, rtol=2e-4)


def test_rocm_info_nce_nt_xent_and_seq2seq_match_reference_on_gpu():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(63)
    q = rng.standard_normal((4, 3)).astype(np.float32)
    p = rng.standard_normal((4, 3)).astype(np.float32)
    neg = rng.standard_normal((4, 5, 3)).astype(np.float32)
    res = rt.launch(
        _artifact(rt, "tessera.info_nce_loss", ("q", "p", "neg"), {"temperature": 0.2}),
        (q, p, neg),
    )
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(res["output"], losses.info_nce_loss(q, p, neg, temperature=0.2), atol=2e-4, rtol=2e-4)

    emb = rng.standard_normal((6, 4)).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2], np.int64)
    res = rt.launch(
        _artifact(rt, "tessera.nt_xent_loss", ("emb", "labels"), {"temperature": 0.5}),
        (emb, labels),
    )
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(res["output"], losses.nt_xent_loss(emb, labels, temperature=0.5), atol=2e-4, rtol=2e-4)

    logits = rng.standard_normal((2, 3, 5)).astype(np.float32)
    targets = np.array([[0, 1, 2], [3, 4, 1]], np.int64)
    mask = np.array([[1, 1, 0], [1, 0, 1]], np.float32)
    res = rt.launch(
        _artifact(rt, "tessera.seq2seq_loss", ("logits", "targets", "mask")),
        (logits, targets, mask),
    )
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(res["output"], losses.seq2seq_loss(logits, targets, mask), atol=2e-4, rtol=2e-4)
