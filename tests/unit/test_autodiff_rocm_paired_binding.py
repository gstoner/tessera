"""ROCm Tier-3 backward kernels satisfy the public paired ABI seam."""

from __future__ import annotations

import numpy as np

import tessera as ts


@ts.jit(target="rocm", autodiff="reverse", wrt=("prediction", "target"))
def _mse(prediction, target):
    return ts.ops.mse_loss(prediction, target, reduction="mean")


def test_rocm_mse_backward_binds_compiled_vjp(monkeypatch):
    import tessera.runtime as runtime

    seen = {}

    def fake_launch(artifact, args):
        seen["metadata"] = artifact.metadata
        seen["args"] = args
        return {
            "ok": True,
            "execution_mode": "hip_runtime",
            "output": (np.ones_like(args[0]), -np.ones_like(args[1])),
        }

    monkeypatch.setattr(runtime, "launch", fake_launch)
    prediction = np.zeros((3, 5), np.float32)
    target = np.ones_like(prediction)
    gradients = _mse.native_backward(
        prediction, target, out_cotangents=np.asarray(1.0, np.float32)
    )
    assert len(gradients) == 2
    assert seen["metadata"]["compiler_path"] == "rocm_regression_loss_bwd_compiled"
    assert seen["metadata"]["ops"][0]["kwargs"]["reduction"] == "mean"
    assert _mse.last_backward_execution["implementation"] == "family_plugin"


@ts.jit(target="rocm", autodiff="reverse", wrt=("q", "k", "v"))
def _flash(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def test_rocm_flash_backward_binds_paired_inputs_to_verified_lane(monkeypatch):
    from types import SimpleNamespace

    from tessera.compiler import native_attention_vjp

    seen = {}

    package = SimpleNamespace(
        operand_names=("q", "k", "v"),
        scheduled=SimpleNamespace(lse_checkpoint_selection="recompute"),
        source_graph_ir_digest="1" * 64,
        schedule_artifact_hash="2" * 64,
        tile_program_digest="3" * 64,
        native_image_digest="4" * 64,
        artifact_hash="5" * 64,
    )

    def fake_build(**kwargs):
        seen["build"] = kwargs
        return package

    def fake_execute(_package, **kwargs):
        seen["execute"] = kwargs
        return tuple(np.zeros_like(value, dtype=np.float32) for value in kwargs["ordered_inputs"])

    monkeypatch.setattr(native_attention_vjp, "build_native_attention_vjp_package", fake_build)
    monkeypatch.setattr(native_attention_vjp, "execute_native_attention_vjp_package", fake_execute)
    q = np.zeros((1, 2, 4, 16), np.float16)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    dout = np.ones_like(q)
    grads = _flash.native_backward(q, k, v, out_cotangents=dout)

    assert len(grads) == 3
    assert seen["build"]["out_cotangent"] is dout
    assert seen["build"]["source"].kwargs["causal"] is True
    assert seen["execute"]["out_cotangent"] is dout
    assert _flash.last_backward_execution["evidence_target"] == "rocm_gfx1151"
    assert _flash.last_backward_execution["implementation"] == "family_plugin"
    assert _flash.last_backward_execution["schedule_consumer"] == (
        "schedule.attention_backward"
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("a", "b"))
def _matmul(a, b):
    return ts.ops.matmul(a, b)


def test_rocm_matmul_backward_is_two_forward_gemm_launches(monkeypatch):
    import tessera.runtime as runtime

    calls = []

    def fake_launch(artifact, args):
        calls.append((artifact.metadata, args))
        return {"ok": True, "execution_mode": "hip_runtime",
                "output": np.asarray(args[0], np.float32) @ np.asarray(args[1], np.float32)}

    monkeypatch.setattr(runtime, "launch", fake_launch)
    a = np.arange(12, dtype=np.float16).reshape(3, 4)
    b = np.arange(20, dtype=np.float16).reshape(4, 5)
    dout = np.ones((3, 5), np.float16)
    da, db = _matmul.native_backward(a, b, out_cotangents=dout)

    assert len(calls) == 2
    assert all(call[0]["compiler_path"] == "rocm_compiled" for call in calls)
    np.testing.assert_array_equal(da, dout.astype(np.float32) @ b.astype(np.float32).T)
    np.testing.assert_array_equal(db, a.astype(np.float32).T @ dout.astype(np.float32))
    assert _matmul.last_backward_execution["implementation"] == (
        "family_plugin_composition"
    )
