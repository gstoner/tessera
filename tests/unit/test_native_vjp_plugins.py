"""E2E-REAL-6 native reverse-product family ownership."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tessera.compiler.jit import JitFn
from tessera.compiler.native_vjp_plugins import (
    execute_native_vjp_family,
    native_vjp_plugin_declarations,
    native_vjp_plugin_owners,
)


def test_normalization_vjp_declares_the_complete_compiler_spine() -> None:
    declarations = native_vjp_plugin_declarations()
    assert set(declarations) == {"layer_norm", "rmsnorm", "rmsnorm_safe"}
    assert set(declarations) == set(native_vjp_plugin_owners())
    for op_name, declaration in declarations.items():
        declaration.validate()
        assert f"tessera.{op_name}" in declaration.graph_consumers
        assert declaration.schedule_consumer.startswith("schedule.")
        assert declaration.tile_consumer.startswith("tile.")
        assert set(declaration.target_consumers) == {
            "x86", "rocm", "nvidia_sm120"
        }


def test_normalization_vjp_package_is_constructed_by_family_plugin(
    monkeypatch,
) -> None:
    import tessera.runtime as runtime

    captured = {}
    dx = np.ones((2, 3), dtype=np.float32)
    dgamma = np.ones((3,), dtype=np.float32)

    def fake_launch(artifact, values):
        captured["metadata"] = artifact.metadata
        captured["values"] = values
        return {
            "ok": True,
            "execution_mode": "cpu_avx512",
            "output": (dx, dgamma),
        }

    monkeypatch.setattr(runtime, "launch", fake_launch)
    source = SimpleNamespace(
        op_name="tessera.rmsnorm",
        result="out",
        kwargs={"eps": 1.0e-5},
    )
    result = execute_native_vjp_family(
        source=source,
        target="x86",
        ordered_inputs=(np.zeros((2, 3), np.float32), np.ones(3, np.float32)),
        arg_names=("x", "gamma"),
        out_cotangents=np.ones((2, 3), np.float32),
        wrt_names=("gamma",),
    )
    assert result is not None
    assert result.gradients == (dgamma,)
    assert result.execution["family"] == "normalization"
    assert result.execution["implementation"] == "family_plugin"
    assert captured["metadata"]["native_vjp_schedule_consumer"].startswith(
        "schedule."
    )
    assert captured["metadata"]["native_vjp_tile_consumer"].startswith("tile.")
    assert captured["metadata"]["native_vjp_target_consumer"] == (
        "x86.avx512_normalization"
    )
    assert len(captured["values"]) == 3


def test_unmigrated_vjp_family_remains_a_compatibility_path() -> None:
    source = SimpleNamespace(op_name="tessera.sgd", result="out", kwargs={})
    assert execute_native_vjp_family(
        source=source,
        target="x86",
        ordered_inputs=(),
        arg_names=(),
        out_cotangents=(),
        wrt_names=(),
    ) is None


def test_jitfn_no_longer_owns_normalization_backward_packaging() -> None:
    assert not hasattr(JitFn, "_native_norm_backward")
