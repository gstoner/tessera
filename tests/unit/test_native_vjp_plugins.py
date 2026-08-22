"""E2E-REAL-6 native reverse-product family ownership."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tessera._jit_boundary import TesseraJitError

from tessera.compiler.jit import JitFn
from tessera.compiler.native_vjp_plugins import (
    execute_native_vjp_family,
    native_vjp_differential_effect_exemptions,
    native_vjp_differential_safe,
    native_vjp_plugin_available,
    native_vjp_plugin_declarations,
    native_vjp_plugin_owners,
)
from tessera.compiler.native_spectral_vjp import (
    build_native_spectral_vjp_package,
    validate_native_spectral_vjp_contract,
)


def test_normalization_vjp_declares_the_complete_compiler_spine() -> None:
    declarations = native_vjp_plugin_declarations()
    normalization = {"layer_norm", "rmsnorm", "rmsnorm_safe"}
    assert normalization < set(declarations)
    assert set(declarations) == set(native_vjp_plugin_owners())
    for op_name in normalization:
        declaration = declarations[op_name]
        declaration.validate()
        assert f"tessera.{op_name}" in declaration.graph_consumers
        assert declaration.schedule_consumer.startswith("schedule.")
        assert declaration.tile_consumer.startswith("tile.")
        assert set(declaration.target_consumers) == {
            "x86", "rocm", "nvidia_sm120", "apple_gpu"
        }


def test_compound_spectral_vjp_declares_only_proven_targets() -> None:
    declarations = native_vjp_plugin_declarations()
    for op_name in ("spectral_filter", "spectral_conv"):
        declaration = declarations[op_name]
        declaration.validate()
        assert declaration.family == "spectral_backward"
        assert declaration.schedule_consumer == "schedule.spectral_backward"
        assert declaration.tile_consumer == "tile.spectral_backward_kernel"
        assert set(declaration.target_consumers) == {
            "x86", "rocm", "nvidia_sm120"
        }


def test_attention_vjp_declares_only_canonical_rank4_consumers() -> None:
    declarations = native_vjp_plugin_declarations()
    for op_name in ("flash_attn", "gqa_attention", "mqa_attention"):
        declaration = declarations[op_name]
        declaration.validate()
        assert declaration.family == "attention_backward"
        assert declaration.schedule_consumer == "schedule.attention_backward"
        assert declaration.tile_consumer == "tile.attention_backward_kernel"
        assert set(declaration.target_consumers) == {"x86", "rocm"}
    # The public rank-3 wrapper still needs an explicit reshape/transpose
    # product before it can truthfully share the rank-4 physical package.
    assert "multi_head_attention" not in declarations


def test_loss_vjps_declare_explicit_family_and_target_ownership() -> None:
    declarations = native_vjp_plugin_declarations()
    for op_name in (
        "mse_loss",
        "smooth_l1_loss",
        "binary_cross_entropy_loss",
        "cross_entropy_loss",
    ):
        declaration = declarations[op_name]
        declaration.validate()
        assert declaration.schedule_consumer == "schedule.loss_backward"
        assert declaration.tile_consumer == "tile.loss_backward_kernel"
        assert set(declaration.target_consumers) == {
            "x86",
            "rocm",
            "nvidia_sm120",
        }
    distribution = declarations["kl_divergence"]
    assert set(distribution.target_consumers) == {"rocm"}


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
    source = SimpleNamespace(op_name="tessera.relu", result="out", kwargs={})
    assert execute_native_vjp_family(
        source=source,
        target="x86",
        ordered_inputs=(),
        arg_names=(),
        out_cotangents=(),
        wrt_names=(),
    ) is None
    assert not native_vjp_plugin_available("tessera.relu", "x86")
    assert native_vjp_plugin_available("tessera.sgd", "x86")
    assert native_vjp_plugin_available("tessera.flash_attn", "x86")
    assert not native_vjp_plugin_available("tessera.flash_attn", "apple_gpu")


def test_attention_differential_policy_rejects_active_dropout() -> None:
    deterministic = SimpleNamespace(
        op_name="tessera.flash_attn", kwargs={"dropout_p": 0.0}
    )
    stochastic = SimpleNamespace(
        op_name="tessera.flash_attn", kwargs={"dropout_p": 0.25, "seed": 7}
    )
    assert native_vjp_differential_safe(deterministic, "x86", "state")
    assert native_vjp_differential_effect_exemptions(
        deterministic, "x86", "state"
    ) == ("tessera.flash_attn",)
    assert not native_vjp_differential_safe(stochastic, "x86", "state")
    assert not native_vjp_differential_effect_exemptions(
        stochastic, "x86", "state"
    )


def test_lion_never_enters_the_reexecuting_differential_gate() -> None:
    lion = SimpleNamespace(op_name="tessera.lion", kwargs={})
    assert not native_vjp_differential_safe(lion, "x86", "pure")
    assert not native_vjp_differential_effect_exemptions(lion, "x86", "pure")


def test_nvidia_stateful_products_are_owned_by_family_plugins() -> None:
    declarations = native_vjp_plugin_declarations()
    assert declarations["lion"].target_consumers["nvidia_sm120"] == (
        "nvidia.sm120_lion_backward"
    )
    for family in (
        "gated_deltanet",
        "kimi_delta_attention",
        "modified_delta_attention",
    ):
        assert declarations[family].target_consumers["nvidia_sm120"] == (
            "nvidia.sm120_sequence_mixer_backward"
        )
        assert native_vjp_plugin_available(f"tessera.{family}", "nvidia_sm120")


def test_nvidia_lion_package_is_constructed_by_family_plugin(monkeypatch) -> None:
    import tessera.runtime as runtime

    captured = {}

    def fake_launch(artifact, values):
        captured["metadata"] = artifact.metadata
        captured["values"] = values
        return {
            "ok": True,
            "execution_mode": "cuda_driver",
            "output": tuple(np.ones(4, np.float32) for _ in range(3)),
        }

    monkeypatch.setattr(runtime, "launch", fake_launch)
    source = SimpleNamespace(
        op_name="tessera.lion", result="out", kwargs={"beta1": 0.9}
    )
    with pytest.raises(TesseraJitError, match="non-reexecuting frontend proof"):
        execute_native_vjp_family(
            source=source,
            target="nvidia_sm120",
            ordered_inputs=tuple(np.ones(4, np.float32) for _ in range(3)),
            arg_names=("param", "grad", "moment"),
            out_cotangents=tuple(np.ones(4, np.float32) for _ in range(2)),
            wrt_names=("param", "moment"),
            source_graph_ir="module { /* tracer authority */ }",
        )
    assert captured == {}


def test_spectral_vjp_package_binds_source_schedule_and_tile_lineage() -> None:
    source = SimpleNamespace(
        op_name="tessera.spectral_filter", result="out", kwargs={"axis": -1}
    )
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    filt = np.array([2 - 1j, -3 + 0.5j], dtype=np.complex64)
    dy = np.array([0.5 + 1j, -2 + 3j], dtype=np.complex64)
    package = build_native_spectral_vjp_package(
        source_graph_ir="module { /* traced */ }",
        source=source,
        target="x86",
        ordered_inputs=(x, filt),
        arg_names=("x", "filter"),
        out_cotangent=dy,
    )
    contract = package.contract()
    validate_native_spectral_vjp_contract(contract)
    assert len(package.source_graph_ir_digest) == 64
    assert len(package.schedule_artifact_hash) == 64
    assert len(package.tile_program_digest) == 64
    tampered = dict(contract, normalization="ortho")
    with np.testing.assert_raises_regex(ValueError, "Schedule artifact"):
        validate_native_spectral_vjp_contract(tampered)


def test_spectral_vjp_plugin_owns_package_and_honors_wrt(monkeypatch) -> None:
    import tessera.runtime as runtime

    captured = {}
    dx = np.array([1 + 0j, 2 + 0j], dtype=np.complex64)
    df = np.array([3 + 0j, 4 + 0j], dtype=np.complex64)

    def fake_launch(artifact, values):
        captured["metadata"] = artifact.metadata
        captured["values"] = values
        return {"ok": True, "execution_mode": "cpu_avx512", "output": (dx, df)}

    monkeypatch.setattr(runtime, "launch", fake_launch)
    source = SimpleNamespace(
        op_name="tessera.spectral_filter", result="out", kwargs={"axis": -1}
    )
    result = execute_native_vjp_family(
        source=source,
        target="x86",
        ordered_inputs=(
            np.ones(2, dtype=np.complex64),
            np.ones(2, dtype=np.complex64),
        ),
        arg_names=("x", "filter"),
        out_cotangents=np.ones(2, dtype=np.complex64),
        wrt_names=("filter",),
        source_graph_ir="module { /* tracer authority */ }",
    )
    assert result is not None
    assert result.gradients == (df,)
    assert result.execution["frontend_authority"] == "tracer"
    assert result.execution["family"] == "spectral_backward"
    contract = captured["metadata"]["native_spectral_vjp"]
    validate_native_spectral_vjp_contract(contract)
    assert len(captured["values"]) == 3


def test_spectral_vjp_rejects_missing_tracer_lineage() -> None:
    source = SimpleNamespace(
        op_name="tessera.spectral_filter", result="out", kwargs={}
    )
    with np.testing.assert_raises_regex(Exception, "tracer-owned source Graph"):
        execute_native_vjp_family(
            source=source,
            target="x86",
            ordered_inputs=(
                np.ones(2, dtype=np.complex64),
                np.ones(2, dtype=np.complex64),
            ),
            arg_names=("x", "filter"),
            out_cotangents=np.ones(2, dtype=np.complex64),
            wrt_names=("x",),
        )


def test_jitfn_no_longer_owns_normalization_backward_packaging() -> None:
    assert not hasattr(JitFn, "_native_norm_backward")
