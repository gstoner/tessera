"""F2: native scheduled unary envelopes, without Graph-owned packaging."""
from dataclasses import replace

import pytest

from tessera.compiler import nvidia_native, scheduled_kernel
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_scheduled_kernel_consumers import _module


@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_nvidia_f32_scheduled_kernel_admission(family):
    module = _module(family=family, target="nvidia_sm120")
    assert scheduled_kernel.supports_scheduled_kernel(module, target="nvidia_sm120")
    arg = module.functions[0].args[0]
    arg.ir_type = replace(arg.ir_type, dtype="bf16")
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="nvidia_sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_nvidia_scheduled_kernel_native_boundary(family, monkeypatch):
    module = _module(family=family, target="nvidia_sm120")
    artifact = scheduled_kernel.lower_scheduled_kernel(module, target="nvidia_sm120")
    assert "func.func" not in artifact.tile_ir
    assert "nvvm.kernel" in artifact.tile_ir
    assert "bufferization" not in artifact.tile_ir
    assert 'exp_mode = "approx_exp2"' in artifact.tile_ir if family == "softmax" else 'schedule = "serial"' in artifact.tile_ir
    # Native replay needs only the durable Schedule IR, not the original object.
    assert run_tessera_opt(find_tessera_opt(), artifact.schedule_ir, "--tessera-schedule-to-tile") == artifact.tile_ir
    module.functions.clear()
    calls = []

    def compile_tile(text, entry):
        calls.append(text)
        return text, "// PTX", {}, "compiler", "toolchain", (), "cold"

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph packaging was called")

    monkeypatch.setattr(nvidia_native, "_compile_tile_ir", compile_tile)
    monkeypatch.setattr(nvidia_native, "package_softmax", forbidden)
    monkeypatch.setattr(nvidia_native, "package_reduction", forbidden)
    package = nvidia_native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert calls == [artifact.tile_ir]
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert nvidia_native.package_scheduled_kernel(
        replace(artifact, graph_ir="discarded"), pipeline_name="tessera-nvidia-pipeline-sm120"
    ) == package
    with pytest.raises(ValueError, match="shape|axis"):
        nvidia_native.package_scheduled_kernel(replace(artifact, input_shape=(99,)), pipeline_name="tessera-nvidia-pipeline-sm120")
    altered = artifact.schedule_ir.replace('workgroup_size = 128', 'workgroup_size = 64', 1)
    with pytest.raises(RuntimeError, match="altered after hashing"):
        run_tessera_opt(find_tessera_opt(), altered, "--tessera-schedule-to-tile")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_nvidia_softmax_schedule_refuses_wrong_exp_policy():
    artifact = scheduled_kernel.lower_scheduled_kernel(
        _module(family="softmax", target="nvidia_sm120"), target="nvidia_sm120"
    )
    with pytest.raises(RuntimeError, match="policy must match"):
        run_tessera_opt(find_tessera_opt(), artifact.schedule_ir.replace('"approx_exp2"', '"accurate"'), "--tessera-schedule-to-tile")


@pytest.mark.parametrize("kind", ["sum", "mean", "max", "min"])
def test_legacy_canonical_reduction_preserves_kind(kind):
    module = _module(family="reduce", target="nvidia_sm120")
    op = module.functions[0].body[0]
    op.op_name = "tessera.reduce"
    op.kwargs["kind"] = kind
    assert nvidia_native._reduction_contract(module)[1] == kind
    op.kwargs["kind"] = "unknown"
    assert nvidia_native._reduction_contract(module) is None


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_nvidia_schedule_does_not_discard_extra_function_work():
    artifact = scheduled_kernel.lower_scheduled_kernel(
        _module(family="softmax", target="nvidia_sm120"), target="nvidia_sm120"
    )
    changed = artifact.schedule_ir.replace("    %0 =", "    %extra = arith.constant 1 : i64\n    %0 =", 1)
    assert changed != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="isolated unary function"):
        run_tessera_opt(find_tessera_opt(), changed, "--tessera-schedule-to-tile")


@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_direct_unary_clients_use_native_schedule(monkeypatch, family):
    module = _module(family=family, target="nvidia_sm120")
    calls = []
    artifact = object()
    result = object()

    def lower(value, *, target):
        assert value is module and target == "nvidia_sm120"
        calls.append("lower")
        return artifact

    def package(value, *, pipeline_name):
        assert value is artifact
        calls.append("package")
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("migrated direct client used Graph emission")

    monkeypatch.setattr(scheduled_kernel, "lower_scheduled_kernel", lower)
    monkeypatch.setattr(nvidia_native, "package_scheduled_kernel", package)
    monkeypatch.setattr(nvidia_native, "_package_graph_softmax", forbidden)
    monkeypatch.setattr(nvidia_native, "_package_graph_reduction", forbidden)
    assert nvidia_native.package_native(module, pipeline_name="pipeline") is result
    assert calls == ["lower", "package"]


@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_direct_migrated_unary_requires_native_compiler(monkeypatch, family):
    monkeypatch.setattr(scheduled_kernel, "find_tessera_opt", lambda: None)
    with pytest.raises(RuntimeError, match="requires production tessera-opt"):
        nvidia_native.package_native(_module(family=family, target="nvidia_sm120"), pipeline_name="pipeline")


def test_explicit_cooperative_reduction_keeps_owning_route(monkeypatch):
    module = _module(family="reduce", target="nvidia_sm120")
    sentinel = object()
    calls = []

    def package(value, *, pipeline_name, schedule):
        calls.append(schedule)
        return sentinel

    monkeypatch.setattr(nvidia_native, "_package_graph_reduction", package)
    assert nvidia_native.package_native(module, pipeline_name="pipeline", options={
        "nvidia_reduction_schedule": "cooperative_128"
    }) is sentinel
    assert calls == ["cooperative_128"]


@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_narrow_direct_unary_retains_unmigrated_route(monkeypatch, family):
    from tessera.compiler.graph_ir import tensor_ir_type

    module = _module(family=family, target="nvidia_sm120")
    fn = module.functions[0]
    fn.args[0].ir_type = tensor_ir_type((2, 3, 5), "fp16")
    if family == "softmax":
        fn.result_types[0] = fn.args[0].ir_type
    sentinel = object()
    monkeypatch.setattr(scheduled_kernel, "find_tessera_opt", lambda: None)
    monkeypatch.setattr(nvidia_native, "_package_graph_softmax", lambda *a, **kw: sentinel)
    monkeypatch.setattr(nvidia_native, "_package_graph_reduction", lambda *a, **kw: sentinel)
    assert nvidia_native.package_native(module, pipeline_name="pipeline") is sentinel
