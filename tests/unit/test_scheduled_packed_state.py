"""Serialized packed/state artifacts must remain native Schedule descendants."""

from dataclasses import replace

import pytest

from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt, lower_scheduled_matmul
from tessera.compiler.scheduled_paged_kv import lower_scheduled_paged_kv
from tests.device.nvidia.test_e2e_spine_native import _int4_module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")


def artifact(kind):
    if kind == "int4":
        return lower_scheduled_matmul(_int4_module(3, 5, 7), target="nvidia_sm120")
    return lower_scheduled_paged_kv(("pages", "table", "out"), (3, 4, 2, 2, 3, 1, 5))


def package(kind, scheduled):
    consumer = native.package_scheduled_int4_matmul if kind == "int4" else native.package_scheduled_paged_kv
    return consumer(scheduled, pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("kind", ["int4", "paged"])
def test_native_packed_state_consumes_serialized_schedule(kind, monkeypatch):
    scheduled = artifact(kind)
    calls = []
    monkeypatch.setattr(
        native,
        "_compile_tile_ir",
        lambda text, entry: (calls.append(text) or text, "// PTX", {}, "compiler", "toolchain", (), "cold"),
    )
    result = package(kind, replace(scheduled, graph_ir="discarded"))
    assert calls == [scheduled.tile_ir]
    assert result.descriptor.provenance["schedule_digest"] == scheduled.schedule_digest
    assert not hasattr(native, "emit_int4_matmul_tile_ir")
    assert not hasattr(native, "emit_paged_kv_read_tile_ir")


@pytest.mark.parametrize("kind", ["int4", "paged"])
def test_native_packed_state_rejects_swapped_pointers(kind, monkeypatch):
    scheduled = artifact(kind)
    text = scheduled.tile_ir.replace("kernel %arg0, %arg1,", "kernel %arg1, %arg0,")
    assert text != scheduled.tile_ir
    monkeypatch.setattr(native, "_compile_tile_ir", lambda *a: pytest.fail("compiled corruption"))
    with pytest.raises(ValueError, match="replay"):
        package(kind, replace(scheduled, tile_ir=text))


@pytest.mark.parametrize("dims", [
    (3, 4, 2, 2, 3, -1, 5),
    (3, 4, 2, 2, 3, -(2**63), 5),
    (3, 4, 2, 2, 3, -6, 5),
    (3, 4, 2, 2, 3, 5, 4),
    (3, 4, 2, 2, 3, 0, 0),
])
def test_native_paged_bounds_rejected(dims):
    with pytest.raises(RuntimeError):
        lower_scheduled_paged_kv(("pages", "table", "out"), dims)


def test_packed_axes_and_odd_k_are_native():
    scheduled = artifact("int4")
    assert "tessera.packing_axes = array<i64: 1, 0>" in scheduled.tile_ir
    assert "tessera.packed_shapes = array<i64: 3, 4, 4, 5>" in scheduled.tile_ir
    with pytest.raises(ValueError, match="dimensions"):
        package("int4", replace(scheduled, k=8))


def test_paged_descriptor_cannot_change_logical_interval():
    scheduled = artifact("paged")
    with pytest.raises(ValueError, match="descriptor"):
        package("paged", replace(scheduled, dims=(3, 4, 2, 2, 3, 2, 5)))


@pytest.mark.parametrize("kind", ["int4", "paged", "checkpoint_forward", "checkpoint_backward"])
def test_driver_retains_native_stage_lineage(kind, monkeypatch):
    from tessera.compiler.driver import compile_graph_module
    from tests.device.nvidia.test_e2e_spine_native import _paged_kv_module
    from tests.unit.test_nvidia_checkpoint_pair import checkpoint_modules

    if kind == "int4":
        module = _int4_module(3, 5, 7)
    elif kind == "paged":
        module = _paged_kv_module(3, 2, 2, 3, 4, 1, 6)
    else:
        module = checkpoint_modules()[int(kind.endswith("backward"))]
    monkeypatch.setattr(
        native, "_compile_tile_ir", lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold")
    )
    bundle = compile_graph_module(
        module,
        source_origin="native_contract_test",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.lineage_complete


@pytest.mark.parametrize("override", ["layout", "strides", "storage_pack"])
def test_int4_refuses_unplumbed_physical_overrides(override):
    module = _int4_module(3, 5, 7)
    module.functions[0].body[0].kwargs[override] = "custom"
    with pytest.raises(ValueError, match="overrides"):
        lower_scheduled_matmul(module, target="nvidia_sm120")


def test_int4_gemm_alias_preserves_native_lowering():
    module = _int4_module(3, 5, 7)
    module.functions[0].body[0].op_name = "tessera.gemm"
    scheduled = lower_scheduled_matmul(module, target="nvidia_sm120")
    assert "tile.matmul_kernel" in scheduled.tile_ir


def test_int4_native_entry_rejects_unrelated_return():
    module = _int4_module(3, 3, 3)
    module.functions[0].return_values = ["%a"]
    with pytest.raises(ValueError, match="return"):
        lower_scheduled_matmul(module, target="nvidia_sm120")


@pytest.mark.parametrize("kind", ["paged", "forward", "backward"])
def test_new_native_packages_reject_unplumbed_argument_layout(kind):
    from tests.device.nvidia.test_e2e_spine_native import _paged_kv_module
    from tests.unit.test_nvidia_checkpoint_pair import checkpoint_modules
    if kind == "paged":
        module = _paged_kv_module(3, 2, 2, 3, 4, 1, 6)
        supports = native.supports_paged_kv_read
    else:
        module = checkpoint_modules()[int(kind == "backward")]
        supports = native.supports_attention_backward_lse if kind == "backward" else native.supports_attention_lse
    module.functions[0].args[0].layout = "column_major"
    assert not supports(module)


def test_physical_paged_read_has_distinct_native_mnemonic():
    scheduled = artifact("paged")
    assert "tessera.paged_kv_read" in scheduled.graph_ir
    assert "tessera.paged_kv_read" in scheduled.schedule_ir
    assert "tessera.kv_cache.read" not in scheduled.graph_ir
    assert "tile.paged_kv_read_kernel" in scheduled.tile_ir
