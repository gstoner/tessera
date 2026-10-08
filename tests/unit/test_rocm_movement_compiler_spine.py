"""Native movement compiler-spine, graph binding and exact-device contracts."""
import copy
from dataclasses import replace
import os

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_e2e_movement import _paged_module, _moe_module
from tessera import runtime as rt
from tessera.compiler import rocm_native
from tessera.compiler.canonical_compile import (
    canonical_compile, _extract_primary_op, _extract_component_ops,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.scheduled_paged_kv import lower_scheduled_paged_kv_graph
from tessera.compiler.scheduled_moe_dispatch import lower_scheduled_moe_dispatch

CASES = (("gfx1151", "paged_kv"), ("gfx1201", "paged_kv"), ("gfx1151", "moe_dispatch"))


def case(arch, family):
    if family == "paged_kv":
        module = _paged_module(4, 4, 3, 8, 1, 5)
        artifact = lower_scheduled_paged_kv_graph(module, target="rocm_"+arch)
        packager = rocm_native.package_paged_kv_read
    else:
        module = _moe_module(7, 9, 13)
        artifact = lower_scheduled_moe_dispatch(module, target="rocm_"+arch)
        packager = rocm_native.package_moe_dispatch
    return module, artifact, packager


@pytest.mark.parametrize("arch,family", CASES)
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
def test_movement_projection_preserves_graph_and_binds_admitted_artifact(arch, family):
    original = _paged_module(4, 4, 3, 8, 1, 5) if family == "paged_kv" else _moe_module(7, 9, 13)
    before = copy.deepcopy(original)
    module, artifact, packager = case(arch, family)
    assert module == before
    changed = replace(artifact, graph_ir=artifact.graph_ir.replace(
        '"'+("rocm_"+arch)+'"', '"rocm_gfx942"'))
    assert changed.graph_ir != artifact.graph_ir
    with pytest.raises(ValueError, match="caller Graph or target"):
        packager(module, pipeline_name="tessera-lower-to-rocm", architecture=arch,
                 scheduled_artifact=changed)
    changed = replace(artifact, tile_ir=artifact.tile_ir.replace(
        artifact.schedule_digest, "f"*64))
    with pytest.raises(ValueError, match="native Schedule replay|Schedule replay"):
        packager(module, pipeline_name="tessera-lower-to-rocm", architecture=arch,
                 scheduled_artifact=changed)
    changed = copy.deepcopy(module)
    changed.functions[0].name += "_different_caller"
    with pytest.raises(ValueError, match="caller Graph or target"):
        packager(changed, pipeline_name="tessera-lower-to-rocm", architecture=arch,
                 scheduled_artifact=artifact)


def test_canonical_gate_names_preserve_registered_dotted_cache_identity():
    module = _paged_module(4, 4, 3, 8, 1, 5)
    assert _extract_primary_op(module) == "kv_cache_read"
    assert _extract_component_ops(module) == ("kv_cache_read",)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("arch,family", CASES)
def test_canonical_movement_spine_executes_on_owning_device(arch, family):
    if os.environ.get("TESSERA_ROCM_MOVEMENT_DEVICE_PROOF") != "1":
        pytest.skip("explicit owning-device movement gate")
    if rt._rocm_live_arch() != arch:
        pytest.skip("different owning architecture")
    module = _paged_module(4, 4, 3, 8, 1, 5) if family == "paged_kv" else _moe_module(7, 9, 13)
    result = canonical_compile(module, target="rocm_"+arch, enable_tool_validation=True)
    assert result.executable, result.reason
    bundle = result.bundle
    assert bundle.execution_kind == "native_gpu"
    assert bundle.schedule.producer == "tessera-opt.tessera-graph-to-schedule"
    assert bundle.schedule.representation == "mlir"
    assert bundle.schedule.input_digest == bundle.graph.output_digest
    assert bundle.tile.producer == "tessera-opt.tessera-schedule-to-tile"
    assert bundle.tile.input_digest == bundle.schedule.output_digest
    assert bundle.target_ir.input_digest == bundle.tile.output_digest
    assert bundle.backend.input_digest == bundle.target_ir.output_digest
    assert bundle.native_image.target == "rocm_"+arch
    assert bundle.launch_descriptor.provenance["schedule_digest"] in bundle.schedule.text
    rng = np.random.default_rng(120509)
    if family == "paged_kv":
        x = rng.normal(size=(4, 4, 3, 8)).astype(np.float32)
        indices = np.array([2, 0, 3, 1], np.int32)
        expected = x[indices].reshape(16, 3, 8)[1:6]
        args = dict(pages=x, page_table=indices, P=4, LP=4, PageSize=4,
                    H=3, D=8, Start=1, Tokens=5, slice=np.zeros_like(expected))
    else:
        x = rng.normal(size=(7, 13)).astype(np.float32)
        indices = rng.integers(0, 7, size=9, dtype=np.int32)
        expected = x[indices]
        args = dict(x=x, token=indices, o=np.zeros_like(expected), T=7, S=9, H=13)
    launch = rt.launch(result.to_runtime_artifact(), args)
    assert launch["ok"], launch
    assert launch["execution_kind"] == "native_gpu"
    np.testing.assert_array_equal(launch["output"], expected)
    rt._clear_rocm_native_image_cache()


def test_rocm_descriptor_matrix_adapter_requires_its_actual_image():
    with pytest.raises(ValueError, match="exact-target native image"):
        rt._execute_rocm_native_descriptor(rt.RuntimeArtifact(), {})


def test_gfx1201_paged_default_admission_keeps_explicit_opt_out(monkeypatch):
    from tessera.compiler.driver import canonical_compile_options

    module = _paged_module(4, 4, 3, 8, 1, 5)
    monkeypatch.setattr(rocm_native, "native_packaging_available", lambda: True)
    assert canonical_compile_options(module, target="rocm_gfx1201")["package_native"]
    assert not canonical_compile_options(
        module, target="rocm_gfx1201", options={"package_native": False})["package_native"]
    monkeypatch.setattr(rocm_native, "native_packaging_available", lambda: False)
    assert not canonical_compile_options(module, target="rocm_gfx1201")["package_native"]
