"""Checked six-buffer package admission and exact-device conversion."""
from dataclasses import replace
import os
import subprocess
import ml_dtypes
import numpy as np
import pytest
from tessera.compiler.rocm_nvfp4_ingest_native import (
    package_nvfp4_ingest_graph, execute_nvfp4_ingest)
from tessera.compiler.rocm_nvfp4_ingest import (
    reference_nvfp4_requantize,nvfp4_requantization_policy)
from tests.unit.test_rocm_graph_nvfp4_ingest import graph

@pytest.fixture(scope="module")
def package():
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip("matching ROCm compiler required")
    help_result=subprocess.run([str(tool),"--help"],text=True,capture_output=True)
    if "--generate-rocm-fpquant-kernel" not in help_result.stdout:
        pytest.skip("ROCm materializer is not built on this host")
    return package_nvfp4_ingest_graph(graph())

def operands():
    rng=np.random.default_rng(120164)
    return (rng.integers(0,256,(7,32),dtype=np.uint8),
        rng.choice(np.array([0,.125,.5,1,2,6]),(7,4)).astype(ml_dtypes.float8_e4m3fn),
        np.array([.5,2.],np.float64))

def test_package_retains_complete_native_contract(package):
    package.validate()
    assert len(package.native.descriptor.buffers) == 6
    assert package.native.descriptor.geometry.workgroup == (256,1,1)
    assert package.native.descriptor.provenance["numeric_policy"] == nvfp4_requantization_policy()
    assert "tile.nvfp4_requantize_kernel" in package.native.tile_ir

@pytest.mark.parametrize("field,value",[
    ("abi_id","different"),
    ("geometry",None),
    ("provenance",{}),
])
def test_package_rejects_changed_descriptor(package,field,value):
    if field == "geometry":
        value=replace(package.native.descriptor.geometry,workgroup=(128,1,1))
    descriptor=replace(package.native.descriptor,**{field:value})
    changed=replace(package,native=replace(package.native,descriptor=descriptor))
    with pytest.raises(ValueError):
        changed.validate()

def test_package_rejects_changed_graph(package):
    with pytest.raises(ValueError):
        replace(package,graph_ir=package.graph_ir+" ").validate()

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires exact gfx1201")
def test_checked_package_executes_and_preserves_inputs(package):
    args=operands()
    before=[x.copy() for x in args]
    expected=reference_nvfp4_requantize(*args,row_offsets=[0,3,7],
        numeric_policy=nvfp4_requantization_policy())
    actual=execute_nvfp4_ingest(package,*args)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])
    np.testing.assert_allclose(actual[2],expected[2],rtol=1e-13,atol=1e-30)
    for a,b in zip(args,before):
        np.testing.assert_array_equal(a,b)

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires exact gfx1201")
@pytest.mark.parametrize("kind",["bad_global","negative_scale","shape","dtype"])
def test_checked_package_rejects_invalid_inputs_before_allocation(package,kind,monkeypatch):
    from tessera import runtime as rt
    args=list(operands())
    if kind=="bad_global":
        args[2][0]=np.inf
    elif kind=="negative_scale":
        args[1][0,0]=-1
    elif kind=="shape":
        args[0]=args[0][:-1]
    else:
        args[0]=args[0].astype(np.int8)
    def forbidden():
        pytest.fail("invalid invocation reached HIP allocation boundary")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    with pytest.raises(ValueError):
        execute_nvfp4_ingest(package,*args)

def test_package_rejects_changed_tile_policy(package):
    tile=package.native.tile_ir.replace("nearest_signed_e2m1_by_weight_sse","different_selection")
    assert tile != package.native.tile_ir
    with pytest.raises(ValueError):
        replace(package,native=replace(package.native,tile_ir=tile)).validate()

def test_package_image_and_descriptor_roundtrip(package):
    from tessera.compiler.native_artifact import NativeImageArtifact,LaunchDescriptor
    restored=replace(package,native=replace(package.native,
        image=NativeImageArtifact.from_dict(package.native.image.to_dict()),
        descriptor=LaunchDescriptor.from_dict(package.native.descriptor.to_dict())))
    restored.validate()

def test_catalog_inferred_graph_executes_through_native_passes(package):
    from tessera.compiler.rocm_nvfp4_ingest_native import build_nvfp4_ingest_graph
    module=build_nvfp4_ingest_graph(7,64,[0,3,7],
        numeric_policy=nvfp4_requantization_policy())
    assert [str(t) for t in module.functions[0].result_types] == [
        "tensor<7x32xui8>","tensor<2x7xui8>","tensor<7x2x2xf64>"]
    before=module.to_mlir(target="rocm_gfx1201",canonical=True)
    packaged=package_nvfp4_ingest_graph(module)
    packaged.validate()
    assert module.to_mlir(target="rocm_gfx1201",canonical=True) == before

def test_unsigned_container_spelling_does_not_promote_dtype():
    from tessera.dtype import canonicalize_dtype,is_planned_gated_dtype
    from tessera.compiler.graph_ir import tensor_ir_type
    assert is_planned_gated_dtype("uint8")
    with pytest.raises(ValueError):
        canonicalize_dtype("uint8")
    assert str(tensor_ir_type((7,32),"uint8")) == "tensor<7x32xui8>"

def test_named_ingest_capability_does_not_inherit_sibling_ready_defaults():
    from tessera.compiler.capabilities import TARGET_CAPABILITIES,supports_op
    for target in TARGET_CAPABILITIES:
        cap=supports_op(target,"tessera.nvfp4_requantize")
        if target=="rocm_gfx1201":
            assert cap.supported and cap.runtime_status=="ready"
        else:
            assert not cap.supported and cap.runtime_status=="unsupported"

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires exact gfx1201")
def test_common_runtime_launch_uses_native_converter(package,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_ingest_native import ingest_runtime_artifact
    from tessera.compiler import rocm_nvfp4_ingest as ingest
    args=operands()
    expected=reference_nvfp4_requantize(*args,row_offsets=[0,3,7],
        numeric_policy=nvfp4_requantization_policy())
    def forbidden(*a,**k):
        pytest.fail("native runtime tried the host semantic oracle")
    monkeypatch.setattr(ingest,"ingest_nvfp4_projections",forbidden)
    monkeypatch.setattr(ingest,"reference_nvfp4_requantize",forbidden)
    names=[b.name for b in package.native.descriptor.buffers]
    arrays=list(args)+[np.empty_like(x) for x in expected]
    receipt=rt.launch(ingest_runtime_artifact(package),
        {"buffers":dict(zip(names,arrays)),"scalars":{}})
    assert receipt.get("ok"),receipt
    assert receipt["execution_kind"]=="native_gpu"
    assert receipt["image_digest"]==package.native.image.image_digest
    np.testing.assert_array_equal(arrays[3],expected[0])
    np.testing.assert_array_equal(arrays[4],expected[1])
    np.testing.assert_allclose(arrays[5],expected[2],rtol=1e-13,atol=1e-30)

def test_common_runtime_refuses_changed_graph_before_device_access(package,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_ingest_native import ingest_runtime_artifact
    artifact=replace(ingest_runtime_artifact(package),graph_ir=package.graph_ir+" ")
    def forbidden():
        pytest.fail("invalid lineage reached HIP")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    receipt=rt.launch(artifact,{"buffers":{},"scalars":{}})
    assert not receipt["ok"]
    assert receipt["runtime_status"]=="invalid_artifact"
    assert receipt["diagnostic_code"]=="E_LAUNCH_BINDING_MISMATCH"

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires exact gfx1201")
def test_common_runtime_rejects_output_alias_before_device_access(package,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_ingest_native import ingest_runtime_artifact
    args=operands()
    arrays=list(args)+[args[0],np.empty((2,7),np.uint8),np.empty((7,2,2),np.float64)]
    names=[b.name for b in package.native.descriptor.buffers]
    def forbidden():
        pytest.fail("aliased output reached HIP")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    receipt=rt.launch(ingest_runtime_artifact(package),
        {"buffers":dict(zip(names,arrays)),"scalars":{}})
    assert not receipt["ok"],receipt
    assert "alias" in receipt["reason"]

def test_canonical_driver_preserves_graph_schedule_tile_lineage(package):
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.rocm_nvfp4_ingest_native import build_nvfp4_ingest_graph
    module=build_nvfp4_ingest_graph(7,64,[0,3,7],numeric_policy=nvfp4_requantization_policy())
    before=module.to_mlir(target="rocm_gfx1201",canonical=True)
    bundle=compile_graph_module(module,source_origin="test",target="rocm_gfx1201",
        options={"package_native":True},enable_tool_validation=False)
    assert bundle.executable and bundle.execution_kind=="native_gpu"
    assert bundle.launch_descriptor.abi_id==package.native.descriptor.abi_id
    assert bundle.artifact("schedule").input_digest==bundle.artifact("graph").output_digest
    assert bundle.artifact("tile").input_digest==bundle.artifact("schedule").output_digest
    assert bundle.artifact("target").input_digest==bundle.artifact("tile").output_digest
    assert module.to_mlir(target="rocm_gfx1201",canonical=True)==before


def test_named_manifest_proof_is_exact_architecture():
    from tessera.compiler.backend_manifest import manifest_for
    from tessera.compiler import pipeline_gates
    rows=manifest_for("nvfp4_requantize")
    proved=[row for row in rows if row.status=="device_verified_jit"]
    assert len(proved)==1 and proved[0].target=="rocm_gfx1201"
    assert proved[0].dtypes==("nvfp4","fp8_e4m3","fp64")
    assert proved[0].execute_compare_fixture=="tests/unit/test_rocm_nvfp4_ingest_package.py"
    for target in ("rocm_gfx1151","rocm_gfx942","nvidia_sm120","apple_gpu","x86"):
        assert not any(row.status=="device_verified_jit"
            for row in pipeline_gates._manifest_entries("nvfp4_requantize",target))


def test_canonical_compile_admits_verified_descriptor(package):
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.compiler.rocm_nvfp4_ingest_native import build_nvfp4_ingest_graph
    module=build_nvfp4_ingest_graph(7,64,[0,3,7],numeric_policy=nvfp4_requantization_policy())
    compiled=canonical_compile(module,target="rocm_gfx1201",enable_tool_validation=False)
    assert compiled.executable,compiled.reason
    assert compiled.launch_descriptor.abi_id==package.native.descriptor.abi_id
    assert compiled.to_runtime_artifact().native_image==compiled.native_image
