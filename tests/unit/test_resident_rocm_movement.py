"""Resident products reject fork, stale packages and unsupported launch forms."""
import os
from types import SimpleNamespace
import pytest
from tessera.compiler.resident_rocm_movement import ResidentMovementCall,views
import numpy as np

def test_resident_fork_guard_precedes_python_lock():
    owner=ResidentMovementCall.__new__(ResidentMovementCall)
    owner.pid=os.getpid()+1;owner.closed=False
    class Forbidden:
        def __enter__(self):pytest.fail("entered inherited lock")
        def __exit__(self,*args):pass
    owner.lock=Forbidden()
    for operation in (lambda:owner.upload(()),lambda:owner.execute(),lambda:owner.read(1),owner.close):
        with pytest.raises(ValueError,match="fork"):operation()

def test_resident_stale_prepared_artifact_is_refused_before_gpu():
    prepared=SimpleNamespace(_finalizer=SimpleNamespace(alive=True),
        _lib=object(),_sealed_artifact_hash="original",artifact=SimpleNamespace(artifact_hash="changed"))
    with pytest.raises(ValueError,match="sealed preparation"):ResidentMovementCall(prepared)

@pytest.mark.parametrize("array",[[],np.ones((1,1,1,1,1),np.float32)])
def test_resident_host_view_rejects_opaque_or_excess_rank(array):
    with pytest.raises(TypeError,match="host tensor"):views((array,))

@pytest.mark.parametrize("array",[
    np.ones((2,3),dtype=">f4"),np.ones((2,3),np.float16)])
def test_resident_view_never_relabels_non_native_storage(array):
    assert views((array,))[0].dtype==0

def test_resident_common_route_rejects_different_artifact_before_execution():
    from tessera import runtime as rt
    owner=ResidentMovementCall.__new__(ResidentMovementCall)
    owner.artifact_hash="original"
    owner.execute=lambda **kwargs:pytest.fail("executed a mismatched package")
    artifact=SimpleNamespace(native_image=object(),
        launch_descriptor=SimpleNamespace(validate_image=lambda image:None),
        metadata={},artifact_hash="changed")
    receipt=rt._launch_native_descriptor(artifact,{"resident_movement":owner},None,0)
    assert not receipt["ok"]
    assert receipt["reason"].startswith("E_LAUNCH_BINDING_MISMATCH")

@pytest.mark.parametrize("change",["shape","arch","policy","buffer_layout","rows"])
def test_softmax_edge_rejects_mismatched_contract_before_gpu(change):
    from dataclasses import replace
    from tessera.compiler.native_artifact import (BufferBinding,ShapeGuard,
        ScalarArgument,LaunchGeometry,OrderingSemantics,WorkspaceRequirement)
    from tessera.compiler.rocm_native import GFX_PAGED_KV_F32_ABI,GFX_SOFTMAX_F32_ABI
    from tessera.compiler.resident_rocm_movement import validate_softmax_consumer
    shape=(5,2,13)
    image=SimpleNamespace(target="rocm_gfx1151",architecture="gfx1151",binary_format="hsaco")
    prepared=SimpleNamespace(output_shape=shape,artifact=SimpleNamespace(
        native_image=image,launch_descriptor=SimpleNamespace(abi_id=GFX_PAGED_KV_F32_ABI)))
    descriptor=SimpleNamespace(validate_image=lambda image:None,abi_id=GFX_SOFTMAX_F32_ABI,
        buffers=(BufferBinding(0,"x","input","fp32",3,"row_major",4),
                 BufferBinding(1,"y","output","fp32",3,"row_major",4)),
        shape_guards=tuple(ShapeGuard(n,a,"eq",d) for n in ("x","y") for a,d in enumerate(shape)),
        scalars=(ScalarArgument(2,"Rows","int64"),ScalarArgument(3,"K","int64")),
        geometry=LaunchGeometry(policy="gfx1151_softmax_workgroup_per_row_256"),
        ordering=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",)),
        workspace=WorkspaceRequirement(),dynamic_local_memory_bytes=0,dynamic_local_memory_expression=None,
        provenance=dict(family="softmax",kind="softmax",axis=-1,storage="f32",accum="f32",
            keepdims=False,shape=shape,output_shape=shape,rows=10,columns=13,exp_mode="accurate",ftz=False))
    artifact=SimpleNamespace(native_image=image,launch_descriptor=descriptor)
    validate_softmax_consumer(prepared,artifact)
    if change=="shape":descriptor.provenance["output_shape"]=(5,2,12)
    elif change=="arch":artifact.native_image=SimpleNamespace(target="rocm_gfx1201",architecture="gfx1201",binary_format="hsaco")
    elif change=="policy":descriptor.provenance["ftz"]=True
    elif change=="rows":descriptor.provenance["rows"]=9
    else:descriptor.buffers=(replace(descriptor.buffers[0],layout="strided"),descriptor.buffers[1])
    with pytest.raises(ValueError,match="owning architecture|extent/ABI"):
        validate_softmax_consumer(prepared,artifact)
