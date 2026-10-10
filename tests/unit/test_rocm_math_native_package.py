"""Checked math package projection; device execution has its own owning lane."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from tessera.compiler.graph_ir import GraphIRModule,GraphIRFunction,IRArg,IROp,IRType
from tessera.compiler import rocm_math_native as math_native
from tessera.compiler import rocm_native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera import runtime as rt

def module(kind="sqrt",shape=(3,17),reverse=False):
    typ=IRType("tensor<"+"x".join(map(str,shape))+"xf32>",tuple(map(str,shape)),"fp32")
    names=["a","b"] if kind in {"add","div"} else ["a"]
    order=list(reversed(names)) if reverse else names
    return GraphIRModule([GraphIRFunction(name="math",
        args=[IRArg(n,typ) for n in names],result_types=[typ],
        body=[IROp(result="out",op_name="tessera."+kind,operands=["%"+n for n in order],
                   operand_types=[str(typ)]*len(names),result_type=str(typ),
                   kwargs={"axis":-1} if kind in {"cumsum","cummax"} else {})],
        return_values=["%out"])])

@pytest.fixture
def tool():
    if find_tessera_opt() is None:
        pytest.skip("requires production compiler")

@pytest.mark.parametrize("target",["rocm_gfx1151","rocm_gfx1201"])
@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
def test_package_request_preserves_caller_graph(tool,target,kind):
    source=module(kind,reverse=True)
    original=deepcopy(source)
    recipe=math_native.lower_math_graph(source,target)
    assert source==original
    info=math_native._info(recipe.tile_ir)
    assert info["shape"]==(3,17)
    assert info["bindings"]==tuple(["a","b","out"] if kind in {"add","div"} else ["a","out"])
    if kind in {"add","div"}:assert info["roles"]==(1,0)
    recipe.validate()

@pytest.mark.parametrize("field",["graph_ir","schedule_ir","tile_ir"])
def test_recipe_mutation_rejected_before_packaging(tool,field):
    recipe=math_native.lower_math_graph(module(),"rocm_gfx1151")
    text=getattr(recipe,field).replace("sqrt","exp")
    with pytest.raises((ValueError,RuntimeError)):
        replace(recipe,**{field:text}).validate()

@pytest.fixture
def packaged():
    if not rocm_native.native_packaging_available():
        pytest.skip("requires built ROCm compiler and device libraries")
    recipe=math_native.lower_math_graph(module("div",reverse=True),"rocm_gfx1151")
    native=math_native.package_math_recipe(recipe,pipeline_name="tessera-lower-to-rocm")
    artifact=rt.RuntimeArtifact(
        graph_ir=recipe.graph_ir,schedule_ir=recipe.schedule_ir,tile_ir=native.tile_ir,
        target_ir=native.target_ir,native_image=native.image,launch_descriptor=native.descriptor,
        metadata={"target":"rocm_gfx1151","execution_kind":"native_gpu"})
    return recipe,native,artifact

def test_portable_lineage_validation_does_not_need_compiler(packaged,monkeypatch):
    _,native,artifact=packaged
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    def forbidden(*a,**k):pytest.fail("portable launch called compiler")
    monkeypatch.setattr(math_native,"find_tessera_opt",forbidden)
    monkeypatch.setattr(math_native,"run_tessera_opt",forbidden)
    math_native.validate_math_runtime_artifact(restored)
    assert [b.name for b in native.descriptor.buffers]==["b","a","out"]

@pytest.mark.parametrize("field",["graph_ir","schedule_ir","tile_ir","target_ir"])
def test_changed_portable_stage_rejected_before_hip(packaged,monkeypatch,field):
    _,_,artifact=packaged
    if field == "target_ir":
        # The image contract rejects changed Target text at construction.
        with pytest.raises(ValueError, match="E_LAUNCH_STALE_IMAGE"):
            replace(artifact, target_ir=artifact.target_ir+"\n// changed\n")
        return
    changed=replace(artifact,**{field:getattr(artifact,field)+"\n// changed\n"})
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:pytest.fail("bad stage reached HIP"))
    receipt=rt.launch(changed,{})
    assert receipt["ok"] is False and receipt["diagnostic_code"]=="E_LAUNCH_BINDING_MISMATCH"

@pytest.mark.parametrize("bad",["scalar","alias","shape","noncompact","readonly","roles"])
def test_runtime_projection_checks_before_hip(packaged,bad):
    _,native,_=packaged
    shape=(3,17)
    arrays={"a":np.ones(shape,np.float32),"b":np.full(shape,2,np.float32),
            "out":np.empty(shape,np.float32)}
    scalars={"N":51}
    desc=native.descriptor
    if bad=="scalar":scalars["N"]=52
    if bad=="alias":arrays["out"]=arrays["a"]
    if bad=="shape":arrays["a"]=np.ones((3,18),np.float32)
    if bad=="noncompact":arrays["a"]=np.ones((3,34),np.float32)[:,::2]
    if bad=="readonly":arrays["out"].flags.writeable=False
    if bad=="roles":
        info=deepcopy(desc.provenance["native_math"]);info["roles"]=[True,0]
        desc=replace(desc,provenance={**desc.provenance,"native_math":info})
    with pytest.raises(ValueError):
        math_native.runtime_projection(native.image,desc,arrays,scalars)


@pytest.mark.parametrize("target",["rocm_gfx1151","rocm_gfx1201"])
@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
def test_native_math_registries_name_exact_device_proof(target,kind):
    from tessera.compiler.backend_manifest import manifest_for
    from tessera.compiler.capabilities import TARGET_CAPABILITIES
    from tessera.compiler.execution_matrix import lookup
    capability=TARGET_CAPABILITIES[target].supported_ops["tessera."+kind]
    expected_ranks = (2, 3, 4) if (target, kind) == ("rocm_gfx1201", "add") else (2, 3)
    assert capability.dtypes==("fp32",) and capability.ranks==expected_ranks
    entries=[entry for entry in manifest_for(kind) if entry.target==target]
    assert len(entries)==1
    assert entries[0].status=="device_verified_jit"
    assert entries[0].dtypes==("fp32",)
    assert entries[0].execute_compare_fixture=="tests/device/rocm/test_native_math_package_jit.py"
    row=lookup(target,"rocm_math_native_descriptor")
    assert row.evidence_target==target and row.op_family=="math"
