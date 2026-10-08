"""Ordinary JIT and portable math package evidence on each owning ROCm GPU."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_math_native as native
from tessera.compiler import rocm_native
from tests.unit.test_rocm_math_native_package import module
from benchmarks.rocm.benchmark_native_math_schedule import expected

def sqrt(a):return ts.ops.sqrt(a)
def exp(a):return ts.ops.exp(a)
def add(a,b):return ts.ops.add(a,b)
def div(a,b):return ts.ops.div(a,b)
def add_reverse(a,b):return ts.ops.add(b,a)
def div_reverse(a,b):return ts.ops.div(b,a)
def cumsum(a):return ts.ops.cumsum(a,axis=-1)
def cummax(a):return ts.ops.cummax(a,axis=-1)

FUNCTIONS={"sqrt":sqrt,"exp":exp,"add":add,"div":div,
           "add_reverse":add_reverse,"div_reverse":div_reverse,
           "cumsum":cumsum,"cummax":cummax}

@pytest.fixture
def target():
    arch=rt._rocm_live_arch()
    if arch not in {"gfx1151","gfx1201"} or not rocm_native.native_packaging_available():
        pytest.skip("requires exact owning ROCm GPU and native compiler")
    return "rocm_"+arch

@pytest.mark.parametrize("name",list(FUNCTIONS))
@pytest.mark.parametrize("shape",[(3,17),(2,3,257)])
def test_ordinary_math_jit_uses_native_package_and_portable_replay(target,name,shape,monkeypatch):
    kind=name.removesuffix("_reverse")
    binary=kind in {"add","div"}
    rng=np.random.default_rng(606)
    args=[rng.uniform(.125,2,shape).astype(np.float32)]
    if binary:args.append(rng.uniform(.5,2,shape).astype(np.float32))
    order=lambda values:list(reversed(values)) if name.endswith("_reverse") else values
    function=ts.jit(target=target)(FUNCTIONS[name])
    actual=function(*args)
    np.testing.assert_allclose(actual,expected(kind,order(args)),rtol=2e-5,atol=2e-5)
    assert function.execution_kind=="native_gpu"
    artifact=function.runtime_artifact()
    assert artifact.native_image is not None and artifact.launch_descriptor is not None
    assert "family=rocm_math" in artifact.schedule_ir
    assert "tile.elementwise_kernel" in artifact.tile_ir or "tile.scan_kernel" in artifact.tile_ir
    native.validate_math_runtime_artifact(artifact)
    assert artifact.metadata["compiler_path"]=="rocm_math_native_descriptor"
    from tessera.compiler.execution_matrix import lookup
    row=lookup(target,"rocm_math_native_descriptor")
    assert row.op_family=="math" and row.evidence_target==target
    image_digest=artifact.native_image.image_digest
    def forbidden(*a,**k):pytest.fail("warm native math used compiler or legacy metadata executor")
    monkeypatch.setattr(native,"run_tessera_opt",forbidden)
    for attr in ("_execute_rocm_compiled_unary","_execute_rocm_compiled_binary","_execute_rocm_compiled_scan"):
        monkeypatch.setattr(rt,attr,forbidden)
    for arg in args:arg*=np.float32(.75)
    again=function(**dict(zip(("a","b"),args)))
    np.testing.assert_allclose(again,expected(kind,order(args)),rtol=2e-5,atol=2e-5)
    assert function.runtime_artifact().native_image.image_digest==image_digest
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    info=restored.launch_descriptor.provenance["native_math"]
    # Bind original Graph arguments, then let the descriptor's SSA roles order them.
    input_names=info["bindings"][:-1]
    buffers=dict(zip(input_names,args))
    output_name=next(b.name for b in restored.launch_descriptor.buffers if b.direction=="output")
    buffers[output_name]=np.empty(shape,np.float32)
    scalar={"Rows":info["rows"],"Columns":info["columns"]} if kind in {"cumsum","cummax"} else {"N":info["elements"]}
    receipt=rt.launch(restored,{"buffers":buffers,"scalars":scalar})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_allclose(buffers[output_name],expected(kind,order(args)),rtol=2e-5,atol=2e-5)
    _,execution_kind=rt._execute_rocm_native_descriptor(restored,{"buffers":buffers,"scalars":scalar})
    assert execution_kind=="native_gpu"
    np.testing.assert_allclose(buffers[output_name],expected(kind,order(args)),rtol=2e-5,atol=2e-5)

@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
def test_math_images_reuse_across_shapes_and_binary_roles(target,kind):
    images=[]
    targets=[]
    for shape,reverse in (((3,17),False),((7,259),kind in {"add","div"}),((2,3,1024),False)):
        recipe=native.lower_math_graph(module(kind,shape,reverse),target)
        package=native.package_math_recipe(recipe,pipeline_name="tessera-lower-to-rocm")
        images.append(package.image.payload)
        targets.append(package.target_ir)
    assert images[0]==images[1]==images[2]
    assert targets[0]==targets[1]==targets[2]
