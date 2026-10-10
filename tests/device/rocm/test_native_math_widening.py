"""Exact-device f16/bf16 casts fused into native f32 math input loads."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_math_native as native
from tests.device.rocm.test_native_math_package_jit import target as _target_fixture
from tests.unit.test_rocm_math_widening import module
from benchmarks.rocm.benchmark_native_math_schedule import expected

@pytest.fixture
def target():
    return _target_fixture.__wrapped__()

def sqrt(a):return ts.ops.sqrt(ts.ops.cast(a,"fp32"))
def exp(a):return ts.ops.exp(ts.ops.cast(a,"fp32"))
def add(a,b):return ts.ops.add(ts.ops.cast(a,"fp32"),ts.ops.cast(b,"fp32"))
def div(a,b):return ts.ops.div(ts.ops.cast(a,"fp32"),ts.ops.cast(b,"fp32"))
def add_reverse(a,b):return ts.ops.add(ts.ops.cast(b,"fp32"),ts.ops.cast(a,"fp32"))
def div_reverse(a,b):return ts.ops.div(ts.ops.cast(b,"fp32"),ts.ops.cast(a,"fp32"))
def cumsum(a):return ts.ops.cumsum(ts.ops.cast(a,"fp32"),axis=-1)
def cummax(a):return ts.ops.cummax(ts.ops.cast(a,"fp32"),axis=-1)

FUNCTIONS={"sqrt":sqrt,"exp":exp,"add":add,"div":div,"add_reverse":add_reverse,
           "div_reverse":div_reverse,"cumsum":cumsum,"cummax":cummax}

@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("name",list(FUNCTIONS))
@pytest.mark.parametrize("shape",[(3,17),(2,3,257)])
def test_widening_jit_and_portable_changed_input(target,storage,name,shape,monkeypatch):
    dtype=np.float16 if storage=="f16" else pytest.importorskip("ml_dtypes").bfloat16
    kind=name.removesuffix("_reverse")
    rng=np.random.default_rng(606)
    arrays=[rng.uniform(.125,2,shape).astype(dtype)]
    if kind in {"add","div"}:arrays.append(rng.uniform(.5,2,shape).astype(dtype))
    def oracle():
        values=[a.astype(np.float32) for a in arrays]
        return expected(kind,list(reversed(values)) if name.endswith("_reverse") else values)
    fn=ts.jit(target=target)(FUNCTIONS[name])
    result=fn(*arrays)
    assert result.dtype==np.float32 and fn.execution_kind=="native_gpu"
    np.testing.assert_allclose(result,oracle(),rtol=2e-5,atol=2e-5)
    artifact=fn.runtime_artifact()
    info=artifact.launch_descriptor.provenance["native_math"]
    assert info["storage"]==storage and info["output_storage"]=="f32"
    assert storage+"_f32" in artifact.launch_descriptor.abi_id
    monkeypatch.setattr(native,"run_tessera_opt",lambda *a,**k:pytest.fail("warm call compiled"))
    for attr in ("_execute_rocm_compiled_unary","_execute_rocm_compiled_binary","_execute_rocm_compiled_scan"):
        monkeypatch.setattr(rt,attr,lambda *a,**k:pytest.fail("metadata fallback"))
    for a in arrays:a[:]=a.astype(np.float32)*.75
    np.testing.assert_allclose(fn(*arrays),oracle(),rtol=2e-5,atol=2e-5)
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    buffers=dict(zip(info["bindings"][:-1],arrays))
    output=next(b.name for b in artifact.launch_descriptor.buffers if b.direction=="output")
    buffers[output]=np.empty(shape,np.float32)
    scalars={"Rows":info["rows"],"Columns":info["columns"]} if kind.startswith("cum") else {"N":info["elements"]}
    receipt=rt.launch(restored,{"buffers":buffers,"scalars":scalars})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_allclose(buffers[output],oracle(),rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
def test_widening_image_identity_reuses_shapes_and_retains_storage(target,storage,kind):
    images=[]
    for shape in ((3,17),(7,259)):
        recipe=native.lower_math_graph(module(kind,storage,shape,reverse=kind in {"add","div"}),target)
        image=native.package_math_recipe(recipe,pipeline_name="tessera-lower-to-rocm").image
        images.append(image.payload)
    assert images[0]==images[1]


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("kind",["sqrt","exp","add","div","cumsum","cummax"])
@pytest.mark.parametrize("columns",[17,257])
def test_widening_ieee_classifications_and_signed_zero(target,storage,kind,columns):
    dtype=np.float16 if storage=="f16" else pytest.importorskip("ml_dtypes").bfloat16
    pattern=np.array([-np.inf,-1.,-0.,0.,.5,1.,np.inf,np.nan],np.float32)
    a=np.resize(pattern,(3,columns)).astype(dtype)
    args=[a]
    if kind in {"add","div"}:args.append(np.ones((3,columns),dtype))
    with np.errstate(invalid="ignore",divide="ignore",over="ignore"):
        oracle=expected(kind,[x.astype(np.float32) for x in args])
        fn=ts.jit(target=target)(FUNCTIONS[kind])
        actual=fn(*args)
    assert fn.execution_kind=="native_gpu" and actual.dtype==np.float32
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(oracle))
    np.testing.assert_array_equal(np.isposinf(actual),np.isposinf(oracle))
    np.testing.assert_array_equal(np.isneginf(actual),np.isneginf(oracle))
    finite=np.isfinite(oracle)
    np.testing.assert_allclose(actual[finite],oracle[finite],rtol=2e-5,atol=2e-5)
    if kind in {"sqrt","div"}:
        zeros=oracle==0
        np.testing.assert_array_equal(np.signbit(actual[zeros]),np.signbit(oracle[zeros]))
