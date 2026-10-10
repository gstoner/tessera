"""Public frontend and common runtime preserve permuted matmul roles."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 GPU required")


@ts.jit(target="nvidia_sm120")
def plain(b,a):
    return ts.ops.matmul(a,b,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def fused(residual,b,a,bias):
    return ts.ops.matmul(a,b,bias=bias,activation="relu",
                         residual=residual,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def half_fused(residual,b,a,bias):
    return ts.ops.matmul(a,b,bias=bias,activation="relu",
                         residual=residual,output_dtype="fp16")


@pytest.mark.parametrize("rhs_layout",["row_major","col_major"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("mode",["plain","fused","half_fused"])
def test_public_permuted_matmul_native_and_portable(dtype,mode,rhs_layout,monkeypatch):
    import ml_dtypes
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(120708)
    a=(rng.normal(size=(17,35))*.2).astype(storage)
    b=np.array((rng.normal(size=(35,19))*.2).astype(storage),
               order="F" if rhs_layout=="col_major" else "C")
    bias=(rng.normal(size=19)*.1).astype(np.float32)
    residual=(rng.normal(size=(17,19))*.05).astype(np.float32)
    epilogue=mode!="plain"
    fn={"plain":plain,"fused":fused,"half_fused":half_fused}[mode]
    args=(residual,b,a,bias) if epilogue else (b,a)
    want=a.astype(np.float64)@b.astype(np.float64)
    if epilogue:want=np.maximum(want+bias.astype(np.float64),0)+residual.astype(np.float64)
    if mode=="half_fused":want=want.astype(np.float16)
    actual=fn(*args)
    assert fn.execution_kind=="native_gpu"
    np.testing.assert_allclose(actual,want,rtol=1e-3 if mode=="half_fused" else 4e-5,atol=4e-5)
    artifact=rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
    assert artifact.launch_descriptor.provenance["b_layout"]==rhs_layout
    assert artifact.launch_descriptor.provenance["route"]=="canonical_scheduled_tile_consumer"
    assert artifact.launch_descriptor.provenance["schedule_digest"]
    assert artifact.launch_descriptor.provenance["tile_ir_digest"]
    assert artifact.metadata["compiler_path"]=="nvidia_sm120_native_descriptor"
    bindings=dict(zip(artifact.metadata["frontend_input_bindings"],args,strict=True))
    output=next(x for x in artifact.launch_descriptor.buffers if x.direction=="output")
    bindings[output.name]=np.empty_like(actual)
    receipt=rt.launch(artifact,{"buffers":bindings,"scalars":{"M":17,"N":19,"K":35}})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_array_equal(actual,receipt["output"])
    def unavailable(*args,**kwargs):
        raise AssertionError("eager fallback on cached frontend call")
    monkeypatch.setattr(fn,"_fn",unavailable)
    import tessera.compiler.canonical_compile as compile_module
    monkeypatch.setattr(compile_module,"canonical_compile",unavailable)
    if fn._native_prepared_matmul_calls:
        monkeypatch.setattr(fn,"_trace_frontend_capture",unavailable)
        monkeypatch.setattr(rt,"launch",unavailable)
    np.testing.assert_array_equal(actual,fn(*args))
    if fn._native_prepared_matmul_calls:
        assert fn._native_descriptor_last_receipt["native_call_binding"]=="prepared_cpp_matmul"
    # Reused packages bind the new buffers, rather than retaining prior values.
    changed_a = np.zeros_like(a)
    changed_args = (residual,b,changed_a,bias) if epilogue else (b,changed_a)
    changed_want = np.zeros((17,19),np.float64)
    if epilogue:
        changed_want = np.maximum(changed_want+bias.astype(np.float64),0)+residual
    if mode=="half_fused":
        changed_want=changed_want.astype(np.float16)
    np.testing.assert_allclose(fn(*changed_args),changed_want,rtol=1e-3,atol=4e-5)
