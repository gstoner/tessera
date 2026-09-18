"""Exact-architecture Schedule ancestry and opt-in gfx1201 execution."""

from dataclasses import replace
import os
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.compiler import scheduled_kernel, rocm_native
from tessera.compiler.native_unary_contract import verify_unary_ancestry
from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_scheduled_kernel_consumers import _module





@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_gfx1201_unary_projection_rejects_stale_metadata(family):
    artifact = scheduled_kernel.lower_scheduled_kernel(_module(family=family, target="rocm"), target="rocm_gfx1201")
    verify_unary_ancestry(artifact, target="rocm", architecture="gfx1201")
    swapped = re.sub(r"(tile\.(?:softmax|reduce)_kernel )(%[\w]+), (%[\w]+)", r"\1\3, \2", artifact.tile_ir, count=1)
    assert swapped != artifact.tile_ir
    for changed in [
        replace(artifact, architecture="gfx1151"),
        replace(artifact, input_shape=(6, 5)),
        replace(artifact, workgroup_size=1),
        replace(artifact, tile_ir=swapped),
    ]:
        with pytest.raises(ValueError):
            verify_unary_ancestry(changed, target="rocm", architecture="gfx1201")


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_gfx1201_scheduled_package_executes(family):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201", "requires the exact owning device"

    artifact = scheduled_kernel.lower_scheduled_kernel(_module(family=family, target="rocm"), target="rocm_gfx1201")
    package = rocm_native.package_scheduled_kernel(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert package.image.target == "rocm_gfx1201"
    x = np.random.default_rng(12).normal(size=(2, 3, 5)).astype(np.float32)
    output = np.zeros(artifact.output_shape, np.float32)
    runtime = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    scalars = (
        {"Rows": artifact.rows, "K": artifact.columns}
        if family == "softmax"
        else {"Outer": artifact.outer, "AxisExtent": artifact.axis_extent, "Inner": artifact.inner}
    )
    result = rt.launch(runtime, {"buffers": {"x": x, "o": output}, "scalars": scalars})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    if family == "reduce":
        expected = x.mean(axis=1)
    else:
        ex = np.exp(x - x.max(axis=-1, keepdims=True))
        expected = ex / ex.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "architecture,abi",
    [("gfx1151", rocm_native.GFX_REDUCE_F32_ABI), ("gfx1201", rocm_native.GFX_SOFTMAX_F16_ABI)],
)
def test_gfx1201_cached_launcher_keeps_architecture_and_family_gate(architecture, abi):
    """A gfx1151 image is never launched as gfx1201, and an ABI without a
    gfx1201 device row (f16 softmax: only the f32 rows are proved here) is
    refused by name rather than run on gfx1151's evidence."""
    from tessera import runtime as rt

    image = SimpleNamespace(target="rocm_gfx1201", architecture=architecture)
    descriptor = SimpleNamespace(abi_id=abi)
    with pytest.raises(ValueError, match="proved unary, matmul, attention, depth-attention or paged-KV ABI"):
        rt._submit_rocm_gfx1151_native(image, descriptor, {}, {}, None)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("activation,bias", [("relu", False), ("gelu", False), ("silu", True), ("none", True), ("gelu", True)])
@pytest.mark.parametrize("shape", [(16,16,16), (17,19,23), (65,48,37)])
def test_gfx1201_scheduled_matmul_package_executes_fused_epilogue(shape, activation, bias):
    """GFX1201-PARITY slice 1: the fused bias/activation epilogue on RDNA4.

    The generator's typed body hands the epilogue to the store and TileToROCM
    applies it per element after resolving this chip's half-wave accumulator
    rows -- the same implementation that serves gfx1151's replicated rows. The
    reference is the fused numpy program, not the bare matmul (the failure
    mode the 2026-09-17 loop found on this chip was exactly a bare matmul
    returned as the fused result)."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module, _epilogue_reference
    assert rt._rocm_live_arch() == "gfx1201"
    m, k, n = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        matmul_module(target="rocm", shape=shape, activation=activation, bias=bias), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.provenance["activation"] == activation
    assert package.descriptor.provenance["bias"] is bias
    rng = np.random.default_rng(713 + len(activation))
    a = (rng.normal(size=(m,k)) * 0.4).astype(np.float16)
    b = (rng.normal(size=(k,n)) * 0.4).astype(np.float16)
    bias_arr = (rng.normal(size=(n,)) * 0.5).astype(np.float32) if bias else None
    output = np.zeros((m,n), np.float32)
    buffers = {"a": a, "b": b, "o": output}
    if bias:
        buffers["bias"] = bias_arr
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_allclose(output, _epilogue_reference(a, b, bias_arr, activation), rtol=0, atol=5e-2)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("storage", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("shape", [(16,16,16), (17,19,23), (65,48,37)])
def test_gfx1201_scheduled_matmul_package_executes_fp8(shape, storage):
    """GFX1201-PARITY slice 5: OCP FP8 storage through the typed route.

    The RDNA4 WMMA datatype audit proved V_WMMA_F32_16X16X16_{FP8,BF8}_{FP8,BF8}
    on this box for one 16x16x16 tile; this is the kernel-shaped consumer --
    a Graph `tessera.matmul` over fp8 operands, scheduled, packaged and launched
    like the f16 one. Products of fp8 values are exact in f32, so the reference
    is the f32 matmul of the same codes; the tolerance covers accumulation order.
    """
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module
    assert rt._rocm_live_arch() == "gfx1201"
    m, k, n = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        matmul_module(target="rocm", shape=shape, dtype=storage), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    assert artifact.storage == ("e4m3" if storage == "fp8_e4m3" else "e5m2")
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.abi_id == (
        rocm_native.GFX_MATMUL_E4M3_F32_ABI if storage == "fp8_e4m3" else rocm_native.GFX_MATMUL_E5M2_F32_ABI)
    np_dtype = ml_dtypes.float8_e4m3fn if storage == "fp8_e4m3" else ml_dtypes.float8_e5m2
    rng = np.random.default_rng(919 + len(storage))
    a = (rng.normal(size=(m,k)) * 0.5).astype(np_dtype)
    b = (rng.normal(size=(k,n)) * 0.5).astype(np_dtype)
    output = np.zeros((m,n), np.float32)
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("shape", [(16,16,16), (17,19,23), (65,48,37)])
def test_gfx1201_scheduled_matmul_package_executes(shape):
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module
    assert rt._rocm_live_arch() == "gfx1201"
    rt._native_launchers.pop("rocm_gfx1201", None)
    artifact = scheduled_matmul.lower_scheduled_matmul(matmul_module(target="rocm", shape=shape), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    for changed in (replace(artifact, architecture="gfx1151"),
                    replace(artifact, macro_tile_m=32),
                    replace(artifact, tile_ir=artifact.tile_ir+"\n// changed")):
        with pytest.raises(ValueError):
            rocm_native.package_scheduled_matmul(changed, pipeline_name="tessera-lower-to-rocm")
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.provenance["macro_tile"] == [16,16]
    m,k,n = shape
    rng = np.random.default_rng(712)
    a = rng.normal(size=(m,k)).astype(np.float16)
    b = rng.normal(size=(k,n)).astype(np.float16)
    output = np.zeros((m,n), np.float32)
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": {"a":a,"b":b,"o":output}, "scalars":{"M":m,"N":n,"K":k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_allclose(output, a.astype(np.float32)@b.astype(np.float32), rtol=2e-4, atol=2e-4)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_gfx1201_scheduled_attention_package_executes(bias, causal, dtype):
    from tessera import runtime as rt
    from tessera.compiler.scheduled_attention import lower_scheduled_attention
    from tests.unit.test_scheduled_attention_consumers import _module as attention_module
    assert rt._rocm_live_arch() == "gfx1201"
    rt._native_launchers.pop("rocm_gfx1201", None)
    module = attention_module(target="rocm", bias=bias)
    if dtype == "bf16":
        from tessera.compiler.graph_ir import IRType
        for arg in module.functions[0].args[:3]:
            old = arg.ir_type
            arg.ir_type = IRType(str(old).replace("xf16>","xbf16>"), old.shape, "bf16")
        op = module.functions[0].body[0]
        op.operand_types = [str(arg.ir_type) for arg in module.functions[0].args]
    import ml_dtypes
    storage = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
    module.functions[0].body[0].kwargs.update(causal=causal, window=(-1,-1))
    artifact = lower_scheduled_attention(module, target="rocm_gfx1201")
    package = rocm_native.package_scheduled_attention(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert artifact.backward_lse_policy == "gfx1201_explicit_lse"
    b,hq,hkv,sq,sk,d,dv = artifact.dims
    rng = np.random.default_rng(517)
    q = (rng.normal(size=(b,hq,sq,d))*0.2).astype(storage)
    k = (rng.normal(size=(b,hkv,sk,d))*0.2).astype(storage)
    v = (rng.normal(size=(b,hkv,sk,dv))*0.2).astype(storage)
    output = np.zeros((b,hq,sq,dv),np.float32)
    buffers = {"q":q,"k":k,"v":v,"o":output}
    scores = (q.astype(np.float64) @ np.repeat(k.astype(np.float64),hq//hkv,axis=1).swapaxes(-1,-2))*artifact.scale
    if bias:
        buffers["bias"] = (rng.normal(size=(b,hq,sq,sk))*0.05).astype(np.float32)
        scores += buffers["bias"]
    if causal:
        scores = np.where(np.arange(sk)[None,:] <= np.arange(sq)[:,None] + max(sk-sq,0), scores, -np.inf)
    weights = np.exp(scores-scores.max(axis=-1,keepdims=True))
    weights /= weights.sum(axis=-1,keepdims=True)
    expected = weights @ np.repeat(v.astype(np.float64),hq//hkv,axis=1)
    runtime = rt.RuntimeArtifact(metadata={"target":package.image.target},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    result = rt.launch(runtime,{"buffers":buffers,"scalars":{"Sq":sq,"Sk":sk,"Scale":artifact.scale,"Causal":int(causal),"Hq":hq,"KvRatio":hq//hkv}})
    assert result["ok"] and result["execution_kind"] == "native_gpu",json.dumps(result, default=str)
    np.testing.assert_allclose(output,expected,rtol=0.01,atol=0.001)


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
@pytest.mark.parametrize("family", ["matmul", "attention"])
def test_gfx1201_driver_uses_adjacent_scheduled_artifacts(family):
    from tessera.compiler.driver import compile_graph_module
    from tests._support.compiler_tool import require_tessera_opt
    require_tessera_opt("tessera-schedule-to-tile", "lower-tile-to-rocm")  # ROCm-capable build only
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module
    from tests.unit.test_scheduled_attention_consumers import _module as attention_module
    module = (matmul_module if family == "matmul" else attention_module)(target="rocm")
    bundle = compile_graph_module(module, source_origin="gfx1201-package-proof",
        target="rocm_gfx1201", options={"package_native":True}, enable_tool_validation=False)
    assert bundle.lineage_complete
    assert bundle.native_image.architecture == "gfx1201"
    assert bundle.launch_descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert bundle.schedule and bundle.tile and bundle.target_ir


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("dropout", [0.0, 0.125])
def test_gfx1201_backward_program(dtype, dropout):
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
    from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module as backward_module, _reference
    module = backward_module(1,4,2,17,19,64,dtype=dtype,dropout_p=dropout,lse_checkpoint="recompute")
    artifact = lower_scheduled_attention_backward(module,target="rocm_gfx1201")
    program = rocm_native.package_scheduled_attention_backward(artifact,pipeline_name="tessera-lower-to-rocm")
    assert program.image.architecture == "gfx1201"
    assert artifact.lse_checkpoint_selection == "recompute"
    rng = np.random.default_rng(617)
    storage = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
    q, k, v, do = [(rng.normal(size=s)*0.2).astype(storage) for s in [(1,4,17,64),(1,2,19,64),(1,2,19,64),(1,4,17,64)]]
    bias = (rng.normal(size=(1,4,17,19))*0.05).astype(np.float32)
    buffers = dict(q=q,key=k,v=v,do=do,bias=bias,dq=np.empty(q.shape,np.float32),dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    result = rt._submit_rocm_gfx1151_attention_backward_program(program,buffers)
    expected = _reference(do,q,k,v,bias,dropout_p=dropout)
    for got, want in zip(result["outputs"],expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=0.04,atol=0.003)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("cotangent_storage", ["captured", "float32"])
def test_gfx1201_public_attention_native_backward(dtype, cotangent_storage):
    import ml_dtypes
    import tessera as ts
    from tessera.compiler.native_vjp_plugins import validate_native_vjp_execution_certificate

    @ts.jit(target="rocm", autodiff="reverse", wrt=("q", "k", "v"))
    def attention(q, k, v):
        return ts.ops.gqa_attention(q, k, v, num_query_heads=4, num_kv_heads=2, causal=True)

    rng = np.random.default_rng(1904)
    storage = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
    q, k, v, dout = [(rng.normal(size=shape)*0.2).astype(storage) for shape in
                    [(1,4,17,64),(1,2,19,64),(1,2,19,64),(1,4,17,64)]]
    actual = attention.native_backward(q, k, v, out_cotangents=dout if cotangent_storage == "captured" else dout.astype(np.float32))
    qf, kf, vf, df = [x.astype(np.float64) for x in (q,k,v,dout)]
    kr, vr = [np.repeat(x,2,axis=1) for x in (kf,vf)]
    score = qf @ kr.swapaxes(-1,-2) / 8
    score = np.where(np.arange(19)[None,:] <= np.arange(17)[:,None]+2, score, -np.inf)
    p = np.exp(score-score.max(axis=-1,keepdims=True)); p /= p.sum(axis=-1,keepdims=True)
    dp = df @ vr.swapaxes(-1,-2)
    ds = p*(dp-(dp*p).sum(axis=-1,keepdims=True))/8
    expected = (ds@kr, (ds.swapaxes(-1,-2)@qf).reshape(1,2,2,19,64).sum(axis=2),
                (p.swapaxes(-1,-2)@df).reshape(1,2,2,19,64).sum(axis=2))
    for got, want in zip(actual,expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=0.04,atol=0.003)
    execution = attention.last_backward_execution
    assert execution["evidence_target"] == "rocm_gfx1201"
    assert execution["target_consumer"] == "rocm.gfx1201_attention_backward_program"
    assert execution["physical_attestation"]["device_arch"] == "gfx1201"
    # Public pairing must retain its source/Schedule/Tile/native artifact identities.
    assert all(len(execution[key]) == 64 for key in
               ("source_graph_ir_digest","schedule_artifact_hash","tile_program_digest","native_image_digest"))
    validate_native_vjp_execution_certificate(execution["execution_certificate"])
    assert execution["execution_certificate"]["evidence_scope"] == "exact_device"



def test_public_attention_rejects_unknown_rocm_architecture(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import native_attention_vjp as owner
    from tessera.compiler.graph_ir import IROp
    monkeypatch.setattr(rt,"_rocm_live_arch",lambda: "gfx9999")
    q = np.zeros((1,1,16,64),np.float16)
    with pytest.raises(ValueError,match="supported exact ROCm architecture"):
        owner.build_native_attention_vjp_package(source_graph_ir="module {}",
            source=IROp(result="o",op_name="tessera.flash_attn",operands=["%q","%k","%v"],
                        operand_types=["tensor<1x1x16x64xf16>"]*3),
            target="rocm",ordered_inputs=(q,q,q),arg_names=("q","k","v"),
            source_arg_names=("q","k","v"),out_cotangent=q)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("checkpoint", ["recompute", "saved"])
def test_gfx1201_reusable_attention_owner(dtype, checkpoint, monkeypatch):
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_attention import ResidentROCmAttentionTape
    from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
    from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module, _reference
    artifact = lower_scheduled_attention_backward(_module(1,4,2,17,19,64,dtype=dtype,
        dropout_p=0,lse_checkpoint=checkpoint),target="rocm_gfx1201")
    program = rocm_native.package_scheduled_attention_backward(artifact,pipeline_name="tessera-lower-to-rocm")
    rng = np.random.default_rng(42)
    storage = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
    q,k,v,do = [(rng.normal(size=s)*0.2).astype(storage) for s in
               [(1,4,17,64),(1,2,19,64),(1,2,19,64),(1,4,17,64)]]
    bias = np.zeros((1,4,17,19),np.float32)
    expected = _reference(do,q,k,v,bias,dropout_p=0)
    buffers = dict(q=q,key=k,v=v,do=do,bias=bias,dq=np.empty(q.shape,np.float32),
                   dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    hip = rt._load_hip_for_launch()
    counts = dict(hipMalloc=0,hipFree=0,hipModuleLoadData=0,hipModuleUnload=0)
    class CountHIP:
        def __getattr__(self,name):
            fn = getattr(hip,name)
            if name not in counts:
                return fn
            def call(*args):
                counts[name] += 1
                return fn(*args)
            return call
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda: CountHIP())
    with ResidentROCmAttentionTape(program,buffers) as tape:
        allocated = dict(counts)
        q.fill(0); k.fill(0); v.fill(0); bias.fill(100)
        first = tape.submit(do)
        second = tape.submit(np.zeros_like(do))
        third = tape.submit((do * 0.5).astype(storage))
        retired = tape.retire()
        with pytest.raises(ValueError,match="retiring"):
            tape.submit(do)
        assert first.result()["device_event_samples_valid"]
        assert not first.result()["device_event_selector_eligible"]
        assert first.result()["resident_forward_launches"] == 1
        assert second.result()["resident_forward_launches"] == (1 if checkpoint == "saved" else 2)
        result = first.result()["outputs"]
        zeros = second.result()["outputs"]
        assert third.result()["resident_forward_launches"] == (1 if checkpoint == "saved" else 3)
        for got,want in zip(third.result()["outputs"],expected,strict=True):
            np.testing.assert_allclose(got,0.5*want,rtol=0.04,atol=0.003)
        retired.result()
        for got,want in zip(result,expected,strict=True):
            np.testing.assert_allclose(got,want,rtol=0.04,atol=0.003)
        for got in zeros:
            np.testing.assert_array_equal(got,0)
        assert counts["hipMalloc"] == allocated["hipMalloc"]
        assert counts["hipModuleLoadData"] == 1
    assert counts["hipFree"] == counts["hipMalloc"]
    assert counts["hipModuleUnload"] == 1


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
def test_gfx1201_dynamic_matmul_reuses_one_image():
    from tessera import runtime as rt
    from tessera.compiler.driver import compile_graph_module
    from tests.unit.test_scheduled_matmul_consumers import _dynamic_module
    bundle = compile_graph_module(_dynamic_module(target="rocm",bounds=(64,64,48)),
        source_origin="gfx1201-dynamic-matmul",target="rocm_gfx1201",
        options={"package_native":True},enable_tool_validation=False)
    assert bundle.launch_descriptor.provenance["shape_policy"] == "bounded_dynamic"
    assert all(g.predicate == "max" for g in bundle.launch_descriptor.shape_guards)
    artifact = rt.RuntimeArtifact(metadata={"target":"rocm_gfx1201"},native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,tile_ir=bundle.tile.text,target_ir=bundle.target_ir.text)
    rng = np.random.default_rng(94)
    for m,n,k in ((37,29,35),(1,7,5),(64,64,48)):
        a = (rng.normal(size=(m,k))*0.2).astype(np.float16)
        b = (rng.normal(size=(k,n))*0.2).astype(np.float16)
        out = np.zeros((m,n),np.float32)
        result = rt.launch(artifact,dict(a=a,b=b,o=out,M=m,N=n,K=k))
        assert result["ok"],result
        np.testing.assert_allclose(out,a.astype(np.float32)@b.astype(np.float32),rtol=0.002,atol=0.001)
    a = np.zeros((65,5),np.float16); b = np.zeros((5,7),np.float16)
    result = rt.launch(artifact,dict(a=a,b=b,o=np.zeros((65,7),np.float32),M=65,N=7,K=5))
    assert not result["ok"]


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("checkpoint", ["recompute", "saved"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_gfx1201_external_reader_orders_reuse_and_retirement(checkpoint, asynchronous, monkeypatch):
    import ctypes as ct
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_attention import ResidentROCmAttentionTape
    from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
    from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module
    schedule=lower_scheduled_attention_backward(_module(1,4,2,17,19,64,dtype="fp16",
        dropout_p=0,lse_checkpoint=checkpoint),target="rocm_gfx1201")
    program=rocm_native.package_scheduled_attention_backward(schedule,pipeline_name="tessera-lower-to-rocm")
    rng=np.random.default_rng(241)
    q,k,v,do=[(rng.normal(size=s)*0.2).astype(np.float16) for s in
              [(1,4,17,64),(1,2,19,64),(1,2,19,64),(1,4,17,64)]]
    buffers=dict(q=q,key=k,v=v,do=do,bias=np.zeros((1,4,17,19),np.float32),
        dq=np.empty(q.shape,np.float32),dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    hip=rt._load_hip_for_launch()
    class NoDeviceSync:
        def __getattr__(self,name):
            if name=="hipDeviceSynchronize":
                raise AssertionError("owner must not synchronize unrelated streams")
            return getattr(hip,name)
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda:NoDeviceSync())
    P=ct.c_void_p
    consumer=P(); copies=[]
    tape=ResidentROCmAttentionTape(program,buffers)
    assert hip.hipStreamCreateWithFlags(ct.byref(consumer),1)==0
    try:
        with pytest.raises(ValueError,match="completed backward"):
            tape.reader(consumer.value)
        expected=tape.backward(do)
        reader=tape.reader(consumer.value)
        with pytest.raises(ValueError,match="active readers"):
            tape.submit(do)
        with pytest.raises(ValueError,match="external readers"):
            tape.close()
        names=("dq","dk","dv")
        for name,want in zip(names,expected,strict=True):
            address,shape,dtype=reader.outputs[name]
            assert shape==want.shape and dtype=="float32"
            dest=P();assert hip.hipMalloc(ct.byref(dest),want.nbytes)==0;copies.append(dest)
            assert hip.hipMemcpyAsync(dest,P(address),want.nbytes,3,consumer)==0
        retirement=tape.retire()
        assert not retirement.done()
        if asynchronous:
            released = reader.release_async()
            assert not released.cancel()
            released.result(timeout=30)
        else:
            reader.close()
        retirement.result(timeout=30)
        with pytest.raises(ValueError,match="released"):
            _=reader.outputs
        for ptr,want in zip(copies,expected,strict=True):
            got=np.empty_like(want)
            assert hip.hipMemcpy(got.ctypes.data_as(P),ptr,got.nbytes,2)==0
            np.testing.assert_array_equal(got,want)
    finally:
        tape.close()
        assert hip.hipStreamSynchronize(consumer)==0
        for ptr in copies: assert hip.hipFree(ptr)==0
        assert hip.hipStreamDestroy(consumer)==0


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("start,end", [(0, 1), (3, 10), (7, 9), (0, 16)])
def test_gfx1201_scheduled_paged_kv_package_executes(start, end):
    """GFX1201-PARITY, the last family: the paged-KV read package on RDNA4.

    The generator is a scalar per-thread gather (no WMMA fragment). This is
    the gfx1201 row the family's promotion rests on; the gfx1151 twin is
    `test_rocm_e2e_spine.py::test_exact_gfx1151_paged_kv_descriptor_matches_permuted_page_oracle`."""
    from tessera import runtime as rt
    from tests.unit.test_rocm_e2e_spine import _paged_kv_module

    assert rt._rocm_live_arch() == "gfx1201"
    package = rocm_native.package_paged_kv_read(
        _paged_kv_module(start=start, end=end), pipeline_name="tessera-lower-to-rocm",
        architecture="gfx1201")
    assert package.image.architecture == "gfx1201"
    assert package.image.target == "rocm_gfx1201"
    assert package.descriptor.abi_id == rocm_native.GFX_PAGED_KV_F32_ABI
    artifact = rt.RuntimeArtifact(
        graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1201"}, native_image=package.image,
        launch_descriptor=package.descriptor)
    rng = np.random.default_rng(2203 + start)
    pages = np.ascontiguousarray(rng.standard_normal((4, 4, 3, 8)), dtype=np.float32)
    table = np.array([2, 0, 3, 1], dtype=np.int32)
    logical = pages[table].reshape(16, 3, 8)
    tokens = end - start
    output = np.zeros((tokens, 3, 8), dtype=np.float32)
    result = rt.launch(artifact, {
        "pages": pages, "page_table": table, "slice": output,
        "P": 4, "LP": 4, "PageSize": 4, "H": 3, "D": 8, "Start": start, "Tokens": tokens})
    assert result["ok"] is True and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_array_equal(output, logical[start:end])


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("storage", ["int8", "int4"])
@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 19, 23), (65, 48, 37)])
def test_gfx1201_scheduled_matmul_package_executes_integer_storage(shape, storage):
    """GFX1201-PARITY slice 1b: int8/int4 storage with i32 accumulation on the
    typed route, RDNA4 (V_WMMA_I32_16X16X16_IU8 / IU4 through rdna4_wmma
    fragments: 8 int8 per lane in two i32 words, 8 int4 nibbles in one).
    Exact against the int32 numpy product; int4 values ride int8 containers."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _integer_operands, _module as matmul_module
    assert rt._rocm_live_arch() == "gfx1201"
    m, k, n = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        matmul_module(target="rocm", shape=shape, dtype=storage, output_dtype="int32"), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.abi_id == (rocm_native.GFX_MATMUL_I8_I32_ABI if storage == "int8" else rocm_native.GFX_MATMUL_I4_I32_ABI)
    a, b, expected = _integer_operands(shape, storage, 97 + len(storage))
    output = np.zeros((m, n), np.int32)
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("activation,bias", [("none", False), ("gelu", True)])
@pytest.mark.parametrize("shape", [(16, 16, 16), (65, 48, 37)])
def test_gfx1201_scheduled_matmul_package_executes_bf16(shape, activation, bias):
    """bf16 storage on the typed route, RDNA4 (slice 1b), plain and fused."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module, _epilogue_reference
    ml_dtypes = pytest.importorskip("ml_dtypes")
    assert rt._rocm_live_arch() == "gfx1201"
    m, k, n = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        matmul_module(target="rocm", shape=shape, dtype="bf16", activation=activation, bias=bias), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == (rocm_native.GFX_MATMUL_BF16_F32_FUSED_ABI if bias else rocm_native.GFX_MATMUL_BF16_F32_ABI)
    rng = np.random.default_rng(311 + len(activation))
    a = (rng.normal(size=(m, k)) * 0.4).astype(ml_dtypes.bfloat16)
    b = (rng.normal(size=(k, n)) * 0.4).astype(ml_dtypes.bfloat16)
    bias_arr = (rng.normal(size=(n,)) * 0.5).astype(np.float32) if bias else None
    output = np.zeros((m, n), np.float32)
    buffers = {"a": a, "b": b, "o": output}
    if bias:
        buffers["bias"] = bias_arr
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    np.testing.assert_allclose(output, _epilogue_reference(a.astype(np.float32), b.astype(np.float32), bias_arr, activation), rtol=0, atol=8e-2)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 owning-device gate")
@pytest.mark.parametrize("shape,panel", [((1024, 1024, 1024), (64, 64)), ((2048, 2048, 2048), (32, 64)), ((1024, 1024, 1000), (16, 16))])
def test_gfx1201_scheduled_matmul_package_executes_the_selected_panel(shape, panel):
    """The panels gfx1201 selects per shape (typed-route gap packets: the
    4x4 in the fully tiled [1024, 2048) band, the 2x4 from 2048 up) execute
    exactly like the 1x1 they replace; a ragged neighbour keeps the 1x1.
    Correctness only."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module
    assert rt._rocm_live_arch() == "gfx1201"
    m, k, n = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(matmul_module(target="rocm", shape=shape), target="rocm_gfx1201")
    scheduled_matmul.verify_matmul_projection(artifact)
    assert (artifact.macro_tile_m, artifact.macro_tile_n) == panel
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.provenance["physical_route"] == f"gfx1201_register_wmma_{panel[0] // 16}x{panel[1] // 16}"
    rng = np.random.default_rng(1024)
    a = (rng.normal(size=(m, k)) * 0.25).astype(np.float16)
    b = (rng.normal(size=(k, n)) * 0.25).astype(np.float16)
    output = np.zeros((m, n), np.float32)
    runtime = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(runtime, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}})
    assert result["ok"] and result["execution_kind"] == "native_gpu", json.dumps(result, default=str)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(output, expected, rtol=0, atol=5e-2 * float(np.abs(expected).max()) / 10 + 2e-2)
