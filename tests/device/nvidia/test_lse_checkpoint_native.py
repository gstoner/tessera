"""Exact-device proof for the explicit SM120 saved-LSE physical ABI."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import launch
from tests._support.nvidia import nvidia_cuda_host_ready


def _types(shape: tuple[int, int, int, int, int, int, int] = (1, 2, 1, 3, 4, 4, 3)) -> tuple[IRType, IRType, IRType, IRType, IRType]:
    b, hq, hkv, sq, sk, d, dv = shape

    def tensor(*dims: int) -> IRType:
        encoded = "x".join(str(dim) for dim in dims)
        return IRType(f"tensor<{encoded}xf32>", tuple(str(dim) for dim in dims), "fp32")

    q = tensor(b, hq, sq, d)
    k = tensor(b, hkv, sk, d)
    v = tensor(b, hkv, sk, dv)
    do = tensor(b, hq, sq, dv)
    lse = tensor(b, hq, sq)
    return q, k, v, do, lse


def _forward_module(*, saved: bool,
                    shape: tuple[int, int, int, int, int, int, int] = (1, 2, 1, 3, 4, 4, 3),
                    bias: bool = False) -> GraphIRModule:
    q, k, v, _, lse = _types(shape)
    b, hq, _, sq, _, _, dv = shape
    out = IRType(f"tensor<{b}x{hq}x{sq}x{dv}xf32>",
                 tuple(str(dim) for dim in (b, hq, sq, dv)), "fp32")
    names = "o,row_lse" if saved else "o"
    result_types = [out, lse] if saved else [out]
    returns = ["%o", "%row_lse"] if saved else ["%o"]
    args = [IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    if bias:
        bt = IRType(f"tensor<{b}x{hq}x{sq}x{shape[4]}xf32>",
                    tuple(map(str, (b, hq, sq, shape[4]))), "fp32")
        args.append(IRArg("bias", bt))
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_lse_forward", args=args,
        result_types=result_types,
        body=[IROp(
            result=names, op_name="tessera.flash_attn", operands=["%" + arg.name for arg in args],
            operand_types=[str(arg.ir_type) for arg in args], result_type=str(out),
            inferred_types=tuple(result_types),
            kwargs={"scale": 0.5, "causal": True, **({"lse_checkpoint": "saved"} if saved else {})},
        )], return_values=returns,
    )])


def _backward_module(*, saved: bool,
                     shape: tuple[int, int, int, int, int, int, int] = (1, 2, 1, 3, 4, 4, 3),
                    bias: bool = False) -> GraphIRModule:
    q, k, v, do, lse = _types(shape)
    args = [IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%do", "%q", "%k", "%v"]
    operand_types = [str(do), str(q), str(k), str(v)]
    kwargs: dict[str, object] = {
        "scale": 0.5, "causal": True, "route": "deterministic_direct",
        "deterministic": True, "workspace_limit_bytes": 0,
    }
    if saved:
        args.extend((IRArg("output", do), IRArg("row_lse", lse)))
        operands.extend(("%output", "%row_lse"))
        operand_types.extend((str(do), str(lse)))
        kwargs["lse_checkpoint"] = "saved"
    if bias:
        b, hq, _, sq, sk, _, _ = shape
        bt = IRType(f"tensor<{b}x{hq}x{sq}x{sk}xf32>",
                    tuple(map(str, (b, hq, sq, sk))), "fp32")
        index = 5 if saved else 4
        args.insert(index, IRArg("bias", bt))
        operands.insert(index, "%bias")
        operand_types.insert(index, str(bt))
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_lse_backward", args=args, result_types=[q, k, v],
        body=[IROp(
            result="dq,dk,dv", op_name="tessera.flash_attn_bwd", operands=operands,
            operand_types=operand_types, result_type=f"({q}, {k}, {v})", kwargs=kwargs,
        )], return_values=["%dq", "%dk", "%dv"],
    )])


def _compile(module: GraphIRModule):
    return compile_graph_module(
        module, source_origin="NVIDIA-LSE-1", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray, bias: np.ndarray | None = None, *, scale: float = 0.5):
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    do64 = np.asarray(do, dtype=np.float64)
    b, hq, sq, d = q64.shape
    _, hkv, sk, _ = k64.shape
    dv = v64.shape[-1]
    out = np.empty((b, hq, sq, dv), dtype=np.float64)
    lse = np.empty((b, hq, sq), dtype=np.float64)
    dq = np.zeros_like(q64); dk = np.zeros_like(k64); dv_out = np.zeros_like(v64)
    # Tessera's causal mask is bottom-right aligned (`tessera.ops.flash_attn`:
    # `np.triu(..., k=1 + max(Sk - Sq, 0))`): query row r sees keys
    # 0..r + (Sk - Sq). This oracle used top-left alignment, which differs
    # whenever Sq < Sk (here 3 < 4) and failed a correct kernel.
    offset = max(sk - sq, 0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                visible = min(row + 1 + offset, sk)
                scores = scale * (k64[batch, kv_head] @ q64[batch, head, row])
                if bias is not None:
                    scores += np.asarray(bias[batch, head, row], dtype=np.float64)
                scores[visible:] = -np.inf
                row_lse = np.log(np.exp(scores - np.max(scores)).sum()) + np.max(scores)
                p = np.exp(scores - row_lse)
                out[batch, head, row] = p @ v[batch, kv_head]
                lse[batch, head, row] = row_lse
                delta = do64[batch, head, row] @ out[batch, head, row]
                for key in range(visible):
                    ds = p[key] * (do64[batch, head, row] @ v64[batch, kv_head, key] - delta)
                    dq[batch, head, row] += scale * ds * k64[batch, kv_head, key]
                    dk[batch, kv_head, key] += scale * ds * q64[batch, head, row]
                    dv_out[batch, kv_head, key] += p[key] * do64[batch, head, row]
    return out, lse, (dq, dk, dv_out)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [
    (1, 2, 1, 3, 4, 4, 3),
    (1, 4, 2, 16, 16, 32, 32),
    (2, 4, 2, 5, 7, 8, 6),
])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("neutral_policy", [False, True])
def test_sm120_saved_lse_forward_backward_matches_recompute_and_oracle(shape, with_bias, permuted, neutral_policy) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    def compile_case(module):
        if neutral_policy:
            module.functions[0].body[0].kwargs.update(
                scale=1, softcap=0, logit_softcap=0, dropout=0, dropout_p=0, window=(-1, -1))
        if permuted and module.functions[0].body[0].kwargs.get("lse_checkpoint") == "saved":
            module.functions[0].args.reverse()
        return _compile(module)
    forward_saved, forward_recompute = (
        compile_case(_forward_module(saved=True, shape=shape, bias=with_bias)),
        compile_case(_forward_module(saved=False, shape=shape, bias=with_bias)),
    )
    backward_saved, backward_recompute = (
        compile_case(_backward_module(saved=True, shape=shape, bias=with_bias)),
        compile_case(_backward_module(saved=False, shape=shape, bias=with_bias)),
    )
    assert forward_saved.launch_descriptor.abi_id == ("tessera.nvidia.attention.q_k_v_bias_o_row_lse_dims.f32.v1" if with_bias else "tessera.nvidia.attention.q_k_v_o_row_lse_dims.f32.v1")
    assert backward_saved.launch_descriptor.abi_id == ("tessera.nvidia.attention_backward.do_q_k_v_output_bias_row_lse_dq_dk_dv_dims.f32.v2" if with_bias else "tessera.nvidia.attention_backward.do_q_k_v_output_row_lse_dq_dk_dv_dims.f32.v2")
    assert len(backward_saved.launch_descriptor.buffers) == 9 + int(with_bias)
    assert forward_saved.launch_descriptor.workspace.bytes == backward_saved.launch_descriptor.workspace.bytes == 0
    for bundle in (forward_saved, backward_saved):
        provenance = bundle.launch_descriptor.provenance
        for key in ("graph_ir_digest", "schedule_digest", "schedule_ir_digest", "tile_ir_digest"):
            assert len(provenance[key]) == 64, (key, provenance)
    for bundle in (forward_recompute, backward_recompute):
        assert len(bundle.launch_descriptor.provenance["tile_ir_digest"]) == 64
    rng = np.random.default_rng(120_001)
    b, hq, hkv, sq, sk, d, dv = shape
    q = (rng.normal(size=(b, hq, sq, d)) * 0.2).astype(np.float32)
    k = (rng.normal(size=(b, hkv, sk, d)) * 0.2).astype(np.float32)
    v = (rng.normal(size=(b, hkv, sk, dv)) * 0.2).astype(np.float32)
    do = (rng.normal(size=(b, hq, sq, dv)) * 0.2).astype(np.float32)
    bias = (rng.normal(size=(b, hq, sq, sk)) * .3).astype(np.float32) if with_bias else None
    extra = {"bias": bias} if with_bias else {}
    scalars = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), shape, strict=True))
    saved_o = np.empty((b, hq, sq, dv), dtype=np.float32); row_lse = np.empty((b, hq, sq), dtype=np.float32)
    recompute_o = np.empty_like(saved_o)
    saved_forward = launch(compile_result_from_bundle(forward_saved, module=_forward_module(saved=True, shape=shape, bias=with_bias)).to_runtime_artifact(),
                           {"q": q, "k": k, "v": v, "o": saved_o, "row_lse": row_lse, **scalars, **extra})
    recompute_forward = launch(compile_result_from_bundle(forward_recompute, module=_forward_module(saved=False, shape=shape, bias=with_bias)).to_runtime_artifact(),
                               {"q": q, "k": k, "v": v, "o": recompute_o, **scalars, **extra})
    assert saved_forward["ok"], saved_forward.get("reason")
    assert recompute_forward["ok"], recompute_forward.get("reason")
    ref_o, ref_lse, ref_grads = _reference(q, k, v, do, bias, scale=1.0 if neutral_policy else 0.5)
    # Exercise saved-LSE on caller-owned CUDA buffers and one explicit stream.
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    with NvidiaDeviceSession() as session:
        qd, kd, vd = session.upload(q), session.upload(k), session.upload(v)
        resident_extra = {"bias": session.upload(bias)} if with_bias else {}
        resident_o = session.empty(saved_o.shape, np.float32)
        resident_lse = session.empty(row_lse.shape, np.float32)
        resident_artifact = compile_result_from_bundle(
            forward_saved, module=_forward_module(saved=True, shape=shape, bias=with_bias)
        ).to_runtime_artifact()
        resident = launch(
            resident_artifact,
            {"q": qd, "k": kd, "v": vd, "o": resident_o,
             "row_lse": resident_lse, **scalars, **resident_extra},
            stream=session.stream,
        )
        assert resident["ok"], resident.get("reason")
        assert resident["execution_kind"] == "native_gpu"
        np.testing.assert_allclose(
            session.download(resident_o), ref_o, rtol=3e-5, atol=3e-5
        )
        np.testing.assert_allclose(
            session.download(resident_lse), ref_lse, rtol=3e-5, atol=3e-5
        )
        resident_backward_artifact = compile_result_from_bundle(
            backward_saved, module=_backward_module(saved=True, shape=shape, bias=with_bias)
        ).to_runtime_artifact()
        resident_do = session.upload(do)
        resident_dq = session.empty(q.shape, np.float32)
        resident_dk = session.empty(k.shape, np.float32)
        resident_dv = session.empty(v.shape, np.float32)
        resident_backward = launch(
            resident_backward_artifact,
            {"do": resident_do, "q": qd, "k": kd, "v": vd,
             "output": resident_o, "row_lse": resident_lse, "dq": resident_dq,
             "dk": resident_dk, "dv": resident_dv, **scalars, **resident_extra},
            stream=session.stream,
        )
        assert resident_backward["ok"], resident_backward.get("reason")
        assert resident_backward["execution_kind"] == "native_gpu"
        for actual, expected in zip(
            (session.download(resident_dq), session.download(resident_dk),
             session.download(resident_dv)), ref_grads, strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(saved_forward["output"][0], ref_o, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(saved_forward["output"][1], ref_lse, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(saved_forward["output"][0], recompute_forward["output"], rtol=0.0, atol=0.0)
    saved_grads = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    recompute_grads = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    saved_backward = launch(compile_result_from_bundle(backward_saved, module=_backward_module(saved=True, shape=shape, bias=with_bias)).to_runtime_artifact(),
                            {"do": do, "q": q, "k": k, "v": v, "output": saved_o, "row_lse": row_lse,
                             "dq": saved_grads[0], "dk": saved_grads[1], "dv": saved_grads[2], **scalars, **extra})
    recompute_backward = launch(compile_result_from_bundle(backward_recompute, module=_backward_module(saved=False, shape=shape, bias=with_bias)).to_runtime_artifact(),
                                {"do": do, "q": q, "k": k, "v": v,
                                 "dq": recompute_grads[0], "dk": recompute_grads[1], "dv": recompute_grads[2], **scalars, **extra})
    assert saved_backward["ok"], saved_backward.get("reason")
    assert recompute_backward["ok"], recompute_backward.get("reason")
    for saved, recompute, oracle in zip(saved_backward["output"], recompute_backward["output"], ref_grads, strict=True):
        np.testing.assert_allclose(saved, oracle, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(saved, recompute, rtol=3e-5, atol=3e-5)


@pytest.mark.hardware_nvidia
def test_sm120_saved_lse_rejects_rank_mismatched_physical_buffer() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    bundle = _compile(_forward_module(saved=True))
    artifact = compile_result_from_bundle(bundle, module=_forward_module(saved=True)).to_runtime_artifact()
    x = np.ones((1, 2, 3, 4), dtype=np.float32)
    result = launch(artifact, {
        "q": x, "k": np.ones((1, 1, 4, 4), np.float32), "v": np.ones((1, 1, 4, 3), np.float32),
        "o": np.empty((1, 2, 3, 3), np.float32), "row_lse": np.empty((1, 2, 3, 1), np.float32),
        "B": 1, "Hq": 2, "Hkv": 1, "Sq": 3, "Sk": 4, "D": 4, "Dv": 3,
    })
    assert not result["ok"]
    assert result["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("with_bias", [False, True])
def test_sm120_checkpoint_pair_retains_saved_output_and_bias_generation(with_bias):
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    from tessera.compiler.nvidia_native import package_attention_checkpoint_pair
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    shape = (2, 4, 2, 5, 7, 8, 6)
    forward = _forward_module(saved=True, shape=shape, bias=with_bias)
    forward.functions[0].body[0].result = "output,row_lse"
    forward.functions[0].return_values = ["%output", "%row_lse"]
    pair = package_attention_checkpoint_pair(
        forward, _backward_module(saved=True, shape=shape, bias=with_bias),
        pipeline_name="tessera-nvidia-pipeline-sm120")
    rng = np.random.default_rng(120_202)
    b,hq,hkv,sq,sk,d,dv = shape
    q,k,v,do = [(rng.normal(size=dims)*.2).astype(np.float32) for dims in
                ((b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv), (b,hq,sq,dv))]
    bias = (rng.normal(size=(b,hq,sq,sk))*.3).astype(np.float32) if with_bias else None
    expected_o, _, expected_grads = _reference(q,k,v,do,bias)
    def download(value):
        import ctypes
        interface = value.__cuda_array_interface__
        host = np.empty(interface["shape"], np.dtype(interface["typestr"]))
        copy = ctypes.CDLL("libcuda.so.1").cuMemcpyDtoH_v2
        copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        copy.restype = ctypes.c_int
        assert copy(ctypes.c_void_p(host.ctypes.data),
                    ctypes.c_void_p(interface["data"][0]), host.nbytes) == 0
        return host
    with NvidiaDeviceSession() as session:
        qd,kd,vd,dod = (session.upload(value) for value in (q,k,v,do))
        bd = session.upload(bias) if with_bias else None
        tape = pair.capture(qd,kd,vd,bias=bd)
        try:
            np.testing.assert_allclose(download(tape.primal), expected_o, rtol=4e-5, atol=4e-5)
            # Caller mutation must not change the captured generation.
            for value in (qd,kd,vd,*([bd] if with_bias else [])):
                import ctypes
                zeros = np.zeros(value.shape, np.float32)
                assert session.lib.tessera_nvidia_device_upload(
                    ctypes.c_void_p(value.ptr), ctypes.c_void_p(zeros.ctypes.data),
                    zeros.nbytes, ctypes.c_void_p(session.stream)) == 0
                assert session.synchronize() == 0
            for _ in range(2):
                grads = tape.backward(dod)
                for actual,expected in zip(grads,expected_grads,strict=True):
                    np.testing.assert_allclose(download(actual), expected, rtol=4e-5, atol=4e-5)
        finally:
            tape.close()
        assert tape.closed


@pytest.mark.parametrize("saved",[False,True])
@pytest.mark.parametrize("backward",[False,True])
@pytest.mark.parametrize("bias",[False,True])
def test_checkpoint_event_profiler_returns_actual_timed_outputs(saved,backward,bias):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 device/toolchain required")
    from tessera.runtime import _nvidia_native_descriptor_device_latency
    shape=(1,4,2,5,7,8,6)
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(507065)
    q,k,v,do=[rng.normal(size=s).astype(np.float32)*.2 for s in
              ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    bias_value=rng.normal(size=(b,hq,sq,sk)).astype(np.float32)*.1 if bias else None
    output,lse,gradients=_reference(q,k,v,do,bias_value)
    module=(_backward_module if backward else _forward_module)(saved=saved,shape=shape,bias=bias)
    bundle=compile_graph_module(module,source_origin="NVIDIA-LSE-1",target="nvidia_sm120",
                                options={"package_native":True},enable_tool_validation=False)
    args={"q":q,"k":k,"v":v,**dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True))}
    if bias:args["bias"]=bias_value
    if backward:
        args["do"]=do
        expected=dict(zip(("dq","dk","dv"),gradients,strict=True))
        if saved:args.update(output=output.astype(np.float32),row_lse=lse.astype(np.float32))
    else:
        expected={"o":output}
        if saved:expected["row_lse"]=lse
    for name,reference in expected.items():args[name]=np.full_like(reference,np.nan,dtype=np.float32)
    latency=_nvidia_native_descriptor_device_latency(bundle.native_image,bundle.launch_descriptor,
                                                     args,reps=5,warmup=2)
    assert np.isfinite(latency) and latency>0
    for name,reference in expected.items():
        assert np.isfinite(args[name]).all(),f"event profiler did not read back {name}"
        np.testing.assert_allclose(args[name],reference,rtol=4e-5,atol=4e-5)
