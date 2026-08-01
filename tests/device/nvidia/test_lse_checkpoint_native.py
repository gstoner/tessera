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
                    shape: tuple[int, int, int, int, int, int, int] = (1, 2, 1, 3, 4, 4, 3)) -> GraphIRModule:
    q, k, v, _, lse = _types(shape)
    b, hq, _, sq, _, _, dv = shape
    out = IRType(f"tensor<{b}x{hq}x{sq}x{dv}xf32>",
                 tuple(str(dim) for dim in (b, hq, sq, dv)), "fp32")
    names = "o,row_lse" if saved else "o"
    result_types = [out, lse] if saved else [out]
    returns = ["%o", "%row_lse"] if saved else ["%o"]
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_lse_forward", args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=result_types,
        body=[IROp(
            result=names, op_name="tessera.flash_attn", operands=["%q", "%k", "%v"],
            operand_types=[str(q), str(k), str(v)], result_type=str(out),
            kwargs={"scale": 0.5, "causal": True, **({"lse_checkpoint": "saved"} if saved else {})},
        )], return_values=returns,
    )])


def _backward_module(*, saved: bool,
                     shape: tuple[int, int, int, int, int, int, int] = (1, 2, 1, 3, 4, 4, 3)) -> GraphIRModule:
    q, k, v, do, lse = _types(shape)
    args = [IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%do", "%q", "%k", "%v"]
    operand_types = [str(do), str(q), str(k), str(v)]
    kwargs: dict[str, object] = {
        "scale": 0.5, "causal": True, "route": "deterministic_direct",
        "deterministic": True, "workspace_limit_bytes": 0,
    }
    if saved:
        args.append(IRArg("row_lse", lse)); operands.append("%row_lse"); operand_types.append(str(lse))
        kwargs["lse_checkpoint"] = "saved"
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


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray):
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    dv = v.shape[-1]
    out = np.empty((b, hq, sq, dv), dtype=np.float32)
    lse = np.empty((b, hq, sq), dtype=np.float32)
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv_out = np.zeros_like(v)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                scores = 0.5 * (k[batch, kv_head] @ q[batch, head, row])
                scores[row + 1 :] = -np.inf
                row_lse = np.log(np.exp(scores - np.max(scores)).sum()) + np.max(scores)
                p = np.exp(scores - row_lse)
                out[batch, head, row] = p @ v[batch, kv_head]
                lse[batch, head, row] = row_lse
                delta = do[batch, head, row] @ out[batch, head, row]
                for key in range(row + 1):
                    ds = p[key] * (do[batch, head, row] @ v[batch, kv_head, key] - delta)
                    dq[batch, head, row] += 0.5 * ds * k[batch, kv_head, key]
                    dk[batch, kv_head, key] += 0.5 * ds * q[batch, head, row]
                    dv_out[batch, kv_head, key] += p[key] * do[batch, head, row]
    return out, lse, (dq, dk, dv_out)


@pytest.mark.hardware_nvidia
def test_sm120_saved_lse_forward_backward_matches_recompute_and_oracle() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    forward_saved, forward_recompute = _compile(_forward_module(saved=True)), _compile(_forward_module(saved=False))
    backward_saved, backward_recompute = _compile(_backward_module(saved=True)), _compile(_backward_module(saved=False))
    assert forward_saved.launch_descriptor.abi_id == "tessera.nvidia.attention.q_k_v_o_row_lse_dims.f32.v1"
    assert backward_saved.launch_descriptor.abi_id == "tessera.nvidia.attention_backward.do_q_k_v_row_lse_dq_dk_dv_dims.f32.v1"
    assert forward_saved.launch_descriptor.workspace.bytes == backward_saved.launch_descriptor.workspace.bytes == 0
    rng = np.random.default_rng(120_001)
    q = (rng.normal(size=(1, 2, 3, 4)) * 0.2).astype(np.float32)
    k = (rng.normal(size=(1, 1, 4, 4)) * 0.2).astype(np.float32)
    v = (rng.normal(size=(1, 1, 4, 3)) * 0.2).astype(np.float32)
    do = (rng.normal(size=(1, 2, 3, 3)) * 0.2).astype(np.float32)
    scalars = {"B": 1, "Hq": 2, "Hkv": 1, "Sq": 3, "Sk": 4, "D": 4, "Dv": 3}
    saved_o = np.empty((1, 2, 3, 3), dtype=np.float32); row_lse = np.empty((1, 2, 3), dtype=np.float32)
    recompute_o = np.empty_like(saved_o)
    saved_forward = launch(compile_result_from_bundle(forward_saved, module=_forward_module(saved=True)).to_runtime_artifact(),
                           {"q": q, "k": k, "v": v, "o": saved_o, "row_lse": row_lse, **scalars})
    recompute_forward = launch(compile_result_from_bundle(forward_recompute, module=_forward_module(saved=False)).to_runtime_artifact(),
                               {"q": q, "k": k, "v": v, "o": recompute_o, **scalars})
    assert saved_forward["ok"], saved_forward.get("reason")
    assert recompute_forward["ok"], recompute_forward.get("reason")
    ref_o, ref_lse, ref_grads = _reference(q, k, v, do)
    np.testing.assert_allclose(saved_forward["output"][0], ref_o, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(saved_forward["output"][1], ref_lse, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(saved_forward["output"][0], recompute_forward["output"], rtol=0.0, atol=0.0)
    saved_grads = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    recompute_grads = [np.empty_like(q), np.empty_like(k), np.empty_like(v)]
    saved_backward = launch(compile_result_from_bundle(backward_saved, module=_backward_module(saved=True)).to_runtime_artifact(),
                            {"do": do, "q": q, "k": k, "v": v, "row_lse": row_lse,
                             "dq": saved_grads[0], "dk": saved_grads[1], "dv": saved_grads[2], **scalars})
    recompute_backward = launch(compile_result_from_bundle(backward_recompute, module=_backward_module(saved=False)).to_runtime_artifact(),
                                {"do": do, "q": q, "k": k, "v": v,
                                 "dq": recompute_grads[0], "dk": recompute_grads[1], "dv": recompute_grads[2], **scalars})
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
