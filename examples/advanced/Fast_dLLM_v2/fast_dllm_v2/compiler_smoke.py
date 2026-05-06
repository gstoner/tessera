"""Compiler-facing Graph IR for the Fast dLLM v2 sample."""

from __future__ import annotations

from tessera.compiler import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type

from .config import FastDLLMConfig, tiny_config


def build_toy_graph_ir(cfg: FastDLLMConfig | None = None) -> GraphIRModule:
    cfg = cfg or tiny_config()
    h = cfg.hidden_size
    ff = cfg.intermediate_size
    tokens = cfg.block_tokens * cfg.branch_count
    x_ty = tensor_ir_type((tokens, h), "f32")
    ff_ty = tensor_ir_type((tokens, ff), "f32")
    w_h_h_ty = tensor_ir_type((h, h), "f32")
    w_h_ff_ty = tensor_ir_type((h, ff), "f32")
    w_ff_h_ty = tensor_ir_type((ff, h), "f32")

    args = [
        IRArg("x", x_ty, layout="row_major"),
        IRArg("denoise_w", w_h_h_ty, layout="row_major"),
        IRArg("confidence_w", w_h_h_ty, layout="row_major"),
        IRArg("ff_in", w_h_ff_ty, layout="row_major"),
        IRArg("ff_out", w_ff_h_ty, layout="row_major"),
    ]
    ops = [
        IROp("denoise", "tessera.matmul", ["%x", "%denoise_w"], [str(x_ty), str(w_h_h_ty)], str(x_ty)),
        IROp("norm0", "tessera.rmsnorm_safe", ["%denoise"], [str(x_ty)], str(x_ty), kwargs={"eps": cfg.rms_norm_eps}),
        IROp("scores", "tessera.matmul", ["%norm0", "%confidence_w"], [str(x_ty), str(w_h_h_ty)], str(x_ty)),
        IROp("confidence", "tessera.softmax", ["%scores"], [str(x_ty)], str(x_ty), kwargs={"axis": -1}),
        IROp("ff_proj", "tessera.matmul", ["%confidence", "%ff_in"], [str(x_ty), str(w_h_ff_ty)], str(ff_ty)),
        IROp("ff_act", "tessera.relu", ["%ff_proj"], [str(ff_ty)], str(ff_ty)),
        IROp("updated", "tessera.matmul", ["%ff_act", "%ff_out"], [str(ff_ty), str(w_ff_h_ty)], str(x_ty)),
        IROp("norm1", "tessera.rmsnorm_safe", ["%updated"], [str(x_ty)], str(x_ty), kwargs={"eps": cfg.rms_norm_eps}),
    ]
    fn = GraphIRFunction(
        name="fast_dllm_v2_confidence_decode_step",
        args=args,
        result_types=[x_ty],
        body=ops,
        return_values=["%norm1"],
        fn_attrs={
            "tessera.example": '"Fast_dLLM_v2"',
            "tessera.decode.branches": f"{cfg.branch_count} : i64",
            "tessera.kv.block_tokens": f"{cfg.block_tokens} : i64",
            "tessera.confidence_tau": f"{cfg.confidence_tau}",
        },
    )
    return GraphIRModule(functions=[fn], module_attrs={
        "tessera.ir.version": '"1.0"',
        "tessera.example": '"Fast_dLLM_v2"',
    })


def compile_toy_graph(*, target: str = "apple_cpu"):
    # The checked-in typed MLIR fixture covers parser validation separately.
    return compile_graph_module(
        build_toy_graph_ir(),
        source_origin="examples/advanced/Fast_dLLM_v2",
        target=target,
        example_id="fast_dllm_v2_tiny",
        enable_tool_validation=False,
    )
