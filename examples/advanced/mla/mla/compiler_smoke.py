"""Compiler-facing Graph IR for the FlashMLA sample."""

from __future__ import annotations

from tessera.compiler import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type

from .config import MLAConfig, tiny_config


def build_toy_graph_ir(cfg: MLAConfig | None = None) -> GraphIRModule:
    cfg = cfg or tiny_config()
    tokens = cfg.batch_size * cfg.seq_len
    d = cfg.model_dim
    c = cfg.latent_dim
    x_ty = tensor_ir_type((tokens, d), "f32")
    latent_ty = tensor_ir_type((tokens, c), "f32")
    scores_ty = tensor_ir_type((tokens, tokens), "f32")
    w_d_c_ty = tensor_ir_type((d, c), "f32")
    w_c_d_ty = tensor_ir_type((c, d), "f32")
    w_d_s_ty = tensor_ir_type((d, tokens), "f32")
    w_d_d_ty = tensor_ir_type((d, d), "f32")

    args = [
        IRArg("x", x_ty, layout="row_major"),
        IRArg("q_down", w_d_c_ty, layout="row_major"),
        IRArg("q_up", w_c_d_ty, layout="row_major"),
        IRArg("kv_down", w_d_c_ty, layout="row_major"),
        IRArg("k_absorb", w_c_d_ty, layout="row_major"),
        IRArg("score_w", w_d_s_ty, layout="row_major"),
        IRArg("v_absorb", w_c_d_ty, layout="row_major"),
        IRArg("out_w", w_d_d_ty, layout="row_major"),
    ]
    ops = [
        IROp("q_latent", "tessera.matmul", ["%x", "%q_down"], [str(x_ty), str(w_d_c_ty)], str(latent_ty)),
        IROp("q_full", "tessera.matmul", ["%q_latent", "%q_up"], [str(latent_ty), str(w_c_d_ty)], str(x_ty)),
        IROp("kv_latent_raw", "tessera.matmul", ["%x", "%kv_down"], [str(x_ty), str(w_d_c_ty)], str(latent_ty)),
        IROp("kv_latent", "tessera.rmsnorm_safe", ["%kv_latent_raw"], [str(latent_ty)], str(latent_ty), kwargs={"eps": cfg.rms_norm_eps}),
        IROp("k_full", "tessera.matmul", ["%kv_latent", "%k_absorb"], [str(latent_ty), str(w_c_d_ty)], str(x_ty)),
        IROp("score_input", "tessera.relu", ["%q_full"], [str(x_ty)], str(x_ty)),
        IROp("scores", "tessera.matmul", ["%score_input", "%score_w"], [str(x_ty), str(w_d_s_ty)], str(scores_ty)),
        IROp("probs", "tessera.softmax", ["%scores"], [str(scores_ty)], str(scores_ty), kwargs={"axis": -1}),
        IROp("v_full", "tessera.matmul", ["%kv_latent", "%v_absorb"], [str(latent_ty), str(w_c_d_ty)], str(x_ty)),
        IROp("context", "tessera.matmul", ["%probs", "%v_full"], [str(scores_ty), str(x_ty)], str(x_ty)),
        IROp("out", "tessera.matmul", ["%context", "%out_w"], [str(x_ty), str(w_d_d_ty)], str(x_ty)),
        IROp("norm_out", "tessera.rmsnorm_safe", ["%out"], [str(x_ty)], str(x_ty), kwargs={"eps": cfg.rms_norm_eps}),
    ]
    fn = GraphIRFunction(
        name="flash_mla_tiny_prefill",
        args=args,
        result_types=[x_ty],
        body=ops,
        return_values=["%norm_out"],
        fn_attrs={
            "tessera.example": '"mla"',
            "tessera.mla.latent_dim": f"{cfg.latent_dim} : i64",
            "tessera.mla.num_q_heads": f"{cfg.num_q_heads} : i64",
            "tessera.mla.num_kv_heads": f"{cfg.num_kv_heads} : i64",
        },
    )
    return GraphIRModule(functions=[fn], module_attrs={
        "tessera.ir.version": '"1.0"',
        "tessera.example": '"mla"',
    })


def compile_toy_graph(*, target: str = "apple_cpu"):
    # The checked-in typed MLIR fixture covers parser validation separately.
    return compile_graph_module(
        build_toy_graph_ir(),
        source_origin="examples/advanced/mla",
        target=target,
        example_id="flash_mla_tiny",
        enable_tool_validation=False,
    )
