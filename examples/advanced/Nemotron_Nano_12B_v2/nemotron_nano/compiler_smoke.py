"""Compiler-facing Graph IR for the Nemotron Nano sample.

The current lightweight compiler supports straight-line tensor op graphs.  This
module maps the tiny Nemotron M/*/- pattern to that supported subset so the
sample exercises Graph IR, Schedule IR, Tile IR, and Apple Target IR today.
"""

from __future__ import annotations

from tessera.compiler import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type

from .config import NemotronNanoConfig, tiny_config


def build_toy_graph_ir(cfg: NemotronNanoConfig | None = None) -> GraphIRModule:
    cfg = cfg or tiny_config()
    h = cfg.hidden_size
    ff = cfg.intermediate_size
    batch_tokens = 32
    x_ty = tensor_ir_type((batch_tokens, h), "f32")
    ff_ty = tensor_ir_type((batch_tokens, ff), "f32")
    out_ty = tensor_ir_type((batch_tokens, h), "f32")
    w_h_ff_ty = tensor_ir_type((h, ff), "f32")
    w_ff_h_ty = tensor_ir_type((ff, h), "f32")
    w_h_h_ty = tensor_ir_type((h, h), "f32")

    args = [
        IRArg("x", x_ty, layout="row_major"),
        IRArg("m_in", w_h_ff_ty, layout="row_major"),
        IRArg("m_out", w_ff_h_ty, layout="row_major"),
        IRArg("attn_q", w_h_h_ty, layout="row_major"),
        IRArg("attn_o", w_h_h_ty, layout="row_major"),
        IRArg("mlp_in", w_h_ff_ty, layout="row_major"),
        IRArg("mlp_out", w_ff_h_ty, layout="row_major"),
    ]
    ops = [
        IROp("m_proj", "tessera.matmul", ["%x", "%m_in"], [str(x_ty), str(w_h_ff_ty)], str(ff_ty)),
        IROp("m_act", "tessera.relu", ["%m_proj"], [str(ff_ty)], str(ff_ty)),
        IROp("m_outv", "tessera.matmul", ["%m_act", "%m_out"], [str(ff_ty), str(w_ff_h_ty)], str(out_ty)),
        IROp("q_proj", "tessera.matmul", ["%m_outv", "%attn_q"], [str(out_ty), str(w_h_h_ty)], str(out_ty)),
        IROp("attn_prob", "tessera.softmax", ["%q_proj"], [str(out_ty)], str(out_ty), kwargs={"axis": -1}),
        IROp("attn_out", "tessera.matmul", ["%attn_prob", "%attn_o"], [str(out_ty), str(w_h_h_ty)], str(out_ty)),
        IROp("mlp_proj", "tessera.matmul", ["%attn_out", "%mlp_in"], [str(out_ty), str(w_h_ff_ty)], str(ff_ty)),
        IROp("mlp_act", "tessera.relu", ["%mlp_proj"], [str(ff_ty)], str(ff_ty)),
        IROp("mlp_outv", "tessera.matmul", ["%mlp_act", "%mlp_out"], [str(ff_ty), str(w_ff_h_ty)], str(out_ty)),
        IROp("norm", "tessera.rmsnorm_safe", ["%mlp_outv"], [str(out_ty)], str(out_ty), kwargs={"eps": cfg.rms_norm_eps}),
    ]
    fn = GraphIRFunction(
        name="nemotron_nano_tiny_m_star_dash",
        args=args,
        result_types=[out_ty],
        body=ops,
        return_values=["%norm"],
        fn_attrs={
            "tessera.example": '"Nemotron_Nano_12B_v2"',
            "tessera.hybrid_pattern": f'"{cfg.hybrid_override_pattern}"',
        },
    )
    return GraphIRModule(functions=[fn], module_attrs={
        "tessera.ir.version": '"1.0"',
        "tessera.example": '"Nemotron_Nano_12B_v2"',
    })


def compile_toy_graph(*, target: str = "apple_cpu"):
    # The smoke validates the Python compiler artifact path.  The checked-in
    # typed MLIR fixture covers tessera-opt parser validation separately.
    return compile_graph_module(
        build_toy_graph_ir(),
        source_origin="examples/advanced/Nemotron_Nano_12B_v2",
        target=target,
        example_id="nemotron_nano_12b_v2_tiny",
        enable_tool_validation=False,
    )
