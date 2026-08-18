from __future__ import annotations

import numpy as np

import tessera as ts
from tessera.compiler.presburger import (
    PresburgerConstraint,
    PresburgerSystem,
    attach_presburger_system,
)
from tessera.compiler.graph_ir import IROp
from tessera.compiler.residual_evaluator import (
    build_region_residual_abi,
    capture_treeverse_forward,
    execute_treeverse_region_adjoint,
    treeverse_candidates,
)
from tessera.compiler.trace import to_graph_ir_module, trace
from tessera.compiler.structured_cfg import (
    StructuredBlock,
    StructuredCFG,
    StructuredEdge,
    recover_structured_cfg,
)


def _traced_cond_module():
    def program(flag, x, weight):
        return ts.control.cond(
            flag,
            lambda: ts.ops.silu(ts.ops.matmul(x, weight)),
            lambda: ts.ops.relu(ts.ops.matmul(x, weight)),
        )

    traced = trace(
        program,
        np.ones((1,), dtype=np.float32),
        np.ones((1, 4), dtype=np.float32),
        np.eye(4, dtype=np.float32),
    )
    return to_graph_ir_module(traced, name="structured_cond")


def test_tracer_recovers_explicit_multiblock_cfg_without_ast_markers() -> None:
    function = _traced_cond_module().functions[0]
    cfg = function.structured_cfg

    assert cfg is not None
    assert len(cfg.blocks) == 5
    assert {block.terminator for block in cfg.blocks} >= {
        "cond_br", "br", "return", "function_exit"
    }
    assert {edge.kind for edge in cfg.edges} == {
        "true", "false", "yield", "return"
    }
    result_name = function.return_values[0].lstrip("%")
    assert any(block.arguments == (result_name,) for block in cfg.blocks)
    assert function.fn_attrs["tessera.structured_cfg.digest"] == f'"{cfg.digest}"'


def test_tracer_recovers_variadic_branch_state_without_ast_reexecution() -> None:
    def program(flag, x, y):
        return ts.control.cond(
            flag,
            lambda: (ts.ops.tanh(x), ts.ops.relu(y)),
            lambda: (ts.ops.relu(x), ts.ops.tanh(y)),
        )

    traced = trace(
        program,
        np.ones((1,), dtype=np.float32),
        np.ones((4,), dtype=np.float32),
        np.ones((4,), dtype=np.float32),
    )
    function = to_graph_ir_module(traced, name="variadic_cond").functions[0]
    cfg = function.structured_cfg
    merge = next(block for block in cfg.blocks if block.block_id.endswith("if_merge"))
    yields = [edge for edge in cfg.edges if edge.target == merge.block_id]

    assert len(function.return_values) == 2
    assert len(merge.arguments) == 2
    assert len(yields) == 2
    assert all(len(edge.values) == 2 for edge in yields)
    assert function.fn_attrs["tessera.structured_cfg.digest"] == f'"{cfg.digest}"'


def test_cfg_digest_carries_effect_alias_mutation_and_stochastic_identity() -> None:
    stateful = IROp(
        result="next_cache",
        op_name="tessera.kv_cache.append",
        operands=["%cache", "%key", "%value"],
        operand_types=["!tessera.kv_cache", "tensor<4xf32>", "tensor<4xf32>"],
        result_type="!tessera.kv_cache",
        kwargs={"tessera.aliasing": "operand_0"},
    )
    random = IROp(
        result="sample",
        op_name="tessera.rng_philox_uniform",
        operands=["%key"],
        operand_types=["tensor<2xi64>"],
        result_type="tensor<4xf32>",
        kwargs={"tessera.stochastic_identity": "philox:key0:counter0"},
    )

    cfg = recover_structured_cfg((stateful, random))
    operations = [operation for block in cfg.blocks for operation in block.operations]

    assert operations[0].effect == "state"
    assert operations[0].aliasing == "operand_0"
    assert operations[0].mutation_operands == ("%cache", "%key", "%value")
    # Keyed/counter-based Philox is replayable and therefore pure.  Its
    # stochastic identity remains an independent scheduling/AD constraint.
    assert operations[1].effect == "pure"
    assert operations[1].stochastic_identity == "philox:key0:counter0"


def test_source_cfg_analysis_detects_a_true_two_entry_irreducible_scc() -> None:
    block_ids = ("entry", "left", "right", "exit")
    cfg = StructuredCFG(
        entry_block="entry",
        exit_block="exit",
        blocks=tuple(
            StructuredBlock(block_id, (), (), "branch")
            for block_id in block_ids
        ),
        edges=(
            StructuredEdge("entry", "left", "true"),
            StructuredEdge("entry", "right", "false"),
            StructuredEdge("left", "right", "backedge"),
            StructuredEdge("right", "left", "backedge"),
            StructuredEdge("right", "exit", "exit"),
        ),
    )

    analysis = cfg.analyze()

    assert analysis.classification == "irreducible"
    assert analysis.cyclic_components == (("left", "right"),)
    assert analysis.irreducible_components == (("left", "right"),)
    assert analysis.requires_bounded_state_machine


def test_source_cfg_analysis_distinguishes_reducible_and_acyclic_graphs() -> None:
    reducible = StructuredCFG(
        entry_block="entry",
        exit_block="exit",
        blocks=tuple(
            StructuredBlock(block_id, (), (), "branch")
            for block_id in ("entry", "header", "body", "exit")
        ),
        edges=(
            StructuredEdge("entry", "header", "enter"),
            StructuredEdge("header", "body", "body"),
            StructuredEdge("body", "header", "backedge"),
            StructuredEdge("header", "exit", "exit"),
        ),
    )
    acyclic = StructuredCFG(
        entry_block="entry",
        exit_block="exit",
        blocks=(
            StructuredBlock("entry", (), (), "branch"),
            StructuredBlock("exit", (), (), "return"),
        ),
        edges=(StructuredEdge("entry", "exit", "return"),),
    )

    assert reducible.analyze().classification == "reducible"
    assert reducible.analyze().requires_bounded_state_machine
    assert acyclic.analyze().classification == "acyclic"
    assert not acyclic.analyze().requires_bounded_state_machine


def test_typed_presburger_system_propagates_through_every_region_block() -> None:
    function = _traced_cond_module().functions[0]
    system = PresburgerSystem(
        symbols=("N",),
        constraints=(PresburgerConstraint("mod", (1,), modulus=4),),
    )

    before = function.structured_cfg.digest
    attach_presburger_system(function, system)
    cfg = function.structured_cfg

    assert cfg.digest != before
    assert cfg.presburger_system == system
    assert all(block.presburger_digest == system.digest for block in cfg.blocks)
    assert function.fn_attrs["tessera.structured_cfg.digest"] == f'"{cfg.digest}"'


def test_save_and_hybrid_residual_abis_execute_against_cfg_identity() -> None:
    cfg_digest = _traced_cond_module().functions[0].structured_cfg.digest
    hybrid, save = treeverse_candidates(
        steps=4,
        state_bytes=16,
        measured_step_work_ns=10,
        memory_budgets=(16, 64),
    )
    hybrid_abi = build_region_residual_abi(
        hybrid,
        cfg_digest=cfg_digest,
        state_dtype="f32",
        state_shape=(4,),
    )
    save_abi = build_region_residual_abi(
        save,
        cfg_digest=cfg_digest,
        state_dtype="f32",
        state_shape=(4,),
    )

    assert hybrid_abi.policy == "hybrid"
    assert save_abi.policy == "save"
    assert hybrid_abi.digest != save_abi.digest
    assert hybrid_abi.compiler_attributes() == {
        "tessera.autodiff.checkpoint_policy": "hybrid",
        "tessera.autodiff.checkpoint_indices": list(
            hybrid_abi.checkpoint_indices
        ),
        "tessera.autodiff.residual_schema": "tessera.region_residual_abi.v1",
        "tessera.autodiff.residual_digest": hybrid_abi.digest,
        "tessera.structured_cfg.digest": cfg_digest,
        "tessera.autodiff.residual_state_dtype": "f32",
        "tessera.autodiff.residual_state_shape": [4],
    }

    initial = np.arange(1, 5, dtype=np.float32)

    def forward_step(index, state):
        return state * np.float32(index + 2)

    def backward_step(index, before, after, cotangent):
        del before, after
        return cotangent * np.float32(index + 2)

    capture = capture_treeverse_forward(
        candidate=hybrid,
        initial_state=initial,
        forward_step=forward_step,
        residual_abi=hybrid_abi,
    )
    execution = execute_treeverse_region_adjoint(
        cotangent=np.ones_like(initial),
        residuals=capture.residuals,
        forward_step=forward_step,
        backward_step=backward_step,
    )

    np.testing.assert_array_equal(
        execution.input_cotangent,
        np.full_like(initial, 2 * 3 * 4 * 5),
    )
    assert execution.checkpoint_indices == hybrid_abi.checkpoint_indices
