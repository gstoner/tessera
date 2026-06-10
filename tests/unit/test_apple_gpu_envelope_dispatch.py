"""P1 (2026-06-09) — descriptor-driven dispatch drift gates.

The Apple GPU envelope tables are single-sourced in
``tessera.compiler.apple_gpu_envelope``; ``driver.py`` (compile gating),
``runtime.py`` (lane dispatch), and ``apple_kernel_descriptor`` (declarative
registry) import them. These tests are the oracle against the legacy paths:

* every runtime op resolves a lane, and every lane has a runtime handler;
* the lane assignment reproduces the legacy elif-chain routing exactly
  (hard-pinned for representative ops of every lane);
* driver/runtime re-export the identical objects (no second truth);
* descriptors carry the lane consistent with the envelope.
"""

from __future__ import annotations

import pytest

from tessera.compiler import apple_gpu_envelope as env
from tessera.compiler import driver
from tessera import runtime as rt


def test_every_runtime_op_has_a_lane_and_a_handler():
    handlers = rt._apple_gpu_lane_handlers()
    for op in sorted(env._APPLE_GPU_RUNTIME_OPS):
        lane = env.APPLE_GPU_LANE_BY_OP.get(op)
        assert lane is not None, f"{op} has no lane"
        assert lane in handlers, f"lane {lane!r} ({op}) has no runtime handler"
    # And no lane exists without a handler nor handler without a lane.
    assert set(handlers) == set(env.APPLE_GPU_LANES)


def test_lane_table_covers_exactly_the_envelope():
    assert frozenset(env.APPLE_GPU_LANE_BY_OP) == env._APPLE_GPU_RUNTIME_OPS


# Hard-pinned oracle: one or more representative ops per legacy elif branch,
# in the historical routing. Editing the envelope must keep these stable.
_LEGACY_ROUTING = {
    "tessera.matmul": "mps",
    "tessera.gemm": "mps",
    "tessera.batched_gemm": "mps",
    "tessera.rope": "rope",
    "tessera.flash_attn": "flash_attn",
    "tessera.softmax": "softmax",
    "tessera.softmax_safe": "softmax",
    "tessera.gelu": "gelu",
    "tessera.relu": "unary",
    "tessera.sigmoid_safe": "unary",
    "tessera.add": "binary",
    "tessera.ge": "binary",
    "tessera.clamp": "clamp",
    "tessera.clip": "clamp",
    "tessera.where": "where",
    "tessera.loss.mse": "loss_compose",
    "tessera.group_norm": "norm_compose",
    "tessera.multi_head_attention": "attn_wrapper",
    "tessera.lightning_attention": "linear_attn",
    "tessera.attn_sliding_window": "masked_attn",
    "tessera.kimi_delta_attention": "delta_attn",
    "tessera.hybrid_attention": "hybrid_attn",
    "tessera.deepseek_sparse_attention": "sparse_attn",
    "tessera.layer_norm": "rowop",
    "tessera.rmsnorm_safe": "rowop",
    "tessera.silu_mul": "silu_mul",
    "tessera.linear_general": "linear_general",
    "tessera.qkv_projection": "qkv_projection",
    "tessera.mean": "reduce",
    "tessera.cummax": "reduce",
    "tessera.conv2d": "conv2d",
    "tessera.conv3d": "conv3d",
    "tessera.cholesky": "linalg",
    "tessera.tri_solve": "linalg",
    "tessera.selective_ssm": "ssm",
    "tessera.moe_swiglu_block": "moe_swiglu_block",
    "tessera.grouped_gemm": "grouped_gemm",
    "tessera.popcount": "popcount",
    "tessera.count_nonzero": "count_nonzero",
    "tessera.loss.z_loss": "z_loss",
    "tessera.loss.asymmetric_bce": "asymmetric_bce",
    "tessera.loss.load_balance_loss": "load_balance_loss",
    "tessera.masked_categorical": "masked_categorical",
    "tessera.clifford_geometric_product": "clifford",
    "tessera.ebm_energy_quadratic": "ebm",
    "tessera.loss.score_matching": "ebm_loss",
}


@pytest.mark.parametrize("op,lane", sorted(_LEGACY_ROUTING.items()))
def test_lane_matches_legacy_elif_routing(op, lane):
    assert env.APPLE_GPU_LANE_BY_OP[op] == lane


def test_driver_and_runtime_reexport_identical_objects():
    for name in (
        "_APPLE_GPU_MPS_OPS", "_APPLE_GPU_MSL_OPS", "_APPLE_GPU_MPSGRAPH_OPS",
        "_APPLE_GPU_PROJECTION_OPS", "_APPLE_GPU_REDUCTION_OPS",
        "_APPLE_GPU_CONV_OPS", "_APPLE_GPU_LINALG_OPS", "_APPLE_GPU_SSM_OPS",
        "_APPLE_GPU_MOE_OPS", "_APPLE_GPU_LDT_OPS", "_APPLE_GPU_CLIFFORD_OPS",
        "_APPLE_GPU_EBM_OPS", "_APPLE_GPU_EBM_LOSS_OPS",
        "_APPLE_GPU_LOSS_COMPOSE_OPS", "_APPLE_GPU_NORM_COMPOSE_OPS",
        "_APPLE_GPU_ATTN_WRAPPER_OPS", "_APPLE_GPU_LINEAR_ATTN_OPS",
        "_APPLE_GPU_MASKED_ATTN_OPS", "_APPLE_GPU_DELTA_ATTN_OPS",
        "_APPLE_GPU_HYBRID_ATTN_OPS", "_APPLE_GPU_SPARSE_ATTN_OPS",
        "_APPLE_GPU_RUNTIME_OPS",
    ):
        assert getattr(driver, name) is getattr(env, name), name
        assert getattr(rt, name) is getattr(env, name), name


def test_lane_for_accepts_bare_and_dotted_names():
    assert env.lane_for("matmul") == "mps"
    assert env.lane_for("tessera.matmul") == "mps"
    assert env.lane_for("gather") is None


def test_descriptors_carry_envelope_lane():
    from tessera.compiler.apple_kernel_descriptor import (
        all_apple_kernel_descriptors,
    )
    for name, desc in all_apple_kernel_descriptors().items():
        expected = env.lane_for(name)
        assert desc.lane == expected, (
            f"descriptor {name!r} lane {desc.lane!r} != envelope {expected!r}")
        # Envelope membership and lane presence agree.
        assert (expected is not None) == (f"tessera.{name}" in env._APPLE_GPU_RUNTIME_OPS)
