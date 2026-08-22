"""Exact-device compiler-JVP proof for the SM120 Philox dropout package."""

from __future__ import annotations

import numpy as np
import pytest

import tessera


@tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("x",))
def _seeded_dropout(x):
    return tessera.ops.dropout(
        x, p=0.25, training=True, seed=0x123456789ABCDEF,
    )


def test_compiler_jvp_replays_philox_mask_on_primal_and_tangent():
    from tessera import runtime

    if runtime._load_nvidia_rng_runtime() is None:
        pytest.skip("NVIDIA Philox runtime is unavailable")
    x = np.linspace(-2.0, 3.0, 257, dtype=np.float32)
    dx = np.linspace(4.0, -1.0, 257, dtype=np.float32)
    primal, tangent = _seeded_dropout.native_jvp(x, tangents=dx)
    scale = np.float32(1.0 / 0.75)
    primal_mask = np.asarray(primal) != 0.0
    tangent_mask = np.asarray(tangent) != 0.0
    np.testing.assert_array_equal(tangent_mask, primal_mask)
    np.testing.assert_array_equal(
        np.asarray(primal), np.where(primal_mask, x * scale, 0.0).astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(tangent), np.where(primal_mask, dx * scale, 0.0).astype(np.float32)
    )


def test_unseeded_training_dropout_jvp_fails_closed():
    from tessera.compiler.graph_ir import IROp
    from tessera.compiler.native_jvp_plugins import plan_native_jvp_family

    source = IROp(
        result="o", op_name="tessera.dropout", operands=["%x"],
        operand_types=["tensor<8xf32>"], result_type="tensor<8xf32>",
        kwargs={"p": 0.25, "training": True},
    )
    with pytest.raises(ValueError, match="explicit key/seed"):
        plan_native_jvp_family(
            source=source, primal_inputs=(np.ones(8, np.float32),),
            wrt_indices=(0,), target="nvidia_sm120", architecture="sm120",
            execution_mode="cuda_runtime",
        )
