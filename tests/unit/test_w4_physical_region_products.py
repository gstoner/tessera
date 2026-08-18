from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tessera.compiler.structured_region_product import (
    build_dynamic_if_product,
    build_hybrid_while_product,
    execute_dynamic_if_product,
    execute_hybrid_while_product,
)


CFG_DIGEST = hashlib.sha256(b"w4-physical-region-cfg").hexdigest()
PAIRED_IR = "paired<scf.if,scf.while,residual-ssa>"


def test_w4_region_product_contract_is_content_addressed_and_fail_closed() -> None:
    product = build_hybrid_while_product(
        target="x86",
        paired_ir=PAIRED_IR,
        cfg_digest=CFG_DIGEST,
        state_shape_bound=(32,),
        steps=5,
        checkpoint_indices=(2, 4),
    )
    assert product.contract()["graph_reentry"] is False
    assert product.contract()["predicate_replay"] is False
    assert product.residual_abi is not None
    assert product.residual_abi.policy == "hybrid"
    assert product.residual_abi.checkpoint_indices == (2, 4)
    assert len(product.artifact_digest) == 64
    with pytest.raises(ValueError, match="x86 and gfx1151 only"):
        build_dynamic_if_product(
            target="cuda", paired_ir=PAIRED_IR, cfg_digest=CFG_DIGEST,
            state_shape_bound=(32,),
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_dynamic_branch_product_consumes_selected_saved_activation(target: str) -> None:
    from tessera import runtime

    if target == "x86" and not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and runtime._rocm_chip() != "gfx1151":
        pytest.skip("exact gfx1151 device unavailable")
    product = build_dynamic_if_product(
        target=target, paired_ir=PAIRED_IR, cfg_digest=CFG_DIGEST,
        state_shape_bound=(64,),
    )
    x = np.linspace(-1.0, 1.0, 37, dtype=np.float32)
    ct = np.linspace(0.25, 1.25, 37, dtype=np.float32)
    inactive = np.empty((0,), dtype=np.float32)
    tanh = np.tanh(x).astype(np.float32)
    actual = execute_dynamic_if_product(
        product, predicate=True, saved_true=tanh, saved_false=inactive,
        cotangent=ct,
    )
    np.testing.assert_allclose(actual, ct * 2.0 * tanh * (1.0 - tanh * tanh),
                               rtol=3e-5, atol=3e-5)
    sigmoid = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    actual = execute_dynamic_if_product(
        product, predicate=False, saved_true=inactive, saved_false=sigmoid,
        cotangent=ct,
    )
    np.testing.assert_allclose(actual, ct * 2.0 * sigmoid * sigmoid * (1.0 - sigmoid),
                               rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_hybrid_while_product_consumes_sparse_tape_on_native_target(target: str) -> None:
    from tessera import runtime

    if target == "x86" and not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    if target == "rocm_gfx1151" and runtime._rocm_chip() != "gfx1151":
        pytest.skip("exact gfx1151 device unavailable")
    product = build_hybrid_while_product(
        target=target, paired_ir=PAIRED_IR, cfg_digest=CFG_DIGEST,
        state_shape_bound=(64,), steps=5, checkpoint_indices=(2, 4),
    )
    initial = np.linspace(0.5, 1.5, 37, dtype=np.float32)
    cotangent = np.linspace(-1.0, 1.0, 37, dtype=np.float32)
    scales = (1.25, 0.5, 1.5, 0.75, 2.0)
    execution = execute_hybrid_while_product(
        product, initial_state=initial, scales=scales, cotangent=cotangent
    )
    np.testing.assert_allclose(
        execution.input_cotangent,
        cotangent * np.prod(np.asarray(scales, dtype=np.float32)),
        rtol=2e-6,
        atol=2e-6,
    )
    assert execution.checkpoint_indices == (2, 4)
    assert execution.replayed_steps == 2
    assert execution.backward_steps == 5
