"""Compatibility coverage for formerly phantom advanced-example APIs."""

from __future__ import annotations

import numpy as np

import tessera as ts


def test_dynamic_depthwise_conv1d_is_real_module():
    layer = ts.nn.DynamicDepthwiseConv1d(2, 3, streaming=True)
    x = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

    y = layer(x)

    assert y.shape == (1, 2, 6)
    assert hasattr(layer, "reset_state")


def test_top_level_autocast_and_checkpoint_compatibility():
    assert ts.autocast is ts.autodiff.autocast
    assert callable(ts.checkpoint)
    assert hasattr(ts.checkpoint, "save_state")

    wrapped = ts.checkpoint(lambda x: x + 1.0)
    np.testing.assert_allclose(wrapped(np.array([1.0], dtype=np.float32)), [2.0])


def test_stateful_optimizer_namespace_adamw_updates_parameters():
    layer = ts.nn.Linear(2, 1)
    params = list(layer.parameters())
    before = params[0].numpy().copy()
    for param in params:
        param.grad = np.ones_like(param.numpy(), dtype=np.float32)

    opt = ts.optimizers.AdamW(params, lr=1e-2)
    opt.step()

    assert isinstance(opt.state, dict)
    assert not np.allclose(params[0].numpy(), before)


def test_distributions_namespace_normal_and_beta_are_available():
    normal = ts.distributions.Normal(0.0, 1.0)
    beta = ts.distributions.Beta(2.0, 3.0)

    assert normal.sample((4,), seed=0).shape == (4,)
    assert beta.sample((4,), seed=0).shape == (4,)
    np.testing.assert_allclose(normal.log_prob(0.0), -0.5 * np.log(2.0 * np.pi))


def test_top_level_tensor_op_aliases_match_ops_namespace():
    assert ts.arange is ts.ops.arange
    assert ts.gather is ts.ops.gather
    assert ts.clip is ts.ops.clip
    assert ts.einsum is ts.ops.einsum
    assert ts.masked_fill is ts.ops.masked_fill


def test_low_precision_dtype_annotation_shorthands_exist():
    assert ts.fp8_e4m3["B", "D"].dtype == "fp8_e4m3"
    assert ts.fp8_e5m2["B", "D"].dtype == "fp8_e5m2"
    assert ts.fp6["B", "D"].dtype == "fp6_e3m2"
    assert ts.fp4["B", "D"].dtype == "fp4_e2m1"
    assert ts.nvfp4["B", "D"].dtype == "nvfp4"
