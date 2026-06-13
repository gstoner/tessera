"""P1 — DFlash target-model hidden-state tap.

Validates tessera.dflash.HiddenStateTap / capture_target_hidden: the multi-layer
feature injection that conditions the DFlash draft (model_mlx _patch_model).
"""
import numpy as np
import pytest

from tessera import dflash as D
from tessera import nn


class _Layer(nn.Module):
    """Deterministic transform layer: x -> x * scale + scale."""

    def __init__(self, scale: float):
        super().__init__()
        self._scale = float(scale)

    def forward(self, x):
        return np.asarray(x) * self._scale + self._scale


class _Target(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(0.5 + 0.3 * i) for i in range(n_layers)])

    def forward(self, x):
        h = np.asarray(x)
        outs = []
        for layer in self.layers:
            h = layer(h)
            outs.append(h)
        return h, outs


def test_tap_captures_and_concats_selected_layers():
    rng = np.random.default_rng(0)
    B, S, Dm = 2, 4, 8
    x = rng.standard_normal((B, S, Dm)).astype(np.float32)
    target = _Target(5)
    layer_ids = [0, 2, 4]

    with D.capture_target_hidden(target.layers, layer_ids) as tap:
        _, ref_outs = target(x)
        got = tap.hidden_states

    expected = np.concatenate([ref_outs[i] for i in layer_ids], axis=-1)
    assert got.shape == (B, S, len(layer_ids) * Dm)
    assert np.allclose(got, expected)


def test_tap_restores_forward_after_exit():
    target = _Target(3)
    x = np.ones((1, 2, 4), np.float32)
    before, _ = target(x)
    with D.capture_target_hidden(target.layers, [0, 1]):
        target(x)
    after, _ = target(x)
    # forward must behave identically once the tap is removed
    assert np.allclose(before, after)
    # and the wrapped layers no longer carry an instance 'forward' attribute
    for i in (0, 1):
        assert "forward" not in target.layers[i].__dict__


def test_tap_reset_and_reuse_across_steps():
    rng = np.random.default_rng(1)
    x1 = rng.standard_normal((1, 3, 4)).astype(np.float32)
    x2 = rng.standard_normal((1, 3, 4)).astype(np.float32)
    target = _Target(4)
    tap = D.capture_target_hidden(target.layers, [1, 3]).install()
    try:
        target(x1)
        h1 = tap.hidden_states.copy()
        tap.reset()
        target(x2)
        h2 = tap.hidden_states
        assert not np.allclose(h1, h2)        # fresh capture per step
    finally:
        tap.remove()


def test_tap_feeds_target_feature_projection():
    """The tapped hidden states are exactly the (B, S, nL*D) shape the draft's
    fc projection consumes."""
    rng = np.random.default_rng(2)
    B, S, Dm, nL = 2, 5, 8, 3
    x = rng.standard_normal((B, S, Dm)).astype(np.float32)
    target = _Target(4)
    fc = rng.standard_normal((nL * Dm, Dm)).astype(np.float32) * 0.1
    hidden_norm = rng.standard_normal(Dm).astype(np.float32) * 0.1 + 1.0

    with D.capture_target_hidden(target.layers, [0, 1, 2]) as tap:
        target(x)
        th = tap.hidden_states
    x_ctx = D.target_feature_projection(th, fc, hidden_norm)
    assert th.shape == (B, S, nL * Dm)
    assert np.asarray(x_ctx).shape == (B, S, Dm)


def test_tap_missing_capture_raises():
    target = _Target(3)
    tap = D.capture_target_hidden(target.layers, [0, 2]).install()
    try:
        with pytest.raises(RuntimeError, match="has not captured"):
            _ = tap.hidden_states
    finally:
        tap.remove()
