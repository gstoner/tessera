from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff.vjp import get_vjp


def _directional_difference(fn, value, direction, eps=1e-5):
    return (fn(value + eps * direction) - fn(value - eps * direction)) / (2 * eps)


@pytest.mark.parametrize(
    ("name", "erase"),
    [("gated_deltanet", False), ("gated_deltanet", True),
     ("modified_delta_attention", True)],
)
def test_delta_mixer_analytic_vjp_matches_directional_difference(name, erase):
    rng = np.random.default_rng(41)
    shape = (1, 1, 3, 2)
    q = rng.standard_normal(shape)
    k = rng.standard_normal(shape)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal(shape)
    gate = rng.standard_normal(shape)
    beta = rng.uniform(0.2, 0.8, shape[:-1])
    decay = rng.uniform(0.7, 0.95, shape[:-1])
    dout = rng.standard_normal(shape)
    op = getattr(ts.ops, name)
    grads = get_vjp(name)(
        dout, q, k, v, gate, beta, decay, erase=erase
    )
    values = [q, k, v, gate, beta, decay]
    for index, (value, gradient) in enumerate(zip(values, grads)):
        direction = rng.standard_normal(value.shape)

        def objective(candidate):
            operands = list(values)
            operands[index] = candidate
            return float(
                np.sum(
                    np.asarray(op(*operands, erase=erase), dtype=np.float64)
                    * dout
                )
            )

        numeric = _directional_difference(objective, value, direction)
        analytic = float(np.sum(np.asarray(gradient) * direction))
        np.testing.assert_allclose(analytic, numeric, rtol=3e-4, atol=3e-5)


def test_sequence_mixer_schedule_carries_forward_and_backward_state_contract():
    @ts.jit
    def mixer(q, k, v):
        return ts.ops.gated_deltanet(
            q, k, v, causal=True, erase=True
        )

    text = mixer.schedule_ir
    assert 'schedule = "linear_recurrence"' in text
    assert 'carried_state = "B,H,Dk,Dv@fp32"' in text
    assert 'backward_form = "reverse_recurrence"' in text
    assert "chunk_size = 64" in text
