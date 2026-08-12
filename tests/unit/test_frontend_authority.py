from __future__ import annotations

import numpy as np

import tessera


@tessera.jit
def _straight_line(
    x: tessera.Tensor["4"], y: tessera.Tensor["4"]
) -> tessera.Tensor["4"]:
    return tessera.ops.relu(tessera.ops.add(x, y))


def test_concrete_straight_line_jit_establishes_cached_tracer_authority():
    x = np.arange(4, dtype=np.float32)
    y = np.ones(4, dtype=np.float32)

    np.testing.assert_allclose(_straight_line(x, y), np.maximum(x + y, 0.0))
    assert _straight_line.frontend_authority == "tracer"
    assert len(_straight_line._traced_frontend_specializations) == 1

    # The explicit gate compares both structure and concrete values and binds
    # the evidence to a content digest.
    certificate = _straight_line.frontend_differential(x, y)
    certificate.validate()
    assert certificate.contract["structural_match"]
    assert certificate.contract["numerical_match"]


def test_tracer_authority_cache_avoids_retracing_a_known_signature(monkeypatch):
    x = np.arange(4, dtype=np.float32)
    y = np.ones(4, dtype=np.float32)
    _straight_line(x, y)

    def fail_trace(*_args, **_kwargs):
        raise AssertionError("cached signatures must not execute the tracer again")

    monkeypatch.setattr("tessera.compiler.trace.trace", fail_trace)
    np.testing.assert_allclose(_straight_line(x, y), np.maximum(x + y, 0.0))
