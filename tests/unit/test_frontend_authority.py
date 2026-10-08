from __future__ import annotations

import numpy as np
import pytest

import tessera


class Static4F32:
    shape = (4,)
    dtype = "fp32"


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


def test_static_annotations_use_symbolic_tracing_before_ast_extraction(
    monkeypatch,
) -> None:
    from tessera.compiler import graph_ir as graph_ir_module
    from tessera.compiler import graph_ir_cache

    graph_ir_cache.clear_graph_ir_cache()

    original_extract = graph_ir_module.GraphIRBuilder._extract_ops
    extraction_count = 0

    def count_effect_analysis_only(*args, **kwargs):
        nonlocal extraction_count
        extraction_count += 1
        if extraction_count > 1:
            raise AssertionError("JIT emission must not enter _OpExtractor")
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(
        graph_ir_module.GraphIRBuilder,
        "_extract_ops",
        count_effect_analysis_only,
    )

    @tessera.jit
    def annotated(x: Static4F32, y: Static4F32):
        return tessera.ops.add(x, y)

    assert annotated.graph_ir.module_attrs["tessera.frontend.authority"] == '"tracer"'
    assert annotated._legacy_graph_ir is None
    assert annotated.arg_names == ["x", "y"]
    assert extraction_count == 1

    # The compatibility AST is delayed until an explicit differential proof
    # asks for it; ordinary compilation never pays for or trusts that capture.
    monkeypatch.setattr(
        graph_ir_module.GraphIRBuilder,
        "_extract_ops",
        original_extract,
    )
    certificate = annotated.frontend_differential(
        np.ones(4, np.float32), np.ones(4, np.float32)
    )
    certificate.validate()
    assert annotated._legacy_graph_ir is not None

@pytest.mark.parametrize("field", ["rtol", "atol"])
def test_differential_cache_retains_requested_numeric_policy(monkeypatch, field):
    monkeypatch.setattr(_straight_line, "_frontend_differential_certificates", {})
    x = np.arange(4, dtype=np.float32)
    y = np.ones(4, dtype=np.float32)
    first = _straight_line.frontend_differential(x, y, rtol=1e-3, atol=1e-3)
    policy = {"rtol": 1e-3, "atol": 1e-3}
    policy[field] = 1e-7
    stricter = _straight_line.frontend_differential(x, y, **policy)
    assert stricter.contract[field] == policy[field]
    assert stricter.digest != first.digest
    assert _straight_line.frontend_differential(x, y, **policy) is stricter


def test_strict_differential_cannot_reuse_loose_numerical_proof(monkeypatch):
    from tessera.compiler import frontend_authority
    monkeypatch.setattr(_straight_line, "_frontend_differential_certificates", {})
    certify = frontend_authority.certify_frontends
    def with_rounding_discrepancy(**kwargs):
        kwargs["tracer_outputs"] = tuple(
            value + np.float32(1e-4) for value in kwargs["tracer_outputs"])
        return certify(**kwargs)
    monkeypatch.setattr(frontend_authority, "certify_frontends", with_rounding_discrepancy)
    x = np.ones(4, dtype=np.float32)
    y = np.ones(4, dtype=np.float32)
    loose = _straight_line.frontend_differential(x, y, rtol=0, atol=1e-3)
    assert loose.contract["numerical_match"]
    from tessera._jit_boundary import TesseraJitError
    with pytest.raises(TesseraJitError, match="lacks differential parity") as exc:
        _straight_line.frontend_differential(x, y, rtol=0, atol=1e-7)
    assert isinstance(exc.value.__cause__, ValueError)
    assert _straight_line.frontend_differential(x, y, rtol=0, atol=1e-3) is loose
