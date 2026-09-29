from pathlib import Path

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera._jit_boundary import TesseraJitError


def _residual(theta, x):
    return x * x * x - theta


def test_tensor_arithmetic_traces_canonical_residual_and_broadcast():
    x = np.linspace(0.25, 1.25, 17, dtype=np.float32)
    theta = np.array([0.5], dtype=np.float32)
    traced = trace(_residual, theta, x)
    assert [op.op_name for op in traced.body] == ["tessera.mul", "tessera.mul", "tessera.sub"]
    module = to_graph_ir_module(traced, name="residual", source_hash="test", target="x86")
    assert module.verify().ok
    np.testing.assert_allclose(traced.output_values[0], x * x * x - theta)
    addition = trace(lambda a, b: a + b, theta, x)
    assert addition.body[0].op_name == "tessera.add"
    np.testing.assert_allclose(addition.output_values[0], theta + x)


def test_persistent_tape_uses_tracer_graph_without_ast_fallback(monkeypatch):
    function = ts.jit(autodiff="reverse")(_residual)
    captured = {}

    def materialize(source, **kwargs):
        captured["source"] = source
        return object()

    def forbidden(*args, **kwargs):
        raise AssertionError("persistent capture entered the AST compatibility path")

    monkeypatch.setattr("tessera.compiler.native_persistent_tape.materialize_persistent_tape", materialize)
    monkeypatch.setattr(function, "_specialized_autodiff_module", forbidden)
    values = np.ones(17, np.float32)
    function.compile_persistent_device_tape(values, values, compiler=Path("unused"),
        llvm_bin=Path("unused"), backend="nvidia", chip="sm_120")
    assert 'tessera.frontend.authority = "tracer"' in captured["source"]
    assert captured["source"].count('"tessera.mul"') == 2


def test_persistent_tape_refuses_untraceable_source(monkeypatch):
    @ts.jit(autodiff="reverse")
    def unsupported(x):
        return x * 2

    def forbidden(*args, **kwargs):
        raise AssertionError("untraceable source reached the native packager")

    monkeypatch.setattr("tessera.compiler.native_persistent_tape.materialize_persistent_tape", forbidden)
    with pytest.raises(TesseraJitError, match="tracer frontend failed"):
        unsupported.compile_persistent_device_tape(np.ones(4, np.float32),
            compiler=Path("unused"), llvm_bin=Path("unused"), backend="nvidia", chip="sm_120")
