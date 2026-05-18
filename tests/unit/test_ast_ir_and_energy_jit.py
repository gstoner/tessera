"""M6 Step 1 + Step 2 tests — shared AST→IR core + energy_jit.

Step 1 locks the contract that ``@clifford_jit`` and ``@energy_jit``
now share a single :mod:`tessera.compiler.ast_ir` core: the same
lowerer, the same SSA-form `%tN` refs, the same inline literal
encoding, the same source-span error messages.

Step 2 locks the energy-function whitelist + decorator surface.
"""

from __future__ import annotations

import pytest

from tessera.compiler import ast_ir
from tessera.compiler.energy_jit import (
    EnergyCompiledCallable,
    EnergyIRProgram,
    EnergyJitError,
    energy_jit,
    lower_energy_function,
)


# ---------------------------------------------------------------------------
# Shared AST→IR core (M6 Step 1)
# ---------------------------------------------------------------------------

def test_lowering_config_is_frontend_agnostic() -> None:
    """The config is the only thing that differs per-frontend."""
    cfg = ast_ir.LoweringConfig(
        namespace="ns",
        attr_to_op_name={"foo": "ns_foo"},
    )
    assert cfg.namespace == "ns"
    assert cfg.error_class is ast_ir.ASTLoweringError


def test_ast_ir_lower_function_smoke() -> None:
    """A minimal program lowers through the shared core without any
    Clifford / energy knowledge."""
    cfg = ast_ir.LoweringConfig(
        namespace="ns",
        attr_to_op_name={"foo": "ns_foo", "bar": "ns_bar"},
    )

    def prog(a, b):
        x = ns.foo(a, b)        # noqa: F821 — only the AST is read
        return ns.bar(x)        # noqa: F821

    ir = ast_ir.lower_function(prog, cfg)
    assert isinstance(ir, ast_ir.IRProgram)
    assert ir.namespace == "ns"
    assert ir.arg_names == ("a", "b")
    assert [op.op_name for op in ir.ops] == ["ns_foo", "ns_bar"]
    assert ir.ops[1].operand_refs == (ir.ops[0].result_name,)


def test_ast_ir_text_uses_namespace_prefix() -> None:
    cfg = ast_ir.LoweringConfig(
        namespace="demo", attr_to_op_name={"foo": "demo_foo"},
    )

    def p(a):
        return demo.foo(a)      # noqa: F821

    text = ast_ir.lower_function(p, cfg).text()
    assert text.startswith("demo_ir(")
    assert "# demo.foo" in text


def test_ast_ir_inline_literals_are_shared() -> None:
    """The int / float / bool literal encoding is one of the
    properties that needs to stay identical across frontends."""
    cfg = ast_ir.LoweringConfig(
        namespace="ns", attr_to_op_name={"foo": "ns_foo"},
    )

    def p(a):
        return ns.foo(a, 2, -3, 0.5, True)   # noqa: F821

    ir = ast_ir.lower_function(p, cfg)
    op = ir.ops[0]
    assert op.operand_refs == ("a", "#int:2", "#int:-3", "#float:0.5", "#bool:1")


def test_ast_ir_rejects_non_namespaced_call() -> None:
    cfg = ast_ir.LoweringConfig(
        namespace="ns", attr_to_op_name={"foo": "ns_foo"},
    )

    def p(a):
        return foo.bar(a)       # noqa: F821 — wrong namespace

    with pytest.raises(ast_ir.ASTLoweringError, match="only ``tessera.ns"):
        ast_ir.lower_function(p, cfg)


def test_ast_ir_resolve_operand_round_trips_literals() -> None:
    env = {"a": 7, "b": 8}
    assert ast_ir.resolve_operand("#int:3", env) == 3
    assert ast_ir.resolve_operand("#int:-5", env) == -5
    assert ast_ir.resolve_operand("#float:1.5", env) == 1.5
    assert ast_ir.resolve_operand("#bool:0", env) is False
    assert ast_ir.resolve_operand("#bool:1", env) is True
    assert ast_ir.resolve_operand("a", env) == 7
    with pytest.raises(KeyError):
        ast_ir.resolve_operand("missing", env)


# ---------------------------------------------------------------------------
# clifford_jit still works through the extracted core (Step 1 regression)
# ---------------------------------------------------------------------------

def test_clifford_jit_unchanged_after_extraction() -> None:
    """Smoke: the existing `@clifford_jit` lowering path produces the
    same IR shape after M6 Step 1's extraction."""
    from tessera import ga
    from tessera.compiler.clifford_jit import lower_function_to_ir

    def f(rotor, points):
        rotated = ga.rotor_sandwich(rotor, points)
        return ga.norm(rotated)

    ir = lower_function_to_ir(f)
    assert ir.arg_names == ("rotor", "points")
    assert [op.op_name for op in ir.ops] == [
        "clifford_rotor_sandwich", "clifford_norm",
    ]
    # text() prefix preserved.
    assert ir.text().startswith("clifford_ir(")


# ---------------------------------------------------------------------------
# energy_jit (M6 Step 2)
# ---------------------------------------------------------------------------

def test_energy_jit_lowers_quadratic_form() -> None:
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y, W):
        return energy.quadratic(y, W)

    assert isinstance(E, EnergyCompiledCallable)
    assert E.artifact.source_name.endswith("E")
    ir = E.artifact.ir
    assert isinstance(ir, EnergyIRProgram)
    assert ir.arg_names == ("y", "W")
    assert [op.op_name for op in ir.ops] == ["energy_quadratic"]


def test_energy_jit_lowers_mlp_head_chain() -> None:
    """A small MLP energy head should lower through the whitelist."""
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y, W1, b1, W2, b2):
        h = energy.linear(y, W1, b1)
        a = energy.relu(h)
        out = energy.linear(a, W2, b2)
        return energy.reduce_sum(out)

    ir = E.artifact.ir
    op_names = [op.op_name for op in ir.ops]
    assert op_names == [
        "energy_linear", "energy_relu", "energy_linear", "energy_reduce_sum",
    ]
    # Each consumer references the prior op's result.
    assert ir.ops[1].operand_refs == (ir.ops[0].result_name,)
    assert ir.ops[2].operand_refs[0] == ir.ops[1].result_name


def test_energy_jit_text_format_is_energy_prefix() -> None:
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y):
        return energy.norm_sq(y)

    text = E.artifact.ir.text()
    assert text.startswith("energy_ir(")
    assert "# energy.norm_sq" in text


def test_energy_jit_metadata_is_json_serializable() -> None:
    import json
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y):
        h = energy.tanh(y)
        return energy.norm(h)

    meta = E.artifact.ir.as_metadata()
    blob = json.dumps(meta)
    reloaded = json.loads(blob)
    assert reloaded["namespace"] == "energy"
    assert [op["op"] for op in reloaded["ops"]] == [
        "energy_tanh", "energy_norm",
    ]


def test_energy_jit_rejects_non_energy_call() -> None:
    """``np.dot(y, y)`` is rejected — the receiver isn't ``energy``."""
    def E(y):
        return np.dot(y, y)     # noqa: F821 — only the AST is read

    with pytest.raises(EnergyJitError, match="only ``tessera.energy"):
        lower_energy_function(E)


def test_energy_jit_rejects_unknown_op() -> None:
    """``energy.softmax(x)`` isn't in the whitelist."""
    def E(y):
        return energy.softmax(y)  # noqa: F821

    with pytest.raises(EnergyJitError):
        lower_energy_function(E)


def test_energy_jit_decorator_returns_callable_wrapping_original() -> None:
    """v1 keeps execution semantics identical to the original Python
    function — codegen is M6 Step 3/4."""
    import numpy as np
    from tessera import energy

    @energy_jit(target="apple_gpu")
    def E(y):
        return energy.norm_sq(y)

    y = np.array([3.0, 4.0])
    assert E(y) == pytest.approx(25.0)


def test_energy_jit_rejects_unsupported_dtype() -> None:
    with pytest.raises(EnergyJitError, match="dtype='f32'"):
        @energy_jit(dtype="fp16")
        def E(y):
            return energy.norm(y)   # noqa: F821 — never reached


def test_energy_jit_whitelist_is_closed() -> None:
    """Adding a new op to the whitelist is a deliberate decision."""
    from tessera.compiler import energy_jit as _ej
    expected = {
        "quadratic", "bilinear", "inner",
        "polynomial", "norm", "norm_sq",
        "relu", "tanh", "sigmoid", "gelu", "softplus",
        "linear", "mlp_head",
        "reduce_sum",
    }
    assert set(_ej._ENERGY_ATTR_TO_OP_NAME) == expected
