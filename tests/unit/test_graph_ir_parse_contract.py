"""Graph IR the front end emits must be valid MLIR -- checked by MLIR.

Decision #19's Target IR gate (W0.9) parses and verifies every emitter's text
with ``tessera-opt``. Graph IR had no equivalent: the driver's
``tessera-opt`` check ran a lowering pipeline over the non-canonical render
with ``--allow-unregistered-dialect``, recorded ``status="failed"``, and
nothing read the record. Measured across the whole Mac unit suite on
2026-10-24, 2,014 of 3,127 distinct modules parsed and verified; most of the
rest were symbolic decoration-time modules, which by design are not for the
parser (``GRAPH_IR_UNRESOLVED_ELEMENT_TYPE``, Decision #21a).

This gate runs the strict, opt-in ``enable_tool_validation=True`` path end to
end over a fixed corpus of *specialized* modules emitted by the real ``@jit``
front end:

* every catalogued elementwise op of arity <= 2 on typed ``f32`` operands;
* typed forms of the foundation compiler examples.

**Ratchet.** ``EXEMPT`` names each corpus case that does not validate today,
with the failure it shows. An exempt case must still fail exactly that way --
once it passes, this gate fails until the entry is deleted -- and the list may
not grow past ``EXEMPT_CEILING``. Lower the ceiling with every removal.

``tessera-opt`` is resolved like the driver resolves it; a host without a
runnable build skips, since it cannot evaluate this contract.
"""

from __future__ import annotations

import inspect

import pytest

import tessera
from tessera.compiler import driver
from tessera.compiler import op_catalog
from tessera.compiler.driver import GraphIRToolValidationError, compile_graph_module

UNDECLARED = "is unknown"

#: corpus case -> substring of the failure it shows today (see module docstring).
EXEMPT: dict[str, str] = {
    # Stage 3 (catalog <= ODS): catalogued Graph ops no dialect declares.
    "elementwise:atan2": UNDECLARED,
    "elementwise:clifford_conjugate": UNDECLARED,
    "elementwise:clifford_exp": UNDECLARED,
    "elementwise:clifford_grade_involution": UNDECLARED,
    "elementwise:clifford_grade_projection": UNDECLARED,
    "elementwise:clifford_hodge_star": UNDECLARED,
    "elementwise:clifford_log": UNDECLARED,
    "elementwise:clifford_reverse": UNDECLARED,
    "elementwise:clip": UNDECLARED,
    "elementwise:complex_abs": UNDECLARED,
    "elementwise:complex_arg": UNDECLARED,
    "elementwise:complex_conjugate": UNDECLARED,
    "elementwise:complex_div": UNDECLARED,
    "elementwise:complex_exp": UNDECLARED,
    "elementwise:complex_log": UNDECLARED,
    "elementwise:complex_mul": UNDECLARED,
    "elementwise:complex_pow": UNDECLARED,
    "elementwise:complex_sqrt": UNDECLARED,
    "elementwise:ebm_inner_step": UNDECLARED,
    "elementwise:ebm_refinement": UNDECLARED,
    "elementwise:mobius_from_three_points": UNDECLARED,
    "elementwise:softcap": UNDECLARED,
    "elementwise:stereographic": UNDECLARED,
    # The corpus synthesizes f32 operands; popcount is integer-only, and its
    # verifier says so. Not an emitter defect.
    "elementwise:popcount": "requires integer operands",
}
EXEMPT_CEILING = 24


def _arity(spec: op_catalog.OpSpec) -> int:
    return 2 if spec.max_arity >= 2 else 1


def _elementwise_cases() -> dict[str, str]:
    cases: dict[str, str] = {}
    for spec in op_catalog.OP_SPECS.values():
        if spec.lowering != "elementwise" or spec.min_arity > 2:
            continue
        if not callable(getattr(tessera.ops, spec.public_name, None)):
            continue
        n = _arity(spec)
        params = ", ".join(f"a{i}: tessera.f32[4, 8]" for i in range(n))
        args = ", ".join(f"a{i}" for i in range(n))
        cases.setdefault(
            f"elementwise:{spec.public_name}",
            f"def f({params}):\n    return tessera.ops.{spec.public_name}({args})\n",
        )
    return cases


_EXAMPLES = {
    "example:mlp": (
        "def f(A: tessera.f32[4, 8], B: tessera.f32[8, 16]):\n"
        "    return tessera.ops.relu(tessera.ops.matmul(A, B))\n"),
    "example:attention_like": (
        "def f(A: tessera.f32[4, 8], B: tessera.f32[8, 16]):\n"
        "    return tessera.ops.softmax(tessera.ops.matmul(A, B))\n"),
    "example:rmsnorm": (
        "def f(X: tessera.f32[4, 8]):\n"
        "    return tessera.ops.rmsnorm_safe(X)\n"),
}

CORPUS = {**_elementwise_cases(), **_EXAMPLES}


def _module(source: str):
    namespace = {"tessera": tessera}
    exec(source, namespace)  # noqa: S102 -- corpus sources are literals above
    fn = tessera.jit(source=source)(namespace["f"])
    assert not fn.graph_ir.frontend_unlowered_calls, fn.graph_ir.frontend_unlowered_calls
    return fn.graph_ir


def _validate(module):
    return compile_graph_module(
        module, source_origin="graph-ir-parse-contract", target="cpu",
        enable_tool_validation=True)


@pytest.fixture(scope="module")
def opt() -> str:
    tool = driver._resolve_tessera_opt()
    if tool is None:
        pytest.skip("no runnable tessera-opt; this host cannot evaluate the Graph IR parse contract")
    return tool


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_specialized_graph_ir_parses_and_verifies(case, opt):
    module = _module(CORPUS[case])
    if case in EXEMPT:
        with pytest.raises(GraphIRToolValidationError) as info:
            _validate(module)
        assert EXEMPT[case] in str(info.value), (
            f"{case} still fails, but differently than recorded: {info.value}")
        return
    bundle = _validate(module)
    assert [inv.status for inv in bundle.tool_invocations] == ["ok"]


def test_exemptions_only_shrink():
    stale = sorted(set(EXEMPT) - set(CORPUS))
    assert not stale, f"exempt cases no longer in the corpus: {stale}"
    assert len(EXEMPT) <= EXEMPT_CEILING, (
        f"{len(EXEMPT)} exemptions exceed the ceiling {EXEMPT_CEILING}; fix the "
        f"emitter or ODS instead of exempting a new failure")


def test_strict_validation_refuses_an_unspecialized_module_by_its_semantic_key(opt):
    source = "def f(a):\n    return tessera.ops.relu(a)\n"
    with pytest.raises(GraphIRToolValidationError, match="GRAPH_IR_UNRESOLVED_ELEMENT_TYPE"):
        _validate(_module(source))


def test_strict_validation_refuses_when_no_tool_can_answer(monkeypatch):
    monkeypatch.setattr(driver, "_resolve_tessera_opt", lambda: None)
    with pytest.raises(GraphIRToolValidationError, match="no runnable tessera-opt"):
        _validate(_module(_EXAMPLES["example:rmsnorm"]))


def test_validation_is_opt_in_and_never_masks_unregistered_ops():
    assert inspect.signature(compile_graph_module).parameters[
        "enable_tool_validation"].default is False
    source = inspect.getsource(driver._validate_graph_ir_with_tessera_opt)
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith(("#", '"""')))
    assert '"--allow-unregistered-dialect"' not in code
