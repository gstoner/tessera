"""Tests for ``JitFn.explain()`` — the P0-3 front-door inspector.

Locks four contracts:

1. ``fn.explain()`` returns an :class:`Explain` object regardless of
   execution kind.
2. The 5-line summary (``str(explain)``) answers the four canonical
   questions in a deterministic shape.
3. Structured fields (``.ir``, ``.kernels``, ``.diagnostics``,
   ``.next_actions``) are populated from the JIT's already-resolved
   state without re-running compilation.
4. ``.as_dict()`` is JSON-serializable.
"""

from __future__ import annotations

import json

import pytest

import tessera
from tessera.compiler.explain import (
    Diagnostic,
    Explain,
    IRLayers,
    Kernel,
    NEXT_INSPECT_IR,
    NEXT_NO_ACTION,
    NextAction,
)


@pytest.fixture
def trivial_jit():
    @tessera.jit
    def f(
        x: tessera.Tensor["B", "D"],
        y: tessera.Tensor["B", "D"],
    ) -> tessera.Tensor["B", "D"]:
        return tessera.ops.relu(tessera.ops.add(x, y))

    return f


class TestExplainShape:
    def test_returns_explain_dataclass(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert isinstance(ex, Explain)

    def test_carries_function_name(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert "f" in ex.function_name

    def test_target_is_a_string(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert isinstance(ex.target, str)
        assert ex.target

    def test_execution_kind_is_one_of_known_values(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert ex.execution_kind in (
            "native_cpu", "native_gpu", "reference_cpu",
            "artifact_only", "fallback_eager",
        )


class TestExplainSummary:
    def test_summary_is_5_lines(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        lines = str(ex).splitlines()
        assert len(lines) == 5

    def test_summary_starts_with_function_name(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        first = str(ex).splitlines()[0]
        # The label format is ``tessera.jit['name' → target]: ...``
        assert "tessera.jit[" in first
        assert ex.function_name in first

    def test_summary_includes_ir_layers_line(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        lines = str(ex).splitlines()
        assert any("IR layers" in l for l in lines)

    def test_summary_includes_kernels_line(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        lines = str(ex).splitlines()
        assert any("Kernels resolved" in l for l in lines)

    def test_summary_includes_next_action(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        lines = str(ex).splitlines()
        assert any("Next:" in l for l in lines)


class TestStructuredFields:
    def test_ir_layers_namespace_present(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert isinstance(ex.ir, IRLayers)

    def test_kernels_populated_from_graph_ir(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert len(ex.kernels) > 0
        for k in ex.kernels:
            assert isinstance(k, Kernel)
            assert isinstance(k.op_name, str)
            assert isinstance(k.runtime_status, str)

    def test_diagnostics_are_typed_diagnostics(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        for d in ex.diagnostics:
            assert isinstance(d, Diagnostic)

    def test_next_actions_carry_stable_codes(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        assert len(ex.next_actions) > 0
        for n in ex.next_actions:
            assert isinstance(n, NextAction)
            assert n.code  # non-empty stable ID

    def test_predicate_properties(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        # exactly one of the four state predicates should be True for
        # a successfully-resolved JIT.
        states = (
            ex.is_native, ex.is_reference,
            ex.is_artifact_only, ex.is_fallback,
        )
        assert sum(states) == 1, (
            f"expected exactly one execution-kind predicate to be True, "
            f"got {dict(zip(['native','reference','artifact','fallback'], states))}"
        )


class TestAsDict:
    def test_as_dict_is_json_serializable(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        d = ex.as_dict()
        # Should not raise.
        text = json.dumps(d)
        assert len(text) > 0

    def test_as_dict_carries_diagnostic_codes_as_strings(
        self, trivial_jit
    ) -> None:
        ex = trivial_jit.explain()
        d = ex.as_dict()
        for entry in d["diagnostics"]:
            assert isinstance(entry["code"], str)

    def test_as_dict_carries_next_action_codes(self, trivial_jit) -> None:
        ex = trivial_jit.explain()
        d = ex.as_dict()
        for entry in d["next_actions"]:
            assert isinstance(entry["code"], str)


class TestNextActionDerivation:
    def test_clean_native_yields_no_action_or_check_support(
        self, trivial_jit
    ) -> None:
        """A clean compile through CPU reference path emits the
        no-action sentinel (no fallback, no issue to fix)."""

        ex = trivial_jit.explain()
        first = ex.next_actions[0]
        # Either NO_ACTION (clean reference run) or CHECK_SUPPORT
        # (clean native run) — both are acceptable for a no-issue
        # path.
        assert first.code in (NEXT_NO_ACTION, "CHECK_TS_COMPILER_SUPPORT")

    def test_artifact_only_yields_inspect_ir_action(self) -> None:
        """When execution_kind is artifact_only, the front door
        surfaces the IR inspection hint."""

        @tessera.jit(target="rocm")
        def g(
            x: tessera.Tensor["B", "D"],
            y: tessera.Tensor["B", "D"],
        ) -> tessera.Tensor["B", "D"]:
            return tessera.ops.add(x, y)

        ex = g.explain()
        if ex.is_artifact_only:
            codes = [n.code for n in ex.next_actions]
            assert NEXT_INSPECT_IR in codes
