"""Tests for F1+G2 — unified Diagnostic across all 3 lanes + source positions.

Locks four contracts:

1. Each lane defines a stable enum (``JitDiagnosticCode``,
   ``FrontendDiagnosticCode``, ``ConstrainedDiagnosticCode``).
2. Every lane exception (Clifford / Complex / Energy / textual DSL)
   exposes a ``.to_diagnostic()`` method returning a unified
   :class:`Diagnostic`.
3. The ``lane`` field is stamped correctly per lane.
4. ``SourceLocation`` plumbs through emission sites that have
   position info, and the explain summary renders it.
"""

from __future__ import annotations

import pytest

import tessera
from tessera.compiler import (
    ConstrainedDiagnosticCode,
    Diagnostic,
    FallbackReason,
    FrontendDiagnosticCode,
    JitDiagnosticCode,
    SourceLocation,
)


class TestEnumStability:
    def test_frontend_code_values_are_stable(self) -> None:
        # Lock the public string values — renaming is a breaking change.
        expected = {
            "LEX_UNEXPECTED_CHAR": "TEXTUAL_LEX_UNEXPECTED_CHAR",
            "PARSE_UNEXPECTED_TOKEN": "TEXTUAL_PARSE_UNEXPECTED_TOKEN",
            "PARSE_UNEXPECTED_EOF": "TEXTUAL_PARSE_UNEXPECTED_EOF",
            "PARSE_EXPECTED_IDENTIFIER": "TEXTUAL_PARSE_EXPECTED_IDENTIFIER",
            "PARSE_UNSUPPORTED_MODULE_DECL": "TEXTUAL_PARSE_UNSUPPORTED_MODULE_DECL",
            "SEMANTIC_UNKNOWN_OP": "TEXTUAL_SEMANTIC_UNKNOWN_OP",
            "SEMANTIC_ARITY_MISMATCH": "TEXTUAL_SEMANTIC_ARITY_MISMATCH",
            "SEMANTIC_RESULT_TYPE_UNRESOLVED": "TEXTUAL_SEMANTIC_RESULT_TYPE_UNRESOLVED",
        }
        for name, value in expected.items():
            assert FrontendDiagnosticCode[name].value == value

    def test_constrained_code_values_are_stable(self) -> None:
        expected = {
            "CLIFFORD_OP_NOT_WHITELISTED": "CLIFFORD_OP_NOT_WHITELISTED",
            "CLIFFORD_EMPTY_OP_PLAN": "CLIFFORD_EMPTY_OP_PLAN",
            "CLIFFORD_TARGET_MISMATCH": "CLIFFORD_TARGET_MISMATCH",
            "CLIFFORD_UNSUPPORTED_TARGET": "CLIFFORD_UNSUPPORTED_TARGET",
            "COMPLEX_NON_HOLOMORPHIC": "COMPLEX_NON_HOLOMORPHIC",
            "COMPLEX_CR_RESIDUAL_TOO_LARGE": "COMPLEX_CR_RESIDUAL_TOO_LARGE",
            "ENERGY_FORBIDDEN_OP": "ENERGY_FORBIDDEN_OP",
            "ENERGY_UNSUPPORTED_DTYPE": "ENERGY_UNSUPPORTED_DTYPE",
        }
        for name, value in expected.items():
            assert ConstrainedDiagnosticCode[name].value == value


class TestDiagnosticShape:
    def test_source_position_is_optional(self) -> None:
        d = Diagnostic(severity="info", code="X", message="m")
        assert d.source_position is None
        assert d.format_position() == ""

    def test_source_position_round_trips(self) -> None:
        pos = SourceLocation(line=12, col=5, source_name="foo.py")
        d = Diagnostic(
            severity="error", code="X", message="m",
            source_position=pos,
        )
        assert d.source_position is pos
        assert "12:5" in d.format_position()
        assert "foo.py" in d.format_position()

    def test_lane_field_is_optional(self) -> None:
        d = Diagnostic(severity="info", code="X", message="m")
        assert d.lane is None

    def test_from_frontend_stamps_lane_textual_dsl(self) -> None:
        d = Diagnostic.from_frontend(
            code=FrontendDiagnosticCode.PARSE_UNEXPECTED_TOKEN,
            message="m",
        )
        assert d.lane == "textual_dsl"
        assert d.severity == "error"

    def test_from_constrained_requires_lane(self) -> None:
        d = Diagnostic.from_constrained(
            code=ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED,
            message="m",
            lane="clifford_jit",
        )
        assert d.lane == "clifford_jit"

    def test_from_jit_stamps_tessera_jit_lane(self) -> None:
        d = Diagnostic.from_jit(
            code=JitDiagnosticCode.SOURCE_PROVIDED,
            message="m",
        )
        assert d.lane == "tessera_jit"


class TestTextualDSLException:
    def test_parser_error_has_to_diagnostic(self) -> None:
        from tessera.compiler.frontend.parser import FrontendSyntaxError
        try:
            raise FrontendSyntaxError(
                "unexpected EOF",
                code=FrontendDiagnosticCode.PARSE_UNEXPECTED_EOF.value,
            )
        except FrontendSyntaxError as exc:
            d = exc.to_diagnostic()
            assert d.code_value == "TEXTUAL_PARSE_UNEXPECTED_EOF"
            assert d.lane == "textual_dsl"


class TestCliffordException:
    def test_clifford_error_has_to_diagnostic(self) -> None:
        from tessera.compiler.clifford_jit import CliffordJitError
        try:
            raise CliffordJitError(
                "test",
                code=ConstrainedDiagnosticCode.CLIFFORD_UNSUPPORTED_TARGET.value,
            )
        except CliffordJitError as exc:
            d = exc.to_diagnostic()
            assert d.code_value == "CLIFFORD_UNSUPPORTED_TARGET"
            assert d.lane == "clifford_jit"

    def test_clifford_unsupported_target_carries_code(self) -> None:
        from tessera.compiler.clifford_jit import clifford_jit

        with pytest.raises(Exception) as excinfo:
            @clifford_jit(target="cuda_xyz")
            def f(a):
                return a
        assert hasattr(excinfo.value, "code"), (
            "CliffordJitError should expose .code attribute"
        )
        assert excinfo.value.code == "CLIFFORD_UNSUPPORTED_TARGET"


class TestComplexJitException:
    def test_not_holomorphic_error_has_to_diagnostic(self) -> None:
        from tessera.compiler.complex_jit import NotHolomorphicError
        try:
            raise NotHolomorphicError(
                op_name="complex_conjugate",
                python_attr="complex_conjugate",
                fn_name="f",
            )
        except NotHolomorphicError as exc:
            d = exc.to_diagnostic()
            assert d.code_value == "COMPLEX_NON_HOLOMORPHIC"
            assert d.lane == "complex_jit"
            assert d.detail["op_name"] == "complex_conjugate"


class TestEnergyJitException:
    def test_energy_error_has_to_diagnostic(self) -> None:
        from tessera.compiler.energy_jit import EnergyJitError
        try:
            raise EnergyJitError(
                "test",
                code=ConstrainedDiagnosticCode.ENERGY_UNSUPPORTED_DTYPE.value,
            )
        except EnergyJitError as exc:
            d = exc.to_diagnostic()
            assert d.code_value == "ENERGY_UNSUPPORTED_DTYPE"
            assert d.lane == "energy_jit"


class TestExplainLaneSurface:
    def test_jit_explain_diagnostics_carry_lane(self) -> None:
        @tessera.jit
        def f(x: tessera.Tensor["B"], y: tessera.Tensor["B"]):
            return tessera.ops.add(x, y)

        ex = f.explain()
        assert ex.diagnostics, "expected at least one diagnostic"
        # Every diagnostic from the tessera.jit lane is lane-stamped.
        for d in ex.diagnostics:
            # The fallback diagnostic (when present) has lane=None
            # because it fires post-frontend.  All others should be
            # lane-stamped.
            if not isinstance(d.code, FallbackReason):
                assert d.lane == "tessera_jit", (
                    f"diagnostic {d.code_value!r} not lane-stamped"
                )

    def test_explain_as_dict_includes_position_and_lane(self) -> None:
        @tessera.jit
        def f(x: tessera.Tensor["B"], y: tessera.Tensor["B"]):
            return tessera.ops.add(x, y)

        d = f.explain().as_dict()
        assert "diagnostics" in d
        for entry in d["diagnostics"]:
            # Both keys must be present (None values OK).
            assert "source_position" in entry
            assert "lane" in entry


class TestPublicNamespace:
    def test_new_symbols_exported_from_tessera_compiler(self) -> None:
        import tessera.compiler as tc
        for name in (
            "ConstrainedDiagnosticCode",
            "FrontendDiagnosticCode",
            "SourceLocation",
        ):
            assert name in tc.__all__, f"{name} not in compiler.__all__"
            assert hasattr(tc, name), f"{name} not importable"
