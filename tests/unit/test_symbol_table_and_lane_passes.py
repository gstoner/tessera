"""Tests for F2+G1 — SymbolTable substrate + Graph IR lane provenance.

Locks two contracts:

1. ``SymbolTable`` supports scoped define/lookup with stable
   inside-out resolution; ``undefined_name_diagnostic`` picks
   lane-appropriate codes.
2. ``GraphIRFunction.lane`` carries lane provenance; the
   lane-aware passes in ``lane_passes`` consume it correctly.
"""

from __future__ import annotations

import pytest

from tessera.compiler import (
    ConstrainedDiagnosticCode,
    FrontendDiagnosticCode,
    SourceLocation,
    SymbolEntry,
    SymbolTable,
)


class TestSymbolTableScoping:
    def test_define_and_lookup(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("x", kind="arg", ir_ref="i32")
        entry = table.lookup("x")
        assert entry is not None
        assert entry.name == "x"
        assert entry.kind == "arg"
        assert entry.ir_ref == "i32"

    def test_lookup_returns_none_for_undefined(self) -> None:
        table = SymbolTable(lane="textual_dsl")
        assert table.lookup("missing") is None

    def test_inside_out_scope_resolution(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("x", ir_ref="outer")
        table.enter_scope()
        table.define("x", ir_ref="inner")
        assert table.lookup("x").ir_ref == "inner"
        table.leave_scope()
        assert table.lookup("x").ir_ref == "outer"

    def test_inner_scope_inherits_outer(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("outer_only")
        table.enter_scope()
        # Outer name is visible.
        assert "outer_only" in table
        # Inner adds its own.
        table.define("inner_only")
        assert "inner_only" in table
        table.leave_scope()
        # Inner name no longer visible after leave_scope.
        assert "inner_only" not in table

    def test_leave_scope_at_function_level_raises(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        with pytest.raises(IndexError):
            table.leave_scope()

    def test_names_in_scope_walks_inside_out(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("a")
        table.define("b")
        table.enter_scope()
        table.define("c")
        names = table.names_in_scope()
        # Inner-first ordering.
        assert names[0] == "c"
        assert "a" in names
        assert "b" in names

    def test_depth_tracks_scope_stack(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        assert table.depth() == 1
        table.enter_scope()
        assert table.depth() == 2
        table.enter_scope()
        assert table.depth() == 3
        table.leave_scope()
        assert table.depth() == 2

    def test_bindings_introduced_in_current_scope(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("a")
        table.enter_scope()
        table.define("b")
        table.define("c")
        current = table.bindings_introduced_in_current_scope()
        names = {e.name for e in current}
        assert names == {"b", "c"}, names
        assert all(isinstance(e, SymbolEntry) for e in current)


class TestUndefinedNameDiagnostic:
    def test_picks_jit_code_for_tessera_jit_lane(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        diag = table.undefined_name_diagnostic("missing")
        assert diag.lane == "tessera_jit"
        # Should reuse the existing JIT-lane "unsupported body" code.
        assert "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY" in diag.code_value

    def test_picks_frontend_code_for_textual_dsl(self) -> None:
        table = SymbolTable(lane="textual_dsl")
        diag = table.undefined_name_diagnostic("missing")
        assert diag.lane == "textual_dsl"
        assert diag.code_value == FrontendDiagnosticCode.SEMANTIC_UNKNOWN_OP.value

    def test_picks_clifford_code_for_clifford_lane(self) -> None:
        table = SymbolTable(lane="clifford_jit")
        diag = table.undefined_name_diagnostic("missing")
        assert diag.lane == "clifford_jit"
        assert diag.code_value == ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value

    def test_detail_carries_names_in_scope(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        table.define("x")
        table.define("y")
        diag = table.undefined_name_diagnostic("z")
        assert diag.detail["undefined_name"] == "z"
        assert "x" in diag.detail["names_in_scope"]
        assert "y" in diag.detail["names_in_scope"]

    def test_source_position_round_trips(self) -> None:
        table = SymbolTable(lane="tessera_jit")
        pos = SourceLocation(line=42, col=7)
        diag = table.undefined_name_diagnostic("missing", source_position=pos)
        assert diag.source_position is pos


class TestGraphIRFunctionLane:
    def test_default_lane_is_tessera_jit(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction

        fn = GraphIRFunction(name="f")
        assert fn.lane == "tessera_jit"

    def test_lane_field_is_settable(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction

        fn = GraphIRFunction(name="f", lane="complex_jit")
        assert fn.lane == "complex_jit"


class TestLaneAwarePasses:
    def test_clifford_pass_noop_on_non_clifford_function(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction
        from tessera.compiler.lane_passes import assert_clifford_ops_only

        # Non-Clifford lane → pass returns no diagnostics regardless of
        # what's in the body.
        fn = GraphIRFunction(name="f", lane="tessera_jit")
        assert assert_clifford_ops_only(fn) == []

    def test_clifford_pass_flags_non_ga_op_in_clifford_function(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction, IROp
        from tessera.compiler.lane_passes import assert_clifford_ops_only

        fn = GraphIRFunction(name="bad", lane="clifford_jit")
        fn.body.append(IROp(
            op_name="tessera.matmul",
            operands=("%a", "%b"),
            operand_types=("tensor", "tensor"),
            result_type="tensor",
            result="r",
        ))
        diagnostics = assert_clifford_ops_only(fn)
        assert len(diagnostics) == 1
        assert diagnostics[0].lane == "clifford_jit"
        assert diagnostics[0].code_value == (
            ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value
        )

    def test_clifford_pass_accepts_clifford_ops(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction, IROp
        from tessera.compiler.lane_passes import assert_clifford_ops_only

        fn = GraphIRFunction(name="ok", lane="clifford_jit")
        fn.body.append(IROp(
            op_name="clifford_geometric_product",
            operands=("%a", "%b"),
            operand_types=("tensor", "tensor"),
            result_type="tensor",
            result="r",
        ))
        assert assert_clifford_ops_only(fn) == []

    def test_complex_pass_flags_non_holomorphic_op(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction, IROp
        from tessera.compiler.lane_passes import (
            assert_complex_jit_holomorphic,
        )

        fn = GraphIRFunction(name="bad", lane="complex_jit")
        fn.body.append(IROp(
            op_name="complex_conjugate",
            operands=("%a",),
            operand_types=("tensor",),
            result_type="tensor",
            result="r",
        ))
        diagnostics = assert_complex_jit_holomorphic(fn)
        assert len(diagnostics) == 1
        assert diagnostics[0].lane == "complex_jit"
        assert diagnostics[0].code_value == (
            ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value
        )

    def test_complex_pass_accepts_holomorphic_op(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction, IROp
        from tessera.compiler.lane_passes import (
            assert_complex_jit_holomorphic,
        )

        fn = GraphIRFunction(name="ok", lane="complex_jit")
        fn.body.append(IROp(
            op_name="complex_exp",
            operands=("%a",),
            operand_types=("tensor",),
            result_type="tensor",
            result="r",
        ))
        assert assert_complex_jit_holomorphic(fn) == []

    def test_run_lane_aware_passes_dispatches_to_both(self) -> None:
        from tessera.compiler.graph_ir import GraphIRFunction, IROp
        from tessera.compiler.lane_passes import run_lane_aware_passes

        # Complex-lane function with a non-holomorphic op.
        fn = GraphIRFunction(name="bad", lane="complex_jit")
        fn.body.append(IROp(
            op_name="complex_abs",
            operands=("%a",),
            operand_types=("tensor",),
            result_type="tensor",
            result="r",
        ))
        result = run_lane_aware_passes(fn)
        assert len(result) == 1
        assert result[0].code_value == (
            ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value
        )


class TestPublicNamespace:
    def test_symbol_table_exported(self) -> None:
        import tessera.compiler as tc
        assert "SymbolTable" in tc.__all__
        assert "SymbolEntry" in tc.__all__
