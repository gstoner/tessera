"""Tests for F3 + U1 + U2 — FrontendLane registry + recommend + explain.lane.

Locks four contracts:

1. The registry has exactly 5 lanes with stable enum values.
2. ``ts.compiler.lanes.recommend(source)`` picks the strongest lane
   the source qualifies for; falls back to ``tessera_jit`` on
   disqualifying ops or general tensor work.
3. ``ts.compiler.lanes.for_op(op_name)`` returns the lane(s) that
   accept an op name.
4. ``fn.explain().lane`` carries the lane provenance.
5. The generated ``docs/reference/tessera_frontend_lanes.md`` is
   drift-gated against ``render_markdown()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tessera
from tessera.compiler import FrontendLane, lanes
from tessera.compiler.frontend_lanes import (
    all_lanes,
    for_lane,
    for_op,
    recommend,
    render_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DOC = REPO_ROOT / "docs" / "reference" / "tessera_frontend_lanes.md"


class TestRegistryShape:
    def test_exactly_five_lanes(self) -> None:
        assert len(all_lanes()) == 5

    def test_lane_enum_values_are_stable(self) -> None:
        assert FrontendLane.TESSERA_JIT.value == "tessera_jit"
        assert FrontendLane.TEXTUAL_DSL.value == "textual_dsl"
        assert FrontendLane.CLIFFORD_JIT.value == "clifford_jit"
        assert FrontendLane.COMPLEX_JIT.value == "complex_jit"
        assert FrontendLane.ENERGY_JIT.value == "energy_jit"

    def test_every_lane_has_a_spec(self) -> None:
        for lane in FrontendLane:
            spec = for_lane(lane)
            assert spec.name is lane
            assert spec.source_format
            assert spec.decorator
            assert spec.emitted_ir
            assert spec.diagnostic_codes


class TestRecommendHeuristic:
    def test_general_tensor_code_falls_back_to_tessera_jit(self) -> None:
        result = recommend("def f(x, y): return ts.ops.matmul(x, y)")
        assert result.name is FrontendLane.TESSERA_JIT

    def test_clifford_lane_for_ga_module(self) -> None:
        result = recommend(
            "def f(a, b): return ga.geometric_product(a, b)"
        )
        assert result.name is FrontendLane.CLIFFORD_JIT

    def test_complex_lane_for_complex_prefix(self) -> None:
        result = recommend(
            "def f(z): return complex_exp(complex_mul(z, z))"
        )
        assert result.name is FrontendLane.COMPLEX_JIT

    def test_complex_lane_for_bare_mobius(self) -> None:
        result = recommend("def f(z): return mobius(z)")
        assert result.name is FrontendLane.COMPLEX_JIT

    def test_energy_lane_for_ebm_module(self) -> None:
        result = recommend("def f(x): return ebm.langevin_step(x)")
        assert result.name is FrontendLane.ENERGY_JIT

    def test_numpy_disqualifies_constrained_lanes(self) -> None:
        """A single numpy reference should kick constrained lanes
        out of contention — they'd reject at decoration time anyway."""

        result = recommend(
            "import numpy as np\n"
            "def f(z): return np.dot(complex_mul(z, z), z)"
        )
        assert result.name is FrontendLane.TESSERA_JIT

    def test_syntax_error_returns_tessera_jit(self) -> None:
        """When the source can't be parsed, ``recommend`` falls
        back to the general lane rather than raising."""

        result = recommend("def f(x: this is not valid")
        assert result.name is FrontendLane.TESSERA_JIT


class TestForOp:
    def test_clifford_op_matches_clifford_lane(self) -> None:
        matches = for_op("clifford_geometric_product")
        assert any(s.name is FrontendLane.CLIFFORD_JIT for s in matches)

    def test_mobius_matches_complex_lane(self) -> None:
        matches = for_op("mobius")
        assert any(s.name is FrontendLane.COMPLEX_JIT for s in matches)

    def test_ebm_op_matches_energy_lane(self) -> None:
        matches = for_op("ebm_langevin_step")
        assert any(s.name is FrontendLane.ENERGY_JIT for s in matches)

    def test_general_op_matches_nothing(self) -> None:
        """``matmul`` has no constrained-lane match — caller is
        expected to fall back to TESSERA_JIT."""

        matches = for_op("matmul")
        assert matches == ()


class TestPublicNamespaceShim:
    def test_lanes_module_exposed(self) -> None:
        """``ts.compiler.lanes`` resolves to the
        ``frontend_lanes`` module so the developer-facing call
        ``ts.compiler.lanes.recommend(...)`` works."""

        assert lanes.recommend is recommend
        assert lanes.all_lanes is all_lanes
        assert lanes.for_op is for_op


class TestExplainLane:
    def test_default_lane_is_tessera_jit(self) -> None:
        @tessera.jit
        def f(x: tessera.Tensor["B"], y: tessera.Tensor["B"]):
            return tessera.ops.add(x, y)

        ex = f.explain()
        assert ex.lane == "tessera_jit"

    def test_lane_appears_in_as_dict(self) -> None:
        @tessera.jit
        def f(x: tessera.Tensor["B"], y: tessera.Tensor["B"]):
            return tessera.ops.add(x, y)

        d = f.explain().as_dict()
        assert d["lane"] == "tessera_jit"

    def test_summary_omits_brackets_for_default_lane(self) -> None:
        """A ``tessera_jit`` lane should NOT show ``[tessera_jit]``
        in the summary — that's the default and adding the bracket
        would just be noise.  Non-default lanes get the bracket."""

        @tessera.jit
        def f(x: tessera.Tensor["B"], y: tessera.Tensor["B"]):
            return tessera.ops.add(x, y)

        text = str(f.explain())
        first_line = text.splitlines()[0]
        assert "[tessera_jit]" not in first_line


class TestGeneratedDocDriftGate:
    def test_generated_doc_matches_render(self) -> None:
        if not GENERATED_DOC.exists():
            pytest.fail(
                f"missing {GENERATED_DOC.relative_to(REPO_ROOT)} — "
                f"regenerate via "
                f"`python -c 'from tessera.compiler.frontend_lanes "
                f"import render_markdown; "
                f"open(\"{GENERATED_DOC.relative_to(REPO_ROOT)}\", "
                f"\"w\").write(render_markdown())'`"
            )
        on_disk = GENERATED_DOC.read_text(encoding="utf-8")
        rendered = render_markdown()
        assert on_disk == rendered, (
            "tessera_frontend_lanes.md is out of date with "
            "tessera.compiler.frontend_lanes.  Regenerate."
        )

    def test_doc_mentions_every_lane(self) -> None:
        text = GENERATED_DOC.read_text(encoding="utf-8")
        for lane in FrontendLane:
            assert lane.value in text, lane.value
