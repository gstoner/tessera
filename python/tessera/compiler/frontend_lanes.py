"""Frontend-lane registry (F3 + U1 + U2, 2026-05-19).

Tessera has three deliberate frontend lanes:

1. ``@tessera.jit`` — general Python tensor programs (CPython AST →
   Graph IR).
2. Textual DSL — serialized IR-like surface used by lit fixtures
   (regex lexer + recursive-descent parser → Graph IR).
3. Constrained math lanes — ``@clifford_jit`` / ``@complex_jit`` /
   ``@energy_jit`` (CPython AST → whitelist-verified IR with
   stricter contracts).

This module is the canonical Python-side answer to the developer
question "which decorator should I use?".  It exposes:

  * :class:`FrontendLane` — enum of the 5 lane names.
  * :class:`FrontendLaneSpec` — per-lane contract (source format,
    verification, emitted IR, diagnostics).
  * :func:`all_lanes` — every spec, in declared order.
  * :func:`recommend(source)` — AST-walking heuristic that picks the
    strongest lane the source qualifies for.
  * :func:`for_op(op_name)` — reverse lookup: which lane should
    handle this op?

The registry is also the input to the generated docs at
``docs/reference/tessera_frontend_lanes.md``; the dashboard is
drift-gated against this module's :func:`render_markdown`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FrontendLane(Enum):
    """The 5 active Tessera frontend lanes.  Values are the same
    strings that flow through ``GraphIRFunction.lane`` and
    ``Diagnostic.lane``."""

    TESSERA_JIT = "tessera_jit"
    TEXTUAL_DSL = "textual_dsl"
    CLIFFORD_JIT = "clifford_jit"
    COMPLEX_JIT = "complex_jit"
    ENERGY_JIT = "energy_jit"


@dataclass(frozen=True)
class FrontendLaneSpec:
    """One row of the lane registry."""

    name: FrontendLane

    source_format: str
    """Human-readable description of the input format."""

    decorator: str
    """Canonical decorator / factory name a developer would type
    (e.g., ``"@tessera.jit"`` or ``"tessera.from_text(...)"``)."""

    verification: tuple[str, ...]
    """Verification contracts the lane enforces at decoration time
    (e.g., ``"holomorphicity"`` for ``@complex_jit``)."""

    emitted_ir: str
    """The IR layer the lane emits (``"GraphIR"`` for the general
    lanes, ``"ConstrainedIRProgram"`` for the math lanes)."""

    diagnostic_codes: tuple[str, ...]
    """Stable diagnostic code prefixes this lane emits — points
    consumers at the matching enum in
    :mod:`tessera.compiler.diagnostics`."""

    explain_supported: bool
    """``True`` when ``fn.explain()`` works on this lane's JitFn /
    callable.  Today only the ``@tessera.jit`` lane has it (the
    constrained lanes ship their own report types — this is the
    follow-up to fold them in)."""

    op_name_patterns: tuple[str, ...] = ()
    """Op-name prefixes / exact names the lane accepts.  Used by
    :func:`recommend` and :func:`for_op` for routing.  Empty means
    "any op" (the general lanes); the constrained lanes carry
    explicit prefixes."""

    notes: str = ""


_LANE_REGISTRY: tuple[FrontendLaneSpec, ...] = (
    FrontendLaneSpec(
        name=FrontendLane.TESSERA_JIT,
        source_format="Python callable (CPython AST)",
        decorator="@tessera.jit",
        verification=(
            "constraints (ConstraintSolver)",
            "effects (EffectLattice)",
            "op-catalog membership",
        ),
        emitted_ir="GraphIR → Schedule IR → Tile IR → Target IR",
        diagnostic_codes=("JIT_*",),
        explain_supported=True,
        op_name_patterns=(),  # general — accepts anything in op_catalog
        notes=(
            "The general lane.  Accepts any function whose AST "
            "lowers to canonical Graph IR ops.  Default choice "
            "unless one of the stricter lanes applies."
        ),
    ),
    FrontendLaneSpec(
        name=FrontendLane.TEXTUAL_DSL,
        source_format="textual IR string",
        decorator="tessera.compiler.frontend.parse_text(...)",
        verification=(
            "lexical (regex)",
            "syntactic (recursive descent)",
            "semantic (op catalog + arity)",
        ),
        emitted_ir="GraphIR",
        diagnostic_codes=("TEXTUAL_*",),
        explain_supported=False,
        op_name_patterns=(),  # general — accepts anything in op_catalog
        notes=(
            "Serialized IR-like surface used by lit fixtures and "
            "round-trip tests.  Authored by hand or emitted by an "
            "external producer."
        ),
    ),
    FrontendLaneSpec(
        name=FrontendLane.CLIFFORD_JIT,
        source_format="Python callable (CPython AST, GA whitelist)",
        decorator="@tessera.compiler.clifford_jit(target=...)",
        verification=(
            "GA op whitelist (every op must be tessera.ga.*)",
            "target-routing (every op must route to the decorator's target)",
            "manifest membership (every op must be fused on target)",
        ),
        emitted_ir="CliffordIRProgram",
        diagnostic_codes=("CLIFFORD_*",),
        explain_supported=False,
        op_name_patterns=("clifford_", "ga.",),
        notes=(
            "GA / Clifford-algebra lane.  Every op must be a "
            "fused Apple GPU MSL kernel; the decorator-time check "
            "refuses functions that mix in tensor / numpy ops."
        ),
    ),
    FrontendLaneSpec(
        name=FrontendLane.COMPLEX_JIT,
        source_format="Python callable (CPython AST, holomorphic whitelist)",
        decorator="@tessera.compiler.complex_jit (alias @analytic_symbolic)",
        verification=(
            "holomorphicity (every op in HOLOMORPHIC_OPS)",
            "Cauchy-Riemann residual (decoration-time, exact)",
        ),
        emitted_ir="ComplexIRProgram",
        diagnostic_codes=("COMPLEX_*",),
        explain_supported=False,
        op_name_patterns=(
            "complex_",
            "mobius",
            "stereographic",
            "cross_ratio",
        ),
        notes=(
            "Visual Complex Analysis lane (M7).  Refuses functions "
            "containing anti-holomorphic ops (conjugate, abs, arg) "
            "at decoration time; the lane's invariant is that the "
            "emitted IR is provably holomorphic."
        ),
    ),
    FrontendLaneSpec(
        name=FrontendLane.ENERGY_JIT,
        source_format="Python callable (CPython AST, energy whitelist)",
        decorator="@tessera.compiler.energy_jit(target=..., dtype=...)",
        verification=(
            "energy op whitelist",
            "dtype restriction (v1 only fp32)",
        ),
        emitted_ir="EnergyIRProgram",
        diagnostic_codes=("ENERGY_*",),
        explain_supported=False,
        op_name_patterns=("energy_", "ebm_"),
        notes=(
            "Energy-based-model lane.  Accepts only ops from the "
            "energy/EBM whitelist; v1 fp32-only.  Pairs with the "
            "EBM Apple GPU fused kernels."
        ),
    ),
)


def all_lanes() -> tuple[FrontendLaneSpec, ...]:
    """Every registered lane, in declared order."""

    return _LANE_REGISTRY


def for_lane(name: FrontendLane) -> FrontendLaneSpec:
    """Return the spec for a given lane enum value."""

    for spec in _LANE_REGISTRY:
        if spec.name is name:
            return spec
    raise KeyError(f"no spec for lane {name!r}")


def for_op(op_name: str) -> tuple[FrontendLaneSpec, ...]:
    """Return every lane whose ``op_name_patterns`` matches ``op_name``.

    The general lanes (TESSERA_JIT, TEXTUAL_DSL) have empty
    pattern lists — they match nothing here, and the caller is
    expected to fall back to TESSERA_JIT when no constrained lane
    matches.
    """

    matches: list[FrontendLaneSpec] = []
    for spec in _LANE_REGISTRY:
        for pattern in spec.op_name_patterns:
            if op_name.startswith(pattern) or op_name == pattern.rstrip("_"):
                matches.append(spec)
                break
    return tuple(matches)


def recommend(source: str) -> FrontendLaneSpec:
    """Heuristic: which lane should compile this Python source?

    Walks the AST and counts how many op references look like they
    belong to each constrained lane.  Returns the lane with the
    highest hit count, or :class:`FrontendLane.TESSERA_JIT` when no
    constrained lane has a clear majority.

    Conservative: a single ``np.dot`` is enough to disqualify the
    GA / complex / energy lanes (they'd reject the function at
    decoration time anyway).  This way the recommendation tracks
    what would actually compile, not what looks superficially
    related.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return for_lane(FrontendLane.TESSERA_JIT)

    # Per-lane hit counters; "disqualified" set tracks lanes that
    # contain at least one obviously-non-lane op.
    hits: dict[FrontendLane, int] = {
        lane.name: 0 for lane in _LANE_REGISTRY
    }
    disqualified: set[FrontendLane] = set()

    constrained_lanes = (
        FrontendLane.CLIFFORD_JIT,
        FrontendLane.COMPLEX_JIT,
        FrontendLane.ENERGY_JIT,
    )

    def _pattern_matches(pattern: str, attr_name: str, module: str) -> bool:
        """Match a lane op-name pattern against an AST attribute
        node's ``attr`` and the resolved ``module`` prefix.

        Patterns ending in ``_`` (e.g., ``"clifford_"``) match by
        prefix; patterns ending in ``.`` (e.g., ``"ga."``) match the
        bare module name *or* the dotted form; bare patterns
        (``"mobius"``) match exact attribute names.
        """

        stripped = pattern.rstrip("._")
        # Module-side match: "ga." or "ga" both match module="ga".
        if module and (module == stripped or module.startswith(pattern)):
            return True
        # Attribute-side match: "clifford_*" prefix or "mobius" exact.
        if attr_name.startswith(pattern) or attr_name == stripped:
            return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            # Resolve "module.attr" for one-level attributes.
            module = ""
            if isinstance(node.value, ast.Name):
                module = node.value.id
            elif isinstance(node.value, ast.Attribute):
                module = node.value.attr

            for lane_enum in constrained_lanes:
                spec = for_lane(lane_enum)
                hit = any(
                    _pattern_matches(p, attr_name, module)
                    for p in spec.op_name_patterns
                )
                if hit:
                    hits[lane_enum] += 1

            # Disqualify constrained lanes on numpy / aten references.
            if isinstance(node.value, ast.Name) and node.value.id in (
                "np", "numpy", "torch", "aten",
            ):
                disqualified.update(constrained_lanes)
        elif isinstance(node, ast.Name):
            # Bare-name reference matches lane patterns on exact names
            # only ("mobius", "stereographic", etc.).  Catches code
            # like `mobius(a, b, c, d, z)` that uses M7 primitives
            # directly without the `complex.` prefix.
            for lane_enum in constrained_lanes:
                spec = for_lane(lane_enum)
                for pattern in spec.op_name_patterns:
                    stripped = pattern.rstrip("._")
                    if node.id == stripped or node.id.startswith(pattern):
                        hits[lane_enum] += 1
                        break

    # Find the constrained lane with the most hits that isn't
    # disqualified.
    best_lane: Optional[FrontendLane] = None
    best_count = 0
    for lane_enum in constrained_lanes:
        if lane_enum in disqualified:
            continue
        if hits[lane_enum] > best_count:
            best_lane = lane_enum
            best_count = hits[lane_enum]

    if best_lane is not None and best_count > 0:
        return for_lane(best_lane)
    return for_lane(FrontendLane.TESSERA_JIT)


# ─────────────────────────────────────────────────────────────────────
# Doc rendering.
# ─────────────────────────────────────────────────────────────────────


_DOC_HEADER = """\
<!-- AUTO-GENERATED by python/tessera/compiler/frontend_lanes.py. DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.frontend_lanes --render -->

# Tessera Frontend Lanes

Tessera has **three deliberate frontend lanes** (plus a textual DSL).
Each lane verifies different invariants at decoration time and emits
to a different IR shape.  This page lists every active lane.

| Lane | Decorator | Source format | Emitted IR | Verification |
|------|-----------|---------------|------------|--------------|
"""


def render_markdown() -> str:
    """Render the lane registry as the canonical
    ``docs/reference/tessera_frontend_lanes.md`` document."""

    lines: list[str] = [_DOC_HEADER]
    for spec in _LANE_REGISTRY:
        verification = "; ".join(f"`{v}`" for v in spec.verification)
        lines.append(
            f"| **{spec.name.value}** | `{spec.decorator}` | "
            f"{spec.source_format} | `{spec.emitted_ir}` | {verification} |"
        )
    lines.append("")
    lines.append("## Per-lane notes")
    lines.append("")
    for spec in _LANE_REGISTRY:
        lines.append(f"### `{spec.name.value}`")
        lines.append("")
        lines.append(spec.notes or "_(no additional notes)_")
        lines.append("")
        lines.append(
            "**Diagnostic codes:** "
            + ", ".join(f"`{p}`" for p in spec.diagnostic_codes)
            + "  "
        )
        lines.append(
            f"**`fn.explain()` supported:** "
            f"{'yes' if spec.explain_supported else 'not yet'}  "
        )
        if spec.op_name_patterns:
            lines.append(
                "**Op-name patterns accepted:** "
                + ", ".join(f"`{p}`" for p in spec.op_name_patterns)
                + "  "
            )
        lines.append("")
    lines.append("## Python query API")
    lines.append("")
    lines.append("```python")
    lines.append("import tessera as ts")
    lines.append("")
    lines.append("ts.compiler.lanes.all()")
    lines.append("ts.compiler.lanes.recommend(source)")
    lines.append("ts.compiler.lanes.for_op('clifford_geometric_product')")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "FrontendLane",
    "FrontendLaneSpec",
    "all_lanes",
    "for_lane",
    "for_op",
    "recommend",
    "render_markdown",
]
