"""Audit-B (2026-05-22) — Effect lattice + determinism audit.

The TSOL spec declares the effect lattice:

    pure < random < movement < state < collective < memory < io < top

Every op in `tessera.ops` and `tessera.nn.*` carries an `effect`
declaration via :data:`tessera.compiler.op_catalog.OP_SPECS`.  This
module audits:

  * Effect distribution across the surface (how many ops at each
    lattice level).
  * Per-category cross-check: do ops in canonical TSOL categories
    declare effects consistent with the spec's lattice examples?
    (e.g., `all_reduce` should be `collective`; `dropout` should be
    `random`; `matmul` should be `pure`.)
  * Determinism contract: which ops accept `deterministic=...` /
    declare deterministic behavior in `numeric_policy`.

The dashboard at ``docs/audit/generated/effect_lattice_audit.md``
surfaces:

  * Per-effect-level op count and a small sample.
  * Mismatched ops where the declared effect doesn't match the
    expected category from the TSOL spec.
  * Determinism coverage: ops with `numeric_policy.deterministic`
    set vs ops in deterministic-critical categories.

Drift gates at ``tests/unit/test_effect_audit.py``:

  * Every op has a parseable Effect declaration.
  * Canonical effect anchors stay locked (`matmul=pure`,
    `dropout=random`, `all_reduce=collective`, etc.).
  * No op silently regresses to `effect=top` (the conservative
    fallback — usable but lossy for compiler analysis).
"""

from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────
# Expected-effect anchors per TSOL spec (Effect Mapping section)
# ─────────────────────────────────────────────────────────────────────────


# Maps an op canonical name → the effect the TSOL spec declares it
# should carry.  Sourced from
# `docs/operations/Tessera_Standard_Operations.md` §Effect Mapping.
# Drift gate enforces these don't regress.
_TSOL_EFFECT_ANCHORS: dict[str, str] = {
    # pure — math + algebra + layout
    "matmul": "pure",
    "gemm": "pure",
    "conv2d": "pure",
    "conv3d": "pure",
    "layer_norm": "pure",
    "rmsnorm": "pure",
    "softmax": "pure",
    "gelu": "pure",
    "relu": "pure",
    "silu": "pure",
    "fft": "pure",
    "ifft": "pure",
    "rfft": "pure",
    "irfft": "pure",
    "transpose": "pure",
    "cast": "pure",
    # random — RNG-bearing ops
    "dropout": "random",
    "rng_uniform": "random",
    "rng_normal": "random",
    # collective — distributed communication
    "all_reduce": "collective",
    "reduce_scatter": "collective",
    "all_gather": "collective",
    "all_to_all": "collective",
    "moe_dispatch": "collective",
    "moe_combine": "collective",
}


# Canonical effect levels for the dashboard.  Order matches the
# lattice order in `effects.py`.
_EFFECT_LEVELS = (
    "pure",
    "random",
    "movement",
    "state",
    "collective",
    "memory",
    "io",
    "top",
)


# ─────────────────────────────────────────────────────────────────────────
# Audit data model
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EffectAuditRow:
    """One op's effect audit snapshot."""

    name: str
    declared_effect: str
    expected_effect: str | None
    matches_anchor: bool
    has_numeric_policy: bool
    determinism_aware: bool


# ─────────────────────────────────────────────────────────────────────────
# Collection
# ─────────────────────────────────────────────────────────────────────────


def collect_effect_audit() -> tuple[EffectAuditRow, ...]:
    """Walk OP_SPECS + cross-check each op against the anchor list."""
    from .op_catalog import OP_SPECS
    from .primitive_coverage import _policy_for_name

    out: list[EffectAuditRow] = []
    for name in sorted(OP_SPECS.keys()):
        spec = OP_SPECS[name]
        declared = spec.effect
        expected = _TSOL_EFFECT_ANCHORS.get(name)
        matches = (expected is None) or (declared == expected)
        # Numeric policy presence — Sprint C2 attaches NumericPolicy
        # to 67 ops with intrinsic dtype/accum coupling.  Determinism-
        # awareness is captured via the `deterministic` field on the
        # policy (default False today; ops can opt in via the policy
        # factory).
        policy = _policy_for_name(name)
        has_policy = policy is not None
        determinism_aware = bool(policy and getattr(policy, "deterministic", False))
        out.append(EffectAuditRow(
            name=name,
            declared_effect=declared,
            expected_effect=expected,
            matches_anchor=matches,
            has_numeric_policy=has_policy,
            determinism_aware=determinism_aware,
        ))
    return tuple(out)


def effect_distribution() -> dict[str, int]:
    """Return ``{effect_level: count}`` across the full op surface."""
    rows = collect_effect_audit()
    out: dict[str, int] = {level: 0 for level in _EFFECT_LEVELS}
    for r in rows:
        out[r.declared_effect] = out.get(r.declared_effect, 0) + 1
    return out


def anchor_mismatches() -> tuple[EffectAuditRow, ...]:
    """Return rows where the declared effect contradicts the TSOL
    spec anchor.  Empty tuple = clean."""
    return tuple(r for r in collect_effect_audit() if not r.matches_anchor)


def top_effect_ops() -> tuple[str, ...]:
    """Return ops at the conservative ``top`` fallback level.
    Ideally every op narrows to something more specific."""
    return tuple(
        r.name for r in collect_effect_audit() if r.declared_effect == "top"
    )


def determinism_aware_ops() -> tuple[str, ...]:
    """Return ops whose numeric_policy declares deterministic=True."""
    return tuple(
        r.name for r in collect_effect_audit() if r.determinism_aware
    )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


#: Stable CSV column order for the effect-lattice audit — append-only.
EFFECT_AUDIT_CSV_COLUMNS: tuple[str, ...] = (
    "name", "declared_effect", "expected_effect", "matches_anchor",
    "has_numeric_policy", "determinism_aware",
)


def render_csv() -> str:
    """Render the canonical machine-readable effect-lattice audit table.

    One row per op, sorted by name.  Drift-gated artifact; the Markdown
    is the human companion.
    """
    import csv as _csv
    import io as _io

    rows = sorted(collect_effect_audit(), key=lambda r: r.name)
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(EFFECT_AUDIT_CSV_COLUMNS)
    for r in rows:
        writer.writerow([
            r.name, r.declared_effect,
            r.expected_effect if r.expected_effect is not None else "",
            "1" if r.matches_anchor else "0",
            "1" if r.has_numeric_policy else "0",
            "1" if r.determinism_aware else "0",
        ])
    return buf.getvalue()


def render_dashboard() -> str:
    rows = collect_effect_audit()
    distribution = effect_distribution()
    mismatches = anchor_mismatches()
    top_ops = top_effect_ops()
    det_ops = determinism_aware_ops()

    lines: list[str] = []
    lines.append("# Effect Lattice + Determinism Audit")
    lines.append("")
    lines.append(
        "Generated from `python/tessera/compiler/effect_audit.py`.  "
        "Don't edit by hand — regenerate via "
        "`python -c \"from tessera.compiler.effect_audit import "
        "render_dashboard; "
        "open('docs/audit/generated/effect_lattice_audit.md', 'w')"
        ".write(render_dashboard())\"`.  "
        "Drift gated by `tests/unit/test_effect_audit.py`."
    )
    lines.append("")

    # ── Headline ──
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **{len(rows)}** ops in `OP_SPECS` carry an effect.")
    lines.append(
        f"- **{len(mismatches)}** mismatch the TSOL spec anchors "
        f"(of {len(_TSOL_EFFECT_ANCHORS)} anchored ops)."
    )
    lines.append(
        f"- **{len(top_ops)}** ops sit at the conservative `top` "
        f"fallback level."
    )
    lines.append(
        f"- **{len(det_ops)}** ops declare deterministic-aware "
        f"numeric policies."
    )
    lines.append("")

    # ── Distribution ──
    lines.append("## Effect distribution")
    lines.append("")
    lines.append("| Effect level | Count | Description |")
    lines.append("|--------------|------:|-------------|")
    descriptions = {
        "pure": "No side effects; output depends only on inputs.",
        "random": "RNG-bearing; result varies across calls.",
        "movement": "Explicit prefetch / async copy / wait.",
        "state": "Reads or writes compiler-visible state (KV cache).",
        "collective": "Async device / rank communication.",
        "memory": "Writes mutable tensors or aliases host memory.",
        "io": "Host I/O or unknown external calls.",
        "top": "Conservative fallback (unknown / unconstrained).",
    }
    for level in _EFFECT_LEVELS:
        count = distribution.get(level, 0)
        lines.append(
            f"| `{level}` | {count} | {descriptions[level]} |"
        )
    lines.append("")

    # ── Anchor mismatches ──
    lines.append("## TSOL spec anchor cross-check")
    lines.append("")
    if not mismatches:
        lines.append(
            "_No mismatches — every TSOL spec anchor (matmul, dropout, "
            "all_reduce, etc.) carries the expected effect._"
        )
    else:
        lines.append("| Op | Declared | Expected (TSOL spec) |")
        lines.append("|----|----------|---------------------|")
        for r in mismatches:
            lines.append(
                f"| `{r.name}` | `{r.declared_effect}` | "
                f"`{r.expected_effect}` |"
            )
    lines.append("")

    # ── top-effect ops (potential bugs) ──
    lines.append("## Ops at `top` (conservative fallback)")
    lines.append("")
    if not top_ops:
        lines.append(
            "_No ops at `top` — every op narrows to a specific lattice "
            "level._"
        )
    else:
        lines.append(
            "Ops below couldn't be narrowed below `top` and inherit "
            "the conservative fallback.  Each one is a potential "
            "optimization opportunity:"
        )
        lines.append("")
        for name in top_ops:
            lines.append(f"- `{name}`")
    lines.append("")

    # ── Determinism coverage ──
    lines.append("## Determinism-aware numeric policies")
    lines.append("")
    lines.append(
        "The TSOL spec promises `deterministic=True` flips ops to "
        "deterministic implementations.  Today's numeric-policy "
        "system (Sprint C2, 2026-05-11) attaches a `NumericPolicy` "
        "to 67 ops; the `deterministic` field controls per-op "
        "behavior."
    )
    lines.append("")
    if not det_ops:
        lines.append(
            "_No ops carry `deterministic=True` on their default "
            "numeric_policy today.  Deterministic mode is "
            "user-controlled via `@jit(deterministic=True)` rather "
            "than per-op default policies; this is intentional._"
        )
    else:
        lines.append(
            f"**{len(det_ops)}** ops declare deterministic-aware "
            f"default policies:"
        )
        lines.append("")
        for name in det_ops:
            lines.append(f"- `{name}`")
    lines.append("")

    # ── Per-anchor verification ──
    lines.append("## Per-anchor verification (TSOL effect map)")
    lines.append("")
    lines.append("| Op | Spec effect | Declared effect | OK |")
    lines.append("|----|-------------|-----------------|----|")
    row_by_name = {r.name: r for r in rows}
    for name in sorted(_TSOL_EFFECT_ANCHORS.keys()):
        row = row_by_name.get(name)
        expected = _TSOL_EFFECT_ANCHORS[name]
        if row is None:
            lines.append(
                f"| `{name}` | `{expected}` | _missing_ | ⚠️ |"
            )
        else:
            ok = "✅" if row.matches_anchor else "❌"
            lines.append(
                f"| `{name}` | `{expected}` | "
                f"`{row.declared_effect}` | {ok} |"
            )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path=None):
    from pathlib import Path
    target = path or (
        Path(__file__).resolve().parents[3]
        / "docs" / "audit" / "generated" / "effect_lattice_audit.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard())
    return target


__all__ = [
    "EffectAuditRow",
    "collect_effect_audit",
    "effect_distribution",
    "anchor_mismatches",
    "top_effect_ops",
    "determinism_aware_ops",
    "render_dashboard",
    "write_dashboard",
]
