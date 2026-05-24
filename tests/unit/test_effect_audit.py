"""Audit-B (2026-05-22) — effect lattice + determinism drift gate.

Pins:

  * Every op in OP_SPECS has a parseable Effect declaration.
  * Zero ops sit at the conservative `top` fallback (the lattice
    correctly narrows every op).
  * Zero mismatches against the TSOL spec effect anchors
    (`matmul=pure`, `dropout=random`, `all_reduce=collective`,
    `kv_cache_*=state`, etc.).
  * Effect distribution floor counts stay at-or-above the 2026-05-22
    baseline.
  * Dashboard at ``docs/audit/generated/effect_lattice_audit.md``
    matches the live registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.effect_audit import (
    EffectAuditRow,
    anchor_mismatches,
    collect_effect_audit,
    determinism_aware_ops,
    effect_distribution,
    render_dashboard,
    top_effect_ops,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = (
    REPO_ROOT / "docs" / "audit" / "generated" / "effect_lattice_audit.md"
)


_VALID_EFFECTS = frozenset({
    "pure", "random", "movement", "state",
    "collective", "memory", "io", "top",
})


# ─────────────────────────────────────────────────────────────────────────
# Structural sanity
# ─────────────────────────────────────────────────────────────────────────


def test_every_op_has_a_valid_effect() -> None:
    """Every row's declared_effect must be a known lattice level."""
    rows = collect_effect_audit()
    bad = [
        (r.name, r.declared_effect)
        for r in rows
        if r.declared_effect not in _VALID_EFFECTS
    ]
    assert not bad, f"Ops with invalid effects: {bad}"


def test_op_count_at_or_above_baseline() -> None:
    """Locked floor — catches accidental OP_SPECS deletions."""
    rows = collect_effect_audit()
    assert len(rows) >= 240, (
        f"Op count dropped below 240 ({len(rows)}); was OP_SPECS "
        f"accidentally truncated?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Lattice correctness — anchors + no `top` fallback
# ─────────────────────────────────────────────────────────────────────────


def test_no_anchor_mismatches() -> None:
    """Every TSOL spec anchor (matmul=pure, dropout=random, etc.)
    must match the declared effect.  A mismatch means the spec or
    the implementation drifted — fix one to match the other."""
    mismatches = anchor_mismatches()
    if mismatches:
        msg_lines = ["TSOL spec anchor mismatches:"]
        for r in mismatches:
            msg_lines.append(
                f"  - {r.name}: declared={r.declared_effect!r}, "
                f"spec expects {r.expected_effect!r}"
            )
        pytest.fail("\n".join(msg_lines))


def test_no_ops_at_top_fallback() -> None:
    """`top` is the conservative fallback for unknown / unconstrained
    effects.  Every op should narrow to something more specific.
    A regression to `top` is an optimization-defeating bug."""
    top = top_effect_ops()
    assert not top, (
        f"Ops at `top` fallback: {top}.  Narrow them to a specific "
        f"lattice level (pure/random/movement/state/collective/"
        f"memory/io)."
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-anchor sentinels
# ─────────────────────────────────────────────────────────────────────────


_LOCKED_EFFECT_SENTINELS = (
    # (op, expected_effect)
    ("matmul",        "pure"),
    ("gemm",          "pure"),
    ("conv2d",        "pure"),
    ("layer_norm",    "pure"),
    ("rmsnorm",       "pure"),
    ("softmax",       "pure"),
    ("gelu",          "pure"),
    ("transpose",     "pure"),
    ("fft",           "pure"),
    ("dropout",       "random"),
    ("rng_uniform",   "random"),
    ("rng_normal",    "random"),
    ("all_reduce",    "collective"),
    ("reduce_scatter", "collective"),
    ("all_gather",    "collective"),
    ("all_to_all",    "collective"),
    ("moe_dispatch",  "collective"),
    ("moe_combine",   "collective"),
)


@pytest.mark.parametrize("name,expected", _LOCKED_EFFECT_SENTINELS)
def test_locked_effect_sentinel(name: str, expected: str) -> None:
    """High-traffic op effects locked at the Audit-B landing.  Any
    rename / restructuring that changes one of these is a contract
    break — surface immediately."""
    rows = {r.name: r for r in collect_effect_audit()}
    row = rows.get(name)
    assert row is not None, f"op {name!r} not in OP_SPECS"
    assert row.declared_effect == expected, (
        f"op {name!r} effect regressed: declared={row.declared_effect!r}, "
        f"expected={expected!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Distribution floors (locked 2026-05-22)
# ─────────────────────────────────────────────────────────────────────────

_DISTRIBUTION_FLOORS = {
    "pure":       200,  # most ops are pure
    "random":       3,  # dropout + 2 RNG samplers
    "movement":     2,  # prefetch + async_copy
    "state":       20,  # KV cache + memory_state family
    "collective":   7,  # all-reduce/all-gather/reduce-scatter/all-to-all/moe
}


@pytest.mark.parametrize(
    "effect,floor", sorted(_DISTRIBUTION_FLOORS.items())
)
def test_effect_distribution_floor(effect: str, floor: int) -> None:
    distribution = effect_distribution()
    actual = distribution.get(effect, 0)
    assert actual >= floor, (
        f"Effect `{effect}` count dropped below floor {floor} "
        f"(got {actual})."
    )


def test_top_effect_count_remains_zero() -> None:
    distribution = effect_distribution()
    assert distribution.get("top", 0) == 0, (
        "Some op regressed to `top` fallback — narrow it."
    )


# ─────────────────────────────────────────────────────────────────────────
# Determinism coverage
# ─────────────────────────────────────────────────────────────────────────


def test_determinism_aware_ops_floor() -> None:
    """At least 20 ops carry deterministic-aware policies today
    (Sprint C2 NumericPolicy work).  Catches a regression where the
    policy factory loses the determinism field."""
    det = determinism_aware_ops()
    assert len(det) >= 20, (
        f"Determinism-aware op count dropped below 20 (got {len(det)}). "
        f"Inspect `_NUMERIC_POLICY_BY_NAME_FACTORIES` in "
        f"`primitive_coverage.py`."
    )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_exists() -> None:
    assert DASHBOARD.exists(), (
        f"Generated dashboard missing: "
        f"{DASHBOARD.relative_to(REPO_ROOT)}.  Regenerate via "
        f"`tessera.compiler.effect_audit.write_dashboard()`."
    )


def test_dashboard_matches_live_registry() -> None:
    if not DASHBOARD.exists():
        pytest.skip("dashboard not yet generated")
    live = render_dashboard()
    on_disk = DASHBOARD.read_text()
    if live == on_disk:
        return
    live_lines = live.splitlines()
    disk_lines = on_disk.splitlines()
    first_diff = next(
        (i for i, (l, d) in enumerate(zip(live_lines, disk_lines))
         if l != d),
        min(len(live_lines), len(disk_lines)),
    )
    pytest.fail(
        f"Effect-audit dashboard drift at line {first_diff + 1}: "
        f"on-disk has {disk_lines[first_diff]!r}, live has "
        f"{live_lines[first_diff]!r}.  Regenerate the dashboard."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    assert DASHBOARD.exists()
    text = DASHBOARD.read_text()
    for phrase in (
        "# Effect Lattice + Determinism Audit",
        "## Headline",
        "## Effect distribution",
        "## TSOL spec anchor cross-check",
        "## Determinism-aware numeric policies",
        "## Per-anchor verification (TSOL effect map)",
    ):
        assert phrase in text, (
            f"effect audit dashboard missing canonical phrase {phrase!r}"
        )
