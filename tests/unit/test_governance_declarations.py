"""Drift gates for the compiler governance decisions adopted in W0.8.

Two of the six governance decisions in ``CLAUDE.md`` are mechanically
checkable, and the plan is explicit that they must be *tests, not
conventions* — the stated risk is that the rules get ignored under delivery
pressure (``INTEGRATED_COMPILER_PLAN.md`` §7).

* **Decision #29 — a declaration must have a consumer.** Every
  ``primitive_coverage`` contract axis names the pass or code path that reads
  it. An axis with no consumer reads as a closed contract in the generated
  dashboards while carrying nothing.

* **Decision #31 — one implementation per boundary.** Each IR level boundary
  has exactly one production lowering; a second is a declared oracle with a
  differential test, or it is deleted. The mechanical half checked here is
  that no two ODS files define the same dialect name.

Both gates are **ratchets**. The known-open cases from the architecture
reviews are listed explicitly with an owning plan item, and the test asserts
the waiver list does not grow. Closing a wave item means deleting its waiver
entry, not editing the assertion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.primitive_coverage import CONTRACT_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Decision #29 — a declaration must have a consumer
# ─────────────────────────────────────────────────────────────────────────────

# Each contract axis maps to the code path that actually reads it. A consumer
# is a repo-relative path; the gate asserts it exists, so a rename that orphans
# an axis fails here rather than silently leaving a dangling claim.
AXIS_CONSUMERS: dict[str, str] = {
    "math_semantics": "python/tessera/compiler/op_catalog.py",
    "dtype_layout_rule": "python/tessera/dtype.py",
    "vjp": "python/tessera/autodiff/vjp.py",
    "jvp": "python/tessera/autodiff/jvp.py",
    "transpose_rule": "python/tessera/autodiff/vjp.py",
    "sharding_rule": "python/tessera/sharding.py",
    "masking_effect_rule": "python/tessera/compiler/effects.py",
    "lowering_rule": "python/tessera/compiler/matmul_pipeline.py",
    "backend_kernel": "python/tessera/compiler/backend_manifest.py",
    "tests": "python/tessera/compiler/primitive_coverage.py",
}

# Axes that are DECLARED but whose nominal consumer does not actually read
# them — root cause A in the integrated plan. Each entry names the wave item
# that closes it. This dict may shrink, never grow.
AXIS_CONSUMER_WAIVERS: dict[str, str] = {
    # `vmap` is a Python for-loop; it never consults the declared rule.
    "batching_rule": "W1 (Autodiff §B3) — batching rule registry with a real vmap consumer",
    # `_infer_result_type` is a five-case if-chain ending in operand_types[0].
    "shape_rule": "W1.2 — one shape-rule registry owned by op_catalog.OpSpec",
}


def test_every_contract_axis_names_a_consumer() -> None:
    """Decision #29: no contract axis may exist without a named consumer."""
    declared = set(CONTRACT_FIELDS)
    accounted = set(AXIS_CONSUMERS) | set(AXIS_CONSUMER_WAIVERS)

    unaccounted = declared - accounted
    assert not unaccounted, (
        "Decision #29 violation: these primitive_coverage axes declare a "
        f"contract with no named consumer: {sorted(unaccounted)}. Either wire "
        "a consumer and add it to AXIS_CONSUMERS, or delete the axis. If the "
        "consumer is genuinely not built yet, add an AXIS_CONSUMER_WAIVERS "
        "entry naming the wave item that closes it."
    )

    stale = accounted - declared
    assert not stale, (
        f"These axes are mapped here but no longer declared in "
        f"CONTRACT_FIELDS: {sorted(stale)}. Remove the stale mapping."
    )


@pytest.mark.parametrize("axis,consumer", sorted(AXIS_CONSUMERS.items()))
def test_declared_axis_consumer_exists(axis: str, consumer: str) -> None:
    """A named consumer must be a real file, so renames cannot orphan an axis."""
    path = REPO_ROOT / consumer
    assert path.exists(), (
        f"Decision #29: axis {axis!r} names consumer {consumer!r}, which does "
        "not exist. Update AXIS_CONSUMERS to the new location or delete the axis."
    )


def test_axis_consumer_waivers_do_not_grow() -> None:
    """The unconsumed-declaration list is a ratchet.

    Closing a wave item means deleting its waiver entry. Adding one means a new
    Decision #29 violation was introduced, which is what this gate exists to
    prevent.
    """
    assert len(AXIS_CONSUMER_WAIVERS) <= 2, (
        "A new unconsumed contract axis was added. Decision #29 requires a "
        "declaration to have a consumer; wire one instead of widening the waiver."
    )
    for axis, owner in AXIS_CONSUMER_WAIVERS.items():
        assert axis in CONTRACT_FIELDS, f"Waived axis {axis!r} is not declared."
        assert re.match(r"^W\d", owner), (
            f"Waiver for {axis!r} must name the owning wave item (e.g. 'W1.2 — …'); "
            f"got {owner!r}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decision #31 — one implementation per boundary
# ─────────────────────────────────────────────────────────────────────────────

# ODS files under these roots are scanned for `let name = "..."` dialect
# declarations. Two files declaring the same dialect name is the accretion
# pattern that produced the duplicate Queue.td/Attn.td trap (W0.6).
ODS_SCAN_ROOTS = ("src",)

_DIALECT_NAME_RE = re.compile(r'let\s+name\s*=\s*"([A-Za-z0-9_.]+)"')


def _iter_ods_dialect_declarations() -> dict[str, list[Path]]:
    """Map dialect name → the ODS files declaring it."""
    by_name: dict[str, list[Path]] = {}
    for root in ODS_SCAN_ROOTS:
        for td in (REPO_ROOT / root).rglob("*.td"):
            # `archive/` is excluded from all build targets by policy.
            if "archive" in td.parts:
                continue
            try:
                text = td.read_text(encoding="utf-8", errors="replace")
            except OSError:  # pragma: no cover - unreadable file
                continue
            # Only a `Dialect` record's `let name` counts; op/type records use
            # mnemonic instead, so scope the match to files defining a Dialect.
            if not re.search(r":\s*Dialect\s*\{", text):
                continue
            for name in _DIALECT_NAME_RE.findall(text):
                by_name.setdefault(name, []).append(td.relative_to(REPO_ROOT))
    return by_name


def test_no_dialect_is_declared_by_two_ods_files() -> None:
    """Decision #31: exactly one ODS file may define a given dialect.

    Two `.td` files declaring the same dialect name is a live trap: the build
    wires one, docs cite the other, and edits land in the copy nothing
    compiles. This is what W0.6 removed.
    """
    duplicates = {
        name: paths
        for name, paths in _iter_ods_dialect_declarations().items()
        if len(paths) > 1
    }
    assert not duplicates, (
        "Decision #31 violation: these dialects are declared by more than one "
        "ODS file:\n"
        + "\n".join(
            f"  {name}: {[str(p) for p in paths]}" for name, paths in sorted(duplicates.items())
        )
        + "\nKeep the one the build wires and delete the other; repoint any docs "
        "that cite the deleted path."
    )


def test_ods_scan_finds_dialects_at_all() -> None:
    """Guard the guard: a scan that silently matches nothing proves nothing."""
    found = _iter_ods_dialect_declarations()
    assert len(found) >= 5, (
        f"ODS dialect scan found only {len(found)} dialects, which suggests the "
        "scan pattern broke rather than that the tree has no dialects."
    )
