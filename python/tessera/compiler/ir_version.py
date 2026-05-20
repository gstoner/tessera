"""Graph IR schema versioning (Issue 2, 2026-05-20).

Tessera's Graph IR carries an explicit schema version so future
field removals / renames have a controlled migration path.  Today
only version ``"1.0"`` exists; the :func:`migrate` function is a
no-op identity transform.  When the first migration is needed,
add a per-version migrator function and update
:data:`IR_VERSION_HISTORY`.

When to bump
------------

The version-bump triggers are deliberately narrow:

  * **Major** (``"2.0"``) — a field is removed or renamed.  Every
    serialized IR from older versions must be migrated.  Breaks
    backward compatibility — requires a migrator + a release-note
    entry.
  * **Minor** (``"1.1"``) — a field is added (with a default) or
    a passive semantic is changed.  Backward-compatible: old
    serialized IRs still parse and produce the same execution.

Phase A explicitly *adds optional fields*, so it stays in
``"1.0"`` (every Phase A field has a default and old IRs round-trip
unchanged).  The first migration trigger is a future removal /
rename — not Phase A, not Phase B, not Phase C.

Public surface
--------------

  * :data:`GRAPH_IR_SCHEMA_VERSION` — the current schema string.
  * :data:`IR_VERSION_HISTORY` — every past version + a one-line
    note about what changed.
  * :func:`migrate(module, *, from_version)` — bring a module up
    to :data:`GRAPH_IR_SCHEMA_VERSION`.  No-op today.

Drift gate
----------

``tests/unit/test_ir_version_contract.py`` enforces:

  * Version constant exists and is a non-empty string.
  * Version constant equals the latest entry in
    :data:`IR_VERSION_HISTORY`.
  * Adding a new ``IR_VERSION_HISTORY`` entry without bumping
    :data:`GRAPH_IR_SCHEMA_VERSION` fails the gate.
  * :func:`migrate` is the identity function when
    ``from_version == GRAPH_IR_SCHEMA_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_ir import GraphIRModule


# ─────────────────────────────────────────────────────────────────────
# Current schema version.  Bump when fields are removed or renamed.
# Adding optional fields stays in 1.0 because old IRs round-trip
# unchanged (the new fields just default to None/empty/zero).
# ─────────────────────────────────────────────────────────────────────


GRAPH_IR_SCHEMA_VERSION: str = "1.0"


@dataclass(frozen=True)
class _IRVersionEntry:
    """One row of the version-history table."""

    version: str
    landed: str
    """ISO date the version was finalized."""

    note: str
    """One-line summary of what changed."""


IR_VERSION_HISTORY: tuple[_IRVersionEntry, ...] = (
    _IRVersionEntry(
        version="1.0",
        landed="2026-05-19",
        note=(
            "Initial schema version coinciding with the Phase A "
            "optional-metadata landing (source_span / numeric_policy / "
            "value_kind / verification_facts on IROp; "
            "verification_facts / source_hash on GraphIRFunction; "
            "lane on GraphIRFunction).  All Phase A fields are "
            "optional with defaults — pre-1.0 producers don't exist "
            "as a labeled population, but anyone constructing IR "
            "without these fields stays valid."
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────
# Migration entry point.  Today this is a no-op because only one
# version exists; the function shape is here so callers don't have
# to rewrite their imports when v2.0 lands.
# ─────────────────────────────────────────────────────────────────────


def migrate(
    module: "GraphIRModule",
    *,
    from_version: str,
) -> "GraphIRModule":
    """Bring ``module`` up to :data:`GRAPH_IR_SCHEMA_VERSION`.

    Today only ``"1.0"`` exists, so this function returns the
    module unchanged.  When the first migration is needed, add a
    branch here that walks the module's ops / functions and
    applies the field rewrite.

    Raises
    ------
    ValueError
        When ``from_version`` is not in :data:`IR_VERSION_HISTORY`.
        A future-dated version (newer than this binary knows about)
        is rejected with a clear error rather than silently coerced.
    """

    known_versions = {entry.version for entry in IR_VERSION_HISTORY}
    if from_version not in known_versions:
        raise ValueError(
            f"unknown Graph IR schema version: {from_version!r}; "
            f"known versions: {sorted(known_versions)}"
        )
    if from_version == GRAPH_IR_SCHEMA_VERSION:
        return module
    # Future migrations land below.  Each branch:
    #   1. Walks ``module`` to apply field rewrites.
    #   2. Sets ``from_version`` to the next version.
    #   3. Falls through to the next branch.
    # No branches exist today because only 1.0 ships.
    raise NotImplementedError(
        f"no migrator from {from_version!r} to "
        f"{GRAPH_IR_SCHEMA_VERSION!r} — add one when the first "
        f"field-removal change lands"
    )


def latest_version_entry() -> _IRVersionEntry:
    """The most recent :class:`_IRVersionEntry`.  Equal to the
    entry whose ``.version == GRAPH_IR_SCHEMA_VERSION`` by
    construction (the drift gate enforces this)."""

    return IR_VERSION_HISTORY[-1]


__all__ = [
    "GRAPH_IR_SCHEMA_VERSION",
    "IR_VERSION_HISTORY",
    "latest_version_entry",
    "migrate",
]
