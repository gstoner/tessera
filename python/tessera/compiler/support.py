"""Public query API for per-op compiler support readiness.

This module is a **thin wrapper** over :mod:`tessera.compiler.audit` —
the source of truth for the 8-axis support matrix.  It exposes the
same data Python developers ask about most often (without forcing
them to learn the audit table's row shape):

  * :func:`support(op_name)` → :class:`OpSupport` with per-axis status
    + per-target breakdown.
  * :func:`tier(op_name, target=None)` → :class:`Tier` rollup —
    "is this op ready?" answered as one enum value.
  * :func:`is_native_supported(op_name, target=...)` → ``bool``
    convenience predicate.

Design notes
------------

* **No parallel registry.**  Everything is derived from
  :func:`tessera.compiler.audit.support_row_for`,
  :data:`tessera.compiler.backend_manifest.manifest_for`, and
  :data:`tessera.compiler.capabilities.TARGET_CAPABILITIES`.  When a
  new fused kernel lands or a new target activates, the query API
  picks it up without an edit here.

* **Conservative tiers.**  ``tier(op)`` returns the best tier across
  any target.  ``tier(op, target="apple_gpu")`` is target-specific.
  Callers asking "is matmul fast?" rarely mean "fast on every
  target", so the rollup defaults to optimistic; the target-specific
  form is for serving-time decisions.

* **Enum, not strings.**  :class:`Tier` is a :class:`enum.Enum` so
  callers can compare with ``is`` and IDEs auto-complete the
  options.  String representation matches the enum value
  (``str(Tier.NATIVE_READY) == "Tier.NATIVE_READY"``); the
  per-target axis status strings (``"fused"`` / ``"reference"`` /
  ``"planned"``) flow through unchanged for callers who need the
  raw audit vocabulary.

Examples
--------

::

    from tessera import compiler

    info = compiler.support("matmul")
    print(info.best_tier)                  # Tier.NATIVE_READY
    print(info.for_target("apple_gpu").tier)  # Tier.NATIVE_READY
    print(info.for_target("cpu").tier)        # Tier.REFERENCE_ONLY

    compiler.tier("matmul")                # Tier.NATIVE_READY
    compiler.tier("matmul", target="apple_gpu")  # Tier.NATIVE_READY
    compiler.is_native_supported("matmul", target="apple_gpu")  # True
    compiler.is_native_supported("complex_log", target="apple_gpu")  # False
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from . import audit as _audit
from . import backend_manifest as _bm
from . import capabilities as _cap


class Tier(Enum):
    """Single-tier readiness rollup.

    Order is strictly best-to-worst — :meth:`best_of` and the
    rank-comparison helpers rely on it.
    """

    NATIVE_READY = "native_ready"
    """Fused kernel at this target *and* runtime reports ``ready``."""

    REFERENCE_ONLY = "reference_only"
    """Runs via the numpy reference path.  Correct, but not a native
    claim — no fused kernel is dispatched."""

    ARTIFACT_ONLY = "artifact_only"
    """Compiles to IR / a target artifact but cannot execute the
    workload today."""

    PLANNED = "planned"
    """Registered in the audit / coverage table but the execution
    path isn't built yet."""

    @classmethod
    def _rank(cls) -> dict["Tier", int]:
        return {
            cls.NATIVE_READY: 0,
            cls.REFERENCE_ONLY: 1,
            cls.ARTIFACT_ONLY: 2,
            cls.PLANNED: 3,
        }

    def is_at_least(self, other: "Tier") -> bool:
        """``True`` iff ``self`` is at least as good as ``other``."""

        rank = self._rank()
        return rank[self] <= rank[other]

    @classmethod
    def best_of(cls, tiers: Iterable["Tier"]) -> "Tier":
        """Pick the best tier from an iterable; defaults to ``PLANNED``
        when empty."""

        rank = cls._rank()
        best: Tier = cls.PLANNED
        for t in tiers:
            if rank[t] < rank[best]:
                best = t
        return best


# ─────────────────────────────────────────────────────────────────────
# Per-target / per-op readiness rows.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetSupport:
    """Readiness of a single op on a single target."""

    target: str
    target_ir: str
    """``fused`` / ``reference`` / ``compileable`` / ``artifact_only`` /
    ``planned`` — the raw backend-manifest status flowing through."""
    runtime: str
    """``ready`` / ``reference`` / ``unsupported`` / ``unknown`` — the
    raw capability registry's runtime claim for this op on this
    target."""
    tier: Tier
    """Single-tier rollup of the two axes above."""


@dataclass(frozen=True)
class OpSupport:
    """Per-op readiness — the 8-axis audit row plus per-target breakdown."""

    op_name: str
    family: str

    # 8-axis status (mirrors audit.LAYER_AXES).  Pulled verbatim from
    # the audit so callers see exactly what the support table renders.
    api: str
    frontend: str
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    runtime: str
    bench: str

    targets: tuple[TargetSupport, ...]
    best_tier: Tier

    def for_target(self, target: str) -> TargetSupport:
        """Return the row for one target.  Raises :class:`KeyError`
        for unknown targets."""

        target = _cap.normalize_target(target) if target in _cap.TARGET_CAPABILITIES else target
        for ts in self.targets:
            if ts.target == target:
                return ts
        raise KeyError(
            f"no support entry for op={self.op_name!r} on target={target!r}; "
            f"known targets: {tuple(ts.target for ts in self.targets)!r}"
        )

    def as_dict(self) -> dict[str, object]:
        """Render as a plain dict — useful for JSON serialization
        in CI artifacts and tprof exports."""

        return {
            "op_name": self.op_name,
            "family": self.family,
            "axes": {
                "api": self.api,
                "frontend": self.frontend,
                "graph_ir": self.graph_ir,
                "schedule_ir": self.schedule_ir,
                "tile_ir": self.tile_ir,
                "target_ir": self.target_ir,
                "runtime": self.runtime,
                "bench": self.bench,
            },
            "best_tier": self.best_tier.value,
            "targets": [
                {
                    "target": ts.target,
                    "target_ir": ts.target_ir,
                    "runtime": ts.runtime,
                    "tier": ts.tier.value,
                }
                for ts in self.targets
            ],
        }


# ─────────────────────────────────────────────────────────────────────
# Tier derivation.
# ─────────────────────────────────────────────────────────────────────

# Statuses that count as "fused / native-equivalent" on the tile/target axes.
_NATIVE_LIKE_TARGET_IR = frozenset({"fused"})
# Statuses that count as "executable but reference-grade".
_REFERENCE_LIKE_TARGET_IR = frozenset({"reference"})
# Statuses that compile to IR but don't actually run.
_ARTIFACT_LIKE_TARGET_IR = frozenset({"compileable", "artifact_only"})


def _derive_tier(target_ir: str, runtime: str) -> Tier:
    """Roll the two execution axes up into a single Tier.

    Precedence:

    * ``target_ir=fused`` is the **canonical** native-ready claim.  If
      a fused kernel exists in the backend manifest, the op is
      :class:`Tier.NATIVE_READY` unless the capability registry
      actively denies it (``runtime=unsupported``).  The capability
      registry's ``unknown`` is a *no-opinion* answer (the target has
      no explicit ``OpCapability`` row for this op) — it does not
      negate the manifest's fused claim.  This matches how
      :func:`tessera.compiler.audit.support_row_for` ranks the
      ``tile_ir`` / ``target_ir`` axes above the ``runtime`` axis for
      tier purposes.

    * ``target_ir=reference`` or ``runtime=reference`` → numpy/CPU
      fallback runs but no native dispatch happens.

    * ``target_ir=compileable`` / ``artifact_only`` → emits IR but
      does not execute.

    * Otherwise → planned.
    """

    if runtime == "unsupported":
        return Tier.PLANNED
    if target_ir in _NATIVE_LIKE_TARGET_IR:
        return Tier.NATIVE_READY
    if runtime == "reference" or target_ir in _REFERENCE_LIKE_TARGET_IR:
        return Tier.REFERENCE_ONLY
    if target_ir in _ARTIFACT_LIKE_TARGET_IR:
        return Tier.ARTIFACT_ONLY
    if runtime in ("ready", "fused"):
        # Runtime claims ready but target_ir is silent — promote to
        # NATIVE_READY (matches how OP_SPECS-only catalog entries
        # interact with the x86/cpu reference path).
        return Tier.NATIVE_READY
    return Tier.PLANNED


def _per_target_rows(op_name: str) -> tuple[TargetSupport, ...]:
    """Build per-target readiness from backend manifest + capabilities."""

    # Honor M7 alias map — ``mobius`` / ``stereographic`` look up the
    # ``complex_*`` entries in the backend manifest.
    backend_name = _audit._backend_lookup_name(op_name)

    # Collect every target the manifest knows about for this op.
    manifest_entries = _bm.manifest_for(backend_name)
    target_status: dict[str, str] = {e.target: e.status for e in manifest_entries}

    rows: list[TargetSupport] = []
    for target_name, target_cap in _cap.TARGET_CAPABILITIES.items():
        target_ir = target_status.get(target_name, "planned")
        # Pull the per-target runtime claim from the capability registry.
        # Mirrors _axis_runtime in audit.py but resolves per-target
        # rather than picking the best across targets.
        op_caps = (
            target_cap.supported_ops.get(f"tessera.{op_name}")
            or target_cap.supported_ops.get(op_name)
        )
        if op_caps is not None:
            runtime = op_caps.runtime_status
        elif (
            "reference_execution" in target_cap.features
            and op_name in _audit.OP_SPECS
        ):
            runtime = "reference"
        else:
            runtime = "unknown"
        tier_val = _derive_tier(target_ir, runtime)
        rows.append(TargetSupport(
            target=target_name,
            target_ir=target_ir,
            runtime=runtime,
            tier=tier_val,
        ))
    # Sort: best tier first, then alphabetical so the output is stable.
    rows.sort(key=lambda r: (Tier._rank()[r.tier], r.target))
    return tuple(rows)


# ─────────────────────────────────────────────────────────────────────
# Public query functions.
# ─────────────────────────────────────────────────────────────────────


def support(op_name: str) -> OpSupport:
    """Return the per-op readiness picture.

    Combines the 8-axis audit row (api / frontend / graph_ir / etc.)
    with per-target target_ir + runtime breakdown.  The audit table
    in ``docs/audit/generated/support_table.md`` is the same data
    rendered as a Markdown grid.
    """

    row = _audit.support_row_for(op_name)
    targets = _per_target_rows(op_name)
    best_tier = Tier.best_of(t.tier for t in targets)
    return OpSupport(
        op_name=op_name,
        family=row.family,
        api=row.cells["api"].status,
        frontend=row.cells["frontend"].status,
        graph_ir=row.cells["graph_ir"].status,
        schedule_ir=row.cells["schedule_ir"].status,
        tile_ir=row.cells["tile_ir"].status,
        target_ir=row.cells["target_ir"].status,
        runtime=row.cells["runtime"].status,
        bench=row.cells["bench"].status,
        targets=targets,
        best_tier=best_tier,
    )


def tier(op_name: str, *, target: str | None = None) -> Tier:
    """Single-tier rollup.

    ``target=None``  → best tier across any target (optimistic).
    ``target=<name>`` → that target's tier (serving-time view).
    """

    info = support(op_name)
    if target is None:
        return info.best_tier
    return info.for_target(target).tier


def is_native_supported(op_name: str, *, target: str) -> bool:
    """True iff the op has a fused kernel + ready runtime on ``target``."""

    return tier(op_name, target=target) is Tier.NATIVE_READY


def known_targets() -> tuple[str, ...]:
    """The set of targets the query API knows about.

    Derived from :data:`tessera.compiler.capabilities.TARGET_CAPABILITIES`
    so this stays in sync with whatever the capability registry ships.
    """

    return tuple(sorted(_cap.TARGET_CAPABILITIES))


__all__ = [
    "OpSupport",
    "TargetSupport",
    "Tier",
    "is_native_supported",
    "known_targets",
    "support",
    "tier",
]
