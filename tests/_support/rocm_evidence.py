"""Re-derivation of ROCm retune verdicts from the evidence they retained.

The gfx1151 retune baseline states its gate as prose --
``"promotion_gate": "correct oracle plus shape-specific repeated-median gate"``
-- so unlike Apple's `promotion_rules` there is no threshold block to hold a
decision to. Each family instead carries a one-line `decision` and a `rows`
list, and until now nothing checked that the two agree.

**Three of six families can be checked today; two cannot, and the reason is
worth stating precisely.** `f32_gemm` and `grouped_gemm` record only the
*chosen* configuration per shape -- one tile per shape, no speedup field -- so
a decision naming a threshold ("2x2 through 256", "tn=1 below 64k") has no
losing measurement to be checked against. That is not a prose problem; it is a
missing-data problem, and closing it needs a re-record on gfx1151 that keeps
both candidates per shape. `kv_moe_transport` has no `rows` at all.

What the remaining families do carry is a per-row speedup and win rate, which
is enough to check the direction of a verdict: a family that promotes should
show every shape winning, and one that rejects should not. That check runs in
both directions on purpose -- a rejection is a claim too, and if a regressed
recording showed the rejected candidate winning outright, nothing today would
notice.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: Field names carrying a candidate-vs-incumbent ratio. They differ per family
#: (`paired_speedup` in g6b, `device_speedup`/`e2e_speedup` elsewhere), which is
#: itself a schema inconsistency worth folding in when these are next recorded.
SPEEDUP_FIELDS = ("device_speedup", "e2e_speedup", "paired_speedup")

#: Fraction of repeated trials the candidate won. `win_rate` in the promoted
#: families, `device_win_rate` in g6c.
WIN_RATE_FIELDS = ("win_rate", "device_win_rate")


def row_speedups(row: Mapping[str, Any]) -> list[float]:
    return [float(row[name]) for name in SPEEDUP_FIELDS
            if isinstance(row.get(name), (int, float))]


def row_win_rates(row: Mapping[str, Any]) -> list[float]:
    return [float(row[name]) for name in WIN_RATE_FIELDS
            if isinstance(row.get(name), (int, float))]


def family_is_derivable(family: Mapping[str, Any]) -> bool:
    """Whether this family retained enough to check its own decision."""
    rows = family.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    return any(row_speedups(row) or row_win_rates(row) for row in rows)


def promotion_verdict_violations(
    family: Mapping[str, Any], *, expect_promotion: bool,
) -> list[str]:
    """Whether a family's rows support the direction of its own `decision`.

    ``expect_promotion`` is the caller's reading of the prose, stated
    explicitly rather than parsed, so a reworded decision fails the caller's
    own text assertion instead of silently changing what is enforced here.

    A promotion requires **every** row to clear 1.0x on every ratio it records
    and to win every repeated trial it reports. A rejection requires at least
    one row not to -- otherwise the evidence says the candidate won and the
    verdict says it lost, and one of the two is wrong.
    """
    rows = family.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["missing_rows"]

    violations: list[str] = []
    unanimous = True
    for index, row in enumerate(rows):
        speedups, wins = row_speedups(row), row_win_rates(row)
        if not speedups and not wins:
            violations.append(f"row[{index}]:no_comparative_evidence")
            continue
        if any(value <= 1.0 for value in speedups) or any(
                value < 1.0 for value in wins):
            unanimous = False
            if expect_promotion:
                violations.append(
                    f"row[{index}]:promoted_without_a_win"
                    f"(speedups={speedups}, win_rates={wins})")
    if not expect_promotion and unanimous:
        violations.append("rejected_although_every_row_wins")
    return violations


def winner_only_families(payload: Mapping[str, Any],
                         names: Sequence[str]) -> list[str]:
    """Families whose decision cannot be checked because only the winner was kept."""
    return [name for name in names
            if isinstance(payload.get(name), Mapping)
            and not family_is_derivable(payload[name])]
