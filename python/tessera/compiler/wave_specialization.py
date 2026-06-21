"""tessera.compiler.wave_specialization — target-parametric warp/wave role split.

Tessera's ``WarpSpecializationPass`` (C++) encodes the SM_90 Hopper model:
a dedicated *producer* warp issues TMA loads and signals an mbarrier while a
*consumer* warpgroup runs WGMMA.  The moonmath CDNA3 attention writeup shows the
same structural idea pays off on AMD wave64 — but in a *ping-pong* form: eight
waves split into two groups of four, and the groups **swap** roles each phase
(group A does PV+QK matmul while group B streams K + does softmax, then they
trade).  Same primitive (producer/consumer role split), two parameterizations.

This module is the **target-parametric descriptor** that captures both: how many
wave groups, how many waves each, the role each group holds, and — for the CDNA
ping-pong schedule — how roles rotate across phases plus the barrier count.  It
is the design contract the warp-specialization lowering should consume so the
role split is no longer hard-coded to NVIDIA; the C++ pass reframe rides on top.

Pure data, stdlib only (no ``tessera`` imports) — duck-types a target profile via
``threads_per_wave`` / ``waves_per_cu`` so it stays backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── roles ────────────────────────────────────────────────────────────────────

PRODUCER = "producer"
CONSUMER = "consumer"
_ROLES: tuple[str, str] = (PRODUCER, CONSUMER)


class TesseraWaveSpecError(ValueError):
    """Raised for a malformed wave-specialization plan."""


# ── wave group ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WaveGroup:
    """A contiguous group of waves holding one initial role.

    ``wave_ids`` are the lane-group (wave/warp) indices in this group; ``role``
    is the group's role in phase 0 (it rotates across phases when the enclosing
    plan is ping-pong).
    """

    group_id: int
    wave_ids: tuple[int, ...]
    role: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise TesseraWaveSpecError(
                f"WaveGroup: role must be one of {_ROLES}, got {self.role!r}")
        if not self.wave_ids:
            raise TesseraWaveSpecError("WaveGroup: wave_ids must be non-empty")

    @property
    def num_waves(self) -> int:
        return len(self.wave_ids)


# ── plan ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WaveSpecializationPlan:
    """A target-parametric warp/wave specialization schedule.

    ``ping_pong=True`` (CDNA attention) → groups swap roles every phase, so the
    matrix-core group and the memory/softmax group alternate; ``ping_pong=False``
    (SM_90) → roles are fixed (a dedicated producer warp + consumer warpgroup).
    """

    total_waves: int
    groups: tuple[WaveGroup, ...]
    ping_pong: bool
    threads_per_wave: int
    barriers_per_iter: int

    def __post_init__(self) -> None:
        if self.total_waves <= 0:
            raise TesseraWaveSpecError(
                f"total_waves must be positive, got {self.total_waves}")
        if not self.groups:
            raise TesseraWaveSpecError("a plan needs at least one wave group")
        covered = [w for g in self.groups for w in g.wave_ids]
        if sorted(covered) != list(range(self.total_waves)):
            raise TesseraWaveSpecError(
                f"groups must partition wave ids 0..{self.total_waves - 1} "
                f"exactly once; got {sorted(covered)}")
        roles = {g.role for g in self.groups}
        if roles != set(_ROLES):
            raise TesseraWaveSpecError(
                f"a specialization needs both roles {_ROLES}; got {sorted(roles)}")
        if self.barriers_per_iter < 1:
            raise TesseraWaveSpecError(
                f"barriers_per_iter must be >= 1, got {self.barriers_per_iter}")

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_phases(self) -> int:
        """Distinct role configurations per K-loop iteration.

        Ping-pong cycles through ``len(roles)`` phases (2); a fixed-role plan has
        a single phase."""
        return len(_ROLES) if self.ping_pong else 1

    def role_at_phase(self, group_id: int, phase: int) -> str:
        """The role ``group_id`` holds in ``phase``.

        Fixed-role plans always return the group's initial role.  Ping-pong
        plans rotate: a group's role advances by ``phase`` through the role list,
        so two groups always hold *complementary* roles each phase.
        """
        if not (0 <= group_id < self.num_groups):
            raise TesseraWaveSpecError(
                f"group_id {group_id} out of range (0..{self.num_groups - 1})")
        if phase < 0:
            raise TesseraWaveSpecError(f"phase must be >= 0, got {phase}")
        base = self.groups[group_id]
        if not self.ping_pong:
            return base.role
        start = _ROLES.index(base.role)
        return _ROLES[(start + phase) % len(_ROLES)]

    def to_mlir_attrs(self) -> str:
        """Inline attr dict describing the specialization for the lowering pass."""
        roles = ", ".join(f'"{g.role}"' for g in self.groups)
        return (
            "{"
            f"tessera.wave_groups = {self.num_groups} : i64, "
            f"tessera.waves_per_group = {self.groups[0].num_waves} : i64, "
            f"tessera.threads_per_wave = {self.threads_per_wave} : i64, "
            f'tessera.schedule = "{"ping_pong" if self.ping_pong else "fixed"}", '
            f"tessera.barriers_per_iter = {self.barriers_per_iter} : i64, "
            f"tessera.initial_roles = [{roles}]"
            "}"
        )

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "wave_specialization_plan",
            "total_waves": self.total_waves,
            "num_groups": self.num_groups,
            "waves_per_group": self.groups[0].num_waves,
            "threads_per_wave": self.threads_per_wave,
            "ping_pong": self.ping_pong,
            "num_phases": self.num_phases,
            "barriers_per_iter": self.barriers_per_iter,
            "groups": [
                {"group_id": g.group_id, "wave_ids": list(g.wave_ids), "role": g.role}
                for g in self.groups
            ],
        }


# ── constructors ─────────────────────────────────────────────────────────────


def plan_wave_specialization(
    *,
    total_waves: int,
    num_groups: int = 2,
    threads_per_wave: int = 64,
    ping_pong: bool = True,
    barriers_per_iter: int | None = None,
) -> WaveSpecializationPlan:
    """Build an evenly-partitioned specialization plan.

    ``total_waves`` waves are split into ``num_groups`` equal contiguous groups
    whose initial roles alternate producer/consumer.  ``ping_pong`` makes the
    groups swap roles each phase (the CDNA attention schedule); otherwise roles
    are fixed (the SM_90 schedule).  ``barriers_per_iter`` defaults to 2 for a
    ping-pong plan (phase handoff + iteration boundary) and 1 for a fixed plan.
    """
    if num_groups != 2:
        # Producer/consumer is a two-role split; >2 groups would need a role
        # assignment policy we don't model yet.  Fail loudly rather than guess.
        raise TesseraWaveSpecError(
            f"plan_wave_specialization currently supports exactly 2 groups "
            f"(producer/consumer); got num_groups={num_groups}")
    if total_waves <= 0 or total_waves % num_groups != 0:
        raise TesseraWaveSpecError(
            f"total_waves ({total_waves}) must be a positive multiple of "
            f"num_groups ({num_groups})")
    if threads_per_wave <= 0:
        raise TesseraWaveSpecError(
            f"threads_per_wave must be positive, got {threads_per_wave}")
    per_group = total_waves // num_groups
    groups = tuple(
        WaveGroup(
            group_id=g,
            wave_ids=tuple(range(g * per_group, (g + 1) * per_group)),
            role=_ROLES[g % len(_ROLES)],
        )
        for g in range(num_groups)
    )
    if barriers_per_iter is None:
        barriers_per_iter = 2 if ping_pong else 1
    return WaveSpecializationPlan(
        total_waves=total_waves,
        groups=groups,
        ping_pong=ping_pong,
        threads_per_wave=threads_per_wave,
        barriers_per_iter=barriers_per_iter,
    )


def cdna_attention_plan(
    profile: Any = None, *, total_waves: int = 8
) -> WaveSpecializationPlan:
    """The moonmath CDNA3 attention schedule: 8 waves, 2 ping-pong groups.

    "Eight waves per workgroup arranged as two groups of four waves" that swap
    matrix-core / memory+softmax roles each phase, with two barriers per
    iteration.  ``profile`` (an ``ROCmTargetProfile``, optional) supplies
    ``threads_per_wave`` (64 on CDNA) when given; otherwise wave64 is assumed.
    """
    tpw = getattr(profile, "threads_per_wave", 64) if profile is not None else 64
    return plan_wave_specialization(
        total_waves=total_waves,
        num_groups=2,
        threads_per_wave=tpw,
        ping_pong=True,
    )


def sm90_attention_plan(*, producer_warps: int = 1, consumer_warps: int = 4) -> WaveSpecializationPlan:
    """The SM_90 schedule: a fixed producer warp + a consumer warpgroup.

    Roles do not rotate (``ping_pong=False``): one warp issues TMA loads, the
    consumer warpgroup runs WGMMA.  ``threads_per_wave`` is 32 (NVIDIA warp).
    """
    if producer_warps <= 0 or consumer_warps <= 0:
        raise TesseraWaveSpecError(
            "producer_warps and consumer_warps must both be positive")
    total = producer_warps + consumer_warps
    producer = WaveGroup(
        group_id=0, wave_ids=tuple(range(producer_warps)), role=PRODUCER)
    consumer = WaveGroup(
        group_id=1,
        wave_ids=tuple(range(producer_warps, total)),
        role=CONSUMER,
    )
    return WaveSpecializationPlan(
        total_waves=total,
        groups=(producer, consumer),
        ping_pong=False,
        threads_per_wave=32,
        barriers_per_iter=1,
    )


__all__ = [
    "PRODUCER",
    "CONSUMER",
    "WaveGroup",
    "WaveSpecializationPlan",
    "TesseraWaveSpecError",
    "plan_wave_specialization",
    "cdna_attention_plan",
    "sm90_attention_plan",
]
