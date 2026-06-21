"""Unit tests for the target-parametric wave-specialization descriptor.

Ported from the moonmath CDNA3 attention writeup: producer/consumer role split
generalizes off SM_90.  CDNA runs it as an 8-wave, 2-group *ping-pong* (groups
swap matrix-core / memory+softmax roles each phase, 2 barriers/iter); SM_90 runs
it with *fixed* roles (a producer warp + a consumer warpgroup).
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_target import AMDArch, ROCmTargetProfile
from tessera.compiler.wave_specialization import (
    CONSUMER,
    PRODUCER,
    TesseraWaveSpecError,
    WaveGroup,
    WaveSpecializationPlan,
    cdna_attention_plan,
    plan_wave_specialization,
    sm90_attention_plan,
)


# ── CDNA ping-pong schedule (the article) ────────────────────────────────────


def test_cdna_plan_shape() -> None:
    p = cdna_attention_plan(ROCmTargetProfile(arch=AMDArch.GFX_942))
    assert p.total_waves == 8
    assert p.num_groups == 2
    assert p.groups[0].num_waves == 4 and p.groups[1].num_waves == 4
    assert p.threads_per_wave == 64  # CDNA wave64
    assert p.ping_pong is True
    assert p.num_phases == 2
    assert p.barriers_per_iter == 2  # phase handoff + iteration boundary


def test_cdna_roles_swap_each_phase() -> None:
    p = cdna_attention_plan()
    # Phase 0: group 0 produces, group 1 consumes.  Phase 1: they trade.
    assert p.role_at_phase(0, 0) == PRODUCER
    assert p.role_at_phase(1, 0) == CONSUMER
    assert p.role_at_phase(0, 1) == CONSUMER
    assert p.role_at_phase(1, 1) == PRODUCER


def test_cdna_groups_always_complementary() -> None:
    p = cdna_attention_plan()
    for phase in range(5):
        assert p.role_at_phase(0, phase) != p.role_at_phase(1, phase)


def test_cdna_plan_defaults_to_wave64_without_profile() -> None:
    assert cdna_attention_plan().threads_per_wave == 64


def test_cdna_plan_honors_total_waves_override() -> None:
    p = cdna_attention_plan(total_waves=16)
    assert p.total_waves == 16
    assert p.groups[0].num_waves == 8


# ── SM_90 fixed-role schedule ────────────────────────────────────────────────


def test_sm90_plan_is_fixed_role() -> None:
    s = sm90_attention_plan()
    assert s.ping_pong is False
    assert s.num_phases == 1
    assert s.threads_per_wave == 32  # NVIDIA warp
    assert s.barriers_per_iter == 1


def test_sm90_roles_do_not_rotate() -> None:
    s = sm90_attention_plan(producer_warps=1, consumer_warps=4)
    for phase in range(3):
        assert s.role_at_phase(0, phase) == PRODUCER
        assert s.role_at_phase(1, phase) == CONSUMER


def test_sm90_warp_counts() -> None:
    s = sm90_attention_plan(producer_warps=2, consumer_warps=6)
    assert s.total_waves == 8
    assert s.groups[0].num_waves == 2
    assert s.groups[1].num_waves == 6


# ── partition invariants + validation ────────────────────────────────────────


def test_groups_partition_all_waves() -> None:
    p = cdna_attention_plan()
    covered = sorted(w for g in p.groups for w in g.wave_ids)
    assert covered == list(range(8))


def test_plan_requires_both_roles() -> None:
    g0 = WaveGroup(group_id=0, wave_ids=(0, 1), role=PRODUCER)
    g1 = WaveGroup(group_id=1, wave_ids=(2, 3), role=PRODUCER)  # both producer
    with pytest.raises(TesseraWaveSpecError, match="both roles"):
        WaveSpecializationPlan(
            total_waves=4, groups=(g0, g1), ping_pong=True,
            threads_per_wave=64, barriers_per_iter=2,
        )


def test_plan_rejects_incomplete_partition() -> None:
    g0 = WaveGroup(group_id=0, wave_ids=(0, 1), role=PRODUCER)
    g1 = WaveGroup(group_id=1, wave_ids=(2,), role=CONSUMER)  # wave 3 missing
    with pytest.raises(TesseraWaveSpecError, match="partition"):
        WaveSpecializationPlan(
            total_waves=4, groups=(g0, g1), ping_pong=True,
            threads_per_wave=64, barriers_per_iter=2,
        )


def test_plan_wave_specialization_rejects_uneven_split() -> None:
    with pytest.raises(TesseraWaveSpecError, match="multiple of"):
        plan_wave_specialization(total_waves=7, num_groups=2)


def test_plan_wave_specialization_rejects_more_than_two_groups() -> None:
    with pytest.raises(TesseraWaveSpecError, match="exactly 2 groups"):
        plan_wave_specialization(total_waves=8, num_groups=4)


def test_wavegroup_rejects_bad_role() -> None:
    with pytest.raises(TesseraWaveSpecError, match="role must be one of"):
        WaveGroup(group_id=0, wave_ids=(0,), role="manager")


def test_role_at_phase_validates_group_id() -> None:
    p = cdna_attention_plan()
    with pytest.raises(TesseraWaveSpecError, match="out of range"):
        p.role_at_phase(5, 0)


# ── emission + metadata ──────────────────────────────────────────────────────


def test_to_mlir_attrs_carries_schedule() -> None:
    attrs = cdna_attention_plan().to_mlir_attrs()
    assert 'tessera.schedule = "ping_pong"' in attrs
    assert "tessera.wave_groups = 2" in attrs
    assert "tessera.barriers_per_iter = 2" in attrs


def test_metadata_round_trip() -> None:
    md = cdna_attention_plan().as_metadata_dict()
    assert md["kind"] == "wave_specialization_plan"
    assert md["ping_pong"] is True
    assert len(md["groups"]) == 2
    assert md["waves_per_group"] == 4
