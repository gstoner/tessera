"""Tests for C3 tile-granular compute/comm overlap + SC-HRF scopes/orderings."""

from __future__ import annotations

import pytest

from tessera.compiler.comm_overlap import (
    MemoryOrdering,
    MemoryScope,
    OverlapKind,
    OverlapPlan,
    SignalOp,
    plan_overlap,
)


# ── MemoryScope nesting / covers ─────────────────────────────────────────────
def test_scope_values():
    assert MemoryScope.WAVEFRONT.value == "wavefront"
    assert MemoryScope.WORKGROUP.value == "workgroup"
    assert MemoryScope.AGENT.value == "agent"
    assert MemoryScope.SYSTEM.value == "system"


def test_scope_covers_self():
    for s in MemoryScope:
        assert s.covers(s)


def test_scope_nesting_broad_covers_narrow():
    assert MemoryScope.SYSTEM.covers(MemoryScope.AGENT)
    assert MemoryScope.SYSTEM.covers(MemoryScope.WAVEFRONT)
    assert MemoryScope.AGENT.covers(MemoryScope.WORKGROUP)
    assert MemoryScope.WORKGROUP.covers(MemoryScope.WAVEFRONT)


def test_scope_narrow_does_not_cover_broad():
    assert not MemoryScope.WORKGROUP.covers(MemoryScope.AGENT)
    assert not MemoryScope.WAVEFRONT.covers(MemoryScope.SYSTEM)
    assert not MemoryScope.AGENT.covers(MemoryScope.SYSTEM)


def test_scope_covers_rejects_non_scope():
    with pytest.raises(ValueError):
        MemoryScope.AGENT.covers("agent")  # type: ignore[arg-type]


# ── MemoryOrdering predicates ────────────────────────────────────────────────
def test_ordering_values():
    assert MemoryOrdering.RELAXED.value == "relaxed"
    assert MemoryOrdering.ACQUIRE.value == "acquire"
    assert MemoryOrdering.RELEASE.value == "release"
    assert MemoryOrdering.ACQ_REL.value == "acq_rel"


def test_ordering_has_acquire_release():
    assert MemoryOrdering.ACQUIRE.has_acquire
    assert MemoryOrdering.ACQ_REL.has_acquire
    assert not MemoryOrdering.RELEASE.has_acquire
    assert not MemoryOrdering.RELAXED.has_acquire

    assert MemoryOrdering.RELEASE.has_release
    assert MemoryOrdering.ACQ_REL.has_release
    assert not MemoryOrdering.ACQUIRE.has_release
    assert not MemoryOrdering.RELAXED.has_release


# ── SignalOp producer / consumer validity ────────────────────────────────────
def test_producer_release_is_valid():
    s = SignalOp(role="producer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELEASE)
    assert s.is_valid_producer()


def test_producer_acq_rel_is_valid():
    s = SignalOp(role="producer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.ACQ_REL)
    assert s.is_valid_producer()


def test_producer_relaxed_rejected():
    with pytest.raises(ValueError):
        SignalOp(role="producer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELAXED)


def test_producer_acquire_rejected():
    with pytest.raises(ValueError):
        SignalOp(role="producer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.ACQUIRE)


def test_consumer_acquire_is_valid():
    s = SignalOp(role="consumer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.ACQUIRE)
    assert s.is_valid_consumer()


def test_consumer_acq_rel_is_valid():
    s = SignalOp(role="consumer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.ACQ_REL)
    assert s.is_valid_consumer()


def test_consumer_release_rejected():
    with pytest.raises(ValueError):
        SignalOp(role="consumer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELEASE)


def test_consumer_relaxed_rejected():
    with pytest.raises(ValueError):
        SignalOp(role="consumer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELAXED)


def test_signal_bad_role():
    with pytest.raises(ValueError):
        SignalOp(role="middle", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELEASE)


def test_signal_bad_kind():
    with pytest.raises(ValueError):
        SignalOp(role="producer", scope=MemoryScope.AGENT,
                 ordering=MemoryOrdering.RELEASE, kind="atomic_xor")


def test_signal_bad_scope_type():
    with pytest.raises(ValueError):
        SignalOp(role="producer", scope="agent",  # type: ignore[arg-type]
                 ordering=MemoryOrdering.RELEASE)


def test_signal_metadata_roundtrips_enum_values():
    s = SignalOp(role="producer", scope=MemoryScope.SYSTEM,
                 ordering=MemoryOrdering.ACQ_REL, kind="atomic_add")
    md = s.as_metadata_dict()
    assert md == {
        "role": "producer",
        "scope": "system",
        "ordering": "acq_rel",
        "kind": "atomic_add",
    }


# ── OverlapKind values ───────────────────────────────────────────────────────
def test_overlap_kind_values():
    assert OverlapKind.SEQUENTIAL_FUSED.value == "sequential_fused"
    assert OverlapKind.WORKGROUP_SPECIALIZED.value == "workgroup_specialized"
    assert OverlapKind.UNFUSED_PRODUCER_CONSUMER.value == "unfused_producer_consumer"


# ── OverlapPlan consistency ──────────────────────────────────────────────────
def _release(scope=MemoryScope.AGENT):
    return SignalOp(role="producer", scope=scope, ordering=MemoryOrdering.RELEASE)


def _acquire(scope=MemoryScope.AGENT):
    return SignalOp(role="consumer", scope=scope, ordering=MemoryOrdering.ACQUIRE)


def test_workgroup_specialized_valid():
    plan = OverlapPlan(
        kind=OverlapKind.WORKGROUP_SPECIALIZED,
        producer_workgroups=48,
        consumer_workgroups=16,
        signal=_release(),
        wait=_acquire(),
    )
    assert plan.producer_workgroups == 48
    assert plan.consumer_workgroups == 16
    assert plan.signal.ordering is MemoryOrdering.RELEASE
    assert plan.wait.ordering is MemoryOrdering.ACQUIRE


def test_workgroup_specialized_requires_consumer_wgs():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.WORKGROUP_SPECIALIZED,
            producer_workgroups=48,
            consumer_workgroups=0,
            signal=_release(),
            wait=_acquire(),
        )


def test_workgroup_specialized_requires_producer_wgs():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.WORKGROUP_SPECIALIZED,
            producer_workgroups=0,
            consumer_workgroups=16,
            signal=_release(),
            wait=_acquire(),
        )


def test_workgroup_specialized_scope_must_cover():
    # producer release at workgroup scope cannot cover consumer acquire at agent
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.WORKGROUP_SPECIALIZED,
            producer_workgroups=48,
            consumer_workgroups=16,
            signal=_release(scope=MemoryScope.WORKGROUP),
            wait=_acquire(scope=MemoryScope.AGENT),
        )


def test_sequential_fused_requires_zero_consumers():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.SEQUENTIAL_FUSED,
            producer_workgroups=64,
            consumer_workgroups=16,
            signal=_release(),
            wait=_acquire(),
        )


def test_sequential_fused_valid_zero_consumers():
    plan = OverlapPlan(
        kind=OverlapKind.SEQUENTIAL_FUSED,
        producer_workgroups=64,
        consumer_workgroups=0,
        signal=_release(),
        wait=_acquire(),
    )
    assert plan.consumer_workgroups == 0


def test_plan_signal_must_be_producer():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
            producer_workgroups=32,
            consumer_workgroups=32,
            signal=_acquire(),  # wrong role
            wait=_acquire(),
        )


def test_plan_wait_must_be_consumer():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
            producer_workgroups=32,
            consumer_workgroups=32,
            signal=_release(),
            wait=_release(),  # wrong role
        )


def test_plan_cu_partition_valid():
    plan = OverlapPlan(
        kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
        producer_workgroups=32,
        consumer_workgroups=32,
        signal=_release(),
        wait=_acquire(),
        cu_partition=(60, 20),
    )
    assert plan.cu_partition == (60, 20)


def test_plan_cu_partition_bad_shape():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
            producer_workgroups=32,
            consumer_workgroups=32,
            signal=_release(),
            wait=_acquire(),
            cu_partition=(60,),  # type: ignore[arg-type]
        )


def test_plan_cu_partition_negative():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
            producer_workgroups=32,
            consumer_workgroups=32,
            signal=_release(),
            wait=_acquire(),
            cu_partition=(60, -1),
        )


def test_plan_negative_workgroups():
    with pytest.raises(ValueError):
        OverlapPlan(
            kind=OverlapKind.UNFUSED_PRODUCER_CONSUMER,
            producer_workgroups=-1,
            consumer_workgroups=32,
            signal=_release(),
            wait=_acquire(),
        )


def test_plan_metadata_roundtrips_enums():
    plan = OverlapPlan(
        kind=OverlapKind.WORKGROUP_SPECIALIZED,
        producer_workgroups=48,
        consumer_workgroups=16,
        signal=_release(scope=MemoryScope.AGENT),
        wait=_acquire(scope=MemoryScope.AGENT),
        cu_partition=(48, 16),
    )
    md = plan.as_metadata_dict()
    assert md["kind"] == "workgroup_specialized"
    assert md["producer_workgroups"] == 48
    assert md["consumer_workgroups"] == 16
    assert md["signal"] == {"role": "producer", "scope": "agent",
                            "ordering": "release", "kind": "atomic_cas"}
    assert md["wait"] == {"role": "consumer", "scope": "agent",
                          "ordering": "acquire", "kind": "atomic_cas"}
    assert md["cu_partition"] == [48, 16]


# ── plan_overlap factory ─────────────────────────────────────────────────────
def test_plan_overlap_builds_matched_pair_default_agent():
    plan = plan_overlap(OverlapKind.WORKGROUP_SPECIALIZED,
                        producer_wgs=48, consumer_wgs=16)
    assert plan.signal.role == "producer"
    assert plan.signal.ordering is MemoryOrdering.RELEASE
    assert plan.signal.scope is MemoryScope.AGENT
    assert plan.wait.role == "consumer"
    assert plan.wait.ordering is MemoryOrdering.ACQUIRE
    assert plan.wait.scope is MemoryScope.AGENT
    # the release scope covers the acquire scope (well-formed handoff)
    assert plan.signal.scope.covers(plan.wait.scope)


def test_plan_overlap_sequential_fused_zero_consumers():
    plan = plan_overlap(OverlapKind.SEQUENTIAL_FUSED,
                        producer_wgs=64, consumer_wgs=0)
    assert plan.kind is OverlapKind.SEQUENTIAL_FUSED
    assert plan.consumer_workgroups == 0


def test_plan_overlap_custom_scope():
    plan = plan_overlap(OverlapKind.UNFUSED_PRODUCER_CONSUMER,
                        producer_wgs=32, consumer_wgs=32,
                        scope=MemoryScope.SYSTEM)
    assert plan.signal.scope is MemoryScope.SYSTEM
    assert plan.wait.scope is MemoryScope.SYSTEM


def test_plan_overlap_uses_atomic_cas():
    plan = plan_overlap(OverlapKind.WORKGROUP_SPECIALIZED,
                        producer_wgs=48, consumer_wgs=16)
    assert plan.signal.kind == "atomic_cas"
    assert plan.wait.kind == "atomic_cas"


def test_plan_overlap_specialized_requires_positive_consumers():
    with pytest.raises(ValueError):
        plan_overlap(OverlapKind.WORKGROUP_SPECIALIZED,
                     producer_wgs=48, consumer_wgs=0)
