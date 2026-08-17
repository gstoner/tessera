from __future__ import annotations

import dataclasses

import pytest

from tessera.compiler.benchmark_row import MeasuredResourceVector
from tessera.compiler.schedule_object import (
    ScheduleAction,
    ScheduleEdge,
    ScheduleObject,
    ScheduleResidency,
    ScheduleRole,
)


def _vector(char: str = "a") -> dict[str, object]:
    return MeasuredResourceVector(
        compute_time_ms=1.0,
        bytes_moved=64,
        communication_bytes=0,
        queue_identity="compute:0",
        resource_identity="device:0",
        timing_provenance={"source": "steady_clock", "domain": "host"},
        artifact_digest=char * 64,
    ).as_dict()


def test_schedule_object_is_content_addressed_and_order_canonical() -> None:
    a = ScheduleAction("load", _vector("a"), op_ref="tile.async_copy", scope="k")
    b = ScheduleAction("mma", _vector("b"), ("load",), op_ref="tile.mma", scope="k")
    edge = ScheduleEdge("load", "mma", "sync", ("async_token",))
    roles = (
        ScheduleRole("producer", ("wave0",)),
        ScheduleRole("consumer", ("wave1",)),
    )
    residency = (ScheduleResidency("fragment", "tile"),)
    left = ScheduleObject("gemm", (a, b), (edge,), roles, residency)
    right = ScheduleObject("gemm", (b, a), (edge,), tuple(reversed(roles)), residency)

    assert left.digest == right.digest
    assert len(left.digest) == 64
    assert left.canonical_payload()["schema"] == "tessera.schedule_object.v1"

    declared = ScheduleObject("declared", (a, b))
    assert declared.edges == (
        ScheduleEdge("load", "mma", "data", ("declared_dependency",)),
    )


def test_schedule_object_digest_binds_roles_edges_resources_and_residency() -> None:
    a = ScheduleAction("load", _vector())
    b = ScheduleAction("mma", _vector("b"), ("load",))
    base = ScheduleObject("gemm", (a, b))
    variants = (
        ScheduleObject("gemm", (dataclasses.replace(a, resource_vector=_vector("c")), b)),
        ScheduleObject("gemm", (a, b), (ScheduleEdge("load", "mma", "data", ("ssa",)),)),
        ScheduleObject("gemm", (a, b), roles=(ScheduleRole("p", ("wave0",)),)),
        ScheduleObject("gemm", (a, b), residency=(ScheduleResidency("x", "layer"),)),
    )
    assert all(item.digest != base.digest for item in variants)


def test_schedule_object_takes_an_immutable_resource_snapshot() -> None:
    vector = _vector()
    action = ScheduleAction("load", vector)
    schedule = ScheduleObject("immutable", (action,))
    digest = schedule.digest

    vector["bytes_moved"] = 4096
    vector["timing_provenance"]["source"] = "mutated"

    assert schedule.digest == digest
    assert schedule.canonical_payload()["actions"][0]["resource_vector"][
        "bytes_moved"
    ] == 64
    with pytest.raises(TypeError):
        action.resource_vector["bytes_moved"] = 128


def test_schedule_object_rejects_cycles_unknown_edges_and_invalid_tiers() -> None:
    a = ScheduleAction("a", _vector(), ("b",))
    b = ScheduleAction("b", _vector("b"), ("a",))
    with pytest.raises(ValueError, match="cycle"):
        ScheduleObject("cycle", (a, b))
    with pytest.raises(ValueError, match="endpoints"):
        ScheduleObject(
            "unknown",
            (ScheduleAction("a", _vector()),),
            (ScheduleEdge("a", "missing", "data", ("ssa",)),),
        )
    with pytest.raises(ValueError, match="depends_on"):
        ScheduleObject(
            "disagree",
            (ScheduleAction("a", _vector()), ScheduleAction("b", _vector("b"))),
            (ScheduleEdge("a", "b", "data", ("ssa",)),),
        )
    with pytest.raises(ValueError, match="tier"):
        ScheduleResidency("x", "register")
