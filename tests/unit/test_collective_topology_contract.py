from __future__ import annotations

import pytest

from tessera.collectives import adapter, collective_topology


def test_collective_topology_has_stable_rank_device_fingerprint() -> None:
    first = collective_topology(backend="nccl", world_size=2, device_ordinals=(3, 5))
    second = collective_topology(backend="nccl", world_size=2, device_ordinals=(3, 5))
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["rank_order"] == [0, 1]
    assert first.to_dict()["device_ordinals"] == [3, 5]


def test_zero_cta_policy_is_content_addressed_and_validated() -> None:
    default = collective_topology(backend="rccl", world_size=2)
    zero = collective_topology(
        backend="rccl", world_size=2, cta_policy="zero"
    )
    assert default.fingerprint != zero.fingerprint
    assert zero.to_dict()["cta_policy"] == "zero"
    with pytest.raises(ValueError, match="CTA policy"):
        collective_topology(
            backend="rccl", world_size=2, cta_policy="best_effort"
        )


@pytest.mark.parametrize("devices", [(0, 0), (0,), (-1, 1)])
def test_collective_topology_rejects_ambiguous_rank_binding(devices) -> None:
    with pytest.raises(ValueError):
        collective_topology(backend="nccl", world_size=2, device_ordinals=devices)


@pytest.mark.parametrize("world_size", [0, -1])
def test_collective_adapter_rejects_nonpositive_world_size(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size must be positive"):
        adapter(backend="mock", world_size=world_size)


def test_collective_adapter_rejects_invalid_mesh_axes() -> None:
    with pytest.raises(ValueError, match="mesh axes"):
        adapter(backend="mock", world_size=2, mesh_axes={"dp": 0})
