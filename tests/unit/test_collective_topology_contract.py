from __future__ import annotations

import pytest

from tessera.collectives import collective_topology


def test_collective_topology_has_stable_rank_device_fingerprint() -> None:
    first = collective_topology(backend="nccl", world_size=2, device_ordinals=(3, 5))
    second = collective_topology(backend="nccl", world_size=2, device_ordinals=(3, 5))
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["rank_order"] == [0, 1]
    assert first.to_dict()["device_ordinals"] == [3, 5]


@pytest.mark.parametrize("devices", [(0, 0), (0,), (-1, 1)])
def test_collective_topology_rejects_ambiguous_rank_binding(devices) -> None:
    with pytest.raises(ValueError):
        collective_topology(backend="nccl", world_size=2, device_ordinals=devices)
