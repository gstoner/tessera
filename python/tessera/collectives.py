"""Runtime-facing collective adapter facade.

The facade makes backend status explicit while keeping the current hardware-free
mock collective path usable for tests and examples. Native NCCL/RCCL execution
is intentionally reported as unavailable unless a future runtime backend wires
and detects it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .testing.mock_collective import MockRank, MockRankGroup


COLLECTIVE_STATUSES = {"mock", "single_process", "backend_unavailable", "hardware_runtime"}


@dataclass(frozen=True)
class CollectiveBackendStatus:
    backend: str
    status: str
    reason: str = ""
    world_size: int = 1

    @property
    def available(self) -> bool:
        return self.status in {"mock", "single_process", "hardware_runtime"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "reason": self.reason,
            "world_size": self.world_size,
            "available": self.available,
        }


class CollectiveAdapter:
    """Small collective adapter with explicit backend status."""

    def __init__(self, *, backend: str = "mock", world_size: int = 1, mesh_axes: dict[str, int] | None = None) -> None:
        self.backend = backend
        self.world_size = int(world_size)
        self.mesh_axes = dict(mesh_axes or {"dp": self.world_size})
        if self.backend == "mock" and self.world_size > 1:
            self._group: MockRankGroup | None = MockRankGroup(self.world_size, self.mesh_axes)
        else:
            self._group = None

    def status(self) -> CollectiveBackendStatus:
        if self.backend == "mock" and self._group is not None:
            return CollectiveBackendStatus(self.backend, "mock", world_size=self.world_size)
        if self.backend in {"mock", "single_process"} and self.world_size == 1:
            return CollectiveBackendStatus(self.backend, "single_process", world_size=1)
        if self.backend in {"nccl", "rccl", "mpi"}:
            return CollectiveBackendStatus(
                self.backend,
                "backend_unavailable",
                reason=f"{self.backend} native collective runtime is not wired in this hardware-free build",
                world_size=self.world_size,
            )
        return CollectiveBackendStatus(
            self.backend,
            "backend_unavailable",
            reason=f"unknown collective backend {self.backend!r}",
            world_size=self.world_size,
        )

    def all_reduce(self, values, *, op: str = "sum"):
        return self._run(lambda rank, value: rank.all_reduce(value, op=op), values)

    def reduce_scatter(self, values, *, axis: int = 0, op: str = "sum"):
        return self._run(lambda rank, value: rank.reduce_scatter(value, axis=axis, op=op), values)

    def all_gather(self, values, *, axis: int = 0):
        return self._run(lambda rank, value: rank.all_gather(value, axis=axis), values)

    def all_to_all(self, values, *, scatter_axis: int = 0, gather_axis: int = 0):
        return self._run(lambda rank, value: rank.all_to_all(value, scatter_axis=scatter_axis, gather_axis=gather_axis), values)

    def _run(self, fn, values):
        status = self.status()
        if status.status == "backend_unavailable":
            raise RuntimeError(status.reason)
        if status.status == "single_process":
            value = _single_value(values)
            return [np.asarray(value)]
        assert self._group is not None
        per_rank = _per_rank_values(values, self.world_size)
        return self._group.run(lambda rank: fn(rank, np.asarray(per_rank[rank.rank])))


def adapter(*, backend: str = "mock", world_size: int = 1, mesh_axes: dict[str, int] | None = None) -> CollectiveAdapter:
    return CollectiveAdapter(backend=backend, world_size=world_size, mesh_axes=mesh_axes)


def query_backend(backend: str = "mock", *, world_size: int = 1) -> dict[str, Any]:
    return adapter(backend=backend, world_size=world_size).status().to_dict()


def _per_rank_values(values, world_size: int) -> list[Any]:
    if isinstance(values, (list, tuple)) and len(values) == world_size:
        return list(values)
    return [values for _ in range(world_size)]


def _single_value(values):
    if isinstance(values, (list, tuple)):
        return values[0]
    return values


__all__ = [
    "COLLECTIVE_STATUSES",
    "CollectiveAdapter",
    "CollectiveBackendStatus",
    "adapter",
    "query_backend",
]
