"""Elastic distributed runtime metadata for Tessera."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional


VALID_BACKENDS = ("k8s", "etcd", "redis", "slurm", "ray", "custom")
VALID_RESHARD_POLICIES = ("consistent_hash", "balanced", "replicate", "none")


class ElasticConfigError(ValueError):
    """Raised when elastic runtime configuration is invalid."""


@dataclass(frozen=True)
class ElasticConfig:
    """Rendezvous and membership policy."""

    backend: str = "k8s"
    group: str = "default"
    min_ranks: int = 1
    max_ranks: int = 1
    rebalance_on_join: bool = True
    rebalance_on_exit: bool = True
    rendezvous: str = ""

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ElasticConfigError(f"backend must be one of {VALID_BACKENDS}")
        if self.min_ranks < 1 or self.max_ranks < self.min_ranks:
            raise ElasticConfigError("require 1 <= min_ranks <= max_ranks")
        if not self.group:
            raise ElasticConfigError("group is required")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "group": self.group,
            "min_ranks": self.min_ranks,
            "max_ranks": self.max_ranks,
            "rebalance_on_join": self.rebalance_on_join,
            "rebalance_on_exit": self.rebalance_on_exit,
            "rendezvous": self.rendezvous,
        }

    def to_ir_attr(self) -> str:
        return (
            "{tessera.elastic = {"
            f'backend = "{self.backend}", group = "{self.group}", '
            f"min_ranks = {self.min_ranks}, max_ranks = {self.max_ranks}}}"
        )


@dataclass(frozen=True)
class ReshardPlan:
    """Plan for remapping mesh shards after elastic membership changes."""

    policy: str = "consistent_hash"
    migrate_async: bool = True
    old_mesh: Mapping[str, int] = field(default_factory=dict)
    new_mesh: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in VALID_RESHARD_POLICIES:
            raise ElasticConfigError(f"policy must be one of {VALID_RESHARD_POLICIES}")
        for mesh_name, mesh in (("old_mesh", self.old_mesh), ("new_mesh", self.new_mesh)):
            for axis, size in mesh.items():
                if not axis or size < 1:
                    raise ElasticConfigError(f"{mesh_name} contains invalid axis {axis!r}={size!r}")

    def moved_fraction(self) -> float:
        old_total = _mesh_total(self.old_mesh)
        new_total = _mesh_total(self.new_mesh)
        if old_total == 0 or new_total == 0:
            return 0.0
        if self.policy == "consistent_hash":
            return abs(new_total - old_total) / max(new_total, old_total)
        if self.policy == "none":
            return 0.0
        return 1.0

    def to_ir_attr(self) -> str:
        async_text = "true" if self.migrate_async else "false"
        return f'{{tessera.reshard = {{policy = "{self.policy}", migrate_async = {async_text}}}}}'


@dataclass(frozen=True)
class TopologyChangePolicy:
    """Policy for hardware/topology changes during elastic recovery."""

    retune_autotuner: bool = True
    invalidate_kernels: bool = True


_CURRENT_CONFIG: Optional[ElasticConfig] = None
_CURRENT_MESH: dict[str, int] = {}


def configure(**kwargs) -> ElasticConfig:
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = ElasticConfig(**kwargs)
    return _CURRENT_CONFIG


@contextlib.contextmanager
def elastic(
    *,
    rendezvous: str,
    min_ranks: int,
    max_ranks: int,
    backend: Optional[str] = None,
    group: str = "default",
) -> Iterator[ElasticConfig]:
    """Context manager for elastic membership."""

    inferred_backend = backend or rendezvous.split("://", 1)[0]
    cfg = configure(
        backend=inferred_backend,
        rendezvous=rendezvous,
        group=group,
        min_ranks=min_ranks,
        max_ranks=max_ranks,
    )
    try:
        yield cfg
    finally:
        pass


def reshard(
    *,
    policy: str = "consistent_hash",
    migrate_async: bool = True,
    old_mesh: Optional[Mapping[str, int]] = None,
    new_mesh: Optional[Mapping[str, int]] = None,
) -> ReshardPlan:
    return ReshardPlan(
        policy=policy,
        migrate_async=migrate_async,
        old_mesh=dict(old_mesh or _CURRENT_MESH),
        new_mesh=dict(new_mesh or _CURRENT_MESH),
    )


def set_current_mesh(mesh: Mapping[str, int]) -> None:
    global _CURRENT_MESH
    _CURRENT_MESH = dict(mesh)


def current_mesh() -> Mapping[str, int]:
    return dict(_CURRENT_MESH)


def world_size() -> int:
    return _mesh_total(_CURRENT_MESH) or 1


def on_topology_change(*, retune_autotuner: bool = True, invalidate_kernels: bool = True) -> TopologyChangePolicy:
    return TopologyChangePolicy(
        retune_autotuner=retune_autotuner,
        invalidate_kernels=invalidate_kernels,
    )


def current_config() -> Optional[ElasticConfig]:
    return _CURRENT_CONFIG


def _mesh_total(mesh: Mapping[str, int]) -> int:
    result = 1
    if not mesh:
        return 0
    for size in mesh.values():
        result *= int(size)
    return result


__all__ = [
    "ElasticConfig",
    "ElasticConfigError",
    "ReshardPlan",
    "TopologyChangePolicy",
    "configure",
    "current_config",
    "current_mesh",
    "elastic",
    "on_topology_change",
    "reshard",
    "set_current_mesh",
    "world_size",
]
