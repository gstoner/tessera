"""Canonical, content-addressed compiler schedule value.

The object is deliberately target-neutral.  Target IR carries only the pieces
that must participate in verification (tokens, roles and residency); measured
resource data stays in this value and is addressed by its digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .benchmark_row import validate_resource_vector


SCHEDULE_OBJECT_SCHEMA = "tessera.schedule_object.v1"
_EDGE_KINDS = frozenset({"data", "sync", "resource"})
_RESIDENCY_TIERS = frozenset({"tile", "layer", "full"})


def _freeze_json(value: Any) -> Any:
    """Take an immutable snapshot of JSON-shaped schedule identity data."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers for hashing and serialization."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ScheduleAction:
    action_id: str
    resource_vector: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()
    op_ref: str = ""
    scope: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("schedule action_id must be non-empty")
        if self.action_id in self.depends_on:
            raise ValueError(f"schedule action {self.action_id!r} cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"schedule action {self.action_id!r} has duplicate dependencies")
        validate_resource_vector(self.resource_vector)
        object.__setattr__(self, "resource_vector", _freeze_json(self.resource_vector))

    @classmethod
    def from_benchmark_row(
        cls,
        action_id: str,
        row: Mapping[str, Any],
        *,
        depends_on: Sequence[str] = (),
        op_ref: str = "",
        scope: str = "",
    ) -> "ScheduleAction":
        hot_path = row.get("hot_path_metadata")
        if not isinstance(hot_path, Mapping):
            raise ValueError("schedule action benchmark row lacks hot_path_metadata")
        vector = hot_path.get("resource_vector")
        if not isinstance(vector, Mapping):
            raise ValueError("schedule action benchmark row lacks a measured resource vector")
        return cls(action_id, vector, tuple(depends_on), op_ref, scope)


@dataclass(frozen=True, order=True)
class ScheduleEdge:
    predecessor: str
    successor: str
    kind: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predecessor or not self.successor:
            raise ValueError("schedule edge endpoints must be non-empty")
        if self.predecessor == self.successor:
            raise ValueError("schedule edge cannot be a self-edge")
        if self.kind not in _EDGE_KINDS:
            raise ValueError(f"schedule edge kind must be one of {sorted(_EDGE_KINDS)}")
        if not self.reasons or any(not reason for reason in self.reasons):
            raise ValueError("schedule edge requires at least one non-empty reason")


@dataclass(frozen=True, order=True)
class ScheduleRole:
    name: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("schedule role name must be non-empty")
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("schedule role members must be non-empty and unique")


@dataclass(frozen=True, order=True)
class ScheduleResidency:
    value: str
    tier: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("schedule residency value must be non-empty")
        if self.tier not in _RESIDENCY_TIERS:
            raise ValueError(
                f"schedule residency tier must be one of {sorted(_RESIDENCY_TIERS)}"
            )


@dataclass(frozen=True)
class ScheduleObject:
    object_id: str
    actions: tuple[ScheduleAction, ...]
    edges: tuple[ScheduleEdge, ...] = ()
    roles: tuple[ScheduleRole, ...] = ()
    residency: tuple[ScheduleResidency, ...] = ()
    schema: str = SCHEDULE_OBJECT_SCHEMA

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("schedule object_id must be non-empty")
        if self.schema != SCHEDULE_OBJECT_SCHEMA:
            raise ValueError(f"unsupported schedule object schema {self.schema!r}")
        ids = tuple(action.action_id for action in self.actions)
        if len(set(ids)) != len(ids):
            raise ValueError("schedule action identities must be unique")
        known = set(ids)
        declared_pairs: set[tuple[str, str]] = set()
        for action in self.actions:
            unknown = set(action.depends_on) - known
            if unknown:
                raise ValueError(
                    f"schedule action {action.action_id!r} has unknown dependencies {sorted(unknown)}"
                )
            declared_pairs.update(
                (predecessor, action.action_id) for predecessor in action.depends_on
            )
        explicit_pairs = {
            (edge.predecessor, edge.successor) for edge in self.edges
        }
        for edge in self.edges:
            if edge.predecessor not in known or edge.successor not in known:
                raise ValueError("schedule edge endpoints must name actions in the object")
            if (edge.predecessor, edge.successor) not in declared_pairs:
                raise ValueError(
                    "schedule edge must also be represented in successor depends_on"
                )
        missing_edges = declared_pairs - explicit_pairs
        if missing_edges:
            object.__setattr__(
                self,
                "edges",
                self.edges
                + tuple(
                    ScheduleEdge(before, after, "data", ("declared_dependency",))
                    for before, after in sorted(missing_edges)
                ),
            )
        if len({role.name for role in self.roles}) != len(self.roles):
            raise ValueError("schedule role names must be unique")
        if len({item.value for item in self.residency}) != len(self.residency):
            raise ValueError("schedule residency values must be unique")
        self._validate_dag()

    def _validate_dag(self) -> None:
        incoming: dict[str, set[str]] = {
            action.action_id: set(action.depends_on) for action in self.actions
        }
        for edge in self.edges:
            incoming[edge.successor].add(edge.predecessor)
        ready = sorted(name for name, deps in incoming.items() if not deps)
        visited: set[str] = set()
        while ready:
            current = ready.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for name, deps in incoming.items():
                if current in deps:
                    deps.remove(current)
                    if not deps and name not in visited:
                        ready.append(name)
            ready.sort()
        if len(visited) != len(incoming):
            raise ValueError("schedule object action graph contains a cycle")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "object_id": self.object_id,
            "actions": [
                {
                    "action_id": action.action_id,
                    "depends_on": sorted(action.depends_on),
                    "op_ref": action.op_ref,
                    "scope": action.scope,
                    "resource_vector": _thaw_json(action.resource_vector),
                }
                for action in sorted(self.actions, key=lambda item: item.action_id)
            ],
            "edges": [
                {
                    "predecessor": edge.predecessor,
                    "successor": edge.successor,
                    "kind": edge.kind,
                    "reasons": sorted(edge.reasons),
                }
                for edge in sorted(self.edges)
            ],
            "roles": [
                {"name": role.name, "members": sorted(role.members)}
                for role in sorted(self.roles)
            ],
            "residency": [
                {"value": item.value, "tier": item.tier}
                for item in sorted(self.residency)
            ],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCHEDULE_OBJECT_SCHEMA",
    "ScheduleAction",
    "ScheduleEdge",
    "ScheduleObject",
    "ScheduleResidency",
    "ScheduleRole",
]
