"""Prune-only cost analysis for measured Tile action DAGs.

R3 consumes the measured resource vectors introduced by R2 and searches legal
topological orders of a Tile action DAG.  It deliberately has no selection or
promotion API: the result may reduce the set of candidates sent to exact-device
measurement, but scalar measured latency remains the final authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .benchmark_row import (
    SCALAR_SELECTOR_AUTHORITY,
    validate_resource_vector,
)


COMPOSITION_MODEL = "tessera.tile_action_dag_cost.v1"


@dataclass(frozen=True)
class CompositionCalibration:
    """Measured bandwidths used to convert byte counts into lane time."""

    memory_bytes_per_ms: float
    communication_bytes_per_ms: float
    provenance: Mapping[str, Any]
    digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_bytes_per_ms", self.memory_bytes_per_ms),
            ("communication_bytes_per_ms", self.communication_bytes_per_ms),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("calibration provenance must be a mapping")
        for name in ("source", "domain"):
            provenance_value = self.provenance.get(name)
            if not isinstance(provenance_value, str) or not provenance_value:
                raise ValueError(f"calibration provenance requires non-empty {name}")
        _require_digest(self.digest, "calibration digest")


@dataclass(frozen=True)
class TileAction:
    """One measured action and its explicit predecessor identities."""

    action_id: str
    resource_vector: Mapping[str, Any]
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("Tile action_id must be non-empty")
        if self.action_id in self.depends_on:
            raise ValueError(f"Tile action {self.action_id!r} cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"Tile action {self.action_id!r} has duplicate dependencies")
        validate_resource_vector(self.resource_vector)

    @classmethod
    def from_benchmark_row(
        cls,
        action_id: str,
        row: Mapping[str, Any],
        *,
        depends_on: Sequence[str] = (),
    ) -> "TileAction":
        hot_path = row.get("hot_path_metadata")
        if not isinstance(hot_path, Mapping):
            raise ValueError("Tile action benchmark row lacks hot_path_metadata")
        vector = hot_path.get("resource_vector")
        if not isinstance(vector, Mapping):
            raise ValueError("Tile action benchmark row lacks a measured resource vector")
        return cls(action_id, vector, tuple(depends_on))


@dataclass(frozen=True)
class CompositionCandidate:
    candidate_id: str
    actions: tuple[TileAction, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("composition candidate_id must be non-empty")
        _validate_dag(self.actions)


@dataclass(frozen=True)
class CompositionEstimate:
    candidate_id: str
    predicted_makespan_ms: float
    action_order: tuple[str, ...]
    orders_examined: int
    exhaustive: bool
    calibration_digest: str
    analysis_digest: str
    method: str = COMPOSITION_MODEL
    selector_authority: str = SCALAR_SELECTOR_AUTHORITY
    eligible_for_promotion: bool = False


@dataclass(frozen=True)
class CompositionPruningResult:
    """Candidate filter result, intentionally without a selected winner."""

    retained: tuple[str, ...]
    pruned: tuple[str, ...]
    estimates: tuple[CompositionEstimate, ...]
    relative_margin: float
    method: str = COMPOSITION_MODEL
    selector_authority: str = SCALAR_SELECTOR_AUTHORITY
    eligible_for_promotion: bool = False


def estimate_composition(
    candidate: CompositionCandidate,
    calibration: CompositionCalibration,
    *,
    max_orders: int = 4096,
) -> CompositionEstimate:
    """Return the best modeled makespan over bounded legal DAG orders."""

    if max_orders < 1:
        raise ValueError("max_orders must be >= 1")
    orders, exhaustive = _topological_orders(candidate.actions, max_orders=max_orders)
    best_order: tuple[str, ...] = ()
    best_ms = math.inf
    for order in orders:
        makespan = _simulate_order(candidate.actions, order, calibration)
        if makespan < best_ms:
            best_ms = makespan
            best_order = order
    if not orders:
        raise ValueError("composition candidate has no legal topological order")
    payload = {
        "candidate_id": candidate.candidate_id,
        "action_order": list(best_order),
        "artifact_digests": [
            action.resource_vector["artifact_digest"] for action in candidate.actions
        ],
        "calibration_digest": calibration.digest,
        "method": COMPOSITION_MODEL,
    }
    analysis_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompositionEstimate(
        candidate_id=candidate.candidate_id,
        predicted_makespan_ms=best_ms,
        action_order=best_order,
        orders_examined=len(orders),
        exhaustive=exhaustive,
        calibration_digest=calibration.digest,
        analysis_digest=analysis_digest,
    )


def prune_composition_candidates(
    candidates: Sequence[CompositionCandidate],
    calibration: CompositionCalibration,
    *,
    relative_margin: float = 0.25,
    max_orders: int = 4096,
) -> CompositionPruningResult:
    """Prune clearly inferior *exhaustively analyzed* candidates.

    Inexact searches are always retained.  The best modeled candidate is also
    retained, but is not selected: callers must exact-device measure every
    retained candidate and use scalar latency to choose among them.
    """

    if not candidates:
        raise ValueError("at least one composition candidate is required")
    if not math.isfinite(relative_margin) or relative_margin < 0.0:
        raise ValueError("relative_margin must be finite and non-negative")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("composition candidate ids must be unique")
    estimates = tuple(
        estimate_composition(candidate, calibration, max_orders=max_orders)
        for candidate in candidates
    )
    exact = [estimate for estimate in estimates if estimate.exhaustive]
    if not exact:
        return CompositionPruningResult(tuple(ids), (), estimates, relative_margin)
    floor = min(estimate.predicted_makespan_ms for estimate in exact)
    threshold = floor * (1.0 + relative_margin)
    by_id = {estimate.candidate_id: estimate for estimate in estimates}
    pruned = tuple(
        candidate_id for candidate_id in ids
        if by_id[candidate_id].exhaustive
        and by_id[candidate_id].predicted_makespan_ms > threshold
    )
    pruned_set = set(pruned)
    retained = tuple(candidate_id for candidate_id in ids if candidate_id not in pruned_set)
    return CompositionPruningResult(retained, pruned, estimates, relative_margin)


def _validate_dag(actions: Sequence[TileAction]) -> None:
    if not actions:
        raise ValueError("composition candidate requires at least one Tile action")
    ids = [action.action_id for action in actions]
    if len(set(ids)) != len(ids):
        raise ValueError("Tile action ids must be unique")
    known = set(ids)
    for action in actions:
        missing = set(action.depends_on) - known
        if missing:
            raise ValueError(
                f"Tile action {action.action_id!r} has unknown dependencies: "
                f"{sorted(missing)}"
            )
    _, exhaustive = _topological_orders(actions, max_orders=1)
    if not exhaustive and len(actions) == 1:
        return
    # A bounded search still produces an order for every acyclic graph. Empty
    # means Kahn's ready set became empty before all actions were emitted.
    orders, _ = _topological_orders(actions, max_orders=1)
    if not orders:
        raise ValueError("Tile action dependencies contain a cycle")


def _topological_orders(
    actions: Sequence[TileAction], *, max_orders: int
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    ids = tuple(action.action_id for action in actions)
    deps = {action.action_id: set(action.depends_on) for action in actions}
    found: list[tuple[str, ...]] = []
    overflow = False

    def visit(prefix: tuple[str, ...], remaining: set[str]) -> None:
        nonlocal overflow
        if overflow:
            return
        if not remaining:
            found.append(prefix)
            if len(found) > max_orders:
                overflow = True
            return
        ready = sorted(node for node in remaining if deps[node].issubset(prefix))
        for node in ready:
            visit(prefix + (node,), remaining - {node})

    visit((), set(ids))
    if overflow:
        return tuple(found[:max_orders]), False
    return tuple(found), True


def _simulate_order(
    actions: Sequence[TileAction],
    order: Sequence[str],
    calibration: CompositionCalibration,
) -> float:
    by_id = {action.action_id: action for action in actions}
    completed: dict[str, float] = {}
    queue_free: dict[str, float] = {}
    lane_free: dict[tuple[str, str], float] = {}
    for action_id in order:
        action = by_id[action_id]
        vector = action.resource_vector
        queue = str(vector["queue_identity"])
        resource = str(vector["resource_identity"])
        durations = {
            "compute": float(vector["compute_time_ms"]),
            "memory": int(vector["bytes_moved"]) / calibration.memory_bytes_per_ms,
            "communication": int(vector["communication_bytes"])
            / calibration.communication_bytes_per_ms,
        }
        active = {name: duration for name, duration in durations.items() if duration > 0.0}
        dependency_ready = max(
            (completed[dependency] for dependency in action.depends_on),
            default=0.0,
        )
        start = max(
            dependency_ready,
            queue_free.get(queue, 0.0),
            *(lane_free.get((resource, lane), 0.0) for lane in active),
        )
        finish = start + max(active.values(), default=0.0)
        completed[action_id] = finish
        queue_free[queue] = finish
        for lane, duration in active.items():
            lane_free[(resource, lane)] = start + duration
    return max(completed.values(), default=0.0)


def _require_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")


__all__ = [
    "COMPOSITION_MODEL",
    "CompositionCalibration",
    "CompositionCandidate",
    "CompositionEstimate",
    "CompositionPruningResult",
    "TileAction",
    "estimate_composition",
    "prune_composition_candidates",
]
