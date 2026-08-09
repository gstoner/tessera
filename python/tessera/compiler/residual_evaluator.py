"""Execution-derived residual-policy evidence for compiler autodiff.

This module measures the two quantities that a save/recompute decision actually
trades: complete backward execution time and retained residual storage.  It is
deliberately independent of TileSight and analytical target models.  Those may
prune the candidate set, but only an execution-derived :class:`ResidualEvidence`
is eligible to select a production policy.

Treeverse plans are emitted as candidates, not verdicts.  Their estimated work
is derived from a measured per-step cost, but the complete backward must still
be run through :func:`measure_residual_candidate` before promotion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import statistics
import time
from typing import Any, Literal


ResidualPolicy = Literal["save", "recompute", "hybrid", "treeverse"]


@dataclass(frozen=True)
class ForwardCapture:
    """Forward result and the exact values retained for its backward."""

    output: Any
    residuals: Any


@dataclass(frozen=True)
class ResidualEvidence:
    """Measured evidence for one exact policy/shape/dtype/target candidate."""

    target: str
    operation: str
    shape_bucket: tuple[int, ...]
    dtype: str
    policy: ResidualPolicy
    retained_residual_bytes: int
    backward_samples_ns: tuple[int, ...]
    timing_domain: str
    provenance: str
    exact_device: bool

    @property
    def backward_median_ns(self) -> int:
        if not self.backward_samples_ns:
            raise ValueError("residual evidence has no backward samples")
        return round(statistics.median(self.backward_samples_ns))

    @property
    def backward_p95_ns(self) -> int:
        if not self.backward_samples_ns:
            raise ValueError("residual evidence has no backward samples")
        ordered = sorted(self.backward_samples_ns)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @property
    def selector_eligible(self) -> bool:
        return (
            self.exact_device
            and bool(self.backward_samples_ns)
            and self.retained_residual_bytes >= 0
            and self.timing_domain not in {"estimated", "tilesight_model"}
        )

    def compiler_attributes(self) -> dict[str, int | str]:
        """Attributes consumed by ActivationRematerializationPass."""
        if not self.selector_eligible:
            raise ValueError(
                "only exact-device, execution-derived residual evidence may "
                "stamp compiler selection attributes"
            )
        return {
            "tessera.backward_work_ns": self.backward_median_ns,
            "tessera.residual.retained_bytes": self.retained_residual_bytes,
            "tessera.autodiff.residual_policy": self.policy,
            "tessera.residual.evidence_provenance": self.provenance,
        }


@dataclass(frozen=True)
class TreeverseCandidate:
    """A candidate checkpoint envelope awaiting complete-backward execution."""

    steps: int
    checkpoint_slots: int
    checkpoint_indices: tuple[int, ...]
    retained_residual_bytes: int
    estimated_replayed_steps: int
    estimated_backward_work_ns: int
    selector_eligible: bool = False


def _retained_bytes(value: Any, seen: set[int]) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(_retained_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_retained_bytes(item, seen) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return value.nbytes if isinstance(value, memoryview) else len(value)

    # NumPy and device-array implementations conventionally expose `nbytes`.
    # Follow `.base` to avoid charging overlapping views as independent saved
    # allocations.  For a foreign device array without `.base`, object identity
    # still prevents duplicate entries in nested residual structures.
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        owner = value
        while getattr(owner, "base", None) is not None:
            owner = owner.base
        identity = id(owner)
        if identity in seen:
            return 0
        seen.add(identity)
        return int(getattr(owner, "nbytes", nbytes))
    return 0


def retained_residual_bytes(residuals: Any) -> int:
    """Return unique retained buffer bytes for a nested residual structure."""
    return _retained_bytes(residuals, set())


def measure_residual_candidate(
    *,
    target: str,
    operation: str,
    shape_bucket: Sequence[int],
    dtype: str,
    policy: ResidualPolicy,
    forward: Callable[[], ForwardCapture],
    backward: Callable[[Any, Any], Any],
    cotangent: Any,
    synchronize: Callable[[], None] | None = None,
    warmup: int = 3,
    repetitions: int = 20,
    timing_domain: str = "synchronized_host_wall",
    provenance: str,
    exact_device: bool,
) -> ResidualEvidence:
    """Measure a complete backward and the residual allocation it consumes.

    ``backward`` owns the policy semantics.  A recompute candidate must perform
    its recomputation inside that callable; the timer therefore covers actual
    complete backward work rather than an isolated forward proxy.
    """
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    sync = synchronize or (lambda: None)
    capture = forward()
    if not isinstance(capture, ForwardCapture):
        raise TypeError("forward must return ForwardCapture(output, residuals)")
    residual_bytes = retained_residual_bytes(capture.residuals)
    samples: list[int] = []
    for iteration in range(warmup + repetitions):
        # Refresh saved values every iteration so mutation and allocation are
        # represented exactly as they are in the requested policy.
        capture = forward()
        sync()
        start = time.perf_counter_ns()
        backward(cotangent, capture.residuals)
        sync()
        elapsed = time.perf_counter_ns() - start
        if iteration >= warmup:
            samples.append(elapsed)
    return ResidualEvidence(
        target=target,
        operation=operation,
        shape_bucket=tuple(int(dim) for dim in shape_bucket),
        dtype=dtype,
        policy=policy,
        retained_residual_bytes=residual_bytes,
        backward_samples_ns=tuple(samples),
        timing_domain=timing_domain,
        provenance=provenance,
        exact_device=exact_device,
    )


def select_residual_policy(
    evidence: Iterable[ResidualEvidence], *, memory_budget_bytes: int
) -> ResidualEvidence:
    """Select the fastest eligible policy that fits the residual budget."""
    if memory_budget_bytes < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    rows = list(evidence)
    if not rows:
        raise ValueError("at least one residual candidate is required")
    identity = {
        (row.target, row.operation, row.shape_bucket, row.dtype) for row in rows
    }
    if len(identity) != 1:
        raise ValueError("residual candidates must describe one exact workload")
    eligible = [
        row
        for row in rows
        if row.selector_eligible
        and row.retained_residual_bytes <= memory_budget_bytes
    ]
    if not eligible:
        raise ValueError("no execution-derived residual policy fits the budget")
    return min(
        eligible,
        key=lambda row: (
            row.backward_median_ns,
            row.retained_residual_bytes,
            row.policy,
        ),
    )


def treeverse_candidates(
    *, steps: int, state_bytes: int, measured_step_work_ns: int,
    memory_budgets: Iterable[int],
) -> tuple[TreeverseCandidate, ...]:
    """Generate checkpoint candidates from measured step work.

    The balanced checkpoint locations are a pruning envelope.  The replay count
    is intentionally an estimate and every returned candidate is promotion-
    ineligible until its complete backward is measured.
    """
    if steps <= 0 or state_bytes <= 0 or measured_step_work_ns <= 0:
        raise ValueError("steps, state_bytes, and measured_step_work_ns must be positive")
    candidates: list[TreeverseCandidate] = []
    for budget in memory_budgets:
        if budget < 0:
            raise ValueError("memory budgets must be non-negative")
        slots = min(max(budget // state_bytes, 0), max(steps - 1, 0))
        if slots:
            indices = tuple(
                sorted({round(i * steps / (slots + 1)) for i in range(1, slots + 1)})
            )
        else:
            indices = ()
        boundaries = (0, *indices, steps)
        replayed = sum(
            length * max(length - 1, 0) // 2
            for length in (
                boundaries[i + 1] - boundaries[i]
                for i in range(len(boundaries) - 1)
            )
        )
        candidates.append(
            TreeverseCandidate(
                steps=steps,
                checkpoint_slots=len(indices),
                checkpoint_indices=indices,
                retained_residual_bytes=len(indices) * state_bytes,
                estimated_replayed_steps=replayed,
                estimated_backward_work_ns=(steps + replayed)
                * measured_step_work_ns,
            )
        )
    return tuple(candidates)


__all__ = [
    "ForwardCapture",
    "ResidualEvidence",
    "ResidualPolicy",
    "TreeverseCandidate",
    "measure_residual_candidate",
    "retained_residual_bytes",
    "select_residual_policy",
    "treeverse_candidates",
]
