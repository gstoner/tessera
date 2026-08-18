"""Execution-derived residual-policy evidence for compiler autodiff.

This module measures the two quantities that a save/recompute decision actually
trades: complete backward execution time and retained residual storage.  It is
deliberately independent of TileSight and analytical target models.  Those may
prune the candidate set, but only an execution-derived :class:`ResidualEvidence`
is eligible to select a production policy.

Treeverse plans begin as pruning-only candidates.  The bounded counted-region
executor can replay a candidate's actual backward; only that execution-derived
row can become selector eligible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import statistics
import time
from typing import Any, Literal


ResidualPolicy = Literal["save", "recompute", "hybrid", "treeverse"]
REGION_RESIDUAL_ABI_SCHEMA = "tessera.region_residual_abi.v1"


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
    forward_steps: int = 0
    replayed_steps: int = 0
    backward_steps: int = 0

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
                "only exact-device, execution-derived residual evidence may stamp compiler selection attributes"
            )
        return {
            "tessera.backward_work_ns": self.backward_median_ns,
            "tessera.residual.retained_bytes": self.retained_residual_bytes,
            "tessera.autodiff.residual_policy": self.policy,
            "tessera.residual.evidence_provenance": self.provenance,
            "tessera.residual.forward_steps": self.forward_steps,
            "tessera.residual.replayed_steps": self.replayed_steps,
            "tessera.residual.backward_steps": self.backward_steps,
        }

    def evidence_record(self) -> dict[str, object]:
        """Return a stable, serializable row for benchmark packets."""
        return {
            "target": self.target,
            "operation": self.operation,
            "shape_bucket": list(self.shape_bucket),
            "dtype": self.dtype,
            "policy": self.policy,
            "retained_residual_bytes": self.retained_residual_bytes,
            "backward_samples_ns": list(self.backward_samples_ns),
            "backward_median_ns": self.backward_median_ns,
            "backward_p95_ns": self.backward_p95_ns,
            "forward_steps": self.forward_steps,
            "replayed_steps": self.replayed_steps,
            "backward_steps": self.backward_steps,
            "timing_domain": self.timing_domain,
            "provenance": self.provenance,
            "exact_device": self.exact_device,
            "selector_eligible": self.selector_eligible,
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


@dataclass(frozen=True)
class RegionResidualSlot:
    """One content-addressed state checkpoint crossing the forward/backward ABI."""

    checkpoint_index: int
    logical_name: str
    retained_bytes: int

    def __post_init__(self) -> None:
        if self.checkpoint_index <= 0 or not self.logical_name:
            raise ValueError("region residual slots require positive identity")
        if self.retained_bytes <= 0:
            raise ValueError("region residual slots require retained bytes")


@dataclass(frozen=True)
class RegionResidualABI:
    """Typed SAVE/HYBRID/RECOMPUTE boundary for one structured CFG."""

    policy: Literal["save", "recompute", "hybrid"]
    cfg_digest: str
    steps: int
    state_dtype: str
    state_shape: tuple[int, ...]
    slots: tuple[RegionResidualSlot, ...]
    schema: str = REGION_RESIDUAL_ABI_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REGION_RESIDUAL_ABI_SCHEMA:
            raise ValueError("unsupported region residual ABI schema")
        if (
            len(self.cfg_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.cfg_digest)
        ):
            raise ValueError("region residual ABI requires a CFG SHA-256 digest")
        if self.steps <= 0 or not self.state_dtype:
            raise ValueError("region residual ABI requires steps and state dtype")
        indices = tuple(slot.checkpoint_index for slot in self.slots)
        if indices != tuple(sorted(set(indices))) or any(
            index >= self.steps for index in indices
        ):
            raise ValueError("region residual slots must be unique interior checkpoints")
        expected = (
            "recompute"
            if not indices
            else "save"
            if len(indices) == self.steps - 1
            else "hybrid"
        )
        if self.policy != expected:
            raise ValueError(
                f"region residual policy {self.policy!r} disagrees with {indices}"
            )

    @property
    def checkpoint_indices(self) -> tuple[int, ...]:
        return tuple(slot.checkpoint_index for slot in self.slots)

    @property
    def retained_bytes(self) -> int:
        return sum(slot.retained_bytes for slot in self.slots)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy": self.policy,
            "cfg_digest": self.cfg_digest,
            "steps": self.steps,
            "state_dtype": self.state_dtype,
            "state_shape": list(self.state_shape),
            "slots": [slot.__dict__ for slot in self.slots],
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def compiler_attributes(self) -> dict[str, object]:
        """Return the lossless Graph/MLIR checkpoint carrier.

        SAVE and HYBRID are not selector labels: the lowering must receive the
        exact structured-CFG identity and retained step set that the evaluator
        executed.  Keep these names synchronized with
        ``LowerControlFlowToSCFPass``; that pass rejects incomplete or
        contradictory contracts instead of falling back to recomputation.
        """

        return {
            "tessera.autodiff.checkpoint_policy": (
                "recompute_all" if self.policy == "recompute" else self.policy
            ),
            "tessera.autodiff.checkpoint_indices": list(self.checkpoint_indices),
            "tessera.autodiff.residual_schema": self.schema,
            "tessera.autodiff.residual_digest": self.digest,
            "tessera.structured_cfg.digest": self.cfg_digest,
            "tessera.autodiff.residual_state_dtype": self.state_dtype,
            "tessera.autodiff.residual_state_shape": list(self.state_shape),
        }


def build_region_residual_abi(
    candidate: TreeverseCandidate,
    *,
    cfg_digest: str,
    state_dtype: str,
    state_shape: Sequence[int],
) -> RegionResidualABI:
    """Materialize a candidate as an explicit executable residual contract."""

    if not candidate.checkpoint_indices:
        policy: Literal["save", "recompute", "hybrid"] = "recompute"
    elif len(candidate.checkpoint_indices) == candidate.steps - 1:
        policy = "save"
    else:
        policy = "hybrid"
    per_slot = (
        candidate.retained_residual_bytes // len(candidate.checkpoint_indices)
        if candidate.checkpoint_indices
        else 0
    )
    slots = tuple(
        RegionResidualSlot(index, f"state_after_{index}", per_slot)
        for index in candidate.checkpoint_indices
    )
    abi = RegionResidualABI(
        policy=policy,
        cfg_digest=cfg_digest,
        steps=candidate.steps,
        state_dtype=state_dtype,
        state_shape=tuple(int(dim) for dim in state_shape),
        slots=slots,
    )
    if abi.retained_bytes != candidate.retained_residual_bytes:
        raise ValueError("candidate retained bytes are not divisible across slots")
    return abi


@dataclass(frozen=True)
class TreeverseExecution:
    """Observed execution of one candidate over a differentiable region."""

    input_cotangent: Any
    forward_steps: int
    replayed_steps: int
    backward_steps: int
    checkpoint_indices: tuple[int, ...]


@dataclass(frozen=True)
class _BorrowedState:
    """Marks the caller-owned initial state as non-retained residual memory."""

    value: Any


def _clone_state(value: Any) -> Any:
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else copy.deepcopy(value)


def capture_treeverse_forward(
    *,
    candidate: TreeverseCandidate,
    initial_state: Any,
    forward_step: Callable[[int, Any], Any],
    clone_state: Callable[[Any], Any] = _clone_state,
    residual_abi: RegionResidualABI | None = None,
) -> ForwardCapture:
    """Execute the forward region and retain exactly the selected checkpoints."""
    if residual_abi is not None and (
        residual_abi.steps != candidate.steps
        or residual_abi.checkpoint_indices != candidate.checkpoint_indices
    ):
        raise ValueError("region residual ABI does not match its treeverse candidate")
    state = clone_state(initial_state)
    checkpoints: dict[int, Any] = {}
    selected = set(candidate.checkpoint_indices)
    for index in range(candidate.steps):
        state = forward_step(index, state)
        if index + 1 in selected:
            checkpoints[index + 1] = clone_state(state)
    return ForwardCapture(
        output=state,
        residuals={
            "candidate": candidate,
            "borrowed_initial": _BorrowedState(initial_state),
            "checkpoints": checkpoints,
            "residual_abi": residual_abi,
        },
    )


def execute_treeverse_region_adjoint(
    *,
    cotangent: Any,
    residuals: Mapping[str, Any],
    forward_step: Callable[[int, Any], Any],
    backward_step: Callable[[int, Any, Any, Any], Any],
    clone_state: Callable[[Any], Any] = _clone_state,
) -> TreeverseExecution:
    """Execute a counted-region adjoint with bounded checkpoint storage.

    Each required primal is replayed from the closest retained checkpoint.
    The implementation intentionally does not retain an entire segment: its
    replay count therefore matches the candidate's triangular work envelope.
    ``backward_step`` receives ``(index, state_before, state_after, cotangent)``.
    """
    candidate = residuals.get("candidate")
    checkpoints = residuals.get("checkpoints")
    borrowed = residuals.get("borrowed_initial")
    residual_abi = residuals.get("residual_abi")
    if (
        not isinstance(candidate, TreeverseCandidate)
        or not isinstance(checkpoints, Mapping)
        or not isinstance(borrowed, _BorrowedState)
    ):
        raise TypeError("treeverse residuals must come from capture_treeverse_forward")
    if residual_abi is not None and (
        not isinstance(residual_abi, RegionResidualABI)
        or residual_abi.steps != candidate.steps
        or residual_abi.checkpoint_indices != candidate.checkpoint_indices
    ):
        raise ValueError("region residual ABI identity does not match the execution")
    if tuple(sorted(int(i) for i in checkpoints)) != candidate.checkpoint_indices:
        raise ValueError("treeverse checkpoint identity does not match the candidate")
    available_checkpoints = {0: borrowed.value, **checkpoints}

    boundaries = (0, *candidate.checkpoint_indices, candidate.steps)
    adjoint = cotangent
    replayed = 0
    backward_steps = 0
    for segment in range(len(boundaries) - 2, -1, -1):
        start, end = boundaries[segment], boundaries[segment + 1]
        start_state = available_checkpoints.get(start)
        if start_state is None:
            raise ValueError(f"missing treeverse checkpoint at step {start}")
        for index in range(end - 1, start - 1, -1):
            state = clone_state(start_state)
            for replay_index in range(start, index):
                state = forward_step(replay_index, state)
                replayed += 1
            after = forward_step(index, state)
            adjoint = backward_step(index, state, after, adjoint)
            backward_steps += 1
    if replayed != candidate.estimated_replayed_steps:
        raise RuntimeError("executed treeverse replay count disagrees with its plan")
    return TreeverseExecution(
        input_cotangent=adjoint,
        forward_steps=candidate.steps + replayed,
        replayed_steps=replayed,
        backward_steps=backward_steps,
        checkpoint_indices=candidate.checkpoint_indices,
    )


def measure_treeverse_region_candidate(
    *,
    target: str,
    operation: str,
    shape_bucket: Sequence[int],
    dtype: str,
    candidate: TreeverseCandidate,
    initial_state: Any,
    forward_step: Callable[[int, Any], Any],
    backward_step: Callable[[int, Any, Any, Any], Any],
    cotangent: Any,
    synchronize: Callable[[], None] | None = None,
    warmup: int = 3,
    repetitions: int = 20,
    timing_domain: str = "synchronized_host_wall",
    provenance: str,
    exact_device: bool,
) -> ResidualEvidence:
    """Measure an actually executed treeverse region adjoint.

    This is the promotion bridge missing from :func:`treeverse_candidates`:
    estimated envelopes remain ineligible, while the returned evidence may
    select only when it was executed on the exact target device.
    """

    def forward() -> ForwardCapture:
        return capture_treeverse_forward(
            candidate=candidate,
            initial_state=initial_state,
            forward_step=forward_step,
        )

    executions: list[TreeverseExecution] = []

    def backward(ct: Any, saved: Any) -> Any:
        execution = execute_treeverse_region_adjoint(
            cotangent=ct,
            residuals=saved,
            forward_step=forward_step,
            backward_step=backward_step,
        )
        executions.append(execution)
        return execution.input_cotangent

    if not candidate.checkpoint_indices:
        executed_policy: ResidualPolicy = "recompute"
    elif candidate.checkpoint_slots >= max(candidate.steps - 1, 0):
        executed_policy = "save"
    else:
        executed_policy = "hybrid"

    evidence = measure_residual_candidate(
        target=target,
        operation=operation,
        shape_bucket=shape_bucket,
        dtype=dtype,
        policy=executed_policy,
        forward=forward,
        backward=backward,
        cotangent=cotangent,
        synchronize=synchronize,
        warmup=warmup,
        repetitions=repetitions,
        timing_domain=timing_domain,
        provenance=provenance,
        exact_device=exact_device,
    )
    if not executions:
        raise RuntimeError("treeverse measurement did not execute a backward")
    work = executions[-1]
    if any(
        (row.forward_steps, row.replayed_steps, row.backward_steps)
        != (work.forward_steps, work.replayed_steps, work.backward_steps)
        for row in executions
    ):
        raise RuntimeError("treeverse execution work changed between samples")
    return replace(
        evidence,
        forward_steps=work.forward_steps,
        replayed_steps=work.replayed_steps,
        backward_steps=work.backward_steps,
    )


def measure_counted_region_policy_cohort(
    *,
    target: str,
    operation: str,
    shape_bucket: Sequence[int],
    dtype: str,
    steps: int,
    state_bytes: int,
    measured_step_work_ns: int,
    memory_budgets: Iterable[int],
    initial_state: Any,
    forward_step: Callable[[int, Any], Any],
    backward_step: Callable[[int, Any, Any, Any], Any],
    cotangent: Any,
    synchronize: Callable[[], None] | None = None,
    warmup: int = 3,
    repetitions: int = 20,
    timing_domain: str = "synchronized_host_wall",
    provenance: str,
    exact_device: bool,
) -> tuple[ResidualEvidence, ...]:
    """Execute the SAVE/RECOMPUTE/HYBRID checkpoint cohort.

    The candidate generator only prunes checkpoint placements. Every returned
    row comes from a complete backward execution, is labelled by the policy it
    actually implements, and carries exact replay/backward work counters.
    """
    candidates = treeverse_candidates(
        steps=steps,
        state_bytes=state_bytes,
        measured_step_work_ns=measured_step_work_ns,
        memory_budgets=memory_budgets,
    )
    return tuple(
        measure_treeverse_region_candidate(
            target=target,
            operation=operation,
            shape_bucket=shape_bucket,
            dtype=dtype,
            candidate=candidate,
            initial_state=initial_state,
            forward_step=forward_step,
            backward_step=backward_step,
            cotangent=cotangent,
            synchronize=synchronize,
            warmup=warmup,
            repetitions=repetitions,
            timing_domain=timing_domain,
            provenance=provenance,
            exact_device=exact_device,
        )
        for candidate in candidates
    )


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


def select_residual_policy(evidence: Iterable[ResidualEvidence], *, memory_budget_bytes: int) -> ResidualEvidence:
    """Select the fastest eligible policy that fits the residual budget."""
    if memory_budget_bytes < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    rows = list(evidence)
    if not rows:
        raise ValueError("at least one residual candidate is required")
    identity = {(row.target, row.operation, row.shape_bucket, row.dtype) for row in rows}
    if len(identity) != 1:
        raise ValueError("residual candidates must describe one exact workload")
    eligible = [row for row in rows if row.selector_eligible and row.retained_residual_bytes <= memory_budget_bytes]
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
    *,
    steps: int,
    state_bytes: int,
    measured_step_work_ns: int,
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
            indices = tuple(sorted({round(i * steps / (slots + 1)) for i in range(1, slots + 1)}))
        else:
            indices = ()
        boundaries = (0, *indices, steps)
        replayed = sum(
            length * max(length - 1, 0) // 2
            for length in (boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))
        )
        candidates.append(
            TreeverseCandidate(
                steps=steps,
                checkpoint_slots=len(indices),
                checkpoint_indices=indices,
                retained_residual_bytes=len(indices) * state_bytes,
                estimated_replayed_steps=replayed,
                estimated_backward_work_ns=(steps + replayed) * measured_step_work_ns,
            )
        )
    return tuple(candidates)


__all__ = [
    "ForwardCapture",
    "ResidualEvidence",
    "ResidualPolicy",
    "REGION_RESIDUAL_ABI_SCHEMA",
    "RegionResidualABI",
    "RegionResidualSlot",
    "TreeverseCandidate",
    "TreeverseExecution",
    "capture_treeverse_forward",
    "build_region_residual_abi",
    "execute_treeverse_region_adjoint",
    "measure_treeverse_region_candidate",
    "measure_counted_region_policy_cohort",
    "measure_residual_candidate",
    "retained_residual_bytes",
    "select_residual_policy",
    "treeverse_candidates",
]
