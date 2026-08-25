"""
tessera.compiler.pipeline_planner — 1F1B pipeline schedule builder.

Implements the one-forward-one-backward (1F1B) pipeline schedule from
GPipe/PipeDream. The schedule determines at each clock cycle which micro-batch
and which pipeline stage is executing a forward or backward pass.

1F1B properties:
  - Steady-state memory: only (num_stages) activations live simultaneously,
    versus GPipe's (num_stages × num_micro_batches) memory footprint
  - Throughput: approaches 100% device utilization as num_micro_batches → ∞
  - Latency: pipeline_depth = num_stages - 1 bubbles before steady state

Interleaved 1F1B (Megatron-LM variant):
  - Each rank holds multiple virtual pipeline stages (chunks)
  - Reduces bubble fraction from (p-1)/m to (p-1)/(m*v) where v = chunks/rank
  - Requires num_micro_batches >= num_stages * num_chunks

Decoupled-stage (local-objective) schedule:
  - Models block-local training where each stage owns a self-contained
    objective and trains directly from data — no cross-stage forward/backward
    activation dependency (DiffusionBlocks, arXiv:2506.14202). Because no stage
    waits on an upstream activation, there is no pipeline fill/drain: every rank
    runs forward+backward on its own micro-batch every step.
  - bubble fraction = 0, warmup = 0. This is the *scheduling* dual of the
    decoupled-block memory lever (checkpoint.CheckpointPolicy.DECOUPLED_BLOCK);
    see docs/audit/roadmap/decoupled_stage_pipeline.md. Experimental: valid only
    for genuinely decoupled per-stage objectives, not standard backprop.

Reference: CLAUDE.md §Phase 4 — PipelinePlan
           src/transforms/lib/PipelineStageInsertionPass.cpp
           "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al. 2021)
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Optional
from .benchmark_row import MeasuredResourceVector
from .schedule_object import ScheduleAction, ScheduleObject, ScheduleRole

# ─────────────────────────────────────────────────────────────────────────────
# Schedule step
# ─────────────────────────────────────────────────────────────────────────────


class Phase(Enum):
    FORWARD = "F"
    BACKWARD = "B"
    IDLE = "_"  # bubble


class ScheduleStep(NamedTuple):
    """
    One unit of work in the 1F1B schedule.

    Attributes:
        clock       : global clock tick
        rank        : which pipeline rank (device) executes this step
        stage       : pipeline stage index
        micro_batch : micro-batch index (0-based)
        phase       : FORWARD or BACKWARD
    """

    clock: int
    rank: int
    stage: int
    micro_batch: int
    phase: Phase


# ─────────────────────────────────────────────────────────────────────────────
# PipelinePlan
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelinePlan:
    """
    1F1B pipeline parallelism schedule.

    Attributes:
        num_stages     : number of pipeline stages (= number of ranks in pp axis)
        num_micro_batches: number of micro-batches per global batch
        interleaved    : if True, use interleaved 1F1B (requires num_chunks > 1)
        num_chunks     : virtual chunks per rank for interleaved schedule
        decoupled      : if True, use the decoupled-stage (local-objective)
                         schedule — zero bubble, no cross-stage activation
                         dependency (mutually exclusive with interleaved)

    Example:
        plan = PipelinePlan(num_stages=4, num_micro_batches=8)
        steps = plan.schedule_steps()
        # Returns list of ScheduleStep objects for all ranks and clocks

        # Inspect stage 0 at clock 0
        step0 = next(s for s in steps if s.rank == 0 and s.clock == 0)
        assert step0.phase == Phase.FORWARD
        assert step0.micro_batch == 0
    """

    num_stages: int
    num_micro_batches: int
    interleaved: bool = False
    num_chunks: int = 1
    decoupled: bool = False

    def __post_init__(self) -> None:
        if self.num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {self.num_stages}")
        if self.num_micro_batches < 1:
            raise ValueError(
                f"num_micro_batches must be >= 1, got {self.num_micro_batches}"
            )
        if self.decoupled and self.interleaved:
            raise ValueError(
                "decoupled and interleaved schedules are mutually exclusive"
            )
        if self.interleaved:
            if self.num_chunks < 2:
                raise ValueError("Interleaved 1F1B requires num_chunks >= 2")
            min_m = self.num_stages * self.num_chunks
            if self.num_micro_batches < min_m:
                raise ValueError(
                    f"Interleaved 1F1B requires num_micro_batches >= "
                    f"num_stages × num_chunks = {min_m}, got {self.num_micro_batches}"
                )

    @property
    def bubble_fraction(self) -> float:
        """
        Fraction of total compute cycles that are idle (pipeline bubbles).

        Standard 1F1B:    bubble = (p - 1) / m
        Interleaved 1F1B: bubble = (p - 1) / (m × v)
        Decoupled-stage:  bubble = 0  (no cross-stage activation dependency)
        where p = num_stages, m = num_micro_batches, v = num_chunks
        """
        if self.decoupled:
            # Stages train from data with no upstream dependency: every rank is
            # busy every clock, so there is no fill/drain bubble.
            return 0.0
        p = self.num_stages
        m = self.num_micro_batches
        if self.interleaved and self.num_chunks > 1:
            return (p - 1) / (m * self.num_chunks)
        return (p - 1) / m

    @property
    def warmup_steps(self) -> int:
        """
        Number of forward-only steps before the first backward can begin.
        This is the pipeline fill time: p - 1 steps (0 for decoupled stages,
        which never wait on an upstream forward).
        """
        if self.decoupled:
            return 0
        return self.num_stages - 1

    def total_clocks(self) -> int:
        """Total clock cycles for the complete schedule (all ranks, all micro-batches)."""
        p = self.num_stages
        m = self.num_micro_batches
        if self.decoupled:
            # Each rank runs F then B per micro-batch with no inter-stage
            # dependency; all ranks proceed in lockstep → 2m clocks, no bubble.
            return 2 * m
        # Fill: (p-1) fwd-only + m*(F+B) steady-state + drain (p-1) bwd-only
        return (p - 1) + m + m + (p - 1)

    def schedule_steps(self) -> List[ScheduleStep]:
        """
        Generate the full 1F1B schedule as a flat list of ScheduleStep objects.

        Each step describes exactly one unit of work: which rank executes a
        forward or backward pass for which micro-batch at which clock tick.

        Returns:
            List[ScheduleStep] in clock order (ties broken by rank).

        The schedule follows the standard GPipe 1F1B pattern:
          Phase 1 (warmup): ranks fill the pipeline with forward passes
          Phase 2 (steady): alternating F and B, one per clock per rank
          Phase 3 (drain):  ranks flush remaining backward passes
        """
        if self.decoupled:
            return self._build_decoupled()
        if self.interleaved:
            return self._build_interleaved()
        return self._build_standard()

    def _build_decoupled(self) -> List[ScheduleStep]:
        """
        Decoupled-stage (local-objective) schedule.

        Each stage owns a self-contained objective and trains directly from
        data, so there is no cross-stage activation dependency: every rank runs
        forward then backward on its own micro-batch with no fill/drain. Rank r
        processes micro-batch mb with forward at clock 2*mb and backward at
        clock 2*mb+1 — identical across ranks, zero bubble.
        """
        p = self.num_stages
        m = self.num_micro_batches
        steps: List[ScheduleStep] = []
        for mb in range(m):
            for rank in range(p):
                steps.append(
                    ScheduleStep(
                        clock=2 * mb,
                        rank=rank,
                        stage=rank,
                        micro_batch=mb,
                        phase=Phase.FORWARD,
                    )
                )
                steps.append(
                    ScheduleStep(
                        clock=2 * mb + 1,
                        rank=rank,
                        stage=rank,
                        micro_batch=mb,
                        phase=Phase.BACKWARD,
                    )
                )
        steps.sort(key=lambda s: (s.clock, s.rank))
        return steps

    def _build_standard(self) -> List[ScheduleStep]:
        """Standard 1F1B (non-interleaved) schedule."""
        p = self.num_stages
        m = self.num_micro_batches
        steps: List[ScheduleStep] = []

        # Per-rank state: next micro-batch to forward and backward
        fwd_mb = list(range(p))  # rank k starts at micro-batch k
        bwd_mb = [0] * p  # backward starts after warmup
        fwd_ptr = [0] * p
        bwd_ptr = [-1] * p  # -1 = not yet started

        # We use a simpler direct formulation:
        # For rank r (stage r), forward of micro-batch m starts at clock (r + m)
        # Backward of micro-batch m starts at clock (r + m + p)

        for mb in range(m):
            for rank in range(p):
                # Forward
                fwd_clock = rank + mb
                steps.append(
                    ScheduleStep(
                        clock=fwd_clock,
                        rank=rank,
                        stage=rank,
                        micro_batch=mb,
                        phase=Phase.FORWARD,
                    )
                )
                # Backward (mirrored: last stage finishes backward first)
                # In standard 1F1B, backward of mb on rank r starts at:
                #   clock = (p - 1 - rank) + mb + p
                bwd_clock = (p - 1 - rank) + mb + p
                steps.append(
                    ScheduleStep(
                        clock=bwd_clock,
                        rank=rank,
                        stage=rank,
                        micro_batch=mb,
                        phase=Phase.BACKWARD,
                    )
                )

        steps.sort(key=lambda s: (s.clock, s.rank))
        return steps

    def _build_interleaved(self) -> List[ScheduleStep]:
        """
        Interleaved 1F1B (Megatron-LM virtual pipeline stages).

        Each rank holds `num_chunks` virtual stages, reducing bubble fraction
        by factor v. Stage assignment: rank r, chunk c → virtual stage r + c*p.
        """
        p = self.num_stages
        m = self.num_micro_batches
        v = self.num_chunks
        steps: List[ScheduleStep] = []

        for chunk in range(v):
            for mb in range(m):
                for rank in range(p):
                    virtual_stage = rank + chunk * p
                    fwd_clock = rank + mb + chunk * p
                    bwd_clock = fwd_clock + p * v
                    steps.append(
                        ScheduleStep(
                            clock=fwd_clock,
                            rank=rank,
                            stage=virtual_stage,
                            micro_batch=mb,
                            phase=Phase.FORWARD,
                        )
                    )
                    steps.append(
                        ScheduleStep(
                            clock=bwd_clock,
                            rank=rank,
                            stage=virtual_stage,
                            micro_batch=mb,
                            phase=Phase.BACKWARD,
                        )
                    )

        steps.sort(key=lambda s: (s.clock, s.rank))
        return steps

    def render_ascii(self, max_clocks: Optional[int] = None) -> str:
        """
        Render the schedule as an ASCII timeline (useful for debugging).

        Each row is a rank (pipeline device), each column is a clock tick.
        F = forward, B = backward, _ = bubble.

        Example (4 stages, 4 micro-batches):
          rank 0: F0 F1 F2 F3 B0 B1 B2 B3
          rank 1: __ F0 F1 F2 B3 B0 B1 B2
          rank 2: __ __ F0 F1 B3 B2 B0 B1
          rank 3: __ __ __ F0 B3 B2 B1 B0
        """
        steps = self.schedule_steps()
        max_clock = max(s.clock for s in steps)
        if max_clocks is not None:
            max_clock = min(max_clock, max_clocks - 1)

        # Build grid[rank][clock] = label
        grid: List[List[str]] = [
            ["__"] * (max_clock + 1) for _ in range(self.num_stages)
        ]
        for step in steps:
            if step.clock > max_clock:
                continue
            label = f"{step.phase.value}{step.micro_batch}"
            grid[step.rank][step.clock] = label.ljust(2)

        lines = []
        for rank, row in enumerate(grid):
            lines.append(f"rank {rank}: " + " ".join(row))
        return "\n".join(lines)

    def to_mlir_attrs(self) -> str:
        """Materialize the digest-bound Schedule Object IR carrier.

        Lowering consumes this carrier directly instead of rebuilding a 1F1B
        schedule from a parallel scalar ``pipeline_plan``. Resource vectors
        remain out-of-band in :attr:`schedule_object`; IR carries the digest
        and the dependency/phase view needed by the pipeline passes.
        """
        schedule = self.schedule_object
        step_by_id = {self._action_id(step): step for step in self.schedule_steps()}
        rendered_steps = []
        for action in schedule.actions:
            step = step_by_id[action.action_id]
            dependencies = ", ".join(
                f'"{dependency}"' for dependency in action.depends_on
            )
            rendered_steps.append(
                "{"
                f'action_id = "{action.action_id}", '
                f"clock = {step.clock}, "
                f"depends_on = [{dependencies}], "
                f"micro_batch = {step.micro_batch}, "
                f'phase = "{step.phase.value}", '
                f"rank = {step.rank}, "
                f"stage = {step.stage}"
                "}"
            )
        return (
            "{"
            f'tessera.schedule_digest = "{schedule.digest}", '
            'tessera.pipeline_schedule_schema = "tessera.pipeline_schedule.v1", '
            f'tessera.pipeline_steps = [{", ".join(rendered_steps)}], '
            f"tessera.pp_num_stages = {self.num_stages}, "
            f"tessera.pp_num_micro_batches = {self.num_micro_batches}, "
            f'tessera.pp_interleaved = {"true" if self.interleaved else "false"}, '
            f'tessera.pp_decoupled = {"true" if self.decoupled else "false"}, '
            f"tessera.pp_num_chunks = {self.num_chunks}"
            "}"
        )

    @staticmethod
    def _action_id(step: ScheduleStep) -> str:
        return (
            f"pipeline:{step.clock}:{step.rank}:{step.stage}:"
            f"{step.micro_batch}:{step.phase.value}"
        )

    @property
    def schedule_object(self) -> ScheduleObject:
        """Return the one content-addressed authority for this pipeline."""

        steps = tuple(self.schedule_steps())
        # Key by the VIRTUAL STAGE, not the physical rank. Under interleaving
        # one rank owns several virtual stages, so a (rank, micro_batch, phase)
        # key collides across chunks: later chunks overwrite earlier ones, and
        # a forward step then resolves its producer to a future chunk (dropped
        # by the ordering filter below) or to nothing at all. Measured on
        # num_stages=4, num_chunks=2: 64 of 128 steps collided and 32 of 56
        # cross-stage forward steps lost their true producer edge, so the
        # emitted Schedule Object permitted a virtual stage to run before the
        # stage that feeds it (PR #626 review).
        by_key = {(step.stage, step.micro_batch, step.phase): step for step in steps}
        order_by_id = {self._action_id(step): index for index, step in enumerate(steps)}
        # Virtual stage count is DERIVED from the emitted schedule: under
        # interleaving it is num_stages x num_chunks, not num_stages.
        total_stages = max((step.stage for step in steps), default=-1) + 1
        inverted_backward: list[tuple[str, str]] = []
        previous_by_rank: dict[int, ScheduleStep] = {}
        actions: list[ScheduleAction] = []
        for step in steps:
            action_id = self._action_id(step)
            dependencies: set[str] = set()
            previous = previous_by_rank.get(step.rank)
            if previous is not None:
                dependencies.add(self._action_id(previous))
            if step.phase == Phase.FORWARD and step.stage > 0:
                upstream = by_key.get(
                    (step.stage - 1, step.micro_batch, Phase.FORWARD)
                )
                if upstream is not None:
                    dependencies.add(self._action_id(upstream))
            if step.phase == Phase.BACKWARD:
                own_forward = by_key.get(
                    (step.stage, step.micro_batch, Phase.FORWARD)
                )
                if own_forward is not None:
                    dependencies.add(self._action_id(own_forward))
                if step.stage + 1 < total_stages:
                    downstream = by_key.get(
                        (step.stage + 1, step.micro_batch, Phase.BACKWARD)
                    )
                    if downstream is not None:
                        downstream_id = self._action_id(downstream)
                        if order_by_id[downstream_id] < order_by_id[action_id]:
                            dependencies.add(downstream_id)
                        else:
                            # KNOWN PLANNER LIMITATION, surfaced by keying on
                            # the virtual stage (PR #626 review). The
                            # interleaved generator emits backward steps in
                            # ASCENDING stage order (num_stages=2, chunks=2,
                            # micro-batch 0: stage 0 B at clock 4 ... stage 3 B
                            # at clock 7), which is the opposite of gradient
                            # flow, so this edge cannot be expressed in the
                            # emitted order. Recording it would claim an
                            # ordering the schedule does not realize, so it is
                            # omitted and counted; fixing the generator's
                            # interleaved backward order is its own change.
                            inverted_backward.append((action_id, downstream_id))

            # A real producer ordered AFTER its consumer is a schedule defect,
            # not something to drop: silently filtering it would emit a
            # Schedule Object that permits the consumer to run first. Fail
            # closed instead (PR #626 review).
            late = sorted(
                dependency
                for dependency in dependencies
                if order_by_id[dependency] >= order_by_id[action_id]
            )
            # Forward/own-forward edges MUST precede their consumer in any
            # correct schedule; an inversion there is a defect, not a
            # limitation, so it fails closed rather than being dropped.
            if late:
                raise ValueError(
                    f"pipeline schedule places {action_id!r} before its "
                    f"producers {late!r}; the emitted Schedule Object would "
                    f"permit a stage to execute ahead of its input"
                )
            identity = json.dumps(
                {
                    "action_id": action_id,
                    "dependencies": sorted(dependencies),
                    "mode": (
                        "decoupled"
                        if self.decoupled
                        else "interleaved" if self.interleaved else "1f1b"
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            vector = MeasuredResourceVector(
                compute_time_ms=1.0,
                bytes_moved=0,
                communication_bytes=0,
                queue_identity=f"pipeline_rank:{step.rank}",
                resource_identity=f"pipeline_stage:{step.stage}",
                timing_provenance={
                    "source": "static_pipeline_model",
                    "domain": "compiler",
                },
                artifact_digest=hashlib.sha256(identity.encode()).hexdigest(),
            ).as_dict()
            actions.append(
                ScheduleAction(
                    action_id,
                    vector,
                    tuple(sorted(dependencies)),
                    op_ref=f"schedule.pipeline.{step.phase.value.lower()}",
                    scope=f"stage:{step.stage}",
                )
            )
            previous_by_rank[step.rank] = step

        mode = (
            "decoupled"
            if self.decoupled
            else "interleaved" if self.interleaved else "1f1b"
        )
        roles = tuple(
            ScheduleRole(f"stage_{stage}", (f"stage{stage}",))
            for stage in range(self.num_stages)
        )
        return ScheduleObject(
            object_id=(
                f"pipeline:{mode}:stages={self.num_stages}:"
                f"micro_batches={self.num_micro_batches}:chunks={self.num_chunks}"
            ),
            actions=tuple(actions),
            roles=roles,
        )

    def __repr__(self) -> str:
        if self.decoupled:
            mode = ", decoupled"
        elif self.interleaved:
            mode = f", interleaved, v={self.num_chunks}"
        else:
            mode = ""
        return (
            f"PipelinePlan(stages={self.num_stages}, "
            f"micro_batches={self.num_micro_batches}{mode}, "
            f"bubble={self.bubble_fraction:.2%})"
        )
