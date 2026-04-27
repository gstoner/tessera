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

Reference: CLAUDE.md §Phase 4 — PipelinePlan
           src/transforms/lib/PipelineStageInsertionPass.cpp
           "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al. 2021)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, NamedTuple, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Schedule step
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    FORWARD  = "F"
    BACKWARD = "B"
    IDLE     = "_"   # bubble


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

    def __post_init__(self) -> None:
        if self.num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {self.num_stages}")
        if self.num_micro_batches < 1:
            raise ValueError(f"num_micro_batches must be >= 1, got {self.num_micro_batches}")
        if self.interleaved:
            if self.num_chunks < 2:
                raise ValueError(
                    "Interleaved 1F1B requires num_chunks >= 2"
                )
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
        where p = num_stages, m = num_micro_batches, v = num_chunks
        """
        p = self.num_stages
        m = self.num_micro_batches
        if self.interleaved and self.num_chunks > 1:
            return (p - 1) / (m * self.num_chunks)
        return (p - 1) / m

    @property
    def warmup_steps(self) -> int:
        """
        Number of forward-only steps before the first backward can begin.
        This is the pipeline fill time: p - 1 steps.
        """
        return self.num_stages - 1

    def total_clocks(self) -> int:
        """Total clock cycles for the complete schedule (all ranks, all micro-batches)."""
        # Fill: (p-1) fwd-only + m*(F+B) steady-state + drain (p-1) bwd-only
        p = self.num_stages
        m = self.num_micro_batches
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
        if self.interleaved:
            return self._build_interleaved()
        return self._build_standard()

    def _build_standard(self) -> List[ScheduleStep]:
        """Standard 1F1B (non-interleaved) schedule."""
        p = self.num_stages
        m = self.num_micro_batches
        steps: List[ScheduleStep] = []

        # Per-rank state: next micro-batch to forward and backward
        fwd_mb = list(range(p))      # rank k starts at micro-batch k
        bwd_mb = [0] * p             # backward starts after warmup
        fwd_ptr = [0] * p
        bwd_ptr = [-1] * p           # -1 = not yet started

        # We use a simpler direct formulation:
        # For rank r (stage r), forward of micro-batch m starts at clock (r + m)
        # Backward of micro-batch m starts at clock (r + m + p)

        for mb in range(m):
            for rank in range(p):
                # Forward
                fwd_clock = rank + mb
                steps.append(ScheduleStep(
                    clock=fwd_clock, rank=rank, stage=rank,
                    micro_batch=mb, phase=Phase.FORWARD,
                ))
                # Backward (mirrored: last stage finishes backward first)
                # In standard 1F1B, backward of mb on rank r starts at:
                #   clock = (p - 1 - rank) + mb + p
                bwd_clock = (p - 1 - rank) + mb + p
                steps.append(ScheduleStep(
                    clock=bwd_clock, rank=rank, stage=rank,
                    micro_batch=mb, phase=Phase.BACKWARD,
                ))

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
                    steps.append(ScheduleStep(
                        clock=fwd_clock, rank=rank, stage=virtual_stage,
                        micro_batch=mb, phase=Phase.FORWARD,
                    ))
                    steps.append(ScheduleStep(
                        clock=bwd_clock, rank=rank, stage=virtual_stage,
                        micro_batch=mb, phase=Phase.BACKWARD,
                    ))

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
            ["__"] * (max_clock + 1)
            for _ in range(self.num_stages)
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
        """Serialize for PipelineStageInsertionPass."""
        return (
            f'{{tessera.pipeline_plan = {{'
            f'num_stages = {self.num_stages}, '
            f'num_micro_batches = {self.num_micro_batches}, '
            f'interleaved = {"true" if self.interleaved else "false"}, '
            f'num_chunks = {self.num_chunks}}}}}'
        )

    def __repr__(self) -> str:
        mode = f", interleaved, v={self.num_chunks}" if self.interleaved else ""
        return (
            f"PipelinePlan(stages={self.num_stages}, "
            f"micro_batches={self.num_micro_batches}{mode}, "
            f"bubble={self.bubble_fraction:.2%})"
        )
