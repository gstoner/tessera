"""
queue_dialect.py — Python bindings for the Tessera Queue dialect.

The Queue dialect models the producer–consumer token flow between warp-
specialized execution stages (e.g., TMA producer warps / WGMMA consumer
warps in a Hopper flash-attention kernel).

Dialect ops:
    tessera.queue.create   — allocate a token queue between P and C warps
    tessera.queue.push     — producer deposits a token (signals data-ready)
    tessera.queue.pop      — consumer waits for a token (stalls until ready)
    tessera.queue.barrier  — full barrier (sync all outstanding tokens)

Each builder function returns a ``QueueNode`` that can be serialised to MLIR
text or composed into a ``WarpPipeline``.

Usage::

    from tessera.compiler.queue_dialect import WarpPipelineBuilder

    pipe = WarpPipelineBuilder(stages=3, queue_depth=2)
    nodes = pipe.build()
    for node in nodes:
        print(node.to_mlir())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Queue role
# ---------------------------------------------------------------------------

class WarpRole(Enum):
    PRODUCER = "producer"   # async-copy / TMA warps
    CONSUMER = "consumer"   # WGMMA / tensor-core warps


# ---------------------------------------------------------------------------
# Queue node descriptors
# ---------------------------------------------------------------------------

@dataclass
class QueueCreateNode:
    """
    tessera.queue.create

    Allocates a token queue between producer and consumer warp groups.

    Attributes
    ----------
    queue_id  : unique integer ID (used for inter-warp identification)
    depth     : maximum number of outstanding tokens (≥1)
    dtype     : element type flowing through the queue (informational)
    producer_warps : number of warp groups on the produce side
    consumer_warps : number of warp groups on the consume side
    """
    queue_id: int
    depth: int = 1
    dtype: str = "bf16"
    producer_warps: int = 1
    consumer_warps: int = 1

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"queue depth must be >= 1, got {self.depth}")
        if self.queue_id < 0:
            raise ValueError(f"queue_id must be >= 0")

    def to_mlir(self) -> str:
        lines = [
            f'%queue_{self.queue_id} = "tessera.queue.create"() {{',
            f'  queue_id = {self.queue_id} : i64,',
            f'  depth = {self.depth} : i64,',
            f'  producer_warps = {self.producer_warps} : i64,',
            f'  consumer_warps = {self.consumer_warps} : i64,',
            f'  dtype = "{self.dtype}"',
            f'}} : () -> !tessera.queue<{self.dtype}>',
        ]
        return "\n".join(lines)


@dataclass
class QueuePushNode:
    """
    tessera.queue.push

    Producer deposits a token (signals that a tile has been loaded).

    Attributes
    ----------
    queue_id  : matching queue_id from QueueCreateNode
    stage     : which pipeline stage this push belongs to (for double-buffer)
    data_val  : MLIR SSA name of the tile data value
    token_val : result name for the async push token
    """
    queue_id: int
    stage: int = 0
    data_val: str = "%tile"
    token_val: str = "%push_token"

    def to_mlir(self) -> str:
        lines = [
            f'{self.token_val} = "tessera.queue.push"({self.data_val}, %queue_{self.queue_id}) {{',
            f'  stage = {self.stage} : i64',
            f'}} : (!tessera.tile, !tessera.queue<bf16>) -> !tessera.async_token',
        ]
        return "\n".join(lines)


@dataclass
class QueuePopNode:
    """
    tessera.queue.pop

    Consumer waits for a token (blocking until the producer has signalled).

    Attributes
    ----------
    queue_id  : matching queue_id
    stage     : pipeline stage
    out_val   : result SSA name of the dequeued tile
    """
    queue_id: int
    stage: int = 0
    out_val: str = "%consumed_tile"

    def to_mlir(self) -> str:
        lines = [
            f'{self.out_val} = "tessera.queue.pop"(%queue_{self.queue_id}) {{',
            f'  stage = {self.stage} : i64',
            f'}} : (!tessera.queue<bf16>) -> !tessera.tile',
        ]
        return "\n".join(lines)


@dataclass
class QueueBarrierNode:
    """
    tessera.queue.barrier

    Full barrier — waits for all outstanding push tokens on this queue
    before proceeding.

    Attributes
    ----------
    queue_id : matching queue_id
    scope    : "warpgroup" | "block" | "device"
    """
    queue_id: int
    scope: str = "warpgroup"

    _VALID_SCOPES = {"warpgroup", "block", "device"}

    def __post_init__(self) -> None:
        if self.scope not in self._VALID_SCOPES:
            raise ValueError(f"scope must be one of {self._VALID_SCOPES}")

    def to_mlir(self) -> str:
        lines = [
            f'"tessera.queue.barrier"(%queue_{self.queue_id}) {{',
            f'  scope = "{self.scope}"',
            f'}} : (!tessera.queue<bf16>) -> ()',
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Warp pipeline descriptor
# ---------------------------------------------------------------------------

@dataclass
class WarpStage:
    """One stage of the warp pipeline."""
    stage_id: int
    role: WarpRole
    queue_id: int
    depth: int = 1

    def push_node(self, data_val: str = "%tile") -> QueuePushNode:
        return QueuePushNode(
            queue_id=self.queue_id,
            stage=self.stage_id,
            data_val=data_val,
            token_val=f"%push_token_{self.stage_id}",
        )

    def pop_node(self) -> QueuePopNode:
        return QueuePopNode(
            queue_id=self.queue_id,
            stage=self.stage_id,
            out_val=f"%consumed_tile_{self.stage_id}",
        )


class WarpPipeline:
    """
    A complete warp-specialised pipeline consisting of N stages,
    each with a queue connecting producer and consumer warps.

    Attributes
    ----------
    stages      : list of WarpStage descriptors
    double_buf  : if True, depth = 2 for all queues (double-buffering)
    """

    def __init__(
        self,
        stages: List[WarpStage],
        double_buf: bool = False,
    ) -> None:
        self.stages = stages
        self.double_buf = double_buf

    def create_nodes(self) -> List[QueueCreateNode]:
        depth = 2 if self.double_buf else 1
        seen = set()
        nodes = []
        for s in self.stages:
            if s.queue_id not in seen:
                nodes.append(QueueCreateNode(
                    queue_id=s.queue_id,
                    depth=depth,
                ))
                seen.add(s.queue_id)
        return nodes

    def to_mlir(self) -> str:
        lines = ["// tessera.queue Warp Pipeline"]
        for node in self.create_nodes():
            lines.append(node.to_mlir())
        lines.append("")
        for stage in self.stages:
            if stage.role == WarpRole.PRODUCER:
                lines.append(f"// Stage {stage.stage_id} — Producer")
                lines.append(stage.push_node().to_mlir())
            else:
                lines.append(f"// Stage {stage.stage_id} — Consumer")
                lines.append(stage.pop_node().to_mlir())
        # Final barrier
        if self.stages:
            last_qid = self.stages[-1].queue_id
            lines.append(QueueBarrierNode(queue_id=last_qid).to_mlir())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class WarpPipelineBuilder:
    """
    Constructs a WarpPipeline with alternating producer/consumer stages.

    Parameters
    ----------
    stages      : total number of pipeline stages (P + C pairs)
    queue_depth : depth per queue (1 = single-buffer, 2 = double-buffer)
    dtype       : element dtype flowing through queues
    """

    def __init__(
        self,
        stages: int = 2,
        queue_depth: int = 1,
        dtype: str = "bf16",
    ) -> None:
        if stages < 1:
            raise ValueError(f"stages must be >= 1, got {stages}")
        if queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1, got {queue_depth}")
        self.n_stages = stages
        self.queue_depth = queue_depth
        self.dtype = dtype

    def build(self) -> WarpPipeline:
        """
        Build a pipeline with alternating producer/consumer stage pairs.

        Stage pattern for n_stages=3:
            stage 0 (queue 0): PRODUCER
            stage 1 (queue 0): CONSUMER → stage 1 (queue 1): PRODUCER
            stage 2 (queue 1): CONSUMER
        """
        warp_stages: List[WarpStage] = []
        for i in range(self.n_stages):
            role = WarpRole.PRODUCER if i % 2 == 0 else WarpRole.CONSUMER
            queue_id = i // 2
            warp_stages.append(
                WarpStage(
                    stage_id=i,
                    role=role,
                    queue_id=queue_id,
                    depth=self.queue_depth,
                )
            )
        return WarpPipeline(
            stages=warp_stages,
            double_buf=(self.queue_depth >= 2),
        )

    def build_fa2_pattern(
        self,
        num_tma_warps: int = 1,
        num_wgmma_warps: int = 1,
    ) -> WarpPipeline:
        """
        Build the canonical FA-2 warp pipeline:
            TMA producer warp  → queue 0 → WGMMA consumer warps

        Parameters
        ----------
        num_tma_warps   : producer warp groups
        num_wgmma_warps : consumer warp groups
        """
        depth = max(self.queue_depth, 2)  # FA-2 always double-buffers
        create = QueueCreateNode(
            queue_id=0,
            depth=depth,
            dtype=self.dtype,
            producer_warps=num_tma_warps,
            consumer_warps=num_wgmma_warps,
        )
        stages = [
            WarpStage(stage_id=0, role=WarpRole.PRODUCER, queue_id=0, depth=depth),
            WarpStage(stage_id=1, role=WarpRole.CONSUMER, queue_id=0, depth=depth),
        ]
        pipe = WarpPipeline(stages=stages, double_buf=(depth >= 2))
        return pipe
