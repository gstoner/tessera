"""Runtime consumption of compiler-emitted 1F1B schedule steps.

Compute follows the compiler's dependency clock.  Backward collectives may run
on an independent transport executor and are joined before completion, making
overlap explicit without weakening the serialized compute proof.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class CollectiveDescriptor:
    """Serializable transport work owned by one emitted schedule step."""

    kind: str
    tensor: str
    axis: int = 0
    op: str = "sum"
    optimizer_shard: bool = False
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"all_reduce", "reduce_scatter", "all_gather", "all_to_all"}:
            raise ValueError(f"unsupported collective kind {self.kind!r}")
        if not self.tensor:
            raise ValueError("collective tensor identity must be non-empty")
        if self.axis < 0:
            raise ValueError("collective axis must be non-negative")
        if self.op not in {"sum", "max", "min"}:
            raise ValueError(f"unsupported collective reduction {self.op!r}")
        if self.optimizer_shard and self.kind not in {"reduce_scatter", "all_gather"}:
            raise ValueError("OptimizerShard transport must reduce-scatter or all-gather")


@dataclass(frozen=True)
class EmittedPipelineStep:
    clock: int
    micro_batch: int
    phase: str
    region: str
    stage: int
    collectives: tuple[CollectiveDescriptor, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PipelineRuntimeResult:
    completed_steps: tuple[EmittedPipelineStep, ...]
    collective_count: int
    peak_inflight_collectives: int


class OptimizerShardTransport:
    """Execute descriptor-bound ZeRO transport with explicit tensor ownership.

    ``tensors`` contains one host-visible value per rank. Native adapters move
    those values through NCCL/RCCL; the deterministic mock adapter exercises
    the same ownership state machine without claiming hardware evidence.
    """

    def __init__(self, adapter: Any, tensors: Mapping[str, Iterable[Any]]) -> None:
        self.adapter = adapter
        self._values = {name: list(values) for name, values in tensors.items()}
        self._ownership = {name: "replicated" for name in self._values}
        self._lock = Lock()

    @property
    def ownership(self) -> Mapping[str, str]:
        with self._lock:
            return dict(self._ownership)

    def values(self, tensor: str) -> tuple[Any, ...]:
        with self._lock:
            return tuple(self._values[tensor])

    def run(self, step: EmittedPipelineStep) -> None:
        for descriptor in step.collectives:
            self._execute(descriptor)

    def _execute(self, descriptor: CollectiveDescriptor) -> None:
        with self._lock:
            if descriptor.tensor not in self._values:
                raise KeyError(f"no runtime value for collective tensor {descriptor.tensor!r}")
            values = self._values[descriptor.tensor]
            ownership = self._ownership[descriptor.tensor]
            if descriptor.kind == "reduce_scatter":
                if descriptor.optimizer_shard and ownership != "replicated":
                    raise RuntimeError(
                        f"{descriptor.tensor} must be replicated before OptimizerShard reduce-scatter"
                    )
                outputs = self.adapter.reduce_scatter(
                    values, axis=descriptor.axis, op=descriptor.op
                )
                if descriptor.normalize:
                    outputs = [value / float(self.adapter.world_size) for value in outputs]
                self._ownership[descriptor.tensor] = "rank_local"
            elif descriptor.kind == "all_gather":
                if descriptor.optimizer_shard and ownership != "rank_local":
                    raise RuntimeError(
                        f"{descriptor.tensor} must be rank-local before OptimizerShard all-gather"
                    )
                outputs = self.adapter.all_gather(values, axis=descriptor.axis)
                self._ownership[descriptor.tensor] = "replicated"
            elif descriptor.kind == "all_reduce":
                outputs = self.adapter.all_reduce(values, op=descriptor.op)
                self._ownership[descriptor.tensor] = "replicated"
            else:
                outputs = self.adapter.all_to_all(
                    values,
                    scatter_axis=descriptor.axis,
                    gather_axis=descriptor.axis,
                )
            self._values[descriptor.tensor] = list(outputs)


def _parse_collectives(row: Mapping[str, Any]) -> tuple[CollectiveDescriptor, ...]:
    raw = row.get("collectives", ())
    if not isinstance(raw, (list, tuple)):
        raise ValueError("pipeline step collectives must be an array")
    return tuple(
        CollectiveDescriptor(
            kind=str(item["kind"]),
            tensor=str(item["tensor"]),
            axis=int(item.get("axis", 0)),
            op=str(item.get("op", "sum")),
            optimizer_shard=bool(item.get("optimizer_shard", False)),
            normalize=bool(item.get("normalize", False)),
        )
        for item in raw
    )


def parse_pipeline_steps(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[EmittedPipelineStep, ...]:
    steps = tuple(
        EmittedPipelineStep(
            clock=int(row["clock"]),
            micro_batch=int(row["micro_batch"]),
            phase=str(row["phase"]),
            region=str(row["region"]),
            stage=int(row["stage"]),
            collectives=_parse_collectives(row),
        )
        for row in rows
    )
    if not steps:
        raise ValueError("pipeline runtime requires emitted schedule steps")
    clocks = [step.clock for step in steps]
    if clocks != list(range(len(steps))):
        raise ValueError("pipeline clocks must be unique, ordered, and contiguous")
    for step in steps:
        if step.phase not in {"forward", "backward"}:
            raise ValueError(f"invalid pipeline phase {step.phase!r}")
        if step.region not in {"warmup", "steady", "cooldown"}:
            raise ValueError(f"invalid pipeline region {step.region!r}")
        if min(step.micro_batch, step.stage) < 0:
            raise ValueError("pipeline micro-batch and stage must be non-negative")
    return steps


def execute_pipeline_steps(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_stage: Callable[[EmittedPipelineStep], Any],
    run_collective: Callable[[EmittedPipelineStep], Any] | None = None,
    collective_after: Callable[[EmittedPipelineStep], bool] | None = None,
    transport_workers: int = 1,
    collective_runtime: OptimizerShardTransport | None = None,
) -> PipelineRuntimeResult:
    """Consume an emitted compute order and overlap selected collectives.

    A collective defaults to every backward step.  The callback receives the
    exact compiler step, so runtime launch descriptors can select the stage,
    micro-batch, and transport group without reconstructing schedule semantics.
    """
    if transport_workers < 1:
        raise ValueError("transport_workers must be positive")
    steps = parse_pipeline_steps(rows)
    if collective_runtime is not None and run_collective is not None:
        raise ValueError("provide run_collective or collective_runtime, not both")
    if collective_runtime is not None:
        run_collective = collective_runtime.run
        should_launch = lambda step: bool(step.collectives)
    else:
        should_launch = collective_after or (lambda step: step.phase == "backward")
    futures: list[Future[Any]] = []
    peak = 0
    with ThreadPoolExecutor(
        max_workers=transport_workers,
        thread_name_prefix="tessera-collective",
    ) as transport:
        for step in steps:
            run_stage(step)
            if run_collective is not None and should_launch(step):
                futures.append(transport.submit(run_collective, step))
                peak = max(
                    peak, sum(1 for future in futures if not future.done())
                )
        for future in futures:
            future.result()
    return PipelineRuntimeResult(
        completed_steps=steps,
        collective_count=len(futures),
        peak_inflight_collectives=peak,
    )


__all__ = [
    "CollectiveDescriptor",
    "EmittedPipelineStep",
    "OptimizerShardTransport",
    "PipelineRuntimeResult",
    "execute_pipeline_steps",
    "parse_pipeline_steps",
]
