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
    mesh_axis: str = "default"
    world_size: int | None = None
    scatter_axis: int | None = None
    gather_axis: int | None = None
    source_tensor: str | None = None
    result_tensor: str | None = None
    dtype: str | None = None
    chunk_bytes: int | None = None
    peer_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"all_reduce", "reduce_scatter", "all_gather", "all_to_all", "collective_permute"}:
            raise ValueError(f"unsupported collective kind {self.kind!r}")
        if not self.tensor:
            raise ValueError("collective tensor identity must be non-empty")
        if self.axis < 0:
            raise ValueError("collective axis must be non-negative")
        if self.op not in {"sum", "max", "min", "prod"}:
            raise ValueError(f"unsupported collective reduction {self.op!r}")
        if self.optimizer_shard and self.kind not in {"reduce_scatter", "all_gather"}:
            raise ValueError("OptimizerShard transport must reduce-scatter or all-gather")
        if not self.mesh_axis:
            raise ValueError("collective mesh axis must be non-empty")
        if self.world_size is not None and self.world_size <= 0:
            raise ValueError("collective world_size must be positive when present")
        if self.scatter_axis is not None and self.scatter_axis < 0:
            raise ValueError("collective scatter_axis must be non-negative")
        if self.gather_axis is not None and self.gather_axis < 0:
            raise ValueError("collective gather_axis must be non-negative")
        if self.source_tensor is not None and not self.source_tensor:
            raise ValueError("collective source tensor identity must be non-empty")
        if self.result_tensor is not None and not self.result_tensor:
            raise ValueError("collective result tensor identity must be non-empty")
        if self.dtype is not None and not self.dtype:
            raise ValueError("collective dtype must be non-empty when present")
        if self.chunk_bytes is not None and self.chunk_bytes <= 0:
            raise ValueError("collective chunk_bytes must be positive when present")
        if self.kind == "collective_permute":
            sources = tuple(source for source, _ in self.peer_pairs)
            targets = tuple(target for _, target in self.peer_pairs)
            if not self.peer_pairs:
                raise ValueError("collective_permute requires a non-empty peer map")
            if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
                raise ValueError("collective_permute peer sources and targets must be unique")
            if any(peer < 0 for pair in self.peer_pairs for peer in pair):
                raise ValueError("collective_permute peers must be non-negative")
            if self.world_size is not None and any(
                peer >= self.world_size for pair in self.peer_pairs for peer in pair
            ):
                raise ValueError("collective_permute peer is outside world_size")
        elif self.peer_pairs:
            raise ValueError("peer_pairs are valid only for collective_permute")

    @classmethod
    def from_target_record(cls, row: Mapping[str, Any]) -> "CollectiveDescriptor":
        """Decode one ``tessera_collective`` Target-IR package record."""
        op_name = str(row.get("op", ""))
        prefix = "tessera_collective."
        if not op_name.startswith(prefix):
            raise ValueError("target collective record requires tessera_collective.* op")
        kind = op_name.removeprefix(prefix)
        if kind == "await":
            raise ValueError("await is a dependency edge, not a transport dispatch")
        axis = int(row.get("tensor_axis", 0))
        world_size = row.get("world_size")
        reduction = str(row.get("reduction", "sum"))
        # The runtime adapter's `op` field is irrelevant for non-reducing
        # collectives but remains a validated enum for one uniform descriptor.
        normalize = reduction == "mean"
        if reduction in {"none", "mean"}:
            reduction = "sum"
        source_peers = tuple(int(peer) for peer in row.get("source_peers", ()))
        target_peers = tuple(int(peer) for peer in row.get("target_peers", ()))
        if len(source_peers) != len(target_peers):
            raise ValueError("collective_permute source/target peer maps differ in length")
        return cls(
            kind=kind,
            tensor=str(row["tensor"]),
            axis=axis,
            op=reduction,
            mesh_axis=str(row.get("mesh_axis", "default")),
            world_size=None if world_size is None else int(world_size),
            scatter_axis=int(row.get("scatter_axis", axis)),
            gather_axis=int(row.get("gather_axis", axis)),
            source_tensor=str(row.get("input", row["tensor"])),
            result_tensor=str(row.get("output", row["tensor"])),
            normalize=normalize,
            dtype=None if row.get("dtype") is None else str(row["dtype"]),
            chunk_bytes=(None if row.get("chunk_bytes") is None else int(row["chunk_bytes"])),
            peer_pairs=tuple(zip(source_peers, target_peers)),
        )

    @classmethod
    def from_sharding_primitive(
        cls,
        name: str,
        *,
        tensor: str,
        mesh_axis: str,
        tensor_axis: int = 0,
        world_size: int | None = None,
        result: str | None = None,
    ) -> "CollectiveDescriptor":
        """Resolve a public S6 alias through the canonical Target contract."""
        from .distributed_contracts import collective_record_for_primitive

        return cls.from_target_record(
            collective_record_for_primitive(
                name,
                tensor=tensor,
                mesh_axis=mesh_axis,
                tensor_axis=tensor_axis,
                world_size=world_size,
                result=result,
            )
        )


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
            self.execute(descriptor)

    def execute(self, descriptor: CollectiveDescriptor) -> None:
        """Execute one compiler-owned Target-IR collective descriptor."""
        with self._lock:
            source_tensor = descriptor.source_tensor or descriptor.tensor
            result_tensor = descriptor.result_tensor or descriptor.tensor
            if source_tensor not in self._values:
                raise KeyError(f"no runtime value for collective tensor {source_tensor!r}")
            adapter_world_size = int(self.adapter.world_size)
            mesh_axes = getattr(self.adapter, "mesh_axes", None)
            if descriptor.mesh_axis != "default" and isinstance(mesh_axes, Mapping):
                if descriptor.mesh_axis not in mesh_axes:
                    raise RuntimeError(f"{descriptor.tensor} references unknown mesh axis {descriptor.mesh_axis!r}")
                mesh_world_size = int(mesh_axes[descriptor.mesh_axis])
                if mesh_world_size != adapter_world_size:
                    raise RuntimeError(
                        f"runtime adapter spans {adapter_world_size} ranks, but mesh "
                        f"axis {descriptor.mesh_axis!r} spans {mesh_world_size}; "
                        "subgroup transport is not implemented"
                    )
                if descriptor.world_size is not None and descriptor.world_size != mesh_world_size:
                    raise RuntimeError(
                        f"{descriptor.tensor} targets world_size={descriptor.world_size}, "
                        f"but mesh axis {descriptor.mesh_axis!r} owns {mesh_world_size} ranks"
                    )
            if descriptor.world_size is not None and descriptor.world_size != adapter_world_size:
                raise RuntimeError(
                    f"{descriptor.tensor} targets world_size={descriptor.world_size}, "
                    f"but runtime adapter owns {adapter_world_size} ranks"
                )
            values = self._values[source_tensor]
            ownership = self._ownership[source_tensor]
            if descriptor.kind == "reduce_scatter":
                if descriptor.optimizer_shard and ownership != "replicated":
                    raise RuntimeError(f"{descriptor.tensor} must be replicated before OptimizerShard reduce-scatter")
                outputs = self.adapter.reduce_scatter(values, axis=descriptor.axis, op=descriptor.op)
                result_ownership = "rank_local"
            elif descriptor.kind == "all_gather":
                if descriptor.optimizer_shard and ownership != "rank_local":
                    raise RuntimeError(f"{descriptor.tensor} must be rank-local before OptimizerShard all-gather")
                outputs = self.adapter.all_gather(values, axis=descriptor.axis)
                result_ownership = "replicated"
            elif descriptor.kind == "all_reduce":
                outputs = self.adapter.all_reduce(values, op=descriptor.op)
                result_ownership = "replicated"
            elif descriptor.kind == "all_to_all":
                outputs = self.adapter.all_to_all(
                    values,
                    scatter_axis=(descriptor.axis if descriptor.scatter_axis is None else descriptor.scatter_axis),
                    gather_axis=(descriptor.axis if descriptor.gather_axis is None else descriptor.gather_axis),
                )
                result_ownership = "rank_local"
            else:
                outputs = self.adapter.collective_permute(values, pairs=descriptor.peer_pairs)
                result_ownership = "rank_local"
            if descriptor.normalize:
                if descriptor.kind not in {"all_reduce", "reduce_scatter"}:
                    raise RuntimeError("normalization is valid only for reducing collectives")
                outputs = [value / float(self.adapter.world_size) for value in outputs]
            self._values[result_tensor] = list(outputs)
            self._ownership[result_tensor] = result_ownership


# The runtime is generic; retain the established name for API compatibility.
CollectiveTransportRuntime = OptimizerShardTransport


@dataclass(frozen=True)
class OneSidedDescriptor:
    """One registered, rank-local RMA action in Target IR."""

    kind: str
    window: str = ""
    buffer: str = ""
    source: str = ""
    bytes: int = 0
    count: int = 0
    dtype: str = "f32"
    peer: int = 0
    peer_window_offset: int = 0
    signal_index: int = 0
    context: int = 0
    operation_count: int = 1
    strict_ordering: bool = True

    def __post_init__(self) -> None:
        kinds = {
            "window.register",
            "window.deregister",
            "put_signal",
            "signal",
            "wait_signal",
        }
        if self.kind not in kinds:
            raise ValueError(f"unsupported one-sided operation {self.kind!r}")
        if self.kind in {"window.register", "window.deregister", "put_signal"}:
            if not self.window:
                raise ValueError(f"{self.kind} requires a window identity")
        if self.kind == "window.register":
            if not self.buffer or self.bytes <= 0:
                raise ValueError("window.register requires a buffer and positive byte extent")
        if self.kind == "put_signal":
            if not self.source or self.count <= 0:
                raise ValueError("put_signal requires a source and positive element count")
            if self.dtype not in {"f32", "f16", "bf16", "i8"}:
                raise ValueError("put_signal dtype must be f32|f16|bf16|i8")
            if self.peer_window_offset < 0:
                raise ValueError("put_signal window offset must be non-negative")
        if self.kind in {"put_signal", "signal", "wait_signal"}:
            if self.peer < 0 or self.signal_index < 0 or self.context < 0:
                raise ValueError("one-sided peer, signal index, and context must be non-negative")
        if self.kind == "wait_signal" and self.operation_count <= 0:
            raise ValueError("wait_signal operation count must be positive")

    @classmethod
    def from_target_record(cls, row: Mapping[str, Any]) -> "OneSidedDescriptor":
        op_name = str(row.get("op", ""))
        prefix = "tessera_collective."
        if not op_name.startswith(prefix):
            raise ValueError("one-sided target record requires tessera_collective.* op")
        return cls(
            kind=op_name.removeprefix(prefix),
            window=str(row.get("window", "")),
            buffer=str(row.get("buffer", "")),
            source=str(row.get("source", "")),
            bytes=int(row.get("bytes", 0)),
            count=int(row.get("count", 0)),
            dtype=str(row.get("dtype", "f32")),
            peer=int(row.get("peer", 0)),
            peer_window_offset=int(row.get("peer_window_offset", 0)),
            signal_index=int(row.get("signal_index", 0)),
            context=int(row.get("rma_context", row.get("context", 0))),
            operation_count=int(row.get("operation_count", 1)),
            strict_ordering=bool(row.get("strict_ordering", True)),
        )


class OneSidedTransportRuntime:
    """Execute ordered RMA records through one rank-local native adapter."""

    def __init__(self, adapter: Any, buffers: Mapping[str, Any]) -> None:
        self.adapter = adapter
        self.buffers = dict(buffers)
        self.windows: dict[str, Any] = {}

    def execute(self, descriptor: OneSidedDescriptor) -> None:
        if descriptor.kind == "window.register":
            if descriptor.window in self.windows:
                raise RuntimeError(f"one-sided window {descriptor.window!r} is already registered")
            if descriptor.buffer not in self.buffers:
                raise KeyError(f"no runtime buffer named {descriptor.buffer!r}")
            register = getattr(self.adapter, "register_symmetric_window", None)
            if not callable(register):
                raise RuntimeError("one-sided adapter does not implement window registration")
            self.windows[descriptor.window] = register(
                self.buffers[descriptor.buffer],
                descriptor.bytes,
                strict_ordering=descriptor.strict_ordering,
            )
            return
        if descriptor.kind == "window.deregister":
            if descriptor.window not in self.windows:
                raise RuntimeError(f"one-sided window {descriptor.window!r} is not registered")
            window = self.windows.pop(descriptor.window)
            close = getattr(window, "close", None)
            if callable(close):
                close()
                return
            deregister = getattr(self.adapter, "deregister_symmetric_window", None)
            if not callable(deregister):
                raise RuntimeError("one-sided adapter cannot deregister its window handle")
            deregister(window)
            return
        if descriptor.kind == "put_signal":
            if descriptor.window not in self.windows:
                raise RuntimeError(f"one-sided window {descriptor.window!r} is not registered")
            if descriptor.source not in self.buffers:
                raise KeyError(f"no runtime buffer named {descriptor.source!r}")
            put = getattr(self.adapter, "put_signal", None)
            if not callable(put):
                raise RuntimeError("one-sided adapter does not implement put_signal")
            put(
                self.buffers[descriptor.source],
                descriptor.count,
                descriptor.dtype,
                descriptor.peer,
                self.windows[descriptor.window],
                descriptor.peer_window_offset,
                signal_index=descriptor.signal_index,
                context=descriptor.context,
            )
            return
        if descriptor.kind == "signal":
            signal = getattr(self.adapter, "signal", None)
            if not callable(signal):
                raise RuntimeError("one-sided adapter does not implement signal")
            signal(
                descriptor.peer,
                signal_index=descriptor.signal_index,
                context=descriptor.context,
            )
            return
        wait = getattr(self.adapter, "wait_signal", None)
        if not callable(wait):
            raise RuntimeError("one-sided adapter does not implement wait_signal")
        wait(
            descriptor.peer,
            operation_count=descriptor.operation_count,
            signal_index=descriptor.signal_index,
            context=descriptor.context,
        )

    def finish(self) -> None:
        if self.windows:
            names = ", ".join(sorted(self.windows))
            raise RuntimeError(f"one-sided artifact leaked registered windows: {names}")


def execute_one_sided_target_records(
    records: Iterable[Mapping[str, Any]],
    *,
    adapter: Any,
    buffers: Mapping[str, Any],
) -> OneSidedTransportRuntime:
    runtime = OneSidedTransportRuntime(adapter, buffers)
    for row in records:
        runtime.execute(OneSidedDescriptor.from_target_record(row))
    runtime.finish()
    return runtime


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
            mesh_axis=str(item.get("mesh_axis", "default")),
            world_size=(None if item.get("world_size") is None else int(item["world_size"])),
            scatter_axis=(None if item.get("scatter_axis") is None else int(item["scatter_axis"])),
            gather_axis=(None if item.get("gather_axis") is None else int(item["gather_axis"])),
            source_tensor=(None if item.get("input") is None else str(item["input"])),
            result_tensor=(None if item.get("output") is None else str(item["output"])),
            dtype=None if item.get("dtype") is None else str(item["dtype"]),
            chunk_bytes=(None if item.get("chunk_bytes") is None else int(item["chunk_bytes"])),
        )
        for item in raw
    )


def execute_target_collectives(
    records: Iterable[Mapping[str, Any]],
    *,
    adapter: Any,
    tensors: Mapping[str, Iterable[Any]],
) -> CollectiveTransportRuntime:
    """Execute ordered portable Target-IR records through one runtime adapter.

    Packaging resolves SSA buffers to stable ``tensor`` identities before
    calling this boundary. The same path drives the deterministic in-process
    adapter and native NCCL/RCCL adapters.
    """
    runtime = CollectiveTransportRuntime(adapter, tensors)
    for row in records:
        runtime.execute(CollectiveDescriptor.from_target_record(row))
    return runtime


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
                peak = max(peak, sum(1 for future in futures if not future.done()))
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
