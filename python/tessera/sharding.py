"""S6 sharding and collective reference primitives.

This module gives Tessera a standalone SPMD vocabulary independent of JAX:
named meshes, partition specs, named sharding, a CPU-reference ``shard_map``,
and primitive collectives. Backend-specific NCCL/RCCL lowering remains a later
contract axis; these objects are the compiler-visible semantic surface.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import control


@dataclass(frozen=True)
class NamedMesh:
    axis_names: tuple[str, ...]
    shape: tuple[int, ...]
    devices: tuple[Any, ...] = ()

    def __init__(
        self,
        axis_names: Sequence[str],
        shape: Sequence[int] | Mapping[str, int],
        devices: Sequence[Any] = (),
    ) -> None:
        names = tuple(str(name) for name in axis_names)
        if isinstance(shape, Mapping):
            dims = tuple(int(shape[name]) for name in names)
        else:
            dims = tuple(int(dim) for dim in shape)
        if len(names) != len(dims):
            raise ValueError("NamedMesh axis_names and shape must have equal length")
        if any(dim <= 0 for dim in dims):
            raise ValueError("NamedMesh dimensions must be positive")
        object.__setattr__(self, "axis_names", names)
        object.__setattr__(self, "shape", dims)
        object.__setattr__(self, "devices", tuple(devices))

    @property
    def size(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out

    def axis_size(self, axis_name: str) -> int:
        return self.shape[self.axis_names.index(axis_name)]


@dataclass(frozen=True)
class PartitionSpec:
    axes: tuple[str | None, ...]

    def __init__(self, *axes: str | None):
        object.__setattr__(self, "axes", tuple(axes))


@dataclass(frozen=True)
class NamedSharding:
    mesh: NamedMesh
    spec: PartitionSpec
    memory_kind: str = "device"


def named_sharding(
    mesh: NamedMesh,
    spec: PartitionSpec | Sequence[str | None],
    *,
    memory_kind: str = "device",
) -> NamedSharding:
    if not isinstance(spec, PartitionSpec):
        spec = PartitionSpec(*spec)
    return NamedSharding(mesh=mesh, spec=spec, memory_kind=str(memory_kind))


def partition_spec(*axes: str | None) -> PartitionSpec:
    return PartitionSpec(*axes)


def _sharded_axis(spec: PartitionSpec, mesh: NamedMesh) -> tuple[int, str] | None:
    for array_axis, mesh_axis in enumerate(spec.axes):
        if mesh_axis is not None:
            if mesh_axis not in mesh.axis_names:
                raise ValueError(f"unknown mesh axis in PartitionSpec: {mesh_axis!r}")
            return array_axis, mesh_axis
    return None


def shard_map(
    fn: Callable,
    *,
    mesh: NamedMesh,
    in_specs: PartitionSpec | Sequence[PartitionSpec],
    out_specs: PartitionSpec | None = None,
    axis_name: str | None = None,
) -> Callable:
    """Reference ``shard_map`` over one partitioned input axis.

    The v1 CPU path maps over the first mesh axis named by the first input
    spec and concatenates shard outputs along the corresponding output axis.
    """
    specs = (in_specs,) if isinstance(in_specs, PartitionSpec) else tuple(in_specs)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if len(specs) != len(args):
            raise ValueError("shard_map in_specs length must match positional args")
        active = None
        for spec in specs:
            active = _sharded_axis(spec, mesh)
            if active is not None:
                break
        if active is None:
            return fn(*args, **kwargs)
        array_axis, mesh_axis = active
        size = mesh.axis_size(mesh_axis)
        name = axis_name or mesh_axis
        outputs = []
        for i in range(size):
            shard_args = []
            for arg, spec in zip(args, specs):
                local = _sharded_axis(spec, mesh)
                if local is None:
                    shard_args.append(arg)
                    continue
                ax, local_mesh_axis = local
                if local_mesh_axis != mesh_axis:
                    raise NotImplementedError("reference shard_map supports one mesh axis")
                shards = np.array_split(np.asarray(arg), size, axis=ax)
                shard_args.append(shards[i])
            token = control._push_axis(name, i, size)
            try:
                outputs.append(fn(*shard_args, **kwargs))
            finally:
                control._pop_axis(token)
        if out_specs is None:
            return outputs
        out_axis = _sharded_axis(out_specs, mesh)
        if out_axis is None:
            return outputs[0]
        return np.concatenate([np.asarray(out) for out in outputs], axis=out_axis[0])

    return wrapped


def _stack(values: Any) -> np.ndarray:
    return np.asarray(values)


def psum(values: Any, axis_name: str | None = None) -> np.ndarray:
    arr = _stack(values)
    return np.sum(arr, axis=0)


def pmean(values: Any, axis_name: str | None = None) -> np.ndarray:
    arr = _stack(values)
    return np.mean(arr, axis=0)


def pmax(values: Any, axis_name: str | None = None) -> np.ndarray:
    return np.max(_stack(values), axis=0)


def pmin(values: Any, axis_name: str | None = None) -> np.ndarray:
    return np.min(_stack(values), axis=0)


def collective_permute(values: Any, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    arr = _stack(values)
    out = np.empty_like(arr)
    for src, dst in pairs:
        out[int(dst)] = arr[int(src)]
    return out


def broadcast_to_axis(value: Any, *, axis_size: int, axis: int = 0) -> np.ndarray:
    return np.stack([np.asarray(value) for _ in range(int(axis_size))], axis=axis)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint D — Memory-bank sharding (2026-05-11).
#
# Tensor data shards are described by `PartitionSpec` + `NamedSharding`
# (above).  Memory-bank state (Titans/Atlas-style learned memory tables
# under `tessera.memory.memory_read/write/evict`) needs a richer
# partition vocabulary:
#
#   - The bank IS the state; it's not a tensor input/output.
#   - Partitioning is content-addressed (by key-hash) rather than
#     positional (by index), so a row's owning rank depends on its key,
#     not its position in the table.
#   - Eviction policies (`lru`/`fifo`/`score`) are sharded — each rank
#     independently evicts its own bucket.
#
# `MemoryShardSpec` captures these.  `MemoryMode` enumerates the supported
# partitioning strategies.
# ─────────────────────────────────────────────────────────────────────────────


class MemoryMode:
    """Partition strategies for memory banks.

    Values are strings so they can travel through IR metadata unchanged.
    """

    BLOCK = "block"            # contiguous slices along the entries axis
    REPLICATED = "replicated"  # every rank holds the full bank
    KEY_HASH = "key_hash"      # row ownership = hash(key) mod num_shards
    BUCKET = "bucket"          # explicit bucket function (caller-supplied)


_MEMORY_MODES: frozenset[str] = frozenset({
    MemoryMode.BLOCK,
    MemoryMode.REPLICATED,
    MemoryMode.KEY_HASH,
    MemoryMode.BUCKET,
})


@dataclass(frozen=True)
class MemoryShardSpec:
    """How a memory bank is partitioned across a mesh axis.

    Attributes
    ----------
    mesh_axis : str
        Mesh axis to partition along.  Validated against the
        ``NamedMesh.axis_names`` at attach time.
    mode : str
        One of ``MemoryMode.*``.  ``KEY_HASH`` is the recommended default
        for content-addressed memory: a row's owning shard is
        ``hash(key) mod num_shards``, so reads with the same key always
        find the row on the same shard.
    eviction : str
        Eviction policy applied independently per shard.  One of
        ``"lru"``, ``"fifo"``, ``"score"``, ``"oldest"``.  Default
        ``"score"`` matches the `tessera.memory.memory_evict` reference.
    persistence : str
        ``"persistent"`` (saved to checkpoints; default), ``"ephemeral"``
        (in-RAM only).  Matches the `STATE_COLLECTION_SPECS["memory_state"]`
        ``persistent=True`` flag.
    bucket_fn : str | None
        For ``MemoryMode.BUCKET``: name of the user-supplied bucket
        function registered via `register_memory_bucket_fn`.
        Required when ``mode == BUCKET``.
    """

    mesh_axis: str
    mode: str = MemoryMode.KEY_HASH
    eviction: str = "score"
    persistence: str = "persistent"
    bucket_fn: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MEMORY_MODES:
            raise ValueError(
                f"MemoryShardSpec.mode must be one of {sorted(_MEMORY_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.persistence not in {"persistent", "ephemeral"}:
            raise ValueError(
                f"MemoryShardSpec.persistence must be 'persistent' or 'ephemeral', "
                f"got {self.persistence!r}"
            )
        if self.eviction not in {"lru", "fifo", "score", "oldest"}:
            raise ValueError(
                f"MemoryShardSpec.eviction must be 'lru' | 'fifo' | 'score' | 'oldest', "
                f"got {self.eviction!r}"
            )
        if self.mode == MemoryMode.BUCKET and self.bucket_fn is None:
            raise ValueError(
                "MemoryShardSpec(mode='bucket') requires a `bucket_fn=` name"
            )
        if self.mode != MemoryMode.BUCKET and self.bucket_fn is not None:
            raise ValueError(
                f"bucket_fn is only meaningful when mode='bucket' (got mode={self.mode!r})"
            )

    def validate_against(self, mesh: NamedMesh) -> None:
        """Confirm `mesh_axis` exists on the given mesh."""
        if self.mesh_axis not in mesh.axis_names:
            raise ValueError(
                f"MemoryShardSpec.mesh_axis={self.mesh_axis!r} not in mesh "
                f"axes {mesh.axis_names}"
            )

    def shard_owner(self, key: np.ndarray, mesh: NamedMesh) -> int:
        """Compute the owning shard index for a single key (or batch).

        For ``KEY_HASH``: stable FNV-1a hash on the key bytes mod
        ``mesh.axis_size(mesh_axis)``.  For ``BUCKET``: dispatches to
        the registered bucket function.  For ``BLOCK`` / ``REPLICATED``:
        returns 0 (caller is responsible for slicing along the entries
        axis externally).
        """
        self.validate_against(mesh)
        n_shards = mesh.axis_size(self.mesh_axis)
        if self.mode == MemoryMode.REPLICATED:
            return 0
        if self.mode == MemoryMode.BLOCK:
            return 0
        if self.mode == MemoryMode.KEY_HASH:
            arr = np.ascontiguousarray(key)
            h = _fnv1a_64(arr.tobytes())
            return int(h % n_shards)
        if self.mode == MemoryMode.BUCKET:
            fn = _MEMORY_BUCKET_FUNCTIONS.get(self.bucket_fn or "")
            if fn is None:
                raise ValueError(
                    f"MemoryShardSpec.bucket_fn={self.bucket_fn!r} is not "
                    f"registered.  Use register_memory_bucket_fn first."
                )
            return int(fn(key, n_shards) % n_shards)
        raise ValueError(f"unexpected mode {self.mode!r}")


def _fnv1a_64(data: bytes) -> int:
    """Deterministic FNV-1a — same algorithm as
    `primitive_coverage.AutotunePass` cache-key for IR-side consistency."""
    h = 0xcbf29ce484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


_MEMORY_BUCKET_FUNCTIONS: dict[str, Callable[[np.ndarray, int], int]] = {}


def register_memory_bucket_fn(
    name: str,
    fn: Callable[[np.ndarray, int], int],
) -> None:
    """Register a user-supplied bucket function for ``MemoryMode.BUCKET``.

    The function takes ``(key_array, num_shards)`` and returns the owning
    shard index (will be reduced mod ``num_shards`` automatically).
    """
    _MEMORY_BUCKET_FUNCTIONS[name] = fn


def get_memory_bucket_fn(name: str) -> Callable[[np.ndarray, int], int] | None:
    return _MEMORY_BUCKET_FUNCTIONS.get(name)


__all__ = [
    "NamedMesh",
    "NamedSharding",
    "PartitionSpec",
    "broadcast_to_axis",
    "collective_permute",
    "named_sharding",
    "partition_spec",
    "pmax",
    "pmean",
    "pmin",
    "psum",
    "shard_map",
    # Sprint D — memory-bank sharding
    "MemoryMode",
    "MemoryShardSpec",
    "register_memory_bucket_fn",
    "get_memory_bucket_fn",
]
