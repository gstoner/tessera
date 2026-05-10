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
]
