"""
ShardSpec and MeshSpec — the glue between logical domains and physical mesh axes.

ShardSpec records HOW a distributed array is partitioned across a mesh.
MeshSpec records the logical mesh (axis names → sizes).

These lower into `schedule.mesh.define` and the distribution attributes on
`tessera.optimizer.shard` ops in Schedule IR.

Design:
  ShardSpec is intentionally minimal. It records:
    - which logical dimensions are partitioned  (partition tuple of dim indices)
    - which mesh axes they are partitioned over (mesh_axes tuple of axis names)

  It does NOT record the actual per-rank slice — that is computed lazily by
  DistributedArray.parts(axis) when the mesh is bound at runtime.

IR correspondence:
  ShardSpec(partition=(0,), mesh_axes=("dp",))
    →  {tessera.shard = {axis = "dp", partition = [0]}}  on the tensor value
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional


@dataclass(frozen=True)
class MeshSpec:
    """
    A logical mesh: maps axis names to their sizes.

    Example:
        mesh = MeshSpec({"dp": 4, "tp": 8, "pp": 2})
        assert mesh.total_ranks == 64
        assert mesh.axes == ("dp", "tp", "pp")
    """
    axes: Dict[str, int]   # axis_name → size

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("MeshSpec must have at least one axis")
        for name, size in self.axes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Axis names must be non-empty strings, got {name!r}")
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"Axis size must be a positive int, got {size!r}")

    @property
    def total_ranks(self) -> int:
        result = 1
        for s in self.axes.values():
            result *= s
        return result

    def axis_size(self, axis: str) -> int:
        if axis not in self.axes:
            raise KeyError(f"Unknown mesh axis {axis!r}. Known axes: {list(self.axes)}")
        return self.axes[axis]

    def to_ir_attr(self) -> str:
        """Emit as a mesh.define attribute string."""
        dims = list(self.axes.values())
        names = list(self.axes.keys())
        return f"dims = {dims}, axis_names = {names}"

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.axes.items())
        return f"MeshSpec({{{parts}}})"


@dataclass(frozen=True)
class ShardSpec:
    """
    Describes how a tensor is partitioned across mesh axes.

    Attributes:
        partition   : tuple of logical dimension indices that are partitioned.
                      len(partition) must equal len(mesh_axes).
        mesh_axes   : tuple of mesh axis names corresponding to each partitioned dim.
        replicated  : if True, tensor is fully replicated (no partition).

    Examples:
        # Column-parallel weight: partition dim-1 over tp axis
        ShardSpec(partition=(1,), mesh_axes=("tp",))

        # Data-parallel activation: partition dim-0 (batch) over dp axis
        ShardSpec(partition=(0,), mesh_axes=("dp",))

        # 2D sharding: batch over dp, hidden over tp
        ShardSpec(partition=(0, 1), mesh_axes=("dp", "tp"))

        # Replicated (e.g., bias terms)
        ShardSpec.replicate()
    """
    partition: Tuple[int, ...]
    mesh_axes: Tuple[str, ...]
    replicated: bool = False
    cyclic: bool = False       # True → round-robin (Cyclic dist), False → contiguous (Block dist)

    def __post_init__(self) -> None:
        if self.replicated:
            return
        if len(self.partition) != len(self.mesh_axes):
            raise ValueError(
                f"partition has {len(self.partition)} dims but "
                f"mesh_axes has {len(self.mesh_axes)} entries"
            )
        if len(set(self.partition)) != len(self.partition):
            raise ValueError(f"Duplicate dimension indices in partition: {self.partition}")
        if len(set(self.mesh_axes)) != len(self.mesh_axes):
            raise ValueError(f"Duplicate mesh axes in shard spec: {self.mesh_axes}")

    @classmethod
    def replicate(cls) -> "ShardSpec":
        """Create a fully-replicated shard spec (no partition)."""
        return cls(partition=(), mesh_axes=(), replicated=True)

    @classmethod
    def cyclic_shard(cls, partition: Tuple[int, ...], mesh_axes: Tuple[str, ...]) -> "ShardSpec":
        """Create a cyclic (round-robin) shard spec."""
        return cls(partition=partition, mesh_axes=mesh_axes, cyclic=True)

    def shard_size(self, logical_dim: int, full_size: int, mesh: MeshSpec) -> int:
        """
        Return the per-rank size of a given logical dimension.
        If the dimension is not partitioned, returns full_size.
        """
        if self.replicated or logical_dim not in self.partition:
            return full_size
        idx = self.partition.index(logical_dim)
        axis = self.mesh_axes[idx]
        axis_size = mesh.axis_size(axis)
        if full_size % axis_size != 0:
            raise ValueError(
                f"Dimension {logical_dim} size {full_size} is not evenly divisible "
                f"by mesh axis {axis!r} size {axis_size}"
            )
        return full_size // axis_size

    def to_ir_attr(self) -> str:
        """Emit as tessera.shard attribute string for Graph/Schedule IR."""
        if self.replicated:
            return '{tessera.shard = "replicated"}'
        axes_str = ", ".join(f'"{a}"' for a in self.mesh_axes)
        dims_str = ", ".join(str(d) for d in self.partition)
        kind = "cyclic" if self.cyclic else "block"
        return (f'{{tessera.shard = {{kind = "{kind}", '
                f'axes = [{axes_str}], dims = [{dims_str}]}}}}')

    def __repr__(self) -> str:
        if self.replicated:
            return "ShardSpec(replicated)"
        kind = "cyclic" if self.cyclic else "block"
        pairs = ", ".join(f"dim{d}→{ax}" for d, ax in zip(self.partition, self.mesh_axes))
        return f"ShardSpec({kind}, {pairs})"
