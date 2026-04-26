"""
DistributedArray — a tensor with an attached ShardSpec.

In Phase 1, physical storage is an eagerly-evaluated numpy array on CPU.
The ShardSpec is metadata only — it tells downstream passes (the
DistributionLoweringPass in Phase 2) how to slice this tensor across ranks.

The .parts(axis) method returns the per-rank slices for a given mesh axis,
enabling index_launch to fan out kernels across shards.

Reference: docs/programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md §4.5
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple, Any
import numpy as np

from .domain import Domain, Rect, Distribution, Block, Replicated
from .shard import ShardSpec, MeshSpec

if TYPE_CHECKING:
    pass


# Supported dtype strings → numpy dtypes
_DTYPE_MAP = {
    "bf16":   np.float32,   # numpy has no bf16; store as f32 in Phase 1
    "fp16":   np.float16,
    "fp32":   np.float32,
    "fp64":   np.float64,
    "int8":   np.int8,
    "uint8":  np.uint8,
    "int32":  np.int32,
    "int64":  np.int64,
    "bool":   np.bool_,
}

_VALID_DTYPES = set(_DTYPE_MAP)


class DistributedArray:
    """
    A logically distributed tensor with shape, dtype, and ShardSpec metadata.

    Phase 1: backed by a numpy array (CPU only). The ShardSpec is carried as
    metadata and used by the compiler — it does not affect eager computation.

    Phase 3: backed by a CudaBuffer or HIP buffer per-rank.

    Attributes:
        shape      : logical (global) shape of the tensor
        dtype      : string dtype ("bf16", "fp16", "fp32", ...)
        shard_spec : ShardSpec describing the partition strategy
        _data      : numpy backing array (Phase 1 CPU only)
    """

    def __init__(
        self,
        data: np.ndarray,
        dtype: str,
        shard_spec: ShardSpec,
        logical_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Valid dtypes: {sorted(_VALID_DTYPES)}"
            )
        self._data = data
        self.dtype = dtype
        self.shard_spec = shard_spec
        self.shape: Tuple[int, ...] = logical_shape if logical_shape is not None \
            else tuple(data.shape)

    @classmethod
    def from_domain(
        cls,
        domain: Domain,
        dtype: str,
        distribution: Distribution,
        fill: str = "zeros",
        mesh: Optional[MeshSpec] = None,
    ) -> "DistributedArray":
        """
        Create a DistributedArray over the given domain with the given distribution.

        Args:
            domain       : the logical iteration space (e.g. Rect((B, S, D)))
            dtype        : storage dtype string ("bf16", "fp16", "fp32", ...)
            distribution : how to partition the domain (Block, Cyclic, Replicated)
            fill         : initial data — "zeros", "ones", "randn", or "empty"
            mesh         : optional MeshSpec; if provided, validates shard sizes

        Returns:
            DistributedArray with attached ShardSpec

        Example:
            D    = tessera.domain.Rect((4, 128, 256))
            dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
            X    = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
            assert X.shape == (4, 128, 256)
            assert X.shard_spec.mesh_axes == ("dp", "tp")
        """
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Valid dtypes: {sorted(_VALID_DTYPES)}"
            )
        shard_spec = distribution.make_shard_spec(domain)
        np_dtype = _DTYPE_MAP[dtype]

        # Validate shard sizes if mesh is provided
        if mesh is not None and not shard_spec.replicated:
            for i, (dim_idx, axis) in enumerate(
                zip(shard_spec.partition, shard_spec.mesh_axes)
            ):
                full = domain.shape[dim_idx]
                _ = shard_spec.shard_size(dim_idx, full, mesh)  # raises if not divisible

        shape = domain.shape
        if fill == "zeros":
            data = np.zeros(shape, dtype=np_dtype)
        elif fill == "ones":
            data = np.ones(shape, dtype=np_dtype)
        elif fill == "randn":
            data = np.random.randn(*shape).astype(np_dtype)
        elif fill == "empty":
            data = np.empty(shape, dtype=np_dtype)
        else:
            raise ValueError(f"Unknown fill strategy {fill!r}. Use 'zeros', 'ones', 'randn', 'empty'.")

        return cls(data=data, dtype=dtype, shard_spec=shard_spec, logical_shape=shape)

    def parts(self, axis: str) -> List["DistributedArray"]:
        """
        Return per-rank slices along the given mesh axis.

        In Phase 1 (CPU, single-process), this returns a list of sub-arrays
        split along the partitioned dimension for that axis.

        In Phase 3 (multi-GPU), this will return per-device buffer views.

        Args:
            axis: mesh axis name (e.g. "tp", "dp")

        Returns:
            list of DistributedArray, one per rank on that axis

        Example:
            X = tessera.array.from_domain(Rect((8, 256)), "fp32", Block(("dp",)))
            shards = X.parts("dp")  # 4 shards if dp=4
            assert shards[0].shape == (2, 256)
        """
        if self.shard_spec.replicated:
            # Replicated → all ranks get the same full array
            return [self]

        if axis not in self.shard_spec.mesh_axes:
            raise ValueError(
                f"Array is not partitioned over axis {axis!r}. "
                f"Partitioned axes: {self.shard_spec.mesh_axes}"
            )

        axis_idx_in_spec = self.shard_spec.mesh_axes.index(axis)
        dim_idx = self.shard_spec.partition[axis_idx_in_spec]

        # Phase 1: split the numpy array along that dimension
        dim_size = self.shape[dim_idx]
        # Infer number of parts from the annotation (default 1 per rank in mock)
        # In Phase 1, we split evenly; Phase 3 will use mesh.axis_size(axis)
        num_parts = getattr(self, "_mesh_size_cache", {}).get(axis, 1)

        if dim_size % num_parts != 0:
            raise ValueError(
                f"Cannot evenly split dimension {dim_idx} of size {dim_size} "
                f"into {num_parts} parts for axis {axis!r}"
            )

        subs = np.array_split(self._data, num_parts, axis=dim_idx)
        result = []
        for sub in subs:
            result.append(DistributedArray(
                data=sub,
                dtype=self.dtype,
                shard_spec=self.shard_spec,
                logical_shape=tuple(sub.shape),
            ))
        return result

    def _bind_mesh(self, mesh: MeshSpec) -> "DistributedArray":
        """
        Attach mesh size information so parts() can compute correct splits.
        Called by index_launch before fanning out.
        """
        self._mesh_size_cache = {
            axis: mesh.axis_size(axis)
            for axis in self.shard_spec.mesh_axes
        }
        return self

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    def numpy(self) -> np.ndarray:
        """Return the backing numpy array (Phase 1 only)."""
        return self._data

    def __repr__(self) -> str:
        return (
            f"DistributedArray(shape={self.shape}, dtype={self.dtype!r}, "
            f"shard={self.shard_spec})"
        )
