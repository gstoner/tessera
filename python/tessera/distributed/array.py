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
from typing import TYPE_CHECKING, Any, Optional, List, Tuple
import numpy as np

from .domain import Domain, Distribution
from .shard import ShardSpec, MeshSpec

if TYPE_CHECKING:
    pass


# Supported dtype strings → numpy dtypes.
#
# Keys are **canonical Tessera dtype names**, normalized via
# ``tessera.dtype.canonicalize_dtype`` at every API entry point (Sprint A0,
# 2026-05-11).  Aliases like ``"f32"``/``"i8"`` are accepted as inputs but
# never stored.  ``uint8`` is in the canonical-numpy-backing set even though
# the tensor-attributes doc lists ``uint*`` as planned/gated — this is the
# one numpy-backing storage slot kept live for the legacy quantization
# helpers; new entries should go through ``canonicalize_dtype(...,
# allow_planned_gated=True)`` and declare ``dtype_status="planned_gated"``.
_DTYPE_MAP = {
    "bf16":      np.float32,   # numpy has no bf16; store as f32 in Phase 1
    "fp16":      np.float16,
    "fp32":      np.float32,
    "fp64":      np.float64,
    "fp8_e4m3":  np.float32,   # no native numpy storage; pinned to f32
    "fp8_e5m2":  np.float32,
    "fp6_e2m3":  np.float32,
    "fp6_e3m2":  np.float32,
    "fp4_e2m1":  np.float32,
    "nvfp4":     np.float32,
    "int8":      np.int8,
    "int16":     np.int16,
    "int32":     np.int32,
    "int64":     np.int64,
    "uint8":     np.uint8,     # legacy quantization helpers
    "bool":      np.bool_,
}

_VALID_DTYPES = set(_DTYPE_MAP)


def _normalize_dtype(dtype: str) -> str:
    """Canonicalize a user-supplied dtype string at the public API boundary.

    Accepts canonical names and registered aliases (``"f32"``/``"i8"``/
    ``"float32"``/etc.).  Returns the canonical spelling that downstream
    storage and IR metadata expect.  Raises ``ValueError`` (via
    ``TesseraDtypeError``) for unknown or compound spellings, and for TF32
    (which is a math_mode, not a storage dtype).
    """
    from tessera.dtype import canonicalize_dtype  # local import → no cycle

    canon = canonicalize_dtype(dtype, allow_planned_gated=True)
    if canon not in _DTYPE_MAP:
        raise ValueError(
            f"Tessera dtype {canon!r} (from {dtype!r}) has no numpy "
            f"storage backing in this DistributedArray runtime.  "
            f"Backed dtypes: {sorted(_VALID_DTYPES)}."
        )
    return canon


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
        dtype = _normalize_dtype(dtype)
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
        dtype = _normalize_dtype(dtype)
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
        data: Any
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

        dim_size = self.shape[dim_idx]
        num_parts = getattr(self, "_mesh_size_cache", {}).get(axis, 1)

        if self.shard_spec.cyclic:
            # ── Cyclic (round-robin) partition ───────────────────────────────
            # Rank k receives rows: k, k + num_parts, k + 2*num_parts, …
            # We pad the dimension to the next multiple of num_parts if needed.
            result = []
            for rank in range(num_parts):
                indices = list(range(rank, dim_size, num_parts))
                if not indices:
                    # Edge case: more ranks than elements — empty shard
                    sub_shape = list(self._data.shape)
                    sub_shape[dim_idx] = 0
                    sub = np.empty(sub_shape, dtype=self._data.dtype)
                else:
                    sub = np.take(self._data, indices, axis=dim_idx)
                result.append(DistributedArray(
                    data=sub,
                    dtype=self.dtype,
                    shard_spec=self.shard_spec,
                    logical_shape=tuple(sub.shape),
                ))
            return result
        else:
            # ── Block (contiguous) partition ─────────────────────────────────
            if dim_size % num_parts != 0:
                raise ValueError(
                    f"Cannot evenly split dimension {dim_idx} of size {dim_size} "
                    f"into {num_parts} parts for axis {axis!r}"
                )
            subs = np.array_split(self._data, num_parts, axis=dim_idx)
            return [
                DistributedArray(
                    data=sub,
                    dtype=self.dtype,
                    shard_spec=self.shard_spec,
                    logical_shape=tuple(sub.shape),
                )
                for sub in subs
            ]

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
