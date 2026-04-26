"""
Domains and Distributions.

A Domain describes the LOGICAL iteration/data space (shape only).
A Distribution describes HOW it maps to a mesh (placement strategy).

These are always separate objects — algorithm vs placement. Never merge.

Reference: docs/programming_guide/Tessera_Programming_Guide_Chapter10_Portability.md
           docs/programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md

IR lowering:
    Rect((B, S, D))  +  Block(mesh_axes=("dp",))
    →  schedule.mesh.define @M dims=[...] axis_names=[...]
       with shard annotation {tessera.shard = {axes=["dp"], dims=[0]}}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, Optional

from .shard import ShardSpec, MeshSpec


# ─────────────────────────────────────────────────────────────────────────────
# Domains
# ─────────────────────────────────────────────────────────────────────────────

class Domain:
    """Base class for all domain types."""

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class Rect(Domain):
    """
    A rectangular (dense) domain with fixed shape.

    This is the most common domain type. It represents a dense tensor of the
    given shape with no sparsity or irregular structure.

    Args:
        dims: tuple of dimension sizes, e.g. (batch, seq_len, d_model)

    Example:
        D = tessera.domain.Rect((4, 128, 256))
        assert D.shape == (4, 128, 256)
        assert D.rank == 3
        assert D.numel == 4 * 128 * 256
    """
    _dims: Tuple[int, ...]

    def __init__(self, dims: Tuple[int, ...]) -> None:
        if not dims:
            raise ValueError("Rect domain must have at least one dimension")
        for i, d in enumerate(dims):
            if not isinstance(d, int) or d < 1:
                raise ValueError(
                    f"Rect dimension {i} must be a positive integer, got {d!r}"
                )
        object.__setattr__(self, "_dims", tuple(dims))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._dims

    @property
    def numel(self) -> int:
        result = 1
        for d in self._dims:
            result *= d
        return result

    def __repr__(self) -> str:
        return f"Rect({self._dims})"


# ─────────────────────────────────────────────────────────────────────────────
# Distributions
# ─────────────────────────────────────────────────────────────────────────────

class Distribution:
    """Base class for all distribution strategies."""

    def make_shard_spec(self, domain: Domain) -> ShardSpec:
        """
        Produce a ShardSpec from this distribution applied to the given domain.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def to_ir_attr(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Block(Distribution):
    """
    Block (contiguous) partition over mesh axes.

    Partitions the first N dimensions of the domain across the given mesh axes,
    one dimension per axis, in order.

    Args:
        mesh_axes: tuple of mesh axis names. The i-th axis partitions dim i
                   of the domain. Length must be <= domain.rank.

    Example:
        # Partition dim-0 (batch) over "dp", dim-1 (seq) over "tp"
        dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
        spec = dist.make_shard_spec(Rect((8, 1024, 256)))
        # → ShardSpec(partition=(0,1), mesh_axes=("dp","tp"))

    Phase 1 scope: only contiguous block partition (no strided, no 2D tiling).
    """
    mesh_axes: Tuple[str, ...]

    def __init__(self, mesh_axes: Tuple[str, ...]) -> None:
        if not mesh_axes:
            raise ValueError("Block distribution requires at least one mesh axis")
        if isinstance(mesh_axes, str):
            raise TypeError(
                "mesh_axes must be a tuple of strings, not a single string. "
                f"Did you mean Block(mesh_axes=({mesh_axes!r},)) ?"
            )
        object.__setattr__(self, "mesh_axes", tuple(mesh_axes))

    def make_shard_spec(self, domain: Domain) -> ShardSpec:
        if len(self.mesh_axes) > domain.rank:
            raise ValueError(
                f"Block distribution has {len(self.mesh_axes)} axes but "
                f"domain has only {domain.rank} dimensions"
            )
        partition = tuple(range(len(self.mesh_axes)))
        return ShardSpec(partition=partition, mesh_axes=self.mesh_axes)

    def to_ir_attr(self) -> str:
        axes = ", ".join(f'"{a}"' for a in self.mesh_axes)
        return f"{{tessera.dist = {{kind = \"block\", axes = [{axes}]}}}}"

    def __repr__(self) -> str:
        return f"Block(mesh_axes={self.mesh_axes})"


@dataclass(frozen=True)
class Cyclic(Distribution):
    """
    Cyclic (round-robin) partition over a mesh axis.

    Distributes elements cyclically rather than in contiguous blocks.
    Useful for load-balanced workloads (e.g., MoE expert assignment).

    Phase 1 scope: defined but make_shard_spec raises NotImplementedError.
    Will be wired in Phase 2 when the MoE A2A backend is built.
    """
    mesh_axes: Tuple[str, ...]

    def __init__(self, mesh_axes: Tuple[str, ...]) -> None:
        object.__setattr__(self, "mesh_axes", tuple(mesh_axes))

    def make_shard_spec(self, domain: Domain) -> ShardSpec:
        raise NotImplementedError(
            "Cyclic distribution sharding is a Phase 2 feature (MoE A2A). "
            "Use Block for Phase 1."
        )

    def __repr__(self) -> str:
        return f"Cyclic(mesh_axes={self.mesh_axes})"


@dataclass(frozen=True)
class Replicated(Distribution):
    """
    No partition — tensor is replicated across all ranks.

    Used for bias terms, small embeddings, and model configurations that
    should be identical on every rank.

    Example:
        dist = tessera.dist.Replicated()
        spec = dist.make_shard_spec(Rect((256,)))
        assert spec.replicated == True
    """

    def make_shard_spec(self, domain: Domain) -> ShardSpec:
        return ShardSpec.replicate()

    def to_ir_attr(self) -> str:
        return '{tessera.dist = {kind = "replicated"}}'

    def __repr__(self) -> str:
        return "Replicated()"
