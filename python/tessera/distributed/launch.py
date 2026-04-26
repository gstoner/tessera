"""
index_launch and @kernel — fan a kernel out across mesh partitions.

index_launch(axis="tp")(fn)(A.parts("tp"), B.parts("tp"), C.parts("tp"))

This is the primary mechanism for expressing tensor-parallel and data-parallel
kernel dispatch. It lowers to a `schedule.mesh.region` op in Schedule IR with
the kernel body as its region.

In Phase 1 (CPU, single-process mock):
  - index_launch iterates over per-rank shard lists sequentially
  - No actual parallelism; verifies the functional path only

In Phase 3 (multi-GPU):
  - Each shard is dispatched to a separate CUDA stream / rank
  - Collectives are inserted automatically at mesh boundaries

Reference: docs/programming_guide/Tessera_Programming_Guide_Chapter4_Execution_Model.md §4.4
"""

from __future__ import annotations
import functools
import inspect
from typing import Callable, Any, List, Optional

from .array import DistributedArray


# ─────────────────────────────────────────────────────────────────────────────
# @kernel decorator
# ─────────────────────────────────────────────────────────────────────────────

class KernelFn:
    """
    A Tessera kernel function — a tile-level computation that operates on
    one shard of a distributed tensor.

    Created by @tessera.kernel. In Phase 1 the kernel body executes as plain
    Python/numpy. In Phase 3 it is compiled to GPU PTX via the Tile IR path.
    """

    def __init__(self, fn: Callable, name: Optional[str] = None) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<TesseraKernel {self.name!r}>"


def kernel(fn: Callable) -> KernelFn:
    """
    Decorator that marks a function as a Tessera tile kernel.

    Kernels operate on individual shards of distributed tensors. They are
    dispatched by index_launch across mesh partitions.

    Usage:
        @tessera.kernel
        def tp_gemm(A: f16[..., ...], B: f16[..., ...], C: mut_f32[..., ...]):
            C[:] = ops.gemm(A, B)

        tessera.index_launch(axis="tp")(tp_gemm)(
            X.parts("tp"), W.parts("tp"), Y.parts("tp")
        )
    """
    return KernelFn(fn)


# ─────────────────────────────────────────────────────────────────────────────
# index_launch
# ─────────────────────────────────────────────────────────────────────────────

class IndexLauncher:
    """
    Returned by index_launch(axis=...) — callable with (kernel)(shards...).

    Phase 1 implementation: iterates over shard lists sequentially.
    Phase 3: dispatches to CUDA streams / MPI ranks in parallel.
    """

    def __init__(self, axis: str) -> None:
        self.axis = axis

    def __call__(self, fn: Callable) -> "_ShardDispatcher":
        """Bind the kernel to this launcher, returning a dispatcher."""
        return _ShardDispatcher(fn=fn, axis=self.axis)

    def __repr__(self) -> str:
        return f"IndexLauncher(axis={self.axis!r})"


class _ShardDispatcher:
    """
    Produced by IndexLauncher(fn). Callable with the shard lists.
    """

    def __init__(self, fn: Callable, axis: str) -> None:
        self._fn = fn
        self.axis = axis

    def __call__(self, *shard_lists: Any) -> List[Any]:
        """
        Execute the kernel over all shards.

        Args:
            *shard_lists: one list of shards per kernel argument.
                          All lists must have the same length (number of ranks).

        Returns:
            List of per-rank results (None for void kernels).

        Phase 1: sequential iteration, no parallelism.

        Example:
            tessera.index_launch(axis="tp")(tp_gemm)(
                A.parts("tp"),   # list of 4 shards
                B.parts("tp"),   # list of 4 shards
                C.parts("tp"),   # list of 4 shards
            )
            # → calls tp_gemm(A[0],B[0],C[0]), tp_gemm(A[1],B[1],C[1]), ...
        """
        # Validate: all inputs must be lists of the same length
        processed = []
        for arg in shard_lists:
            if isinstance(arg, list):
                processed.append(arg)
            elif isinstance(arg, DistributedArray):
                # Convenience: auto-call .parts(axis) if not already done
                processed.append(arg.parts(self.axis))
            else:
                # Scalar or non-distributed arg → replicate across all ranks
                processed.append(None)  # handled below

        # Determine number of ranks from the first list argument
        n_ranks = None
        for p in processed:
            if p is not None:
                n_ranks = len(p)
                break

        if n_ranks is None:
            raise ValueError("index_launch requires at least one list-of-shards argument")

        # Validate all lists have the same length
        for i, p in enumerate(processed):
            if p is not None and len(p) != n_ranks:
                raise ValueError(
                    f"Shard list {i} has {len(p)} entries but expected {n_ranks} "
                    f"(from axis {self.axis!r})"
                )

        # Phase 1: sequential dispatch
        results = []
        for rank_idx in range(n_ranks):
            rank_args = [
                p[rank_idx] if p is not None else shard_lists[i]
                for i, p in enumerate(processed)
            ]
            result = self._fn(*rank_args)
            results.append(result)

        return results

    def __repr__(self) -> str:
        name = getattr(self._fn, "name", getattr(self._fn, "__name__", "?"))
        return f"<ShardDispatcher kernel={name!r} axis={self.axis!r}>"


def index_launch(axis: str) -> IndexLauncher:
    """
    Fan a kernel out across all partitions of a distributed tensor on `axis`.

    Usage:
        tessera.index_launch(axis="tp")(my_kernel)(
            A.parts("tp"),
            B.parts("tp"),
            C.parts("tp"),
        )

    This distributes my_kernel across all tensor-parallel ranks.
    Collectives (all_gather, reduce_scatter) are inserted automatically by the
    compiler at mesh region boundaries (Phase 2+).

    Args:
        axis: mesh axis name to fan out over (e.g. "tp", "dp", "pp")

    Returns:
        IndexLauncher — call it with your kernel to get a ShardDispatcher
    """
    return IndexLauncher(axis=axis)
