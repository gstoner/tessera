"""
tessera.testing.mock_collective — thread-based fake multi-rank group.

Provides MockRankGroup: a simple in-process multi-rank simulator for testing
index_launch and collective semantics without NCCL, MPI, or CUDA.

Design:
  - Each "rank" is represented by a Python thread sharing an in-process
    barrier and shared memory dict.
  - Collectives (all_reduce, reduce_scatter, all_gather) use a barrier +
    per-rank buffer approach.
  - Phase 1 scope: CPU tensors (numpy arrays) only.

Reference: CLAUDE.md §CPU Collective Mock
           tests/phase1/conftest.py — uses MockRankGroup(n=4, mesh_axes={"dp": 4})
"""

from __future__ import annotations
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class MockCollectiveError(Exception):
    """Raised when a mock collective operation fails (e.g., shape mismatch)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MockRank — per-rank view
# ─────────────────────────────────────────────────────────────────────────────

class MockRank:
    """
    Represents a single rank within a MockRankGroup.

    Provides collective operations that synchronise across all ranks in the
    group using Python threading primitives.

    Attributes:
        rank      : this rank's index (0-based)
        world_size: total number of ranks in the group
        mesh_axes : dict of axis_name → size (e.g. {"dp": 4})
    """

    def __init__(
        self,
        rank: int,
        group: "MockRankGroup",
    ) -> None:
        self.rank = rank
        self.world_size = group.world_size
        self.mesh_axes = group.mesh_axes
        self._group = group

    # ── Collective operations ────────────────────────────────────────────────

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """
        Synchronise all ranks and reduce tensor across all of them.

        Args:
            tensor : local contribution (numpy array)
            op     : reduction op — "sum", "max", "min", "prod"

        Returns:
            Reduced numpy array, same shape as input.
        """
        return self._group._all_reduce(self.rank, tensor, op)

    def reduce_scatter(
        self,
        tensor: np.ndarray,
        axis: int = 0,
        op: str = "sum",
    ) -> np.ndarray:
        """
        Reduce across all ranks and scatter one slice to each rank.

        Args:
            tensor : local contribution (full size)
            axis   : dimension to scatter along
            op     : reduction op

        Returns:
            1/world_size slice of the reduced tensor for this rank.
        """
        return self._group._reduce_scatter(self.rank, tensor, axis, op)

    def all_gather(self, tensor: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Gather tensors from all ranks and concatenate along axis.

        Args:
            tensor : local shard
            axis   : dimension to gather along

        Returns:
            Concatenated tensor of all shards.
        """
        return self._group._all_gather(self.rank, tensor, axis)

    def barrier(self) -> None:
        """Wait until all ranks reach this barrier."""
        self._group._barrier()

    def __repr__(self) -> str:
        return f"MockRank(rank={self.rank}, world_size={self.world_size})"


# ─────────────────────────────────────────────────────────────────────────────
# MockRankGroup
# ─────────────────────────────────────────────────────────────────────────────

class MockRankGroup:
    """
    A simulated multi-rank group for testing distributed Tessera programs.

    Creates `n` fake ranks backed by Python threads that share in-process
    numpy buffers. Collectives are implemented as barrier + numpy reduction.

    Usage:
        group = MockRankGroup(n=4, mesh_axes={"dp": 4})

        def worker(rank: MockRank):
            local = np.ones((256,), dtype=np.float32) * rank.rank
            result = rank.all_reduce(local, op="sum")
            assert result.sum() == 256 * (0 + 1 + 2 + 3)

        group.run(worker)

    Args:
        n         : total number of ranks (world_size)
        mesh_axes : logical mesh mapping axis_name → count.
                    Must multiply to n (e.g. {"dp": 2, "tp": 2} for n=4).

    Raises:
        ValueError: if mesh_axes product != n
    """

    def __init__(
        self,
        n: int,
        mesh_axes: Optional[Dict[str, int]] = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"MockRankGroup requires n >= 1, got {n}")

        if mesh_axes is None:
            mesh_axes = {"dp": n}

        # Validate mesh_axes product == n
        product = 1
        for size in mesh_axes.values():
            product *= size
        if product != n:
            raise ValueError(
                f"mesh_axes product ({product}) != n ({n}). "
                f"mesh_axes={mesh_axes}"
            )

        self.world_size = n
        self.mesh_axes = dict(mesh_axes)

        # Shared state for collectives
        self._barrier_obj = threading.Barrier(n)
        self._lock = threading.Lock()
        self._shared_buffers: Dict[str, List[Optional[np.ndarray]]] = {}

        # Build rank objects
        self.ranks: List[MockRank] = [
            MockRank(rank=i, group=self) for i in range(n)
        ]

    def run(
        self,
        fn: Callable[[MockRank], Any],
        *,
        timeout: Optional[float] = 30.0,
    ) -> List[Any]:
        """
        Run fn on each rank in a separate thread and collect results.

        Args:
            fn      : function taking a MockRank, returns a result
            timeout : per-thread timeout in seconds (default 30s)

        Returns:
            List of per-rank results in rank order.

        Raises:
            MockCollectiveError: if any rank raises an exception.
        """
        results: List[Any] = [None] * self.world_size
        errors: List[Optional[Exception]] = [None] * self.world_size

        def _run_rank(rank: MockRank) -> None:
            try:
                results[rank.rank] = fn(rank)
            except Exception as exc:
                errors[rank.rank] = exc

        threads = [
            threading.Thread(target=_run_rank, args=(r,), daemon=True)
            for r in self.ranks
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        # Check for errors
        errs = [(i, e) for i, e in enumerate(errors) if e is not None]
        if errs:
            rank_idx, exc = errs[0]
            raise MockCollectiveError(
                f"Rank {rank_idx} raised {type(exc).__name__}: {exc}"
            ) from exc

        # Check for hung threads
        hung = [i for i, t in enumerate(threads) if t.is_alive()]
        if hung:
            raise MockCollectiveError(
                f"Ranks {hung} did not complete within {timeout}s timeout"
            )

        return results

    # ── Internal collective implementations ─────────────────────────────────

    def _barrier(self) -> None:
        self._barrier_obj.wait()

    def _all_reduce(self, rank: int, tensor: np.ndarray, op: str) -> np.ndarray:
        """All-reduce: all ranks contribute, all get the result."""
        key = "all_reduce"
        self._deposit(key, rank, tensor)
        self._barrier_obj.wait()

        buffers = self._shared_buffers[key]
        result = self._reduce(buffers, op)
        self._barrier_obj.wait()

        self._withdraw(key)
        return result

    def _reduce_scatter(
        self, rank: int, tensor: np.ndarray, axis: int, op: str
    ) -> np.ndarray:
        """Reduce-scatter: all-reduce then slice rank's portion."""
        reduced = self._all_reduce(rank, tensor, op)
        # Slice rank's portion along axis
        dim_size = reduced.shape[axis]
        if dim_size % self.world_size != 0:
            raise MockCollectiveError(
                f"reduce_scatter: axis {axis} size {dim_size} not divisible "
                f"by world_size {self.world_size}"
            )
        shard_size = dim_size // self.world_size
        idx = [slice(None)] * reduced.ndim
        idx[axis] = slice(rank * shard_size, (rank + 1) * shard_size)
        return reduced[tuple(idx)].copy()

    def _all_gather(self, rank: int, tensor: np.ndarray, axis: int) -> np.ndarray:
        """All-gather: collect shard from each rank, concatenate."""
        key = "all_gather"
        self._deposit(key, rank, tensor)
        self._barrier_obj.wait()

        buffers = self._shared_buffers[key]
        result = np.concatenate(buffers, axis=axis)  # type: ignore[arg-type]
        self._barrier_obj.wait()

        self._withdraw(key)
        return result

    # ── Buffer management ────────────────────────────────────────────────────

    def _deposit(self, key: str, rank: int, tensor: np.ndarray) -> None:
        """Thread-safe deposit of a tensor into the shared buffer slot."""
        with self._lock:
            if key not in self._shared_buffers:
                self._shared_buffers[key] = [None] * self.world_size
            self._shared_buffers[key][rank] = tensor.copy()

    def _withdraw(self, key: str) -> None:
        """Clear the shared buffer slot (called after all ranks have read)."""
        with self._lock:
            self._shared_buffers.pop(key, None)

    @staticmethod
    def _reduce(
        buffers: List[Optional[np.ndarray]], op: str
    ) -> np.ndarray:
        """Reduce a list of numpy arrays with the given op."""
        arrays = [b for b in buffers if b is not None]
        if not arrays:
            raise MockCollectiveError("No buffers to reduce")
        result = arrays[0].copy()
        for arr in arrays[1:]:
            if op == "sum":
                result = result + arr
            elif op == "max":
                result = np.maximum(result, arr)
            elif op == "min":
                result = np.minimum(result, arr)
            elif op == "prod":
                result = result * arr
            else:
                raise MockCollectiveError(f"Unknown reduction op {op!r}")
        return result

    def __repr__(self) -> str:
        return f"MockRankGroup(n={self.world_size}, mesh_axes={self.mesh_axes})"
