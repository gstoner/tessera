"""Megatron-style tensor-parallel rewrite (Workstream E).

The audit's #5 lesson, re-scoped per the P2 feedback. Tessera already has the
*backward* collective machinery (``AdjointCollectiveInsertionPass`` inserts
reduce-scatter / all-gather / all-reduce on cotangents) and a manual
``DistributedPlan`` sharding contract. The two gaps:

  E1 — an **automatic rewrite** of a plain linear into column/row/sequence
       parallel + the right collectives, instead of a hand-authored plan.
  E2 — **numeric cross-rank gradient equivalence**: a test that the sharded
       backward gradients, recombined, equal the single-rank gradients.

This module supplies both. ``rewrite_linear(W, TPSpec)`` produces a
``ParallelLinear`` whose per-rank forward/backward use real collectives (run over
``MockRankGroup`` threads, Decision #6); ``verify_tp_gradient_equivalence`` is the
E2 oracle.

Megatron rules implemented:

  * **column** — shard W along output cols. Forward: ``Y_r = X @ W_r``,
    ``Y = all_gather(Y_r)``. Backward: ``dW_r = Xᵀ @ dY_r`` (local),
    ``dX = all_reduce(dY_r @ W_rᵀ)``.
  * **row** — shard W along input rows, X along its columns. Forward:
    ``Y = all_reduce(X_r @ W_r)``. Backward: ``dW_r = X_rᵀ @ dY`` (local),
    ``dX_r = dY @ W_rᵀ`` (local shard).
  * **sequence** — shard activations along tokens, W replicated. Forward:
    ``Y = all_gather_0(X_r @ W)``. Backward: ``dW = all_reduce(X_rᵀ @ dY_r)``,
    ``dX_r = dY_r @ Wᵀ`` (local).

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream E).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np


class TPMode(enum.Enum):
    COLUMN = "column"
    ROW = "row"
    SEQUENCE = "sequence"
    REPLICATED = "replicated"


@dataclass(frozen=True)
class TPSpec:
    """How a linear is sharded across the tensor-parallel mesh axis."""

    mode: TPMode
    world_size: int
    mesh_axis: str = "tp"

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", TPMode(self.mode))
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")


def _even_split(total: int, world_size: int) -> int:
    if total % world_size != 0:
        raise ValueError(f"dim {total} not divisible by world_size {world_size}")
    return total // world_size


@dataclass
class ParallelLinear:
    """A plain linear rewritten for tensor parallelism.

    Holds the full weight (the rewrite input) and the spec; the per-rank methods
    take a ``MockRank`` and use its collectives so combination happens in-band —
    every rank returns the *full* combined ``Y`` / ``dX`` plus its local ``dW``
    shard.
    """

    W: np.ndarray            # (C_in, C_out)
    spec: TPSpec

    # ── sharding ──
    def weight_shard(self, rank: int) -> np.ndarray:
        W = self.W
        ws = self.spec.world_size
        if self.spec.mode is TPMode.COLUMN:
            n = _even_split(W.shape[1], ws)
            return W[:, rank * n:(rank + 1) * n]
        if self.spec.mode is TPMode.ROW:
            n = _even_split(W.shape[0], ws)
            return W[rank * n:(rank + 1) * n, :]
        return W  # sequence / replicated: weight is replicated

    def input_shard(self, rank: int, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        ws = self.spec.world_size
        if self.spec.mode is TPMode.ROW:
            n = _even_split(X.shape[1], ws)           # shard along C_in
            return X[:, rank * n:(rank + 1) * n]
        if self.spec.mode is TPMode.SEQUENCE:
            n = _even_split(X.shape[0], ws)           # shard along tokens
            return X[rank * n:(rank + 1) * n, :]
        return X                                      # column / replicated: full X

    def grad_slice(self, rank: int, dY: np.ndarray) -> np.ndarray:
        """The upstream-gradient slice this rank owns (mirror of the output layout)."""
        dY = np.asarray(dY, np.float64)
        ws = self.spec.world_size
        if self.spec.mode is TPMode.COLUMN:
            n = _even_split(dY.shape[1], ws)          # output sharded along cols
            return dY[:, rank * n:(rank + 1) * n]
        if self.spec.mode is TPMode.SEQUENCE:
            n = _even_split(dY.shape[0], ws)          # output sharded along tokens
            return dY[rank * n:(rank + 1) * n, :]
        return dY                                     # row: output replicated

    # ── per-rank forward (returns the full combined Y on every rank) ──
    def forward(self, mr: Any, X: np.ndarray) -> np.ndarray:
        Wr = self.weight_shard(mr.rank)
        Xr = self.input_shard(mr.rank, X)
        if self.spec.mode is TPMode.COLUMN:
            return mr.all_gather(Xr @ Wr, axis=1)          # concat output cols
        if self.spec.mode is TPMode.ROW:
            return mr.all_reduce(Xr @ Wr, op="sum")        # sum partial outputs
        if self.spec.mode is TPMode.SEQUENCE:
            return mr.all_gather(Xr @ Wr, axis=0)          # concat token rows
        return Xr @ Wr                                     # replicated

    # ── per-rank backward → (full dX, local dW shard) ──
    def backward(self, mr: Any, X: np.ndarray, dY: np.ndarray):
        Wr = self.weight_shard(mr.rank)
        Xr = self.input_shard(mr.rank, X)
        dYr = self.grad_slice(mr.rank, dY)
        if self.spec.mode is TPMode.COLUMN:
            dW_shard = Xr.T @ dYr                           # local (C_in, n_out)
            dX = mr.all_reduce(dYr @ Wr.T, op="sum")        # sum across col shards
            return dX, dW_shard
        if self.spec.mode is TPMode.ROW:
            dW_shard = Xr.T @ dYr                           # local (n_in, C_out)
            dX_shard = dYr @ Wr.T                            # local (N, n_in)
            dX = mr.all_gather(dX_shard, axis=1)            # concat C_in shards
            return dX, dW_shard
        if self.spec.mode is TPMode.SEQUENCE:
            dW = mr.all_reduce(Xr.T @ dYr, op="sum")        # full dW (replicated)
            dX_shard = dYr @ Wr.T                            # local token rows
            dX = mr.all_gather(dX_shard, axis=0)            # concat token rows
            return dX, dW
        return Xr.T @ dY, dYr @ Wr.T


def rewrite_linear(W: np.ndarray, spec: TPSpec) -> ParallelLinear:
    """Rewrite a plain linear weight into a tensor-parallel plan.

    This is the automatic-rewrite contract E1 was missing: a caller hands a plain
    ``(C_in, C_out)`` weight + a :class:`TPSpec`, and gets back a plan that knows
    how to shard, run, and differentiate it with the correct collectives — no
    hand-authored ``DistributedPlan`` required.
    """
    W = np.asarray(W, np.float64)
    if W.ndim != 2:
        raise ValueError(f"weight must be 2-D (C_in, C_out); got {W.shape}")
    return ParallelLinear(W=W, spec=spec)


# ── E2 oracle: sharded gradients, recombined, equal single-rank gradients ─────


@dataclass(frozen=True)
class TPGradVerdict:
    relation: str            # "equivalent" | "divergent"
    max_dx_err: float
    max_dw_err: float
    max_y_err: float
    detail: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.relation == "equivalent"


def _reassemble_dw(dw_shards: list[np.ndarray], spec: TPSpec) -> np.ndarray:
    if spec.mode is TPMode.COLUMN:
        return np.concatenate(dw_shards, axis=1)   # shards are output-col blocks
    if spec.mode is TPMode.ROW:
        return np.concatenate(dw_shards, axis=0)   # shards are input-row blocks
    return dw_shards[0]                              # sequence: dW replicated


def verify_tp_gradient_equivalence(
    X: np.ndarray, W: np.ndarray, spec: TPSpec, *, seed: int = 0, tol: float = 1e-9,
) -> TPGradVerdict:
    """Run the sharded forward+backward over MockRankGroup and compare to single-rank.

    The cross-rank gradient equivalence test P2 named as the real gap: combined
    ``dX``/``dW`` from the sharded program must equal the monolithic autodiff
    gradients of ``Y = X @ W``.
    """
    from ..testing.mock_collective import MockRankGroup

    X = np.asarray(X, np.float64)
    W = np.asarray(W, np.float64)
    rng = np.random.default_rng(seed)
    Y_ref = X @ W
    dY = rng.standard_normal(Y_ref.shape)            # arbitrary upstream gradient
    dW_ref = X.T @ dY
    dX_ref = dY @ W.T

    pl = rewrite_linear(W, spec)
    group = MockRankGroup(spec.world_size, {spec.mesh_axis: spec.world_size})

    def worker(mr: Any):
        Y = pl.forward(mr, X)
        dX, dW_shard = pl.backward(mr, X, dY)
        return Y, dX, dW_shard

    results = group.run(worker)
    ys, dxs, dw_shards = zip(*results)

    # Every rank returns the full combined Y and dX → all equal to the reference.
    max_y_err = max(float(np.max(np.abs(y - Y_ref))) for y in ys)
    max_dx_err = max(float(np.max(np.abs(dx - dX_ref))) for dx in dxs)
    dW_combined = _reassemble_dw(list(dw_shards), spec)
    max_dw_err = float(np.max(np.abs(dW_combined - dW_ref)))

    worst = max(max_y_err, max_dx_err, max_dw_err)
    rel = "equivalent" if worst <= tol else "divergent"
    detail = (f"Y/dX/dW errs = {max_y_err:.1e}/{max_dx_err:.1e}/{max_dw_err:.1e} "
              f"(≤ {tol:.0e})" if rel == "equivalent" else
              f"sharded grads diverge from single-rank (worst {worst:.1e} > {tol:.0e})")
    return TPGradVerdict(rel, max_dx_err, max_dw_err, max_y_err, detail)


__all__ = [
    "TPMode",
    "TPSpec",
    "ParallelLinear",
    "rewrite_linear",
    "verify_tp_gradient_equivalence",
    "TPGradVerdict",
]
