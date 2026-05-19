"""DDP / FSDP wrappers — Phase I of `docs/audit/execution_roadmap.md`.

Wraps a `tessera.nn.Module` and applies distributed-gradient collectives on
each backward pass. The forward pass is unchanged; backward inserts
`all_reduce` (DDP) or `reduce_scatter` (FSDP) on every `Parameter.grad`
before the user calls the optimizer step.

Today these wrappers run against the `tessera.testing.mock_collective`
in-process rank simulator — they're testable end-to-end on CPU without
needing real NCCL/RCCL. The C++ effect-aware adjoint collective insertion
pass (Phase F5) will eventually emit the same collectives at IR level for
compiled paths; the Python wrappers stay as the user-facing surface.

Limitations of this v1 implementation:

* DDP keeps full parameter copies on each rank. Gradient sync is
  `all_reduce(sum) / world_size`.
* FSDP stores 1/world_size of each parameter per rank. Forward `gather`s
  the full weight just-in-time; backward `reduce_scatter`s the gradient
  back into the local shard.
* No overlap of comm with compute (Phase G + F4-IR-rewrite work).
* No per-bucket gradient grouping (`bucket_cap_mb` etc.).
* Modules must be `tessera.nn.Module` instances. Plain functions don't have
  `parameters()` to iterate over.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..nn.module import Module


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _check_module(module: Module) -> None:
    if not isinstance(module, Module):
        raise TypeError(
            f"DDP/FSDP wrap expects a tessera.nn.Module, got {type(module).__name__}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DDP — Data parallel: replicated weights, sharded data, all_reduce gradients
# ─────────────────────────────────────────────────────────────────────────────


class DDP(Module):
    """Data-parallel wrapper.

    Each rank holds a full copy of every `Parameter`. Forward is a straight
    pass-through. After backward, call :meth:`sync_grads(mock_rank)` to
    `all_reduce` gradients across the DP mesh axis (mean-reduction).

    Example:

        ddp = tessera.distributed.DDP(model, mesh_axis="dp")
        with tape() as t:
            y = ddp(x)
            t.backward(y, cotangent=dy)
        ddp.sync_grads(mock_rank)
        # Now every rank has identical mean-reduced grads
    """

    def __init__(self, module: Module, mesh_axis: str = "dp") -> None:
        super().__init__()
        _check_module(module)
        self.module = module
        self.mesh_axis = str(mesh_axis)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def sync_grads(self, rank) -> None:
        """All-reduce every Parameter's `.grad` across the DP mesh axis,
        then divide by `world_size` (mean reduction).

        ``rank`` is a ``tessera.testing.mock_collective.MockRank`` for tests
        or any object exposing the same ``all_reduce(tensor, op="sum")`` /
        ``world_size`` surface for production wiring.
        """
        if rank is None or rank.world_size <= 1:
            return  # no-op on single-rank
        for p in self.module.parameters():
            if p.grad is None:
                continue
            local = p.grad.numpy()
            reduced = rank.all_reduce(local, op="sum")
            mean = reduced / float(rank.world_size)
            # Write back into the parameter's grad buffer in place
            p.grad._data[...] = mean.astype(p.grad.numpy().dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# FSDP — Fully sharded: each rank stores 1/world_size of every parameter
# ─────────────────────────────────────────────────────────────────────────────


class FSDP(Module):
    """Fully-Sharded Data-Parallel wrapper.

    Each rank stores 1/world_size of every parameter (sharded along the
    leading dim). On forward, gather the full weight just-in-time, run the
    forward, drop the gathered copy. On backward, the user runs the
    standard backward, then calls :meth:`sync_grads(mock_rank)` to
    `reduce_scatter` each gradient into the local 1/world_size shard.

    Sharding axis: 0 (the leading dim). For non-leading-dim sharding,
    transpose into the desired layout before wrapping.

    ZeRO stage mapping (deferred-items plan, Item 3):

    * ``stage=2`` — Phase I2 default; gradients + optimizer state are
      sharded across DP ranks (parameters held full per-rank). Numerical
      behaviour matches Phase I2's original FSDP v1.
    * ``stage=3`` — also shards parameters; sets
      ``self.zero_config.partition_parameters = True`` so the
      ``OptimizerShardPass`` IR-side machinery knows to emit
      ``tessera_sr.params_sharded`` annotations on the wrapped module's
      ops. Today the Python wrapper holds full params per rank between
      gather/reshard pairs (numerical behavior identical to stage 2);
      the actual NCCL all-gather of parameters before forward lands in
      Phase G. The ``stage=3`` flag is what production training expects
      to set even before that NCCL upgrade ships.

    Limitations (v1):

    * Initial state is unsharded — call :meth:`shard(mock_rank)` once
      before the first forward to discard the rank-non-local part of each
      parameter.
    * Forward gather is not overlapped with compute. Phase G adds the
      streaming version.
    """

    def __init__(
        self,
        module: Module,
        mesh_axis: str = "dp",
        *,
        stage: int = 2,
    ) -> None:
        super().__init__()
        _check_module(module)
        if stage not in (2, 3):
            raise ValueError(
                f"FSDP stage must be 2 or 3 (use DDP for stage 1); got {stage}"
            )
        self.module = module
        self.mesh_axis = str(mesh_axis)
        self.stage = int(stage)
        self._sharded = False
        # Lazy-build the ZeROConfig so this Python module doesn't pull in
        # solver_config at top-level import. Resolved on first attribute
        # access.
        self._zero_config_cache = None

    @property
    def zero_config(self):
        """Lazy-built :class:`tessera.compiler.solver_config.ZeROConfig`
        matching the wrapper's stage. Populates with
        ``partition_parameters=True`` when ``stage == 3`` so the IR-side
        ``OptimizerShardPass`` annotates the module accordingly."""
        if self._zero_config_cache is None:
            from ..compiler.solver_config import ZeROConfig
            self._zero_config_cache = ZeROConfig(
                stage=self.stage,
                dp_axis=self.mesh_axis,
                num_dp_ranks=1,  # filled in by `compile_bundle` at jit time
                partition_optimizer_states=True,
                partition_gradients=True,
                partition_parameters=(self.stage == 3),
            )
        return self._zero_config_cache

    def shard(self, rank) -> None:
        """Initial shard — drop everything but rank-local 1/world_size of
        each parameter along axis 0.

        Idempotent: calling more than once is a no-op after the first.
        """
        if self._sharded or rank is None or rank.world_size <= 1:
            self._sharded = True
            return
        ws = rank.world_size
        for p in self.module.parameters():
            buf = p._data._data
            if buf.shape[0] % ws != 0:
                raise ValueError(
                    f"FSDP: parameter shape[0]={buf.shape[0]} not divisible by "
                    f"world_size={ws}; cannot shard along leading dim"
                )
            shard_size = buf.shape[0] // ws
            local = buf[rank.rank * shard_size:(rank.rank + 1) * shard_size].copy()
            # Re-allocate the parameter's storage to the smaller shape
            new_shape = (shard_size,) + buf.shape[1:]
            new_buf = np.empty(new_shape, dtype=buf.dtype)
            new_buf[...] = local
            p._data._data = new_buf
        self._sharded = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # In a real implementation, forward would all_gather every
        # parameter just-in-time, run the forward, then drop. The numpy
        # reference path doesn't have a way to swap parameter storage
        # transparently per-call, so v1 requires the user to gather
        # explicitly via :meth:`gather_for_forward(rank)` before calling
        # the wrapped module. After the call, parameters can be re-sharded
        # via the next backward + sync_grads.
        return self.module(*args, **kwargs)

    def gather_for_forward(self, rank) -> None:
        """All-gather each parameter to its full size for the duration of
        the forward pass. Pair with :meth:`reshard_after_forward(rank)` to
        return to sharded storage.
        """
        if rank is None or rank.world_size <= 1:
            return
        for p in self.module.parameters():
            local = p._data._data
            full = rank.all_gather(local, axis=0)
            p._data._data = full

    def reshard_after_forward(self, rank) -> None:
        """Drop the all-gathered full parameter back to the rank-local shard.
        Inverse of :meth:`gather_for_forward`."""
        if rank is None or rank.world_size <= 1:
            return
        ws = rank.world_size
        for p in self.module.parameters():
            buf = p._data._data
            if buf.shape[0] % ws != 0:
                continue  # already resharded or non-shardable
            shard_size = buf.shape[0] // ws
            local = buf[rank.rank * shard_size:(rank.rank + 1) * shard_size].copy()
            new_shape = (shard_size,) + buf.shape[1:]
            new_buf = np.empty(new_shape, dtype=buf.dtype)
            new_buf[...] = local
            p._data._data = new_buf

    def sync_grads(self, rank) -> None:
        """Reduce-scatter each Parameter's `.grad` into the local shard
        (mean reduction).

        Assumes gradients were computed on the full (gathered) parameter
        and live at full shape. After this call, each rank's `.grad` has
        the local 1/world_size slice scaled to the mean.
        """
        if rank is None or rank.world_size <= 1:
            return
        ws = rank.world_size
        for p in self.module.parameters():
            if p.grad is None:
                continue
            full_grad = p.grad.numpy()
            local_grad = rank.reduce_scatter(full_grad, axis=0, op="sum") / float(ws)
            # Replace .grad with the smaller local shard
            p.grad = local_grad


class ZeRO3(FSDP):
    """DeepSpeed-style explicit alias for ``FSDP(module, stage=3)``.

    Functionally identical to ``FSDP(module, stage=3)``; provides the
    ZeRO-3 naming production training code typically uses. The wrapper's
    ``zero_config`` will report ``stage=3`` and
    ``partition_parameters=True``, which the IR-side
    ``OptimizerShardPass`` reads to emit ``tessera_sr.params_sharded``
    annotations on the module's ops.
    """

    def __init__(self, module: Module, mesh_axis: str = "dp") -> None:
        super().__init__(module, mesh_axis=mesh_axis, stage=3)


__all__ = ["DDP", "FSDP", "ZeRO3"]
