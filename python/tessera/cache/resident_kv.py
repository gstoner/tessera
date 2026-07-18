"""Device-resident latent KV cache — R4.

The final phase of the GPU-resident architecture: the compressed latent
``c_kv`` and the shared decoupled-RoPE key slice ``k_rope`` live in **resident
device buffers** that persist across decode steps. So:

  * **append writes only the new token** straight into the resident buffer (a
    host write through the unified-memory `.numpy()` view — no upload, the GPU
    sees it immediately), instead of re-sending the whole window every step; and
  * the decode reads the populated window as a **zero-copy prefix view** of the
    resident buffer (no per-step cache copy).

Contrast with the host caches (`MLAPagedDecoder` / `MLABlockPagedCache`), which
hold numpy state and re-wrap/re-upload the window each step. `ResidentLatentKVCache`
is the device-resident backing those would adopt for serving.

This covers the **contiguous** latent cache. A device-resident *block-paged*
cache (non-contiguous block-table gather on-device) is the documented follow-on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import runtime as R


class ResidentLatentKVCache:
    """Growing, device-resident ``c_kv`` + ``k_rope`` cache.

    Parameters
    ----------
    latent_dim, rope_dim
        Widths of the compressed latent and the shared rope key slice.
    max_seq
        Capacity in tokens. ``append`` past this raises (eviction is a follow-on).
    """

    def __init__(self, *, latent_dim: int, rope_dim: int, max_seq: int) -> None:
        if latent_dim <= 0 or rope_dim <= 0 or max_seq <= 0:
            raise ValueError("latent_dim / rope_dim / max_seq must be positive")
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.max_seq = int(max_seq)
        self.current_seq = 0
        dt = R.DeviceTensor
        self._latent = dt.empty((self.max_seq, self.latent_dim), np.float32)
        self._rope = dt.empty((self.max_seq, self.rope_dim), np.float32)
        self._resident = self._latent is not None and self._rope is not None
        if self._latent is not None and self._rope is not None:
            # views over the resident storage — host writes go straight to the
            # buffer the GPU reads (unified memory), no upload.
            self._latent_view = self._latent.numpy()
            self._rope_view = self._rope.numpy()
        else:  # portable fallback: plain numpy backing
            self._latent_view = np.zeros((self.max_seq, self.latent_dim), np.float32)
            self._rope_view = np.zeros((self.max_seq, self.rope_dim), np.float32)

    # ------------------------------------------------------------------
    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def shape(self) -> tuple[int, int]:
        return (self.current_seq, self.latent_dim)

    def cache_bytes_per_token(self) -> int:
        return (self.latent_dim + self.rope_dim) * 4

    # ------------------------------------------------------------------
    def append(self, c_kv: Any, k_rope: Any) -> "ResidentLatentKVCache":
        """Append ``n`` tokens' latent + rope key. The data is written **in
        place** into the resident device buffer — no upload, no reallocation."""
        c = np.asarray(c_kv, np.float32)
        r = np.asarray(k_rope, np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        if c.shape[1] != self.latent_dim or r.shape[1] != self.rope_dim:
            raise ValueError(
                f"expected c_kv [*,{self.latent_dim}] / k_rope [*,{self.rope_dim}]; "
                f"got {c.shape} / {r.shape}")
        if c.shape[0] != r.shape[0]:
            raise ValueError("c_kv and k_rope must append the same token count")
        n = c.shape[0]
        if self.current_seq + n > self.max_seq:
            raise ValueError(
                f"append would exceed max_seq={self.max_seq}: "
                f"current={self.current_seq}, appending={n}")
        s, e = self.current_seq, self.current_seq + n
        self._latent_view[s:e] = c
        self._rope_view[s:e] = r
        self.current_seq = e
        return self

    # ------------------------------------------------------------------
    def latent_window(self) -> Any:
        """Resident ``DeviceTensor`` view of the populated latent
        ``[current_seq, latent_dim]`` (zero-copy prefix). Falls back to a numpy
        slice when not resident."""
        if self._resident and self._latent is not None:
            return self._latent.prefix_view(self.current_seq)
        return self._latent_view[:self.current_seq]

    def rope_window(self) -> Any:
        if self._resident and self._rope is not None:
            return self._rope.prefix_view(self.current_seq)
        return self._rope_view[:self.current_seq]

    def latent_numpy(self) -> Any:
        """Zero-copy numpy view of the populated latent window (no download)."""
        return self._latent_view[:self.current_seq]

    def rope_numpy(self) -> Any:
        return self._rope_view[:self.current_seq]

    def free(self) -> None:
        for t in (self._latent, self._rope):
            if t is not None:
                t.free()
        self._latent = self._rope = None
        self._resident = False

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"ResidentLatentKVCache(latent_dim={self.latent_dim}, "
                f"rope_dim={self.rope_dim}, current_seq={self.current_seq}/"
                f"{self.max_seq}, resident={self._resident})")


class ResidentBlockPagedKVCacheError(RuntimeError):
    """Raised on block-pool exhaustion or invalid sequence operations."""


class _SeqState:
    __slots__ = ("block_table", "length", "table_tensor", "table_view")

    def __init__(self, table_capacity: int, resident: bool) -> None:
        self.block_table: list[int] = []
        self.length: int = 0
        self.table_tensor = (
            R.DeviceTensor.empty((table_capacity,), np.int32) if resident else None
        )
        self.table_view = (
            self.table_tensor.numpy() if self.table_tensor is not None
            else np.full((table_capacity,), -1, dtype=np.int32)
        )
        self.table_view[...] = -1


class ResidentBlockPagedKVCache:
    """Device-resident, multi-sequence block-paged latent KV cache (vLLM-style,
    on-GPU gather).

    The physical block pool — latent ``[num_blocks, block_size, latent_dim]`` and
    rope ``[num_blocks, block_size, rope_dim]`` — is a **resident device buffer**.
    Each sequence owns a block table; ``append`` writes the new token in place
    (no upload); ``gather_latent`` / ``gather_rope`` assemble a sequence's
    (possibly non-contiguous) window into a contiguous resident ``DeviceTensor``
    **on-GPU** via the block-table gather kernel — no host round-trip.
    """

    def __init__(self, *, latent_dim: int, rope_dim: int, num_blocks: int,
                 block_size: int) -> None:
        if min(latent_dim, rope_dim, num_blocks, block_size) <= 0:
            raise ValueError("all dims must be positive")
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        dt = R.DeviceTensor
        self._lat_pool = dt.empty((self.num_blocks, self.block_size, self.latent_dim), np.float32)
        self._rope_pool = dt.empty((self.num_blocks, self.block_size, self.rope_dim), np.float32)
        self._resident = self._lat_pool is not None and self._rope_pool is not None
        if self._lat_pool is not None and self._rope_pool is not None:
            self._lat_view = self._lat_pool.numpy()
            self._rope_view = self._rope_pool.numpy()
        else:
            self._lat_view = np.zeros((self.num_blocks, self.block_size, self.latent_dim), np.float32)
            self._rope_view = np.zeros((self.num_blocks, self.block_size, self.rope_dim), np.float32)
        self._free: list[int] = list(range(self.num_blocks - 1, -1, -1))
        self._seqs: dict[Any, _SeqState] = {}
        self.last_gather_execution = "uninitialized"
        self.last_gather_telemetry: dict[str, Any] | None = None
        self.last_attention_execution = "uninitialized"
        self.last_attention_telemetry: dict[str, Any] | None = None
        self._attention_calls = 0

    # ------------------------------------------------------------------
    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def block_table(self, seq_id: Any) -> list[int]:
        return list(self._seq(seq_id).block_table)

    def sequence_length(self, seq_id: Any) -> int:
        return self._seq(seq_id).length

    def _seq(self, seq_id: Any) -> _SeqState:
        st = self._seqs.get(seq_id)
        if st is None:
            raise ResidentBlockPagedKVCacheError(f"unknown sequence {seq_id!r}")
        return st

    # ------------------------------------------------------------------
    def add_sequence(self, seq_id: Any) -> None:
        if seq_id in self._seqs:
            raise ResidentBlockPagedKVCacheError(f"sequence {seq_id!r} exists")
        self._seqs[seq_id] = _SeqState(self.num_blocks, self._resident)

    def free_sequence(self, seq_id: Any) -> None:
        st = self._seqs.pop(seq_id, None)
        if st is None:
            raise ResidentBlockPagedKVCacheError(f"unknown sequence {seq_id!r}")
        for blk in st.block_table:
            self._lat_view[blk] = 0
            self._rope_view[blk] = 0
            self._free.append(blk)
        if st.table_tensor is not None:
            st.table_tensor.free()
            st.table_tensor = None

    def append(self, seq_id: Any, c_kv: Any, k_rope: Any) -> None:
        """Append tokens to ``seq_id``, allocating blocks on demand and writing
        them in place into the resident pool (no upload)."""
        st = self._seq(seq_id)
        c = np.asarray(c_kv, np.float32)
        r = np.asarray(k_rope, np.float32)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        if c.shape[1] != self.latent_dim or r.shape[1] != self.rope_dim:
            raise ValueError("c_kv / k_rope last-dim mismatch")
        if c.shape[0] != r.shape[0]:
            raise ValueError("c_kv and k_rope token-count mismatch")
        n = c.shape[0]
        needed = (st.length + n + self.block_size - 1) // self.block_size
        additional = needed - len(st.block_table)
        if additional > len(self._free):
            # Reservation is transactional: a failed multi-block append must
            # not consume a prefix of the remaining pool.
            raise ResidentBlockPagedKVCacheError("block pool exhausted")
        while len(st.block_table) < needed:
            block = self._free.pop()
            st.table_view[len(st.block_table)] = block
            st.block_table.append(block)
        for t in range(n):
            logical = st.length + t
            blk = st.block_table[logical // self.block_size]
            off = logical % self.block_size
            self._lat_view[blk, off] = c[t]
            self._rope_view[blk, off] = r[t]
        st.length += n

    # ------------------------------------------------------------------
    def _gather(self, st: _SeqState, pool: Any, view: Any, dim: int) -> Any:
        """Return a resident DeviceTensor window [length, dim] (or numpy when
        not resident)."""
        self.last_gather_execution = "reference_cpu"
        self.last_gather_telemetry = None
        n_blk = len(st.block_table)
        if n_blk == 0:
            raise ResidentBlockPagedKVCacheError("empty sequence")
        bt = (R.DeviceTensor.from_numpy(np.asarray(st.block_table, np.int32))
              if self._resident else None)
        if self._resident and pool is not None and bt is not None:
            win = R._apple_gpu_gather_blocks_device(
                pool, bt, self.num_blocks, n_blk, self.block_size, dim)
            bt.free()
            if win is not None:
                from .._apple_gpu_dispatch import read_dispatch_telemetry
                telemetry = read_dispatch_telemetry()
                pipeline = telemetry.get("resources")
                telemetry["resources"] = {
                    "api": "MPSGraph.gatherWithUpdatesTensor",
                    "pipeline_limits": pipeline,
                    "pipeline_limits_unavailable_reason": (
                        None if pipeline is not None
                        else "framework_pipeline_not_publicly_exposed"),
                }
                self.last_gather_execution = "native_gpu"
                self.last_gather_telemetry = telemetry
                # [n_blk, block_size, dim] -> [n_blk*block_size, dim] -> [length, dim]
                flat = win.reshape_view(n_blk * self.block_size, dim)
                out = flat.prefix_view(st.length)
                # keep the gathered buffer owner alive via the returned view's handle;
                # caller frees `win` is awkward — instead copy ownership semantics:
                return _GatheredWindow(win, out)
        # numpy fallback gather
        rows = np.empty((st.length, dim), np.float32)
        for i in range(st.length):
            blk = st.block_table[i // self.block_size]
            off = i % self.block_size
            rows[i] = view[blk, off]
        return rows

    def gather_latent(self, seq_id: Any) -> Any:
        return self._gather(self._seq(seq_id), self._lat_pool, self._lat_view,
                            self.latent_dim)

    def gather_rope(self, seq_id: Any) -> Any:
        return self._gather(self._seq(seq_id), self._rope_pool, self._rope_view,
                            self.rope_dim)

    def attention(
        self,
        seq_id: Any,
        q_rope: Any,
        *,
        scale: float | None = None,
        causal: bool = True,
        causal_offset: int | None = None,
        window: int | None = None,
        route: str = "auto",
    ) -> Any:
        """Attend directly over the resident physical page pool.

        ``q_rope`` is ``[Q, rope_dim]``. Scores are formed against the paged
        ``k_rope`` pool and weight the paged ``c_kv`` values, producing
        ``[Q, latent_dim]``. The native candidate follows the resident int32
        page table inside one MSL dispatch; it never gathers/stages a dense KV
        window. ``causal_offset`` is the absolute key position aligned with
        query row zero (default: right-aligned ``length - Q``). ``window`` is
        the inclusive number of visible keys and must be positive when set.
        """
        st = self._seq(seq_id)
        if st.length == 0:
            raise ResidentBlockPagedKVCacheError("empty sequence")
        q = np.asarray(q_rope, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[1] != self.rope_dim:
            raise ValueError(
                f"q_rope must be [Q,{self.rope_dim}]; got {q.shape}")
        if window is not None and int(window) <= 0:
            raise ValueError("window must be positive when provided")
        q_len = int(q.shape[0])
        if route == "auto":
            from ..compiler.apple_route_selector import production_route_for
            route = production_route_for(
                op="resident_paged_kv",
                shape=f"{st.length}x{self.latent_dim}x{self.rope_dim}x{q_len}",
                dtype="f32", incumbent_route="staged")
        if route not in {"direct", "staged"}:
            raise ValueError("route must be 'auto', 'direct', or 'staged'")
        offset = (
            max(st.length - q_len, 0) if causal_offset is None
            else int(causal_offset)
        )
        if causal and offset < 0:
            raise ValueError("causal_offset must be non-negative")
        factor = float(scale) if scale is not None else self.rope_dim ** -0.5
        self.last_attention_execution = "reference_cpu"
        self.last_attention_telemetry = None
        self._attention_calls += 1

        if (route == "direct" and self._resident and R.DeviceTensor.is_metal()
                and self._lat_pool is not None
                and self._rope_pool is not None and st.table_tensor is not None):
            dq = R.DeviceTensor.from_numpy(np.ascontiguousarray(q))
            if dq is not None:
                out = R._apple_gpu_paged_latent_attention_device(
                    dq, self._lat_pool, self._rope_pool, st.table_tensor,
                    num_blocks=self.num_blocks,
                    n_blocks=len(st.block_table), block_size=self.block_size,
                    logical_length=st.length, latent_dim=self.latent_dim,
                    rope_dim=self.rope_dim, causal_offset=(offset if causal else -1),
                    window=(int(window) if window is not None else 0),
                    scale=factor,
                )
                dq.free()
                if out is not None:
                    from .._apple_gpu_dispatch import read_dispatch_telemetry
                    self.last_attention_execution = "native_gpu"
                    self.last_attention_telemetry = {
                        **read_dispatch_telemetry(),
                        "device_time_scope": "single_direct_page_table_attention",
                        "subdispatches": 1,
                    }
                    return out

        if route == "staged" and self._resident and R.DeviceTensor.is_metal():
            latent = self.gather_latent(seq_id)
            latent_telemetry = self.last_gather_telemetry
            rope = self.gather_rope(seq_id)
            rope_telemetry = self.last_gather_telemetry
            if hasattr(latent, "tensor") and hasattr(rope, "tensor"):
                dq = R.DeviceTensor.from_numpy(np.ascontiguousarray(q))
                out = (R._apple_gpu_dense_latent_attention_device(
                    dq, latent.tensor, rope.tensor, logical_length=st.length,
                    latent_dim=self.latent_dim, rope_dim=self.rope_dim,
                    causal_offset=(offset if causal else -1),
                    window=(int(window) if window is not None else 0),
                    scale=factor) if dq is not None else None)
                if dq is not None:
                    dq.free()
                latent.free(); rope.free()
                if out is not None:
                    from .._apple_gpu_dispatch import read_dispatch_telemetry
                    self.last_attention_execution = "native_gpu"
                    dense_telemetry = read_dispatch_telemetry()
                    parts = [latent_telemetry, rope_telemetry, dense_telemetry]
                    device_times: list[int] = []
                    for part in parts:
                        if isinstance(part, dict):
                            value = part.get("device_time_ns")
                            if isinstance(value, int):
                                device_times.append(value)
                    self.last_attention_telemetry = {
                        **dense_telemetry,
                        "device_time_ns": (
                            sum(device_times)
                            if len(device_times) == 3
                            else None),
                        "device_time_scope": "two_gathers_plus_dense_attention",
                        "subdispatches": 3,
                        "subdispatch_resources": [
                            p.get("resources") if isinstance(p, dict) else None
                            for p in parts
                        ],
                    }
                    return out

        # Shared numerical oracle. It deliberately follows the non-identity
        # page table rather than calling gather(), so direct and native paths
        # prove the same physical-page semantics.
        result = np.zeros((q_len, self.latent_dim), dtype=np.float32)
        for qi in range(q_len):
            limit = min(offset + qi, st.length - 1) if causal else st.length - 1
            first = max(0, limit - int(window) + 1) if window is not None else 0
            positions = range(first, limit + 1)
            scores = []
            values = []
            for pos in positions:
                block = st.block_table[pos // self.block_size]
                slot = pos % self.block_size
                scores.append(float(np.dot(q[qi], self._rope_view[block, slot])) * factor)
                values.append(self._lat_view[block, slot])
            logits = np.asarray(scores, dtype=np.float64)
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            result[qi] = weights @ np.asarray(values, dtype=np.float64)
        return result

    def lifecycle_telemetry(self) -> dict[str, int]:
        """Allocation/accounting snapshot used by exhaustion and leak gates."""
        return {
            "pool_blocks": self.num_blocks,
            "used_blocks": self.num_used_blocks,
            "free_blocks": self.num_free_blocks,
            "live_sequences": len(self._seqs),
            "resident_page_tables": sum(
                st.table_tensor is not None for st in self._seqs.values()),
            "attention_calls": self._attention_calls,
        }

    def free(self) -> None:
        for st in self._seqs.values():
            if st.table_tensor is not None:
                st.table_tensor.free()
                st.table_tensor = None
        self._seqs.clear()
        self._free = list(range(self.num_blocks - 1, -1, -1))
        for t in (self._lat_pool, self._rope_pool):
            if t is not None:
                t.free()
        self._lat_pool = self._rope_pool = None
        self._resident = False

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass


class _GatheredWindow:
    """Holds the gathered block buffer (owner) + a prefix view of the populated
    rows. ``.tensor`` is the view DeviceTensor; ``.numpy()`` materializes; freed
    together."""

    __slots__ = ("_owner", "tensor")

    def __init__(self, owner: Any, view: Any) -> None:
        self._owner = owner   # owns the gathered [n_blk*block_size, dim] buffer
        self.tensor = view    # non-owning prefix view [length, dim]

    def numpy(self) -> Any:
        return self.tensor.numpy()

    def free(self) -> None:
        if self._owner is not None:
            self._owner.free()
            self._owner = None
