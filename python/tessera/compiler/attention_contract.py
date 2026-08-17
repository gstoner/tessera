"""Canonical streaming-attention and deterministic backward contracts.

This module is target-neutral.  It defines the numerical recurrence used by
fixture-backed compiler tests and the launch-workspace ownership that physical
backends must consume.  CUDA, ROCm, and Metal may choose different schedules,
but they must not redefine head distribution, split ownership, or reduction
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_DROPOUT_COUNTER_MULTIPLIER = np.uint64(1664525)
_DROPOUT_COUNTER_INCREMENT = np.uint64(1013904223)
_UINT32_RANGE = 4294967296.0


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class AttentionWorkspaceSlice:
    """One explicitly owned region of launch-lifetime attention workspace."""

    name: str
    offset: int
    bytes: int
    initialization: str
    producer: str
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.offset < 0 or self.bytes <= 0:
            raise ValueError("attention workspace slices require a positive extent")
        if self.initialization not in {"undefined", "zero"}:
            raise ValueError("attention workspace initialization must be undefined or zero")
        if not self.producer or not self.consumers:
            raise ValueError("attention workspace ownership requires producer and consumers")


@dataclass(frozen=True)
class AttentionBackwardWorkspacePlan:
    """Canonical deterministic split/reduced backward launch contract."""

    bytes: int
    alignment: int
    split_count: int
    reduction_order: tuple[int, ...]
    slices: tuple[AttentionWorkspaceSlice, ...]

    def __post_init__(self) -> None:
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("attention workspace alignment must be a power of two")
        if self.split_count < 2:
            raise ValueError("split/reduced attention requires at least two splits")
        if self.reduction_order != tuple(range(self.split_count)):
            raise ValueError("attention partials must reduce in ascending split order")
        end = 0
        for item in self.slices:
            if item.offset % self.alignment:
                raise ValueError("attention workspace slices must be aligned")
            if item.offset < end:
                raise ValueError("attention workspace slices overlap")
            end = item.offset + item.bytes
        if _align(end, self.alignment) != self.bytes:
            raise ValueError("attention workspace allocation does not cover its slices")


def plan_attention_backward_workspace(
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_rows: int,
    key_rows: int,
    head_dim: int,
    value_dim: int,
    split_count: int = 2,
    alignment: int = 256,
) -> AttentionBackwardWorkspacePlan:
    """Plan target-neutral backward workspace with fixed reduction ordering.

    Split zero writes directly to final dK/dV.  Each additional split owns one
    workspace dK/dV pair, so the extra footprint is exactly
    ``(split_count - 1) * (dK + dV)``.  This preserves the proven two-split
    one-extra-pair contract while making larger deterministic split counts
    representable.
    """

    dims = (
        batch,
        query_heads,
        kv_heads,
        query_rows,
        key_rows,
        head_dim,
        value_dim,
    )
    if any(dim <= 0 for dim in dims):
        raise ValueError("attention workspace dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if split_count < 2:
        raise ValueError("split_count must be at least two")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")

    output_elements = batch * query_heads * query_rows * value_dim
    row_elements = batch * query_heads * query_rows
    dk_elements = batch * kv_heads * key_rows * head_dim
    dv_elements = batch * kv_heads * key_rows * value_dim
    extra_partials = split_count - 1
    specs = (
        (
            "forward_o",
            output_elements * 4,
            "undefined",
            "forward_recompute",
            ("row_prepass", "dq"),
        ),
        (
            "row_lse",
            row_elements * 4,
            "undefined",
            "forward_recompute",
            ("row_prepass", "dkdv_split", "dq"),
        ),
        (
            "row_delta",
            row_elements * 4,
            "undefined",
            "row_prepass",
            ("dkdv_split", "dq"),
        ),
        (
            "partial_dk",
            extra_partials * dk_elements * 4,
            "zero",
            "dkdv_split",
            ("dkdv_fixed_order_reduce",),
        ),
        (
            "partial_dv",
            extra_partials * dv_elements * 4,
            "zero",
            "dkdv_split",
            ("dkdv_fixed_order_reduce",),
        ),
    )
    slices: list[AttentionWorkspaceSlice] = []
    offset = 0
    for name, byte_count, initialization, producer, consumers in specs:
        offset = _align(offset, alignment)
        slices.append(
            AttentionWorkspaceSlice(
                name,
                offset,
                byte_count,
                initialization,
                producer,
                consumers,
            )
        )
        offset += byte_count
    return AttentionBackwardWorkspacePlan(
        bytes=_align(offset, alignment),
        alignment=alignment,
        split_count=split_count,
        reduction_order=tuple(range(split_count)),
        slices=tuple(slices),
    )


def attention_dropout_replay(
    *,
    batch: int,
    query_heads: int,
    query_rows: int,
    key_rows: int,
    dropout_p: float,
    seed: int,
) -> np.ndarray:
    """Return the canonical deterministic attention-dropout replay scale.

    The element counter is the row-major logical probability identity
    ``(((batch * Hq + head) * Sq + query) * Sk + key)``.  It is independent of
    tile shape, split count, and physical scheduling, so forward and every
    backward stage replay the same mask without storing it.
    """

    dimensions = (batch, query_heads, query_rows, key_rows)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("attention dropout dimensions must be positive")
    probability = float(dropout_p)
    if not np.isfinite(probability) or probability < 0.0 or probability >= 1.0:
        raise ValueError("attention dropout probability must satisfy 0 <= p < 1")
    if probability == 0.0:
        return np.ones(dimensions, dtype=np.float32)

    bh = np.arange(batch * query_heads, dtype=np.uint64)[:, None, None]
    query = np.arange(query_rows, dtype=np.uint64)[None, :, None]
    key = np.arange(key_rows, dtype=np.uint64)[None, None, :]
    counter = (bh * np.uint64(query_rows) + query) * np.uint64(key_rows) + key
    hashed = (
        counter * _DROPOUT_COUNTER_MULTIPLIER
        + np.uint64(seed) + _DROPOUT_COUNTER_INCREMENT
    ) & np.uint64(0xFFFFFFFF)
    threshold = np.uint64(int(probability * _UINT32_RANGE))
    replay = (hashed >= threshold).reshape(dimensions).astype(np.float32)
    replay *= np.float32(1.0 / (1.0 - probability))
    return replay


def _masked_scores(
    q: np.ndarray,
    key: np.ndarray,
    *,
    scale: float,
    bias: np.ndarray | None,
    causal: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    key_offset: int = 0,
    query_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = scale * (q @ key.T)
    if bias is not None:
        raw = raw + bias
    if softcap > 0.0:
        tanh = np.tanh(raw / softcap)
        scores = softcap * tanh
        derivative = 1.0 - tanh * tanh
    else:
        scores = raw
        derivative = np.ones_like(raw, dtype=np.float32)
    qpos = query_offset + np.arange(q.shape[0], dtype=np.int64)[:, None]
    kpos = key_offset + np.arange(key.shape[0], dtype=np.int64)[None, :]
    valid = np.ones(scores.shape, dtype=np.bool_)
    if causal:
        valid &= kpos <= qpos
    if window_left >= 0:
        valid &= kpos >= qpos - window_left
    if window_right >= 0:
        valid &= kpos <= qpos + window_right
    return np.where(valid, scores, -np.inf), valid, derivative


def reference_streaming_attention(
    q: Any,
    key: Any,
    value: Any,
    *,
    block_size: int,
    scale: float | None = None,
    bias: Any | None = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> np.ndarray:
    """Execute the canonical rank-4 KV-block online-softmax recurrence."""

    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(key, dtype=np.float32)
    vf = np.asarray(value, dtype=np.float32)
    if qf.ndim != 4 or kf.ndim != 4 or vf.ndim != 4:
        raise ValueError("streaming attention requires rank-4 Q/K/V")
    batch, query_heads, query_rows, head_dim = qf.shape
    if kf.shape[:1] != (batch,) or vf.shape[:1] != (batch,):
        raise ValueError("Q/K/V batch dimensions must match")
    kv_heads, key_rows = kf.shape[1:3]
    if (
        query_heads % kv_heads
        or kf.shape[3] != head_dim
        or vf.shape[:3] != (batch, kv_heads, key_rows)
        or block_size <= 0
    ):
        raise ValueError("invalid GQA or KV streaming shape")
    value_dim = vf.shape[3]
    scale_value = head_dim**-0.5 if scale is None else float(scale)
    bias_array = None if bias is None else np.asarray(bias, dtype=np.float32)
    if bias_array is not None and bias_array.shape != (
        batch,
        query_heads,
        query_rows,
        key_rows,
    ):
        raise ValueError("attention bias must have shape [B,Hq,Sq,Sk]")
    dropout_replay = attention_dropout_replay(
        batch=batch,
        query_heads=query_heads,
        query_rows=query_rows,
        key_rows=key_rows,
        dropout_p=dropout_p,
        seed=dropout_seed,
    )

    output = np.zeros(
        (batch, query_heads, query_rows, value_dim), dtype=np.float32
    )
    group = query_heads // kv_heads
    for batch_index in range(batch):
        for query_head in range(query_heads):
            kv_head = query_head // group
            running_max = np.full((query_rows,), -np.inf, dtype=np.float32)
            running_sum = np.zeros((query_rows,), dtype=np.float32)
            accumulator = np.zeros((query_rows, value_dim), dtype=np.float32)
            for start in range(0, key_rows, block_size):
                stop = min(start + block_size, key_rows)
                block_bias = (
                    None
                    if bias_array is None
                    else bias_array[batch_index, query_head, :, start:stop]
                )
                scores, valid, _ = _masked_scores(
                    qf[batch_index, query_head],
                    kf[batch_index, kv_head, start:stop],
                    scale=scale_value,
                    bias=block_bias,
                    causal=causal,
                    window_left=window_left,
                    window_right=window_right,
                    softcap=softcap,
                    key_offset=start,
                    query_offset=max(key_rows - query_rows, 0),
                )
                block_max = np.max(scores, axis=1)
                new_max = np.maximum(running_max, block_max)
                # A leading KV block can be wholly outside a ragged causal
                # window. Avoid forming -inf - -inf while preserving the exact
                # zero contribution required by online softmax.
                old_delta = np.full_like(running_max, -np.inf)
                finite_old = np.isfinite(running_max) & np.isfinite(new_max)
                np.subtract(
                    running_max, new_max, out=old_delta, where=finite_old
                )
                old_scale = np.exp(old_delta)
                score_delta = np.full_like(scores, -np.inf)
                finite_rows = np.isfinite(new_max)[:, None]
                np.subtract(
                    scores,
                    new_max[:, None],
                    out=score_delta,
                    where=valid & finite_rows,
                )
                weights = np.exp(score_delta)
                output_weights = weights * dropout_replay[
                    batch_index, query_head, :, start:stop
                ]
                accumulator = (
                    accumulator * old_scale[:, None]
                    + output_weights @ vf[batch_index, kv_head, start:stop]
                )
                running_sum = running_sum * old_scale + np.sum(weights, axis=1)
                running_max = new_max
            output[batch_index, query_head] = (
                accumulator / running_sum[:, None]
            )
    return output


def reference_attention_backward_split_reduced(
    do: Any,
    q: Any,
    key: Any,
    value: Any,
    *,
    split_count: int = 2,
    scale: float | None = None,
    bias: Any | None = None,
    causal: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference the canonical dQ plus split/fixed-order dK/dV algorithm."""

    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(key, dtype=np.float32)
    vf = np.asarray(value, dtype=np.float32)
    dof = np.asarray(do, dtype=np.float32)
    if qf.ndim != 4 or kf.ndim != 4 or vf.ndim != 4 or dof.ndim != 4:
        raise ValueError("attention backward requires rank-4 tensors")
    batch, query_heads, query_rows, head_dim = qf.shape
    kv_heads, key_rows = kf.shape[1:3]
    if (
        dof.shape[:3] != (batch, query_heads, query_rows)
        or dof.shape[3] != vf.shape[3]
        or query_heads % kv_heads
        or split_count < 2
    ):
        raise ValueError("invalid attention backward shape or split count")
    scale_value = head_dim**-0.5 if scale is None else float(scale)
    bias_array = None if bias is None else np.asarray(bias, dtype=np.float32)
    if bias_array is not None and bias_array.shape != (
        batch,
        query_heads,
        query_rows,
        key_rows,
    ):
        raise ValueError("attention bias must have shape [B,Hq,Sq,Sk]")
    dropout_replay = attention_dropout_replay(
        batch=batch,
        query_heads=query_heads,
        query_rows=query_rows,
        key_rows=key_rows,
        dropout_p=dropout_p,
        seed=dropout_seed,
    )
    dq = np.zeros_like(qf, dtype=np.float32)
    partial_dk = np.zeros((split_count, *kf.shape), dtype=np.float32)
    partial_dv = np.zeros((split_count, *vf.shape), dtype=np.float32)
    group = query_heads // kv_heads
    for batch_index in range(batch):
        for query_head in range(query_heads):
            kv_head = query_head // group
            head_bias = (
                None if bias_array is None else bias_array[batch_index, query_head]
            )
            scores, valid, score_derivative = _masked_scores(
                qf[batch_index, query_head],
                kf[batch_index, kv_head],
                scale=scale_value,
                bias=head_bias,
                causal=causal,
                window_left=window_left,
                window_right=window_right,
                softcap=softcap,
                query_offset=max(key_rows - query_rows, 0),
            )
            row_max = np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores - row_max)
            weights = np.where(valid, weights, 0.0)
            weights /= np.sum(weights, axis=1, keepdims=True)
            replay = dropout_replay[batch_index, query_head]
            output_weights = weights * replay
            output = output_weights @ vf[batch_index, kv_head]
            dp = (
                dof[batch_index, query_head] @ vf[batch_index, kv_head].T
            ) * replay
            delta = np.sum(
                output * dof[batch_index, query_head], axis=1, keepdims=True
            )
            ds = weights * (dp - delta) * score_derivative
            dq[batch_index, query_head] = (
                scale_value * (ds @ kf[batch_index, kv_head])
            )
            for split in range(split_count):
                rows = np.arange(query_rows) % split_count == split
                if not np.any(rows):
                    continue
                partial_dk[split, batch_index, kv_head] += scale_value * (
                    ds[rows].T @ qf[batch_index, query_head, rows]
                )
                partial_dv[split, batch_index, kv_head] += (
                    output_weights[rows].T
                    @ dof[batch_index, query_head, rows]
                )
    dk = np.zeros_like(kf, dtype=np.float32)
    dv = np.zeros_like(vf, dtype=np.float32)
    for split in range(split_count):
        dk += partial_dk[split]
        dv += partial_dv[split]
    return dq, dk, dv


__all__ = [
    "AttentionBackwardWorkspacePlan",
    "AttentionWorkspaceSlice",
    "attention_dropout_replay",
    "plan_attention_backward_workspace",
    "reference_attention_backward_split_reduced",
    "reference_streaming_attention",
]
