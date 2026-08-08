"""Content-addressed chunked STFT policy and state transition.

This is the target-neutral streaming contract.  It deliberately supports the
causal/non-centred policy first: centred reflection requires future samples and
must not be silently approximated by a physical package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .scheduled_matmul import digest_text


@dataclass(frozen=True)
class StreamingSTFTPolicy:
    axis: int
    n_fft: int
    window_length: int
    hop: int
    onesided: bool = True
    center: bool = False
    pad_mode: str = "constant"
    max_chunk_samples: int = 0

    def __post_init__(self) -> None:
        if self.n_fft < self.window_length or self.window_length <= 0:
            raise ValueError("streaming STFT requires n_fft >= window_length > 0")
        if self.hop <= 0 or self.max_chunk_samples < 0:
            raise ValueError("streaming STFT requires a positive hop and nonnegative bound")
        if self.pad_mode not in {"constant", "reflect"}:
            raise ValueError("streaming STFT pad_mode must be constant or reflect")
        if self.center:
            raise ValueError(
                "streaming centred STFT fails closed until lookahead lineage is explicit"
            )

    @property
    def digest(self) -> str:
        return digest_text(
            "schema=tessera.streaming_stft_policy.v1;"
            f"axis={self.axis};n_fft={self.n_fft};window={self.window_length};"
            f"hop={self.hop};onesided={int(self.onesided)};"
            f"center={int(self.center)};pad_mode={self.pad_mode};"
            f"max_chunk_samples={self.max_chunk_samples}"
        )


@dataclass(frozen=True)
class StreamingSTFTState:
    policy_digest: str
    tail: np.ndarray
    samples_consumed: int
    frames_emitted: int


def stream_stft_chunk(
    chunk: Any,
    window: Any,
    policy: StreamingSTFTPolicy,
    state: StreamingSTFTState | None = None,
) -> tuple[np.ndarray, StreamingSTFTState]:
    """Consume one chunk and retain exactly the overlap needed by the next."""
    values = np.asarray(chunk)
    win = np.asarray(window)
    if win.ndim != 1 or win.shape[0] != policy.window_length:
        raise ValueError("streaming STFT window does not match the policy")
    axis = policy.axis if policy.axis >= 0 else values.ndim + policy.axis
    if axis < 0 or axis >= values.ndim:
        raise ValueError("streaming STFT axis is out of range")
    if policy.max_chunk_samples and values.shape[axis] > policy.max_chunk_samples:
        raise ValueError("streaming STFT chunk exceeds its bounded policy")
    moved = np.moveaxis(values, axis, -1)
    consumed = 0
    emitted = 0
    if state is not None:
        if state.policy_digest != policy.digest:
            raise ValueError("streaming STFT state belongs to a different policy")
        if state.tail.shape[:-1] != moved.shape[:-1]:
            raise ValueError("streaming STFT batch shape changed across chunks")
        moved = np.concatenate([state.tail, moved], axis=-1)
        consumed = state.samples_consumed
        emitted = state.frames_emitted
    padded_window = np.zeros(policy.n_fft, dtype=win.dtype)
    offset = (policy.n_fft - policy.window_length) // 2
    padded_window[offset : offset + policy.window_length] = win
    frame_count = max(0, (moved.shape[-1] - policy.n_fft) // policy.hop + 1)
    bins = policy.n_fft // 2 + 1 if policy.onesided else policy.n_fft
    if frame_count:
        frames = np.stack(
            [
                moved[..., index * policy.hop : index * policy.hop + policy.n_fft]
                * padded_window
                for index in range(frame_count)
            ],
            axis=-2,
        )
        transform = np.fft.rfft if policy.onesided else np.fft.fft
        spectra = transform(frames, axis=-1)
    else:
        spectra = np.empty(moved.shape[:-1] + (0, bins), dtype=np.complex64)
    tail_start = frame_count * policy.hop
    tail = np.ascontiguousarray(moved[..., tail_start:])
    next_state = StreamingSTFTState(
        policy_digest=policy.digest,
        tail=tail,
        samples_consumed=consumed + values.shape[axis],
        frames_emitted=emitted + frame_count,
    )
    lead_rank = values.ndim - 1
    order = (
        list(range(axis))
        + [lead_rank, lead_rank + 1]
        + list(range(axis, lead_rank))
    )
    return np.transpose(spectra.astype(np.complex64), order), next_state
