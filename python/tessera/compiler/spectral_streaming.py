"""Content-addressed chunked STFT policy and state transition.

This is the target-neutral streaming contract.  It deliberately supports the
causal/non-centred policy first: centred reflection requires future samples and
must not be silently approximated by a physical package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
        if self.hop <= 0 or self.hop > self.n_fft or self.max_chunk_samples < 0:
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
    artifact_digest: str
    target: str
    architecture: str
    tail: np.ndarray
    samples_consumed: int
    frames_emitted: int
    parent_state_digest: str
    window_digest: str
    tail_digest: str
    state_digest: str

    @property
    def execution_certificate(self) -> dict[str, Any]:
        return {
            "schema": "tessera.streaming_stft_execution.v1",
            "origin": "runtime",
            "artifact_digest": self.artifact_digest,
            "state_digest": self.state_digest,
            "target": self.target,
            "architecture_identity": self.architecture,
        }


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    header = (
        f"dtype={array.dtype.str};shape="
        + "x".join(str(int(dim)) for dim in array.shape)
        + ";"
    ).encode()
    hasher = hashlib.sha256(header)
    hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return hasher.hexdigest()


def _state_digest(
    *, policy_digest: str, artifact_digest: str, target: str, architecture: str,
    parent_state_digest: str,
    window_digest: str, tail_digest: str, samples_consumed: int,
    frames_emitted: int,
) -> str:
    return digest_text(
        "schema=tessera.streaming_stft_state.v2;"
        f"policy={policy_digest};artifact={artifact_digest};target={target};"
        f"architecture={architecture};"
        f"parent={parent_state_digest};"
        f"window={window_digest};tail={tail_digest};"
        f"samples={samples_consumed};frames={frames_emitted}"
    )


def validate_streaming_stft_state(state: StreamingSTFTState) -> None:
    if (len(state.policy_digest) != 64 or len(state.artifact_digest) != 64 or
            len(state.parent_state_digest) != 64):
        raise ValueError("streaming STFT state lineage identity is malformed")
    tail_digest = _array_digest(state.tail)
    if tail_digest != state.tail_digest:
        raise ValueError("streaming STFT state tail was altered")
    expected = _state_digest(
        policy_digest=state.policy_digest,
        artifact_digest=state.artifact_digest,
        target=state.target,
        architecture=state.architecture,
        parent_state_digest=state.parent_state_digest,
        window_digest=state.window_digest,
        tail_digest=tail_digest,
        samples_consumed=state.samples_consumed,
        frames_emitted=state.frames_emitted,
    )
    if expected != state.state_digest:
        raise ValueError("streaming STFT state lineage digest mismatch")


def _stream_stft_chunk_x86(
    values: np.ndarray,
    window: np.ndarray,
    policy: StreamingSTFTPolicy,
    axis: int,
    tail: np.ndarray | None,
    artifact_digest: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    import ctypes
    from tessera import runtime

    if str(values.dtype) not in {"float16", "bfloat16", "float32"} or (
        window.dtype != values.dtype
    ):
        raise ValueError("x86 streaming STFT requires matching f16/bf16/f32 storage")
    lib = runtime._load_x86_elementwise()
    if lib is None:
        raise RuntimeError("x86 streaming STFT physical package is unavailable")

    def descriptor(array: np.ndarray) -> tuple[Any, Any]:
        if any(stride % array.itemsize for stride in array.strides):
            raise ValueError("streaming STFT strides must be element aligned")
        strides = tuple(int(stride // array.itemsize) for stride in array.strides)
        if any(stride == 0 and array.shape[dim] > 1
               for dim, stride in enumerate(strides)):
            raise ValueError("streaming STFT overlapping layouts are unsupported")
        return ((ctypes.c_int64 * array.ndim)(*map(int, array.shape)),
                (ctypes.c_int64 * array.ndim)(*strides))

    shape, strides = descriptor(values)
    window_shape, window_strides = descriptor(window)
    moved_shape = values.shape[:axis] + values.shape[axis + 1 :]
    tail_samples = 0 if tail is None else int(tail.shape[-1])
    if tail is not None and (tail.shape[:-1] != moved_shape or tail.dtype != values.dtype):
        raise ValueError("streaming STFT tail ABI does not match the chunk")
    combined = tail_samples + values.shape[axis]
    frames = max(0, (combined - policy.n_fft) // policy.hop + 1)
    bins = policy.n_fft // 2 + 1 if policy.onesided else policy.n_fft
    output_shape = (
        values.shape[:axis] + (frames, bins) + values.shape[axis + 1 :]
    )
    output = np.empty(output_shape, np.complex64)
    next_samples = combined - frames * policy.hop
    next_tail = np.empty(moved_shape + (next_samples,), dtype=values.dtype)
    empty_tail = np.empty((0,), dtype=values.dtype) if tail is None else tail
    storage = {"float32": 0, "float16": 1, "bfloat16": 2}[str(values.dtype)]
    rc = lib.tessera_x86_streaming_stft_broadcast_layout_storage(
        artifact_digest.encode(), values.ctypes.data_as(ctypes.c_void_p),
        empty_tail.ctypes.data_as(ctypes.c_void_p),
        window.ctypes.data_as(ctypes.c_void_p),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        next_tail.ctypes.data_as(ctypes.c_void_p),
        values.ndim, shape, strides, axis, tail_samples,
        window.ndim, window_shape, window_strides,
        policy.n_fft, policy.hop, frames, storage, ctypes.c_float(1.0),
        int(policy.onesided),
    )
    if rc:
        raise RuntimeError(f"x86 streaming STFT package failed rc={rc}")
    return output, next_tail, frames


def _stream_stft_chunk_rocm(
    values: np.ndarray, window: np.ndarray, policy: StreamingSTFTPolicy,
    axis: int, tail: np.ndarray | None, artifact_digest: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    import ctypes
    from tessera.compiler.emit import spectral_candidates

    if str(values.dtype) not in {"float16", "bfloat16", "float32"} or (
        window.dtype != values.dtype
    ):
        raise ValueError("gfx1151 streaming STFT requires matching f16/bf16/f32 storage")
    lib = spectral_candidates._amd_composite_lib()
    if lib is None or not hasattr(
        lib, "ts_streaming_stft_hostptr_broadcast_layout_storage_amd"
    ):
        raise RuntimeError("gfx1151 streaming STFT physical package is unavailable")

    def descriptor(array: np.ndarray) -> tuple[Any, Any]:
        if any(stride % array.itemsize for stride in array.strides):
            raise ValueError("streaming STFT strides must be element aligned")
        element_strides = tuple(
            int(stride // array.itemsize) for stride in array.strides
        )
        if any(stride == 0 and array.shape[dim] > 1
               for dim, stride in enumerate(element_strides)):
            raise ValueError("streaming STFT overlapping layouts are unsupported")
        return ((ctypes.c_int64 * array.ndim)(*map(int, array.shape)),
                (ctypes.c_int64 * array.ndim)(*element_strides))

    shape, strides = descriptor(values)
    window_shape, window_strides = descriptor(window)
    batch_shape = values.shape[:axis] + values.shape[axis + 1 :]
    tail_samples = 0 if tail is None else int(tail.shape[-1])
    if tail is not None and (tail.shape[:-1] != batch_shape or tail.dtype != values.dtype):
        raise ValueError("streaming STFT tail ABI does not match the chunk")
    combined = tail_samples + values.shape[axis]
    frames = max(0, (combined - policy.n_fft) // policy.hop + 1)
    bins = policy.n_fft // 2 + 1 if policy.onesided else policy.n_fft
    output = np.empty(
        values.shape[:axis] + (frames, bins) + values.shape[axis + 1 :],
        np.complex64,
    )
    next_tail = np.empty(
        batch_shape + (combined - frames * policy.hop,), dtype=values.dtype
    )
    empty_tail = np.empty((0,), dtype=values.dtype) if tail is None else tail
    storage = {"float32": 0, "float16": 1, "bfloat16": 2}[str(values.dtype)]
    rc = lib.ts_streaming_stft_hostptr_broadcast_layout_storage_amd(
        artifact_digest.encode(), spectral_candidates._cptr(values),
        spectral_candidates._cptr(empty_tail), spectral_candidates._cptr(window),
        spectral_candidates._cptr(output), spectral_candidates._cptr(next_tail),
        values.ndim, shape, strides, axis, tail_samples, window.ndim,
        window_shape, window_strides, policy.n_fft, policy.hop, frames,
        storage, ctypes.c_float(1.0), int(policy.onesided),
    )
    if rc:
        raise RuntimeError(f"gfx1151 streaming STFT package failed rc={rc}")
    return output, next_tail, frames


def _stream_stft_chunk_nvidia(
    values: np.ndarray, window: np.ndarray, policy: StreamingSTFTPolicy,
    axis: int, tail: np.ndarray | None, artifact_digest: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    import ctypes
    from tessera import runtime

    if values.dtype != np.float32 or window.dtype != np.float32:
        raise ValueError("SM120 streaming STFT v1 requires float32 storage")
    lib = runtime._load_nvidia_fft_runtime()
    if (lib is None or
            not hasattr(lib, "tessera_nvidia_streaming_stft_broadcast_layout_f32") or
            lib.tessera_nvidia_spectral_arch() != 120):
        raise RuntimeError("SM120 streaming STFT physical package is unavailable")

    def descriptor(array: np.ndarray) -> tuple[Any, Any]:
        if any(stride % array.itemsize for stride in array.strides):
            raise ValueError("streaming STFT strides must be element aligned")
        element_strides = tuple(
            int(stride // array.itemsize) for stride in array.strides
        )
        if any(stride == 0 and array.shape[dim] > 1
               for dim, stride in enumerate(element_strides)):
            raise ValueError("streaming STFT overlapping layouts are unsupported")
        return ((ctypes.c_int64 * array.ndim)(*map(int, array.shape)),
                (ctypes.c_int64 * array.ndim)(*element_strides))

    shape, strides = descriptor(values)
    window_shape, window_strides = descriptor(window)
    batch_shape = values.shape[:axis] + values.shape[axis + 1 :]
    tail_samples = 0 if tail is None else int(tail.shape[-1])
    if tail is not None and (tail.shape[:-1] != batch_shape or
                             tail.dtype != values.dtype):
        raise ValueError("streaming STFT tail ABI does not match the chunk")
    combined = tail_samples + values.shape[axis]
    frames = max(0, (combined - policy.n_fft) // policy.hop + 1)
    bins = policy.n_fft // 2 + 1 if policy.onesided else policy.n_fft
    output = np.empty(
        values.shape[:axis] + (frames, bins) + values.shape[axis + 1 :],
        np.complex64,
    )
    next_tail = np.empty(
        batch_shape + (combined - frames * policy.hop,), dtype=np.float32
    )
    empty_tail = np.empty((0,), dtype=np.float32) if tail is None else tail
    pointer = ctypes.POINTER(ctypes.c_float)
    rc = lib.tessera_nvidia_streaming_stft_broadcast_layout_f32(
        artifact_digest.encode("ascii"), values.ctypes.data_as(pointer),
        empty_tail.ctypes.data_as(pointer), window.ctypes.data_as(pointer),
        output.view(np.float32).ctypes.data_as(pointer),
        next_tail.ctypes.data_as(pointer), values.ndim, shape, strides, axis,
        tail_samples, window.ndim, window_shape, window_strides, policy.n_fft,
        policy.hop, frames, ctypes.c_float(1.0), int(policy.onesided),
    )
    if rc:
        raise RuntimeError(f"SM120 streaming STFT package failed rc={rc}")
    return output, next_tail, frames


def stream_stft_chunk(
    chunk: Any,
    window: Any,
    policy: StreamingSTFTPolicy,
    state: StreamingSTFTState | None = None,
    *,
    target: str = "reference",
) -> tuple[np.ndarray, StreamingSTFTState]:
    """Consume one chunk and retain exactly the overlap needed by the next."""
    values = np.asarray(chunk)
    win = np.asarray(window)
    if win.ndim < 1 or win.shape[-1] != policy.window_length:
        raise ValueError("streaming STFT window does not match the policy")
    axis = policy.axis if policy.axis >= 0 else values.ndim + policy.axis
    if axis < 0 or axis >= values.ndim:
        raise ValueError("streaming STFT axis is out of range")
    if policy.max_chunk_samples and values.shape[axis] > policy.max_chunk_samples:
        raise ValueError("streaming STFT chunk exceeds its bounded policy")
    moved = np.moveaxis(values, axis, -1)
    batch_shape = moved.shape[:-1]
    if win.ndim - 1 > len(batch_shape) or any(
        extent not in {1, batch_shape[len(batch_shape) - win.ndim + 1 + dim]}
        for dim, extent in enumerate(win.shape[:-1])
    ):
        raise ValueError("streaming STFT window batch dimensions do not broadcast")
    if target not in {"reference", "x86", "rocm", "nvidia_sm120"}:
        raise ValueError("streaming STFT target is unsupported")
    architecture = (
        "zen5-avx512" if target == "x86" else
        "gfx1151" if target == "rocm" else
        "sm_120" if target == "nvidia_sm120" else "reference"
    )
    artifact_digest = digest_text(
        "schema=tessera.streaming_stft_artifact.v1;"
        f"target={target};arch={architecture};"
        f"policy={policy.digest};window_broadcast=trailing_batch_broadcast_v1;"
        "state_abi=canonical_batch_tail_v1;accum=fp32"
    )
    window_digest = _array_digest(win)
    consumed = 0
    emitted = 0
    parent_state_digest = digest_text(
        "schema=tessera.streaming_stft_state_root.v1;"
        f"policy={policy.digest};artifact={artifact_digest};window={window_digest}"
    )
    if state is not None:
        validate_streaming_stft_state(state)
        if state.policy_digest != policy.digest:
            raise ValueError("streaming STFT state belongs to a different policy")
        if state.artifact_digest != artifact_digest:
            raise ValueError("streaming STFT state belongs to a different physical artifact")
        if state.window_digest != window_digest:
            raise ValueError("streaming STFT window changed across chunks")
        if state.tail.shape[:-1] != moved.shape[:-1]:
            raise ValueError("streaming STFT batch shape changed across chunks")
        consumed = state.samples_consumed
        emitted = state.frames_emitted
        parent_state_digest = state.state_digest
    prior_tail = None if state is None else state.tail
    if target == "x86":
        spectra, tail, frame_count = _stream_stft_chunk_x86(
            values, win, policy, axis, prior_tail, artifact_digest
        )
    elif target == "rocm":
        spectra, tail, frame_count = _stream_stft_chunk_rocm(
            values, win, policy, axis, prior_tail, artifact_digest
        )
    elif target == "nvidia_sm120":
        spectra, tail, frame_count = _stream_stft_chunk_nvidia(
            values, win, policy, axis, prior_tail, artifact_digest
        )
    else:
        if prior_tail is not None:
            moved = np.concatenate([prior_tail, moved], axis=-1)
        expanded_window = np.broadcast_to(win, batch_shape + (win.shape[-1],))
        padded_window = np.zeros(batch_shape + (policy.n_fft,), dtype=win.dtype)
        offset = (policy.n_fft - policy.window_length) // 2
        padded_window[..., offset : offset + policy.window_length] = expanded_window
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
            packed_spectra = transform(frames, axis=-1)
        else:
            packed_spectra = np.empty(batch_shape + (0, bins), dtype=np.complex64)
        tail_start = frame_count * policy.hop
        tail = np.ascontiguousarray(moved[..., tail_start:])
        lead_rank = values.ndim - 1
        order = (list(range(axis)) + [lead_rank, lead_rank + 1]
                 + list(range(axis, lead_rank)))
        spectra = np.transpose(packed_spectra.astype(np.complex64), order)
    samples_consumed = consumed + values.shape[axis]
    frames_emitted = emitted + frame_count
    tail_digest = _array_digest(tail)
    next_state = StreamingSTFTState(
        policy_digest=policy.digest,
        artifact_digest=artifact_digest,
        target=target,
        architecture=architecture,
        tail=tail,
        samples_consumed=samples_consumed,
        frames_emitted=frames_emitted,
        parent_state_digest=parent_state_digest,
        window_digest=window_digest,
        tail_digest=tail_digest,
        state_digest=_state_digest(
            policy_digest=policy.digest,
            artifact_digest=artifact_digest,
            target=target,
            architecture=architecture,
            parent_state_digest=parent_state_digest,
            window_digest=window_digest,
            tail_digest=tail_digest,
            samples_consumed=samples_consumed,
            frames_emitted=frames_emitted,
        ),
    )
    return spectra, next_state
