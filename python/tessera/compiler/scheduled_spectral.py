"""Content-addressed compound TSOL programs over canonical FFT artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .scheduled_fft import lower_scheduled_fft, validate_scheduled_fft_metadata
from .scheduled_matmul import digest_text
from .spectral_plan import next_power_of_two


_OPS = (
    "tessera.spectral_filter",
    "tessera.dct",
    "tessera.spectral_conv",
    "tessera.stft",
    "tessera.istft",
)


@dataclass(frozen=True)
class ScheduledSpectralArtifact:
    op_name: str
    target: str
    architecture: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    axis: int
    padding: tuple[int, int]
    crop: tuple[int, int]
    window_length: int
    hop: int
    frames: int
    normalization: str
    complex_layout: str
    accumulation: str
    workspace_bytes: int
    workspace_policy: str
    mutation_lineage: str
    native_entry: str
    child_ffts: tuple[Mapping[str, Any], ...]
    schedule_digest: str

    def _identity(self) -> dict[str, Any]:
        return {
            "schema": "tessera.scheduled_spectral.v1",
            "op_name": self.op_name,
            "target": self.target,
            "architecture": self.architecture,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shape": list(self.output_shape),
            "axis": self.axis,
            "padding": list(self.padding),
            "crop": list(self.crop),
            "window_length": self.window_length,
            "hop": self.hop,
            "frames": self.frames,
            "normalization": self.normalization,
            "complex_layout": self.complex_layout,
            "accumulation": self.accumulation,
            "workspace_bytes": self.workspace_bytes,
            "workspace_policy": self.workspace_policy,
            "mutation_lineage": self.mutation_lineage,
            "native_entry": self.native_entry,
            "child_fft_digests": [
                str(child["schedule_digest"]) for child in self.child_ffts
            ],
        }

    def validate(self) -> None:
        if self.op_name not in _OPS:
            raise ValueError("TSOL package operation identity mismatch")
        if self.target != "rocm" or self.architecture != "gfx1151":
            raise ValueError("TSOL package supports only exact gfx1151")
        if digest_text(
            json.dumps(self._identity(), sort_keys=True, separators=(",", ":"))
        ) != self.schedule_digest:
            raise ValueError("TSOL package content identity mismatch")
        for child in self.child_ffts:
            validate_scheduled_fft_metadata(
                child, target="rocm", input_shape=child["input_shape"]
            )
        if self.complex_layout != "interleaved_f32x2":
            raise ValueError("TSOL package complex layout mismatch")
        if self.normalization != "backward":
            raise ValueError("TSOL package normalization mismatch")
        if self.mutation_lineage != "inputs_immutable_output_fresh_v1":
            raise ValueError("TSOL package mutation lineage mismatch")

    def to_metadata(self) -> dict[str, Any]:
        result = self._identity()
        result.update(
            {
                "child_ffts": [dict(child) for child in self.child_ffts],
                "schedule_digest": self.schedule_digest,
            }
        )
        return result


def _child(op_name: str, shape: tuple[int, ...], *, n: int | None = None) -> dict[str, Any]:
    return lower_scheduled_fft(
        target="rocm",
        op_name=op_name,
        input_shape=shape,
        axis=-1,
        n=n,
        input_name="spectral_child_input",
        output_name="spectral_child_output",
    ).to_metadata()


@lru_cache(maxsize=128)
def lower_scheduled_spectral(
    *,
    target: str,
    op_name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    axis: int = -1,
    hop: int | None = None,
) -> ScheduledSpectralArtifact:
    if target not in {"rocm", "rocm_gfx1151"}:
        raise ValueError(
            "scheduled TSOL composites support only rocm_gfx1151; "
            "gfx1200/gfx1250 require architecture-owned profiles and evidence"
        )
    if op_name not in _OPS:
        raise ValueError(f"unsupported scheduled TSOL operation {op_name!r}")
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
    if not shapes or any(not shape or any(dim <= 0 for dim in shape) for shape in shapes):
        raise ValueError("scheduled TSOL requires positive static input shapes")
    normalized_axis = axis if axis >= 0 else len(shapes[0]) + axis
    if normalized_axis != len(shapes[0]) - 1:
        raise ValueError("gfx1151 TSOL v1 requires the transformed axis to be last")

    padding = (0, 0)
    crop = (0, 0)
    win = 0
    stride = 0
    frames = 0
    children: tuple[Mapping[str, Any], ...] = ()
    accumulation = "f32"
    policy = "call_workspace"

    if op_name == "tessera.spectral_filter":
        if len(shapes) != 2 or shapes[0] != shapes[1]:
            raise ValueError("spectral_filter requires two equal complex shapes")
        output = shapes[0]
        workspace = 3 * math.prod(output) * 8
        entry = "ts_spectral_filter_hostptr_amd"
    elif op_name == "tessera.dct":
        if len(shapes) != 1:
            raise ValueError("dct requires one input")
        n = shapes[0][-1]
        child_shape = (*shapes[0][:-1], 2 * n)
        children = (_child("tessera.fft", child_shape),)
        output = shapes[0]
        padding = (0, n)
        crop = (0, n)
        # Host staging (input/output) plus two complex device buffers at 2N.
        workspace = math.prod(shapes[0][:-1] or (1,)) * n * 40
        entry = "ts_dct_hostptr_batch_amd"
    elif op_name == "tessera.spectral_conv":
        if len(shapes) != 2 or shapes[0][:-1] != shapes[1][:-1]:
            raise ValueError("spectral_conv requires matching static batch dimensions")
        output_n = shapes[0][-1] + shapes[1][-1] - 1
        fft_n = next_power_of_two(output_n)
        rows = math.prod(shapes[0][:-1] or (1,))
        child_shape = (rows, fft_n)
        children = (
            _child("tessera.fft", child_shape),
            _child("tessera.ifft", child_shape),
        )
        output = (*shapes[0][:-1], output_n)
        padding = (fft_n - shapes[0][-1], fft_n - shapes[1][-1])
        crop = (0, fft_n - output_n)
        # Three real staging buffers and six complex FFT-sized buffers.
        workspace = rows * (
            (shapes[0][-1] + shapes[1][-1] + output_n) * 4
            + 6 * fft_n * 8
        )
        entry = "ts_spectral_conv_hostptr_batch_amd"
    elif op_name == "tessera.stft":
        if len(shapes) != 2 or len(shapes[1]) != 1:
            raise ValueError("stft requires a signal and rank-1 window")
        win = shapes[1][0]
        stride = int(hop or 0)
        if stride <= 0 or win > shapes[0][-1]:
            raise ValueError("stft requires 0 < hop and window <= signal")
        frames = (shapes[0][-1] - win) // stride + 1
        batch = math.prod(shapes[0][:-1] or (1,))
        children = (_child("tessera.rfft", (batch * frames, win)),)
        output = (*shapes[0][:-1], frames, win // 2 + 1)
        workspace = (
            batch * shapes[0][-1] * 4
            + win * 4
            + batch * frames * (2 * win * 8 + (win // 2 + 1) * 8)
        )
        entry = "ts_stft_hostptr_batch_amd"
    else:
        if len(shapes) != 2 or len(shapes[1]) != 1:
            raise ValueError("istft requires spectra and a rank-1 window")
        win = shapes[1][0]
        stride = int(hop or 0)
        frames = shapes[0][-2]
        if stride <= 0 or shapes[0][-1] != win // 2 + 1:
            raise ValueError("istft spectrum/window/hop contract mismatch")
        batch = math.prod(shapes[0][:-2] or (1,))
        children = (_child("tessera.irfft", (batch * frames, win // 2 + 1), n=win),)
        samples = (frames - 1) * stride + win
        output = (*shapes[0][:-2], samples)
        workspace = (
            win * 4
            + batch * samples * 4
            + batch * frames * ((win // 2 + 1) * 8 + 2 * win * 8)
        )
        accumulation = "deterministic_f32_ascending_frames"
        entry = "ts_istft_hostptr_batch_amd"

    provisional = ScheduledSpectralArtifact(
        op_name=op_name,
        target="rocm",
        architecture="gfx1151",
        input_shapes=shapes,
        output_shape=output,
        axis=normalized_axis,
        padding=padding,
        crop=crop,
        window_length=win,
        hop=stride,
        frames=frames,
        normalization="backward",
        complex_layout="interleaved_f32x2",
        accumulation=accumulation,
        workspace_bytes=workspace,
        workspace_policy=policy,
        mutation_lineage="inputs_immutable_output_fresh_v1",
        native_entry=entry,
        child_ffts=children,
        schedule_digest="",
    )
    identity = json.dumps(provisional._identity(), sort_keys=True, separators=(",", ":"))
    artifact = ScheduledSpectralArtifact(
        **{**provisional.__dict__, "schedule_digest": digest_text(identity)}
    )
    artifact.validate()
    return artifact


def validate_scheduled_spectral_metadata(
    metadata: Mapping[str, Any], *, input_shapes: Sequence[Sequence[int]]
) -> Mapping[str, Any]:
    if metadata.get("schema") != "tessera.scheduled_spectral.v1":
        raise ValueError("TSOL package requires tessera.scheduled_spectral.v1 metadata")
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
    if tuple(tuple(shape) for shape in metadata.get("input_shapes") or ()) != shapes:
        raise ValueError("TSOL package input shape mismatch")
    rebuilt = lower_scheduled_spectral(
        target="rocm",
        op_name=str(metadata.get("op_name")),
        input_shapes=shapes,
        axis=int(metadata.get("axis", -1)),
        hop=int(metadata.get("hop", 0)) or None,
    )
    expected = rebuilt.to_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"TSOL package contract mismatch for {key}")
    return metadata
