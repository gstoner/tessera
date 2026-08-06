"""Content-addressed compound TSOL programs over canonical FFT artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .scheduled_fft import lower_scheduled_fft, validate_scheduled_fft_metadata
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt
from .spectral_plan import next_power_of_two


_OPS = (
    "tessera.spectral_filter",
    "tessera.dct",
    "tessera.spectral_conv",
    "tessera.stft",
    "tessera.istft",
)

_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


def _packed_workspace_bytes(*segments: tuple[int, int]) -> int:
    offset = 0
    for element_bytes, elements in segments:
        offset = (offset + element_bytes - 1) // element_bytes * element_bytes
        offset += element_bytes * elements
    return offset


@dataclass(frozen=True)
class ScheduledSpectralArtifact:
    schedule_ir: str
    tile_ir: str
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

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def _input_shapes_text(self) -> str:
        return "|".join("x".join(str(dim) for dim in shape) for shape in self.input_shapes)

    def _child_digests_text(self) -> str:
        return ",".join(str(child["schedule_digest"]) for child in self.child_ffts)

    def _identity_payload(self) -> str:
        output = "x".join(str(dim) for dim in self.output_shape)
        return (
            f"schema=tessera.scheduled_spectral.v2;op={self.op_name};"
            f"target={self.target};arch={self.architecture};"
            f"inputs={self._input_shapes_text()};output={output};axis={self.axis};"
            f"padding={self.padding[0]},{self.padding[1]};"
            f"crop={self.crop[0]},{self.crop[1]};window={self.window_length};"
            f"hop={self.hop};frames={self.frames};normalization={self.normalization};"
            f"complex_layout={self.complex_layout};accumulation={self.accumulation};"
            f"workspace_bytes={self.workspace_bytes};"
            f"workspace_policy={self.workspace_policy};"
            f"mutation_lineage={self.mutation_lineage};native_entry={self.native_entry};"
            f"child_fft_digests={self._child_digests_text()};"
            f"workgroup={1 if self.target == 'x86' else 256}"
        )

    def _identity(self) -> dict[str, Any]:
        return {
            "schema": "tessera.scheduled_spectral.v2",
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
        if (self.target, self.architecture) not in {
            ("rocm", "gfx1151"),
            ("x86", "zen5-avx512"),
        }:
            raise ValueError("TSOL package requires exact gfx1151 or Zen 5 AVX-512")
        if digest_text(self._identity_payload()) != self.schedule_digest:
            raise ValueError("TSOL package content identity mismatch")
        for child in self.child_ffts:
            validate_scheduled_fft_metadata(
                child, target=self.target, input_shape=child["input_shape"]
            )
        if self.complex_layout != "interleaved_f32x2":
            raise ValueError("TSOL package complex layout mismatch")
        if self.normalization != "backward":
            raise ValueError("TSOL package normalization mismatch")
        if self.workspace_policy != "persistent_artifact_workspace":
            raise ValueError("TSOL package workspace policy mismatch")
        if self.mutation_lineage != "inputs_immutable_output_fresh_v1":
            raise ValueError("TSOL package mutation lineage mismatch")
        if self.schedule_ir.count("schedule.spectral_program") != 1:
            raise ValueError("TSOL package requires one Schedule program edge")
        if self.schedule_ir.count("schedule.artifact") != 1:
            raise ValueError("TSOL package requires one durable schedule artifact")
        if self.tile_ir.count("tile.spectral_program_kernel") != 1:
            raise ValueError("TSOL package requires one launch-level Tile program")
        if any(name in self.tile_ir for name in ("schedule.spectral_program", "schedule.artifact")):
            raise ValueError("TSOL Tile package retained Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("TSOL Tile package has stale schedule identity")

    def to_metadata(self) -> dict[str, Any]:
        result = self._identity()
        result.update(
            {
                "child_ffts": [dict(child) for child in self.child_ffts],
                "schedule_digest": self.schedule_digest,
                "schedule_ir": self.schedule_ir,
                "schedule_ir_digest": self.schedule_ir_digest,
                "tile_ir": self.tile_ir,
                "tile_digest": self.tile_digest,
            }
        )
        return result


def _child(
    target: str, op_name: str, shape: tuple[int, ...], *, n: int | None = None
) -> dict[str, Any]:
    return lower_scheduled_fft(
        target=target,
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
    if target not in {"x86", "rocm", "rocm_gfx1151"}:
        raise ValueError(
            "scheduled TSOL composites support Zen 5 AVX-512 and rocm_gfx1151; "
            "gfx1200/gfx1250 require architecture-owned profiles and evidence"
        )
    compiler_target = "x86" if target == "x86" else "rocm"
    architecture = "zen5-avx512" if compiler_target == "x86" else "gfx1151"
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
    policy = "persistent_artifact_workspace"

    if op_name == "tessera.spectral_filter":
        if len(shapes) != 2 or shapes[0] != shapes[1]:
            raise ValueError("spectral_filter requires two equal complex shapes")
        output = shapes[0]
        elements = math.prod(output)
        workspace = _packed_workspace_bytes(*((8, elements),) * 3)
        entry = (
            "tessera_x86_spectral_filter_f32"
            if compiler_target == "x86"
            else "ts_spectral_filter_plan_hostptr_amd"
        )
    elif op_name == "tessera.dct":
        if len(shapes) != 1:
            raise ValueError("dct requires one input")
        n = shapes[0][-1]
        child_shape = (*shapes[0][:-1], 2 * n)
        children = (_child(compiler_target, "tessera.fft", child_shape),)
        output = shapes[0]
        padding = (0, n)
        crop = (0, n)
        # Host staging (input/output) plus two complex device buffers at 2N.
        elements = math.prod(shapes[0][:-1] or (1,)) * n
        workspace = _packed_workspace_bytes(
            (4, elements), (4, elements), (8, 2 * elements), (8, 2 * elements)
        )
        entry = (
            "tessera_x86_dct_f32"
            if compiler_target == "x86"
            else "ts_dct_plan_hostptr_batch_amd"
        )
    elif op_name == "tessera.spectral_conv":
        if len(shapes) != 2 or shapes[0][:-1] != shapes[1][:-1]:
            raise ValueError("spectral_conv requires matching static batch dimensions")
        output_n = shapes[0][-1] + shapes[1][-1] - 1
        fft_n = next_power_of_two(output_n)
        rows = math.prod(shapes[0][:-1] or (1,))
        child_shape = (rows, fft_n)
        children = (
            _child(compiler_target, "tessera.fft", child_shape),
            _child(compiler_target, "tessera.ifft", child_shape),
        )
        output = (*shapes[0][:-1], output_n)
        padding = (fft_n - shapes[0][-1], fft_n - shapes[1][-1])
        crop = (0, fft_n - output_n)
        # Three real staging buffers and six complex FFT-sized buffers.
        workspace = _packed_workspace_bytes(
            (4, rows * shapes[0][-1]),
            (4, rows * shapes[1][-1]),
            (4, rows * output_n),
            *((8, rows * fft_n),) * 6,
        )
        entry = (
            "tessera_x86_spectral_conv_f32"
            if compiler_target == "x86"
            else "ts_spectral_conv_plan_hostptr_batch_amd"
        )
    elif op_name == "tessera.stft":
        if len(shapes) != 2 or len(shapes[1]) != 1:
            raise ValueError("stft requires a signal and rank-1 window")
        win = shapes[1][0]
        stride = int(hop or 0)
        if stride <= 0 or win > shapes[0][-1]:
            raise ValueError("stft requires 0 < hop and window <= signal")
        frames = (shapes[0][-1] - win) // stride + 1
        batch = math.prod(shapes[0][:-1] or (1,))
        children = (_child(compiler_target, "tessera.rfft", (batch * frames, win)),)
        output = (*shapes[0][:-1], frames, win // 2 + 1)
        workspace = _packed_workspace_bytes(
            (4, batch * shapes[0][-1]),
            (4, win),
            (8, batch * frames * win),
            (8, batch * frames * win),
            (8, batch * frames * (win // 2 + 1)),
        )
        entry = (
            "tessera_x86_stft_f32"
            if compiler_target == "x86"
            else "ts_stft_plan_hostptr_batch_amd"
        )
    else:
        if len(shapes) != 2 or len(shapes[1]) != 1:
            raise ValueError("istft requires spectra and a rank-1 window")
        win = shapes[1][0]
        stride = int(hop or 0)
        frames = shapes[0][-2]
        if stride <= 0 or shapes[0][-1] != win // 2 + 1:
            raise ValueError("istft spectrum/window/hop contract mismatch")
        batch = math.prod(shapes[0][:-2] or (1,))
        children = (
            _child(
                compiler_target,
                "tessera.irfft",
                (batch * frames, win // 2 + 1),
                n=win,
            ),
        )
        samples = (frames - 1) * stride + win
        output = (*shapes[0][:-2], samples)
        workspace = _packed_workspace_bytes(
            (4, win),
            (4, batch * samples),
            (8, batch * frames * (win // 2 + 1)),
            (8, batch * frames * win),
            (8, batch * frames * win),
        )
        accumulation = "deterministic_f32_ascending_frames"
        entry = (
            "tessera_x86_istft_f32"
            if compiler_target == "x86"
            else "ts_istft_plan_hostptr_batch_amd"
        )

    provisional = ScheduledSpectralArtifact(
        schedule_ir="",
        tile_ir="",
        op_name=op_name,
        target=compiler_target,
        architecture=architecture,
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
    identity = provisional._identity_payload()
    schedule_digest = digest_text(identity)
    input_names = tuple(f"a{index}" for index in range(len(shapes)))
    input_elements = (
        ("complex<f32>", "complex<f32>")
        if op_name == "tessera.spectral_filter"
        else ("complex<f32>", "f32")
        if op_name == "tessera.istft"
        else tuple("f32" for _ in shapes)
    )
    output_element = (
        "complex<f32>"
        if op_name in {"tessera.spectral_filter", "tessera.stft"}
        else "f32"
    )
    input_types = tuple(
        "tensor<" + "x".join([*(str(dim) for dim in shape), element]) + ">"
        for shape, element in zip(shapes, input_elements)
    )
    output_type = "tensor<" + "x".join(
        [*(str(dim) for dim in output), output_element]
    ) + ">"
    operands = ", ".join(f"%{name}" for name in input_names)
    function_args = ", ".join(
        f"%{name}: {type_name}" for name, type_name in zip(input_names, input_types)
    )
    operand_types = ", ".join(input_types)
    output_shape_text = ", ".join(str(dim) for dim in output)
    padding_text = ", ".join(str(value) for value in padding)
    crop_text = ", ".join(str(value) for value in crop)
    child_digests = ",".join(str(child["schedule_digest"]) for child in children)
    attrs = (
        f'artifact_hash = "{schedule_digest}", target = "{compiler_target}", '
        f'arch = "{architecture}", kind = "{op_name}", '
        f'input_shapes = "{provisional._input_shapes_text()}", '
        f'output_shape = array<i64: {output_shape_text}>, axis = {normalized_axis} : i64, '
        f'padding = array<i64: {padding_text}>, crop = array<i64: {crop_text}>, '
        f'window_length = {win} : i64, hop = {stride} : i64, frames = {frames} : i64, '
        f'normalization = "backward", complex_layout = "interleaved_f32x2", '
        f'accumulation = "{accumulation}", workspace_bytes = {workspace} : i64, '
        f'workspace_policy = "{policy}", mutation_lineage = "inputs_immutable_output_fresh_v1", '
        f'native_entry = "{entry}", child_fft_digests = "{child_digests}", '
        f'workgroup_size = {1 if compiler_target == "x86" else 256} : i64'
    )
    schedule_ir = (
        f'module attributes {{tessera.target = "{compiler_target}", '
        f'tessera.arch = "{architecture}"}} {{\n'
        f"  func.func @scheduled_spectral({function_args}) -> {output_type} {{\n"
        f'    %result = "schedule.spectral_program"({operands}) {{{attrs}}} : '
        f"({operand_types}) -> {output_type}\n"
        f'    "schedule.artifact"() {{hash = "{schedule_digest}", '
        f'arch = "{architecture}", shape_key = "{provisional._input_shapes_text()}", '
        f'numeric_policy = "{accumulation};backward"}} : () -> ()\n'
        f"    return %result : {output_type}\n"
        f"  }}\n"
        f"}}\n"
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled TSOL lowering requires production tessera-opt")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    artifact = ScheduledSpectralArtifact(
        **{
            **provisional.__dict__,
            "schedule_ir": schedule_ir,
            "tile_ir": tile_ir,
            "schedule_digest": schedule_digest,
        }
    )
    artifact.validate()
    return artifact


def validate_scheduled_spectral_metadata(
    metadata: Mapping[str, Any], *, input_shapes: Sequence[Sequence[int]]
) -> Mapping[str, Any]:
    if metadata.get("schema") != "tessera.scheduled_spectral.v2":
        raise ValueError("TSOL package requires tessera.scheduled_spectral.v2 metadata")
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
    if tuple(tuple(shape) for shape in metadata.get("input_shapes") or ()) != shapes:
        raise ValueError("TSOL package input shape mismatch")
    rebuilt = lower_scheduled_spectral(
        target=str(metadata.get("target")),
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
