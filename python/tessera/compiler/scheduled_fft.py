"""Canonical one-dimensional FFT through Graph -> Schedule -> launch Tile."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt
from .spectral_plan import mixed_radix_sequence, next_power_of_two, plan_fft


_FFT_OPS = ("tessera.fft", "tessera.ifft", "tessera.rfft", "tessera.irfft")
_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


def _prefer_x86_mixed_radix(length: int, radices: tuple[int, ...] | None) -> bool:
    """Evidence-derived work gate for the Zen 5 mixed-radix candidate.

    Direct work is proportional to N*sum(radix); Bluestein executes three
    padded radix-2 transforms.  The 0.4 boundary selects every measured win in
    the committed shape corpus and rejects N=255, its only measured loss.
    """
    if length <= 8 or radices is None or length & (length - 1) == 0:
        return False
    padded = next_power_of_two(2 * length - 1)
    direct_work = length * sum(radices)
    bluestein_work = 3 * padded * (padded.bit_length() - 1)
    return 5 * direct_work <= 2 * bluestein_work


@dataclass(frozen=True)
class ScheduledFFTArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    op_name: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: int
    length: int
    physical_length: int
    batch: int
    mode: str
    real_transform_policy: str
    hermitian_layout: str
    inverse: bool
    normalization: str
    scale: float
    radix_policy: str
    strategy: str
    algorithm: str
    radix_sequence: tuple[int, ...]
    bluestein_m: int
    workspace_elems: int
    workspace_policy: str
    residency: str
    twiddle_policy: str
    kernel_family: str
    workgroup_size: int
    schedule_digest: str

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        if self.schedule_ir.count("schedule.fft") != 1:
            raise ValueError("scheduled FFT requires one schedule.fft SSA edge")
        if self.schedule_ir.count("schedule.artifact {") != 1:
            raise ValueError("scheduled FFT requires one durable schedule artifact")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("scheduled FFT has incomplete content identity")
        if self.tile_ir.count("tile.fft_kernel") != 1:
            raise ValueError("scheduled FFT requires one launch-level Tile package")
        if any(
            op in self.tile_ir
            for op in ("schedule.fft", "schedule.artifact", *_FFT_OPS)
        ):
            raise ValueError("scheduled FFT Tile package retained Graph or Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("scheduled FFT Tile package has a stale schedule digest")
        for name, value in (
            ("axis", self.axis),
            ("length", self.length),
            ("physical_length", self.physical_length),
            ("batch", self.batch),
            ("bluestein_m", self.bluestein_m),
            ("workspace_elems", self.workspace_elems),
            ("tessera.workgroup_size", self.workgroup_size),
        ):
            if not re.search(rf"{re.escape(name)} = {value} : i64", self.tile_ir):
                raise ValueError(f"scheduled FFT Tile package has stale {name}")
        for attr_name, attr_value in (
            ("mode", self.mode),
            ("real_transform_policy", self.real_transform_policy),
            ("hermitian_layout", self.hermitian_layout),
            ("normalization", self.normalization),
            ("radix_policy", self.radix_policy),
            ("strategy", self.strategy),
            ("algorithm", self.algorithm),
            ("workspace_policy", self.workspace_policy),
            ("residency", self.residency),
            ("twiddle_policy", self.twiddle_policy),
            ("kernel_family", self.kernel_family),
        ):
            if f'{attr_name} = "{attr_value}"' not in self.tile_ir:
                raise ValueError(f"scheduled FFT Tile package has stale {attr_name}")
        inverse = "true" if self.inverse else "false"
        if f"inverse = {inverse}" not in self.tile_ir:
            raise ValueError("scheduled FFT Tile package has stale direction")
        radix = ", ".join(str(value) for value in self.radix_sequence)
        expected = f"radix_sequence = array<i64: {radix}>" if radix else "radix_sequence = array<i64>"
        if expected not in self.tile_ir:
            raise ValueError("scheduled FFT Tile package has stale radix sequence")
        if self.tile_digest == self.schedule_ir_digest:
            raise ValueError("Schedule and Tile FFT artifacts must be distinct")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": "tessera.scheduled_fft.v3",
            "target": self.target,
            "architecture": self.architecture,
            "op_name": self.op_name,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "axis": self.axis,
            "length": self.length,
            "physical_length": self.physical_length,
            "batch": self.batch,
            "mode": self.mode,
            "real_transform_policy": self.real_transform_policy,
            "hermitian_layout": self.hermitian_layout,
            "inverse": self.inverse,
            "normalization": self.normalization,
            "scale": self.scale,
            "radix_policy": self.radix_policy,
            "strategy": self.strategy,
            "algorithm": self.algorithm,
            "radix_sequence": list(self.radix_sequence),
            "bluestein_m": self.bluestein_m,
            "workspace_elems": self.workspace_elems,
            "workspace_policy": self.workspace_policy,
            "residency": self.residency,
            "twiddle_policy": self.twiddle_policy,
            "kernel_family": self.kernel_family,
            "workgroup_size": self.workgroup_size,
            "schedule_digest": self.schedule_digest,
            "tile_digest": self.tile_digest,
            "tile_ir": self.tile_ir,
        }


def _shape_type(shape: Sequence[int], element: str) -> str:
    return "tensor<" + "x".join([*(str(int(dim)) for dim in shape), element]) + ">"


@lru_cache(maxsize=128)
def lower_scheduled_fft(
    *,
    target: str,
    op_name: str,
    input_shape: tuple[int, ...],
    axis: int = -1,
    n: int | None = None,
    input_name: str = "x",
    output_name: str = "o",
) -> ScheduledFFTArtifact:
    if op_name not in _FFT_OPS:
        raise ValueError(f"unsupported scheduled FFT op {op_name!r}")
    shape = tuple(int(dim) for dim in input_shape)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("scheduled FFT requires a positive static input shape")
    normalized_axis = axis if axis >= 0 else len(shape) + axis
    if normalized_axis < 0 or normalized_axis >= len(shape):
        raise ValueError("scheduled FFT axis is out of range")
    compiler_target = "x86" if target == "x86" else "rocm"
    if target not in {"x86", "rocm", "rocm_gfx1151"}:
        raise ValueError("scheduled FFT supports x86 and rocm_gfx1151")
    architecture = "zen5-avx512" if compiler_target == "x86" else "gfx1151"
    mode = {"tessera.rfft": "r2c", "tessera.irfft": "c2r"}.get(op_name, "c2c")
    inverse = op_name in {"tessera.ifft", "tessera.irfft"}
    if op_name == "tessera.irfft":
        length = int(n if n is not None else 2 * (shape[normalized_axis] - 1))
    else:
        length = shape[normalized_axis]
    output_shape = list(shape)
    if op_name == "tessera.rfft":
        output_shape[normalized_axis] = length // 2 + 1
    elif op_name == "tessera.irfft":
        output_shape[normalized_axis] = length
    output = tuple(output_shape)
    batch = math.prod(dim for index, dim in enumerate(shape) if index != normalized_axis)
    real_mode = mode in {"r2c", "c2r"}
    candidate_physical_length = length // 2 if real_mode and length % 2 == 0 else length
    candidate_x86_radices = (
        mixed_radix_sequence(candidate_physical_length)
        if compiler_target == "x86"
        else None
    )
    packed_real = real_mode and length % 2 == 0 and (
        compiler_target == "rocm"
        or candidate_physical_length & (candidate_physical_length - 1) == 0
        or _prefer_x86_mixed_radix(candidate_physical_length, candidate_x86_radices)
    )
    physical_length = candidate_physical_length if packed_real else length
    real_transform_policy = (
        "packed_even_n2_hermitian_v1"
        if packed_real
        else "full_complex_hermitian_fallback"
        if real_mode
        else "not_applicable"
    )
    hermitian_layout = (
        "half_spectrum_nyquist_explicit" if real_mode else "full_complex"
    )
    x86_radices = (
        mixed_radix_sequence(physical_length) if compiler_target == "x86" else None
    )
    x86_mixed = _prefer_x86_mixed_radix(physical_length, x86_radices)
    radix_policy = (
        "mixed_radix" if compiler_target != "x86" or x86_mixed else "radix2"
    )
    plan = plan_fft(
        physical_length,
        axis=normalized_axis,
        mode=mode,
        inverse=inverse,
        radix_policy=radix_policy,
    )
    workspace = plan.workspace_elems
    if compiler_target == "rocm" and plan.strategy == "bluestein":
        # Persistent native plan: three mutable M buffers plus one immutable
        # pretransformed chirp. Host/device staging is capacity-managed by the
        # package and is not algorithm workspace.
        workspace = 4 * plan.bluestein_m
    if compiler_target == "x86" and plan.strategy == "mixed_radix":
        workspace = 2 * physical_length
    algorithm = (
        "stockham_autosort"
        if compiler_target != "x86" or plan.strategy == "mixed_radix"
        else "cooley_tukey_dit"
    )
    residency = (
        "host_thread_local_ping_pong"
        if compiler_target == "x86" and plan.strategy == "mixed_radix"
        else "host_inplace"
        if compiler_target == "x86"
        else "persistent_device_plan_fused_lds_hermitian_batch"
        if real_transform_policy == "packed_even_n2_hermitian_v1"
        and plan.strategy != "bluestein"
        and physical_length <= 1024
        and physical_length & (physical_length - 1) == 0
        else "persistent_device_plan_fused_lds_batch"
        if plan.strategy != "bluestein"
        and physical_length <= 1024
        and physical_length & (physical_length - 1) == 0
        else "persistent_device_plan"
    )
    twiddle_policy = (
        "thread_local_cached_f32"
        if compiler_target == "x86"
        else "device_sincos_per_butterfly"
    )
    if compiler_target == "rocm":
        workspace_policy = (
            "persistent_plan_4m"
            if plan.strategy == "bluestein"
            else "persistent_plan_n"
        )
        twiddle_policy = (
            "persistent_device_chirp_fft"
            if plan.strategy == "bluestein"
            else twiddle_policy
        )
    else:
        workspace_policy = (
            "host_temporary_m"
            if plan.strategy == "bluestein"
            else "thread_local_2n"
            if plan.strategy == "mixed_radix"
            else "inplace_no_scratch"
        )
    kernel_family = (
        "zen5_avx512_fft_v4"
        if compiler_target == "x86"
        else "gfx1151_stockham_bluestein_v6"
    )
    input_element = "f32" if op_name == "tessera.rfft" else "complex<f32>"
    output_element = "f32" if op_name == "tessera.irfft" else "complex<f32>"
    input_type = _shape_type(shape, input_element)
    output_type = _shape_type(output, output_element)
    attrs = [f"axis = {axis} : i64"]
    if op_name == "tessera.irfft":
        attrs.append(f"n = {length} : i64")
    graph_ir = (
        f'module attributes {{tessera.target = "{compiler_target}", '
        f'tessera.arch = "{architecture}"}} {{\n'
        f"  func.func @scheduled_fft(%{input_name}: {input_type}) -> {output_type} {{\n"
        f'    %result = "{op_name}"(%{input_name}) {{{", ".join(attrs)}}} : '
        f"({input_type}) -> {output_type}\n"
        f"    return %result : {output_type}\n"
        f"  }}\n"
        f"}}\n"
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled FFT lowering requires production tessera-opt")
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled FFT lowering lost its schedule identity")
    artifact = ScheduledFFTArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=compiler_target,
        architecture=architecture,
        op_name=op_name,
        input_name=input_name,
        output_name=output_name,
        input_shape=shape,
        output_shape=output,
        axis=normalized_axis,
        length=length,
        physical_length=physical_length,
        batch=batch,
        mode=mode,
        real_transform_policy=real_transform_policy,
        hermitian_layout=hermitian_layout,
        inverse=inverse,
        normalization="backward",
        scale=1.0 / length if inverse else 1.0,
        radix_policy=radix_policy,
        strategy=plan.strategy,
        algorithm=algorithm,
        radix_sequence=tuple(int(value) for value in plan.radix_seq),
        bluestein_m=plan.bluestein_m,
        workspace_elems=workspace,
        workspace_policy=workspace_policy,
        residency=residency,
        twiddle_policy=twiddle_policy,
        kernel_family=kernel_family,
        workgroup_size=1 if compiler_target == "x86" else 256,
        schedule_digest=hashes[0],
    )
    artifact.validate()
    return artifact


def validate_scheduled_fft_metadata(
    metadata: Mapping[str, Any],
    *,
    target: str,
    input_shape: Sequence[int],
) -> Mapping[str, Any]:
    expected_target = "x86" if target == "x86" else "rocm"
    expected_architecture = "zen5-avx512" if expected_target == "x86" else "gfx1151"
    if metadata.get("schema") != "tessera.scheduled_fft.v3":
        raise ValueError("FFT package requires tessera.scheduled_fft.v3 metadata")
    if metadata.get("target") != expected_target:
        raise ValueError("FFT package target identity mismatch")
    if metadata.get("architecture") != expected_architecture:
        raise ValueError("FFT package architecture identity mismatch")
    shape = tuple(int(dim) for dim in input_shape)
    if tuple(metadata.get("input_shape") or ()) != shape:
        raise ValueError("FFT package input shape mismatch")
    op_name = str(metadata.get("op_name") or "")
    if op_name not in _FFT_OPS:
        raise ValueError("FFT package operation identity mismatch")
    axis = int(metadata.get("axis", -1))
    if axis < 0 or axis >= len(shape):
        raise ValueError("FFT package axis is not normalized")
    length = int(metadata.get("length", 0))
    physical_length = int(metadata.get("physical_length", 0))
    # An rFFT spectrum with ``bins`` entries can represent either an even
    # length ``2*(bins-1)`` or the adjacent odd length. The explicit ``n`` is
    # part of the content-addressed Tile artifact, so validate the bin/length
    # relation rather than silently forcing the even default.
    valid_length = (
        shape[axis] == length // 2 + 1
        if op_name == "tessera.irfft"
        else length == shape[axis]
    )
    if not valid_length:
        raise ValueError("FFT package transform length mismatch")
    mode = str(metadata.get("mode") or "")
    real_mode = mode in {"r2c", "c2r"}
    policy = str(metadata.get("real_transform_policy") or "")
    packed = policy == "packed_even_n2_hermitian_v1"
    if packed and (not real_mode or length % 2 or physical_length != length // 2):
        raise ValueError("FFT package packed-real physical length mismatch")
    if not packed and physical_length != length:
        raise ValueError("FFT package fallback physical length mismatch")
    expected_layout = "half_spectrum_nyquist_explicit" if real_mode else "full_complex"
    if metadata.get("hermitian_layout") != expected_layout:
        raise ValueError("FFT package Hermitian layout mismatch")
    expected_batch = math.prod(dim for index, dim in enumerate(shape) if index != axis)
    if int(metadata.get("batch", 0)) != expected_batch:
        raise ValueError("FFT package batch mismatch")
    expected_output = list(shape)
    if op_name == "tessera.rfft":
        expected_output[axis] = length // 2 + 1
    elif op_name == "tessera.irfft":
        expected_output[axis] = length
    if tuple(metadata.get("output_shape") or ()) != tuple(expected_output):
        raise ValueError("FFT package output shape mismatch")
    expected_mode = {"tessera.rfft": "r2c", "tessera.irfft": "c2r"}.get(
        op_name, "c2c"
    )
    if metadata.get("mode") != expected_mode:
        raise ValueError("FFT package mode mismatch")
    expected_inverse = op_name in {"tessera.ifft", "tessera.irfft"}
    if bool(metadata.get("inverse")) != expected_inverse:
        raise ValueError("FFT package direction mismatch")
    if metadata.get("normalization") != "backward":
        raise ValueError("FFT package normalization mismatch")
    if not str(metadata.get("input_name") or ""):
        raise ValueError("FFT package input binding is missing")
    tile_ir = str(metadata.get("tile_ir") or "")
    if digest_text(tile_ir) != metadata.get("tile_digest"):
        raise ValueError("FFT package Tile digest mismatch")
    schedule_digest = str(metadata.get("schedule_digest") or "")
    if _HASH_RE.findall(tile_ir) != [schedule_digest]:
        raise ValueError("FFT package schedule identity mismatch")
    for key in (
        "mode",
        "real_transform_policy",
        "hermitian_layout",
        "radix_policy",
        "strategy",
        "algorithm",
        "workspace_policy",
        "residency",
        "twiddle_policy",
        "kernel_family",
        "normalization",
    ):
        if f'{key} = "{metadata.get(key)}"' not in tile_ir:
            raise ValueError(f"FFT package Tile contract mismatch for {key}")
    for key in ("axis", "length", "physical_length", "batch", "bluestein_m", "workspace_elems"):
        if not re.search(rf"{key} = {int(metadata.get(key, -1))} : i64", tile_ir):
            raise ValueError(f"FFT package Tile contract mismatch for {key}")
    inverse = "true" if expected_inverse else "false"
    if f"inverse = {inverse}" not in tile_ir:
        raise ValueError("FFT package Tile direction mismatch")
    radix = ", ".join(str(int(value)) for value in metadata.get("radix_sequence") or ())
    expected_radix = (
        f"radix_sequence = array<i64: {radix}>"
        if radix
        else "radix_sequence = array<i64>"
    )
    if expected_radix not in tile_ir:
        raise ValueError("FFT package Tile radix sequence mismatch")
    if f'tessera.workgroup_size = {int(metadata.get("workgroup_size", -1))} : i64' not in tile_ir:
        raise ValueError("FFT package Tile workgroup mismatch")
    if any(op in tile_ir for op in ("schedule.fft", "schedule.artifact", *_FFT_OPS)):
        raise ValueError("FFT package re-entered Graph or Schedule IR")
    return metadata
