"""Content-addressed compound TSOL programs over canonical FFT artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .benchmark_row import MeasuredResourceVector
from .composition_cost import InferredActionDAG, infer_action_dag
from .graph_ir import GraphIRFunction, IRArg, IROp, IRType
from .schedule_object import ScheduleObject, ScheduleRole
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

_NORMALIZATIONS = ("backward", "forward", "ortho")
_STORAGE_POLICIES = ("f32", "f16", "bf16")


def _spectral_graph_function(
    *, object_id: str, op_name: str, dct_type: int
) -> tuple[GraphIRFunction, tuple[str, ...]]:
    """Represent the fused physical producer as registered Graph actions."""

    specs: tuple[tuple[str, str, tuple[str, ...]], ...]
    if op_name == "tessera.spectral_filter":
        specs = (
            ("complex_multiply", "tessera.mul", ("%input0", "%input1")),
            ("materialize_output", "tessera.reshape", ("%complex_multiply",)),
        )
    elif op_name == "tessera.dct" and dct_type == 2:
        specs = (
            ("even_extend", "tessera.pad", ("%input0",)),
            ("fft", "tessera.fft", ("%even_extend",)),
            ("phase_correct", "tessera.mul", ("%fft", "%phase")),
            ("crop", "tessera.slice", ("%phase_correct",)),
        )
    elif op_name == "tessera.dct":
        specs = (
            ("direct_cosine", "tessera.dct", ("%input0",)),
            ("materialize_output", "tessera.reshape", ("%direct_cosine",)),
        )
    elif op_name == "tessera.spectral_conv":
        specs = (
            ("pad_signal", "tessera.pad", ("%input0",)),
            ("pad_kernel", "tessera.pad", ("%input1",)),
            ("rfft_signal", "tessera.rfft", ("%pad_signal",)),
            ("rfft_kernel", "tessera.rfft", ("%pad_kernel",)),
            ("complex_multiply", "tessera.mul", ("%rfft_signal", "%rfft_kernel")),
            ("inverse", "tessera.irfft", ("%complex_multiply",)),
            ("crop", "tessera.slice", ("%inverse",)),
        )
    elif op_name == "tessera.stft":
        specs = (
            ("frame", "tessera.reshape", ("%input0",)),
            ("apply_window", "tessera.mul", ("%frame", "%input1")),
            ("transform", "tessera.rfft", ("%apply_window",)),
        )
    else:
        specs = (
            ("inverse", "tessera.irfft", ("%input0",)),
            ("apply_window", "tessera.mul", ("%inverse", "%input1")),
            ("overlap_add", "tessera.reduce_sum", ("%apply_window",)),
        )
    tensor = IRType("tensor<*xf32>")
    names = {operand.lstrip("%") for _, _, operands in specs for operand in operands}
    produced = {action_id for action_id, _, _ in specs}
    args = [IRArg(name, tensor) for name in sorted(names - produced)]
    ops = [
        IROp(
            result=f"%{action_id}",
            op_name=graph_op,
            operands=list(operands),
            operand_types=["tensor<*xf32>"] * len(operands),
            result_type="tensor<*xf32>",
        )
        for action_id, graph_op, operands in specs
    ]
    return (
        GraphIRFunction(
            name=object_id,
            args=args,
            body=ops,
            return_values=[f"%{specs[-1][0]}"],
        ),
        tuple(action_id for action_id, _, _ in specs),
    )


def infer_spectral_action_dag(
    *,
    semantic_digest: str,
    target: str,
    architecture: str,
    op_name: str,
    dct_type: int,
    workspace_bytes: int,
) -> tuple[InferredActionDAG, ScheduleObject]:
    """Infer the representative spectral producer DAG and bind SO identity."""

    object_id = f"spectral:{semantic_digest}"
    function, action_ids = _spectral_graph_function(
        object_id=object_id, op_name=op_name, dct_type=dct_type
    )
    bytes_per_action = workspace_bytes // max(1, len(action_ids))
    vectors = tuple(
        MeasuredResourceVector(
            compute_time_ms=1.0,
            bytes_moved=bytes_per_action,
            communication_bytes=0,
            queue_identity=f"{target}:spectral:0",
            resource_identity=architecture,
            timing_provenance={
                "source": "static_spectral_model",
                "domain": "compiler",
            },
            artifact_digest=digest_text(
                f"{semantic_digest}:{action_id}:{target}:{architecture}"
            ),
        ).as_dict()
        for action_id in action_ids
    )
    inferred = infer_action_dag(function, vectors, action_ids=action_ids)
    schedule = ScheduleObject(
        object_id=object_id,
        actions=inferred.actions,
        edges=inferred.schedule_object.edges,
        roles=(
            ScheduleRole("spectral_compute", (architecture,)),
            ScheduleRole("spectral_queue", (f"{target}:spectral:0",)),
        ),
    )
    return inferred, schedule


@dataclass(frozen=True)
class SpectralArchitectureProfile:
    target: str
    compiler_target: str
    architecture: str
    execution_status: str
    native_package_abi: str | None
    package_status: str
    reason: str


_ARCHITECTURE_PROFILES = {
    "x86": SpectralArchitectureProfile(
        "x86",
        "x86",
        "zen5-avx512",
        "ready",
        "tessera.x86.spectral_composite.v8",
        "exact_device_validated",
        "exact Zen 5 package",
    ),
    "rocm": SpectralArchitectureProfile(
        "rocm",
        "rocm",
        "gfx1151",
        "ready",
        "tessera.rocm.spectral_composite.v7",
        "exact_device_validated",
        "exact gfx1151 package",
    ),
    "rocm_gfx1151": SpectralArchitectureProfile(
        "rocm_gfx1151",
        "rocm",
        "gfx1151",
        "ready",
        "tessera.rocm.spectral_composite.v7",
        "exact_device_validated",
        "exact gfx1151 package",
    ),
    "rocm_gfx1200": SpectralArchitectureProfile(
        "rocm_gfx1200",
        "rocm",
        "gfx1200",
        "fail_closed",
        "tessera.rocm.spectral_composite.v7",
        "build_only",
        "architecture-stamped package exists; RDNA 4 schedule and exact-device "
        "evidence are required for execution",
    ),
    "rocm_gfx1250": SpectralArchitectureProfile(
        "rocm_gfx1250",
        "rocm",
        "gfx1250",
        "fail_closed",
        "tessera.rocm.spectral_composite.v7",
        "build_only",
        "architecture-stamped package exists; gfx1250 schedule and exact-device "
        "evidence are required for execution",
    ),
}


def spectral_architecture_profile(target: str) -> SpectralArchitectureProfile:
    try:
        return _ARCHITECTURE_PROFILES[target]
    except KeyError as error:
        raise ValueError(f"unsupported scheduled TSOL target {target!r}") from error


def spectral_output_scale(
    op_name: str, normalization: str, transform_length: int
) -> float:
    """Return the package-owned scale around the canonical backward FFT child."""
    if normalization not in _NORMALIZATIONS or transform_length <= 0:
        raise ValueError("invalid TSOL normalization scale request")
    if op_name in {"tessera.spectral_filter", "tessera.spectral_conv"}:
        return 1.0
    if normalization == "backward":
        return 1.0
    root = math.sqrt(float(transform_length))
    if op_name == "tessera.istft":
        return float(transform_length) if normalization == "forward" else root
    return 1.0 / float(transform_length) if normalization == "forward" else 1.0 / root


@dataclass(frozen=True)
class SpectralProgramContract:
    """Target-neutral TSOL shape and numeric contract.

    ``-1`` dimensions are bounded dynamic dimensions. Physical packages remain
    exact specializations: this contract validates a concrete shape before the
    content-addressed Schedule→Tile artifact is built. That makes dynamic-shape
    lineage explicit without pretending a static native image accepts an
    unbounded tensor or transferring x86/gfx1151 evidence to another target.
    """

    op_name: str
    input_signature: tuple[tuple[int, ...], ...]
    shape_bounds: tuple[tuple[int, ...], ...]
    axis: int
    dct_type: int
    storage: str
    normalization: str

    @property
    def shape_policy(self) -> str:
        return (
            "bounded_runtime_specialization_v1"
            if any(dim == -1 for shape in self.input_signature for dim in shape)
            else "exact_runtime_specialization_v1"
        )

    @property
    def template_digest(self) -> str:
        signature = "|".join(
            "x".join(str(dim) for dim in shape) for shape in self.input_signature
        )
        bounds = "|".join(
            "x".join(str(dim) for dim in shape) for shape in self.shape_bounds
        )
        return digest_text(
            f"schema=tessera.spectral_program_template.v1;op={self.op_name};"
            f"signature={signature};bounds={bounds};axis={self.axis};"
            f"dct_type={self.dct_type};"
            f"storage={self.storage};normalization={self.normalization}"
        )

    def specialize(
        self, input_shapes: Sequence[Sequence[int]]
    ) -> tuple[tuple[int, ...], ...]:
        concrete = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
        if len(concrete) != len(self.input_signature):
            raise ValueError("TSOL specialization input count mismatch")
        for index, (shape, signature, bounds) in enumerate(
            zip(concrete, self.input_signature, self.shape_bounds)
        ):
            if len(shape) != len(signature) or len(shape) != len(bounds):
                raise ValueError(f"TSOL specialization rank mismatch for input {index}")
            for dim, expected, bound in zip(shape, signature, bounds):
                if dim <= 0 or dim > bound or (expected != -1 and dim != expected):
                    raise ValueError(
                        f"TSOL specialization shape {shape!r} violates "
                        f"signature {signature!r} with bounds {bounds!r}"
                    )
        return concrete


def define_spectral_program_contract(
    *,
    op_name: str,
    input_signature: Sequence[Sequence[int | None]],
    shape_bounds: Sequence[Sequence[int]] | None = None,
    axis: int = -1,
    dct_type: int | None = None,
    storage: str = "f32",
    normalization: str = "backward",
) -> SpectralProgramContract:
    """Define a target-neutral, bounded TSOL specialization envelope."""
    if op_name not in _OPS:
        raise ValueError(f"unsupported scheduled TSOL operation {op_name!r}")
    signatures = tuple(
        tuple(-1 if dim is None else int(dim) for dim in shape)
        for shape in input_signature
    )
    expected_inputs = 1 if op_name == "tessera.dct" else 2
    if len(signatures) != expected_inputs or any(not shape for shape in signatures):
        raise ValueError(f"{op_name} requires {expected_inputs} non-scalar inputs")
    if any(dim == 0 or dim < -1 for shape in signatures for dim in shape):
        raise ValueError("TSOL shape signatures use positive extents or -1")
    selected_dct_type = int(dct_type if dct_type is not None else 2)
    if op_name == "tessera.dct":
        if selected_dct_type not in {1, 2, 3, 4}:
            raise ValueError("scheduled DCT requires dct_type in {1, 2, 3, 4}")
    elif dct_type is not None:
        raise ValueError("dct_type applies only to tessera.dct")
    else:
        selected_dct_type = 0
    if shape_bounds is None:
        if any(dim == -1 for shape in signatures for dim in shape):
            raise ValueError("dynamic TSOL dimensions require explicit shape bounds")
        bounds = signatures
    else:
        bounds = tuple(tuple(int(dim) for dim in shape) for shape in shape_bounds)
    if len(bounds) != len(signatures) or any(
        len(bound) != len(signature) for bound, signature in zip(bounds, signatures)
    ):
        raise ValueError("TSOL shape bounds must match every input signature")
    if any(dim <= 0 for shape in bounds for dim in shape):
        raise ValueError("TSOL shape bounds must be positive")
    for signature, bound in zip(signatures, bounds):
        if any(s != -1 and s != b for s, b in zip(signature, bound)):
            raise ValueError("static TSOL dimensions must equal their shape bounds")
    rank = len(signatures[0])
    normalized_axis = int(axis) if int(axis) >= 0 else rank + int(axis)
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ValueError(f"TSOL axis {axis} is invalid for rank {rank}")
    if storage not in _STORAGE_POLICIES:
        raise ValueError(f"unsupported TSOL storage policy {storage!r}")
    if normalization not in _NORMALIZATIONS:
        raise ValueError(f"unsupported TSOL normalization {normalization!r}")
    return SpectralProgramContract(
        op_name=op_name,
        input_signature=signatures,
        shape_bounds=bounds,
        axis=normalized_axis,
        dct_type=selected_dct_type,
        storage=storage,
        normalization=normalization,
    )


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
    input_signature: tuple[tuple[int, ...], ...]
    shape_bounds: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    axis: int
    dct_type: int
    shape_policy: str
    storage: str
    padding: tuple[int, int]
    crop: tuple[int, int]
    transform_length: int
    window_length: int
    window_broadcast: str
    hop: int
    frames: int
    center: bool
    onesided: bool
    pad_mode: str
    output_length: int
    normalization: str
    complex_layout: str
    accumulation: str
    workspace_bytes: int
    workspace_policy: str
    fusion_topology: str
    mutation_lineage: str
    native_entry: str
    child_ffts: tuple[Mapping[str, Any], ...]
    template_digest: str
    graph_analysis_digest: str
    schedule_object: Mapping[str, Any]
    schedule_digest: str

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    @property
    def abi_storage(self) -> str:
        return "f32"

    @property
    def storage_conversion(self) -> str:
        return (
            "native_f32"
            if self.storage == "f32"
            else "native_package_cast_f32_accumulate_cast_output_v1"
        )

    @property
    def numeric_policy(self) -> dict[str, str]:
        return {
            "storage": {"f32": "fp32", "f16": "fp16", "bf16": "bf16"}[
                self.storage
            ],
            "accum": "fp32",
        }

    @property
    def axis_packing(self) -> str:
        if self.op_name in {"tessera.stft", "tessera.istft"}:
            return "native_runtime_stride_descriptor_v1"
        return (
            "none_contiguous"
            if self.axis == len(self.input_shapes[0]) - 1
            else "native_package_host_pack_v1"
        )

    def _input_shapes_text(self) -> str:
        return "|".join(
            "x".join(str(dim) for dim in shape) for shape in self.input_shapes
        )

    def _child_digests_text(self) -> str:
        return ",".join(str(child["schedule_digest"]) for child in self.child_ffts)

    @staticmethod
    def _shapes_text(shapes: Sequence[Sequence[int]]) -> str:
        return "|".join("x".join(str(dim) for dim in shape) for shape in shapes)

    def _identity_payload(self) -> str:
        output = "x".join(str(dim) for dim in self.output_shape)
        return (
            f"schema=tessera.scheduled_spectral.v7;op={self.op_name};"
            f"target={self.target};arch={self.architecture};"
            f"inputs={self._input_shapes_text()};output={output};axis={self.axis};"
            f"dct_type={self.dct_type};"
            f"shape_policy={self.shape_policy};storage={self.storage};"
            f"abi_storage={self.abi_storage};storage_conversion={self.storage_conversion};"
            f"axis_packing={self.axis_packing};"
            f"input_signature={self._shapes_text(self.input_signature)};"
            f"shape_bounds={self._shapes_text(self.shape_bounds)};"
            f"template_digest={self.template_digest};"
            f"padding={self.padding[0]},{self.padding[1]};"
            f"crop={self.crop[0]},{self.crop[1]};n_fft={self.transform_length};"
            f"window={self.window_length};"
            f"window_broadcast={self.window_broadcast};"
            f"hop={self.hop};frames={self.frames};center={int(self.center)};"
            f"onesided={int(self.onesided)};"
            f"pad_mode={self.pad_mode};output_length={self.output_length};"
            f"normalization={self.normalization};"
            f"complex_layout={self.complex_layout};"
            f"numeric_storage={self.numeric_policy['storage']};"
            f"numeric_accum={self.numeric_policy['accum']};"
            f"accumulation={self.accumulation};"
            f"workspace_bytes={self.workspace_bytes};"
            f"workspace_policy={self.workspace_policy};"
            f"fusion_topology={self.fusion_topology};"
            f"mutation_lineage={self.mutation_lineage};native_entry={self.native_entry};"
            f"child_fft_digests={self._child_digests_text()};"
            f"workgroup={1 if self.target == 'x86' else 256}"
        )

    def _identity(self) -> dict[str, Any]:
        return {
            "schema": "tessera.scheduled_spectral.v7",
            "op_name": self.op_name,
            "target": self.target,
            "architecture": self.architecture,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "input_signature": [list(shape) for shape in self.input_signature],
            "shape_bounds": [list(shape) for shape in self.shape_bounds],
            "output_shape": list(self.output_shape),
            "axis": self.axis,
            "dct_type": self.dct_type,
            "shape_policy": self.shape_policy,
            "storage": self.storage,
            "abi_storage": self.abi_storage,
            "storage_conversion": self.storage_conversion,
            "axis_packing": self.axis_packing,
            "padding": list(self.padding),
            "crop": list(self.crop),
            "transform_length": self.transform_length,
            "window_length": self.window_length,
            "window_broadcast": self.window_broadcast,
            "hop": self.hop,
            "frames": self.frames,
            "center": self.center,
            "onesided": self.onesided,
            "pad_mode": self.pad_mode,
            "output_length": self.output_length,
            "normalization": self.normalization,
            "complex_layout": self.complex_layout,
            "numeric_policy": self.numeric_policy,
            "accumulation": self.accumulation,
            "workspace_bytes": self.workspace_bytes,
            "workspace_policy": self.workspace_policy,
            "fusion_topology": self.fusion_topology,
            "mutation_lineage": self.mutation_lineage,
            "native_entry": self.native_entry,
            "child_fft_digests": [
                str(child["schedule_digest"]) for child in self.child_ffts
            ],
            "template_digest": self.template_digest,
        }

    def validate(self) -> None:
        if self.op_name not in _OPS:
            raise ValueError("TSOL package operation identity mismatch")
        if (self.target, self.architecture) not in {
            ("rocm", "gfx1151"),
            ("x86", "zen5-avx512"),
        }:
            raise ValueError("TSOL package requires exact gfx1151 or Zen 5 AVX-512")
        semantic_digest = digest_text(self._identity_payload())
        if self.schedule_object.get("object_id") != f"spectral:{semantic_digest}":
            raise ValueError("TSOL Schedule Object semantic identity mismatch")
        encoded_schedule = json.dumps(
            self.schedule_object, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if hashlib.sha256(encoded_schedule).hexdigest() != self.schedule_digest:
            raise ValueError("TSOL Schedule Object content identity mismatch")
        if len(self.graph_analysis_digest) != 64:
            raise ValueError("TSOL inferred Graph analysis identity mismatch")
        for child in self.child_ffts:
            validate_scheduled_fft_metadata(
                child, target=self.target, input_shape=child["input_shape"]
            )
        if self.complex_layout != "interleaved_f32x2":
            raise ValueError("TSOL package complex layout mismatch")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError("TSOL package normalization mismatch")
        if self.storage not in _STORAGE_POLICIES or self.shape_policy not in {
            "exact_runtime_specialization_v1",
            "bounded_runtime_specialization_v1",
        }:
            raise ValueError("TSOL physical package policy is not evidence-backed")
        if self.numeric_policy["accum"] != "fp32":
            raise ValueError("TSOL physical package requires f32 accumulation")
        semantic = define_spectral_program_contract(
            op_name=self.op_name,
            input_signature=self.input_signature,
            shape_bounds=self.shape_bounds,
            axis=self.axis,
            dct_type=self.dct_type if self.op_name == "tessera.dct" else None,
            storage=self.storage,
            normalization=self.normalization,
        )
        semantic.specialize(self.input_shapes)
        if semantic.template_digest != self.template_digest:
            raise ValueError("TSOL package template identity mismatch")
        if self.workspace_policy != "persistent_artifact_workspace":
            raise ValueError("TSOL package workspace policy mismatch")
        expected_fusion = {
            "tessera.spectral_filter": "standalone_complex_multiply_v1",
            "tessera.dct": (
                "even_extension_c2c_phase_corrected_v2"
                if self.dct_type == 2
                else "direct_cosine_kernel_v1"
            ),
            "tessera.spectral_conv": "packed_rfft_cmul_irfft_single_artifact_v1",
            "tessera.stft": (
                "frame_window_packed_rfft_single_artifact_v2"
                if self.onesided else "frame_window_c2c_single_artifact_v1"
            ),
            "tessera.istft": (
                "packed_irfft_window_ola_single_artifact_v2"
                if self.onesided else "c2c_ifft_window_ola_single_artifact_v1"
            ),
        }[self.op_name]
        if self.fusion_topology != expected_fusion:
            raise ValueError("TSOL package fusion topology mismatch")
        if self.mutation_lineage != "inputs_immutable_output_fresh_v1":
            raise ValueError("TSOL package mutation lineage mismatch")
        if self.schedule_ir.count("schedule.spectral_program") != 1:
            raise ValueError("TSOL package requires one Schedule program edge")
        if self.schedule_ir.count("schedule.artifact") != 1:
            raise ValueError("TSOL package requires one durable schedule artifact")
        if self.tile_ir.count("tile.spectral_program_kernel") != 1:
            raise ValueError("TSOL package requires one launch-level Tile program")
        if any(
            name in self.tile_ir
            for name in ("schedule.spectral_program", "schedule.artifact")
        ):
            raise ValueError("TSOL Tile package retained Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("TSOL Tile package has stale schedule identity")
        if f"dct_type = {self.dct_type} : i64" not in self.tile_ir:
            raise ValueError("TSOL Tile package has stale DCT type identity")

    def to_metadata(self) -> dict[str, Any]:
        result = self._identity()
        result.update(
            {
                "child_ffts": [dict(child) for child in self.child_ffts],
                "schedule_digest": self.schedule_digest,
                "graph_analysis_digest": self.graph_analysis_digest,
                "schedule_object": dict(self.schedule_object),
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
    dct_type: int | None = None,
    input_signature: tuple[tuple[int | None, ...], ...] | None = None,
    shape_bounds: tuple[tuple[int, ...], ...] | None = None,
    storage: str = "f32",
    normalization: str = "backward",
    center: bool = False,
    pad_mode: str = "constant",
    output_length: int | None = None,
    n_fft: int | None = None,
    onesided: bool = True,
) -> ScheduledSpectralArtifact:
    profile = spectral_architecture_profile(target)
    if profile.execution_status != "ready":
        raise ValueError(
            f"{profile.architecture} TSOL profile fails closed: {profile.reason}"
        )
    compiler_target = profile.compiler_target
    architecture = profile.architecture
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
    semantic_contract = define_spectral_program_contract(
        op_name=op_name,
        input_signature=input_signature or shapes,
        shape_bounds=shape_bounds,
        axis=axis,
        dct_type=dct_type,
        storage=storage,
        normalization=normalization,
    )
    semantic_contract.specialize(shapes)
    normalized_axis = semantic_contract.axis
    if op_name == "tessera.spectral_filter" and storage != "f32":
        raise ValueError("spectral_filter requires interleaved complex f32 storage")

    padding = (0, 0)
    crop = (0, 0)
    win = 0
    fft_n = 0
    stride = 0
    frames = 0
    children: tuple[Mapping[str, Any], ...] = ()
    accumulation = "f32"
    policy = "persistent_artifact_workspace"
    fusion_topology = ""
    centered = bool(center)
    selected_pad_mode = str(pad_mode)
    selected_output_length = -1
    window_broadcast = "not_applicable"
    if selected_pad_mode not in {"constant", "reflect"}:
        raise ValueError("scheduled STFT pad_mode must be constant or reflect")
    if op_name != "tessera.stft" and selected_pad_mode != "constant":
        raise ValueError("pad_mode applies only to scheduled STFT")
    if op_name not in {"tessera.stft", "tessera.istft"} and centered:
        raise ValueError("center applies only to scheduled STFT/ISTFT")
    if op_name != "tessera.istft" and output_length is not None:
        raise ValueError("output_length applies only to scheduled ISTFT")
    if op_name not in {"tessera.stft", "tessera.istft"} and n_fft is not None:
        raise ValueError("n_fft applies only to scheduled STFT/ISTFT")
    if op_name not in {"tessera.stft", "tessera.istft"} and not onesided:
        raise ValueError("onesided applies only to scheduled STFT/ISTFT")

    if op_name == "tessera.spectral_filter":
        fusion_topology = "standalone_complex_multiply_v1"
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
        fusion_topology = (
            "even_extension_c2c_phase_corrected_v2"
            if semantic_contract.dct_type == 2
            else "direct_cosine_kernel_v1"
        )
        if len(shapes) != 1:
            raise ValueError("dct requires one input")
        n = shapes[0][normalized_axis]
        batch_shape = shapes[0][:normalized_axis] + shapes[0][normalized_axis + 1 :]
        if semantic_contract.dct_type == 2:
            child_shape = (*batch_shape, 2 * n)
            children = (_child(compiler_target, "tessera.fft", child_shape),)
        output = shapes[0]
        padding = (0, n)
        crop = (0, n)
        elements = math.prod(batch_shape or (1,)) * n
        workspace = (
            _packed_workspace_bytes(
                (4, elements),
                (4, elements),
                (8, 2 * elements),
                (8, 2 * elements),
            )
            if semantic_contract.dct_type == 2
            else _packed_workspace_bytes((4, elements), (4, elements))
        )
        entry = (
            "tessera_x86_dct_strided_storage"
            if compiler_target == "x86"
            else "ts_dct_plan_hostptr_strided_storage_amd"
        )
    elif op_name == "tessera.spectral_conv":
        fusion_topology = "packed_rfft_cmul_irfft_single_artifact_v1"
        if len(shapes) != 2 or len(shapes[0]) != len(shapes[1]):
            raise ValueError("spectral_conv requires equal input ranks")
        x_batch = shapes[0][:normalized_axis] + shapes[0][normalized_axis + 1 :]
        w_batch = shapes[1][:normalized_axis] + shapes[1][normalized_axis + 1 :]
        if x_batch != w_batch:
            raise ValueError("spectral_conv requires matching static batch dimensions")
        output_n = shapes[0][normalized_axis] + shapes[1][normalized_axis] - 1
        fft_n = next_power_of_two(output_n)
        rows = math.prod(x_batch or (1,))
        child_shape = (rows, fft_n)
        children = (
            _child(compiler_target, "tessera.rfft", child_shape),
            _child(
                compiler_target,
                "tessera.irfft",
                (rows, fft_n // 2 + 1),
                n=fft_n,
            ),
        )
        output = (
            shapes[0][:normalized_axis] + (output_n,) + shapes[0][normalized_axis + 1 :]
        )
        padding = (
            fft_n - shapes[0][normalized_axis],
            fft_n - shapes[1][normalized_axis],
        )
        crop = (0, fft_n - output_n)
        # Host staging, two padded-real inputs, three packed spectra, and the
        # packed inverse's real output. No full-complex N-point intermediate.
        workspace = _packed_workspace_bytes(
            (4, rows * shapes[0][normalized_axis]),
            (4, rows * shapes[1][normalized_axis]),
            (4, rows * output_n),
            (4, rows * fft_n),
            (4, rows * fft_n),
            *((8, rows * (fft_n // 2 + 1)),) * 3,
            (4, rows * fft_n),
        )
        entry = (
            "tessera_x86_spectral_conv_strided_storage"
            if compiler_target == "x86"
            else "ts_spectral_conv_plan_hostptr_strided_storage_amd"
        )
    elif op_name == "tessera.stft":
        if len(shapes) != 2 or len(shapes[1]) < 1:
            raise ValueError("stft requires a signal and trailing-dimension window")
        win = shapes[1][-1]
        fft_n = int(n_fft if n_fft is not None else win)
        if fft_n < win:
            raise ValueError("stft requires n_fft >= window length")
        fusion_topology = (
            "frame_window_packed_rfft_single_artifact_v2"
            if onesided else "frame_window_c2c_single_artifact_v1"
        )
        stride = int(hop or 0)
        samples = shapes[0][normalized_axis]
        if stride <= 0:
            raise ValueError("stft requires a positive hop")
        pad = fft_n // 2 if centered else 0
        if centered and selected_pad_mode == "reflect" and samples <= pad:
            raise ValueError("centered reflect STFT requires signal length > n_fft/2")
        framed_samples = max(samples + 2 * pad, fft_n)
        frames = (framed_samples - fft_n) // stride + 1
        padding = (pad, pad)
        batch_shape = shapes[0][:normalized_axis] + shapes[0][normalized_axis + 1 :]
        window_batch = shapes[1][:-1]
        if len(window_batch) > len(batch_shape):
            raise ValueError("stft window batch rank exceeds signal batch rank")
        aligned_window_batch = (1,) * (len(batch_shape) - len(window_batch)) + window_batch
        if any(w not in {1, b} for w, b in zip(aligned_window_batch, batch_shape)):
            raise ValueError("stft window batch dimensions are not broadcastable")
        window_broadcast = "trailing_batch_broadcast_v1"
        batch = math.prod(batch_shape or (1,))
        child_op = "tessera.rfft" if onesided else "tessera.fft"
        children = (_child(compiler_target, child_op, (batch * frames, fft_n)),)
        bins = fft_n // 2 + 1 if onesided else fft_n
        output = (
            shapes[0][:normalized_axis]
            + (frames, bins)
            + shapes[0][normalized_axis + 1 :]
        )
        workspace = (
            _packed_workspace_bytes(
                (4, batch * framed_samples),
                (4, math.prod(shapes[1])),
                (4, batch * frames * fft_n),
                (8, batch * frames * bins),
            )
            if onesided
            else _packed_workspace_bytes(
                (4, batch * framed_samples),
                (4, math.prod(shapes[1])),
                (8, batch * frames * fft_n),
                (8, batch * frames * fft_n),
                (8, batch * frames * bins),
            )
        )
        entry = (
            "tessera_x86_stft_policy_broadcast_layout_storage"
            if compiler_target == "x86"
            else "ts_stft_plan_hostptr_broadcast_layout_storage_amd"
        )
    else:
        if len(shapes) != 2 or len(shapes[1]) < 1:
            raise ValueError("istft requires spectra and a trailing-dimension window")
        win = shapes[1][-1]
        fft_n = int(n_fft if n_fft is not None else win)
        if fft_n < win:
            raise ValueError("istft requires n_fft >= window length")
        fusion_topology = (
            "packed_irfft_window_ola_single_artifact_v2"
            if onesided else "c2c_ifft_window_ola_single_artifact_v1"
        )
        stride = int(hop or 0)
        if normalized_axis <= 0:
            raise ValueError("istft frequency axis requires a preceding frame axis")
        frame_axis = normalized_axis - 1
        frames = shapes[0][frame_axis]
        expected_bins = fft_n // 2 + 1 if onesided else fft_n
        if stride <= 0 or shapes[0][normalized_axis] != expected_bins:
            raise ValueError("istft spectrum/window/hop contract mismatch")
        batch_shape = tuple(
            dim
            for index, dim in enumerate(shapes[0])
            if index not in {frame_axis, normalized_axis}
        )
        window_batch = shapes[1][:-1]
        if len(window_batch) > len(batch_shape):
            raise ValueError("istft window batch rank exceeds spectrum batch rank")
        aligned_window_batch = (1,) * (len(batch_shape) - len(window_batch)) + window_batch
        if any(w not in {1, b} for w, b in zip(aligned_window_batch, batch_shape)):
            raise ValueError("istft window batch dimensions are not broadcastable")
        window_broadcast = "trailing_batch_broadcast_v1"
        batch = math.prod(batch_shape or (1,))
        children = (
            _child(
                compiler_target,
                "tessera.irfft" if onesided else "tessera.ifft",
                (batch * frames, expected_bins),
                n=fft_n,
            ),
        )
        raw_samples = (frames - 1) * stride + fft_n
        pad = fft_n // 2 if centered else 0
        available = raw_samples - 2 * pad
        if available <= 0:
            raise ValueError("centered ISTFT has no samples after trimming")
        selected_output_length = (
            available if output_length is None else int(output_length)
        )
        if selected_output_length <= 0 or selected_output_length > available:
            raise ValueError(
                "scheduled ISTFT output_length must crop within the available output"
            )
        crop = (pad, raw_samples - pad - selected_output_length)
        output = (
            shapes[0][:frame_axis]
            + (selected_output_length,)
            + shapes[0][normalized_axis + 1 :]
        )
        workspace = (
            _packed_workspace_bytes(
                (4, math.prod(shapes[1])),
                (4, batch * raw_samples),
                (8, batch * frames * expected_bins),
                (4, batch * frames * fft_n),
            )
            if onesided
            else _packed_workspace_bytes(
                (4, math.prod(shapes[1])),
                (4, batch * raw_samples),
                (8, batch * frames * expected_bins),
                (8, batch * frames * fft_n),
                (8, batch * frames * fft_n),
            )
        )
        accumulation = "deterministic_f32_ascending_frames"
        entry = (
            "tessera_x86_istft_policy_broadcast_layout_storage"
            if compiler_target == "x86"
            else "ts_istft_plan_hostptr_broadcast_layout_storage_amd"
        )

    provisional = ScheduledSpectralArtifact(
        schedule_ir="",
        tile_ir="",
        op_name=op_name,
        target=compiler_target,
        architecture=architecture,
        input_shapes=shapes,
        input_signature=semantic_contract.input_signature,
        shape_bounds=semantic_contract.shape_bounds,
        output_shape=output,
        axis=normalized_axis,
        dct_type=semantic_contract.dct_type,
        shape_policy=semantic_contract.shape_policy,
        storage=semantic_contract.storage,
        padding=padding,
        crop=crop,
        transform_length=fft_n,
        window_length=win,
        window_broadcast=window_broadcast,
        hop=stride,
        frames=frames,
        center=centered,
        onesided=bool(onesided),
        pad_mode=selected_pad_mode,
        output_length=selected_output_length,
        normalization=semantic_contract.normalization,
        complex_layout="interleaved_f32x2",
        accumulation=accumulation,
        workspace_bytes=workspace,
        workspace_policy=policy,
        fusion_topology=fusion_topology,
        mutation_lineage="inputs_immutable_output_fresh_v1",
        native_entry=entry,
        child_ffts=children,
        template_digest=semantic_contract.template_digest,
        graph_analysis_digest="",
        schedule_object={},
        schedule_digest="",
    )
    identity = provisional._identity_payload()
    semantic_digest = digest_text(identity)
    inferred, schedule_object = infer_spectral_action_dag(
        semantic_digest=semantic_digest,
        target=compiler_target,
        architecture=architecture,
        op_name=op_name,
        dct_type=semantic_contract.dct_type,
        workspace_bytes=workspace,
    )
    schedule_digest = schedule_object.digest
    input_names = tuple(f"a{index}" for index in range(len(shapes)))
    real_element = semantic_contract.storage
    input_elements = (
        ("complex<f32>", "complex<f32>")
        if op_name == "tessera.spectral_filter"
        else (
            ("complex<f32>", real_element)
            if op_name == "tessera.istft"
            else tuple(real_element for _ in shapes)
        )
    )
    output_element = (
        "complex<f32>"
        if op_name in {"tessera.spectral_filter", "tessera.stft"}
        else real_element
    )
    input_types = tuple(
        "tensor<" + "x".join([*(str(dim) for dim in shape), element]) + ">"
        for shape, element in zip(shapes, input_elements)
    )
    output_type = (
        "tensor<" + "x".join([*(str(dim) for dim in output), output_element]) + ">"
    )
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
        f'input_signature = "{provisional._shapes_text(provisional.input_signature)}", '
        f'shape_bounds = "{provisional._shapes_text(provisional.shape_bounds)}", '
        f'template_digest = "{provisional.template_digest}", '
        f"output_shape = array<i64: {output_shape_text}>, axis = {normalized_axis} : i64, "
        f"dct_type = {semantic_contract.dct_type} : i64, "
        f'shape_policy = "{semantic_contract.shape_policy}", storage = "{semantic_contract.storage}", '
        f'abi_storage = "{provisional.abi_storage}", '
        f'storage_conversion = "{provisional.storage_conversion}", '
        f'axis_packing = "{provisional.axis_packing}", '
        f"padding = array<i64: {padding_text}>, crop = array<i64: {crop_text}>, "
        f"transform_length = {fft_n} : i64, window_length = {win} : i64, "
        f'window_broadcast = "{window_broadcast}", '
        f"hop = {stride} : i64, frames = {frames} : i64, "
        f"center = {str(centered).lower()}, onesided = {str(bool(onesided)).lower()}, "
        f"pad_mode = \"{selected_pad_mode}\", "
        f"output_length = {selected_output_length} : i64, "
        f'normalization = "{semantic_contract.normalization}", complex_layout = "interleaved_f32x2", '
        f'numeric_policy = {{storage = "{provisional.numeric_policy["storage"]}", '
        f'accum = "{provisional.numeric_policy["accum"]}"}}, '
        f'accumulation = "{accumulation}", workspace_bytes = {workspace} : i64, '
        f'workspace_policy = "{policy}", mutation_lineage = "inputs_immutable_output_fresh_v1", '
        f'fusion_topology = "{fusion_topology}", '
        f'native_entry = "{entry}", child_fft_digests = "{child_digests}", '
        f'workgroup_size = {1 if compiler_target == "x86" else 256} : i64'
    )
    # Carry the exact semantic payload the physical launch consumes,
    # so the Schedule->Tile consumer can re-verify the attributes it
    # uses instead of trusting a digest string alone (PR #626 review).
    # Carry the two preimages that make the consumer's verification a CHAIN
    # rather than a string comparison (PR #626 follow-up):
    #   sha256(schedule_payload) == tessera.schedule_digest, and that payload
    #   contains object_id "spectral:<sha256(identity)>", and the identity
    #   text carries the policy fields the launch consumes.
    # Editing any link breaks a hash, so the attribute and its declaration
    # can no longer be co-edited.
    def _mlir_escape(text: str) -> str:
        return text.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))

    semantic_payload = _mlir_escape(identity)
    schedule_payload = _mlir_escape(
        json.dumps(
            schedule_object.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    schedule_ir = (
        f'module attributes {{tessera.target = "{compiler_target}", '
        f'tessera.arch = "{architecture}", '
        f'tessera.schedule_digest = "{schedule_digest}", tessera.spectral_semantic = "{semantic_payload}", tessera.schedule_payload = "{schedule_payload}"}} {{\n'
        f"  func.func @scheduled_spectral({function_args}) -> {output_type} {{\n"
        f'    %result = "schedule.spectral_program"({operands}) {{{attrs}}} : '
        f"({operand_types}) -> {output_type}\n"
        f'    "schedule.artifact"() {{hash = "{schedule_digest}", '
        f'arch = "{architecture}", shape_key = "{provisional._input_shapes_text()}", '
        # NUMPOL-CARRIER-1 (queue row 3b): this used to emit
        #     numeric_policy = "f32;ortho"
        # a StringAttr holding a private semicolon-delimited encoding, under
        # the name of a well-defined DictionaryAttr. Two consequences, both
        # measured: every numeric_policy consumer reads the attribute with
        # `getAttrOfType<DictionaryAttr>`, which returns null for a string —
        # so this contract was silently skipped by the schema validator and
        # unreadable by the accumulator consumers. And it is not a Decision
        # #15a numeric_policy in the first place: `accumulation` can be
        # "deterministic_f32_ascending_frames", a reduction-ORDER contract,
        # not a dtype.
        #
        # So it is renamed rather than reshaped. Squeezing an order contract
        # into the dtype attribute would make the collision permanent; giving
        # it its own name makes both readable. The schedule digest is computed
        # from `schedule_object.digest` and not from this text, so identities
        # are unchanged.
        f'tessera.spectral_accumulation = "{accumulation}", '
        f'tessera.spectral_normalization = "{semantic_contract.normalization}"}} : () -> ()\n'
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
            "graph_analysis_digest": inferred.graph_analysis_digest,
            "schedule_object": schedule_object.canonical_payload(),
            "schedule_digest": schedule_digest,
        }
    )
    artifact.validate()
    return artifact


def validate_scheduled_spectral_metadata(
    metadata: Mapping[str, Any], *, input_shapes: Sequence[Sequence[int]]
) -> Mapping[str, Any]:
    if metadata.get("schema") != "tessera.scheduled_spectral.v7":
        raise ValueError("TSOL package requires tessera.scheduled_spectral.v7 metadata")
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
    declared_shapes = tuple(
        tuple(int(dim) for dim in shape) for shape in metadata.get("input_shapes") or ()
    )
    signature = tuple(
        tuple(int(dim) for dim in shape)
        for shape in metadata.get("input_signature") or ()
    )
    bounds = tuple(
        tuple(int(dim) for dim in shape) for shape in metadata.get("shape_bounds") or ()
    )
    semantic = define_spectral_program_contract(
        op_name=str(metadata.get("op_name")),
        input_signature=signature,
        shape_bounds=bounds,
        axis=int(metadata.get("axis", -1)),
        dct_type=int(metadata.get("dct_type", 0)) or None,
        storage=str(metadata.get("storage", "f32")),
        normalization=str(metadata.get("normalization", "backward")),
    )
    semantic.specialize(declared_shapes)
    declared = lower_scheduled_spectral(
        target=str(metadata.get("target")),
        op_name=str(metadata.get("op_name")),
        input_shapes=declared_shapes,
        axis=int(metadata.get("axis", -1)),
        hop=int(metadata.get("hop", 0)) or None,
        dct_type=semantic.dct_type if semantic.op_name == "tessera.dct" else None,
        input_signature=semantic.input_signature,
        shape_bounds=semantic.shape_bounds,
        storage=semantic.storage,
        normalization=semantic.normalization,
        center=bool(metadata.get("center", False)),
        pad_mode=str(metadata.get("pad_mode", "constant")),
        output_length=(
            int(metadata["output_length"])
            if int(metadata.get("output_length", -1)) >= 0
            else None
        ),
        n_fft=(
            int(metadata["transform_length"])
            if str(metadata.get("op_name")) in {"tessera.stft", "tessera.istft"}
            else None
        ),
        onesided=bool(metadata.get("onesided", True)),
    )
    declared_metadata = declared.to_metadata()
    for key, value in declared_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"TSOL package contract mismatch for {key}")
    if shapes == declared_shapes:
        return declared_metadata

    semantic.specialize(shapes)
    expected = lower_scheduled_spectral(
        target=str(metadata.get("target")),
        op_name=str(metadata.get("op_name")),
        input_shapes=shapes,
        axis=int(metadata.get("axis", -1)),
        hop=int(metadata.get("hop", 0)) or None,
        dct_type=semantic.dct_type if semantic.op_name == "tessera.dct" else None,
        input_signature=semantic.input_signature,
        shape_bounds=semantic.shape_bounds,
        storage=semantic.storage,
        normalization=semantic.normalization,
        center=bool(metadata.get("center", False)),
        pad_mode=str(metadata.get("pad_mode", "constant")),
        output_length=(
            int(metadata["output_length"])
            if int(metadata.get("output_length", -1)) >= 0
            else None
        ),
        n_fft=(
            int(metadata["transform_length"])
            if str(metadata.get("op_name")) in {"tessera.stft", "tessera.istft"}
            else None
        ),
        onesided=bool(metadata.get("onesided", True)),
    ).to_metadata()
    return expected
