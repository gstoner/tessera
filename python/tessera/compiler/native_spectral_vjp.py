"""Content-addressed native packages for compound spectral reverse products.

The compiler already owns ``tessera.spectral_backward`` and its
Schedule->Tile lowering.  This module connects that carrier to public
``JitFn.native_backward`` without reconstructing a forward Graph in runtime.
The initial physical envelope intentionally matches the C++ verifier:

* identical complex64 tensors for ``spectral_filter``;
* unbroadcast, equal-rank float32 full convolution on the final axis.
"""

from __future__ import annotations

import ctypes
import base64
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


_SCHEMA = "tessera.native_spectral_vjp.v2"
_MUTATION = "inputs_immutable_outputs_fresh_v1"
_AXIS_PACKING = "native_runtime_stride_descriptor_v1"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tensor_type(value: Any) -> str:
    import numpy as np

    array = np.asarray(value)
    element = {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "complex64": "complex<f32>",
    }.get(str(array.dtype))
    if element is None:
        raise ValueError(
            "native compound spectral VJP supports f16/bf16/f32 or complex64 storage"
        )
    dimensions = "x".join(str(int(dim)) for dim in array.shape)
    return f"tensor<{dimensions}x{element}>"


def _policy_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0"}:
            return False
        if lowered in {"true", "1"}:
            return True
        raise ValueError(f"invalid boolean spectral policy {value!r}")
    return bool(value)


def _algorithm_identity(
    kind: str, logical_length: int, target: str, spectrum_layout: str
) -> str:
    if spectrum_layout == "full_complex" and kind in {
        "tessera.stft", "tessera.istft"
    }:
        return "full_complex_direct_dft_v1"
    if kind == "tessera.stft":
        if target == "rocm":
            return "direct_stored_bin_gfx1151_v1"
        if target == "nvidia_sm120":
            return "direct_stored_bin_sm120_v1"
        return (
            "packed_c2r_stored_bin_v1"
            if logical_length % 2 == 0
            else "direct_stored_bin_odd_tail_v1"
        )
    if kind == "tessera.istft":
        if target == "rocm":
            return "normalized_overlap_add_direct_dft_gfx1151_v1"
        if target == "nvidia_sm120":
            return "normalized_overlap_add_direct_dft_sm120_v1"
        return "normalized_overlap_add_r2c_v1"
    return "native_direct_v1"


@dataclass(frozen=True)
class NativeSpectralVJPPackage:
    schema: str
    target: str
    arch: str
    kind: str
    axis: int
    logical_length: int
    normalization: str
    spectrum_layout: str
    axis_packing: str
    window_broadcast: str
    hop: int
    center: bool
    onesided: bool
    pad_mode: str
    output_length: int
    algorithm: str
    numeric_storage: str
    numeric_accum: str
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source_graph_ir: str
    source_graph_ir_digest: str
    schedule_artifact_hash: str
    tile_program_digest: str
    native_image: bytes | None = None
    native_symbol: str | None = None

    def contract(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target": self.target,
            "arch": self.arch,
            "kind": self.kind,
            "axis": self.axis,
            "logical_length": self.logical_length,
            "normalization": self.normalization,
            "spectrum_layout": self.spectrum_layout,
            "axis_packing": self.axis_packing,
            "window_broadcast": self.window_broadcast,
            "hop": self.hop,
            "center": self.center,
            "onesided": self.onesided,
            "pad_mode": self.pad_mode,
            "output_length": self.output_length,
            "algorithm": self.algorithm,
            "numeric_policy": {
                "storage": self.numeric_storage,
                "accum": self.numeric_accum,
            },
            "input_types": list(self.input_types),
            "output_types": list(self.output_types),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "source_graph_ir": self.source_graph_ir,
            "source_graph_ir_digest": self.source_graph_ir_digest,
            "schedule_artifact_hash": self.schedule_artifact_hash,
            "tile_program_digest": self.tile_program_digest,
            "mutation_lineage": _MUTATION,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        path = f"{self.target}_spectral_backward_compiled"
        metadata = {
            "target": self.target,
            "compiler_path": path,
            "executable": True,
            "execution_kind": "native_cpu" if self.target == "x86" else "native_gpu",
            "execution_mode": ("cpu_avx512" if self.target == "x86" else
                               "cuda_runtime" if self.target == "nvidia_sm120" else
                               "hip_runtime"),
            "arg_names": ["dy", *self.input_names],
            "output_names": list(self.output_names),
            "native_spectral_vjp": self.contract(),
        }
        if self.native_image is not None:
            metadata["native_image_b64"] = base64.b64encode(
                self.native_image
            ).decode("ascii")
            metadata["native_symbol"] = self.native_symbol
        return metadata


def _spectral_schedule_hash(
    *,
    kind: str,
    target: str,
    arch: str,
    input_types: Sequence[str],
    output_types: Sequence[str],
    axis: int,
    logical_length: int,
    normalization: str,
    spectrum_layout: str,
    hop: int = -1,
    center: bool = False,
    onesided: bool = True,
    pad_mode: str = "constant",
    output_length: int = -1,
    numeric_storage: str = "fp32",
    numeric_accum: str = "fp32",
    window_broadcast: str = "not_applicable",
) -> str:
    # Keep byte-for-byte parity with spectralBackwardDigest() in PMPasses.cpp.
    contract = (
        f"schema=tessera.spectral_backward.v3;kind={kind};target={target};"
        f"arch={arch};inputs={','.join(input_types)};"
        f"outputs={','.join(output_types)};axis={axis};"
        f"logical_length={logical_length};normalization={normalization};"
        f"spectrum_layout={spectrum_layout};axis_packing={_AXIS_PACKING};"
        f"hop={hop};center={int(center)};"
        f"onesided={int(onesided)};pad_mode={pad_mode};"
       f"output_length={output_length};mutation_lineage={_MUTATION}"
       f";numeric_storage={numeric_storage};numeric_accum={numeric_accum}"
       f";window_broadcast={window_broadcast}"
    )
    return hashlib.sha256(contract.encode()).hexdigest()


def build_native_spectral_vjp_package(
    *,
    source_graph_ir: str,
    source: Any,
    target: str,
    ordered_inputs: Sequence[Any],
    arg_names: Sequence[str],
    out_cotangent: Any,
) -> NativeSpectralVJPPackage:
    import numpy as np

    bare = source.op_name.removeprefix("tessera.")
    if bare not in {"spectral_filter", "spectral_conv", "stft", "istft"}:
        raise ValueError(f"unsupported compound spectral VJP {bare!r}")
    if bare in {"stft", "istft"} and target not in {
        "x86", "rocm", "nvidia_sm120"
    }:
        raise ValueError(
            f"native {bare.upper()} VJP is admitted only on x86 AVX-512 "
            "or an exact GPU spectral policy package"
        )
    if target not in {"x86", "rocm", "nvidia_sm120"}:
        raise ValueError(f"native compound spectral VJP has no {target!r} package")
    if len(ordered_inputs) != 2 or len(arg_names) != 2:
        raise ValueError("compound spectral VJP requires exactly two forward operands")
    arrays = tuple(np.asarray(value) for value in ordered_inputs)
    dy = np.asarray(out_cotangent)
    axis = int(source.kwargs.get("axis", -1))
    normalized_axis = axis if axis >= 0 else arrays[0].ndim + axis
    normalization = str(
        source.kwargs.get("normalization", source.kwargs.get("norm", "backward"))
    )
    if normalization not in {"backward", "forward", "ortho"}:
        raise ValueError("compound spectral VJP has an invalid normalization")
    hop = -1
    center = _policy_bool(source.kwargs.get("center", False))
    onesided = _policy_bool(source.kwargs.get("onesided", True))
    pad_mode = str(source.kwargs.get("pad_mode", "constant"))
    output_length = -1
    window_broadcast = "not_applicable"
    if bare == "spectral_filter":
        if normalized_axis != arrays[0].ndim - 1:
            raise ValueError("initial native compound spectral VJP requires the final axis")
        if any(value.dtype != np.complex64 for value in (*arrays, dy)):
            raise ValueError("native spectral-filter VJP requires complex64 tensors")
        if arrays[0].shape != arrays[1].shape or arrays[0].shape != dy.shape:
            raise ValueError(
                "native spectral-filter VJP requires identical unbroadcast tensors"
            )
        logical_length = int(source.kwargs.get("n") or arrays[0].shape[-1])
        layout = "full_complex"
    elif bare == "spectral_conv":
        if normalized_axis != arrays[0].ndim - 1:
            raise ValueError("initial native compound spectral VJP requires the final axis")
        if any(value.dtype != np.float32 for value in (*arrays, dy)):
            raise ValueError("native spectral-conv VJP requires float32 tensors")
        if arrays[0].ndim != arrays[1].ndim or arrays[0].ndim != dy.ndim:
            raise ValueError("native spectral-conv VJP requires equal-rank tensors")
        if arrays[0].shape[:-1] != arrays[1].shape[:-1] or arrays[0].shape[:-1] != dy.shape[:-1]:
            raise ValueError("native spectral-conv VJP does not permit batch broadcasting")
        full_length = arrays[0].shape[-1] + arrays[1].shape[-1] - 1
        if dy.shape[-1] != full_length:
            raise ValueError("native spectral-conv VJP requires a full-convolution cotangent")
        logical_length = 1 << int(math.ceil(math.log2(max(full_length, 1))))
        layout = "half_spectrum_nyquist_explicit"
    elif bare == "stft":
        x, window = arrays
        if normalized_axis < 0 or normalized_axis >= x.ndim:
            raise ValueError("native STFT VJP axis is out of range")
        real_storage = {"float16", "bfloat16", "float32"}
        if (
            str(x.dtype) not in real_storage
            or window.dtype != x.dtype
            or dy.dtype != np.complex64
        ):
            raise ValueError(
                "native STFT VJP requires matching f16/bf16/f32 signal/window "
                "and complex64 cotangent"
            )
        if (
            x.ndim < 1
            or window.ndim < 1
            or dy.ndim != x.ndim + 1
            or normalized_axis < 0
            or normalized_axis >= x.ndim
        ):
            raise ValueError("native STFT VJP requires a signal and window")
        logical_length = int(
            source.kwargs.get("logical_length", source.kwargs.get("n_fft"))
            or window.shape[-1]
        )
        hop = int(source.kwargs.get("hop", source.kwargs.get("hop_length", -1)))
        if (
            pad_mode not in {"constant", "reflect"}
            or logical_length < window.shape[-1]
            or hop <= 0
            or (
                center
                and pad_mode == "reflect"
                and x.shape[normalized_axis] <= logical_length // 2
            )
        ):
            raise ValueError(
                "native STFT VJP requires bounded centered/uncentered policy, "
                "n_fft >= window, and explicit hop"
            )
        batch_shape = x.shape[:normalized_axis] + x.shape[normalized_axis + 1 :]
        if window.ndim - 1 > len(batch_shape) or any(
            extent not in {1, batch_shape[len(batch_shape) - window.ndim + 1 + dim]}
            for dim, extent in enumerate(window.shape[:-1])
        ):
            raise ValueError("native STFT VJP window batch dimensions do not broadcast")
        samples = x.shape[normalized_axis]
        framed_samples = max(samples + (logical_length if center else 0), logical_length)
        frames = (framed_samples - logical_length) // hop + 1
        bins = logical_length // 2 + 1 if onesided else logical_length
        expected_dy = (
            x.shape[:normalized_axis]
            + (frames, bins)
            + x.shape[normalized_axis + 1 :]
        )
        if dy.shape != expected_dy:
            raise ValueError("native STFT VJP cotangent shape disagrees with framing")
        layout = (
            "half_spectrum_nyquist_explicit" if onesided else "full_complex"
        )
        window_broadcast = "trailing_batch_broadcast_v1"
    else:
        spectrum, window = arrays
        if normalized_axis < 0 or normalized_axis >= spectrum.ndim:
            raise ValueError("native ISTFT VJP axis is out of range")
        real_storage = {"float16", "bfloat16", "float32"}
        if (
            spectrum.dtype != np.complex64
            or str(window.dtype) not in real_storage
            or dy.dtype != window.dtype
        ):
            raise ValueError(
                "native ISTFT VJP requires complex64 spectrum and matching "
                "f16/bf16/f32 window/cotangent"
            )
        if (
            spectrum.ndim < 2
            or window.ndim < 1
            or dy.ndim != spectrum.ndim - 1
            or normalized_axis <= 0
            or normalized_axis >= spectrum.ndim
        ):
            raise ValueError(
                "native ISTFT VJP requires a spectrum with a preceding frame axis"
            )
        logical_length = int(
            source.kwargs.get("logical_length", source.kwargs.get("n_fft"))
            or window.shape[-1]
        )
        hop = int(source.kwargs.get("hop", source.kwargs.get("hop_length", -1)))
        frame_axis = normalized_axis - 1
        raw_length = (spectrum.shape[frame_axis] - 1) * hop + logical_length
        trim = logical_length // 2 if center else 0
        available_length = raw_length - 2 * trim
        requested = source.kwargs.get("length", source.kwargs.get("output_length"))
        output_length = available_length if requested is None else int(requested)
        if (
            logical_length < window.shape[-1]
            or hop <= 0
            or output_length <= 0
            or output_length > available_length
        ):
            raise ValueError(
                "native ISTFT VJP requires bounded centered/uncentered cropped "
                "policy, n_fft >= window, and explicit hop"
            )
        batch_shape = (
            spectrum.shape[:frame_axis] + spectrum.shape[normalized_axis + 1 :]
        )
        if window.ndim - 1 > len(batch_shape) or any(
            extent not in {1, batch_shape[len(batch_shape) - window.ndim + 1 + dim]}
            for dim, extent in enumerate(window.shape[:-1])
        ):
            raise ValueError("native ISTFT VJP window batch dimensions do not broadcast")
        expected_bins = logical_length // 2 + 1 if onesided else logical_length
        if (
            spectrum.shape[normalized_axis] != expected_bins
            or dy.shape
            != spectrum.shape[:frame_axis]
            + (output_length,)
            + spectrum.shape[normalized_axis + 1 :]
        ):
            raise ValueError("native ISTFT VJP cotangent shape disagrees with overlap-add")
        layout = (
            "half_spectrum_nyquist_explicit" if onesided else "full_complex"
        )
        window_broadcast = "trailing_batch_broadcast_v1"
    input_types = tuple(_tensor_type(value) for value in (dy, *arrays))
    output_types = tuple(_tensor_type(value) for value in arrays)
    storage_dtype = arrays[0].dtype if bare == "stft" else arrays[1].dtype
    numeric_storage = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "complex64": "fp32",
    }[str(storage_dtype)]
    numeric_accum = "fp32"
    arch = ("zen5-avx512" if target == "x86" else
            "sm120" if target == "nvidia_sm120" else "gfx1151")
    kind = f"tessera.{bare}"
    source_digest = hashlib.sha256(source_graph_ir.encode()).hexdigest()
    schedule_hash = _spectral_schedule_hash(
        kind=kind,
        target=target,
        arch=arch,
        input_types=input_types,
        output_types=output_types,
        axis=axis,
        logical_length=logical_length,
        normalization=normalization,
        spectrum_layout=layout,
        hop=hop,
        center=center,
        onesided=onesided,
        pad_mode=pad_mode,
        output_length=output_length,
        numeric_storage=numeric_storage,
        numeric_accum=numeric_accum,
        window_broadcast=window_broadcast,
    )
    algorithm = _algorithm_identity(kind, logical_length, target, layout)
    tile_digest = _digest({
        "family": "spectral_backward",
        "source_graph_ir_digest": source_digest,
        "schedule_artifact_hash": schedule_hash,
        "consumer": "tile.spectral_backward_kernel",
        "algorithm": algorithm,
        "window_broadcast": window_broadcast,
        "numeric_policy": {
            "storage": numeric_storage,
            "accum": numeric_accum,
        },
        "input_types": input_types,
        "output_types": output_types,
    })
    return NativeSpectralVJPPackage(
        schema=_SCHEMA,
        target=target,
        arch=arch,
        kind=kind,
        axis=axis,
        logical_length=logical_length,
        normalization=normalization,
        spectrum_layout=layout,
        axis_packing=_AXIS_PACKING,
        window_broadcast=window_broadcast,
        hop=hop,
        center=center,
        onesided=onesided,
        pad_mode=pad_mode,
        output_length=output_length,
        algorithm=algorithm,
        numeric_storage=numeric_storage,
        numeric_accum=numeric_accum,
        input_types=input_types,
        output_types=output_types,
        input_names=tuple(arg_names),
        output_names=tuple(f"d_{name}" for name in arg_names),
        source_graph_ir=source_graph_ir,
        source_graph_ir_digest=source_digest,
        schedule_artifact_hash=schedule_hash,
        tile_program_digest=tile_digest,
    )


def validate_native_spectral_vjp_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != _SCHEMA:
        raise ValueError("native spectral VJP package schema mismatch")
    source = contract.get("source_graph_ir")
    if not isinstance(source, str) or hashlib.sha256(source.encode()).hexdigest() != contract.get(
        "source_graph_ir_digest"
    ):
        raise ValueError("native spectral VJP source Graph lineage mismatch")
    policy = contract.get("numeric_policy")
    if not isinstance(policy, Mapping) or policy.get("accum") != "fp32":
        raise ValueError("native spectral VJP requires explicit fp32 accumulation")
    if policy.get("storage") not in {"fp16", "bf16", "fp32"}:
        raise ValueError("native spectral VJP numeric storage is unsupported")
    if contract.get("axis_packing") != _AXIS_PACKING:
        raise ValueError("native spectral VJP stride ABI identity mismatch")
    expected = _spectral_schedule_hash(
        kind=str(contract["kind"]),
        target=str(contract["target"]),
        arch=str(contract["arch"]),
        input_types=tuple(contract["input_types"]),
        output_types=tuple(contract["output_types"]),
        axis=int(contract["axis"]),
        logical_length=int(contract["logical_length"]),
        normalization=str(contract["normalization"]),
        spectrum_layout=str(contract["spectrum_layout"]),
        hop=int(contract.get("hop", -1)),
        center=bool(contract.get("center", False)),
        onesided=bool(contract.get("onesided", True)),
        pad_mode=str(contract.get("pad_mode", "constant")),
        output_length=int(contract.get("output_length", -1)),
        numeric_storage=str(policy.get("storage", "")),
        numeric_accum=str(policy.get("accum", "")),
        window_broadcast=str(contract.get("window_broadcast", "")),
    )
    if expected != contract.get("schedule_artifact_hash"):
        raise ValueError("native spectral VJP Schedule artifact was altered")
    tile_digest = _digest({
        "family": "spectral_backward",
        "source_graph_ir_digest": contract["source_graph_ir_digest"],
        "schedule_artifact_hash": expected,
        "consumer": "tile.spectral_backward_kernel",
        "algorithm": str(contract.get("algorithm", "")),
        "window_broadcast": str(contract.get("window_broadcast", "")),
        "numeric_policy": dict(policy),
        "input_types": tuple(contract["input_types"]),
        "output_types": tuple(contract["output_types"]),
    })
    if tile_digest != contract.get("tile_program_digest"):
        raise ValueError("native spectral VJP Tile artifact was altered")
    expected_algorithm = _algorithm_identity(
        str(contract.get("kind", "")), int(contract.get("logical_length", 0)),
        str(contract.get("target", "")), str(contract.get("spectrum_layout", "")),
    )
    if contract.get("algorithm") != expected_algorithm:
        raise ValueError("native spectral VJP algorithm identity was altered")


def _graph_carrier(package: NativeSpectralVJPPackage) -> str:
    args = ", ".join(
        f"%{name}: {ty}"
        for name, ty in zip(("dy", *package.input_names), package.input_types)
    )
    results = ", ".join(package.output_types)
    lhs = ", ".join(f"%{name}" for name in package.output_names)
    operands = ", ".join(("%dy", *(f"%{name}" for name in package.input_names)))
    symbol = package.kind.removeprefix("tessera.") + "_bwd"
    policy = (
        f'kind = "{package.kind}", axis = {package.axis} : i64, '
        f'logical_length = {package.logical_length} : i64, '
        f'normalization = "{package.normalization}", '
        f'spectrum_layout = "{package.spectrum_layout}", '
        f'axis_packing = "{package.axis_packing}"'
    )
    if package.kind in {"tessera.stft", "tessera.istft"}:
        policy += (
            f', hop = {package.hop} : i64, center = {str(package.center).lower()}, '
            f'onesided = {str(package.onesided).lower()}, '
            f'pad_mode = "{package.pad_mode}", '
            f'window_broadcast = "{package.window_broadcast}"'
        )
        if package.output_length >= 0:
            policy += f", output_length = {package.output_length} : i64"
    policy += (
        f', numeric_policy = {{storage = "{package.numeric_storage}", '
        f'accum = "{package.numeric_accum}"}}'
    )
    return f'''module attributes {{tessera.target = "{package.target}", tessera.arch = "{package.arch}"}} {{
  func.func @{symbol}({args}) -> ({results}) {{
    {lhs} = "tessera.spectral_backward"({operands}) {{{policy}}} : ({', '.join(package.input_types)}) -> ({results})
    return {lhs} : {results}
  }}
}}
'''


def compile_rocm_native_spectral_vjp(
    package: NativeSpectralVJPPackage,
) -> NativeSpectralVJPPackage:
    if package.target != "rocm" or package.arch != "gfx1151":
        raise ValueError("ROCm spectral VJP compilation requires exact gfx1151")
    from .rocm_native import _extract_hsaco
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt

    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("tessera-opt is unavailable for ROCm spectral VJP packaging")
    ir = _graph_carrier(package)
    for option in (
        "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile",
        "--lower-tile-to-rocm=arch=gfx1151",
        "--generate-rocm-spectral-backward-kernel",
    ):
        ir = run_tessera_opt(tool, ir, option)
    pipeline = (
        "builtin.module(convert-scf-to-cf,"
        "gpu.module(convert-gpu-to-rocdl,reconcile-unrealized-casts),"
        "rocdl-attach-target{chip=gfx1151},gpu-module-to-binary)"
    )
    result = subprocess.run(
        [str(tool), "-", f"--pass-pipeline={pipeline}"],
        input=ir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "ROCm spectral VJP serialization failed: "
            + (result.stderr.strip() or str(result.returncode))
        )
    return replace(
        package,
        native_image=_extract_hsaco(result.stdout),
        native_symbol=package.kind.removeprefix("tessera.") + "_bwd",
    )


def execute_x86_native_spectral_vjp(metadata: Mapping[str, Any], args: Sequence[Any]):
    import numpy as np
    from tessera import runtime

    contract = metadata.get("native_spectral_vjp")
    if not isinstance(contract, Mapping):
        raise ValueError("x86 spectral VJP executor requires a package contract")
    validate_native_spectral_vjp_contract(contract)
    values = tuple(np.asarray(value) for value in args)
    if len(values) != 3:
        raise ValueError("x86 spectral VJP executor requires dy and two operands")
    dy, x, parameter = values
    dx = np.empty(x.shape, dtype=x.dtype, order="C")
    dparameter = np.empty(parameter.shape, dtype=parameter.dtype, order="C")
    lib = runtime._load_x86_elementwise()
    if lib is None:
        raise RuntimeError("x86 AVX-512 spectral VJP package is unavailable")
    ptr = ctypes.POINTER(ctypes.c_float)
    def address(value: Any) -> Any:
        return value.ctypes.data_as(ptr)
    def void_address(value: Any) -> Any:
        return value.ctypes.data_as(ctypes.c_void_p)
    def storage_code(value: Any) -> int:
        return {"float32": 0, "float16": 1, "bfloat16": 2}.get(
            str(value.dtype), -1
        )
    def layout_descriptor(value: Any) -> tuple[Any, Any]:
        if any(stride % value.itemsize for stride in value.strides):
            raise ValueError("spectral VJP strides must be element aligned")
        strides = tuple(int(stride // value.itemsize) for stride in value.strides)
        if any(stride == 0 and int(value.shape[dim]) > 1
               for dim, stride in enumerate(strides)):
            raise ValueError("spectral VJP overlapping broadcast layout is unsupported")
        ordered = sorted(
            (abs(strides[dim]), int(value.shape[dim]))
            for dim in range(value.ndim) if value.shape[dim] > 1
        )
        span = 1
        for stride, extent in ordered:
            if stride < span:
                raise ValueError("spectral VJP overlapping layouts are unsupported")
            span += (extent - 1) * stride
        shape = (ctypes.c_int64 * value.ndim)(*map(int, value.shape))
        descriptor = (ctypes.c_int64 * value.ndim)(*strides)
        return shape, descriptor
    kind = str(contract["kind"])
    if kind == "tessera.spectral_filter":
        if any(not value.flags.c_contiguous for value in values):
            raise ValueError("spectral-filter VJP requires contiguous arguments")
        lib.tessera_x86_avx512_spectral_filter_bwd_c64(
            address(dy), address(x), address(parameter), address(dx),
            address(dparameter), x.size,
        )
    elif kind == "tessera.spectral_conv":
        if any(not value.flags.c_contiguous for value in values):
            raise ValueError("spectral-conv VJP requires contiguous arguments")
        batch = int(np.prod(x.shape[:-1], dtype=np.int64))
        scale = {
            "backward": 1.0,
            "forward": 1.0 / int(contract["logical_length"]),
            "ortho": 1.0 / math.sqrt(int(contract["logical_length"])),
        }[str(contract["normalization"])]
        lib.tessera_x86_avx512_spectral_conv_bwd_f32(
            address(dy), address(x), address(parameter), address(dx),
            address(dparameter), batch, dy.shape[-1], x.shape[-1],
            parameter.shape[-1], scale,
        )
    elif kind == "tessera.stft":
        n = int(contract["logical_length"])
        axis = int(contract["axis"]) % x.ndim
        frames = dy.shape[axis]
        storage = storage_code(x)
        if storage < 0 or parameter.dtype != x.dtype or dy.dtype != np.complex64:
            raise ValueError("x86 STFT VJP runtime dtype mismatch")
        x_shape, x_strides = layout_descriptor(x)
        dy_shape, dy_strides = layout_descriptor(dy)
        window_shape, window_strides = layout_descriptor(parameter)
        scale = {
            "backward": 1.0,
            "forward": 1.0 / n,
            "ortho": 1.0 / math.sqrt(n),
        }[str(contract["normalization"])]
        rc = lib.tessera_x86_avx512_stft_bwd_policy_layout_storage(
            str(contract["schedule_artifact_hash"]).encode(), address(dy),
            void_address(x), void_address(parameter), void_address(dx),
            void_address(dparameter), x.ndim, x_shape, x_strides, axis,
            dy.ndim, dy_shape, dy_strides,
            parameter.ndim, window_shape, window_strides,
            n, parameter.shape[-1],
            int(contract["hop"]), storage, scale, int(bool(contract["center"])),
            {"constant": 0, "reflect": 1}[str(contract["pad_mode"])],
            int(bool(contract["onesided"])),
        )
        if rc:
            raise RuntimeError(f"x86 STFT backward package failed rc={rc}")
    elif kind == "tessera.istft":
        n = int(contract["logical_length"])
        axis = int(contract["axis"]) % x.ndim
        frame_axis = axis - 1
        storage = storage_code(parameter)
        if storage < 0 or dy.dtype != parameter.dtype or x.dtype != np.complex64:
            raise ValueError("x86 ISTFT VJP runtime dtype mismatch")
        dy_output_axis = frame_axis
        dy_shape, dy_strides = layout_descriptor(dy)
        spectrum_shape, spectrum_strides = layout_descriptor(x)
        window_shape, window_strides = layout_descriptor(parameter)
        inverse_scale = {
            "backward": 1.0 / n,
            "forward": 1.0,
            "ortho": 1.0 / math.sqrt(n),
        }[str(contract["normalization"])]
        rc = lib.tessera_x86_avx512_istft_bwd_policy_layout_storage(
            str(contract["schedule_artifact_hash"]).encode(), void_address(dy),
            address(x), void_address(parameter), address(dx),
            void_address(dparameter), dy.ndim, dy_shape, dy_strides,
            dy_output_axis, x.ndim, spectrum_shape, spectrum_strides,
            frame_axis, axis, parameter.ndim, window_shape, window_strides,
            n, parameter.shape[-1],
            int(contract["hop"]), storage, inverse_scale,
            int(bool(contract["center"])), int(bool(contract["onesided"])),
        )
        if rc:
            raise RuntimeError(f"x86 ISTFT backward package failed rc={rc}")
    else:
        raise ValueError(f"x86 spectral VJP kind {kind!r} is unsupported")
    return dx, dparameter


def execute_rocm_native_spectral_vjp(metadata: Mapping[str, Any], args: Sequence[Any]):
    import numpy as np
    from tessera import runtime

    contract = metadata.get("native_spectral_vjp")
    if not isinstance(contract, Mapping):
        raise ValueError("ROCm spectral VJP executor requires a package contract")
    validate_native_spectral_vjp_contract(contract)
    encoded_image = metadata.get("native_image_b64")
    symbol = metadata.get("native_symbol")
    if not isinstance(encoded_image, str) or not isinstance(symbol, str):
        raise ValueError("ROCm spectral VJP requires a prebuilt native image")
    try:
        image = base64.b64decode(encoded_image, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("ROCm spectral VJP image encoding is invalid") from exc
    if not image.startswith(b"\x7fELF"):
        raise ValueError("ROCm spectral VJP image is not an ELF package")
    values = [np.asarray(value) for value in args]
    if tuple(_tensor_type(value) for value in values) != tuple(contract["input_types"]):
        raise ValueError("ROCm spectral VJP runtime dtype or shape mismatch")
    kind = str(contract["kind"])
    if kind in {"tessera.stft", "tessera.istft"}:
        from tessera.compiler.emit import spectral_candidates

        lib = spectral_candidates._amd_composite_lib()
        if lib is None or lib.ts_spectral_composite_arch_amd() != b"gfx1151":
            raise RuntimeError("exact gfx1151 spectral reverse package is unavailable")

        def descriptor(value: Any) -> tuple[Any, Any]:
            if any(stride % value.itemsize for stride in value.strides):
                raise ValueError("ROCm spectral VJP strides must be element aligned")
            strides = tuple(int(stride // value.itemsize) for stride in value.strides)
            if any(stride == 0 and value.shape[dim] > 1
                   for dim, stride in enumerate(strides)):
                raise ValueError("ROCm spectral VJP overlapping layouts are unsupported")
            return ((ctypes.c_int64 * value.ndim)(*map(int, value.shape)),
                    (ctypes.c_int64 * value.ndim)(*strides))

        dy, x, window = values
        dx = np.empty_like(x, order="C")
        dwindow = np.empty_like(window, order="C")
        storage = {"float32": 0, "float16": 1, "bfloat16": 2}.get(
            str(window.dtype), -1
        )
        if storage < 0:
            raise ValueError("ROCm spectral VJP storage is unsupported")
        n = int(contract["logical_length"])
        window_shape, window_strides = descriptor(window)
        digest = str(contract["schedule_artifact_hash"]).encode()
        if kind == "tessera.stft":
            axis = int(contract["axis"]) % x.ndim
            x_shape, x_strides = descriptor(x)
            dy_shape, dy_strides = descriptor(dy)
            scale = {
                "backward": 1.0,
                "forward": 1.0 / n,
                "ortho": 1.0 / math.sqrt(n),
            }[str(contract["normalization"])]
            rc = lib.ts_stft_backward_hostptr_broadcast_layout_storage_amd(
                digest, spectral_candidates._cptr(dy),
                spectral_candidates._cptr(x), spectral_candidates._cptr(window),
                spectral_candidates._cptr(dx), spectral_candidates._cptr(dwindow),
                x.ndim, x_shape, x_strides, axis,
                dy.ndim, dy_shape, dy_strides,
                window.ndim, window_shape, window_strides,
                n, int(contract["hop"]), storage, ctypes.c_float(scale),
                int(bool(contract["center"])),
                {"constant": 0, "reflect": 1}[str(contract["pad_mode"])],
                int(bool(contract["onesided"])),
            )
        else:
            axis = int(contract["axis"]) % x.ndim
            frame_axis = axis - 1
            dy_axis = frame_axis
            dy_shape, dy_strides = descriptor(dy)
            x_shape, x_strides = descriptor(x)
            inverse_scale = {
                "backward": 1.0 / n,
                "forward": 1.0,
                "ortho": 1.0 / math.sqrt(n),
            }[str(contract["normalization"])]
            rc = lib.ts_istft_backward_hostptr_broadcast_layout_storage_amd(
                digest, spectral_candidates._cptr(dy),
                spectral_candidates._cptr(x), spectral_candidates._cptr(window),
                spectral_candidates._cptr(dx), spectral_candidates._cptr(dwindow),
                dy.ndim, dy_shape, dy_strides, dy_axis,
                x.ndim, x_shape, x_strides, frame_axis, axis,
                window.ndim, window_shape, window_strides,
                n, int(contract["hop"]), storage,
                ctypes.c_float(inverse_scale), int(bool(contract["center"])),
                int(bool(contract["onesided"])),
            )
        if rc:
            raise RuntimeError(f"gfx1151 spectral reverse package failed rc={rc}")
        return dx, dwindow
    if any(not value.flags.c_contiguous for value in values):
        raise ValueError("ROCm spectral VJP runtime requires contiguous arguments")
    outputs = [np.empty_like(values[1]), np.empty_like(values[2])]
    hip = runtime._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("ROCm HIP runtime is unavailable")
    module = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), image) != 0:
        raise RuntimeError("ROCm spectral VJP image is not loadable")
    function = ctypes.c_void_p()
    if hip.hipModuleGetFunction(ctypes.byref(function), module, symbol.encode()) != 0:
        raise RuntimeError("ROCm spectral VJP symbol is missing")
    host = [*values, *outputs]
    device: list[ctypes.c_void_p] = []
    for value in host:
        pointer = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(pointer), value.nbytes) != 0:
            raise RuntimeError("ROCm spectral VJP allocation failed")
        device.append(pointer)
    try:
        for value, pointer in zip(values, device):
            hip.hipMemcpy(pointer, value.ctypes.data_as(ctypes.c_void_p), value.nbytes, 1)
        packed_values: list[Any] = []
        for pointer, value in zip(device, host):
            packed_values.extend((
                ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
                ctypes.c_int64(0), ctypes.c_int64(value.size), ctypes.c_int64(1),
            ))
        packed = (ctypes.c_void_p * len(packed_values))()
        for index, value in enumerate(packed_values):
            packed[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        launch = hip.hipModuleLaunchKernel
        launch.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        kind = str(contract["kind"])
        if kind == "tessera.spectral_filter":
            work_items = int(np.asarray(values[1]).size)
        elif kind == "tessera.spectral_conv":
            work_items = int(np.asarray(values[1]).size + np.asarray(values[2]).size)
        elif kind == "tessera.stft":
            work_items = int(np.asarray(values[1]).size + np.asarray(values[2]).size)
        elif kind == "tessera.istft":
            work_items = int(2 * np.asarray(values[1]).size + np.asarray(values[2]).size)
        else:  # validate_native_spectral_vjp_contract owns the vocabulary.
            raise ValueError(f"ROCm spectral VJP kind {kind!r} is unsupported")
        blocks = (work_items + 255) // 256
        if launch(function, blocks, 1, 1, 256, 1, 1, 0, None, packed, None) != 0:
            raise RuntimeError("ROCm spectral VJP launch failed")
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("ROCm spectral VJP synchronization failed")
        for output, pointer in zip(outputs, device[-2:]):
            hip.hipMemcpy(output.ctypes.data_as(ctypes.c_void_p), pointer, output.nbytes, 2)
    finally:
        for pointer in device:
            hip.hipFree(pointer)
    return tuple(outputs)


def execute_nvidia_native_spectral_vjp(metadata: Mapping[str, Any], args: Sequence[Any]):
    """Analytic reverse products whose convolution transforms execute via cuFFT."""
    import numpy as np
    from tessera import runtime

    contract = metadata.get("native_spectral_vjp")
    if not isinstance(contract, Mapping):
        raise ValueError("NVIDIA spectral VJP executor requires a package contract")
    validate_native_spectral_vjp_contract(contract)
    values = [np.asarray(value) for value in args]
    dy, x, parameter = values
    kind = str(contract["kind"])
    if kind in {"tessera.stft", "tessera.istft"}:
        lib = runtime._load_nvidia_fft_runtime()
        if (lib is None or
                lib.tessera_nvidia_spectral_package_abi() !=
                b"tessera.nvidia.spectral_policy.v1" or
                lib.tessera_nvidia_spectral_arch() != 120):
            raise RuntimeError("exact SM120 spectral reverse package is unavailable")

        def descriptor(value: Any) -> tuple[Any, Any]:
            if any(stride % value.itemsize for stride in value.strides):
                raise ValueError("NVIDIA spectral VJP strides must be element aligned")
            strides = tuple(int(stride // value.itemsize) for stride in value.strides)
            if any(stride == 0 and value.shape[dim] > 1
                   for dim, stride in enumerate(strides)):
                raise ValueError("NVIDIA spectral VJP overlapping layouts are unsupported")
            return ((ctypes.c_int64 * value.ndim)(*map(int, value.shape)),
                    (ctypes.c_int64 * value.ndim)(*strides))

        storage_name = str(contract["numeric_policy"]["storage"])
        storage_dtype_name = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }[storage_name]
        storage_code = {"fp32": 0, "fp16": 1, "bf16": 2}[storage_name]
        if kind == "tessera.stft":
            valid_storage = (
                str(x.dtype) == storage_dtype_name
                and str(parameter.dtype) == storage_dtype_name
                and dy.dtype == np.dtype(np.complex64)
            )
        else:
            valid_storage = (
                x.dtype == np.dtype(np.complex64)
                and str(parameter.dtype) == storage_dtype_name
                and str(dy.dtype) == storage_dtype_name
            )
        if not valid_storage:
            raise ValueError(
                "NVIDIA spectral VJP operands disagree with the scheduled numeric policy"
            )
        dx = np.empty_like(x, order="C")
        dparameter = np.empty_like(parameter, order="C")
        pointer = ctypes.POINTER(ctypes.c_float)
        digest = str(contract["schedule_artifact_hash"]).encode("ascii")
        window_shape, window_strides = descriptor(parameter)
        n = int(contract["logical_length"])
        if kind == "tessera.stft":
            axis = int(contract["axis"]) % x.ndim
            x_shape, x_strides = descriptor(x)
            dy_shape, dy_strides = descriptor(dy)
            forward_scale = {
                "backward": 1.0,
                "forward": 1.0 / n,
                "ortho": 1.0 / math.sqrt(n),
            }[str(contract["normalization"])]
            rc = lib.tessera_nvidia_stft_backward_broadcast_layout_storage(
                digest, dy.ctypes.data_as(pointer),
                ctypes.c_void_p(x.ctypes.data),
                ctypes.c_void_p(parameter.ctypes.data),
                ctypes.c_void_p(dx.ctypes.data),
                ctypes.c_void_p(dparameter.ctypes.data),
                x.ndim, x_shape, x_strides, axis, dy.ndim, dy_shape,
                dy_strides, parameter.ndim, window_shape, window_strides,
                n, int(contract["hop"]), storage_code,
                ctypes.c_float(forward_scale),
                int(bool(contract["center"])),
                {"constant": 0, "reflect": 1}[str(contract["pad_mode"])],
                int(bool(contract["onesided"])),
            )
        else:
            axis = int(contract["axis"]) % x.ndim
            frame_axis = axis - 1
            dy_axis = frame_axis
            dy_shape, dy_strides = descriptor(dy)
            x_shape, x_strides = descriptor(x)
            inverse_scale = {
                "backward": 1.0 / n,
                "forward": 1.0,
                "ortho": 1.0 / math.sqrt(n),
            }[str(contract["normalization"])]
            rc = lib.tessera_nvidia_istft_backward_broadcast_layout_storage(
                digest, ctypes.c_void_p(dy.ctypes.data),
                x.ctypes.data_as(pointer),
                ctypes.c_void_p(parameter.ctypes.data),
                dx.ctypes.data_as(pointer),
                ctypes.c_void_p(dparameter.ctypes.data), dy.ndim, dy_shape,
                dy_strides, dy_axis, x.ndim, x_shape, x_strides, frame_axis,
                axis, parameter.ndim, window_shape, window_strides, n,
                int(contract["hop"]), storage_code,
                ctypes.c_float(inverse_scale),
                int(bool(contract["center"])),
                int(bool(contract["onesided"])),
            )
        if rc:
            raise RuntimeError(f"SM120 spectral reverse package failed rc={rc}")
        return dx, dparameter
    dy, x, parameter = (np.ascontiguousarray(value) for value in values)
    if kind == "tessera.spectral_filter":
        return ((dy * np.conj(parameter)).astype(np.complex64),
                (dy * np.conj(x)).astype(np.complex64))

    def convolution(lhs: Any, rhs: Any) -> Any:
        return runtime._spectral_composite(
            "tessera.spectral_conv", [lhs, rhs], {}, runtime._nvidia_fftexec, np
        )

    x_length, parameter_length = int(x.shape[-1]), int(parameter.shape[-1])
    dx_full = convolution(dy.astype(np.float32), np.flip(parameter, axis=-1).copy())
    dp_full = convolution(dy.astype(np.float32), np.flip(x, axis=-1).copy())
    dx = dx_full[..., parameter_length - 1:parameter_length - 1 + x_length]
    dp = dp_full[..., x_length - 1:x_length - 1 + parameter_length]
    return dx.astype(np.float32), dp.astype(np.float32)


__all__ = [
    "NativeSpectralVJPPackage",
    "build_native_spectral_vjp_package",
    "compile_rocm_native_spectral_vjp",
    "execute_nvidia_native_spectral_vjp",
    "execute_rocm_native_spectral_vjp",
    "execute_x86_native_spectral_vjp",
    "validate_native_spectral_vjp_contract",
]
