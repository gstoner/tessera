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


_SCHEMA = "tessera.native_spectral_vjp.v1"
_MUTATION = "inputs_immutable_outputs_fresh_v1"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tensor_type(value: Any) -> str:
    import numpy as np

    array = np.asarray(value)
    element = {
        np.dtype(np.float32): "f32",
        np.dtype(np.complex64): "complex<f32>",
    }.get(array.dtype)
    if element is None:
        raise ValueError(
            "native compound spectral VJP supports float32 or complex64 storage"
        )
    dimensions = "x".join(str(int(dim)) for dim in array.shape)
    return f"tensor<{dimensions}x{element}>"


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
) -> str:
    # Keep byte-for-byte parity with spectralBackwardDigest() in PMPasses.cpp.
    contract = (
        f"schema=tessera.spectral_backward.v1;kind={kind};target={target};"
        f"arch={arch};inputs={','.join(input_types)};"
        f"outputs={','.join(output_types)};axis={axis};"
        f"logical_length={logical_length};normalization={normalization};"
        f"spectrum_layout={spectrum_layout};hop=-1;center=0;onesided=1;"
        f"pad_mode=constant;output_length=-1;mutation_lineage={_MUTATION}"
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
    if bare not in {"spectral_filter", "spectral_conv"}:
        raise ValueError(f"unsupported compound spectral VJP {bare!r}")
    if target not in {"x86", "rocm", "nvidia_sm120"}:
        raise ValueError(f"native compound spectral VJP has no {target!r} package")
    if len(ordered_inputs) != 2 or len(arg_names) != 2:
        raise ValueError("compound spectral VJP requires exactly two forward operands")
    arrays = tuple(np.asarray(value) for value in ordered_inputs)
    dy = np.asarray(out_cotangent)
    axis = int(source.kwargs.get("axis", -1))
    normalized_axis = axis if axis >= 0 else arrays[0].ndim + axis
    if normalized_axis != arrays[0].ndim - 1:
        raise ValueError("initial native compound spectral VJP requires the final axis")
    normalization = str(
        source.kwargs.get("normalization", source.kwargs.get("norm", "backward"))
    )
    if normalization not in {"backward", "forward", "ortho"}:
        raise ValueError("compound spectral VJP has an invalid normalization")
    if bare == "spectral_filter":
        if any(value.dtype != np.complex64 for value in (*arrays, dy)):
            raise ValueError("native spectral-filter VJP requires complex64 tensors")
        if arrays[0].shape != arrays[1].shape or arrays[0].shape != dy.shape:
            raise ValueError(
                "native spectral-filter VJP requires identical unbroadcast tensors"
            )
        logical_length = int(source.kwargs.get("n") or arrays[0].shape[-1])
        layout = "full_complex"
    else:
        if any(value.dtype != np.float32 for value in (*arrays, dy)):
            raise ValueError("native spectral-conv VJP requires float32 tensors")
        if arrays[0].ndim != arrays[1].ndim or arrays[0].ndim != dy.ndim:
            raise ValueError("native spectral-conv VJP requires equal-rank tensors")
        if arrays[0].shape[:-1] != arrays[1].shape[:-1] or arrays[0].shape[:-1] != dy.shape[:-1]:
            raise ValueError("native spectral-conv VJP does not permit batch broadcasting")
        output_length = arrays[0].shape[-1] + arrays[1].shape[-1] - 1
        if dy.shape[-1] != output_length:
            raise ValueError("native spectral-conv VJP requires a full-convolution cotangent")
        logical_length = 1 << int(math.ceil(math.log2(max(output_length, 1))))
        layout = "half_spectrum_nyquist_explicit"
    input_types = tuple(_tensor_type(value) for value in (dy, *arrays))
    output_types = tuple(_tensor_type(value) for value in arrays)
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
    )
    tile_digest = _digest({
        "family": "spectral_backward",
        "source_graph_ir_digest": source_digest,
        "schedule_artifact_hash": schedule_hash,
        "consumer": "tile.spectral_backward_kernel",
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
    )
    if expected != contract.get("schedule_artifact_hash"):
        raise ValueError("native spectral VJP Schedule artifact was altered")
    tile_digest = _digest({
        "family": "spectral_backward",
        "source_graph_ir_digest": contract["source_graph_ir_digest"],
        "schedule_artifact_hash": expected,
        "consumer": "tile.spectral_backward_kernel",
        "input_types": tuple(contract["input_types"]),
        "output_types": tuple(contract["output_types"]),
    })
    if tile_digest != contract.get("tile_program_digest"):
        raise ValueError("native spectral VJP Tile artifact was altered")


def _graph_carrier(package: NativeSpectralVJPPackage) -> str:
    args = ", ".join(
        f"%{name}: {ty}"
        for name, ty in zip(("dy", *package.input_names), package.input_types)
    )
    results = ", ".join(package.output_types)
    lhs = ", ".join(f"%{name}" for name in package.output_names)
    operands = ", ".join(("%dy", *(f"%{name}" for name in package.input_names)))
    symbol = package.kind.removeprefix("tessera.") + "_bwd"
    return f'''module attributes {{tessera.target = "{package.target}", tessera.arch = "{package.arch}"}} {{
  func.func @{symbol}({args}) -> ({results}) {{
    {lhs} = "tessera.spectral_backward"({operands}) {{kind = "{package.kind}", axis = {package.axis} : i64, logical_length = {package.logical_length} : i64, normalization = "{package.normalization}", spectrum_layout = "{package.spectrum_layout}"}} : ({', '.join(package.input_types)}) -> ({results})
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
    values = tuple(np.ascontiguousarray(np.asarray(value)) for value in args)
    if len(values) != 3:
        raise ValueError("x86 spectral VJP executor requires dy and two operands")
    dy, x, parameter = values
    dx = np.empty_like(x)
    dparameter = np.empty_like(parameter)
    lib = runtime._load_x86_elementwise()
    if lib is None:
        raise RuntimeError("x86 AVX-512 spectral VJP package is unavailable")
    ptr = ctypes.POINTER(ctypes.c_float)
    def address(value: Any) -> Any:
        return value.ctypes.data_as(ptr)
    if contract["kind"] == "tessera.spectral_filter":
        lib.tessera_x86_avx512_spectral_filter_bwd_c64(
            address(dy), address(x), address(parameter), address(dx),
            address(dparameter), x.size,
        )
    else:
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
    values = [np.ascontiguousarray(np.asarray(value)) for value in args]
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
        if launch(function, 1, 1, 1, 256, 1, 1, 0, None, packed, None) != 0:
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
    dy, x, parameter = (np.ascontiguousarray(np.asarray(value)) for value in args)
    if contract["kind"] == "tessera.spectral_filter":
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
