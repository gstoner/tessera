"""Compiler-owned CUDA training packages for the SM120 native-image seam.

The shared Graph/Linalg contracts define the mathematics.  This module owns
CUDA-specific source generation, PTX assembly, cache identity, launch ABI, and
resource evidence.  The correctness-first kernels deliberately avoid importing
ROCm wave/LDS schedules; later CUDA tuning can replace a package only after
two-domain evidence justifies a selector change.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ResourceRecord,
    ScalarArgument,
    ShapeGuard,
)


_STORAGE_DTYPES = ("f32", "f16", "bf16")


def _training_abi(family: str, signature: str, storage: str) -> str:
    return f"tessera.nvidia.training.{family}.{signature}.{storage}.v1"


SM120_NORM_BWD_ABIS = {
    storage: _training_abi(
        "norm_backward",
        "x_gamma_dy_dx_dgamma_dbeta_rows_cols",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_LOSS_BWD_ABIS = {
    storage: _training_abi(
        "loss_backward",
        "prediction_target_dy_dprediction_dtarget_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_CLASS_BWD_ABIS = {
    storage: _training_abi(
        "class_backward",
        "logits_labels_dy_dlogits_rows_classes_i64",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_BROADCAST_REDUCE_ABIS = {
    storage: _training_abi(
        "broadcast_reduce",
        "input_output_input_n_output_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_SGD_ABIS = {
    storage: _training_abi(
        "optimizer.sgd", "param_grad_output_n", storage
    )
    for storage in _STORAGE_DTYPES
}
SM120_MOMENTUM_ABIS = {
    storage: _training_abi(
        "optimizer.momentum",
        "param_grad_velocity_output_velocity_output_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_ADAM_ABIS = {
    storage: _training_abi(
        "optimizer.adam",
        "param_grad_m1_m2_output_m1_output_m2_output_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_LION_BWD_ABIS = {
    storage: _training_abi(
        "optimizer.lion_backward",
        "param_grad_moment_dparam_out_dmoment_out_dparam_dgrad_dmoment_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_OPTIMIZER_BWD_ABIS = {
    kind: _training_abi(
        f"optimizer.{kind}_backward",
        {
            "sgd": "param_grad_dparam_out_dparam_dgrad_n",
            "momentum": "param_grad_velocity_dparam_out_dvelocity_out_dparam_dgrad_dvelocity_n",
            "nesterov": "param_grad_velocity_dparam_out_dvelocity_out_dparam_dgrad_dvelocity_n",
            "adam": "param_grad_m1_m2_dparam_out_dm1_out_dm2_out_dparam_dgrad_dm1_dm2_n",
            "adamw": "param_grad_m1_m2_dparam_out_dm1_out_dm2_out_dparam_dgrad_dm1_dm2_n",
        }[kind],
        "f32",
    )
    for kind in ("sgd", "momentum", "nesterov", "adam", "adamw")
}
SM120_ADAFACTOR_BWD_ABIS = {
    "full": _training_abi(
        "optimizer.adafactor_backward",
        "param_grad_state_dparam_out_dparam_dgrad_dstate_n",
        "f32",
    ),
    "factored": _training_abi(
        "optimizer.adafactor_backward",
        "param_grad_row_col_dparam_out_dparam_dgrad_drow_dcol_rows_cols",
        "f32",
    ),
}
SM120_DELTANET_BWD_ABIS = {
    "f32_v1": _training_abi(
        "deltanet_backward", "q_k_v_do_dq_dk_dv_b_h_s_dqk_dv", "f32"
    ),
    "f32": _training_abi(
        "deltanet_backward",
        "q_k_v_gate_beta_decay_do_dq_dk_dv_dgate_dbeta_ddecay_dims_flags",
        "f32",
    ),
}
SM120_FUSED_LOSS_SGD_ABIS = {
    storage: _training_abi(
        "fused_loss_sgd",
        "prediction_target_dy_param_output_dtarget_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}
SM120_FUSED_LOSS_ADAMW_ABIS = {
    storage: _training_abi(
        "fused_loss_adamw",
        "prediction_target_dy_param_m1_m2_output_m1_output_m2_output_dtarget_n",
        storage,
    )
    for storage in _STORAGE_DTYPES
}

SM120_NORM_BWD_F32_ABI = SM120_NORM_BWD_ABIS["f32"]
SM120_LOSS_BWD_F32_ABI = SM120_LOSS_BWD_ABIS["f32"]
SM120_CLASS_BWD_F32_ABI = SM120_CLASS_BWD_ABIS["f32"]
SM120_SGD_F32_ABI = SM120_SGD_ABIS["f32"]
SM120_MOMENTUM_F32_ABI = SM120_MOMENTUM_ABIS["f32"]
SM120_ADAM_F32_ABI = SM120_ADAM_ABIS["f32"]
SM120_LION_BWD_F32_ABI = SM120_LION_BWD_ABIS["f32"]
SM120_FUSED_LOSS_SGD_F32_ABI = SM120_FUSED_LOSS_SGD_ABIS["f32"]
SM120_FUSED_LOSS_ADAMW_F32_ABI = SM120_FUSED_LOSS_ADAMW_ABIS["f32"]

_ABI_STORAGE = {
    abi: storage
    for family in (
        SM120_NORM_BWD_ABIS,
        SM120_LOSS_BWD_ABIS,
        SM120_CLASS_BWD_ABIS,
        SM120_BROADCAST_REDUCE_ABIS,
        SM120_SGD_ABIS,
        SM120_MOMENTUM_ABIS,
        SM120_ADAM_ABIS,
        SM120_LION_BWD_ABIS,
        SM120_DELTANET_BWD_ABIS,
        SM120_FUSED_LOSS_SGD_ABIS,
        SM120_FUSED_LOSS_ADAMW_ABIS,
    )
    for storage, abi in family.items()
}
_ABI_STORAGE.update(
    {abi: "f32" for abi in (*SM120_OPTIMIZER_BWD_ABIS.values(),
                             *SM120_ADAFACTOR_BWD_ABIS.values())}
)
SM120_TRAINING_ABIS = tuple(_ABI_STORAGE)

_LOSS_KINDS = {
    "mse",
    "mae",
    "huber",
    "smooth_l1",
    "bce",
    "kl",
    "js",
}
_cache: dict[str, tuple[str, dict[str, object], str, str]] = {}


@dataclass(frozen=True)
class NVIDIATrainingPackage:
    contract: str
    cuda_source: str
    ptx: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


def _nvcc() -> Path:
    found = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    path = Path(found)
    if not path.is_file():
        raise RuntimeError("CUDA training packaging requires nvcc")
    return path


def _ptxas() -> Path:
    found = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    path = Path(found)
    if not path.is_file():
        raise RuntimeError("CUDA training packaging requires ptxas")
    return path


def _version(path: Path) -> str:
    result = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=False
    )
    text = result.stdout + result.stderr
    return hashlib.sha256(text.encode()).hexdigest()


def _resource_metrics(stderr: str) -> dict[str, object]:
    def number(pattern: str) -> int:
        match = re.search(pattern, stderr)
        return int(match.group(1)) if match else 0

    return {
        "registers_per_thread": number(r"Used\s+(\d+)\s+registers"),
        "static_shared_memory_bytes": number(r"(\d+)\s+bytes smem"),
        "spill_store_bytes": number(r"(\d+)\s+bytes spill stores"),
        "spill_load_bytes": number(r"(\d+)\s+bytes spill loads"),
    }


def _compile(source: str, entry: str) -> tuple[str, dict[str, object], str, str, str]:
    nvcc, ptxas = _nvcc(), _ptxas()
    compiler_fp = "nvidia-training-cuda-v1:" + hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    toolchain_fp = f"nvcc={_version(nvcc)};ptxas={_version(ptxas)};arch=sm_120a"
    cache_key = hashlib.sha256(
        f"{compiler_fp}\n{toolchain_fp}\n{source}".encode()
    ).hexdigest()
    cached = _cache.get(cache_key)
    compile_state = "warm_cache" if cached is not None else "cold"
    if cached is None:
        with tempfile.TemporaryDirectory(prefix="tessera-sm120-training-") as tmp:
            root = Path(tmp)
            cu = root / "kernel.cu"
            ptx_path = root / "kernel.ptx"
            cubin = root / "kernel.cubin"
            cu.write_text(source, encoding="utf-8")
            emitted = subprocess.run(
                [
                    str(nvcc),
                    "-std=c++17",
                    "-arch=sm_120",
                    "-O3",
                    "--fmad=false",
                    "--ptx",
                    str(cu),
                    "-o",
                    str(ptx_path),
                ],
                capture_output=True,
                text=True,
            )
            if emitted.returncode:
                raise RuntimeError(
                    "CUDA training PTX generation failed: " + emitted.stderr.strip()
                )
            ptx = ptx_path.read_text(encoding="ascii")
            if f".visible .entry {entry}" not in ptx:
                raise RuntimeError(f"CUDA training PTX is missing entry {entry}")
            assembled = subprocess.run(
                [
                    str(ptxas),
                    "-arch=sm_120a",
                    "-v",
                    str(ptx_path),
                    "-o",
                    str(cubin),
                ],
                capture_output=True,
                text=True,
            )
            if assembled.returncode:
                raise RuntimeError(
                    "CUDA training PTX assembly failed: " + assembled.stderr.strip()
                )
            cached = (
                ptx,
                _resource_metrics(assembled.stderr),
                compiler_fp,
                toolchain_fp,
            )
            _cache[cache_key] = cached
    ptx, metrics, compiler_fp, toolchain_fp = cached
    return ptx, metrics, compiler_fp, toolchain_fp, compile_state


def _f32(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("CUDA training scalar attributes must be finite")
    text = f"{float(value):.17g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def training_storage_for_abi(abi_id: str) -> str:
    try:
        return _ABI_STORAGE[abi_id]
    except KeyError as exc:
        raise ValueError(f"unknown NVIDIA training ABI {abi_id!r}") from exc


def _validate_storage(storage: str) -> str:
    if storage not in _STORAGE_DTYPES:
        raise ValueError(
            "CUDA training storage must be f32, f16, or bf16"
        )
    return storage


def _storage_prelude(storage: str) -> str:
    _validate_storage(storage)
    if storage == "f32":
        return """using storage_t = float;
__device__ __forceinline__ float load_storage(storage_t value) { return value; }
__device__ __forceinline__ storage_t store_storage(float value) { return value; }
"""
    if storage == "f16":
        return """using storage_t = __half;
__device__ __forceinline__ float load_storage(storage_t value) {
  return __half2float(value);
}
__device__ __forceinline__ storage_t store_storage(float value) {
  return __float2half_rn(value);
}
"""
    return """using storage_t = __nv_bfloat16;
__device__ __forceinline__ float load_storage(storage_t value) {
  return __bfloat162float(value);
}
__device__ __forceinline__ storage_t store_storage(float value) {
  return __float2bfloat16_rn(value);
}
"""


def _loss_local(kind: str, parameter: float) -> str:
    if kind not in _LOSS_KINDS:
        raise ValueError(f"unsupported CUDA training loss kind {kind!r}")
    if kind == "mse":
        return "local = 2.0f * error; target_local = -local;"
    if kind == "mae":
        return (
            "local = (error > 0.0f) - (error < 0.0f); "
            "target_local = -local;"
        )
    if kind == "huber":
        threshold = _f32(parameter)
        return (
            f"local = fabsf(error) <= {threshold} ? error : "
            f"{threshold} * ((error > 0.0f) - (error < 0.0f)); "
            "target_local = -local;"
        )
    if kind == "smooth_l1":
        beta = _f32(parameter)
        return (
            f"local = fabsf(error) < {beta} ? error / {beta} : "
            "((error > 0.0f) - (error < 0.0f)); target_local = -local;"
        )
    if kind == "bce":
        return (
        "float sigmoid = prediction_value >= 0.0f "
        "? 1.0f / (1.0f + expf(-prediction_value)) "
        ": expf(prediction_value) / (1.0f + expf(prediction_value)); "
        "local = sigmoid - target_value; "
        "target_local = -prediction_value;"
        )
    if kind == "kl":
        return (
            "float probability = expf(prediction_value); "
            "float safe_target = fmaxf(target_value, 1.0e-12f); "
            "local = probability * "
            "(prediction_value - logf(safe_target) + 1.0f); "
            "target_local = -probability / safe_target;"
        )
    return (
        "float midpoint = 0.5f * (prediction_value + target_value); "
        "float safe_prediction = fmaxf(prediction_value, 1.0e-12f); "
        "float safe_target = fmaxf(target_value, 1.0e-12f); "
        "float safe_midpoint = fmaxf(midpoint, 1.0e-12f); "
        "local = 0.5f * (logf(safe_prediction) - logf(safe_midpoint)); "
        "target_local = 0.5f * (logf(safe_target) - logf(safe_midpoint));"
    )


def _header() -> str:
    return """#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
"""


def _loss_source(
    entry: str,
    *,
    kind: str,
    parameter: float,
    reduction: str,
    storage: str,
    mean_denominator: int | None,
) -> str:
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("CUDA loss backward reduction must be none|sum|mean")
    dy_index = "i" if reduction == "none" else "0"
    if mean_denominator is not None and mean_denominator < 1:
        raise ValueError("CUDA loss mean denominator must be positive")
    mean_scale = (
        f"(1.0f / {_f32(float(mean_denominator))})"
        if reduction == "mean" and mean_denominator is not None
        else "(1.0f / (float)n)"
        if reduction == "mean"
        else "1.0f"
    )
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}(
    const storage_t* prediction, const storage_t* target, const storage_t* dy,
    storage_t* dprediction, storage_t* dtarget, int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float prediction_value = load_storage(prediction[i]);
  float target_value = load_storage(target[i]);
  float error = prediction_value - target_value;
  float local = 0.0f, target_local = 0.0f;
  {_loss_local(kind, parameter)}
  float upstream = load_storage(dy[{dy_index}]) * {mean_scale};
  dprediction[i] = store_storage(local * upstream);
  dtarget[i] = store_storage(target_local * upstream);
}}
"""
    )


def _norm_source(
    entry: str, *, kind: str, epsilon: float, storage: str
) -> str:
    if kind not in {"rmsnorm", "layernorm"}:
        raise ValueError("CUDA norm backward kind must be rmsnorm|layernorm")
    layer = kind == "layernorm"
    center = (
        "load_storage(x[index]) - mean"
        if layer
        else "load_storage(x[index])"
    )
    mean_body = (
        "for (int64_t j = 0; j < columns; ++j) "
        "mean += load_storage(x[row * columns + j]); "
        "mean /= (float)columns;"
        if layer
        else ""
    )
    variance_term = (
        "float centered = load_storage(x[row * columns + j]) - mean; "
        "variance += centered * centered;"
        if layer
        else "float centered = load_storage(x[row * columns + j]); "
        "variance += centered * centered;"
    )
    beta_body = (
        "float beta_sum = 0.0f; "
        "for (int64_t row = 0; row < rows; ++row) "
        "beta_sum += load_storage(dy[row * columns + col]); "
        "dbeta[col] = beta_sum;"
        if layer
        else "dbeta[col] = 0.0f;"
    )
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}(
    const storage_t* x, const storage_t* gamma, const storage_t* dy,
    storage_t* dx,
    float* dgamma, float* dbeta, int64_t rows, int64_t columns) {{
  int64_t owner = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (owner < rows) {{
    int64_t row = owner;
    float mean = 0.0f;
    {mean_body}
    float variance = 0.0f;
    for (int64_t j = 0; j < columns; ++j) {{
      {variance_term}
    }}
    float rstd = rsqrtf(variance / (float)columns + {_f32(epsilon)});
    float sum_gdy = 0.0f, sum_gdy_xhat = 0.0f;
    for (int64_t j = 0; j < columns; ++j) {{
      int64_t index = row * columns + j;
      float xhat = ({center}) * rstd;
      float gdy = load_storage(dy[index]) * load_storage(gamma[j]);
      sum_gdy += gdy;
      sum_gdy_xhat += gdy * xhat;
    }}
    sum_gdy /= (float)columns;
    sum_gdy_xhat /= (float)columns;
    for (int64_t j = 0; j < columns; ++j) {{
      int64_t index = row * columns + j;
      float xhat = ({center}) * rstd;
      float dx_value = {'(load_storage(dy[index]) * load_storage(gamma[j]) - sum_gdy - xhat * sum_gdy_xhat) * rstd' if layer else '(load_storage(dy[index]) * load_storage(gamma[j]) - xhat * sum_gdy_xhat) * rstd'};
      dx[index] = store_storage(dx_value);
    }}
  }}
  if (owner < columns) {{
    int64_t col = owner;
    float gamma_sum = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {{
      float mean = 0.0f;
      {'for (int64_t j = 0; j < columns; ++j) mean += load_storage(x[row * columns + j]); mean /= (float)columns;' if layer else ''}
      float variance = 0.0f;
      for (int64_t j = 0; j < columns; ++j) {{
        float centered = {'load_storage(x[row * columns + j]) - mean' if layer else 'load_storage(x[row * columns + j])'};
        variance += centered * centered;
      }}
      float rstd = rsqrtf(variance / (float)columns + {_f32(epsilon)});
      float xhat = ({'load_storage(x[row * columns + col]) - mean' if layer else 'load_storage(x[row * columns + col])'}) * rstd;
      gamma_sum += load_storage(dy[row * columns + col]) * xhat;
    }}
    dgamma[col] = gamma_sum;
    {beta_body}
  }}
}}
"""
    )


def _class_source(
    entry: str, *, label_smoothing: float, reduction: str, storage: str
) -> str:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("class backward reduction must be none|sum|mean")
    dy_value = (
        "load_storage(dy[row])"
        if reduction == "none"
        else "load_storage(dy[0])"
    )
    scale = "(1.0f / (float)rows)" if reduction == "mean" else "1.0f"
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}(
    const storage_t* logits, const int64_t* labels, const storage_t* dy,
    storage_t* dlogits, int64_t rows, int64_t classes) {{
  int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float maximum = -INFINITY;
  for (int64_t col = 0; col < classes; ++col)
    maximum = fmaxf(maximum, load_storage(logits[row * classes + col]));
  float denominator = 0.0f;
  for (int64_t col = 0; col < classes; ++col)
    denominator += expf(load_storage(logits[row * classes + col]) - maximum);
  int64_t label = labels[row];
  float smooth = {_f32(label_smoothing)} / (float)classes;
  for (int64_t col = 0; col < classes; ++col) {{
    float probability = expf(
        load_storage(logits[row * classes + col]) - maximum) / denominator;
    float target = smooth + (col == label ? {_f32(1.0 - label_smoothing)} : 0.0f);
    dlogits[row * classes + col] = store_storage(
        (probability - target) * {dy_value} * {scale});
  }}
}}
"""
    )


def _broadcast_index_expression(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> str:
    if not input_shape or len(output_shape) > len(input_shape):
        raise ValueError("broadcast reduction requires compatible ranks")
    padded = (1,) * (len(input_shape) - len(output_shape)) + output_shape
    for source, target in zip(input_shape, padded, strict=True):
        if target not in {1, source}:
            raise ValueError(
                f"shape {output_shape} does not broadcast to {input_shape}"
            )
    terms: list[str] = []
    output_stride = 1
    input_strides: list[int] = []
    stride = 1
    for extent in reversed(input_shape):
        input_strides.append(stride)
        stride *= extent
    input_strides.reverse()
    output_strides: list[int] = []
    for extent in reversed(padded):
        output_strides.append(output_stride)
        output_stride *= extent
    output_strides.reverse()
    for source, target, in_stride, out_stride in zip(
        input_shape, padded, input_strides, output_strides, strict=True
    ):
        if target == 1:
            continue
        terms.append(
            f"((linear / {in_stride}LL) % {source}LL) * {out_stride}LL"
        )
    return " + ".join(terms) if terms else "0"


def _broadcast_reduce_source(
    entry: str,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    storage: str,
) -> str:
    index = _broadcast_index_expression(input_shape, output_shape)
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}(
    const storage_t* input, storage_t* output,
    int64_t input_n, int64_t output_n) {{
  int64_t owner = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (owner >= output_n) return;
  float total = 0.0f;
  for (int64_t linear = 0; linear < input_n; ++linear) {{
    int64_t output_index = {index};
    if (output_index == owner) total += load_storage(input[linear]);
  }}
  output[owner] = store_storage(total);
}}
"""
    )


def _optimizer_body(
    kind: str,
    *,
    lr: float,
    momentum: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    step: int,
) -> tuple[str, str, tuple[str, ...]]:
    if kind == "sgd":
        return (
            "const storage_t* param, const storage_t* grad, storage_t* output",
            "float param_value = load_storage(param[i]); "
            "float grad_value = load_storage(grad[i]); "
            f"output[i] = store_storage(param_value - {_f32(lr)} * grad_value);",
            ("param", "grad", "output"),
        )
    if kind in {"momentum", "nesterov"}:
        update = "grad[i] + mu * velocity_new" if kind == "nesterov" else "velocity_new"
        update = update.replace("grad[i]", "grad_value")
        return (
            "const storage_t* param, const storage_t* grad, "
            "const float* velocity, storage_t* output, "
            "float* velocity_output",
            "float param_value = load_storage(param[i]); "
            "float grad_value = load_storage(grad[i]); "
            f"float mu = {_f32(momentum)}; "
            "float velocity_new = mu * velocity[i] + grad_value; "
            "velocity_output[i] = velocity_new; "
            f"output[i] = store_storage(param_value - {_f32(lr)} * ({update}));",
            ("param", "grad", "velocity", "output", "velocity_output"),
        )
    if kind not in {"adam", "adamw"}:
        raise ValueError(f"unsupported CUDA optimizer kind {kind!r}")
    if step < 1:
        raise ValueError("Adam step must be positive")
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    decay = weight_decay if kind == "adamw" else 0.0
    body = (
        "float param_value = load_storage(param[i]); "
        "float grad_value = load_storage(grad[i]); "
        f"float m_new = {_f32(beta1)} * moment1[i] + "
        f"{_f32(1.0 - beta1)} * grad_value; "
        f"float v_new = {_f32(beta2)} * moment2[i] + "
        f"{_f32(1.0 - beta2)} * grad_value * grad_value; "
        "moment1_output[i] = m_new; moment2_output[i] = v_new; "
        f"float update = (m_new / {_f32(correction1)}) / "
        f"(sqrtf(v_new / {_f32(correction2)}) + {_f32(epsilon)}) + "
        f"{_f32(decay)} * param_value; "
        f"output[i] = store_storage(param_value - {_f32(lr)} * update);"
    )
    return (
        "const storage_t* param, const storage_t* grad, "
        "const float* moment1, const float* moment2, storage_t* output, "
        "float* moment1_output, "
        "float* moment2_output",
        body,
        (
            "param",
            "grad",
            "moment1",
            "moment2",
            "output",
            "moment1_output",
            "moment2_output",
        ),
    )


def _optimizer_source(
    entry: str, *, kind: str, storage: str, **kwargs: Any
) -> tuple[str, tuple[str, ...]]:
    signature, body, names = _optimizer_body(kind, **kwargs)
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}({signature}, int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  {body}
}}
""",
        names,
    )


def _lion_backward_source(
    entry: str, *, lr: float, beta2: float, weight_decay: float, storage: str
) -> str:
    """SM120 materialization of Lion's shared stop-sign VJP policy.

    ``param``, ``grad``, and ``moment`` are retained in the ABI so the native
    package has the same operand ownership as the Graph operation.  The
    stop-sign convention makes their values unnecessary for this VJP.
    """
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}(
    const storage_t* param, const storage_t* grad, const float* moment,
    const float* dparam_out, const float* dmoment_out,
    float* dparam, float* dgrad, float* dmoment, int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Keep the retained forward operands physically present in the ABI.  A
  // volatile load prevents NVCC from deleting those formally-unused pointers.
  volatile float retained_param = load_storage(param[i]);
  volatile float retained_grad = load_storage(grad[i]);
  volatile float retained_moment = moment[i];
  (void)retained_param; (void)retained_grad; (void)retained_moment;
  dparam[i] = dparam_out[i] * {_f32(1.0 - lr * weight_decay)};
  dgrad[i] = dmoment_out[i] * {_f32(1.0 - beta2)};
  dmoment[i] = dmoment_out[i] * {_f32(beta2)};
}}
"""
    )


def _optimizer_backward_source(
    entry: str,
    *,
    kind: str,
    lr: float,
    momentum: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    step: int,
) -> tuple[str, tuple[str, ...]]:
    """Correctness-first analytic VJPs for the shared explicit-state ABI."""
    if kind == "sgd":
        names = ("param", "grad", "dparam_out", "dparam", "dgrad")
        signature = (
            "const float* param, const float* grad, const float* dparam_out, "
            "float* dparam, float* dgrad"
        )
        body = (
            "volatile float retained = param[i] + grad[i]; (void)retained; "
            "dparam[i] = dparam_out[i]; "
            f"dgrad[i] = -{_f32(lr)} * dparam_out[i];"
        )
    elif kind in {"momentum", "nesterov"}:
        names = (
            "param", "grad", "velocity", "dparam_out", "dvelocity_out",
            "dparam", "dgrad", "dvelocity",
        )
        signature = (
            "const float* param, const float* grad, const float* velocity, "
            "const float* dparam_out, const float* dvelocity_out, "
            "float* dparam, float* dgrad, float* dvelocity"
        )
        factor = 1.0 + momentum if kind == "nesterov" else 1.0
        velocity_factor = momentum if kind == "nesterov" else 1.0
        body = (
            "volatile float retained = param[i] + grad[i] + velocity[i]; "
            "(void)retained; float from_param = "
            f"-{_f32(lr)} * dparam_out[i]; "
            "dparam[i] = dparam_out[i]; "
            f"dgrad[i] = {_f32(factor)} * from_param + dvelocity_out[i]; "
            f"dvelocity[i] = {_f32(momentum)} * "
            f"({_f32(velocity_factor)} * from_param + dvelocity_out[i]);"
        )
    else:
        if kind not in {"adam", "adamw"} or step < 1:
            raise ValueError(f"unsupported CUDA optimizer VJP kind {kind!r}")
        names = (
            "param", "grad", "moment1", "moment2", "dparam_out",
            "dmoment1_out", "dmoment2_out", "dparam", "dgrad",
            "dmoment1", "dmoment2",
        )
        signature = (
            "const float* param, const float* grad, const float* moment1, "
            "const float* moment2, const float* dparam_out, "
            "const float* dmoment1_out, const float* dmoment2_out, "
            "float* dparam, float* dgrad, float* dmoment1, float* dmoment2"
        )
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        decay = weight_decay if kind == "adamw" else 0.0
        body = (
            "float g = grad[i]; "
            f"float m_new = {_f32(beta1)} * moment1[i] + {_f32(1.0 - beta1)} * g; "
            f"float v_new = {_f32(beta2)} * moment2[i] + {_f32(1.0 - beta2)} * g * g; "
            f"float normalized_v = v_new / {_f32(correction2)}; "
            "float root = sqrtf(normalized_v); "
            f"float denom = root + {_f32(epsilon)}; "
            f"float dm_new = dmoment1_out[i] + dparam_out[i] * "
            f"(-{_f32(lr / correction1)}) / denom; "
            f"float droot = normalized_v > 0.0f ? 0.5f / ({_f32(correction2)} * root) : 0.0f; "
            f"float dv_new = dmoment2_out[i] + dparam_out[i] * {_f32(lr)} * "
            f"(m_new / {_f32(correction1)}) * droot / (denom * denom); "
            f"dparam[i] = dparam_out[i] * {_f32(1.0 - lr * decay)}; "
            f"dgrad[i] = {_f32(1.0 - beta1)} * dm_new + "
            f"2.0f * {_f32(1.0 - beta2)} * g * dv_new; "
            f"dmoment1[i] = {_f32(beta1)} * dm_new; "
            f"dmoment2[i] = {_f32(beta2)} * dv_new; "
            "volatile float retained = param[i]; (void)retained;"
        )
    return (
        _header()
        + f'''extern "C" __global__ void {entry}({signature}, int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  {body}
}}
''',
        names,
    )


def _adafactor_full_backward_source(
    entry: str, *, lr: float, beta2: float, epsilon: float
) -> str:
    return _header() + f'''extern "C" __global__ void {entry}(
    const float* param, const float* grad, const float* state,
    const float* dparam_out, float* dparam, float* dgrad, float* dstate,
    int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = grad[i];
  float moment = {_f32(beta2)} * state[i] + {_f32(1.0 - beta2)} * g * g;
  float safe = (isnan(moment) || moment > {_f32(epsilon)}) ? moment : {_f32(epsilon)};
  float root = sqrtf(safe), denom = root + {_f32(epsilon)};
  float dm = moment > {_f32(epsilon)}
      ? {_f32(lr)} * dparam_out[i] * g /
        (2.0f * fmaxf(root, 1.0e-30f) * denom * denom)
      : 0.0f;
  volatile float retained = param[i]; (void)retained;
  dparam[i] = dparam_out[i];
  dgrad[i] = -{_f32(lr)} * dparam_out[i] / denom
           + 2.0f * {_f32(1.0 - beta2)} * g * dm;
  dstate[i] = {_f32(beta2)} * dm;
}}
'''


def _adafactor_factored_backward_source(
    entry: str, *, lr: float, beta2: float, epsilon: float
) -> str:
    """One deterministic launch; thread zero owns every ordered reduction."""
    return _header() + f'''
__device__ __forceinline__ float adafactor_row(
    const float* g, const float* old_row, int64_t r,
    int64_t rows, int64_t cols) {{
  float total = 0.0f;
  for (int64_t c = 0; c < cols; ++c) {{ float v = g[r * cols + c]; total += v * v; }}
  return {_f32(beta2)} * old_row[r] + {_f32(1.0 - beta2)} * total / (float)cols;
}}
__device__ __forceinline__ float adafactor_col(
    const float* g, const float* old_col, int64_t c,
    int64_t rows, int64_t cols) {{
  float total = 0.0f;
  for (int64_t r = 0; r < rows; ++r) {{ float v = g[r * cols + c]; total += v * v; }}
  return {_f32(beta2)} * old_col[c] + {_f32(1.0 - beta2)} * total / (float)rows;
}}
__device__ __forceinline__ float adafactor_floor(float value, float floor) {{
  return (isnan(value) || value > floor) ? value : floor;
}}
extern "C" __global__ void {entry}(
    const float* param, const float* grad, const float* old_row,
    const float* old_col, const float* dparam_out, float* dparam,
    float* dgrad, float* d_old_row, float* d_old_col,
    int64_t rows, int64_t cols) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  volatile float retained = param[0]; (void)retained;
  float mean = 0.0f;
  for (int64_t r = 0; r < rows; ++r)
    mean += adafactor_row(grad, old_row, r, rows, cols);
  mean /= (float)rows;
  float safe_mean = adafactor_floor(mean, {_f32(epsilon)});
  float dmean = 0.0f;
  for (int64_t r = 0; r < rows; ++r) {{
    float safe_row = adafactor_floor(
        adafactor_row(grad, old_row, r, rows, cols), {_f32(epsilon)});
    for (int64_t c = 0; c < cols; ++c) {{
      int64_t i = r * cols + c;
      float safe_col = adafactor_floor(
          adafactor_col(grad, old_col, c, rows, cols), {_f32(epsilon)});
      float scale = safe_row * safe_col / safe_mean;
      float root = sqrtf(scale), denom = root + {_f32(epsilon)};
      float ds = {_f32(lr)} * dparam_out[i] * grad[i] /
          (2.0f * adafactor_floor(root, 1.0e-30f) * denom * denom);
      dmean -= ds * safe_row * safe_col / (safe_mean * safe_mean);
    }}
  }}
  for (int64_t r = 0; r < rows; ++r) {{
    float row = adafactor_row(grad, old_row, r, rows, cols);
    float drow = 0.0f;
    float safe_row = adafactor_floor(row, {_f32(epsilon)});
    for (int64_t c = 0; c < cols; ++c) {{
      int64_t i = r * cols + c;
      float safe_col = adafactor_floor(
          adafactor_col(grad, old_col, c, rows, cols), {_f32(epsilon)});
      float root = sqrtf(safe_row * safe_col / safe_mean);
      float denom = root + {_f32(epsilon)};
      float ds = {_f32(lr)} * dparam_out[i] * grad[i] /
          (2.0f * adafactor_floor(root, 1.0e-30f) * denom * denom);
      drow += ds * safe_col / safe_mean;
    }}
    if (row <= {_f32(epsilon)}) drow = 0.0f;
    else if (mean > {_f32(epsilon)}) drow += dmean / (float)rows;
    d_old_row[r] = drow;
  }}
  for (int64_t c = 0; c < cols; ++c) {{
    float col = adafactor_col(grad, old_col, c, rows, cols);
    float dcol = 0.0f;
    float safe_col = adafactor_floor(col, {_f32(epsilon)});
    for (int64_t r = 0; r < rows; ++r) {{
      int64_t i = r * cols + c;
      float safe_row = adafactor_floor(
          adafactor_row(grad, old_row, r, rows, cols), {_f32(epsilon)});
      float root = sqrtf(safe_row * safe_col / safe_mean);
      float denom = root + {_f32(epsilon)};
      float ds = {_f32(lr)} * dparam_out[i] * grad[i] /
          (2.0f * adafactor_floor(root, 1.0e-30f) * denom * denom);
      dcol += ds * safe_row / safe_mean;
    }}
    if (col <= {_f32(epsilon)}) dcol = 0.0f;
    d_old_col[c] = dcol;
  }}
  for (int64_t r = 0; r < rows; ++r) {{
    float row = adafactor_row(grad, old_row, r, rows, cols);
    float safe_row = adafactor_floor(row, {_f32(epsilon)});
    float drow = d_old_row[r];
    for (int64_t c = 0; c < cols; ++c) {{
      int64_t i = r * cols + c;
      float col = adafactor_col(grad, old_col, c, rows, cols);
      float safe_col = adafactor_floor(col, {_f32(epsilon)});
      float denom = sqrtf(safe_row * safe_col / safe_mean) + {_f32(epsilon)};
      float dcol = d_old_col[c];
      dparam[i] = dparam_out[i];
      dgrad[i] = -{_f32(lr)} * dparam_out[i] / denom
          + 2.0f * {_f32(1.0 - beta2)} * grad[i]
            * (drow / (float)cols + dcol / (float)rows);
    }}
  }}
  for (int64_t r = 0; r < rows; ++r) d_old_row[r] *= {_f32(beta2)};
  for (int64_t c = 0; c < cols; ++c) d_old_col[c] *= {_f32(beta2)};
}}
'''


def _deltanet_backward_source(entry: str) -> str:
    """Bounded, correctness-first CUDA VJP for affine DeltaNet variants.

    The physical v2 ABI is fixed for all DeltaNet variants.  This entry owns
    the affine reverse stage (optional gate, beta, and decay) and the bounded
    serial-fill derivative for erase and modified updates. One CUDA thread
    owns each B/H recurrence so the state is private and no cross-block
    reduction changes the ABI.
    """
    return _header() + f"""
extern \"C\" __global__ void {entry}(
    const float* q, const float* k, const float* v, const float* gate,
    const float* beta, const float* decay, const float* dy,
    float* dq, float* dk, float* dv, float* dgate, float* dbeta,
    float* ddecay, int64_t B, int64_t H, int64_t S, int64_t Dqk,
    int64_t Dv, int64_t HasGate, int64_t HasBeta, int64_t HasDecay,
    int64_t Erase, int64_t Modified) {{
  if (threadIdx.x != 0) return;
  const int64_t bh = blockIdx.x, BH = B * H;
  // This bounded envelope lets one thread carry the analytic reverse state.
  if (bh >= BH || Dqk > 8 || Dv > 8) return;
  float dstate[16][16] = {{0.0f}};
  for (int64_t t = S; t-- > 0;) {{
    float old[16][16] = {{0.0f}};
    // Replay the causal recurrence only through t-1 to obtain S_t.
    for (int64_t u = 0; u < t; ++u) {{
      const float a = HasDecay ? decay[bh*S+u] : 1.0f;
      const float b = HasBeta ? beta[bh*S+u] : 1.0f;
      float target[16], update[16][16], delta[16][16];
      for (int64_t e = 0; e < Dv; ++e) {{
        target[e] = v[(bh*S+u)*Dv+e];
        if (Erase) {{
          float vhat = 0.0f;
          for (int64_t d = 0; d < Dqk; ++d)
            vhat += k[(bh*S+u)*Dqk+d] * old[d][e];
          target[e] -= a * vhat;
        }}
      }}
      float norm2 = 0.0f;
      for (int64_t d = 0; d < Dqk; ++d)
        for (int64_t e = 0; e < Dv; ++e) {{
          update[d][e] = k[(bh*S+u)*Dqk+d] * target[e];
          norm2 += update[d][e] * update[d][e];
        }}
      const float denom = Modified ? 1.0f + sqrtf(norm2) : 1.0f;
      for (int64_t d = 0; d < Dqk; ++d)
        for (int64_t e = 0; e < Dv; ++e)
          old[d][e] = a * old[d][e] + b * update[d][e] / denom;
    }}
    const float a = HasDecay ? decay[bh*S+t] : 1.0f;
    const float b = HasBeta ? beta[bh*S+t] : 1.0f;
    float target[16], vhat[16], update[16][16], delta[16][16];
    for (int64_t e = 0; e < Dv; ++e) {{
      target[e] = v[(bh*S+t)*Dv+e];
      vhat[e] = 0.0f;
      if (Erase) {{
        for (int64_t d = 0; d < Dqk; ++d)
          vhat[e] += k[(bh*S+t)*Dqk+d] * old[d][e];
        target[e] -= a * vhat[e];
      }}
    }}
    float norm2 = 0.0f;
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e) {{
        update[d][e] = k[(bh*S+t)*Dqk+d] * target[e];
        norm2 += update[d][e] * update[d][e];
      }}
    const float norm = Modified ? sqrtf(norm2) : 0.0f;
    const float denom = Modified ? 1.0f + norm : 1.0f;
    float state[16][16];
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e)
        {{ delta[d][e] = update[d][e] / denom;
           state[d][e] = a * old[d][e] + b * delta[d][e]; }}

    float dnew[16][16];
    for (int64_t d = 0; d < Dqk; ++d) {{
      for (int64_t e = 0; e < Dv; ++e) {{
        const int64_t v_offset = (bh*S+t)*Dv+e;
        const float g = gate[v_offset];
        const float sigmoid = HasGate ? 1.0f / (1.0f + expf(-g)) : 1.0f;
        const float dyt = dy[v_offset] * sigmoid;
        dnew[d][e] = dstate[d][e] + q[(bh*S+t)*Dqk+d] * dyt;
      }}
    }}

    float beta_grad = 0.0f, decay_grad = 0.0f;
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e) {{
        beta_grad += dnew[d][e] * delta[d][e];
        decay_grad += dnew[d][e] * old[d][e];
    }}
    dbeta[bh*S+t] = HasBeta ? beta_grad : 0.0f;

    float dupdate[16][16], dtarget[16];
    float projection = 0.0f;
    if (Modified)
      for (int64_t d = 0; d < Dqk; ++d)
        for (int64_t e = 0; e < Dv; ++e)
          projection += b * dnew[d][e] * update[d][e];
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e) {{
        dupdate[d][e] = b * dnew[d][e] / denom;
        if (Modified && norm > 0.0f)
          dupdate[d][e] -= update[d][e] * projection / (norm * denom * denom);
      }}
    for (int64_t e = 0; e < Dv; ++e) {{
      dtarget[e] = 0.0f;
      for (int64_t d = 0; d < Dqk; ++d)
        dtarget[e] += dupdate[d][e] * k[(bh*S+t)*Dqk+d];
      dv[(bh*S+t)*Dv+e] = dtarget[e];
      if (Erase) decay_grad -= dtarget[e] * vhat[e];
    }}
    ddecay[bh*S+t] = HasDecay ? decay_grad : 0.0f;
    for (int64_t d = 0; d < Dqk; ++d) {{
      float dk_value = 0.0f, dq_value = 0.0f;
      for (int64_t e = 0; e < Dv; ++e) {{
        const int64_t v_offset = (bh*S+t)*Dv+e;
        const float g = gate[v_offset];
        const float sigmoid = HasGate ? 1.0f / (1.0f + expf(-g)) : 1.0f;
        const float dyt = dy[v_offset] * sigmoid;
        dq_value += state[d][e] * dyt;
        dk_value += dupdate[d][e] * target[e];
      }}
      dq[(bh*S+t)*Dqk+d] = dq_value;
      dk[(bh*S+t)*Dqk+d] = dk_value;
    }}
    for (int64_t e = 0; e < Dv; ++e) {{
      const int64_t v_offset = (bh*S+t)*Dv+e;
      const float g = gate[v_offset];
      if (HasGate) {{
        const float sigmoid = 1.0f / (1.0f + expf(-g));
        float raw = 0.0f;
        for (int64_t d = 0; d < Dqk; ++d)
          raw += q[(bh*S+t)*Dqk+d] * state[d][e];
        dgate[v_offset] = dy[v_offset] * raw * sigmoid * (1.0f - sigmoid);
      }} else {{
        dgate[v_offset] = 0.0f;
      }}
    }}
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e) {{
        const float dvhat = Erase ? -a * dtarget[e] : 0.0f;
        dk[(bh*S+t)*Dqk+d] += old[d][e] * dvhat;
        dstate[d][e] = a * dnew[d][e] + k[(bh*S+t)*Dqk+d] * dvhat;
      }}
  }}
}}
"""


def _fused_source(
    entry: str,
    *,
    loss_kind: str,
    optimizer: str,
    parameter: float,
    reduction: str,
    storage: str,
    **kwargs: Any,
) -> tuple[str, tuple[str, ...]]:
    if optimizer not in {"sgd", "adamw"}:
        raise ValueError("fused CUDA training supports SGD or AdamW")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("fused reduction must be none|sum|mean")
    dy_index = "i" if reduction == "none" else "0"
    scale = "(1.0f / (float)n)" if reduction == "mean" else "1.0f"
    names: tuple[str, ...]
    if optimizer == "sgd":
        signature = (
            "const storage_t* prediction, const storage_t* target, "
            "const storage_t* dy, const storage_t* param, "
            "storage_t* output, storage_t* dtarget"
        )
        update = (
            "float param_value = load_storage(param[i]); "
            "output[i] = store_storage(param_value - "
            f"{_f32(float(kwargs['lr']))} * gradient);"
        )
        names = ("prediction", "target", "dy", "param", "output", "dtarget")
    else:
        _, optimizer_body, _ = _optimizer_body("adamw", **kwargs)
        signature = (
            "const storage_t* prediction, const storage_t* target, "
            "const storage_t* dy, const storage_t* param, "
            "const float* moment1, const float* moment2, "
            "storage_t* output, float* moment1_output, "
            "float* moment2_output, storage_t* dtarget"
        )
        update = optimizer_body.replace(
            "float grad_value = load_storage(grad[i]);",
            "float grad_value = gradient;",
        )
        names = (
            "prediction",
            "target",
            "dy",
            "param",
            "moment1",
            "moment2",
            "output",
            "moment1_output",
            "moment2_output",
            "dtarget",
        )
    return (
        _header()
        + _storage_prelude(storage)
        + f"""
extern "C" __global__ void {entry}({signature}, int64_t n) {{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float prediction_value = load_storage(prediction[i]);
  float target_value = load_storage(target[i]);
  float error = prediction_value - target_value;
  float local = 0.0f, target_local = 0.0f;
  {_loss_local(loss_kind, parameter)}
  float upstream = load_storage(dy[{dy_index}]) * {scale};
  float gradient = local * upstream;
  dtarget[i] = store_storage(target_local * upstream);
  {update}
}}
""",
        names,
    )


def _package(
    *,
    contract: str,
    source: str,
    entry: str,
    abi_id: str,
    buffer_names: tuple[str, ...],
    scalar_names: tuple[str, ...],
    provenance: dict[str, object],
    storage: str,
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    ptx, metrics, compiler_fp, toolchain_fp, compile_state = _compile(source, entry)
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name="tessera-nvidia-pipeline-sm120",
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        resource_record=ResourceRecord(
            provenance="nvcc --ptx sm_120 + ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    output_names = {
        "dprediction",
        "dtarget",
        "dx",
        "dgamma",
        "dbeta",
        "dlogits",
        "output",
        "velocity_output",
        "moment1_output",
        "moment2_output",
        "dparam",
        "dgrad",
        "dmoment",
        "dvelocity",
        "dmoment1",
        "dmoment2",
        "dstate0",
        "dstate1",
        "dq", "dk", "dv", "dgate", "ddecay",
    }

    def buffer_rank(name: str) -> int:
        if abi_id in SM120_DELTANET_BWD_ABIS.values():
            return 3 if name in {"beta", "decay", "dbeta", "ddecay"} else 4
        if abi_id in SM120_NORM_BWD_ABIS.values() and name in {"x", "dy", "dx"}:
            return 2
        if abi_id in SM120_CLASS_BWD_ABIS.values() and name in {"logits", "dlogits"}:
            return 2
        return 1

    fp32_buffers = {
        "dgamma",
        "dbeta",
        "velocity",
        "velocity_output",
        "moment1",
        "moment2",
        "moment1_output",
        "moment2_output",
        "moment",
        "dparam_out",
        "dmoment_out",
        "dparam",
        "dgrad",
        "dmoment",
        "dvelocity",
        "dmoment1",
        "dmoment2",
        "state0",
        "state1",
        "dstate0",
        "dstate1",
    }

    def buffer_dtype(name: str) -> tuple[str, int]:
        if name == "labels":
            return "int64", 8
        if name in fp32_buffers:
            return "fp32", 4
        return ("fp32", 4) if storage == "f32" else (storage, 2)

    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=tuple(
            BufferBinding(
                index,
                name,
                "output" if name in output_names else "input",
                buffer_dtype(name)[0],
                buffer_rank(name),
                "row_major",
                buffer_dtype(name)[1],
            )
            for index, name in enumerate(buffer_names)
        ),
        scalars=tuple(
            ScalarArgument(len(buffer_names) + index, name, "int64")
            for index, name in enumerate(scalar_names)
        ),
        shape_guards=tuple(
            ShapeGuard(name, 0, "min", 1) for name in buffer_names
        ) + (
            (ShapeGuard("q", 3, "max", 8), ShapeGuard("v", 3, "max", 8))
            if abi_id in SM120_DELTANET_BWD_ABIS.values() else ()
        ),
        geometry=LaunchGeometry(policy="sm120_training_threads_128"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-TRAINING-AUTODIFF",
            "schedule": "correctness_first_threads_128",
            "storage": storage,
            "accum": "fp32",
            "selector_changed": False,
            **provenance,
        },
    )
    return NVIDIATrainingPackage(contract, source, ptx, image, descriptor)


def package_loss_backward(
    *,
    kind: str,
    parameter: float = 1.0,
    reduction: str = "none",
    storage: str = "f32",
    mean_denominator: int | None = None,
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    entry = f"tessera_cuda_training_loss_{kind}_{reduction}_{storage}"
    source = _loss_source(
        entry,
        kind=kind,
        parameter=parameter,
        reduction=reduction,
        storage=storage,
        mean_denominator=mean_denominator,
    )
    return _package(
        contract=f"tile.training.loss_backward.{kind}.{reduction}.{storage}",
        source=source,
        entry=entry,
        abi_id=SM120_LOSS_BWD_ABIS[storage],
        buffer_names=("prediction", "target", "dy", "dprediction", "dtarget"),
        scalar_names=("N",),
        provenance={
            "family": "loss_backward",
            "kind": kind,
            "parameter": parameter,
            "reduction": reduction,
            "mean_denominator": mean_denominator,
        },
        storage=storage,
    )


def package_norm_backward(
    *, kind: str, epsilon: float = 1.0e-5, storage: str = "f32"
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    entry = f"tessera_cuda_training_norm_{kind}_{storage}"
    return _package(
        contract=(
            f"tile.training.norm_backward.{kind}.dynamic_affine.{storage}"
        ),
        source=_norm_source(
            entry, kind=kind, epsilon=epsilon, storage=storage
        ),
        entry=entry,
        abi_id=SM120_NORM_BWD_ABIS[storage],
        buffer_names=("x", "gamma", "dy", "dx", "dgamma", "dbeta"),
        scalar_names=("Rows", "Columns"),
        provenance={
            "family": "norm_backward",
            "kind": kind,
            "epsilon": epsilon,
            "affine": True,
            "deterministic": True,
        },
        storage=storage,
    )


def package_class_backward(
    *,
    label_smoothing: float = 0.0,
    reduction: str = "none",
    storage: str = "f32",
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    suffix = hashlib.sha256(f"{label_smoothing:.17g}".encode()).hexdigest()[:8]
    entry = (
        f"tessera_cuda_training_class_cross_entropy_"
        f"{reduction}_{storage}_{suffix}"
    )
    return _package(
        contract=f"tile.training.class_backward.cross_entropy.{storage}_i64",
        source=_class_source(
            entry,
            label_smoothing=label_smoothing,
            reduction=reduction,
            storage=storage,
        ),
        entry=entry,
        abi_id=SM120_CLASS_BWD_ABIS[storage],
        buffer_names=("logits", "labels", "dy", "dlogits"),
        scalar_names=("Rows", "Classes"),
        provenance={
            "family": "class_backward",
            "kind": "cross_entropy",
            "label_smoothing": label_smoothing,
            "reduction": reduction,
        },
        storage=storage,
    )


def package_broadcast_gradient(
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    storage: str = "f32",
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    _broadcast_index_expression(input_shape, output_shape)
    if any(extent < 1 for extent in (*input_shape, *output_shape)):
        raise ValueError("broadcast reduction shapes must be positive")
    digest = hashlib.sha256(
        repr((input_shape, output_shape, storage)).encode()
    ).hexdigest()[:8]
    entry = f"tessera_cuda_training_broadcast_reduce_{storage}_{digest}"
    return _package(
        contract=(
            f"tile.training.broadcast_gradient.{storage}."
            f"r{len(input_shape)}_to_r{len(output_shape)}"
        ),
        source=_broadcast_reduce_source(
            entry,
            input_shape=input_shape,
            output_shape=output_shape,
            storage=storage,
        ),
        entry=entry,
        abi_id=SM120_BROADCAST_REDUCE_ABIS[storage],
        buffer_names=("input", "output"),
        scalar_names=("InputN", "OutputN"),
        provenance={
            "family": "broadcast_gradient",
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "deterministic": True,
        },
        storage=storage,
    )


def package_optimizer(
    *,
    kind: str,
    lr: float,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    storage: str = "f32",
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    values = {
        "lr": lr,
        "momentum": momentum,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
        "step": step,
    }
    digest = hashlib.sha256(repr(sorted(values.items())).encode()).hexdigest()[:8]
    entry = f"tessera_cuda_training_optimizer_{kind}_{storage}_{digest}"
    source, names = _optimizer_source(
        entry, kind=kind, storage=storage, **values
    )
    abi = (
        SM120_SGD_ABIS[storage]
        if kind == "sgd"
        else SM120_MOMENTUM_ABIS[storage]
        if kind in {"momentum", "nesterov"}
        else SM120_ADAM_ABIS[storage]
    )
    return _package(
        contract=f"tile.training.optimizer.{kind}.{storage}",
        source=source,
        entry=entry,
        abi_id=abi,
        buffer_names=names,
        scalar_names=("N",),
        provenance={"family": "optimizer", "kind": kind, **values},
        storage=storage,
    )


def package_lion_backward(
    *,
    lr: float = 1.0e-4,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    storage: str = "f32",
) -> NVIDIATrainingPackage:
    """Build the CUDA-owned, stop-sign Lion VJP package for SM120."""
    _validate_storage(storage)
    values = {"lr": lr, "beta2": beta2, "weight_decay": weight_decay}
    digest = hashlib.sha256(repr(sorted(values.items())).encode()).hexdigest()[:8]
    entry = f"tessera_cuda_training_lion_backward_{storage}_{digest}"
    return _package(
        contract=f"tile.training.optimizer.lion_backward.{storage}",
        source=_lion_backward_source(entry, storage=storage, **values),
        entry=entry,
        abi_id=SM120_LION_BWD_ABIS[storage],
        buffer_names=(
            "param", "grad", "moment", "dparam_out", "dmoment_out",
            "dparam", "dgrad", "dmoment",
        ),
        scalar_names=("N",),
        provenance={
            "family": "optimizer_backward", "kind": "lion",
            "policy": "stop_sign", **values,
        },
        storage=storage,
    )


def package_optimizer_backward(
    *, kind: str, lr: float, momentum: float = 0.0, beta1: float = 0.9,
    beta2: float = 0.999, epsilon: float = 1.0e-8,
    weight_decay: float = 0.0, step: int = 1,
) -> NVIDIATrainingPackage:
    values = {
        "lr": lr, "momentum": momentum, "beta1": beta1, "beta2": beta2,
        "epsilon": epsilon, "weight_decay": weight_decay, "step": step,
    }
    digest = hashlib.sha256(
        repr((kind, sorted(values.items()))).encode()
    ).hexdigest()[:8]
    entry = f"tessera_cuda_training_optvjp_{kind}_f32_{digest}"
    source, names = _optimizer_backward_source(entry, kind=kind, **values)
    return _package(
        contract=f"tile.training.optimizer_vjp.{kind}.f32",
        source=source, entry=entry, abi_id=SM120_OPTIMIZER_BWD_ABIS[kind],
        buffer_names=names, scalar_names=("N",),
        provenance={"family": "optimizer_vjp", "kind": kind, **values},
        storage="f32",
    )


def package_adafactor_backward(
    *, topology: str, lr: float = 1.0e-3, beta2: float = 0.999,
    epsilon: float = 1.0e-30,
) -> NVIDIATrainingPackage:
    if topology not in {"full", "factored"}:
        raise ValueError("CUDA Adafactor VJP topology must be full or factored")
    values = {"lr": lr, "beta2": beta2, "epsilon": epsilon}
    digest = hashlib.sha256(
        repr((topology, sorted(values.items()))).encode()
    ).hexdigest()[:8]
    entry = f"tessera_cuda_training_adafactorvjp_{topology}_f32_{digest}"
    if topology == "full":
        source = _adafactor_full_backward_source(entry, **values)
        names = (
            "param", "grad", "state0", "dparam_out", "dparam", "dgrad",
            "dstate0",
        )
        scalars = ("N",)
    else:
        source = _adafactor_factored_backward_source(entry, **values)
        names = (
            "param", "grad", "state0", "state1", "dparam_out", "dparam",
            "dgrad", "dstate0", "dstate1",
        )
        scalars = ("Rows", "Columns")
    return _package(
        contract=f"tile.training.adafactor_vjp.{topology}.f32",
        source=source, entry=entry, abi_id=SM120_ADAFACTOR_BWD_ABIS[topology],
        buffer_names=names, scalar_names=scalars,
        provenance={"family": "adafactor_vjp", "topology": topology, **values},
        storage="f32",
    )


def package_deltanet_backward() -> NVIDIATrainingPackage:
    """Build the versioned SM120 four-stage DeltaNet package envelope."""
    entry = "tessera_cuda_training_deltanet_backward_f32_v2"
    return _package(
        contract="tile.training.deltanet_backward.causal.flags.f32",
        source=_deltanet_backward_source(entry), entry=entry,
        abi_id=SM120_DELTANET_BWD_ABIS["f32"],
        buffer_names=("q", "k", "v", "gate", "beta", "decay", "dy",
                      "dq", "dk", "dv", "dgate", "dbeta", "ddecay"),
        scalar_names=("B", "H", "S", "Dqk", "Dv", "HasGate", "HasBeta",
                      "HasDecay", "Erase", "Modified"),
        provenance={"family": "deltanet_backward", "causal": True,
                    "variants": "plain,gate,beta,decay,erase,modified; Dqk,Dv<=8",
                    "stages": ["checkpoint", "affine_summary", "serial_fill", "reverse"],
                    "schedule": "bounded_serial_fill_reverse"},
        storage="f32",
    )
def package_fused_loss_optimizer(
    *,
    loss_kind: str,
    optimizer: str,
    parameter: float,
    reduction: str,
    lr: float,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
    weight_decay: float = 0.0,
    step: int = 1,
    storage: str = "f32",
) -> NVIDIATrainingPackage:
    _validate_storage(storage)
    values = {
        "lr": lr,
        "momentum": momentum,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
        "step": step,
    }
    digest = hashlib.sha256(
        repr((loss_kind, optimizer, parameter, reduction, sorted(values.items()))).encode()
    ).hexdigest()[:8]
    entry = (
        f"tessera_cuda_training_fused_{loss_kind}_{reduction}_"
        f"{optimizer}_{storage}_{digest}"
    )
    source, names = _fused_source(
        entry,
        loss_kind=loss_kind,
        optimizer=optimizer,
        parameter=parameter,
        reduction=reduction,
        storage=storage,
        **values,
    )
    return _package(
        contract=(
            f"tile.training.fused.{loss_kind}.{optimizer}."
            f"{reduction}.{storage}"
        ),
        source=source,
        entry=entry,
        abi_id=(
            SM120_FUSED_LOSS_SGD_ABIS[storage]
            if optimizer == "sgd"
            else SM120_FUSED_LOSS_ADAMW_ABIS[storage]
        ),
        buffer_names=names,
        scalar_names=("N",),
        provenance={
            "family": "fused_loss_optimizer",
            "loss_kind": loss_kind,
            "optimizer": optimizer,
            "parameter": parameter,
            "reduction": reduction,
            **values,
        },
        storage=storage,
    )


__all__ = [
    "NVIDIATrainingPackage",
    "SM120_ADAM_ABIS",
    "SM120_ADAM_F32_ABI",
    "SM120_LION_BWD_ABIS",
    "SM120_DELTANET_BWD_ABIS",
    "SM120_LION_BWD_F32_ABI",
    "SM120_CLASS_BWD_ABIS",
    "SM120_CLASS_BWD_F32_ABI",
    "SM120_FUSED_LOSS_ADAMW_ABIS",
    "SM120_FUSED_LOSS_ADAMW_F32_ABI",
    "SM120_FUSED_LOSS_SGD_ABIS",
    "SM120_FUSED_LOSS_SGD_F32_ABI",
    "SM120_LOSS_BWD_ABIS",
    "SM120_LOSS_BWD_F32_ABI",
    "SM120_MOMENTUM_ABIS",
    "SM120_MOMENTUM_F32_ABI",
    "SM120_NORM_BWD_ABIS",
    "SM120_NORM_BWD_F32_ABI",
    "SM120_SGD_ABIS",
    "SM120_SGD_F32_ABI",
    "SM120_TRAINING_ABIS",
    "SM120_BROADCAST_REDUCE_ABIS",
    "package_broadcast_gradient",
    "package_class_backward",
    "package_fused_loss_optimizer",
    "package_adafactor_backward",
    "package_loss_backward",
    "package_lion_backward",
    "package_optimizer_backward",
    "package_deltanet_backward",
    "package_norm_backward",
    "package_optimizer",
    "training_storage_for_abi",
]
