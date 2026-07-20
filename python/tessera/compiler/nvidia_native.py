"""Compiler-owned SM120 native-image packaging.

NVIDIA-E2E-1 moves the production Tile -> NVIDIA -> NVVM -> PTX pipeline out
of the runtime. Canonical rank-2 Tensor Core matmuls, block-scaled NVFP4, and
row softmax are packaged with explicit storage-specific launch ABIs. The
runtime consumes only the resulting :class:`NativeImageArtifact` and
:class:`LaunchDescriptor`.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding,
    DeviceLibraryRecord,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ResourceRecord,
    ScalarArgument,
    ShapeGuard,
    WorkspaceRequirement,
)
from .nvidia_math_contract import CUDA_MATH_CONTRACT_VERSION


SM120_F16_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.v1"
SM120_BF16_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.bf16.v1"
SM120_TF32_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.f32_tf32.v1"
SM120_FP8_E4M3_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.fp8_e4m3.v1"
SM120_FP8_E5M2_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.fp8_e5m2.v1"
SM120_INT8_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.int8_i32.v1"
SM120_FP64_ABI = "tessera.nvidia.matmul.a_b_d_m_n_k.fp64.v1"
SM120_NVFP4_ABI = "tessera.nvidia.nvfp4.a_b_scale_a_scale_b_d_m_n_k.v1"
SM120_FP6_E2M3_ABI = "tessera.nvidia.mxfp6_e2m3.a_b_scale_a_scale_b_d_m_n_k.v1"
SM120_FP6_E3M2_ABI = "tessera.nvidia.mxfp6_e3m2.a_b_scale_a_scale_b_d_m_n_k.v1"
SM120_MXFP4_ABI = "tessera.nvidia.mxfp4.a_b_scale_a_scale_b_d_m_n_k.v1"
SM120_SOFTMAX_F16_ABI = "tessera.nvidia.softmax.x_o_rows_k.f16.v1"
SM120_SOFTMAX_F32_ABI = "tessera.nvidia.softmax.x_o_rows_k.f32.v1"
SM120_REDUCE_F16_ABI = "tessera.nvidia.reduce.x_o_outer_axis_inner.f16_f32acc.v2"
SM120_REDUCE_F32_ABI = "tessera.nvidia.reduce.x_o_outer_axis_inner.f32_f32acc.v2"
SM120_ATTN_F16_ABI = "tessera.nvidia.attention.q_k_v_o_dims.f16_f32acc.v1"
SM120_ATTN_F32_ABI = "tessera.nvidia.attention.q_k_v_o_dims.f32_f32acc.v1"
SM120_ATTN_BIAS_F16_ABI = "tessera.nvidia.attention.q_k_v_bias_o_dims.f16_f32acc.v1"
SM120_ATTN_BIAS_F32_ABI = "tessera.nvidia.attention.q_k_v_bias_o_dims.f32_f32acc.v1"
SM120_ATTN_BWD_F32_ABI = "tessera.nvidia.attention_backward.do_q_k_v_dq_dk_dv_dims.f32.v1"
SM120_ATTN_BWD_BIAS_F32_ABI = "tessera.nvidia.attention_backward.do_q_k_v_bias_dq_dk_dv_dims.f32.v1"
SM120_ATTN_BWD_F16_ABI = "tessera.nvidia.attention_backward.do_q_k_v_dq_dk_dv_dims.f16_f32acc.v2"
SM120_ATTN_BWD_BIAS_F16_ABI = "tessera.nvidia.attention_backward.do_q_k_v_bias_dq_dk_dv_dims.f16_f32acc.v2"
SM120_PAGED_KV_F32_ABI = "tessera.nvidia.paged_kv.pages_table_o_dims.f32_i32.v1"
SM120_PAGED_ATTN_F32_ABI = "tessera.nvidia.paged_attention.q_kp_vp_table_indices_o_dims.f32_i32_i64.v1"
SM120_REPLAY_DECODE_F32_ABI = "tessera.nvidia.replay_ssm.delta_x_b_s0_c_a_y_dims.f32.v1"
SM120_REPLAY_FLUSH_F32_ABI = "tessera.nvidia.replay_ssm.delta_x_b_s0_a_dims.f32.v1"
SM120_MOE_DISPATCH_F32_ABI = "tessera.nvidia.moe.dispatch.x_token_o_dims.f32_i32.v1"
SM120_MOE_COMBINE_F32_ABI = "tessera.nvidia.moe.combine.partials_token_weight_o_dims.f32_i32.v1"
SM120_GROUPED_GEMM_F32_ABI = "tessera.nvidia.moe.grouped_gemm.x_w_offsets_o_dims.f32_i32.v1"
SM120_MOE_DTYPES = ("fp16", "bf16", "fp32")
SM120_MOE_ABIS = tuple(
    f"tessera.nvidia.moe.{route}.{suffix}.{storage_ir}{tail}"
    for storage_ir in ("f16", "bf16")
    for route, suffix, tail in (
        ("dispatch", "x_token_o_dims", "_i32.v2"),
        ("combine", "partials_token_weight_o_dims", "_i32.v2"),
        ("grouped_gemm", "x_w_offsets_o_dims", "_f32acc_i32.v2"),
    )
) + (SM120_MOE_DISPATCH_F32_ABI, SM120_MOE_COMBINE_F32_ABI, SM120_GROUPED_GEMM_F32_ABI)
SM120_EPILOGUE_ABIS = tuple(
    f"tessera.nvidia.matmul.a_b_{suffix}_d_m_n_k.{storage}.v1"
    for storage in ("f16", "bf16", "tf32", "e4m3", "e5m2")
    for suffix in ("bias", "residual", "bias_residual")
)


@dataclass(frozen=True)
class NVIDIANativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


_cache: dict[
    str,
    tuple[
        str,
        str,
        Mapping[str, object],
        str,
        str,
        tuple[DeviceLibraryRecord, ...],
    ],
] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tool(name: str) -> Path | None:
    env_names = {
        "tessera-nvidia-opt": "TESSERA_NVIDIA_OPT",
        "mlir-opt": "MLIR_OPT",
        "mlir-translate": "MLIR_TRANSLATE",
        "llc": "LLC",
        "llvm-link": "LLVM_LINK",
        "ptxas": "PTXAS",
    }
    configured = os.environ.get(env_names[name])
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = _repo_root()
    candidates = {
        "tessera-nvidia-opt": root
        / ("build-nvidia-cuda/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt"),
        "mlir-opt": Path("/usr/lib/llvm-23/bin/mlir-opt"),
        "mlir-translate": Path("/usr/lib/llvm-23/bin/mlir-translate"),
        "llc": Path("/usr/lib/llvm-23/bin/llc"),
        "llvm-link": Path("/usr/lib/llvm-23/bin/llvm-link"),
        "ptxas": Path("/usr/local/cuda/bin/ptxas"),
    }
    candidate = candidates[name]
    if candidate.is_file():
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def tools_available() -> bool:
    return all(
        _tool(name) is not None
        for name in (
            "tessera-nvidia-opt",
            "mlir-opt",
            "mlir-translate",
            "llc",
            "llvm-link",
            "ptxas",
        )
    )


def _version(tool: Path) -> str:
    result = subprocess.run(
        [str(tool), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return hashlib.sha256(output.encode()).hexdigest() if output else tool.name


def _run(command: list[str], source: bytes) -> bytes:
    result = subprocess.run(command, input=source, capture_output=True)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"NVIDIA native packaging failed: {detail}")
    return result.stdout


def _cuda_libdevice() -> Path | None:
    configured = os.environ.get("TESSERA_CUDA_LIBDEVICE")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    roots = tuple(
        Path(value).expanduser()
        for value in (
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_PATH"),
            "/usr/local/cuda",
        )
        if value
    )
    for root in roots:
        path = root / "nvvm/libdevice/libdevice.10.bc"
        if path.is_file():
            return path
    return None


def _library_record(path: Path) -> DeviceLibraryRecord:
    return DeviceLibraryRecord(
        logical_name="cuda.libdevice",
        content_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        link_mode="llvm_link_only_needed",
    )


def _link_cuda_device_library_if_needed(
    llvm_ir: bytes,
    *,
    llvm_link: Path,
    libdevice: Path | None,
) -> tuple[bytes, tuple[DeviceLibraryRecord, ...]]:
    """Resolve retained ``__nv_*`` calls before NVPTX code generation."""
    if re.search(rb"(?m)^declare\b[^@]*@__nv_[A-Za-z0-9_]+", llvm_ir) is None:
        return llvm_ir, ()
    if libdevice is None:
        raise RuntimeError(
            "NVIDIA native packaging requires CUDA libdevice for retained __nv_* calls; "
            "set TESSERA_CUDA_LIBDEVICE or CUDA_HOME"
        )
    linked = _run(
        [
            str(llvm_link),
            "--only-needed",
            "-",
            str(libdevice),
            "-o",
            "-",
        ],
        llvm_ir,
    )
    return linked, (_library_record(libdevice),)


def emit_matmul_tile_ir(
    *,
    entry: str,
    storage: str,
    schedule: str = "shared",
    bias: bool = False,
    residual: bool = False,
    activation: str = "none",
) -> str:
    """Emit a typed production Tile matmul consumed by LowerTileToNVIDIA."""
    if storage not in {"f64", "f16", "bf16", "tf32", "e4m3", "e5m2", "s8"}:
        raise ValueError(f"unsupported SM120 canonical matmul storage {storage!r}")
    if schedule not in {"direct", "shared"}:
        raise ValueError(f"unsupported SM120 matmul schedule {schedule!r}")
    if storage in {"f64", "tf32", "e4m3", "e5m2", "s8"} and schedule != "direct":
        raise ValueError(f"SM120 {storage} canonical matmul requires direct schedule")
    if (bias or residual or activation != "none") and storage not in {"f16", "bf16", "tf32", "e4m3", "e5m2"}:
        raise ValueError("SM120 canonical fused epilogues require f16/bf16/TF32/FP8 matmul storage")
    if activation not in {"none", "relu", "gelu", "silu"}:
        raise ValueError(f"unsupported SM120 epilogue activation {activation!r}")
    warps, staging = (1, "global") if schedule == "direct" else (4, "shared")
    fragment_k = {
        "f64": 4,
        "f16": 16,
        "bf16": 16,
        "tf32": 8,
        "e4m3": 32,
        "e5m2": 32,
        "s8": 32,
    }[storage]
    accum = "s32" if storage == "s8" else storage if storage == "f64" else "f32"
    output = "i32" if storage == "s8" else storage if storage == "f64" else "f32"
    fragment_m = 8 if storage == "f64" else 16
    optional_args = ("%bias: !llvm.ptr, " if bias else "") + ("%residual: !llvm.ptr, " if residual else "")
    optional_operands = ("%bias, " if bias else "") + ("%residual, " if residual else "")
    residual_attrs = ', residual = true, epilogue_order = "matmul_bias_activation_residual"' if residual else ""
    return f'''module {{
  llvm.func @{entry}(%a: !llvm.ptr, %b: !llvm.ptr, {optional_args}%d: !llvm.ptr,
                     %m: i64, %n: i64, %k: i64) attributes {{nvvm.kernel}} {{
    tile.matmul_kernel %a, %b, {optional_operands}%d, %m, %n, %k {{
      mma = #tile.mma_desc<family = "mma_sync", m = {fragment_m}, n = 8, k = {fragment_k}, a = "{storage}", b = "{storage}", acc = "{accum}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = {str(bias).lower()}, activation = "{activation}", output = "{output}">,
      warps = {warps} : i64, staging = "{staging}"{residual_attrs}
    }} : !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (1 + int(bias) + int(residual))}i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_f16_matmul_tile_ir(*, entry: str, schedule: str = "shared") -> str:
    """Compatibility wrapper for the canonical f16 Tile matmul."""
    return emit_matmul_tile_ir(entry=entry, storage="f16", schedule=schedule)


def emit_nvfp4_matmul_tile_ir(*, entry: str) -> str:
    """Emit the typed general-shape NVFP4 launch kernel."""
    return f"""module {{
  llvm.func @{entry}(%a: !llvm.ptr, %b: !llvm.ptr,
                     %scale_a: !llvm.ptr, %scale_b: !llvm.ptr,
                     %d: !llvm.ptr, %m: i64, %n: i64, %k: i64)
      attributes {{nvvm.kernel}} {{
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {{
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
"""


def emit_softmax_tile_ir(*, entry: str, storage: str) -> str:
    """Emit the typed stable row-softmax launch envelope."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported SM120 softmax storage {storage!r}")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %rows: i64, %columns: i64) attributes {{nvvm.kernel}} {{
    tile.softmax_kernel %x, %o, %rows, %columns {{
      storage = "{storage}", accum = "f32", axis = -1 : i64,
      exp_mode = "approx_exp2", ftz = false
    }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''


def emit_mx_matmul_tile_ir(*, entry: str, storage: str) -> str:
    """Emit a typed FP6/MXFP4 launch kernel with explicit UE8M0 scales."""
    if storage not in {"e2m3", "e3m2", "fp4_e2m1"}:
        raise ValueError(f"unsupported SM120 MX matmul storage {storage!r}")
    fragment_k = 64 if storage == "fp4_e2m1" else 32
    return f'''module {{
  llvm.func @{entry}(%a: !llvm.ptr, %b: !llvm.ptr,
                     %scale_a: !llvm.ptr, %scale_b: !llvm.ptr,
                     %d: !llvm.ptr, %m: i64, %n: i64, %k: i64)
      attributes {{nvvm.kernel}} {{
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {{
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {fragment_k}, a = "{storage}", b = "{storage}", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_reduce_tile_ir(*, entry: str, storage: str, kind: str, axis: int = 0,
                        keepdims: bool = False, schedule: str = "serial") -> str:
    """Emit a typed arbitrary-axis reduction with explicit shape semantics."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported SM120 reduction storage {storage!r}")
    if kind not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported SM120 reduction kind {kind!r}")
    if axis < 0 or schedule not in {"serial", "cooperative_128"}:
        raise ValueError("SM120 reduction requires normalized axis and a proven schedule")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %outer: i64, %axis_extent: i64, %inner: i64) attributes {{nvvm.kernel}} {{
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {{
      storage = "{storage}", accum = "f32", kind = "{kind}",
      axis = {axis} : i64, keepdims = {str(keepdims).lower()},
      schedule = "{schedule}", nan_mode = "propagate"
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_attention_tile_ir(
    *,
    entry: str,
    storage: str,
    scale: float,
    causal: bool,
    bias: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> str:
    """Emit the correctness-first typed SDPA launch envelope."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported SM120 attention storage {storage!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SM120 attention scale must be finite and positive")
    if window_left < -1 or window_right < -1:
        raise ValueError("SM120 attention windows must be >= -1")
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("SM120 attention softcap must be finite and nonnegative")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("SM120 attention dropout_p must be in [0, 1)")
    optional_arg = "%bias: !llvm.ptr, " if bias else ""
    optional_operand = "%bias, " if bias else ""
    return f'''module {{
  llvm.func @{entry}(%q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr,
                     {optional_arg}%o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64,
                     %sq: i64, %sk: i64, %d: i64, %dv: i64)
      attributes {{nvvm.kernel}} {{
    tile.attention_kernel %q, %key, %v, {optional_operand}%o, %b, %hq, %hkv, %sq, %sk, %d, %dv {{
      storage = "{storage}", accum = "f32", scale = {scale:.17g} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {float(softcap)!r} : f32, dropout_p = {float(dropout_p)!r} : f32,
      dropout_seed = {dropout_seed} : i64
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (1 + int(bias))}i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_paged_kv_read_tile_ir(*, entry: str) -> str:
    return f'''module {{
  llvm.func @{entry}(%pages: !llvm.ptr, %table: !llvm.ptr, %o: !llvm.ptr,
                     %p: i64, %lp: i64, %ps: i64, %h: i64, %d: i64,
                     %start: i64, %tokens: i64) attributes {{nvvm.kernel}} {{
    tile.paged_kv_read_kernel %pages, %table, %o, %p, %lp, %ps, %h, %d, %start, %tokens {{
      storage = "f32", table_storage = "i32", route = "direct"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_paged_attention_tile_ir(*, entry: str, scale: float, causal: bool) -> str:
    """Emit the compiler-owned fused causal-offset paged-attention envelope."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SM120 paged attention scale must be finite and positive")
    return f'''module {{
  llvm.func @{entry}(%q: !llvm.ptr, %kp: !llvm.ptr, %vp: !llvm.ptr,
                     %table: !llvm.ptr, %indices: !llvm.ptr, %o: !llvm.ptr,
                     %p: i64, %lp: i64, %ps: i64, %h: i64, %qlen: i64,
                     %tokens: i64, %d: i64, %causal_offset: i64)
      attributes {{nvvm.kernel}} {{
    tile.paged_attention_kernel %q, %kp, %vp, %table, %indices, %o,
        %p, %lp, %ps, %h, %qlen, %tokens, %d, %causal_offset {{
      storage = "f32", accum = "f32", table_storage = "i32",
      token_index_storage = "i64", scale = {scale:.17g} : f32,
      causal = {str(causal).lower()}, route = "fused_direct"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        i64, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_replay_ssm_tile_ir(*, decode_entry: str, flush_entry: str) -> tuple[str, str]:
    """Emit the two compiler-owned kernels behind one resident ReplaySSM ring."""
    decode = f'''module {{
  llvm.func @{decode_entry}(%delta: !llvm.ptr, %x: !llvm.ptr, %bcoef: !llvm.ptr,
      %s0: !llvm.ptr, %c: !llvm.ptr, %a: !llvm.ptr, %y: !llvm.ptr,
      %batch: i64, %channels: i64, %state: i64, %tokens: i64)
      attributes {{nvvm.kernel}} {{
    tile.replay_ssm_decode_kernel %delta, %x, %bcoef, %s0, %c, %a, %y,
        %batch, %channels, %state, %tokens {{
      storage = "f32", route = "output_only"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }}
}}
'''
    flush = f'''module {{
  llvm.func @{flush_entry}(%delta: !llvm.ptr, %x: !llvm.ptr, %bcoef: !llvm.ptr,
      %s0: !llvm.ptr, %a: !llvm.ptr, %batch: i64, %channels: i64,
      %state: i64, %tokens: i64) attributes {{nvvm.kernel}} {{
    tile.replay_ssm_flush_kernel %delta, %x, %bcoef, %s0, %a, %batch,
        %channels, %state, %tokens {{
      storage = "f32", route = "state_and_output", deterministic = true
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }}
}}
'''
    return decode, flush


def emit_moe_tile_ir(*, entry: str, route: str, storage: str = "f32") -> str:
    """Emit one canonical local-device MoE transport/compute kernel."""
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported canonical MoE storage {storage!r}")
    if route == "dispatch":
        signature = "%x: !llvm.ptr, %token: !llvm.ptr, %o: !llvm.ptr, %t: i64, %s: i64, %h: i64"
        body = f'''tile.moe_dispatch_kernel %x, %token, %o, %t, %s, %h {{
      storage = "{storage}", index_storage = "i32"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64'''
    elif route == "combine":
        signature = "%partials: !llvm.ptr, %token: !llvm.ptr, %weights: !llvm.ptr, %o: !llvm.ptr, %t: i64, %s: i64, %h: i64"
        body = f'''tile.moe_combine_kernel %partials, %token, %weights, %o, %t, %s, %h {{
      storage = "{storage}", index_storage = "i32", deterministic = true
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64'''
    elif route == "grouped_gemm":
        signature = "%x: !llvm.ptr, %w: !llvm.ptr, %offsets: !llvm.ptr, %o: !llvm.ptr, %t: i64, %k: i64, %n: i64, %e: i64"
        body = f'''tile.grouped_gemm_kernel %x, %w, %offsets, %o, %t, %k, %n, %e {{
      storage = "{storage}", accum = "f32", index_storage = "i32"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64'''
    else:
        raise ValueError(f"unsupported canonical MoE route {route!r}")
    return f'''module {{
  llvm.func @{entry}({signature}) attributes {{nvvm.kernel}} {{
    {body}
    llvm.return
  }}
}}
'''


def emit_attention_backward_tile_ir(
    *, entry: str, scale: float, causal: bool, storage: str = "f32", bias: bool = False,
    window_left: int = -1, window_right: int = -1, softcap: float = 0.0,
    dropout_p: float = 0.0, dropout_seed: int = 0,
) -> str:
    """Emit the deterministic f16/f32 reference VJP through canonical Tile IR."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported SM120 attention backward storage {storage!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SM120 attention backward scale must be finite and positive")
    if window_left < -1 or window_right < -1:
        raise ValueError("SM120 attention backward windows must be >= -1")
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("SM120 attention backward softcap must be finite and nonnegative")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("SM120 attention backward dropout_p must be in [0, 1)")
    optional_arg = "%bias: !llvm.ptr, " if bias else ""
    optional_operand = "%bias, " if bias else ""
    return f'''module {{
  llvm.func @{entry}(%do: !llvm.ptr, %q: !llvm.ptr, %key: !llvm.ptr,
                     %v: !llvm.ptr, {optional_arg}%dq: !llvm.ptr,
                     %dk: !llvm.ptr, %dv: !llvm.ptr, %b: i64, %hq: i64,
                     %hkv: i64, %sq: i64, %sk: i64, %d: i64, %dv_dim: i64)
      attributes {{nvvm.kernel}} {{
    tile.attention_backward_kernel %do, %q, %key, %v, {optional_operand}%dq, %dk, %dv,
        %b, %hq, %hkv, %sq, %sk, %d, %dv_dim {{
      storage = "{storage}", accum = "f32", scale = {scale:.17g} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {float(softcap)!r} : f32,
      dropout_p = {float(dropout_p)!r} : f32, dropout_seed = {dropout_seed} : i64,
      route = "deterministic_direct",
      deterministic = true, workspace_bytes = 0 : i64
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (3 + int(bias))}i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_f32_softmax_tile_ir(*, entry: str) -> str:
    return emit_softmax_tile_ir(entry=entry, storage="f32")


def emit_f16_softmax_tile_ir(*, entry: str) -> str:
    return emit_softmax_tile_ir(entry=entry, storage="f16")


def _shape(module: GraphIRModule, name: str) -> tuple[int, ...] | None:
    fn = module.functions[0]
    arg = next((item for item in fn.args if item.name == name), None)
    rank = None if arg is None else arg.ir_type.rank
    if arg is None or rank is None or rank < 1:
        return None
    try:
        dims = tuple(int(dim) for dim in arg.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return dims if all(dim > 0 for dim in dims) else None


def _static_shape(module: GraphIRModule, name: str) -> tuple[int, int] | None:
    dims = _shape(module, name)
    return (dims[0], dims[1]) if dims is not None and len(dims) == 2 else None


def requests_softmax(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    op = module.functions[0].body[0]
    return op.op_name in {"tessera.softmax", "tessera.softmax_safe"}


def requests_reduction(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return module.functions[0].body[0].op_name in {
        "tessera.reduce",
        "tessera.sum",
        "tessera.mean",
        "tessera.max",
        "tessera.amax",
    }


def requests_attention(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return module.functions[0].body[0].op_name == "tessera.flash_attn"


def requests_attention_backward(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.flash_attn_bwd"
    )


def requests_paged_kv_read(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.kv_cache.read"
    )


def _paged_kv_contract(
    module: GraphIRModule,
) -> tuple[str, str, tuple[int, int, int, int, int, int, int]] | None:
    if not requests_paged_kv_read(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) != 2 or len(fn.result_types) != 1:
        return None
    pages_name, table_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    pages = args.get(pages_name)
    table = args.get(table_name)
    pages_shape = _shape(module, pages_name)
    table_shape = _shape(module, table_name)
    if (
        pages is None or table is None or pages.ir_type.dtype != "fp32"
        or table.ir_type.dtype != "int32" or pages_shape is None
        or len(pages_shape) != 4 or table_shape is None or len(table_shape) != 1
    ):
        return None
    p, page_size, heads, dim = pages_shape
    logical_pages = table_shape[0]
    start = int(op.kwargs.get("start", -1))
    end = int(op.kwargs.get("end", -1))
    tokens = end - start
    result = fn.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if (
        start < 0 or tokens <= 0 or end > logical_pages * page_size
        or result.dtype != "fp32" or result_shape != (tokens, heads, dim)
    ):
        return None
    return pages_name, table_name, (p, logical_pages, page_size, heads, dim, start, tokens)


def supports_paged_kv_read(module: GraphIRModule) -> bool:
    return _paged_kv_contract(module) is not None


def _attention_contract(
    module: GraphIRModule,
) -> tuple[
    str,
    tuple[int, int, int, int, int, int, int],
    float,
    bool,
    str | None,
    int,
    int,
    float,
    float,
    int,
] | None:
    if not requests_attention(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) not in {3, 4} or len(fn.result_types) != 1:
        return None
    names = tuple(value.removeprefix("%") for value in op.operands[:3])
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in names):
        return None
    storages = {args[name].ir_type.dtype for name in names}
    if len(storages) != 1 or storages.isdisjoint({"fp16", "fp32"}):
        return None
    storage = storages.pop()
    if storage is None:
        return None
    if storage not in {"fp16", "fp32"}:
        return None
    shapes = tuple(_shape(module, name) for name in names)
    if any(shape is None or len(shape) != 4 for shape in shapes):
        return None
    q_shape, k_shape, v_shape = shapes
    assert q_shape is not None and k_shape is not None and v_shape is not None
    b, hq, sq, d = q_shape
    bk, hkv, sk, dk = k_shape
    bv, hv, sv, dv = v_shape
    if b != bk or b != bv or hkv != hv or sk != sv or d != dk or hq % hkv:
        return None
    result = fn.result_types[0]
    try:
        result_shape = tuple(int(dim) for dim in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or result_shape != (b, hq, sq, dv):
        return None
    bias_name = op.operands[3].removeprefix("%") if len(op.operands) == 4 else None
    if bias_name is not None:
        bias_arg = args.get(bias_name)
        bias_shape = _shape(module, bias_name)
        if bias_arg is None or bias_arg.ir_type.dtype != "fp32" or bias_shape != (b, hq, sq, sk):
            return None
    window = op.kwargs.get("window")
    if window is None:
        window_left = int(op.kwargs.get("window_left", -1))
        window_right = int(op.kwargs.get("window_right", -1))
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window_left, window_right = (int(value) for value in window)
    else:
        window_left = window_right = int(window)
    if window_left < -1 or window_right < -1:
        return None
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    dropout_seed = int(op.kwargs.get("dropout_seed", op.kwargs.get("seed", 0)))
    if not math.isfinite(softcap) or softcap < 0.0:
        return None
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        return None
    scale = float(op.kwargs.get("scale", 1.0 / math.sqrt(float(d))))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return (
        storage, (b, hq, hkv, sq, sk, d, dv), scale,
        bool(op.kwargs.get("causal", False)), bias_name,
        window_left, window_right, softcap, dropout, dropout_seed,
    )


def supports_attention(module: GraphIRModule) -> bool:
    return _attention_contract(module) is not None


def _attention_backward_contract(
    module: GraphIRModule,
) -> tuple[
    str, tuple[str, ...], tuple[int, int, int, int, int, int, int], float, bool,
    str | None, int, int, float, float, int,
] | None:
    if not requests_attention_backward(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) not in {4, 5} or len(fn.result_types) != 3:
        return None
    names = tuple(value.removeprefix("%") for value in op.operands[:4])
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in names):
        return None
    storages = {args[name].ir_type.dtype for name in names}
    if len(storages) != 1 or not storages <= {"fp16", "fp32"}:
        return None
    storage = storages.pop()
    if storage is None:
        return None
    shapes = tuple(_shape(module, name) for name in names)
    if any(shape is None or len(shape) != 4 for shape in shapes):
        return None
    do_shape, q_shape, k_shape, v_shape = shapes
    assert do_shape is not None and q_shape is not None
    assert k_shape is not None and v_shape is not None
    b, hq, sq, d = q_shape
    bk, hkv, sk, dk = k_shape
    bv, hv, sv, dv = v_shape
    if (
        b != bk or b != bv or hkv != hv or sk != sv or d != dk or hq % hkv
        or do_shape != (b, hq, sq, dv)
    ):
        return None
    expected_results = (q_shape, k_shape, v_shape)
    for result, expected in zip(fn.result_types, expected_results):
        try:
            result_shape = tuple(int(dim) for dim in result.shape)
        except (TypeError, ValueError):
            return None
        if result.dtype != storage or result_shape != expected:
            return None
    bias_name = op.operands[4].removeprefix("%") if len(op.operands) == 5 else None
    if bias_name is not None:
        bias_arg = args.get(bias_name)
        if (
            bias_arg is None or bias_arg.ir_type.dtype != "fp32"
            or _shape(module, bias_name) != (b, hq, sq, sk)
        ):
            return None
    window = op.kwargs.get("window")
    if window is None:
        window_left = int(op.kwargs.get("window_left", -1))
        window_right = int(op.kwargs.get("window_right", -1))
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window_left, window_right = (int(value) for value in window)
    else:
        window_left = window_right = int(window)
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    scale = float(op.kwargs.get("scale", 1.0 / math.sqrt(float(d))))
    route = str(op.kwargs.get("route", "deterministic_direct"))
    deterministic = bool(op.kwargs.get("deterministic", True))
    workspace_limit = int(op.kwargs.get("workspace_limit_bytes", 0))
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    dropout_seed = int(op.kwargs.get("dropout_seed", op.kwargs.get("seed", 0)))
    if (
        window_left < -1 or window_right < -1 or not math.isfinite(softcap)
        or softcap < 0.0 or not math.isfinite(scale) or scale <= 0.0
        or route != "deterministic_direct" or not deterministic
        or workspace_limit < 0 or not math.isfinite(dropout)
        or not 0.0 <= dropout < 1.0
    ):
        return None
    return (
        storage, names, (b, hq, hkv, sq, sk, d, dv), scale,
        bool(op.kwargs.get("causal", False)), bias_name,
        window_left, window_right, softcap, dropout, dropout_seed,
    )


def supports_attention_backward(module: GraphIRModule) -> bool:
    return _attention_backward_contract(module) is not None


def _reduction_contract(module: GraphIRModule) -> tuple[str, str, int, bool] | None:
    if not requests_reduction(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) != 1 or len(fn.result_types) != 1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((item for item in fn.args if item.name == input_name), None)
    shape = _shape(module, input_name)
    if arg is None or arg.ir_type.dtype not in {"fp16", "fp32"} or shape is None or len(shape) < 1:
        return None
    raw_axis = op.kwargs.get("axis", -1)
    if not isinstance(raw_axis, int) or isinstance(raw_axis, bool):
        return None
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(shape): return None
    keepdims = bool(op.kwargs.get("keepdims", False))
    result = fn.result_types[0]
    try:
        result_shape = tuple(int(dim) for dim in result.shape)
    except (TypeError, ValueError):
        return None
    expected = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    if result.dtype != "fp32" or result_shape != expected:
        return None
    kind = (
        "max"
        if op.op_name in {"tessera.max", "tessera.amax"}
        else "sum"
        if op.op_name in {"tessera.reduce", "tessera.sum"}
        else "mean"
    )
    return arg.ir_type.dtype, kind, axis, keepdims


def supports_reduction(module: GraphIRModule) -> bool:
    return _reduction_contract(module) is not None


def _softmax_storage(module: GraphIRModule) -> str | None:
    if not requests_softmax(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) != 1 or op.kwargs.get("axis", -1) != -1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args or args[name].ir_type.dtype not in {"fp16", "fp32"}:
        return None
    storage = args[name].ir_type.dtype
    if fn.result_types and fn.result_types[0].dtype != storage:
        return None
    return storage if _shape(module, name) is not None else None


def supports_f32_softmax(module: GraphIRModule) -> bool:
    return _softmax_storage(module) == "fp32"


def supports_f16_softmax(module: GraphIRModule) -> bool:
    return _softmax_storage(module) == "fp16"


def _matmul_storage(module: GraphIRModule) -> str | None:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if op.op_name not in {"tessera.matmul", "tessera.gemm"} or len(op.operands) != 2:
        return None
    names = tuple(value[1:] if value.startswith("%") else value for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in names):
        return None
    storages = {args[name].ir_type.dtype for name in names}
    if len(storages) != 1:
        return None
    storage = storages.pop()
    if storage == "fp32":
        policy_mode = getattr(op.numeric_policy, "math_mode", None)
        kw_mode = op.kwargs.get("math_mode")
        if policy_mode != "tf32" and kw_mode != "tf32":
            return None
        storage = "tf32"
    if storage not in {
        "fp64",
        "fp16",
        "bf16",
        "tf32",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8",
    }:
        return None
    result_storage = "int32" if storage == "int8" else "fp64" if storage == "fp64" else "fp32"
    if fn.result_types and fn.result_types[0].dtype != result_storage:
        return None
    a_shape, b_shape = (_static_shape(module, name) for name in names)
    if not (a_shape and b_shape and a_shape[1] == b_shape[0]):
        return None
    return storage


def supports_matmul(module: GraphIRModule, *, storage: str | None = None) -> bool:
    selected = _matmul_storage(module)
    return selected is not None and (storage is None or selected == storage)


def supports_f16_matmul(module: GraphIRModule) -> bool:
    return supports_matmul(module, storage="fp16")


def supports_bf16_matmul(module: GraphIRModule) -> bool:
    return supports_matmul(module, storage="bf16")


def supports_tf32_matmul(module: GraphIRModule) -> bool:
    return supports_matmul(module, storage="tf32")


def supports_fp8_matmul(module: GraphIRModule) -> bool:
    return _matmul_storage(module) in {"fp8_e4m3", "fp8_e5m2"}


def supports_int8_matmul(module: GraphIRModule) -> bool:
    return supports_matmul(module, storage="int8")


def supports_fp64_matmul(module: GraphIRModule) -> bool:
    return supports_matmul(module, storage="fp64")


def _scale_names(module: GraphIRModule) -> tuple[str, str] | None:
    op = module.functions[0].body[0]
    values = (op.kwargs.get("scale_a"), op.kwargs.get("scale_b"))
    if any(not isinstance(value, str) for value in values):
        return None
    scale_a, scale_b = values
    assert isinstance(scale_a, str) and isinstance(scale_b, str)
    return (
        scale_a[1:] if scale_a.startswith("%") else scale_a,
        scale_b[1:] if scale_b.startswith("%") else scale_b,
    )


def requests_nvfp4_matmul(module: GraphIRModule) -> bool:
    """Return whether the Graph request selects the NVFP4 packaging lane.

    This intentionally checks only the operation and matrix storage dtype.  A
    malformed NVFP4 scale/epilogue contract must reach the NVFP4 packager and
    be rejected there instead of falling through to an unrelated f16 error.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn = module.functions[0]
    op = fn.body[0]
    if op.op_name not in {"tessera.matmul", "tessera.gemm"} or len(op.operands) != 2:
        return False
    args = {arg.name: arg for arg in fn.args}
    names = tuple(value.removeprefix("%") for value in op.operands)
    return all(name in args and args[name].ir_type.dtype == "nvfp4" for name in names)


def requests_mx_matmul(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn = module.functions[0]
    op = fn.body[0]
    if op.op_name not in {"tessera.matmul", "tessera.gemm"} or len(op.operands) != 2:
        return False
    args = {arg.name: arg for arg in fn.args}
    names = tuple(value.removeprefix("%") for value in op.operands)
    storages = {args[name].ir_type.dtype for name in names if name in args}
    return len(storages) == 1 and storages <= {
        "fp6_e2m3",
        "fp6_e3m2",
        "fp4_e2m1",
    }


def supports_mx_matmul(module: GraphIRModule) -> bool:
    if not requests_mx_matmul(module):
        return False
    fn = module.functions[0]
    op = fn.body[0]
    matrix_names = tuple(value.removeprefix("%") for value in op.operands)
    scale_names = _scale_names(module)
    if scale_names is None:
        return False
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in (*matrix_names, *scale_names)):
        return False
    if any(args[name].ir_type.dtype != "uint8" for name in scale_names):
        return False
    if fn.result_types and fn.result_types[0].dtype != "fp32":
        return False
    if (
        op.kwargs.get("bias") not in {None, False}
        or op.kwargs.get("residual") not in {None, False}
        or op.kwargs.get("activation", "none") != "none"
    ):
        return False
    a_shape, b_shape = (_static_shape(module, name) for name in matrix_names)
    sa_shape, sb_shape = (_static_shape(module, name) for name in scale_names)
    if not a_shape or not b_shape or a_shape[1] != b_shape[0]:
        return False
    m, k = a_shape
    n = b_shape[1]
    scale_k = (k + 31) // 32
    return sa_shape == (m, scale_k) and sb_shape == (scale_k, n)


def supports_nvfp4_matmul(module: GraphIRModule) -> bool:
    if not requests_nvfp4_matmul(module):
        return False
    fn = module.functions[0]
    op = fn.body[0]
    if op.op_name not in {"tessera.matmul", "tessera.gemm"} or len(op.operands) != 2:
        return False
    matrix_names = tuple(value[1:] if value.startswith("%") else value for value in op.operands)
    scale_names = _scale_names(module)
    if scale_names is None:
        return False
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in (*matrix_names, *scale_names)):
        return False
    if any(args[name].ir_type.dtype != "uint8" for name in scale_names):
        return False
    if fn.result_types and fn.result_types[0].dtype != "fp32":
        return False
    if (
        op.kwargs.get("bias") not in {None, False}
        or op.kwargs.get("residual") not in {None, False}
        or op.kwargs.get("activation", "none") != "none"
    ):
        return False
    a_shape, b_shape = (_static_shape(module, name) for name in matrix_names)
    sa_shape, sb_shape = (_static_shape(module, name) for name in scale_names)
    if not a_shape or not b_shape or a_shape[1] != b_shape[0]:
        return False
    m, k = a_shape
    n = b_shape[1]
    scale_k = (k + 15) // 16
    return sa_shape == (m, scale_k) and sb_shape == (scale_k, n)


def _resource_metrics(stderr: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    registers = re.search(r"Used\s+(\d+)\s+registers", stderr)
    if registers:
        metrics["registers_per_thread"] = int(registers.group(1))
    smem = re.search(r"(\d+)\s+bytes smem", stderr)
    metrics["static_shared_memory_bytes"] = int(smem.group(1)) if smem else 0
    spill_store = re.search(r"(\d+)\s+bytes spill stores", stderr)
    spill_load = re.search(r"(\d+)\s+bytes spill loads", stderr)
    metrics["spill_store_bytes"] = int(spill_store.group(1)) if spill_store else 0
    metrics["spill_load_bytes"] = int(spill_load.group(1)) if spill_load else 0
    return metrics


def _compile_tile_ir(
    tile_ir: str,
    entry: str,
) -> tuple[
    str,
    str,
    Mapping[str, object],
    str,
    str,
    tuple[DeviceLibraryRecord, ...],
    str,
]:
    tools = {
        name: _tool(name)
        for name in (
            "tessera-nvidia-opt",
            "mlir-opt",
            "mlir-translate",
            "llc",
            "llvm-link",
            "ptxas",
        )
    }
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise RuntimeError(f"NVIDIA native packaging tools unavailable: {missing}")
    paths = {name: path for name, path in tools.items() if path is not None}
    compiler_fp = "lower-tile-to-nvidia:" + hashlib.sha256(Path(paths["tessera-nvidia-opt"]).read_bytes()).hexdigest()
    toolchain_fp = ";".join(
        f"{name}={_version(paths[name])}" for name in ("mlir-opt", "mlir-translate", "llc", "llvm-link", "ptxas")
    )
    libdevice = _cuda_libdevice()
    libdevice_fp = _library_record(libdevice).content_digest if libdevice is not None else "unavailable"
    toolchain_fp += f";cuda.libdevice={libdevice_fp}"
    codegen_contract = "tessera.nvidia.native.llc-sm_120a.v1;" + CUDA_MATH_CONTRACT_VERSION
    cache_key = hashlib.sha256(f"{compiler_fp}\n{toolchain_fp}\n{codegen_contract}\n{tile_ir}".encode()).hexdigest()
    cached = _cache.get(cache_key)
    compile_state = "warm_cache" if cached is not None else "cold"
    if cached is None:
        lowered = _run(
            [
                str(paths["tessera-nvidia-opt"]),
                "--tessera-lower-to-nvidia-sm120",
            ],
            tile_ir.encode(),
        ).decode()
        llvm_mlir = _run(
            [
                str(paths["mlir-opt"]),
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-cf-to-llvm",
                "--reconcile-unrealized-casts",
            ],
            lowered.encode(),
        )
        llvm_ir = _run(
            [
                str(paths["mlir-translate"]),
                "--mlir-to-llvmir",
            ],
            llvm_mlir,
        )
        llvm_ir, device_libraries = _link_cuda_device_library_if_needed(
            llvm_ir,
            llvm_link=paths["llvm-link"],
            libdevice=libdevice,
        )
        ptx = _run(
            [
                str(paths["llc"]),
                "-mtriple=nvptx64-nvidia-cuda",
                "-mcpu=sm_120a",
                "-O3",
            ],
            llvm_ir,
        ).decode()
        if f".visible .entry {entry}" not in ptx:
            raise RuntimeError(f"NVIDIA native PTX is missing entry {entry}")
        with tempfile.TemporaryDirectory(prefix="tessera-sm120-package-") as tmp:
            ptx_path = Path(tmp) / "kernel.ptx"
            cubin_path = Path(tmp) / "kernel.cubin"
            ptx_path.write_text(ptx)
            assembled = subprocess.run(
                [
                    str(paths["ptxas"]),
                    "-arch=sm_120a",
                    "-v",
                    str(ptx_path),
                    "-o",
                    str(cubin_path),
                ],
                capture_output=True,
                text=True,
            )
            if assembled.returncode:
                raise RuntimeError(f"NVIDIA native PTX assembly failed: {assembled.stderr.strip()}")
            metrics: Mapping[str, object] = _resource_metrics(assembled.stderr)
        cached = (
            lowered,
            ptx,
            metrics,
            compiler_fp,
            toolchain_fp,
            device_libraries,
        )
        _cache[cache_key] = cached
    lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries = cached
    return (
        lowered,
        ptx,
        metrics,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    )


def package_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    schedule: str = "auto",
) -> NVIDIANativePackage:
    """Compile and package one canonical static-shape Graph matmul."""
    storage = _matmul_storage(module)
    if storage is None:
        raise ValueError(
            "SM120 native packaging requires one static rank-2 proven-storage "
            "matmul with its required accumulator output"
        )
    storage_ir = {
        "fp64": "f64",
        "fp16": "f16",
        "bf16": "bf16",
        "tf32": "tf32",
        "fp8_e4m3": "e4m3",
        "fp8_e5m2": "e5m2",
        "int8": "s8",
    }[storage]
    fn = module.functions[0]
    op = fn.body[0]
    activation = str(op.kwargs.get("activation", "none"))
    bias_value = op.kwargs.get("bias")
    residual_value = op.kwargs.get("residual")
    bias_name = bias_value.removeprefix("%") if isinstance(bias_value, str) else None
    residual_name = residual_value.removeprefix("%") if isinstance(residual_value, str) else None
    fused = bias_value not in {None, False} or residual_value not in {None, False} or activation != "none"
    if fused:
        if storage not in {"fp16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("SM120 canonical fused epilogues require f16/bf16/TF32/FP8 matmul storage")
        if bias_value not in {None, False} and bias_name is None:
            raise ValueError("SM120 fused bias must name a Graph argument")
        if residual_value not in {None, False} and residual_name is None:
            raise ValueError("SM120 fused residual must name a Graph argument")
        if activation not in {"none", "relu", "gelu", "silu"}:
            raise ValueError(f"unsupported SM120 fused activation {activation!r}")
    abi_id = {
        "fp64": SM120_FP64_ABI,
        "fp16": SM120_F16_ABI,
        "bf16": SM120_BF16_ABI,
        "tf32": SM120_TF32_ABI,
        "fp8_e4m3": SM120_FP8_E4M3_ABI,
        "fp8_e5m2": SM120_FP8_E5M2_ABI,
        "int8": SM120_INT8_ABI,
    }[storage]
    if fused and (bias_name or residual_name):
        suffix = "bias_residual" if bias_name and residual_name else "bias" if bias_name else "residual"
        abi_id = f"tessera.nvidia.matmul.a_b_{suffix}_d_m_n_k.{storage_ir}.v1"
    if schedule == "auto":
        schedule = "shared" if storage in {"fp16", "bf16"} else "direct"
    entry = (
        f"tessera_tile_matmul_fused_{storage_ir}_{activation}_b{int(bool(bias_name))}_r{int(bool(residual_name))}"
        if fused
        else f"tessera_tile_matmul_{schedule}_{storage_ir}"
    )
    tile_ir = emit_matmul_tile_ir(
        entry=entry,
        storage=storage_ir,
        schedule=schedule,
        bias=bool(bias_name),
        residual=bool(residual_name),
        activation=activation,
    )
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )

    # The package identity is tied to the actual production-pass Target IR.
    target_ir_digest = hashlib.sha256(lowered.encode()).hexdigest()
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=target_ir_digest,
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    a_name, b_name = tuple(value[1:] if value.startswith("%") else value for value in op.operands)
    a_shape = _static_shape(module, a_name)
    b_shape = _static_shape(module, b_name)
    assert a_shape is not None and b_shape is not None
    m, k = a_shape
    _, n = b_shape
    args = {arg.name: arg for arg in fn.args}
    if bias_name and (
        bias_name not in args or _shape(module, bias_name) != (n,) or args[bias_name].ir_type.dtype != "fp32"
    ):
        raise ValueError("SM120 fused bias requires an fp32 [N] argument")
    if residual_name and (
        residual_name not in args
        or _shape(module, residual_name) != (m, n)
        or args[residual_name].ir_type.dtype != "fp32"
    ):
        raise ValueError("SM120 fused residual requires an fp32 [M,N] argument")
    output_name = op.result or "output"
    buffer_rows = [
        (
            a_name,
            "input",
            "fp32" if storage == "tf32" else storage,
            2,
            8 if storage == "fp64" else 4 if storage == "tf32" else 2 if storage in {"fp16", "bf16"} else 1,
            "row_major",
        ),
        (
            b_name,
            "input",
            "fp32" if storage == "tf32" else storage,
            2,
            8 if storage == "fp64" else 4 if storage == "tf32" else 2 if storage in {"fp16", "bf16"} else 1,
            "col_major",
        ),
    ]
    if bias_name:
        buffer_rows.append((bias_name, "input", "fp32", 1, 4, "row_major"))
    if residual_name:
        buffer_rows.append((residual_name, "input", "fp32", 2, 4, "row_major"))
    buffer_rows.append(
        (
            output_name,
            "output",
            "int32" if storage == "int8" else storage if storage == "fp64" else "fp32",
            2,
            8 if storage == "fp64" else 4,
            "row_major",
        )
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=tuple(
            BufferBinding(index, name, role, dtype, rank, layout, alignment)
            for index, (name, role, dtype, rank, alignment, layout) in enumerate(buffer_rows)
        ),
        scalars=(
            ScalarArgument(len(buffer_rows), "M", "int64"),
            ScalarArgument(len(buffer_rows) + 1, "N", "int64"),
            ScalarArgument(len(buffer_rows) + 2, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard(a_name, 0, "eq", m),
            ShapeGuard(a_name, 1, "eq", k),
            ShapeGuard(b_name, 0, "eq", k),
            ShapeGuard(b_name, 1, "eq", n),
            *(() if bias_name is None else (ShapeGuard(bias_name, 0, "eq", n),)),
            *(
                ()
                if residual_name is None
                else (
                    ShapeGuard(residual_name, 0, "eq", m),
                    ShapeGuard(residual_name, 1, "eq", n),
                )
            ),
            ShapeGuard(output_name, 0, "eq", m),
            ShapeGuard(output_name, 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy=f"sm120_matmul_{schedule}_mn"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": schedule,
            "shape": [m, n, k],
            "storage": storage_ir,
            "epilogue": {
                "bias": bool(bias_name),
                "activation": activation,
                "residual": bool(residual_name),
                "order": ["matmul", "bias", "activation", "residual"],
            },
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_f16_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    schedule: str = "shared",
) -> NVIDIANativePackage:
    if not supports_f16_matmul(module):
        raise ValueError("SM120 f16 matmul packaging requires fp16 storage")
    return package_matmul(module, pipeline_name=pipeline_name, schedule=schedule)


def package_bf16_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    schedule: str = "shared",
) -> NVIDIANativePackage:
    if not supports_bf16_matmul(module):
        raise ValueError("SM120 bf16 matmul packaging requires bf16 storage")
    return package_matmul(module, pipeline_name=pipeline_name, schedule=schedule)


def package_nvfp4_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    """Compile and package one static logical-shape NVFP4 Graph matmul."""
    if not supports_nvfp4_matmul(module):
        raise ValueError("SM120 NVFP4 packaging requires one static rank-2 matmul with logical scale_a/scale_b views")
    entry = "tessera_tile_matmul_nvfp4"
    tile_ir = emit_nvfp4_matmul_tile_ir(entry=entry)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, SM120_NVFP4_ABI),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    fn = module.functions[0]
    op = fn.body[0]
    a_name, b_name = tuple(value[1:] if value.startswith("%") else value for value in op.operands)
    scale_names = _scale_names(module)
    assert scale_names is not None
    scale_a_name, scale_b_name = scale_names
    a_shape = _static_shape(module, a_name)
    b_shape = _static_shape(module, b_name)
    assert a_shape is not None and b_shape is not None
    m, k = a_shape
    n = b_shape[1]
    packed_k = (k + 1) // 2
    scale_k = (k + 15) // 16
    output_name = op.result or "output"
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=SM120_NVFP4_ABI,
        buffers=(
            BufferBinding(0, a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, scale_a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(3, scale_b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, output_name, "output", "fp32", 2, "row_major", 4),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard(a_name, 0, "eq", m),
            ShapeGuard(a_name, 1, "eq", packed_k),
            ShapeGuard(b_name, 0, "eq", packed_k),
            ShapeGuard(b_name, 1, "eq", n),
            ShapeGuard(scale_a_name, 0, "eq", m),
            ShapeGuard(scale_a_name, 1, "eq", scale_k),
            ShapeGuard(scale_b_name, 0, "eq", scale_k),
            ShapeGuard(scale_b_name, 1, "eq", n),
            ShapeGuard(output_name, 0, "eq", m),
            ShapeGuard(output_name, 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy="sm120_nvfp4_m16n8k64"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-1",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "warp_m16n8_k64",
            "shape": [m, n, k],
            "scale_vector_size": 16,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_mx_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    """Compile and package one static FP6/MXFP4 matmul with UE8M0 scales."""
    if not supports_mx_matmul(module):
        raise ValueError(
            "SM120 MX packaging requires one static rank-2 FP6/MXFP4 matmul with logical UE8M0 scale_a/scale_b views"
        )
    fn = module.functions[0]
    op = fn.body[0]
    a_name, b_name = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    storage = args[a_name].ir_type.dtype
    if storage not in {"fp6_e2m3", "fp6_e3m2", "fp4_e2m1"}:
        raise ValueError("SM120 MX packaging received an unsupported logical storage dtype")
    physical = {
        "fp6_e2m3": "e2m3",
        "fp6_e3m2": "e3m2",
        "fp4_e2m1": "fp4_e2m1",
    }[storage]
    abi_id = {
        "fp6_e2m3": SM120_FP6_E2M3_ABI,
        "fp6_e3m2": SM120_FP6_E3M2_ABI,
        "fp4_e2m1": SM120_MXFP4_ABI,
    }[storage]
    entry = f"tessera_tile_matmul_mx_{physical}"
    tile_ir = emit_mx_matmul_tile_ir(entry=entry, storage=physical)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    scale_names = _scale_names(module)
    assert scale_names is not None
    scale_a_name, scale_b_name = scale_names
    a_shape = _static_shape(module, a_name)
    b_shape = _static_shape(module, b_name)
    assert a_shape is not None and b_shape is not None
    m, k = a_shape
    n = b_shape[1]
    physical_k = k if storage.startswith("fp6") else (k + 1) // 2
    scale_k = (k + 31) // 32
    output_name = op.result or "output"
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, scale_a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(3, scale_b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, output_name, "output", "fp32", 2, "row_major", 4),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard(a_name, 0, "eq", m),
            ShapeGuard(a_name, 1, "eq", physical_k),
            ShapeGuard(b_name, 0, "eq", physical_k),
            ShapeGuard(b_name, 1, "eq", n),
            ShapeGuard(scale_a_name, 0, "eq", m),
            ShapeGuard(scale_a_name, 1, "eq", scale_k),
            ShapeGuard(scale_b_name, 0, "eq", scale_k),
            ShapeGuard(scale_b_name, 1, "eq", n),
            ShapeGuard(output_name, 0, "eq", m),
            ShapeGuard(output_name, 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy="sm120_mx_m16n8"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "warp_m16n8_mx",
            "shape": [m, n, k],
            "storage": storage,
            "scale_dtype": "ue8m0",
            "scale_vector_size": 32,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_reduction(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    schedule: str = "serial",
) -> NVIDIANativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 reduction packaging requires static f16/f32 input, f32 output, "
            "one normalized axis and sum/mean/max semantics"
        )
    storage, kind, axis, keepdims = contract
    if schedule not in {"serial", "cooperative_128"}:
        raise ValueError("SM120 reduction schedule must be serial or cooperative_128")
    storage_ir = "f16" if storage == "fp16" else "f32"
    entry = f"tessera_tile_reduce_{kind}_{storage_ir}_{schedule}"
    abi_id = SM120_REDUCE_F16_ABI if storage == "fp16" else SM120_REDUCE_F32_ABI
    tile_ir = emit_reduce_tile_ir(
        entry=entry, storage=storage_ir, kind=kind, axis=axis,
        keepdims=keepdims, schedule=schedule,
    )
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    fn = module.functions[0]
    op = fn.body[0]
    input_name = op.operands[0].removeprefix("%")
    output_name = op.result or "output"
    shape = _shape(module, input_name)
    assert shape is not None
    outer = math.prod(shape[:axis]) if axis else 1
    axis_extent = shape[axis]
    inner = math.prod(shape[axis + 1:]) if axis + 1 < len(shape) else 1
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", storage, len(shape), "row_major", 2 if storage == "fp16" else 4),
            BufferBinding(1, output_name, "output", "fp32", len(output_shape), "row_major", 4),
        ),
        scalars=(ScalarArgument(2, "Outer", "int64"),
                 ScalarArgument(3, "AxisExtent", "int64"),
                 ScalarArgument(4, "Inner", "int64")),
        shape_guards=tuple(
            [ShapeGuard(input_name, axis, "eq", extent) for axis, extent in enumerate(shape)]
            + [ShapeGuard(output_name, axis, "eq", extent) for axis, extent in enumerate(output_shape)]
        ),
        geometry=LaunchGeometry(policy=f"sm120_reduce_{schedule}"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": schedule,
            "shape": list(shape),
            "storage": storage_ir,
            "accum": "f32",
            "kind": kind,
            "axis": axis,
            "keepdims": keepdims,
            "nan_mode": "propagate",
            "outer": outer,
            "axis_extent": axis_extent,
            "inner": inner,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_attention(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    contract = _attention_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 attention packaging requires static rank-4 f16/f32 Q/K/V, "
            "f32 output, MHA/GQA-compatible heads, and scale/causal semantics; "
            "bias, window, softcap, and dropout remain planned"
        )
    (
        storage, dims, scale, causal, bias_name, window_left, window_right,
        softcap, dropout_p, dropout_seed,
    ) = contract
    storage_ir = "f16" if storage == "fp16" else "f32"
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:{window_right}:"
        f"{softcap:.17g}:{dropout_p:.17g}:{dropout_seed}".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_attention_{storage_ir}_{'causal' if causal else 'full'}_{semantic_key}"
    abi_id = (
        SM120_ATTN_BIAS_F16_ABI if storage == "fp16" else SM120_ATTN_BIAS_F32_ABI
    ) if bias_name else (
        SM120_ATTN_F16_ABI if storage == "fp16" else SM120_ATTN_F32_ABI
    )
    tile_ir = emit_attention_tile_ir(
        entry=entry, storage=storage_ir, scale=scale, causal=causal,
        bias=bias_name is not None, window_left=window_left,
        window_right=window_right, softcap=softcap,
        dropout_p=dropout_p, dropout_seed=dropout_seed,
    )
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v", metrics=metrics
        ),
    )
    fn = module.functions[0]
    op = fn.body[0]
    q_name, k_name, v_name = (value.removeprefix("%") for value in op.operands[:3])
    output_name = op.result or "output"
    b, hq, hkv, sq, sk, d, dv = dims
    alignment = 2 if storage == "fp16" else 4
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=tuple([
            BufferBinding(0, q_name, "input", storage, 4, "row_major", alignment),
            BufferBinding(1, k_name, "input", storage, 4, "row_major", alignment),
            BufferBinding(2, v_name, "input", storage, 4, "row_major", alignment),
        ] + ([BufferBinding(3, bias_name, "input", "fp32", 4, "row_major", 4)] if bias_name else [])
          + [BufferBinding(3 + int(bias_name is not None), output_name, "output", "fp32", 4, "row_major", 4)]),
        scalars=tuple(
            ScalarArgument(4 + int(bias_name is not None) + index, name, "int64")
            for index, name in enumerate(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"))
        ),
        shape_guards=tuple([
            ShapeGuard(q_name, 0, "eq", b), ShapeGuard(q_name, 1, "eq", hq),
            ShapeGuard(q_name, 2, "eq", sq), ShapeGuard(q_name, 3, "eq", d),
            ShapeGuard(k_name, 0, "eq", b), ShapeGuard(k_name, 1, "eq", hkv),
            ShapeGuard(k_name, 2, "eq", sk), ShapeGuard(k_name, 3, "eq", d),
            ShapeGuard(v_name, 0, "eq", b), ShapeGuard(v_name, 1, "eq", hkv),
            ShapeGuard(v_name, 2, "eq", sk), ShapeGuard(v_name, 3, "eq", dv),
            ShapeGuard(output_name, 0, "eq", b), ShapeGuard(output_name, 1, "eq", hq),
            ShapeGuard(output_name, 2, "eq", sq), ShapeGuard(output_name, 3, "eq", dv),
        ] + ([
            ShapeGuard(bias_name, 0, "eq", b), ShapeGuard(bias_name, 1, "eq", hq),
            ShapeGuard(bias_name, 2, "eq", sq), ShapeGuard(bias_name, 3, "eq", sk),
        ] if bias_name else [])),
        geometry=LaunchGeometry(policy="sm120_attention_thread_per_output_128"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("completion",)
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "thread_per_output_128",
            "storage": storage_ir,
            "accum": "f32",
            "output": "f32",
            "shape": list(dims),
            "scale": scale,
            "causal": causal,
            "bias": bias_name is not None,
            "window_left": window_left,
            "window_right": window_right,
            "softcap": softcap,
            "dropout_p": dropout_p,
            "dropout_seed": dropout_seed,
            "dropout_rng": "lcg32_counter_v1",
            "limitations": [],
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_attention_backward(
    module: GraphIRModule, *, pipeline_name: str
) -> NVIDIANativePackage:
    contract = _attention_backward_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 canonical attention backward requires static rank-4 matching f16/f32 "
            "dO/Q/K/V and gradients, deterministic_direct, valid dropout, and a "
            "nonnegative workspace limit"
        )
    (storage, names, dims, scale, causal, bias_name, window_left, window_right,
     softcap, dropout_p, dropout_seed) = contract
    storage_ir = "f16" if storage == "fp16" else "f32"
    do_name, q_name, k_name, v_name = names
    b, hq, hkv, sq, sk, d, dv = dims
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:{window_right}:"
        f"{softcap:.17g}:{dropout_p:.17g}:{dropout_seed}:deterministic_direct".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_attention_backward_{storage_ir}_deterministic_{semantic_key}"
    if storage == "fp16":
        abi_id = SM120_ATTN_BWD_BIAS_F16_ABI if bias_name else SM120_ATTN_BWD_F16_ABI
    else:
        abi_id = SM120_ATTN_BWD_BIAS_F32_ABI if bias_name else SM120_ATTN_BWD_F32_ABI
    tile_ir = emit_attention_backward_tile_ir(
        entry=entry, storage=storage_ir, scale=scale, causal=causal, bias=bias_name is not None,
        window_left=window_left, window_right=window_right, softcap=softcap,
        dropout_p=dropout_p, dropout_seed=dropout_seed,
    )
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp, toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),), compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    result_names = module.functions[0].body[0].result_names
    if len(result_names) != 3:
        raise ValueError("SM120 attention backward needs dQ,dK,dV SSA result names")
    dq_name, dk_name, dv_name = result_names
    input_bindings = [
        BufferBinding(0, do_name, "input", storage, 4, "row_major", 2 if storage == "fp16" else 4),
        BufferBinding(1, q_name, "input", storage, 4, "row_major", 2 if storage == "fp16" else 4),
        BufferBinding(2, k_name, "input", storage, 4, "row_major", 2 if storage == "fp16" else 4),
        BufferBinding(3, v_name, "input", storage, 4, "row_major", 2 if storage == "fp16" else 4),
    ]
    if bias_name:
        input_bindings.append(
            BufferBinding(4, bias_name, "input", "fp32", 4, "row_major", 4)
        )
    output_base = 4 + int(bias_name is not None)
    buffers = tuple(input_bindings + [
        BufferBinding(output_base, dq_name, "output", storage, 4, "row_major", 2 if storage == "fp16" else 4),
        BufferBinding(output_base + 1, dk_name, "output", storage, 4, "row_major", 2 if storage == "fp16" else 4),
        BufferBinding(output_base + 2, dv_name, "output", storage, 4, "row_major", 2 if storage == "fp16" else 4),
    ])
    scalar_base = output_base + 3
    guards = [
        ShapeGuard(do_name, 0, "eq", b), ShapeGuard(do_name, 1, "eq", hq),
        ShapeGuard(do_name, 2, "eq", sq), ShapeGuard(do_name, 3, "eq", dv),
        ShapeGuard(q_name, 0, "eq", b), ShapeGuard(q_name, 1, "eq", hq),
        ShapeGuard(q_name, 2, "eq", sq), ShapeGuard(q_name, 3, "eq", d),
        ShapeGuard(k_name, 0, "eq", b), ShapeGuard(k_name, 1, "eq", hkv),
        ShapeGuard(k_name, 2, "eq", sk), ShapeGuard(k_name, 3, "eq", d),
        ShapeGuard(v_name, 0, "eq", b), ShapeGuard(v_name, 1, "eq", hkv),
        ShapeGuard(v_name, 2, "eq", sk), ShapeGuard(v_name, 3, "eq", dv),
        ShapeGuard(dq_name, 0, "eq", b), ShapeGuard(dq_name, 1, "eq", hq),
        ShapeGuard(dq_name, 2, "eq", sq), ShapeGuard(dq_name, 3, "eq", d),
        ShapeGuard(dk_name, 0, "eq", b), ShapeGuard(dk_name, 1, "eq", hkv),
        ShapeGuard(dk_name, 2, "eq", sk), ShapeGuard(dk_name, 3, "eq", d),
        ShapeGuard(dv_name, 0, "eq", b), ShapeGuard(dv_name, 1, "eq", hkv),
        ShapeGuard(dv_name, 2, "eq", sk), ShapeGuard(dv_name, 3, "eq", dv),
    ]
    if bias_name:
        guards.extend([
            ShapeGuard(bias_name, 0, "eq", b), ShapeGuard(bias_name, 1, "eq", hq),
            ShapeGuard(bias_name, 2, "eq", sq), ShapeGuard(bias_name, 3, "eq", sk),
        ])
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=abi_id,
        buffers=buffers,
        scalars=tuple(ScalarArgument(scalar_base + index, name, "int64")
                      for index, name in enumerate(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"))),
        shape_guards=tuple(guards),
        geometry=LaunchGeometry(policy="sm120_attention_backward_deterministic_direct_128"),
        workspace=WorkspaceRequirement(bytes=0, alignment=4),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("completion",)
        ),
        provenance={
            "work_item": "NVIDIA-PARITY-ATTN-BWD", "sync_key": "E2E-SPINE-2026-07-18",
            "route": "deterministic_direct", "candidate_role": "canonical_reference",
            "deterministic": True, "dk_dv_reduction": "single_owner_fixed_order",
            "workspace_bytes": 0, "storage": storage_ir, "accum": "f32",
            "shape": list(dims), "scale": scale, "causal": causal,
            "bias": bias_name is not None, "window_left": window_left,
            "window_right": window_right, "softcap": softcap,
            "dropout_p": dropout_p, "dropout_seed": dropout_seed,
            "dropout_rng": "lcg32_counter_v1", "limitations": [],
            "comparison_candidates": ["atomic", "split_reduced"],
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_paged_kv_read(
    module: GraphIRModule, *, pipeline_name: str
) -> NVIDIANativePackage:
    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 paged-KV packaging requires static f32 [P,PS,H,D] pages, "
            "rank-1 int32 page table, explicit valid start/end, and f32 output"
        )
    pages_name, table_name, dims = contract
    p, logical_pages, page_size, heads, dim, start, tokens = dims
    entry = "tessera_tile_paged_kv_read_f32_direct"
    tile_ir = emit_paged_kv_read_tile_ir(entry=entry)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp, toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, SM120_PAGED_KV_F32_ABI),),
        compile_state=compile_state, device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    output_name = module.functions[0].body[0].result or "output"
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=SM120_PAGED_KV_F32_ABI,
        buffers=(
            BufferBinding(0, pages_name, "input", "fp32", 4, "row_major", 4),
            BufferBinding(1, table_name, "input", "int32", 1, "row_major", 4),
            BufferBinding(2, output_name, "output", "fp32", 3, "row_major", 4),
        ),
        scalars=tuple(ScalarArgument(3 + index, name, "int64") for index, name in enumerate(
            ("P", "LP", "PageSize", "H", "D", "Start", "Tokens")
        )),
        shape_guards=(
            ShapeGuard(pages_name, 0, "eq", p), ShapeGuard(pages_name, 1, "eq", page_size),
            ShapeGuard(pages_name, 2, "eq", heads), ShapeGuard(pages_name, 3, "eq", dim),
            ShapeGuard(table_name, 0, "eq", logical_pages),
            ShapeGuard(output_name, 0, "eq", tokens), ShapeGuard(output_name, 1, "eq", heads),
            ShapeGuard(output_name, 2, "eq", dim),
        ),
        geometry=LaunchGeometry(policy="sm120_paged_kv_direct_256"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("completion",)
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2", "sync_key": "E2E-SPINE-2026-07-18",
            "route": "direct", "shape": list(dims), "storage": "f32",
            "table_storage": "i32", "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_paged_attention(
    *, physical_pages: int, logical_pages: int, page_size: int, heads: int,
    query_length: int, tokens: int, dim: int, scale: float,
    causal: bool, causal_offset: int, pipeline_name: str,
) -> NVIDIANativePackage:
    """Package a true fused page-table/causal-offset attention image."""
    if min(physical_pages, logical_pages, page_size, heads, query_length, tokens, dim) <= 0:
        raise ValueError("SM120 paged attention dimensions must be positive")
    if tokens > logical_pages * page_size:
        raise ValueError("SM120 paged attention tokens exceed logical page capacity")
    if causal_offset < 0 or (causal and causal_offset + query_length > tokens):
        raise ValueError("SM120 paged attention causal offset is outside the logical token range")
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{causal_offset}".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_paged_attention_f32_fused_{semantic_key}"
    tile_ir = emit_paged_attention_tile_ir(entry=entry, scale=scale, causal=causal)
    lowered, ptx, image = _package_direct_image(
        tile_ir, entry=entry, abi_id=SM120_PAGED_ATTN_F32_ABI,
        pipeline_name=pipeline_name,
    )
    dims = (physical_pages, logical_pages, page_size, heads, query_length,
            tokens, dim, causal_offset)
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry,
        abi_id=SM120_PAGED_ATTN_F32_ABI,
        buffers=(
            BufferBinding(0, "Q", "input", "fp32", 3, "row_major", 4),
            BufferBinding(1, "K_pages", "input", "fp32", 4, "row_major", 4),
            BufferBinding(2, "V_pages", "input", "fp32", 4, "row_major", 4),
            BufferBinding(3, "page_table", "input", "int32", 1, "row_major", 4),
            BufferBinding(4, "token_indices", "input", "int64", 1, "row_major", 8),
            BufferBinding(5, "O", "output", "fp32", 3, "row_major", 4),
        ),
        scalars=tuple(ScalarArgument(6 + index, name, "int64") for index, name in enumerate(
            ("P", "LP", "PageSize", "H", "QueryLength", "Tokens", "D", "CausalOffset")
        )),
        shape_guards=(
            ShapeGuard("Q", 0, "eq", heads), ShapeGuard("Q", 1, "eq", query_length),
            ShapeGuard("Q", 2, "eq", dim), ShapeGuard("K_pages", 0, "eq", physical_pages),
            ShapeGuard("K_pages", 1, "eq", page_size), ShapeGuard("K_pages", 2, "eq", heads),
            ShapeGuard("K_pages", 3, "eq", dim), ShapeGuard("V_pages", 0, "eq", physical_pages),
            ShapeGuard("V_pages", 1, "eq", page_size), ShapeGuard("V_pages", 2, "eq", heads),
            ShapeGuard("V_pages", 3, "eq", dim), ShapeGuard("page_table", 0, "eq", logical_pages),
            ShapeGuard("token_indices", 0, "eq", tokens), ShapeGuard("O", 0, "eq", heads),
            ShapeGuard("O", 1, "eq", query_length), ShapeGuard("O", 2, "eq", dim),
        ),
        geometry=LaunchGeometry(policy="sm120_paged_attention_fused_direct_128"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="inputs", synchronization=("completion",)
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2", "sync_key": "E2E-SPINE-2026-07-18",
            "route": "fused_direct", "shape": list(dims), "scale": scale,
            "causal": causal, "causal_offset": causal_offset,
            "table_storage": "i32", "token_index_storage": "i64",
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def _package_direct_image(
    tile_ir: str, *, entry: str, abi_id: str, pipeline_name: str,
) -> tuple[str, str, NativeImageArtifact]:
    """Compile a direct launch-level Tile kernel and retain its full fingerprint."""
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp, toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),), compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    return lowered, ptx, image


def package_replay_ssm_kernels(
    *, batch: int, channels: int, state_dim: int, capacity: int,
    async_slots: int, pipeline_name: str,
) -> tuple[NVIDIANativePackage, NVIDIANativePackage]:
    """Package deterministic decode/flush images for one persistent ReplaySSM ABI."""
    from .ssm_replay import replay_state_descriptor

    state = replay_state_descriptor(
        target="nvidia_sm120", batch=batch, channels=channels, state_dim=state_dim,
        capacity=capacity, async_slots=async_slots,
    )
    decode_entry = "tessera_tile_replay_ssm_decode_f32"
    flush_entry = "tessera_tile_replay_ssm_flush_f32"
    decode_ir, flush_ir = emit_replay_ssm_tile_ir(
        decode_entry=decode_entry, flush_entry=flush_entry
    )
    specs = (
        (decode_ir, decode_entry, SM120_REPLAY_DECODE_F32_ABI, "output_only"),
        (flush_ir, flush_entry, SM120_REPLAY_FLUSH_F32_ABI, "state_and_output"),
    )
    packages: list[NVIDIANativePackage] = []
    for tile_ir, entry, abi_id, route in specs:
        lowered, ptx, image = _package_direct_image(
            tile_ir, entry=entry, abi_id=abi_id, pipeline_name=pipeline_name
        )
        buffers: tuple[BufferBinding, ...]
        if route == "output_only":
            buffers = (
                BufferBinding(0, "delta", "input", "fp32", 3, "row_major", 4),
                BufferBinding(1, "x", "input", "fp32", 3, "row_major", 4),
                BufferBinding(2, "B", "input", "fp32", 3, "row_major", 4),
                BufferBinding(3, "S0", "input", "fp32", 3, "row_major", 4),
                BufferBinding(4, "C", "input", "fp32", 2, "row_major", 4),
                BufferBinding(5, "A", "input", "fp32", 1, "row_major", 4),
                BufferBinding(6, "Y", "output", "fp32", 3, "row_major", 4),
            )
        else:
            buffers = (
                BufferBinding(0, "delta", "input", "fp32", 3, "row_major", 4),
                BufferBinding(1, "x", "input", "fp32", 3, "row_major", 4),
                BufferBinding(2, "B", "input", "fp32", 3, "row_major", 4),
                BufferBinding(3, "S0", "inout", "fp32", 3, "row_major", 4),
                BufferBinding(4, "A", "input", "fp32", 1, "row_major", 4),
            )
        scalar_base = len(buffers)
        descriptor = LaunchDescriptor(
            image_digest=image.image_digest, entry_symbol=entry, abi_id=abi_id,
            buffers=buffers,
            scalars=tuple(ScalarArgument(scalar_base + i, name, "int64") for i, name in enumerate(
                ("Batch", "Channels", "State", "Tokens")
            )),
            geometry=LaunchGeometry(policy="sm120_replay_thread_per_batch_channel_128"),
            workspace=state.workspace,
            ordering=state.ordering,
            provenance={
                "work_item": "NVIDIA-E2E-2", "sync_key": "E2E-SPINE-2026-07-18",
                "route": route, "state_descriptor": state.as_metadata_dict(),
                "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
            },
        )
        packages.append(NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor))
    return packages[0], packages[1]


def package_moe_kernels(
    *, num_tokens: int, num_slots: int, hidden: int, expert_count: int,
    expert_k: int, expert_n: int, group_offsets: tuple[int, ...],
    pipeline_name: str, storage: str = "fp32",
) -> tuple[NVIDIANativePackage, NVIDIANativePackage, NVIDIANativePackage]:
    """Package local dispatch/combine/grouped-GEMM images over canonical metadata."""
    if min(num_tokens, hidden, expert_count, expert_k, expert_n) <= 0 or num_slots < 0:
        raise ValueError("canonical MoE package dimensions are invalid")
    if storage not in SM120_MOE_DTYPES:
        raise ValueError(f"canonical MoE storage must be one of {SM120_MOE_DTYPES}")
    if len(group_offsets) != expert_count + 1 or group_offsets[0] != 0:
        raise ValueError("canonical MoE group offsets must contain E+1 entries starting at zero")
    if any(a > b for a, b in zip(group_offsets, group_offsets[1:])) or group_offsets[-1] != num_slots:
        raise ValueError("canonical MoE group offsets must monotonically partition all kept slots")
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    alignment = 2 if storage != "fp32" else 4
    entries = (
        ("dispatch", f"tessera_tile_moe_dispatch_{storage_ir}",
         SM120_MOE_DISPATCH_F32_ABI if storage == "fp32" else
         f"tessera.nvidia.moe.dispatch.x_token_o_dims.{storage_ir}_i32.v2"),
        ("combine", f"tessera_tile_moe_combine_{storage_ir}",
         SM120_MOE_COMBINE_F32_ABI if storage == "fp32" else
         f"tessera.nvidia.moe.combine.partials_token_weight_o_dims.{storage_ir}_i32.v2"),
        ("grouped_gemm", f"tessera_tile_grouped_gemm_{storage_ir}",
         SM120_GROUPED_GEMM_F32_ABI if storage == "fp32" else
         f"tessera.nvidia.moe.grouped_gemm.x_w_offsets_o_dims.{storage_ir}_f32acc_i32.v2"),
    )
    packages: list[NVIDIANativePackage] = []
    for route, entry, abi_id in entries:
        tile_ir = emit_moe_tile_ir(entry=entry, route=route, storage=storage_ir)
        lowered, ptx, image = _package_direct_image(
            tile_ir, entry=entry, abi_id=abi_id, pipeline_name=pipeline_name
        )
        buffers: tuple[BufferBinding, ...]
        dims: tuple[str, ...]
        if route == "dispatch":
            buffers = (
                BufferBinding(0, "X", "input", storage, 2, "row_major", alignment),
                BufferBinding(1, "token_of_slot", "input", "int32", 1, "row_major", 4),
                BufferBinding(2, "dispatched", "output", storage, 2, "row_major", alignment),
            )
            dims = ("Tokens", "Slots", "Hidden")
        elif route == "combine":
            buffers = (
                BufferBinding(0, "partials", "input", storage, 2, "row_major", alignment),
                BufferBinding(1, "token_of_slot", "input", "int32", 1, "row_major", 4),
                BufferBinding(2, "combine_weights", "input", "fp32", 1, "row_major", 4),
                BufferBinding(3, "O", "output", storage, 2, "row_major", alignment),
            )
            dims = ("Tokens", "Slots", "Hidden")
        else:
            buffers = (
                BufferBinding(0, "X", "input", storage, 2, "row_major", alignment),
                BufferBinding(1, "W", "input", storage, 3, "row_major", alignment),
                BufferBinding(2, "group_offsets", "input", "int32", 1, "row_major", 4),
                BufferBinding(3, "O", "output", storage, 2, "row_major", alignment),
            )
            dims = ("GroupedTokens", "K", "N", "Experts")
        base = len(buffers)
        descriptor = LaunchDescriptor(
            image_digest=image.image_digest, entry_symbol=entry, abi_id=abi_id,
            buffers=buffers,
            scalars=tuple(ScalarArgument(base + i, name, "int64") for i, name in enumerate(dims)),
            geometry=LaunchGeometry(policy="sm120_moe_thread_per_output_256"),
            ordering=OrderingSemantics(
                ordered_submission=True, residency="inputs",
                synchronization=("dispatch_before_expert_compute", "expert_compute_before_combine", "completion"),
            ),
            provenance={
                "work_item": "NVIDIA-E2E-2", "sync_key": "E2E-SPINE-2026-07-18",
                "route": route, "num_tokens": num_tokens, "num_slots": num_slots,
                "hidden": hidden, "expert_count": expert_count, "expert_k": expert_k,
                "expert_n": expert_n, "group_offsets": list(group_offsets),
                "storage": storage_ir, "accum": "f32",
                "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
            },
        )
        packages.append(NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor))
    return packages[0], packages[1], packages[2]


def package_softmax(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    """Compile and package one static f16/f32 last-axis softmax request."""
    storage = _softmax_storage(module)
    if storage is None:
        raise ValueError("SM120 native softmax packaging requires one static f16/f32 last-axis softmax")
    storage_ir = "f16" if storage == "fp16" else "f32"
    entry = f"tessera_tile_softmax_{storage_ir}"
    abi_id = SM120_SOFTMAX_F16_ABI if storage == "fp16" else SM120_SOFTMAX_F32_ABI
    alignment = 2 if storage == "fp16" else 4
    tile_ir = emit_softmax_tile_ir(entry=entry, storage=storage_ir)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    fn = module.functions[0]
    op = fn.body[0]
    input_name = op.operands[0].removeprefix("%")
    output_name = op.result or "output"
    shape = _shape(module, input_name)
    assert shape is not None
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    guards = tuple(
        ShapeGuard(name, axis, "eq", extent) for name in (input_name, output_name) for axis, extent in enumerate(shape)
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", storage, len(shape), "row_major", alignment),
            BufferBinding(1, output_name, "output", storage, len(shape), "row_major", alignment),
        ),
        scalars=(
            ScalarArgument(2, "Rows", "int64"),
            ScalarArgument(3, "K", "int64"),
        ),
        shape_guards=guards,
        geometry=LaunchGeometry(policy="sm120_softmax_thread_per_row_128"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "thread_per_row_128",
            "shape": list(shape),
            "storage": storage_ir,
            "accum": "f32",
            "axis": -1,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def package_f32_softmax(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    if not supports_f32_softmax(module):
        raise ValueError("SM120 f32 softmax packaging requires f32 storage")
    return package_softmax(module, pipeline_name=pipeline_name)


def package_f16_softmax(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    if not supports_f16_softmax(module):
        raise ValueError("SM120 f16 softmax packaging requires f16 storage")
    return package_softmax(module, pipeline_name=pipeline_name)


__all__ = [
    "NVIDIANativePackage",
    "SM120_ATTN_F16_ABI",
    "SM120_ATTN_F32_ABI",
    "SM120_ATTN_BIAS_F16_ABI",
    "SM120_ATTN_BIAS_F32_ABI",
    "SM120_ATTN_BWD_F32_ABI",
    "SM120_ATTN_BWD_BIAS_F32_ABI",
    "SM120_ATTN_BWD_F16_ABI",
    "SM120_ATTN_BWD_BIAS_F16_ABI",
    "SM120_BF16_ABI",
    "SM120_EPILOGUE_ABIS",
    "SM120_F16_ABI",
    "SM120_FP8_E4M3_ABI",
    "SM120_FP8_E5M2_ABI",
    "SM120_NVFP4_ABI",
    "SM120_FP64_ABI",
    "SM120_FP6_E2M3_ABI",
    "SM120_FP6_E3M2_ABI",
    "SM120_MXFP4_ABI",
    "SM120_PAGED_KV_F32_ABI",
    "SM120_PAGED_ATTN_F32_ABI",
    "SM120_REPLAY_DECODE_F32_ABI",
    "SM120_REPLAY_FLUSH_F32_ABI",
    "SM120_MOE_DISPATCH_F32_ABI",
    "SM120_MOE_COMBINE_F32_ABI",
    "SM120_GROUPED_GEMM_F32_ABI",
    "SM120_MOE_ABIS",
    "SM120_REDUCE_F16_ABI",
    "SM120_REDUCE_F32_ABI",
    "SM120_INT8_ABI",
    "SM120_TF32_ABI",
    "SM120_SOFTMAX_F16_ABI",
    "SM120_SOFTMAX_F32_ABI",
    "emit_f16_matmul_tile_ir",
    "emit_attention_tile_ir",
    "emit_attention_backward_tile_ir",
    "emit_f16_softmax_tile_ir",
    "emit_matmul_tile_ir",
    "emit_mx_matmul_tile_ir",
    "emit_reduce_tile_ir",
    "emit_f32_softmax_tile_ir",
    "emit_nvfp4_matmul_tile_ir",
    "emit_paged_kv_read_tile_ir",
    "emit_paged_attention_tile_ir",
    "emit_replay_ssm_tile_ir",
    "emit_moe_tile_ir",
    "emit_softmax_tile_ir",
    "package_bf16_matmul",
    "package_attention",
    "package_attention_backward",
    "package_f16_matmul",
    "package_f16_softmax",
    "package_matmul",
    "package_mx_matmul",
    "package_reduction",
    "package_f32_softmax",
    "package_nvfp4_matmul",
    "package_paged_kv_read",
    "package_paged_attention",
    "package_replay_ssm_kernels",
    "package_moe_kernels",
    "package_softmax",
    "requests_softmax",
    "requests_attention",
    "requests_attention_backward",
    "requests_mx_matmul",
    "requests_reduction",
    "requests_paged_kv_read",
    "supports_bf16_matmul",
    "supports_attention",
    "supports_attention_backward",
    "supports_f16_matmul",
    "supports_f16_softmax",
    "supports_fp64_matmul",
    "supports_fp8_matmul",
    "supports_matmul",
    "supports_mx_matmul",
    "supports_reduction",
    "supports_paged_kv_read",
    "supports_int8_matmul",
    "supports_tf32_matmul",
    "supports_f32_softmax",
    "supports_nvfp4_matmul",
    "tools_available",
]
