"""Compiler-owned gfx1151 native packaging for ROCM-E2E-1/-2.

The pilot deliberately covers one semantic family: static f16/f32 last-axis
softmax.  Python describes the typed Tile launch envelope, while the registered
ROCm passes own the Tile-to-directive adaptation, kernel generation, ROCDL
lowering, and HSACO production.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .attention_contract import plan_attention_backward_workspace
from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding,
    DeviceLibraryRecord,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
    WorkspaceRequirement,
)


GFX1151_SOFTMAX_F16_ABI = "tessera.rocm.softmax.x_o_rows_k.f16.v1"
GFX1151_SOFTMAX_F32_ABI = "tessera.rocm.softmax.x_o_rows_k.f32.v1"
GFX1151_REDUCE_F32_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.f32.v1"
GFX1151_REDUCE_F16_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.f16_f32out.v1"
GFX1151_REDUCE_BF16_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.bf16_f32out.v1"
GFX1151_PAGED_KV_F32_ABI = "tessera.rocm.paged_kv.pages_table_o_dims.f32_i32.v1"
GFX1151_MOE_DISPATCH_F32_ABI = "tessera.rocm.moe_dispatch.x_token_o_t_s_h.f32_i32.v1"
GFX1151_ATTN_F16_ABI = "tessera.rocm.attention.q_k_v_o_dims.f16_f32out.v1"
GFX1151_ATTN_BF16_ABI = "tessera.rocm.attention.q_k_v_o_dims.bf16_f32out.v1"
GFX1151_ATTN_BWD_PRE_ABI = "tessera.rocm.attention_backward.pre.v1"
GFX1151_ATTN_BWD_DKDV_ABI = "tessera.rocm.attention_backward.dkdv_split.v1"
GFX1151_ATTN_BWD_REDUCE_ABI = "tessera.rocm.attention_backward.dkdv_reduce.v1"
GFX1151_ATTN_BWD_DQ_ABI = "tessera.rocm.attention_backward.dq.v1"


@dataclass(frozen=True)
class ROCMNativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


@dataclass(frozen=True)
class ROCMWorkspaceSlice:
    name: str
    offset: int
    bytes: int
    initialization: str

    def __post_init__(self) -> None:
        if not self.name or self.offset < 0 or self.bytes <= 0:
            raise ValueError("ROCm workspace slices require a name and positive extent")
        if self.initialization not in {"undefined", "zero"}:
            raise ValueError("ROCm workspace slice initialization must be undefined or zero")


@dataclass(frozen=True)
class ROCMNativeProgram:
    """One compiler-owned image plus an ordered multi-kernel launch plan."""

    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptors: tuple[LaunchDescriptor, ...]
    workspace: WorkspaceRequirement
    workspace_slices: tuple[ROCMWorkspaceSlice, ...]

    def __post_init__(self) -> None:
        symbols = {entry.symbol for entry in self.image.entry_points}
        planned = tuple(descriptor.entry_symbol for descriptor in self.descriptors)
        if not planned or any(symbol not in symbols for symbol in planned):
            raise ValueError("ROCm native program descriptors must name image entry points")
        if any(descriptor.image_digest != self.image.image_digest for descriptor in self.descriptors):
            raise ValueError("ROCm native program descriptor/image identity mismatch")
        ordered = sorted(self.workspace_slices, key=lambda item: item.offset)
        if tuple(ordered) != self.workspace_slices:
            raise ValueError("ROCm native program workspace slices must be offset ordered")
        end = 0
        for item in ordered:
            if item.offset < end:
                raise ValueError("ROCm native program workspace slices overlap")
            end = item.offset + item.bytes
        if end > self.workspace.bytes:
            raise ValueError("ROCm native program workspace slices exceed allocation")


_cache: dict[
    str,
    tuple[str, str, bytes, str, str, tuple[DeviceLibraryRecord, ...]],
] = {}

_BUILTIN_BITCODE_RE = re.compile(r'"-mlink-builtin-bitcode"\s+"([^"]+\.bc)"')


def _mlir_float(value: float) -> str:
    literal = f"{value:.17g}"
    return literal if any(char in literal for char in ".eE") else literal + ".0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tessera_opt() -> Path | None:
    configured = os.environ.get("TESSERA_OPT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = _repo_root()
    for path in (
        root / "build/tools/tessera-opt/tessera-opt",
        root / "build/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt",
    ):
        if path.is_file():
            return path
    found = shutil.which("tessera-opt")
    return Path(found) if found else None


def tools_available() -> bool:
    return _tessera_opt() is not None


def native_packaging_available() -> bool:
    """Whether a ROCm native package can actually be built on this host.

    `tools_available` answers only "is there a tessera-opt", but packaging also
    needs AMD clang to fingerprint the OCML/OCKL/OCLC bitcode the driver
    selects. A caller that checks the first and not the second gets a
    `RuntimeError` out of `_driver_selected_device_libraries` — correct, but it
    reads as a broken test on any host without ROCm rather than an absent
    toolchain."""
    return tools_available() and _rocm_clang(_rocm_path()) is not None


def native_package_kind(module: GraphIRModule) -> str | None:
    """Return the canonical single-descriptor gfx1151 family for ``module``."""

    if supports_softmax(module):
        return "softmax"
    if supports_reduction(module):
        return "reduction"
    if supports_paged_kv_read(module):
        return "paged_kv"
    if supports_attention(module):
        return "attention"
    if supports_moe_dispatch(module):
        return "moe_dispatch"
    return None


def supports_native_package(module: GraphIRModule) -> bool:
    """Whether the canonical single-launch gfx1151 package accepts ``module``."""

    return native_package_kind(module) is not None


def package_native(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> ROCMNativePackage:
    """Compile the one canonical gfx1151 descriptor selected for ``module``."""

    kind = native_package_kind(module)
    if kind == "softmax":
        return package_softmax(module, pipeline_name=pipeline_name)
    if kind == "reduction":
        return package_reduction(module, pipeline_name=pipeline_name)
    if kind == "paged_kv":
        return package_paged_kv_read(module, pipeline_name=pipeline_name)
    if kind == "attention":
        return package_attention(module, pipeline_name=pipeline_name)
    if kind == "moe_dispatch":
        return package_moe_dispatch(module, pipeline_name=pipeline_name)
    raise ValueError(
        "gfx1151 native packaging requires one supported static Graph contract"
    )


def _rocm_path() -> Path:
    configured = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).expanduser()
    for candidate in (configured, configured / "core"):
        if (candidate / "amdgcn/bitcode").is_dir():
            return candidate
    return configured


def _rocm_clang(rocm_path: Path) -> Path | None:
    configured = os.environ.get("TESSERA_ROCM_CLANG")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    for path in (
        rocm_path / "bin/amdclang++",
        rocm_path / "llvm/bin/clang++",
        Path("/opt/rocm/core/bin/amdclang++"),
    ):
        if path.is_file():
            return path
    found = shutil.which("amdclang++")
    return Path(found) if found else None


def _driver_selected_device_libraries(*, arch: str = "gfx1151") -> tuple[DeviceLibraryRecord, ...]:
    """Fingerprint the exact builtin bitcode set selected by AMD clang.

    The driver owns selection because its OCLC control libraries encode the
    target ISA, ABI, wavefront, finite-only, and unsafe-math policy. Paths are
    used only for discovery and never persisted in the native-image contract.
    """
    rocm_path = _rocm_path()
    clang = _rocm_clang(rocm_path)
    if clang is None:
        raise RuntimeError(
            "ROCm native packaging requires AMD clang to identify OCML/OCKL/OCLC; set TESSERA_ROCM_CLANG or ROCM_PATH"
        )
    result = subprocess.run(
        [
            str(clang),
            "-###",
            "-x",
            "hip",
            "--offload-device-only",
            f"--offload-arch={arch}",
            f"--rocm-path={rocm_path}",
            "-c",
            "-",
        ],
        input="",
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = "\n".join((result.stdout, result.stderr))
    selected = tuple(Path(value) for value in _BUILTIN_BITCODE_RE.findall(transcript))
    if result.returncode or not selected:
        detail = result.stderr.strip() or f"AMD clang exited {result.returncode}"
        raise RuntimeError(f"ROCm device-library discovery failed: {detail}")
    missing = tuple(path for path in selected if not path.is_file())
    if missing:
        raise RuntimeError("ROCm driver selected missing device libraries: " + ", ".join(path.name for path in missing))
    stems = {path.stem for path in selected}
    if not {"ocml", "ockl"}.issubset(stems) or not any(stem.startswith("oclc_") for stem in stems):
        raise RuntimeError("ROCm driver selection omitted required OCML/OCKL/OCLC libraries")
    return tuple(
        DeviceLibraryRecord(
            logical_name=f"rocm.{path.stem}",
            content_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            link_mode="compiler_driver",
        )
        for path in selected
    )


def _version_fingerprint(tool: Path) -> str:
    result = subprocess.run([str(tool), "--version"], capture_output=True, text=True, check=False)
    text = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return hashlib.sha256((text or str(tool)).encode()).hexdigest()


def _run_opt(tool: Path, source: str, pipeline: str) -> str:
    result = subprocess.run(
        [str(tool), "-", f"--pass-pipeline={pipeline}"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "ROCm native packaging failed: " + (result.stderr.strip() or f"tessera-opt exited {result.returncode}")
        )
    return result.stdout


def _extract_hsaco(text: str) -> bytes:
    marker = 'bin = "'
    if marker not in text:
        raise RuntimeError("ROCm native packaging produced no gpu.object binary")
    index = text.index(marker) + len(marker)
    output = bytearray()
    hexdigits = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while index < len(text) and text[index] != '"':
        char = text[index]
        if char == "\\":
            escaped = text[index + 1 : index + 3]
            if len(escaped) == 2 and all(value in hexdigits for value in escaped):
                output.append(int(escaped, 16))
                index += 3
                continue
            next_char = text[index + 1]
            if next_char in simple:
                output.append(simple[next_char])
                index += 2
                continue
        output.append(ord(char))
        index += 1
    payload = bytes(output)
    if not payload.startswith(b"\x7fELF"):
        raise RuntimeError("ROCm native packaging output is not an ELF HSACO")
    return payload


def emit_softmax_tile_ir(*, entry: str, storage: str) -> str:
    """Emit the shared semantic softmax envelope with ROCm-owned math intent."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported gfx1151 softmax storage {storage!r}")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %rows: i64, %columns: i64) {{
    tile.softmax_kernel %x, %o, %rows, %columns {{
      storage = "{storage}", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''


def emit_reduce_tile_ir(
    *, entry: str, storage: str, kind: str, axis: int, keepdims: bool, inner_is_one: bool = False
) -> str:
    """Emit the shared arbitrary-axis mixed-precision reduction envelope."""
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported gfx1151 reduction storage {storage!r}")
    if kind not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported gfx1151 reduction kind {kind!r}")
    if axis < 0:
        raise ValueError("gfx1151 reduction requires a normalized axis")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %outer: i64, %axis_extent: i64, %inner: i64) {{
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {{
      storage = "{storage}", accum = "f32", kind = "{kind}",
      axis = {axis} : i64, keepdims = {str(keepdims).lower()},
      schedule = "serial", nan_mode = "propagate",
      inner_is_one = {str(inner_is_one).lower()}
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_paged_kv_read_tile_ir(*, entry: str) -> str:
    """Emit the shared direct f32 paged-KV gather envelope."""
    return f"""module {{
  llvm.func @{entry}(%pages: !llvm.ptr, %table: !llvm.ptr, %o: !llvm.ptr,
                     %p: i64, %lp: i64, %ps: i64, %h: i64, %d: i64,
                     %start: i64, %tokens: i64) {{
    tile.paged_kv_read_kernel %pages, %table, %o, %p, %lp, %ps, %h, %d, %start, %tokens {{
      storage = "f32", table_storage = "i32", route = "direct"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
"""


def emit_moe_dispatch_tile_ir(*, entry: str) -> str:
    """Emit the shared direct f32/i32 MoE token-gather envelope."""
    return f"""module {{
  llvm.func @{entry}(%x: !llvm.ptr, %token: !llvm.ptr, %o: !llvm.ptr,
                     %t: i64, %s: i64, %h: i64) {{
    tile.moe_dispatch_kernel %x, %token, %o, %t, %s, %h {{
      storage = "f32", index_storage = "i32"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
"""


def emit_attention_tile_ir(
    *,
    entry: str,
    storage: str,
    dims: tuple[int, int, int, int, int, int, int],
    scale: float,
    causal: bool,
    bias: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> str:
    """Emit the canonical attention carrier plus gfx1151 schedule buckets.

    B/H/S remain runtime launch operands. ``head_dim`` and ``value_dim`` are
    compile/cache buckets used only to select the physical WMMA descriptor.
    """
    if storage not in {"f16", "bf16"}:
        raise ValueError(f"unsupported gfx1151 attention storage {storage!r}")
    _, hq, hkv, _, _, head_dim, value_dim = dims
    optional_arg = ", %bias: !llvm.ptr" if bias else ""
    optional_operand = ", %bias" if bias else ""
    scale_literal = _mlir_float(scale)
    softcap_literal = _mlir_float(softcap)
    dropout_literal = _mlir_float(dropout_p)
    return f'''module {{
  llvm.func @{entry}(%q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr{optional_arg},
                     %o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64,
                     %sq: i64, %sk: i64, %d: i64, %dv: i64) {{
    tile.attention_kernel %q, %key, %v{optional_operand}, %o, %b, %hq, %hkv,
        %sq, %sk, %d, %dv {{
      storage = "{storage}", accum = "f32", scale = {scale_literal} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {softcap_literal} : f32,
      dropout_p = {dropout_literal} : f32,
      dropout_seed = {dropout_seed} : i64, head_dim = {head_dim} : i64,
      value_dim = {value_dim} : i64, gqa = {str(hq != hkv).lower()}
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr{", !llvm.ptr" if bias else ""},
        !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_attention_graph_ir(
    *,
    entry: str,
    storage: str,
    dims: tuple[int, int, int, int, int, int, int],
    scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    bias: bool = False,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
    tile_kv: int = 16,
) -> str:
    """Emit canonical rank-4 attention for direct shared-to-target lowering.

    This is the semantic entry path: the shared Tile IR lowering owns
    batch/head distribution, GQA mapping, the KV-block recurrence, ragged
    zero-fill, and pipeline SSA. ROCm lowering owns only its physical schedule.
    Additive bias and softcap are direct shared score-recurrence operations;
    target lowering observes them but does not reconstruct their ordering.
    """
    if storage not in {"f16", "bf16"}:
        raise ValueError(f"unsupported gfx1151 attention storage {storage!r}")
    b, hq, hkv, sq, sk, head_dim, value_dim = dims
    if tile_kv <= 0:
        raise ValueError("canonical attention tile_kv must be positive")
    scale_literal = _mlir_float(scale)
    softcap_literal = _mlir_float(softcap)
    dropout_literal = _mlir_float(dropout_p)
    bias_arg = f",\n      %bias: tensor<{b}x{hq}x{sq}x{sk}xf32>" if bias else ""
    bias_operand = ", %bias" if bias else ""
    bias_type = f",\n          tensor<{b}x{hq}x{sq}x{sk}xf32>" if bias else ""
    return f"""module attributes {{
  tessera.ir.version = "1.0",
  tessera.target = {{sm = 90 : i32, warps = 1 : i32,
                    smem = 65536 : i64, pipeline_stages = 2 : i32}}
}} {{
  func.func @{entry}(
      %q: tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
      %key: tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
      %v: tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>{bias_arg}
  ) -> tensor<{b}x{hq}x{sq}x{value_dim}xf32> {{
    %o = "tessera.flash_attn"(%q, %key, %v{bias_operand})
        <{{operandSegmentSizes = array<i32: 1, 1, 1, {int(bias)}>}}> {{
      causal = {str(causal).lower()},
      dropout_p = {dropout_literal} : f64,
      dropout_seed = {dropout_seed} : i64,
      head_dim = {head_dim} : i64,
      scale = {scale_literal} : f32,
      softcap = {softcap_literal} : f32,
      tessera.tile_q = {sq} : i32,
      tessera.tile_kv = {tile_kv} : i32,
      window_left = {window_left} : i64,
      window_right = {window_right} : i64
    }} : (tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>{bias_type})
          -> tensor<{b}x{hq}x{sq}x{value_dim}xf32>
    return %o : tensor<{b}x{hq}x{sq}x{value_dim}xf32>
  }}
}}
"""


def emit_attention_backward_tile_ir(
    *,
    forward_entry: str,
    backward_entry: str,
    storage: str,
    dims: tuple[int, int, int, int, int, int, int],
    scale: float,
    causal: bool,
    bias: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> str:
    """Emit a self-contained forward-recompute plus deterministic VJP program.

    The portable backward carrier records the same launch-owned workspace,
    split count, block-loop structure, and reduction order consumed by the
    multi-entry :class:`ROCMNativeProgram`.
    """
    if storage not in {"f16", "bf16"}:
        raise ValueError(f"unsupported gfx1151 attention backward storage {storage!r}")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("gfx1151 attention backward dropout must satisfy 0 <= p < 1")
    _, hq, hkv, _, _, head_dim, value_dim = dims
    if head_dim != value_dim:
        raise ValueError("gfx1151 optimized attention backward requires D == Dv")
    optional_arg = ", %bias: !llvm.ptr" if bias else ""
    optional_operand = ", %bias" if bias else ""
    scale_literal = _mlir_float(scale)
    softcap_literal = _mlir_float(softcap)
    dropout_literal = _mlir_float(dropout_p)
    workspace = plan_attention_backward_workspace(
        batch=dims[0],
        query_heads=dims[1],
        kv_heads=dims[2],
        query_rows=dims[3],
        key_rows=dims[4],
        head_dim=dims[5],
        value_dim=dims[6],
        split_count=2,
    )
    common_attrs = f'''
      storage = "{storage}", accum = "f32", scale = {scale_literal} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {softcap_literal} : f32,
      dropout_p = {dropout_literal} : f32,
      dropout_seed = {dropout_seed} : i64, head_dim = {head_dim} : i64,
      value_dim = {value_dim} : i64, gqa = {str(hq != hkv).lower()}'''
    return f"""module {{
  llvm.func @{forward_entry}(%q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr{optional_arg},
                     %o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64,
                     %sq: i64, %sk: i64, %d: i64, %dv: i64) {{
    tile.attention_kernel %q, %key, %v{optional_operand}, %o, %b, %hq, %hkv,
        %sq, %sk, %d, %dv {{{common_attrs}
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr{", !llvm.ptr" if bias else ""},
        !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
  llvm.func @{backward_entry}(%do: !llvm.ptr, %q: !llvm.ptr, %key: !llvm.ptr,
                     %v: !llvm.ptr{optional_arg}, %dq: !llvm.ptr,
                     %dk: !llvm.ptr, %dv_out: !llvm.ptr, %b: i64, %hq: i64,
                     %hkv: i64, %sq: i64, %sk: i64, %d: i64, %dv: i64) {{
    tile.attention_backward_kernel %do, %q, %key, %v{optional_operand},
        %dq, %dk, %dv_out, %b, %hq, %hkv, %sq, %sk, %d, %dv {{{common_attrs},
      route = "deterministic_split_reduced", deterministic = true,
      workspace_bytes = {workspace.bytes} : i64,
      workspace_owner = "program_launch", split_count = 2 : i64,
      reduction_order = array<i64: 0, 1>, query_block = 16 : i64,
      key_block = 16 : i64,
      loop_order = ["batch_kv_head", "split", "query_block",
                    "key_block", "fixed_order_reduce"]
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr{", !llvm.ptr" if bias else ""},
        !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
"""


def emit_attention_backward_graph_ir(
    *,
    forward_entry: str,
    backward_entry: str,
    storage: str,
    dims: tuple[int, int, int, int, int, int, int],
    scale: float,
    causal: bool,
    bias: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
    query_block: int = 16,
    key_block: int = 16,
    split_count: int = 2,
    save_lse: bool = True,
) -> str:
    """Emit the shared tensor-valued forward-recompute and backward program.

    This is the canonical target-neutral source for optimized backward
    packaging. Target lowering consumes the resulting ``scf.for`` phase bodies
    and owns only its physical schedule and launch ABI. The launch-level Tile
    emitter above remains a compatibility/reference seam.
    """
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported attention backward storage {storage!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("attention backward scale must be finite and positive")
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("attention backward softcap must be finite and nonnegative")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("attention backward dropout must satisfy 0 <= p < 1")
    if dropout_seed < 0:
        raise ValueError("attention backward dropout seed must be nonnegative")
    if query_block <= 0 or key_block <= 0 or split_count < 2:
        raise ValueError("attention backward needs positive blocks and at least two splits")
    b, hq, hkv, sq, sk, head_dim, value_dim = dims
    if head_dim != value_dim:
        raise ValueError("optimized attention backward requires D == Dv")

    scale_literal = _mlir_float(scale)
    softcap_literal = _mlir_float(softcap)
    dropout_literal = _mlir_float(dropout_p)
    bias_argument = f",\n      %bias: tensor<{b}x{hq}x{sq}x{sk}xf32>" if bias else ""
    forward_bias_operand = ", %bias" if bias else ""
    forward_bias_segment = 1 if bias else 0
    forward_bias_type = f", tensor<{b}x{hq}x{sq}x{sk}xf32>" if bias else ""
    if bias:
        backward_bias_value = "%bias"
        backward_bias_setup = ""
        backward_bias_type = f"tensor<{b}x{hq}x{sq}x{sk}xf32>"
    else:
        backward_bias_value = "%zero_bias"
        backward_bias_type = f"tensor<1x{sq}x{sk}xf32>"
        backward_bias_setup = f"    %zero_bias = arith.constant dense<0.000000e+00> : {backward_bias_type}\n"

    return f'''module {{
  func.func @{forward_entry}(
      %q: tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
      %key: tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
      %v: tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>{bias_argument}
  ) -> tensor<{b}x{hq}x{sq}x{value_dim}xf32>
      attributes {{tessera.lse_checkpoint = "{"saved" if save_lse else "recompute"}"}} {{
    %o = "tessera.flash_attn"(%q, %key, %v{forward_bias_operand})
        <{{operandSegmentSizes = array<i32: 1, 1, 1, {forward_bias_segment}>}}> {{
      causal = {str(causal).lower()},
      dropout_p = {dropout_literal} : f64,
      dropout_seed = {dropout_seed} : i64,
      head_dim = {head_dim} : i64,
      scale = {scale_literal} : f32,
      softcap = {softcap_literal} : f32,
      tessera.tile_q = {sq} : i32,
      tessera.tile_kv = {key_block} : i32,
      window_left = {window_left} : i64,
      window_right = {window_right} : i64
    }} : (tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>{forward_bias_type})
          -> tensor<{b}x{hq}x{sq}x{value_dim}xf32>
    return %o : tensor<{b}x{hq}x{sq}x{value_dim}xf32>
  }}

  func.func @{backward_entry}(
      %do: tensor<{b}x{hq}x{sq}x{value_dim}x{storage}>,
      %q: tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
      %key: tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
      %v: tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>{bias_argument}
  ) -> (tensor<{b}x{hq}x{sq}x{head_dim}xf32>,
        tensor<{b}x{hkv}x{sk}x{head_dim}xf32>,
        tensor<{b}x{hkv}x{sk}x{value_dim}xf32>)
      attributes {{tessera.lse_checkpoint = "{"saved" if save_lse else "recompute"}"}} {{
{backward_bias_setup}    %dq, %dk, %dv_out = "tessera_attn.backward"(
        %do, %q, %key, %v, {backward_bias_value}) {{
      causal = {str(causal).lower()},
      dropout_p = {dropout_literal} : f32,
      dropout_seed = {dropout_seed} : i64,
      key_block = {key_block} : i64,
      query_block = {query_block} : i64,
      scale = {scale_literal} : f32,
      softcap = {softcap_literal} : f32,
      split_count = {split_count} : i64,
      window_left = {window_left} : i64,
      window_right = {window_right} : i64
    }} : (tensor<{b}x{hq}x{sq}x{value_dim}x{storage}>,
          tensor<{b}x{hq}x{sq}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{head_dim}x{storage}>,
          tensor<{b}x{hkv}x{sk}x{value_dim}x{storage}>,
          {backward_bias_type})
          -> (tensor<{b}x{hq}x{sq}x{head_dim}xf32>,
              tensor<{b}x{hkv}x{sk}x{head_dim}xf32>,
              tensor<{b}x{hkv}x{sk}x{value_dim}xf32>)
    return %dq, %dk, %dv_out : tensor<{b}x{hq}x{sq}x{head_dim}xf32>,
                                tensor<{b}x{hkv}x{sk}x{head_dim}xf32>,
                                tensor<{b}x{hkv}x{sk}x{value_dim}xf32>
  }}
}}
'''


def requests_softmax(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name in {"tessera.softmax", "tessera.softmax_safe"}
    )


def requests_reduction(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name
        in {"tessera.reduce", "tessera.sum", "tessera.mean", "tessera.max", "tessera.amax"}
    )


def requests_paged_kv_read(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.kv_cache.read"
    )


def requests_moe_dispatch(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.moe_dispatch"
    )


def requests_attention(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.flash_attn"
    )


def requests_attention_backward(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1
        and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name in {"tessera.flash_attn_bwd", "tessera.flash_attn_vjp"}
    )


def _shape(module: GraphIRModule, name: str) -> tuple[int, ...] | None:
    arg = next((item for item in module.functions[0].args if item.name == name), None)
    if arg is None or arg.ir_type.rank is None or arg.ir_type.rank < 1:
        return None
    try:
        shape = tuple(int(dim) for dim in arg.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if all(dim > 0 for dim in shape) else None


def _softmax_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, ...]] | None:
    if not requests_softmax(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 1 or op.kwargs.get("axis", -1) != -1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((item for item in function.args if item.name == input_name), None)
    shape = _shape(module, input_name)
    if (
        arg is None
        or shape is None
        or arg.ir_type.dtype not in {"fp16", "fp32"}
        or not function.result_types
        or function.result_types[0].dtype != arg.ir_type.dtype
    ):
        return None
    return input_name, op.result or "output", arg.ir_type.dtype, shape


def supports_softmax(module: GraphIRModule) -> bool:
    return _softmax_contract(module) is not None


def _reduction_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, str, tuple[int, ...], tuple[int, ...], int, bool] | None:
    if not requests_reduction(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 1 or len(function.result_types) != 1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((item for item in function.args if item.name == input_name), None)
    shape = _shape(module, input_name)
    if arg is None or arg.ir_type.dtype not in {"fp16", "bf16", "fp32"} or shape is None:
        return None
    raw_axis = op.kwargs.get("axis", -1)
    if not isinstance(raw_axis, int) or isinstance(raw_axis, bool):
        return None
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(shape):
        return None
    keepdims = bool(op.kwargs.get("keepdims", False))
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
    result = function.result_types[0]
    try:
        declared_output_shape = tuple(int(dim) for dim in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or declared_output_shape != output_shape:
        return None
    kind = "max" if op.op_name in {"tessera.max", "tessera.amax"} else "mean" if op.op_name == "tessera.mean" else "sum"
    return (
        input_name,
        op.result or "output",
        arg.ir_type.dtype,
        kind,
        shape,
        output_shape,
        axis,
        keepdims,
    )


def supports_reduction(module: GraphIRModule) -> bool:
    return _reduction_contract(module) is not None


def _paged_kv_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, int, int, int, int, int, int]] | None:
    if not requests_paged_kv_read(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    pages_name, table_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    pages = args.get(pages_name)
    table = args.get(table_name)
    pages_shape = _shape(module, pages_name)
    table_shape = _shape(module, table_name)
    if (
        pages is None
        or table is None
        or pages.ir_type.dtype != "fp32"
        or table.ir_type.dtype != "int32"
        or pages_shape is None
        or len(pages_shape) != 4
        or table_shape is None
        or len(table_shape) != 1
    ):
        return None
    physical_pages, page_size, heads, dim = pages_shape
    logical_pages = table_shape[0]
    start = op.kwargs.get("start")
    end = op.kwargs.get("end")
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        return None
    tokens = end - start
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if (
        start < 0
        or tokens <= 0
        or end > logical_pages * page_size
        or result.dtype != "fp32"
        or result_shape != (tokens, heads, dim)
    ):
        return None
    return (
        pages_name,
        table_name,
        op.result or "output",
        (physical_pages, logical_pages, page_size, heads, dim, start, tokens),
    )


def supports_paged_kv_read(module: GraphIRModule) -> bool:
    return _paged_kv_contract(module) is not None


def _moe_dispatch_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, int, int]] | None:
    if not requests_moe_dispatch(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    x_name, token_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    x, token = args.get(x_name), args.get(token_name)
    x_shape, token_shape = _shape(module, x_name), _shape(module, token_name)
    if (
        x is None
        or token is None
        or x.ir_type.dtype != "fp32"
        or token.ir_type.dtype != "int32"
        or x_shape is None
        or len(x_shape) != 2
        or token_shape is None
        or len(token_shape) != 1
    ):
        return None
    tokens, hidden = x_shape
    slots = token_shape[0]
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or result_shape != (slots, hidden):
        return None
    return x_name, token_name, op.result or "output", (tokens, slots, hidden)


def supports_moe_dispatch(module: GraphIRModule) -> bool:
    return _moe_dispatch_contract(module) is not None


def _attention_contract(
    module: GraphIRModule,
) -> (
    tuple[
        tuple[str, str, str],
        str | None,
        str,
        tuple[int, int, int, int, int, int, int],
        float,
        bool,
        int,
        int,
        float,
        float,
        int,
    ]
    | None
):
    if not requests_attention(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) not in {3, 4} or len(function.result_types) != 1:
        return None
    q_name, k_name, v_name = (value.removeprefix("%") for value in op.operands[:3])
    args = {arg.name: arg for arg in function.args}
    if any(name not in args for name in (q_name, k_name, v_name)):
        return None
    dtypes = {args[name].ir_type.dtype for name in (q_name, k_name, v_name)}
    if len(dtypes) != 1 or not dtypes <= {"fp16", "bf16"}:
        return None
    # The set-membership guard above excludes the optional/unknown dtype
    # state carried by Graph IR. Preserve that proof for static consumers.
    dtype = cast(str, dtypes.pop())
    q_shape = _shape(module, q_name)
    k_shape = _shape(module, k_name)
    v_shape = _shape(module, v_name)
    if any(shape is None or len(shape) != 4 for shape in (q_shape, k_shape, v_shape)):
        return None
    assert q_shape is not None and k_shape is not None and v_shape is not None
    b, hq, sq, d = q_shape
    bk, hkv, sk, dk = k_shape
    bv, hv, sv, dv = v_shape
    if b != bk or b != bv or hkv != hv or sk != sv or d != dk or hq % hkv or d != dv or d <= 0 or d % 16:
        return None
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or result_shape != (b, hq, sq, dv):
        return None
    bias_name = op.operands[3].removeprefix("%") if len(op.operands) == 4 else None
    if bias_name is not None:
        bias_arg = args.get(bias_name)
        if bias_arg is None or bias_arg.ir_type.dtype != "fp32" or _shape(module, bias_name) != (b, hq, sq, sk):
            return None
    window = op.kwargs.get("window")
    if window is None:
        window_left = int(op.kwargs.get("window_left", -1))
        window_right = int(op.kwargs.get("window_right", -1))
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window_left, window_right = (int(value) for value in window)
    else:
        window_left = window_right = int(window)
    causal = bool(op.kwargs.get("causal", False))
    if not ((window_left == -1 and window_right == -1) or (causal and window_left >= 0 and window_right == 0)):
        return None
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    dropout_seed = int(op.kwargs.get("dropout_seed", op.kwargs.get("seed", 0)) or 0)
    scale = float(op.kwargs.get("scale", 1.0 / math.sqrt(float(d))))
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(softcap)
        or softcap < 0.0
        or not math.isfinite(dropout)
        or not 0.0 <= dropout < 1.0
    ):
        return None
    return (
        (q_name, k_name, v_name),
        bias_name,
        dtype,
        (b, hq, hkv, sq, sk, d, dv),
        scale,
        causal,
        window_left,
        window_right,
        softcap,
        dropout,
        dropout_seed,
    )


def supports_attention(module: GraphIRModule) -> bool:
    return _attention_contract(module) is not None


def _attention_backward_contract(
    module: GraphIRModule,
) -> (
    tuple[
        str,
        tuple[str, str, str, str],
        str | None,
        tuple[str, str, str],
        tuple[int, int, int, int, int, int, int],
        float,
        bool,
        int,
        int,
        float,
        float,
        int,
    ]
    | None
):
    if not requests_attention_backward(module):
        return None
    fn = module.functions[0]
    op = fn.body[0]
    if len(op.operands) not in {4, 5} or len(fn.result_types) != 3:
        return None
    names = tuple(value.removeprefix("%") for value in op.operands[:4])
    if len(names) != 4:
        return None
    do_name, q_name, k_name, v_name = names
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in names):
        return None
    storages = {args[name].ir_type.dtype for name in names}
    if len(storages) != 1 or not storages <= {"fp16", "bf16"}:
        return None
    storage = cast(str, storages.pop())
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
        b != bk
        or b != bv
        or hkv != hv
        or sk != sv
        or d != dk
        or d != dv
        or d <= 0
        or d % 16
        or hq % hkv
        or do_shape != (b, hq, sq, dv)
    ):
        return None
    expected_results = (q_shape, k_shape, v_shape)
    for result, expected in zip(fn.result_types, expected_results, strict=True):
        try:
            result_shape = tuple(int(dim) for dim in result.shape)
        except (TypeError, ValueError):
            return None
        if result.dtype != "fp32" or result_shape != expected:
            return None
    result_names = tuple(op.result_names)
    if len(result_names) != 3 or any(not name for name in result_names):
        return None
    bias_name = op.operands[4].removeprefix("%") if len(op.operands) == 5 else None
    if bias_name is not None:
        bias_arg = args.get(bias_name)
        if bias_arg is None or bias_arg.ir_type.dtype != "fp32" or _shape(module, bias_name) != (b, hq, sq, sk):
            return None
    window = op.kwargs.get("window")
    if window is None:
        window_left = int(op.kwargs.get("window_left", -1))
        window_right = int(op.kwargs.get("window_right", -1))
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window_left, window_right = (int(value) for value in window)
    else:
        window_left = window_right = int(window)
    causal = bool(op.kwargs.get("causal", False))
    if not ((window_left == -1 and window_right == -1) or (causal and window_left >= 0 and window_right == 0)):
        return None
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    scale = float(op.kwargs.get("scale", 1.0 / math.sqrt(float(d))))
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    dropout_seed = int(op.kwargs.get("dropout_seed", op.kwargs.get("seed", 0)) or 0)
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(softcap)
        or softcap < 0.0
        or not math.isfinite(dropout)
        or dropout < 0.0
        or dropout >= 1.0
        or str(op.kwargs.get("route", "deterministic_direct")) != "deterministic_direct"
        or not bool(op.kwargs.get("deterministic", True))
    ):
        return None
    return (
        storage,
        (do_name, q_name, k_name, v_name),
        bias_name,
        result_names,
        (b, hq, hkv, sq, sk, d, dv),
        scale,
        causal,
        window_left,
        window_right,
        softcap,
        dropout,
        dropout_seed,
    )


def supports_attention_backward(module: GraphIRModule) -> bool:
    return _attention_backward_contract(module) is not None


def _compile_native_tile_ir(
    tile_ir: str,
    *,
    directive: str,
    generator: str,
    semantic_pipeline: str = "",
) -> tuple[
    str,
    str,
    bytes,
    str,
    str,
    tuple[DeviceLibraryRecord, ...],
    str,
]:
    tool = _tessera_opt()
    if tool is None:
        raise RuntimeError("tessera-opt is required for ROCm native packaging")
    device_libraries = _driver_selected_device_libraries()
    library_identity = "|".join(
        f"{item.logical_name}:{item.content_digest}:{item.link_mode}" for item in device_libraries
    )
    key = hashlib.sha256(
        (f"{tile_ir}|{directive}|{generator}|{semantic_pipeline}|{library_identity}").encode()
    ).hexdigest()
    cached = _cache.get(key)
    if cached is not None:
        target_ir, backend_ir, payload, compiler_fp, toolchain_fp, libraries = cached
        return (
            target_ir,
            backend_ir,
            payload,
            compiler_fp,
            toolchain_fp,
            libraries,
            "warm_cache",
        )

    semantic_prefix = f"{semantic_pipeline}," if semantic_pipeline else ""
    target_pipeline = (
        "builtin.module("
        f"{semantic_prefix}rocm-wave-lds-pipeline,rocm-wave-lds-legality,"
        "lower-tile-to-rocm{arch=gfx1151})"
    )
    native_pipeline = (
        f"builtin.module({semantic_prefix}"
        "rocm-wave-lds-pipeline,rocm-wave-lds-legality,"
        f"lower-tile-to-rocm{{arch=gfx1151}},{generator},"
        "lower-tessera-target-to-rocdl,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts,rocm-materialize-dynamic-lds),"
        "rocdl-attach-target{chip=gfx1151},"
        "gpu-module-to-binary)"
    )
    target_ir = _run_opt(tool, tile_ir, target_pipeline)
    if directive not in target_ir:
        raise RuntimeError(f"ROCm native packaging did not produce typed {directive} Target IR")
    backend_ir = _run_opt(tool, tile_ir, native_pipeline)
    payload = _extract_hsaco(backend_ir)
    compiler_fp = _version_fingerprint(tool)
    clang = _rocm_clang(_rocm_path())
    driver_fp = _version_fingerprint(clang) if clang is not None else "missing"
    toolchain_fp = hashlib.sha256(f"{compiler_fp}|{driver_fp}|gfx1151|{library_identity}".encode()).hexdigest()
    _cache[key] = (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
    )
    return (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        "cold",
    )


def _compile_tile_ir(tile_ir: str):
    """Compatibility wrapper retained for the ROCM-E2E-1 test seam."""
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.softmax",
        generator="generate-rocm-softmax-kernel",
    )


def _compile_reduction_tile_ir(tile_ir: str):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.reduce",
        generator="generate-rocm-reduce-kernel",
    )


def _compile_paged_kv_tile_ir(tile_ir: str):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.paged_kv_read",
        generator="generate-rocm-paged-kv-read-kernel",
    )


def _compile_moe_dispatch_tile_ir(tile_ir: str):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.moe_dispatch",
        generator="generate-rocm-moe-kernel",
    )


def _compile_attention_tile_ir(tile_ir: str):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.flash_attn",
        generator="generate-wmma-flash-attn-kernel",
    )


def _compile_attention_graph_ir(tile_ir: str, *, tile_q: int, tile_kv: int):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.flash_attn",
        generator="generate-wmma-flash-attn-kernel",
        semantic_pipeline=(f"tessera-tile-ir-lowering{{tile-q={tile_q} tile-kv={tile_kv} sm=90}}"),
    )


def _compile_attention_backward_tile_ir(tile_ir: str):
    return _compile_native_tile_ir(
        tile_ir,
        directive="tessera_rocm.flash_attn_bwd",
        generator=("generate-wmma-flash-attn-kernel,generate-wmma-flash-attn-bwd-kernel"),
    )


def _compile_attention_backward_graph_ir(graph_ir: str, *, tile_q: int, tile_kv: int):
    return _compile_native_tile_ir(
        graph_ir,
        directive="tessera_rocm.flash_attn_bwd",
        generator=("generate-wmma-flash-attn-kernel,generate-wmma-flash-attn-bwd-kernel"),
        semantic_pipeline=(f"tessera-tile-ir-lowering{{tile-q={tile_q} tile-kv={tile_kv} sm=90}}"),
    )


def package_softmax(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _softmax_contract(module)
    if contract is None:
        raise ValueError("gfx1151 native packaging requires one static f16/f32 last-axis softmax")
    input_name, output_name, dtype, shape = contract
    storage = "f16" if dtype == "fp16" else "f32"
    entry = f"tessera_tile_softmax_{storage}"
    abi_id = GFX1151_SOFTMAX_F16_ABI if dtype == "fp16" else GFX1151_SOFTMAX_F32_ABI
    alignment = 2 if dtype == "fp16" else 4
    tile_ir = emit_softmax_tile_ir(entry=entry, storage=storage)
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", dtype, len(shape), "row_major", alignment),
            BufferBinding(1, output_name, "output", dtype, len(shape), "row_major", alignment),
        ),
        scalars=(
            ScalarArgument(2, "Rows", "int64"),
            ScalarArgument(3, "K", "int64"),
        ),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name in (input_name, output_name)
            for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="gfx1151_softmax_workgroup_per_row_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-1",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "workgroup_per_row_256",
            "shape": list(shape),
            "storage": storage,
            "accum": "f32",
            "axis": -1,
            "exp_mode": "accurate",
            "ftz": False,
            "rows": rows,
            "columns": columns,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)


def package_reduction(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 reduction packaging requires one static f16/bf16/f32 "
            "sum/mean/max with f32 output and one normalized axis"
        )
    input_name, output_name, dtype, kind, shape, output_shape, axis, keepdims = contract
    storage = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    entry = f"tessera_tile_reduce_{kind}_{storage}"
    abi_id = {
        "fp16": GFX1151_REDUCE_F16_ABI,
        "bf16": GFX1151_REDUCE_BF16_ABI,
        "fp32": GFX1151_REDUCE_F32_ABI,
    }[dtype]
    outer = math.prod(shape[:axis]) if axis else 1
    axis_extent = shape[axis]
    inner = math.prod(shape[axis + 1 :]) if axis + 1 < len(shape) else 1
    tile_ir = emit_reduce_tile_ir(
        entry=entry,
        storage=storage,
        kind=kind,
        axis=axis,
        keepdims=keepdims,
        inner_is_one=inner == 1,
    )
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_reduction_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(
                0,
                input_name,
                "input",
                dtype,
                len(shape),
                "row_major",
                2 if dtype in {"fp16", "bf16"} else 4,
            ),
            BufferBinding(1, output_name, "output", "fp32", len(output_shape), "row_major", 4),
        ),
        scalars=(
            ScalarArgument(2, "Outer", "int64"),
            ScalarArgument(3, "AxisExtent", "int64"),
            ScalarArgument(4, "Inner", "int64"),
        ),
        shape_guards=tuple(
            [ShapeGuard(input_name, index, "eq", extent) for index, extent in enumerate(shape)]
            + [ShapeGuard(output_name, index, "eq", extent) for index, extent in enumerate(output_shape)]
        ),
        geometry=LaunchGeometry(policy="gfx1151_reduce_workgroup_per_output_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "workgroup_per_output_256",
            "shape": list(shape),
            "storage": storage,
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
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)


def package_paged_kv_read(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 paged-KV packaging requires static f32 [P,PS,H,D] pages, "
            "rank-1 int32 page table, explicit valid start/end, and f32 output"
        )
    pages_name, table_name, output_name, dims = contract
    physical_pages, logical_pages, page_size, heads, dim, start, tokens = dims
    entry = "tessera_tile_paged_kv_read_f32_direct"
    tile_ir = emit_paged_kv_read_tile_ir(entry=entry)
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_paged_kv_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX1151_PAGED_KV_F32_ABI),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=GFX1151_PAGED_KV_F32_ABI,
        buffers=(
            BufferBinding(0, pages_name, "input", "fp32", 4, "row_major", 4),
            BufferBinding(1, table_name, "input", "int32", 1, "row_major", 4),
            BufferBinding(2, output_name, "output", "fp32", 3, "row_major", 4),
        ),
        scalars=tuple(
            ScalarArgument(3 + index, name, "int64")
            for index, name in enumerate(("P", "LP", "PageSize", "H", "D", "Start", "Tokens"))
        ),
        shape_guards=(
            ShapeGuard(pages_name, 0, "eq", physical_pages),
            ShapeGuard(pages_name, 1, "eq", page_size),
            ShapeGuard(pages_name, 2, "eq", heads),
            ShapeGuard(pages_name, 3, "eq", dim),
            ShapeGuard(table_name, 0, "eq", logical_pages),
            ShapeGuard(output_name, 0, "eq", tokens),
            ShapeGuard(output_name, 1, "eq", heads),
            ShapeGuard(output_name, 2, "eq", dim),
        ),
        geometry=LaunchGeometry(policy="gfx1151_paged_kv_direct_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "route": "direct",
            "shape": list(dims),
            "storage": "f32",
            "table_storage": "i32",
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)


def package_attention(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _attention_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 attention packaging requires static rank-4 f16/bf16 Q/K/V, "
            "f32 output, equal head/value dimensions divisible by 16, compatible "
            "MHA/GQA heads, deterministic dropout metadata, and a supported "
            "causal/window policy"
        )
    (
        names,
        bias_name,
        dtype,
        dims,
        scale,
        causal,
        window_left,
        window_right,
        softcap,
        dropout_p,
        dropout_seed,
    ) = contract
    q_name, k_name, v_name = names
    storage = {"fp16": "f16", "bf16": "bf16"}[dtype]
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:"
        f"{window_right}:{softcap:.17g}:{dropout_p:.17g}:"
        f"{dropout_seed}".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_attention_{storage}_{'causal' if causal else 'full'}_{semantic_key}"
    abi_id = GFX1151_ATTN_F16_ABI if dtype == "fp16" else GFX1151_ATTN_BF16_ABI
    canonical_route = (window_left < 0 and window_right < 0) or (causal and window_left >= 0 and window_right == 0)
    if canonical_route:
        tile_kv = 16
        tile_ir = emit_attention_graph_ir(
            entry=entry,
            storage=storage,
            dims=dims,
            scale=scale,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
            bias=bias_name is not None,
            softcap=softcap,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed,
            tile_kv=tile_kv,
        )
        compiled = _compile_attention_graph_ir(tile_ir, tile_q=dims[3], tile_kv=tile_kv)
    else:
        tile_ir = emit_attention_tile_ir(
            entry=entry,
            storage=storage,
            dims=dims,
            scale=scale,
            causal=causal,
            bias=bias_name is not None,
            window_left=window_left,
            window_right=window_right,
            softcap=softcap,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed,
        )
        compiled = _compile_attention_tile_ir(tile_ir)
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = compiled
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    function = module.functions[0]
    output_name = function.body[0].result or "output"
    b, hq, hkv, sq, sk, d, dv = dims
    alignment = 2
    descriptor_buffers = [
        BufferBinding(0, q_name, "input", dtype, 4, "row_major", alignment),
        BufferBinding(1, k_name, "input", dtype, 4, "row_major", alignment),
        BufferBinding(2, v_name, "input", dtype, 4, "row_major", alignment),
    ]
    if bias_name is not None:
        descriptor_buffers.append(BufferBinding(3, bias_name, "input", "fp32", 4, "row_major", 4))
    descriptor_buffers.append(
        BufferBinding(
            3 + int(bias_name is not None),
            output_name,
            "output",
            "fp32",
            4,
            "row_major",
            4,
        )
    )
    scalars = [
        ScalarArgument(4 + int(bias_name is not None), "Sq", "int64"),
        ScalarArgument(5 + int(bias_name is not None), "Sk", "int64"),
        ScalarArgument(6 + int(bias_name is not None), "Scale", "float32"),
        ScalarArgument(7 + int(bias_name is not None), "Causal", "int64"),
    ]
    if hq != hkv:
        scalars.extend(
            (
                ScalarArgument(8 + int(bias_name is not None), "Hq", "int64"),
                ScalarArgument(9 + int(bias_name is not None), "KvRatio", "int64"),
            )
        )
    if window_left >= 0:
        scalars.append(ScalarArgument(8 + int(bias_name is not None) + 2 * int(hq != hkv), "Window", "int64"))
    if softcap > 0.0:
        scalars.append(
            ScalarArgument(
                8 + int(bias_name is not None) + 2 * int(hq != hkv) + int(window_left >= 0),
                "Softcap",
                "float32",
            )
        )
    if dropout_p > 0.0:
        dropout_base = 8 + int(bias_name is not None) + 2 * int(hq != hkv) + int(window_left >= 0) + int(softcap > 0.0)
        scalars.extend(
            (
                ScalarArgument(dropout_base, "DropoutP", "float32"),
                ScalarArgument(dropout_base + 1, "DropoutSeed", "int64"),
            )
        )
    guards = [
        ShapeGuard(q_name, 0, "eq", b),
        ShapeGuard(q_name, 1, "eq", hq),
        ShapeGuard(q_name, 2, "eq", sq),
        ShapeGuard(q_name, 3, "eq", d),
        ShapeGuard(k_name, 0, "eq", b),
        ShapeGuard(k_name, 1, "eq", hkv),
        ShapeGuard(k_name, 2, "eq", sk),
        ShapeGuard(k_name, 3, "eq", d),
        ShapeGuard(v_name, 0, "eq", b),
        ShapeGuard(v_name, 1, "eq", hkv),
        ShapeGuard(v_name, 2, "eq", sk),
        ShapeGuard(v_name, 3, "eq", dv),
        ShapeGuard(output_name, 0, "eq", b),
        ShapeGuard(output_name, 1, "eq", hq),
        ShapeGuard(output_name, 2, "eq", sq),
        ShapeGuard(output_name, 3, "eq", dv),
    ]
    if bias_name is not None:
        guards.extend(
            (
                ShapeGuard(bias_name, 0, "eq", b),
                ShapeGuard(bias_name, 1, "eq", hq),
                ShapeGuard(bias_name, 2, "eq", sq),
                ShapeGuard(bias_name, 3, "eq", sk),
            )
        )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=tuple(descriptor_buffers),
        scalars=tuple(scalars),
        shape_guards=tuple(guards),
        geometry=LaunchGeometry(policy="gfx1151_attention_wmma_query_tile_wave32"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": (
                "CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26"
                if canonical_route
                else "ROCM-E2E-ATTENTION-CARRIERS-2026-07-26"
            ),
            "schedule": ("gfx1151_wmma_canonical_streaming" if canonical_route else "gfx1151_wmma_streaming"),
            "semantic_route": (
                "canonical_rank4_kv_scf_for" if canonical_route else "tile.attention_kernel_compatibility"
            ),
            "shape": list(dims),
            "storage": storage,
            "accum": "f32",
            "output": "f32",
            "scale": scale,
            "causal": causal,
            "gqa": hq != hkv,
            "kv_ratio": hq // hkv,
            "bias": bias_name is not None,
            "window_left": window_left,
            "window_right": window_right,
            "softcap": softcap,
            "dropout_p": dropout_p,
            "dropout_seed": dropout_seed,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)


def package_attention_backward(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativeProgram:
    """Package the gfx1151 forward-recompute + split/reduced VJP program."""
    contract = _attention_backward_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 optimized attention backward requires static rank-4 "
            "f16/bf16 dO/Q/K/V, fp32 dQ/dK/dV, D == Dv divisible by 16, "
            "compatible MHA/GQA heads, deterministic dropout replay, "
            "and a supported causal/window policy"
        )
    (
        dtype,
        names,
        bias_name,
        result_names,
        dims,
        scale,
        causal,
        window_left,
        window_right,
        softcap,
        dropout_p,
        dropout_seed,
    ) = contract
    do_name, q_name, k_name, v_name = names
    dq_name, dk_name, dv_name = (name.removeprefix("%") for name in result_names)
    storage = {"fp16": "f16", "bf16": "bf16"}[dtype]
    b, hq, hkv, sq, sk, d, dv = dims
    requested_checkpoint = module.functions[0].body[0].kwargs.get("lse_checkpoint", "auto")
    if requested_checkpoint not in {"auto", "saved", "recompute"}:
        raise ValueError(
            "gfx1151 attention backward lse_checkpoint must be 'auto', "
            f"'saved', or 'recompute'; got {requested_checkpoint!r}"
        )
    # Exact gfx1151 host-wall sweeps show a stable saved-LSE win at 128x128 and
    # 256x256 for both f16 and bf16, while shorter lengths are mixed.
    selected_checkpoint = (
        "saved"
        if requested_checkpoint == "auto" and max(sq, sk) >= 128
        else "recompute"
        if requested_checkpoint == "auto"
        else requested_checkpoint
    )
    save_lse = selected_checkpoint == "saved"
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:"
        f"{window_right}:{softcap:.17g}:{dropout_p:.17g}:"
        f"{dropout_seed}:split_reduced:{selected_checkpoint}".encode()
    ).hexdigest()[:10]
    forward_entry = f"tessera_tile_attention_bwd_recompute_{storage}_{semantic_key}"
    backward_entry = f"tessera_tile_attention_backward_{storage}_{semantic_key}"
    stage_symbols = (
        forward_entry,
        f"{backward_entry}_pre",
        f"{backward_entry}_dkdv",
        f"{backward_entry}_dkdv_reduce",
        f"{backward_entry}_dq",
    )
    stage_abis = (
        GFX1151_ATTN_F16_ABI if dtype == "fp16" else GFX1151_ATTN_BF16_ABI,
        GFX1151_ATTN_BWD_PRE_ABI,
        GFX1151_ATTN_BWD_DKDV_ABI,
        GFX1151_ATTN_BWD_REDUCE_ABI,
        GFX1151_ATTN_BWD_DQ_ABI,
    )
    tile_ir = emit_attention_backward_graph_ir(
        forward_entry=forward_entry,
        backward_entry=backward_entry,
        storage=storage,
        dims=dims,
        scale=scale,
        causal=causal,
        bias=bias_name is not None,
        window_left=window_left,
        window_right=window_right,
        softcap=softcap,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        save_lse=save_lse,
    )
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_attention_backward_graph_ir(tile_ir, tile_q=sq, tile_kv=16)
    if (
        'source = "canonical_rank4_kv_scf_for"' not in target_ir
        or 'source = "canonical_tensor_backward_scf_for"' not in target_ir
        or "canonical_phase_loops = true" not in target_ir
    ):
        raise RuntimeError(
            "gfx1151 attention backward packaging requires direct shared "
            "forward and tensor-valued backward Target-IR consumers"
        )
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=tuple(
            NativeEntryPoint(symbol, abi) for symbol, abi in zip(stage_symbols, stage_abis, strict=True)
        ),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )

    shared_workspace = plan_attention_backward_workspace(
        batch=b,
        query_heads=hq,
        kv_heads=hkv,
        query_rows=sq,
        key_rows=sk,
        head_dim=d,
        value_dim=dv,
        split_count=2,
    )
    slices = [
        ROCMWorkspaceSlice(item.name, item.offset, item.bytes, item.initialization) for item in shared_workspace.slices
    ]
    workspace = WorkspaceRequirement(
        bytes=shared_workspace.bytes,
        alignment=shared_workspace.alignment,
        lifetime="launch",
        initialization="undefined",
    )

    alignment = 2
    common_guards = (
        ShapeGuard(do_name, 0, "eq", b),
        ShapeGuard(do_name, 1, "eq", hq),
        ShapeGuard(do_name, 2, "eq", sq),
        ShapeGuard(do_name, 3, "eq", dv),
        ShapeGuard(q_name, 0, "eq", b),
        ShapeGuard(q_name, 1, "eq", hq),
        ShapeGuard(q_name, 2, "eq", sq),
        ShapeGuard(q_name, 3, "eq", d),
        ShapeGuard(k_name, 0, "eq", b),
        ShapeGuard(k_name, 1, "eq", hkv),
        ShapeGuard(k_name, 2, "eq", sk),
        ShapeGuard(k_name, 3, "eq", d),
        ShapeGuard(v_name, 0, "eq", b),
        ShapeGuard(v_name, 1, "eq", hkv),
        ShapeGuard(v_name, 2, "eq", sk),
        ShapeGuard(v_name, 3, "eq", dv),
        ShapeGuard(dq_name, 0, "eq", b),
        ShapeGuard(dq_name, 1, "eq", hq),
        ShapeGuard(dq_name, 2, "eq", sq),
        ShapeGuard(dq_name, 3, "eq", d),
        ShapeGuard(dk_name, 0, "eq", b),
        ShapeGuard(dk_name, 1, "eq", hkv),
        ShapeGuard(dk_name, 2, "eq", sk),
        ShapeGuard(dk_name, 3, "eq", d),
        ShapeGuard(dv_name, 0, "eq", b),
        ShapeGuard(dv_name, 1, "eq", hkv),
        ShapeGuard(dv_name, 2, "eq", sk),
        ShapeGuard(dv_name, 3, "eq", dv),
    )
    bias_guard = (
        (
            ShapeGuard(bias_name, 0, "eq", b),
            ShapeGuard(bias_name, 1, "eq", hq),
            ShapeGuard(bias_name, 2, "eq", sq),
            ShapeGuard(bias_name, 3, "eq", sk),
        )
        if bias_name is not None
        else ()
    )
    guards = common_guards + bias_guard
    user_buffers = {
        do_name: BufferBinding(0, do_name, "input", dtype, 4, "row_major", alignment),
        q_name: BufferBinding(0, q_name, "input", dtype, 4, "row_major", alignment),
        k_name: BufferBinding(0, k_name, "input", dtype, 4, "row_major", alignment),
        v_name: BufferBinding(0, v_name, "input", dtype, 4, "row_major", alignment),
        dq_name: BufferBinding(0, dq_name, "output", "fp32", 4, "row_major", 4),
        dk_name: BufferBinding(0, dk_name, "output", "fp32", 4, "row_major", 4),
        dv_name: BufferBinding(0, dv_name, "output", "fp32", 4, "row_major", 4),
    }
    if bias_name is not None:
        user_buffers[bias_name] = BufferBinding(0, bias_name, "input", "fp32", 4, "row_major", 4)

    def bindings(*names_and_directions: tuple[str, str, str, int]) -> tuple[BufferBinding, ...]:
        result: list[BufferBinding] = []
        for ordinal, (name, direction, dtype_name, rank) in enumerate(names_and_directions):
            if name in user_buffers:
                base = user_buffers[name]
                result.append(
                    BufferBinding(
                        ordinal,
                        name,
                        direction,
                        base.dtype,
                        base.rank,
                        base.layout,
                        base.alignment,
                    )
                )
            else:
                result.append(
                    BufferBinding(
                        ordinal,
                        name,
                        direction,
                        dtype_name,
                        rank,
                        "program_workspace",
                        4,
                    )
                )
        return tuple(result)

    def guards_for(
        names_and_directions: tuple[tuple[str, str, str, int], ...],
    ) -> tuple[ShapeGuard, ...]:
        stage_names = {item[0] for item in names_and_directions}
        return tuple(guard for guard in guards if guard.binding in stage_names)

    option_buffers = ((bias_name, "input", "fp32", 4),) if bias_name is not None else ()
    forward_specs = (
        (q_name, "input", dtype, 4),
        (k_name, "input", dtype, 4),
        (v_name, "input", dtype, 4),
        *option_buffers,
        ("forward_o", "output", "fp32", 3),
        *((("row_lse", "output", "fp32", 2),) if save_lse else ()),
    )
    stage_buffer_specs = (
        forward_specs,
        (
            (q_name, "input", dtype, 4),
            (k_name, "input", dtype, 4),
            (do_name, "input", dtype, 4),
            ("forward_o", "input", "fp32", 3),
            ("row_lse", "input" if save_lse else "output", "fp32", 2),
            ("row_delta", "output", "fp32", 2),
            *option_buffers,
        ),
        (
            (q_name, "input", dtype, 4),
            (k_name, "input", dtype, 4),
            (v_name, "input", dtype, 4),
            (do_name, "input", dtype, 4),
            ("row_lse", "input", "fp32", 2),
            ("row_delta", "input", "fp32", 2),
            (dk_name, "inout", "fp32", 4),
            (dv_name, "inout", "fp32", 4),
            ("partial_dk", "output", "fp32", 3),
            ("partial_dv", "output", "fp32", 3),
            *option_buffers,
        ),
        (
            (dk_name, "inout", "fp32", 4),
            (dv_name, "inout", "fp32", 4),
            ("partial_dk", "input", "fp32", 3),
            ("partial_dv", "input", "fp32", 3),
        ),
        (
            (q_name, "input", dtype, 4),
            (k_name, "input", dtype, 4),
            (v_name, "input", dtype, 4),
            (do_name, "input", dtype, 4),
            ("row_lse", "input", "fp32", 2),
            ("row_delta", "input", "fp32", 2),
            (dq_name, "output", "fp32", 4),
            *option_buffers,
        ),
    )
    stage_policies = (
        "gfx1151_attention_bwd_forward_recompute_wave32",
        "gfx1151_attention_bwd_pre_query_tile_wave32",
        "gfx1151_attention_bwd_dkdv_two_split_wave32",
        "gfx1151_attention_bwd_reduce_256",
        "gfx1151_attention_bwd_dq_query_tile_wave32",
    )
    stage_names = ("forward_recompute", "pre", "dkdv_split", "dkdv_reduce", "dq")
    base_provenance = {
        "work_item": "ROCM-E2E-ATTENTION",
        "sync_key": "ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26",
        "schedule": "gfx1151_wmma_backward_split_reduced",
        "semantic_route": "canonical_tensor_backward_scf_for",
        "route": "deterministic_split_reduced",
        "deterministic": True,
        "shape": list(dims),
        "storage": storage,
        "accum": "f32",
        "gradient_storage": "f32",
        "scale": scale,
        "causal": causal,
        "gqa": hq != hkv,
        "kv_ratio": hq // hkv,
        "bias": bias_name is not None,
        "window_left": window_left,
        "window_right": window_right,
        "softcap": softcap,
        "dropout_p": dropout_p,
        "dropout_seed": dropout_seed,
        "workspace_owner": "program_launch",
        "workspace_bytes": workspace.bytes,
        "workspace_contract": "canonical_attention_backward_split_v1",
        "lse_checkpoint": selected_checkpoint,
        "lse_checkpoint_policy": requested_checkpoint,
        "source_ir_kind": "canonical_attention_tensor_program",
        "split_count": shared_workspace.split_count,
        "reduction_order": list(shared_workspace.reduction_order),
        "workspace_slices": [
            {
                "name": item.name,
                "offset": item.offset,
                "bytes": item.bytes,
                "initialization": item.initialization,
                "producer": shared.producer,
                "consumers": list(shared.consumers),
            }
            for item, shared in zip(slices, shared_workspace.slices, strict=True)
        ],
        "shared_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        # Compatibility key retained for artifact-schema readers that have not
        # yet generalized the historical Tile-only source field.
        "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
    }
    descriptors = tuple(
        LaunchDescriptor(
            image_digest=image.image_digest,
            entry_symbol=symbol,
            abi_id=abi,
            buffers=bindings(*buffer_specs),
            scalars=(),
            shape_guards=guards_for(buffer_specs),
            geometry=LaunchGeometry(policy=policy),
            workspace=workspace,
            ordering=OrderingSemantics(
                ordered_submission=True,
                residency="all",
                synchronization=("program_order", "completion"),
            ),
            provenance={
                **base_provenance,
                "stage": stage_name,
                "stage_index": stage_index,
            },
        )
        for stage_index, (symbol, abi, buffer_specs, policy, stage_name) in enumerate(
            zip(
                stage_symbols,
                stage_abis,
                stage_buffer_specs,
                stage_policies,
                stage_names,
                strict=True,
            )
        )
    )
    return ROCMNativeProgram(
        tile_ir,
        target_ir,
        backend_ir,
        image,
        descriptors,
        workspace,
        tuple(slices),
    )


def package_moe_dispatch(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _moe_dispatch_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 MoE dispatch packaging requires static f32 [T,H] input, "
            "rank-1 int32 token indices, and f32 [S,H] output"
        )
    x_name, token_name, output_name, dims = contract
    tokens, slots, hidden = dims
    entry = "tessera_tile_moe_dispatch_f32_direct"
    tile_ir = emit_moe_dispatch_tile_ir(entry=entry)
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_moe_dispatch_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX1151_MOE_DISPATCH_F32_ABI),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=GFX1151_MOE_DISPATCH_F32_ABI,
        buffers=(
            BufferBinding(0, x_name, "input", "fp32", 2, "row_major", 4),
            BufferBinding(1, token_name, "input", "int32", 1, "row_major", 4),
            BufferBinding(2, output_name, "output", "fp32", 2, "row_major", 4),
        ),
        scalars=tuple(ScalarArgument(3 + index, name, "int64") for index, name in enumerate(("T", "S", "H"))),
        shape_guards=(
            ShapeGuard(x_name, 0, "eq", tokens),
            ShapeGuard(x_name, 1, "eq", hidden),
            ShapeGuard(token_name, 0, "eq", slots),
            ShapeGuard(output_name, 0, "eq", slots),
            ShapeGuard(output_name, 1, "eq", hidden),
        ),
        geometry=LaunchGeometry(policy="gfx1151_moe_dispatch_direct_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "route": "direct_gather",
            "shape": list(dims),
            "storage": "f32",
            "index_storage": "i32",
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)


__all__ = [
    "GFX1151_ATTN_BF16_ABI",
    "GFX1151_ATTN_BWD_DKDV_ABI",
    "GFX1151_ATTN_BWD_DQ_ABI",
    "GFX1151_ATTN_BWD_PRE_ABI",
    "GFX1151_ATTN_BWD_REDUCE_ABI",
    "GFX1151_ATTN_F16_ABI",
    "GFX1151_MOE_DISPATCH_F32_ABI",
    "GFX1151_PAGED_KV_F32_ABI",
    "GFX1151_REDUCE_BF16_ABI",
    "GFX1151_REDUCE_F16_ABI",
    "GFX1151_REDUCE_F32_ABI",
    "GFX1151_SOFTMAX_F16_ABI",
    "GFX1151_SOFTMAX_F32_ABI",
    "ROCMNativePackage",
    "ROCMNativeProgram",
    "ROCMWorkspaceSlice",
    "emit_attention_backward_tile_ir",
    "emit_attention_backward_graph_ir",
    "emit_attention_tile_ir",
    "emit_moe_dispatch_tile_ir",
    "emit_reduce_tile_ir",
    "emit_paged_kv_read_tile_ir",
    "emit_softmax_tile_ir",
    "package_moe_dispatch",
    "package_native",
    "package_attention",
    "package_attention_backward",
    "package_reduction",
    "package_paged_kv_read",
    "package_softmax",
    "requests_moe_dispatch",
    "requests_attention",
    "requests_attention_backward",
    "requests_reduction",
    "requests_paged_kv_read",
    "requests_softmax",
    "native_package_kind",
    "supports_moe_dispatch",
    "supports_attention",
    "supports_attention_backward",
    "supports_reduction",
    "supports_paged_kv_read",
    "supports_softmax",
    "supports_native_package",
    "native_packaging_available",
    "tools_available",
]
