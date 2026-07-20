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
)


GFX1151_SOFTMAX_F16_ABI = "tessera.rocm.softmax.x_o_rows_k.f16.v1"
GFX1151_SOFTMAX_F32_ABI = "tessera.rocm.softmax.x_o_rows_k.f32.v1"
GFX1151_REDUCE_F32_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.f32.v1"
GFX1151_REDUCE_F16_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.f16_f32out.v1"
GFX1151_REDUCE_BF16_ABI = "tessera.rocm.reduce.x_o_outer_axis_inner.bf16_f32out.v1"
GFX1151_PAGED_KV_F32_ABI = "tessera.rocm.paged_kv.pages_table_o_dims.f32_i32.v1"
GFX1151_MOE_DISPATCH_F32_ABI = "tessera.rocm.moe_dispatch.x_token_o_t_s_h.f32_i32.v1"


@dataclass(frozen=True)
class ROCMNativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


_cache: dict[
    str,
    tuple[str, str, bytes, str, str, tuple[DeviceLibraryRecord, ...]],
] = {}

_BUILTIN_BITCODE_RE = re.compile(r'"-mlink-builtin-bitcode"\s+"([^"]+\.bc)"')


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
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %token: !llvm.ptr, %o: !llvm.ptr,
                     %t: i64, %s: i64, %h: i64) {{
    tile.moe_dispatch_kernel %x, %token, %o, %t, %s, %h {{
      storage = "f32", index_storage = "i32"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
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


def _compile_native_tile_ir(
    tile_ir: str,
    *,
    directive: str,
    generator: str,
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
    key = hashlib.sha256(f"{tile_ir}|{library_identity}".encode()).hexdigest()
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

    target_pipeline = "builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})"
    native_pipeline = (
        "builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,"
        f"lower-tile-to-rocm{{arch=gfx1151}},{generator},"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1151},"
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
        scalars=tuple(
            ScalarArgument(3 + index, name, "int64")
            for index, name in enumerate(("T", "S", "H"))
        ),
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
    "GFX1151_MOE_DISPATCH_F32_ABI",
    "GFX1151_PAGED_KV_F32_ABI",
    "GFX1151_REDUCE_BF16_ABI",
    "GFX1151_REDUCE_F16_ABI",
    "GFX1151_REDUCE_F32_ABI",
    "GFX1151_SOFTMAX_F16_ABI",
    "GFX1151_SOFTMAX_F32_ABI",
    "ROCMNativePackage",
    "emit_moe_dispatch_tile_ir",
    "emit_reduce_tile_ir",
    "emit_paged_kv_read_tile_ir",
    "emit_softmax_tile_ir",
    "package_moe_dispatch",
    "package_reduction",
    "package_paged_kv_read",
    "package_softmax",
    "requests_moe_dispatch",
    "requests_reduction",
    "requests_paged_kv_read",
    "requests_softmax",
    "supports_moe_dispatch",
    "supports_reduction",
    "supports_paged_kv_read",
    "supports_softmax",
    "tools_available",
]
