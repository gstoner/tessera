"""Compiler-owned Apple GPU native-library descriptors for APPLE-E2E-1."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .graph_ir import GraphIRModule
from .native_artifact import (BufferBinding, LaunchDescriptor, LaunchGeometry,
                              NativeEntryPoint, NativeImageArtifact,
                              OrderingSemantics, ShapeGuard)

APPLE_BMM_F32_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.f32.v1"
APPLE_BMM_F32_SYMBOL = "tessera_apple_gpu_bmm_f32"
APPLE_BMM_F16_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.f16.v1"
APPLE_BMM_F16_SYMBOL = "tessera_apple_gpu_bmm_f16"
APPLE_BMM_BF16_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.bf16.v1"
APPLE_BMM_BF16_SYMBOL = "tessera_apple_gpu_bmm_bf16"
APPLE_SOFTMAX_F32_ABI = "tessera.apple.softmax.x_o_rows_columns.f32.v1"
APPLE_SOFTMAX_F32_SYMBOL = "tessera_apple_gpu_softmax_f32"
APPLE_SOFTMAX_F16_ABI = "tessera.apple.softmax.x_o_rows_columns.f16.v1"
APPLE_SOFTMAX_F16_SYMBOL = "tessera_apple_gpu_softmax_f16"
APPLE_SOFTMAX_BF16_ABI = "tessera.apple.softmax.x_o_rows_columns.bf16.v1"
APPLE_SOFTMAX_BF16_SYMBOL = "tessera_apple_gpu_softmax_bf16"
APPLE_TRANSPOSE_F32_ABI = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f32.v1"
APPLE_TRANSPOSE_F32_SYMBOL = "tessera_apple_gpu_mpsgraph_transpose_f32"
APPLE_TRANSPOSE_F16_ABI = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f16.v1"
APPLE_TRANSPOSE_F16_SYMBOL = "tessera_apple_gpu_mpsgraph_transpose_f16"
APPLE_TRANSPOSE_BF16_ABI = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.bf16.v1"
APPLE_TRANSPOSE_BF16_SYMBOL = APPLE_TRANSPOSE_F16_SYMBOL

_BMM_VARIANTS: dict[str, tuple[str, str]] = {
    "fp32": (APPLE_BMM_F32_SYMBOL, APPLE_BMM_F32_ABI),
    "fp16": (APPLE_BMM_F16_SYMBOL, APPLE_BMM_F16_ABI),
    "bf16": (APPLE_BMM_BF16_SYMBOL, APPLE_BMM_BF16_ABI),
}
_SOFTMAX_VARIANTS = {
    "fp32": (APPLE_SOFTMAX_F32_SYMBOL, APPLE_SOFTMAX_F32_ABI),
    "fp16": (APPLE_SOFTMAX_F16_SYMBOL, APPLE_SOFTMAX_F16_ABI),
    "bf16": (APPLE_SOFTMAX_BF16_SYMBOL, APPLE_SOFTMAX_BF16_ABI),
}
_TRANSPOSE_VARIANTS = {
    "fp32": (APPLE_TRANSPOSE_F32_SYMBOL, APPLE_TRANSPOSE_F32_ABI),
    "fp16": (APPLE_TRANSPOSE_F16_SYMBOL, APPLE_TRANSPOSE_F16_ABI),
    "bf16": (APPLE_TRANSPOSE_BF16_SYMBOL, APPLE_TRANSPOSE_BF16_ABI),
}

# APPLE-E2E-1 owns an explicit descriptor disposition for every currently
# value-lowered family.  Only BMM has a descriptor today; the other entries
# prevent a value-producing Target-IR symbol (or a host-specific probe) from
# being silently mistaken for descriptor-first execution.
APPLE_VALUE_DESCRIPTOR_STATES: dict[str, str] = {
    "tessera.transpose": "descriptor_ready",
    "tessera.rl.ppo_policy_loss": "descriptor_ready",
    "tessera.ebm.energy_quadratic": "descriptor_ready",
    "tessera.ebm.langevin_step": "descriptor_ready",
    "tessera.ebm.refinement": "descriptor_ready",
    "tessera.ebm.partition_exact": "descriptor_ready",
    "tessera.clifford.geometric_product": "descriptor_ready",
    "tessera.cholesky": "descriptor_ready",
    "tessera.tri_solve": "descriptor_ready",
    "tessera.cholesky_solve": "descriptor_ready",
}


def value_descriptor_state(op_name: str) -> str:
    """Return the APPLE-E2E-1 descriptor disposition for one value-lane op."""
    return APPLE_VALUE_DESCRIPTOR_STATES.get(op_name, "unsupported")

@dataclass(frozen=True)
class AppleNativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor

def _runtime_library_path() -> Path | None:
    configured = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = Path(__file__).resolve().parents[3]
    for path in (
        root / "build-apple/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.dylib",
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.dylib",
    ):
        if path.is_file():
            return path
    return None

def tools_available() -> bool:
    return _runtime_library_path() is not None

def _contract(module: GraphIRModule):
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name not in {"tessera.batched_gemm", "tessera.matmul", "tessera.gemm"} or len(op.operands) != 2:
        return None
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if any(name not in args for name in names) or len(fn.result_types) != 1:
        return None
    dtype = args[names[0]].ir_type.dtype
    if (dtype not in _BMM_VARIANTS or args[names[1]].ir_type.dtype != dtype
            or fn.result_types[0].dtype != dtype):
        return None
    try:
        a_shape = tuple(int(v) for v in args[names[0]].ir_type.shape)
        b_shape = tuple(int(v) for v in args[names[1]].ir_type.shape)
        out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return None
    if len(a_shape) != 3 or len(b_shape) != 3:
        return None
    batch, m, k = a_shape
    b_batch, bk, n = b_shape
    if bk != k or b_batch not in {1, batch} or out_shape != (batch, m, n):
        return None
    return names[0], names[1], op.result or "output", (batch, m, n, k), b_batch == 1, dtype


def _softmax_contract(module: GraphIRModule):
    """Return the static last-axis f32 softmax contract, if present.

    The underlying C ABI accepts only ``rows`` and ``columns``.  Keeping this
    descriptor deliberately rank-2 and axis-last prevents an apparent package
    from silently flattening a higher-rank or non-contiguous Graph IR value.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name != "tessera.softmax" or len(op.operands) != 1:
        return None
    axis = op.kwargs.get("axis", -1)
    if axis not in {-1, 1} or len(fn.result_types) != 1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args or args[name].ir_type.dtype not in _SOFTMAX_VARIANTS or fn.result_types[0].dtype != args[name].ir_type.dtype:
        return None
    try:
        shape = tuple(int(v) for v in args[name].ir_type.shape)
        out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return None
    if len(shape) != 2 or out_shape != shape:
        return None
    return name, op.result or "output", shape


def _transpose_contract(module: GraphIRModule):
    """Return one static rank-2/rank-3 MPSGraph permute contract."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name != "tessera.transpose" or len(op.operands) != 1 or len(fn.result_types) != 1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args:
        return None
    dtype = args[name].ir_type.dtype
    if dtype not in _TRANSPOSE_VARIANTS or fn.result_types[0].dtype != dtype:
        return None
    try:
        shape = tuple(int(v) for v in args[name].ir_type.shape)
        out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return None
    if len(shape) not in {2, 3}:
        return None
    raw_axes = op.kwargs.get("axes", tuple(range(len(shape) - 1, -1, -1)))
    if not isinstance(raw_axes, (list, tuple)):
        return None
    try:
        axes = tuple(int(axis) % len(shape) for axis in raw_axes)
    except (TypeError, ValueError):
        return None
    if len(axes) != len(shape) or set(axes) != set(range(len(shape))) or out_shape != tuple(shape[axis] for axis in axes):
        return None
    return name, op.result or "output", shape, axes, dtype

def supports_native_package(module: GraphIRModule) -> bool:
    if _contract(module) is not None:
        return True
    if _softmax_contract(module) is not None:
        return True
    if _transpose_contract(module) is not None:
        return True
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return value_descriptor_state(module.functions[0].body[0].op_name) == "descriptor_ready"


_VALUE_SYMBOLS = {
    "tessera.rl.ppo_policy_loss": ("ppo_policy_loss", "tessera_apple_gpu_ppo_policy_loss_f32"),
    "tessera.ebm.energy_quadratic": ("ebm_energy_quadratic", "tessera_apple_gpu_ebm_energy_quadratic_value_f32"),
    "tessera.ebm.langevin_step": ("ebm_langevin_step", "tessera_apple_gpu_ebm_langevin_step_value_f32"),
    "tessera.ebm.refinement": ("ebm_refinement", "tessera_apple_gpu_ebm_refinement_value_f32"),
    "tessera.ebm.partition_exact": ("ebm_partition_exact", "tessera_apple_gpu_ebm_partition_exact_value_f32"),
    "tessera.cholesky": ("cholesky", "tessera_apple_gpu_cholesky_f32"),
    "tessera.tri_solve": ("tri_solve", "tessera_apple_gpu_tri_solve_f32"),
    "tessera.cholesky_solve": ("cholesky_solve", "tessera_apple_gpu_solve_cholesky_f32"),
    "tessera.clifford.geometric_product": ("clifford_geometric_product", "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"),
}


def package_native(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    if _contract(module) is not None:
        return package_batched_gemm(module, pipeline_name=pipeline_name)
    if _softmax_contract(module) is not None:
        return package_softmax(module, pipeline_name=pipeline_name)
    if _transpose_contract(module) is not None:
        return package_transpose(module, pipeline_name=pipeline_name)
    fn, op = module.functions[0], module.functions[0].body[0]
    entry = _VALUE_SYMBOLS.get(op.op_name)
    if entry is None or len(fn.result_types) != 1:
        raise ValueError(f"APPLE-E2E-1 has no descriptor contract for {op.op_name}")
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if not names or any(name not in args or args[name].ir_type.dtype != "fp32" for name in names) or fn.result_types[0].dtype != "fp32":
        raise ValueError("Apple value descriptors require static f32 tensor operands/results")
    try:
        shapes = {name: tuple(int(v) for v in args[name].ir_type.shape) for name in names}
        out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Apple value descriptors require static shapes") from exc
    kind, symbol = entry
    value_kwargs = dict(op.kwargs)
    if op.op_name == "tessera.rl.ppo_policy_loss" and len(names) > 3:
        if len(names) > 6:
            raise ValueError("Apple PPO descriptors accept at most mask, ref_logp, and entropy side tensors")
        symbol = "tessera_apple_gpu_ppo_policy_loss_ex_f32"
        value_kwargs.update({
            "has_mask": len(names) >= 4,
            "has_ref_kl": len(names) >= 5,
            "has_entropy": len(names) >= 6,
        })
    abi = f"tessera.apple.value.{kind}.f32.v1"
    out = op.result or "output"; target_ir = f'tessera_apple.gpu.kernel_call @{symbol} {{abi = "{abi}", status = "executable"}}'
    library = _runtime_library_path()
    if library is None: raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    payload = library.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name, compiler_fingerprint="apple-runtime-abi-v1", toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(), target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object", payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged")
    bindings = tuple(BufferBinding(i, name, "input", "fp32", len(shapes[name]), "row_major", 4) for i, name in enumerate(names)) + (BufferBinding(len(names), out, "output", "fp32", len(out_shape), "row_major", 4),)
    guards = tuple(ShapeGuard(name, axis, "eq", extent) for name, shape in list(shapes.items()) + [(out, out_shape)] for axis, extent in enumerate(shape))
    descriptor = LaunchDescriptor(image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi, buffers=bindings, shape_guards=guards, geometry=LaunchGeometry(policy="apple_value_executor"), ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)), provenance={"work_item":"APPLE-E2E-1", "route":"apple_value_executor", "op_kind":kind, "value_call":{"op":"tessera_apple.gpu.kernel_call", "op_kind":kind, "symbol":symbol, "status":"executable", **value_kwargs}})
    return AppleNativePackage("apple.value", target_ir, target_ir, image, descriptor)

def package_batched_gemm(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    contract = _contract(module)
    if contract is None:
        raise ValueError("Apple native package requires one static f32/f16/bf16 rank-3 batched_gemm")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    a, b, out, (batch, m, n, k), broadcast, dtype = contract
    symbol, abi = _BMM_VARIANTS[dtype]
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", storage = "{dtype}", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(BufferBinding(0, a, "input", dtype, 3, "row_major", 4 if dtype == "fp32" else 2),
                 BufferBinding(1, b, "input", dtype, 3, "row_major", 4 if dtype == "fp32" else 2),
                 BufferBinding(2, out, "output", dtype, 3, "row_major", 4 if dtype == "fp32" else 2)),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", extent) for name, shape in ((a, (batch,m,k)), (b, ((1 if broadcast else batch),k,n)), (out, (batch,m,n))) for axis, extent in enumerate(shape)),
        geometry=LaunchGeometry(policy="apple_mps_bmm"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-E2E-1", "route": "mps_bmm_native_library", "shape": [batch,m,n,k], "broadcast_b": broadcast, "storage": dtype},
    )
    return AppleNativePackage("tile.matmul_kernel", target_ir, target_ir, image, descriptor)


def package_softmax(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package one static rank-2 f32 softmax through its named Apple ABI."""
    contract = _softmax_contract(module)
    if contract is None:
        raise ValueError("Apple softmax package requires one static f32 rank-2 last-axis softmax")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    x, out, (rows, columns) = contract
    dtype = str(module.functions[0].args[0].ir_type.dtype or "")
    if dtype not in _SOFTMAX_VARIANTS:
        raise ValueError("Apple softmax package requires f32, f16, or bf16 storage")
    symbol, abi = _SOFTMAX_VARIANTS[dtype]
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", storage = "{dtype}", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(BufferBinding(0, x, "input", dtype, 2, "row_major", 4 if dtype == "fp32" else 2),
                 BufferBinding(1, out, "output", dtype, 2, "row_major", 4 if dtype == "fp32" else 2)),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name, shape in ((x, (rows, columns)), (out, (rows, columns)))
            for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="apple_msl_softmax"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-E2E-1", "route": "apple_softmax_native_library",
                    "shape": [rows, columns], "storage": dtype},
    )
    return AppleNativePackage("tile.softmax_kernel", target_ir, target_ir, image, descriptor)


def package_transpose(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package one static rank-2/rank-3 MPSGraph transpose ABI."""
    contract = _transpose_contract(module)
    if contract is None:
        raise ValueError("Apple transpose package requires one static f32/f16/bf16 rank-2/rank-3 permutation")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    x, out, shape, axes, dtype = contract
    symbol, abi = _TRANSPOSE_VARIANTS[dtype]
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", storage = "{dtype}", rank = {len(shape)}, status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )
    alignment = 4 if dtype == "fp32" else 2
    out_shape = tuple(shape[axis] for axis in axes)
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(BufferBinding(0, x, "input", dtype, len(shape), "row_major", alignment),
                 BufferBinding(1, out, "output", dtype, len(shape), "row_major", alignment)),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name, guarded_shape in ((x, shape), (out, out_shape))
            for axis, extent in enumerate(guarded_shape)
        ),
        geometry=LaunchGeometry(policy="apple_mpsgraph_transpose"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-E2E-1", "route": "apple_mpsgraph_transpose_native_library",
                    "shape": list(shape), "axes": list(axes), "storage": dtype},
    )
    return AppleNativePackage("tile.transpose_kernel", target_ir, target_ir, image, descriptor)
