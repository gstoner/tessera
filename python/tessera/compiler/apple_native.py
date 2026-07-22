"""Compiler-owned Apple GPU native-library descriptors for Apple E2E work."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .graph_ir import GraphIRModule
from .native_artifact import (BufferBinding, LaunchDescriptor, LaunchGeometry,
                              NativeEntryPoint, NativeImageArtifact,
                              OrderingSemantics, ScalarArgument, ShapeGuard)

APPLE_BMM_F32_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.f32.v1"
APPLE_BMM_F32_SYMBOL = "tessera_apple_gpu_bmm_f32"
APPLE_BMM_F16_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.f16.v1"
APPLE_BMM_F16_SYMBOL = "tessera_apple_gpu_bmm_f16"
APPLE_BMM_BF16_ABI = "tessera.apple.bmm.a_b_o_batch_m_n_k.bf16.v1"
APPLE_BMM_BF16_SYMBOL = "tessera_apple_gpu_bmm_bf16"
APPLE_SOFTMAX_F32_ABI = "tessera.apple.softmax.x_o_rows_columns.f32.v1"
APPLE_SOFTMAX_F32_SYMBOL = "tessera_apple_gpu_softmax_f32"
APPLE_SOFTMAX_DYNAMIC_F32_ABI = "tessera.apple.softmax.x_o_rows_columns.dynamic.f32.v1"
APPLE_SOFTMAX_F16_ABI = "tessera.apple.softmax.x_o_rows_columns.f16.v1"
APPLE_SOFTMAX_F16_SYMBOL = "tessera_apple_gpu_softmax_f16"
APPLE_SOFTMAX_BF16_ABI = "tessera.apple.softmax.x_o_rows_columns.bf16.v1"
APPLE_SOFTMAX_BF16_SYMBOL = "tessera_apple_gpu_softmax_bf16"
APPLE_GELU_F32_ABI = "tessera.apple.gelu.x_o_elements.f32.v1"
APPLE_GELU_F32_SYMBOL = "tessera_apple_gpu_gelu_f32"
APPLE_GELU_DYNAMIC_F32_ABI = "tessera.apple.gelu.x_o_elements.dynamic.f32.v1"
APPLE_GELU_F16_ABI = "tessera.apple.gelu.x_o_elements.f16.v1"
APPLE_GELU_F16_SYMBOL = "tessera_apple_gpu_gelu_f16"
APPLE_GELU_DYNAMIC_F16_ABI = "tessera.apple.gelu.x_o_elements.dynamic.f16.v1"
APPLE_GELU_BF16_ABI = "tessera.apple.gelu.x_o_elements.bf16.v1"
APPLE_GELU_BF16_SYMBOL = "tessera_apple_gpu_gelu_bf16"
APPLE_GELU_DYNAMIC_BF16_ABI = "tessera.apple.gelu.x_o_elements.dynamic.bf16.v1"
APPLE_POPCOUNT_DYNAMIC_I32_ABI = "tessera.apple.popcount.x_o_elements.dynamic.i32.v1"
APPLE_POPCOUNT_I32_SYMBOL = "tessera_apple_gpu_popcount_i32"
APPLE_COUNT_NONZERO_DYNAMIC_F32_I32_ABI = (
    "tessera.apple.count_nonzero.x_o_outer_axis_extent.dynamic.f32_i32.v1"
)
APPLE_COUNT_NONZERO_F32_SYMBOL = "tessera_apple_gpu_count_nonzero_lastaxis_f32"
APPLE_TOPK_DYNAMIC_F32_I32_ABI = (
    "tessera.apple.top_k.x_values_indices_rows_columns_k.dynamic.f32_i32.v1"
)
APPLE_TOPK_F32_SYMBOL = "tessera_apple_gpu_topk_f32"
APPLE_SVD_REDUCED_F32_ABI = "tessera.apple.svd.a_u_s_vh_m_n.f32.v1"
APPLE_SVD_REDUCED_F32_SYMBOL = "tessera_apple_gpu_svd_reduced_f32"
APPLE_SVD_REDUCED_BATCHED_F32_ABI = "tessera.apple.svd.a_u_s_vh_batch_m_n.f32.v1"
APPLE_SVD_REDUCED_BATCHED_F32_SYMBOL = "tessera_apple_gpu_svd_reduced_batched_f32"
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
_GELU_VARIANTS = {
    "fp32": (APPLE_GELU_F32_SYMBOL, APPLE_GELU_F32_ABI),
    "fp16": (APPLE_GELU_F16_SYMBOL, APPLE_GELU_F16_ABI),
    "bf16": (APPLE_GELU_BF16_SYMBOL, APPLE_GELU_BF16_ABI),
}
_GELU_DYNAMIC_VARIANTS = {
    "fp32": (APPLE_GELU_F32_SYMBOL, APPLE_GELU_DYNAMIC_F32_ABI),
    "fp16": (APPLE_GELU_F16_SYMBOL, APPLE_GELU_DYNAMIC_F16_ABI),
    "bf16": (APPLE_GELU_BF16_SYMBOL, APPLE_GELU_DYNAMIC_BF16_ABI),
}
_TRANSPOSE_VARIANTS = {
    "fp32": (APPLE_TRANSPOSE_F32_SYMBOL, APPLE_TRANSPOSE_F32_ABI),
    "fp16": (APPLE_TRANSPOSE_F16_SYMBOL, APPLE_TRANSPOSE_F16_ABI),
    "bf16": (APPLE_TRANSPOSE_BF16_SYMBOL, APPLE_TRANSPOSE_BF16_ABI),
}

# Apple E2E work owns an explicit descriptor disposition for every admitted
# value-lowered family.  Dedicated dynamic and multi-result contracts are
# checked before this table so a generic value symbol cannot broaden them.
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
    "tessera.svd": "descriptor_ready",
    "tessera.popcount": "descriptor_ready",
    "tessera.count_nonzero": "descriptor_ready",
    "tessera.top_k": "descriptor_ready",
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


def _dynamic_softmax_contract(module: GraphIRModule):
    """Rank-2 dynamic f32 last-axis softmax with explicit row/column scalars."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if (op.op_name != "tessera.softmax" or len(op.operands) != 1
            or len(fn.result_types) != 1 or op.kwargs.get("axis", -1) not in {-1, 1}):
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if (name not in args or args[name].ir_type.dtype != "fp32"
            or fn.result_types[0].dtype != "fp32"):
        return None
    shape = tuple(args[name].ir_type.shape)
    if shape != ("?", "?") or tuple(fn.result_types[0].shape) != shape:
        return None
    return name, op.result or "output"


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


def _gelu_contract(module: GraphIRModule):
    """Return the static rank-2 GELU ABI contract.

    The native ABI is a flattened elementwise call.  Descriptor admission keeps
    rank-2 and non-empty extents explicit so the Graph-IR shape proof and the
    numerical oracle remain aligned with the existing MSL execution envelope.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name != "tessera.gelu" or len(op.operands) != 1 or len(fn.result_types) != 1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args:
        return None
    dtype = args[name].ir_type.dtype
    if dtype not in _GELU_VARIANTS or fn.result_types[0].dtype != dtype:
        return None
    try:
        shape = tuple(int(v) for v in args[name].ir_type.shape)
        out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return None
    if len(shape) != 2 or any(extent <= 0 for extent in shape) or out_shape != shape:
        return None
    return name, op.result or "output", shape, dtype


def _dynamic_gelu_contract(module: GraphIRModule):
    """Return the explicit rank-2 dynamic GELU scalar-ABI contract.

    Dynamic extents are not inferred by the launcher: callers bind the ABI's
    ``Elements`` scalar and the runtime verifies it against the input/output
    buffers before submission.  Requiring both dimensions dynamic keeps this
    distinct from a static descriptor whose guard was merely omitted.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name != "tessera.gelu" or len(op.operands) != 1 or len(fn.result_types) != 1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args:
        return None
    dtype = args[name].ir_type.dtype
    if dtype not in _GELU_DYNAMIC_VARIANTS or fn.result_types[0].dtype != dtype:
        return None
    shape = tuple(args[name].ir_type.shape)
    out_shape = tuple(fn.result_types[0].shape)
    if shape != ("?", "?") or out_shape != shape:
        return None
    return name, op.result or "output", dtype


def _dynamic_popcount_contract(module: GraphIRModule):
    """Rank-1 dynamic i32 popcount with explicit ``Elements`` ABI scalar."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name != "tessera.popcount" or len(op.operands) != 1 or len(fn.result_types) != 1:
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if (name not in args or args[name].ir_type.dtype != "int32"
            or fn.result_types[0].dtype != "int32"):
        return None
    shape, out_shape = tuple(args[name].ir_type.shape), tuple(fn.result_types[0].shape)
    if shape != ("?",) or out_shape != shape:
        return None
    return name, op.result or "output"


def _dynamic_count_nonzero_contract(module: GraphIRModule):
    """Rank-2 dynamic f32 last-axis reduction with explicit shape scalars."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if (op.op_name != "tessera.count_nonzero" or len(op.operands) != 1
            or len(fn.result_types) != 1 or op.kwargs.get("axis", -1) not in {-1, 1}
            or bool(op.kwargs.get("keepdims", False))):
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if (name not in args or args[name].ir_type.dtype != "fp32"
            or fn.result_types[0].dtype != "int32"):
        return None
    if (tuple(args[name].ir_type.shape) != ("?", "?")
            or tuple(fn.result_types[0].shape) != ("?",)):
        return None
    return name, op.result or "output"


def _dynamic_topk_contract(module: GraphIRModule):
    """Ordered rank-2 f32 top-k with static K and dynamic row/axis extents."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if (op.op_name != "tessera.top_k" or len(op.operands) != 1
            or len(fn.result_types) != 2 or len(fn.return_values) != 2
            or op.kwargs.get("axis", -1) not in {-1, 1}):
        return None
    try:
        k = int(op.kwargs["k"])
    except (KeyError, TypeError, ValueError):
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if (name not in args or args[name].ir_type.dtype != "fp32" or k <= 0
            or tuple(args[name].ir_type.shape) != ("?", "?")
            or fn.result_types[0].dtype != "fp32"
            or fn.result_types[1].dtype != "int32"):
        return None
    expected_shape = ("?", str(k))
    if any(tuple(result.shape) != expected_shape for result in fn.result_types):
        return None
    outputs = tuple(value.removeprefix("%") for value in fn.return_values)
    if len(set(outputs)) != 2:
        return None
    return name, outputs, k


def _svd_contract(module: GraphIRModule):
    """Return one reduced f32 SVD contract with explicitly ordered outputs.

    The GPU Jacobi implementation factors only tall matrices.  The adapter ABI
    owns descending singular-value order and converts its internal right-vector
    columns into Graph IR's ``Vh`` result, so this contract must not admit wide
    matrices or full-matrix SVD under a reduced descriptor name.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    if (op.op_name != "tessera.svd" or len(op.operands) != 1
            or len(fn.result_types) != 3 or len(fn.return_values) != 3
            or bool(op.kwargs.get("full_matrices", False))):
        return None
    name = op.operands[0].removeprefix("%")
    args = {arg.name: arg for arg in fn.args}
    if name not in args or args[name].ir_type.dtype != "fp32":
        return None
    if any(result.dtype != "fp32" for result in fn.result_types):
        return None
    try:
        shape = tuple(int(v) for v in args[name].ir_type.shape)
        output_shapes = tuple(tuple(int(v) for v in result.shape) for result in fn.result_types)
    except (TypeError, ValueError):
        return None
    if len(shape) not in {2, 3} or any(extent <= 0 for extent in shape):
        return None
    batch = 1 if len(shape) == 2 else shape[0]
    m, n = shape[-2:]
    if m < n:
        return None
    expected = ((m, n), (n,), (n, n)) if len(shape) == 2 else (
        (batch, m, n), (batch, n), (batch, n, n)
    )
    if output_shapes != expected:
        return None
    return name, tuple(value.removeprefix("%") for value in fn.return_values), shape

def supports_native_package(module: GraphIRModule) -> bool:
    if _contract(module) is not None:
        return True
    if _softmax_contract(module) is not None:
        return True
    if _dynamic_softmax_contract(module) is not None:
        return True
    if _transpose_contract(module) is not None:
        return True
    if _gelu_contract(module) is not None:
        return True
    if _dynamic_gelu_contract(module) is not None:
        return True
    if _dynamic_popcount_contract(module) is not None:
        return True
    if _dynamic_count_nonzero_contract(module) is not None:
        return True
    if _dynamic_topk_contract(module) is not None:
        return True
    if _svd_contract(module) is not None:
        return True
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name in {
        "tessera.batched_gemm", "tessera.matmul", "tessera.gemm", "tessera.softmax",
        "tessera.transpose", "tessera.gelu", "tessera.popcount",
        "tessera.count_nonzero", "tessera.top_k", "tessera.svd",
    }:
        return False
    # Multi-result families are descriptor-supported only through a dedicated
    # ordered-binding contract above; the generic value path has one output.
    return len(fn.result_types) == 1 and value_descriptor_state(op.op_name) == "descriptor_ready"


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
    if _dynamic_softmax_contract(module) is not None:
        return package_dynamic_softmax(module, pipeline_name=pipeline_name)
    if _transpose_contract(module) is not None:
        return package_transpose(module, pipeline_name=pipeline_name)
    if _gelu_contract(module) is not None:
        return package_gelu(module, pipeline_name=pipeline_name)
    if _dynamic_gelu_contract(module) is not None:
        return package_dynamic_gelu(module, pipeline_name=pipeline_name)
    if _dynamic_popcount_contract(module) is not None:
        return package_dynamic_popcount(module, pipeline_name=pipeline_name)
    if _dynamic_count_nonzero_contract(module) is not None:
        return package_dynamic_count_nonzero(module, pipeline_name=pipeline_name)
    if _dynamic_topk_contract(module) is not None:
        return package_dynamic_topk(module, pipeline_name=pipeline_name)
    if _svd_contract(module) is not None:
        return package_svd(module, pipeline_name=pipeline_name)
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
        provenance={"work_item": "APPLE-E2E-1", "route": "mps_bmm_native_library",
                    "op_kind": "batched_gemm", "shape": [batch,m,n,k],
                    "broadcast_b": broadcast, "storage": dtype},
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
                    "op_kind": "softmax", "shape": [rows, columns], "storage": dtype},
    )
    return AppleNativePackage("tile.softmax_kernel", target_ir, target_ir, image, descriptor)


def package_dynamic_softmax(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package dynamic rank-2 f32 row-softmax with verified shape scalars."""
    contract = _dynamic_softmax_contract(module)
    if contract is None:
        raise ValueError("Apple dynamic softmax requires rank-2 ?x? f32 last-axis input and output")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, out = contract
    target_ir = (f'tessera_apple.gpu.kernel_call @{APPLE_SOFTMAX_F32_SYMBOL} '
                 f'{{abi = "{APPLE_SOFTMAX_DYNAMIC_F32_ABI}", '
                 'scalars = "Rows,Columns", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload,
        entry_points=(NativeEntryPoint(APPLE_SOFTMAX_F32_SYMBOL, APPLE_SOFTMAX_DYNAMIC_F32_ABI),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=APPLE_SOFTMAX_F32_SYMBOL,
        abi_id=APPLE_SOFTMAX_DYNAMIC_F32_ABI,
        buffers=(BufferBinding(0, x, "input", "fp32", 2, "row_major", 4),
                 BufferBinding(1, out, "output", "fp32", 2, "row_major", 4)),
        scalars=(ScalarArgument(2, "Rows", "int32"), ScalarArgument(3, "Columns", "int32")),
        geometry=LaunchGeometry(policy="apple_msl_softmax_dynamic"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("return",),
        ),
        provenance={"work_item": "APPLE-NATIVE-E2E-2",
                    "route": "apple_softmax_native_library", "op_kind": "softmax",
                    "dynamic_shape": True,
                    "scalar_contract": "Rows=x.shape[0],Columns=x.shape[1]",
                    "storage": "fp32"},
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
                    "op_kind": "transpose", "shape": list(shape), "axes": list(axes),
                    "storage": dtype},
    )
    return AppleNativePackage("tile.transpose_kernel", target_ir, target_ir, image, descriptor)


def package_gelu(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package one static rank-2 GELU MSL ABI."""
    contract = _gelu_contract(module)
    if contract is None:
        raise ValueError("Apple GELU package requires one static non-empty rank-2 f32/f16/bf16 contract")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, out, shape, dtype = contract
    symbol, abi = _GELU_VARIANTS[dtype]
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", storage = "{dtype}", accumulation = "fp32", '
                 'status = "executable"}}')
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
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(BufferBinding(0, x, "input", dtype, 2, "row_major", alignment),
                 BufferBinding(1, out, "output", dtype, 2, "row_major", alignment)),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name, guarded_shape in ((x, shape), (out, shape))
            for axis, extent in enumerate(guarded_shape)
        ),
        geometry=LaunchGeometry(policy="apple_msl_gelu"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-NATIVE-E2E-2", "route": "apple_gelu_native_library",
                    "op_kind": "gelu", "shape": list(shape), "storage": dtype,
                    "accumulation": "fp32"},
    )
    return AppleNativePackage("tile.gelu_kernel", target_ir, target_ir, image, descriptor)


def package_dynamic_gelu(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package dynamic rank-2 GELU with an explicit ``Elements`` scalar."""
    contract = _dynamic_gelu_contract(module)
    if contract is None:
        raise ValueError("Apple dynamic GELU package requires rank-2 ?x? f32/f16/bf16 input and output")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, out, dtype = contract
    symbol, abi = _GELU_DYNAMIC_VARIANTS[dtype]
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", storage = "{dtype}", accumulation = "fp32", '
                 'scalar = "Elements", status = "executable"}}')
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
        scalars=(ScalarArgument(2, "Elements", "int64"),),
        geometry=LaunchGeometry(policy="apple_msl_gelu_dynamic"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-NATIVE-E2E-2", "route": "apple_gelu_native_library",
                    "op_kind": "gelu", "dynamic_shape": True, "scalar_contract": "Elements=x.size",
                    "storage": dtype, "accumulation": "fp32"},
    )
    return AppleNativePackage("tile.gelu_kernel", target_ir, target_ir, image, descriptor)


def package_dynamic_popcount(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package rank-1 dynamic i32 popcount with scalar-verified extent."""
    contract = _dynamic_popcount_contract(module)
    if contract is None:
        raise ValueError("Apple dynamic popcount requires rank-1 ? i32 input and output")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, out = contract
    target_ir = (f'tessera_apple.gpu.kernel_call @{APPLE_POPCOUNT_I32_SYMBOL} '
                 f'{{abi = "{APPLE_POPCOUNT_DYNAMIC_I32_ABI}", scalar = "Elements", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(APPLE_POPCOUNT_I32_SYMBOL, APPLE_POPCOUNT_DYNAMIC_I32_ABI),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=APPLE_POPCOUNT_I32_SYMBOL,
        abi_id=APPLE_POPCOUNT_DYNAMIC_I32_ABI,
        buffers=(BufferBinding(0, x, "input", "int32", 1, "row_major", 4),
                 BufferBinding(1, out, "output", "int32", 1, "row_major", 4)),
        scalars=(ScalarArgument(2, "Elements", "int32"),),
        geometry=LaunchGeometry(policy="apple_msl_popcount_dynamic"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-NATIVE-E2E-2", "route": "apple_popcount_native_library",
                    "op_kind": "popcount", "dynamic_shape": True, "scalar_contract": "Elements=x.size"},
    )
    return AppleNativePackage("tile.popcount_kernel", target_ir, target_ir, image, descriptor)


def package_dynamic_count_nonzero(
    module: GraphIRModule, *, pipeline_name: str,
) -> AppleNativePackage:
    """Package dynamic rank-2 f32 last-axis count-nonzero reduction."""
    contract = _dynamic_count_nonzero_contract(module)
    if contract is None:
        raise ValueError(
            "Apple dynamic count_nonzero requires rank-2 ?x? f32 input, rank-1 ? i32 "
            "output, axis=-1, and keepdims=False"
        )
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, out = contract
    target_ir = (f'tessera_apple.gpu.kernel_call @{APPLE_COUNT_NONZERO_F32_SYMBOL} '
                 f'{{abi = "{APPLE_COUNT_NONZERO_DYNAMIC_F32_I32_ABI}", '
                 'scalars = "Outer,AxisExtent", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload,
        entry_points=(NativeEntryPoint(
            APPLE_COUNT_NONZERO_F32_SYMBOL, APPLE_COUNT_NONZERO_DYNAMIC_F32_I32_ABI,
        ),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=APPLE_COUNT_NONZERO_F32_SYMBOL,
        abi_id=APPLE_COUNT_NONZERO_DYNAMIC_F32_I32_ABI,
        buffers=(BufferBinding(0, x, "input", "fp32", 2, "row_major", 4),
                 BufferBinding(1, out, "output", "int32", 1, "row_major", 4)),
        scalars=(ScalarArgument(2, "Outer", "int32"),
                 ScalarArgument(3, "AxisExtent", "int32")),
        geometry=LaunchGeometry(policy="apple_msl_count_nonzero_lastaxis_dynamic"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("return",),
        ),
        provenance={"work_item": "APPLE-NATIVE-E2E-2",
                    "route": "apple_count_nonzero_native_library",
                    "op_kind": "count_nonzero", "axis": -1, "keepdims": False,
                    "dynamic_shape": True,
                    "scalar_contract": "Outer=x.shape[0],AxisExtent=x.shape[1]"},
    )
    return AppleNativePackage("tile.count_nonzero_kernel", target_ir, target_ir, image, descriptor)


def package_dynamic_topk(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package deterministic ordered top-k with explicit output bindings."""
    contract = _dynamic_topk_contract(module)
    if contract is None:
        raise ValueError(
            "Apple dynamic top_k requires rank-2 ?x? f32 input, ordered rank-2 "
            "f32/i32 outputs with static K, and last-axis selection"
        )
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, (values, indices), k = contract
    target_ir = (f'tessera_apple.gpu.kernel_call @{APPLE_TOPK_F32_SYMBOL} '
                 f'{{abi = "{APPLE_TOPK_DYNAMIC_F32_I32_ABI}", '
                 'result_order = "values,indices", scalars = "Rows,Columns,K", '
                 'ordering = "descending,nan_last,lower_index_ties", '
                 'status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload,
        entry_points=(NativeEntryPoint(APPLE_TOPK_F32_SYMBOL, APPLE_TOPK_DYNAMIC_F32_I32_ABI),),
        compile_state="prepackaged",
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=APPLE_TOPK_F32_SYMBOL,
        abi_id=APPLE_TOPK_DYNAMIC_F32_I32_ABI,
        buffers=(BufferBinding(0, x, "input", "fp32", 2, "row_major", 4),
                 BufferBinding(1, values, "output", "fp32", 2, "row_major", 4),
                 BufferBinding(2, indices, "output", "int32", 2, "row_major", 4)),
        scalars=(ScalarArgument(3, "Rows", "int32"),
                 ScalarArgument(4, "Columns", "int32"),
                 ScalarArgument(5, "K", "int32")),
        geometry=LaunchGeometry(policy="apple_msl_topk_deterministic_dynamic"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("return",),
        ),
        provenance={"work_item": "APPLE-NATIVE-E2E-2",
                    "route": "apple_topk_deterministic_native_library",
                    "op_kind": "top_k", "dynamic_shape": True, "axis": -1,
                    "result_order": ["values", "indices"], "k": k,
                    "value_order": "descending", "nan_policy": "last",
                    "tie_policy": "lower_index",
                    "scalar_contract": "Rows=x.shape[0],Columns=x.shape[1],K=graph.k"},
    )
    return AppleNativePackage("tile.top_k_kernel", target_ir, target_ir, image, descriptor)


def package_svd(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Package static reduced f32 SVD with ordered ``(U, S, Vh)`` bindings."""
    contract = _svd_contract(module)
    if contract is None:
        raise ValueError("Apple GPU SVD package requires static tall reduced f32 (U, S, Vh) contract")
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    x, outputs, shape = contract
    is_batched = len(shape) == 3
    batch = 1 if not is_batched else shape[0]
    m, n = shape[-2:]
    symbol = APPLE_SVD_REDUCED_BATCHED_F32_SYMBOL if is_batched else APPLE_SVD_REDUCED_F32_SYMBOL
    abi = APPLE_SVD_REDUCED_BATCHED_F32_ABI if is_batched else APPLE_SVD_REDUCED_F32_ABI
    output_shapes = ((m, n), (n,), (n, n)) if not is_batched else (
        (batch, m, n), (batch, n), (batch, n, n)
    )
    target_ir = (f'tessera_apple.gpu.kernel_call @{symbol} '
                 f'{{abi = "{abi}", result_order = "u,s,vh", status = "executable"}}')
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )
    buffers = (
        BufferBinding(0, x, "input", "fp32", len(shape), "row_major", 4),
        *(BufferBinding(index + 1, name, "output", "fp32", len(output_shapes[index]), "row_major", 4)
          for index, name in enumerate(outputs)),
    )
    scalars = ((ScalarArgument(4, "M", "int32"), ScalarArgument(5, "N", "int32"))
               if not is_batched else
               (ScalarArgument(4, "Batch", "int32"), ScalarArgument(5, "M", "int32"),
                ScalarArgument(6, "N", "int32")))
    guards = tuple(
        ShapeGuard(name, axis, "eq", extent)
        for name, guarded_shape in ((x, shape), *zip(outputs, output_shapes))
        for axis, extent in enumerate(guarded_shape)
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=buffers, scalars=scalars, shape_guards=guards,
        geometry=LaunchGeometry(policy="apple_gpu_svd_reduced"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-NATIVE-E2E-2", "route": "apple_gpu_svd_reduced_adapter",
                    "op_kind": "svd", "result_order": ["u", "s", "vh"],
                    "full_matrices": False, "tall_only": True, "batch": batch},
    )
    return AppleNativePackage("apple.svd_reduced", target_ir, target_ir, image, descriptor)
