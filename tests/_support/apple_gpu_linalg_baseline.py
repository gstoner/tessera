"""Frozen Graph-owned Apple GPU linalg descriptor constructors (Decision #31(a) oracle).

E2E-REAL-6 Apple GPU linalg (2026-11-04). Before this slice,
``apple_native.package_svd`` and the generic value branch of
``apple_native.package_native`` read the Python Graph object directly to build
the launch descriptor for reduced ``svd`` and for ``cholesky``, ``tri_solve``
and ``cholesky_solve``. Those families now lower through the native
Graph -> Schedule -> Tile -> Target boundaries (``NativeAppleGPUKernel.h``) and
package the replayed artifact
(``apple_native.package_scheduled_apple_gpu_kernel``). This module freezes the
retired constructors, for exactly those families, so
``tests/unit/test_apple_gpu_kernel_migration.py`` can prove the compiled route
reproduces them. It is test-only: nothing under ``python/`` imports it.
"""

from __future__ import annotations

import hashlib

from tessera.compiler import apple_native
from tessera.compiler.apple_native import (
    APPLE_SVD_REDUCED_BATCHED_F32_ABI, APPLE_SVD_REDUCED_BATCHED_F32_SYMBOL,
    APPLE_SVD_REDUCED_F32_ABI, APPLE_SVD_REDUCED_F32_SYMBOL, AppleNativePackage,
)
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)

#: The retired generic value-branch symbols for these three families.
_VALUE_SYMBOLS = {
    "tessera.cholesky": ("cholesky", "tessera_apple_gpu_cholesky_f32"),
    "tessera.tri_solve": ("tri_solve", "tessera_apple_gpu_tri_solve_f32"),
    "tessera.cholesky_solve": ("cholesky_solve", "tessera_apple_gpu_solve_cholesky_f32"),
}


def retired_svd_contract(module: GraphIRModule):
    """The retired ``_svd_contract``: one reduced tall f32 SVD, ordered outputs."""
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


def retired_value_contract(module: GraphIRModule) -> bool:
    """The retired generic value branch's admission for these three families."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name not in _VALUE_SYMBOLS or len(fn.result_types) != 1:
        return False
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if (not names or any(name not in args or args[name].ir_type.dtype != "fp32" for name in names)
            or fn.result_types[0].dtype != "fp32"):
        return False
    try:
        [tuple(int(v) for v in args[name].ir_type.shape) for name in names]
        tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return False
    return True


def _image(target_ir: str, symbol: str, abi: str, pipeline_name: str) -> NativeImageArtifact:
    library = apple_native._runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return NativeImageArtifact(
        target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )


def package_value(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """The retired generic value branch of ``package_native``, for these families."""
    if not retired_value_contract(module):
        raise ValueError("Apple value descriptors require static f32 tensor operands/results")
    fn, op = module.functions[0], module.functions[0].body[0]
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    shapes = {name: tuple(int(v) for v in args[name].ir_type.shape) for name in names}
    out_shape = tuple(int(v) for v in fn.result_types[0].shape)
    kind, symbol = _VALUE_SYMBOLS[op.op_name]
    value_kwargs = dict(op.kwargs)
    abi = f"tessera.apple.value.{kind}.f32.v1"
    out = op.result or "output"
    target_ir = f'tessera_apple.gpu.kernel_call @{symbol} {{abi = "{abi}", status = "executable"}}'
    image = _image(target_ir, symbol, abi, pipeline_name)
    bindings = tuple(BufferBinding(i, name, "input", "fp32", len(shapes[name]), "row_major", 4)
                     for i, name in enumerate(names)) + (
        BufferBinding(len(names), out, "output", "fp32", len(out_shape), "row_major", 4),)
    guards = tuple(ShapeGuard(name, axis, "eq", extent)
                   for name, shape in list(shapes.items()) + [(out, out_shape)]
                   for axis, extent in enumerate(shape))
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi, buffers=bindings,
        shape_guards=guards, geometry=LaunchGeometry(policy="apple_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-E2E-1", "route": "apple_value_executor", "op_kind": kind,
                    "value_call": {"op": "tessera_apple.gpu.kernel_call", "op_kind": kind,
                                   "symbol": symbol, "status": "executable", **value_kwargs}})
    return AppleNativePackage("apple.value", target_ir, target_ir, image, descriptor)


def package_svd(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """The retired ``apple_native.package_svd``."""
    contract = retired_svd_contract(module)
    if contract is None:
        raise ValueError("Apple GPU SVD package requires static tall reduced f32 (U, S, Vh) contract")
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
    image = _image(target_ir, symbol, abi, pipeline_name)
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


def package(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """Dispatch to the retired constructor for one of the four families."""
    if module.functions and module.functions[0].body and module.functions[0].body[0].op_name == "tessera.svd":
        return package_svd(module, pipeline_name=pipeline_name)
    return package_value(module, pipeline_name=pipeline_name)
