"""Frozen pre-E2E-REAL-6 Apple GPU static GELU packaging (differential tests only).

This is the static half of ``apple_native.lower_gelu_artifact`` /
``package_gelu_artifact`` as it stood on main ``b498bc6f0``: it read the
Python Graph object, ran ``--tessera-gelu-to-apple_gpu`` (Graph straight to
the Target ``func.call`` of the runtime MSL kernel, bypassing Schedule and
Tile) and built the launch descriptor from that text. Production static GELU
now lowers through the native Graph -> Schedule -> Tile activation route and
packages the replayed ``ScheduledKernelArtifact``
(``apple_native.package_scheduled_kernel``); this module is the **declared
oracle** Decision #31(a) allows. Its only consumers are the differential tests
in ``tests/unit/test_apple_gelu_migration.py``.

The dynamic GELU route still uses the production functions; only the static
admission was removed from them, so the copy below keeps it verbatim.
"""
from __future__ import annotations

import copy
import hashlib
import re

from tessera.compiler.apple_native import (
    _GELU_DYNAMIC_VARIANTS,
    _GELU_VARIANTS,
    AppleGeluArtifact,
    AppleNativePackage,
    _gelu_contract,
)
from tessera.compiler import apple_native
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def lower_gelu_artifact(module: GraphIRModule) -> AppleGeluArtifact:
    contract = _gelu_contract(module)
    if contract is None:
        raise ValueError("Apple GELU requires a rank-two contract")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("Apple GELU requires native lowering")
    targeted = copy.deepcopy(module)
    targeted.module_attrs['tessera.target'] = '"apple_gpu"'
    targeted.module_attrs['tessera.arch'] = '"apple7"'
    parent = run_tessera_opt(tool, targeted.to_mlir(canonical=True), '--canonicalize')
    native = run_tessera_opt(tool, parent, '--tessera-gelu-to-apple_gpu')
    return AppleGeluArtifact(parent, native, contract[0], contract[1])


def package_gelu_artifact(artifact: AppleGeluArtifact, *, pipeline_name: str) -> AppleNativePackage:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("Apple GELU requires native replay")
    parent = run_tessera_opt(tool, artifact.parent_ir, '--canonicalize')
    native = run_tessera_opt(tool, parent, '--tessera-gelu-to-apple_gpu')
    if native != artifact.native_ir:
        raise ValueError("Apple GELU native product disagrees with parent replay")
    tensor = r'tensor<(\?|[1-9][0-9]*)x(\?|[1-9][0-9]*)x(f32|f16|bf16)>'
    match = re.fullmatch(
        r'\s*module attributes \{tessera.arch = "apple7", (?:tessera.ir.version = "1.0", )?tessera.target = "apple_gpu"\} \{\s*'
        r'func.func @\w+\(%(\w+): ' + tensor
        + r'(?: \{tessera.dim_names = \["[A-Za-z_0-9]+", "[A-Za-z_0-9]+"\]\})?'
        + r'\) -> (tensor<[^>]+>) \{\s*'
        r'%(\w+) = tessera.gelu %(\w+)(?: \{tessera.effect_kind = "pure"\})? : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)\s*'
        r'return %(\w+) : (tensor<[^>]+>)\s*\}\s*\}\s*', parent)
    if match is None:
        raise ValueError("Apple GELU descriptor requires one native rank-two GELU parent")
    arg, m, k, storage, result, value, operand, in_ty, out_ty, returned, ret_ty = match.groups()
    expected_ty = f'tensor<{m}x{k}x{storage}>'
    if operand != arg or returned != value or any(t != expected_ty for t in (result, in_ty, out_ty, ret_ty)):
        raise ValueError("Apple GELU native tensor/SSA contract disagrees")
    dtype = {'f32': 'fp32', 'f16': 'fp16', 'bf16': 'bf16'}[storage]
    dynamic = m == "?" or k == "?"
    shape = tuple(None if d == "?" else int(d) for d in (m, k))
    x, out = artifact.input_name, artifact.output_name
    if any(type(n) is not str or not n.isidentifier() for n in (x, out)) or x == out:
        raise ValueError("Apple GELU host aliases must be distinct identifiers")
    symbol, abi = (_GELU_DYNAMIC_VARIANTS if dynamic else _GELU_VARIANTS)[dtype]
    if (f'call @{symbol}(' not in native or 'cf.assert ' not in native
            or 'tessera.gelu ' in native):
        raise ValueError("Apple GELU native producer did not lower its status boundary")
    # Looked up through the module so a test's runtime fixture reaches both routes.
    library = apple_native._runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-NATIVE-E2E-2 requires a fresh Tessera Apple GPU runtime dylib")
    target_ir = native
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
            for axis, extent in enumerate(guarded_shape) if extent is not None
        ),
        scalars=(ScalarArgument(2, "Elements", "int64"),) if dynamic else (),
        geometry=LaunchGeometry(policy="apple_msl_gelu_dynamic" if dynamic else "apple_msl_gelu"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": "APPLE-NATIVE-E2E-2", "route": "apple_gelu_native_library",
                    "op_kind": "gelu", "shape": list(shape), "storage": dtype,
                    "dynamic_shape": dynamic, "scalar_contract": "Elements=x.size" if dynamic else None,
                    "accumulation": "fp32",
                    "parent_ir_digest": hashlib.sha256(parent.encode()).hexdigest(),
                    "native_ir_digest": hashlib.sha256(native.encode()).hexdigest()},
    )
    return AppleNativePackage(native, target_ir, target_ir, image, descriptor)


def package_gelu(module: GraphIRModule, *, pipeline_name: str) -> AppleNativePackage:
    """The retired production ``apple_native.package_gelu`` (static half)."""
    return package_gelu_artifact(lower_gelu_artifact(module), pipeline_name=pipeline_name)
