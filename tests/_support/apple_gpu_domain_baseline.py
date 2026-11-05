"""Frozen Graph-owned Apple GPU domain descriptor constructor (Decision #31(a) oracle).

E2E-REAL-6 Apple GPU domain families (2026-11-04). Before this slice the
generic value branch of ``apple_native.package_native`` read the Python Graph
object directly to build the launch descriptor for
``tessera.rl.ppo_policy_loss``, ``tessera.ebm.{energy_quadratic,
langevin_step, refinement, partition_exact}`` and
``tessera.clifford.geometric_product``. Those families now lower through the
native Graph -> Schedule -> Tile boundaries (``NativeAppleGPUDomain.h``) and the
Apple GPU value lowering, and package the replayed artifact
(``apple_native.package_scheduled_apple_gpu_domain``). This module freezes the
retired constructor verbatim (main ``4606fc710``), for exactly those families,
so ``tests/unit/test_apple_gpu_domain_migration.py`` can prove the compiled
route reproduces it. It is test-only: nothing under ``python/`` imports it.
"""

from __future__ import annotations

import hashlib

from tessera.compiler import apple_native
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ShapeGuard,
)

_VALUE_SYMBOLS = {
    "tessera.rl.ppo_policy_loss": ("ppo_policy_loss", "tessera_apple_gpu_ppo_policy_loss_f32"),
    "tessera.ebm.energy_quadratic": ("ebm_energy_quadratic", "tessera_apple_gpu_ebm_energy_quadratic_value_f32"),
    "tessera.ebm.langevin_step": ("ebm_langevin_step", "tessera_apple_gpu_ebm_langevin_step_value_f32"),
    "tessera.ebm.refinement": ("ebm_refinement", "tessera_apple_gpu_ebm_refinement_value_f32"),
    "tessera.ebm.partition_exact": ("ebm_partition_exact", "tessera_apple_gpu_ebm_partition_exact_value_f32"),
    "tessera.clifford.geometric_product": ("clifford_geometric_product", "tessera_apple_gpu_clifford_geo_product_cl30_value_f32"),
}


def retired_contract(module: GraphIRModule) -> bool:
    """The envelope the retired classifier admitted for these families: one
    Graph function holding one of these ops with a single result."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn, op = module.functions[0], module.functions[0].body[0]
    return op.op_name in _VALUE_SYMBOLS and len(fn.result_types) == 1


def package(module: GraphIRModule, *, pipeline_name: str) -> apple_native.AppleNativePackage:
    """The retired generic value branch of ``package_native``, unchanged."""
    if not retired_contract(module):
        raise ValueError("the retired Apple GPU value constructor did not admit this module")
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
    library = apple_native._runtime_library_path()
    if library is None: raise RuntimeError("APPLE-E2E-1 requires a fresh Tessera Apple GPU runtime dylib")
    payload = library.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(target="apple_gpu", architecture="apple_gpu", pipeline_name=pipeline_name, compiler_fingerprint="apple-runtime-abi-v1", toolchain_fingerprint=hashlib.sha256(("apple_gpu|" + digest).encode()).hexdigest(), target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object", payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged")
    bindings = tuple(BufferBinding(i, name, "input", "fp32", len(shapes[name]), "row_major", 4) for i, name in enumerate(names)) + (BufferBinding(len(names), out, "output", "fp32", len(out_shape), "row_major", 4),)
    guards = tuple(ShapeGuard(name, axis, "eq", extent) for name, shape in list(shapes.items()) + [(out, out_shape)] for axis, extent in enumerate(shape))
    descriptor = LaunchDescriptor(image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi, buffers=bindings, shape_guards=guards, geometry=LaunchGeometry(policy="apple_value_executor"), ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)), provenance={"work_item":"APPLE-E2E-1", "route":"apple_value_executor", "op_kind":kind, "value_call":{"op":"tessera_apple.gpu.kernel_call", "op_kind":kind, "symbol":symbol, "status":"executable", **value_kwargs}})
    return apple_native.AppleNativePackage("apple.value", target_ir, target_ir, image, descriptor)
