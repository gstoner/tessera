"""Retired Graph-authored SM120 attention packaging, kept only as a #31(a) oracle.

E2E-REAL-6 (sync ``E2E-REAL-6-sm120-attention-2026-11-03``). Production
``nvidia_native.package_scheduled_checkpoint`` and
``package_scheduled_attention_recompute`` read every launch field from a
validated replayed artifact. Before that cut the Graph object was the author:

* the saved-LSE checkpoints (``attention_lse`` / ``attention_backward_lse``)
  took names, dims, scale, causal and bias from the Graph admission tuple and
  synthesized a fresh f32 checkpoint Graph from them
  (``scheduled_checkpoint.lower_scheduled_checkpoint``, as at 8eee7d584^);
* the recompute backward (``attention_backward``) built its entry symbol,
  Tile IR (the retired Python constructor in ``nvidia_attention_baseline``)
  and descriptor from the Graph admission tuple.

Typed-storage and scaled FP8 K/V forwards never had a Graph-authored packager,
so :func:`graph_authored_forward_launch` derives their launch contract (buffer
order, dtype, rank, alignment, shape guards and scalar names) from the Graph
function signature alone, independent of any native contract.
"""

from __future__ import annotations

import hashlib

from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint, NativeImageArtifact,
    OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard, WorkspaceRequirement,
)
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_checkpoint import lower_scheduled_checkpoint
from tests._support.nvidia_attention_baseline import emit_attention_backward_tile_ir

_DIMS = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv")
_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}


def _graph_bias_shape(module: GraphIRModule, name: str | None) -> tuple[int, ...] | None:
    return None if name is None else native._shape(module, name)


def baseline_attention_lse(module: GraphIRModule, *, pipeline_name: str) -> native.NVIDIANativePackage:
    """The f32 saved-LSE forward, authored from the Graph admission tuple."""
    contract = native._attention_lse_contract(module)
    if contract is None:
        raise ValueError("retired saved-LSE forward requires the canonical paired ABI")
    names, dims, scale, causal = contract
    bias = len(names) == 6
    return native.package_scheduled_checkpoint(lower_scheduled_checkpoint(
        names, dims, scale, causal, backward=False, bias=bias,
        bias_shape=_graph_bias_shape(module, names[3] if bias else None)),
        pipeline_name=pipeline_name)


def baseline_attention_backward_lse(module: GraphIRModule, *, pipeline_name: str) -> native.NVIDIANativePackage:
    """The f32 saved-LSE backward, authored from the Graph admission tuple."""
    contract = native._attention_backward_lse_contract(module)
    if contract is None:
        raise ValueError("retired saved-LSE backward requires the canonical paired ABI")
    names, dims, scale, causal = contract
    bias = len(names) == 10
    return native.package_scheduled_checkpoint(lower_scheduled_checkpoint(
        names, dims, scale, causal, backward=True, bias=bias,
        bias_shape=_graph_bias_shape(module, names[5] if bias else None)),
        pipeline_name=pipeline_name)


def baseline_attention_backward(module: GraphIRModule, *, pipeline_name: str) -> native.NVIDIANativePackage:
    """The recompute VJP as packaged at 8eee7d584^: Graph tuple -> Python Tile IR."""
    contract = native._attention_backward_contract(module)
    if contract is None:
        raise ValueError("retired recompute backward requires the canonical contract")
    (storage, names, dims, scale, causal, bias_name, window_left, window_right,
     softcap, dropout_p, dropout_seed) = contract
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    do_name, q_name, k_name, v_name = names
    b, hq, hkv, sq, sk, d, dv = dims
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:{window_right}:"
        f"{softcap:.17g}:{dropout_p:.17g}:{dropout_seed}:deterministic_direct".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_attention_backward_{storage_ir}_deterministic_{semantic_key}"
    abi_id = {
        "fp16": (native.SM120_ATTN_BWD_F16_ABI, native.SM120_ATTN_BWD_BIAS_F16_ABI),
        "bf16": (native.SM120_ATTN_BWD_BF16_ABI, native.SM120_ATTN_BWD_BIAS_BF16_ABI),
        "fp32": (native.SM120_ATTN_BWD_F32_ABI, native.SM120_ATTN_BWD_BIAS_F32_ABI),
    }[storage][bias_name is not None]
    tile_ir = emit_attention_backward_tile_ir(
        entry=entry, storage=storage_ir, scale=scale, causal=causal, bias=bias_name is not None,
        window_left=window_left, window_right=window_right, softcap=softcap,
        dropout_p=dropout_p, dropout_seed=dropout_seed,
    )
    lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state = (
        native._compile_tile_ir(tile_ir, entry))
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp, toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),), compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    dq_name, dk_name, dv_name = module.functions[0].body[0].result_names
    width = _BYTES[storage]
    roles = [(do_name, "input", storage, (b, hq, sq, dv)), (q_name, "input", storage, (b, hq, sq, d)),
             (k_name, "input", storage, (b, hkv, sk, d)), (v_name, "input", storage, (b, hkv, sk, dv))]
    if bias_name:
        roles.append((bias_name, "input", "fp32", (b, hq, sq, sk)))
    roles += [(dq_name, "output", storage, (b, hq, sq, d)), (dk_name, "output", storage, (b, hkv, sk, d)),
              (dv_name, "output", storage, (b, hkv, sk, dv))]
    buffers = tuple(BufferBinding(i, name, direction, dtype, 4, "row_major", 4 if dtype == "fp32" else width)
                    for i, (name, direction, dtype, _shape) in enumerate(roles))
    # The retired packager listed bias guards last; order is not semantic.
    guards = tuple(ShapeGuard(name, axis, "eq", extent)
                   for name, _, _, shape in sorted(roles, key=lambda role: role[0] == bias_name)
                   for axis, extent in enumerate(shape))
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=abi_id, buffers=buffers,
        scalars=tuple(ScalarArgument(len(buffers) + i, name, "int64") for i, name in enumerate(_DIMS)),
        shape_guards=guards,
        geometry=LaunchGeometry(policy="sm120_attention_backward_deterministic_direct_128"),
        workspace=WorkspaceRequirement(bytes=0, alignment=4),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
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
    return native.NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def graph_authored_forward_launch(module: GraphIRModule) -> dict[str, object]:
    """Buffers, guards and scalar names a Graph-side saved-LSE forward would author.

    Read only from the Graph signature: inputs in ``tessera.flash_attn`` operand
    order (the scaled chain contributes its codes then, after any bias, its
    scales), then results in return order. No native contract is consulted.
    """
    fn = module.functions[0]
    args = {arg.name: arg for arg in fn.args}
    attention = fn.body[-1]
    operands = [value.removeprefix("%") for value in attention.operands]
    scales: list[str] = []
    if len(fn.body) == 3:
        sources = {op.result_names[0].removeprefix("%"): op for op in fn.body[:2]}
        operands = [operands[0], *(sources[name].operands[0].removeprefix("%") for name in operands[1:3]),
                    *operands[3:]]
        scales = [op.operands[1].removeprefix("%") for op in fn.body[:2]]
    inputs = [(name, args[name].ir_type) for name in operands + scales]
    outputs = [(value.removeprefix("%"), result) for value, result in zip(fn.return_values, fn.result_types, strict=True)]
    buffers: list[BufferBinding] = []
    guards: list[ShapeGuard] = []
    for direction, items in (("input", inputs), ("output", outputs)):
        for name, ir_type in items:
            shape = tuple(int(extent) for extent in ir_type.shape)
            buffers.append(BufferBinding(len(buffers), name, direction, ir_type.dtype, len(shape),
                                         "row_major", _BYTES[ir_type.dtype]))
            guards.extend(ShapeGuard(name, axis, "eq", extent) for axis, extent in enumerate(shape))
    scalar_names = _DIMS
    q_shape = tuple(int(extent) for extent in args[operands[0]].ir_type.shape)
    b, hq, sq, _ = q_shape
    sk = int(args[operands[1]].ir_type.shape[2])
    bias = len(operands) == 4
    bias_shape = tuple(int(extent) for extent in args[operands[3]].ir_type.shape) if bias else ()
    if scales or (bias and bias_shape != (b, hq, sq, sk)):
        scalar_names += ("BiasB", "BiasH", "BiasQ", "BiasK") if bias else ()
    if scales:
        scalar_names += ("KScaleB", "KScaleH", "KScaleS", "KScaleD",
                         "VScaleB", "VScaleH", "VScaleS", "VScaleD")
    return {"buffers": tuple(buffers), "guards": frozenset(guards), "scalars": scalar_names}
