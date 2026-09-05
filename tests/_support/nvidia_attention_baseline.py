"""Pre-native attention packaging retained only for differential device tests."""
import hashlib
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint, NativeImageArtifact,
    OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard,
)
from tessera.compiler.nvidia_native import (
    NVIDIANativePackage, SM120_ATTN_F16_ABI, SM120_ATTN_BF16_ABI, SM120_ATTN_F32_ABI,
    SM120_ATTN_BIAS_F16_ABI, SM120_ATTN_BIAS_BF16_ABI, SM120_ATTN_BIAS_F32_ABI,
    _attention_contract, emit_attention_tile_ir, _compile_tile_ir,
)

def baseline_attention(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    contract = _attention_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 attention packaging requires static rank-4 f16/bf16/f32 Q/K/V, "
            "f32 output, MHA/GQA-compatible heads, and scale/causal semantics; "
            "bias, window, softcap, and dropout remain planned"
        )
    (
        storage, dims, scale, causal, bias_name, window_left, window_right,
        softcap, dropout_p, dropout_seed,
    ) = contract
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:{window_right}:"
        f"{softcap:.17g}:{dropout_p:.17g}:{dropout_seed}".encode()
    ).hexdigest()[:10]
    entry = f"tessera_tile_attention_{storage_ir}_{'causal' if causal else 'full'}_{semantic_key}"
    abi_id = ({
        "fp16": SM120_ATTN_BIAS_F16_ABI,
        "bf16": SM120_ATTN_BIAS_BF16_ABI,
        "fp32": SM120_ATTN_BIAS_F32_ABI,
    } if bias_name else {
        "fp16": SM120_ATTN_F16_ABI,
        "bf16": SM120_ATTN_BF16_ABI,
        "fp32": SM120_ATTN_F32_ABI,
    })[storage]
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
    alignment = 2 if storage in {"fp16", "bf16"} else 4
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


