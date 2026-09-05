"""Pre-native attention packaging retained only for differential device tests."""
import hashlib
import math
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint, NativeImageArtifact,
    OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard,
)
from tessera.compiler.nvidia_native import (
    NVIDIANativePackage, SM120_ATTN_F16_ABI, SM120_ATTN_BF16_ABI, SM120_ATTN_F32_ABI,
    SM120_ATTN_BIAS_F16_ABI, SM120_ATTN_BIAS_BF16_ABI, SM120_ATTN_BIAS_F32_ABI,
    _attention_contract, _compile_tile_ir,
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
    lse_checkpoint: str = "recompute",
) -> str:
    """Emit the correctness-first typed SDPA launch envelope."""
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported SM120 attention storage {storage!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SM120 attention scale must be finite and positive")
    if window_left < -1 or window_right < -1:
        raise ValueError("SM120 attention windows must be >= -1")
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("SM120 attention softcap must be finite and nonnegative")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("SM120 attention dropout_p must be in [0, 1)")
    if lse_checkpoint not in {"recompute", "saved"}:
        raise ValueError("SM120 attention lse_checkpoint must be 'recompute' or 'saved'")
    optional_arg = "%bias: !llvm.ptr, " if bias else ""
    optional_operand = "%bias, " if bias else ""
    lse_arg = "%row_lse: !llvm.ptr, " if lse_checkpoint == "saved" else ""
    lse_operand = "%row_lse, " if lse_checkpoint == "saved" else ""
    return f'''module {{
  llvm.func @{entry}(%q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr,
                     {optional_arg}%o: !llvm.ptr, {lse_arg}%b: i64, %hq: i64, %hkv: i64,
                     %sq: i64, %sk: i64, %d: i64, %dv: i64)
      attributes {{nvvm.kernel}} {{
    tile.attention_kernel %q, %key, %v, {optional_operand}%o, {lse_operand}%b, %hq, %hkv, %sq, %sk, %d, %dv {{
      storage = "{storage}", accum = "f32", scale = {scale:.17g} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {float(softcap)!r} : f32, dropout_p = {float(dropout_p)!r} : f32,
      dropout_seed = {dropout_seed} : i64, lse_checkpoint = "{lse_checkpoint}"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (1 + int(bias) + int(lse_checkpoint == "saved"))}i64, i64, i64, i64, i64, i64, i64
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



def emit_attention_backward_tile_ir(
    *, entry: str, scale: float, causal: bool, storage: str = "f32", bias: bool = False,
    window_left: int = -1, window_right: int = -1, softcap: float = 0.0,
    dropout_p: float = 0.0, dropout_seed: int = 0,
    lse_checkpoint: str = "recompute",
) -> str:
    """Emit the deterministic f16/bf16/f32 reference VJP through Tile IR."""
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported SM120 attention backward storage {storage!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SM120 attention backward scale must be finite and positive")
    if window_left < -1 or window_right < -1:
        raise ValueError("SM120 attention backward windows must be >= -1")
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("SM120 attention backward softcap must be finite and nonnegative")
    if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
        raise ValueError("SM120 attention backward dropout_p must be in [0, 1)")
    if lse_checkpoint not in {"recompute", "saved"}:
        raise ValueError("SM120 attention backward lse_checkpoint must be 'recompute' or 'saved'")
    optional_arg = "%bias: !llvm.ptr, " if bias else ""
    optional_operand = "%bias, " if bias else ""
    lse_arg = "%row_lse: !llvm.ptr, " if lse_checkpoint == "saved" else ""
    lse_operand = "%row_lse, " if lse_checkpoint == "saved" else ""
    return f'''module {{
  llvm.func @{entry}(%do: !llvm.ptr, %q: !llvm.ptr, %key: !llvm.ptr,
                     %v: !llvm.ptr, {optional_arg}{lse_arg}%dq: !llvm.ptr,
                     %dk: !llvm.ptr, %dv: !llvm.ptr, %b: i64, %hq: i64,
                     %hkv: i64, %sq: i64, %sk: i64, %d: i64, %dv_dim: i64)
      attributes {{nvvm.kernel}} {{
    tile.attention_backward_kernel %do, %q, %key, %v, {optional_operand}{lse_operand}%dq, %dk, %dv,
        %b, %hq, %hkv, %sq, %sk, %d, %dv_dim {{
      storage = "{storage}", accum = "f32", scale = {scale:.17g} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window_left} : i64, window_right = {window_right} : i64,
      softcap = {float(softcap)!r} : f32,
      dropout_p = {float(dropout_p)!r} : f32, dropout_seed = {dropout_seed} : i64,
      route = "deterministic_direct",
      deterministic = true, workspace_bytes = 0 : i64,
      workspace_owner = "output_element", lse_checkpoint = "{lse_checkpoint}"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (3 + int(bias) + int(lse_checkpoint == "saved"))}i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''
