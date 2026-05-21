
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
// Phase F4 — AdjointInterface op trait for reverse-mode autodiff.
// Each differentiable op below opts in via DeclareOpInterfaceMethods.
include "Tessera/AdjointInterface.td"

def Tessera_Dialect : Dialect {
  let name = "tessera";
  let cppNamespace = "::tessera";
  let summary = "Tessera Graph IR dialect";
  let useDefaultTypePrinterParser = 1;
  let isExtensible = 1;
  let description = [{
    Tessera Graph IR is the top layer of the canonical Tessera compiler IR
    stack. It models high-level tensor operations before Schedule IR, Tile IR,
    and Target IR lowering.
  }];
}

def Tessera_EpilogueKind : I32EnumAttr<
  "EpilogueKind",
  "Which post-op is fused",
  [
    I32EnumAttrCase<"None", 0>,
    I32EnumAttrCase<"Relu", 1>,
    I32EnumAttrCase<"Gelu", 2>,
    I32EnumAttrCase<"Silu", 3>
  ]
> {
  let convertFromStorage = [{(tessera::EpilogueKind)$_self;}];
  let cppNamespace = "::tessera";
}
def Tessera_EpilogueKindAttr : EnumAttr<Tessera_Dialect, Tessera_EpilogueKind, "epilogue">;

def Tessera_StorageDType : I32EnumAttr<
  "StorageDType",
  "storage dtype",
  [
    I32EnumAttrCase<"F32", 0, "f32">,
    I32EnumAttrCase<"BF16", 1, "bf16">,
    I32EnumAttrCase<"F16", 2, "f16">,
    I32EnumAttrCase<"FP8E4M3", 3, "fp8_e4m3">,
    I32EnumAttrCase<"FP8E5M2", 4, "fp8_e5m2">,
    I32EnumAttrCase<"INT8", 5, "int8">,
    I32EnumAttrCase<"NVFP4", 6, "nvfp4">
  ]
> {
  let cppNamespace = "::tessera";
}
def Tessera_StorageDTypeAttr : EnumAttr<Tessera_Dialect, Tessera_StorageDType, "storage_dtype">;

def Tessera_AccumDType : I32EnumAttr<
  "AccumDType",
  "accumulator dtype",
  [
    I32EnumAttrCase<"F32", 0, "f32">,
    I32EnumAttrCase<"F16", 1, "f16">,
    I32EnumAttrCase<"BF16", 2, "bf16">,
    I32EnumAttrCase<"S32", 3, "s32">
  ]
> {
  let cppNamespace = "::tessera";
}
def Tessera_AccumDTypeAttr : EnumAttr<Tessera_Dialect, Tessera_AccumDType, "accum_dtype">;

def Tessera_RoundingMode : I32EnumAttr<
  "RoundingMode",
  "rounding mode",
  [
    I32EnumAttrCase<"NearestEven", 0, "nearest_even">,
    I32EnumAttrCase<"Stochastic", 1, "stochastic">,
    I32EnumAttrCase<"TowardZero", 2, "toward_zero">
  ]
> {
  let cppNamespace = "::tessera";
}
def Tessera_RoundingModeAttr : EnumAttr<Tessera_Dialect, Tessera_RoundingMode, "rounding_mode">;

def Tessera_QuantAxis : I32EnumAttr<
  "QuantAxis",
  "quantization scale axis",
  [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"PerTensor", 1, "per_tensor">,
    I32EnumAttrCase<"PerChannel0", 2, "per_channel_0">,
    I32EnumAttrCase<"PerChannel1", 3, "per_channel_1">,
    I32EnumAttrCase<"Blockwise", 4, "blockwise">
  ]
> {
  let cppNamespace = "::tessera";
}
def Tessera_QuantAxisAttr : EnumAttr<Tessera_Dialect, Tessera_QuantAxis, "quant_axis">;

def Tessera_NumericPolicyAttr
    : DictionaryAttrBase<CPred<"::llvm::isa<::mlir::DictionaryAttr>($_self)">,
                         "Tessera numeric policy dictionary">;

def TensorType : AnyTypeOf<[AnyRankedTensor, AnyUnrankedTensor]>;

def Tessera_KVCacheType : TypeDef<Tessera_Dialect, "KVCache"> {
  let mnemonic = "kv_cache";
  let summary = "Stateful KV cache handle with rolling-window/page semantics";
}

def Tessera_CachePageType : TypeDef<Tessera_Dialect, "CachePage"> {
  let mnemonic = "cache_page";
  let summary = "Logical cache page handle";
}

def Tessera_RingType : TypeDef<Tessera_Dialect, "Ring"> {
  let mnemonic = "ring";
  let summary = "Stateful ring-buffer handle";
}

def Tessera_ArchParamType : TypeDef<Tessera_Dialect, "ArchParam"> {
  let mnemonic = "arch_param";
  let summary = "FP32 architecture-search parameter vector";
}

def Tessera_ArchGateType : TypeDef<Tessera_Dialect, "ArchGate"> {
  let mnemonic = "arch_gate";
  let summary = "Differentiable architecture gate vector";
}

def Tessera_NeighborsTopologyType : TypeDef<Tessera_Dialect, "NeighborsTopology"> {
  let mnemonic = "neighbors.topology";
  let summary = "Opaque handle for tessera.neighbors topology extension ops";
}

def Tessera_NeighborsHaloType : TypeDef<Tessera_Dialect, "NeighborsHalo"> {
  let mnemonic = "neighbors.halo";
  let summary = "Opaque handle for tessera.neighbors halo extension ops";
}

def Tessera_ArchRelax : I32EnumAttr<
  "ArchRelax",
  "architecture relaxation",
  [
    I32EnumAttrCase<"Softmax", 0, "softmax">,
    I32EnumAttrCase<"Gumbel", 1, "gumbel">,
    I32EnumAttrCase<"HardConcrete", 2, "hard_concrete">,
    I32EnumAttrCase<"Ste", 3, "ste">
  ]
> {
  let cppNamespace = "::tessera";
}
def Tessera_ArchRelaxAttr : EnumAttr<Tessera_Dialect, Tessera_ArchRelax, "arch_relax">;

def Tessera_MatmulOp : Op<
  Tessera_Dialect, "matmul",
  [Pure,
   // B3 v2 (2026-05-20): explicit method list — required under MLIR 21
   // for TilingInterface to emit per-Op method declarations on the
   // generated header.  See src/compiler/ir/TilingInterface_NOTES.md.
   DeclareOpInterfaceMethods<TilingInterface, [
     "getLoopIteratorTypes",
     "getIterationDomain",
     "getTiledImplementation",
     "getResultTilePosition",
   ]>,
   DeclareOpInterfaceMethods<Tessera_AdjointInterface>]
> {
  let summary = "Matrix multiply (A @ B)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs,
                        OptionalAttr<I64Attr>:$tile_k,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "false">:$transposeA,
                        DefaultValuedAttr<BoolAttr, "false">:$transposeB);
  let results = (outs TensorType:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_Conv2DNHWCOp : Op<
  Tessera_Dialect, "conv2d_nhwc",
  [Pure,
   // B3 v2 (2026-05-20): same explicit method list as MatmulOp.
   // ``getTiledImplementation`` returns ``failure()`` until the
   // stride/pad-aware window reconstruction lands (deferred v3 work).
   DeclareOpInterfaceMethods<TilingInterface, [
     "getLoopIteratorTypes",
     "getIterationDomain",
     "getTiledImplementation",
     "getResultTilePosition",
   ]>]
> {
  let summary = "2D convolution NHWC x HWCF -> NHWC";
  let arguments = (ins TensorType:$input, TensorType:$filter,
                        ArrayAttr:$strides, ArrayAttr:$dilations,
                        OptionalAttr<Tessera_EpilogueKindAttr>:$epilogue);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_FlashAttnOp : Op<
  Tessera_Dialect, "flash_attn",
  [Pure]
> {
  let summary = "FlashAttention fused op";
  let arguments = (ins TensorType:$q, AnyType:$k_or_cache, Variadic<AnyType>:$v,
                        I64Attr:$head_dim,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        OptionalAttr<F64Attr>:$dropout_p,
                        DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_RopeOp : Op<Tessera_Dialect, "rope", [Pure]> {
  let summary = "Rotary position embedding over the innermost dimension";
  let arguments = (ins TensorType:$x, TensorType:$theta);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_NTKRopeOp : Op<Tessera_Dialect, "ntk_rope", [Pure]> {
  let summary = "NTK-scaled rotary position embedding";
  let arguments = (ins TensorType:$x, TensorType:$theta,
                        DefaultValuedAttr<F64Attr, "1.0">:$scale);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_ALiBiOp : Op<Tessera_Dialect, "alibi", [Pure]> {
  let summary = "ALiBi positional bias helper";
  let arguments = (ins I64Attr:$num_heads, I64Attr:$seq_len);
  let results = (outs TensorType:$bias);
  let assemblyFormat = "attr-dict `:` type($bias)";
}

def Tessera_RopeSplitOp : Op<Tessera_Dialect, "rope_split", [Pure]> {
  let summary = "Split last dimension into RoPE and non-RoPE partitions";
  let arguments = (ins TensorType:$x, I64Attr:$rope_dim);
  let results = (outs TensorType:$rope_part, TensorType:$no_rope_part);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_RopeMergeOp : Op<Tessera_Dialect, "rope_merge", [Pure]> {
  let summary = "Merge RoPE and non-RoPE partitions along the last dimension";
  let arguments = (ins TensorType:$rope_part, TensorType:$no_rope_part);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MultiHeadAttentionOp : Op<Tessera_Dialect, "multi_head_attention", [Pure]> {
  let summary = "Multi-head attention wrapper over flash attention";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$num_heads,
                        OptionalAttr<F64Attr>:$scale,
                        DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_GQAAttentionOp : Op<Tessera_Dialect, "gqa_attention", [Pure]> {
  let summary = "Grouped-query attention wrapper";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$num_query_heads,
                        I64Attr:$num_kv_heads);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MQAAttentionOp : Op<Tessera_Dialect, "mqa_attention", [Pure]> {
  let summary = "Multi-query attention wrapper";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MLADecodeOp : Op<Tessera_Dialect, "mla_decode", [Pure]> {
  let summary = "MLA decode wrapper over latent K/V expansion and attention";
  let arguments = (ins TensorType:$q, TensorType:$k_latent, TensorType:$v_latent,
                        Variadic<TensorType>:$weights);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_QuantizeFP8Op : Op<Tessera_Dialect, "quantize_fp8", [Pure]> {
  let summary = "Reference FP8 quantize with dequantized fp32 payload plus scale";
  let arguments = (ins TensorType:$x, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x_q, AnyType:$scale);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_DequantizeFP8Op : Op<Tessera_Dialect, "dequantize_fp8", [Pure]> {
  let summary = "Reference FP8 dequantize";
  let arguments = (ins TensorType:$x_q, AnyType:$scale, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_QuantizeFP4Op : Op<Tessera_Dialect, "quantize_fp4", [Pure]> {
  let summary = "Reference FP4 quantize with dequantized fp32 payload plus scale";
  let arguments = (ins TensorType:$x, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x_q, AnyType:$scale);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_DequantizeFP4Op : Op<Tessera_Dialect, "dequantize_fp4", [Pure]> {
  let summary = "Reference FP4 dequantize";
  let arguments = (ins TensorType:$x_q, AnyType:$scale, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MoeDispatchOp : Op<Tessera_Dialect, "moe_dispatch", []> {
  let summary = "MoE token dispatch transport op";
  let arguments = (ins TensorType:$x, TensorType:$route);
  let results = (outs TensorType:$partials);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MoeCombineOp : Op<Tessera_Dialect, "moe_combine", []> {
  let summary = "MoE token combine transport op";
  let arguments = (ins TensorType:$partials, TensorType:$inverse_route,
                        DefaultValuedAttr<StrAttr, "\"sum\"">:$reduce);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// attention_variants_plan, LA-1 — Linear / kernel-feature attention.
// Distinct ODS op so backend passes can match it independently of
// flash_attn. The `feature_map` string attribute selects the φ function;
// `causal` selects between recurrent and folded forms. Decay / state /
// chunk_size live as additional optional attrs on the op for future
// kernel variants.
def Tessera_LinearAttnOp : Op<
  Tessera_Dialect, "linear_attn",
  [Pure]
> {
  let summary = "Linear / kernel-feature attention (causal recurrence or folded)";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        DefaultValuedAttr<StrAttr, "\"elu\"">:$feature_map,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        OptionalAttr<I64Attr>:$chunk_size);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_LinearAttnStateOp : Op<
  Tessera_Dialect, "linear_attn_state",
  [Pure]
> {
  let summary = "Linear-attention companion op returning the post-update state";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        DefaultValuedAttr<StrAttr, "\"elu\"">:$feature_map,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        OptionalAttr<I64Attr>:$chunk_size);
  let results = (outs TensorType:$state);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// attention_variants_plan, LA-4 — promotion of the example
// `power_retention` dialect into the canonical Tessera dialect.
// Power attention (Buckman/Edelmuth 2024) and Retention (Sun et al.,
// RetNet 2023) are both linear-cost causal attention variants; share
// the (Q, K, V, optional decay/log_g, optional state) input shape
// with `linear_attn`. The distinct op_names anchor backend-specific
// kernels (the example folder ships CUDA + HIP sketches under
// `examples/advanced/power_retention/src/kernels/`).

def Tessera_PowerAttnOp : Op<
  Tessera_Dialect, "power_attn",
  [Pure]
> {
  let summary = "Symmetric power attention (linear-cost; deg + state width)";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$state,
                        OptionalAttr<I64Attr>:$window,
                        DefaultValuedAttr<I64Attr, "2">:$deg,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_RetentionOp : Op<
  Tessera_Dialect, "retention",
  [Pure]
> {
  let summary = "RetNet-style retention with multiplicative decay (linear-cost)";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        Optional<TensorType>:$log_g,
                        DefaultValuedAttr<I64Attr, "2">:$deg,
                        DefaultValuedAttr<I64Attr, "128">:$chunk,
                        OptionalAttr<I64Attr>:$switch_over,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  // (output, state, sum_of_keys) — matches RetNet's training/inference contract.
  let results = (outs TensorType:$o, TensorType:$state, TensorType:$sum_of_keys);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_LightningAttentionOp : Op<
  Tessera_Dialect, "lightning_attention",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "Lightning linear attention with optional recurrent state";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        Optional<TensorType>:$state,
                        Optional<TensorType>:$decay,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        DefaultValuedAttr<BoolAttr, "false">:$return_state,
                        DefaultValuedAttr<StrAttr, "\"fp32\"">:$state_dtype);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_GatedAttentionOp : Op<
  Tessera_Dialect, "gated_attention",
  [Pure]
> {
  let summary = "Softmax attention multiplied by a learned gate";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v, TensorType:$gate,
                        OptionalAttr<F64Attr>:$scale,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        DefaultValuedAttr<StrAttr, "\"sigmoid\"">:$gate_activation);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Tessera_DeltaAttentionOp<string mnemonic, string summaryText>
  : Op<Tessera_Dialect, mnemonic, [Pure, AttrSizedOperandSegments]> {
  let summary = summaryText;
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        Optional<TensorType>:$gate,
                        Optional<TensorType>:$beta,
                        Optional<TensorType>:$decay,
                        Optional<TensorType>:$state,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        DefaultValuedAttr<BoolAttr, "false">:$return_state,
                        DefaultValuedAttr<StrAttr, "\"fp32\"">:$state_dtype);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_GatedDeltaNetOp
  : Tessera_DeltaAttentionOp<"gated_deltanet", "Gated DeltaNet recurrent attention">;
def Tessera_KimiDeltaAttentionOp
  : Tessera_DeltaAttentionOp<"kimi_delta_attention", "Kimi Delta Attention recurrent op">;
def Tessera_ModifiedDeltaAttentionOp
  : Tessera_DeltaAttentionOp<"modified_delta_attention", "Modified bounded Delta Attention recurrent op">;

def Tessera_HybridAttentionOp : Op<
  Tessera_Dialect, "hybrid_attention",
  [Pure]
> {
  let summary = "Named hybrid attention policy wrapper for Ling/Kimi schedules";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        DefaultValuedAttr<StrAttr, "\"auto\"">:$pattern,
                        DefaultValuedAttr<I64Attr, "0">:$layer_index,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        DefaultValuedAttr<BoolAttr, "false">:$return_state,
                        DefaultValuedAttr<StrAttr, "\"fp32\"">:$state_dtype);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_FusedEpilogueOp : Op<
  Tessera_Dialect, "fused_epilogue",
  [Pure]
> {
  let summary = "Fused epilogue for matmul/conv with optional bias";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs, TensorType:$bias,
                        Tessera_EpilogueKindAttr:$epilogue,
                        DefaultValuedAttr<BoolAttr, "false">:$has_bias);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

class Tessera_UnaryPureOp<string mnemonic, string summaryText> : Op<
  Tessera_Dialect, mnemonic,
  [Pure, DeclareOpInterfaceMethods<Tessera_AdjointInterface>]
> {
  let summary = summaryText;
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_LayerNormOp : Tessera_UnaryPureOp<"layer_norm", "LayerNorm over the innermost dimension"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<F64Attr>:$eps,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
}

def Tessera_SoftmaxOp : Tessera_UnaryPureOp<"softmax", "Softmax over an explicit or innermost axis"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
}

def Tessera_GeluOp : Tessera_UnaryPureOp<"gelu", "Gaussian error linear unit">;
def Tessera_ReluOp : Tessera_UnaryPureOp<"relu", "Rectified linear unit">;
def Tessera_SigmoidOp : Tessera_UnaryPureOp<"sigmoid", "Sigmoid activation">;
def Tessera_SinOp : Tessera_UnaryPureOp<"sin", "Sine elementwise op">;

// Phase 8.4.8 Stage 2a — fused silu-and-multiply primitive.
// Anchors the SwiGLU `matmul → silu_mul → matmul` chain that Stage 2b's
// SwigluFusionPass collapses into Tessera_SwigluFusedOp. Adjoint is
// handled today by the Python numpy-tape autodiff (see
// python/tessera/autodiff/vjp.py); IR-level adjoint via AdjointInterface
// is a follow-up tied to Phase F4.
def Tessera_SiluMulOp : Op<
  Tessera_Dialect, "silu_mul",
  [Pure]
> {
  let summary = "Fused silu(a) * b primitive (SwiGLU activation step)";
  let arguments = (ins TensorType:$a, TensorType:$b);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Phase 8.4.8 Stage 2b — SwiGLU 3-op fusion target. Result of
// SwigluFusionPass collapsing matmul → matmul → silu_mul → matmul into a
// single op carrying (x, W_gate, W_up, W_down). Backends with a fused
// MLP-block kernel (Apple GPU MSL — Stage 3, NVIDIA WGMMA epilogue, ROCm
// MFMA epilogue) lower this directly; backends without one can expand it
// back to the chain in their own lowering pipelines.
def Tessera_SwigluFusedOp : Op<
  Tessera_Dialect, "swiglu_fused",
  [Pure]
> {
  let summary = "Fused SwiGLU MLP block: gemm(silu(x@Wg) * (x@Wu), Wd)";
  let arguments = (ins TensorType:$x,
                       TensorType:$w_gate,
                       TensorType:$w_up,
                       TensorType:$w_down);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Theme 5 — MLA primitives (Python ops landed; ODS for fusion-pass
// match support). Each is matmul-shaped; the distinct op_names anchor
// the FlashMLA target pass match in MLAFusionPass.
def Tessera_LatentKVCompressOp : Op<
  Tessera_Dialect, "latent_kv_compress",
  [Pure]
> {
  let summary = "MLA latent compression: c = x @ W_dkv";
  let arguments = (ins TensorType:$x, TensorType:$w_dkv);
  let results = (outs TensorType:$c);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_LatentKVExpandKOp : Op<
  Tessera_Dialect, "latent_kv_expand_k",
  [Pure]
> {
  let summary = "MLA latent K expansion: K = c @ W_uk";
  let arguments = (ins TensorType:$c, TensorType:$w_uk);
  let results = (outs TensorType:$k);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_LatentKVExpandVOp : Op<
  Tessera_Dialect, "latent_kv_expand_v",
  [Pure]
> {
  let summary = "MLA latent V expansion: V = c @ W_uv";
  let arguments = (ins TensorType:$c, TensorType:$w_uv);
  let results = (outs TensorType:$v);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion target.
// Result of MLAFusionPass collapsing the 3-op chain
//   c = latent_kv_compress(x, W_dkv)
//   K = latent_kv_expand_k(c, W_uk)
//   V = latent_kv_expand_v(c, W_uv)
//   O = flash_attn(Q, K, V)
// into a single op carrying (x, W_dkv, W_uk, W_uv, Q). Backends with a
// FlashMLA-style absorb-K kernel (Hopper, future Blackwell) materialize
// just `c` and reuse `W_uk` / `W_uv` inside the score kernel; backends
// without one expand back to the 4-op chain.
def Tessera_MLADecodeFusedOp : Op<
  Tessera_Dialect, "mla_decode_fused",
  [Pure]
> {
  let summary = "Fused MLA decode block: latent compress → expand → flash_attn";
  let arguments = (ins TensorType:$x,
                       TensorType:$w_dkv,
                       TensorType:$w_uk,
                       TensorType:$w_uv,
                       TensorType:$q,
                       OptionalAttr<F64Attr>:$scale,
                       DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// attention_variants_plan, NSA — branch + fusion ODS ops.
def Tessera_AttnSlidingWindowOp : Op<
  Tessera_Dialect, "attn_sliding_window",
  [Pure]
> {
  let summary = "NSA branch 1 — sliding-window dense attention";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$window_size,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AttnCompressedBlocksOp : Op<
  Tessera_Dialect, "attn_compressed_blocks",
  [Pure]
> {
  let summary = "NSA branch 2 — attention over compressed-block summaries";
  let arguments = (ins TensorType:$q, TensorType:$k_c, TensorType:$v_c);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AttnTopKBlocksOp : Op<
  Tessera_Dialect, "attn_top_k_blocks",
  [Pure]
> {
  let summary = "NSA branch 3 — top-k block-selected attention";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        TensorType:$scores,
                        I64Attr:$top_k,
                        I64Attr:$block_size,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Gap 4 / Sub-1 (2026-05-20) — 2D local-window attention for spatial
// grids.  Each query at position (h, w) attends to a
// (2*window_h+1) × (2*window_w+1) neighbourhood centred at (h, w).
// Same halo / sharding shape as 2D stencils — the same
// HaloMeshIntegrationPass picks it up via the `halo_aware` metadata in
// primitive_coverage.py.
//
// Operand shapes:
//   q, k, v : rank-5 tensors  (B, H, Hq, Wq, D)
//   o       : rank-5 tensor   (B, H, Hq, Wq, D)  — same shape as q
//
// Optional attributes:
//   halo.width : I64ArrayAttr [rh, rw] mirroring the Python kwarg, so
//                the integration pass can read width without re-parsing.
//   numeric_policy : optional dictionary carrying {storage, accum,
//                math_mode} per the NumericPolicy spec.
def Tessera_AttnLocalWindow2DOp : Op<
  Tessera_Dialect, "attn_local_window_2d",
  [Pure]
> {
  let summary = "2D local-window attention for spatial grids "
                "(weather/climate / ViT local bias / neural CA)";
  let description = [{
    Computes per-(h, w) local attention over a fixed-size 2D neighbourhood:

        scores(h, w, ph, pw) = Q(h, w) · K(h + ph - rh, w + pw - rw) / √D
        weights              = masked-softmax over (ph, pw)
        O(h, w)              = Σ_(ph, pw) weights · V(h + ph - rh, w + pw - rw)

    where ``(rh, rw) = window``.  Edge positions are masked (the gather
    is clipped to in-range indices and the masked-softmax zeros their
    contribution), matching the numpy reference in
    ``tessera.ops.attn_local_window_2d``.

    The op carries an optional ``halo.width = [rh, rw]`` attribute so
    the ``HaloMeshIntegrationPass`` can insert ``halo.exchange`` ops on
    rank boundaries without re-parsing kwargs.
  }];
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64ArrayAttr:$window);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_DeepSeekSparseAttentionOp : Op<
  Tessera_Dialect, "deepseek_sparse_attention",
  [Pure]
> {
  let summary = "DeepSeek/NSA sparse attention wrapper over sliding/compressed/top-k branches";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        Optional<TensorType>:$gate_logits,
                        I64Attr:$window_size,
                        I64Attr:$block_size,
                        I64Attr:$top_k,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// NSA-4 fusion target — collapses the three NSA branches + gating into
// a single op for backends with a fused NSA kernel.
def Tessera_NativeSparseAttnFusedOp : Op<
  Tessera_Dialect, "native_sparse_attn_fused",
  [Pure]
> {
  let summary = "Fused NSA block (sliding + compressed + selected + gating)";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        TensorType:$gate_logits,
                        I64Attr:$window_size,
                        I64Attr:$block_size,
                        I64Attr:$top_k,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Phase F-MoR — Mixture of Recursions primitives.
def Tessera_MorRouterOp : Op<
  Tessera_Dialect, "mor_router",
  [Pure]
> {
  let summary = "MoR token-choice router: argmax of x @ w_router → per-token depth";
  let arguments = (ins TensorType:$x, TensorType:$w_router,
                        I64Attr:$max_depth);
  let results = (outs TensorType:$depth);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MorPartitionOp : Op<
  Tessera_Dialect, "mor_partition",
  [Pure]
> {
  let summary = "MoR active-token mask at recursion step";
  let arguments = (ins TensorType:$x, TensorType:$depth,
                        I64Attr:$step);
  let results = (outs TensorType:$mask);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_MorScatterOp : Op<
  Tessera_Dialect, "mor_scatter",
  [Pure]
> {
  let summary = "MoR scatter — write updated values back into full at masked positions";
  let arguments = (ins TensorType:$full, TensorType:$updated, TensorType:$mask);
  let results = (outs TensorType:$out);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_TransposeOp : Op<
  Tessera_Dialect, "transpose",
  [Pure]
> {
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CastOp : Op<
  Tessera_Dialect, "cast",
  [Pure]
> {
  let arguments = (ins TensorType:$x,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_DropoutOp : Op<Tessera_Dialect, "dropout", []> {
  let summary = "Reference random dropout with optional deterministic seed";
  let arguments = (ins TensorType:$x,
                       OptionalAttr<F64Attr>:$p,
                       OptionalAttr<I64Attr>:$seed,
                       DefaultValuedAttr<BoolAttr, "true">:$training);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

class Tessera_CollectiveOp<string mnemonic, string summaryText> : Op<
  Tessera_Dialect, mnemonic,
  []
> {
  let summary = summaryText;
  let arguments = (ins TensorType:$x,
                       OptionalAttr<StrAttr>:$axis,
                       OptionalAttr<StrAttr>:$op);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AllReduceOp : Tessera_CollectiveOp<"all_reduce", "Single-rank/reference all-reduce semantic op">;
def Tessera_ReduceScatterOp : Tessera_CollectiveOp<"reduce_scatter", "Single-rank/reference reduce-scatter semantic op">;
def Tessera_AllGatherOp : Tessera_CollectiveOp<"all_gather", "Single-rank/reference all-gather semantic op">;

class Tessera_SpectralUnaryOp<string mnemonic, string summaryText> : Op<
  Tessera_Dialect, mnemonic,
  [Pure]
> {
  let summary = summaryText;
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_FFTOp : Tessera_SpectralUnaryOp<"fft", "FFT spectral transform">;
def Tessera_IFFTOp : Tessera_SpectralUnaryOp<"ifft", "Inverse FFT spectral transform">;
def Tessera_RFFTOp : Tessera_SpectralUnaryOp<"rfft", "Real FFT spectral transform">;
def Tessera_IRFFTOp : Tessera_SpectralUnaryOp<"irfft", "Inverse real FFT spectral transform">;
def Tessera_DCTOp : Tessera_SpectralUnaryOp<"dct", "Discrete cosine transform"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<I64Attr>:$type);
}

def Tessera_SpectralConvOp : Op<Tessera_Dialect, "spectral_conv", [Pure]> {
  let summary = "Reference spectral convolution semantic op";
  let arguments = (ins TensorType:$x, TensorType:$filter);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AdamOp : Op<Tessera_Dialect, "adam", []> {
  let summary = "Functional Adam optimizer step";
  let arguments = (ins TensorType:$param, TensorType:$grad,
                       TensorType:$moment1, TensorType:$moment2,
                       OptionalAttr<F64Attr>:$lr,
                       OptionalAttr<F64Attr>:$beta1,
                       OptionalAttr<F64Attr>:$beta2,
                       OptionalAttr<F64Attr>:$eps,
                       OptionalAttr<I64Attr>:$step,
                       OptionalAttr<StrAttr>:$compute_dtype,
                       OptionalAttr<StrAttr>:$state_dtype,
                       OptionalAttr<StrAttr>:$master_dtype,
                       OptionalAttr<BoolAttr>:$cast_updates_to_param_dtype);
  let results = (outs TensorType:$new_param, TensorType:$new_moment1, TensorType:$new_moment2);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Tessera_OptimizerTreeOp<string mnemonic, string summaryText>
  : Op<Tessera_Dialect, mnemonic, []> {
  let summary = summaryText;
  let arguments = (ins TensorType:$params, TensorType:$grads,
                       Optional<TensorType>:$state,
                       OptionalAttr<F64Attr>:$lr,
                       OptionalAttr<F64Attr>:$beta1,
                       OptionalAttr<F64Attr>:$beta2,
                       OptionalAttr<F64Attr>:$eps,
                       OptionalAttr<F64Attr>:$momentum,
                       OptionalAttr<F64Attr>:$weight_decay,
                       OptionalAttr<StrAttr>:$compute_dtype,
                       OptionalAttr<StrAttr>:$state_dtype,
                       OptionalAttr<StrAttr>:$master_dtype,
                       OptionalAttr<BoolAttr>:$cast_updates_to_param_dtype);
  let results = (outs TensorType:$new_params, TensorType:$new_state);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AdamWOp : Tessera_OptimizerTreeOp<"adamw", "Functional AdamW optimizer tree step">;
def Tessera_MomentumOp : Tessera_OptimizerTreeOp<"momentum", "Functional SGD-with-momentum optimizer tree step">;
def Tessera_AdafactorOp : Tessera_OptimizerTreeOp<"adafactor", "Functional Adafactor optimizer tree step">;
def Tessera_LionOp : Tessera_OptimizerTreeOp<"lion", "Functional Lion optimizer tree step">;

def Tessera_RMSNormSafeOp : Op<
  Tessera_Dialect, "rmsnorm_safe",
  [Pure]
> {
  let summary = "Numerically safe RMSNorm with policy-controlled stats/rounding";
  let arguments = (ins TensorType:$x,
                        OptionalAttr<F64Attr>:$eps,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_SoftmaxSafeOp : Op<
  Tessera_Dialect, "softmax_safe",
  [Pure]
> {
  let summary = "Numerically safe softmax/logsumexp primitive";
  let arguments = (ins TensorType:$x,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_KVCacheCreateOp : Op<Tessera_Dialect, "kv_cache.create", []> {
  let summary = "Create a stateful KV cache with rolling-window/page policy";
  let arguments = (ins I64Attr:$max_seq,
                       I64Attr:$head_dim,
                       OptionalAttr<StrAttr>:$eviction,
                       OptionalAttr<I64Attr>:$page_size,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs Tessera_KVCacheType:$cache);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tessera_KVCacheAppendOp : Op<Tessera_Dialect, "kv_cache.append", []> {
  let summary = "Append key/value tensors to a stateful KV cache";
  let arguments = (ins Tessera_KVCacheType:$cache, TensorType:$k, TensorType:$v);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_KVCachePruneOp : Op<Tessera_Dialect, "kv_cache.prune", []> {
  let summary = "Prune a rolling-window KV cache to a fixed window";
  let arguments = (ins Tessera_KVCacheType:$cache, I64Attr:$window);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CachePageLookupOp : Op<Tessera_Dialect, "cache.page_lookup", []> {
  let summary = "Lookup a logical page in a stateful cache";
  let arguments = (ins Tessera_KVCacheType:$cache, Index:$position);
  let results = (outs Tessera_CachePageType:$page);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_RingCreateOp : Op<Tessera_Dialect, "ring.create", []> {
  let summary = "Create a stateful ring buffer";
  let arguments = (ins I64Attr:$capacity);
  let results = (outs Tessera_RingType:$ring);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchParameterOp : Op<Tessera_Dialect, "arch.parameter", []> {
  let summary = "Declare FP32 architecture logits separate from model weights";
  let arguments = (ins I64Attr:$size,
                       OptionalAttr<StrAttr>:$name,
                       DefaultValuedAttr<F64Attr, "0.0">:$init);
  let results = (outs Tessera_ArchParamType:$alpha);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchGumbelSoftmaxOp : Op<Tessera_Dialect, "arch.gumbel_softmax", []> {
  let summary = "Differentiable Gumbel-Softmax relaxation over architecture logits";
  let arguments = (ins Tessera_ArchParamType:$alpha,
                       F64Attr:$temperature,
                       OptionalAttr<I64Attr>:$seed);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchHardConcreteOp : Op<Tessera_Dialect, "arch.hard_concrete", []> {
  let summary = "Hard-concrete sparse relaxation over architecture logits";
  let arguments = (ins Tessera_ArchParamType:$alpha,
                       F64Attr:$temperature);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchSTEOneHotOp : Op<Tessera_Dialect, "arch.ste_one_hot", []> {
  let summary = "Straight-through one-hot architecture gate";
  let arguments = (ins Tessera_ArchParamType:$alpha);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchWeightedSumOp : Op<Tessera_Dialect, "arch.weighted_sum", []> {
  let summary = "Weighted merge of candidate tensor outputs by architecture gate";
  let arguments = (ins Variadic<TensorType>:$candidates,
                       Tessera_ArchGateType:$gate);
  let results = (outs TensorType:$result);
  let assemblyFormat = "$candidates `by` $gate attr-dict `:` functional-type($candidates, $result)";
  let hasVerifier = 1;
}

def Tessera_ArchSwitchOp : Op<Tessera_Dialect, "arch.switch", []> {
  let summary = "Soft or hard architecture candidate switch";
  let arguments = (ins Variadic<TensorType>:$candidates,
                       Tessera_ArchGateType:$gate,
                       DefaultValuedAttr<BoolAttr, "false">:$hard);
  let results = (outs TensorType:$result);
  let assemblyFormat = "$candidates `by` $gate attr-dict `:` functional-type($candidates, $result)";
  let hasVerifier = 1;
}

def Tessera_ArchMixedOp : Op<Tessera_Dialect, "arch.mixed", []> {
  let summary = "Search-space mixed operator carrying candidate names and relaxation";
  let arguments = (ins AnyType:$input,
                       Tessera_ArchGateType:$gate,
                       ArrayAttr:$candidates,
                       Tessera_ArchRelaxAttr:$relax);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `by` $gate attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// Neighbors extension marker ops. The standalone TesseraNeighbors component
// owns semantic passes for these names; defining generic shells in the core
// tessera dialect lets MLIR parse quoted generic IR such as
// "tessera.neighbors.stencil.apply" while those passes remain string-name
// driven and lightweight.
class Tessera_NeighborsGenericOp<string mnemonic>
    : Op<Tessera_Dialect, mnemonic, []> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
}

def Tessera_NeighborsTopologyCreateOp
    : Tessera_NeighborsGenericOp<"neighbors.topology.create">;
def Tessera_NeighborsHaloRegionOp
    : Tessera_NeighborsGenericOp<"neighbors.halo.region">;
def Tessera_NeighborsHaloExchangeOp
    : Tessera_NeighborsGenericOp<"neighbors.halo.exchange">;
// Sub-4 (2026-05-20) — halo transport triple lowered from halo.exchange.
def Tessera_NeighborsHaloPackOp
    : Tessera_NeighborsGenericOp<"neighbors.halo.pack">;
def Tessera_NeighborsHaloTransportOp
    : Tessera_NeighborsGenericOp<"neighbors.halo.transport">;
def Tessera_NeighborsHaloUnpackOp
    : Tessera_NeighborsGenericOp<"neighbors.halo.unpack">;
def Tessera_NeighborsNeighborReadOp
    : Tessera_NeighborsGenericOp<"neighbors.neighbor.read">;
def Tessera_NeighborsStencilDefineOp
    : Tessera_NeighborsGenericOp<"neighbors.stencil.define">;
def Tessera_NeighborsStencilApplyOp
    : Tessera_NeighborsGenericOp<"neighbors.stencil.apply">;
def Tessera_NeighborsPipelineConfigOp
    : Tessera_NeighborsGenericOp<"neighbors.pipeline.config">;

// ── Phase F4 — Custom adjoint placeholder ─────────────────────────────────
//
// Emitted by the AutodiffPass when an op carries a `customAdjointName()`
// (i.e., its VJP is registered from Python via
// `tessera.autodiff.custom_rule`). The runtime resolves this op against the
// Python registry; the C++ pass treats it as opaque.
//
// Operands: forward inputs + forward output cotangents (interleaved by
// position; the `name` attr disambiguates).
// Results: one cotangent per forward input (Variadic).
def Tessera_CustomAdjointCallOp : Op<
  Tessera_Dialect, "custom_adjoint_call",
  [Pure]
> {
  let summary = "Placeholder for a Python-registered VJP — resolved at runtime";
  let description = [{
    Emitted by the AutodiffPass when a forward op opts into a
    Python-registered VJP via `AdjointInterface::customAdjointName()`. The
    `name` attribute is the registry key; the runtime lowers this op by
    calling the registered Python callable with `(dout, *forward_inputs)`
    and unpacking the returned tuple of input cotangents.
  }];
  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$cotangents);
  let assemblyFormat = "$name `(` operands `)` attr-dict `:` functional-type(operands, results)";
}
