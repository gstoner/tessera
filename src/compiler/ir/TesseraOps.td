
include "mlir/IR/OpBase.td"
include "mlir/IR/OpImplementation.h.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def Tessera_Dialect : Dialect {
  let name = "tessera";
  let cppNamespace = "::tessera";
  let summary = "Tessera Graph IR dialect";
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
  let stringify = [{tessera::stringifyEpilogueKind($_self);}];
  let cppNamespace = "::tessera";
}
def Tessera_EpilogueKindAttr : EnumAttr<Tessera_Dialect, Tessera_EpilogueKind, "epilogue">;

def Tessera_StorageDType : StrEnumAttr<"StorageDType", "storage dtype",
  ["f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "int8", "nvfp4"]>;
def Tessera_AccumDType : StrEnumAttr<"AccumDType", "accumulator dtype",
  ["f32", "f16", "bf16", "s32"]>;
def Tessera_RoundingMode : StrEnumAttr<"RoundingMode", "rounding mode",
  ["nearest_even", "stochastic", "toward_zero"]>;
def Tessera_QuantAxis : StrEnumAttr<"QuantAxis", "quantization scale axis",
  ["none", "per_tensor", "per_channel_0", "per_channel_1", "blockwise"]>;

def Tessera_NumericPolicyAttr : StructAttr<"NumericPolicy", Tessera_Dialect, [
  StructFieldAttr<"storage", Tessera_StorageDType>,
  StructFieldAttr<"accum", Tessera_AccumDType>,
  StructFieldAttr<"rounding", Tessera_RoundingMode>,
  StructFieldAttr<"scale", F64Attr>,
  StructFieldAttr<"quant_axis", Tessera_QuantAxis>,
  StructFieldAttr<"deterministic", BoolAttr>
]>;

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

def Tessera_MatmulOp : Op<
  Tessera_Dialect, "matmul",
  [Pure, NoSideEffect, DeclareOpInterfaceMethods<TilingInterface>]
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
  [Pure, NoSideEffect, DeclareOpInterfaceMethods<TilingInterface>]
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
  [Pure, NoSideEffect]
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

def Tessera_FusedEpilogueOp : Op<
  Tessera_Dialect, "fused_epilogue",
  [Pure, NoSideEffect]
> {
  let summary = "Fused epilogue for matmul/conv with optional bias";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs, TensorType:$bias,
                        Tessera_EpilogueKindAttr:$epilogue,
                        DefaultValuedAttr<BoolAttr, "false">:$has_bias);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_TransposeOp : Op<
  Tessera_Dialect, "transpose",
  [Pure, NoSideEffect]
> {
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CastOp : Op<
  Tessera_Dialect, "cast",
  [Pure, NoSideEffect]
> {
  let arguments = (ins TensorType:$x,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_RMSNormSafeOp : Op<
  Tessera_Dialect, "rmsnorm_safe",
  [Pure, NoSideEffect]
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
  [Pure, NoSideEffect]
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
