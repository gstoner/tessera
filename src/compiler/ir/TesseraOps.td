
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

// DeepGEMM-inspired scale-layout contract as a first-class IR attribute (2026-06).
// Mirrors the python `grouped_layout.ScaleLayout` dataclass: granularity
// (per_tensor/per_row/per_channel/block), block shape, packed scale element
// format (none/e4m3/e5m2/e8m0/ue8m0), vector_size, alignment, and whether the
// scale tensor is TMA-ready transposed (MN-major).  Carried alongside (and
// describing the layout of) the optional scale-tensor operands on grouped GEMM
// ops so LayoutLegalityPass can insert repack/transpose/align — promoting scale
// from a `quant` dtype-name string to a verifiable layout type.
def Tessera_ScaleLayoutAttr
    : DictionaryAttrBase<CPred<"::llvm::isa<::mlir::DictionaryAttr>($_self)">,
                         "Tessera scale-layout dictionary">;

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
  // Per-op hook twin of CanonicalizeTesseraIR's TransposeIntoMatmul, so the
  // transpose→flag fold also fires under the generic --canonicalize the CPU JIT
  // runs (reaching the executed path, not just the custom tessera-canonicalize).
  let hasCanonicalizer = 1;
}

// Apple Value Target IR sprint 6 (2026-06-03): batched matrix multiply.
// Rank-3 C[b] = A[b] @ B[b] over a shared leading batch dim. Kept minimal at
// the Graph-IR layer ([Pure] + a shape verifier); the value lane lowers the
// static rank-3 f32 envelope to the Accelerate batched GEMM C ABI. No
// broadcasting and no transpose attrs in this sprint (both are gated).
def Tessera_BatchedGemmOp : Op<
  Tessera_Dialect, "batched_gemm",
  [Pure]
> {
  let summary = "Batched matrix multiply (C[b] = A[b] @ B[b])";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// DeepGEMM-inspired grouped GEMM (MoE expert FFN) with the grouped-layout
// contract as first-class IR attributes (2026-06).  `grouped_kind` selects the
// family (dense / contiguous / masked / k_grouped); `grouped_alignment` is the
// per-group tile alignment; `quant` names the low-precision scale-layout dtype
// (the full ScaleLayout is resolved from the dtype at lowering).  The verifier
// enforces the family set, a power-of-two alignment, and (T,K)·(E,K,N)->(T,N)
// shape consistency — making the contract verifiable at the IR level, not just
// runtime kwargs.
def Tessera_GroupedGemmOp : Op<
  Tessera_Dialect, "grouped_gemm",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "Ragged grouped matmul (MoE expert FFN) with a grouped-layout contract";
  let description = [{
    Ragged grouped matmul. The optional `x_scale` / `w_scale` operands are the
    low-precision (de)quantization scale tensors for the activations and expert
    weights; their layout is described by the `scale_layout` attribute (granularity
    / block / packing / transposed). `numeric_policy` records the storage/accum
    coupling (e.g. fp8 storage with fp32 accumulation — DeepGEMM's two-level
    accumulation). `AttrSizedOperandSegments` separates the three required operands
    from the optional scale operands so existing 3-operand forms keep their
    semantics. The legacy `quant` dtype-name string is retained for back-compat;
    new emitters should attach `scale_layout` + the scale operands instead.
  }];
  let arguments = (ins TensorType:$x,
                        TensorType:$weights,
                        TensorType:$group_sizes,
                        Optional<TensorType>:$x_scale,
                        Optional<TensorType>:$w_scale,
                        DefaultValuedAttr<StrAttr, "\"contiguous\"">:$grouped_kind,
                        OptionalAttr<I64Attr>:$grouped_alignment,
                        OptionalAttr<StrAttr>:$quant,
                        OptionalAttr<Tessera_ScaleLayoutAttr>:$scale_layout,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$result);
  // Custom format keeps the bare 3-operand form unchanged (back-compat) and adds
  // an optional `scales(%x_scale, %w_scale)` clause; operandSegmentSizes is
  // auto-managed (not printed) since presence is derived from the optional group.
  let assemblyFormat = [{
    $x `,` $weights `,` $group_sizes
    (`scales` `(` $x_scale^ `,` $w_scale `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

// Local SwiGLU-fused MoE expert-FFN block (MegaMoE precursor, 2026-06).  Fuses
// the expert feed-forward core into one Graph-IR op: gate/up grouped GEMMs ->
// silu_mul -> down grouped GEMM, carrying the same grouped-layout contract as
// GroupedGemmOp.  x (T,K); w_gate/w_up (E,K,F); w_down (E,F,N); group_sizes
// (E,) -> result (T,N).
def Tessera_MoeSwigluBlockOp : Op<
  Tessera_Dialect, "moe_swiglu_block",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "Local SwiGLU-fused MoE expert FFN over grouped tokens";
  let description = [{
    SwiGLU-fused MoE expert FFN. Optional scale operands carry the low-precision
    (de)quant scales for the activations (`x_scale`) and the three expert weight
    tensors (`w_gate_scale` / `w_up_scale` / `w_down_scale`); `scale_layout`
    describes their packed layout and `numeric_policy` the storage/accum coupling.
    `AttrSizedOperandSegments` keeps the existing 5-operand form valid. Legacy
    `quant` string retained for back-compat.
  }];
  let arguments = (ins TensorType:$x,
                        TensorType:$w_gate,
                        TensorType:$w_up,
                        TensorType:$w_down,
                        TensorType:$group_sizes,
                        Optional<TensorType>:$x_scale,
                        Optional<TensorType>:$w_gate_scale,
                        Optional<TensorType>:$w_up_scale,
                        Optional<TensorType>:$w_down_scale,
                        DefaultValuedAttr<StrAttr, "\"contiguous\"">:$grouped_kind,
                        OptionalAttr<I64Attr>:$grouped_alignment,
                        OptionalAttr<StrAttr>:$quant,
                        OptionalAttr<Tessera_ScaleLayoutAttr>:$scale_layout,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$result);
  // Bare 5-operand form unchanged; optional all-or-nothing scale clause for the
  // activation + 3 expert weights. operandSegmentSizes auto-managed.
  let assemblyFormat = [{
    $x `,` $w_gate `,` $w_up `,` $w_down `,` $group_sizes
    (`scales` `(` $x_scale^ `,` $w_gate_scale `,` $w_up_scale `,` $w_down_scale `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

// Fused dequantize-into-GEMM (model-class roadmap M5.1, 2026-06): the packed
// low-precision weight codes + a separate scale operand are consumed in one op
// that dequantizes inside the contraction with an fp32 accumulator. Distinct
// from MatmulOp (dense) — the weight is a packed INT4/INT8/FP8 code tensor and
// `w_scale` is a load-bearing per-group scale operand whose layout rides on
// `scale_layout`. `weight_dtype` names the packed code format. The reference +
// Apple GPU composed lane lives in python/tessera/stdlib/quant.py.
def Tessera_DequantMatmulOp : Op<
  Tessera_Dialect, "dequant_matmul",
  [Pure]
> {
  let summary = "Fused dequantize-into-GEMM: y = x @ dequant(w_codes) (fp32 accum)";
  let description = [{
    `y = x @ dequantize(w_codes, w_scale)` with an fp32 accumulator. `w_codes`
    holds packed low-precision weight codes (INT4 nibble-packed, INT8, or an FP8
    grid); the optional `w_scale` operand is the separate per-group / per-channel
    scale tensor whose packed layout is described by `scale_layout`.
    `weight_dtype` is one of {int4, int8, fp8_e4m3, fp8_e5m2}; `quant_group_size`
    (when present) is the group extent along the contraction axis (omitted =
    per-channel). `numeric_policy` records the storage/accum coupling.
  }];
  let arguments = (ins TensorType:$x,
                        TensorType:$w_codes,
                        Optional<TensorType>:$w_scale,
                        StrAttr:$weight_dtype,
                        OptionalAttr<I64Attr>:$quant_group_size,
                        OptionalAttr<Tessera_ScaleLayoutAttr>:$scale_layout,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$result);
  let assemblyFormat = [{
    $x `,` $w_codes (`scale` `(` $w_scale^ `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

// Quantized M-grouped grouped GEMM (M5.1): the MoE expert-FFN core where each
// expert's weight is a packed low-precision code tensor + per-group scales.
// Token block e is dequant-matmul'd by expert e. Mirrors GroupedGemmOp's
// grouped-layout contract but with the weight carried as packed codes.
def Tessera_DequantGroupedGemmOp : Op<
  Tessera_Dialect, "dequant_grouped_gemm",
  [Pure]
> {
  let summary = "Quantized ragged grouped matmul (packed expert weights + scales)";
  let description = [{
    Quantized grouped GEMM: x (T,K) tokens sorted by expert; `w_codes` (E,K,N)
    packed low-precision expert weights; `group_sizes` (E,) per-expert token
    counts; optional `w_scale` per-group scale operand (`scale_layout` describes
    its layout). `weight_dtype` names the packed code format; `numeric_policy`
    the storage/accum coupling. Row block e of the result is the dequant-matmul
    of x[block e] by expert e.
  }];
  let arguments = (ins TensorType:$x,
                        TensorType:$w_codes,
                        TensorType:$group_sizes,
                        Optional<TensorType>:$w_scale,
                        DefaultValuedAttr<StrAttr, "\"contiguous\"">:$grouped_kind,
                        StrAttr:$weight_dtype,
                        OptionalAttr<I64Attr>:$quant_group_size,
                        OptionalAttr<Tessera_ScaleLayoutAttr>:$scale_layout,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$result);
  let assemblyFormat = [{
    $x `,` $w_codes `,` $group_sizes (`scale` `(` $w_scale^ `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

// DiffusionGemma block-diffusion canvas-denoise step — a structured region the
// compiler sees as one op (vs. the unrolled per-layer attention/MoE graph). It
// marks the bidirectional canvas denoiser over committed encoder/context KV;
// lowering recognizes it as an Apple GPU metal_runtime region (the per-head
// attention rides the attn_bias/flash_attn lane, the MoE the moe_swiglu_block
// lane). See python/tessera/models/block_diffusion_runtime.py for the executable
// reference.
def Tessera_DiffusionBlockStepOp : Op<
  Tessera_Dialect, "diffusion_block_step",
  [Pure]
> {
  let summary = "DiffusionGemma block-diffusion canvas-denoise step (structured region)";
  let arguments = (ins TensorType:$canvas,
                        TensorType:$encoder_k,
                        TensorType:$encoder_v,
                        I64Attr:$num_denoise_layers,
                        I64Attr:$num_attention_heads,
                        I64Attr:$num_kv_heads,
                        I64Attr:$head_dim,
                        DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$out);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Stage 13 — RL post-training loss Graph IR surface. These ops make PPO/GRPO/
// CISPO compiler-visible at the Graph layer. Only PPO's strict fp32 mean
// envelope is promoted to Apple GPU value execution in this stage; GRPO/CISPO
// remain compiler-visible / non-executable until they either decompose cleanly
// or gain their own runtime proof.
def Tessera_RLNormalizeGroupAdvantagesOp : Op<
  Tessera_Dialect, "rl.normalize_group_advantages",
  [Pure]
> {
  let summary = "Normalize rewards/advantages across a group axis";
  let arguments = (ins TensorType:$rewards,
                        I64Attr:$group_axis,
                        OptionalAttr<F64Attr>:$eps);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_RLPPOPolicyLossOp : Op<
  Tessera_Dialect, "rl.ppo_policy_loss",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "PPO clipped policy loss";
  let arguments = (ins TensorType:$logp_new, TensorType:$logp_old,
                        TensorType:$advantages,
                        Optional<TensorType>:$mask,
                        Optional<TensorType>:$ref_logp,
                        Optional<TensorType>:$entropy,
                        OptionalAttr<F64Attr>:$clip_epsilon,
                        OptionalAttr<F64Attr>:$kl_coef,
                        OptionalAttr<F64Attr>:$entropy_coef,
                        OptionalAttr<StrAttr>:$reduction);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_RLGRPOPolicyLossOp : Op<
  Tessera_Dialect, "rl.grpo_policy_loss",
  [Pure]
> {
  let summary = "GRPO policy loss";
  let arguments = (ins TensorType:$logp_new, TensorType:$logp_old,
                        TensorType:$rewards,
                        I64Attr:$group_axis,
                        OptionalAttr<F64Attr>:$clip_epsilon,
                        OptionalAttr<F64Attr>:$kl_coef,
                        OptionalAttr<StrAttr>:$reduction);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_RLCISPOPolicyLossOp : Op<
  Tessera_Dialect, "rl.cispo_policy_loss",
  [Pure]
> {
  let summary = "CISPO policy loss";
  let arguments = (ins TensorType:$logp_new, TensorType:$logp_old,
                        TensorType:$rewards,
                        I64Attr:$group_axis,
                        OptionalAttr<F64Attr>:$epsilon_high,
                        OptionalAttr<F64Attr>:$kl_coef,
                        OptionalAttr<StrAttr>:$reduction);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// EBM value-lane pilot — generic Graph IR spelling for the tensor-shaped EBM
// kernels that can traverse Graph -> Tile -> Apple Value Target IR. These are
// deliberately separate from the older `tessera_ebm` solver dialect: the solver
// dialect remains the domain-specific inspection/annotation surface, while
// these ops participate in the canonical value handoff.
def Tessera_EBMEnergyQuadraticOp : Op<
  Tessera_Dialect, "ebm.energy_quadratic",
  [Pure]
> {
  let summary = "Quadratic EBM energy E_b = 0.5 * ||x_b - y_b||^2";
  let arguments = (ins TensorType:$x, TensorType:$y);
  let results = (outs TensorType:$energies);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMLangevinStepOp : Op<
  Tessera_Dialect, "ebm.langevin_step",
  [Pure]
> {
  let summary = "One affine EBM Langevin step y - eta*grad + noise_scale*noise";
  let arguments = (ins TensorType:$y, TensorType:$grad,
                        Optional<TensorType>:$noise,
                        F64Attr:$eta,
                        OptionalAttr<F64Attr>:$temperature,
                        OptionalAttr<F64Attr>:$noise_scale);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMInnerStepOp : Op<
  Tessera_Dialect, "ebm.inner_step",
  [Pure]
> {
  let summary = "EBM inner step y - eta*grad with optional injected noise";
  let arguments = (ins TensorType:$y, TensorType:$grad,
                        Optional<TensorType>:$noise,
                        F64Attr:$eta,
                        OptionalAttr<F64Attr>:$noise_scale);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMRefinementOp : Op<
  Tessera_Dialect, "ebm.refinement",
  [Pure]
> {
  let summary = "T-step EBM refinement loop with optional noise controls";
  let arguments = (ins TensorType:$y, TensorType:$grad,
                        Optional<TensorType>:$noise,
                        F64Attr:$eta,
                        I64Attr:$steps,
                        OptionalAttr<F64Attr>:$temperature,
                        OptionalAttr<F64Attr>:$noise_scale);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMLangevinStepPhiloxOp : Op<
  Tessera_Dialect, "ebm.langevin_step_philox",
  [Pure]
> {
  let summary = "EBM Langevin step with Philox RNG state";
  let arguments = (ins TensorType:$y, TensorType:$grad,
                        TensorType:$seed, TensorType:$counter,
                        F64Attr:$eta,
                        F64Attr:$temperature,
                        OptionalAttr<F64Attr>:$noise_scale);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMDecodeInitOp : Op<
  Tessera_Dialect, "ebm.decode_init",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "Initialize K EBM candidate trajectories";
  let arguments = (ins TensorType:$x,
                        Optional<TensorType>:$base,
                        Optional<TensorType>:$noise,
                        I64Attr:$steps,
                        StrAttr:$strategy,
                        OptionalAttr<I64Attr>:$seed);
  let results = (outs TensorType:$candidates);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMSelfVerifyOp : Op<
  Tessera_Dialect, "ebm.self_verify",
  [Pure]
> {
  let summary = "Select or reduce EBM candidates from per-candidate energies";
  let arguments = (ins TensorType:$energies, TensorType:$candidates,
                        OptionalAttr<StrAttr>:$reduction,
                        OptionalAttr<F64Attr>:$temperature);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMPartitionExactOp : Op<
  Tessera_Dialect, "ebm.partition_exact",
  [Pure]
> {
  let summary = "Exact EBM partition function / logsumexp over energies";
  let arguments = (ins TensorType:$energies,
                        OptionalAttr<F64Attr>:$temperature,
                        OptionalAttr<StrAttr>:$reduction);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMBivectorLangevinStepOp : Op<
  Tessera_Dialect, "ebm.bivector_langevin_step",
  [Pure]
> {
  let summary = "Langevin step constrained to a bivector grade subspace";
  let arguments = (ins TensorType:$state, TensorType:$grad,
                        Optional<TensorType>:$noise,
                        F64Attr:$eta,
                        OptionalAttr<F64Attr>:$temperature,
                        OptionalAttr<F64Attr>:$noise_scale,
                        OptionalAttr<I64Attr>:$grade,
                        OptionalAttr<StrAttr>:$projection,
                        OptionalAttr<StrAttr>:$manifold,
                        OptionalAttr<StrAttr>:$metric);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_EBMSphereLangevinStepOp : Op<
  Tessera_Dialect, "ebm.sphere_langevin_step",
  [Pure]
> {
  let summary = "Riemannian Langevin step on a unit sphere";
  let arguments = (ins TensorType:$state, TensorType:$grad,
                        Optional<TensorType>:$noise,
                        F64Attr:$eta,
                        OptionalAttr<F64Attr>:$temperature,
                        OptionalAttr<F64Attr>:$noise_scale,
                        OptionalAttr<StrAttr>:$projection,
                        OptionalAttr<StrAttr>:$manifold,
                        OptionalAttr<StrAttr>:$metric,
                        OptionalAttr<BoolAttr>:$normalized_state);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// GA / Clifford Graph IR hardening. These ops pin algebra metadata at the
// canonical Graph IR layer; backend execution still goes through the existing
// GA manifest/JIT bridge until a value handoff is explicitly promoted.
class Tessera_CliffordBinaryOp<string mnemonic, string summaryText>
    : Op<Tessera_Dialect, mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins TensorType:$lhs, TensorType:$rhs,
                        I64Attr:$p, I64Attr:$q,
                        OptionalAttr<StrAttr>:$coefficient_layout,
                        OptionalAttr<I64ArrayAttr>:$signature);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

class Tessera_CliffordUnaryOp<string mnemonic, string summaryText>
    : Op<Tessera_Dialect, mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins TensorType:$input,
                        I64Attr:$p, I64Attr:$q,
                        OptionalAttr<StrAttr>:$coefficient_layout,
                        OptionalAttr<I64ArrayAttr>:$signature);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_CliffordGeometricProductOp
    : Tessera_CliffordBinaryOp<"clifford.geometric_product",
                               "Clifford geometric product">;
def Tessera_CliffordOuterProductOp
    : Tessera_CliffordBinaryOp<"clifford.outer_product",
                               "Clifford outer/wedge product">;
def Tessera_CliffordInnerProductOp
    : Tessera_CliffordBinaryOp<"clifford.inner_product",
                               "Clifford inner product">;
def Tessera_CliffordReverseOp
    : Tessera_CliffordUnaryOp<"clifford.reverse",
                              "Clifford reverse involution">;
def Tessera_CliffordNormOp
    : Tessera_CliffordUnaryOp<"clifford.norm",
                              "Clifford multivector norm">;

def Tessera_CliffordGradeProjectOp : Op<
  Tessera_Dialect, "clifford.grade_project",
  [Pure]
> {
  let summary = "Project Clifford coefficients to an allowed grade mask";
  let arguments = (ins TensorType:$input,
                        I64Attr:$p, I64Attr:$q,
                        I64ArrayAttr:$grade_mask,
                        OptionalAttr<StrAttr>:$coefficient_layout,
                        OptionalAttr<I64ArrayAttr>:$signature);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_CliffordRotorSandwichOp : Op<
  Tessera_Dialect, "clifford.rotor_sandwich",
  [Pure]
> {
  let summary = "Apply rotor sandwich R x reverse(R) to a multivector";
  let arguments = (ins TensorType:$rotor, TensorType:$value,
                        I64Attr:$p, I64Attr:$q,
                        OptionalAttr<StrAttr>:$coefficient_layout,
                        OptionalAttr<I64ArrayAttr>:$signature,
                        OptionalAttr<I64ArrayAttr>:$grade_mask);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// L-series linalg pilot (2026-06-02): Cholesky is the carrier op for vetting
// the full Graph → Schedule → Tile → Target spine on the Apple CPU + GPU
// backends.  Kept deliberately minimal at the Graph-IR layer ([Pure] + a
// shape verifier); TilingInterface / AdjointInterface are added by the steps
// (L3 / autodiff) that actually need them.
def Tessera_CholeskyOp : Op<
  Tessera_Dialect, "cholesky",
  [Pure]
> {
  let summary = "Cholesky factorization (A = L Lᵀ for SPD A; returns L)";
  let arguments = (ins TensorType:$a,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "true">:$lower);
  let results = (outs TensorType:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// L-series linalg family (2026-06-02): the next 5 linalg members, each carried
// through the full Graph→Schedule→Tile→Target Apple spine via the table-driven
// TileLinalg pattern + kLinalgSpecs.  Kept minimal at the Graph-IR layer
// ([Pure] + a shape verifier).  Multi-operand (tri_solve / cholesky_solve) and
// multi-result (lu / qr / svd) ops build out the foundation for those shapes.

// Triangular solve: solve op(A) X = B for X, A lower/upper triangular.
def Tessera_TriSolveOp : Op<Tessera_Dialect, "tri_solve", [Pure]> {
  let summary = "Triangular solve op(A) X = B (A triangular, NxN; B/X NxK)";
  let arguments = (ins TensorType:$a, TensorType:$b,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "true">:$lower,
                        DefaultValuedAttr<BoolAttr, "false">:$trans,
                        DefaultValuedAttr<BoolAttr, "false">:$unit_diag);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// SPD solve: solve A X = B for X via the Cholesky factorization of SPD A.
def Tessera_CholeskySolveOp : Op<Tessera_Dialect, "cholesky_solve", [Pure]> {
  let summary = "SPD solve A X = B via Cholesky (A SPD NxN; B/X NxK)";
  let arguments = (ins TensorType:$a, TensorType:$b,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "true">:$lower);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// LU factorization with partial pivoting: A = P L U (packed LU + pivots).
def Tessera_LUOp : Op<Tessera_Dialect, "lu", [Pure]> {
  let summary = "LU factorization with partial pivoting (A NxN → packed LU + pivots)";
  let arguments = (ins TensorType:$a,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$lu, TensorType:$pivots);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// QR factorization (reduced): A = Q R, Q (MxN) orthonormal cols, R (NxN) upper.
def Tessera_QROp : Op<Tessera_Dialect, "qr", [Pure]> {
  let summary = "QR factorization (A MxN → Q MxN, R NxN), reduced";
  let arguments = (ins TensorType:$a,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "false">:$full_matrices);
  let results = (outs TensorType:$q, TensorType:$r);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Singular value decomposition: A = U diag(S) Vᵀ.
def Tessera_SVDOp : Op<Tessera_Dialect, "svd", [Pure]> {
  let summary = "Singular value decomposition (A MxN → U, S, V)";
  let arguments = (ins TensorType:$a,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        DefaultValuedAttr<BoolAttr, "false">:$full_matrices);
  let results = (outs TensorType:$u, TensorType:$s, TensorType:$v);
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
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "FlashAttention fused op";
  let description = [{
    Fused scaled-dot-product attention. When the optional `attn_bias` operand is
    present it is an additive score bias of shape `(B, Sq, Sk)` (broadcastable
    from `(1, Sq, Sk)`), applied as `softmax(scale * Q*K^T + attn_bias) * V` —
    matching SDPA's additive-mask convention. This is the general substrate for
    structured masks (e.g. DFlash sliding-layer block masks); the bias-free path
    is unchanged. `AttrSizedOperandSegments` separates the `v` variadic from the
    optional bias, so existing 2-/3-operand forms keep their semantics.
  }];
  let arguments = (ins TensorType:$q, AnyType:$k_or_cache, Variadic<AnyType>:$v,
                        Optional<TensorType>:$attn_bias,
                        I64Attr:$head_dim,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        OptionalAttr<F64Attr>:$dropout_p,
                        DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_VarlenSdpaOp : Op<
  Tessera_Dialect, "varlen_sdpa", [Pure]
> {
  let summary = "Variable-length (packed-sequence) scaled-dot-product attention";
  let description = [{
    Packed-sequence SDPA: queries, keys and values are *packed* streams of shape
    `(H, total_q, head_dim)` / `(H, total_k, head_dim)` (heads folded into the
    leading axis), and the `cu_seqlens_q` / `cu_seqlens_k` rank-1 integer operands
    are the cumulative per-sample length offsets (prefix sums, first element 0).
    Sample `i` owns query rows `[cu_q[i], cu_q[i+1])` and key rows
    `[cu_k[i], cu_k[i+1])`; a query attends only within its own sample's key block
    (block-diagonal), bidirectionally by default or bottom-right-aligned causal
    when `causal` is set.

    The two cu_seqlens operands are **independent** — the rectangular case
    (`cu_seqlens_q != cu_seqlens_k`, each query block attending a longer key
    block) is first-class, not an approximation. This is the IR contract behind
    the NVIDIA Cosmos 3 "two-way flat attention" (§5.2.2): a causal launch over
    the Reasoner tokens and a bidirectional launch over the interleaved
    `[Reasoner; Generator]` key/value block. The matching additive-mask form is
    `tessera.flash_attn` with `attn_bias`; the two are provably equal (the
    metamorphic oracle in tests/unit/test_varlen_sdpa.py).

    Lowering: the reference + Apple GPU `metal_runtime` lane decompose this into
    per-block `flash_attn` dispatches; a fused varlen kernel (FA-3 on Hopper,
    NATTEN/CUTLASS on Blackwell) stays behind the Phase G/H backend gate.
  }];
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        TensorType:$cu_seqlens_q, TensorType:$cu_seqlens_k,
                        I64Attr:$head_dim,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy,
                        OptionalAttr<F64Attr>:$scale,
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
  // Sprint V4b (2026-05-22): shape-preserving position embedding.
  // Input and output ranks + static dim sizes must agree.
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def Tessera_GQAAttentionOp : Op<Tessera_Dialect, "gqa_attention", [Pure]> {
  let summary = "Grouped-query attention wrapper";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$num_query_heads,
                        I64Attr:$num_kv_heads);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_MQAAttentionOp : Op<Tessera_Dialect, "mqa_attention", [Pure]> {
  let summary = "Multi-query attention wrapper";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_MLADecodeOp : Op<Tessera_Dialect, "mla_decode", [Pure]> {
  let summary = "MLA decode wrapper over latent K/V expansion and attention";
  let arguments = (ins TensorType:$q, TensorType:$k_latent, TensorType:$v_latent,
                        Variadic<TensorType>:$weights);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_QuantizeFP8Op : Op<Tessera_Dialect, "quantize_fp8", [Pure]> {
  let summary = "Reference FP8 quantize with dequantized fp32 payload plus scale";
  let arguments = (ins TensorType:$x, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x_q, AnyType:$scale);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_DequantizeFP8Op : Op<Tessera_Dialect, "dequantize_fp8", [Pure]> {
  let summary = "Reference FP8 dequantize";
  let arguments = (ins TensorType:$x_q, AnyType:$scale, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_QuantizeFP4Op : Op<Tessera_Dialect, "quantize_fp4", [Pure]> {
  let summary = "Reference FP4 quantize with dequantized fp32 payload plus scale";
  let arguments = (ins TensorType:$x, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x_q, AnyType:$scale);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_DequantizeFP4Op : Op<Tessera_Dialect, "dequantize_fp4", [Pure]> {
  let summary = "Reference FP4 dequantize";
  let arguments = (ins TensorType:$x_q, AnyType:$scale, OptionalAttr<StrAttr>:$format);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_MoeDispatchOp : Op<Tessera_Dialect, "moe_dispatch", [MemoryEffects<[MemWrite]>]> {
  let summary = "MoE token dispatch transport op";
  let arguments = (ins TensorType:$x, TensorType:$route);
  let results = (outs TensorType:$partials);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  // Sprint V1 (2026-05-22): leading-axis token-count match between
  // the input token tensor ``x`` and the expert routing tensor
  // ``route``.  Both must agree on dim 0 when statically known.
  let hasVerifier = 1;
}

def Tessera_MoeCombineOp : Op<Tessera_Dialect, "moe_combine", [MemoryEffects<[MemWrite]>]> {
  let summary = "MoE token combine transport op";
  let arguments = (ins TensorType:$partials, TensorType:$inverse_route,
                        DefaultValuedAttr<StrAttr, "\"sum\"">:$reduce);
  let results = (outs TensorType:$x);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
                        DefaultValuedAttr<StrAttr, "\"fp32\"">:$state_dtype,
                        // Track L L3.1 — `erase=false` (default) is gated linear
                        // attention (backward-compatible); `erase=true` is the
                        // genuine DeltaNet `(I − β k kᵀ)` rule.  The runtime
                        // honors this (apple_gpu routes to the genuine kernel).
                        DefaultValuedAttr<BoolAttr, "false">:$erase);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_GatedDeltaNetOp
  : Tessera_DeltaAttentionOp<"gated_deltanet", "Gated DeltaNet recurrent attention"> {
  let hasVerifier = 1;
}
def Tessera_KimiDeltaAttentionOp
  : Tessera_DeltaAttentionOp<"kimi_delta_attention", "Kimi Delta Attention recurrent op"> {
  let hasVerifier = 1;
}
def Tessera_ModifiedDeltaAttentionOp
  : Tessera_DeltaAttentionOp<"modified_delta_attention", "Modified bounded Delta Attention recurrent op"> {
  let hasVerifier = 1;
}

// Mamba-2 selective state-space model (SSD).  Track L L4 — this is the op the
// coverage registry previously *claimed* had landed; it now genuinely exists.
def Tessera_SelectiveSsmOp : Op<
  Tessera_Dialect, "selective_ssm",
  [Pure, AttrSizedOperandSegments]
> {
  let summary = "Mamba-2 selective state-space model (SSD) recurrence";
  let description = [{
    Selective state-space recurrence (Mamba-2 / SSD).  Per token t, channel d,
    state dim n:
      A_bar = exp(delta[b,t,d] * a[d,n]);  B_bar = delta[b,t,d] * b[b,t,n]
      h[t,d,n] = A_bar * h[t-1,d,n] + B_bar * x[b,t,d]
      y[t,d]   = sum_n c[b,t,n] * h[t,d,n]
    `a` is the (negative) state-matrix diagonal — rank-1 `(D)` (scalar-state, the
    common per-head config that lowers to the chunked-parallel SSD matmul form)
    or rank-2 `(D,N)`.  `b`/`c` are the input/output projections `(B,S,N)`,
    `delta` the selective time-step `(B,S,D)`.  Optional `gate` (output gate,
    shape of `x`) and `state` (initial carry `(B,D,N)`).  Reference:
    `tessera.ops.selective_ssm`; chunked-parallel lowering:
    `python/tessera/_mamba_ssd.py`.
  }];
  let arguments = (ins TensorType:$x,
                        TensorType:$a,
                        TensorType:$b,
                        TensorType:$c,
                        TensorType:$delta,
                        Optional<TensorType>:$gate,
                        Optional<TensorType>:$state,
                        DefaultValuedAttr<I64Attr, "128">:$chunk_size);
  let results = (outs TensorType:$y);
  let assemblyFormat = [{
    $x `,` $a `,` $b `,` $c `,` $delta
    (`gate` `(` $gate^ `)`)? (`init` `(` $state^ `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

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
  let hasVerifier = 1;
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
  // Sprint V1 (2026-05-22): shape-preserving normalization.  Input
  // and output ranks + static dims must agree.  ``eps`` (when set)
  // must be positive — required for ``rsqrt(var + eps)`` stability
  // per MEMORY_MODEL_SPEC §7 numerics.
  let hasVerifier = 1;
}

def Tessera_SoftmaxOp : Tessera_UnaryPureOp<"softmax", "Softmax over an explicit or innermost axis"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  // Sprint V4b (2026-05-22): shape-preserving normalization over an
  // explicit axis.  If ``axis`` is set, ``-rank <= axis < rank``.
  // Input and output ranks + static dim sizes must agree.
  let hasVerifier = 1;
}

def Tessera_GeluOp : Tessera_UnaryPureOp<"gelu", "Gaussian error linear unit">;
def Tessera_ReluOp : Tessera_UnaryPureOp<"relu", "Rectified linear unit">;
def Tessera_SigmoidOp : Tessera_UnaryPureOp<"sigmoid", "Sigmoid activation">;
def Tessera_SinOp : Tessera_UnaryPureOp<"sin", "Sine elementwise op">;
// Tier-1 activations lowered by the Apple GPU MPSGraph lane (2026-05-29).
def Tessera_SiluOp : Tessera_UnaryPureOp<"silu", "SiLU / swish activation (x * sigmoid(x))">;
def Tessera_TanhOp : Tessera_UnaryPureOp<"tanh", "Hyperbolic tangent activation">;
def Tessera_SoftplusOp : Tessera_UnaryPureOp<"softplus", "Softplus activation (log(1 + exp(x)))">;

// Phase 0 production-compiler spine (docs/spec/PRODUCTION_COMPILER_PLAN.md):
// the minimal *total* binary elementwise op used to prove the
// tessera -> linalg -> bufferize -> llvm -> ExecutionEngine boundary
// (RUNTIME_ABI_SPEC.md §12). Deliberately trivial — no broadcasting;
// SameOperandsAndResultType enforces the elementwise contract without a
// custom verifier.
def Tessera_AddOp : Op<
  Tessera_Dialect, "add",
  [Pure]
> {
  let summary = "Elementwise tensor addition (a + b)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}

// Phase 1 production-compiler binary elementwise family. Same shape as
// Tessera_AddOp — trivial, total, lowered by `tessera-to-linalg` through a
// shared name→arith-kind table. sub/mul are the minimum needed before
// softmax/normalization arrive (softmax = max → sub → exp → div).
def Tessera_SubOp : Op<Tessera_Dialect, "sub", [Pure]> {
  let summary = "Elementwise tensor subtraction (a - b)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}
def Tessera_MulOp : Op<Tessera_Dialect, "mul", [Pure]> {
  let summary = "Elementwise tensor multiplication (a * b)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}
def Tessera_ScoreCombineOp : Op<Tessera_Dialect, "score_combine", [Pure]> {
  let summary = "Diffusion guidance score composition (base + gamma * delta)";
  let arguments = (ins TensorType:$base, TensorType:$delta,
                       DefaultValuedAttr<F64Attr, "1.0">:$gamma);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// CGG orchestration marker. This is intentionally not a fused model kernel: it
// records that a library-level guided-denoise region produced the visible
// score-combine chain for a given timestep/schedule/gamma. Lowerings may use it
// for audit/scheduling metadata, but the executable math remains the ordinary
// denoiser outputs plus `tessera.score_combine`.
def Tessera_GuidedDenoiseRegionOp : Op<
  Tessera_Dialect, "guided_denoise_region",
  [Pure]
> {
  let summary = "Guided denoise orchestration marker for CGG-style score composition";
  let arguments = (ins TensorType:$ref_score, TensorType:$favored_score,
                       TensorType:$unfavored_score,
                       I64Attr:$timestep,
                       StrAttr:$schedule,
                       F64Attr:$gamma,
                       OptionalAttr<StrAttr>:$preference);
  let results = (outs TensorType:$guided_score);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def Tessera_DivOp : Op<Tessera_Dialect, "div", [Pure]> {
  let summary = "Elementwise tensor division (a / b)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}

// Phase 2 Sprint 2.1 — data-parallel conditional (attention masking). select is
// the general elementwise ternary (cond != 0 ? a : b); masked_fill is the
// attention-masking form (mask != 0 ? x : value), with `value` a scalar attr
// (e.g. a large negative for causal masking before softmax).
def Tessera_SelectOp : Op<Tessera_Dialect, "select", [Pure]> {
  let summary = "Elementwise select (cond != 0 ? a : b)";
  let arguments = (ins TensorType:$cond, TensorType:$a, TensorType:$b);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}
def Tessera_MaskedFillOp : Op<Tessera_Dialect, "masked_fill", [Pure]> {
  let summary = "Elementwise masked fill (mask != 0 ? x : value)";
  let arguments = (ins TensorType:$x, TensorType:$mask, F64Attr:$value);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Phase 2 state — functional KV-cache update. Writes `value` (1 x D) into row
// `row` of `buffer` (T x D), returning the updated buffer (value-semantic; the
// caller threads the cache across decode steps). Lowers to tensor.insert_slice.
def Tessera_WriteRowOp : Op<Tessera_Dialect, "write_row", [Pure]> {
  let summary = "Functional row update: buffer with row `row` set to value";
  let arguments = (ins TensorType:$buffer, TensorType:$value, I64Attr:$row);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Phase 1 Sprint 1.2 — single-axis reduction. One parameterized op
// (`kind` ∈ {sum, max, min, mean}, `axis`) lowered to linalg.reduce + an
// identity-fill init, mirroring how linalg.reduce is itself parameterized
// (dimensions + combiner region). This is the first production-lane op whose
// result rank differs from its input rank.
def Tessera_ReduceOp : Op<Tessera_Dialect, "reduce", [Pure]> {
  let summary = "Reduction over a single axis (sum/max/min/mean)";
  let arguments = (ins TensorType:$input, StrAttr:$kind, I64Attr:$axis);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// RMSNorm (unweighted) over the innermost dimension; eps optional.
def Tessera_RmsNormOp : Tessera_UnaryPureOp<"rmsnorm", "RMSNorm over the innermost dimension"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<F64Attr>:$eps,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let hasVerifier = 1;
}

// Log-softmax over an explicit or innermost axis.
def Tessera_LogSoftmaxOp : Tessera_UnaryPureOp<"log_softmax", "Log-softmax over an explicit or innermost axis"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let hasVerifier = 1;
}

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
                       TensorType:$w_down,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
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
  let hasVerifier = 1;
}

def Tessera_LatentKVExpandKOp : Op<
  Tessera_Dialect, "latent_kv_expand_k",
  [Pure]
> {
  let summary = "MLA latent K expansion: K = c @ W_uk";
  let arguments = (ins TensorType:$c, TensorType:$w_uk);
  let results = (outs TensorType:$k);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_LatentKVExpandVOp : Op<
  Tessera_Dialect, "latent_kv_expand_v",
  [Pure]
> {
  let summary = "MLA latent V expansion: V = c @ W_uv";
  let arguments = (ins TensorType:$c, TensorType:$w_uv);
  let results = (outs TensorType:$v);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
                       DefaultValuedAttr<BoolAttr, "false">:$causal,
                       OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def Tessera_AttnCompressedBlocksOp : Op<
  Tessera_Dialect, "attn_compressed_blocks",
  [Pure]
> {
  let summary = "NSA branch 2 — attention over compressed-block summaries";
  let arguments = (ins TensorType:$q, TensorType:$k_c, TensorType:$v_c);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

// MiniMax Sparse Attention (MSA, arXiv:2606.13392). Three Graph IR ops mirror
// the reference Python surface (see docs/msa.md): an exp-free per-GQA-group
// Index Branch scorer, a deterministic forced-local Top-k block selector, and
// an exact block-sparse Main Branch. Unlike deepseek_sparse_attention (which
// blends sliding/compressed/top-k branches per head), MSA is a single exact
// path with per-GQA-group selection. `force_local_block` is carried as a
// *semantic* attribute (not a heuristic) so Schedule/Tile IR may rely on each
// query's own block always being resident. Pure per call.
def Tessera_MSAIndexScoresOp : Op<
  Tessera_Dialect, "msa_index_scores",
  [Pure]
> {
  let summary = "MSA Index Branch — exp-free per-GQA-group KV-block scores";
  let arguments = (ins TensorType:$q, TensorType:$k,
                        I64Attr:$block_size,
                        OptionalAttr<F64Attr>:$scale);
  let results = (outs TensorType:$scores);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_MSASelectBlocksOp : Op<
  Tessera_Dialect, "msa_select_blocks",
  [Pure]
> {
  let summary = "MSA Top-k block selection (per GQA group, forced-local, deterministic)";
  let arguments = (ins TensorType:$scores,
                        I64Attr:$top_k,
                        I64Attr:$block_size,
                        DefaultValuedAttr<BoolAttr, "true">:$force_local_block,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$block_ids);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_MSASparseAttentionOp : Op<
  Tessera_Dialect, "msa_sparse_attention",
  [Pure]
> {
  let summary = "MiniMax Sparse Attention — exact block-sparse Main Branch over selected blocks";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$block_size,
                        I64Attr:$top_k,
                        DefaultValuedAttr<BoolAttr, "true">:$force_local_block,
                        DefaultValuedAttr<BoolAttr, "true">:$causal,
                        OptionalAttr<F64Attr>:$scale);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Lookahead Sparse Attention (LSA) — experimental, inference-only composite
// attention policy. Each query attends over the union of its causal local
// window and the tokens of the sigmoid-threshold-selected historical blocks.
// Pure per call (D2): tau is the caller-owned re-selection cadence and does not
// change a single forward's math. See docs/audit/domain/archive/lsa_scope.md.
def Tessera_LookaheadSparseAttentionOp : Op<
  Tessera_Dialect, "lookahead_sparse_attention",
  [Pure]
> {
  let summary = "Lookahead sparse attention — local window ∪ threshold-selected historical blocks";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$window_size,
                        I64Attr:$block_size,
                        DefaultValuedAttr<I64Attr, "64">:$tau,
                        DefaultValuedAttr<F64Attr, "0.5">:$threshold,
                        DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def Tessera_MorScatterOp : Op<
  Tessera_Dialect, "mor_scatter",
  [Pure]
> {
  let summary = "MoR scatter — write updated values back into full at masked positions";
  let arguments = (ins TensorType:$full, TensorType:$updated, TensorType:$mask);
  let results = (outs TensorType:$out);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_TransposeOp : Op<
  Tessera_Dialect, "transpose",
  [Pure]
> {
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  // Sprint V1 (2026-05-22): rank-preserving permutation.  Input and
  // output must have the same rank; the multiset of static dim sizes
  // must be a permutation (dynamic dims are unconstrained).  Element
  // type must match.
  let hasVerifier = 1;
  // Phase 1 (front-to-back closure plan): a no-perm (reverse-dims) transpose
  // is its own inverse — `transpose(transpose(x)) -> x`. Picked up by
  // `createCanonicalizerPass`, now wired into the tessera_jit pipeline so it
  // is observable end-to-end on the executed CPU path.
  let hasCanonicalizer = 1;
}

// Sprint V6a (2026-05-22): ReshapeOp registration.  V5
// `SymbolicDimEqualityPass` already dispatches on `tessera.reshape`
// by op name; this ODS registration activates the dispatch branch
// end-to-end (lit fixtures no longer need
// `--allow-unregistered-dialect`).
//
// Contract: input and output must have the same element type and
// the same product of static dims when both are concrete.  The
// reshape is shape-changing (rank may differ) but element-count
// preserving.  Per-op `tessera.dim_names_in` / `tessera.dim_names_out`
// attributes (read by SymbolicDimEqualityPass) are optional discardable
// attrs — they don't appear in the ODS argument list.
def Tessera_ReshapeOp : Op<
  Tessera_Dialect, "reshape",
  [Pure]
> {
  let summary = "Element-count-preserving tensor reshape";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  // Phase 1 (front-to-back closure plan): broaden folder coverage beyond the
  // arithmetic/cast ops. An identity reshape (result type == input type) folds
  // to its input — the same structural identity as CastOp — and a chain
  // reshape(reshape(x)) collapses to a single reshape to the final type. Both
  // are guarded against dropping the optional tessera.dim_names_in/out symbolic
  // -dim annotations that SymbolicDimEqualityPass reads.
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Tessera_CastOp : Op<
  Tessera_Dialect, "cast",
  [Pure]
> {
  let arguments = (ins TensorType:$x,
                        OptionalAttr<Tessera_NumericPolicyAttr>:$numeric_policy);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  // Sprint V4b (2026-05-22): shape-preserving element-type
  // conversion.  Input and output must have the same rank + static
  // dim sizes; only the element type may differ.
  let hasVerifier = 1;
  // Phase 1 (front-to-back closure plan): an identity cast (same type, no
  // numeric policy) folds to its input. A numeric_policy may carry a real
  // rounding/math-mode change, so identity-fold only when it is absent.
  let hasFolder = 1;
}

// ── Structural / layout-view ops (P1a, S_SERIES_GAP_CLOSURE_PLAN §6.A) ────────
// The 0-view metadata family — pure tensor reshape/permute views (the shape
// transform is encoded in the result type, mirroring ReshapeOp; an optional
// attr-dict carries axis/shape/perm detail). These give the 28 structural ops a
// Graph IR identity so a @jit body can lower through the compiled pipeline + the
// view/copy analysis (§6.B) instead of dropping to the numpy reference. Pure:
// no FLOP, no memory effect (their backend_kernel is `not_applicable`, P0).
// Frontend emission + the view-elision lowering are the P1b/§6.B follow-ups.
def Tessera_SqueezeOp : Op<Tessera_Dialect, "squeeze", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Remove size-1 dimensions (optional `axes` attr selects which)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_UnsqueezeOp : Op<Tessera_Dialect, "unsqueeze", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Insert size-1 dimensions (the `axes` attr selects positions)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_ExpandOp : Op<Tessera_Dialect, "expand", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Broadcast-expand to a larger shape (stride-0 view, no copy)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_BroadcastOp : Op<Tessera_Dialect, "broadcast", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Broadcast to a target shape (stride-0 view, no copy)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_PermuteOp : Op<Tessera_Dialect, "permute", [Pure, SameOperandsAndResultElementType]> {
  let summary = "General axis permutation (the `perm` attr is the new order)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_FlattenOp : Op<Tessera_Dialect, "flatten", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Collapse a contiguous dim range (optional `start`/`end` attrs)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_ViewOp : Op<Tessera_Dialect, "view", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Reshape sharing storage (contiguous view, no copy; `shape` attr)";
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  // A no-copy contiguous view cannot change the element count — mirror
  // ReshapeOp's static product check so an impossible alias shape (e.g.
  // 2x3 -> 5) is rejected, not trusted by later view-elision/lowering.
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Tessera_DropoutOp : Op<Tessera_Dialect, "dropout", [MemoryEffects<[MemWrite]>]> {
  let summary = "Reference random dropout with optional deterministic seed";
  // Sprint V4b (2026-05-22): dropout probability must be in [0, 1).
  // Shape is preserved; element type is preserved.
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
  [MemoryEffects<[MemWrite]>]
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
  let hasVerifier = 1;
}

def Tessera_FFTOp : Tessera_SpectralUnaryOp<"fft", "FFT spectral transform">;
def Tessera_IFFTOp : Tessera_SpectralUnaryOp<"ifft", "Inverse FFT spectral transform">;
def Tessera_RFFTOp : Tessera_SpectralUnaryOp<"rfft", "Real FFT spectral transform">;
def Tessera_IRFFTOp : Tessera_SpectralUnaryOp<"irfft", "Inverse real FFT spectral transform">;
def Tessera_DCTOp : Tessera_SpectralUnaryOp<"dct", "Discrete cosine transform"> {
  let arguments = (ins TensorType:$x,
                       OptionalAttr<I64Attr>:$axis,
                       OptionalAttr<I64Attr>:$type);
  let hasVerifier = 1;
}

def Tessera_SpectralConvOp : Op<Tessera_Dialect, "spectral_conv", [Pure]> {
  let summary = "Reference spectral convolution semantic op";
  let arguments = (ins TensorType:$x, TensorType:$filter);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_AdamOp : Op<Tessera_Dialect, "adam", [Pure]> {
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
  : Op<Tessera_Dialect, mnemonic, [Pure]> {
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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def Tessera_KVCacheCreateOp : Op<Tessera_Dialect, "kv_cache.create", [MemoryEffects<[MemWrite]>]> {
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

def Tessera_KVCacheAppendOp : Op<Tessera_Dialect, "kv_cache.append", [MemoryEffects<[MemWrite]>]> {
  let summary = "Append key/value tensors to a stateful KV cache";
  let arguments = (ins Tessera_KVCacheType:$cache, TensorType:$k, TensorType:$v);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_KVCachePruneOp : Op<Tessera_Dialect, "kv_cache.prune", [MemoryEffects<[MemWrite]>]> {
  let summary = "Prune a rolling-window KV cache to a fixed window";
  let arguments = (ins Tessera_KVCacheType:$cache, I64Attr:$window);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_CacheCommitOp : Op<
  Tessera_Dialect, "cache.commit", [MemoryEffects<[MemWrite]>]
> {
  let summary = "Commit an accepted speculative prefix — advance the cache cursor (SD1-3)";
  let description = [{
    Speculative-decode commit: advance a KV/SSM cache handle's cursor to keep
    exactly `accepted_length` tokens — the accepted prefix becomes part of the
    sequence and the rejected draft tail is discarded. This is the IR-visible,
    **typed-effect** form of `tessera.speculative.advance_kv`: it carries
    `MemoryEffects<[MemWrite]>` (the `EffectLattice` classifies it as `state`, not
    `pure`), so the speculative loop lowers WITHOUT hidden Python cache mutation
    (CF0: "cache mutation only through typed cache handles"). The handle is
    threaded value-to-value (`cache → updated`) so the mutation is linear.
  }];
  let arguments = (ins Tessera_KVCacheType:$cache, Index:$accepted_length);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CacheRollbackOp : Op<
  Tessera_Dialect, "cache.rollback", [MemoryEffects<[MemWrite]>]
> {
  let summary = "Roll back rejected speculative drafts — rewind the cache cursor (SD1-3)";
  let description = [{
    Speculative-decode rollback: rewind a KV/SSM cache handle's cursor over the
    last `num_rejected` draft tokens (KV: trim; SSM/ReplaySSM: ring-buffer cursor
    rewind, no per-position snapshot). The IR-visible, **typed-effect** form of
    the speculative rollback that `tessera.speculative.advance_ssm` /
    `KVCacheHandle.trim` perform — `MemoryEffects<[MemWrite]>`, classified `state`
    by the `EffectLattice` (CF0: "speculative rollback via cursor/state handles,
    not implicit Python state"). The handle is threaded `cache → updated`.
  }];
  let arguments = (ins Tessera_KVCacheType:$cache, Index:$num_rejected);
  let results = (outs Tessera_KVCacheType:$updated);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CachePageLookupOp : Op<Tessera_Dialect, "cache.page_lookup", [MemoryEffects<[MemRead]>]> {
  let summary = "Lookup a logical page in a stateful cache";
  let arguments = (ins Tessera_KVCacheType:$cache, Index:$position);
  let results = (outs Tessera_CachePageType:$page);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_RingCreateOp : Op<Tessera_Dialect, "ring.create", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a stateful ring buffer";
  let arguments = (ins I64Attr:$capacity);
  let results = (outs Tessera_RingType:$ring);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchParameterOp : Op<Tessera_Dialect, "arch.parameter", [MemoryEffects<[MemWrite]>]> {
  let summary = "Declare FP32 architecture logits separate from model weights";
  let arguments = (ins I64Attr:$size,
                       OptionalAttr<StrAttr>:$name,
                       DefaultValuedAttr<F64Attr, "0.0">:$init);
  let results = (outs Tessera_ArchParamType:$alpha);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchGumbelSoftmaxOp : Op<Tessera_Dialect, "arch.gumbel_softmax", [MemoryEffects<[MemWrite]>]> {
  let summary = "Differentiable Gumbel-Softmax relaxation over architecture logits";
  let arguments = (ins Tessera_ArchParamType:$alpha,
                       F64Attr:$temperature,
                       OptionalAttr<I64Attr>:$seed);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchHardConcreteOp : Op<Tessera_Dialect, "arch.hard_concrete", [MemoryEffects<[MemWrite]>]> {
  let summary = "Hard-concrete sparse relaxation over architecture logits";
  let arguments = (ins Tessera_ArchParamType:$alpha,
                       F64Attr:$temperature);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  let hasVerifier = 1;
}

def Tessera_ArchSTEOneHotOp : Op<Tessera_Dialect, "arch.ste_one_hot", [Pure]> {
  let summary = "Straight-through one-hot architecture gate";
  let arguments = (ins Tessera_ArchParamType:$alpha);
  let results = (outs Tessera_ArchGateType:$gate);
  let assemblyFormat = "$alpha attr-dict";
  // No custom verifier: ArchParam/ArchGate are parameterless opaque types, so
  // the ODS operand/result constraints fully constrain the op (no scalar/shape
  // invariant to add). A `verify(){return success();}` would be a trivial stub.
}

def Tessera_ArchWeightedSumOp : Op<Tessera_Dialect, "arch.weighted_sum", [Pure]> {
  let summary = "Weighted merge of candidate tensor outputs by architecture gate";
  let arguments = (ins Variadic<TensorType>:$candidates,
                       Tessera_ArchGateType:$gate);
  let results = (outs TensorType:$result);
  let assemblyFormat = "$candidates `by` $gate attr-dict `:` functional-type($candidates, $result)";
  let hasVerifier = 1;
}

def Tessera_ArchSwitchOp : Op<Tessera_Dialect, "arch.switch", [Pure]> {
  let summary = "Soft or hard architecture candidate switch";
  let arguments = (ins Variadic<TensorType>:$candidates,
                       Tessera_ArchGateType:$gate,
                       DefaultValuedAttr<BoolAttr, "false">:$hard);
  let results = (outs TensorType:$result);
  let assemblyFormat = "$candidates `by` $gate attr-dict `:` functional-type($candidates, $result)";
  let hasVerifier = 1;
}

def Tessera_ArchMixedOp : Op<Tessera_Dialect, "arch.mixed", [Pure]> {
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

// Phase-G G-B — Graph-IR bounded control-flow loop. The loop body is a
// symbol-referenced `func.func` (`body`), not a region, so the op stays a
// value-semantic leaf that the apple_gpu pipeline lowers to a
// `tessera_apple.gpu.control_loop` Target-IR op (Decision #19). Carries the
// iter-args (loop-carried state) + the static range; the result is the final
// carry. Matching runtime capability: `tessera_apple_gpu_run_graph_loop_f32`
// (exercised today via the GraphFn lane; an MLIR-driven executor is a follow-on).
def Tessera_ControlForOp : Op<
  Tessera_Dialect, "control_for",
  [Pure]
> {
  let summary = "Bounded control-flow for-loop carried by iter_args (Phase-G G-B)";
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$body,
                       I64Attr:$start, I64Attr:$stop, I64Attr:$step,
                       // G-B.2 executable payload — the loop body serialized to
                       // the run_graph_loop op-list ABI so the lowered Target op
                       // is self-contained + dispatchable (carry id = #operands;
                       // body op j → #operands+1+j). Optional: omit for IR-only.
                       OptionalAttr<I64Attr>:$carry_arg_index,
                       OptionalAttr<DenseI32ArrayAttr>:$body_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$body_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$body_fattr,
                       OptionalAttr<I64Attr>:$body_out_id);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_ControlIfOp : Op<
  Tessera_Dialect, "control_if",
  [Pure]
> {
  let summary = "Divergent control-flow if/else, single carried result (Phase-G close-out C)";
  let description = [{
    `O = flag>0 ? then(args) : else(args)`. Only the taken branch executes
    (divergent, not a data-parallel select). The flag is the function arg at
    `flag_arg_index` (a shape-(1,) tensor; `flag[0] > 0` selects `then`). Both
    branch bodies are serialized to the run_graph_cond op-list ABI as the
    executable payload (branch op j → operand-id `#operands + j`), so the lowered
    Apple Target op (`tessera_apple.gpu.control_if`) is self-contained +
    dispatchable via its recorded `symbol`. Optional payload: omit for IR-only.
  }];
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$then_branch,
                       FlatSymbolRefAttr:$else_branch,
                       I64Attr:$flag_arg_index,
                       OptionalAttr<DenseI32ArrayAttr>:$then_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$then_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$then_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$then_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$then_fattr,
                       OptionalAttr<I64Attr>:$then_out_id,
                       OptionalAttr<DenseI32ArrayAttr>:$else_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$else_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$else_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$else_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$else_fattr,
                       OptionalAttr<I64Attr>:$else_out_id,
                       // The merged result shape (the branch output is not an
                       // operand, so unlike control_for it can't be recovered
                       // from an arg — carry it explicitly).
                       OptionalAttr<DenseI64ArrayAttr>:$out_shape);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_ControlWhileOp : Op<
  Tessera_Dialect, "control_while",
  [Pure]
> {
  let summary = "Bounded while-loop carried by a single iter_arg (Phase-G close-out D)";
  let description = [{
    `for up to max_iters: while cond(carry)>0: carry = body(carry)`. Lowers (on
    Apple) to an MPSGraph forLoop + select-masking (native `while` is unstable):
    once the predicate goes false the carry freezes. The carry init is the
    function arg at `carry_arg_index` (carry id = #operands; body/cond op j →
    `#operands + 1 + j`). Both the body and cond bodies are serialized to the
    run_graph_while op-list ABI as the executable payload, so the lowered Apple
    Target op (`tessera_apple.gpu.control_while`) is self-contained +
    dispatchable via its recorded `symbol`. Optional payload: omit for IR-only.
  }];
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$body,
                       FlatSymbolRefAttr:$cond,
                       I64Attr:$carry_arg_index,
                       I64Attr:$max_iters,
                       OptionalAttr<DenseI32ArrayAttr>:$body_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$body_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$body_fattr,
                       OptionalAttr<I64Attr>:$body_out_id,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$cond_fattr,
                       OptionalAttr<I64Attr>:$cond_out_id);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_ControlScanOp : Op<
  Tessera_Dialect, "control_scan",
  [Pure]
> {
  let summary = "Bounded scan: stream xs[t], carry a state, stack ys[t] (4th control primitive)";
  let description = [{
    `(carry, ys) = scan(body, init, xs)`: for `t in 0..trip-1`,
    `(carry, y) = body(carry, xs[t])` and `ys[t] = y`. Unlike `control_for`, scan
    CONSUMES a per-step input slice `xs[t]` and PRODUCES a stacked per-step output
    `ys` (shape `(trip, *y.shape)`) alongside the final carry — the streaming I/O
    a sequence recurrence (RNN/SSM/linear-attention) needs. The carry init is
    operand `carry_arg_index` (0); `xs` is the remaining operand. The body is
    serialized to the run_graph_scan op-list ABI as the executable payload
    (operand ids `0..n-1`; carry id = `n`; per-step `xt` id = `n+1`; body op j →
    `n+2+j`; `body_out_id`/`y_out_id` pick the next carry / per-step y), so the
    lowered Target op is self-contained + dispatchable. Optional payload: omit for
    IR-only.

    `captures` are loop-invariant operands (weight matrices, biases) the body
    closes over — distinct from the per-step `xs` stream. The body signature is
    `(carry, xt, captures...) -> (carry', y)`. With captures the recurrence is a
    real cell, e.g. the linear state-space update `h_t = h_{t-1} @ W + x_t`.
  }];
  let arguments = (ins AnyType:$init, AnyType:$xs,
                       Variadic<AnyType>:$captures,
                       FlatSymbolRefAttr:$body,
                       I64Attr:$trip,
                       I64Attr:$carry_arg_index,
                       OptionalAttr<DenseI32ArrayAttr>:$body_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$body_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$body_fattr,
                       OptionalAttr<I64Attr>:$body_out_id,
                       OptionalAttr<I64Attr>:$y_out_id);
  let results = (outs AnyType:$carry_out, AnyType:$ys);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_SpecAcceptOp : Op<
  Tessera_Dialect, "spec_accept",
  [Pure]
> {
  let summary = "Speculative-decode acceptance: greedy longest-prefix path + bonus (SD1)";
  let description = [{
    Greedy (argmax-match) speculative-decode verification over a tree of draft
    paths — SD1's standalone `spec_accept` primitive. For each path `p` the
    accepted length is the run of leading positions where `draft[p,i] ==
    target[p,i]`; the op selects the path with the **longest** accepted prefix
    (first wins ties) and returns the packed `tensor<3xi32>`
    `[accepted_path_idx, accepted_prefix_length, bonus_token]`, where the bonus is
    `target[path, length]` — the corrected/next token at the first divergence.
    `draft` is `P×D` and `target` is `P×(D+1)` (one extra column) so the bonus is
    always defined, even when a whole path of length `D` is accepted.

    Deterministic — a pure function of `draft`/`target`, no RNG, no cache
    mutation. It mirrors the `tessera.speculative` greedy reference
    (`_ref_spec_accept` / `dflash_linear_verify`, generalized to multi-path); the
    distribution-preserving Leviathan / rejection-sampling form (with RNG and a
    residual bonus) is the SD1-2 follow-up.
  }];
  let arguments = (ins AnyType:$draft, AnyType:$target);
  let results = (outs AnyType:$result);  // tensor<3xi32>: [path, length, bonus]
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_SpecAcceptSampleOp : Op<
  Tessera_Dialect, "spec_accept_sample",
  [Pure]
> {
  let summary = "Speculative-decode acceptance: distribution-preserving rejection sampling (SD1-2)";
  let description = [{
    Distribution-preserving (Leviathan) speculative-decode verification of a
    LINEAR draft chain — SD1-2. Per drafted position `i` the draft token `d_i` is
    accepted iff `accept_u[i] <= min(1, p_target/p_draft)` (with `p_draft > 0`); on
    the first rejection a corrected token is drawn from the residual
    `normalize(relu(p_target[i] - p_draft[i]))`; if the whole chain is accepted a
    bonus token is drawn from `target_probs`'s extra row. The marginal of every
    emitted token equals the target distribution.

    **RNG is explicit (CF0 contract).** The two uniform streams are op operands —
    `accept_u` (one per position, the per-token accept draw) and `resid_u` (one
    uniform for the single categorical draw that occurs, residual-on-reject or
    bonus-on-full-accept). The categorical draw is **CDF inversion**: the smallest
    `k` with `cumsum(weights)[k] > u·Σweights`. This makes the op a deterministic,
    device-bit-exact function of its inputs (it does NOT reproduce numpy's
    `Generator.choice` algorithm — the caller draws the uniforms; the op defines
    the inversion). Mirrors `python/tessera/dflash.py:dflash_speculative_verify` /
    `speculative.py:rejection_verify_chain` under CDF-inversion sampling.

    Shapes: `draft` `D` i32; `target_probs` `(D+1)×V` f32; `draft_probs` `D×V` f32;
    `accept_u` `D` f32; `resid_u` `1` f32. Result is `(D+2)` i32:
    `[accepted_prefix_length, t0, t1, …, t_D]` — `accepted` accepted draft tokens
    at `t[0..accepted-1]` then one corrected/bonus token at `t[accepted]`
    (`accepted+1` valid tokens; trailing slots are 0).
  }];
  let arguments = (ins AnyType:$draft, AnyType:$target_probs,
                       AnyType:$draft_probs, AnyType:$accept_u, AnyType:$resid_u);
  let results = (outs AnyType:$result);  // tensor<(D+2)xi32>: [accepted, tokens...]
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_SpecAcceptTreeSampleOp : Op<
  Tessera_Dialect, "spec_accept_tree_sample",
  [Pure]
> {
  let summary = "Speculative-decode acceptance: tree (multi-path) Leviathan rejection sampling (SD1)";
  let description = [{
    The **tree (multi-path)** distribution-preserving (Leviathan) form of
    speculative-decode acceptance — the device form of
    `tessera.speculative.batch_verify`. Over a draft tree of `P` paths each `D`
    deep, the draft token at `(p, i)` is accepted iff
    `accept_u[p,i] <= exp(target_log_probs[p,i] - draft_log_probs[p,i])` (the
    division-free `accept_u <= min(1, p_target/p_draft)`). Each path's accepted
    prefix is the run of leading accepts; the op selects the path with the
    **longest** accepted prefix (first wins ties) and returns the packed
    `tensor<2xi32>` `[accepted_path_idx, accepted_prefix_length]` — the caller
    advances the cache by the accepted prefix (`cache.commit`).

    `target_log_probs` / `draft_log_probs` are `P × D` (the log-prob of each path's
    chosen draft token under the target / draft at each step); `accept_u` is the
    explicit per-`(path, step)` uniform (CF0: RNG through explicit operands, so the
    op is deterministic + device-bit-exact). This generalizes the linear
    `spec_accept_sample` to a tree and the greedy multi-path `spec_accept` to the
    stochastic Leviathan rule.
  }];
  let arguments = (ins AnyType:$target_log_probs, AnyType:$draft_log_probs,
                       AnyType:$accept_u);
  let results = (outs AnyType:$result);  // tensor<2xi32>: [path_idx, prefix_length]
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_TargetVerifyOp : Op<
  Tessera_Dialect, "target_verify",
  [Pure]
> {
  let summary = "Speculative-decode target verification: I/O contract for the target scoring (SD1-4)";
  let description = [{
    SD1-4 — the **input/output contract** for the speculative-decode target
    verification step. `target_verify` "runs the target model over the current
    token plus the draft prefix": given the verified-position context `tokens`
    (the current committed token followed by the `D` draft tokens, `S = D+1`
    positions) and the target model's raw `logits` at those positions (`S × V`),
    it emits the contract-shaped per-position target **probabilities**
    `target_probs` (`S × V`, a `softmax` over the vocab `V`) — exactly the
    `(prefix_len+1) × V` *probability* tensor that `spec_accept_sample` consumes as
    its `target_probs` operand (the accept ratio `accept_u·p_draft ≤ p_target` and
    the residual `relu(p_target − p_draft)` read probabilities, not log-probs).
    Callers that want log-probs (e.g. `spec_accept_tree_sample` /
    `speculative.batch_verify`) take the log of the result.

    This is a **composed model call**, not a fused kernel: the op pins the
    batching shapes (S positions × V vocab) so the target scoring is explicit
    enough to batch and fuse, but the target model itself is external. `[Pure]`
    (a function of the tokens + logits). A fused target-verify kernel is a later
    DK-track concern.
  }];
  let arguments = (ins AnyType:$tokens, AnyType:$logits);
  let results = (outs AnyType:$target_probs);  // tensor<SxVxf32>
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
