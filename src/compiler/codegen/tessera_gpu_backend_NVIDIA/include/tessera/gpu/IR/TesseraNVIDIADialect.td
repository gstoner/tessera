//===- TesseraNVIDIADialect.td - Tessera NVIDIA Target IR ----*- tablegen -*-===//
//
// Hardware-free NVIDIA Target IR contract dialect (Decision #19).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_NVIDIA_DIALECT
#define TESSERA_NVIDIA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def TesseraNVIDIADialect : Dialect {
  let name = "tessera_nvidia";
  let cppNamespace = "::tessera::nvidia";
  let summary = "Tessera NVIDIA Target IR contract dialect";
  let description = [{
    Hardware-free NVIDIA Target IR contracts spanning Hopper (`wgmma`),
    datacenter Blackwell (`tcgen05_mma` + TMEM), and consumer Blackwell
    (warp-level `mma_sync`) matmul lowering, plus the `tma_async_copy` /
    `mbarrier` async companions.

    Decision #19: a backend exposes a *typed*, hardware-free Target IR dialect
    between Tile IR and hardware-specific (NVVM) lowering. Until now this
    dialect was `isExtensible` with zero registered ops, so `LowerTileToNVIDIA`
    built generic operations that no verifier constrained. These op definitions
    give it a real typed surface: `LowerTileToNVIDIA` still populates them via
    the generic `OperationState` builders, but they now round-trip, verify, and
    parse without `--allow-unregistered-dialect`. `LowerNVIDIAToNVVM` rewrites
    each to its NVVM artifact marker.
  }];

  // Plain attribute storage (not op-properties) so the ops round-trip and the
  // generic `OperationState` builders in NVIDIALowering.cpp populate the
  // attributes without the `getProperties()` machinery — mirrors the Apple
  // reference dialect (TesseraAppleOps.td).
}

// Target values crossing the NVIDIA boundary are data buffers, scalar/vector
// fragments, or LLVM ABI values. This deliberately excludes arbitrary dialect
// types (for example async tokens and symbol-like handles): synchronization is
// represented by registered TMA/mbarrier ops and scalar controls are attributes.
// The union covers every physical ABI emitted today, including LLVM aggregate
// accumulator returns from warp-level MMA.
def NVIDIA_TargetValue : AnyTypeOf<[
    AnyRankedTensor, AnyUnrankedTensor, AnyMemRef, AnyVectorOfAnyRank,
    AnyInteger, AnyFloat, LLVM_AnyPointer, LLVM_AnyStruct
  ], "a NVIDIA data buffer, scalar/vector fragment, or LLVM ABI value">;

// v1 typed contract. Each op carries the region's supported target values as
// variadic operands and yields supported target values; the specialization
// metadata (arch / shape / dtype / warpgroup / …) rides as optional attributes.
// The attribute set is the union the lowering can emit — an op ignores those it
// does not use — which tolerates the per-SM-branch combinations
// `LowerTileToNVIDIA` produces. Ops carry no side-effect trait so generic DCE
// never drops a marker whose result is unused.
class TesseraNVIDIA_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraNVIDIADialect, mnemonic, traits> {
  let arguments = (ins
      Variadic<NVIDIA_TargetValue>:$inputs,
      OptionalAttr<StrAttr>:$arch,
      OptionalAttr<StrAttr>:$source,
      OptionalAttr<StrAttr>:$shape,
      OptionalAttr<StrAttr>:$dtype_ab,
      OptionalAttr<StrAttr>:$dtype_c,
      OptionalAttr<StrAttr>:$accum,
      OptionalAttr<StrAttr>:$format,
      OptionalAttr<StrAttr>:$epilogue,
      OptionalAttr<StrAttr>:$mask,
      OptionalAttr<StrAttr>:$src_space,
      OptionalAttr<StrAttr>:$dst_space,
      OptionalAttr<BoolAttr>:$block_scaled,
      OptionalAttr<I64Attr>:$warpgroup,
      OptionalAttr<I64Attr>:$cta_group,
      OptionalAttr<I64Attr>:$columns,
      OptionalAttr<I64Attr>:$bytes);
  let results = (outs Variadic<NVIDIA_TargetValue>:$outputs);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
}

class NVIDIA_EnumStrAttr<list<string> cases, string summaryText>
    : StringBasedAttr<
        CPred<"::llvm::is_contained(::llvm::ArrayRef<::llvm::StringRef>({"
              # !interleave(!foreach(c, cases, "\"" # c # "\""), ", ") #
              "}), ::llvm::cast<::mlir::StringAttr>($_self).getValue())">,
        summaryText>;
def NVIDIA_CudaMathKindAttr
    : NVIDIA_EnumStrAttr<[
        "abs_i32", "min_i32", "max_i32", "brev_u32", "byte_perm_u32",
        "clz_u32", "ffs_u32", "popc_u32", "funnelshift_l_wrap_u32",
        "dp2a_lo_s32", "dp4a_s32", "bitcast_f32_i32", "bitcast_i32_f32",
        "cvt_f32_i32_rn", "cvt_f32_i32_rd", "cvt_f32_i32_ru",
        "cvt_f32_i32_rz", "vadd2_u16x2", "vadd4_u8x4", "vadd4_s8x4",
        "vaddss4_s8x4", "vsub4_u8x4", "vsub4_s8x4", "vsubss4_s8x4",
        "vabsdiff4_u8x4", "vcmpeq4_u8x4_mask", "vcmpeq4_u8x4_pred"],
        "typed CUDA math operation kind">;

def NVIDIA_MmaSyncOp : TesseraNVIDIA_Op<"mma_sync"> {
  let summary = "Consumer Blackwell (sm_120) warp-level mma.sync contract";
}
def NVIDIA_BlockCoordinateOp : Op<TesseraNVIDIADialect, "block_coordinate", [Pure]> {
  let summary = "Typed SM120 macro-tile block-coordinate contract";
  let description = [{
    Materializes the row and column bases for the canonical SM120 physical
    tile.  The exact attributes deliberately make grid interpretation a
    target-owned, fail-closed contract rather than implicit producer
    arithmetic.
  }];
  let arguments = (ins
      StrAttr:$arch,
      I64Attr:$tile_m,
      I64Attr:$tile_n,
      StrAttr:$grid_order);
  let results = (outs I64:$row_base, I64:$col_base);
  let hasVerifier = 1;
}
def NVIDIA_MacroCTAMatmulOp : Op<TesseraNVIDIADialect, "macro_cta_matmul"> {
  let summary = "SM120 four-warp shared-panel matmul contract";
  let description = [{
    Owns the physical 32x32 CTA decomposition used to reuse one A[32,16] and
    one B[16,32] panel across four warps. The operation is deliberately exact:
    changing its warp ownership, staging space, grid order, or instruction tile
    requires a new contract rather than reinterpretation by a lowering.
  }];
  let arguments = (ins
      LLVM_AnyPointer:$a,
      LLVM_AnyPointer:$b,
      LLVM_AnyPointer:$d,
      I64:$m,
      I64:$n,
      I64:$k,
      Variadic<I64>:$leading_dimensions,
      StrAttr:$arch,
      I64Attr:$cta_m,
      I64Attr:$cta_n,
      I64Attr:$tile_m,
      I64Attr:$tile_n,
      I64Attr:$tile_k,
      I64Attr:$warps,
      StrAttr:$warp_ownership,
      StrAttr:$storage,
      StrAttr:$accum,
      StrAttr:$staging,
      I64Attr:$stages,
      StrAttr:$completion,
      StrAttr:$bounds,
      StrAttr:$grid_order);
  let results = (outs);
  let hasVerifier = 1;
}
def NVIDIA_MmaFusedOp : TesseraNVIDIA_Op<"mma_fused"> {
  let summary = "sm_120 mma.sync GEMM with a fused pointwise epilogue";
}
def NVIDIA_MmaAttentionOp : TesseraNVIDIA_Op<"mma_attention"> {
  let summary = "sm_120 two-MMA attention with f32 softmax accumulation";
}
def NVIDIA_FpQuantOp : TesseraNVIDIA_Op<"fpquant"> {
  let summary = "sm_120 FP8/FP6/FP4 storage quantize or dequantize contract";
}
def NVIDIA_Nvfp4BlockScaleMmaOp : TesseraNVIDIA_Op<"nvfp4_block_scale_mma"> {
  let summary = "sm_120a NVFP4 E2M1 MMA with UE4M3 block scales";
}
def NVIDIA_MxBlockScaleMmaOp : TesseraNVIDIA_Op<"mx_block_scale_mma"> {
  let summary = "sm_120a MX FP6/FP4 MMA with UE8M0 block scales";
}
def NVIDIA_WgmmaOp : TesseraNVIDIA_Op<"wgmma"> {
  let summary = "Hopper (sm_90a) warpgroup wgmma contract";
}
def NVIDIA_Tcgen05MmaOp : TesseraNVIDIA_Op<"tcgen05_mma"> {
  let summary = "Datacenter Blackwell (sm_100a) tcgen05 MMA contract";
}
def NVIDIA_WmmaOp : TesseraNVIDIA_Op<"wmma"> {
  let summary = "Pre-Hopper warp-level wmma contract";
}
def NVIDIA_TmaAsyncCopyOp : TesseraNVIDIA_Op<"tma_async_copy"> {
  let summary = "TMA cp.async.bulk.tensor async-copy contract (sm_90+)";
}
def NVIDIA_MbarrierOp : TesseraNVIDIA_Op<"mbarrier"> {
  let summary = "mbarrier arrive/wait contract";
}
def NVIDIA_TmemAllocOp : TesseraNVIDIA_Op<"tmem_alloc"> {
  let summary = "Blackwell TMEM allocation contract (sm_100a)";
}
def NVIDIA_TmemLoadOp : TesseraNVIDIA_Op<"tmem_load"> {
  let summary = "Blackwell TMEM load contract (sm_100a)";
}
def NVIDIA_TmemStoreOp : TesseraNVIDIA_Op<"tmem_store"> {
  let summary = "Blackwell TMEM store contract (sm_100a)";
}
def NVIDIA_CudaKernelOp : TesseraNVIDIA_Op<"cuda_kernel"> {
  let summary = "Generic CUDA kernel artifact contract";
}
def NVIDIA_TrainingKernelOp : TesseraNVIDIA_Op<"training_kernel"> {
  let summary = "SM120 state-lineage training kernel artifact contract";
}

def NVIDIA_PhiloxModeAttr
    : NVIDIA_EnumStrAttr<[
        "uniform_core", "uniform_range", "normal", "dropout"],
        "supported NVIDIA Philox generation mode">;

// Stateless key/counter RNG directive. The generator deliberately begins at
// the uniform core: distribution transforms are separate native consumers so
// their numerical policies cannot silently alter the canonical bit stream.
def NVIDIA_PhiloxOp : Op<TesseraNVIDIADialect, "philox"> {
  let summary = "Stateless Philox4x32-10 CUDA kernel generation directive";
  let arguments = (ins StrAttr:$name, NVIDIA_PhiloxModeAttr:$mode);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

// Typed CUDA Math API execution contract. Portable casts/bitcasts are
// normalized before this boundary where possible; CUDA-only funnel, packed
// dot, byte-lane SIMD, saturation, and predicate-result semantics remain
// explicit here. The verifier owns a closed kind/type/rounding/saturation/
// lane/predicate matrix rather than accepting an open string dispatch.
def NVIDIA_CudaMathKernelOp : Op<TesseraNVIDIADialect, "cuda_math_kernel"> {
  let summary = "Typed NVIDIA CUDA integer, bitcast, and packed-SIMD kernel";
  let arguments = (ins
      Variadic<NVIDIA_TargetValue>:$inputs,
      NVIDIA_CudaMathKindAttr:$kind,
      StrAttr:$input_storage,
      StrAttr:$output_storage,
      StrAttr:$rounding,
      BoolAttr:$saturation,
      I64Attr:$lane_width,
      StrAttr:$signedness,
      StrAttr:$predicate_form);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}
def NVIDIA_ControlForOp : TesseraNVIDIA_Op<"control_for"> {
  let summary = "Single-launch bounded for-loop CUDA contract";
}
def NVIDIA_ControlIfOp : TesseraNVIDIA_Op<"control_if"> {
  let summary = "Single-launch divergent conditional CUDA contract";
}
def NVIDIA_ControlWhileOp : TesseraNVIDIA_Op<"control_while"> {
  let summary = "Single-launch bounded while-loop CUDA contract";
}
def NVIDIA_ControlScanOp : TesseraNVIDIA_Op<"control_scan"> {
  let summary = "Single-launch streaming scan CUDA contract";
}

// Wrapper / probe ops. The **Python** Target IR emitters (``target_ir.py`` /
// ``value_target_contract.py``) wrap the contract body in ``tessera_nvidia.func``
// and may add ``tessera_nvidia.kernel_call`` / ``tessera_nvidia.profiler_probe``.
// The C++ ``LowerTileToNVIDIA`` never emits these — but registering them keeps the
// *whole* emitted ``tessera_nvidia`` surface typed and parseable now that the
// dialect is no longer ``isExtensible`` (otherwise a Python-emitted module would
// fail to parse through the registered dialect).
def NVIDIA_FuncOp : Op<TesseraNVIDIADialect, "func", [NoTerminator]> {
  let summary = "NVIDIA Target IR function wrapper holding the kernel-contract body";
  let arguments = (ins OptionalAttr<StrAttr>:$sym_name);
  let regions = (region AnyRegion:$body);
  // No assemblyFormat: the emitter writes the generic form, which every
  // registered op parses; keeping it generic avoids constraining the emitter.
}
def NVIDIA_DelegateBindingAttr
    : NVIDIA_EnumStrAttr<["cuda_kernel", "c_abi"],
        "how the delegate is reached: device kernel launch or host C-ABI call">;

def NVIDIA_DelegateProvenanceAttr
    : NVIDIA_EnumStrAttr<["vendor_library", "handwritten_kernel"],
        "what the delegate is: a vendor library entry or an in-tree hand-written kernel">;

def NVIDIA_DelegateAccuracyAttr
    : NVIDIA_EnumStrAttr<["reference_exact", "tolerance_bounded"],
        "numerical claim the delegate makes against the compiled reference">;

def NVIDIA_KernelCallOp : Op<TesseraNVIDIADialect, "kernel_call"> {
  let summary = "Delegation to a named CUDA kernel or host C-ABI symbol";
  let description = [{
    A **fast path out of the compiler**, made visible to it.

    Tessera's architecture is Graph -> Schedule -> Tile -> Target driven by
    `tessera-opt`. Some work legitimately does not come from that pipeline: a
    vendor library entry, or a hand-tuned kernel that beats anything the
    compiler will emit. Decision #28 keeps those as first-class **Tier-3
    candidates** — but only if a *measured, accuracy-budgeted arbiter* can
    score them against compiled output. An arbiter cannot score what it
    cannot see, which is what this op exists to prevent.

    This op previously carried a summary line and nothing else. `callee` — the
    single fact that says *what is being delegated to* — rode in the inherited
    `attr-dict` as an unvalidated discardable attribute, so an emitter could
    name any symbol, or none, and still verify. That is a declaration with no
    consumer (Decision #29): it reads in review as a closed contract while
    carrying nothing.

    The attributes are the arbiter's inputs, and each is required because a
    missing one silently removes this candidate's comparability rather than
    failing:

    * `callee` / `arch` — identity. Without them the candidate cannot be
      cached, re-measured, or bound.
    * `binding` — which of the two pathways this is. A device kernel launch
      and a host C-ABI call have different launch costs and different
      failure modes; collapsing them would make a measurement meaningless.
    * `provenance` — vendor library vs in-tree hand-written kernel. The
      arbiter's tiering, and the thing that distinguishes "delegated" from
      "compiler-generated" when scoring.
    * `accuracy` — the *budget* half of "fastest in-budget candidate". A
      delegate claiming `tolerance_bounded` must state its `tolerance`;
      being faster is not a reason to select a candidate whose numerical
      claim is unstated. This is a semantic key and never defaults
      (Decision #21a).

    For inline PTX, see `tessera_nvidia.inline_ptx` — the other pathway. That
    is a distinct op rather than a mode of this one because the delegate is
    *text embedded in the artifact* rather than a binding resolved elsewhere,
    so the two have different verifiers and different failure modes.
  }];
  let arguments = (ins
      Variadic<NVIDIA_TargetValue>:$inputs,
      StrAttr:$callee,
      StrAttr:$arch,
      NVIDIA_DelegateBindingAttr:$binding,
      NVIDIA_DelegateProvenanceAttr:$provenance,
      NVIDIA_DelegateAccuracyAttr:$accuracy,
      OptionalAttr<F64Attr>:$tolerance);
  let results = (outs Variadic<NVIDIA_TargetValue>:$outputs);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def NVIDIA_InlinePtxOp : Op<TesseraNVIDIADialect, "inline_ptx"> {
  let summary = "Delegation to inline PTX embedded in the compiled artifact";
  let description = [{
    The second fast path, and deliberately a separate op from
    `tessera_nvidia.kernel_call`.

    Inline PTX is not a variant of a symbol call: the delegate is *text
    carried in the artifact* rather than a binding resolved at link or launch
    time. It has no callee to cache on, its side effects are not inferable
    from a symbol, and an empty asm string is a silently-successful no-op
    rather than an unresolved-symbol error. Folding both into one op with a
    mode attribute would force a verifier that checks which half of its own
    attributes to trust — the shape that lets a malformed candidate through.

    `constraints` is required for the same reason it is in LLVM: an inline
    asm block whose operand constraints are unstated cannot be bound
    correctly, and guessing them is how register clobbers become silent
    miscompiles. `has_side_effects` is explicit because PTX that touches
    memory or barriers must not be reordered or eliminated, and absence of
    evidence is not purity.

    Accuracy carries the same contract as `kernel_call`: an arbiter selects
    the fastest *in-budget* candidate, so a delegate with no stated numerical
    claim is not selectable (Decision #21a — a semantic key never defaults).
  }];
  let arguments = (ins
      Variadic<NVIDIA_TargetValue>:$inputs,
      StrAttr:$ptx,
      StrAttr:$constraints,
      StrAttr:$arch,
      NVIDIA_DelegateAccuracyAttr:$accuracy,
      OptionalAttr<F64Attr>:$tolerance,
      UnitAttr:$has_side_effects);
  let results = (outs Variadic<NVIDIA_TargetValue>:$outputs);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}
def NVIDIA_ProfilerProbeOp : TesseraNVIDIA_Op<"profiler_probe"> {
  let summary = "Profiler probe marker (device-timer instrumentation point)";
}

#endif // TESSERA_NVIDIA_DIALECT
