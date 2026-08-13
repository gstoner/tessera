//===- TileOps.td - Tessera Tile IR dialect ---------------*- tablegen -*-===//
//
// Sprint 9: registered Tile IR ops for the value-lane lowering spine. These are
// the transient ops produced by the value-mode TilingPass and consumed by the
// Tile->Apple Target lowering. Registering them removes the
// `--allow-unregistered-dialect` dependency from the Apple value lane (the
// real Graph -> Schedule -> Tile -> Target compiler handoff).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TILE_OPS
#define TESSERA_TILE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Tile_Dialect : Dialect {
  let name = "tile";
  let cppNamespace = "::tessera::tile";
  let summary = "Tessera Tile IR — the layer between Schedule IR and Target IR";
  let description = [{
    Transient tile-level ops for the value-preserving lowering spine. Operands
    and results are the original SSA tensors (so the Tile->Target pass can
    replaceOp cleanly). Op-specific metadata (`source`, `transposeA/B`, linalg
    solver flags, ordinal) rides as discardable attributes.
  }];
  // C1 (2026-06-23) — the dialect now owns first-class layout attributes
  // (#tile.layout / #tile.swizzle), so it needs the generated attribute
  // printer/parser hooks.
  let useDefaultAttributePrinterParser = 1;
  // Phase B (2026-06-23) — the dialect now owns the !tile.async_token type
  // (the SSA completion edge for async_copy→wait/mma), so it needs the
  // generated type printer/parser hooks.
  let useDefaultTypePrinterParser = 1;
}

class Tile_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tessera_Tile_Dialect, mnemonic, traits>;

class Tile_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tessera_Tile_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Phase B (2026-06-23) — async-copy completion token. Payload-free SSA value: it
// carries "this async copy completed", NOT a count (NV expect-bytes stays in
// #tile.barrier<kind=tma,expect=…>; ROCm vmcnt stays in the legality's
// outstanding-set arithmetic). Threading it from tile.async_copy's result into
// tile.wait_async / tile.mma operands turns the copy→consumer synchronization
// dependency into an SSA def-use edge that a legality PASS can check by
// dataflow instead of program order.
//
// Scope (accurate state, not aspiration): this is an ADDITIVE, optimization-
// fence SSA edge. It is checked by the opt-in legality passes
// (tessera-warpspec-legality on NV, rocm-wave-lds-legality on ROCm), NOT by an
// op verifier — the type rides the ops' existing Variadic<AnyType> slots, so
// nothing structurally forces single-producer / produced-before-consumed beyond
// MLIR's generic SSA dominance. And it does NOT yet retire the string markers:
// on the ROCm path the tile.barrier_id / tile.depends_on FIFO is still the
// load-bearing dependency model and the token is redundant decoration there;
// the token replaced program-order reasoning on the NV warp-spec path. Both
// models coexist as of Phase C/D; full unification is future work. Kept
// backend-neutral (one type, no NVIDIA/ROCm privilege).
def Tile_AsyncTokenType : Tile_Type<"AsyncToken", "async_token"> {
  let summary = "Completion token for an asynchronous tile copy";
}

// Portable tile-fragment ABI (2026-07-14).  These types deliberately carry no
// target register shape: `!tile.fragment` is a logical cooperative-matrix
// operand, while `#tile.mma_desc` on its defining/consuming operation supplies
// the role and instruction contract.  NVIDIA vector fragments and AMD VGPR
// orderings are target-lowering details, never part of shared Tile IR.
def Tile_TileValueType : Tile_Type<"TileValue", "tile"> {
  let summary = "Logical distributed tile value";
}

// W1.1 step 1 (2026-08-03) — the fragment carries its own instruction contract.
//
// It was `Opaque`, which made `!tile.fragment == !tile.fragment` ALWAYS TRUE,
// so MLIR's own type equality checked nothing and `MMAOp::verify()` recovered
// the contract by chasing each operand's producing op. That chase cannot cross
// a block-argument edge, and a K-loop accumulator is an `scf.for` iter-arg by
// construction -- so the typed form was unusable by every real GEMM. See
// docs/audit/compiler/W1_1_TYPING_DESIGN.md §2.
//
// Split rule: the TYPE carries what makes two fragments interchangeable; the
// `#tile.mma_desc` ATTRIBUTE keeps how the operation executes (`k_blocks`).
//
// `family` IS a type parameter, deliberately. An earlier draft put it on the
// attribute, reasoning that instruction selection is an operation-local choice.
// That is wrong: `family` selects a physical register ABI -- wave 32 for
// RDNA3/RDNA4/gfx125x WMMA vs 64 for CDNA MFMA, with different element counts
// and formats (`ROCMFragmentLayout.h`) -- and `TileToROCM.cpp` states that those
// descriptors "are intentionally non-interchangeable". Leaving it off the type
// would let a loop init packed for `mma_sync` feed a body `tile.mma` selecting
// `wgmma`, with the types comparing equal. "The op chooses it" does not imply
// "the value does not carry it".
//
// The BARE spelling `!tile.fragment` remains legal and means all-unknown, so
// the existing fixtures and the Python text emitters keep parsing while
// producers migrate (W1.1 steps 3-4). `family = "auto"` is the legal
// pre-resolution value and compares equal only to itself, which reproduces
// today's descriptor-equality semantics rather than approximating them.
def Tile_FragmentType : Tile_Type<"Fragment", "fragment"> {
  let summary = "Cooperative-matrix fragment and its instruction contract";
  let parameters = (ins "int64_t":$m, "int64_t":$n, "int64_t":$k,
                        StringRefParameter<"">:$elem,
                        StringRefParameter<"">:$acc,
                        StringRefParameter<"">:$role,
                        StringRefParameter<"">:$layout,
                        StringRefParameter<"">:$family);
  let hasCustomAssemblyFormat = 1;
  // Domain validation, mirroring TileMmaDescAttr::verify (PR #502 review).
  // Without it, `get()` accepted a nonpositive shape, an empty dtype, an
  // unknown role/layout/family, and -- worst -- a PARTIALLY populated tuple,
  // which `isUnknown()` does not recognise and step 2's type-based checks would
  // therefore treat as a stated contract.
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The legacy all-unknown fragment, spelled `!tile.fragment`. Producers
    /// that have not migrated yet still emit this; `MMAOp::verify()` falls back
    /// to producer-chasing for it.
    static FragmentType getUnknown(::mlir::MLIRContext *ctx) {
      return get(ctx, 0, 0, 0, "", "", "", "", "");
    }
    /// True when no parameter is stated -- i.e. the bare spelling.
    bool isUnknown() const {
      return getM() == 0 && getN() == 0 && getK() == 0 && getElem().empty() &&
             getAcc().empty() && getRole().empty() && getLayout().empty() &&
             getFamily().empty();
    }
  }];
}

// SSA handle for a physical allocation. The handle carries identity through
// def-use rather than through the legacy #tile.buffer_ref name convention.
// Its address space and byte extent remain properties of tile.alloc.
def Tile_BufferType : Tile_Type<"Buffer", "buffer"> {
  let summary = "Opaque SSA handle for a physical Tile allocation";
}

// SSA state threaded through a staged producer/consumer ring. Unlike the
// legacy #tile.pipeline_state annotation, this value establishes ordering and
// state ownership through ordinary SSA def-use.
def Tile_PipelineStateType : Tile_Type<"PipelineState", "pipeline_state"> {
  let summary = "SSA state for a staged producer/consumer pipeline";
}

def Tile_TMADescriptorType : Tile_Type<"TMADescriptor", "tma_descriptor"> {
  let summary = "Portable SSA handle for one tensor-memory-access descriptor";
}

def Tile_MBarrierType : Tile_Type<"MBarrier", "mbarrier"> {
  let summary = "SSA handle for a transaction barrier with one or more slots";
}

def Tile_MBarrierTokenType : Tile_Type<"MBarrierToken", "mbarrier_token"> {
  let summary = "SSA token returned by an mbarrier arrival";
}

def Tile_TMEMType : Tile_Type<"TMEM", "tmem"> {
  let summary = "Opaque SSA handle for datacenter-Blackwell tensor memory";
}

// Physical storage and staged-pipeline ownership. These operations are the
// canonical def-use identity. #tile.buffer_ref remains parser-visible only for
// migration diagnostics and archived IR; active passes do not consume it.
def Tile_AllocOp : Tile_Op<"alloc"> {
  let summary = "Allocate a physical Tile buffer and return its SSA handle";
  let arguments = (ins StrAttr:$space, I64Attr:$bytes, AnyAttr:$layout);
  let results = (outs Tile_BufferType:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
  let hasVerifier = 1;
}

def Tile_DeallocOp : Tile_Op<"dealloc"> {
  let summary = "End the lifetime of a physical Tile buffer";
  let arguments = (ins Tile_BufferType:$buffer);
  let results = (outs);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
  let hasVerifier = 1;
}

def Tile_PipelineInitOp : Tile_Op<"pipeline_init"> {
  let summary = "Create an SSA-owned producer or consumer pipeline state";
  let arguments = (ins I64Attr:$depth, I64Attr:$stage, I64Attr:$phase,
                        StrAttr:$role);
  let results = (outs Tile_PipelineStateType:$state);
  let assemblyFormat = "attr-dict `:` type($state)";
  let hasVerifier = 1;
}

def Tile_PipelineAdvanceOp : Tile_Op<"pipeline_advance"> {
  let summary = "Advance staged pipeline state after explicit dependencies";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Tile_PipelineStateType:$next);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $next)";
  let hasVerifier = 1;
}

// Target-neutral multi-rank transport.  Schedule lowering must resolve the
// overloaded Graph `axis` into separate mesh and tensor axes before reaching
// this layer.  `world_size` is optional because launch packaging may bind a
// dynamic process group; when present the verifier proves gather/scatter shape
// scaling exactly.  Target/runtime ownership remains architecture-specific.
class Tile_CollectiveOp<string mnemonic, string summaryText> : Tile_Op<
    mnemonic, [MemoryEffects<[MemWrite]>]> {
  let summary = summaryText;
  let arguments = (ins AnyType:$input,
                       StrAttr:$mesh_axis,
                       I64Attr:$tensor_axis,
                       DefaultValuedAttr<StrAttr, "\"none\"">:$reduction,
                       OptionalAttr<I64Attr>:$world_size);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}

def Tile_AllReduceOp : Tile_CollectiveOp<
    "all_reduce", "Reduce a tensor across one named mesh axis">;
def Tile_ReduceScatterOp : Tile_CollectiveOp<
    "reduce_scatter", "Reduce then scatter one tensor axis across a mesh axis">;
def Tile_AllGatherOp : Tile_CollectiveOp<
    "all_gather", "Gather one tensor axis across a named mesh axis">;
def Tile_AllToAllOp : Tile_CollectiveOp<
    "all_to_all", "Exchange equal tensor-axis partitions across a mesh axis">;

def Tile_CollectivePermuteOp : Tile_Op<
    "collective_permute", [MemoryEffects<[MemWrite]>]> {
  let summary = "Send each listed source rank to one listed destination rank";
  let arguments = (ins AnyType:$input, StrAttr:$mesh_axis,
                       DenseI64ArrayAttr:$source_peers,
                       DenseI64ArrayAttr:$target_peers,
                       OptionalAttr<I64Attr>:$world_size);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}

// Target-neutral SSA vocabulary for asynchronous tensor movement. Availability
// and physical lowering remain target owned: NVIDIA maps these to TMA and
// mbarrier on supporting architectures, while other backends may reject or
// select an architecture-specific implementation.
def Tile_TMADescriptorOp : Tile_Op<"tma.descriptor"> {
  let summary = "Create an SSA tensor-memory-access descriptor";
  let arguments = (ins AnyType:$source, I64Attr:$tile_rows,
                        I64Attr:$tile_cols, OptionalAttr<I64Attr>:$slot,
                        OptionalAttr<I64Attr>:$expect_tx);
  let results = (outs Tile_TMADescriptorType:$descriptor);
  let assemblyFormat =
      "$source attr-dict `:` type($source) `->` type($descriptor)";
  let hasVerifier = 1;
}

def Tile_TMACopyAsyncOp
    : Tile_Op<"tma.copy_async", [AttrSizedOperandSegments]> {
  let summary = "Start a descriptor-driven asynchronous tile copy";
  let arguments = (ins Tile_TMADescriptorType:$descriptor,
                        Optional<Tile_MBarrierType>:$barrier,
                        Variadic<AnyType>:$dependencies,
                        I64Attr:$mbarrier_slot,
                        OptionalAttr<I64Attr>:$expect_tx);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tile_MBarrierInitOp : Tile_Op<"mbarrier.init"> {
  let summary = "Create an SSA transaction-barrier array";
  let arguments = (ins I64Attr:$slots, I64Attr:$phase_bits);
  let results = (outs Tile_MBarrierType:$barrier);
  let assemblyFormat = "attr-dict `:` type($barrier)";
  let hasVerifier = 1;
}

def Tile_MBarrierArriveExpectTxOp
    : Tile_Op<"mbarrier.arrive_expect_tx"> {
  let summary = "Arrive and bind the expected transaction byte count";
  let arguments = (ins Tile_MBarrierType:$barrier, I64Attr:$slot,
                        I64Attr:$bytes);
  let results = (outs Tile_MBarrierTokenType:$token);
  let assemblyFormat =
      "$barrier attr-dict `:` type($barrier) `->` type($token)";
  let hasVerifier = 1;
}

def Tile_MBarrierWaitOp
    : Tile_Op<"mbarrier.wait", [AttrSizedOperandSegments]> {
  let summary = "Wait for one transaction-barrier slot and dependency token";
  let arguments = (ins Optional<Tile_MBarrierType>:$barrier,
                        Variadic<AnyType>:$dependencies, I64Attr:$slot);
  let results = (outs);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
  let hasVerifier = 1;
}

def Tile_MBarrierTryWaitOp : Tile_Op<"mbarrier.try_wait"> {
  let summary = "Test whether an mbarrier arrival token has completed";
  let arguments = (ins Tile_MBarrierType:$barrier,
                        Tile_MBarrierTokenType:$token, I64Attr:$slot);
  let results = (outs I1:$ready);
  let assemblyFormat =
      "$barrier `,` $token attr-dict `:` type($barrier) `,` type($token)";
  let hasVerifier = 1;
}

// SM100-only physical vocabulary. These are registered in shared Tile IR so
// WarpSpec can express and thread the contract, but exact execution remains an
// NVIDIA target decision and SM120 must reject it.
def Tile_TMEMAllocOp : Tile_Op<"tmem.allocate"> {
  let summary = "Allocate a typed tensor-memory region";
  let arguments = (ins I64Attr:$bytes, I64Attr:$alignment);
  let results = (outs Tile_TMEMType:$handle);
  let assemblyFormat = "attr-dict `:` type($handle)";
  let hasVerifier = 1;
}

def Tile_TMEMLoadOp : Tile_Op<"tmem.load"> {
  let summary = "Load a logical value from tensor memory";
  let arguments = (ins Tile_TMEMType:$handle, Variadic<AnyType>:$indices);
  let results = (outs AnyType:$value);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tile_TMEMStoreOp : Tile_Op<"tmem.store"> {
  let summary = "Store a logical value into tensor memory";
  let arguments = (ins AnyType:$value, Tile_TMEMType:$handle,
                        Variadic<AnyType>:$indices);
  let results = (outs);
  let assemblyFormat =
      "$value `,` $handle (`[` $indices^ `]`)? attr-dict `:` type(operands)";
  let hasVerifier = 1;
}

def Tile_TCGen05MMAOp : Tile_Op<"tcgen05.mma"> {
  let summary = "Typed datacenter-Blackwell TCGen05 MMA contract";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Tile_TMEMType:$accumulator,
                        I64Attr:$cta_group, AnyAttr:$mma);
  let results = (outs Tile_TMEMType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `,` $accumulator attr-dict `:` "
      "type($lhs) `,` type($rhs) `,` type($accumulator) `->` type($result)";
  let hasVerifier = 1;
}

// Rank-2 dense matmul: C = A @ B (non-transposed value envelope).
def Tile_MatmulOp : Tile_Op<"matmul", [Pure]> {
  let summary = "Rank-2 tile-level matmul (value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Alias spelling (gemm) — same contract as matmul.
def Tile_GemmOp : Tile_Op<"gemm", [Pure]> {
  let summary = "Rank-2 tile-level gemm (value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Rank-3 batched matmul: C[b] = A[b] @ B[b].
def Tile_BatchedGemmOp : Tile_Op<"batched_gemm", [Pure]> {
  let summary = "Rank-3 tile-level batched matmul (value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Stage 13 — PPO policy loss value envelope. This tile op is produced only for
// static contiguous fp32 PPO mean-reduction shapes and consumed only by the
// Apple GPU value lane.
def Tile_PPOPolicyLossOp : Tile_Op<"ppo_policy_loss", [Pure]> {
  let summary = "PPO clipped policy loss (Apple GPU value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// EBM value lane — strict fp32/static tensor-shaped kernels with real Metal
// dispatch probes before runtime success is claimed.
def Tile_EBMEnergyQuadraticOp : Tile_Op<"ebm_energy_quadratic", [Pure]> {
  let summary = "Quadratic EBM energy (Apple GPU value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Tile_EBMLangevinStepOp : Tile_Op<"ebm_langevin_step", [Pure]> {
  let summary = "One affine EBM Langevin step (Apple GPU value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Tile_EBMRefinementOp : Tile_Op<"ebm_refinement", [Pure]> {
  let summary = "T-step affine EBM refinement (Apple GPU value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Tile_EBMPartitionExactOp : Tile_Op<"ebm_partition_exact", [Pure]> {
  let summary = "Exact EBM partition/logsumexp scalar (Apple GPU value lane)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// GA / Clifford value Tile seam. These are registered Tile IR carriers only:
// only clifford_geometric_product is promoted to an Apple GPU value executor in
// Stage 16E; the other carriers remain compiler-visible and gated in TileToApple.
class Tile_CliffordBinaryOp<string mnemonic, string summaryText>
    : Tile_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

class Tile_CliffordUnaryOp<string mnemonic, string summaryText>
    : Tile_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Tile_CliffordGeometricProductOp
    : Tile_CliffordBinaryOp<"clifford_geometric_product",
                            "Clifford geometric product (Tile value seam)">;
def Tile_CliffordOuterProductOp
    : Tile_CliffordBinaryOp<"clifford_outer_product",
                            "Clifford outer/wedge product (Tile value seam)">;
def Tile_CliffordInnerProductOp
    : Tile_CliffordBinaryOp<"clifford_inner_product",
                            "Clifford inner product (Tile value seam)">;
def Tile_CliffordReverseOp
    : Tile_CliffordUnaryOp<"clifford_reverse",
                           "Clifford reverse involution (Tile value seam)">;
def Tile_CliffordNormOp
    : Tile_CliffordUnaryOp<"clifford_norm",
                           "Clifford norm (Tile value seam)">;
def Tile_CliffordGradeProjectOp
    : Tile_CliffordUnaryOp<"clifford_grade_project",
                           "Clifford grade projection (Tile value seam)">;
def Tile_CliffordRotorSandwichOp
    : Tile_CliffordBinaryOp<"clifford_rotor_sandwich",
                            "Clifford rotor sandwich (Tile value seam)">;

// L-series linalg family — sequential factorizations/solves that lower 1:1 from
// the Graph IR op (already shape-verified there). Kept permissive at the Tile
// layer (operand/result counts + shapes already validated upstream).
class Tile_LinalgOp<string mnemonic> : Tile_Op<mnemonic, [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
}

def Tile_CholeskyOp      : Tile_LinalgOp<"cholesky">;
def Tile_TriSolveOp      : Tile_LinalgOp<"tri_solve">;
def Tile_CholeskySolveOp : Tile_LinalgOp<"cholesky_solve">;
def Tile_LUOp            : Tile_LinalgOp<"lu">;
def Tile_QROp            : Tile_LinalgOp<"qr">;
def Tile_SVDOp           : Tile_LinalgOp<"svd">;

// CF2/CF3 — portable device-control carriers.  These ops are the explicit
// Schedule/SCF -> Tile seam for bounded control flow.  The executable body is
// still symbol/payload described by discardable attributes copied from the
// verified Graph op; operands/results retain the value-semantic tensor ABI.
class Tile_ControlOp<string mnemonic> : Tile_Op<mnemonic, [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Tile_ControlForOp   : Tile_ControlOp<"control_for">;
def Tile_ControlIfOp    : Tile_ControlOp<"control_if">;
def Tile_ControlWhileOp : Tile_ControlOp<"control_while">;
def Tile_ControlScanOp  : Tile_ControlOp<"control_scan">;

//===----------------------------------------------------------------------===//
// Tile synchronization ops (TIRx op-layer convergence, 2026-06-23, Phase A0).
//
// The copy/wait/mma/barrier seam was previously emitted as opaque generic ops
// (`"tile.async_copy"(...)`), legal only because TesseraTileDialect calls
// allowUnknownOperations(true). Registering them makes the seam a first-class op
// — a precondition for giving tile.async_copy a typed completion result (the
// !tile.async_token SSA edge added in Phase B), the additive dependency edge
// that supersedes program-order reasoning on the NV path and coexists with the
// `tile.barrier_id` / `tile.depends_on` FIFO still load-bearing on ROCm (see the
// AsyncToken type doc above — full unification of the two is future work).
//
// DELIBERATELY NOT `[Pure]`, unlike the matmul/gemm value carriers above. These
// ops carry synchronization/ordering semantics; `[Pure]` (= NoMemoryEffect +
// AlwaysSpeculatable) would let DCE delete a wait/barrier, CSE merge two mmas, or
// LICM hoist an mma past its wait. Carrying no effect interface keeps them
// conservatively side-effecting (isMemoryEffectFree == false) — identical to the
// opaque behavior they had as unregistered ops, so registration changes nothing
// the optimizer is allowed to do.
//
// Permissive `Variadic<AnyType>` envelope + no verifier: every existing
// generic-form fixture stays valid across the value lane (tensors), the ROCm
// lowering (memref/ptr/index/token), and the NV path. The contract tightens when
// the typed async-token operand/result lands (Phase B/C).
//===----------------------------------------------------------------------===//

// Asynchronous global→shared/LDS tile copy (cp.async / TMA / buffer_load). Lands
// the staged tile as its result; a typed completion token is threaded in Phase B.
def Tile_AsyncCopyOp : Tile_Op<"async_copy"> {
  let summary = "Asynchronous global→shared/LDS tile copy";
  let description = [{
    THE DECLARED CONTRACT (single source of truth — reconciled 2026-08-10;
    `AsyncCopyOp::verify`/`WaitAsyncOp::verify` in TileOps.cpp are the only
    verifiers of these ops). One op, two dependency encodings, mirroring
    `tile.mma`'s W1.1 dual form:

      * **Typed form (production)** -- the op returns a `!tile.async_token`
        (last result) that `tile.wait_async` / `tile.mma` consume, so the
        copy/wait dependency lives in SSA def-use instead of program order.
        Convention: one source operand, staged tile + token as results
        (`TileIRLoweringPass::emitAsyncCopy`); a token-only result list is also
        legal. Matches `tessera_rocm.async_copy`, which already yields a
        `!tessera_rocm.token` threaded into its `wait`.
      * **Legacy form (declared compatibility envelope)** -- no token result;
        the dependency is carried outside SSA by grouping keys:
        `tile.barrier_id` / `tile.depends_on` (the ROCm wave-LDS FIFO lane,
        still load-bearing there) or an integer `stage` (the Python reference
        spine's encoding; consumed conservatively by `TileBufferReusePass`,
        which treats a key-less copy/wait as matching anything — the
        lifetime-extending choice, so absence fails safe).

    All attributes are OPTIONAL at this level; when present they must be well
    formed (`stage` >= 0 — enforced by the verifier). `tile_rows` /
    `tile_cols` / `tile.layout` are performance keys read by
    `AsyncCopyLoweringPass` with declared defaults; they never change what the
    copy means. Operand types are deliberately open: the value lane stages
    tensors, the ROCm lane memrefs/buffers.

    What is deliberately NOT the contract: a *required* `stage` or
    memref-only operands. That model (formerly a second name-dispatched
    verifier in `ScheduleOps.cpp`, deleted) had no in-tree producer and
    rejected the production emitter's output.

    Until W1.1's producer migration completes, absence of a token is NOT an
    error. What the verifier does enforce is that a copy which *claims* the
    typed form is well formed.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Wait on outstanding async copies (waitcnt(vmcnt) / mbarrier / cp.async.wait).
def Tile_WaitAsyncOp : Tile_Op<"wait_async"> {
  let summary = "Wait on outstanding asynchronous tile copies";
  let description = [{
    The consumer half of `tile.async_copy`, under the same single contract
    (see that op's description). Typed form: every `!tile.async_token` operand
    must be produced by a `tile.async_copy` -- a wait on a token from anywhere
    else is a dependency the backend cannot honor. Legacy form: no token
    operand; the wait retires copies matched by its grouping key
    (`tile.barrier_id` on ROCm, `stage` on the Python spine), and a key-less
    wait retires everything outstanding -- the conservative reading every
    consumer implements. `stage`, when present, must be >= 0.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Tile-level tensor-core MMA (WGMMA / MFMA / WMMA / tcgen05).
def Tile_MMAOp : Tile_Op<"mma"> {
  let summary = "Tile-level matrix multiply-accumulate (tensor-core)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// A logical subregion of a source, distributed according to `tile.layout`.
// It is a pure coordinate/layout view, not a physical register-pack operation.
def Tile_ViewOp : Tile_Op<"view", [Pure]> {
  let summary = "Create a logical distributed tile view";
  let description = [{
    Pointer-backed form (`tile.memory` present) takes either

      `(base, rowOrigin, colOrigin)`                         -- unbounded, or
      `(base, rowOrigin, colOrigin, rowBound, colBound)`     -- bounded.

    The bounds are the matrix's LOGICAL extents and are runtime values on a
    ragged problem. A backend that materializes fragments from a bounded view
    must mask loads against them; one that cannot must say so with a named
    diagnostic rather than ignoring the operands, since dropping them reads
    past the edge of the matrix.

    Arity is checked by `ViewOp::verify`, not by each consumer -- a shared-IR
    op whose accepted shape is decided per backend makes the same valid Tile IR
    compile on one target and fail on another.

    An optional `tile.linear_base` unit attribute inserts a precomputed linear
    origin immediately after `base`. The logical row/column origins remain
    present for bounds checks, while fragment materializers may reuse the
    supplied address origin instead of reconstructing `row * leading_dim +
    column` independently for every sibling fragment.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Tile_TileValueType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

// Decode a packed physical buffer into a logical tile. Inputs are:
//   unscaled: base, row-origin, column-origin, logical-rows, logical-columns
//   scaled:   base, scale-base, row-origin, column-origin,
//             logical-rows, logical-columns
// `tile.packed_view` owns the logical/container/scale contract and
// `tile.memory` owns the physical row/column addressing. The result stays
// target-neutral; a following tile.fragment_pack may select a target fragment.
def Tile_PackedLoadOp : Tile_Op<"packed_load"> {
  let summary = "Decode a packed physical buffer into a logical tile";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Tile_TileValueType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

// Encode a logical tile into packed physical storage. Inputs are:
//   tile, base, row-origin, column-origin, logical-rows, logical-columns.
// Scale-bearing stores are deliberately rejected until a portable scale
// production/rounding contract exists; scale-bearing formats are load-only.
def Tile_PackedStoreOp : Tile_Op<"packed_store"> {
  let summary = "Encode a logical tile into a packed physical buffer";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Materialize a logical tile in the target-selected cooperative-matrix ABI.
// The `role` and `mma` attributes are checked by the op verifier and are the
// portable contract consumed by NVIDIA/ROCm pack lowerings. Block-scaled
// descriptors may carry additional logical scale_a/scale_b fragments without
// exposing either backend's physical scale-selector registers.
def Tile_FragmentPackOp : Tile_Op<"fragment_pack"> {
  let summary = "Pack a logical tile into an opaque MMA fragment";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Tile_FragmentType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

// Construct the additive-identity accumulator without imposing a physical
// memory mapping. This is portable across cooperative-matrix backends and is
// the canonical initial C operand for the first A/B materialization slice.
def Tile_FragmentZeroOp : Tile_Op<"fragment_zero", [Pure]> {
  let summary = "Create a zero-initialized MMA accumulator fragment";
  let arguments = (ins);
  let results = (outs Tile_FragmentType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

// Convert an accumulator fragment back to a logical tile for an epilogue or
// store. Target lowering owns the register unpack; Tile IR owns the logical
// result layout and descriptor compatibility.
def Tile_FragmentUnpackOp : Tile_Op<"fragment_unpack"> {
  let summary = "Unpack an MMA accumulator fragment into a logical tile";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Tile_TileValueType:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $result)";
  let hasVerifier = 1;
}

// Store a logical tile to a physical destination. For a pointer-backed store,
// inputs are (tile, base pointer, row origin, column origin) and `tile.memory`
// supplies address space/order/leading dimension. A zero attribute leading
// dimension means the final SSA operand supplies it at runtime. Conservatively
// side-effecting.
def Tile_StoreOp : Tile_Op<"store"> {
  let summary = "Store a logical tile through a physical memory layout";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level contraction envelope. Unlike the value-lane tile.matmul, this
// form names the pointer ABI and runtime M/N/K dimensions needed to materialize
// a grid of cooperative-matrix tiles. Inputs are A, B, [bias], D, M, N, K; the
// optional bias operand is controlled by #tile.epilogue.
def Tile_MatmulKernelOp : Tile_Op<"matmul_kernel"> {
  let summary = "Launch-level tiled matmul with a portable epilogue contract";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level stable row-softmax envelope. Inputs are source, destination,
// flattened row count, and last-axis width. The storage/accumulation contract
// is explicit so target lowering never infers precision from a host array.
def Tile_SoftmaxKernelOp : Tile_Op<"softmax_kernel"> {
  let summary = "Launch-level stable row softmax";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level arbitrary-axis reduction envelope. Inputs are source,
// destination, and the canonical contiguous outer/axis/inner decomposition.
// Kind, shape retention, schedule, and non-finite behavior are explicit.
def Tile_ReduceKernelOp : Tile_Op<"reduce_kernel"> {
  let summary = "Launch-level typed arbitrary-axis reduction";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level spectral package. Inputs are source/destination pointers plus
// i64 batch and transform length. The attributes bind the full architecture-
// owned plan; a physical consumer must not refactor N or select another
// algorithm after this boundary.
def Tile_FFTKernelOp : Tile_Op<"fft_kernel"> {
  let summary = "Launch-level typed FFT package with fixed physical plan";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level compound spectral package. The variadic operands are the
// source pointers, destination pointer, and i64 workspace size. All framing,
// padding, child-FFT identities, numeric policy, and mutation ownership are
// fixed before this boundary.
def Tile_SpectralProgramKernelOp : Tile_Op<"spectral_program_kernel"> {
  let summary = "Launch-level compound spectral package with fixed child FFTs";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Multi-output compound spectral adjoint. Inputs are the output cotangent and
// retained forward operands followed by one destination pointer per gradient.
// Backend conversion must fail closed until the named architecture provides a
// native package for the exact content-addressed contract.
def Tile_SpectralBackwardKernelOp : Tile_Op<"spectral_backward_kernel"> {
  let summary = "Launch-level typed compound spectral adjoint package";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level flat elementwise envelope. Inputs are one or two source
// pointers, destination, and the flattened element count. Family and kind are
// semantic strings; storage/output_storage make mixed predicate output
// explicit instead of inferring it from a backend symbol.
def Tile_ElementwiseKernelOp : Tile_Op<"elementwise_kernel"> {
  let summary = "Launch-level typed flat elementwise operation";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// NVIDIA/CUDA integer, cast, packed-dot, and packed-SIMD carrier. Inputs are
// A, B, C/accumulator, destination, and flattened N. Unused sources remain in
// the stable ABI; explicit kind/storage/rounding attributes prevent C++ API
// overload resolution from leaking into portable Tile semantics.
def Tile_CudaIntrinsicKernelOp : Tile_Op<"cuda_intrinsic_kernel"> {
  let summary = "Launch-level typed CUDA integer/cast/packed intrinsic";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level index reduction. Inputs are source, i32 destination, flattened
// row count, and reduction width. The first-occurrence tie policy is explicit.
def Tile_ArgReduceKernelOp : Tile_Op<"argreduce_kernel"> {
  let summary = "Launch-level typed argmax or argmin reduction";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level inclusive row scan. Inputs are source, destination, flattened
// row count, and scan width.
def Tile_ScanKernelOp : Tile_Op<"scan_kernel"> {
  let summary = "Launch-level typed inclusive prefix scan";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level unweighted row normalization. Inputs are source, destination,
// flattened row count, row width, and epsilon.
def Tile_NormKernelOp : Tile_Op<"norm_kernel"> {
  let summary = "Launch-level typed RMSNorm or LayerNorm";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level interleaved-pair rotary embedding. Inputs are X, theta,
// destination, flattened row count, and even feature width.
def Tile_RopeKernelOp : Tile_Op<"rope_kernel"> {
  let summary = "Launch-level typed interleaved-pair rotary embedding";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level ALiBi bias construction. Inputs are slopes, destination, head
// count, and sequence length; output is [H,S,S].
def Tile_AlibiKernelOp : Tile_Op<"alibi_kernel"> {
  let summary = "Launch-level typed ALiBi positional bias";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Target-owned escape hatch for the broad x86 stable-ABI inventory.  Unlike
// portable semantic Tile carriers, operands are already ordered exactly as the
// C ABI expects.  The required `effects` attribute makes mutation/state
// ownership explicit for FFT, sort, cache, optimizer, and other non-pure
// entries; this op is deliberately not marked Pure.
def Tile_X86ABIKernelOp : Tile_Op<"x86_abi_kernel"> {
  let summary = "Typed launch of a versioned x86 stable ABI";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level scaled dot-product attention envelope. Inputs are Q, K, V,
// optional bias, O, and (when lse_checkpoint="saved") row_lse, followed by B,
// Hq, Hkv, Sq, Sk, D, and Dv. row_lse is a physical f32[B,Hq,Sq] output, not
// metadata: its presence changes both the pointer ABI and kernel arguments.
def Tile_AttentionKernelOp : Tile_Op<"attention_kernel"> {
  let summary = "Launch-level typed scaled dot-product attention";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level BLOCK-ATTNRES-1 depth-attention envelope. Inputs are query,
// sources, output, source_count, rows, and width.  The two recurrence strings
// freeze the statistics and associative merge algorithms chosen in Schedule
// IR, so a target consumer may tune their physical realization but cannot
// replace the mathematics after the content-addressed boundary.
def Tile_DepthAttentionKernelOp : Tile_Op<"depth_attention_kernel"> {
  let summary = "Launch-level typed depth-attention statistics/merge program";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level scaled dot-product attention VJP envelope. Inputs are dO, Q,
// K, V, optional dense f32 bias, optional saved f32 row_lse, dQ, dK, dV,
// followed by B, Hq, Hkv, Sq, Sk, D, and Dv. The deterministic-direct compatibility route assigns one output
// gradient element to one thread. The canonical split/reduced route instead
// makes launch-owned workspace, query/KV block loops, split count, and fixed
// reduction order explicit as attributes; physical backends schedule those
// loops but do not redefine their ownership or ordering.
def Tile_AttentionBackwardKernelOp : Tile_Op<"attention_backward_kernel"> {
  let summary = "Launch-level deterministic scaled dot-product attention VJP";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level stateful-training envelope. Lion VJP uses five input pointers
// (p/g/m/dp_out/dm_out), three fresh output pointers, and an i64 element count.
// The content hash binds the complete logical-buffer lineage; mutation and
// aliasing are explicit rather than inferred from pointer equality.
def Tile_TrainingKernelOp : Tile_Op<"training_kernel"> {
  let summary = "Launch-level typed stateful training program";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level implicit-function VJP. Inputs are parameter, converged
// solution, output cotangent, residual scratch, matrix-free solve scratch,
// parameter cotangent, and element count. The residual model and solver policy
// are fixed by the content hash; targets must fail closed on unknown models.
def Tile_SolverIFTKernelOp : Tile_Op<"solver_ift_kernel"> {
  let summary = "Launch-level typed residual/linear-solve/residual-adjoint chain";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Launch-level EGGROLL W2 correction. Inputs are x/member-ids/key/output
// pointers followed by P, rows-per-member, in-dim, and out-dim. The RNG and
// numeric identities are attributes covered by tessera.schedule_hash.
def Tile_ESLowRankCorrectionKernelOp
    : Tile_Op<"es_low_rank_correction_kernel"> {
  let summary = "Launch-level typed member-keyed ES rank-1 correction";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Canonical logical-page gather. Inputs are physical pages, i32 page table,
// output, and P/LP/page-size/H/D/start/token-count dimensions.
def Tile_PagedKVReadKernelOp : Tile_Op<"paged_kv_read_kernel"> {
  let summary = "Launch-level typed paged-KV logical-to-physical gather";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Compiler-owned fused paged attention. Inputs are Q, K pages, V pages, i32
// logical-to-physical page table, i64 logical token indices, O, followed by
// P/LP/page-size/H/Q/T/D/causal-offset. The offset is an explicit absolute
// query-position contract; no runtime route may infer it from allocation size.
def Tile_PagedAttentionKernelOp : Tile_Op<"paged_attention_kernel"> {
  let summary = "Launch-level fused paged-KV causal attention";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// ReplaySSM output reconstruction from a session-resident checkpoint and ring.
// Inputs are delta, x, B, S0, C, A, Y, followed by B/D/N/M dimensions.
def Tile_ReplaySSMDecodeKernelOp : Tile_Op<"replay_ssm_decode_kernel"> {
  let summary = "Launch-level ReplaySSM output-only reconstruction";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Deterministic checkpoint fold. Inputs are delta, x, B, S0, A, followed by
// B/D/N/M. S0 is the only written persistent buffer.
def Tile_ReplaySSMFlushKernelOp : Tile_Op<"replay_ssm_flush_kernel"> {
  let summary = "Launch-level deterministic ReplaySSM checkpoint fold";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Canonical local-device MoE gather. Inputs are X, token-of-slot, O, T, S, H.
def Tile_MoEDispatchKernelOp : Tile_Op<"moe_dispatch_kernel"> {
  let summary = "Launch-level canonical MoE token gather";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Canonical deterministic weighted combine. Inputs are partials,
// token-of-slot, weights, O, T, S, H.
def Tile_MoECombineKernelOp : Tile_Op<"moe_combine_kernel"> {
  let summary = "Launch-level deterministic MoE weighted combine";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Canonical contiguous/ragged grouped expert GEMM. Inputs are X, expert
// weights, int32 group offsets, O, T, K, N, E.
def Tile_GroupedGemmKernelOp : Tile_Op<"grouped_gemm_kernel"> {
  let summary = "Launch-level ragged grouped expert GEMM";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 1;
}

// Workgroup / wave barrier (s_barrier / bar.sync) — drains all outstanding work.
def Tile_SBarrierOp : Tile_Op<"s_barrier"> {
  let summary = "Workgroup/wave barrier (drains outstanding async work)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
}

//===----------------------------------------------------------------------===//
// C1 (2026-06-23) — TileLayout algebra (TIRx review, COMPILER_AUDIT item C1)
//
// Replaces the flat `tessera.layout` *string* enum (row_major/bhsd/…) with a
// structured logical→physical placement object, faithful to the book's
// `S[(extents):(strides@axis)] (+ R[count:stride@axis]) + offset` and the
// separate, non-affine `ComposeLayout(swizzle, tile)`:
//
//   shard   (S) — partitions logical indices across named hardware axes;
//   replica (R) — one-to-many copies (broadcast / multicast) at extra offsets;
//   offset  (O) — a constant placement shift;
//   swizzle     — an XOR bank-conflict permutation applied AFTER the affine
//                 stride map (kept a *separate* attribute, never folded in).
//
// Named hardware axes (the verifier's accept-set) make placement unambiguous:
//   memory/TMEM  : m, tlane, tcol           (storage axes — what aliases)
//   thread       : laneid, warpid, reg, tid_in_wg, wid_in_wg
//   grid/cluster : bx, by, bz, cbx, cby, cbz
//   device       : gpuid_x, gpuid_y         (same algebra as ShardSpec, #3)
//===----------------------------------------------------------------------===//

class Tile_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Tessera_Tile_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Tile_SwizzleAttr : Tile_Attr<"TileSwizzle", "swizzle"> {
  let summary = "XOR swizzle composed atop a TileLayout (non-affine)";
  let description = [{
    `addr = (x2 << per_element) | low, x2 = x ^ ((x >> ...) & mask)` — the
    standard shared-memory bank-conflict permutation, parameterized by
    `per_element` / `len` / `atom`. Derived from dtype + descriptor mode, not
    hand-tuned. Held separately from the affine `#tile.layout` so the layout
    stays analyzable.
  }];
  let parameters = (ins "int64_t":$perElement,
                        "int64_t":$swizzleLen,
                        "int64_t":$atomLen);
  let assemblyFormat =
      "`<` `per_element` `=` $perElement `,` `len` `=` $swizzleLen `,` "
      "`atom` `=` $atomLen `>`";
}

def Tile_LayoutAttr : Tile_Attr<"TileLayout", "layout"> {
  let summary = "Logical→physical tile layout: shard ⊕ replica ⊕ offset (+swizzle)";
  let description = [{
    `L(x) = { D(x) + r + offset | r in replica }` where `D` walks the shard
    extents in row-major order and accumulates `stride` onto the named hardware
    `axis`. Example — a tensor-core fragment across lanes/registers:
    `#tile.layout<shard = [8, 4, 2] : [4, 1, 1] @ ["laneid", "laneid", "reg"],
                  replica = [] : [] @ [], offset = 0>`.
  }];
  let parameters = (ins
      ArrayRefParameter<"int64_t">:$shardExtents,
      ArrayRefParameter<"int64_t">:$shardStrides,
      ArrayRefParameter<"::mlir::StringAttr">:$shardAxes,
      ArrayRefParameter<"int64_t">:$replicaCounts,
      ArrayRefParameter<"int64_t">:$replicaStrides,
      ArrayRefParameter<"::mlir::StringAttr">:$replicaAxes,
      "int64_t":$offset,
      OptionalParameter<"::tessera::tile::TileSwizzleAttr">:$swizzle);
  let genVerifyDecl = 1;
  // Hand-written parse/print (TileDialect.cpp): the default ArrayRefParameter
  // parser rejects empty `[]` lists, but a layout with no replica term
  // (replica = []) is the common case.
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// C3 (2026-06-23) — typed barrier domains + PipelineState (TIRx review, item C3)
//
// Three barrier primitives with distinct completion semantics, plus the
// ring-buffer PipelineState the pipeline lowering threads through async loops.
// The key invariant — producer initial phase = 1, consumer initial phase = 0 —
// is the packaged fix for the classic off-by-one ring deadlock; it is checked
// across ops by TilePipelineLegalityPass (the attribute verifier only checks the
// per-attribute structural bounds).
//===----------------------------------------------------------------------===//

def Tile_BarrierAttr : Tile_Attr<"TileBarrier", "barrier"> {
  let summary = "A typed pipeline barrier — completion semantics by kind";
  let description = [{
    Barrier domains (TIRx FA-4):
      `tma`      — byte-count / async-transaction signaled (TMABar);
      `tcgen05`  — MMA-completion signaled (TCGen05Bar);
      `mbarrier` — thread-arrival signaled (MBarrier).
    `expect` is the arrival / byte count the barrier waits on.
  }];
  let parameters = (ins StringRefParameter<"">:$kind, "int64_t":$expect);
  let assemblyFormat = "`<` `kind` `=` $kind `,` `expect` `=` $expect `>`";
  let genVerifyDecl = 1;
}

def Tile_PipelineStateAttr
    : Tile_Attr<"TilePipelineState", "pipeline_state"> {
  let summary = "Ring-buffer pipeline state: (depth, stage, phase, role)";
  let description = [{
    `role` is `producer` | `consumer`. The producer's initial `phase` must be 1
    and the consumer's 0 so the first wait falls through (the off-by-one deadlock
    fix). `stage` indexes the ring (`< depth`); `phase` is the parity bit (0|1).
  }];
  let parameters = (ins "int64_t":$depth, "int64_t":$stage, "int64_t":$phase,
                        StringRefParameter<"">:$role);
  let assemblyFormat = "`<` `depth` `=` $depth `,` `stage` `=` $stage `,` "
                       "`phase` `=` $phase `,` `role` `=` $role `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// C5 scaffold (2026-06-23) — independent per-ring pipeline depths (TIRx item C5)
//
// The book's insight: FA-4 runs three *independent* rings (Q depth 2, KV depth
// 3, TMEM depth 2), not one global `pipeline_stages`. This attribute is the
// first-class IR vocabulary for that — the hardware-free half of C5 (the IR
// surface + the Python autotuner sweep). The measured per-ring sweep + the
// kernel that consumes these depths stay hardware-gated (Phase G/H).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Typed buffer-reference contract (2026-06-23) — strengthens the C2/C6 marker
// conventions (`tile.buffer` string + `tile.access` string) into one typed
// attribute whose memory space and access mode are a closed, verifier-checked
// set. The buffer identity is still a name today; the next step is an SSA
// `!tile.buffer` handle produced by `tile.alloc` / consumed by `tile.dealloc`
// (def-use lifetimes instead of name matching) — see COMPILER_AUDIT.
//===----------------------------------------------------------------------===//

def Tile_BufferRefAttr : Tile_Attr<"TileBufferRef", "buffer_ref"> {
  let summary = "A typed staged-buffer reference: name + memory space + access";
  let description = [{
    Replaces the bare `tile.buffer="…"` + `tile.access="…"` string markers. The
    `space` (smem / tmem / gmem) and `access` (read / write / free) are typed,
    closed sets enforced by the verifier — a stronger contract than two free
    strings. Deprecated migration syntax: active lowering and legality passes
    require `!tile.buffer` def-use identity.
  }];
  let parameters = (ins StringRefParameter<"">:$name,
                        StringRefParameter<"">:$space,
                        StringRefParameter<"">:$access);
  let assemblyFormat =
      "`<` `name` `=` $name `,` `space` `=` $space `,` `access` `=` $access `>`";
  let genVerifyDecl = 1;
}

// Portable block-scale addressing contract. This does not encode an NVIDIA
// scale-selector register or an AMD instruction form. `block_size` is measured
// in logical values and `stride` in scale elements. `axis=-1`, dtype/layout
// "none", and block_size=0 describe a format with no scale operand.
def Tile_ScaleLayoutAttr : Tile_Attr<"TileScaleLayout", "scale_layout"> {
  let summary = "Portable block-scale storage and indexing contract";
  let parameters = (ins StringRefParameter<"">:$dtype,
                        "int64_t":$blockSize,
                        "int64_t":$axis,
                        StringRefParameter<"">:$layout,
                        "int64_t":$stride,
                        "int64_t":$alignment,
                        "int64_t":$offset);
  let assemblyFormat =
      "`<` `dtype` `=` $dtype `,` `block_size` `=` $blockSize `,` "
      "`axis` `=` $axis `,` `layout` `=` $layout `,` `stride` `=` $stride `,` "
      "`alignment` `=` $alignment `,` `offset` `=` $offset `>`";
  let genVerifyDecl = 1;
}

// Target-neutral packed-format contract. Terminal dtype legalization can know
// this portion without inventing an A/B-specific axis or scale layout.
// `logical_bits` is independent of `elements_per_container`: FP6 in an i8
// container therefore remains a six-bit format even though its pack factor is
// one.
def Tile_PackedFormatAttr : Tile_Attr<"TilePackedFormat", "packed_format"> {
  let summary = "Structured logical dtype and physical container format";
  let parameters = (ins StringRefParameter<"">:$logicalType,
                        StringRefParameter<"">:$containerType,
                        "int64_t":$logicalBits,
                        "int64_t":$elementsPerContainer,
                        StringRefParameter<"">:$signedness,
                        StringRefParameter<"">:$encoding,
                        StringRefParameter<"">:$laneOrder);
  let assemblyFormat =
      "`<` `logical` `=` $logicalType `,` `container` `=` $containerType `,` "
      "`logical_bits` `=` $logicalBits `,` "
      "`elements_per_container` `=` $elementsPerContainer `,` "
      "`signedness` `=` $signedness `,` `encoding` `=` $encoding `,` "
      "`lane_order` `=` $laneOrder `>`";
  let genVerifyDecl = 1;
}

// Target-neutral physical view for one concrete packed buffer. Axes, strides,
// offsets, and scale layout are per-view rather than incorrectly shared by A
// and B at a launch-level matmul operation.
def Tile_PackedPhysicalViewAttr
    : Tile_Attr<"TilePackedPhysicalView", "packed_view"> {
  let summary = "Structured logical-to-packed physical storage descriptor";
  let parameters = (ins "::tessera::tile::TilePackedFormatAttr":$format,
                        "int64_t":$packingAxis,
                        ArrayRefParameter<"int64_t">:$strides,
                        "int64_t":$alignment,
                        "int64_t":$offset,
                        "::tessera::tile::TileScaleLayoutAttr":$scale);
  let assemblyFormat =
      "`<` `format` `=` $format `,` `packing_axis` `=` $packingAxis `,` "
      "`strides` `=` `[` $strides `]` `,` `alignment` `=` $alignment `,` "
      "`offset` `=` $offset `,` `scale` `=` $scale `>`";
  let genVerifyDecl = 1;
}

// The target-neutral cooperative-matrix instruction contract. `family=auto`
// is resolved by the target lowering (mma.sync/WGMMA/tcgen05 on NVIDIA;
// WMMA/MFMA on ROCm). The descriptor records logical compatibility only, never
// lane or register order.
def Tile_MmaDescAttr : Tile_Attr<"TileMmaDesc", "mma_desc"> {
  let summary = "Logical cooperative-matrix instruction descriptor";
  let parameters = (ins StringRefParameter<"">:$family,
                        "int64_t":$m,
                        "int64_t":$n,
                        "int64_t":$k,
                        StringRefParameter<"">:$aType,
                        StringRefParameter<"">:$bType,
                        StringRefParameter<"">:$accType,
                        StringRefParameter<"">:$aLayout,
                        StringRefParameter<"">:$bLayout,
                        "int64_t":$kBlocks);
  let assemblyFormat =
      "`<` `family` `=` $family `,` `m` `=` $m `,` `n` `=` $n `,` `k` `=` $k `,` "
      "`a` `=` $aType `,` `b` `=` $bType `,` `acc` `=` $accType `,` "
      "`a_layout` `=` $aLayout `,` `b_layout` `=` $bLayout `,` "
      "`k_blocks` `=` $kBlocks `>`";
  let genVerifyDecl = 1;
}

// Physical address contract for a pointer-backed tile.view. `leading_dim` is
// measured in elements; zero means a dynamic leading dimension is supplied as
// the final SSA operand of tile.view/tile.store. Origin coordinates and bounds
// are SSA operands.
def Tile_MemoryLayoutAttr : Tile_Attr<"TileMemoryLayout", "memory_layout"> {
  let summary = "Physical storage layout for a pointer-backed logical tile";
  let parameters = (ins StringRefParameter<"">:$space,
                        StringRefParameter<"">:$order,
                        "int64_t":$leadingDim);
  let assemblyFormat = "`<` `space` `=` $space `,` `order` `=` $order `,` "
                       "`leading_dim` `=` $leadingDim `>`";
  let genVerifyDecl = 1;
}

// Post-MMA transformation before the final Tile store. Physical accumulator
// registers remain target-owned; bias, activation, and output conversion are
// portable semantics shared by NVIDIA and ROCm materializers.
def Tile_EpilogueAttr : Tile_Attr<"TileEpilogue", "epilogue"> {
  let summary = "Portable matrix-accumulator epilogue contract";
  let parameters = (ins "bool":$bias,
                        StringRefParameter<"">:$activation,
                        StringRefParameter<"">:$outputType);
  let assemblyFormat = "`<` `bias` `=` $bias `,` `activation` `=` $activation `,` "
                       "`output` `=` $outputType `>`";
  let genVerifyDecl = 1;
}

def Tile_PipelineDepthsAttr
    : Tile_Attr<"TilePipelineDepths", "pipeline_depths"> {
  let summary = "Independent per-ring pipeline depths (FA-4: Q / KV / TMEM)";
  let description = [{
    `q` / `kv` / `tmem` are the three independent ring depths a warp-specialized
    attention kernel double-buffers separately (vs. one global `pipeline_stages`
    knob). Each must be >= 1. Book defaults: q=2, kv=3, tmem=2.
  }];
  let parameters = (ins "int64_t":$q, "int64_t":$kv, "int64_t":$tmem);
  let assemblyFormat = "`<` `q` `=` $q `,` `kv` `=` $kv `,` `tmem` `=` $tmem `>`";
  let genVerifyDecl = 1;
}

#endif // TESSERA_TILE_OPS
