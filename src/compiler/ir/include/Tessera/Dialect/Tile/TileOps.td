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
  // Plain attributes (no op-properties storage) — keeps the dialect header-only
  // and matches the Tessera / Apple dialects.
  let usePropertiesForAttributes = 0;
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
// dependency into a verifier-checkable SSA def-use edge, replacing the
// transitional tile.barrier_id / tile.depends_on string markers + program order.
// It rides the ops' existing Variadic<AnyType> slots, so no op signature changes
// and every generic-form fixture stays valid; the planner starts emitting it in
// Phase C. Kept backend-neutral (one type, no NVIDIA/ROCm privilege).
def Tile_AsyncTokenType : Tile_Type<"AsyncToken", "async_token"> {
  let summary = "Completion token for an asynchronous tile copy";
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

//===----------------------------------------------------------------------===//
// Tile synchronization ops (TIRx op-layer convergence, 2026-06-23, Phase A0).
//
// The copy/wait/mma/barrier seam was previously emitted as opaque generic ops
// (`"tile.async_copy"(...)`), legal only because TesseraTileDialect calls
// allowUnknownOperations(true). Registering them makes the seam a first-class op
// — a precondition for giving tile.async_copy a typed completion result (the
// !tile.async_token SSA edge added in Phase B), which replaces the transitional
// `tile.barrier_id` / `tile.depends_on` string markers + program-order reasoning.
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
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
}

// Wait on outstanding async copies (waitcnt(vmcnt) / mbarrier / cp.async.wait).
def Tile_WaitAsyncOp : Tile_Op<"wait_async"> {
  let summary = "Wait on outstanding asynchronous tile copies";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
}

// Tile-level tensor-core MMA (WGMMA / MFMA / WMMA / tcgen05).
def Tile_MMAOp : Tile_Op<"mma"> {
  let summary = "Tile-level matrix multiply-accumulate (tensor-core)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
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
    strings. Carried as `tile.buf = #tile.buffer_ref<…>` on staged writes/frees.
  }];
  let parameters = (ins StringRefParameter<"">:$name,
                        StringRefParameter<"">:$space,
                        StringRefParameter<"">:$access);
  let assemblyFormat =
      "`<` `name` `=` $name `,` `space` `=` $space `,` `access` `=` $access `>`";
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
