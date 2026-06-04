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
}

class Tile_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tessera_Tile_Dialect, mnemonic, traits>;

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

#endif // TESSERA_TILE_OPS
