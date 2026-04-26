//===- AttnOps.cpp — FA-4 attention op verifiers + helpers ───────────────===//
//
// Implements verifiers for the Phase 3 attention ops defined in Attn.td:
//   LseSaveOp, ScaledDotProductOp, OnlineSoftmaxOp (FA-2 invariants)
//
// All ops use ODS-generated accessors; this file only fills hasVerifier=1 slots
// and provides the FA-2 online softmax documentation helpers.
//
// FA-2 online softmax algorithm (Dao et al. 2022):
//
//   For each KV tile j:
//     S_j  = Q · K_j^T / sqrt(d_k)               [ScaledDotProductOp]
//     m_j  = max(m_{j-1}, rowmax(S_j))
//     l_j  = exp(m_{j-1} - m_j) * l_{j-1}
//            + rowsum(exp(S_j - m_j))
//     O_j  = diag(exp(m_{j-1} - m_j)) * O_{j-1}
//            + exp(S_j - m_j) * V_j               [OnlineSoftmaxOp]
//   Final: O = diag(1/l) * O,  LSE = m + log(l)  [LseAccumulateOp]
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

// ODS-generated declarations (produced by mlir-tblgen from Attn.td).
// Include the generated header; actual path depends on CMake binary dir.
#include "AttnOps.h.inc"

namespace tessera {
namespace attn {

// ── LseSaveOp verifier ────────────────────────────────────────────────────
// scores must be a ranked tensor; LSE result has one fewer dimension (row vec).
mlir::LogicalResult LseSaveOp::verify() {
  auto srcType = mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  if (!srcType)
    return emitOpError("scores must be a ranked tensor");
  if (srcType.getRank() < 2)
    return emitOpError("scores tensor must have rank >= 2 (at least [rows, cols])");
  return mlir::success();
}

// ── ScaledDotProductOp verifier ───────────────────────────────────────────
// query: [tile_q × d_k]  key: [tile_kv × d_k]  → scores: [tile_q × tile_kv]
mlir::LogicalResult ScaledDotProductOp::verify() {
  auto qType = mlir::dyn_cast<mlir::RankedTensorType>(getQuery().getType());
  auto kType = mlir::dyn_cast<mlir::RankedTensorType>(getKey().getType());
  if (!qType || !kType)
    return emitOpError("query and key must be ranked tensors");
  if (qType.getRank() != 2 || kType.getRank() != 2)
    return emitOpError("query and key must be 2-D tensors [tile × head_dim]");
  // d_k dimension must match (dim 1 of Q == dim 1 of K).
  if (!qType.isDynamicDim(1) && !kType.isDynamicDim(1) &&
      qType.getDimSize(1) != kType.getDimSize(1))
    return emitOpError("query and key head_dim mismatch: ")
           << qType.getDimSize(1) << " vs " << kType.getDimSize(1);
  if (getScale() <= 0.0f && getScale() != -1.0f)
    return emitOpError("scale must be positive (or -1.0 as sentinel)");
  return mlir::success();
}

// ── OnlineSoftmaxOp verifier ──────────────────────────────────────────────
// Verifies the FA-2 loop-invariant shapes are consistent.
mlir::LogicalResult OnlineSoftmaxOp::verify() {
  // scores: [tile_q × tile_kv]
  auto sType = mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  if (!sType || sType.getRank() != 2)
    return emitOpError("scores must be a 2-D ranked tensor");

  // running_m / running_l: scalar f32 or 1-D [tile_q] — accept both.
  // acc_out: [tile_q × d_v] — opaque check only.
  return mlir::success();
}

// ── Register dialect ops from ODS-generated definitions ──────────────────
#include "AttnDialect.cpp.inc"
#include "AttnOps.cpp.inc"

} // namespace attn
} // namespace tessera
