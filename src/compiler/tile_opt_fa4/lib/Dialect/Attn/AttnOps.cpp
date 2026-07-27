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

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

// Sprint V7b (2026-05-22): the Attn dialect anchors its eager-load
// extension on the parent `tessera` (Graph IR) dialect.  Including
// TesseraOps.h pulls in the TesseraDialect class declaration without
// dragging the full IR header surface.
#include "Tessera/IR/TesseraOps.h"

// ODS-generated declarations (produced by mlir-tblgen from Attn.td).
// Include the generated header; actual path depends on CMake binary dir.
#include "AttnDialect.h.inc"
#define GET_OP_CLASSES
#include "AttnOps.h.inc"

namespace tessera {
namespace attn {

static mlir::LogicalResult verifyFloatScalarOrRankedTensor(
    mlir::Operation *op, mlir::Type type, llvm::StringRef label,
    int64_t maxRank);

static mlir::LogicalResult verifyLseCheckpoint(
    mlir::Operation *op, mlir::MemRefType checkpoint, llvm::StringRef identity,
    llvm::StringRef memorySpace, llvm::StringRef scope,
    mlir::StringAttr cachePolicy) {
  if ((checkpoint.getRank() != 1 && checkpoint.getRank() != 2) ||
      !checkpoint.getElementType().isF32())
    return op->emitOpError(
        "checkpoint must be a rank-1/2 f32 memref with flattened or "
        "batch/head-row layout");
  if (identity.empty())
    return op->emitOpError("checkpoint identity must be non-empty");
  if (memorySpace != "global")
    return op->emitOpError(
        "persistent cross-entry LSE currently requires memory_space = "
        "\"global\"");
  if (scope != "program_launch" && scope != "session")
    return op->emitOpError(
        "checkpoint scope must be \"program_launch\" or \"session\"");
  if (cachePolicy && cachePolicy.getValue() != "default" &&
      cachePolicy.getValue() != "streaming" &&
      cachePolicy.getValue() != "cache")
    return op->emitOpError(
        "cache_policy must be \"default\", \"streaming\", or \"cache\"");
  return mlir::success();
}

mlir::LogicalResult LseSaveOp::verify() {
  if (failed(verifyFloatScalarOrRankedTensor(
          getOperation(), getLse().getType(), "lse", /*maxRank=*/1)))
    return mlir::failure();
  return verifyLseCheckpoint(
      getOperation(), mlir::cast<mlir::MemRefType>(getDestination().getType()),
      getIdentity(),
      getMemorySpace(), getScope(), getCachePolicyAttr());
}

static bool attnDimsAgree(int64_t a, int64_t b) {
  return mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b) ||
         a == b;
}

static bool isFloatElementType(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(type);
}

static mlir::LogicalResult verifyFloatScalarOrRankedTensor(mlir::Operation *op,
                                                           mlir::Type type,
                                                           llvm::StringRef label,
                                                           int64_t maxRank) {
  if (isFloatElementType(type))
    return mlir::success();
  if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (ranked.getRank() > maxRank)
      return op->emitOpError()
             << label << " must have rank <= " << maxRank;
    if (!isFloatElementType(ranked.getElementType()))
      return op->emitOpError() << label << " element type must be floating";
    return mlir::success();
  }
  return op->emitOpError()
         << label << " must be a floating scalar or ranked tensor";
}

static mlir::LogicalResult verifySameRankedTensor(mlir::Operation *op,
                                                  mlir::RankedTensorType a,
                                                  mlir::RankedTensorType b,
                                                  llvm::StringRef label) {
  if (a.getRank() != b.getRank())
    return op->emitOpError() << label << " ranks must match";
  if (a.getElementType() != b.getElementType())
    return op->emitOpError() << label << " element types must match";
  for (int64_t i = 0, e = a.getRank(); i < e; ++i)
    if (!attnDimsAgree(a.getDimSize(i), b.getDimSize(i)))
      return op->emitOpError() << label << " shapes must match";
  return mlir::success();
}

// Sprint V6c (2026-05-22) — target-aware tile size limits.
//
// Per-SM (tile_q_max, tile_kv_max) for the FA-4 ScaledDotProduct kernel.
// Generalizes the V3 FlashAttnOp head_dim pattern to the FA-4 Tile IR
// op family.  Numbers come from the canonical FA-2/FA-3/FA-4 kernel
// shapes documented in `docs/backends/nvidia/kernel-inventory.md` +
// CLAUDE.md Phase 3 description.
//
// sm_70 / sm_75 / sm_80 / sm_86 / sm_89  — FA-2 baseline: 64 × 128
// sm_90 / sm_90a / sm_100 / sm_100a / sm_120 / sm_120a — FA-3/FA-4: 128 × 256
// no SM tag — CPU reference; no tile-size limit applied.
//
// Returns {0, 0} when the SM is unknown (so the verifier skips the
// check rather than rejecting unrecognized strings).
static std::pair<int64_t, int64_t>
maxTileSizesForTargetSm(llvm::StringRef sm) {
  return llvm::StringSwitch<std::pair<int64_t, int64_t>>(sm)
      .Case("sm_70",   {64, 128})
      .Case("sm_75",   {64, 128})
      .Case("sm_80",   {64, 128})
      .Case("sm_86",   {64, 128})
      .Case("sm_89",   {64, 128})
      .Case("sm_90",   {128, 256})
      .Case("sm_90a",  {128, 256})
      .Case("sm_100",  {128, 256})
      .Case("sm_100a", {128, 256})
      .Case("sm_120",  {128, 256})
      .Case("sm_120a", {128, 256})
      .Default({0, 0});  // unknown / no limit applied
}

// ── ScaledDotProductOp verifier ───────────────────────────────────────────
// query: [tile_q × d_k]  key: [tile_kv × d_k]  → scores: [tile_q × tile_kv]
//
// Sprint V6c (2026-05-22): also walks the parent op chain for a
// `tessera.target_sm` attribute and enforces the per-SM (tile_q, tile_kv)
// ceiling.  The Q tile size is read from query.dim(0); the KV tile
// size from key.dim(0).  Functions without the attribute (CPU
// reference path) skip the target-aware check.
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
  double scale = getScale().convertToDouble();
  if (scale <= 0.0 && scale != -1.0)
    return emitOpError("scale must be positive (or -1.0 as sentinel)");
  // Sprint V6c — target-aware tile size limits.  Walk parent op chain
  // for `tessera.target_sm`; enforce per-SM (tile_q_max, tile_kv_max).
  mlir::Operation *parent = (*this)->getParentOp();
  while (parent && !parent->hasAttr("tessera.target_sm"))
    parent = parent->getParentOp();
  if (parent) {
    if (auto attr = mlir::dyn_cast<mlir::StringAttr>(
            parent->getAttr("tessera.target_sm"))) {
      auto [tileQMax, tileKvMax] = maxTileSizesForTargetSm(attr.getValue());
      if (tileQMax > 0 && !qType.isDynamicDim(0)
          && qType.getDimSize(0) > tileQMax)
        return emitOpError("tile_q=")
               << qType.getDimSize(0)
               << " exceeds the SM " << attr.getValue()
               << " ScaledDotProduct kernel limit of " << tileQMax
               << " (Sprint V6c FA-4 tile size table)";
      if (tileKvMax > 0 && !kType.isDynamicDim(0)
          && kType.getDimSize(0) > tileKvMax)
        return emitOpError("tile_kv=")
               << kType.getDimSize(0)
               << " exceeds the SM " << attr.getValue()
               << " ScaledDotProduct kernel limit of " << tileKvMax
               << " (Sprint V6c FA-4 tile size table)";
    }
  }
  return mlir::success();
}

mlir::LogicalResult ScoreBiasOp::verify() {
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(getBias().getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(getBiasedScores().getType());
  if (!scoresType || !biasType || !resultType || scoresType.getRank() != 2 ||
      biasType.getRank() != 2)
    return emitOpError("scores, bias, and result must be rank-2 tensors");
  if (!scoresType.getElementType().isF32() ||
      !biasType.getElementType().isF32())
    return emitOpError("scores and bias must use f32");
  if (failed(verifySameRankedTensor(getOperation(), scoresType, biasType,
                                    "scores/bias")) ||
      failed(verifySameRankedTensor(getOperation(), scoresType, resultType,
                                    "scores/result")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult SoftcapOp::verify() {
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(getCappedScores().getType());
  if (!scoresType || !resultType || scoresType.getRank() != 2)
    return emitOpError("scores and result must be rank-2 tensors");
  if (!scoresType.getElementType().isF32())
    return emitOpError("scores must use f32");
  if (!getCap().isFinite() || getCap().convertToDouble() <= 0.0)
    return emitOpError("cap must be finite and positive");
  return verifySameRankedTensor(getOperation(), scoresType, resultType,
                                "scores/result");
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

mlir::LogicalResult StreamingUpdateOp::verify() {
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto valueType =
      mlir::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  auto accType =
      mlir::dyn_cast<mlir::RankedTensorType>(getAccOut().getType());
  auto newAccType =
      mlir::dyn_cast<mlir::RankedTensorType>(getNewAcc().getType());
  auto runningMType =
      mlir::dyn_cast<mlir::RankedTensorType>(getRunningM().getType());
  auto runningLType =
      mlir::dyn_cast<mlir::RankedTensorType>(getRunningL().getType());
  auto newMType =
      mlir::dyn_cast<mlir::RankedTensorType>(getNewM().getType());
  auto newLType =
      mlir::dyn_cast<mlir::RankedTensorType>(getNewL().getType());
  if (!scoresType || !valueType || !accType || !newAccType)
    return emitOpError(
        "scores, value, accumulator, and new accumulator must be ranked tensors");
  if (scoresType.getRank() != 2 || valueType.getRank() != 2 ||
      accType.getRank() != 2 || newAccType.getRank() != 2)
    return emitOpError(
        "expects scores[Q,KV], value[KV,Dv], and accumulators[Q,Dv]");
  if (!scoresType.isDynamicDim(1) && !valueType.isDynamicDim(0) &&
      scoresType.getDimSize(1) != valueType.getDimSize(0))
    return emitOpError("score KV width must match value KV rows");
  if (!scoresType.isDynamicDim(0) && !accType.isDynamicDim(0) &&
      scoresType.getDimSize(0) != accType.getDimSize(0))
    return emitOpError("score query rows must match accumulator rows");
  if (!valueType.isDynamicDim(1) && !accType.isDynamicDim(1) &&
      valueType.getDimSize(1) != accType.getDimSize(1))
    return emitOpError("value width must match accumulator width");
  if (failed(verifySameRankedTensor(getOperation(), accType, newAccType,
                                    "acc/new_acc")))
    return mlir::failure();
  if (!runningMType || !runningLType || !newMType || !newLType ||
      runningMType.getRank() != 1 || runningLType.getRank() != 1 ||
      newMType.getRank() != 1 || newLType.getRank() != 1)
    return emitOpError(
        "running and updated max/sum must be rank-1 per-query tensors");
  const std::pair<mlir::RankedTensorType, llvm::StringRef> statistics[] = {
      {runningMType, "running_m"},
      {runningLType, "running_l"},
      {newMType, "new_m"},
      {newLType, "new_l"}};
  for (auto [type, label] : statistics) {
    if (!type.getElementType().isF32())
      return emitOpError() << label << " must use FP32 accumulation";
    if (!scoresType.isDynamicDim(0) && !type.isDynamicDim(0) &&
        scoresType.getDimSize(0) != type.getDimSize(0))
      return emitOpError() << label << " length must match score query rows";
  }
  if (!scoresType.getElementType().isF32() ||
      !accType.getElementType().isF32())
    return emitOpError("scores and output accumulator must use FP32");
  return mlir::success();
}

mlir::LogicalResult LseLoadOp::verify() {
  if (failed(verifyFloatScalarOrRankedTensor(
          getOperation(), getLse().getType(), "lse", /*maxRank=*/1)))
    return mlir::failure();
  return verifyLseCheckpoint(
      getOperation(), mlir::cast<mlir::MemRefType>(getSource().getType()),
      getIdentity(), getMemorySpace(), getScope(), getCachePolicyAttr());
}

mlir::LogicalResult LseAccumulateOp::verify() {
  // acc / output are the running attention accumulator [tile_q × d_v] — always
  // 2-D ranked tensors of matching type.
  auto accType = mlir::dyn_cast<mlir::RankedTensorType>(getAcc().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (!accType || !outType)
    return emitOpError("acc and output must be ranked tensors");
  if (accType.getRank() != 2 || outType.getRank() != 2)
    return emitOpError("acc and output must be 2-D tensors [tile_q, d_v]");
  if (failed(verifySameRankedTensor(getOperation(), accType, outType,
                                    "acc/output")))
    return mlir::failure();

  // running_m / running_l / lse are per-row softmax statistics — a scalar f32
  // or a rank-1 [tile_q] tensor. This matches OnlineSoftmaxOp / LseLoadOp and
  // what TileIRLoweringPass emits (scalar f32 in the reduced-loop form); a
  // strict rank-1-only rule here rejected the pass's own output.
  if (failed(verifyFloatScalarOrRankedTensor(
          getOperation(), getRunningM().getType(), "running_m", /*maxRank=*/1)) ||
      failed(verifyFloatScalarOrRankedTensor(
          getOperation(), getRunningL().getType(), "running_l", /*maxRank=*/1)) ||
      failed(verifyFloatScalarOrRankedTensor(
          getOperation(), getLse().getType(), "lse", /*maxRank=*/1)))
    return mlir::failure();

  // Cross-shape checks apply to whichever statistics carry a per-row shape. A
  // scalar (or rank-0) value broadcasts and carries none, but every rank-1
  // statistic is a per-row [tile_q] vector: it must match the acc tile_q and
  // agree with the other rank-1 statistics — independent of which (if any) of
  // the three are scalar. Keying these off running_m alone would let a scalar
  // running_m mask a mismatched rank-1 running_l/lse.
  const std::pair<mlir::Type, llvm::StringRef> stats[] = {
      {getRunningM().getType(), "running_m"},
      {getRunningL().getType(), "running_l"},
      {getLse().getType(), "lse"}};
  mlir::RankedTensorType ref;
  llvm::StringRef refLabel;
  for (const auto &[type, label] : stats) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || ranked.getRank() == 0)
      continue; // scalar / rank-0 — no per-row [tile_q] shape to cross-check
    if (!attnDimsAgree(accType.getDimSize(0), ranked.getDimSize(0)))
      return emitOpError() << label << " length must match acc tile_q";
    if (!ref) {
      ref = ranked;
      refLabel = label;
      continue;
    }
    if (failed(verifySameRankedTensor(
            getOperation(), ref, ranked,
            (llvm::Twine(refLabel) + "/" + label).str())))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult DropoutMaskOp::verify() {
  double p = getDropoutP().convertToDouble();
  if (!(p >= 0.0) || !(p < 1.0))
    return emitOpError("dropout_p must satisfy 0.0 <= p < 1.0");
  if (getSeed() < 0)
    return emitOpError("seed must be non-negative");
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto maskedType =
      mlir::dyn_cast<mlir::RankedTensorType>(getMaskedScores().getType());
  if (!scoresType || !maskedType)
    return emitOpError("scores and masked_scores must be ranked tensors");
  if (scoresType.getRank() != 2)
    return emitOpError("scores must be a 2-D attention tile");
  return verifySameRankedTensor(getOperation(), scoresType, maskedType,
                                "dropout scores/masked_scores");
}

mlir::LogicalResult CausalMaskOp::verify() {
  if (getQOffset() < 0 || getKvOffset() < 0)
    return emitOpError("q_offset and kv_offset must be non-negative");
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto maskedType =
      mlir::dyn_cast<mlir::RankedTensorType>(getMaskedScores().getType());
  if (!scoresType || !maskedType)
    return emitOpError("scores and masked_scores must be ranked tensors");
  if (scoresType.getRank() != 2)
    return emitOpError("scores must be a 2-D attention tile");
  return verifySameRankedTensor(getOperation(), scoresType, maskedType,
                                "causal scores/masked_scores");
}

mlir::LogicalResult BoundaryMaskOp::verify() {
  if (static_cast<int64_t>(getWindowLeft()) < -1 ||
      static_cast<int64_t>(getWindowRight()) < -1)
    return emitOpError("window_left and window_right must be >= -1");
  if (getLogicalSk() <= 0)
    return emitOpError("logical_sk must be positive");
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto maskedType =
      mlir::dyn_cast<mlir::RankedTensorType>(getMaskedScores().getType());
  if (!scoresType || !maskedType || scoresType.getRank() != 2)
    return emitOpError("scores and masked_scores must be rank-2 tensors");
  return verifySameRankedTensor(getOperation(), scoresType, maskedType,
                                "boundary scores/masked_scores");
}

mlir::LogicalResult BlockDropoutOp::verify() {
  double p = getDropoutP().convertToDouble();
  if (!(p >= 0.0) || !(p < 1.0))
    return emitOpError("dropout_p must satisfy 0.0 <= p < 1.0");
  if (getSeed() < 0)
    return emitOpError("seed must be non-negative");
  auto scoresType =
      mlir::dyn_cast<mlir::RankedTensorType>(getScores().getType());
  auto maskedType =
      mlir::dyn_cast<mlir::RankedTensorType>(getMaskedScores().getType());
  if (!scoresType || !maskedType || scoresType.getRank() != 2)
    return emitOpError("scores and masked_scores must be rank-2 tensors");
  return verifySameRankedTensor(getOperation(), scoresType, maskedType,
                                "dropout scores/masked_scores");
}

mlir::LogicalResult BackwardOp::verify() {
  auto ranked4 = [&](mlir::Type type, llvm::StringRef label)
      -> mlir::FailureOr<mlir::RankedTensorType> {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || ranked.getRank() != 4) {
      emitOpError() << label << " must be a rank-4 [B,H,S,D] tensor";
      return mlir::failure();
    }
    return ranked;
  };
  auto dOut = ranked4(getDOutput().getType(), "d_output");
  auto query = ranked4(getQuery().getType(), "query");
  auto key = ranked4(getKey().getType(), "key");
  auto value = ranked4(getValue().getType(), "value");
  auto dQuery = ranked4(getDQuery().getType(), "d_query");
  auto dKey = ranked4(getDKey().getType(), "d_key");
  auto dValue = ranked4(getDValue().getType(), "d_value");
  auto bias = mlir::dyn_cast<mlir::RankedTensorType>(getBias().getType());
  if (mlir::failed(dOut) || mlir::failed(query) || mlir::failed(key) ||
      mlir::failed(value) || mlir::failed(dQuery) || mlir::failed(dKey) ||
      mlir::failed(dValue))
    return mlir::failure();
  if (!bias || (bias.getRank() != 3 && bias.getRank() != 4) ||
      !bias.getElementType().isF32())
    return emitOpError(
        "bias must be an f32 tensor [B|1,Sq,Sk] or [B|1,Hq|1,Sq,Sk]");
  auto q = *query;
  auto k = *key;
  auto v = *value;
  auto go = *dOut;
  if (!attnDimsAgree(q.getDimSize(0), k.getDimSize(0)) ||
      !attnDimsAgree(q.getDimSize(0), v.getDimSize(0)) ||
      !attnDimsAgree(q.getDimSize(1), go.getDimSize(1)) ||
      !attnDimsAgree(q.getDimSize(2), go.getDimSize(2)) ||
      !attnDimsAgree(k.getDimSize(1), v.getDimSize(1)) ||
      !attnDimsAgree(k.getDimSize(2), v.getDimSize(2)) ||
      !attnDimsAgree(q.getDimSize(3), k.getDimSize(3)) ||
      !attnDimsAgree(v.getDimSize(3), go.getDimSize(3)))
    return emitOpError("Q/K/V/dO batch, head, sequence, or feature dimensions disagree");
  int64_t biasSequenceDim = bias.getRank() - 2;
  if ((!bias.isDynamicDim(0) && bias.getDimSize(0) != 1 &&
       !attnDimsAgree(bias.getDimSize(0), q.getDimSize(0))) ||
      (bias.getRank() == 4 && !bias.isDynamicDim(1) &&
       bias.getDimSize(1) != 1 &&
       !attnDimsAgree(bias.getDimSize(1), q.getDimSize(1))) ||
      !attnDimsAgree(bias.getDimSize(biasSequenceDim), q.getDimSize(2)) ||
      !attnDimsAgree(bias.getDimSize(biasSequenceDim + 1), k.getDimSize(2)))
    return emitOpError(
        "bias must have shape [B|1,Sq,Sk] or [B|1,Hq|1,Sq,Sk]");
  if (!q.isDynamicDim(1) && !k.isDynamicDim(1) &&
      (q.getDimSize(1) % k.getDimSize(1)) != 0)
    return emitOpError("query head count must be divisible by KV head count");
  auto sameShape = [](mlir::RankedTensorType lhs,
                      mlir::RankedTensorType rhs) {
    for (int64_t i = 0; i < lhs.getRank(); ++i)
      if (!attnDimsAgree(lhs.getDimSize(i), rhs.getDimSize(i)))
        return false;
    return true;
  };
  if (!sameShape(q, *dQuery) || !sameShape(k, *dKey) ||
      !sameShape(v, *dValue))
    return emitOpError("dQ/dK/dV shapes must match Q/K/V respectively");
  if (!dQuery->getElementType().isF32() || !dKey->getElementType().isF32() ||
      !dValue->getElementType().isF32())
    return emitOpError("dQ/dK/dV must use f32 accumulation");
  if (!getScale().isFinite() || getScale().convertToDouble() <= 0.0)
    return emitOpError("scale must be finite and positive");
  if (static_cast<int64_t>(getWindowLeft()) < -1 ||
      static_cast<int64_t>(getWindowRight()) < -1)
    return emitOpError("window bounds must be >= -1");
  if (!getSoftcap().isFinite() || getSoftcap().convertToDouble() < 0.0)
    return emitOpError("softcap must be finite and nonnegative");
  double dropout = getDropoutP().convertToDouble();
  if (!getDropoutP().isFinite() || dropout < 0.0 || dropout >= 1.0)
    return emitOpError("dropout_p must satisfy 0 <= p < 1");
  if (getDropoutSeed() < 0)
    return emitOpError("dropout_seed must be nonnegative");
  if (getSplitCount() < 2 || getQueryBlock() <= 0 || getKeyBlock() <= 0)
    return emitOpError("split_count >= 2 and positive block sizes are required");
  return mlir::success();
}

mlir::LogicalResult BackwardDQBlockOp::verify() {
  if (getInputs().size() != 10)
    return emitOpError("expects dO,Q,K,V,bias,batch,q_head,q_offset,kv_offset,dQ");
  for (mlir::Value index : getInputs().slice(5, 4))
    if (!index.getType().isIndex())
      return emitOpError("batch/head/block coordinates must be index values");
  if (getInputs()[9].getType() != getDQuery().getType())
    return emitOpError("dQ result must preserve the loop-carried dQ tensor type");
  auto dQuery =
      mlir::dyn_cast<mlir::RankedTensorType>(getDQuery().getType());
  if (!dQuery || dQuery.getRank() != 4 ||
      !dQuery.getElementType().isF32())
    return emitOpError("loop-carried dQ must be a rank-4 f32 tensor");
  return mlir::success();
}

mlir::LogicalResult BackwardDKDVBlockOp::verify() {
  if (getInputs().size() != 12)
    return emitOpError(
        "expects dO,Q,K,V,bias,batch,kv_head,split,q_offset,kv_offset,partial_dK,partial_dV");
  for (mlir::Value index : getInputs().slice(5, 5))
    if (!index.getType().isIndex())
      return emitOpError("batch/head/split/block coordinates must be index values");
  if (getInputs()[10].getType() != getPartialDKey().getType() ||
      getInputs()[11].getType() != getPartialDValue().getType())
    return emitOpError("partial results must preserve loop-carried workspace tensor types");
  for (mlir::Type type :
       {getPartialDKey().getType(), getPartialDValue().getType()}) {
    auto partial = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!partial || partial.getRank() != 5 ||
        !partial.getElementType().isF32())
      return emitOpError("split partials must be rank-5 f32 tensors");
  }
  return mlir::success();
}

mlir::LogicalResult BackwardReduceSplitOp::verify() {
  if (getInputs().size() != 7)
    return emitOpError("expects partial_dK,partial_dV,batch,kv_head,split,dK,dV");
  for (mlir::Value index : getInputs().slice(2, 3))
    if (!index.getType().isIndex())
      return emitOpError("batch/head/split coordinates must be index values");
  if (getInputs()[5].getType() != getDKey().getType() ||
      getInputs()[6].getType() != getDValue().getType())
    return emitOpError("reduction results must preserve loop-carried dK/dV types");
  for (mlir::Type type : {getDKey().getType(), getDValue().getType()}) {
    auto gradient = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!gradient || gradient.getRank() != 4 ||
        !gradient.getElementType().isF32())
      return emitOpError("reduced gradients must be rank-4 f32 tensors");
  }
  return mlir::success();
}

void TesseraAttnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AttnOps.cpp.inc"
      >();
}

// Sprint V7 (2026-05-22): public registration entry so external
// tools (tessera-opt and future translation drivers) can load this
// dialect via `tessera::attn::registerAttnDialect(registry)`.
// Mirrors the canonical Apple backend pattern
// (`tessera::apple::registerAppleDialect()`).
//
// Do not eagerly load this dialect when the parent `tessera` Graph IR dialect
// loads.  MLIR 23 validates dialect namespaces at construction time and rejects
// the legacy dotted namespace `tessera.attn`; eager loading therefore made any
// ordinary `tessera.*` input abort before the Apple lowering pipeline ran.
//
// Keep the dialect registered for tools that explicitly use it, but isolate its
// legacy namespace until the attention IR spelling can be migrated as its own
// compatibility change.
void registerAttnDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraAttnDialect>();
}

} // namespace attn
} // namespace tessera

// ── Register dialect ops from ODS-generated definitions ──────────────────
#include "AttnDialect.cpp.inc"
#define GET_OP_CLASSES
#include "AttnOps.cpp.inc"
