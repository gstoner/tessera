//===- StencilLoopMaterializePass.cpp — Materialize stencil loops ---------===//
//
// Lowers a BC-lowered ``tessera.neighbors.stencil.apply`` op into a real
// ``scf.for``-nested loop body that reads each tap (with per-axis
// boundary-condition fixups) and writes the accumulated value into an
// output tensor.  This is the pass that finally consumes the BC ABI
// emitted by ``BoundaryConditionLowerPass``:
//
//   * periodic    → ((raw % N) + N) % N
//   * reflect     → clamp(raw, 0, N-1)              [first-cut clamp]
//   * dirichlet(v)→ if raw OOB on a dirichlet axis, return constant v
//   * neumann(v)  → if raw OOB on a neumann   axis, return extract+v
//
// Rank-2 only this drop.  Extension to rank-1 / rank-3 is the same loop
// nest with one more induction var; see
// ``docs/architecture/stencil_materialize_and_window_lowering.md``.
//
// Sentinel: ``stencil.materialized = true`` so the pass is idempotent.
//
// Pass argument: ``-tessera-stencil-loop-materialize``.
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace tessera {
namespace neighbors {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Value idxConst(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

static Value f32Const(OpBuilder &b, Location loc, FloatType ty, double v) {
  return b.create<arith::ConstantOp>(loc, ty, b.getFloatAttr(ty, v));
}

static Value i1Const(OpBuilder &b, Location loc, bool v) {
  return b.create<arith::ConstantIntOp>(loc, /*value=*/v ? 1 : 0, /*width=*/1);
}

/// Per-axis fixup result.  `fixedIdx` is always a valid in-range index
/// suitable for ``tensor.extract``.  The two boolean flags fire only for
/// the matching BC mode so the caller can drive the value-side rule
/// (dirichlet replace / neumann add).
struct AxisFixup {
  Value fixedIdx;        // index in [0, N)
  Value dirichletOOB;    // i1, true iff axis mode == "dirichlet" AND raw OOB
  Value neumannOOB;      // i1, true iff axis mode == "neumann"   AND raw OOB
};

static AxisFixup applyBCFixup(OpBuilder &b, Location loc,
                              Value rawIdx, Value N, StringRef mode) {
  AxisFixup out;
  Value zero  = idxConst(b, loc, 0);
  Value oneI  = idxConst(b, loc, 1);
  Value cFalse = i1Const(b, loc, false);

  // inBounds = 0 <= rawIdx && rawIdx < N
  Value geZero  = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                           rawIdx, zero);
  Value ltN     = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           rawIdx, N);
  Value inBnds  = b.create<arith::AndIOp>(loc, geZero, ltN);
  // OOB = !inBnds via xor with constant 1.
  Value oneI1   = i1Const(b, loc, true);
  Value oob     = b.create<arith::XOrIOp>(loc, inBnds, oneI1);

  if (mode == "periodic") {
    // signed remainder may be negative for negative dividends; correct.
    Value rem        = b.create<arith::RemSIOp>(loc, rawIdx, N);
    Value isNeg      = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                rem, zero);
    Value remPlusN   = b.create<arith::AddIOp>(loc, rem, N);
    out.fixedIdx     = b.create<arith::SelectOp>(loc, isNeg, remPlusN, rem);
    out.dirichletOOB = cFalse;
    out.neumannOOB   = cFalse;
    return out;
  }

  // reflect / dirichlet / neumann use clamp to [0, N-1] for the index;
  // the value-side rule differentiates them via the OOB flag.
  Value Nm1     = b.create<arith::SubIOp>(loc, N, oneI);
  Value cl0     = b.create<arith::MaxSIOp>(loc, rawIdx, zero);
  out.fixedIdx  = b.create<arith::MinSIOp>(loc, cl0, Nm1);

  if (mode == "dirichlet") {
    out.dirichletOOB = oob;
    out.neumannOOB   = cFalse;
  } else if (mode == "neumann") {
    out.dirichletOOB = cFalse;
    out.neumannOOB   = oob;
  } else {
    // reflect (or anything we don't recognise) — clamp only.
    out.dirichletOOB = cFalse;
    out.neumannOOB   = cFalse;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct StencilLoopMaterializePass
    : public PassWrapper<StencilLoopMaterializePass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StencilLoopMaterializePass)

  StringRef getArgument() const final {
    return "tessera-stencil-loop-materialize";
  }
  StringRef getDescription() const final {
    return "Materialize stencil.apply into scf.for loops with BC-aware reads "
           "(rank-2 first cut; periodic/reflect/dirichlet(v)/neumann(v))";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect first; we'll erase as we go, so don't mutate during walk.
    SmallVector<Operation *> applyOps;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.neighbors.stencil.apply"
          && !op->hasAttr("stencil.materialized"))
        applyOps.push_back(op);
    });

    for (Operation *op : applyOps)
      materializeOne(op);
  }

  void materializeOne(Operation *op) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    // Prerequisites: BC ABI must be lowered.
    auto bcModes  = op->getAttrOfType<ArrayAttr>("stencil.bc.modes");
    auto bcValues = op->getAttrOfType<ArrayAttr>("stencil.bc.values");
    if (!bcModes || !bcValues) {
      op->emitWarning() << "stencil.apply not BC-lowered; run "
          "-tessera-boundary-condition-lower first";
      return;
    }
    int64_t rank = static_cast<int64_t>(bcModes.size());
    if (rank != 2) {
      // Rank-2 only this drop; document the limitation.
      op->emitWarning() << "StencilLoopMaterializePass supports rank-2 "
          "stencils only this drop; got rank " << rank;
      return;
    }

    Operation *stencilDef = op->getOperand(0).getDefiningOp();
    if (!stencilDef) {
      op->emitWarning() << "stencil.apply's stencil operand has no defining op";
      return;
    }
    auto tapsAttr = stencilDef->getAttrOfType<ArrayAttr>("taps");
    if (!tapsAttr || tapsAttr.empty()) {
      op->emitWarning() << "stencil.define has no taps; skipping";
      return;
    }

    Value field = op->getOperand(1);
    auto fieldTy = llvm::dyn_cast<RankedTensorType>(field.getType());
    if (!fieldTy || fieldTy.getRank() != 2) {
      op->emitWarning() << "stencil field must be a ranked rank-2 tensor";
      return;
    }
    auto floatTy = llvm::dyn_cast<FloatType>(fieldTy.getElementType());
    if (!floatTy) {
      op->emitWarning() << "stencil materialize first-cut requires "
          "float-typed fields";
      return;
    }

    // ---- Runtime dim sizes (dynamic-friendly) ----
    Value N0   = b.create<tensor::DimOp>(loc, field, idxConst(b, loc, 0));
    Value N1   = b.create<tensor::DimOp>(loc, field, idxConst(b, loc, 1));
    Value zero = idxConst(b, loc, 0);
    Value oneI = idxConst(b, loc, 1);

    // ---- Init result tensor of the same type as field ----
    SmallVector<OpFoldResult> sizes = {OpFoldResult(N0), OpFoldResult(N1)};
    Value initT = b.create<tensor::EmptyOp>(loc, sizes, floatTy);

    // ---- Outer scf.for over axis 0 ----
    auto outerFor = b.create<scf::ForOp>(loc, zero, N0, oneI,
                                          ValueRange{initT});
    OpBuilder ob(outerFor.getBody(), outerFor.getBody()->begin());
    Value i        = outerFor.getInductionVar();
    Value accOuter = outerFor.getRegionIterArg(0);

    // ---- Inner scf.for over axis 1 ----
    auto innerFor = ob.create<scf::ForOp>(loc, zero, N1, oneI,
                                            ValueRange{accOuter});
    OpBuilder ib(innerFor.getBody(), innerFor.getBody()->begin());
    Value j        = innerFor.getInductionVar();
    Value accInner = innerFor.getRegionIterArg(0);

    // ---- Per-axis BC constants ----
    StringRef m0 = llvm::cast<StringAttr>(bcModes[0]).getValue();
    StringRef m1 = llvm::cast<StringAttr>(bcModes[1]).getValue();
    double v0 = llvm::cast<FloatAttr>(bcValues[0]).getValueAsDouble();
    double v1 = llvm::cast<FloatAttr>(bcValues[1]).getValueAsDouble();
    Value bcConst0 = f32Const(ib, loc, floatTy, v0);
    Value bcConst1 = f32Const(ib, loc, floatTy, v1);
    Value zeroF    = f32Const(ib, loc, floatTy, 0.0);

    // ---- Accumulator = sum over taps ----
    Value sumVal = zeroF;
    for (Attribute tapAttr : tapsAttr) {
      auto denseTap = llvm::dyn_cast<DenseIntElementsAttr>(tapAttr);
      if (!denseTap) continue;
      auto vals = denseTap.getValues<int64_t>();
      if (vals.size() != 2) continue;
      int64_t d0 = vals[0], d1 = vals[1];

      Value cd0 = idxConst(ib, loc, d0);
      Value cd1 = idxConst(ib, loc, d1);
      Value raw0 = ib.create<arith::AddIOp>(loc, i, cd0);
      Value raw1 = ib.create<arith::AddIOp>(loc, j, cd1);

      AxisFixup ax0 = applyBCFixup(ib, loc, raw0, N0, m0);
      AxisFixup ax1 = applyBCFixup(ib, loc, raw1, N1, m1);

      // Extract field[ax0.fixedIdx, ax1.fixedIdx].
      Value tapVal = ib.create<tensor::ExtractOp>(
          loc, field, ValueRange{ax0.fixedIdx, ax1.fixedIdx});

      // ── Neumann fixup: extract + (axis-wise BC value when OOB) ──
      Value n0v = ib.create<arith::SelectOp>(loc, ax0.neumannOOB,
                                              bcConst0, zeroF);
      Value n1v = ib.create<arith::SelectOp>(loc, ax1.neumannOOB,
                                              bcConst1, zeroF);
      Value nSum       = ib.create<arith::AddFOp>(loc, n0v, n1v);
      Value neumannVal = ib.create<arith::AddFOp>(loc, tapVal, nSum);
      Value neumannAny = ib.create<arith::OrIOp>(loc,
          ax0.neumannOOB, ax1.neumannOOB);
      tapVal = ib.create<arith::SelectOp>(loc, neumannAny, neumannVal, tapVal);

      // ── Dirichlet fixup: replace with BC value if any dirichlet axis OOB ──
      // axis 1's value wins over axis 0 when both fire — arbitrary but
      // deterministic; documented in the architecture doc.
      Value dirAx1 = ib.create<arith::SelectOp>(loc, ax1.dirichletOOB,
                                                  bcConst1, tapVal);
      tapVal       = ib.create<arith::SelectOp>(loc, ax0.dirichletOOB,
                                                  bcConst0, dirAx1);

      sumVal = ib.create<arith::AddFOp>(loc, sumVal, tapVal);
    }

    // ---- Insert sumVal into accumulator at (i, j); yield ----
    Value updated = ib.create<tensor::InsertOp>(
        loc, sumVal, accInner, ValueRange{i, j});
    ib.create<scf::YieldOp>(loc, ValueRange{updated});

    // Outer yields the inner loop's result.
    ob.create<scf::YieldOp>(loc, innerFor.getResults());

    // ---- Replace stencil.apply with the loop result ----
    op->getResult(0).replaceAllUsesWith(outerFor.getResult(0));
    // Carry over the structured attributes from stencil.apply onto the
    // outer scf.for so downstream passes can still inspect them.
    outerFor->setAttr("stencil.materialized", b.getBoolAttr(true));
    if (auto a = op->getAttrOfType<ArrayAttr>("stencil.halo_width"))
      outerFor->setAttr("stencil.halo_width", a);
    if (auto a = op->getAttrOfType<IntegerAttr>("stencil.tap_count"))
      outerFor->setAttr("stencil.tap_count", a);
    op->erase();
  }
};

} // anonymous namespace

void registerStencilLoopMaterializePass() {
  PassRegistration<StencilLoopMaterializePass>();
}

} // namespace neighbors
} // namespace tessera
