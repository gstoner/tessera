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
// Rank-N (1..6 today; cap is a sanity guard, not a fundamental limit).
// The loop nest is built by ``buildLoopNest`` below, a recursive helper
// that emits one ``scf.for`` per axis and threads the accumulator iter
// arg all the way through.  The same BC-fixup machinery applies to
// vertical-level (rank-3) and time-aware (rank-4) fields.
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
           "(rank-N; periodic/reflect/dirichlet(v)/neumann(v))";
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
    // Sanity cap: rank > 6 indicates a malformed BC list (or a stencil
    // wider than anything atmospheric science has thrown at us).  Lift
    // the cap if a real workload needs it.
    if (rank < 1 || rank > 6) {
      op->emitWarning() << "StencilLoopMaterializePass: unsupported rank "
                        << rank << " (supports rank 1..6)";
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
    if (!fieldTy || fieldTy.getRank() != rank) {
      op->emitWarning() << "stencil field must be a ranked rank-" << rank
                        << " tensor (matching BC list); got "
                        << (fieldTy ? fieldTy.getRank() : -1);
      return;
    }
    auto floatTy = llvm::dyn_cast<FloatType>(fieldTy.getElementType());
    if (!floatTy) {
      op->emitWarning() << "stencil materialize requires float-typed fields";
      return;
    }

    // ---- Per-axis runtime dim sizes (dynamic-friendly) ----
    SmallVector<Value> Ns;
    Ns.reserve(rank);
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);
    for (int64_t a = 0; a < rank; ++a) {
      Value Na = b.create<tensor::DimOp>(loc, field, idxConst(b, loc, a));
      Ns.push_back(Na);
      sizes.push_back(OpFoldResult(Na));
    }
    Value zero = idxConst(b, loc, 0);
    Value oneI = idxConst(b, loc, 1);

    // ---- Init result tensor of the same type as field ----
    Value initT = b.create<tensor::EmptyOp>(loc, sizes, floatTy);

    // ---- Per-axis BC constants ----
    SmallVector<StringRef> modes;
    SmallVector<Value> bcConsts;
    modes.reserve(rank);
    bcConsts.reserve(rank);
    for (int64_t a = 0; a < rank; ++a) {
      modes.push_back(llvm::cast<StringAttr>(bcModes[a]).getValue());
      bcConsts.push_back(
          f32Const(b, loc, floatTy,
                   llvm::cast<FloatAttr>(bcValues[a]).getValueAsDouble()));
    }

    // ---- Recursively build the rank-N loop nest ----
    //
    // ``buildNest`` walks axes outer-to-inner.  At each level it creates
    // one ``scf.for`` that carries the running accumulator tensor as an
    // iter arg.  When ``axis == rank`` we are inside the innermost body
    // and emit the tap accumulation + tensor.insert.
    SmallVector<Value> ivs;
    Value rootResult = buildNest(b, loc, /*axis=*/0, rank, ivs, initT,
                                  field, floatTy, Ns, modes, bcConsts,
                                  tapsAttr, zero, oneI);

    // ---- Replace stencil.apply with the loop result ----
    op->getResult(0).replaceAllUsesWith(rootResult);
    // Carry over the structured attributes from stencil.apply onto the
    // outermost scf.for (which produced rootResult) so downstream passes
    // can still inspect them.
    if (auto outerFor = rootResult.getDefiningOp<scf::ForOp>()) {
      outerFor->setAttr("stencil.materialized", b.getBoolAttr(true));
      outerFor->setAttr("stencil.rank", b.getI64IntegerAttr(rank));
      if (auto a = op->getAttrOfType<ArrayAttr>("stencil.halo_width"))
        outerFor->setAttr("stencil.halo_width", a);
      if (auto a = op->getAttrOfType<IntegerAttr>("stencil.tap_count"))
        outerFor->setAttr("stencil.tap_count", a);
    }
    op->erase();
  }

  // -------------------------------------------------------------------
  // buildNest — recursive loop-nest builder
  //
  // Outer-to-inner.  At each level it creates one ``scf.for`` carrying
  // ``acc`` as iter_arg, pushes the induction variable onto ``ivs``, and
  // either recurses or, at the leaf, emits the tap accumulation +
  // tensor.insert + scf.yield chain.
  // -------------------------------------------------------------------
  Value buildNest(OpBuilder &b, Location loc,
                  int64_t axis, int64_t rank,
                  SmallVectorImpl<Value> &ivs, Value acc,
                  Value field, FloatType floatTy,
                  ArrayRef<Value> Ns,
                  ArrayRef<StringRef> modes,
                  ArrayRef<Value> bcConsts,
                  ArrayAttr tapsAttr,
                  Value zero, Value oneI) {
    if (axis == rank) {
      // Leaf: compute one output element and tensor.insert it.
      Value sumVal = f32Const(b, loc, floatTy, 0.0);
      Value zeroF  = sumVal;
      for (Attribute tapAttr : tapsAttr) {
        auto denseTap = llvm::dyn_cast<DenseIntElementsAttr>(tapAttr);
        if (!denseTap) continue;
        auto vals = denseTap.getValues<int64_t>();
        if (static_cast<int64_t>(vals.size()) != rank) continue;

        // Compute per-axis fixed indices.
        SmallVector<AxisFixup> fixups;
        fixups.reserve(rank);
        for (int64_t a = 0; a < rank; ++a) {
          Value cd  = idxConst(b, loc, vals[a]);
          Value raw = b.create<arith::AddIOp>(loc, ivs[a], cd);
          fixups.push_back(applyBCFixup(b, loc, raw, Ns[a], modes[a]));
        }

        // Extract field[fixed_indices].
        SmallVector<Value> idxs;
        idxs.reserve(rank);
        for (auto &f : fixups) idxs.push_back(f.fixedIdx);
        Value tapVal = b.create<tensor::ExtractOp>(loc, field,
                                                    ValueRange(idxs));

        // Neumann fixup: sum of per-axis BC offsets gated on OOB flag.
        Value nSum = zeroF;
        Value neumannAny = b.create<arith::ConstantIntOp>(
            loc, /*value=*/0, /*width=*/1);
        for (int64_t a = 0; a < rank; ++a) {
          Value nv = b.create<arith::SelectOp>(
              loc, fixups[a].neumannOOB, bcConsts[a], zeroF);
          nSum = b.create<arith::AddFOp>(loc, nSum, nv);
          neumannAny = b.create<arith::OrIOp>(loc, neumannAny,
                                                fixups[a].neumannOOB);
        }
        Value neumannVal = b.create<arith::AddFOp>(loc, tapVal, nSum);
        tapVal = b.create<arith::SelectOp>(loc, neumannAny, neumannVal,
                                             tapVal);

        // Dirichlet fixup: replace with axis-a BC constant if any
        // dirichlet axis OOB.  Walk axes innermost-to-outermost so the
        // outermost-OOB axis wins (matches the rank-2 semantics where
        // axis 0 wins over axis 1).
        for (int64_t a = rank - 1; a >= 0; --a) {
          tapVal = b.create<arith::SelectOp>(
              loc, fixups[a].dirichletOOB, bcConsts[a], tapVal);
        }

        sumVal = b.create<arith::AddFOp>(loc, sumVal, tapVal);
      }

      Value updated = b.create<tensor::InsertOp>(loc, sumVal, acc,
                                                   ValueRange(ivs));
      b.create<scf::YieldOp>(loc, ValueRange{updated});
      return updated;  // unused at leaf — caller relies on yield result.
    }

    // Inner-or-outer: emit one scf.for over axis `axis`.
    auto forOp = b.create<scf::ForOp>(loc, zero, Ns[axis], oneI,
                                        ValueRange{acc});
    OpBuilder body(forOp.getBody(), forOp.getBody()->begin());
    ivs.push_back(forOp.getInductionVar());
    Value innerAcc = forOp.getRegionIterArg(0);

    if (axis == rank - 1) {
      // Recursion bottoms out in the leaf, which emits its own scf.yield
      // inside this body.
      buildNest(body, loc, axis + 1, rank, ivs, innerAcc, field, floatTy,
                Ns, modes, bcConsts, tapsAttr, zero, oneI);
    } else {
      // The recursive call creates a nested scf.for whose single result
      // we must yield from this body.
      Value nested = buildNest(body, loc, axis + 1, rank, ivs, innerAcc,
                                 field, floatTy, Ns, modes, bcConsts,
                                 tapsAttr, zero, oneI);
      body.create<scf::YieldOp>(loc, ValueRange{nested});
    }
    ivs.pop_back();
    return forOp.getResult(0);
  }
};

} // anonymous namespace

void registerStencilLoopMaterializePass() {
  PassRegistration<StencilLoopMaterializePass>();
}

} // namespace neighbors
} // namespace tessera
