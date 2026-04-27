
//===- NVFP4Vectorize.cpp - Legalize matmul/attn tiles to NVFP4 MMA forms -===//
//
// Scans for matmul-like ops (linalg.matmul, linalg.batch_matmul, and the
// Tessera tessera.attn.* ops) whose element types are bf16 or f16.  For each
// such op it:
//
//   1. Inserts a `tessera.cast bf16/f16 → nvfp4` op on the A and B operands.
//   2. Keeps the accumulator in f32 (CPX MMA: fp4 × fp4 + fp32 → fp32).
//   3. Annotates the op with `{tessera.nvfp4_accel = true}` so downstream
//      emission knows to use CPX tensor-unit instructions.
//
// This is a best-effort legalization pass; ops whose shapes are not
// 16-multiple-aligned are left unchanged with a warning remark.
//
//===-----------------------------------------------------------------------===//

#include "tessera/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvfp4-vectorize"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if \p type is bf16 or f16 — eligible for NVFP4 MMA lowering.
static bool isLowPrecisionFloat(Type type) {
  return type.isBF16() || type.isF16();
}

/// Returns the inner element type of a memref or tensor type, or the type
/// itself if it is already a scalar.
static Type getElementType(Type type) {
  if (auto mr = type.dyn_cast<MemRefType>())   return mr.getElementType();
  if (auto rt = type.dyn_cast<RankedTensorType>()) return rt.getElementType();
  return type;
}

/// Returns true if all spatial dimensions of \p type are divisible by 16
/// (required for CPX MMA tile alignment).
static bool isMMAAligned(Type type) {
  auto shape = [&]() -> ArrayRef<int64_t> {
    if (auto mr = type.dyn_cast<MemRefType>())   return mr.getShape();
    if (auto rt = type.dyn_cast<RankedTensorType>()) return rt.getShape();
    return {};
  }();
  for (int64_t d : shape)
    if (d != ShapedType::kDynamic && d % 16 != 0) return false;
  return !shape.empty();
}

/// Insert a `tessera.cast` op that reinterprets \p val's element type as nvfp4.
/// Since the CPX dialect's NVFP4Type is not yet fully lowered to LLVM at this
/// stage, we represent the cast as a generic op with an attribute tag so that
/// the final emitter can handle it.
static Value insertCastToNVFP4(OpBuilder &b, Location loc, Value val) {
  // Emit a generic op named "tessera.cast" carrying a "to" attribute.
  OperationState state(loc, "tessera.cast");
  state.addOperands(val);
  // Result type: same shape with !tessera.target.cpx.nvfp4 element
  // For now we use i4 as a placeholder until the CPX dialect is fully lowered
  // (the verifier for this cast op lives in CPXTargetIROps.cpp).
  state.addTypes(val.getType()); // preserved shape; emitter handles type swap
  state.addAttribute("to", b.getStringAttr("nvfp4"));
  return b.create(state)->getResult(0);
}

//===----------------------------------------------------------------------===//
// NVFP4VectorizePass
//===----------------------------------------------------------------------===//

struct NVFP4VectorizePass
    : public PassWrapper<NVFP4VectorizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVFP4VectorizePass)

  StringRef getArgument() const override { return "tessera-vectorize-nvfp4"; }
  StringRef getDescription() const override {
    return "Legalize matmul/attention tiles to NVFP4 MMA forms with "
           "F16/BF16 inputs and F32 accumulators on CPX";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    SmallVector<Operation *> candidates;

    // Collect candidate ops: linalg.matmul, linalg.batch_matmul, and
    // generic ops tagged with tessera.attn or tessera.matmul
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "linalg.matmul" ||
          name == "linalg.batch_matmul" ||
          name.contains("tessera.attn") ||
          name.contains("tessera.matmul"))
        candidates.push_back(op);
    });

    unsigned legalized = 0, skipped = 0;

    for (Operation *op : candidates) {
      // Check operand types: A and B inputs must be bf16 or f16
      if (op->getNumOperands() < 2) continue;
      Value opA = op->getOperand(0);
      Value opB = op->getOperand(1);

      Type elemA = getElementType(opA.getType());
      Type elemB = getElementType(opB.getType());

      if (!isLowPrecisionFloat(elemA) || !isLowPrecisionFloat(elemB)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[nvfp4] skip " << op->getName()
                   << " (non-lowp-float inputs)\n");
        ++skipped;
        continue;
      }

      // Check MMA alignment
      if (!isMMAAligned(opA.getType()) || !isMMAAligned(opB.getType())) {
        op->emitRemark("nvfp4-vectorize: skipping — input shapes not "
                       "16-multiple-aligned; cannot use CPX MMA units");
        ++skipped;
        continue;
      }

      // Insert casts before the op
      b.setInsertionPoint(op);
      Location loc = op->getLoc();
      Value castA = insertCastToNVFP4(b, loc, opA);
      Value castB = insertCastToNVFP4(b, loc, opB);
      op->setOperand(0, castA);
      op->setOperand(1, castB);

      // Tag the op so the final emitter uses CPX MMA tensor unit
      op->setAttr("tessera.nvfp4_accel", UnitAttr::get(op->getContext()));

      ++legalized;
      LLVM_DEBUG(llvm::dbgs()
                 << "[nvfp4] legalized " << op->getName() << " at "
                 << op->getLoc() << "\n");
    }

    if (legalized + skipped > 0)
      module.emitRemark("nvfp4-vectorize: legalized ")
          << legalized << " op(s), skipped " << skipped << " op(s)";
  }
};

PassRegistration<NVFP4VectorizePass> nvfp4VectorizePassReg;

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createNVFP4VectorizePass() {
  return std::make_unique<NVFP4VectorizePass>();
}
} // namespace tessera
