//===- GenerateROCMCompareKernel.cpp - elementwise comparison kernel ------===//
//
// Expands a `tessera_rocm.compare` directive into a flat 2-operand elementwise
// gpu kernel applying a pointwise comparison over N elements (one thread per
// element, strided grid) — the standalone S2 comparison family, producing a
// boolean (i8 0/1) result:
//
//   eq = a == b   ne = a != b   lt = a < b   le = a <= b   gt = a > b   ge = a >= b
//
// The operands share f16/bf16/f32, signed i32, or unsigned u32 storage. Float
// comparisons run in f32; integer order is selected explicitly by dtype rather
// than inferred from signless LLVM storage. The result is an `i8` mask (0/1), matching numpy's bool
// output (1 byte/element). NaN semantics follow numpy: every predicate is
// ORDERED (NaN → false) EXCEPT `ne`, which is UNORDERED-or-not-equal (NaN →
// true), so `np.not_equal(nan, x)` and `np.equal(nan, x)` both match. N is a
// runtime index arg. Validated vs numpy on gfx1151.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

enum class Cmp { Eq, Ne, Lt, Le, Gt, Ge };

void emitCompareBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                     Cmp cmp, bool isUnsignedInteger) {
  Type f32 = b.getF32Type();
  Type i8 = b.getIntegerType(8);
  bool isFloat = isa<FloatType>(storeTy);
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  // numpy NaN semantics: ordered everywhere except `ne` (unordered-or-ne).
  arith::CmpFPredicate pred;
  switch (cmp) {
  case Cmp::Eq: pred = arith::CmpFPredicate::OEQ; break;
  case Cmp::Ne: pred = arith::CmpFPredicate::UNE; break;
  case Cmp::Lt: pred = arith::CmpFPredicate::OLT; break;
  case Cmp::Le: pred = arith::CmpFPredicate::OLE; break;
  case Cmp::Gt: pred = arith::CmpFPredicate::OGT; break;
  case Cmp::Ge: pred = arith::CmpFPredicate::OGE; break;
  }

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value ra = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value rb = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value bit;
  if (isFloat) {
    Value a = isF32 ? ra : b.create<arith::ExtFOp>(loc, f32, ra);
    Value bb = isF32 ? rb : b.create<arith::ExtFOp>(loc, f32, rb);
    bit = b.create<arith::CmpFOp>(loc, pred, a, bb);
  } else {
    arith::CmpIPredicate intPred;
    switch (cmp) {
    case Cmp::Eq: intPred = arith::CmpIPredicate::eq; break;
    case Cmp::Ne: intPred = arith::CmpIPredicate::ne; break;
    case Cmp::Lt: intPred = isUnsignedInteger ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt; break;
    case Cmp::Le: intPred = isUnsignedInteger ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle; break;
    case Cmp::Gt: intPred = isUnsignedInteger ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt; break;
    case Cmp::Ge: intPred = isUnsignedInteger ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge; break;
    }
    bit = b.create<arith::CmpIOp>(loc, intPred, ra, rb);
  }
  Value sv = b.create<arith::ExtUIOp>(loc, i8, bit);       // 0/1 byte
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMCompareKernelPass
    : PassWrapper<GenerateROCMCompareKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMCompareKernelPass)

  StringRef getArgument() const final { return "generate-rocm-compare-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.compare directive into a flat 2-operand "
           "elementwise comparison gpu kernel with i8 boolean output "
           "(compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.compare")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.compare missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "eq";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Cmp cmp = llvm::StringSwitch<Cmp>(kindStr)
                    .Case("eq", Cmp::Eq)
                    .Case("ne", Cmp::Ne)
                    .Case("lt", Cmp::Lt)
                    .Case("le", Cmp::Le)
                    .Case("gt", Cmp::Gt)
                    .Case("ge", Cmp::Ge)
                    .Default(Cmp::Eq);
      static const llvm::StringSet<> kValid = {"eq", "ne", "lt",
                                               "le", "gt", "ge"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-compare-kernel: unknown kind '")
            << kindStr << "' (eq/ne/lt/le/gt/ge)";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      bool isUnsignedInteger = false;
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt == "i32" || dt == "int32")
          storeTy = b.getI32Type();
        else if (dt == "u32" || dt == "uint32") {
          storeTy = b.getI32Type();
          isUnsignedInteger = true;
        }
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-compare-kernel: dtype must be f32, f16, "
                        "bf16, i32, or u32 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      Type i8 = b.getIntegerType(8);
      auto inTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto outTy = MemRefType::get({ShapedType::kDynamic}, i8);
      // (A, B : memref<?xstore>, O : memref<?xi8>, N : index)
      auto fnTy = b.getFunctionType({inTy, inTy, outTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitCompareBody(body, loc, gpuFunc, storeTy, cmp, isUnsignedInteger);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMCompareKernelPass() {
  return std::make_unique<GenerateROCMCompareKernelPass>();
}
