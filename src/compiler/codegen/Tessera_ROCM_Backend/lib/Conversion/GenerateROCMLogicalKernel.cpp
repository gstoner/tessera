//===- GenerateROCMLogicalKernel.cpp - elementwise logical kernel ---------===//
//
// Expands a `tessera_rocm.logical` directive into a flat elementwise gpu kernel
// applying a pointwise logical operation over N elements (one thread per
// element, strided grid) — the standalone S2 logical family, operating on i8
// booleans:
//
//   and = a && b    or = a || b    xor = a ^ b    (binary)
//   not = !a                                       (unary)
//
// Inputs are normalized to a boolean via `a != 0` (matching numpy, where any
// nonzero value is true), so the kernel is correct for arbitrary i8 inputs, not
// just {0,1}. The result is an i8 mask (0/1). `not` takes one operand; the rest
// take two. N is a runtime index arg. Validated vs numpy on gfx1151.
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

enum class Lg { And, Or, Xor, Not };

void emitLogicalBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Lg lg) {
  Type i8 = b.getIntegerType(8);
  Type i1 = b.getIntegerType(1);
  bool isNot = (lg == Lg::Not);
  auto slt = arith::CmpIPredicate::slt;
  auto ne = arith::CmpIPredicate::ne;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0);
  Value O, N;
  if (isNot) {
    O = f.getArgument(1);
    N = f.getArgument(2);
  } else {
    O = f.getArgument(2);
    N = f.getArgument(3);
  }

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value z8 = b.create<arith::ConstantOp>(loc, i8, b.getI8IntegerAttr(0));
  Value ra = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value ba = b.create<arith::CmpIOp>(loc, ne, ra, z8);   // a != 0  -> i1
  Value bit;
  if (isNot) {
    Value one1 = b.create<arith::ConstantOp>(loc, i1, b.getBoolAttr(true));
    bit = b.create<arith::XOrIOp>(loc, ba, one1);          // !a
  } else {
    Value B = f.getArgument(1);
    Value rb = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
    Value bb = b.create<arith::CmpIOp>(loc, ne, rb, z8);   // b != 0 -> i1
    switch (lg) {
    case Lg::And: bit = b.create<arith::AndIOp>(loc, ba, bb); break;
    case Lg::Or:  bit = b.create<arith::OrIOp>(loc, ba, bb); break;
    case Lg::Xor: bit = b.create<arith::XOrIOp>(loc, ba, bb); break;
    default:      bit = ba; break;
    }
  }
  Value sv = b.create<arith::ExtUIOp>(loc, i8, bit);       // 0/1 byte
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMLogicalKernelPass
    : PassWrapper<GenerateROCMLogicalKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMLogicalKernelPass)

  StringRef getArgument() const final { return "generate-rocm-logical-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.logical directive into a flat elementwise "
           "logical (and/or/xor/not) gpu kernel over i8 booleans "
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
      if (op->getName().getStringRef() == "tessera_rocm.logical")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.logical missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "and";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Lg lg = llvm::StringSwitch<Lg>(kindStr)
                  .Case("and", Lg::And)
                  .Case("or", Lg::Or)
                  .Case("xor", Lg::Xor)
                  .Case("not", Lg::Not)
                  .Default(Lg::And);
      static const llvm::StringSet<> kValid = {"and", "or", "xor", "not"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-logical-kernel: unknown kind '")
            << kindStr << "' (and/or/xor/not)";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      Type i8 = b.getIntegerType(8);
      auto memTy = MemRefType::get({ShapedType::kDynamic}, i8);
      // not: (A, O : memref<?xi8>, N : index); else (A, B, O, N)
      FunctionType fnTy =
          (kindStr == "not")
              ? b.getFunctionType({memTy, memTy, idxTy}, {})
              : b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitLogicalBody(body, loc, gpuFunc, lg);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMLogicalKernelPass() {
  return std::make_unique<GenerateROCMLogicalKernelPass>();
}
