//===- GenerateROCMBitwiseKernel.cpp - elementwise bitwise kernel ---------===//
//
// Expands a `tessera_rocm.bitwise` directive into a flat elementwise gpu kernel
// applying a pointwise bitwise operation over N elements (one thread per
// element, strided grid) — the standalone S2 bitwise family, operating on i32
// integers:
//
//   and = a & b    or = a | b    xor = a ^ b    (binary)
//   not = ~a                                     (unary)
//
// Unlike the logical lane, operands are NOT normalized — the op acts on the full
// integer bit pattern. `not` is `a ^ -1` (all ones). `not` takes one operand;
// the rest take two. N is a runtime index arg. Validated vs numpy on gfx1151.
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

enum class Bw { And, Or, Xor, Not };

void emitBitwiseBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Bw bw) {
  Type i32 = b.getIntegerType(32);
  bool isNot = (bw == Bw::Not);
  auto slt = arith::CmpIPredicate::slt;

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

  Value a = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value y;
  if (isNot) {
    Value negOne = b.create<arith::ConstantOp>(loc, i32,
                                               b.getI32IntegerAttr(-1));
    y = b.create<arith::XOrIOp>(loc, a, negOne);   // ~a
  } else {
    Value B = f.getArgument(1);
    Value bb = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
    switch (bw) {
    case Bw::And: y = b.create<arith::AndIOp>(loc, a, bb); break;
    case Bw::Or:  y = b.create<arith::OrIOp>(loc, a, bb); break;
    case Bw::Xor: y = b.create<arith::XOrIOp>(loc, a, bb); break;
    default:      y = a; break;
    }
  }
  b.create<memref::StoreOp>(loc, y, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBitwiseKernelPass
    : PassWrapper<GenerateROCMBitwiseKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMBitwiseKernelPass)

  StringRef getArgument() const final { return "generate-rocm-bitwise-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.bitwise directive into a flat elementwise "
           "bitwise (and/or/xor/not) gpu kernel over i32 integers "
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
      if (op->getName().getStringRef() == "tessera_rocm.bitwise")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.bitwise missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "and";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Bw bw = llvm::StringSwitch<Bw>(kindStr)
                  .Case("and", Bw::And)
                  .Case("or", Bw::Or)
                  .Case("xor", Bw::Xor)
                  .Case("not", Bw::Not)
                  .Default(Bw::And);
      static const llvm::StringSet<> kValid = {"and", "or", "xor", "not"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-bitwise-kernel: unknown kind '")
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
      Type i32 = b.getIntegerType(32);
      auto memTy = MemRefType::get({ShapedType::kDynamic}, i32);
      // not: (A, O : memref<?xi32>, N : index); else (A, B, O, N)
      FunctionType fnTy =
          (kindStr == "not")
              ? b.getFunctionType({memTy, memTy, idxTy}, {})
              : b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitBitwiseBody(body, loc, gpuFunc, bw);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBitwiseKernelPass() {
  return std::make_unique<GenerateROCMBitwiseKernelPass>();
}
