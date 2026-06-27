//===- GenerateROCMWhereKernel.cpp - elementwise ternary select ----------===//
//
// Expands a `tessera_rocm.where` directive into a flat 3-operand elementwise gpu
// kernel computing the numpy `where`/select over N elements (one thread per
// element, strided grid):
//
//   where(cond, a, b)[i] = cond[i] ? a[i] : b[i]
//
// `cond` is an i8 boolean (normalized via != 0, matching numpy where any nonzero
// is true); `a`/`b`/the output share a float storage dtype (f16/bf16/f32). N is
// a runtime index arg. Validated vs np.where on gfx1151.
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

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitWhereBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  Type i8 = b.getIntegerType(8);
  auto slt = arith::CmpIPredicate::slt;
  auto ne = arith::CmpIPredicate::ne;

  b.setInsertionPointToStart(&f.getBody().front());
  Value C = f.getArgument(0), A = f.getArgument(1), B = f.getArgument(2),
        O = f.getArgument(3), N = f.getArgument(4);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value z8 = b.create<arith::ConstantOp>(loc, i8, b.getI8IntegerAttr(0));
  Value cv = b.create<memref::LoadOp>(loc, C, ValueRange{gid});
  Value pred = b.create<arith::CmpIOp>(loc, ne, cv, z8);  // cond != 0
  Value av = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value bv = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value sel = b.create<arith::SelectOp>(loc, pred, av, bv);
  b.create<memref::StoreOp>(loc, sel, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMWhereKernelPass
    : PassWrapper<GenerateROCMWhereKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMWhereKernelPass)

  StringRef getArgument() const final { return "generate-rocm-where-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.where directive into a flat 3-operand "
           "elementwise ternary-select gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.where")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.where missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16") storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16") storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-where-kernel: dtype must be f32, f16, or "
                        "bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      Type i8 = b.getIntegerType(8);
      auto condTy = MemRefType::get({ShapedType::kDynamic}, i8);
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (cond : memref<?xi8>, A, B, O : memref<?xstore>, N : index)
      auto fnTy = b.getFunctionType({condTy, memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitWhereBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMWhereKernelPass() {
  return std::make_unique<GenerateROCMWhereKernelPass>();
}
