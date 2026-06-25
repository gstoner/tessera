//===- GenerateROCMRopeKernel.cpp - rotary position embedding kernel ------===//
//
// Expands a `tessera_rocm.rope` directive into an elementwise-per-pair gpu
// kernel applying rotary position embedding over a rank-2 [M, D] input (D even).
// One workgroup per row (blockIdx.x = m); the lanes stride the D/2 pairs. For
// pair p (e = X[m,2p], o = X[m,2p+1], angle a = Theta[m,2p]):
//
//   O[m,2p]   = e·cos(a) − o·sin(a)
//   O[m,2p+1] = e·sin(a) + o·cos(a)
//
// Matches the reference `_runtime_rope` (interleaved pairs; the even-indexed
// theta entry is the pair angle). cos/sin run in f32 and lower through
// convert-math-to-rocdl. M/D are runtime index args. Validated vs numpy on
// gfx1151.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

void emitRopeBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value X = f.getArgument(0), Theta = f.getArgument(1), O = f.getArgument(2);
  Value M = f.getArgument(3), D = f.getArgument(4);

  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);

  Value m = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value rowInb = b.create<arith::CmpIOp>(loc, slt, m, M);
  auto rowIf = b.create<scf::IfOp>(loc, rowInb, /*withElse=*/false);
  b.setInsertionPointToStart(rowIf.thenBlock());
  Value base = b.create<arith::MulIOp>(loc, m, D);
  Value halfD = b.create<arith::DivUIOp>(loc, D, c2);

  auto loadF32 = [&](Value idx) -> Value {
    Value v = b.create<memref::LoadOp>(loc, X, ValueRange{idx});
    return isF32 ? v : b.create<arith::ExtFOp>(loc, f32, v);
  };
  auto storeFromF32 = [&](Value val, Value idx) {
    Value sv = isF32 ? val : b.create<arith::TruncFOp>(loc, storeTy, val);
    b.create<memref::StoreOp>(loc, sv, O, ValueRange{idx});
  };

  // for p = tid; p < D/2; p += BD
  auto lp = b.create<scf::ForOp>(loc, tid, halfD, cBD);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(lp.getBody());
    Value p = lp.getInductionVar();
    Value ie = b.create<arith::AddIOp>(loc, base,
                                       b.create<arith::MulIOp>(loc, p, c2));
    Value io = b.create<arith::AddIOp>(loc, ie, c1);
    Value e = loadF32(ie);
    Value o = loadF32(io);
    // Angle = Theta at the even position (matches _runtime_rope's theta[0::2]).
    Value araw = b.create<memref::LoadOp>(loc, Theta, ValueRange{ie});
    Value a = isF32 ? araw : b.create<arith::ExtFOp>(loc, f32, araw);
    Value c = b.create<math::CosOp>(loc, a);
    Value s = b.create<math::SinOp>(loc, a);
    // out_even = e*c - o*s ; out_odd = e*s + o*c
    Value oe = b.create<arith::SubFOp>(loc, b.create<arith::MulFOp>(loc, e, c),
                                       b.create<arith::MulFOp>(loc, o, s));
    Value oo = b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, e, s),
                                       b.create<arith::MulFOp>(loc, o, c));
    storeFromF32(oe, ie);
    storeFromF32(oo, io);
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMRopeKernelPass
    : PassWrapper<GenerateROCMRopeKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMRopeKernelPass)

  StringRef getArgument() const final { return "generate-rocm-rope-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.rope directive into an elementwise-per-pair "
           "rotary-position-embedding gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.rope")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.rope missing name");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-rope-kernel: dtype must be f32, f16, or "
                        "bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (X, Theta, O : memref<?xstore>, M, D : index)
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitRopeBody(body, loc, gpuFunc, storeTy);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMRopeKernelPass() {
  return std::make_unique<GenerateROCMRopeKernelPass>();
}
