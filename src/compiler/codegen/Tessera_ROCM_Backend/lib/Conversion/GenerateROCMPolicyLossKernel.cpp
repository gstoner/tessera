//===- GenerateROCMPolicyLossKernel.cpp - PPO / CISPO surrogate -----------===//
//
// Expands `tessera_rocm.policy_loss` into a flat 3-operand elementwise per-
// element RL policy-loss surrogate over (logp_new, logp_old, advantages):
//
//   ratio = exp(ln − lo)
//   kind 0 = ppo    c = clip(ratio, 1−ε, 1+ε); loss = −min(ratio·adv, c·adv)
//   kind 1 = cispo  w = min(ratio, ε_high);    loss = −(w·adv·ln)
//
// `kind`/`clip` are compile-time attrs. The runtime reduces (none/mean/sum) and
// does grpo's advantage normalization on the norm lane. f32 compute; exp lowers
// through math→ROCDL. CPU analog: avx512_policy_loss_f32.
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

void emitPolicyLossBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                        Type storeTy, int64_t kind, double clip) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;
  b.setInsertionPointToStart(&f.getBody().front());
  Value LN = f.getArgument(0), LO = f.getArgument(1), ADV = f.getArgument(2),
        O = f.getArgument(3), N = f.getArgument(4);
  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  auto ld = [&](Value M) {
    Value r = b.create<memref::LoadOp>(loc, M, ValueRange{gid});
    return isF32 ? r : b.create<arith::ExtFOp>(loc, f32, r).getResult();
  };
  Value ln = ld(LN), lo = ld(LO), adv = ld(ADV);
  auto cst = [&](double v) {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v))
        .getResult();
  };
  Value zero = cst(0.0);
  Value r = b.create<math::ExpOp>(loc, b.create<arith::SubFOp>(loc, ln, lo));
  Value y;
  if (kind == 0) {  // ppo
    Value c = b.create<arith::MinimumFOp>(
        loc, b.create<arith::MaximumFOp>(loc, r, cst(1.0 - clip)),
        cst(1.0 + clip));
    Value s = b.create<arith::MinimumFOp>(loc, b.create<arith::MulFOp>(loc, r, adv),
                                          b.create<arith::MulFOp>(loc, c, adv));
    y = b.create<arith::SubFOp>(loc, zero, s);
  } else {          // cispo
    Value w = b.create<arith::MinimumFOp>(loc, r, cst(clip));
    y = b.create<arith::SubFOp>(
        loc, zero, b.create<arith::MulFOp>(loc, b.create<arith::MulFOp>(loc, w, adv), ln));
  }
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});
  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMPolicyLossKernelPass
    : PassWrapper<GenerateROCMPolicyLossKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMPolicyLossKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-policy-loss-kernel";
  }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.policy_loss directive into a flat 3-operand "
           "elementwise PPO / CISPO policy-loss surrogate gpu kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.policy_loss")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.policy_loss missing name");
        return signalPassFailure();
      }
      int64_t kind = 0;
      if (auto k = op->getAttrOfType<IntegerAttr>("kind")) kind = k.getInt();
      if (kind < 0 || kind > 1) {
        op->emitError("generate-rocm-policy-loss-kernel: kind must be 0 or 1");
        return signalPassFailure();
      }
      double clip = 0.2;
      if (auto a = op->getAttrOfType<FloatAttr>("clip"))
        clip = a.getValueAsDouble();
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
          op->emitError("generate-rocm-policy-loss-kernel: dtype must be f32, "
                        "f16, or bf16 (got '") << dt << "')";
          return signalPassFailure();
        }
      }
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitPolicyLossBody(body, loc, gpuFunc, storeTy, kind, clip);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMPolicyLossKernelPass() {
  return std::make_unique<GenerateROCMPolicyLossKernelPass>();
}
