#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Stage J — lower a `tessera_rocm.wmma` that carries REAL RDNA WMMA fragment
// vectors to the real `rocdl.wmma.f32.16x16x16.{f16,bf16}` intrinsic op (which
// translates to `llvm.amdgcn.wmma.f32.16x16x16.*`, the same instruction
// rocdl_emit.py emits and llc proves). The gfx11 / RDNA 3.5 ABI: A/B fragments
// are `vector<16x{f16,bf16}>`, accumulator + result `vector<8xf32>`; the bf16
// intrinsic takes the bit-pattern as `vector<16xi16>` (NOT native bf16), so bf16
// fragments are bitcast first — matching rocdl_emit's `_WMMA_F32_INPUT` map.
//
// Returns true if it emitted the real op (and replaced/erased `op`); false when
// the operands are NOT real fragments (abstract / scalar contract-level IR, e.g.
// a `tile.mma` on scalars before fragment materialization) — the caller then
// falls through to the artifact-marker path, which is the honest lowering at
// that abstraction level.
bool lowerRealWMMA(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  Value a = op->getOperand(0), b = op->getOperand(1), c = op->getOperand(2);
  Type resTy = op->getResult(0).getType();
  auto isVec = [](Type t, int64_t n, Type elt) {
    auto v = dyn_cast<VectorType>(t);
    return v && v.getRank() == 1 && v.getNumElements() == n &&
           v.getElementType() == elt;
  };
  Type f32 = rewriter.getF32Type();
  // Accumulator + result must both be the WMMA f32 accumulator vector<8xf32>.
  if (!isVec(c.getType(), 8, f32) || !isVec(resTy, 8, f32))
    return false;

  Location loc = op->getLoc();
  Type f16 = rewriter.getF16Type();
  Type bf16 = rewriter.getBF16Type();

  if (isVec(a.getType(), 16, f16) && isVec(b.getType(), 16, f16)) {
    Operation *real = rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
        loc, resTy, ValueRange{a, b, c});
    rewriter.replaceOp(op, real->getResults());
    return true;
  }
  if (isVec(a.getType(), 16, bf16) && isVec(b.getType(), 16, bf16)) {
    // RDNA bf16 WMMA takes the bf16 bit-pattern as <16 x i16>.
    Type i16Vec = VectorType::get({16}, rewriter.getIntegerType(16));
    Value ai = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, a);
    Value bi = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, b);
    Operation *real = rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
        loc, resTy, ValueRange{ai, bi, c});
    rewriter.replaceOp(op, real->getResults());
    return true;
  }
  return false;
}

LLVM::LLVMFuncOp declareVoidMarker(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(module.getContext()), {}, false);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

void replaceResultUsesWithUndef(Operation *op, PatternRewriter &rewriter) {
  for (Value result : op->getResults()) {
    if (result.use_empty())
      continue;
    auto undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), result.getType());
    result.replaceAllUsesExcept(undef, undef);
  }
}

struct LoweringPass : PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringPass)

  StringRef getArgument() const final { return "lower-tessera-target-to-rocdl"; }

  StringRef getDescription() const final {
    return "Lower Tessera ROCm target ops to ROCDL: real rocdl.wmma for WMMA "
           "ops carrying fragment vectors, artifact markers otherwise";
  }

  void runOnOperation() override {
    getContext().loadDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
    ModuleOp module = getOperation();
    SmallVector<Operation *> waitOps;
    SmallVector<Operation *> rocmOps;

    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera_rocm.wait")
        waitOps.push_back(op);
      else if (name == "tessera_rocm.mfma" ||
               name == "tessera_rocm.wmma" ||
               name == "tessera_rocm.async_copy" ||
               name == "tessera_rocm.buffer_load" ||
               name == "tessera_rocm.ds_read_tr")
        rocmOps.push_back(op);
    });
    waitOps.append(rocmOps.begin(), rocmOps.end());
    rocmOps = std::move(waitOps);

    for (Operation *op : rocmOps) {
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);

      StringRef opName = op->getName().getStringRef();

      // Stage J: a WMMA op carrying real fragment vectors lowers to the real
      // rocdl.wmma intrinsic; only abstract/scalar WMMA falls through to the
      // marker below.
      if (opName == "tessera_rocm.wmma" && lowerRealWMMA(op, rewriter))
        continue;

      StringRef markerName = "llvm.tessera.rocm.unknown";
      if (opName == "tessera_rocm.mfma")
        markerName = "llvm.amdgcn.mfma.contract";
      else if (opName == "tessera_rocm.wmma")
        markerName = "llvm.amdgcn.wmma.contract";
      else if (opName == "tessera_rocm.async_copy")
        markerName = "llvm.amdgcn.raw.buffer.copy.contract";
      else if (opName == "tessera_rocm.buffer_load")
        markerName = "llvm.amdgcn.raw.buffer.load.contract";
      else if (opName == "tessera_rocm.ds_read_tr")
        markerName = "llvm.amdgcn.ds.read.tr.contract";
      else if (opName == "tessera_rocm.wait") {
        // A targeted counter wait (vmcnt / lgkmcnt) lets the matrix core keep
        // issuing past an in-flight copy; only a wait with no counter class is
        // a true synchronization point that drains the wavefront (s_barrier).
        StringRef counter;
        if (auto attr = op->getAttrOfType<StringAttr>("counter"))
          counter = attr.getValue();
        if (counter == "vmcnt")
          markerName = "llvm.amdgcn.s.waitcnt.vmcnt.contract";
        else if (counter == "lgkmcnt")
          markerName = "llvm.amdgcn.s.waitcnt.lgkmcnt.contract";
        else if (counter.empty())
          markerName = "llvm.amdgcn.s.barrier.contract";
        else {
          op->emitError("tessera_rocm.wait: unknown counter class '")
              << counter << "' (expected 'vmcnt', 'lgkmcnt', or none)";
          signalPassFailure();
          return;
        }
      }

      auto marker = declareVoidMarker(module, markerName);
      rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                    SymbolRefAttr::get(marker), ValueRange{});
      replaceResultUsesWithUndef(op, rewriter);
      rewriter.eraseOp(op);
    }

    bool leakedROCMOp = false;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("tessera_rocm.")) {
        op->emitError("unsupported ROCm target op after ROCDL lowering");
        leakedROCMOp = true;
      }
    });
    if (leakedROCMOp)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTesseraToROCDLImpl() {
  return std::make_unique<LoweringPass>();
}
