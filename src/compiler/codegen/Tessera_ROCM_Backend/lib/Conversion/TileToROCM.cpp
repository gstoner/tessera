#include "TesseraROCM/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

// ── FP8 flavor derivation (B4 — arch-keyed FNUZ vs OCP) ─────────────────────
// The SAME canonical fp8 dtype encodes different bits across AMD generations.
// The *base* (e4m3 / e5m2) comes from the operand element type; the *suffix*
// (fnuz vs OCP-plain) comes from the target arch.  This C++ table is the
// emission-side mirror of the Python source of truth
// `tessera.compiler.rocm_target._FP8_SEMANTICS` and is held in sync by
// tests/unit/test_rocm_fp8_cpp_python_consistency.py.

static bool isFP8Element(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  return isa<Float8E4M3FNType, Float8E5M2Type, Float8E4M3FNUZType,
             Float8E5M2FNUZType>(t);
}

static std::string fp8Base(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (isa<Float8E4M3FNType, Float8E4M3FNUZType>(t))
    return "e4m3";
  if (isa<Float8E5M2Type, Float8E5M2FNUZType>(t))
    return "e5m2";
  return "";
}

// "fnuz" (CDNA 3) | "ocp" (CDNA 4 / RDNA 4 / gfx125x) | "none" (no FP8 path).
static llvm::StringRef fp8SemanticsForArch(llvm::StringRef arch) {
  if (arch == "gfx940" || arch == "gfx942")
    return "fnuz"; // CDNA 3 — E4M3FNUZ / E5M2FNUZ
  if (arch == "gfx950" || arch == "gfx1200" || arch == "gfx1250" ||
      arch == "gfx1251")
    return "ocp"; // CDNA 4 / RDNA 4 / gfx125x — OCP E4M3 / E5M2
  return "none";  // gfx90a, gfx1100, gfx1151 — no FP8 matrix path
}

struct LowerTileToROCMPass
    : PassWrapper<LowerTileToROCMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToROCMPass)

  // A pass carrying ``Option`` members is not implicitly copyable, but the pass
  // manager clones passes — provide the canonical copy ctor that re-inits the
  // options via their in-class initializers.
  LowerTileToROCMPass() = default;
  LowerTileToROCMPass(const LowerTileToROCMPass &other)
      : PassWrapper<LowerTileToROCMPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "lower-tile-to-rocm"; }
  StringRef getDescription() const final {
    return "Lower Tessera Tile IR matmul movement contracts to ROCm Target IR";
  }

  // Target gfx arch — drives the emitted `arch` attribute and the arch-keyed
  // FP8 flavor (FNUZ vs OCP).  Defaults to gfx90a for backward compatibility
  // with existing fixtures.
  Option<std::string> archOpt{
      *this, "arch",
      llvm::cl::desc("target gfx arch (gfx90a/gfx942/gfx950/gfx1200/...)"),
      llvm::cl::init("gfx90a")};

  void runOnOperation() override {
    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tile.mma" || name == "tile.async_copy" ||
          name == "tile.wait_async" || name == "tile.kv_cache" ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    StringRef arch = archOpt;
    Value lastAsyncToken;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      StringRef name = op->getName().getStringRef();

      if (name == "tile.mma") {
        if (op->getNumOperands() < 2 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.mma(lhs, rhs) -> result");
          signalPassFailure();
          return;
        }

        // Derive the arch-keyed FP8 flavor when the operands are FP8.  An FP8
        // matmul on an arch with no FP8 matrix path is a hard, named error
        // (Decision #21) — never a silent flavor guess.
        std::string fp8Flavor;
        if (isFP8Element(op->getOperand(0).getType())) {
          llvm::StringRef sem = fp8SemanticsForArch(arch);
          if (sem == "none") {
            op->emitError("ROCm lowering: FP8 matmul requested on arch '")
                << arch
                << "' which has no FP8 matrix path (see "
                   "tessera.compiler.rocm_target._FP8_SEMANTICS)";
            signalPassFailure();
            return;
          }
          std::string base = fp8Base(op->getOperand(0).getType());
          fp8Flavor = (sem == "fnuz") ? base + "fnuz" : base;
        }

        OperationState state(op->getLoc(), "tessera_rocm.mfma");
        // The v1 contract carries a scalar accumulator operand. Until Tile IR
        // models explicit accumulator SSA, use lhs as the artifact accumulator.
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           op->getOperand(0)});
        state.addTypes(op->getResultTypes());
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("shape", builder.getStringAttr("m16n16k16"));
        state.addAttribute("accum", builder.getStringAttr("f32"));
        state.addAttribute("source", builder.getStringAttr("tessera.matmul"));
        state.addAttribute("ordinal", builder.getI64IntegerAttr(0));
        if (!fp8Flavor.empty())
          state.addAttribute("fp8_flavor", builder.getStringAttr(fp8Flavor));
        Operation *rocmOp = builder.create(state);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.async_copy") {
        if (op->getNumOperands() < 3 || op->getNumResults() != 1) {
          op->emitError("ROCm lowering requires tile.async_copy(dst, src, bytes) -> token");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.async_copy");
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           op->getOperand(2)});
        state.addTypes(op->getResultTypes());
        state.addAttribute("src_space", builder.getStringAttr("global"));
        state.addAttribute("dst_space", builder.getStringAttr("lds"));
        state.addAttribute("arch", builder.getStringAttr(arch));
        Operation *rocmOp = builder.create(state);
        lastAsyncToken = rocmOp->getResult(0);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.wait_async") {
        if (!lastAsyncToken) {
          op->emitError("ROCm lowering requires tile.wait_async after tile.async_copy");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.wait");
        state.addOperands(lastAsyncToken);
        // The async copy is global→LDS (see the async_copy lowering above),
        // which retires on the vector-memory counter — so gate on vmcnt rather
        // than draining the wavefront with a full barrier.  This is the
        // decoupled-wait lever from the CDNA3 attention writeup: the matrix
        // core keeps issuing while the copy is still in flight.
        state.addAttribute("counter", builder.getStringAttr("vmcnt"));
        builder.create(state);
        op->erase();
        continue;
      }

      if (name == "tile.kv_cache") {
        op->emitError("ROCm lowering does not implement KV-cache artifacts in this phase");
        signalPassFailure();
        return;
      }

      if (name.starts_with("tile.tmem.")) {
        op->emitError("ROCm lowering does not support TMEM operations");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTileToROCMImpl() {
  return std::make_unique<LowerTileToROCMPass>();
}
