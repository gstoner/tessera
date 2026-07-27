// CanonicalGemmToAppleGPU.cpp — APPLE-TILE-2 (2026-07-26).
//
// Apple consumption of the canonical M/N/K GEMM reduction contract
// (`CORE-GEMM-KLOOP-2026-07-25`). The shared TilingPass now states a dense
// contraction as three nested `scf.for` loops carrying an FP32 accumulator, a
// staged `!tile.pipeline_state`, ragged zero-pad guards, and structured
// per-step slices. NVIDIA and ROCm consume that loop by mapping each K step
// onto a Tensor Core / MFMA fragment issue.
//
// Apple's physical answer is different, and deliberately so. The loop is a
// *semantic contract* — "reduce over K in FP32, zero-padding the ragged tail"
// — not a schedule Metal must reproduce statement by statement. The Apple7+
// `simdgroup_matrix` GEMM this backend already emits (`msl_gemm_emit.py`)
// implements exactly that reduction, with its own cooperative staging and
// edge-masked loads. So the Apple consumer *recognizes* the canonical
// reduction and re-forms it as one simdgroup dispatch, rather than emitting a
// literal three-loop MSL nest that would be slower and no more correct.
//
// Standing incumbent rule (APPLE-TILE-2): recognizing the loop does NOT change
// production route selection. The value-mode path — whole-tensor `tile.matmul`
// into Accelerate/MPS — remains the incumbent for every measured row until a
// two-run paired corpus shows the simdgroup route winning in the intended
// timing domain. This pass makes the loop form *reach* an Apple route so the
// two can be compared at all; the comparison, not this pass, promotes anything.

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera::apple {
namespace {

bool isForOp(Operation *op) {
  return op && op->getName().getStringRef() == "scf.for";
}

// The K-step marker the shared TilingPass stamps on the canonical inner
// contraction. Its presence is what distinguishes the canonical reduction from
// an ordinary user-written loop containing a matmul.
constexpr StringRef kCanonicalKStep = "tessera.canonical_k_step";

// Climb from the marked inner matmul to the outermost loop of the canonical
// nest, requiring exactly the three-deep M/N/K structure. Returns null when the
// surrounding shape is anything else — an unrecognized nest is left alone for a
// later pass, not misclaimed.
Operation *canonicalNestRoot(Operation *matmul) {
  Operation *kLoop = matmul->getParentOp();
  if (!isForOp(kLoop))
    return nullptr;
  Operation *nLoop = kLoop->getParentOp();
  if (!isForOp(nLoop))
    return nullptr;
  Operation *mLoop = nLoop->getParentOp();
  if (!isForOp(mLoop))
    return nullptr;
  // The K loop must carry the staged pipeline state alongside the accumulator;
  // that is the SSA evidence this is the canonical reduction rather than a
  // coincidental triple loop.
  bool carriesPipelineState = false;
  for (Value result : kLoop->getResults()) {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    result.getType().print(stream);
    if (StringRef(stream.str()).contains("tile.pipeline_state"))
      carriesPipelineState = true;
  }
  if (!carriesPipelineState)
    return nullptr;
  if (mLoop->getNumResults() != 1)
    return nullptr;
  return mLoop;
}

// Recover the whole-tensor A and B the nest slices. Every `tensor.extract_slice`
// feeding the marked matmul must read a value defined outside the nest;
// anything else means the operands are themselves computed per step and the
// contraction is not a single dense GEMM.
bool collectNestOperands(Operation *matmul, Operation *root, Value &a, Value &b) {
  if (matmul->getNumOperands() != 2)
    return false;
  SmallVector<Value, 2> sources;
  for (Value operand : matmul->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || def->getName().getStringRef() != "tensor.extract_slice")
      return false;
    Value source = def->getOperand(0);
    Operation *sourceDef = source.getDefiningOp();
    // Defined outside the nest: either a block argument of an enclosing
    // function, or an operation that is not nested inside the root loop.
    if (sourceDef && root->isProperAncestor(sourceDef))
      return false;
    sources.push_back(source);
  }
  a = sources[0];
  b = sources[1];
  return true;
}

struct CanonicalGemmToAppleGPUPass
    : public PassWrapper<CanonicalGemmToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalGemmToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-apple-canonical-gemm";
  }
  StringRef getDescription() const override {
    return "APPLE-TILE-2 — recognize the canonical M/N/K GEMM reduction and "
           "re-form it as one Apple simdgroup_matrix dispatch.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> marked;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.matmul" &&
          op->hasAttr(kCanonicalKStep))
        marked.push_back(op);
    });

    for (Operation *matmul : marked) {
      Operation *root = canonicalNestRoot(matmul);
      if (!root)
        continue; // Not the canonical nest; leave it for another consumer.

      Value a, b;
      if (!collectNestOperands(matmul, root, a, b)) {
        matmul->emitOpError(
            "APPLE_CANONICAL_GEMM_UNRECOGNIZED: the canonical K step does not "
            "slice two loop-invariant operands, so it is not a single dense "
            "contraction apple_gpu can re-form");
        signalPassFailure();
        return;
      }

      auto aType = llvm::dyn_cast<RankedTensorType>(a.getType());
      auto bType = llvm::dyn_cast<RankedTensorType>(b.getType());
      auto resultType =
          llvm::dyn_cast<RankedTensorType>(root->getResult(0).getType());
      if (!aType || !bType || !resultType || aType.getRank() != 2 ||
          bType.getRank() != 2 || resultType.getRank() != 2 ||
          !aType.hasStaticShape() || !bType.hasStaticShape() ||
          !resultType.hasStaticShape()) {
        root->emitOpError(
            "APPLE_CANONICAL_GEMM_SHAPE_UNSUPPORTED: the simdgroup route "
            "requires static rank-2 operands and result");
        signalPassFailure();
        return;
      }

      Type storage = aType.getElementType();
      StringRef symbol;
      StringRef dtype;
      if (storage.isF16() && bType.getElementType().isF16()) {
        symbol = "tessera_apple_gpu_tile_simdgroup_gemm_f16";
        dtype = "fp16";
      } else if (storage.isBF16() && bType.getElementType().isBF16()) {
        symbol = "tessera_apple_gpu_tile_simdgroup_gemm_bf16";
        dtype = "bf16";
      } else {
        root->emitOpError(
            "APPLE_CANONICAL_GEMM_DTYPE_UNSUPPORTED: apple_gpu re-forms the "
            "canonical reduction only for fp16/bf16 storage with fp32 "
            "accumulation (simdgroup_matrix contract)");
        signalPassFailure();
        return;
      }
      if (!resultType.getElementType().isF32()) {
        root->emitOpError(
            "APPLE_CANONICAL_GEMM_ACCUM_UNSUPPORTED: the canonical reduction "
            "must accumulate in fp32 for the Apple simdgroup route");
        signalPassFailure();
        return;
      }

      OpBuilder builder(root);
      OperationState state(root->getLoc(), "tessera_apple.gpu.kernel_call");
      state.addOperands({a, b});
      state.addTypes({resultType});
      state.addAttribute("op_kind", builder.getStringAttr("tile_simdgroup_gemm"));
      state.addAttribute("symbol", builder.getStringAttr(symbol));
      state.addAttribute("abi", builder.getStringAttr("tile_simdgroup_msl"));
      state.addAttribute("status", builder.getStringAttr("executable"));
      state.addAttribute("framework", builder.getStringAttr("Metal"));
      state.addAttribute("dtype", builder.getStringAttr(dtype));
      // Provenance: this dispatch answers the shared canonical reduction, and
      // says so, so a reader never has to infer which contract it satisfies.
      state.addAttribute("tessera_apple.canonical_k_loop",
                         builder.getBoolAttr(true));
      state.addAttribute("tessera_apple.accumulate",
                         builder.getStringAttr("fp32"));
      // Carry the loop's own tile decision rather than re-deriving one.
      for (StringRef tileAttr :
           {"tessera.tile_m", "tessera.tile_n", "tessera.tile_k"}) {
        if (auto value = matmul->getAttrOfType<IntegerAttr>(tileAttr))
          state.addAttribute(
              ("tessera_apple." + tileAttr.drop_front(strlen("tessera."))).str(),
              value);
      }
      if (matmul->hasAttr("tessera.ragged_zero_pad"))
        state.addAttribute("tessera_apple.ragged_zero_pad",
                           builder.getBoolAttr(true));

      Operation *call = builder.create(state);
      root->getResult(0).replaceAllUsesWith(call->getResult(0));
      root->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCanonicalGemmToAppleGPUPass() {
  return std::make_unique<CanonicalGemmToAppleGPUPass>();
}

} // namespace tessera::apple
