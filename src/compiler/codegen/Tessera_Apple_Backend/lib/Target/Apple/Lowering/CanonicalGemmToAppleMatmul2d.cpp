//===- CanonicalGemmToAppleMatmul2d.cpp - canonical GEMM → MPP matmul2d --===//
//
// APPLE-MATMUL2D-1 (2026-09-15). Apple consumption of the shared canonical
// M/N/K GEMM reduction (`CORE-GEMM-KLOOP-2026-07-25`) onto the Metal 4
// cooperative-tensor lane: the loop is re-formed as two
// `tessera_apple.gpu.tensor_view` bindings and one `tessera_apple.gpu.matmul2d`,
// carrying the storage pair, the fp32 accumulator and Apple's layout quantum
// in verified IR. This is the E2E-REAL-6 Apple family slice: the first GEMM
// whose contracts descend Graph -> Schedule -> Tile -> Target instead of being
// reconstructed by the Python packager.
//
// Recognition is not promotion (APPLE-TILE-2's standing rule). The incumbent
// production route is unchanged; this pass is registered standalone and is
// not in the default `tessera-lower-to-apple_gpu` pipeline until a paired
// corpus admits it.
//
// Layout is decided here and fails closed: a packed operand whose row stride
// is not on Apple's 128-byte quantum cannot be a legal MTLTensor view, so the
// pass refuses with the same reason the verifier states rather than emitting
// an op that would be rejected -- or, on 4-bit data, bound to the wrong bytes.
//===----------------------------------------------------------------------===//
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera::apple {
namespace {

constexpr StringRef kCanonicalKStep = "tessera.canonical_k_step";

bool isForOp(Operation *op) {
  return op && op->getName().getStringRef() == "scf.for";
}

// Same structural test as CanonicalGemmToAppleGPU: a marked matmul inside the
// three-deep M/N/K nest whose K loop carries the staged pipeline state.
Operation *canonicalNestRoot(Operation *matmul) {
  Operation *kLoop = matmul->getParentOp();
  if (!isForOp(kLoop)) return nullptr;
  Operation *nLoop = kLoop->getParentOp();
  if (!isForOp(nLoop)) return nullptr;
  Operation *mLoop = nLoop->getParentOp();
  if (!isForOp(mLoop)) return nullptr;
  bool carriesPipelineState = false;
  for (Value result : kLoop->getResults()) {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    result.getType().print(stream);
    if (StringRef(stream.str()).contains("tile.pipeline_state"))
      carriesPipelineState = true;
  }
  if (!carriesPipelineState || mLoop->getNumResults() != 1) return nullptr;
  return mLoop;
}

bool collectNestOperands(Operation *matmul, Operation *root, Value &a, Value &b) {
  if (matmul->getNumOperands() != 2) return false;
  SmallVector<Value, 2> sources;
  for (Value operand : matmul->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || def->getName().getStringRef() != "tensor.extract_slice") return false;
    Value source = def->getOperand(0);
    Operation *sourceDef = source.getDefiningOp();
    if (sourceDef && root->isProperAncestor(sourceDef)) return false;
    sources.push_back(source);
  }
  a = sources[0];
  b = sources[1];
  return true;
}

constexpr int64_t kTileM = 64, kTileN = 64, kSimdgroups = 4;  // measured best on M1 Max (M6)

struct CanonicalGemmToAppleMatmul2dPass
    : public PassWrapper<CanonicalGemmToAppleMatmul2dPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalGemmToAppleMatmul2dPass)

  StringRef getArgument() const override { return "tessera-apple-canonical-gemm-matmul2d"; }
  StringRef getDescription() const override {
    return "APPLE-MATMUL2D-1 — recognize the canonical M/N/K GEMM reduction and "
           "re-form it as Metal 4 tensor_view + matmul2d Target IR.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> marked;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.matmul" && op->hasAttr(kCanonicalKStep))
        marked.push_back(op);
    });
    for (Operation *matmul : marked) {
      Operation *root = canonicalNestRoot(matmul);
      if (!root) continue;
      Value a, b;
      if (!collectNestOperands(matmul, root, a, b)) {
        matmul->emitOpError("APPLE_MATMUL2D_UNRECOGNIZED: the canonical K step does not "
                            "slice two loop-invariant operands");
        signalPassFailure();
        return;
      }
      auto aType = dyn_cast<RankedTensorType>(a.getType());
      auto bType = dyn_cast<RankedTensorType>(b.getType());
      auto resultType = dyn_cast<RankedTensorType>(root->getResult(0).getType());
      if (!aType || !bType || !resultType || aType.getRank() != 2 || bType.getRank() != 2 ||
          resultType.getRank() != 2 || !aType.hasStaticShape() || !bType.hasStaticShape() ||
          !resultType.hasStaticShape()) {
        root->emitOpError("APPLE_MATMUL2D_SHAPE_UNSUPPORTED: matmul2d requires static "
                          "rank-2 operands and result");
        signalPassFailure();
        return;
      }
      Type aElem = aType.getElementType(), bElem = bType.getElementType();
      if (appleMatmul2dPairCode(aElem, bElem) < 0) {
        root->emitOpError("APPLE_MATMUL2D_PAIR_UNSUPPORTED: the Metal 4 matmul2d lane "
                          "accepts same-type f16/bf16/f8E4M3FN/f8E5M2/f4E2M1FN pairs and "
                          "f16 x {f8E4M3FN, f8E5M2, f4E2M1FN}");
        signalPassFailure();
        return;
      }
      if (!resultType.getElementType().isF32()) {
        root->emitOpError("APPLE_MATMUL2D_ACCUM_UNSUPPORTED: the canonical reduction must "
                          "accumulate in fp32 for the Metal 4 matmul2d lane");
        signalPassFailure();
        return;
      }
      const int64_t M = aType.getDimSize(0), K = aType.getDimSize(1), N = bType.getDimSize(1);
      // Packed row-major operands: extents innermost-first, strides {1, ld}.
      const int64_t aExt[2] = {K, M}, aStr[2] = {1, K};
      const int64_t bExt[2] = {N, K}, bStr[2] = {1, N};
      struct Candidate { Type elem; ArrayRef<int64_t> ext; ArrayRef<int64_t> str; const char *role; };
      const Candidate candidates[2] = {
          {aElem, ArrayRef<int64_t>(aExt, 2), ArrayRef<int64_t>(aStr, 2), "a"},
          {bElem, ArrayRef<int64_t>(bExt, 2), ArrayRef<int64_t>(bStr, 2), "b"}};
      for (const Candidate &c : candidates) {
        const std::string reason = appleTensorViewLayoutReason(c.elem, c.ext, c.str, 0);
        if (!reason.empty()) {
          root->emitOpError("APPLE_MATMUL2D_LAYOUT_UNSUPPORTED: operand ")
              << c.role << " cannot be bound as an MTLTensor view: " << reason;
          signalPassFailure();
          return;
        }
      }

      OpBuilder builder(root);
      MLIRContext *ctx = builder.getContext();
      auto makeView = [&](Value buffer, Type elem, ArrayRef<int64_t> ext, ArrayRef<int64_t> str) {
        OperationState state(root->getLoc(), "tessera_apple.gpu.tensor_view");
        state.addOperands({buffer});
        state.addTypes({TensorViewType::get(ctx, elem)});
        state.addAttribute("byte_offset", builder.getI64IntegerAttr(0));
        state.addAttribute("extents", DenseI64ArrayAttr::get(ctx, ext));
        state.addAttribute("strides", DenseI64ArrayAttr::get(ctx, str));
        return builder.create(state)->getResult(0);
      };
      Value aView = makeView(a, aElem, ArrayRef<int64_t>(aExt, 2), ArrayRef<int64_t>(aStr, 2));
      Value bView = makeView(b, bElem, ArrayRef<int64_t>(bExt, 2), ArrayRef<int64_t>(bStr, 2));
      OperationState state(root->getLoc(), "tessera_apple.gpu.matmul2d");
      state.addOperands({aView, bView});
      state.addTypes({resultType});
      state.addAttribute("tile_m", builder.getI64IntegerAttr(kTileM));
      state.addAttribute("tile_n", builder.getI64IntegerAttr(kTileN));
      state.addAttribute("simdgroups", builder.getI64IntegerAttr(kSimdgroups));
      state.addAttribute("accumulate", builder.getStringAttr("f32"));
      // Provenance: this op answers the shared canonical reduction.
      state.addAttribute("tessera_apple.canonical_k_loop", builder.getBoolAttr(true));
      if (matmul->hasAttr("tessera.ragged_zero_pad"))
        state.addAttribute("tessera_apple.ragged_zero_pad", builder.getBoolAttr(true));
      Operation *mm = builder.create(state);
      root->getResult(0).replaceAllUsesWith(mm->getResult(0));
      root->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCanonicalGemmToAppleMatmul2dPass() {
  return std::make_unique<CanonicalGemmToAppleMatmul2dPass>();
}

} // namespace tessera::apple
