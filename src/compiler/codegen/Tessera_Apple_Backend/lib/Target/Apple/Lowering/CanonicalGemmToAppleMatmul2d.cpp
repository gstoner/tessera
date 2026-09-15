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
// Follow-through (same day):
//  * Ragged M/N. The shared TilingPass zero-pads a ragged operand into a
//    tile-multiple constant (`tensor.insert_slice %src into %zeros`) and slices
//    the padded product back. MPP matmul2d edge-checks partial tiles itself,
//    so the pass looks through that padding: the view binds the ORIGINAL
//    operand with its true extents, the result is the true [M, N], and the
//    trailing `tensor.extract_slice` disappears. The op is tagged
//    `tessera_apple.ragged_tail` so the runtime contract (partial-tile store)
//    is named in IR. When the unpadded operand is not a legal MTLTensor view
//    (a packed 8-bit K off the 32-element quantum) the padded form is kept and
//    tagged `tessera_apple.ragged_zero_pad`, as before.
//  * Sub-block origins. An operand that is a static unit-stride
//    `tensor.extract_slice` of a larger rank-2 tensor is bound as a view INTO
//    the parent: nonzero `byte_offset`, the parent's row stride. Nothing is
//    copied, and the layout verifier decides whether that origin is reachable
//    (128-byte data-plane alignment; 256 bytes for 4-bit data).
//
// Layout is decided here and fails closed: a packed operand whose row stride
// is not on Apple's 128-byte quantum cannot be a legal MTLTensor view, so the
// pass refuses with the same reason the verifier states rather than emitting
// an op that would be rejected -- or, on 4-bit data, bound to the wrong bytes.
//===----------------------------------------------------------------------===//
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
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

bool isNamed(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
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
    if (!isNamed(def, "tensor.extract_slice")) return false;
    Value source = def->getOperand(0);
    Operation *sourceDef = source.getDefiningOp();
    if (sourceDef && root->isProperAncestor(sourceDef)) return false;
    sources.push_back(source);
  }
  a = sources[0];
  b = sources[1];
  return true;
}

// Static offsets/sizes/strides of a tensor.insert_slice / extract_slice with
// no dynamic operands (the shape the tiling pass emits). False otherwise.
bool staticSliceParams(Operation *slice, unsigned nonDynamicOperands,
                       ArrayRef<int64_t> &offsets, ArrayRef<int64_t> &sizes,
                       ArrayRef<int64_t> &strides) {
  if (slice->getNumOperands() != nonDynamicOperands) return false;
  auto off = slice->getAttrOfType<DenseI64ArrayAttr>("static_offsets");
  auto siz = slice->getAttrOfType<DenseI64ArrayAttr>("static_sizes");
  auto str = slice->getAttrOfType<DenseI64ArrayAttr>("static_strides");
  if (!off || !siz || !str) return false;
  offsets = off.asArrayRef();
  sizes = siz.asArrayRef();
  strides = str.asArrayRef();
  return offsets.size() == 2 && sizes.size() == 2 && strides.size() == 2;
}

bool isZeroSplatConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!isNamed(def, "arith.constant")) return false;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(def->getAttr("value"));
  if (!dense || !dense.isSplat()) return false;
  if (isa<FloatType>(dense.getElementType()))
    return dense.getSplatValue<APFloat>().isZero();
  if (isa<IntegerType>(dense.getElementType()))
    return dense.getSplatValue<APInt>().isZero();
  return false;
}

// The tiling pass's ragged zero-pad: `tensor.insert_slice %src into %zeros
// [0, 0] [src sizes] [1, 1]`. Returns %src, or null.
Value zeroPadSource(Value padded) {
  Operation *ins = padded.getDefiningOp();
  if (!isNamed(ins, "tensor.insert_slice")) return nullptr;
  ArrayRef<int64_t> offsets, sizes, strides;
  if (!staticSliceParams(ins, /*source, dest*/ 2, offsets, sizes, strides)) return nullptr;
  Value src = ins->getOperand(0), dest = ins->getOperand(1);
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  if (!srcType || srcType.getRank() != 2 || !srcType.hasStaticShape()) return nullptr;
  if (offsets[0] != 0 || offsets[1] != 0 || strides[0] != 1 || strides[1] != 1 ||
      sizes[0] != srcType.getDimSize(0) || sizes[1] != srcType.getDimSize(1))
    return nullptr;
  if (!isZeroSplatConstant(dest)) return nullptr;
  return src;
}

// One operand as the view will bind it: `buffer` is what the kernel_call
// receives; rows x cols is the logical operand; `ld` its row stride in
// elements of the buffer; `byteOffset` where the window starts.
struct Binding {
  Value buffer;
  Type elem;
  int64_t rows = 0, cols = 0, ld = 0, byteOffset = 0;
  bool unpadded = false;  // looked through the tiling pass's zero-pad
  bool subBlock = false;  // folded a static extract_slice origin
};

bool bindingOf(Value operand, Binding &b) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape()) return false;
  b.buffer = operand;
  b.elem = type.getElementType();
  b.rows = type.getDimSize(0);
  b.cols = type.getDimSize(1);
  b.ld = b.cols;
  b.byteOffset = 0;
  return true;
}

// Fold static unit-stride extract_slice origins into the view. Returns false
// (with `why` set) only when an origin is not representable in bytes.
bool foldSubBlock(Binding &b, std::string &why) {
  const int64_t bits = appleTensorStorageBits(b.elem);
  for (;;) {
    Operation *ex = b.buffer.getDefiningOp();
    if (!isNamed(ex, "tensor.extract_slice")) return true;
    ArrayRef<int64_t> offsets, sizes, strides;
    if (!staticSliceParams(ex, /*source*/ 1, offsets, sizes, strides)) return true;
    auto parent = dyn_cast<RankedTensorType>(ex->getOperand(0).getType());
    if (!parent || parent.getRank() != 2 || !parent.hasStaticShape() ||
        parent.getElementType() != b.elem || strides[0] != 1 || strides[1] != 1 ||
        sizes[0] != b.rows || sizes[1] != b.cols)
      return true;
    // The window's element origin inside the parent, at the parent's stride.
    const int64_t elementOrigin = offsets[0] * parent.getDimSize(1) + offsets[1];
    const int64_t bitOrigin = elementOrigin * bits + b.byteOffset * 8;
    if (bitOrigin % 8 != 0) {
      why = "a 4-bit sub-block origin at an odd element has no byte address";
      return false;
    }
    b.byteOffset = bitOrigin / 8;
    b.ld = parent.getDimSize(1);
    b.buffer = ex->getOperand(0);
    b.subBlock = true;
  }
}

std::string layoutReasonA(const Binding &a) {
  const int64_t ext[2] = {a.cols, a.rows}, str[2] = {1, a.ld};
  return appleTensorViewLayoutReason(a.elem, ArrayRef<int64_t>(ext, 2), ArrayRef<int64_t>(str, 2),
                                     a.byteOffset);
}

constexpr int64_t kTileM = 64, kTileN = 64, kSimdgroups = 4;  // measured best on M1 Max (M6)

struct CanonicalGemmToAppleMatmul2dPass
    : public PassWrapper<CanonicalGemmToAppleMatmul2dPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalGemmToAppleMatmul2dPass)

  CanonicalGemmToAppleMatmul2dPass() = default;
  CanonicalGemmToAppleMatmul2dPass(const CanonicalGemmToAppleMatmul2dPass &other)
      : PassWrapper(other) {}
  explicit CanonicalGemmToAppleMatmul2dPass(StringRef admitPairs) { admit = admitPairs.str(); }

  // Admission is measured, not assumed (paired corpus, M1 Max, 2026-09-15,
  // `benchmarks/baselines/apple_matmul2d_route_corpus_20260915/`): against
  // what production dispatches today the compiled route loses on every f16
  // GEMM shape and is within +-10% of the runtime's own bf16 entry, so those
  // pairs keep their incumbent; the 8/4-bit pairs have NO incumbent (the
  // pipeline used to emit an MPSGraph claim it cannot execute), so they are
  // admitted. `all` is for standalone use and the corpus itself.
  Option<std::string> admit{
      *this, "admit",
      llvm::cl::desc("Operand pairs to re-form: 'all' (default) or 'lowp' (only the "
                     "f8E4M3FN / f8E5M2 / f4E2M1FN pairs; the default pipeline's admission)"),
      llvm::cl::init("all")};

  StringRef getArgument() const override { return "tessera-apple-canonical-gemm-matmul2d"; }
  StringRef getDescription() const override {
    return "APPLE-MATMUL2D-1 — recognize the canonical M/N/K GEMM reduction and "
           "re-form it as Metal 4 tensor_view + matmul2d Target IR.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    if (admit != "all" && admit != "lowp") {
      getOperation().emitError("APPLE_MATMUL2D_ADMIT: admit must be 'all' or 'lowp', got '")
          << admit << "'";
      signalPassFailure();
      return;
    }
    SmallVector<Operation *> marked;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.matmul" && op->hasAttr(kCanonicalKStep))
        marked.push_back(op);
    });
    for (Operation *matmul : marked) {
      Operation *root = canonicalNestRoot(matmul);
      if (!root) continue;
      Value aVal, bVal;
      if (!collectNestOperands(matmul, root, aVal, bVal)) {
        matmul->emitOpError("APPLE_MATMUL2D_UNRECOGNIZED: the canonical K step does not "
                            "slice two loop-invariant operands");
        signalPassFailure();
        return;
      }
      auto resultType = dyn_cast<RankedTensorType>(root->getResult(0).getType());
      Binding a, b;
      if (!bindingOf(aVal, a) || !bindingOf(bVal, b) || !resultType || resultType.getRank() != 2 ||
          !resultType.hasStaticShape()) {
        root->emitOpError("APPLE_MATMUL2D_SHAPE_UNSUPPORTED: matmul2d requires static "
                          "rank-2 operands and result");
        signalPassFailure();
        return;
      }
      // Pairs outside the admitted set are left for the incumbent lowering.
      if (admit == "lowp" && appleMatmul2dPairCode(a.elem, b.elem) >= 10) continue;
      if (appleMatmul2dPairCode(a.elem, b.elem) < 0) {
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

      // Ragged operands: prefer the true extents when they are a legal view.
      auto tryUnpad = [&](Binding &bind) {
        Value src = zeroPadSource(bind.buffer);
        if (!src) return;
        Binding candidate;
        std::string why;
        if (!bindingOf(src, candidate) || !foldSubBlock(candidate, why)) return;
        candidate.unpadded = true;
        // Both operands must still agree on K after unpadding; that is checked
        // below on the chosen bindings, so only legality is judged here.
        if (layoutReasonA(candidate).empty()) bind = candidate;
      };
      tryUnpad(a);
      tryUnpad(b);
      if (a.unpadded != b.unpadded && (a.unpadded ? b.rows : a.cols) != (a.unpadded ? a.cols : b.rows)) {
        // One side was unpadded on K and the other stayed padded: K disagrees.
        // Keep the padded forms so the IR stays consistent.
        if (a.unpadded) bindingOf(aVal, a);
        if (b.unpadded) bindingOf(bVal, b);
      }
      std::string why;
      if (!foldSubBlock(a, why) || !foldSubBlock(b, why)) {
        root->emitOpError("APPLE_MATMUL2D_LAYOUT_UNSUPPORTED: ") << why;
        signalPassFailure();
        return;
      }
      if (a.cols != b.rows) {
        root->emitOpError("APPLE_MATMUL2D_SHAPE_UNSUPPORTED: operands disagree on K (")
            << a.cols << " vs " << b.rows << ")";
        signalPassFailure();
        return;
      }
      for (const auto &[bind, role] : {std::pair{&a, "a"}, std::pair{&b, "b"}}) {
        const std::string reason = layoutReasonA(*bind);
        if (!reason.empty()) {
          root->emitOpError("APPLE_MATMUL2D_LAYOUT_UNSUPPORTED: operand ")
              << role << " cannot be bound as an MTLTensor view: " << reason;
          signalPassFailure();
          return;
        }
      }
      const int64_t M = a.rows, N = b.cols;
      const bool unpadded = a.unpadded || b.unpadded;
      // A zero-padded operand that stayed padded (its true extents were not a
      // legal view) is what `ragged_zero_pad` now names: the IR states padding
      // only when padding is actually consumed.
      const bool consumesPadding = (!a.unpadded && zeroPadSource(aVal)) ||
                                   (!b.unpadded && zeroPadSource(bVal));

      // With true extents the padded product's only legal consumer is the
      // identity slice back to [M, N]; anything else keeps the padded form.
      Operation *tailSlice = nullptr;
      if (unpadded) {
        if (root->getResult(0).hasOneUse()) {
          Operation *user = *root->getResult(0).getUsers().begin();
          ArrayRef<int64_t> offsets, sizes, strides;
          if (isNamed(user, "tensor.extract_slice") &&
              staticSliceParams(user, 1, offsets, sizes, strides) && offsets[0] == 0 &&
              offsets[1] == 0 && strides[0] == 1 && strides[1] == 1 && sizes[0] == M &&
              sizes[1] == N)
            tailSlice = user;
        }
        if (!tailSlice) {
          root->emitOpError("APPLE_MATMUL2D_RAGGED_UNSUPPORTED: the zero-padded product is "
                            "consumed by something other than the slice back to its true "
                            "extents, so the padding cannot be looked through");
          signalPassFailure();
          return;
        }
      }

      OpBuilder builder(root);
      MLIRContext *ctx = builder.getContext();
      auto makeView = [&](const Binding &bind) {
        const int64_t ext[2] = {bind.cols, bind.rows}, str[2] = {1, bind.ld};
        OperationState state(root->getLoc(), "tessera_apple.gpu.tensor_view");
        state.addOperands({bind.buffer});
        state.addTypes({TensorViewType::get(ctx, bind.elem)});
        state.addAttribute("byte_offset", builder.getI64IntegerAttr(bind.byteOffset));
        state.addAttribute("extents", DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>(ext, 2)));
        state.addAttribute("strides", DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>(str, 2)));
        return builder.create(state)->getResult(0);
      };
      Value aView = makeView(a);
      Value bView = makeView(b);
      OperationState state(root->getLoc(), "tessera_apple.gpu.matmul2d");
      state.addOperands({aView, bView});
      state.addTypes({RankedTensorType::get({M, N}, resultType.getElementType())});
      state.addAttribute("tile_m", builder.getI64IntegerAttr(kTileM));
      state.addAttribute("tile_n", builder.getI64IntegerAttr(kTileN));
      state.addAttribute("simdgroups", builder.getI64IntegerAttr(kSimdgroups));
      state.addAttribute("accumulate", builder.getStringAttr("f32"));
      // Provenance: this op answers the shared canonical reduction.
      state.addAttribute("tessera_apple.canonical_k_loop", builder.getBoolAttr(true));
      if (unpadded)
        state.addAttribute("tessera_apple.ragged_tail", builder.getBoolAttr(true));
      if (consumesPadding)
        state.addAttribute("tessera_apple.ragged_zero_pad", builder.getBoolAttr(true));
      Operation *mm = builder.create(state);
      if (tailSlice) {
        tailSlice->getResult(0).replaceAllUsesWith(mm->getResult(0));
        tailSlice->erase();
      } else {
        root->getResult(0).replaceAllUsesWith(mm->getResult(0));
      }
      root->erase();
      // Padding constants, insert_slices and folded sub-block slices are dead
      // once the view binds the true operand; drop them so the IR states one
      // binding, not a binding plus the copies it replaced.
      for (Value v : {aVal, bVal}) {
        Operation *def = v.getDefiningOp();
        while (def && def->use_empty() &&
               (isNamed(def, "tensor.insert_slice") || isNamed(def, "tensor.extract_slice"))) {
          Value zeros = isNamed(def, "tensor.insert_slice") ? def->getOperand(1) : Value();
          Value next = def->getOperand(0);
          def->erase();
          if (zeros)
            if (Operation *z = zeros.getDefiningOp(); z && z->use_empty()) z->erase();
          def = next.getDefiningOp();
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCanonicalGemmToAppleMatmul2dPass(StringRef admit) {
  return std::make_unique<CanonicalGemmToAppleMatmul2dPass>(admit);
}

} // namespace tessera::apple
