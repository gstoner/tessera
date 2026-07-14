#include "TesseraROCM/Passes.h"

#include "Tessera/Dialect/Tile/TileDialect.h"
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

// RDNA arches use the WMMA matrix instruction; CDNA arches use MFMA. The
// matmul tile lowering must pick the right matrix op per arch — emitting MFMA
// on RDNA (which has no matrix-fused-multiply-add core) is a silent miscompile.
// RDNA = gfx11xx (RDNA 3 / 3.5) and gfx12xx (RDNA 4); CDNA = gfx9xx.
static bool isWmmaArch(llvm::StringRef arch) {
  return arch.starts_with("gfx11") || arch.starts_with("gfx12");
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

static LogicalResult rejectUnconsumedStoragePack(Operation *op) {
  if (!op->hasAttr("tessera.storage_packed"))
    return success();
  if (op->hasAttr("tessera.storage_pack"))
    return success();
  op->emitOpError(
      "ROCM_LOWERING_UNCONSUMED_STORAGE_PACK: packed low-precision storage "
      "reached ROCm lowering without a tessera.storage_pack consumer "
      "descriptor; run tessera-storage-pack-consume or add an explicit ROCm "
      "consumer before lowering.");
  return failure();
}

static void copyAttrIfPresent(OperationState &state, Operation *op,
                              StringRef name) {
  if (Attribute attr = op->getAttr(name))
    state.addAttribute(name, attr);
}

static bool layoutHasLdsAxis(tessera::tile::TileLayoutAttr layout) {
  for (StringAttr axis : layout.getShardAxes())
    if (axis.getValue() == "lds")
      return true;
  return false;
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::tile::TesseraTileDialect>();
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
      if (name == "tile.mma" || name == "tile.view" ||
          name == "tile.fragment_pack" || name == "tile.fragment_unpack" ||
          name == "tile.matmul_kernel" ||
          name == "tile.async_copy" ||
          name == "tile.wait_async" || name == "tile.kv_cache" ||
          name.starts_with("tile.tmem."))
        worklist.push_back(op);
    });

    StringRef arch = archOpt;
    // FIFO of outstanding async copies (oldest first), keyed by the stamped
    // tile.barrier_id from rocm-wave-lds-pipeline. A wait retires the id it
    // names (or the oldest if idless) — NOT "the last token", so each wait
    // gates the right copy and double-buffering is correct.
    SmallVector<std::pair<std::string, Value>> outstanding;
    for (Operation *op : worklist) {
      OpBuilder builder(op);
      StringRef name = op->getName().getStringRef();

      if (name == "tile.view" || name == "tile.fragment_pack" ||
          name == "tile.fragment_unpack") {
        op->emitError("ROCm lowering requires the Tile-to-fragment pack "
                      "lowering before backend lowering");
        signalPassFailure();
        return;
      }

      if (name == "tile.matmul_kernel") {
        op->emitError("ROCm tile.matmul_kernel pack/loop/epilogue materializer "
                      "is not implemented for this target");
        signalPassFailure();
        return;
      }

      if (name == "tile.mma") {
        if (llvm::any_of(op->getOperandTypes(), [](Type type) {
              return isa<tessera::tile::FragmentType>(type);
            })) {
          op->emitError("ROCm lowering requires Tile-to-fragment pack "
                        "lowering before lowering typed tile.mma");
          signalPassFailure();
          return;
        }
        if (failed(rejectUnconsumedStoragePack(op))) {
          signalPassFailure();
          return;
        }
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

        // RDNA -> WMMA, CDNA -> MFMA. Same 16x16x16 v1 artifact contract; only
        // the target matrix op (and its eventual ROCDL marker) differs.
        StringRef matrixOp =
            isWmmaArch(arch) ? "tessera_rocm.wmma" : "tessera_rocm.mfma";
        OperationState state(op->getLoc(), matrixOp);
        // Accumulator: a 3-operand tile.mma(lhs, rhs, acc) carries the real
        // accumulator SSA value (the executable Fork-A form the GEMM generator
        // emits in --via-tile mode) — thread it straight through so the lowered
        // tessera_rocm.wmma is bit-identical to the direct generator's. The
        // legacy 2-operand artifact form has no accumulator SSA yet, so it falls
        // back to lhs as a typed placeholder (IR-contract lowering, not run).
        Value acc = op->getNumOperands() >= 3 ? op->getOperand(2)
                                              : op->getOperand(0);
        state.addOperands({op->getOperand(0), op->getOperand(1), acc});
        state.addTypes(op->getResultTypes());
        state.addAttribute("arch", builder.getStringAttr(arch));
        state.addAttribute("shape", builder.getStringAttr("m16n16k16"));
        state.addAttribute("accum", builder.getStringAttr("f32"));
        state.addAttribute("source", builder.getStringAttr("tessera.matmul"));
        state.addAttribute("ordinal", builder.getI64IntegerAttr(0));
        if (!fp8Flavor.empty())
          state.addAttribute("fp8_flavor", builder.getStringAttr(fp8Flavor));
        copyAttrIfPresent(state, op, "numeric_policy");
        copyAttrIfPresent(state, op, "tessera.storage_pack");
        copyAttrIfPresent(state, op, "tile.pipeline_depths");
        copyAttrIfPresent(state, op, "tile.rocm_matrix_path");
        Operation *rocmOp = builder.create(state);
        op->replaceAllUsesWith(rocmOp->getResults());
        op->erase();
        continue;
      }

      if (name == "tile.async_copy") {
        if (failed(rejectUnconsumedStoragePack(op))) {
          signalPassFailure();
          return;
        }
        // The planner may append a !tile.async_token result (the SSA completion
        // edge). It is the trailing result; the leading result is the staged
        // tile the rocm op produces. Require exactly one data result.
        unsigned numResults = op->getNumResults();
        bool hasTileToken =
            numResults >= 1 &&
            isa<tessera::tile::AsyncTokenType>(
                op->getResult(numResults - 1).getType());
        unsigned numData = hasTileToken ? numResults - 1 : numResults;
        if (op->getNumOperands() < 3 || numData != 1) {
          op->emitError("ROCm lowering requires tile.async_copy(dst, src, bytes) -> token");
          signalPassFailure();
          return;
        }

        OperationState state(op->getLoc(), "tessera_rocm.async_copy");
        state.addOperands({op->getOperand(0), op->getOperand(1),
                           op->getOperand(2)});
        state.addTypes(op->getResult(0).getType());
        state.addAttribute("src_space", builder.getStringAttr("global"));
        state.addAttribute("dst_space", builder.getStringAttr("lds"));
        state.addAttribute("arch", builder.getStringAttr(arch));
        if (auto buf = op->getAttrOfType<tessera::tile::TileBufferRefAttr>(
                "tile.buf")) {
          if (buf.getSpace() != "lds") {
            op->emitOpError(
                "ROCM_LOWERING_NON_LDS_BUFFER: tile.async_copy expected "
                "#tile.buffer_ref<space = \"lds\"> for ROCm global-to-LDS "
                "movement.");
            signalPassFailure();
            return;
          }
          if (buf.getAccess() != "write") {
            op->emitOpError(
                "ROCM_LOWERING_NON_WRITE_BUFFER: tile.async_copy expected "
                "#tile.buffer_ref access = \"write\" for the LDS destination.");
            signalPassFailure();
            return;
          }
          state.addAttribute("buffer", builder.getStringAttr(buf.getName()));
        }
        if (auto layout =
                op->getAttrOfType<tessera::tile::TileLayoutAttr>(
                    "tile.layout")) {
          if (!layoutHasLdsAxis(layout)) {
            op->emitOpError(
                "ROCM_LOWERING_LAYOUT_NOT_LDS: ROCm async copy can only "
                "consume #tile.layout placements that include the lds axis.");
            signalPassFailure();
            return;
          }
          state.addAttribute("uses_tile_layout", builder.getBoolAttr(true));
          state.addAttribute("layout_storage", builder.getStringAttr("lds"));
          copyAttrIfPresent(state, op, "tile.layout");
        }
        copyAttrIfPresent(state, op, "numeric_policy");
        copyAttrIfPresent(state, op, "tessera.storage_pack");
        copyAttrIfPresent(state, op, "tile.pipeline_depths");
        Operation *rocmOp = builder.create(state);
        // Record this copy in the FIFO keyed by its stamped barrier id and its
        // SSA token (the rocm result). A wait retires it by SSA value when its
        // operand names the token, else by id/order.
        std::string id;
        if (auto a = op->getAttrOfType<tessera::tile::TileBufferRefAttr>(
                "tile.buf"))
          id = a.getName().str();
        if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
          id = a.getValue().str();
        outstanding.push_back({id, rocmOp->getResult(0)});
        // Redirect the data result and, if present, the planner's
        // !tile.async_token result to the rocm token — so a consuming wait/mma's
        // token operand resolves to this copy's SSA value after lowering.
        op->getResult(0).replaceAllUsesWith(rocmOp->getResult(0));
        if (hasTileToken)
          op->getResult(numResults - 1).replaceAllUsesWith(rocmOp->getResult(0));
        op->erase();
        continue;
      }

      if (name == "tile.wait_async") {
        if (outstanding.empty()) {
          op->emitError("ROCm lowering requires tile.wait_async after tile.async_copy");
          signalPassFailure();
          return;
        }

        // Retire the copy this wait consumes. Prefer the SSA token operand (the
        // planner threaded it, post-lowering it resolves to the copy's rocm
        // token) — a precise def-use retirement. Else fall back to the stamped
        // tile.barrier_id, else the oldest outstanding — never "the last token".
        Value token;
        for (Value operand : op->getOperands()) {
          auto it = llvm::find_if(outstanding, [&](const auto &e) {
            return e.second == operand;
          });
          if (it != outstanding.end()) {
            token = it->second;
            outstanding.erase(it);
            break;
          }
        }
        if (!token) {
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            StringRef want = a.getValue();
            auto it = llvm::find_if(outstanding, [&](const auto &e) {
              return e.first == want;
            });
            if (it != outstanding.end()) {
              token = it->second;
              outstanding.erase(it);
            }
          }
        }
        if (!token) {
          token = outstanding.front().second; // oldest
          outstanding.erase(outstanding.begin());
        }

        OperationState state(op->getLoc(), "tessera_rocm.wait");
        state.addOperands(token);
        // The async copy is global→LDS, which retires on the vector-memory
        // counter — gate on vmcnt rather than draining the wavefront with a full
        // barrier (the decoupled-wait lever: the matrix core keeps issuing while
        // the copy is still in flight).
        StringRef counter = "vmcnt";
        if (auto counterAttr =
                op->getAttrOfType<StringAttr>("tile.wait_counter"))
          counter = counterAttr.getValue();
        state.addAttribute("counter", builder.getStringAttr(counter));
        // Preserve the barrier-id + waitcnt threshold for ROCDL contract
        // lowering (vmcnt(threshold) metadata), so targeted waits stay targeted.
        if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
          state.addAttribute("barrier_id", a);
        if (auto a = op->getAttrOfType<IntegerAttr>("tile.waitcnt_threshold"))
          state.addAttribute("threshold", a);
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
