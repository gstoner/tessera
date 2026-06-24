//===- NVTMADescriptorPass.cpp — Phase 3 ─────────────────────────────────===//
//
// Hoists TMA descriptor setup to the kernel preamble and assigns unique
// mbarrier slots.
//
// Design contract (CLAUDE.md §Phase 3 Design Contract 2):
//   "TMA descriptors are generated once per kernel, not per tile."
//   This pass implements that contract by collecting all tile.tma.descriptor
//   ops, deduplicating them by (src_ptr, tile_rows, tile_cols), and hoisting
//   a single cuTensorMapEncode call to the function entry block.
//
// Input:
//   func.func @flash_attn_kernel(...) {
//     ...
//     %desc_q = tile.tma.descriptor(%Q) {tile_rows=64, tile_cols=64}
//     tile.tma.copy_async %desc_q {mbarrier_slot=0}
//     %desc_q2 = tile.tma.descriptor(%Q) {tile_rows=64, tile_cols=64}  // dup
//     tile.tma.copy_async %desc_q2 {mbarrier_slot=0}
//   }
//
// Output:
//   func.func @flash_attn_kernel(%Q_desc: i64, ...) {
//     // Preamble: mbarrier init for each unique descriptor slot.
//     tile.mbarrier.init {slots=2, phase_bits=2}
//     ...
//     // Inner loop: just the copy_async referencing the hoisted descriptor.
//     tile.tma.copy_async %Q_desc {mbarrier_slot=0, expect_tx=8192}
//   }
//
// Registration: --tessera-nvtma-descriptor
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace tessera {

namespace {

// C6 join (2026-06-23): stamp a typed #tile.barrier (kind = tma, transaction
// byte-count `expect`) + a per-slot tile.barrier_id on a TMA barrier site. The
// setup_descriptor (init) and copy_async (arrive) for one slot carry the SAME
// (kind, expect, id), so TilePipelineLegality (C3) verifies kind consistency and
// WarpSpecLegality (C6) verifies arrival-count == init-count, both live on real
// lowering output.
static void stampTmaBarrier(OpBuilder &b, Operation *op, int64_t slot,
                            int64_t expectTx) {
  if (expectTx < 0)
    return; // #tile.barrier verifier requires expect >= 0.
  op->setAttr("tile.barrier_id",
              b.getStringAttr(("mbar." + Twine(slot)).str()));
  op->setAttr("tile.barrier",
              tile::TileBarrierAttr::get(b.getContext(), "tma", expectTx));
}

// Key for descriptor deduplication: (source SSA value, tile_rows, tile_cols).
struct DescriptorKey {
  Value src;
  int64_t tileRows;
  int64_t tileCols;

  bool operator==(const DescriptorKey &o) const {
    return src == o.src &&
           tileRows == o.tileRows &&
           tileCols == o.tileCols;
  }
};

struct DescriptorKeyInfo : public llvm::DenseMapInfo<DescriptorKey> {
  static DescriptorKey getEmptyKey() {
    return {Value(), -1, -1};
  }
  static DescriptorKey getTombstoneKey() {
    return {Value(), -2, -2};
  }
  static unsigned getHashValue(const DescriptorKey &k) {
    return llvm::hash_combine(
        llvm::hash_value(k.src.getAsOpaquePointer()),
        llvm::hash_value(k.tileRows),
        llvm::hash_value(k.tileCols));
  }
  static bool isEqual(const DescriptorKey &a, const DescriptorKey &b) {
    return a == b;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct NVTMADescriptorPass
    : public PassWrapper<NVTMADescriptorPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVTMADescriptorPass)

  StringRef getArgument() const override { return "tessera-nvtma-descriptor"; }
  StringRef getDescription() const override {
    return "Hoist TMA descriptors to kernel preamble; assign mbarrier slots";
  }
  // C6 join: the pass now constructs #tile.barrier, so the Tile dialect must be
  // loaded.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    OpBuilder b(ctx);

    // Collect all tile.tma.descriptor ops in program order.
    SmallVector<Operation *> descOps;
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tile.tma.descriptor")
        descOps.push_back(op);
    });

    if (descOps.empty())
      return;

    // Deduplicate by key → canonical descriptor value.
    llvm::DenseMap<DescriptorKey, Value, DescriptorKeyInfo> canonMap;
    llvm::DenseMap<DescriptorKey, int64_t, DescriptorKeyInfo> slotMap;
    llvm::DenseMap<Value, int64_t> descriptorSlotMap;
    int64_t nextSlot = 0;

    // Find insertion point: just before the first non-argument instruction.
    Block &entryBlock = funcOp.getBody().front();
    b.setInsertionPointToStart(&entryBlock);

    // Emit mbarrier.init once for all unique descriptors.
    // (Actual slot count filled in after dedup.)
    Operation *mbarrierInitPlaceholder = nullptr;
    {
      OperationState st(funcOp.getLoc(), "tile.mbarrier.init");
      st.addAttribute("slots", b.getI64IntegerAttr(0)); // placeholder
      st.addAttribute("phase_bits", b.getI64IntegerAttr(2));
      mbarrierInitPlaceholder = b.create(st);
    }

    // Process each descriptor op.
    for (Operation *desc : descOps) {
      if (desc->getNumOperands() < 1)
        continue;
      Value src = desc->getOperand(0);
      int64_t tileRows = 64, tileCols = 64;
      if (auto a = desc->getAttrOfType<IntegerAttr>("tile_rows"))
        tileRows = a.getInt();
      if (auto a = desc->getAttrOfType<IntegerAttr>("tile_cols"))
        tileCols = a.getInt();

      DescriptorKey key{src, tileRows, tileCols};
      auto it = canonMap.find(key);
      if (it == canonMap.end()) {
        // Hoist a new descriptor setup to preamble.
        b.setInsertionPoint(mbarrierInitPlaceholder);
        OperationState st(desc->getLoc(), "tile.tma.setup_descriptor");
        st.addOperands({src});
        st.addAttribute("tile_rows", b.getI64IntegerAttr(tileRows));
        st.addAttribute("tile_cols", b.getI64IntegerAttr(tileCols));
        st.addAttribute("mbarrier_slot", b.getI64IntegerAttr(nextSlot));
        // expect_tx = tile_rows * tile_cols * sizeof(bf16)
        int64_t expectTx = tileRows * tileCols * 2;
        st.addAttribute("expect_tx", b.getI64IntegerAttr(expectTx));
        st.addTypes(b.getIntegerType(64));
        Operation *setup = b.create(st);
        // C6: the per-slot setup is this barrier's init site (declares the
        // expected transaction count).
        if (tileRows > 0 && tileCols > 0)
          stampTmaBarrier(b, setup, nextSlot, expectTx);
        canonMap[key] = setup->getResult(0);
        slotMap[key] = nextSlot++;
        descriptorSlotMap[setup->getResult(0)] = slotMap[key];
        desc->replaceAllUsesWith(setup->getResults());
        desc->erase();
      } else {
        // Replace duplicate with the canonical value.
        descriptorSlotMap[it->second] = slotMap[key];
        desc->replaceAllUsesWith(ValueRange{it->second});
        desc->erase();
      }
    }

    // Update mbarrier.init slot count now that we know the real number.
    if (mbarrierInitPlaceholder) {
      mbarrierInitPlaceholder->setAttr(
          "slots", b.getI64IntegerAttr(nextSlot));
    }

    // Update all tile.tma.copy_async ops with their correct mbarrier slot.
    // (Slot 0 is the default; the setup ops assigned sequential slots above.)
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tile.tma.copy_async") {
        if (op->getNumOperands() == 0)
          return;
        Value descriptor = op->getOperand(0);
        auto slotIt = descriptorSlotMap.find(descriptor);
        if (slotIt == descriptorSlotMap.end())
          return;
        op->setAttr("mbarrier_slot", b.getI64IntegerAttr(slotIt->second));
        if (auto setup = descriptor.getDefiningOp()) {
          if (auto expectTx = setup->getAttrOfType<IntegerAttr>("expect_tx")) {
            op->setAttr("expect_tx", expectTx);
            // C6: the copy_async is this barrier's arrive site — same (kind,
            // expect, id) as the init, so arrival-count == init-count.
            stampTmaBarrier(b, op, slotIt->second, expectTx.getInt());
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createNVTMADescriptorPass() {
  return std::make_unique<NVTMADescriptorPass>();
}

} // namespace tessera
