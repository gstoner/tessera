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
    SmallVector<tile::TMADescriptorOp> descOps;
    funcOp.walk(
        [&](tile::TMADescriptorOp op) { descOps.push_back(op); });

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
    tile::MBarrierInitOp mbarrierInitPlaceholder;
    {
      OperationState st(funcOp.getLoc(),
                        tile::MBarrierInitOp::getOperationName());
      st.addAttribute("slots", b.getI64IntegerAttr(1)); // valid placeholder
      st.addAttribute("phase_bits", b.getI64IntegerAttr(2));
      st.addTypes(tile::MBarrierType::get(ctx));
      mbarrierInitPlaceholder =
          cast<tile::MBarrierInitOp>(b.create(st));
    }

    // Process each descriptor op.
    Operation *lastPreamble = mbarrierInitPlaceholder;
    for (tile::TMADescriptorOp desc : descOps) {
      Value src = desc.getSource();
      int64_t tileRows = desc.getTileRows();
      int64_t tileCols = desc.getTileCols();
      if (auto blockArg = dyn_cast<BlockArgument>(src);
          !blockArg || blockArg.getOwner() != &entryBlock) {
        desc.emitError(
            "TMA descriptor hoisting requires a kernel entry-block argument "
            "source; computed descriptors must be materialized before this pass");
        signalPassFailure();
        return;
      }

      DescriptorKey key{src, tileRows, tileCols};
      auto it = canonMap.find(key);
      if (it == canonMap.end()) {
        // Hoist one canonical typed descriptor to the preamble.
        b.setInsertionPointAfter(lastPreamble);
        OperationState st(desc.getLoc(),
                          tile::TMADescriptorOp::getOperationName());
        st.addOperands({src});
        st.addAttribute("tile_rows", b.getI64IntegerAttr(tileRows));
        st.addAttribute("tile_cols", b.getI64IntegerAttr(tileCols));
        if (auto shape =
                desc->getAttrOfType<DenseI64ArrayAttr>("source_shape"))
          st.addAttribute("source_shape", shape);
        st.addAttribute("slot", b.getI64IntegerAttr(nextSlot));
        int64_t expectTx = tileRows * tileCols * 2;
        st.addAttribute("expect_tx", b.getI64IntegerAttr(expectTx));
        st.addTypes(tile::TMADescriptorType::get(ctx));
        auto setup = cast<tile::TMADescriptorOp>(b.create(st));
        lastPreamble = setup;
        stampTmaBarrier(b, setup, nextSlot, expectTx);
        canonMap[key] = setup.getDescriptor();
        slotMap[key] = nextSlot++;
        descriptorSlotMap[setup.getDescriptor()] = slotMap[key];
        desc->replaceAllUsesWith(setup->getResults());
        desc.erase();
      } else {
        // Replace duplicate with the canonical value.
        descriptorSlotMap[it->second] = slotMap[key];
        desc->replaceAllUsesWith(ValueRange{it->second});
        desc.erase();
      }
    }

    // Update mbarrier.init slot count now that we know the real number.
    if (mbarrierInitPlaceholder) {
      mbarrierInitPlaceholder->setAttr(
          "slots", b.getI64IntegerAttr(nextSlot));
    }

    // Rebuild copies with an explicit SSA mbarrier operand. The descriptor,
    // barrier, async token, and staged value now form one def-use chain.
    SmallVector<tile::TMACopyAsyncOp> copies;
    funcOp.walk([&](tile::TMACopyAsyncOp op) { copies.push_back(op); });
    SmallVector<Value> completionTokens;
    for (tile::TMACopyAsyncOp copy : copies) {
      Value descriptor = copy.getDescriptor();
      auto slotIt = descriptorSlotMap.find(descriptor);
      if (slotIt == descriptorSlotMap.end()) {
        copy.emitError("references a descriptor not owned by this kernel");
        signalPassFailure();
        return;
      }
      auto setup = descriptor.getDefiningOp<tile::TMADescriptorOp>();
      int64_t expectTx = setup.getExpectTx().value_or(0);
      SmallVector<Value> operands = {
          descriptor, mbarrierInitPlaceholder.getBarrier()};
      operands.append(copy.getDependencies().begin(),
                      copy.getDependencies().end());
      SmallVector<NamedAttribute> attrs;
      // tile.pending_mbarrier is AsyncCopyLowering's "barrier retrofit
      // pending" declaration; the rebuilt copy binds the real SSA barrier, so
      // the marker is resolved here.
      for (NamedAttribute attr : copy->getAttrs())
        if (attr.getName() != "operandSegmentSizes" &&
            attr.getName() != "mbarrier_slot" &&
            attr.getName() != "expect_tx" &&
            attr.getName() != "tile.pending_mbarrier")
          attrs.push_back(attr);
      attrs.push_back(
          b.getNamedAttr("mbarrier_slot",
                         b.getI64IntegerAttr(slotIt->second)));
      attrs.push_back(
          b.getNamedAttr("expect_tx", b.getI64IntegerAttr(expectTx)));
      attrs.push_back(b.getNamedAttr(
          "operandSegmentSizes",
          b.getDenseI32ArrayAttr(
              {1, 1, static_cast<int32_t>(copy.getDependencies().size())})));
      b.setInsertionPoint(copy);
      OperationState st(copy.getLoc(),
                        tile::TMACopyAsyncOp::getOperationName());
      st.addOperands(operands);
      st.addTypes(copy->getResultTypes());
      st.addAttributes(attrs);
      Operation *replacement = b.create(st);
      stampTmaBarrier(b, replacement, slotIt->second, expectTx);
      for (Value result : replacement->getResults())
        if (isa<tile::AsyncTokenType>(result.getType()))
          completionTokens.push_back(result);
      copy->replaceAllUsesWith(replacement->getResults());
      copy.erase();
    }

    SmallVector<tile::MBarrierWaitOp> waits;
    funcOp.walk([&](tile::MBarrierWaitOp op) { waits.push_back(op); });
    for (tile::MBarrierWaitOp wait : waits) {
      if (wait.getBarrier())
        continue;
      SmallVector<Value> operands = {mbarrierInitPlaceholder.getBarrier()};
      SmallVector<Value> dependencies(wait.getDependencies().begin(),
                                      wait.getDependencies().end());
      if (dependencies.empty())
        dependencies.append(completionTokens.begin(), completionTokens.end());
      operands.append(dependencies);
      SmallVector<NamedAttribute> attrs;
      // Drop tile.retire_all ONLY when the marker's "everything outstanding"
      // meaning was actually realized as a concrete completion-token set. If
      // the copies in this kernel returned no tokens, the wait still means
      // "retire everything" and the marker must survive — stripping it here
      // would build a barrier-only wait that MBarrierWaitOp::verify()
      // correctly rejects as TILE_WAIT_GATES_ON_NOTHING.
      const bool syncRealized = !dependencies.empty();
      for (NamedAttribute attr : wait->getAttrs())
        if (attr.getName() != "operandSegmentSizes" &&
            !(syncRealized && attr.getName() == "tile.retire_all"))
          attrs.push_back(attr);
      attrs.push_back(b.getNamedAttr(
          "operandSegmentSizes",
          b.getDenseI32ArrayAttr(
              {1, 0, static_cast<int32_t>(dependencies.size())})));
      b.setInsertionPoint(wait);
      OperationState st(wait.getLoc(),
                        tile::MBarrierWaitOp::getOperationName());
      st.addOperands(operands);
      st.addAttributes(attrs);
      b.create(st);
      wait.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createNVTMADescriptorPass() {
  return std::make_unique<NVTMADescriptorPass>();
}

} // namespace tessera
