// StreamingAttentionToAppleGPU.cpp — APPLE-ATTN-STREAM-1 (2026-07-26).
//
// Apple consumption of the shared streaming-attention contract
// (`CORE-STREAMING-ATTN-2026-07-26`). The shared lowering now states rank-2
// FlashAttention as a KV-block `scf.for` carrying the FP32 output accumulator,
// the running maximum and normalization sum, producer/consumer
// `!tile.pipeline_state`, and an absolute boundary offset, with
// `tessera_attn.boundary_mask` owning causal/sliding-window state,
// `tessera_attn.block_dropout` owning offset-keyed dropout, and
// `tessera_attn.streaming_update` consuming both scores and V.
//
// Before this pass, `FlashAttnToAppleGPU` rewrote `tessera.flash_attn` straight
// from Graph IR to a monolithic runtime ABI call, so Apple attention never
// crossed the Tile layer and re-derived its own causal/window semantics. That
// made the boundary rules a second, independent implementation of a contract
// the shared ops already own.
//
// Two architecture-owned facts drive the design:
//
//  1. Metal has no asynchronous copy engine and stages through a threadgroup
//     ping-pong pair, so the shared KV ring's depth-3 producer/consumer state
//     has no Metal realization (APPLE-PIPE-1 rejects depth > 2). Apple must
//     therefore *re-form* the recurrence, exactly as it re-forms the canonical
//     GEMM reduction — the loop is a semantic contract, not a schedule.
//  2. Apple already owns online-softmax flash-attention MSL kernels that
//     implement precisely this recurrence. The work here is routing and
//     ownership, not a new kernel.
//
// So this pass recognizes the marked streaming loop and re-forms it as one
// Apple flash-attention dispatch, carrying the boundary semantics *read off the
// shared ops* rather than re-derived. Recognition is not promotion: the
// numerical contract is proven against the incumbent ABI route before any
// selector changes.

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera::apple {
namespace {

// Apple's fused-chain score cap, derived from the 1 KiB per-thread fp32 stack
// budget (`apple_target.apple_fused_chain_score_cap`). Head dims beyond it have
// no Apple fused route.
constexpr int64_t kAppleHeadDimCap = 256;

// Walk back through the movement ops the shared lowering inserts (staging
// copies and the ragged zero-fill slice pair) to the tensor the loop actually
// reads. Returns null when the chain leaves the recognized vocabulary, which
// means the operand is computed per step rather than being a whole K/V/Q.
Value traceToSource(Value value, Operation *root) {
  unsigned guard = 0;
  while (guard++ < 16) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return value; // Block argument of an enclosing function: loop-invariant.
    if (!root->isProperAncestor(def) && def->getBlock() != root->getBlock())
      return value;
    StringRef name = def->getName().getStringRef();
    if (name == "tile.async_copy" || name == "tensor.extract_slice" ||
        name == "tensor.insert_slice") {
      // insert_slice's first operand is the inserted source; extract_slice's
      // and async_copy's is the value being read.
      value = def->getOperand(0);
      continue;
    }
    return root->isProperAncestor(def) ? Value() : value;
  }
  return Value();
}

// The staging the shared lowering hoists ahead of the KV loop — the Q copy and
// the producer/consumer ring initialisation. None of it is `Pure`, so once the
// recurrence is re-formed these ops linger as debris that no DCE will collect.
//
// Leaving them is not merely untidy: the ring the shared attention path builds
// is depth 3, which APPLE-PIPE-1 rejects because Metal stages through a
// ping-pong pair. A dead depth-3 `tile.pipeline_init` left in the function
// would fail `APPLE_STAGE_DEPTH_UNSUPPORTED` the moment both passes run in one
// pipeline — a rejection describing a schedule the program no longer has.
bool isStagingOp(StringRef name) {
  return name == "tile.async_copy" || name == "tile.wait_async" ||
         name == "tile.pipeline_init" || name == "tile.pipeline_advance";
}

void eraseDeadStaging(Block *block) {
  // Consumers before producers (wait_async reads the copy's token), so iterate
  // to a fixpoint rather than assuming one reverse sweep suffices.
  bool changed = true;
  unsigned guard = 0;
  while (changed && guard++ < 8) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
      if (!isStagingOp(op.getName().getStringRef()))
        continue;
      if (!op.use_empty())
        continue;
      op.erase();
      changed = true;
    }
  }
}

bool hasAttentionDistribution(Operation *op, StringRef distribution) {
  if (!op || op->getName().getStringRef() != "scf.for")
    return false;
  auto attr = op->getAttrOfType<StringAttr>("tessera.attention_distribution");
  return attr && attr.getValue() == distribution;
}

bool isStaticRank4(Value value) {
  auto type = llvm::dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 4 && type.hasStaticShape();
}

bool hasLiveLse(Operation *root) {
  bool live = false;
  root->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tessera_attn.lse_accumulate" &&
        op->getNumResults() > 1 && !op->getResult(1).use_empty())
      live = true;
  });
  return live;
}

struct StreamingAttentionToAppleGPUPass
    : public PassWrapper<StreamingAttentionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamingAttentionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-apple-streaming-attention";
  }
  StringRef getDescription() const override {
    return "APPLE-ATTN-STREAM-1 — recognize the shared KV-block streaming "
           "attention recurrence and re-form it as one Apple flash-attention "
           "dispatch, carrying the shared boundary semantics.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TesseraAppleDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> loops;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "scf.for" &&
          op->hasAttr("tessera.streaming_attention"))
        loops.push_back(op);
    });

    for (Operation *loop : loops) {
      Operation *score = nullptr;
      Operation *mask = nullptr;
      Operation *update = nullptr;
      Operation *dropout = nullptr;
      loop->walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        if (name == "tessera_attn.scaled_dot_product")
          score = op;
        else if (name == "tessera_attn.boundary_mask")
          mask = op;
        else if (name == "tessera_attn.streaming_update")
          update = op;
        else if (name == "tessera_attn.block_dropout")
          dropout = op;
      });
      if (!score || !update) {
        loop->emitOpError(
            "APPLE_STREAMING_ATTN_UNRECOGNIZED: a streaming-attention loop "
            "must contain tessera_attn.scaled_dot_product and "
            "tessera_attn.streaming_update");
        signalPassFailure();
        return;
      }
      if (dropout) {
        // Offset-keyed block dropout is part of the shared contract but has no
        // Apple fused route; claiming the loop would silently drop the mask.
        loop->emitOpError(
            "APPLE_STREAMING_ATTN_DROPOUT_UNSUPPORTED: apple_gpu has no fused "
            "offset-keyed block-dropout attention route");
        signalPassFailure();
        return;
      }

      Value q = traceToSource(score->getOperand(0), loop);
      Value k = traceToSource(score->getOperand(1), loop);
      Value v = traceToSource(update->getOperand(1), loop);
      if (!q || !k || !v) {
        loop->emitOpError(
            "APPLE_STREAMING_ATTN_UNRECOGNIZED: Q/K/V do not trace to "
            "loop-invariant tensors, so the recurrence is not a single "
            "attention Apple can re-form");
        signalPassFailure();
        return;
      }

      // APPLE-ATTN-STREAM-2: a rank-4 shared attention lowering is not a
      // rank-4 KV loop. The marked loop still carries one [Sq, D] slice; the
      // enclosing query-head and batch loops own distribution and assemble the
      // [B, Hq, Sq, D] result. Claiming the inner loop would replace one slice
      // with a rank-4 call and is therefore structurally invalid.
      Operation *headLoop = loop->getParentOp();
      Operation *batchLoop = headLoop ? headLoop->getParentOp() : nullptr;
      if (hasAttentionDistribution(headLoop, "query_head") &&
          hasAttentionDistribution(batchLoop, "batch")) {
        if (!isStaticRank4(q) || !isStaticRank4(k) || !isStaticRank4(v) ||
            batchLoop->getNumResults() != 1 ||
            !isStaticRank4(batchLoop->getResult(0))) {
          batchLoop->emitOpError(
              "APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED: rank-4 Apple GQA "
              "requires static [B,H,S,D] Q/K/V and one rank-4 batch result");
          signalPassFailure();
          return;
        }
        auto qType = llvm::cast<RankedTensorType>(q.getType());
        auto kType = llvm::cast<RankedTensorType>(k.getType());
        auto vType = llvm::cast<RankedTensorType>(v.getType());
        auto outType = llvm::cast<RankedTensorType>(batchLoop->getResult(0).getType());
        const int64_t batch = qType.getDimSize(0);
        const int64_t qHeads = qType.getDimSize(1);
        const int64_t sq = qType.getDimSize(2);
        const int64_t headDim = qType.getDimSize(3);
        const int64_t kvHeads = kType.getDimSize(1);
        const int64_t sk = kType.getDimSize(2);
        if (qType.getElementType() != Float32Type::get(&getContext()) ||
            kType.getElementType() != qType.getElementType() ||
            vType.getElementType() != qType.getElementType() ||
            outType.getElementType() != qType.getElementType()) {
          batchLoop->emitOpError(
              "APPLE_STREAMING_ATTN_DTYPE_UNSUPPORTED: rank-4 Apple GQA "
              "currently requires f32 Q/K/V and f32 output");
          signalPassFailure();
          return;
        }
        if (kType.getDimSize(0) != batch || vType.getDimSize(0) != batch ||
            vType.getDimSize(1) != kvHeads || vType.getDimSize(2) != sk ||
            kType.getDimSize(3) != headDim || vType.getDimSize(3) != headDim ||
            outType.getShape() != qType.getShape() || qHeads <= 0 ||
            kvHeads <= 0 || qHeads % kvHeads != 0) {
          batchLoop->emitOpError(
              "APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED: rank-4 Apple GQA "
              "requires matching batch/head dimensions and an integral Q/KV "
              "head ratio");
          signalPassFailure();
          return;
        }
        if (headDim > kAppleHeadDimCap) {
          batchLoop->emitOpError("APPLE_STREAMING_ATTN_HEAD_DIM_UNSUPPORTED: "
                                 "rank-4 head_dim ")
              << headDim << " exceeds the Apple fused-chain score cap of "
              << kAppleHeadDimCap;
          signalPassFailure();
          return;
        }
        if (hasLiveLse(batchLoop)) {
          batchLoop->emitOpError(
              "APPLE_STREAMING_ATTN_LSE_UNSUPPORTED: rank-4 Apple GQA "
              "returns attention output only, but this distribution retains "
              "a live per-row log-sum-exp");
          signalPassFailure();
          return;
        }

        // The established GQA ABI carries causal semantics and scale, but not
        // sliding windows, score bias, or softcap. Refuse those modifiers at
        // this boundary rather than dropping shared semantics.
        const auto hasNonDefaultWindow = [&](StringRef name) {
          auto attr = mask ? mask->getAttrOfType<IntegerAttr>(name) : nullptr;
          return attr && attr.getInt() != -1;
        };
        if (hasNonDefaultWindow("window_left") ||
            hasNonDefaultWindow("window_right") ||
            (score && (score->hasAttr("bias") || score->hasAttr("softcap")))) {
          batchLoop->emitOpError(
              "APPLE_STREAMING_ATTN_RANK4_MODIFIER_UNSUPPORTED: rank-4 "
              "Apple GQA currently supports causal and scale only; window, "
              "bias, and softcap remain on the shared decomposed path");
          signalPassFailure();
          return;
        }

        OpBuilder builder(batchLoop);
        OperationState state(batchLoop->getLoc(),
                             "tessera_apple.gpu.kernel_call");
        state.addOperands({q, k, v});
        state.addTypes({batchLoop->getResult(0).getType()});
        state.addAttribute("op_kind", builder.getStringAttr("flash_attn_gqa"));
        state.addAttribute("symbol", builder.getStringAttr(
                                       "tessera_apple_gpu_flash_attn_gqa_f32"));
        state.addAttribute("abi", builder.getStringAttr("msl_gqa"));
        state.addAttribute("status", builder.getStringAttr("executable"));
        state.addAttribute("framework", builder.getStringAttr("Metal"));
        state.addAttribute("dtype", builder.getStringAttr("f32"));
        state.addAttribute("tessera_apple.streaming_recurrence",
                           builder.getBoolAttr(true));
        state.addAttribute("tessera_apple.rank4_distribution",
                           builder.getBoolAttr(true));
        state.addAttribute("tessera_apple.batch", builder.getI64IntegerAttr(batch));
        state.addAttribute("tessera_apple.q_heads", builder.getI64IntegerAttr(qHeads));
        state.addAttribute("tessera_apple.kv_heads", builder.getI64IntegerAttr(kvHeads));
        state.addAttribute("tessera_apple.sq", builder.getI64IntegerAttr(sq));
        state.addAttribute("tessera_apple.sk", builder.getI64IntegerAttr(sk));
        state.addAttribute("tessera_apple.head_dim",
                           builder.getI64IntegerAttr(headDim));
        state.addAttribute("tessera_apple.gqa_group_size",
                           builder.getI64IntegerAttr(qHeads / kvHeads));
        if (auto scale = score->getAttr("scale"))
          state.addAttribute("tessera_apple.scale", scale);
        if (mask) {
          for (StringRef attr : {"causal", "logical_sk"})
            if (Attribute value = mask->getAttr(attr))
              state.addAttribute(("tessera_apple." + attr).str(), value);
        } else {
          state.addAttribute("tessera_apple.causal", builder.getBoolAttr(false));
        }
        if (auto block = loop->getAttrOfType<IntegerAttr>("tessera.kv_block"))
          state.addAttribute("tessera_apple.kv_block", block);

        Operation *call = builder.create(state);
        batchLoop->getResult(0).replaceAllUsesWith(call->getResult(0));
        Block *block = batchLoop->getBlock();
        batchLoop->erase();
        eraseDeadStaging(block);
        continue;
      }

      auto qType = llvm::dyn_cast<RankedTensorType>(q.getType());
      if (!qType || qType.getRank() != 2 || !qType.hasStaticShape()) {
        loop->emitOpError("APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED: the Apple "
                          "fused route requires a static rank-2 Q");
        signalPassFailure();
        return;
      }
      int64_t headDim = qType.getDimSize(1);
      if (headDim > kAppleHeadDimCap) {
        loop->emitOpError("APPLE_STREAMING_ATTN_HEAD_DIM_UNSUPPORTED: head_dim ")
            << headDim << " exceeds the Apple fused-chain score cap of "
            << kAppleHeadDimCap;
        signalPassFailure();
        return;
      }

      Type storage = qType.getElementType();
      StringRef dtype;
      if (storage.isF32())
        dtype = "f32";
      else if (storage.isF16())
        dtype = "f16";
      else if (storage.isBF16())
        dtype = "bf16";
      else {
        loop->emitOpError(
            "APPLE_STREAMING_ATTN_DTYPE_UNSUPPORTED: the Apple fused "
            "flash-attention route accepts f32/f16/bf16 storage");
        signalPassFailure();
        return;
      }

      // The loop's own result is consumed by tessera_attn.lse_accumulate, whose
      // first result is the attention output and whose second is the per-row
      // log-sum-exp.
      Operation *consumer = nullptr;
      for (Operation *user : loop->getResult(0).getUsers())
        if (user->getName().getStringRef() == "tessera_attn.lse_accumulate")
          consumer = user;
      // Apple's fused flash-attention ABI returns the output only, so a live
      // per-row LSE means we cannot re-form: replacing just the output would
      // leave the whole recurrence alive to recompute the LSE, and the program
      // would do the attention twice.
      //
      // Any LSE use is now real: lse.save has an explicit destination,
      // identity, scope, and MemWrite effect. Apple must reject a package that
      // requests it until the Metal ABI grows a checkpoint output.
      if (consumer) {
        bool liveLse = !consumer->getResult(1).use_empty();
        if (liveLse) {
          loop->emitOpError(
              "APPLE_STREAMING_ATTN_LSE_UNSUPPORTED: the Apple fused route "
              "returns the attention output only, but this program also "
              "consumes the per-row log-sum-exp; re-forming would recompute "
              "the whole recurrence");
          signalPassFailure();
          return;
        }
      }
      Value replaced = consumer ? consumer->getResult(0) : loop->getResult(0);

      OpBuilder builder(consumer ? consumer : loop);
      OperationState state(loop->getLoc(), "tessera_apple.gpu.kernel_call");
      state.addOperands({q, k, v});
      state.addTypes({replaced.getType()});
      state.addAttribute("op_kind", builder.getStringAttr("flash_attn"));
      state.addAttribute(
          "symbol", builder.getStringAttr(
                        ("tessera_apple_gpu_flash_attn_" + dtype).str()));
      state.addAttribute("abi", builder.getStringAttr("msl"));
      state.addAttribute("status", builder.getStringAttr("executable"));
      state.addAttribute("framework", builder.getStringAttr("Metal"));
      state.addAttribute("dtype", builder.getStringAttr(dtype));
      state.addAttribute("tessera_apple.streaming_recurrence",
                         builder.getBoolAttr(true));
      state.addAttribute("tessera_apple.head_dim",
                         builder.getI64IntegerAttr(headDim));
      // Boundary semantics are READ OFF the shared op, never re-derived. This
      // is the whole point of routing Apple attention through the Tile layer:
      // causal/window/logical-length rules get exactly one owner.
      if (mask) {
        for (StringRef attr :
             {"causal", "logical_sk", "window_left", "window_right"})
          if (Attribute value = mask->getAttr(attr))
            state.addAttribute(("tessera_apple." + attr).str(), value);
      } else {
        state.addAttribute("tessera_apple.causal", builder.getBoolAttr(false));
      }
      // The KV blocking the shared loop chose is carried for the emitter.
      if (auto block = loop->getAttrOfType<IntegerAttr>("tessera.kv_block"))
        state.addAttribute("tessera_apple.kv_block", block);

      Operation *call = builder.create(state);
      replaced.replaceAllUsesWith(call->getResult(0));
      // The recurrence is now dead: erase it rather than leaving a second,
      // unreachable computation of the same attention in the module. Order
      // matters — the dead saves read the accumulator, so they go first.
      if (consumer)
        consumer->erase();
      Block *block = loop->getBlock();
      loop->erase();
      eraseDeadStaging(block);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createStreamingAttentionToAppleGPUPass() {
  return std::make_unique<StreamingAttentionToAppleGPUPass>();
}

} // namespace tessera::apple
