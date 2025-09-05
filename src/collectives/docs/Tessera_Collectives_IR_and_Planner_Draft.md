# Tessera Collectives IR + Planner Pass (Draft)

**Version:** v1  
**Status:** Draft for design review  
**Deliverables:** TableGen op definitions (sketch), dialect types, and a tiny planner pass outline that (a) slices memrefs into chunked views and (b) inserts `await` only at true use sites.

---

## 1. Dialect Overview

- Dialect: `tessera.collective`
- Key types:
  - `!tessera.future<T>` — asynchronous handle producing `T`.
  - `!tessera.shard<T>` — logical shard view of `T`.
- Conventions:
  - Collectives return **futures**; blocking uses must insert `tessera.await`.
  - Ops are **pure** w.r.t. host (side-effects are on CommQ, modeled by an op interface).

---

## 2. TableGen: Types & Traits (sketch)

```tablegen
//===- TesseraCollectiveBase.td --------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def Tessera_Collective_Dialect : Dialect {
  let name = "tessera.collective";
  let cppNamespace = "tessera::collective";
}

def Tessera_FutureType : TypeDef<Tessera_Collective_Dialect, "Future"> {
  let summary = "Async future type";
  let parameters = (ins Type:$valueType);
  let assemblyFormat = "`<` type($valueType) `>`";
}

def Tessera_ShardType : TypeDef<Tessera_Collective_Dialect, "Shard"> {
  let summary = "Logical shard view type";
  let parameters = (ins Type:$valueType);
  let assemblyFormat = "`<` type($valueType) `>`";
}

// Common attributes
def Attr_Algo     : StrEnumAttr<"Algo", "collective algo", ["auto","ring","tree","hier"]>;
def Attr_Path     : StrEnumAttr<"Path", "transport path", ["auto","nvlink","pcie","rdma"]>;
def Attr_Scope    : StrEnumAttr<"Scope","hierarchical scope",["intra_sm","intra_gpu","node","rack"]>;
def Attr_DType    : StrEnumAttr<"WireDType","wire dtype",["fp32","bf16","fp16","fp8","i8"]>;
def Attr_Overlap  : StrEnumAttr<"Overlap","overlap policy",["compute","none"]>;
def Attr_Priority : StrEnumAttr<"Priority","sched prio",["latency","throughput"]>;

def Attr_ChunkBytes : I64Attr;
def Attr_MaxInflight : I64Attr;
def Attr_QoS : I64Attr; // e.g., token budget
```

---

## 3. TableGen: Ops (sketch)

```tablegen
class TesseraCollectiveOp<string mnemonic, list<OpTrait> traits = []>
  : Op<Tessera_Collective_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 0;
}

// all_reduce
def AllReduceOp : TesseraCollectiveOp<"all_reduce", [AttrSizedOperandSegments]> {
  let summary = "All-reduce with async future result";
  let arguments = (ins AnyMemRef:$input);
  let results   = (outs
    Type<"tessera::collective::FutureType">:$result);

  let assemblyFormat = [{ $input attr-dict }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "tessera.collective.all_reduce"; }
  }];
}

// reduce_scatter
def ReduceScatterOp : TesseraCollectiveOp<"reduce_scatter", []> {
  let summary = "Reduce-scatter producing a shard future";
  let arguments = (ins AnyMemRef:$input);
  let results   = (outs
    Type<"tessera::collective::FutureType">:$result);
}

// all_gather
def AllGatherOp : TesseraCollectiveOp<"all_gather", []> {
  let summary = "All-gather producing a future";
  let arguments = (ins AnyMemRef:$input);
  let results   = (outs
    Type<"tessera::collective::FutureType">:$result);
}

// await
def AwaitOp : Op<Tessera_Collective_Dialect, "await", []> {
  let summary = "Await a future to get materialized value";
  let arguments = (ins
    Type<"tessera::collective::FutureType">:$future);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{ $future attr-dict }];
}
```

**Notes**
- The concrete result type of `await` is derived from the future’s payload (`Future<T> → T`).  
- Wire precision / algo / path / scope appear as optional attrs on collective ops:
  - `algo : #tessera.collective<Algo "auto">`
  - `path : #tessera.collective<Path "auto">`
  - `dtype : #tessera.collective<WireDType "bf16">`
  - `chunk_bytes : i64`  
  - `max_inflight : i64`

---

## 4. Planner Pass: Problem Statement

Given a function containing collective ops that return futures:
1. **Slice** large memrefs into **chunked subviews** (aligned to tiles or byte size).
2. **Replace** a monolithic collective with a **loop of chunked collectives** that each produce a future.
3. **Insert `await`** only at **true use sites** where a non-async consumer needs the materialized value.
4. **Optionally** set per-chunk attributes (`algo`, `path`, `priority`) from a topology-aware LUT.

---

## 5. Planner Pass: C++ Skeleton

```cpp
//===- PlanChunkedCollectives.cpp --------------------------------*- C++ -*-===//
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "TesseraCollectiveDialect.h" // generated headers (types/ops)

using namespace mlir;
using namespace tessera::collective;

namespace {

struct PlanChunkedCollectivesPass
    : public PassWrapper<PlanChunkedCollectivesPass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "tessera-plan-collectives"; }
  StringRef getDescription() const final {
    return "Slice memrefs into chunked views and insert awaits at true use sites";
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    OpBuilder b(func.getContext());

    // Simple walk: find collective ops
    SmallVector<Operation*> worklist;
    func.walk([&](Operation *op) {
      if (isa<AllReduceOp, ReduceScatterOp, AllGatherOp>(op))
        worklist.push_back(op);
    });

    for (Operation *op : worklist) {
      b.setInsertionPoint(op);

      // Read attributes
      auto chunkBytesAttr = op->getAttrOfType<IntegerAttr>("chunk_bytes");
      int64_t chunkBytes = chunkBytesAttr ? chunkBytesAttr.getInt() : (1 << 20);

      // Infer element size and compute chunk length (very simplified)
      Value input = op->getOperand(0);
      auto mrty = dyn_cast<MemRefType>(input.getType());
      if (!mrty) continue;
      int64_t elemBytes = mrty.getElementTypeBitWidth() / 8;
      int64_t elemsPerChunk = std::max<int64_t>(1, chunkBytes / std::max<int64_t>(1, elemBytes));

      // Assume last dimension chunking for demo
      int64_t N = mrty.getShape().back();
      int64_t numChunks = (N + elemsPerChunk - 1) / elemsPerChunk;

      // Build loop over chunks
      Location loc = op->getLoc();
      auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
      auto one  = b.create<arith::ConstantIndexOp>(loc, 1);
      auto ub   = b.create<arith::ConstantIndexOp>(loc, numChunks);

      auto forOp = b.create<scf::ForOp>(loc, zero, ub, one, ValueRange{});
      b.setInsertionPointToStart(forOp.getBody());

      // Subview slice [*, *, i*sz : min(sz, rem)]
      Value iv = forOp.getInductionVar();
      Value chunkSzVal = b.create<arith::ConstantIndexOp>(loc, elemsPerChunk);

      // Compute offset on last dim
      Value off = b.create<arith::MulIOp>(loc, iv, chunkSzVal);
      SmallVector<OpFoldResult> offsets(mrty.getRank(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(mrty.getRank(), b.getIndexAttr(1));
      for (int64_t d = 0; d < mrty.getRank(); ++d) {
        if (d == mrty.getRank() - 1) {
          sizes.push_back(chunkSzVal);
          offsets[d] = off;
        } else {
          sizes.push_back(b.getIndexAttr(mrty.getDimSize(d)));
        }
      }
      Value sub = b.create<memref::SubViewOp>(loc, input, offsets, sizes, strides);

      // Emit a chunked collective op of the same kind on 'sub'
      Operation *chunkOp = nullptr;
      if (auto ar = dyn_cast<AllReduceOp>(op)) {
        chunkOp = b.create<AllReduceOp>(loc, FutureType::get(sub.getType(), b.getContext()), sub);
      } else if (auto rs = dyn_cast<ReduceScatterOp>(op)) {
        chunkOp = b.create<ReduceScatterOp>(loc, FutureType::get(sub.getType(), b.getContext()), sub);
      } else if (auto ag = dyn_cast<AllGatherOp>(op)) {
        chunkOp = b.create<AllGatherOp>(loc, FutureType::get(sub.getType(), b.getContext()), sub);
      }

      // Keep futures in a vector for later wiring (if needed)
      // In practice, you'd collect them and stitch materialization as needed.

      // End loop (No explicit yield payload in this sketch)
      b.setInsertionPointAfter(forOp);

      // Erase original monolithic op
      op->erase();
    }

    // Second pass: Insert await at true use sites
    func.walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        Value v = operand.get();
        if (auto fty = v.getType().dyn_cast<FutureType>()) {
          // If 'op' cannot accept async operand, insert await just before 'op'
          if (!op->hasTrait<OpTrait::tessera::AcceptsAsync>()) {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(op);
            auto await = b.create<AwaitOp>(op->getLoc(), fty.getValueType(), v);
            operand.set(await.getResult());
          }
        }
      }
    });
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> createPlanChunkedCollectivesPass() {
  return std::make_unique<PlanChunkedCollectivesPass>();
}
```

**Notes**
- The above is a **didactic skeleton**: real code would handle dynamic shapes, partial tail chunks, alignment, multi‑dim chunking, attributes propagation, and composition with `scf::Forall` or `gml_st` tiling.
- **True-use detection:** we rely on a trait `AcceptsAsync` for ops that can forward futures; all others trigger an `await` insertion right before the use.
- A production pass would also **coalesce** adjacent small chunks and emit **max_inflight** throttling via an attribute or via an explicit `tessera.qos.reserve` op.

---

## 6. Example IR Before/After

### Before
```mlir
%f = tessera.collective.all_reduce %dw
        {op="sum", chunk_bytes = 1048576, dtype="bf16"}
; ... unrelated compute ...
%dw_red = tessera.await %f
use %dw_red
```

### After (conceptual)
```mlir
scf.for %i = 0 to %numChunks step 1 {
  %sub = memref.subview %dw [..., %i*sz] [ ..., %sz ] [ ..., 1 ]
  %f_i = tessera.collective.all_reduce %sub {dtype="bf16"}
  ; optionally enqueue for overlap; await only at true consumer
}

; ... unrelated compute ...

%dw_red = tessera.await %f_j  // only where required
use %dw_red
```

---

## 7. Lit Test Sketch

```mlir
// RUN: tessera-opt %s -tessera-plan-collectives | FileCheck %s

// CHECK: scf.for
// CHECK: memref.subview
// CHECK: tessera.collective.all_reduce
// CHECK: tessera.await
```

---

## 8. Next Steps

- Add `tessera.collective` ODS files and generated headers into the repo.
- Implement `AcceptsAsync` op trait for key compute ops in Tessera.
- Flesh out planner with multi-dim tiles and `max_inflight` throttling.
- NCCL/RCCL adapters that accept **chunked submissions** + completion callbacks.
```

