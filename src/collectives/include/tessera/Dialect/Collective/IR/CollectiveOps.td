include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tessera/Dialect/Collective/IR/CollectiveTypes.td"

class TesseraCollectiveOp<string m, list<Trait> T = []>
    : Op<Tessera_Collective_Dialect, m, T>;

//===----------------------------------------------------------------------===//
// Canonical target-neutral collective dispatches
//===----------------------------------------------------------------------===//

class TargetCollectiveOp<string mnemonic, string summaryText>
    : TesseraCollectiveOp<mnemonic,
        [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = summaryText;
  let description = [{
    This is the portable Target-IR dispatch boundary. `mesh_axis` selects the
    process-group dimension and `tensor_axis` selects the payload dimension.
    `world_size`, when present, makes static gather/scatter shape verification
    exact. The result is asynchronous and must be consumed by
    `tessera_collective.await` before entering ordinary tensor computation.
  }];
  let arguments = (ins AnyType:$input,
                       StrAttr:$mesh_axis,
                       I64Attr:$tensor_axis,
                       DefaultValuedAttr<StrAttr, "\"none\"">:$reduction,
                       OptionalAttr<I64Attr>:$world_size,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def AllReduceOp : TargetCollectiveOp<
    "all_reduce", "Reduce a tensor across a named mesh axis">;
def ReduceScatterOp : TargetCollectiveOp<
    "reduce_scatter", "Reduce then scatter a tensor across ranks">;
def AllGatherOp : TargetCollectiveOp<
    "all_gather", "Gather rank-local tensor shards onto every rank">;
def AllToAllOp : TargetCollectiveOp<
    "all_to_all", "Exchange equal tensor partitions across ranks">;

//===----------------------------------------------------------------------===//
// Rank-local registered-window and one-sided Target operations
//===----------------------------------------------------------------------===//

def WindowRegisterOp : TesseraCollectiveOp<"window.register",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Register a rank-local buffer as a symmetric transport window";
  let arguments = (ins AnyMemRef:$buffer, StrAttr:$window_id, I64Attr:$bytes,
                       DefaultValuedAttr<BoolAttr, "true">:$strict_ordering);
  let results = (outs Tessera_WindowType:$window);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer) `->` type($window)";
  let hasVerifier = 1;
}

def WindowDeregisterOp : TesseraCollectiveOp<"window.deregister",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Deregister a symmetric transport window";
  let arguments = (ins Tessera_WindowType:$window);
  let results = (outs);
  let assemblyFormat = "$window attr-dict `:` type($window)";
}

def PutSignalOp : TesseraCollectiveOp<"put_signal",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Put into a peer registered window and signal completion";
  let arguments = (ins AnyMemRef:$source, Tessera_WindowType:$peer_window,
                       I64Attr:$count, StrAttr:$dtype, I64Attr:$peer,
                       I64Attr:$peer_window_offset,
                       DefaultValuedAttr<I64Attr, "0">:$signal_index,
                       DefaultValuedAttr<I64Attr, "0">:$rma_context);
  let results = (outs);
  let assemblyFormat = "$source `to` $peer_window attr-dict `:` type($source) `,` type($peer_window)";
  let hasVerifier = 1;
}

def SignalOp : TesseraCollectiveOp<"signal", [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal a peer without transferring a payload";
  let arguments = (ins I64Attr:$peer,
                       DefaultValuedAttr<I64Attr, "0">:$signal_index,
                       DefaultValuedAttr<I64Attr, "0">:$rma_context);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WaitSignalOp : TesseraCollectiveOp<"wait_signal",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Wait for one or more peer signals in stream order";
  let arguments = (ins I64Attr:$peer,
                       DefaultValuedAttr<I64Attr, "1">:$operation_count,
                       DefaultValuedAttr<I64Attr, "0">:$signal_index,
                       DefaultValuedAttr<I64Attr, "0">:$rma_context);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Utility ops (existing)
//===----------------------------------------------------------------------===//

def PackCastOp : Op<Tessera_Collective_Dialect, "pack_cast", []> {
  let summary = "Cast and pack elements to wire dtype before transport";
  let arguments = (ins AnyMemRef:$input, OptionalAttr<StrAttr>:$to_dtype);
  let results   = (outs AnyMemRef:$packed);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($packed) }];
}

def AwaitOp : Op<Tessera_Collective_Dialect, "await",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Await completion of an async collective future";
  let arguments = (ins Tessera_FutureType:$future);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{ $future attr-dict `:` type($future) `->` type($value) }];
  let hasVerifier = 1;
}

def ShardViewOp : Op<Tessera_Collective_Dialect, "shard_view", []> {
  let summary = "Create a typed logical shard view of a tensor/memref value";
  let arguments = (ins AnyType:$input, StrAttr:$mesh_axis, I64Attr:$dim);
  let results = (outs Tessera_ShardType:$shard);
  let assemblyFormat = "$input `on` $mesh_axis `dim` $dim attr-dict `:` type($input) `->` type($shard)";
  let hasVerifier = 1;
}

def MaterializeShardOp : Op<Tessera_Collective_Dialect, "materialize_shard", []> {
  let summary = "Materialize a typed shard back into its payload value";
  let arguments = (ins Tessera_ShardType:$shard);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$shard attr-dict `:` type($shard) `->` type($value)";
  let hasVerifier = 1;
}

def QoSLimitOp : Op<Tessera_Collective_Dialect, "qos.limit", []> {
  let summary = "Set maximum in-flight collective chunks for QoS";
  let arguments = (ins I64Attr:$max_inflight);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
}

def QoSAcquireOp : Op<Tessera_Collective_Dialect, "qos.acquire", []> {
  let summary = "Acquire a collective QoS token before issuing a chunk";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}

def QoSReleaseOp : Op<Tessera_Collective_Dialect, "qos.release", []> {
  let summary = "Release a collective QoS token after a chunk completes";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}
