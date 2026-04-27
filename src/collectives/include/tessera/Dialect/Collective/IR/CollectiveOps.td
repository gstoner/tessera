include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "CollectiveTypes.td"

class TesseraCollectiveOp<string m, list<OpTrait> T = []>
    : Op<Tessera_Collective_Dialect, m, T>;

def Attr_WireDType : StrEnumAttr<"WireDType", "wire dtype",
    ["fp32", "bf16", "fp16", "fp8", "i8"]>;

def Attr_ReduceOp : StrEnumAttr<"ReduceOp", "reduction operation",
    ["sum", "max", "min", "prod"]>;

//===----------------------------------------------------------------------===//
// Point-to-all collectives (existing)
//===----------------------------------------------------------------------===//

def AllToAllOp : TesseraCollectiveOp<"all_to_all",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "All-to-all scatter: each rank sends a distinct shard to every other rank";
  let arguments = (ins AnyMemRef:$input,
                       OptionalAttr<Attr_WireDType>:$dtype,
                       OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = [{ $input attr-dict }];
}

//===----------------------------------------------------------------------===//
// Reduction collectives (Phase 4 additions)
//===----------------------------------------------------------------------===//

def AllReduceOp : TesseraCollectiveOp<"all_reduce",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "All-reduce: reduce a tensor across all ranks; each rank gets the result";
  let description = [{
    Reduces `input` across all ranks using `reduce_op` and writes the result
    into every rank's output buffer.  Equivalent to NCCL ncclAllReduce.

    `mesh_axis` names the mesh axis along which ranks are connected.
    `chunk_bytes` controls the transport chunk size for the ring algorithm.
  }];
  let arguments = (ins AnyMemRef:$input,
                       Attr_ReduceOp:$reduce_op,
                       StrAttr:$mesh_axis,
                       OptionalAttr<Attr_WireDType>:$dtype,
                       OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = [{
    $input `,` $reduce_op `on` $mesh_axis attr-dict
  }];
}

def ReduceScatterOp : TesseraCollectiveOp<"reduce_scatter",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Reduce-scatter: reduce across ranks then scatter one shard to each rank";
  let description = [{
    Reduces `input` across all ranks, then scatters rank k's shard of the
    result to rank k.  Equivalent to NCCL ncclReduceScatter.

    Used at data-parallel boundaries during the backward pass: each rank
    accumulates its local gradient, then reduce-scatter distributes the
    per-rank gradient slice.

    `scatter_dim` is the tensor dimension to scatter along (typically 0).
  }];
  let arguments = (ins AnyMemRef:$input,
                       Attr_ReduceOp:$reduce_op,
                       StrAttr:$mesh_axis,
                       I64Attr:$scatter_dim,
                       OptionalAttr<Attr_WireDType>:$dtype,
                       OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = [{
    $input `,` $reduce_op `on` $mesh_axis `dim` $scatter_dim attr-dict
  }];
}

def AllGatherOp : TesseraCollectiveOp<"all_gather",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "All-gather: each rank contributes a shard; every rank gets the full tensor";
  let description = [{
    Concatenates each rank's `input` shard along `gather_dim` and makes the
    full tensor available on all ranks.  Equivalent to NCCL ncclAllGather.

    Used before tensor-parallel layers that need full activations (row-parallel
    linear: gather the distributed output before the next layer).

    `gather_dim` is the tensor dimension to gather along (typically 0 or 1).
  }];
  let arguments = (ins AnyMemRef:$input,
                       StrAttr:$mesh_axis,
                       I64Attr:$gather_dim,
                       OptionalAttr<Attr_WireDType>:$dtype,
                       OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = [{
    $input `on` $mesh_axis `dim` $gather_dim attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Utility ops (existing)
//===----------------------------------------------------------------------===//

def PackCastOp : Op<Tessera_Collective_Dialect, "pack_cast", []> {
  let summary = "Cast and pack elements to wire dtype before transport";
  let arguments = (ins AnyMemRef:$input, OptionalAttr<Attr_WireDType>:$to_dtype);
  let results   = (outs AnyMemRef:$packed);
  let assemblyFormat = [{ $input attr-dict }];
}

def AwaitOp : Op<Tessera_Collective_Dialect, "await", []> {
  let summary = "Await completion of an async collective future";
  let arguments = (ins Tessera_FutureType:$future);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{ $future attr-dict }];
}

def QoSLimitOp : Op<Tessera_Collective_Dialect, "qos.limit", []> {
  let summary = "Set maximum in-flight collective chunks for QoS";
  let arguments = (ins I64Attr:$max_inflight);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
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
