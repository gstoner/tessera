//===- ScheduleMeshPipelineOps.td - Mesh & pipeline ops ------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def ScheduleDialect : Dialect {
  let name = "schedule";
  let cppNamespace = "::tessera::schedule";
}

def Schedule_MeshDefineOp : Op<ScheduleDialect, "mesh.define"> {
  let summary = "Define a logical mesh";
  let arguments = (ins ArrayAttr:$dims, ArrayAttr:$axis_names);
  let hasVerifier = 1;
}

def Schedule_MeshRegionOp : Op<ScheduleDialect, "mesh.region"> {
  let summary = "Enter a mesh axis region";
  let arguments = (ins SymbolRefAttr:$mesh, StrAttr:$axis);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_PipelineRegionOp : Op<ScheduleDialect, "pipeline.region"> {
  let summary = "Pipeline-parallel region with schedule";
  let arguments = (ins StrAttr:$schedule, I32Attr:$micro_batches);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_StageOp : Op<ScheduleDialect, "stage"> {
  let summary = "A pipeline stage";
  let arguments = (ins ArrayAttr:$devices);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_YieldOp : Op<ScheduleDialect, "yield",
    [Pure, Terminator, ReturnLike]> {
  let summary = "Terminator for schedule region bodies";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [OpBuilder<(ins), [{
    $_state.addTypes({});
  }]>];
}

def Schedule_MemorySpaceAttr : StrEnumAttr<"MemorySpace", "memory space",
  ["register", "shared", "lds", "global", "managed", "host", "tmem"]>;

def Schedule_OverlapAttr : StrEnumAttr<"OverlapPolicy", "overlap policy",
  ["none", "compute", "collective"]>;

def Schedule_PrefetchOp : Op<ScheduleDialect, "prefetch"> {
  let summary = "Prefetch a value/cache into an explicit memory space";
  let arguments = (ins AnyType:$source,
                       Schedule_MemorySpaceAttr:$into,
                       Schedule_OverlapAttr:$overlap,
                       OptionalAttr<I32Attr>:$stage,
                       OptionalAttr<I32Attr>:$vector);
  let results = (outs AnyType:$staged);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($staged)";
  let hasVerifier = 1;
}

def Schedule_AsyncCopyOp : Op<ScheduleDialect, "async_copy"> {
  let summary = "Schedule-level asynchronous movement before Tile IR lowering";
  let arguments = (ins AnyType:$src, AnyType:$dst,
                       Schedule_MemorySpaceAttr:$src_space,
                       Schedule_MemorySpaceAttr:$dst_space,
                       I32Attr:$stage,
                       Schedule_OverlapAttr:$overlap);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst) `->` type($token)";
  let hasVerifier = 1;
}

def Schedule_AwaitMovementOp : Op<ScheduleDialect, "await_movement"> {
  let summary = "Wait for a schedule-level movement token at the true use site";
  let arguments = (ins AnyType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
  let hasVerifier = 1;
}

def Schedule_ArtifactOp : Op<ScheduleDialect, "artifact"> {
  let summary = "Durable schedule artifact hash and selected tuning knobs";
  let arguments = (ins StrAttr:$hash,
                       StrAttr:$arch,
                       StrAttr:$shape_key,
                       OptionalAttr<DictionaryAttr>:$tile,
                       OptionalAttr<DictionaryAttr>:$movement,
                       OptionalAttr<StrAttr>:$numeric_policy);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}
