<!-- MERGE_START: Tessera Scaling & Resilience Spec (Part 2/2) -->

## 7. Passes & Pipelines
- `-tessera-optimizer-shard`: propagates optimizer sharding, inserts partitioned states, and collective annotations.
- `-tessera-insert-recompute`: reads checkpoint/recompute hints, inserts save/recompute ops with barriers.
- `-tessera-resilience-restart`: threads tokens, adds async export/import, and wraps critical regions with restart metadata.
- `-tessera-export-deployment-manifest`: walks module IR and writes JSON (mesh, shards, pipeline, collectives).

**Pipeline Alias:** `-tessera-sr-pipeline`  
`-canonicalize -cse -tessera-optimizer-shard -tessera-insert-recompute -tessera-resilience-restart -tessera-export-deployment-manifest`

## 8. Minimal ODS Sketch (extract)
```tablegen
//===- tessera_scaling.td -----------------------------------------------===//
def Tessera_Scaling_Dialect : Dialect {
  let name = "tessera_sr";
  let cppNamespace = "::mlir::tessera::sr";
}

def ResTokenType : TypeDef<Tessera_Scaling_Dialect, "ResToken"> {
  let mnemonic = "token";
  let summary = "Resilience token used to order replay/restart regions";
}

def CheckpointOp : Op<Tessera_Scaling_Dialect, "checkpoint",
  [SingleBlockImplicitTerminator<"mlir::func::ReturnOp">]> {
  let summary = "Marks a region for recomputation";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "`{` $body `}` attr-dict";
}

def RecomputeHintOp : Op<Tessera_Scaling_Dialect, "recompute_hint"> {
  let summary = "Advises the recompute insertion pass";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def ResilienceRegionOp : Op<Tessera_Scaling_Dialect, "resilience_region"> {
  let summary = "Critical section with restart/replay semantics";
  let arguments = (ins);
  let results = (outs ResTokenType:$token);
  let assemblyFormat = "`{` $body `}` attr-dict `:` type($token)";
}

def ExportManifestOp : Op<Tessera_Scaling_Dialect, "export_manifest"> {
  let summary = "Emit deployment manifest JSON";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}
```

## 9. Example IR Snippets
### 9.1 Checkpoint
```mlir
tessera_sr.checkpoint {
  %y = "tessera.tile.gemm"(%a, %b) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf32>
  "tessera.tile.relu"(%y) : (tensor<?x?xf32>) -> tensor<?x?xf32>
}
```

### 9.2 Optimizer sharding hint
```mlir
%p = "tessera.param"() : () -> tensor<*, f16>
%m = "tessera.opt.state"() {kind = "adam_m"} : () -> !tessera.opt.state
"tessera.optimizer.shard"(%p, %m) { axis = "data", policy = "zero2" } :
  (tensor<*xf16>, !tessera.opt.state) -> (tensor<*xf16>, !tessera.opt.state)
```

### 9.3 Resilience region + token threading
```mlir
%t = tessera_sr.resilience_region {
  // pipeline stage work
} : !tessera_sr.token

// enforce ordering across async export/import
"tessera.target.export"(%t) : (!tessera_sr.token) -> ()
```

### 9.4 Export manifest
```mlir
tessera_sr.export_manifest { path = "manifest.json", include = ["mesh","optimizer"] }
```

## 10. Runtime Notes
- Tokens lower to Async Dialect ops to guarantee restart ordering.
- Manifest includes **logical mesh**, **param/state shards**, **collectives**, and **stage graphs**.
- Passes are conservative if attrs are missing (no-ops).

## 11. Tests (FileCheck)
See `tests/*.mlir` for patterns.

<!-- MERGE_END -->