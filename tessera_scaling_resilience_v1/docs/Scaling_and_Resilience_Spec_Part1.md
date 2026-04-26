<!-- MERGE_START: Tessera Scaling & Resilience Spec (Part 1/2) -->

# Tessera Scaling & Resilience Extensions
**Version:** v1  
**Scope:** IR semantics + passes for optimizer sharding, checkpoint/recompute, resilience/restart, and deployment manifest export.

## 1. Goals
- Make scaling strategies first-class in IR (not hidden in runtime glue).
- Encode *where/how* optimizer and activations live across devices.
- Provide **resilience semantics**: recoverable regions, restart tokens, and event hooks.
- Enable *system-aware* planning via a **Deployment Manifest**.

## 2. New IR Concepts (High-level)
We extend **Schedule IR** and **Target IR** with new ops/attrs:

- `tessera.schedule.checkpoint` (region): marks recompute-safe segments.
- `tessera.schedule.recompute_hint(policy=...)`: policies: `full`, `selective`, `none`.
- `tessera.optimizer.shard(param|state, axis|mesh, policy=...)`: declares optimizer sharding.
- `tessera.resilience.region(mode= "replay" | "restart")`: recovery semantics.
- `tessera.resilience.token` SSA value to link export/import for restart ordering.
- `tessera.system.export_manifest(...)`: emits JSON manifest describing sharding, mesh, pipeline, and optimizer placements.

These attach to existing Tessera IR levels without breaking existing code.

## 3. Optimizer State Sharding
**Motivation:** Treat optimizer states (m, v, moments, fp32 masters) as explicit IR objects.
### 3.1 Types & Attrs
- Type: `!tessera.opt.state<{of: tensor<...>, kind: "adam_m"|"adam_v"|...}>`
- Attrs:
  - `#tessera.opt.shard<axis="data" | "model" | "pipeline", mesh="dataxmodel">`
  - `#tessera.opt.partition<"zero1"|"zero2"|"zero3">`
  - `#tessera.opt.offload<"host"|"nvme">` (future)

### 3.2 Op: `tessera.optimizer.shard`
```
tessera.optimizer.shard %param, %state
  { axis = "data", policy = "zero2", mesh = ["data","model"] } :
  (!t.tensor, !tessera.opt.state) -> (!t.tensor, !tessera.opt.state)
```
**Effect:** registers a sharding map for param/state across the mesh. Legalization lowers to collective annotations and per-rank slices.

## 4. Checkpoint / Recomputation
### 4.1 Op: `tessera.schedule.checkpoint` (region)
```
tessera.schedule.checkpoint { policy = "selective" } {
  // body: ops allowed to be recomputed on backward
}
```
Lowering inserts:
- Save minimal boundary tensors
- Tag region for **selective recompute** during backward
- Materialize **barriers** to ensure replay order

### 4.2 Op: `tessera.schedule.recompute_hint`
```
tessera.schedule.recompute_hint { policy = "full" }
```
Advises -tessera-insert-recompute for automatic placement.

## 5. Resilience Semantics
### 5.1 Op: `tessera.resilience.region`
```
%tok = tessera.resilience.region { mode = "replay" } {
  // critical stage (e.g., pipeline stage, expert dispatch)
} : !tessera.res.token
```
- Produces a **token** used to order export→import on restart.
- Legalization wires to async/await tokens in Target IR.

### 5.2 Events
- `tessera.resilience.on_failure(handler = @fn)`
- `tessera.resilience.on_recovery(handler = @fn)`

## 6. Deployment Manifest Export
### 6.1 Op: `tessera.system.export_manifest`
```
tessera.system.export_manifest {
  path = "manifest.json",
  include = ["mesh","pipeline","optimizer","checkpoint","collectives"]
}
```
Generates a JSON describing all sharding/placement needed by runtime/orchestrator.

<!-- MERGE_END -->