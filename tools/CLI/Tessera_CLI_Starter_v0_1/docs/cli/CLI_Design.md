<!-- MERGE_START -->
# Tessera CLI Suite — Formal Design (v0.1)

**Scope:** Introduce a cohesive CLI suite to complement Tessera’s Python + MLIR UX, enable CI-friendly pipelines, and standardize artifact layouts.

## Goals
- Unix-friendly: `stdin → stdout` default, composable with `mlir-*` tools.
- Layer-aware: GraphIR / ScheduleIR / TileIR / TargetIR.
- Deterministic: stable exit codes, `--json` machine output, `--seed`.
- Reproducible artifacts: predictable `--out-dir` layout with a manifest.
- Perf-first: built-in hooks for roofline, Perfetto, occupancy estimates.
- Extensible: flags map cleanly to pass manager and backend codegen options.

## Tools
- **tessera-opt**: Apply Tessera/MLIR passes and dump IR snapshots.
- **tessera-compile**: Lower from any IR layer to PTX/CUBIN, HSACO, or CPU objects; optionally emit host stubs & CMake.
- **tessera-profiler**: Run compiled kernels; produce roofline CSV + Perfetto trace + HTML report.
- **tessera-run**: Load artifact dir, feed sample inputs, validate numerics (golden or tolerance-based).
- **tessera-autotune**: Search parameter spaces, populate `tune.db` (SQLite) cache keyed by schedule/arch/input-shape.
- **tessera-inspect**: Summarize IR/kernels (regs, smem, occupancy) into a single table (markdown/CSV/JSON).

## Common Flags
```
-I, --include <dir>...      Add import paths
-o, --output <file|->       Output file (default: -)
    --out-dir <dir>         Root for multi-file artifacts
-O, --opt <0|1|2|3>         Optimization level (default: 2)
    --verify                Run verifiers after each phase
    --require-ir-version X.Y.Z
    --time-passes           Print per-pass timings
    --print-pipeline        Print the resolved pipeline and exit
    --seed <int>            Seed for randomized transforms
    --json                  Emit machine-readable summary
    --cache-dir <dir>       Pass/codegen cache location
    --temp-dir <dir>        Temporary workspace
    --fail-on-warn          Promote warnings to errors
-v, --verbose               Increase log verbosity
    --log-level {error,warn,info,debug,trace}
```

## Layer Semantics
`--from {graph,schedule,tile,target,auto}` and `--to {graph,schedule,tile,target}` control interpretation and output when applicable.

## Pipelines (aliases)
- `graph-to-schedule`: `-tessera-verify, -tessera-migrate-ir, -tessera-graph-canonicalize, -tessera-lower-graph-to-schedule, -tessera-cleanup`
- `schedule-to-tile`: `-tessera-verify, -tessera-schedule-normalize, -tessera-lower-schedule-to-tile, -tessera-tiling-interface, -tessera-cleanup`
- `tile-to-target`: `-tessera-verify, -tessera-lower-tile-to-target, -tessera-target-canonicalize, -tessera-cleanup`

## Artifact Layout
```
<out-dir>/
  ir/              # optional IR snapshots
  kernels/         # .ptx, .cubin, .hsaco, .o, .ll
  host/            # host launch stubs
  cmake/           # generated CMake project
  reports/         # roofline.csv, perfetto.json, occupancy.json, sizes.json
  tune/            # tune.db (SQLite)
  meta/            # compile.json (manifest)
```

## Exit Codes (uniform)
- `0` success
- `1` parse/verify error (IR/flags)
- `2` pass pipeline error
- `3` I/O error (read/write/perm)
- `10` lowering failure
- `11` backend toolchain failure
- `12` invalid target/arch
- `20` runtime execution failure (profiler/run)
- `30` autotune failure

## Tool Details

### tessera-opt
- **Input**: `.mlir` (any Tessera layer), stdin/file
- **Output**: transformed `.mlir` to stdout or `-o`
- **Flags**:
  - `--from/--to` layers; `--alias=<name>`; `--pipeline=<mlir-opt-syntax>`
  - `--add-pass <p>` / `--disable-pass <p>`
  - `--dump {graph,schedule,tile,target,llvm} --dump-dir <dir>`
  - `--canonicalize --migrate-ir --cleanup --verify-only`

### tessera-compile
- **Input**: `.mlir` (graph/schedule/tile/target)
- **Outputs**: target artifacts + manifest + (opt) host/CMake
- **Flags**:
  - Target: `--platform {cuda,hip,cpu}`, `--arch {sm_90,gfx1100,avx2,...}`, `--to {ptx,cubin,hsaco,cpu-obj,llvm-ir}`
  - Features: `--enable-tensor-cores --enable-wgmma --enable-tma --mfma-profile {bf16,fp16,tf32,f32}`
  - Extras: `--emit-host --emit-cmake --link-runtime {shared,static,none}`
  - Reporting: `--profile --occupancy-report --size-report`

### tessera-profiler
- **Input**: `<artifact-dir>` (from `tessera-compile`)
- **Action**: Launch kernels with synthetic or provided shapes, collect times, derive arithmetic intensity & achieved BW, export:
  - `reports/roofline.csv`, `reports/roofline.html` (template)
  - `reports/perfetto.json` (trace events)
  - `reports/summary.json` (aggregated metrics)

### tessera-run
- **Input**: `<artifact-dir>` + optional input JSON (`inputs.json`) or random generator spec
- **Action**: Execute kernels; compare numerics to a golden or reference tolerance (`--rtol --atol`); emit `reports/validate.json`

### tessera-autotune
- **Input**: IR/kernels + space spec (flags or YAML)
- **Action**: Explore schedules/tiling params; write `tune/tune.db (SQLite)` and `tune/summary.json`
- **Searchers**: grid, random, Hyperband (pluggable)
- **Keys**: `(platform, arch, op, shape, candidate-hash)`

### tessera-inspect
- **Input**: `<artifact-dir>`
- **Action**: Summarize per-kernel: regs/thread, smem, max blocks/SM, occupancy estimate, code size, entry name. Emit table: md/csv/json.

## JSON Manifests

### `meta/compile.json`
```json
{
  "tessera": {"version":"0.3.1"},
  "input": {"file":"model.mlir","from":"graph"},
  "pipeline": ["verify","migrate","graph->schedule","schedule->tile","tile->target","codegen"],
  "target": {"platform":"cuda","arch":"sm_90","to":"ptx"},
  "features": {"tensor_cores":true,"wgmma":true},
  "artifacts": {"kernels":["kernels/attn.ptx"],"host":["host/launch.cu"]}
}
```

### `reports/summary.json`
```json
{
  "kernels":[
    {"name":"attn","time_ms":0.83,"bytes":1.2e9,"flops":6.4e12,"ai":5.3,
     "achieved_bw_gbps":950,"achieved_tflops":7.7}
  ]
}
```

## Security & Sandboxing
- Don’t execute untrusted kernels by default; require `--allow-exec`.
- Sandboxed temp dirs; redact env; cap runtime via `--timeout`.

## Open Questions
- How deeply to introspect PTX/SASS for regs/smem if toolchains absent?
- Perfetto schema details; CUPTI/Nsight optional integration.
- Reference impl for numerics (CPU path vs. high-level interpreter).

<!-- MERGE_END -->
