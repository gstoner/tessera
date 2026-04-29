# Tessera CLI Suite — Design Reference (v0.4.0)

**Status:** Active development — all 7 tools implemented, roofline + Perfetto
output wired, autotune grid/random searchers functional.

---

## Goals

- **Unix-friendly** — `stdin → stdout` default; composable with `mlir-*` tools.
- **Layer-aware** — GraphIR / ScheduleIR / TileIR / TargetIR.
- **Deterministic** — stable exit codes, `--json` machine output, `--seed`.
- **Reproducible artifacts** — predictable `--out-dir` layout with a manifest.
- **Perf-first** — built-in hooks for roofline, Perfetto, occupancy estimates.
- **Extensible** — flags map cleanly to the pass manager and backend codegen.

---

## Tools

| Tool | Purpose |
|------|---------|
| `tessera-opt`      | Apply Tessera/MLIR passes; dump IR snapshots |
| `tessera-compile`  | Lower IR to PTX/CUBIN/HSACO/CPU-obj; emit host stubs |
| `tessera-profiler` | Roofline CSV + Perfetto trace + HTML report |
| `tessera-run`      | Execute kernels; validate numerics (tolerance or golden) |
| `tessera-autotune` | Grid/random/Hyperband tile search; write `tune.db` |
| `tessera-inspect`  | Per-kernel: regs, smem, occupancy, size (md/csv/json/table) |
| `tessera-new`      | Scaffold a new Tessera kernel project |

---

## Common Flags (all tools)

```
-h, --help                 Show usage and exit
    --version              Print tessera CLI version and exit
-o, --output <file|->      Output file (default: -)
    --out-dir <dir>        Root for multi-file artifacts (default: out/)
-I <dir>                   Add import path (repeatable)
    --json                 Emit machine-readable JSON summary to stdout
    --dry-run              Print what would happen; do not write files
    --fail-on-warn         Promote warnings to errors
-v, --verbose              Set log level to debug
    --log-level <lvl>      {error,warn,info,debug,trace} (default: info)
    --temp-dir <dir>       Temporary workspace (default: /tmp/tessera)
    --cache-dir <dir>      Pass/codegen cache location
```

---

## Artifact Layout

```
<out-dir>/
  ir/              # .mlir IR snapshots per layer
  kernels/         # .ptx, .cubin, .hsaco, .o, .ll
  host/            # host launch stubs (.cu / .cpp)
  cmake/           # generated CMakeLists.txt
  reports/         # roofline.csv, roofline.html, perfetto.json,
                   # summary.json, occupancy.json, sizes.json,
                   # inspect.{md,csv,json}, validate.json
  tune/            # tune.db (SQLite), schema.sql, summary.json
  meta/            # compile.json (manifest)
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0  | Success |
| 1  | Parse / verify error (flags or IR) |
| 2  | Pass pipeline failure |
| 3  | I/O error (read / write / permissions) |
| 10 | Lowering failure |
| 11 | Backend toolchain failure |
| 12 | Invalid target / arch |
| 20 | Runtime execution failure |
| 30 | Autotune failure |

---

## Tool Reference

### tessera-opt

Apply Tessera/MLIR passes to IR files and write the transformed IR.

```
tessera-opt [options] [input.mlir ...]
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--pipeline <str>` | Pass-pipeline string (mlir-opt syntax) |
| `--alias <name>` | Named pipeline: `neighbors-pipeline`, `graph-to-schedule`, `schedule-to-tile`, `tile-to-target`, `pm-verify`, `pm-legalize`, `full` |
| `--from <layer>` | Input layer: `graph`, `schedule`, `tile`, `target`, `auto` |
| `--to <layer>` | Output layer |
| `--add-pass <p>` | Append pass (repeatable) |
| `--disable-pass <p>` | Remove pass (repeatable) |
| `-O <0-3>` | Optimization level (default 2) |
| `--verify` | Run verifiers after each phase |
| `--verify-only` | Verify IR without transforming |
| `--print-pipeline` | Print resolved pipeline and exit |
| `--time-passes` | Annotate output with pass-timing note |
| `--dump <layer>` | Also write IR snapshot for `<layer>` |
| `--dump-dir <dir>` | Directory for `--dump` output |
| `--canonicalize` | Append `-canonicalize` |
| `--migrate-ir` | Append `-tessera-migrate-ir` |
| `--cleanup` | Append `-tessera-cleanup` |

**Examples:**

```bash
# Apply halo-infer to a neighbors IR file
tessera-opt stencil.mlir --alias=neighbors-pipeline -o out.mlir

# Verify only, no transforms
tessera-opt model.mlir --verify-only

# Print the resolved graph-to-schedule pipeline without running it
tessera-opt --alias=graph-to-schedule --print-pipeline

# Pipe from stdin
cat model.mlir | tessera-opt --pipeline=-cse,-canonicalize -o -
```

---

### tessera-compile

Lower Tessera IR to target artifacts.

```
tessera-compile [options] input.mlir
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--platform <p>` | `cuda`, `hip`, `cpu` (default `cuda`) |
| `--arch <a>` | `sm_90`, `sm_80`, `gfx1100`, `avx2`, … (default `sm_90`) |
| `--to <fmt>` | `ptx`, `cubin`, `hsaco`, `cpu-obj`, `llvm-ir` (default `ptx`) |
| `--from <layer>` | Input IR layer (default `auto`) |
| `--mfma-profile <p>` | `bf16`, `fp16`, `tf32`, `f32` (default `bf16`) |
| `-O <0-3>` | Optimization level (default 2) |
| `--enable-tensor-cores` | Enable WMMA / tensor core instructions |
| `--enable-wgmma` | Enable WGMMA (sm_90+) |
| `--enable-tma` | Enable TMA async bulk-copy (sm_90+) |
| `--emit-host` | Generate CUDA/HIP/CPU host launch stub |
| `--emit-cmake` | Generate CMakeLists.txt for the artifact |
| `--occupancy-report` | Write `reports/occupancy.json` |
| `--size-report` | Write `reports/sizes.json` |

**Examples:**

```bash
# Compile GEMM graph IR to PTX for sm_90 with TMA + host stub
tessera-compile matmul.mlir \
  --platform=cuda --arch=sm_90 --to=ptx \
  --enable-tensor-cores --enable-tma --emit-host \
  --out-dir build/

# CPU path with AVX2, emit CMake project
tessera-compile model.mlir \
  --platform=cpu --arch=avx2 --to=cpu-obj \
  --emit-host --emit-cmake --out-dir build/

# Dry-run: see what would be written
tessera-compile model.mlir --platform=cuda --arch=sm_80 --dry-run
```

---

### tessera-profiler

Profile compiled artifacts and generate roofline reports.

```
tessera-profiler [options] [artifact-dir]
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--peak-tflops <f>` | Hardware peak TFLOPs/s (default 989 = H100 BF16) |
| `--peak-bw <f>` | Hardware peak memory BW GB/s (default 3350 = H100 HBM3) |
| `--kernel <name>` | Filter: profile only matching kernel |
| `--warmup <n>` | Warmup iterations (default 3) |
| `--iters <n>` | Timed iterations (default 10) |
| `--shapes <spec>` | Input shape for launch sizing |

**Outputs:**

- `reports/roofline.csv` — per-kernel AI, latency, achieved TFLOPs/BW, bound
- `reports/roofline.html` — self-contained interactive roofline chart
- `reports/perfetto.json` — load at [ui.perfetto.dev](https://ui.perfetto.dev)
- `reports/summary.json` — aggregated metrics including MFU

**Examples:**

```bash
tessera-compile model.mlir --platform=cuda --arch=sm_90 --out-dir build/
tessera-profiler build/ --peak-tflops=989 --peak-bw=3350

# Open the chart
open build/reports/roofline.html

# Load trace
# → ui.perfetto.dev → Open trace file → build/reports/perfetto.json
```

---

### tessera-run

Execute compiled kernels and validate numerics.

```
tessera-run [options] [artifact-dir]
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--shapes <MxNxK>` | Problem shape for synthetic inputs (default 1024x1024x1024) |
| `--inputs <file>` | Input tensor spec JSON |
| `--golden <file>` | Golden reference JSON |
| `--rtol <f>` | Relative tolerance (default 1e-3) |
| `--atol <f>` | Absolute tolerance (default 1e-5) |
| `--timeout <s>` | Max execution seconds (default 60) |
| `--allow-exec` | Actually execute kernels (requires toolchain) |

**Examples:**

```bash
tessera-run build/ --shapes=2048x2048x2048 --rtol=1e-4

# Strict validation
tessera-run build/ --atol=1e-6 --fail-on-warn --json
```

---

### tessera-autotune

Search tile and schedule parameter spaces.

```
tessera-autotune [options] input.mlir
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--searcher <s>` | `grid`, `random`, `hyperband` (default `random`) |
| `--trials <n>` | Candidates to evaluate (default 32) |
| `--seed <n>` | Random seed (default 42) |
| `--shapes <MxNxK>` | Problem shape (default 2048x2048x2048) |
| `--platform <p>` | Target platform |
| `--arch <a>` | Target arch |
| `--peak-tflops <f>` | Hardware peak TFLOPs/s |
| `--peak-bw <f>` | Hardware peak memory BW GB/s |
| `--budget <dur>` | Time budget (e.g. `30s`, `5m`) — overrides `--trials` |
| `--db <file>` | SQLite cache file |
| `--export <file>` | Export best config JSON |

**Outputs:**

- `tune/schema.sql` — SQLite schema (run against `--db` to initialise)
- `tune/summary.json` — evaluations count, best candidate + metric

**Examples:**

```bash
# 64-trial random search with fixed seed
tessera-autotune matmul.mlir \
  --searcher=random --trials=64 --seed=42 \
  --shapes=4096x4096x4096 \
  --platform=cuda --arch=sm_90 \
  --export=best.json

# Full grid search
tessera-autotune model.mlir --searcher=grid --out-dir build/
```

---

### tessera-inspect

Summarize compiled kernels: registers, shared memory, occupancy, size.

```
tessera-inspect [options] [artifact-dir | file.ptx]
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--format <f>` | `table` (default), `md`, `csv`, `json` |
| `--kernel <name>` | Filter to matching kernel |
| `--show-ir` | List IR snapshot files |
| `--show-ptx` | Print full PTX to stdout |

**Examples:**

```bash
# Human-readable table to stdout
tessera-inspect build/

# JSON to file for CI comparison
tessera-inspect build/ --format=json -o reports/inspect.json

# Inspect a single PTX file
tessera-inspect build/kernels/flash_attn.ptx --format=md
```

---

### tessera-new

Scaffold a new Tessera kernel project.

```
tessera-new <name> [options]
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--template <t>` | `gemm` (default), `flash_attn`, `conv2d` |
| `--platform <p>` | Target platform (default `cuda`) |
| `--arch <a>` | Target arch (default `sm_90`) |

**Creates:**

```
<name>/
  <name>.mlir       — Graph IR skeleton
  <name>.py         — Python entry-point (tessera.jit)
  Makefile          — compile / opt / profile / run / tune / inspect
  CMakeLists.txt    — CMake integration
```

**Examples:**

```bash
tessera-new my_attn --template=flash_attn --platform=cuda --arch=sm_90
cd my_attn
make compile   # → tessera-compile my_attn.mlir --platform=cuda ...
make profile   # → tessera-profiler build/ ...
make tune      # → tessera-autotune my_attn.mlir --searcher=random ...
```

---

## Common Workflows

### Full compile → profile pipeline

```bash
tessera-compile model.mlir \
  --platform=cuda --arch=sm_90 --to=ptx \
  --enable-tensor-cores --enable-tma --emit-host \
  --occupancy-report --size-report \
  --out-dir build/

tessera-profiler build/ --peak-tflops=989 --peak-bw=3350
tessera-inspect  build/ --format=md
tessera-run      build/ --shapes=2048x2048x2048 --rtol=1e-3
```

### CI validation script

```bash
#!/bin/bash
set -e
tessera-compile model.mlir --platform=cuda --arch=sm_90 \
  --out-dir /tmp/ci_out --json > compile.json
tessera-run /tmp/ci_out --rtol=1e-4 --atol=1e-6 --json > run.json
tessera-inspect /tmp/ci_out --format=json -o inspect.json
```

### Autotune then recompile with best config

```bash
tessera-autotune model.mlir --searcher=random --trials=128 \
  --shapes=4096x4096x4096 --export=best.json

# best.json → feed tile sizes back into compile (future: --config=best.json)
tessera-compile model.mlir --platform=cuda --arch=sm_90 --out-dir build_tuned/
```

---

## Pipeline Aliases

| Alias | Expanded passes |
|-------|----------------|
| `neighbors-pipeline` | `-tessera-halo-infer, -tessera-stencil-lower, -tessera-pipeline-overlap, -tessera-dynamic-topology, -canonicalize` |
| `graph-to-schedule` | `-tessera-verify, -tessera-migrate-ir, -tessera-graph-canonicalize, -tessera-lower-graph-to-schedule, -tessera-cleanup` |
| `schedule-to-tile` | `-tessera-verify, -tessera-schedule-normalize, -tessera-lower-schedule-to-tile, -tessera-tiling-interface, -tessera-cleanup` |
| `tile-to-target` | `-tessera-verify, -tessera-lower-tile-to-target, -tessera-target-canonicalize, -tessera-cleanup` |
| `pm-verify` | `-tessera-pm-verify, -cse, -canonicalize` |
| `pm-legalize` | `-tessera-pm-verify, -tessera-graph-to-schedule, -tessera-schedule-to-tile, -canonicalize` |
| `full` | All of the above end-to-end |

---

## JSON Manifests

### `meta/compile.json`

```json
{
  "tessera": {"version": "0.4.0"},
  "input":   {"file": "model.mlir", "from": "graph"},
  "pipeline": ["verify", "migrate", "graph->target", "codegen"],
  "target":  {"platform": "cuda", "arch": "sm_90", "to": "ptx"},
  "features": {"tensor_cores": true, "wgmma": true, "tma": true},
  "artifacts": {
    "kernels": ["kernels/demo.ptx"],
    "host":    ["host/launch.cu"]
  },
  "timestamp": "2026-04-29T14:00:00Z"
}
```

### `reports/summary.json`

```json
{
  "peak_tflops": 989.0,
  "peak_bw_gbps": 3350.0,
  "kernels": [
    {
      "name":             "demo_kernel",
      "time_ms":          0.8312,
      "flops":            6.4e12,
      "bytes":            1.2e9,
      "ai":               5.333,
      "achieved_tflops":  7.699,
      "achieved_gbps":    950.1,
      "bound":            "compute",
      "mfu":              0.0078
    }
  ]
}
```

### `reports/validate.json`

```json
{
  "summary": {
    "total": 1, "passed": 1, "failed": 0,
    "rtol": 0.001, "atol": 1e-5
  },
  "kernels": [
    {
      "name":        "demo_kernel",
      "status":      "pass",
      "max_abs_err": 0.0,
      "max_rel_err": 0.0
    }
  ]
}
```

---

## Security & Sandboxing

- Kernels are **not executed** without `--allow-exec`.
- All writes go under `--out-dir`; no writes to system paths.
- Redact environment in subprocess calls; cap runtime via `--timeout`.
- `tessera-new` validates project names against `[a-zA-Z0-9_-]` only.

---

## Shell Completion

Generate completions for your shell (pipe into your rc file):

```bash
# Bash
eval "$(tessera-opt --generate-completions bash)"   # coming in v0.5

# Zsh
eval "$(tessera-opt --generate-completions zsh)"
```

Until `--generate-completions` is implemented, a static completion script is
available at `docs/cli/completions/tessera.bash`.

---

## Building

```bash
cmake -B build -S tools/CLI/Tessera_CLI_Starter_v0_1 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
cmake --install build --prefix ~/.local

# Run unit tests
ctest --test-dir build -V

# Run lit tests (requires llvm-lit in PATH)
cmake --build build --target check-tessera-cli-lit
```

---

## Changelog

| Version | Notes |
|---------|-------|
| 0.4.0 | All 7 tools with real flags; shared `Args` parser; proper exit codes; roofline HTML chart; Perfetto JSON; numeric validation; grid/random autotune; `tessera-new` scaffolding; recursive `mkdir_p`; `jsonEscape`; updated lit tests; CMake install + LTO |
| 0.3.1 | Artifact layout, skeleton `--json` output, CMake stub |
| 0.1.0 | Initial design document |
