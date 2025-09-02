
# Compiler/Runtime Flow (Part 2/2)

## Compile-time
1. **Discover tunables** on ops.
2. **Generate candidates** via transform recipes (GEMM meta-schedule).
3. **Static checks**: SMEM/regs/occupancy estimators prune invalid configs.
4. **Materialize variants** with selected concrete values; lower to target.

## Runtime
1. **Search**: Grid/Random/**Hyperband** (budget = iters) produces candidates.
2. **Measure**: CUDA events wall-time; (optional) counters.
3. **Cache**: Write each trial to **SQLite** with a **schedule_key** = SHA256 of
   `{kernel_id, problem_signature, config, device_class, objective}`.
4. **Select**: pick best metric; persist `best.json` and `report.html`.

## File map
- `include/tessera/TunableAttrs.td` – attribute schema (sketch).
- `mlir/gemm_meta_schedule.mlir` – transform dialect script (illustrative).
- `tessera_autotuner/` – Python tuner with SQLite cache + Hyperband.

## Notes
- The transform script uses placeholder references (`#attr.tessera.*`), meant
  to be resolved by a small pass/materialization layer that reads the concrete
  values chosen by the tuner and rewrites the transform ops.
- The SQLite cache enables multi-run, multi-host reuse and db-driven reports.

<!-- MERGE:END Tessera_Autotune_Architecture.md -->
