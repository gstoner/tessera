# Tessera Benchmarks

This folder contains benchmark families at different maturity levels. The active
portable path is CPU-first and uses the current Python compiler surface where it
is available.

## Current Compiler Support

| Benchmark | Status | Compiler fit |
|---|---|---|
| `benchmark_gemm.py` | Active | Can exercise `@tessera.jit` through the current CPU `matmul -> relu` lowering path with `--use-compiler` via `run_all.py`. |
| `benchmark_attention.py` | Active proxy | Roofline model; can emit current flash-attention Graph IR/lowering diagnostics with `--use-compiler`, but executable Tile/Target lowering is not wired yet. |
| `benchmark_collective.py` | Active proxy | Alpha-beta communication model; should connect to runtime collectives once C ABI/runtime hooks are ready. |
| `run_all.py` | Active | Orchestrates GEMM, attention, and collective suites; pass `--use-compiler` for current compiler artifact checks. |
| `common/` | Active contract | Shared row schema, correctness helpers, and compiler artifact hooks used by benchmark suites. |
| `Tessera_SuperBench/` | Active harness | GEMM uses the current JIT CPU path with telemetry/autotune artifacts; FlashAttention and Conv2D emit artifact-only compiler rows while running NumPy reference timing/correctness; collectives default to the Tessera mock facade. |
| `spectral/` | Active benchmark | NumPy/PyTorch FFT/DCT/convolution benchmark with `--backend tessera-artifact` for Graph IR artifact rows. Native FFT lowering remains artifact-only until Tile/Target runtime support lands. |
| `Tessera_Operator_Benchmarks/` | Active C++ harness | CPU reference timing for all 7 registered op groups, Graph IR artifact coverage, `tessera.telemetry.v1` JSON summaries, and explicit `backend_unavailable` Tessera-runtime status. |
| `../archive/benchmarks/matrix_multiplication/` | Archived | Blackwell concept sketch using non-existent APIs; future Blackwell work should land as Target IR tests/runtime kernels/operator cases. |
| `DeepScholar-Bench/` | Active CPU smoke | Current-API research-synthesis smoke using `@tessera.jit`, matmul, softmax, layer_norm, and NumPy text/source embeddings. LOTUS integration remains optional and guarded. |
| `lattice_reasoning_core/` | Active current-compiler probe | LDT-style lattice step plus MOPD, Mamba-2, GQA, and Latent MoE primitive microbenchmarks. Emits NumPy reference rows, public Tessera primitive rows, Apple GPU native rows when `metal_runtime` is observed, and an artifact-only integrated-step row for remaining LDT fusion work. |

## Quick Checks

```bash
PYTHONPATH=python python3 benchmarks/run_all.py --json-only --no-save --use-compiler --smoke
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera.py --m=64 --n=64 --k=64 --repeat=1
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/runner/bench_run.py --config benchmarks/Tessera_SuperBench/configs/compiler_smoke.yaml --out /tmp/tessera_superbench_smoke
cmake -S benchmarks/Tessera_Operator_Benchmarks -B /tmp/tessera_opbench_build && cmake --build /tmp/tessera_opbench_build -j
PYTHONPATH=python python3 benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py --config benchmarks/Tessera_Operator_Benchmarks/scripts/configs/quick_sweep.yaml --bin /tmp/tessera_opbench_build/opbench --out /tmp/tessera_opbench_quick
PYTHONPATH=python python3 benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py --output /tmp/tessera_deepscholar_smoke.json
PYTHONPATH=python python3 benchmarks/spectral/spectral_bench.py --backend tessera-artifact --ops fft1d,dct2,conv1d_fft --sizes 64 --device cpu --repeats 1 --warmup 0 --outcsv /tmp/tessera_spectral.csv
PYTHONPATH=python python3 benchmarks/lattice_reasoning_core/benchmark_lattice_reasoning.py --smoke --json /tmp/tessera_lattice_reasoning_smoke.json
```

## Library-layer benchmarks (Phase 7)

A separate family from the operator/roofline benchmarks above.  Each
**category** is a generic compiler-surface label that captures the
primitive composition under test; each **proving workload** is a small,
domain-specific instantiation that anchors the category to a real paper
or canonical model.  The category label is what summaries, audit docs,
and talks should lead with — the workload name preserves the external
anchor for "yes, real surface, real reference."

| Category                | Proving workload(s)                                  | Source                                  |
|-------------------------|------------------------------------------------------|-----------------------------------------|
| Gridded-AI core         | (generic; no domain-specific anchor yet)             | `benchmarks/grid_ai_core/`              |
| **Diffusion grid core** | `corrdiff_core` (NVIDIA CorrDiff regional weather)   | `benchmarks/corrdiff/`                  |
| Clifford / GA core      | (generic; Cl(3, 0) rotor-sandwich chain)             | `benchmarks/clifford_core/`             |
| Energy / EBM core       | (generic; quadratic energy + annealed Langevin)      | `benchmarks/energy_core/`               |
| Cross-lane core         | `visual_complex_core` (M7 visual-complex milestone)  | `benchmarks/visual_complex_core/`       |
| Lattice reasoning core  | LDT step + MOPD/Mamba-2/GQA/Latent-MoE primitives    | `benchmarks/lattice_reasoning_core/`    |

Each library-layer benchmark ships:
  * A small Python core (config + model + oracle + harness).
  * An IR-visible lit fixture in `tests/tessera-ir/phase7/`.
  * A Python guard exercising forward determinism + oracle parity +
    canonical Architecture-Decision-#12 JSON schema.

The audit doc
(`docs/audit/compiler/COMPILER_AUDIT.md`) tracks the
six-layer compiler-correctness coverage for each category.

## Refactor Direction

- Promote compiler-backed benchmark kernels into small Python modules that expose
  Graph/Schedule/Tile/Target artifacts alongside timing rows.
- Keep analytical roofline/proxy benchmarks, but label them explicitly.
- Move purely speculative benchmark concepts to `archive/benchmarks/` once they
  are no longer feeding active compiler/runtime work.
