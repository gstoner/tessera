# Installed compiler and owning-device evidence — 2026-09-10

Owners: COMPILER-DEVEX-1 and EVIDENCE-PACKET-1.
Sync: INSTALLED-DEVICE-GATES-2026-09-10.

`installed.json` records Super-Bear's compiler-tools component after relocation
to a temporary prefix. Both executables report their versions and the installed
optimizer feeds the installed LLVM translator successfully. Loader overrides
and Tessera/Python environment overrides are removed. LLVM dependencies are
provided by the recorded assertions toolchain; this is not proof of a standalone
redistributable LLVM distribution. All 474 assertions lit fixtures also pass.

Reproduce with the configured build and a fresh prefix:

```sh
cmake --build BUILD --target tessera-opt tessera-translate-mlir
cmake --install BUILD --prefix PREFIX --component compiler-tools
python scripts/check_installed_compiler.py PREFIX --output installed.json
```

Native device measurements deliberately have different workloads and are not
cross-device performance comparisons:

| Owning device | Bounded workload | Evidence and outcome |
| --- | --- | --- |
| Apple M1 Max / apple7 | GQA attention backward, fp16/bf16 inputs and fp32 bias | Both direct and package native status and numerical checks pass. Seven paired trials: fp16 direct/package 2.958/3.130 ms; bf16 1.395/1.576 ms. |
| RTX 5070 / SM120 | 16x8 ANN, fused row-parallel transformed candidate | Nine independent processes: median speedup 0.999666x, lower bound 0.977299x; incumbent retained. |
| Princess-Luna / gfx1151 | Same logical 16x8 ANN, its own native HIP artifact | Nine independent processes: median speedup 0.997695x, lower bound 0.983878x; incumbent retained. |

`nvidia/` and `rocm/` contain every run and the existing scoped arbiter's
selection result. Each run verifies the numerical oracle before and after
measurement; candidates are retired after selection. `*-ann.json` are the
initial exploratory seed-101 measurements, excluded from the nine-run decision.
`apple-backward.json` retains runtime/compiler/image ancestry and per-trial data.
Two additional native Metal static/dynamic GELU descriptor tests passed.

Timing is warm complete-call host time (including H2D/D2H for ANN), not device
kernel time. Neither GPU ANN lower bound clears the 1.02 admission margin.
Apple has one process with paired trials, not independent cross-run promotion
proof. No production route was promoted and no device-clock, counter, general
backend coverage or uncertain-driver recovery claim is made.
