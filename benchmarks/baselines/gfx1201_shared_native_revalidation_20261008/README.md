# Shared native compiler: gfx1201 revalidation

## Shared native compiler gfx1201 parity — 2026-10-08

Owner ROCM-NVFP4-INGEST-1 / W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SHARED-NATIVE-PROGRAM-PARITY-2026-10-08.
The latest Super-Bear-built LLVM/MLIR 23.1.1 compiler and matching layout
library were transferred with current source into a new Tajasaurus scratch
checkout, preserving the prior proof. Live hardware is RX 9070 XT/gfx1201,
UUID GPU-28d9e7efbf2ef716. Compiler SHA256:
65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
Native whole-Graph partition plus public NVFP4 resident JIT checks pass:
27 passed, one timeout-plugin configuration warning. The first collection
attempt lacked a shared benchmark helper; the unchanged selection passes
after transferring that source dependency. No test was weakened or skipped.
Six JIT/portable profiles pass independent conversion/storage/folded float64
numerics before and after timing; maximum absolute output error 0.0.
Converter/storage/consumer event windows, graph-dispatch and cold/warm/portable
wall times are separate. No kernel speedup or selector promotion is claimed.
This checks gfx1201 NVFP4 parity after the shared native SM120 dynamic changes;
it does not establish ROCm dynamic producer capacity, wider layouts,
model-quality acceptance, general AD or complete five-slice closure.
Evidence: benchmarks/baselines/gfx1201_shared_native_revalidation_20261008/README.md.
