# Dtype arithmetic and mixed-integer ownership — 2026-09-11

Owners: E2E-REAL-6F, E2E-REAL-6, NUMPOL-CARRIER-1, LAYOUT-ALG-1.
Sync: `DTYPE-CODEGEN-2026-09-11`.

`nvidia.json` (RTX 5070 / SM120) and `rocm.json` (Radeon 8060S / gfx1151)
record independent WSL executions of explicit MLIR scalar and two-lane vector
add/subtract/multiply/divide, compiled through LLVM into native device images.
Each has 24 passing rows: fp64, fp32, fp16, bf16 and signed/unsigned 8/16/32/64-bit
integers. Comparison requires exact values, matching signed zero, and matching
NaN classification. The inputs include rounding ties, finite extremes, subnormals,
infinities, NaNs and modular integer overflow. Integer division excludes zero
and signed-minimum/-1, for which the primitive has no defined result.

Image/binding hashes and disassembled instruction inventories bind results to
emitted code. CUDA FP16/BF16 vector rows contain HADD2/HMUL2 variants. gfx1151
FP16 contains packed arithmetic; BF16 expands through fp32 operations and rounding.
These are not interchangeable arithmetic or matrix acceleration claims. Two-lane
IR does not guarantee every operation is implemented by a packed instruction.

Each device has four **compile failures**, not numerical passes: FP8 E4M3/E5M2
scalar/vector byte-storage conversions reach an unresolved conversion cast at
LLVM translation in this generic pipeline. This does not invalidate separately
implemented FP8 matrix/conversion routes. FP4/FP6/INT4 and NV/MX scaled formats
still need their packing/scale-aware producers; complex and bool need their
own semantic probes. Apple has no new execution packet in this increment.

`x86_matmul.json` records nine diagnostic BF16, u8*s8 and FP64 comparisons on
Princess-Luna's Ryzen AI Max+ 395. Packages now carry replay-verified Schedule
identity, including the mixed u8*s8 recipe. The benchmark's historical i8/i8
Graph spelling was corrected to ui8/i8. All timings are host wall time, and
all packets are **ineligible for performance promotion**.

Additional focused regressions execute ragged 3x9x5 and 2x17x19 mixed products
and a 70001-term overflow case against an independent int64/modulo oracle.
A UBSan executable checks positive/negative accumulation and beta overflow in
the scalar reference. It now uses defined unsigned arithmetic to match VNNI's
non-saturating modulo-2^32 semantics.

Reproduce arithmetic with `benchmarks/record_dtype_arithmetic.py --backend
nvidia|rocm --compiler <tessera-opt> --llvm-bin <LLVM23/bin> --artifacts <dir>
--output <json>`. The recorder bypasses general frontend capture; it does not
promote planned unsigned Graph dtypes or claim complete dtype coverage.
Reproduce matrix comparisons with `benchmarks/x86/benchmark_x86_e2e_dtype_matmul.py`.

The arithmetic recorder writes every outcome and exits nonzero if any selected
row fails compilation or numerical comparison. Select a supported dtype subset
with `--dtypes` when using it as a passing regression gate.

`package-census.json` retains the current F0 inventory. A Graph-input wrapper
can delegate to a compiled consumer: x86 `package_matmul` now has no local
emitter path. Input-boundary counts alone therefore do not measure remaining
semantic reconstruction.
