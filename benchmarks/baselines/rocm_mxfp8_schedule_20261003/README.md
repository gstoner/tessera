# gfx1201 MXFP8 native compiler integration

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-SCHEDULE-2026-10-03.

The textual frontend now carries FP8 E4M3 operands and raw signless-i8 E8M0
scales through verified native Graph -> Schedule -> Tile -> Target -> ROCDL/LLVM
passes into an HSACO. Native Schedule derives separate KN and NK physical
contracts with K32 groups and per-column RHS scales. Typed fragments retain
an isolated K32 partial and native E8M0 scale evaluation in f64 before the f32
join. The ABI identifier explicitly names wide_scale and differs from FP8's
fp32-scale package ABI. The integration seed uses one 16x16 wave; no measured
FP8 or folded-MXFP4 physical schedule is transferred.

Twelve RX 9070 XT/gfx1201 cases pass exact numerical comparisons:
M/N/K=17/19/64 16/16/128 and 16/2/64, KN/NK weight storage, f32/bf16 output.
Nonuniform scale groups and reciprocal code-0/code-254 pairs are included.
Operands are quantized small integers, making each K32 partial exact so group
order and scale indexing are directly checked. BF16 is rounded at the final
store and compared bitwise. Each native image is disassembled and required
to contain the RDNA4 FP8 WMMA instruction. numerical.json records device,
compiler/source/image fingerprints; the complete IR stages are retained.

Four compiler boundary FileCheck checks passed, plus eight malformed-contract
cases for buffer dtype/extent, N/K grouping, granularity, numerical mode and explicit fp32 accumulator policy.
The combined device/registry/regression suite passed 608 tests with one legacy
production-lane test skipped because libtessera_rocm_gemm.so was unavailable.
The integrated shared-pass source preserves the newer NVIDIA producer work.

Remaining: a production checked package descriptor, runtime.launch binding and
validation, image cache identity and kernel/end-to-end benchmarks. Execution
here uses a raw HIP diagnostic launcher; it is not checked production runtime
evidence. Dynamic or partial trailing K32 groups, performance promotion,
canonical MXFP8 storage registration and sibling architectures remain open.
FP8, MXFP8 and MXFP4 still require separate matched evaluation before selecting
a persistent or short/long strategy.

Reproduce the numerical/IR packet with
python benchmarks/rocm/record_gfx1201_mxfp8_native_boundary.py --output-dir PATH,
using the matching compiler and gfx1201 validation environment.

Sibling regression: Super-Bear RTX 5070 (CUDA 13.3, compute capability 12.0)
passed 86 existing scheduled matmul/attention device cases after the integrated
compiler build finished. Shared registry gates passed 322 tests. An earlier
NVIDIA run overlapped linking and hit nine PermissionError cases; it is not
counted as validation. The complete fresh run is nvidia-regression-tests.txt.
This proves existing NVIDIA routes retained parity, not MXFP8 support on CUDA.

Follow-on: [checked runtime package, image reuse and matched timing](../rocm_mxfp8_checked_package_20261003/README.md) now proves checked launch integration. This packet retains the earlier raw-diagnostic boundary and its original receipts.
