# gfx1201 FP8 shape-independent physical image characterization

Owner: E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: ROCM-FP8-TARGET-IDENTITY-20261014.

## Evidence

[gfx1201.json](gfx1201.json) records 60 cases: all four E4M3/E5M2
operand pairings, register K-unroll 1/2/4/8 and LDS K-unroll 1, each at
M/K/N = 16/32/16, 17/31/19 and 48/33/16. Tajasaurus exposes gfx1201
(RX 9070 XT); the packet contains the live architecture and full GPU report.
Every case passes an independent FP32 oracle before timing. Maximum
absolute error is zero for these seeded, finite quantized inputs.

Twenty physical images cover the twenty pairing/recipe combinations.
Each image is reused for its three shapes; new-shape packages report
warm_cache. Target identity includes both storage formats; compiler,
device libraries, unroll and staging configuration remain cache inputs.
Schedule/Tile hashes remain shape-bound and are recorded independently.

The full source-pinned recorder run has new-shape package wall times
33.35–55.50 ms, HIP-event batch means 3.68–13.00 us, and completed host
launch medians 0.567–0.747 ms. These characterize different regions.
The event batches can include CPU enqueue gaps. They are not pure-kernel
latencies and no speedup, default schedule selection or broad closure is claimed.

Twelve default-register rows also time ordinary textual FP8 @jit calls.
Every public call reports native_gpu through the canonical descriptor;
warm completed calls are 0.851–0.904 ms. First public calls follow same-case
image priming and are not cold compiler-process measurements. Two public
tests cover both generic rocm and explicit rocm_gfx1201 targets, keyword
binding, all format pairs and three shapes, plus unchanged inputs and
retained outputs. FP16/BF16 public sibling calls also pass.

The final packet uses a matching LLVM/MLIR 23.1.1 compiler with NVIDIA,
ROCm, x86, EBM and Clifford enabled. The selected HIP device is checked
at the generic native launch boundary after argument validation and before
module acquisition or allocations.

[gfx1151.json](gfx1151.json) is a separate Princess-Luna proof: six
FP16/BF16 LDS cases, maximum error 4.77e-7, two reused physical images and
a relabeled gfx1201 image refused on the live gfx1151 device. Packages were
cross-compiled on Tajasaurus and executed with the pinned candidate runtime.
[sibling_packages.json](sibling_packages.json) stores each image only once.

## Compiler boundary

Typed Graph IR -> native Schedule replay -> Tile -> ROCm Target
-> native WMMA fragment lowering -> LLVM/AMDGPU HSACO -> checked launch ABI.
Python constructs a typed Graph and binds buffers; it emits no GPU arithmetic.
The Target directive carries rhs_storage. The LDS body independently preserves
RHS scalar, vector, shared allocation and fragment types. Register launches
use 32 threads; LDS uses its descriptor's 128-thread block and block-tile grid.

The JSON pins compiler/loader binaries, source files, Target/Tile/HSACO hashes,
entry symbols, ABIs, grids and workgroups. The source revision is the merged
base; source hashes bind the working implementation used for measurement.

## Reproduction

From the repository root in owning gfx1201 WSL, set TESSERA_OPT,
TESSERA_ROCM_OPT, LLVM_BIN, TESSERA_ROCM_CHIP=gfx1201 and the native image
loader provider, then run:

    python benchmarks/rocm/benchmark_rocm_fp8_shape_cache.py --output /path/to/packet.json

Defaults are five event samples of 100 resident launches, ten warmup launches
and five completed-host samples of ten launches. Packages are created once
per shape in a fresh process. Input quantization precedes oracle construction.

## Validation and remaining work

577 focused dtype/diagnostic/pass/LDS/cache gates pass under the five-backend configuration.
83 guard/dispatch gates pass after the live-architecture correction.
The final available-lane regression group passes 230 tests (113 skips),
including the available AVX-512 native runtime cases. Focused mypy is clean. The numerical test
matrix also exercises FP16/BF16 homogeneous recipes (90 device launches total).
82 native cache/Target tests passed before adding those sibling storage cases.

Larger shapes, wider dynamic/layout envelopes, split-K FP8, general attention
image keys and public Graph/Schedule admission cost remain open. Ordinary textual FP8 @jit now reaches the same native package path.
The integer public API keeps its existing wrapping dtype semantics; the
separate i32 package profile is not silently substituted. No gfx1151,
NVIDIA, Apple or x86 device result is inferred from this packet.


### Reproduce the gfx1151 sibling proof

On a matching native compiler host:

    python -c 'from pathlib import Path; from benchmarks.rocm.benchmark_rocm_fp8_shape_cache import export_gfx1151_lds_packages; export_gfx1151_lds_packages(Path("packages.json"))'

Transfer packages.json and the candidate Python package to owning gfx1151 WSL,
set PYTHONPATH and the ROCm environment, then run:

    python benchmarks/rocm/execute_rocm_lds_sibling.py --packages packages.json --output gfx1151.json

The executor verifies numerical agreement, unchanged inputs, one image per
storage across shapes, and refusal of a relabeled image. It makes no timing claim.
