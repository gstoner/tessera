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

[sm120-shared-control.json](sm120-shared-control.json) records a separate
Super-Bear RTX 5070 regression: 28 scheduled-activation tests pass with the
candidate jit.py/driver.py overlaid on PR947's Python tree and its matching
compiler/runtime. The primary checkout stays unchanged. The receipt pins
both overlaid files, binaries and the test log; it adds no timing claim.


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

### Broad-lane ISTFT profile repair

[istft-profile.json](istft-profile.json) and
[istft-profile-tests.txt](istft-profile-tests.txt) record 63 passing ISTFT
contract cases after four matching-profile cases failed against the original
guard. The public gfx1201 primal/JVP cases compare with the eager reference
and finite differences. Matching compiler-owned profiles are preserved;
conflicting and partial profiles are refused. The exact gfx1201 spectral
provider is built from the pinned existing HIP sources, separate from the
five-backend compiler build. No kernel or ABI changed and no timing is claimed.

### Frozen broad-run integration repair

The first complete gfx1201 host sweep reports 28,418 passes, 5,330 skips and
31 failures; it is not a passing gate. The targeted repair lane passes
81 tests, with 14 architecture/environment skips. Eleven assembly failures
came from the ROCm development serializer; all 25 architecture-fragment
checks pass with MLIR_OPT pointing to matching LLVM/MLIR 23.1.1.
FP32 snapshots now pin their actual gfx1151 capability row. Owning gfx1151
tests check the live GPU before compiling. The cache test stays inside one
macro-K recipe and separately asserts that differing k_blocks remain distinct;
it does not erase physical policy to manufacture reuse. A fresh process tests
global clear/capacity behavior without invalidating another owner's leases.

[gfx1151-control.json](gfx1151-control.json) records three recurrence images
cross-compiled by the current matching compiler and executed on Princess-Luna:
zero maximum error for all three. [gfx1151-control-packages.json](gfx1151-control-packages.json)
retains their byte-identical images, generated-source hashes and launch helper.
This remains independent of the gfx1201 proof and makes no timing claim.
Reproduce on a matching current gfx1151 build with
pytest -q tests/unit/test_rocm_control_for_wmma_exec.py.

[resident-bf16-dynamic-n.json](resident-bf16-dynamic-n.json), bound by
[resident-bf16-manifest.json](resident-bf16-manifest.json), records the repaired
public-traced BF16 bounded-N resident route. Updating the complete inferred
result tuple fixes the previously malformed SSA return without changing
native arithmetic. One M128/K256/N<=256 package handles N128 and N256.
All outputs pass the independent FP32 RMSNorm/materialized-BF16/matmul oracle
before recording 100 samples after 25 warmups. Producer/consumer HIP-event
medians are 8.48/10.76 us for N128 and 11.36/14.36 us for N256.
These are event windows that may include enqueue gaps, not pure kernel or
completed public-call latency, and no speedup/default selection is claimed.
The packet's worktree hash binds its measured working state; the companion's
six source and two compiler hashes bind the code, not later documentation edits.

### Integration onto merged PR947

PR947 is merged at 43c57caac44f3af1e57211bf5575994b37950f79. The ROCm
candidate is rebased onto that main. Existing packets above retain their
original source/binary fingerprints and recorded working-state identities;
they are not relabeled as post-merge measurements. The just-started
old-base full run was deliberately interrupted when this merge unblocked
integration (2,920 passed, 1,865 skipped); it is not a passing full gate.
A matching integrated compiler rebuild and fresh evidence/gates are required.

### Post-merge integrated certificates

[post947-gfx1201.json](post947-gfx1201.json) records the integrated five-backend
LLVM/MLIR 23.1.1 build on revision 94caaf2ddbb61f547609e4925c084fe16790f25e:
60 seeded finite FP8 cases pass with zero maximum error, retaining twenty
pairing/physical-recipe images across three shapes. New-shape warm package
costs span 31.20–49.94 ms; event-window medians span 3.75–12.62 us; completed
host-launch medians span 0.594–0.884 ms. These remain distinct timing domains,
with possible CPU enqueue gaps inside event windows; no speedup or default
promotion is claimed.

[post947-gfx1151.json](post947-gfx1151.json) and its
[package bundle](post947-gfx1151-packages.json) independently bind the fresh
integrated compiler to six FP16/BF16 LDS launches on Princess-Luna. Maximum
error is below 4.8e-7, images are reused across shapes, and a relabeled gfx1201
image is rejected. No gfx1151 FP8 or timing claim is added.

[post947-full-unit.txt](post947-full-unit.txt) records 28,493 passes, 5,320
skips and 874 deselections in 1,121.51 seconds for the frozen integrated
candidate. Its [start manifest](post947-full-unit-manifest.txt) records the
revision, tracked diff digest and toolchain context. This is the broad gate
before the subsequent exact-profile alias and prepared-fixture repairs.
It does not certify later source merely because that source is compatible.

The subsequent native/Python profile repair preserves compiler-owned module
spelling for sm_120 and zen5_avx512 while emitting the existing canonical
ISTFT contract. Other/partial profiles remain rejected. The owning gfx1201
[post-alias gate](post-alias-gates.txt) passes 432 cases, with 37 owning-gfx1151
skips; it includes numerical ISTFT, native JVP, independent FP8 contracts,
cache capacity/lifetime and diagnostic/pass registry checks after rebuilding.
Existing measurement packets retain their actual compiler hashes; they are
not relabeled as measurements of this later profile-only build.

### Owning SM120 integration after profile repair

[sm120-integration.json](sm120-integration.json) and
[its terminal test log](sm120-integration-tests.txt) record 630 passing cases
and six skips on Super-Bear's RTX 5070. Source and compiler/provider hashes,
exact test command, device UUID and LLVM/MLIR 23.1.1 tool paths are recorded.
The custom prefix provides MLIR tools; the matching Ubuntu 23.1.1 companions
provide clang/llc/llvm-link. The explicit owning-device proof flag is enabled.
The only skips are two x86 spectral provider cases, two foreign ROCm cases,
and two rank-two cases with no batch transform.

Coverage includes the repaired public ISTFT primal/JVP finite difference,
new exact-profile alias preservation/refusal tests, scheduled activation,
saved-LSE public forward/backward and trace-at-one replay, resident/prepared
ownership, native NVFP4 and required diagnostic/pass registry gates. Four
prepared reverse JSON/NPZ pairs are compiled in temporary storage and verified
against independent FP64 gradients before replay; tests no longer depend on
missing untracked local numerical fixtures. Earlier tool-path/fixture failures
are repaired without skipping the owning lane or weakening numerical checks.
This final explicit-source certificate supersedes the earlier isolated shared
control receipt for current integration claims. No new timing, speedup or
universal W1.1/attention closure is asserted.
