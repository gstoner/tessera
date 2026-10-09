# Single-sided resident tensor producers — SM120

Owner: W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-SINGLE-SIDED-RESIDENT-PRODUCERS-20261009.
Stacked on the bounded resident frontend in PR 928.

## Native architectural integration

Public static and bounded JIT now admit
`matmul(producer_chain(source), raw_rhs)` with compact resident CUDA roots.
The raw RHS carries an explicit row-major storage contract; authored conflicting
layouts reject rather than silently changing interpretation.

Original frontend Graph is exported by native MLIR into the existing
Schedule -> Tile -> NVIDIA Target -> LLVM/PTX packages. Native C++ owns the
LHS intermediate and result, orders all input producer streams (including raw
RHS and epilogues), checks allocation context/capacity/aliasing and returns an
independent completed host result. Python performs metadata projection and
binding; no Python arithmetic or kernel construction supplies execution.

Three new ordered LHS exports cover resident invocation, completed host result,
and distinct program/member event profiles. Two-sided DAG exports retain their
own owner contract. Native mode checks reject cross-use before result writes.

A bounded column-major package created from host inputs is retained separately
from its row-major resident package. Shape reuse then retains each physical
contract. Resident-first host transitions reuse the already selected row route.

## Correctness and ownership

Super-Bear: actual NVIDIA GeForce RTX 5070, SM120; CUDA 13.3,
LLVM/MLIR 23.1.1. The core compiler is unchanged; the matching CUDA provider
was rebuilt atomically from the captured sources.

- 542 focused host tests pass, one skips, including runtime ABI, native bounded
  projection, frontend authority and operation/diagnostic/pass registries.
- 44 new owning device tests pass for FP16/BF16; static/bounded RMSNorm,
  LayerNorm, softmax, two/three-stage chains, fused bias/ReLU/residual FP16
  output and reordered arguments.
- 81 adjacent two-sided static/bounded resident and saved-LSE attention JVP
  device regressions pass with the rebuilt provider.
- Raw-RHS delayed writes must complete before launch. Both resident output
  capacity and full-capacity host staging are primed before these tests.
- Warm shape changes forbid compiler/tracer/eager entry. Package and owner
  identity, reported scratch accounting and held outputs remain stable.
- Host-first column-to-row transition creates one distinct resident package,
  then both layouts reuse their native programs across active shapes.
- Direct native tests reject wrong LHS/DAG owners and forged root allocation
  capacity without modifying sentinel host results; valid calls recover.

## Measured evidence

Two isolated fresh processes each record six dtype/producer programs and
30 active frames. Variants are RMSNorm, softmax and a three-stage
LayerNorm/RMSNorm/softmax chain, each in FP16 and BF16.
MNK bounds are (129,65,513); frames are
(17,19,35), (129,65,513), (1,1,1), (63,31,255), then (17,19,35).
Each program retains one contract digest and native plan across frames.

Seven counterbalanced public-call samples and 128-repeat native CUDA-event
windows are recorded for each frame. Grouped producer/consumer event windows
are separately recorded. An independent FP64 stage oracle passes;
resident/host outputs agree bitwise. Worst recorded absolute error: 0.005788.

| Domain | Median range across both packets |
| --- | --- |
| Native whole-program CUDA events | 16.9–41.8 microseconds |
| Resident public completed wall time | 0.493–1.075 milliseconds |
| Host-input public completed wall time | 0.420–1.041 milliseconds |

Resident/host wall ratios span 0.966–1.230, so these measurements do not support
a general resident speedup. Public time includes input ordering, checks and
result download; native events are a distinct timing domain.
Source, tool/provider hashes, actual GPU UUID, native plans and image digests
are captured. Native plans are stored once per program rather than duplicated
in every changing-frame row.

## Remaining scope

Mixed host/device roots, pitched/column-major public resident inputs, arbitrary
producer graphs and resident AD remain open. FP8/MXFP8/MXFP4/NVFP4 physical
schedules and attention kernels are unchanged. CUDA execution supplies no
Apple, gfx1151/gfx1201 or x86 physical parity. W1.1 and the five-slice compiler
program remain open.

Recorder: `benchmarks/nvidia/record_single_sided_resident_tensor.py`.
Device tests: `tests/device/nvidia/test_single_sided_resident_tensor.py`.
Host tests: `tests/unit/test_single_sided_resident_tensor.py`.
