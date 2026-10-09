# gfx1201 native scaled-product reshape carriers

Owners: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SCALED-RESHAPE-CARRIER-20261009.

## Executed contract

Textual frontend flat f32 A/B/scales pass through actual Graph reshape nodes,
native primal/JVP/reverse outlining, Schedule/Tile structured carriers, ROCm
Target IR, LLVM and HSACO. C++ owns all intermediates and performs pitched
input byte packing. Python supplies metadata, roots and returned buffers.

The named product has M=3, N=5, K=7, block=[2,4] with ragged K/column groups.
All four continuous operands may be active; vector, (5,3) reshaped-matrix and
(1,3,5) singleton-prefix output shapes preserve flat element order.
Compact and positive pitched vector roots execute. Capacity, dtype, logical
byte equality, native geometry, cotangent lineage and output lifetimes are
checked. The vector-span opt-in retains matrix-only defaults for other users.

## Evidence

Tajasaurus RX 9070 XT gfx1201, matching LLVM/MLIR 23.1.1 and a freshly built
checked HIP owner: 18 public numerical/warm-replay cases pass. Independent
float64 block products, four-role JVPs and transpose oracles validate changed
inputs/seeds and retained outputs; warm eager/compiler calls are forbidden.

Compiler/host groups: 18 initial native primal/JVP/reverse packages;
53 reshape/vector-packer checks; 488 adjacent broadcast/continuous SSA cases;
81 shared trace/registration/span checks passed, 23 skipped. Ruff and mypy pass.
These groups overlap and are not a unique-test total.

Recorder: benchmarks/rocm/record_scaled_reshape_carrier.py.
Nine arms compare independent oracles before/after every timing window.
The packet queries actual GPU name/UUID and binds twelve source hashes,
compiler/runtime hashes, native manifests and every native image digest.
Native event program and grouped member timings are separate from warm public
wall times. Grouped repeated members are not additive interleaved program time.
This is functional characterization without a speedup/default-promotion claim.

## Remaining

Generic scaled batching/transpose closure, dynamic shapes, encoded byte
derivatives, broader aliases/layouts and sibling physical AD execution remain
open. This static continuous carrier migration does not close those programs.
FP8/MXFP8/MXFP4 promotion gates and the original five-slice goals remain intact.
