# SM120 resident public attention JVP

Synchronization key: SM120-RESIDENT-ATTENTION-JVP-20261009.
Owners: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.

Public native_jvp accepts compact rank-four FP32 CUDA Q/K/V and active tangent
roots. Abstract metadata capture produces typed semantic Graph IR; existing
native MLIR AD, Schedule/Tile, Target and LLVM/PTX packages supply forward and
saved-LSE JVP images. No resident input is coerced to a host array or numerically
evaluated by the Python frontend.

The structural certificate records zero concrete frontend executions and
requires physical-package numerical authority. It compares retained AST and
tracer topology/policy and pins the typed Graph. It is not a numerical frontend
differential certificate.

The prepared C++ owner checks allocation context, device memory type, extent,
alignment, private-arena aliasing and producer stream context/count. Events
order producer writes before device-to-device snapshots into private storage.
Private forward output/LSE and active tangent snapshots remain owned until JVP
completes. Calls return independent host output arrays after synchronization.
Borrowed roots must remain live throughout the call; concurrent future writes
are not automatically ordered by this synchronous contract.

## Proof

101 adjacent host tests passed before four additional negative certificate
cases; those four plus the resident host matrix passed (42 tests).
43 RTX5070 device cases passed: all six argument permutations, K=5/129, causal
GQA, single/all active tangent roles, delayed producer writes, held outputs,
allocation/alignment/stream-count rejection and recovery, and adjacent host-
upload prepared attention regressions. Numerical forward reference uses
independent float64 attention; JVP uses central finite differences.
The two packets have maximum absolute error below 2.6e-8.
Ruff and mypy ratchet passed; final registry/document gates are recorded in the PR.

## Timing

Run1/run2 are isolated processes with matching source/compiler/provider pins.
Each contains four programs, nine alternating resident/host timing rounds,
actual RTX5070 UUID/SM120/driver and numerical checks before timing.

Resident completed public calls span 1.720–1.946 ms; host calls span
1.381–1.572 ms. Resident/host median ratios span 1.213–1.293.
Resident forward kernel medians span 11.14–54.59 us and JVP medians 3.58–13.38 us.
CUDA events exclude producer waits and input snapshots; public wall time
includes certificate/binding, snapshots, native calls and host downloads.
This establishes execution coverage, not a resident speedup. Profiling
frontend certificate/binding overhead remains follow-up work.

Reproduce on Super-Bear with matching compiler and prepared native provider:
python -m benchmarks.nvidia.record_resident_attention_jvp --output /scratch/run.json

## Remaining scope

Resident ordinary tuple forward and public reverse AD, dynamic shapes, mixed
host/device roots, pitched storage, composed/nested attention AD and automatic
external future-write tracking remain open. This does not close the five-slice
program or transfer CUDA evidence to gfx1151/gfx1201, Apple or x86.
