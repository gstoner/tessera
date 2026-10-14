# Named FP32 scaled-product tensor gate — gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: NATIVE-SCALED-TENSOR-GATE-20261014.

The public textual frontend traces a block-[4,4] FP32 scaled matmul followed
by same-shaped FP32 tensor multiplication. Native MLIR owns primal/JVP/VJP
projection, Schedule/Tile members and ROCm LLVM/HSACO generation. The sealed
program checks operand binding, cotangent linearity and last-read lifetimes.
Python binds the package and provides independent test oracles.

Matching LLVM/MLIR 23.1.1 compilers were rebuilt on Super-Bear and Tajasaurus.
Fresh HIP providers were compiled from the current candidate on Tajasaurus.
Eight RX 9070 XT/gfx1201 device cases pass: all operand transpose combinations,
selected/reordered AD roles, independent FP64/finite-difference comparisons,
changed warm values without compiler subprocesses and retained generations.
329 final compiler/program, diagnostic/pass registry and recorder tests pass.

## Timing

Both fresh packets measure the named M=2,K=9,N=5 JVP. Maximum absolute error
is 2.572921015131513e-9. Source and binary pins, program digest and native
member identities are identical across the two runs.

| Domain | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completed public JVP median | 1.113088 ms | 1.069234 ms |
| Interleaved native program event median | 27.633032 us | 27.445782 us |
| Captured member medians | 1.982–2.287 us | 1.981–2.280 us |

Interleaved program events include host dispatch gaps. Captured members use
a distinct grouped device schedule with 2048 repeats. Neither domain is a
completed public-call measurement. Correctness is checked before and after
every timing domain; warm public compilation is forbidden.

Recorder: benchmarks/rocm/record_native_scaled_tensor_gate.py.
Evidence: run1.json and run2.json.

This is latency characterization, with no speedup or route promotion claim.
Encoded FP8/MXFP8/MXFP4 coefficients, mapped/dynamic/wider compositions,
higher-order AD and gfx1151/NVIDIA/Apple/x86 execution remain separate gates.
Existing public capability/manifest claims are unchanged; this proves a
native member inside the named gfx1201 scaled-product program.

## Shared SSA traversal

Native reverse root collection now visits each SSA value once. The depth-32
regression times out after 10 seconds with the pre-repair compiler and passes
with the repaired compiler. Exact exports at depths 8,16,24 are byte-identical.
The named runtime program and image identities also remain unchanged.

Same-host Super-Bear fresh-process export samples at depth 24 are 7.219084 s
before and 0.015252 s after the repair. Depth 32 completes in 0.017687 s.
These single samples characterize the traversal defect; they exclude image
generation and do not claim general compiler, kernel or public-call speedup.
dag-export-attribution.json records compiler/header/input/output hashes;
dag-depth*.mlir preserve exact inputs. dag-control.patch reverses only the
visited-set repair so the control compiler can be rebuilt from this slice.

The full pre-traversal-repair WSL non-slow lane records 26,052 passed,
7,568 skipped and one recorder-naming failure. That tracked-file gate passes
after indexing this packet README. Final repaired-compiler focused gates are
recorded separately; the broad lane is not presented as a fresh final-binary
run. All 15 combined compiler and gfx1201 execution cases pass after repair.
