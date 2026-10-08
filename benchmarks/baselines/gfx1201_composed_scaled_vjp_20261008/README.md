# Composed public scale VJP — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key GFX1201-COMPOSED-SCALED-VJP-2026-10-08.

The real frontend traces two exact-per-block FP8 E4M3FN scaled products plus
add. Independent FP32 scales and a LHS scale shared across the two products
execute through public native_backward and verified native paired AD.
The exporter derives the primal/cotangent frame from the forward signature,
selects requested/reordered scale roles, outlines actual reduction SSA and
preserves shared contributions as native sum members. Strict tensor arith.addf
is reified as registered tessera.add in the outlined member; the unmodified
backward root remains the witness. Schedule/Tile handles each reduction/sum.
No Python backend arithmetic or kernel constructor is added.

The checked program validates captures, selected role ordering, sum lineage,
private scratch lifetime, returned storage, native member witness equality,
image format and launch geometry. Public routing certifies the whole Graph;
package identity includes it. The cotangent slot follows the actual primal
argument count. Encoded matrix-storage derivatives are not admitted.

Validation: 419 host compiler/transpose/plugin/registry tests, 86 owning gfx1201
reverse/forward regression tests, and eight additional wave schedule owning
checks pass. Seven host recorder contention-guard tests pass. Owning pytest
warns that pytest-timeout is absent; no timeout enforcement claim is made.

The packets query the actual AMD Radeon RX 9070 XT and live gfx1201 architecture,
record HIP UUID, source/recorder/oracle/compiler/runtime/image hashes and five
raw windows per timing scope. The identical matching LLVM/MLIR 23.1.1 assertion
compiler SHA is 6075bf1450770b22635f9a22959638e4845287030d716d37c513cd7a8db51eb0.
The transferred linked layout library is separately bound by build-identities.
Independent float64 scale-gradient comparisons pass before and after timing;
maximum absolute error across final serial/wave profiles is below 3.0e-7.

| Shape M,N,K | Shared LHS scale | Serial native event, ms | Wave native event, ms | Serial public host, ms | Wave public host, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 17,19,256 | False | 7.674031 | 0.264343 | 10.009285 | 2.502413 |
| 17,19,256 | True | 7.703217 | 0.271469 | 9.655783 | 2.280451 |
| 3,5,37 | False | 0.313478 | 0.043125 | 2.480759 | 2.322077 |
| 3,5,37 | True | 0.320901 | 0.096156 | 2.341254 | 2.129883 |

Native events cover the complete four-reduction program plus its sum where
needed, with retained prepared storage. Public host walls include copies,
module/owner setup, launch and completion. These fresh-process serial and wave
runs characterize explicit schedules; they are not counterbalanced A/B,
isolated individual-kernel results or a default-selection promotion.
The default remains serial_per_scale_element.

Reproduce benchmarks/rocm/benchmark_composed_scaled_vjp.py --output PATH on
the matching gfx1201 environment. TESSERA_ROCM_SCALE_VJP_SCHEDULE selects
serial_per_scale_element or wave_per_scale_element explicitly.

Build and terminal failure histories are retained: initial fixed-frame export,
native tensor sum admission, Schedule artifact replay and recorder contention
false positive. Subsequent matching-build and owning successes supersede those
specific gaps; no full-unit green claim follows.

Open: broader shape/storage/layout/format evaluation, arbitrary composed AD,
dynamic/nonleading maps, generic batching/transpose closure, FP8/MXFP8/MXFP4
strategy decisions, model-quality acceptance and focused PR delivery.
No gfx1151, NVIDIA, Apple or x86 physical parity follows from this packet.
