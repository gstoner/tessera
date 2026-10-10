# Canonical ROCm movement route admission

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key ROCM-MOVEMENT-SPINE-2026-10-05.

## Integration fixed

Direct package functions had native Schedule ancestry, but the main compiler
bundle did not retain it. The driver now admits the original typed physical-page
read or token-gather Graph, retains the native Schedule artifact, and binds
packaging to that exact caller Graph and target. Graph, Schedule, Tile, Target
and backend artifact digests are adjacent, with native pass producers.

Canonical gate identity now uses the existing op catalog normalization:
tessera.kv_cache.read maps to registered kv_cache_read instead of an
unregistered dotted tail. This applies consistently to primary and component
operations. gfx1151/gfx1201 movement capabilities, numerical fixtures, exact
manifest rows and the checked descriptor execution-matrix adapter agree.
gfx1151 is removed from the wholly unimplemented target list. No new semantic
op, dtype, target, pass, diagnostic or GPU body is introduced.

Default native admission includes bounded gfx1201 paged reads; explicit
package_native=False remains respected. gfx1151 token gather uses its existing
native admission, now with a complete canonical artifact chain. Native host
staging is the separately tested C++ service from the preceding packet.

## Exact-device proof

AMD Radeon 8060S / gfx1151 and RX 9070 XT / gfx1201 both execute the canonical
compiler result and checked runtime projection. Static PLHD f32 pages with
i32 logical page tables use explicit contiguous valid start/end ranges.
gfx1151 MoE uses static f32[T,H] with explicit i32[S] token-of-slot indices.
This is local movement, not distributed DispatchPlan transport.

Every final row passes an independent NumPy oracle bit-exactly before and
after each trial. Native counters verify zero warm allocations/frees and three
buffer reuses per compiled call. Source and compiler/library fingerprints,
Graph/Schedule/Tile/Target/backend snapshots and HSACOs are retained.

## Retained production-route comparison

Ten alternating-order warm trials, ten whole calls per arm, compare the
canonical compiled result with the established retained paged-read helper or
retained flat row gather. All nine envelopes satisfy the existing 10% host
non-regression threshold. Historical failed descriptor measurements remain
unchanged; these packets record the new native orchestration and complete
canonical route.

| Architecture | Family | Shape | Retained wall ms | Compiled wall ms | Retained / compiled |
| --- | --- | --- | ---: | ---: | ---: |
| gfx1151 | paged_kv | [4, 4, 3, 8, 1, 5] | 0.6388 | 0.5303 | 1.205× |
| gfx1151 | paged_kv | [32, 16, 4, 64, 3, 31] | 1.3797 | 1.2699 | 1.086× |
| gfx1151 | paged_kv | [64, 32, 8, 128, 7, 249] | 3.2515 | 2.5226 | 1.289× |
| gfx1151 | moe_dispatch | [7, 9, 13] | 1.7935 | 0.5289 | 3.391× |
| gfx1151 | moe_dispatch | [64, 128, 256] | 3.9470 | 1.2239 | 3.225× |
| gfx1151 | moe_dispatch | [512, 768, 1024] | 8.0570 | 2.3961 | 3.363× |
| gfx1201 | paged_kv | [4, 4, 3, 8, 1, 5] | 0.6310 | 0.5210 | 1.211× |
| gfx1201 | paged_kv | [32, 16, 4, 64, 3, 31] | 1.6055 | 1.5081 | 1.065× |
| gfx1201 | paged_kv | [64, 32, 8, 128, 7, 249] | 3.8632 | 2.8684 | 1.347× |

The retained page helper performs its existing native warmup/event submission
even in a normal call; the compiler performs one checked operation. These are
the actual warm API call costs. Compilation is outside the measured windows.
Host wall results include checks, transfer, dispatch, completion and download;
they do not establish application-wide speedups.

Resident events are diagnostic and separately stored. The retained page helper
batches submissions in native HIP; the descriptor diagnostic submits from
Python. Their differing host dispatch overhead prevents treating that ratio
as an isolated kernel-speedup or resident-kernel admission certificate.

## Tests and scope

Owning-device spine tests: gfx1151 eight passed, one other-architecture skip;
gfx1201 seven passed, two other-architecture skips. Graph/target substitution,
Schedule/Tile tampering and caller-graph mutation are refused. Explicit native
opt-out and unavailable toolchain admission remain tested. 541 shared canonical, operation/dtype, capability, manifest, conformance,
execution-matrix, native artifact/lifetime and metadata tests pass; five
owning-hardware/tool skips are recorded in registry.txt. Initial missing registry rows and generated/lifecycle drift
failures are preserved before their corrected runs.

Python remains the frontend, Graph projection and thin package/runtime
binding. The GPU body follows native MLIR/LLVM lowering. The new executor
adapter invokes the existing image/descriptor validation, including exact
target/ABI and buffer guards.

Remaining: public Python tensor-call shape/eager contracts, general page
layouts/arbitrary token maps, asynchronous/resident APIs and controlled native
batch event measurements. Existing DispatchPlan transports and general
retained helpers are not retired by this bounded admission. Apple/NVIDIA/x86
have no physical movement change or transferred ROCm timings; shared catalog
name normalization is covered by canonical host tests.
Full five-slice compiler closure remains open.

Graphify refresh is unavailable in the authoritative WSL scratch checkout:
the graphify CLI is not installed there. No refreshed graph evidence is claimed.
