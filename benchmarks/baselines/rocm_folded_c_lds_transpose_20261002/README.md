# Folded C transpose through a reused LDS slab

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-C-LDS-2026-10-02.

## Architecture and outcome

Two compiler-only prototypes materialize scaled f32 typed fragments into LDS,
then gather contiguous vectors for masked global BF16 stores. A raw byte
allocation has typed FP8/f32 views: the A slab's lifetime ends before C reuses
the same capacity. The padded C rows fit in the existing 20,480-byte A slab.
Scale overflow/underflow recovery precedes the single final BF16 conversion.

The first variant gathers C collectively across waves with workgroup fences.
The retune gives each wave its own C cells and native wave-scoped fences.
All lanes participate, including inactive edge lanes. One extra workgroup
fence retains the A-to-C cross-wave lifetime boundary. This introduces checked
f32 LDS fragment-store support in the experimental native lowering; it checks
memory-space agreement and rejects integer/fused LDS stores.

Both prototypes remain experimental patches, removed from active source.
The wave-private variant reduces physical VGPRs but has not closed short-K
performance. It deserves further BF16 staging/prologue/persistence retunes.
No new Schedule policy, default promotion, or complete compiler gate is claimed.

## Exact-device numerical evidence

Tajasaurus RX 9070 XT, live gfx1201, matching LLVM/MLIR 23.1.1.
The initial FP8-to-f32 memref.view was invalid; tests.txt records that failure.
A byte slab with typed views repaired it. tests-byte-view.txt records 86 passes
and three structural failures expecting the original global tile.store bounds.
Those expectations were preserved. All 38 owning-device tests pass in
tests-wave.txt, including ragged shapes, runtime K reuse, and exceptional
scale recovery. A retained implementation needs an explicit Schedule policy
and structural tests for its LDS/global store boundary before publication.

## Measurements

Seven alternating graph-window trials, three rotating resident copies,
independent sampled oracle, poisoned pre-timing outputs, bitwise native/HIP
agreement, and post-timing rechecks. Marked windows last at least 5 ms and
device-clock/event witnesses agree within 5%. Public launch wall samples
are separate. Graph windows include dispatch/markers; no isolated kernel,
hardware-counter, occupancy, or Radiance comparison is claimed.

### Collective C gather

162 physical VGPRs, 29 SGPRs, 25,600 LDS bytes, no scratch/spills.
Twelve static split-barrier signals and waits.

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 24.004 | 23.038 | 1.0419 |
| 256 x 4096 x 2048 | 35.510 | 35.257 | 1.0072 |
| 256 x 4096 x 5120 | 79.366 | 77.165 | 1.0285 |
| 256 x 8192 x 5120 | 152.600 | 150.841 | 1.0117 |

### Wave-private C gather

145 physical VGPRs, 29 SGPRs, 25,600 LDS bytes, no scratch/spills.
Four static split-barrier signals and waits, versus three in the reference.
The maximum virtual scheduled pressure is 151. Native instruction streams
match the separate LLVM pressure probe. LDS reads/stores and WMMA counts are
recorded in wave-pressure/pressure.json. Static instruction counts are not
dynamic barrier event counts; the checked full-K64 loop executes its two
workgroup pairs for each K64 slab, with fixed prologue/epilogue pairs outside.

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 24.300 | 23.080 | 1.0528 |
| 256 x 4096 x 2048 | 36.619 | 35.283 | 1.0379 |
| 256 x 4096 x 5120 | 78.157 | 77.382 | 1.0100 |
| 256 x 8192 x 5120 | 148.492 | 150.462 | 0.9869 |

## Next engineering gates

Stage BF16 only after checked scale recovery; retain numerical parity.
Overlap first-slab A/B issue in the prologue as a separate ablation.
Persistent tile iteration needs an explicit verified Schedule/Tile contract,
checked worker launch geometry, and repeated device/E2E measurements.
Keep output masks and all barrier participation/lifetime checks intact.
Apple, NVIDIA, x86 and gfx1151 have no new lowering or exact-device proof.

## Restored reference verification

The rebuilt reference passes 89 focused compiler/device checks. Its selected
native instruction-stream SHA256 and physical resources exactly match the
pre-retune baseline. See restored-tests.txt and restored-pressure/pressure.json.
