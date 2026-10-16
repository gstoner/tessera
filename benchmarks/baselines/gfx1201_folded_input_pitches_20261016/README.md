# gfx1201 folded input-pitch attribution

Owner: ROCM-MXFP4-W4A8-1 / LAYOUT-ALG-1.
Synchronization key: GFX1201-FOLDED-INPUT-VIEWS-20261016.

## Source and execution scope

Frozen product/recorder source: 920bdadc4212fb89a22bb65dc05cffd169af6bf1.
Every raw packet records equal source-before/source-after hashes and the
compiler binary fingerprint. The selected device was queried as
AMD Radeon RX 9070 XT / gfx1201. These are WSL2 results, not hardware-counter
or per-phase attribution.

The opt-in Graph tensor.extract_slice route carries dense parent storage
through native Schedule/Tile, the distinct pitched Target ABI, LLVM/ROCDL
and HSACO. All four A/B padding combinations and all ten measured shapes
use exactly one pitched HSACO:
f80b91072c88878e2be6193b83e5ca72a6d59b033da0859a3b05ad31aba1b1fc.
Each shape also has an existing contiguous-package control, byte-identical
to the immutable native reference compiler output.

Seven alternating trials were run in both shape/arm orders, with three
rotating resident copies per arm. Captured bracketed device-clock windows
have independent HIP-event witnesses; plain HIP-event windows are retained.
All three outputs were poisoned and checked for both capture forms before
timing, checked after timing, and compared fully against the contiguous
control plus sampled independent decoded checkpoint math. Three separate
checked-public-launch wall samples per arm include allocation, upload,
validation, execution, download and module lifetime.

## Results

| M,N,K | B-only / dense, forward | B-only / dense, reverse |
| --- | ---: | ---: |
| 256,8192,4032 | 1.0129 | 1.0121 |
| 256,8192,4096 | 0.5551 | 0.5614 |
| 256,8192,4160 | 1.0162 | 1.0134 |
| 256,8192,5120 | 0.9507 | 0.9697 |
| 256,8192,8192 | 0.8018 | 0.7898 |
| 256,16384,4032 | 1.0111 | 1.0082 |
| 256,16384,4096 | 0.6847 | 0.7178 |
| 256,16384,4160 | 1.0077 | 1.0070 |
| 256,16384,5120 | 0.9932 | 0.9937 |
| 256,16384,8192 | 0.8909 | 0.8908 |

Ratios below one favor B-only padding. At K4096, B-only padding reduces device
windows by about 44% at N8192 and 28–32% at N16384. A-only padding makes these
cases slower; neighboring K4032/4160 do not benefit. Long-K effects differ:
K8192 improves about 11–21%, while N16384/K5120 improves less than 1%.

The experiment changes row pitches and physical span sizes under identical
kernel bytes. It isolates an input-address/layout contribution, not a cache
partition, cache-hit rate, epilogue cost or LDS counter mechanism. Physical
sizes, resources, exact ISA hashes, launch geometry and all raw samples are
retained. The pitched image has 177 VGPRs, 33 SGPRs, 25,600 LDS bytes and no
spills; the existing contiguous control retains its original bytes.

**No default promotion:** checked public launch wall time did not improve.
Padding may add logical-weight validation/copy overhead. The broader M256
per-column gap, especially K5120/N16384, remains open. A prepared/resident
public route and wider layout/format controls are required before adopting
padding as an optimization.

## Existing-route controls and historical scope

legacy-image-parity.json proves exact Tile, Target, entry, ABI and HSACO
equality for all 22 prior CU-policy shapes and all 12 FP8/MXFP8 KN/NK controls.
These are compiled-image controls, not new format performance measurements.
The prior CU timing packet remains tied to its own 68fb44443 source; its
new source snapshot preserves the actual historical file bytes and digests.
No old timing is relabeled as a measurement of the new compiler.

## Reproduction

Use host WSL with the owning gfx1201 visible and a matching compiler build.
Run from the repository root. Supply explicit compiler/toolchain paths:

    python -m benchmarks.rocm.benchmark_folded_input_pitches --tessera-opt PATH --reference-opt FROZEN_PATH --llvm-bin LLVM_BIN --trials 7 --output forward.json
    python -m benchmarks.rocm.benchmark_folded_input_pitches --tessera-opt PATH --reference-opt FROZEN_PATH --llvm-bin LLVM_BIN --trials 7 --reverse --output reverse.json

For the long-K controls add --n 8192 --n 16384 --k 5120 --k 8192.
The frozen reference compiler SHA256 is
4e5eeed3ea09f496d6f67a144338ae647281618a2d0503e572ba9fbf7139fb6a.
Its age warning is expected: it is an immutable comparison arm, never the
matching compiler for the new route. The recorder requires identical
contiguous output images before admitting each comparison.

Recorder ownership:

| Recorder | Outputs |
| --- | --- |
| benchmarks/rocm/benchmark_folded_input_pitches.py | forward/reverse and long-forward/long-reverse JSON |
| verify_legacy_images.py | post-canonicalization-image-parity.json and log |

The diagnostic smoke run is not included in the published timing set.

## Post-measurement canonicalization proof

Canonicalization hardening is source b50c2cc1349bc7406b7d7ef0e6c1fa40ec2183fa,
separate from the 920bdadc4 timing source. The post-canonicalization image
packet records unchanged source/compiler witnesses, all 22 legacy CU and
12 format identities, and 80 pitched variants (ten shapes, four pitch
combinations, canonicalized and uncanonicalized). Every pitched variant
retains the exact measured f80b9107 HSACO. This is an image-equivalence proof;
the timing packets and their source fingerprints remain unchanged.

## Final dense-layout boundary review

Source f0d6fb7e5f1d3547575d5219650415e4d1c45fbd rejects encoded physical roots,
logical views, scale planes and results before dense bufferization. Seven
red-to-green regressions cover that boundary. The corrected source retains
all 22 CU, 12 format and 80 pitched/canonicalized image identities; the compact
equivalence certificate references the retained earlier rows, while the
compressed raw observation keeps the final source/compiler hashes.
48 owning-gfx1201 regression tests pass with this compiler; 343 focused
view/evidence/diagnostic/pass checks pass. Their raw logs are retained.

Validation history is explicit: the 528b902a5 serial sweep was intentionally
interrupted for the encoding correction, with its original receipt, partial
log and reason preserved. The f0d6fb7e5 four-worker sweep completed with
28,594 passes, 5,873 skips and three failures, all caused by the new
specification's missing date. The document now carries last_updated, without
an allow-list exception. The repaired date/audit/view gate passes 58 tests.
These are historical, source-bound results, not a claim that the final full
gate has passed. The new native-view tests also join the required production
compiler-route lane and reject a missing explicitly configured compiler.

## Final local release gate

Frozen 6a9610363c2e70632b905deb0d57eb96ff143843 completes the full non-slow
unit lane: 28,603 passed, 5,873 skipped, 11 warnings. Source/working-tree,
native compiler and product-tree witnesses are unchanged before/after.
The raw compressed log and receipt preserve this exact source scope.

The required host-free production compiler-route lane passes 856 cases;
the CI execution checker confirms every selected test executed. The owning
gfx1201 lane passes 49 numerical/admission regressions, now including pitched
CU mode and K32 fragment groups. The final native preflight rejects misplaced
scaled-contract tags without assertions while preserving empty-tag ordinary
matmul. All 22 legacy CU, 12 format and 80 pitched/canonicalized image
identities remain unchanged. All 33 generated documents are in sync.

This green receipt supersedes the historical undated-document failure for
its own source only. Later publication commits add evidence/docs/consumers;
they do not change production compiler/runtime/recorder sources. Required
GitHub checks still gate merge.
