---
last_updated: 2026-10-15
---

# Native gfx1201 folded fragment scheduling and device-cost attribution

Owner: ROCM-MXFP4-W4A8-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: GFX1201-FOLDED-FRAGMENT-GROUPS-20261015.

## Current result

The explicit approximate folded route executes Graph -> Schedule -> Tile ->
ROCm Target -> ROCDL/LLVM -> HSACO through the checked ABI. Native Tile-to-ROCm
chooses the instruction schedule; Python reports the Target decision and binds
images. No Python backend math or Tile constructor implements this optimization.

The measured policy selects K32 fragment-read groups for full M256 multiples,
N>=8192 divisible by 4096, and K>=1024 divisible by 64. One native predicate owns
selection and static admission. Other legal shapes retain their native baseline
route and images. There is no automatic logical-MXFP4 approximation promotion.

The optimization improves selected regular-grid medians modestly, but does not
close the independent-reference gap. At M256/K5120 and N8192/16384, captured
native execution remains about 20-22 percent slower than the pinned Radiance
reference. Python submission gaps therefore do not explain the remaining cost.
Native address-generation and alias proofs are the next investigation.

Contract: docs/spec/ir/gfx1201_folded_fragment_schedule.md.
All four backend queues distinguish this gfx1201 change from sibling execution.

## Source-scoped validation

| Frozen source | Scope | Result |
| --- | --- | --- |
| 3b2fa3b70 | Native regular-grid compiler contract and numerical execution | 458 focused checks, 44 RX9070XT device tests, 19 selection/image checks, four negative admissions and two image-reuse classes |
| 3b2fa3b70 | Representative FP8/MXFP8 KN/NK packages | Twelve Tile/Target/image/entry/ABI comparisons are byte-identical to the immutable baseline |
| 3b2fa3b70 | Native instruction scheduling characterization | Eight shapes, forward/reversed compiler roles and shape order, seven alternating trials |
| 7ac37d9e2 | Optional reference recorder and metadata | Thirty host-free argument/census/documentation checks; explicit-stream all-three-copy capture and independent oracle parity |
| 7ac37d9e2 | Independent captured comparison | M256/K5120, three N sizes, both compiler/shape orders, seven trials; reference epilogue on/off, unchanged source/tools/module |

The frozen validation JSON records exact commands, live GPU, source/tool hashes
and terminal results. Source stays clean and unchanged through each run.
Historical 221-device format checks and earlier focused logs retain their
original scope; they are not a full-suite certificate for later source.
Broad regression, coordinating-main synchronization and publication CI remain.

## Captured independent-reference curve

Actual device: AMD Radeon RX 9070 XT / gfx1201, host WSL.
Reference module SHA256:
c9f91bc882ad39f331769043b4d51f9bd3813af48a37b4f3ee5f640598eac3a6.
Declared upstream: dfdfa3832922c9a4253133f09c1f5c0d39748fc7.
The module bytes are authoritative; a source revision declaration alone is not
a build attestation. Dispatch tuning inputs are recorded explicitly.

| M,N,K | Native/reference graph ratio, forward | Reverse |
| --- | ---: | ---: |
| 256,4096,5120 | 1.1160 | 1.1076 |
| 256,8192,5120 | 1.2038 | 1.2045 |
| 256,16384,5120 | 1.2180 | 1.2145 |

| N8192->16384 marginal cost | Forward ns/column | Reverse ns/column |
| --- | ---: | ---: |
| Current native | 16.217 | 16.293 |
| Immutable native baseline | 16.774 | 16.739 |
| Independent reference | 13.141 | 13.291 |

All engines match full BF16 output bits and the sampled independent FP64 oracle
on identical losslessly foldable quantized inputs. These are not full-model
quality results or proof that arbitrary folded weights equal their source.

## Timing domains and control limits

- Ordinary device-clock windows include CPU submission gaps.
- Captured graph windows include GPU graph dispatch, kernels and optional
  markers; capture/instantiation and repeated Python submission are excluded.
- Completed public Tessera launch time includes transfers, device allocation
  and module loading, but is separate from frontend compilation.
- Reference rows have an empty public-API sample list and null median; no
  Tessera public reference API is invented.
- Every resident output copy is poisoned before bracketed/plain captures.
  All three copies are checked before and after timing without a corrective
  relaunch.
- Two native compiler roles and shape order are reversed. Raw executable
  hashes are separate from common compiler/toolchain fingerprints.
- Identical-image controls vary by up to about 3.6 percent across the retained
  runs. Individual small gains are characterization, not per-shape guarantees.
- There are no hardware-counter or dynamically weighted stall measurements.

The reference epilogue experiment changes only RADIANCE_MXFP4_EPIFAST from 1
to 0 in fresh processes. Outputs and native MLIR images remain identical.
Legacy HIP control ELF bytes differ between recompiles, while selected ISA
streams and outputs match; the packet preserves that distinction.

Disabling the reference address-fast epilogue makes it 5.3-8.6 percent slower.
Its marginal cost rises to 13.763 / 13.965 ns per column. This prices part of
the reference optimization, not the whole native gap. The native Tile epilogue
already guards wide recovery and specifies its own FP32 multiply order and
zero/nonfinite rules. Neither may be replaced by the reference arithmetic.

## Timed native ISA attribution

The exported images are hash-matched to the actual native timing packet before
disassembly. In the single WMMA-containing backward branch region:

| Static property | Current | Baseline |
| --- | ---: | ---: |
| Instructions | 95 | 100 |
| DS waits | 7 | 11 |
| WMMAs | 32 | 32 |
| LDS reads | 12 | 12 |
| Barrier signal/wait pairs | 2 | 2 |
| VGPR / SGPR | 177 / 29 | 177 / 29 |
| LDS bytes | 25,600 | 25,600 |
| Spills | 0 | 0 |

The full symbol retains three barrier pairs, including the prologue.
This is static instruction evidence, not a priced barrier/stall explanation.

## Rejected policies and retained evidence

The original N>=4096 policy regresses M256/N4096/K1024 by 4.35 percent in
both orders, with regressions also at K2048/3072. A later minimum-N8192/full-N64
policy regresses N8256/K1088 by 2.4-4.0 percent. Independent N/K trials show
that K alignment or a universal long-K rule does not explain the effect:
odd K64 slab counts can improve at regular N, and irregular N can regress even
at long K. The final policy preserves baseline images for those counterexamples.

- regular-grid-frozen-validation.json and regular-grid-policy-checks.json:
  native implementation, source scope and all fourteen excluded images.
- regular-grid-decision.json and m256-slope-decision.json: role-normalized
  native-only observations with unchanged-image controls.
- radiance-curve-decision.json and the two radiance frozen validation files:
  current independent-reference curves and epilogue controls.
- regular-grid-isa-census.json and compressed assembly: exact timed-image
  instructions and resources.
- counterexample-decision.json: broader policies that were not promoted.
- compressed-evidence.json: stored and uncompressed SHA256 identities for
  deterministic gzip raw packets. Read with Python gzip.open(path, "rt").

## Reproduce

Recorder: benchmarks/rocm/record_gfx1201_folded_native_package.py.
Configure absolute tool/module paths and a task scratch output directory,
then run from the repository root on the owning gfx1201:

    export RADIANCE_MXFP4_WPERM=1 RADIANCE_MXFP4_EPIFAST=1
    export RADIANCE_MXFP4_TN4_MIN_M=2048 RADIANCE_MXFP4_DECODE_MAX_M=0 RADIANCE_MXFP4_DECODE_NT=0
    python benchmarks/rocm/record_gfx1201_folded_native_package.py \
      --tessera-opt "$TESSERA_OPT" --native-reference-opt "$FROZEN_REFERENCE_OPT" \
      --llvm-bin "$TESSERA_LLVM_BIN" --hip-graph-windows --trials 7 \
      --radiance-module "$RADIANCE_REFERENCE_SO" --radiance-revision "$RADIANCE_REFERENCE_REV" \
      --case prefill:256x4096x5120 --case prefill:256x8192x5120 \
      --case prefill:256x16384x5120 --output "$TASK_SCRATCH/reference-curve.json"

Reverse the two compiler paths and shape order for the control run. Compare
named native/reference engines rather than the recorder's printed native/HIP
ratio. Keep the old baseline immutable; its stale-source warning is expected
in this controlled comparison. No sibling hardware proof is inferred.
