# gfx1201 folded native CU calibration

Owner: ROCM-MXFP4-W4A8-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: GFX1201-FOLDED-FRAGMENT-GROUPS-20261015.

## Frozen production route

Benchmark source: 68fb444434bbb20f396399e8a1df186c67e00f94.
Current compiler SHA256:
1d709b7513b2be8787c629a5b2f048be77db4968b9077f65081fdb289d9ddae7.
Immutable control compiler SHA256:
4e5eeed3ea09f496d6f67a144338ae647281618a2d0503e572ba9fbf7139fb6a.

The unchanged explicitly approximate folded semantic contract follows native
Graph -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM -> HSACO and a checked
runtime ABI. Native Tile-to-ROCm selects CU for M256 and regular N8192..32768
in steps of 4096, K1472/1536/1600; N8192/K2048 is also selected. Existing
multi-row CU selection remains. This does not promote folded approximation
to the default logical MXFP4 algorithm.

## Captured device evidence

The full 21-shape rectangle plus one K2048 row passes seven alternating trials
in both compiler and shape orders. All selected rows improve: approximately
6.3-18.5 percent lower captured device time than the immutable WGP control.
Each native mode reuses one image across all 22 shapes. Image keys retain
workgroup mode and fragment scheduling, while descriptors bind actual shapes.

Both native arms use identical device root pointers and three rotating output
buffers. Host downloads are independent snapshots. Complete bitwise comparison
and a sampled independent numerical oracle precede timing; all three outputs
are poisoned and checked for bracketed and plain captured graphs. Graphs close
before engine leases, and every final frame report records zero leases and a
closed pool. Public API wall time, ordinary device windows and captured device
windows remain separate.

Nine excluded classes preserve byte-identical Tile/Target/image/entry/ABI:
small and irregular N, K1024, N16384/K2048, K5120, M200, existing M512 and
N36864. Twelve independent FP8/MXFP8 KN/NK controls also preserve identical
native packages. These are compiler controls, not new format speedup claims.

Static CU/WGP resource evidence agrees: 177 VGPRs, 29 SGPRs, 25,600 LDS bytes
and no spills. Selected entry symbols each contain 32 WMMA instructions and
three barrier signal/wait pairs. No WSL hardware counters or measured
occupancy are available; static agreement does not establish the underlying
cache or scheduling mechanism.

## Validation gates

- 473 affected native/ownership/format/registry/lifecycle checks passed.
- 44 owning RX9070XT numerical tests passed.
- 12 FP8/MXFP8 image controls passed.
- Nine excluded-image controls passed.
- Nine source/protocol evidence checks passed, including independent reference
  layout/binary and full-unit log bindings.
- Full non-slow unit result: 28930 passed, 5482 skipped, 876 deselected, 12 warnings in 1160.81s (0:19:20).
- Independent Radiance comparison: six source-locked cases in both compiler/shape orders.

| M,N,K | Current / Radiance forward | Current / Radiance reversed |
| --- | ---: | ---: |
| [256, 8192, 1536] | 1.0087 | 0.9677 |
| [256, 16384, 1536] | 1.1704 | 1.1378 |
| [256, 8192, 2048] | 1.0340 | 0.9887 |
| [256, 16384, 2048] | 1.2640 | 1.2649 |
| [256, 8192, 5120] | 1.1962 | 1.1984 |
| [256, 16384, 5120] | 1.2067 | 1.2070 |

Original failure receipts in the earlier fragment-group packet remain
historical and unchanged. Each certificate retains its own source scope.

## Rejected output-store experiments

Historical compressed records retain the unapplied four-file prototype patch
and its original source hashes. The 128-row two-phase LDS slab reduced register
use but added three workgroup barriers. The 256-row one-phase slab used 32 KiB
of LDS and one extra barrier; its short-K/held-out rows were inconsistent and
included regressions. Neither output-store policy is shipped or selected.

The prototype metadata includes an earlier pending-linked-validation state;
later successful linked builds/numerical checks and negative timing decisions
retain their own revision scope. Those dirty experiments are not current-source
validation. Earlier shared-frame exploratory timings also predate the fixed
independent host snapshot contract; the frozen grid is the decisive CU witness.

## Reproduction and limits


The archived drivers use repository identifier-style names. Recorded commands
retain their original scratch paths; the table maps each shipped copy to its
purpose and output.

| Recorder | Purpose | Output |
| --- | --- | --- |
| [run_frozen_cu_grid.py](run_frozen_cu_grid.py) | Strict 22-shape forward/reversed CU/WGP calibration | cu-grid-forward.json.gz, cu-grid-reverse.json.gz, paired-summary.json |
| [run_frozen_cu_radiance.py](run_frozen_cu_radiance.py) | Strict six-shape independent reference comparison | radiance-forward.json.gz, radiance-reverse.json.gz, radiance-summary.json |
| [compare_format_controls.py](compare_format_controls.py) | FP8/MXFP8 KN/NK image and ABI controls | format-image-equality.json |
| [verify_excluded_images.py](verify_excluded_images.py) | Excluded shape-class image and ABI controls | excluded-image-equality.json |


Forward/reverse command JSON files identify the actual compiler roles and
all shapes. Compressed packets retain source hashes and actual device identity.
The compression manifest checks both raw and compressed bytes. The reproduction
drivers and recorded validation commands accompany the packet; imports are normalized
for repository lint. Recorded commands, source revisions and binary hashes
remain the authority for the measured runs.

The comparison is specific to RX9070XT/gfx1201 and the named native folded
contract. No sibling schedule or device evidence is transferred. The K5120 native/reference ratio remains about 1.20-1.21,
so long-K Radiance attribution, wider W8A8 short/ragged coverage, general cache/layout
families and broader frontend/AD integration remain open.

## Review revalidation

The reviewed recorder compares the complete selected_schedule receipt, including
fragment_read_schedule. Missing fragment policy means llvm_default. CU/WGP
comparison cannot hide a fragment change. A separate explicit
--compare-native-fragment-read-policy switch enables that experiment alone.
All fresh calibration/reference packets keep that switch disabled.

Five guard cases fail on the prior recorder; all 13 focused ownership/admission
cases pass with the fix; 347 affected publication checks pass. Production compiler/runtime source under src/ and
python/ is unchanged from the full-unit/owning-device certificate at 8e53d33556d04066c4a6cd0335b7be1b164b1f80.
That original full-suite receipt retains its revision; the reviewed benchmark
freeze is 68fb444434bbb20f396399e8a1df186c67e00f94. Earlier measurement hashes are preserved in
pre-review-measurement-scope.json and their Git artifact revision.

## Historical source scope after input-view development

The calibration remains bound to 68fb444434bbb20f396399e8a1df186c67e00f94.
The source-snapshot-68fb44443.json.gz witness contains the actual files read
from that Git revision, with every byte digest checked against the original
packet before archival. Its compression-manifest entry and unit consumer
validate both compressed and expanded data. New compiler source does not
retroactively replace the calibration source. Follow-on current-image parity
and pitched-input timings belong to the input-pitches packet.
