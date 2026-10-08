# gfx1201 NVFP4 ingest schedule proof

**Owner:** ROCM-NVFP4-INGEST-1
**Device:** Tajasarus, AMD Radeon RX 9070 XT, gfx1201
**Source revision:** 58b848ccbc7682db03d3b1e350a5421ded56984d
**Source state:** dirty scratch checkout; packet carries source-file hashes and sets source_dirty=true.

The converter maps each source NVFP4 E2M1/K16 + E4M3-scale projection into the MXFP4 E2M1/K32 + E8M0 ABI. For each destination K32 block it searches E8M0 exponents within four powers of two around the existing code-energy-weighted scale seed, selects nearest signed E2M1 values for source-decoded weights, and minimizes decoded-weight SSE over those candidates. An exhaustive host regression checks eight deterministic blocks against all 254 E8M0 exponents. Gate/up projection global scales and row boundaries stay independent.

On Tajasarus, the 17x19x64 synthetic gate/up case passed native Graph -> Schedule -> Tile -> gfx1201 Target IR packaging and numerical checks before timing and after every end-to-end sample. The consumer output exactly matched the decoded ingested weights. HSACO digest: 90185689ec64670b553ec1f7261d57247b2fd0001ac380589b390b158617b1c3.

- New joint code/scale selection relative RMS: 0.09064 gate and 0.06701 up; SQNR 20.85 dB and 23.48 dB.
- Preserve-source-code baseline, with its own SSE-selected E8M0 scale: relative RMS 0.36791 gate and 0.42102 up.
- Ingest time for this small synthetic projection pair: 3.44 ms.
- Graph/Schedule/Tile/Target packaging: 108.95 ms.
- Persistent allocation HIP-event median: 4.54 us.
- runtime.launch median: 2.45 ms, including module load, allocations, transfers, synchronization, descriptor validation, and Python dispatch.

The error comparison is against values decoded from synthetic NVFP4 inputs. There is no original BF16 checkpoint in the available Tajasarus scratch stores, so source-BF16 and direct BF16-to-MXFP4 comparisons remain open. The small ingest timing is not representative of a model-sized checkpoint. Exponent search is bounded around the seed; no claim of global optimality or selector promotion follows.

Packet: gfx1201_nvfp4_joint_requantization_20261001.json.
Older preserve-code packets remain available as historical baselines: initial run, source-snapshot recheck, Tajasarus recheck, and weighted-scale experiment.

## Fresh Tajasaurus recheck — 2026-10-01

The same deterministic 17x19x64 gate/up case was rerun on the RX 9070 XT with
the branch-built compiler (SHA-256
`31b835d5c05e1863ab272b98eb52efb932b95d5ea37d6cb2cbbab66b9f66ee7b`). The
numerical results and HSACO digest were unchanged: relative RMS 0.09064/0.06701
versus 0.36791/0.42102 for the preserve-code baseline; image
`90185689ec64670b553ec1f7261d57247b2fd0001ac380589b390b158617b1c3`.
The recheck measured 2.97 ms ingest, 237.25 ms compile/package, 4.40 us median
persistent HIP events, and 2.46 ms median `runtime.launch`. Two host E2E samples
were near 10 ms; report the median as diagnostic only. Source is dirty and the
packet repeats the source-file hashes.

[Fresh packet](gfx1201_nvfp4_joint_recheck_20261001.json).

## Fresh current-build recheck

Tajasaurus reran the same synthetic 17x19x64 gate/up projections with the
branch-built gfx1201 compiler. The outputs passed exact comparison before
timing and after every host end-to-end sample; the HSACO digest remained
90185689ec64670b553ec1f7261d57247b2fd0001ac380589b390b158617b1c3.
Ingest took 2.96 ms, Graph/Schedule/Tile/Target package construction 236.59
ms, and persistent HIP-event median 4.31 us. Host runtime.launch median
was 2.47 ms with first-call outliers near 10 ms. The focused ingest suite
passed 10 tests on gfx1201. This confirms the small synthetic contract only;
model-checkpoint accuracy and larger-shape costs remain open. Packet:
[current-build recheck](gfx1201_nvfp4_current_recheck_20261001.json).

## Larger synthetic exact-device run

The recorder now accepts shape-matched Graph fixture dimensions, so both
ingest and the native package can be evaluated beyond the original small
case. On Tajasaurus gfx1201, M/N/K=64/96/128 passed exact output comparison
before timing and after each end-to-end sample. The producer converted 48 gate
and 48 up rows; the scheduled Graph/Schedule/Tile/Target package launched on a
6x4 grid. It reused the same HSACO digest as the 17x19x64 packet, while this
run still compiled a new shape-specific package and descriptor.

Synthetic relative RMS was 0.0930/0.0717 for gate/up, versus 0.3568/0.4186
for the preserve-source-code scale baseline. Ingest took 26.50 ms, package
construction 240.36 ms, persistent HIP-event median 4.71 us, and
runtime.launch median 2.44 ms. First host launches were near 10 ms; timing is
diagnostic. The original BF16 checkpoint remains unavailable, so this does
not establish model-checkpoint quality. Packets:
[64x96x128](gfx1201_nvfp4_m64n96k128_20261001.json) and
[default-shape regression](gfx1201_nvfp4_default_after_cli_20261001.json).


## Larger ragged-K synthetic workload

On Tajasaurus gfx1201, the shape-matched recorder passed exact output comparison
before timing and after every end-to-end sample for M/N/K=200/2048/1536. The
native Graph/Schedule/Tile package used a 128x2 grid and 256-thread workgroup.
The HSACO digest was `72eb832d8601355875a9b441cda9572305fa9d1a2729b769f3e986f195a32e12`.
Synthetic gate/up relative RMS was 0.09396/0.07200, compared with
0.35583/0.41749 for the source-code-preserving scale baseline.

Ingest took 6.703 s on the CPU; Graph/Schedule/Tile/Target packaging took
217.69 ms. Persistent HIP-event median was 54.10 us; `runtime.launch` median
was 6.17 ms, with two initial samples around 13 ms. The event samples ranged
47.68–56.99 us, so this is a bounded diagnostic measurement, not a promotion
claim. This synthetic test demonstrates a larger ragged-K route and conversion
cost, not model quality: a real checkpoint and source-BF16 comparison remain
open. [Packet](gfx1201_nvfp4_m200n2048k1536_20261001.json).


## Vectorized ragged-K ingest recheck

The same Tajasaurus gfx1201 M/N/K=200/2048/1536 synthetic case was rerun
after replacing per-row/per-K-block Python dispatch with bounded 64-row
NumPy batches. A randomized regression compares the vectorized result exactly
against the retained scalar selection path, including packed codes and
E8M0 exponents. All 11 focused ingest tests passed on gfx1201.

CPU ingest fell from 6702.96 ms to 487.31 ms (13.76x faster, 92.7% less time).
Gate/up relative RMS remained exactly 0.09396468/0.07199587, and the preserve-
code baseline remained 0.35582517/0.41749312. The HSACO digest, Target IR
digest, and Tile IR digest were unchanged, confirming that the host conversion
change did not alter the native package. Package construction was 238.30 ms;
persistent HIP-event median was 53.51 us; host runtime.launch median was
6.13 ms with the same initial ~13 ms outliers. These results establish an
ingest CPU improvement only; they make no GPU-kernel promotion claim and still
do not replace a real checkpoint/source-BF16 comparison.

[Vectorized exact-device packet](gfx1201_nvfp4_m200n2048k1536_vectorized_20261001.json).

## Exhaustive scale-search stress check — 2026-10-02

A host-side deterministic audit compared the production joint requantizer with
all 254 finite E8M0 exponents for 16,384 seeded K32 blocks. Source scales were
sampled from positive representable E4M3 values and multiplied by per-block
powers of two spanning 2^-20 through 2^20; packed E2M1 codes were randomized.
No bounded-search result missed the exhaustive minimum SSE. A retained unit
test repeats the comparison for 128 seeded blocks. This is a regression check
for the sampled synthetic envelope, not a proof over arbitrary checkpoints;
the source-BF16/model-checkpoint comparison and gfx1201 execution of the latest
unit change remain open.


## 2026-10-02 exact gfx1201 compiler-refresh recheck

Tajasaurus WSL2 (RX 9070 XT, gfx1201) reran the 17x19x64 boundary row and
200x2048x1536 workload after syncing the active-slice Schedule/Tile source and
rebuilding `tessera-opt`. The focused host suite passed 11 tests. Both native
packages passed output comparison before timing and after every end-to-end
sample. The refreshed compiler SHA-256 is
`1e9b32318ce4a15b8466dcb79a2d55b504c0b6cce5f563abf853abc10bf9b397`; both
shape runs retain the same Tile and Target IR digests as the pre-refresh build.

For 17x19x64, ingest was 1.03 ms, package construction 201.39 ms, resident
HIP-event median 4.61 us, and `runtime.launch` median 2.33 ms (first-call
outliers are included in the raw packet). For 200x2048x1536, ingest was
485.93 ms, package construction 193.68 ms, event median 52.98 us, and
end-to-end median 6.15 ms. Gate/up relative RMS remained 0.09396/0.07200;
the preserve-source-code baseline was 0.35583/0.41749. This validates the
synthetic source conversion and scheduled consumer on the refreshed compiler.
The producer still dominates end-to-end work; these synthetic weights do not
prove real-checkpoint quality or justify route promotion.

[Small compiler-refresh packet](gfx1201_nvfp4_small_compiler_refresh_20261002.json)
· [Large compiler-refresh packet](gfx1201_nvfp4_m200n2048k1536_compiler_refresh_20261002.json).


## 2026-10-02 Schedule provenance refresh

The earlier compiler-refresh packets passed native execution and correctness but
left `schedule_digest` null because the package descriptor did not expose the
intermediate Schedule IR. The recorder now captures the Graph-to-Schedule
output and hashes that exact IR separately from Tile and Target IR. Both
regenerated Tajasaurus gfx1201 packets contain nonempty Schedule, Tile, Target,
and image identities and pass output checks before and after end-to-end timing.

The M/N/K=200/2048/1536 run measured 495.69 ms CPU ingest, 54.83 us median
HIP events, and 6.27 ms median `runtime.launch`; the small 64/96/128 run
measured 2.78 ms ingest, 4.55 us HIP events, and 2.41 ms end-to-end. Variation
from the earlier compiler-refresh samples is retained as measurement spread,
not interpreted as a kernel regression or improvement. The workload remains
synthetic and does not establish checkpoint-level conversion quality.

[Large packet](gfx1201_nvfp4_m200n2048k1536_compiler_refresh_20261002_schedhash.json)
· [Small packet](gfx1201_nvfp4_small_compiler_refresh_20261002_schedhash.json).


## Tajasaurus owner-host property-test refresh — 2026-10-02

The latest 12-test NVFP4 ingest property suite passed in the Tajasaurus WSL2
Python environment; the implementation source hash matched the active
Super-Bear worktree. The suite covers vectorized/scalar parity, joint
requantization, merged projection scales, and exhaustive E8M0 scale selection.
These are host-side numeric properties (the tests do not launch a GPU kernel).
The prior exact gfx1201 scheduled-package packets remain the hardware evidence;
a real model checkpoint/source-BF16 quality comparison is still open.

[Host-test packet](gfx1201_nvfp4_unit_recheck_20261002.json).

## Real Qwen3-8B q-projection checkpoint comparison — 2026-10-02

On Tajasaurus (RX 9070 XT, gfx1201), the recorder fetched only the selected
tensor byte ranges for `model.layers.0.self_attn.q_proj.weight` from pinned
NVFP4 and BF16 revisions; 42,991,620 tensor bytes were read and no shard was
saved. The source NVFP4 tensor has shape 4096x2048 with E4M3 K16 scales and a
scalar FP32 global scale; the corresponding BF16 source is 4096x4096.

Relative RMS against the pinned BF16 projection is 9.50% for shipped NVFP4,
11.22% for direct BF16-to-MXFP4, and 14.99% after NVFP4-to-MXFP4 conversion.
The direct baseline uses the same bounded joint E2M1/E8M0 search, seeded per
K32 block by `floor(log2(max(abs(block))/6))`. Converted MXFP4 is 11.29%
relative RMS from shipped NVFP4 and 18.03% from direct BF16-to-MXFP4. This
isolates material additional error in the ingest path for this tensor; it does
not meet a whole-model quality gate. Do not promote NVFP4-to-MXFP4 ingest as a
default route from this projection result.

Both the NVFP4-ingest weights and a direct BF16-to-MXFP4 baseline passed
through the same Graph -> Schedule -> Tile -> gfx1201 Target package and native
execution. Each output matched its own decoded-weight reference with maximum
absolute error 0.0. Package/image, Schedule, Tile, and Target digests are in
the packet. Ingest
took 2555.53 ms, package construction 145.28 ms, and source range fetching
6402.95 ms. With five checked E2E warmups, median persistent HIP-event time was
31.29 us; median `runtime.launch` was 6.281 ms (0.61% CV). Activation input is
deterministic synthetic FP8, and this single q-projection is not a whole-model
quality or throughput result.

[Exact gfx1201 packet](gfx1201_qwen3_8b_q_proj_20261002.json); benchmark:
`benchmarks/rocm/benchmark_rocm_nvfp4_checkpoint.py`.
