# NVIDIA attention saved LSE and backward — 2026-10-01

Exact-device proof ran on Super-Bear (RTX 5070, sm_120). The native checkpoint
suite passed two cases covering saved versus recomputed LSE forward outputs and
gradients against its independent oracle. The correctness-gated E2E spine
benchmark exercised full and bottom-right causal attention for fp16/fp32
storage at regular `[1,4,4,16,16,32,32]` and ragged
`[1,4,1,17,19,31,29]` extents. Every output passed an independent fp64 softmax
reference check before timing; maximum absolute errors ranged from 2.98e-8 to
5.96e-8.

The packet records device-event and end-to-end distributions separately. Device
event medians ranged from 25.21 to 32.44 us. Device-event stability was within
the 3% policy for all eight rows; seven of eight end-to-end rows met that policy,
with one fp16 causal regular row at 3.12%. Compile/package and host enqueue are
included in end-to-end measurements. This is correctness and attribution
coverage, not a comparative speedup claim or selector promotion.

The benchmark ran at source commit `58b848ccbc7682db03d3b1e350a5421ded56984d`
with local changes to add the numerical oracle and its test, so the worktree was
dirty; timings are tied to the packet and this source state, not a clean merged
revision. The packet also records exact GPU, driver, ABI, image, resources,
compile times, correctness tolerances and per-sample timings.

A fresh RTX 5070 recheck used nine samples per run, two interleaved cohorts,
and 20 device launches per sample. All eight fp16/fp32, full/causal, regular/
ragged rows again passed the independent fp64 oracle; max absolute error was
5.96e-8. Device-event medians across cohorts ranged from 25.18 to 33.01 us.
Five of eight rows met the 3% device stability policy and six of eight met the
end-to-end policy; host-side outliers reached 3.77 ms. This is diagnostic
correctness and route evidence, with no selector promotion. See the separate
[recheck packet](recheck_20261001.json).

The exact-device paired-checkpoint test separately validates saved-LSE forward
and backward gradients against recomputation and an independent oracle. It
passes on the RTX 5070. The test is
`tests/device/nvidia/test_lse_checkpoint_native.py`.

Raw [initial attention packet](attention.json).

Saved-LSE backward timing was added in a separate exact-device packet on the
same RTX 5070. Saved and recompute backward packages both executed as native
GPU packages and matched the independent oracle; maximum gradient absolute
error was 1.49e-7. The descriptor CUDA-event median was 1.089 ms (0.07% CV) for
B/Hq/Hkv/Sq/Sk/D/Dv=1/4/2/16/16/32/32. End-to-end host-array timing was noisy
(48% CV, including a 6.66 ms outlier), so only the kernel event result is
usable for attribution. The event benchmark uses persistent device scratch
allocations managed by the launch bridge; it does not yet use the explicit
resident-stream API for backward. See [backward packet](backward_recheck_20261001.json).

The backward resident launch was then added to the NVIDIA bridge and remeasured
using the actual caller-owned buffers and stream. The native saved-LSE forward
produced LSE in that same session; backward consumed it directly, and the
resident gradients passed the oracle (maximum absolute error 1.49e-7). Across
nine samples of 20 device launches, the CUDA-event median was 1.088 ms at
0.02% CV. Host-array E2E median was 2.292 ms at 4.03% CV. This small-shape
measurement is attribution evidence, not a comparative speedup or selector
promotion. See [resident backward packet](backward_resident_20261001.json).

## 2026-10-01 deeper timing refresh

The saved-LSE checkpoint device suite passed 2/2 on the RTX 5070. It checks
saved and recomputed forward outputs and all backward gradients against the
independent oracle; the fresh backward packet reports maximum gradient error
1.49e-7. The backward event median was 1.0871 ms (0.013% CV) across 21
samples of 100 launches using resident buffers. Host-array runtime.launch
median was 2.313 ms with 48.2% CV and multi-millisecond outliers; use the
device-event figure for attribution and treat host E2E as unstable. Packet:
[backward recheck](backward_recheck_20261001.json).

The new forward recorder covers eight full/causal, fp16/fp32, regular/ragged
rows. All passed the independent fp64 oracle before timing with maximum
absolute error 5.96e-8. Two interleaved runs at 100 launches per CUDA-event
sample gave row medians between 24.86 and 32.70 us. The JSON retains both
timing domains and per-sample values; this refresh makes no comparative or
selector claim. Packet: [forward recheck](forward_recheck_20261001.json).

## Float64 correctness-gated saved-LSE checkpoint recheck

The recorder now validates saved and recomputed forward outputs, saved row LSE,
and all dq/dk/dv gradients against an independent float64 scalar oracle before
timing each shape. The packet covers three shapes, including ragged
Sq/Sk=15/17, and records the exact RTX 5070 identity, target, per-result
maximum error, and timing distributions. All checks passed: maximum output
error 3.10e-8, row-LSE error 2.74e-7, and gradient error 9.53e-8. Five-sample
event medians varied by shape and stage; end-to-end timing was noisy, so this
packet supports route and correctness evidence only, with no saved-versus-
recompute performance claim or selector promotion. Saved-route rows also
require complete Graph IR, Schedule IR, Schedule digest, and Tile IR provenance.
The recompute control rows do not expose the same lineage and are explicitly
labeled as comparator packages with partial provenance.

[Correctness-gated packet](saved_lse_recheck_20261002.json).


## Larger exact-device saved-LSE extension — 2026-10-02

The correctness-gated recorder was widened with two SM120 shapes:
[1,4,2,127,131,64,64] (ragged) and [1,4,2,256,256,64,64].
On Super-Bear's RTX 5070, saved and recompute forward outputs passed the
independent float64 oracle; saved row-LSE and all dq/dk/dv gradients passed
as well. At 256x256, maximum output, LSE, and gradient errors were
5.22e-8, 1.15e-6, and 7.06e-7. Saved packages retained full
Graph/Schedule/Tile ancestry.

This establishes native numerical execution over larger sequence lengths but
exposes a severe backward scaling gap. At 127x131, saved/recompute forward
CUDA-event medians were 0.791/0.785 ms; backward medians were 189.3/308.8 ms.
At 256x256, forward was 2.843/2.826 ms and backward was 1221.5/1927.4 ms.
End-to-end medians closely track the device-event medians at these sizes, so
the backward kernel dominates. The route is correct, but its current
deterministic-direct backward kernel is not operationally performant at these
lengths. Source inspection found scalar D and Dv dot-product loops nested
inside per-key scans in the SM120 target materializer; this is a concrete
optimization lead, not proof of sole causation. No selector or default-route
change follows.

The packet records seven source/build hashes, exact GPU identity, timing
samples, and per-shape correctness. It ran with five samples of 25 launches
and took about 19 minutes for all five shapes.
[Extended exact-device packet](saved_lse_extended_recheck_20261002.json).

## 2026-10-02 ABI-preserving correctness and timing recheck

After testing a saved-output `dO·O` row-delta optimization, exact RTX 5070
backward results differed from both the independent oracle and the recompute
route (maximum absolute error 2.80e-3). The optimization and its extra ABI
operand were removed. The original saved-LSE reduction path then passed all 48
focused tests, including exact-device saved/recompute backward and fp64-oracle
checks.

A fresh five-shape recorder run passed all forward, saved row-LSE, and backward
oracle gates. At 127x131, saved/recompute backward CUDA-event medians were
188.55/306.50 ms; at 256x256 they were 1218.60/1926.13 ms. Forward medians at
those shapes were 0.790/0.786 ms and 2.800/2.771 ms. Three samples with two
launch repetitions are diagnostic and do not close the large backward scaling
gap or justify route promotion.

[Fresh five-shape packet](saved_lse_abi_recheck_20261002.json)

## Fresh exact-device attention refresh — 2026-10-02

The recorder ran on Super-Bear RTX 5070 (sm_120), with two interleaved cohorts,
nine samples per cohort, 20 CUDA-event launches per sample, three E2E
repetitions, and five warmups. All eight full/causal, fp16/fp32, regular/ragged
rows passed the independent fp64 output oracle before timing; max absolute error
was 5.96e-8. Device-event median was 25.3–32.4 us. All eight device timing
rows met the 3% stability policy. Six of eight E2E rows met it; fp16 full
attention was 3.7% on the regular case and 8.2% on the ragged case. This is
route and correctness evidence only, with no comparative speedup claim or
selector promotion.

[2026-10-02 packet](attention_lse_followup_20261002.json)


## 2026-10-02 larger backward profile and stability recheck

A focused saved-LSE backward run on Super-Bear RTX 5070 extended the exact
shape to B/Hq/Hkv/Sq/Sk/D/Dv=1/4/2/128/128/64/64. Five CUDA-event samples
of one resident launch each measured 183.165 ms median (0.012% CV); host-array
end-to-end measured 185.285 ms (0.029% CV). Saved and recompute outputs passed
the independent oracle, with maximum dq/dk/dv errors 4.19e-9, 8.38e-9, and
2.09e-7. This confirms the large cost is in device work rather than transfer
or host launch. Packet: [shape-128 backward recheck](attention_backward_shape128_recheck_20261002.json).

Nsight Compute profiled the saved-LSE kernel
`tessera_tile_attention_backward_lse_355a87010f`: 512 blocks of 128 threads,
56 registers/thread, 22.7% achieved occupancy, 34.7% compute throughput, and
0.00% DRAM throughput. The profiled duration (208.9 ms) includes profiler
replay and is not a benchmark timing. The launch is latency/occupancy limited;
source inspection finds nested scalar D/Dv reductions repeated across output
work. A tiled backward schedule is required before any performance claim or
route promotion. No implementation or selector change is included in this
packet.


## 2026-10-02 backward dV row-delta elimination

The saved-LSE dV path now loads the row LSE directly for probability
reconstruction and skips the dO·V row-delta reduction, which dV does not use.
On Super-Bear RTX 5070, shape [1,4,2,128,128,64,64], five correctness-gated
samples matched saved-LSE, recompute, and the independent oracle (maximum
gradient absolute errors 4.19e-9, 8.38e-9, and 2.09e-7). The median was
183.179 ms resident and 185.364 ms end-to-end; both timing CVs were below
0.14%. This is a correctness-preserving neutral timing result. The scalar
dQ/dK loops still recompute row statistics repeatedly and dominate the
architecture work.

[Packet](attention_backward_shape128_dv_delta_elided_20261002.json).


## Saved-output row-delta optimization — 2026-10-02

The backward package now consumes both saved forward output O and row LSE.
The kernel computes the dQ/dK row delta as dO·O, avoiding the prior repeated
Sk-by-Dv reduction. The first rerun exposed a host PTX bridge ABI bug: the
nine-buffer saved-output entry had been mistaken for a bias-plus-LSE entry,
which copied only part of O and shifted the gradient output slots. The bridge
now identifies the backward_lse_output entry, sizes O independently, and
preserves the saved-output/LSE/output pointer order for host, resident, and
timing launches. This was a launcher marshalling defect; it was not a
numerical counterexample to the dO·O identity.

The expanded exact-device test passes both the original small case and
[1,4,2,16,16,32,32], including host arrays and caller-owned resident buffers.
At [1,4,2,16,16,32,32], all gradients match recompute and the float64 oracle
(maximum errors 2.79e-9, 4.66e-9, 1.49e-7); median device time is 0.0804 ms
and end-to-end is 1.304 ms. At the previously profiled
[1,4,2,128,128,64,64], all gradients again match (maximum errors
5.59e-9, 7.45e-9, 2.09e-7); median device time is 2.378 ms and end-to-end
is 4.079 ms. Against the same-shape pre-optimization saved-LSE/dV-elided
packet (183.179 ms device, 185.364 ms end-to-end), these are 77.0x and 45.4x
reductions. Selector/default routing remains unchanged pending wider exact
shape, dtype, and checkpoint-envelope coverage.

[Shape-16 packet](attention_backward_saved_output_recheck_20261002.json).
[Shape-128 packet](attention_backward_saved_output_shape128_20261002.json).


## Host-bridge capacity rebuild — 2026-10-02

After rebuilding the PTX launcher with ten-slot pointer arrays for resident
and event paths (matching the ten-buffer maximum accepted by saved-O + bias +
LSE), the small saved-output case again passed forward-output, row-LSE,
saved/recompute, and independent-oracle checks on Super-Bear RTX 5070. Maximum
gradient errors were 2.79e-9/4.66e-9/1.49e-7. Five resident samples at 100
launches each measured 0.08015 ms median with 0.075% CV. End-to-end median was
1.419 ms with 53.3% CV due to one host outlier; this remains diagnostic and is
not used for a speedup claim.

[Post-rebuild packet](attention_backward_saved_output_bridge_refresh_20261002.json)

## Bias and saved-output checkpoint integration — 2026-10-02

RTX 5070 (sm_120) native Graph → Schedule → Tile → NVIDIA Target → PTX proof, with explicit exact-shape f32 bias. Both forward output/LSE and dQ/dK/dV pass independent fp64 oracles before timing. Host and resident launches are covered. Paired capture owns private Q/K/V/O/LSE/bias; caller mutation and repeated backward tests pass. The tape's prior omitted saved-output buffer is corrected.

| B/Hq/Hkv/Sq/Sk/D/Dv | Saved backward event median (ms) | E2E median (ms) | Event CV | E2E CV |
| --- | ---: | ---: | ---: | ---: |
| 1/2/1/3/4/4/3 | 0.01112 | 1.27415 | 32.0% | 31.2% |
| 1/4/2/16/16/32/32 | 0.08043 | 2.80940 | 0.6% | 40.2% |
| 2/4/2/5/7/8/6 | 0.01337 | 2.16813 | 8.6% | 92.0% |

Seven samples, thirty repetitions, ten device warmups per row. CUDA events use native repeated dispatch; E2E includes per-call host binding, allocation, transfer and synchronization. Timing is diagnostic; no selector promotion. Automatic bias AD and bias JVP remain open. The full focused lane passed 473 tests with three environment-gated skips; see the transcript for the executed scope.

[Raw packet](attention_bias_saved_output_20261002.json), [validation fingerprints](attention_bias_saved_output_validation_20261002.json), [test transcript](attention_bias_saved_output_tests_20261002.txt).
