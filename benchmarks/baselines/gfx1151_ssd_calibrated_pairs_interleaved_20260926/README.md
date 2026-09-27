# gfx1151 SSD calibrated pairs, interleaved protocol

Princess-Luna, **AMD Radeon(TM) 8060S Graphics (gfx1151)**. The architecture
was queried (`hipGetDevicePropertiesR0600`, ordinal 0), not assumed. The host
is Ubuntu 26.04 under **WSL2, with no `/dev/kfd`**, running ROCm 10 at
`/opt/rocm/core`. `tessera-opt` is a Release build against apt LLVM/MLIR 23 in
a dedicated worktree (`~/programming/tessera-ssd151`). Source commit
`0f9c29cf` (clean tree), recorded 2026-09-26. Every timing run was serialized
with `flock /tmp/tessera-timing.lock`, because another worker times x86 on this
box. Sync `GFX1201-SSD-CALIBRATION-2026-09-26`, follow-up 1.

This packet re-records `../gfx1151_ssd_calibrated_pairs_20260926/` under the
protocol gfx1201 used. The earlier packet is kept as history. It still
validates, but admission now refuses it with
`SSD_CALIBRATION_WINDOW_PROTOCOL_LEGACY`, because its calibrations carry no
`window_protocol`.

**Result: the production SSD selector admits the cooperative candidate** over
the serial incumbent. The recorded reason is "exact-artifact paired
measurements and native calibration admitted".

- The paired speedup lower bound is **9.89x** at confidence 0.980.
- The median is 9.91x, and the pair range is 9.86–9.95x.
- The decision is in `admission.json`. `replay.json` holds the identical
  decision, replayed by `check_ssd_admission.py` from `comparison.json`.

The earlier gfx1151 packet reported 9.75x under the old protocol at 100
launches per window. The two used different methods, so their ratio is **not**
a claim.

## What was measured
- **Pairs.** Nine independent-process serial/cooperative pairs, alternating
  which variant runs first. The SSD shape is `T,H,N,P = 32,2,16,4`, chunk 8.
  This is the same shape and structure as both earlier packets. Each process
  runs **7 windows of 1000 launches** (`--launches 1000`).
- **Calibration per process.** The plain HIP-event windows (the comparison
  row) are interleaved with windows bracketed by the compiler-built
  device-clock marker. The order alternates per window, and every window
  follows the same span-reset and synchronize gap. The clean image is never
  modified. The span is read on `llvm.readsteadycounter` at 100000 kHz.
- **The protocol stamp admission now reads.** Every calibration carries
  `timing.environment.window_protocol =
  interleaved_alternating_plain_bracketed`. Its launch count appears as
  `timing.batch_size = 1000` and in the device clock's
  `provenance.launches_per_window = 1000`, and it equals the row's
  `launches_per_window`. The two calibration-side values sit in the part
  `timing_sha256` covers, and admission validates that digest before reading
  them; the row's own count (in `comparison.json`) is not digest-covered.
  The digests are unkeyed SHA-256, so they catch an unresealed edit, not a
  deliberate reseal. The count is checked for consistency (including one
  value across all 18 rows), not against the durations, which are stored per
  launch.
- **Per-process agreement.** All 18 packets are `promotable` on the
  `device_clock_witness` route with no ineligibility reasons. Each names its
  row's `run_id` and records source commit `0f9c29cf…` with a clean tree.
  - Serial: device clock 0.04–0.08% below the HIP event. The
    bracketed/plain ratio is 0.9997–1.0001.
  - Cooperative: device clock 0.29–0.62% below the HIP event. The ratio is
    0.9975–1.0020.
  - Row medians: serial 133.75–133.81 µs/launch, cooperative 13.45–13.56
    µs/launch.
- **Correctness.** Every row's maximum absolute error is 0 on all three outputs.
- **Profiler reasons are diagnostic, not blocking.** `BARE_METAL_REQUIRED` and
  `ROCPROFILER_*` appear as `diagnostic_gaps`.

## Why 1000 launches per window (measured on gfx1151)
The event bracket and the marker span differ by an offset per window that
grows slowly with window length (cooperative 29.9 -> 41.6 µs, serial 54.1 ->
73.7 µs from 100 to 1000 launches) while its share of the window falls. `diagnostics/launches_probe/` measured it on this box at this commit,
with one clean process per variant at 100 and at 1000 launches. The summary
is in `offset_summary.txt`:

| launches | variant | offset per window (median, min–max) | window | median offset share |
|---|---|---|---|---|
| 100 | serial | 54.1 µs (47.1–110.4) | 13.4 ms | 0.40% |
| 100 | cooperative | 29.9 µs (29.1–30.4) | 1.38 ms | **2.17%** |
| 1000 | serial | 73.7 µs (61.7–103.6) | 133.8 ms | 0.06% |
| 1000 | cooperative | 41.6 µs (30.4–56.4) | 13.1 ms | **0.32%** |

At 100 launches, the ~30 µs offset is 2.2% of a cooperative window. That is
inside the 5% band, but it uses almost half of it. At 1000 launches it is
0.3%, the same regime as the gfx1201 packet, so 1000 was chosen. The committed
gfx1151 packet's own windows show the same offset: a median of 29.6 µs for
cooperative and 52.5 µs for serial at 100 launches. The probe files are
diagnostics only, not admission evidence.

## Not claimed
- **No gfx1201 claim, and no transfer between chips.** gfx1201 evidence is
  `../gfx1201_ssd_calibrated_pairs_20260926/`.
- **No ratio against the earlier gfx1151 packet.** The window length and the
  plain-window timing both changed.
- **No NVIDIA claim.** No bare-metal comparison.
- **No speedup at other shapes.** The serial incumbent's native tape GPU
  lowering caps temporaries at 4096 bytes.
- **No kernel-only time.** The span and the event both cover the whole window,
  including launch gaps.
- **Power state is not pinned.** Interleaving makes plain and bracketed
  windows share one state, but it does not fix that state.
- **`comparison.json` fields.** `promotion_eligible: false` and
  `missing_gates` are `summarize()`'s calibration-free view. Admission
  re-derives both gates from the packets.

## Files
- `record.txt`: the run log. The host, commit, clean-tree check and
  `ROCM_PATH` are on its first line. The repo ignores `*.log`.
- `diagnostics/launches_probe/`: `run.sh.txt` (the script), `run_output.txt`,
  the four rows with their calibrations, and `offset_summary.txt`.

## Reproduce (on Princess-Luna)
```
source ~/programming/tessera/.venv/bin/activate && source scripts/_rocm_env.sh
export PYTHONPATH=python:.
flock /tmp/tessera-timing.lock python benchmarks/record_ssd_rocm_calibrated_pairs.py \
  --compiler build/tools/tessera-opt/tessera-opt --output-dir /tmp/pairs \
  --shape 32 2 16 4 --chunk 8 --launches 1000
python benchmarks/check_ssd_admission.py --comparison /tmp/pairs/comparison.json \
  --compiler build/tools/tessera-opt/tessera-opt --output /tmp/pairs/replay.json
```
