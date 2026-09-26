# gfx1201 SSD calibrated pairs: device-clock witness admission

Tajasarus, **AMD Radeon RX 9070 XT (gfx1201)**. The architecture was queried
(`hipGetDevicePropertiesR0600`, ordinal 0), not assumed. The host is Ubuntu
26.04 under **WSL2, with no `/dev/kfd`**, running ROCm 10.0 / HIP 7.15.
`tessera-opt` is a Release build against LLVM/MLIR 23.1.1 (the non-assertions
prefix, mirroring the host's `build/`). Source commit `856c2b2a` (clean tree),
recorded 2026-09-26. Sync `GFX1201-SSD-CALIBRATION-2026-09-26` (follows
`DEVICE-CLOCK-MARKER-2026-09-26`).

**Result: the production SSD selector admits the cooperative candidate** over
the serial incumbent. The reason recorded is "exact-artifact paired
measurements and native calibration admitted". The paired speedup lower bound
is **9.73x** at confidence 0.980, with a median of 9.77x and pair range
9.69–9.94x. The decision is in `admission.json`, and `replay.json` is an
identical decision replayed by `check_ssd_admission.py` from `comparison.json`.
This is gfx1201's own evidence. It is not derived from, and does not transfer
to or from, the gfx1151 packet (`../gfx1151_ssd_calibrated_pairs_20260926/`).

## What was measured
- **Pairs.** Nine independent-process serial/cooperative pairs, alternating
  which variant runs first. The SSD shape is `T,H,N,P = 32,2,16,4`, chunk 8.
  Each process runs **7 windows of 1000 launches** (`--launches 1000`; the rows
  record `launches_per_window`).
- **Calibration per process.** The process's plain HIP-event windows (its
  comparison row) are **interleaved** with windows bracketed by the
  compiler-built device-clock marker. The order alternates per window, and
  every window follows the same span-reset + synchronize gap
  (`window_protocol: interleaved_alternating_plain_bracketed`). The span is
  read on `llvm.readsteadycounter` at 100000 kHz. The clean image is never
  modified.
- **Per-process agreement.** All 18 packets are eligible on the
  `device_clock_witness` route, and each names its row's `run_id`. They share
  source commit `856c2b2a…` with the comparison.
  - Serial: device clock 0.03–0.08% below the HIP event.
  - Cooperative: device clock 0.29–0.61% below.
  - Bracketed/plain duration ratio: serial 0.9941–1.0043, cooperative
    0.9960–1.0034.
- **Correctness.** Every row's maximum absolute error is 0 on all three outputs.
- **Profiler reasons are diagnostic, not blocking.** `BARE_METAL_REQUIRED` and
  `ROCPROFILER_*` appear as `diagnostic_gaps`.

## Shape envelope on gfx1201 (probed before recording)
`materialize_ssd(..., chip='gfx1201')` for serial and cooperative:

| Shape `T,H,N,P` | Chunk | Serial | Cooperative |
|---|---|---|---|
| 32,2,16,4 | 8, 16 | ok | ok |
| 16,2,16,4 | 8 | ok | ok |
| 32,2,16,4 | 4 | refused | ok |
| 64,2,16,4 | 8 | refused | ok |
| 32,2,32,4 | 8 | refused | ok |
| 32,2,16,8 | 8 | refused | ok |
| 128,2,16,4 | 32 | refused | ok |
| 512,2,32,8 | 32 | refused | ok |

Every serial refusal is the native tape GPU lowering's "at most 4096 temporary
bytes" limit, the same limit that bounds gfx1151. `32,2,16,4` chunk 8 was
chosen to match the gfx1151 packet's shape.

## Two superseded attempts (kept under `superseded/`, not admission evidence)
1. **`first_cf6ac5a8_power_state/`** (100 launches per window; plain windows
   first, then the bracketed ones). **Admission refused:
   `INSTRUMENTATION_OVERHEAD_EXCEEDED` on all 18 processes.** The
   bracketed/plain ratios were 1.0845–1.0882 for serial and 1.9862–2.0825 for
   cooperative. The device clock still agreed with the event: serial 0.22–0.50%
   below it, cooperative 1.21–2.65%. The lower bound was 9.01x.
   - A same-process diagnostic attributes the ratio to **device power state, not
     the markers** (`diagnostics/bracket_power_state_probe.{py,out}`,
     diagnostic only).
   - The recorder timed the plain windows warm, right after the checked call.
     It then compiled the marker (subprocesses), which left the GPU idle for
     seconds, and only then ran the bracketed windows.
   - After a 3 s idle, a **plain** cooperative window measures 34.6–34.8
     µs/launch. That is the same as the bracketed 33.8, and about twice the
     warm ~16.5–18.8.
   - The recorder now interleaves the two window kinds (commit `6e6904dc`).
2. **`second_6e6904dc_short_window/`** (interleaved windows, 100 launches per
   window). Pair 0 was admissible: ratios 0.9991 and 0.9744, cooperative device
   clock 4.40% below the event. Pair 1's cooperative process then **aborted**:
   the timing contract refuses a sample whose witnesses disagree by more than
   5%, and this one disagreed by 6.8% (`record.log`).
   - The event bracket and the marker span differ by a roughly fixed amount per
     window, about 60 µs here. That is 3.5–7% of a 1.6 ms cooperative window.
   - 1000-launch windows (`--launches`, commit `856c2b2a`) cut it below 0.7%.
     The default stays 100, so the gfx1151 protocol is unchanged.

## Not claimed
- **No gfx1151 claim.** This packet says nothing new about gfx1151. The recorder
  changes (interleaving, `--launches`) were not re-run on Princess-Luna, and
  the committed gfx1151 packet stands as recorded at `54442ef5`.
- **No NVIDIA claim.** The `%globaltimer` marker is still owed on Super-Bear.
- **No bare-metal comparison.**
- **No speedup at other shapes.** The serial incumbent cannot compile them.
- **No kernel-only time.** The span and the event both cover the whole window,
  including launch gaps.
- **Power state is part of the measurement.** In the recorded runs the first
  plain cooperative window is ~21 µs/launch against a ~16 µs median. Medians
  are reported. Interleaving makes plain and bracketed windows share the
  state, but it does not pin the state.
- **What the probe record is.** The "instrumented" record is the same clean
  image measured under marker bracketing. The ratio bounds timing overhead; it
  cannot detect a codegen change, because the image is not changed.
- **`comparison.json` fields.** `promotion_eligible: false` and `missing_gates`
  are `summarize()`'s calibration-free view. Admission re-derives both gates
  from the packets.

## Reproduce (on Tajasarus)
```
source ~/programming/tessera/.venv/bin/activate
export ROCM_PATH=/opt/rocm/core-10.0 && source scripts/_rocm_env.sh
export TESSERA_LLVM_BIN=$HOME/.local/share/tessera-toolchains/llvm-23.1.1/bin
export TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1 PYTHONPATH=python:.
python benchmarks/record_ssd_rocm_calibrated_pairs.py \
  --compiler build/tools/tessera-opt/tessera-opt --output-dir /tmp/pairs \
  --shape 32 2 16 4 --chunk 8 --launches 1000
python benchmarks/check_ssd_admission.py --comparison /tmp/pairs/comparison.json \
  --compiler build/tools/tessera-opt/tessera-opt --output /tmp/pairs/replay.json
```
