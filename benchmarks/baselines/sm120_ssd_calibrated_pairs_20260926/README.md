# sm_120 SSD calibrated pairs: `%globaltimer` device-clock witness admission

The-Super-Bear, **NVIDIA GeForce RTX 5070 (sm_120, cc 12.0, UUID
`cba12639821a7a104cd3f918f9c0a545`)**. The identity was queried at record time
(`cuDeviceGetName` / `cuDeviceGetAttribute` / `cuDeviceGetUuid_v2`, ordinal 0)
and is stored in every calibration's `timing.environment.device_identity`; it
was not assumed. The host is Ubuntu 26.04 under **WSL2** (`/dev/dxg`, no
profiler), CUDA 13.4 toolkit, driver 610.88 (`cuDriverGetVersion` 13030).
`tessera-opt` is the worktree's `build/` against apt LLVM/MLIR 23.1.1. Source
commit `de66d702` (clean tree), recorded 2026-09-26. Sync
`NVIDIA-GLOBALTIMER-MARKER-2026-09-26` (follows `DEVICE-CLOCK-MARKER-2026-09-26`
and `WSL-TIMING-ADMISSION-2026-09-26`).

**Result: the production SSD selector admits the cooperative candidate** over
the serial incumbent, on the `device_clock_witness` route: "exact-artifact
paired measurements and %globaltimer device-clock calibration admitted". The
paired speedup lower bound is **4.33x** at confidence 0.980; the median is
7.48x and the nine pair ratios range 4.27-8.29x. `admission.json` is the
decision; `replay.json` is an identical decision (same selected binding)
replayed by `check_ssd_admission.py` from `comparison.json`. This is sm_120's
own evidence; it does not transfer to or from any gfx packet, nor to another
cc 12.0 part.

## The marker was validated first (`diagnostics/`)

`benchmarks/probe_nvidia_globaltimer_marker.py`, same commit, same box, under
the timing lock (`globaltimer_marker_probe.{json,txt}`):

- **Builds and reads the clock twice into one span.** The marker comes from
  `native_device_clock.build_device_clock_marker(backend='nvidia',
  chip='sm_120')` (`tessera-opt --tessera-device-clock-span=backend=nvidia` ->
  NVVM -> `gpu-module-to-binary`). Its SASS has exactly two
  `CS2R Rn, SR_GLOBALTIMERLO` reads and two `REDG.E.MIN/MAX.64.STRONG.SYS` span
  updates; the builder refuses any image without them. One marker launch wrote
  an ordered span (96 ns).
- **Resolution: 32 ns.** A one-thread kernel read `%globaltimer` back to back
  65,536 times (three runs, ~9.8 ns per read): every non-zero increment was
  exactly 32 ns and ~69% of consecutive reads were equal. WSL2 does not
  coarsen it.
- **Agreement with CUDA events, windows of 1-1000 launches of a spin kernel
  (7 windows each):** the span is shorter than the event interval by a roughly
  fixed **~10-16 us per window** (up to ~46 us in the longest). Every window of
  about **1 ms or longer agreed within the 5% band** (worst single window
  3.2%, median error 0.05-1.0%); every configuration of about 0.36 ms or
  shorter had at least one window outside it (up to 76% at a single 69 us
  launch). So the marker is admissible only for long windows, which is what
  `--launches 1000` gives here.

## What was measured

- **Pairs.** Nine independent-process serial/cooperative pairs, alternating
  which variant runs first. SSD shape `T,H,N,P = 32,2,16,4`, chunk 8 (the
  gfx1151/gfx1201 packets' shape). Each process runs **7 windows of 1000
  launches**; rows record `launches_per_window`.
- **Calibration per process.** Plain CUDA-event windows (the comparison row)
  interleaved with windows bracketed by the marker, alternating order, each
  after the same span-reset + synchronize gap
  (`window_protocol: interleaved_alternating_plain_bracketed`). The clean image
  is never modified.
- **Windows:** serial ~99.2 us/launch (99 ms windows); cooperative 12.0-23.2
  us/launch across processes (12-23 ms windows).
- **Per-process agreement.** All 18 packets are eligible, name their row's
  `run_id` and the measured image, and share source commit `de66d702` with the
  comparison. Device clock below the CUDA event by 0.030-0.066% (serial) and
  0.027-0.292% (cooperative). Bracketed/plain duration ratio: serial
  1.0002-1.0010, cooperative 0.9558-1.0128.
- **Correctness.** Every row's maximum absolute error is 0 on all three outputs.

## Not claimed

- **No kernel-only time.** The span and the event both cover the whole
  window, launch gaps included.
- **The cooperative variant's per-process spread (12.0-23.2 us/launch) is not
  explained.** It moved the pair ratios from 4.3x to 8.3x; the lower bound
  uses the second-smallest ratio, so admission is conservative to it, but the
  cause (clock/power state is the likely one on this part) is not measured.
- **The lowest bracketed/plain ratio, 0.9558 (cooperative), sits close to the
  two-sided band edge (1/1.05 = 0.952).** It passed; a noisier run could refuse
  on `INSTRUMENTATION_CHANGED_THE_KERNEL` even though the image is identical,
  since this ratio compares two window kinds, not two images.
- **No bare-metal comparison, no Nsight activity-window packet.** The Nsight
  route (`record_ssd_calibrated_pairs.py`) stays separate and unrecorded here.
- **No other shape or part.** Nothing here transfers to a 5070 Ti or any
  other cc 12.0 GPU.
- **`comparison.json` fields.** `promotion_eligible: false` and
  `missing_gates` are `summarize()`'s calibration-free view; admission
  re-derives both gates from the packets.

## Reproduce (on The-Super-Bear)
```
source scripts/_nvidia_env.sh
export PYTHONPATH=python:. TESSERA_LLVM_BIN=/usr/lib/llvm-23/bin
flock /tmp/tessera-timing.lock python benchmarks/probe_nvidia_globaltimer_marker.py \
  --compiler build/tools/tessera-opt/tessera-opt --output /tmp/probe.json
flock /tmp/tessera-timing.lock python benchmarks/record_ssd_rocm_calibrated_pairs.py \
  --backend nvidia --compiler build/tools/tessera-opt/tessera-opt \
  --output-dir /tmp/pairs --shape 32 2 16 4 --chunk 8 --launches 1000
python benchmarks/check_ssd_admission.py --comparison /tmp/pairs/comparison.json \
  --compiler build/tools/tessera-opt/tessera-opt --output /tmp/pairs/replay.json
```
