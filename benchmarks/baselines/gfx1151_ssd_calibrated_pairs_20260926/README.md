# gfx1151 SSD calibrated pairs: device-clock witness admission

Princess-Luna, Radeon 8060S (gfx1151), Ubuntu 26.04 under **WSL2, no
`/dev/kfd`**, ROCm 10.0. Source commit `54442ef5` (clean tree), recorded
2026-09-26. The device architecture was queried (`hipGetDevicePropertiesR0600`
→ gfx1151), not assumed. Sync `WSL-TIMING-ADMISSION-2026-09-26`.

**Result: the production SSD selector admits the cooperative candidate** over
the serial incumbent. The admission reason is "exact-artifact paired
measurements and native calibration admitted", and the paired speedup lower
bound is 9.75x at confidence 0.980 (see `admission.json`). The previous gfx1151
packet (`../gpu_heap_ssd_ad_20260910/rocm-admission.json`) measured the same
~9.7x bound but was refused: "each measured process requires native HIP clock
calibration". This packet supplies that calibration without a profiler.

## What was measured
- **Pairs:** nine independent-process serial/cooperative pairs (alternating
  order), SSD shape `T,H,N,P = 32,2,16,4`, chunk 8.
  - Each process times its clean image in 7 windows of 100 launches with HIP
    events. This is the comparison row, `<i>-<variant>.json`.
  - The shape is the largest the serial native tape GPU lowering admits. It
    caps temporaries at 4096 bytes, and T=48 or any larger state already fails.
- **Calibration per process:** in the same process, each window is bracketed by
  two launches of a **compiler-built device-clock marker**. The marker is an
  empty kernel stamped by `tessera-opt --tessera-device-clock-span`, and its
  span buffer yields the window on the constant-rate device clock
  (`llvm.readsteadycounter`, 100000 kHz).
  - The clean image itself is never modified. Stamping the kernel directly was
    measured to change its codegen: 2512 → 924 instructions when stamped at the
    block start (running 2.4x faster), and still ~1230 when placed after the
    entry allocas.
  - Output: `<i>-<variant>-calibration.json`, `tessera.profiler_rocm_packet.v1`
    on the `device_clock_witness` route.
- **Per-process agreement** (all 18 admissible):
  - Serial: device clock 0.30–0.51% below the HIP event.
  - Cooperative: 2.05–3.12% below. That window is ~10x shorter, so the markers
    are a larger share of it.
  - Marker-bracketed / plain duration ratio: 0.9965–1.0083, within the
    two-sided 5% band.
  - Every calibration names its comparison row's `run_id`, and all share one
    source commit; admission checks both.
  - Host wall > event > device in every window, as expected.
- **Profiler reasons are recorded, not blocking.** `BARE_METAL_REQUIRED` and
  `ROCPROFILER_*` appear as `diagnostic_gaps` per the owner direction
  (MASTER_AUDIT 2026-09-25).

## Not claimed
- **No gfx1201 evidence.** The marker builds for gfx1201, but the packet and SSD
  adapter are gfx1151-only.
- **No NVIDIA evidence.** The `%globaltimer` marker is owed on Super-Bear.
- **No bare-metal comparison.**
- **No speedup at other shapes.** The serial incumbent cannot compile them.
- **No kernel-only time.** The span, like the event, covers the whole window
  including launch gaps.
- **What the probe record is.** The "instrumented" record is the same clean
  image measured under marker bracketing. Its `resource_delta` is zero by
  construction, and the two-sided gate bounds the markers' timing overhead; it
  cannot detect a codegen change, because the image is not changed.
- **Smaller shapes.** Cooperative disagreement (~2–3%) is systematic and grows as
  the window shrinks, so much shorter kernels would exceed the 5% band.
- **`comparison.json` fields.** Its `promotion_eligible: false` / `missing_gates`
  fields are `summarize()`'s calibration-free view. The admission decision
  re-derives both gates from the packets.

## Reproduce
```
source scripts/_rocm_env.sh
PYTHONPATH=python python benchmarks/record_ssd_rocm_calibrated_pairs.py \
  --compiler build/tools/tessera-opt/tessera-opt --output-dir /tmp/pairs \
  --shape 32 2 16 4 --chunk 8
PYTHONPATH=python python benchmarks/check_ssd_admission.py \
  --comparison /tmp/pairs/comparison.json \
  --compiler build/tools/tessera-opt/tessera-opt --output decision.json
```
