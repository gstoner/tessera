---
audit_role: plan
plan_state: open
last_updated: 2026-09-15
---

# Native RDNA4 commissioning

> **Stale as a status record — corrected 2026-09-15.** The box landed as
> **Tajasarus**: RX 9070 XT (**gfx1201**), Ubuntu 26.04.1 **under WSL2** (not
> the native Linux this plan assumed), ROCm 10.0.0 / HIP 7.15.26333 at
> `/opt/rocm/core-10.0`, assertions-enabled LLVM/MLIR 23.1.1, commissioned
> 2026-09-13 (`todo.md` §`GFX1201-FOUNDATION-2026-09-13`, packets under
> `benchmarks/baselines/gfx1201_*`). "Hardware ordered … no measurements exist"
> below is history. What this plan still owns is unchanged in substance: the
> **profiler-attribution obstacle is not removed** — WSL2 has no `/dev/kfd`, so
> `rocprofv3` returns no dispatch/counter records on Tajasarus either, and no
> performance promotion is admissible from it. A native-Linux install of the
> same box remains the route to counters; the commissioning order below is
> the checklist for that, not a description of the current host.

Owner: W2.4a / IR-NATIVE-FOUNDATION-1; [ROCm queue](todo.md).
Hardware ordered: Ryzen 7 9800X3D, 32 GB DDR5-6000, RX 9070 XT 16 GB;
planned native Ubuntu 26.04, arrival expected late next week. No host or device
measurements exist yet. User explicitly selected native Linux for this lane;
existing Princess-Luna WSL evidence remains a separate gfx1151 lane.

The RX 9070 XT is **gfx1201**, not gfx1200. Confirm with `rocminfo`, not a forced
HSA target override. AMD's [ROCm 7.14.1 release notes](https://rocm.docs.amd.com/en/docs-7.14.1/about/release-notes.html)
list gfx1201 and Ubuntu 26.04. Recheck the actual driver/kernel/SDK combination
at installation; do not mix the existing WSL SDK libraries into this host.
Use the repository RDNA4 ISA archive for instruction availability. Native Linux
removes the current WSL attribution obstacle but does not prove any counter is
valid for this GPU or workload.

## Commissioning order

1. Install the supported AMD driver, ROCm SDK, ROCprofiler-SDK (`rocprofv3`),
   ROCm Systems Profiler (`rocprof-sys-run`) and build dependencies. Configure
   user access to `/dev/kfd` and render nodes using the distribution's groups.
   Establish key-based SSH and verify CPU, GPU, VRAM and ROCm identity.
2. Run `PATH=/opt/rocm/bin:$PATH python3 scripts/probe_rocm_native_host.py
   --expected-gfx gfx1201 --output /tmp/rdna4-host.json`. This read-only probe
   records tool versions/help and available counters; it refuses WSL, a missing
   target, inaccessible KFD or failed probes. A ready result only permits testing.
3. Build an assertions-enabled LLVM/MLIR and Tessera, recording revisions and
   executable hashes. Run focused IR/ABI gates before exact-device tests.
4. Run correctness first, then matched queue experiments on gfx1201. The
   existing queue recorder still defaults its HIP compiler chip to gfx1151;
   use the explicit chip option added for this lane. Never execute a gfx1151
   image and label it gfx1201 evidence.
5. Capture `rocprofv3 --kernel-trace --output-format csv -- <workload>` and
   `rocprofv3 --sys-trace -- <workload>`. Preserve kernel names, agent/queue IDs,
   correlation IDs, timestamps and allocation/copy records. Match serial and
   parallel controls to dispatch records before reporting kernel overlap.
6. Enumerate `rocprofv3 --list-avail`, choose counters exposed by this device,
   then collect an isolated dispatch with the installed tool's input format.
   Reject empty/all-zero output on a known active workload; distinguish real
   zeros from unsupported counters. Do not reuse CDNA/gfx1151 counter names.
7. Capture the same workload with `rocprof-sys-run --trace -- <workload>` after
   verifying installed help. Keep system tracing separate from counter replay.
   Repeat unprofiled measurements in fresh processes; quantify profiler overhead.

References: [rocprofv3 tracing and counter enumeration](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html),
[Systems Profiler workflow](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/doxygen/html/index.html).

Acceptance requires exact output oracles, real dispatch/counter records and
source/compiler/image identities. No backend selector or MSW-9 native status
may be promoted from installation, enumeration or reference algebra alone.
