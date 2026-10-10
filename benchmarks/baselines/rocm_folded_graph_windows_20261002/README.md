# gfx1201 checked folded package HIP graph windows

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-GRAPH-WINDOWS-2026-10-02.

[gfx1201.json](gfx1201.json) records seven alternating ordinary/graph trials on
the queried AMD Radeon RX 9070 XT (gfx1201). Both paths execute the same checked
package images, descriptor geometry and three rotating resident copies.
Before timing, both bracketed and plain graphs overwrite all three poisoned
outputs and match the ordinary result bitwise. Ordinary outputs also match the
independent sampled oracle. Captured node counts equal package launches plus
two markers where requested. After timing, every resident output is checked
without issuing another kernel.

Capture uses an explicit nonblocking HIP stream. Marker and timing events are
ordered on that stream. Graph capture/instantiation wall time is separate;
replay uses one host graph launch per window. Graph executables and graphs are
destroyed before their borrowed buffers/modules and clock storage. There is
no fallback if graph support, capture, node census or marker verification fails.

Each accepted marked window is at least 5 ms and agrees with the HIP-event
witness within 5%. Graph windows include GPU graph dispatch, scheduling and
the marker boundary; they are not profiler-isolated kernel instruction time.
The ordinary windows include repeated Python host dispatch gaps. Public
runtime wall timing still includes allocation, transfers, module lifecycle and
completion. No hardware counter or Radiance comparison is claimed.

| MxNxK | Native graph us/launch | HIP graph us/launch | Native/HIP graph | Native/HIP ordinary |
|---|---:|---:|---:|---:|
| 256x4096x1024 | 23.057 | 20.877 | 1.1044 | 1.1122 |
| 256x4096x2048 | 35.075 | 32.702 | 1.0726 | 1.0580 |
| 256x4096x5120 | 79.221 | 72.731 | 1.0892 | 1.0743 |
| 256x8192x5120 | 150.799 | 147.513 | 1.0223 | 1.0207 |

Graph/ordinary ratios are approximately 0.985–1.012 for native and
0.992–1.004 for HIP in this run. Removing repeated host launches does not
close the native/HIP difference (2.2–10.4% in graph medians). Mixed direction
and small changes do not justify a universal graph speedup claim.
[gfx1201_smoke.json](gfx1201_smoke.json) retains the initial three-trial
bracketed-output smoke; the repeated packet strengthens verification to both
graph modes.

## Reproduce

From the owning gfx1201 WSL checkout with its matching compiler and LLVM:

    python benchmarks/rocm/record_gfx1201_folded_native_package.py --tessera-opt "$TESSERA_OPT" --llvm-bin "$LLVM_BIN" --hip-graph-windows --trials 7 --case prefill:256x4096x1024 --case prefill:256x4096x2048 --case prefill:256x4096x5120 --case prefill:256x8192x5120 --output benchmarks/baselines/rocm_folded_graph_windows_20261002/gfx1201.json

## Native versus HIP resource boundary

The same packet's selected-entry evidence reports 177 native VGPRs versus
123 HIP VGPRs, equal 25,600-byte LDS, and zero scratch/spills for both.
Static entry instruction counts include three native barrier signal/wait
pairs versus two HIP pairs, with 32 FP8 WMMA instructions in both entries.
These are compiled resource/static instruction differences, not executed
instruction counts, measured occupancy, or proof of the performance cause.
The next native experiment should investigate barrier placement and live
fragment ranges while preserving all synchronization and numerical contracts.

## Remaining

Native instruction scheduling/epilogue investigation, exact per-K32 migration,
short/ragged-K widening and wider layout/model-quality proof remain open.
This benchmark adapter does not change the public runtime or shared ABI.
gfx1151, NVIDIA, Apple and x86 have no execution parity from this packet.
