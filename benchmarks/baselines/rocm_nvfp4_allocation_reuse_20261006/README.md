# gfx1201 native NVFP4 allocation reuse

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6;
sync ROCM-NVFP4-ALLOCATION-REUSE-2026-10-06.

Native C++ now retains bounded idle owners below the Python frontend. Ordinary
JIT and portable replay use the same checked path. Checkout keys contain full
component images/entries, dimensions and launch geometry, with live HIP
device/context checks. Checkout assigns a fresh handle, snapshots and uploads
all five inputs and invalidates derived weights/output. It never uses caller
pointer identity to skip changed values.

Retention is at most four idle owners and 64 MiB of accounted device buffers,
host staging, retained image bytes and readback; bookkeeping and driver/module
overhead are outside this byte accounting. Graph-owning or poisoned sessions
are cleaned instead of cached. Cache clear is explicit and context checked.
Failed HIP completion/cleanup retains ownership for retry. Process checks run
before the inherited mutex, preserving fork refusal.

## Evidence

- build.txt: matching gfx1201 native runtime build succeeds.
- tests.txt: 22 initial ownership/rebinding/cache and fault-injection checks pass.
- jit-device.txt: final 35 native/JIT/portable checks pass, including three
  public full-input mutation/reordered cases; a pre-existing pytest timeout
  configuration warning is recorded.
- shared-tests.txt: 37 ABI/frontend/fault-injection gates pass, 22 device skips
  on the NVIDIA WSL host; those skips are not ROCm device proof.
- runtime-abi-refresh.txt: owning runtime ABI CSV/Markdown generator succeeds.
- gfx1201.json: live RX9070XT name/ordinal/opaque HIP UUID, source and runtime
  hashes, identical component images, independent numerical oracles,
  eleven alternating wall samples per arm and separate resident HIP events.

## Timing

- [128, 32, 256]: public cached/control wall 6.444/11.797 ms (0.546x); explicit native session ratio 0.268x.
- [257, 80, 1024]: public cached/control wall 7.384/12.788 ms (0.577x); explicit native session ratio 0.383x.
- [256, 64, 64]: public cached/control wall 6.319/11.437 ms (0.552x); explicit native session ratio 0.255x.

Public timing includes ordinary JIT dispatch, checked artifacts, full input
rebinding/uploads, three native kernels, output readback and release. Both arms
retain the same compiled images; the control disables only native allocation
reuse. Array addresses are unchanged while all five input values alternate.
Oracles are outside each wall window. Kernel events exclude transfers and
include native dispatch gaps. Event ordering/clock variability makes these
diagnostic; the claim is host/runtime wall improvement, not kernel speedup.

General layouts/packing, dynamic shapes, wider AD/residency and the complete
five-slice program remain open. No gfx1151 or sibling physical parity is inferred.

## Fresh-process repeat and drift gates

gfx1201-repeat.json repeats the matched eleven-sample recorder in a fresh process.
- [128, 32, 256]: public cached/control wall ratio 0.577x.
- [257, 80, 1024]: public cached/control wall ratio 0.574x.
- [256, 64, 64]: public cached/control wall ratio 0.560x.

All repeated calls pass the independent full-input oracle. Focused audit/op/dtype/
runtime-ABI gates: 62 passed (drift-gates.txt). Ruff F/E9 and Git whitespace
checks pass. Graphify update is unavailable because its CLI is absent on WSL.

## Static program retention

Profiling of the previous allocating-cache implementation identified repeated
program_from_manifest reconstruction and static semantic verification as the
largest warm CPU costs (public-profile.txt). This cProfile trace is attribution,
not an uninstrumented timing packet.

The common checked runtime retains 24 validated detached program contracts.
Exact type/value snapshots preserve list/tuple, bool/int and float signed zero.
Changed metadata, Graph or argument names miss retention and undergo validation.
Snapshots used for the key also feed the parser, so later caller mutation cannot
alter the admitted static product. Native session input and lifetime checks
continue on every call; the numerical images and recipes are unchanged.

- program-retention-unit.txt: 19 metadata/Graph frontend tests pass.
- program-retention-device.txt: 47 owning-host tests pass: 37 hardware cases,
  nine host retention cases and one native C++ fault-injection case.
- program-retention-shared.txt: 332 diagnostic/pass/ABI/frontend gates pass.
- static-program-gfx1201.json and static-program-repeat-gfx1201.json:
  fresh-process eleven-sample balanced public-JIT wall comparisons. Both arms
  reuse native allocations, carry the same images and consume alternating
  full-input values at the same addresses. The control reconstructs the static
  program each call. Numerical oracles run after every timing window.

- Packet 1, [128, 32, 256]: retained/control 3.285/6.293 ms (0.522x).
- Packet 1, [257, 80, 1024]: retained/control 4.557/7.478 ms (0.609x).
- Packet 1, [256, 64, 64]: retained/control 3.120/6.087 ms (0.513x).
- Packet 2, [128, 32, 256]: retained/control 3.286/6.494 ms (0.506x).
- Packet 2, [257, 80, 1024]: retained/control 4.428/7.266 ms (0.609x).
- Packet 2, [256, 64, 64]: retained/control 3.201/6.142 ms (0.521x).

This is static host-contract cost reduction; kernel/event performance is unchanged
by design. Broader composition/dynamic/layout/AD and the five-slice closure remain open.
