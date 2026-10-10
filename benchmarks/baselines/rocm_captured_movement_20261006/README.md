# Captured native ROCm movement measurements

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key ROCM-CAPTURED-MOVEMENT-2026-10-06.

## Route and ownership

Ordinary native-required JIT emits typed Graph and native Schedule/Tile/ROCm
Target/LLVM HSACO packages. The recorder checks every adjacent compiler digest
and the native Graph-to-Schedule producer, then launches the exact checked
descriptor symbol with the original three memref/scalar ABI on resident buffers.

A private HIP stream captures 256 ordered kernel nodes. The graph, executable
and timing events remain alive until stream completion, then retire before
resident allocations and module. The benchmark helper now accepts an explicit
stream while retaining its existing default-stream behavior. No runtime selector,
GPU body, operation, dtype, pass, diagnostic or C ABI export changes.

Each of eight owning cases has nine alternating driver-loop/captured trials.
Independent gather/slice references are checked bitwise before and after every
arm, including NaN payloads, signed zero and infinities. After timing, sign-bit
changes to source values and reversed resident indices prove the captured
kernels read updated buffers; restoration is also bit-exact.

The full-read profile uses 32 physical pages, a 64-entry logical table with
repeated physical pages, page size 16, eight heads and width 128, reading all
1024 logical tokens. This adds a named LP>P static envelope. Its explicit
static bound does not prove frontend shape-expression or dynamic layout closure.

## Exact-device timing

gfx1151: AMD Radeon 8060S, PCI 0000:c5:00.0, Princess-Luna WSL.
gfx1201: RX 9070 XT, PCI 0000:03:00.0, Tajasaurus WSL.

| Architecture | Case | Driver-loop event us/launch | Captured event us/node | Captured/loop |
| --- | --- | ---: | ---: | ---: |
| gfx1151 | paged small | 2.574 | 2.041 | 0.7928 |
| gfx1151 | paged large | 2.547 | 2.098 | 0.8236 |
| gfx1151 | full large | 37.773 | 37.664 | 0.9971 |
| gfx1151 | dispatched small | 2.222 | 1.683 | 0.7574 |
| gfx1151 | dispatched large | 2.414 | 2.050 | 0.8493 |
| gfx1201 | paged small | 3.057 | 2.947 | 0.9643 |
| gfx1201 | paged large | 3.562 | 3.229 | 0.9063 |
| gfx1201 | full large | 26.036 | 26.229 | 1.0074 |

Event windows include GPU dispatch and kernel execution. The driver-loop arm
may contain host-feed gaps; graph replay removes repeated host submission.
These are controlled resident event windows, not isolated instruction timings.
Separate resident submission/completion wall samples and cold public compile/
host-call and graph-preparation times are retained in each packet. None is
compared as a kernel speedup or transferred across architectures.

Small gfx1151 cases improve roughly 15–24%; small gfx1201 paged cases improve
roughly 4–9%. The large full read changes by less than 1% on both. This supports
a measured short/long distinction and further native resident ownership work,
not universal graph/default promotion.

## Validation and environment

Eight owning-device numerical/rebind/timing cases pass, with 256 captured nodes
per graph and current native images/adjacent IR retained. Each packet fingerprints
the actual owning compiler and frontend/runtime/recorder sources. Recorder and
stream-helper fingerprints match between both devices and the integration tree;
owning runtime sources and compiler binaries retain architecture-specific hashes.

Shared movement/compiler-spine/lifetime/frontend/performance tests: 30 pass,
13 hardware/environment skips on Super-Bear WSL. Those skips are not GPU proof;
the eight packets are the owning exact-device evidence. Changed-file Ruff and
whitespace checks pass.

Tajasaurus uses a scratch-local Python environment. Its compiler initially could
not load libz3.so.4; the existing scratch library search path restores the
LLVM 23.1.1 executable. Prior pre-lineage/pre-rebind packets are retained as historical evidence, not promoted. Setup/frontend failures are described here and recorded in the thread tool history. Graphify remains unavailable
in the authoritative integration WSL checkout.

## Remaining work

This is benchmark resident ownership, not an ordinary captured runtime route.
Native persistent/resident movement packaging, general layouts, asynchronous
lifetime/stream contracts, native movement AD, distributed transport and broader
ROCm route/performance obligations remain open. Quantized FP8/MXFP8/MXFP4 gates
and the original complete five-slice objective remain independent and active.

Apple/NVIDIA/x86 HIP capture is not applicable physical work; their native
ownership and exact-device obligations remain architecture-owned. No sibling
schedule or timing is transferred.

## Native owner follow-on

[Native resident owner](../rocm_resident_owner_20261006/README.md) adds public
preparation and common-runtime admission for the static movement envelope.
This captured packet retains historical source fingerprints from before that
runtime change; it is not current-source or product graph-capture proof.
