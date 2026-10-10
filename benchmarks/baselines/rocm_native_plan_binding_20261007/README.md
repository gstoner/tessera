# Immutable native scaled-program ABI binding

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Sync ROCM-NATIVE-PLAN-BINDING-2026-10-07.

## Measured cause and change

Instrumented warm-call attribution identified repeated native scaled-program
validation, JSON/base64 decoding and ctypes plan construction. The initial
profile is diagnostic instrumentation, not a device-performance measurement.

Complete immutable manifest contents key the checked decode cache. Complete
frozen package contents key readonly ctypes buffer/step/image bindings. Each
cache admits at most 16 entries, with a 1 MiB serialized/binding admission
limit per entry; larger packages use uncached validation/marshaling. Active
owners retain their bindings; public mutable manifest changes create a new
key and are validated. Owner metadata snapshots and native handles remain
distinct. Native C++ preparation still checks SSA lifetimes, capacities,
device/context identity and completion on every call. The compiler route,
native images, allocation ownership and numerical kernel bodies are unchanged.

TESSERA_ROCM_PROGRAM_BINDING_CACHE defaults to 1; 0 is the measured bypass.
This is ABI marshaling reuse at the existing runtime boundary.

## Owning proof and paired benchmark

RX 9070 XT/gfx1201, GPU-28d9e7efbf2ef716. The default lane passes 68 public
primal/JVP tests including active-owner isolation. 24 package checks cover
roundtrip, consistent-witness corruption, mutable-manifest cache invalidation,
bypass and byte admission. 386 shared semantic/registry checks pass.

Six paired public-call rows use 21 alternating-order windows and 10 calls per
arm/window. Independent float64 oracles precede timing; compiler subprocesses
are forbidden in warm windows. Recorded actual HSACO hashes agree across
binding and bypass modes for every row. FP8/MXFP8 primal median paired ratios
are 0.873-0.932 (about 7-13% lower host cost); FP32 scale-JVP ratios are
0.799-0.853 (about 15-20% lower). Individual noisy windows remain in the
packet, including the large LDS primal case. These are public wall-clock
costs, not isolated kernel speedups. Source/compiler/runtime hashes bind
the current owning receipt. The initial paired packet remains unchanged.

Rerun on owning host after sourcing its matching validation environment:
python benchmarks/rocm/benchmark_native_plan_binding.py --output packet.json

## Boundaries

The measured class owns typed E4M3 FP8/MXFP8 f32 products and scale-JVP sums.
Folded MXFP4 BF16 output and NVFP4 ingest use separate execution owners and
need their own attribution. gfx1151 lacks FP8 WMMA. Apple/NVIDIA/x86 use
separate package/runtime contracts and receive no physical performance claim.
Dynamic/nested batching, general transpose/composed AD, the larger ROCm
performance programs, fresh full-unit closure and aggregate PR delivery remain open.
