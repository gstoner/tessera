# Quantized host binding reuse — gfx1201, 2026-10-17

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-QUANTIZED-BINDING-CACHE-20261017. Depends on PR972.

The cache memoizes full package validation and host ABI construction. Original
Graph capture, per-call input checks/uploads, native device identity, private
readiness and execution still run. It holds no native handles or runtime
tensor values. The existing `TESSERA_ROCM_PROGRAM_BINDING_CACHE` switch
selects the original uncached control. Each decorator has sixteen entries
and a 4 MiB content budget per entry; unsupported/oversized data uses the
original path. Keys include all frozen dataclass fields, raw image bytes,
recipe shape, and mutable nested descriptor metadata.

A profiler at the preceding c7e0b6e57b source recorded 120 underlying component
validations across twenty warm MLP calls. Profile time is instrumented
attribution, not a benchmark denominator. The cache addresses that repeated
work and ABI reconstruction.

## Exact-source paired measurements

Owning RX9070XT / gfx1201, UUID GPU-28d9e7efbf2ef716. Final source is
f136213ed645a1f98cbe9a7808202c247b6cd780; compiler/provider are pinned in each
record. All seven image/descriptor receipts, complete portable artifacts and
numerical certificates are byte-identical between every arm and the existing
rocm_quantized_mlp_20261017 packet. Portable payloads are referenced there,
not duplicated here.

| M / H / K / N | Cache off public ms | Cache on public ms | Public speedup | Device program off/on us |
|---|---:|---:|---:|---|
| 1 / 16 / 16 / 1 | 11.537 | 3.778 | 3.05x | 36.11 / 36.38 |
| 3 / 32 / 48 / 19 | 11.665 | 3.823 | 3.05x | 37.37 / 37.20 |
| 256 / 128 / 12288 / 256 | 13.348 | 4.851 | 2.75x | 177.63 / 177.79 |

The earlier 3c568a0e2a four-arm off/on/on/off repeat records retain their
source scope and show 2.65–3.15x in both orders. Final keyword/process guard
fixes were retested in `final-off` / `final-on`. Thirty owning device tests
pass on the final source, covering quantized v1/v2/v3/v4 public/portable and
lifetime behavior. Twelve final host cache guards and three native mutation
cases cover nested provenance, recipe-shape changes, bool/int separation,
keyword calls, entry bounds, disabled/oversized fallback and process reset.

Public samples include frontend capture, input checks/copies, preparation,
native execution and readback. Ordinary program events and captured member
windows are separate device scopes. The gain is host-side only; no image,
physical schedule, numerical policy or runtime ABI changed. The original
flat-reference disagreements and model-quality limits are unchanged.

Broad final-source CPU, lifecycle/generated-doc and publication gates remain
active. Checkpoint/model quality, remaining host costs, wider/dynamic/AD and
the original five-slice objective remain open. No sibling performance claim.
