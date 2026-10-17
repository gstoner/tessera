# Quantized host binding reuse — gfx1201, 2026-10-17

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-QUANTIZED-BINDING-CACHE-20261017. Follows merged PR972.

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
95be6a2f3b6ed94b7fd157a372bb3b1cd31c2fdb (`zero-off` / `zero-on`);
compiler/provider are pinned in each record. All seven receipts, complete
portable artifacts and numerical certificates match every earlier arm and
the canonical rocm_quantized_mlp_20261017 packet byte-for-byte. Portable
payloads are referenced there, not duplicated here.

| M / H / K / N | Cache off public ms | Cache on public ms | Public speedup | Device program off/on us |
|---|---:|---:|---:|---|
| 1 / 16 / 16 / 1 | 11.761 | 4.219 | 2.79x | 35.89 / 35.73 |
| 3 / 32 / 48 / 19 | 11.847 | 3.987 | 2.97x | 37.41 / 37.58 |
| 256 / 128 / 12288 / 256 | 12.981 | 5.340 | 2.43x | 177.43 / 178.28 |

Earlier records retain their source scope: 3c568a0e2a off/on/on/off repeats,
f136213ed6 keyword/process guards in final-off/final-on, and 92bb69760d
UTF-8/key-budget guards in budget-off/budget-on. Final signed-zero source
passes thirty owning device cases. The final focused gate passes 387 cases:
76 cache/native package controls and 311 audit/metadata/diagnostic checks.
Fifteen host guards include typed numeric content, nested mutation, keyword
calls, process reset and UTF-8/key bounds. Four actual native mutation cases
refuse changed descriptor metadata, recipe shape and signed-zero metadata.

The signed-zero case was reproduced red: Python's numeric equality treated
+0.0/-0.0 metadata as the same snapshot although descriptor digests differ.
Float keys now preserve binary64 bits. The genuine native-package regression
passes; no validation rule or numerical tolerance was weakened.

Public samples include frontend capture, input checks/copies, preparation,
native execution and readback. Ordinary program events and captured member
windows are separate device scopes. The gain is host-side only; no image,
physical schedule, numerical policy or runtime ABI changed. The original
flat-reference disagreements and model-quality limits are unchanged.

Final-source WSL CPU validation passes 22,223 tests (9,946 skipped;
11 warnings). All 33 generated documents pass drift checks. Public CI and
review gates remain active. Checkpoint/model quality, remaining host costs, wider/dynamic/AD and
the original five-slice objective remain open. No sibling performance claim.
