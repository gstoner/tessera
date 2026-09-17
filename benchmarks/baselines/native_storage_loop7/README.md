# Native storage loop 7 — device tapes and generated attention AD

| File | Holds |
|---|---|
| `device_tapes_nvidia.json`, `device_tapes_rocm.json` | `record_native_device_tape.py`: 12 cases each on the RTX 5070 and gfx1151 (square/tanh/sum/mean, widths 32/64/256, post-capture mutation, two cotangents, nested-frame cleanup), with compiler/recorder/implementation digests; `higher_order_ad` and `control_flow_tape` are both `false`. |
| `generated_attention_nvidia.json` | `record_generated_attention_ad.py`: six RTX 5070 cases from a fresh reverse-marked MLIR function through the exported checkpoint product (`resident_lse_tape` recorded). |

Recorded by `benchmarks/record_native_device_tape.py` (`recorder_sha256`
matches a tracked revision) and `benchmarks/record_generated_attention_ad.py`
(named by the follow-up log; the packet's `recorder_sha256` matches no tracked
revision of the script).

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` ("Persistent snapshots and
generated attention exports — 2026-09-06").
