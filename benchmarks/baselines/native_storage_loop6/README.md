# Native storage loop 6 — pending cohorts and queue intervals

RTX 5070 (sm_120) and gfx1151 packets for the multi-cohort pending-storage
proof and the serial-versus-parallel native queue measurement.

| File | Holds |
|---|---|
| `pending_cohort2_nvidia.json`, `pending_cohort3_nvidia.json`, `pending_cohort4_nvidia.json` | `record_rotating_storage.py --outstanding {2,3,4} --nested` on the RTX 5070: seven cases per cohort count, with source/compiler/package digests and `selector_promotion`. |
| `queue_intervals_nvidia.json`, `queue_intervals_rocm.json` | `record_native_queue_overlap.py` serial/parallel launch rows (8/32/128 blocks, five repetitions) with timing, `hardware_counter_attribution` and `selector_promotion`. |
| `nsight_systems.json` | The Nsight Systems capture of the CUDA queue run: the `nsys profile … record_native_queue_overlap.py …` command, SQLite digest, kernel intervals and cross-stream intersections. |
| `nsight_compute.csv` | The Nsight Compute export for the same run. |

Recorded by `benchmarks/record_rotating_storage.py` (the `pending_cohort*`
packets; their `recorder_sha256` matches a tracked revision of the script) and
`benchmarks/record_native_queue_overlap.py` (the `queue_intervals_*` packets,
per the command inside `nsight_systems.json`; their `recorder_sha256` matches no
tracked revision of the script).

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` (the pending-cohort
"Packets" link and "Queue measurements and attribution").
