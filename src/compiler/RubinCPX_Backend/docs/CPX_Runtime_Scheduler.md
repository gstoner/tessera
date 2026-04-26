
# CPX Runtime & Scheduler

<!-- MERGE-START: CPX_Runtime_Scheduler -->
- **Two-pool executor**: `Pool{CPX}` and `Pool{Rubin}` plus a KV handoff stage.
- **KV bridge**: zero-copy rings (16–64MB chunks), credit-based backpressure.
- **Rack-aware placement**: NVL144 CPX tray awareness for shard affinity.
- **Precision probe**: enable NVFP4 when present; fallback to FP8/FP16.
<!-- MERGE-END: CPX_Runtime_Scheduler -->
