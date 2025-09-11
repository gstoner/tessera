
# CPX Autotuning & Profiling

<!-- MERGE-START: CPX_Autotune_Profiling -->
Knobs: token-block size, attention tiling, KV page size, pipeline depth.
Counters: PCIe/CX‑9 throughput, KV spill rate, prefill utilization, decode latency, NVFP4 utilization.
Perfetto: lanes for **Context (CPX)**, **Generation (Rubin)**, **KV‑bridge** with chunk ids.
Roofline: include PCIe Gen6 and CX‑9 bandwidth bands.
<!-- MERGE-END: CPX_Autotune_Profiling -->
