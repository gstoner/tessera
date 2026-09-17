# Native storage loop 10 — automatic Q/K product binding

| File | Holds |
|---|---|
| `automatic_attention_jvp_nvidia.json` | Eight RTX 5070 cases (32 directional probes, including 129 keys and both causal policies) where the native `TangentInterface` product from `export-attention-jvp` is bound to the resident forward generation: rows, `backend: nvidia_sm120`, `resident_lse_tape`, `automatic_jvp`, recorder / implementation digests. |

Recorded by `benchmarks/record_generated_attention_ad.py --resident --jvp
--automatic-jvp` (`recorder_sha256` matches a tracked revision of the script).

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` ("Loop 10: automatic Q/K
product binding and persistent failure isolation").
