# Native storage loop 8 — resident attention backward

| File | Holds |
|---|---|
| `resident_attention_nvidia.json` | Six RTX 5070 cases (Sq/Sk = 3/5, 5/3, 4/4, each causal and noncausal) for the resident `AttentionCheckpointPair.capture` backward: rows, `backend: nvidia_sm120`, `resident_lse_tape`, and the recorder / implementation digests. Correctness and ownership evidence only. |

Recorded by `benchmarks/record_generated_attention_ad.py --resident` (the
payload layout is that script's; the packet's `recorder_sha256` matches no
tracked revision of it).

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` ("resident attention packet").
