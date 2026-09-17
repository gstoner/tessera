# Native storage loop 9 — resident attention JVP

| File | Holds |
|---|---|
| `resident_attention_jvp_nvidia.json` | Eight RTX 5070 cases (causal/noncausal Sq/Sk = 3/5, 5/3, 4/4, 3/129), each with the Q, K, Q+K and Q+K+V tangent modes (32 probes), for the resident attention JVP: rows, `backend: nvidia_sm120`, `resident_lse_tape`, recorder / implementation digests. |

Recorded by `benchmarks/record_generated_attention_ad.py --resident --jvp` (the
payload layout is that script's; the packet's `recorder_sha256` matches no
tracked revision of it).

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` ("resident JVP packet").
