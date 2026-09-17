# Native storage loop 11 — split persistent tapes and JIT-owned attention

| File | Holds |
|---|---|
| `persistent_tape_nvidia.json`, `persistent_tape_rocm.json` | Widths 4, 8 and 16 of a two-level SAVE loop (two outer, three inner iterations) through the split persistent tensor products on the RTX 5070 and gfx1151: rows, compiler digest, `source_hashes`, recorder digest. Serial one-thread correctness evidence, not latency. |
| `jit_attention_nvidia.json` | Ten RTX 5070 cases (Q, K, Q/K, reversed K/Q and Q/K/V requests, 5 and 129 keys, grouped heads, aligned causal masks) starting from an ordinary decorated function through `compile_native_attention_jvp`. |

Recorded by `benchmarks/record_persistent_split_tape.py` (`persistent_tape_*`)
and `benchmarks/record_jit_attention_program.py` (`jit_attention_nvidia.json`);
both packets' `recorder_sha256` match a tracked revision of their script.

Cited by `benchmarks/NATIVE_STORAGE_FOLLOWUP.md` ("Loop 11: split persistent
tensor products and JIT-owned attention").
