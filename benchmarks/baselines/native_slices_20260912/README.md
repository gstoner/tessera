# Native numerical and packaging slices

Owners: NUMPOL-CARRIER-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Parent: merged `145b32777413e93d256a36860529b8060f167d26`, plus the source
fingerprints in `sources.json`. Packets are correctness evidence; none promotes
a performance candidate or transfers evidence between architectures.

- `apple_denormal.json`: owning M1 Max, fresh compiler-generated MSL and fresh
  status-returning bridge. Both explicit policies pass 133,376 boundary/random
  pairs for each of add/sub/mul/div, checking signed zero and special values.
  See `benchmarks/apple_gpu/record_denormal_policy.py`. The host unit test
  compiles the exact emitted integer helper against a separate IEEE f32 oracle.
- `x86_absolute.json`: Princess-Luna Ryzen AI Max+ 395 CPU; three ranks/shapes
  cross vector boundaries and compare bitwise magnitude, including NaNs,
  infinities, subnormals and negative zero. See `record_absolute_migration.py`.
- `attention_3_5.json` and `attention_5_3.json`: Super-Bear SM120, CUDA 13.4.1,
  driver 610.88, rebuilt native runtime bridge. B=2, Hq=4, Hkv=2, causal/window
  masks plus a shared 1x1xQxK additive mask execute without host expansion.
  The f32 v2 ABI copies physical bias storage and rejects empty composed rows.
  Run `test_attention_broadcast.py` with `TESSERA_TEST_RAISED_ATTENTION=1` and
  `TESSERA_BROADCAST_EVIDENCE` naming the output directory.

The Apple carrier is bounded scalar f32 arithmetic; unimplemented floating
operations and unsafe flags refuse explicit policy. Query/key-axis broadcast,
Boolean/padding masks, other attention backends/storage and the remaining
Graph-owned packages remain open. WSL timings and these correctness probes are
not selector-grade performance evidence. No full IEEE input-space enumeration
or general exception-flag/rounding-mode support is claimed.
