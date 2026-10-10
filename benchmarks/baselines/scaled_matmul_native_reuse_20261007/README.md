# Native scaled-program owner reuse

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. Owning device gfx1201 / RX 9070 XT.
Native C++ retains up to four idle owners within a 128 MiB allocation/key
budget. Matching uses exact image bytes and ABI fields, with process/device/
context identity. Every acquire receives a fresh handle; generations persist.
Uploads replace all inputs, active owners remain distinct, and failed completion
or cleanup owners remain quarantined. Explicit cache clear releases idle owners.

Ten owning-device tests pass, including scale JVP numerics, finite differences,
changed tangent inputs, fresh/stale handles and generations, concurrent live
allocation isolation, active-owner survival across idle cache clear, and cleanup.

Fresh-process cache off/on medians, milliseconds per public warm call:
| M/N/K | Off | On |
| --- | --- | --- |
| 17/19/256 | 6.69721 | 1.80197 |
| 32/32/256 | 6.80617 | 1.89503 |
| 200/19/256 | 6.80098 | 2.05390 |

Public warm calls prohibit compiler subprocesses. Separate prepared readback
and native-event windows accompany these results; event windows include native
enqueue gaps and are not isolated kernel measurements. These are named static
public-call gains, not general compiler/backend closure. No sibling execution,
generic batching/transpose, full-suite, or broad schedule promotion is claimed.
Recorder: benchmarks/rocm/benchmark_public_scaled_jvp.py.
Raw tests/build logs and source/runtime fingerprints accompany the packet.
