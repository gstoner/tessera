# Composed scaled-product public JVP — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key GFX1201-COMPOSED-SCALED-JVP-2026-10-08.

The actual frontend traces two scaled_matmul products sharing encoded FP8
matrices, with independent FP32 scale pairs, followed by add. Public native_jvp
retains the complete Graph for verified native AD and Schedule/Tile packaging.
The ten-image HIP program returns the primal and selected/reordered scale JVP.
Frontend inspection validates SSA, product contracts and scale argument roles;
it does not implement backend arithmetic. Cache identity includes the entire
composed Graph. Warm calls forbid compiler subprocesses and reference execution.
No operation/dtype admission or generic coverage label is broadened.

Validation: five focused host tests, 357 shared contract/registry tests and
29 owning-device tests pass. The initial host fixture failure is retained;
its serialization target was repaired from the default CPU target to gfx1201.
Owning pytest reports an absent pytest-timeout configuration plugin warning;
these receipts do not establish timeout enforcement.

The owning packet records the actual AMD Radeon RX 9070 XT, HIP UUID, compiler
and runtime hashes, source hashes, ten image hashes and raw timing windows.
Both shapes pass independent float64 oracle comparison before and after timing.

| M,N,K | Ten-member native event median, ms | Public host median, ms | Maximum absolute error |
| --- | ---: | ---: | ---: |
| 17,19,256 | 0.0571272410 | 2.7485189070 | 3.36428e-7 |
| 3,5,37 | 0.0875226557 | 2.9781288256 | 4.52641e-8 |

Native event windows cover the whole ten-launch native program; they are not
isolated individual kernel timings. Public host timing includes copies,
completion and metadata. There is no matched-control speedup claim.

Reproduce with benchmarks/rocm/benchmark_composed_scaled_jvp.py on gfx1201,
using the matching source/compiler/runtime environment. Unit fixture:
tests/unit/test_composed_scaled_jvp.py. Owning regression:
tests/device/rocm/test_composed_scaled_jvp.py and test_public_scaled_jvp.py.

Open: composed reverse AD, general public maps, dynamic/nonleading/storage
derivatives, arbitrary Graph composition and generic batching/transpose closure.
No gfx1151, CUDA, Metal or x86 physical parity follows from this packet.

Documentation validation: 12 audit/recorder naming tests pass. The initial invocation named a nonexistent benchmark gate and ran no tests; that terminal log is retained separately.
