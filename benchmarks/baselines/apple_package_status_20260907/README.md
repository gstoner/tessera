# Native package preparation after the status boundary

Five independent M1 Max processes compare the complete native attention-backward
package against the identical direct runtime ABI (five paired trials, 20 calls
per trial). Each report binds runtime/toolchain, package image, Schedule and Tile
identity. These are host complete-call measurements, not kernel timestamps.
All reported calls passed native status and numerical comparison.

Package/direct median latency ratios:

| Run | f16 | bf16 |
|---|---:|---:|
| 1 | 1.159 | 1.124 |
| 2 | 1.126 | 1.139 |
| 3 | 0.926 | 1.126 |
| 4 | 1.450 | 1.101 |
| 5 | 0.590 | 1.104 |

**No promotion.** bf16 loses all five runs; f16 has mixed wins and high variation.
The direct incumbent remains selected. No timing threshold was relaxed.

Exploratory cProfile runs identified NumPy dtype spelling and buffer-contract
preparation as avoidable host work. Only immutable dtype spelling is now cached;
shape, layout, address alignment and invocation validation are still read for
every call. Descriptor provenance and dynamic expressions remain mutable JSON,
so their identities are recomputed rather than returning a stale cached digest.
The exploratory profiles had different warm/build conditions and are not a
controlled before/after speedup measurement. The five uninstrumented reports
above are the package-versus-incumbent comparison, not a claim about isolated
cache speedup.

Reproduce with `benchmarks/apple_gpu/benchmark_native_backward_package.py
--trials 5 --reps 20 --output <report.json>` in five fresh processes, with
`TESSERA_OPT` and `TESSERA_APPLE_GPU_RUNTIME_LIB` pointing to the intended fresh
compiler and shared runtime. This experiment does not replace the separately
sealed fleet packet; runtime source must be committed before that packet is
remeasured.
