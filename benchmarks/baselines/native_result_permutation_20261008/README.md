# Native gfx1201 computed-result permutation

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Sync MAP-RESULT-PERMUTATION-20261008.

Textual Graph packages execute MXFP8 E4M3/E8M0 K32 scaled products followed
by native f32 transpose through Schedule, Tile, GPU Target, ROCDL/LLVM and
checked HIP ownership. Python marshals bytes and metadata.

Owning RX 9070 XT/gfx1201: five numerical/lifetime cases passed:
17x19x64, 1x33x32, 33x1x65, 31x47x95, 64x65x128.
Six native compiler tests cover ownership, bindings, invalid axes and altered
Schedule/Tile replay. Profiling checks results after grouped captured execution
and subsequent ordinary invocation.

gfx1201.json records live inventory, compiler/provider digests and timing.
Captured member medians in microseconds:

| M,N,K | Product | Movement | Upload/execute/readback |
| --- | ---: | ---: | ---: |
| 17,19,64 | 10.475 | 2.771 | 354.521 |
| 31,47,95 | 8.628 | 2.807 | 398.931 |
| 64,65,128 | 16.907 | 1.818 | 415.521 |

Captured windows exceed 20 ms (minimum 29.032 ms). Capture and instantiation
are outside event windows; device graph dispatch remains included.
Grouped member timing is diagnostic attribution, distinct from ordinary
interleaved program event timing. Both are recorded. Host timing includes
update/upload, invocation and copied readback, excluding compilation.
No speedup/default-route claim is made.

Reproduce: benchmarks/rocm/record_scaled_result_permutation.py --output
<scratch>/packet.json, with matching native tools/provider on gfx1201.

Public nonleading output-axis JIT and reverse cotangent integration remain
open. Textual packages do not prove those routes. No sibling parity is inferred.
