# Evidence consumer, NVIDIA fragment, and residual slice — 2026-09-29

Owners: EVIDENCE-PACKET-1, W1.1, FRONTEND-IR-MEDIUM-1, AD-RESIDUAL-EVAL-1.
Synchronization key: `COMPILER-EVIDENCE-FRAGMENT-RESIDUAL-2026-09-29`.

## ROCm physical math

The gfx1151 `sum` diagnostic now constructs a typed Graph module, lowers it
through native Schedule and Tile IR, reloads the native image and checked
descriptor from JSON, and requires the launch receipt to match all three
artifact, image and descriptor identities. The benchmark packet covers 21
rows (seven operations × f32/f16/bf16); the three `sum` rows carry serialized
package receipts. The other 18 rows remain explicitly labeled metadata
runtime probes. The packet cannot promote a selector.

Princess-Luna gfx1151 used a compiler rebuilt from this source and passed
f32/f16/bf16 `sum` against NumPy, followed by five warm host-wrapper samples
per row in `math_gfx1151.json`. Maximum absolute errors were 0.000244140625,
0 and 0 respectively. The measured warm medians were 2.832, 2.690 and
2.345 ms. Those include runtime transport and synchronization; they are not
kernel times. ROCm `exp`, binary, and scan rows still need native package
contracts, as do gfx1201 math rows.

## SM120 fragment producer boundary

The existing `tessera-nvidia-pipeline-sm120` still emits tensor-valued
`tile.mma` with async tokens for the two legacy TileIRLoweringPass producers.
Passing this form directly to `--lower-tile-to-nvidia=sm=120` previously
created an invalid `tessera_nvidia.mma_sync` with an async token in an MMA
data slot. The NVIDIA lowering now refuses that form at `tile.mma`, naming
the missing typed fragment and accumulator materialization.

Super-Bear built the NVIDIA compiler with CUDA 13.4. The captured legacy
producer fixture fails with the new diagnostic. The typed accumulator-loop
fixture still passes FileCheck. This is a fail-closed boundary, not migration
of the two producers. Their tensor-to-fragment materializer and accumulator
threading remain W1.1 work; the pointer-backed typed route is unchanged.

## Public frontend residual

The public traced tape now has a second exact-device diagnostic: the coupled
residual `(x*x - theta) * (x + theta)`. This tests shared inputs, four
canonical Graph operations, a nonzero primal, saved-input isolation after
caller mutation, repeated backward, and closed-frame refusal. Independent
analytic derivatives are checked at every backward call. Width 17, f32 uses
136 saved-input bytes and no exported residual tensor bytes.

`coupled_gfx1151.json` and `coupled_sm120.json` retain compiler, LLVM,
lineage, ABI and binding identities plus raw samples. Median synchronized
host backward calls were 0.411 ms on Princess-Luna gfx1151 and 0.194 ms on
Super-Bear sm_120. These include allocation, launch and synchronization;
they do not establish device kernel time or a cross-host performance result.
General solver/frontend wiring, dynamic layouts and aliases remain open.
