# gfx1201 folded frontend, layout-verified timing, and staging census

Tajasarus (RX 9070 XT, `gfx1201`) refreshed the matched packet at clean source
revision `6f8ba84674da52a710370dc07b83ac7ec2e13637` with the selected
ROCm toolkit and a freshly rebuilt LLVM/MLIR 23.1.1 compiler from parent
revision `89b2f2fd9f829a2316c42fb31a9879fb4a82cfea` (the follow-on changes
do not alter compiler sources). The authored Python frontend constructs a
`tessera.scaled_matmul` Graph op from typed operand shapes, lowers it through
Schedule, Tile, and Target, and packages the existing production folded HIP
kernel. Its receipt binds the selected BM256/BN64/BK64 schedule, Tile/Target
hashes, approximate policy, ABI, entry, the SHA-256 of the actual HSACO bytes,
and a separately named composite artifact-image digest. The expanded device
file passes 13/13: the previous 11 plus ragged `65×48×64` and multi-stage-K
`257×80×192` frontend cases. The exact K32 route remains the default and
oracle; the generic exact selector explicitly refuses the folded layout.

The [matched timing packet](matched.json) explicitly requires
`RADIANCE_MXFP4_WPERM=1`, matching the fragment-order weights handed to
pinned Radiance revision `dfdfa3832922c9a4253133f09c1f5c0d39748fc7`.
It verifies a sampled independent FP32-dequantized reference and full BF16
output equality on these lossless-fold inputs before alternating HIP-event
windows. The earlier v1 packet lacked that layout provenance and is not
selector-admissible.

| M×N×K | exact K32 | folded frontend | Radiance | folded/Radiance |
|---|---:|---:|---:|---:|
| 256×5120×8704 | 0.520409 ms | 0.181164 ms | 0.141740 ms | 1.28× |
| 1024×17408×5120 | 4.361921 ms | 1.244548 ms | 0.891527 ms | 1.40× |

The [static staging census](staging_census.json) disassembles Tessera's
selected HSACO and Radiance's exact TN2/WPERM/fast-epilogue symbol from its
pinned binary. Both have 32 FP8 WMMAs, eight LDS reads of each b64 form, and
two RDNA4 barrier pairs. Tessera has five static `global_load_b128` sites;
Radiance has twelve plus three `global_load_b64` sites. Those are static
instruction sites, not dynamic counts, and do not by themselves attribute
loads to A or B. Source inspection shows 16-byte A/B staging in Tessera;
Radiance stages packed E2M1 B with 16-byte/8-byte vector reads and folds it
to E4M3 in LDS.

For the two production shapes, the source/schedule model estimates Tessera's
expanded B requests at 44.6/356.5 MB, versus Radiance's packed B at
22.3/178.3 MB plus 1.4/11.1 MB of block scales. The same model estimates
178.3/1426.1 MB of A requests for each engine because A is restaged for each
N tile. These are requested global bytes before cache effects, not measured
DRAM traffic. The doubled B storage is a real difference but cannot alone
be assigned the measured timing gap. Next controlled experiments should vary
one A/B staging mechanism at a time and retain the same layout, inputs,
selected ISA, and HIP-event method.

The first [single-lever ablation](b_cache_ablation.json) did exactly that:
one B `global_load_b128` changed to the emitted RDNA4 non-temporal form,
with no other source change. The ISA guard found zero such sites in production
and one in the variant; all four engines had identical BF16 output on both
shapes. Alternating timing gave baseline 0.174913/1.130776 ms and variant
0.210501/2.308749 ms, a 20%/104% regression. The variant remains unselected.
This refutes B non-temporal caching as the next production lever on these
matched shapes, not the possibility that B traffic matters.

The older ablation's receipt metadata was corrected mechanically from its
already recorded payload `image_sha256`: `hsaco_sha256` now names those bytes,
and its former composite value is retained as `artifact_image_digest`. No
ablation timing or selection verdict was changed by that metadata repair.

Tajasarus is a WSL2 guest with neither `/dev/kfd` nor `/dev/dri`; there are
no admitted PC samples,
cross-CU clock result, or phase fractions in these packets. Neither the
approximate route nor profiler-derived selector training is promoted.
