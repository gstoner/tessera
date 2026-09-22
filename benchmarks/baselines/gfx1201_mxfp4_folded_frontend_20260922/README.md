# gfx1201 folded frontend, layout-verified timing, and staging census

Tajasarus (RX 9070 XT, `gfx1201`) tested clean source revision
`489c386595bf57890d8727ede97eec0f7c5a2a97` with LLVM/MLIR 23.1.1 and
the selected ROCm toolkit. The authored Python frontend constructs a
`tessera.scaled_matmul` Graph op from typed operand shapes, lowers it through
Schedule, Tile, and Target, and packages the existing production folded HIP
kernel. Its receipt binds the selected BM256/BN64/BK64 schedule, Tile/Target
hashes, approximate policy, ABI, entry, and HSACO identity. The full device
file passed 11/11: three broader lossless shapes (including both prefill
dimensions), one deliberately lossy frontend oracle, and the seven previous
physical-ABI cases. The exact K32 route remains the default and oracle.

The [matched timing packet](matched.json) explicitly requires
`RADIANCE_MXFP4_WPERM=1`, matching the fragment-order weights handed to
pinned Radiance revision `dfdfa3832922c9a4253133f09c1f5c0d39748fc7`.
It verifies a sampled independent FP32-dequantized reference and full BF16
output equality on these lossless-fold inputs before alternating HIP-event
windows. The earlier v1 packet lacked that layout provenance and is not
selector-admissible.

| M×N×K | exact K32 | folded frontend | Radiance | folded/Radiance |
|---|---:|---:|---:|---:|
| 256×5120×8704 | 0.517379 ms | 0.182356 ms | 0.142856 ms | 1.28× |
| 1024×17408×5120 | 4.359824 ms | 1.242242 ms | 0.888937 ms | 1.40× |

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

Tajasarus still exposes no `/dev/kfd`; there are no admitted PC samples,
cross-CU clock result, or phase fractions in these packets. Neither the
approximate route nor profiler-derived selector training is promoted.
