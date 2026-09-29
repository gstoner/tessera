# gfx1151 math and residual follow-up, 2026-09-29

Princess-Luna gfx1151, host WSL, PR #882 merged source with the device
identity follow-up. The physical math recorder used three warm host-wall samples per row. The corrected
recorder queries the selected HIP device with `hipGetDevicePropertiesR0600`,
requires live `gfx1151`, and rejects a different configured compiler chip
before recording any row; the packet stores both identities. It produced 21 rows:
three serialized Graph/Schedule/Tile/Target `sum` packages and 18 explicit
metadata runtime probes across `sqrt`, `exp`, `add`, `div`, `cumsum`,
and `cummax` at f32, f16, and bf16. All passed their numerical gates.
These timings include Python and launch overhead; they are diagnostic and
do not establish kernel speed or selector promotion. The remaining six
operation families need native package ownership before they can count as
evidence consumers.

The persistent split tape recorder passed three exact-device nested SAVE
cases (widths 4, 8, 16) on each of Princess-Luna gfx1151 and Super-Bear sm_120, including residual capture, repeated backward,
caller mutation isolation, and a controlled device-residual mutation. The
runtime now checks that forward and backward product contracts agree on
residual source identity, primal result count, and every exported residual
slot before binding and at package validation.

The NVIDIA TileIRLoweringPass still has two tensor-valued `tile.mma`
producers. The sm_120 guard introduced in the parent PR refuses their
unsupported physical lowering. No typed producer migration or NVIDIA fragment proof is claimed by this packet.

Files: `gfx1151_math.json`, `gfx1151_residual.json`, `sm120_residual.json`.
