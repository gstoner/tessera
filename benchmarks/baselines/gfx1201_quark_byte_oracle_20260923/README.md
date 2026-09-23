# Pinned Quark checkpoint byte-oracle preflight

Owner: `ROCM-MXFP4-W4A8-1 / IKF-1`. Sync key:
`GFX1201-QUARK-BYTE-ORACLE-2026-09-23`.

This is a host-only candidate decode, not a checkpoint conversion certificate,
W4A4 executable ABI, or gfx1201 numerical proof. Automatic selection remains
closed. The source is the [AMD Qwen3.8-27B Quark AWQ MXFP4 checkpoint](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4/tree/5233554c5fa56afda40150556b95573c2d7d29c0)
at revision `5233554c5fa56afda40150556b95573c2d7d29c0`. Its
`config.json` SHA-256 is
`4a139d3e01df039e17b4c8b8362f7944674d76051a8cd96f213c6bc332d69f55`
and identifies Quark exporter version `0.13+unknown`. The safetensors file's
HTTP-linked SHA-256 is
`be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272`;
its first 219,640 bytes (8-byte length plus JSON header) hash to
`01453e3d08a6275905b19dd15ac582fba065a551bc392ca22af0588d476f2083`.
The JSON header length is 219,632 bytes. Only bounded HTTP ranges were
downloaded; the full-file hash was not recomputed locally.

The header gives gate projection weight `[17408,2560]` at data offset
`6970630624` and scale `[17408,160]` at `7015195104`; down projection
weight `[5120,8704]` starts at `6923280864` and scale `[5120,544]` at
`6967845344`. Offsets below include the 219,640-byte safetensors prefix:

| Tensor | Absolute HTTP byte range | First-row bytes | SHA-256 |
| --- | --- | --- | --- |
| Gate weight, K=0..127 | `6970850264-6970850327` | `ca2c11c63a2569da...` | `3d2eb71ad973ad544f41d22be1d9ab9c82e0d67fa6862c415d84fff70f591a55` |
| Gate scales, first four K32 groups | `7015414744-7015414747` | `77777777` | `6b4a1673b225e8bf5f093b91be8c864427df32ca41b17cc0b82112b8f0185e41` |
| Down weight, K=0..127 | `6923500504-6923500567` | `0abf6eefaf61a577...` | `a461a6ce74bdcb5190cfac9ab90b018405cc91ebffcca9dfa4889ccd8a7ccd9b` |
| Down scales, first four K32 groups | `6968064984-6968064987` | `76777777` | `9b8b7e4c918c83e315361b93cdac8821c918c77d3b5eab00e76546d121cffd57` |

The complete 64-byte weight samples and their hashes are pinned in
`tests/unit/test_rocm_mxfp4_quark.py`. Under the low-even, biased-E8M0
*hypothesis*, gate starts `[-1,-2,-2,1,0.5,0.5,4,-2] × 2^-8`, and down
starts `[-1,0,-6,-1.5,-4,4,-6,-4] × 2^-9`. The host oracle computes all
16 E2M1 code values independently and refuses scale codes 0 and 255, whose
checkpoint-specific semantics are not established here.

The [Quark 0.12 FP4 packer](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/utils/pack.py#L489-L522)
places even K in the low nibble and ignores its `reorder` argument for FP4.
The checkpoint declares `dtype=fp4`, which in Quark 0.12 dispatches through
[the scaled real quantizer](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/export/nn/modules/realquantizer.py#L627-L700),
not the distinct `dtype=mx` packer. Its
[E8M0 conversion](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/utils/numerics.py#L60-L68)
adds bias 127, consistent with sampled codes `0x76`–`0x78`. The 0.12 source
therefore supports the candidate byte interpretation, but the checkpoint
names a later `0.13+unknown` exporter. It is not an exact producer
certificate, and the checkpoint README supplies no independent projection
output fixture.

Next gate: obtain the matching 0.13 export implementation or an independently
dequantized projection slice with its source/scale policy; compare every code
and scale edge, then certify conversion separately. Only after that should a
distinct W4A4 activation/scale carrier and gfx1201 BF16 proof be attempted.
