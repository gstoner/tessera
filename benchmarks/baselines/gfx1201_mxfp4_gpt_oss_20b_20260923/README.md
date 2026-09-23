# GPT-OSS-20B MXFP4 capacity on Tajasarus

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-GPT-OSS-20B-MXFP4-CAPACITY-2026-09-23`.
The [inventory](model_inventory.json) is pinned to the
`openai/gpt-oss-20b` checkpoint revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. It was derived from
`config.json`, `model.safetensors.index.json`, and the HTTP range-fetched
headers of all three safetensors shards, not from a downloaded or loaded
checkpoint. The inventory records their SHA-256 digests; `header_bytes`
and `header_sha256` include each safetensors header's 8-byte length prefix.
All 24 layers have 32 experts, each with a down projection and a gate/up
projection:

| Projection | Source block shape | Source scale shape | Tessera N×K | Count |
| --- | --- | --- | ---: | ---: |
| down | `[32,2880,90,16]` U8 | `[32,2880,90]` U8 | 2880×2880 | 768 |
| gate/up | `[32,5760,90,16]` U8 | `[32,5760,90]` U8 | 5760×2880 | 768 |

The source's last block dimension holds 16 packed bytes for K32. Tessera's
generic W4A8 layer accounting additionally keeps one derived row-reference
byte per output row. The [exact-device packet](evidence.json) used the
Tajasarus RX 9070 XT/gfx1201 HIP memory capacity. Its `source_revision`
identifies the isolated worktree's base commit; the recorder and inventory
hashes bind the actual files used. Bytes are decimal:

| Quantity | Bytes |
| --- | ---: |
| Pinned checkpoint tensor bytes, all dtypes | 13,761,264,768 |
| Expert packed blocks, scales, and derived row references | 10,158,981,120 |
| Expert expanded weights alone | 19,110,297,600 |
| Expanded weights plus row references | 19,116,933,120 |
| Packed plus expanded expert representations | 29,275,914,240 |
| Device physical capacity | 16,974,905,344 |

The expanded expert weights alone exceed device capacity by 2,135,392,256
bytes. This is an absolute impossibility for a full-model expanded prefill
representation on this GPU, independent of other weights, activations,
fragmentation, or post-load headroom. It does not require a 13.76 GB
checkpoint download or a model-loaded memory reading to rule out that
specific design. The live free-byte snapshot is recorded for provenance
only; the model was **not loaded**, and it is not treated as an available
model budget. On a larger gfx1201 device the recorder still refuses
selection until model-owned post-load headroom is measured.

This says nothing yet about a bounded hot-expert cache, streaming expanded
tiles, or a selective per-layer materialization. Those need an allocation
policy, actual router/expert traffic, reserve for runtime state, and
model-specific BF16/edge-scale proof. The checkpoint's physical
`[expert,N,K/32,16]` blocks and `[expert,N,K/32]` scales also require an
explicit conversion contract before Tessera's current W4A8 package can
consume real GPT-OSS-20B bytes. The earlier six-shape kernel sweep used
synthetic exact-fold inputs at different N/K; it is not a GPT-OSS-20B
model-performance result. Automatic selection remains closed and exact
per-K32 numerical execution remains the oracle.
