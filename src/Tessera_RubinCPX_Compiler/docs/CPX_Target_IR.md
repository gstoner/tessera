
# CPX Target-IR (Device kinds, types, caps, and ops)

<!-- MERGE-START: CPX_Target_IR -->
This document defines the **NVIDIA Rubin CPX** extensions for Tessera Target‑IR.

- **Device kinds:** `#tessera.target.device<"nvidia.rubin_cpx">`
- **Types:** `nvfp4`, `nvfp6`
- **Capabilities:** `has_context_accel, has_hw_{venc,vdec}, mem.gddr7, p2p.pcie_gen6, link.cx9`
- **Memory spaces:** `gddr7.global`, `hbm.global`, `nvlink.scratch`, `host.pcie`

## Ops
- `tessera.kv.cache(role=keys|values, layout=row_major|paged)`
- `tessera.kv.export(policy="pcie+cx9"|"nvlink", chunk_bytes)` / `tessera.kv.import`
- `tessera.kv.prefetch(range)`
- `tessera.attn.prefill_fused(q,k,v,kv_cache,seq_len)`
- `tessera.video.decode/encode(codec)`

### Verification notes
- `kv.export` requires `chunk_bytes > 0`.
- `attn.prefill_fused` must write into a declared `kv.cache` region when present.
<!-- MERGE-END: CPX_Target_IR -->
