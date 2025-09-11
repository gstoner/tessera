
# CPX Pass Pipelines

<!-- MERGE-START: CPX_Pass_Pipelines -->
## -tessera-partition-longcontext
Partitions long‑context subgraphs to **CPX** and decode loops to **Rubin**; inserts `kv.export`/`kv.import`.

## -tessera-lower-kv-transport
Chooses **PCIe Gen6 + CX‑9** vs **NVLink** and lowers KV ops to runtime calls. Exposes `--chunksize`, `--prefetch_distance`.

## -tessera-vectorize-nvfp4
Legalizes matmul/attention tiles to NVFP4 MMAs with FP16/FP32 accumulators.

## -tessera-fuse-video-ingest
Fuses `video.decode → patchify → tokenizer → prefill_fused` on CPX.
<!-- MERGE-END: CPX_Pass_Pipelines -->
