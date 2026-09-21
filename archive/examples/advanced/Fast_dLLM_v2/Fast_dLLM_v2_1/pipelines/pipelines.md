# Pipelines (updated with new pass skeletons)

```bash
tessera-opt input.mlir \
  -tessera-kv-cache-blockify \
  -tessera-parallel-decode-expand='K=4' \
  -tessera-attn-bidir-windowing='window=16' \
  -tessera-conf-validate-merge='tau=0.75,window=8' \
  -tessera-cow-cache-dedup
```
