# Recommended tessera pipelines

## Build‑time
- Enable new passes in your `tessera-opt`:
  - `-tessera-kv-cache-blockify`
  - `-tessera-parallel-decode-expand`
  - `-tessera-attn-bidir-windowing`
  - `-tessera-conf-validate-merge`
  - `-tessera-cow-cache-dedup`

## Example CLI
```bash
tessera-opt input.mlir   -tessera-kv-cache-blockify   -tessera-parallel-decode-expand='K=4'   -tessera-attn-bidir-windowing='window=16'   -tessera-conf-validate-merge='tau=0.75,window=8'   -tessera-cow-cache-dedup   | tessera-compile -target nvidia-sm90 -o fast_dllm_kernel
```
