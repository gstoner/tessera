# gfx1151 typed fragment-store timing, 2026-09-28

Princess-Luna (`gfx1151`, WSL2), Tessera `2a70e08a`, freshly rebuilt
`build/tools/tessera-opt/tessera-opt` (LLVM/MLIR 23.1.1). The adjacent
[`typed_route_gap.json`](typed_route_gap.json) is the unedited output of:

```bash
TESSERA_OPT=$PWD/build/tools/tessera-opt/tessera-opt TESSERA_ROCM_CHIP=gfx1151 \
  .venv/bin/python benchmarks/rocm/record_typed_route_gap.py \
  --shapes 513x769x257,1009x1537x1025 --panels 32x64,64x64 \
  --dtype fp16 --runs 3 --iters 30 \
  --output /tmp/tessera-gfx1151-typed-edge-store-prodpanel-20260928.json
```

The `typed:*` variants originate in Graph → Schedule → Tile and consume the
`TileToROCM` bounded fragment store changed in #873. The `directive:*`
variants use a different generator and are context only. The production
Schedule panel for both selected shapes is 32x64; the recorder also measures
re-panelled 64x64 and LDS variants. For the production panel:

| M×N×K | Typed 32x64 | Directive 32x64 | Typed relative error |
| --- | ---: | ---: | ---: |
| 513×769×257 | 0.0311 ms | 0.0312 ms | 1.32e-6 |
| 1009×1537×1025 | 0.2208 ms | 0.2162 ms | 3.69e-6 |

These are medians of three fresh-process runs, each with interleaved batches
of 30 launches, using synchronized host wall time. The recorder marks both
correctness and performance promotion ineligible: its numerical comparison
is a screen, not the owning device conformance gate, and WSL2 exposes no
usable HIP-event interval or device counters. No pre-#873 binary was measured
in this packet, so it does not establish the store's speedup or regression.
