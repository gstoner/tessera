# Native quantizer -> NVFP4 expansion -> scaled product (gfx1201)

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-QUANTIZE-NVFP4-PRODUCT-20261017.

The public four-root Graph contains the original BF16 K16 max/normalize/E4M3
quantizer, packed NVFP4 weight expansion, and exact FP32 scaled product. C++
partitions the original SSA into three native members and derives all nine
buffer frames. The four intermediate code/scale planes remain private to one
native HIP owner. Python does not reconstruct numerical Tile/Target bodies or
copy intermediate results through host arrays.

Native program `tessera.native.quantized_product.v2`, portable package v2 and
frontend v2 preserve the existing two-member v1 meanings. Root role order is
compiler-derived. Independently valid incompatible members, missing producers,
resealed private lifetimes and invalid expansion scales/globals are rejected.
All members lower through Schedule/Tile and ROCm Target IR to LLVM/HSACO.

## Validation

Exact device: RX 9070 XT, gfx1201, GPU-28d9e7efbf2ef716. HIP reports
7.15.26333; compiler foundation is LLVM/MLIR 23.1.1. The native partition/v1
gate passes 19 tests. The package/v1 gate passed 35 tests before the additional
shared receipt-classification guard; the 64-case owning combined run includes
that guard. Six final new owning-device tests pass: four numerical shapes,
unchanged roots and retained results, complete Graph/Schedule/Tile/Target
reports, compiler-disabled warm and portable calls, fresh-process portable
replay, and private-plane readiness/invalidation after root updates.

The earlier metadata property reported fallback despite correct native receipts;
the property now classifies execution from the current complete native receipts.
A test initially counted Schedule metadata attributes as operation statements;
it now checks the actual native Schedule statements. Neither issue is presented
as numerical hardware failure or hidden by weakening native execution checks.

## Timings

Seven windows per shape; warm public windows average seven calls each.
Captured per-member windows average 128 replays and include device graph
dispatch. Ordinary program event windows average 50 repetitions and may include
CPU enqueue gaps. The native API already averages; no second division occurs.
Prepared invoke/read and complete public calls are recorded separately.

| M,N,K | Warm public ms | Quantizer / expansion / product captured microseconds | Ordinary program event microseconds |
|---|---:|---:|---:|
| 3,17,32 | 4.005 | 4.67 / 3.09 / 4.59 | 12.34 |
| 19,33,64 | 3.916 | 5.59 / 3.74 / 10.37 | 19.91 |
| 256,128,12288 | 4.694 | 27.61 / 7.83 / 115.67 | 154.98 |

Correctness gates timing. These are distinct regions, with no kernel-speedup or
default-strategy promotion claim. Source hashes, native image/descriptor receipts,
compiler SHA, portable artifact hashes and raw samples accompany each packet.
The compiler-age warning reflects source checkout mtimes after rebasing; the
native partition contents match the compiled implementation, and the warning is
retained. Frozen timing packets retain their measured source identity.

## Reproduce

On the owning host with the matching core compiler and multi-output HIP runtime:

```sh
python -m pytest tests/device/rocm/test_native_quantized_nvfp4_product_jit.py -q
python benchmarks/rocm/benchmark_public_quantized_nvfp4_product.py --output-dir ~/scratch/quantize-nvfp4-product --repeats 7
```

This proves the named static four-root product. Full gate/up/activation/down MLP,
model-quality acceptance, encoded AD, dynamic/general layouts and sibling
backend physical adoption remain open.

The final route-label rerun retains identical native member image digests.
Small-shape ordinary event windows varied from about 22.24 to 12.34 us and
35.10 to 19.91 us across the two runs, while the largest remained about 155 us.
`timing-repeat-control.json` preserves both observations. These measurements
are not evidence of an optimization or a stable speedup.

## Final shared gates and synchronization

The clean host CPU unit lane passes 22,202 tests (9,946 legitimate skips).
The focused package/registry/audit lane passes 363 tests and all 33 generated
documents are in sync. Fresh matching core and NVIDIA tools pass 12 existing
RTX5070 active-source JVP/ragged/ReLU controls; this is existing-profile
regression parity, with no adoption of the new gfx1201 program. Core rebuild
SHA is byte-identical to the measured gfx1201 compiler. `validation-gates.json`
proves full source-tree identity when rebasing this slice onto merged PR969;
`sm120-regression.json` binds the separate owning CUDA evidence.
