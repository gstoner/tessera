# Native resident quantizer/product evidence

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-RESIDENT-QUANT-PRODUCT-20261017.

RX 9070 XT, gfx1201, UUID GPU-28d9e7efbf2ef716. Each packet records the live
rocminfo, compiler/runtime/recorder hashes, original Graph, native SSA plan
and both image hashes. The complete authored Graph is partitioned by C++;
Python marshals its member descriptors into the existing native owner.

The standalone quantizer semantic matcher is shared with this composed
partition. No Graph arithmetic, intermediate device allocation or numerical
conversion is performed by Python. Quantized code and scale buffers remain
private device storage between the two native members.

| M×N×K | Output | Max absolute error | Captured producer µs | Captured consumer µs | Ordinary composed event µs | Prepared invoke/read ms |
|---|---|---:|---:|---:|---:|---:|
| 19×33×64 | bf16 | 0.00771492 | 4.712 | 6.879 | 11.703 | 0.244 |
| 256×128×12288 | f32 | 6.32876e-05 | 28.217 | 348.462 | 378.159 | 0.603 |
| 3×17×32 | f32 | 1.45057e-07 | 4.693 | 4.878 | 10.198 | 0.231 |

Every case compares three changed-input generations with an independent
float64 block-scaled oracle before timing, retains prior returned arrays,
and verifies native status 10 for a consumer launched before producer outputs
are initialized. BF16 output uses rtol 0.005 / atol 0.01; FP32 uses 0.0002 /
0.0002. These are tolerance-based composed-product comparisons, not bitwise
FP32 accumulation proofs.

Captured windows group one member in a device graph and include graph dispatch;
they exclude capture, instantiation and host copies. Ordinary event windows
include host launch gaps. The native owner already averages over repetitions.
Prepared timings include invocation and returned-output readback and exclude
compilation and owner preparation. These domains are not interchangeable.

Reproduce with the checked-in recorder and corresponding authored source:

```sh
python benchmarks/rocm/benchmark_resident_quantized_product.py \
  --source benchmarks/baselines/rocm_resident_quantized_product_20261017/resident-product-3-17-32.mlir \
  --output ~/scratch/resident-product.json --repeats 50
```

This proves the named authored Graph and prepared C++ owner. Public JIT and
portable composed-package integration remain required, as do resident chaining
through gate/up/activation/down, full MLP and model-quality acceptance. No
speedup, sibling architecture parity or complete NVFP4-ingest claim is made.
