# Public resident quantizer/product integration

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-RESIDENT-QUANT-PRODUCT-20261017.

The authored Python function emits the original BF16 max/scale/normalize/RNE
Graph and exact E4M3 scaled product. Native MLIR owns semantic admission,
two-member partition, Schedule/Tile lowering, private buffer lifetimes and
ROCDL/LLVM/HSACO generation. The existing checked C++ owner executes both
members on its private HIP stream; Python performs no quantization or matmul. The new consumer uses unprojected
checked native Target IR; it does not use Python shape-free Target rewriting.

## Exact-device checks

RX 9070 XT / gfx1201 / GPU-28d9e7efbf2ef716. Five public device tests cover
three shapes, changed roots, retained results, no-eager-quantization traps,
keyword and positional portable bindings, compiler-disabled warm calls and
fresh-process replay. A mutated inspection artifact cannot rewrite the cached
private compiler product. Nineteen package/contract tests cover source admission,
re-sealed buffer/lifetime corruption and parent-stage mutation before provider
access. FP32 composed outputs use rtol/atol 0.0002; this is a tolerance-based
comparison with an independent float64 block-scaled oracle.

## Public-call overhead

| M×N×K | Warm public call ms | Serialized-dispatch control ms | Max absolute error |
|---|---:|---:|---:|
| 19×33×64 | 3.153 | 5.962 | 3.33765e-07 |
| 256×128×12288 | 5.625 | 6.714 | 5.47557e-05 |
| 3×17×32 | 3.270 | 5.968 | 1.01893e-07 |

The control reconstructs the artifact and uses portable dispatch with the same
native members. It omits frontend dispatch, so it is not a reconstruction of
every cost of the old public call. The optimized ordinary call reuses its
verified private program and creates an artifact only for inspection/export.
Both paths validate the program and use the same checked C++ owner with current
host roots. Portable artifact validation remains in the portable launch path.

Public windows include checked owner preparation, host-root copies, native
invocation and readback. Cold windows also include tracing and compilation,
but exclude decorator construction. These are not device-kernel timings.

The resident-product JSON packets separately record captured producer/consumer
device windows, ordinary launch-inclusive HIP windows, and prepared invoke/read
windows through the production package. The native API already averages over
repetitions; the recorder does not divide its result a second time.

Portable public artifacts are checked in beside the packets. Source hashes
bind the measured Python package, runtime integration and original native
partition/quantizer sources. Native compiler SHA and runtime SHA are recorded
in each prepared packet. The compiler-age warning reflects checkout mtimes;
the native source hashes remain those of the compiled 554c74ef9f implementation.

## Reproduction

```sh
python benchmarks/rocm/benchmark_public_quantized_product.py \
  --output-dir ~/scratch/public-quantized-product --repeats 7
python benchmarks/rocm/benchmark_resident_quantized_product.py \
  --source benchmarks/baselines/rocm_resident_quantized_product_20261017/resident-product-3-17-32.mlir \
  --output ~/scratch/prepared-quantized-product.json --repeats 50
```

This closes the named public three-input static FP32-output edge. Native BF16
output has separate prepared-route proof in the earlier packet. Bounded or
general layouts, nested frontend helper inlining, encoded AD, NVFP4 expansion
chaining, full gate/up/activation/down MLP and model-quality acceptance remain
open. No sibling hardware parity, kernel speedup or complete NVFP4-ingest
closure is claimed.

## Shared-runtime regression gate

Twelve existing RTX5070 SM120 producer/ragged/ReLU controls pass with matching
core and NVIDIA tools. `sm120-regression.json` records the exact UUID, source
revision and compiler/provider/fixture hashes. This is existing-profile
regression evidence; the new quantized-product program remains gfx1201-only.

## Cross-member interface check

`producer-interface-counterexample.json` records an individually valid K64
quantizer substituted into a K32 composed plan. The old validator admitted the
re-sealed record; the new validator compares native producer shape/storage/byte
spans with all three shared interfaces and rejects it before provider access.
The malformed program was not launched.

## PR969 review corrections

Per-call quantized-product state, receipts and its inspection cache are reset
before call validation, including a rejected later specialization. The portable
consumer now requires the checked Target witness to equal the Target bound to
its executable image. Both regressions fail before the fix and pass afterward;
the complete affected package/device lane passes 26 tests (21 package, five
owning gfx1201). `review-source-identity.json` binds the corrected sources and
`review-regressions.json` records this validation. Existing timing packets keep
their measured pre-review source hashes; no native arithmetic or timing claim
changes.
