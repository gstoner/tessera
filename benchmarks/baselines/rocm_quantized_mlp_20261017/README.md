# Native quantized MLP — gfx1201, 2026-10-17

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-QUANTIZED-MLP-20261017. Follows merged PR971.

The authored Python Graph executes through native MLIR partition, seven
Schedule/Tile/ROCm Target/LLVM/HSACO members and one checked C++ HIP owner:
input quantization, gate/up expansion, BF16 parent product/store, SiLU BF16
activation, activation quantization, down expansion and BF16 down product/store.
Eighteen buffers include seven roots; logical views and FP32 accumulators do
not escape into allocations. The public route is
`canonical_rocm_quantized_mlp_program`; native/package/frontend v4 is additive.

Owning GPU: RX9070XT / gfx1201, UUID GPU-28d9e7efbf2ef716. HIP reports
7.15.26333. Packets pin the live device, source/compiler/provider/recorder and
reference hashes, all seven receipts and portable artifacts.

## Numerical evidence

Eight device cases pass prepared and public execution, changed roots, retained
outputs, downstream readiness invalidation, compiler-free warm/portable replay,
fresh-process replay, nested metadata isolation and eager-math traps.

Checks follow explicit rounding boundaries: independent FP64 products check
the BF16 parent; FP32 SiLU is checked at the actual stored parent; the down
product is checked at its actual activation. Resident results equal staged
native components bitwise. Per-stage rtol 0.015625 / atol 0.00002 is unchanged.
The final certificate bounds decoded quantized-input perturbations by the
absolute weight product, plus measured terminal-store rounding and the
per-stage allowance. Diagnostic staged owners run before timing; production
execution has one owner and no intermediate host readback.

The large case has 13 outputs outside the same flat relative gate against
an ideal earlier FP64 activation; its maximum full-reference error is 0.0625.
That disagreement is retained, not reported as a flat-oracle pass. In a
separate retained-update seed, a parent rounding-midpoint crossing (-6.5 in
the FP64-rounded reference versus -6.46875 native BF16) is amplified by SiLU.
The activation at the actual native parent matches NumPy exactly in that
attribution. These results are not checkpoint/model-quality acceptance.

## Separate timing scopes

| M / H / K / N | Public warm ms | Program event us | Captured seven members us |
|---|---:|---:|---|
| 1 / 16 / 16 / 1 | 11.686 | 36.24 | 5.57 / 3.96 / 6.83 / 2.38 / 5.55 / 3.80 / 5.34 |
| 3 / 32 / 48 / 19 | 11.496 | 36.79 | 5.60 / 4.22 / 8.22 / 2.48 / 5.58 / 4.07 / 5.93 |
| 256 / 128 / 12288 / 256 | 12.934 | 177.70 | 27.42 / 15.13 / 118.09 / 2.06 / 3.94 / 2.83 / 4.51 |

Member order follows the seven stages above. Public calls include validation,
root upload, owner preparation, execution and readback. Prepared invoke/read
excludes preparation and root upload. Ordinary event windows include enqueue
gaps and are already averaged by the native API. Captured members include
graph dispatch and exclude capture/instantiation/copies. Medians from these
domains are not interchangeable. No speedup or default-selector promotion.

## Validation and remaining work

Focused native/public capture/store/legacy/registry gate: 362 passed. Dedicated
execution matrix gate: 15 passed. Fresh matching core/NVIDIA build passes the
85-case existing RTX5070 active-source JVP/runtime fixture. Broad WSL CPU validation passes 22,206 tests (9,946 skipped; 11 warnings);
audit/metadata/diagnostic gates pass 311 tests. All 33 generated documents pass the final drift check. Apple/x86 physical adoption is not applicable
to this profile; gfx1151 lacks FP8 WMMA. No sibling physical parity is inferred.

Checkpoint/model quality, representative performance and host dispatch cost,
wider layouts/dynamic shapes, encoded AD and the original five-slice
NVIDIA/ROCm requirements remain open.
