# MLIR Verifier Coverage Dashboard

Human-readable view. The canonical machine-readable artifact is `verifier_coverage.csv` in this directory — that CSV is what the drift gate compares. Don't edit either by hand; run `python -m tessera.compiler.audit verifier_coverage --write` (or `scripts/check_generated_docs.sh --write`) to refresh both. Drift is gated by `tests/unit/test_verifier_coverage.py`.

## Summary

| Status | Count | Meaning |
|--------|-------|---------|
| `real` | 47 | `hasVerifier = 1;` + substantive `verify()` body. |
| `trivial_stub` | 9 | `hasVerifier = 1;` + trivial `return success();` stub. |
| `absent` | 0 | `hasVerifier = 1;` but no `verify()` body (build error risk). |
| `no_verifier` | 80 | No verifier declared.  TD constraints suffice — fine for many ops. |
| **Total** | 136 | |

## Per-dialect details

### `src/compiler/ir/TesseraOps.td`

| Op | Status |
|----|--------|
| `ALiBiOp` | `no_verifier` |
| `AdafactorOp` | `no_verifier` |
| `AdamOp` | `no_verifier` |
| `AdamWOp` | `no_verifier` |
| `AllGatherOp` | `no_verifier` |
| `AllReduceOp` | `no_verifier` |
| `ArchGumbelSoftmaxOp` | `trivial_stub` |
| `ArchHardConcreteOp` | `trivial_stub` |
| `ArchMixedOp` | `trivial_stub` |
| `ArchParameterOp` | `trivial_stub` |
| `ArchSTEOneHotOp` | `trivial_stub` |
| `ArchSwitchOp` | `trivial_stub` |
| `ArchWeightedSumOp` | `trivial_stub` |
| `AttnCompressedBlocksOp` | `no_verifier` |
| `AttnLocalWindow2DOp` | `real` |
| `AttnSlidingWindowOp` | `no_verifier` |
| `AttnTopKBlocksOp` | `no_verifier` |
| `BatchedGemmOp` | `real` |
| `CachePageLookupOp` | `no_verifier` |
| `CastOp` | `real` |
| `CholeskyOp` | `real` |
| `CholeskySolveOp` | `real` |
| `CliffordGeometricProductOp` | `real` |
| `CliffordGradeProjectOp` | `real` |
| `CliffordInnerProductOp` | `real` |
| `CliffordNormOp` | `real` |
| `CliffordOuterProductOp` | `real` |
| `CliffordReverseOp` | `real` |
| `CliffordRotorSandwichOp` | `real` |
| `Conv2DNHWCOp` | `real` |
| `CustomAdjointCallOp` | `no_verifier` |
| `DCTOp` | `no_verifier` |
| `DeepSeekSparseAttentionOp` | `no_verifier` |
| `DequantizeFP4Op` | `no_verifier` |
| `DequantizeFP8Op` | `no_verifier` |
| `DropoutOp` | `real` |
| `EBMBivectorLangevinStepOp` | `real` |
| `EBMDecodeInitOp` | `real` |
| `EBMEnergyQuadraticOp` | `real` |
| `EBMInnerStepOp` | `real` |
| `EBMLangevinStepOp` | `real` |
| `EBMLangevinStepPhiloxOp` | `real` |
| `EBMPartitionExactOp` | `real` |
| `EBMRefinementOp` | `real` |
| `EBMSelfVerifyOp` | `real` |
| `EBMSphereLangevinStepOp` | `real` |
| `FFTOp` | `no_verifier` |
| `FlashAttnOp` | `real` |
| `FusedEpilogueOp` | `real` |
| `GQAAttentionOp` | `no_verifier` |
| `GatedAttentionOp` | `no_verifier` |
| `GatedDeltaNetOp` | `no_verifier` |
| `GeluOp` | `no_verifier` |
| `HybridAttentionOp` | `no_verifier` |
| `IFFTOp` | `no_verifier` |
| `IRFFTOp` | `no_verifier` |
| `KVCacheAppendOp` | `no_verifier` |
| `KVCacheCreateOp` | `trivial_stub` |
| `KVCachePruneOp` | `no_verifier` |
| `KimiDeltaAttentionOp` | `no_verifier` |
| `LUOp` | `real` |
| `LatentKVCompressOp` | `no_verifier` |
| `LatentKVExpandKOp` | `no_verifier` |
| `LatentKVExpandVOp` | `no_verifier` |
| `LayerNormOp` | `real` |
| `LightningAttentionOp` | `no_verifier` |
| `LinearAttnOp` | `no_verifier` |
| `LinearAttnStateOp` | `no_verifier` |
| `LionOp` | `no_verifier` |
| `LogSoftmaxOp` | `no_verifier` |
| `MLADecodeFusedOp` | `no_verifier` |
| `MLADecodeOp` | `no_verifier` |
| `MQAAttentionOp` | `no_verifier` |
| `MatmulOp` | `real` |
| `ModifiedDeltaAttentionOp` | `no_verifier` |
| `MoeCombineOp` | `no_verifier` |
| `MoeDispatchOp` | `real` |
| `MomentumOp` | `no_verifier` |
| `MorPartitionOp` | `no_verifier` |
| `MorRouterOp` | `no_verifier` |
| `MorScatterOp` | `no_verifier` |
| `MultiHeadAttentionOp` | `no_verifier` |
| `NTKRopeOp` | `no_verifier` |
| `NativeSparseAttnFusedOp` | `no_verifier` |
| `NeighborsHaloExchangeOp` | `no_verifier` |
| `NeighborsHaloPackOp` | `no_verifier` |
| `NeighborsHaloRegionOp` | `no_verifier` |
| `NeighborsHaloTransportOp` | `no_verifier` |
| `NeighborsHaloUnpackOp` | `no_verifier` |
| `NeighborsNeighborReadOp` | `no_verifier` |
| `NeighborsPipelineConfigOp` | `no_verifier` |
| `NeighborsStencilApplyOp` | `no_verifier` |
| `NeighborsStencilDefineOp` | `no_verifier` |
| `NeighborsTopologyCreateOp` | `no_verifier` |
| `PowerAttnOp` | `no_verifier` |
| `QROp` | `real` |
| `QuantizeFP4Op` | `no_verifier` |
| `QuantizeFP8Op` | `no_verifier` |
| `RFFTOp` | `no_verifier` |
| `RLCISPOPolicyLossOp` | `real` |
| `RLGRPOPolicyLossOp` | `real` |
| `RLNormalizeGroupAdvantagesOp` | `real` |
| `RLPPOPolicyLossOp` | `real` |
| `RMSNormSafeOp` | `no_verifier` |
| `ReduceScatterOp` | `no_verifier` |
| `ReluOp` | `no_verifier` |
| `ReshapeOp` | `real` |
| `RetentionOp` | `no_verifier` |
| `RingCreateOp` | `trivial_stub` |
| `RmsNormOp` | `no_verifier` |
| `RopeMergeOp` | `no_verifier` |
| `RopeOp` | `real` |
| `RopeSplitOp` | `no_verifier` |
| `SVDOp` | `real` |
| `SigmoidOp` | `no_verifier` |
| `SiluMulOp` | `no_verifier` |
| `SiluOp` | `no_verifier` |
| `SinOp` | `no_verifier` |
| `SoftmaxOp` | `real` |
| `SoftmaxSafeOp` | `no_verifier` |
| `SoftplusOp` | `no_verifier` |
| `SpectralConvOp` | `no_verifier` |
| `SwigluFusedOp` | `no_verifier` |
| `TanhOp` | `no_verifier` |
| `TransposeOp` | `real` |
| `TriSolveOp` | `real` |

### `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`

| Op | Status |
|----|--------|
| `CausalMaskOp` | `no_verifier` |
| `DropoutMaskOp` | `no_verifier` |
| `LseAccumulateOp` | `no_verifier` |
| `LseLoadOp` | `no_verifier` |
| `LseSaveOp` | `real` |
| `OnlineSoftmaxOp` | `real` |
| `ScaledDotProductOp` | `real` |

### `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`

| Op | Status |
|----|--------|
| `CreateOp` | `real` |
| `PopOp` | `real` |
| `PushOp` | `real` |
