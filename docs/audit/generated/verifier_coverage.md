# MLIR Verifier Coverage Dashboard

Human-readable view. The canonical machine-readable artifact is `verifier_coverage.csv` in this directory — that CSV is what the drift gate compares. Don't edit either by hand; run `python -m tessera.compiler.audit verifier_coverage --write` (or `scripts/check_generated_docs.sh --write`) to refresh both. Drift is gated by `tests/unit/test_verifier_coverage.py`.

## Summary

| Status | Count | Meaning |
|--------|-------|---------|
| `real` | 73 | `hasVerifier = 1;` + substantive `verify()` body. |
| `trivial_stub` | 9 | `hasVerifier = 1;` + trivial `return success();` stub. |
| `absent` | 0 | `hasVerifier = 1;` but no `verify()` body (build error risk). |
| `no_verifier` | 63 | No verifier declared.  TD constraints suffice — fine for many ops. |
| **Total** | 145 | |

## Per-dialect details

### `src/compiler/ir/TesseraOps.td`

| Op | Status |
|----|--------|
| `ALiBiOp` | `no_verifier` |
| `AdafactorOp` | `no_verifier` |
| `AdamOp` | `no_verifier` |
| `AdamWOp` | `no_verifier` |
| `AddOp` | `no_verifier` |
| `AllGatherOp` | `no_verifier` |
| `AllReduceOp` | `no_verifier` |
| `ArchGumbelSoftmaxOp` | `trivial_stub` |
| `ArchHardConcreteOp` | `trivial_stub` |
| `ArchMixedOp` | `trivial_stub` |
| `ArchParameterOp` | `trivial_stub` |
| `ArchSTEOneHotOp` | `trivial_stub` |
| `ArchSwitchOp` | `trivial_stub` |
| `ArchWeightedSumOp` | `trivial_stub` |
| `AttnCompressedBlocksOp` | `real` |
| `AttnLocalWindow2DOp` | `real` |
| `AttnSlidingWindowOp` | `real` |
| `AttnTopKBlocksOp` | `real` |
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
| `ControlForOp` | `no_verifier` |
| `Conv2DNHWCOp` | `real` |
| `CustomAdjointCallOp` | `no_verifier` |
| `DCTOp` | `no_verifier` |
| `DeepSeekSparseAttentionOp` | `real` |
| `DequantizeFP4Op` | `no_verifier` |
| `DequantizeFP8Op` | `no_verifier` |
| `DivOp` | `no_verifier` |
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
| `GQAAttentionOp` | `real` |
| `GatedAttentionOp` | `real` |
| `GatedDeltaNetOp` | `real` |
| `GeluOp` | `no_verifier` |
| `HybridAttentionOp` | `real` |
| `IFFTOp` | `no_verifier` |
| `IRFFTOp` | `no_verifier` |
| `KVCacheAppendOp` | `real` |
| `KVCacheCreateOp` | `trivial_stub` |
| `KVCachePruneOp` | `real` |
| `KimiDeltaAttentionOp` | `real` |
| `LUOp` | `real` |
| `LatentKVCompressOp` | `real` |
| `LatentKVExpandKOp` | `real` |
| `LatentKVExpandVOp` | `real` |
| `LayerNormOp` | `real` |
| `LightningAttentionOp` | `real` |
| `LinearAttnOp` | `real` |
| `LinearAttnStateOp` | `real` |
| `LionOp` | `no_verifier` |
| `LogSoftmaxOp` | `no_verifier` |
| `MLADecodeFusedOp` | `real` |
| `MLADecodeOp` | `real` |
| `MQAAttentionOp` | `real` |
| `MaskedFillOp` | `no_verifier` |
| `MatmulOp` | `real` |
| `ModifiedDeltaAttentionOp` | `real` |
| `MoeCombineOp` | `real` |
| `MoeDispatchOp` | `real` |
| `MomentumOp` | `no_verifier` |
| `MorPartitionOp` | `no_verifier` |
| `MorRouterOp` | `no_verifier` |
| `MorScatterOp` | `no_verifier` |
| `MulOp` | `no_verifier` |
| `MultiHeadAttentionOp` | `real` |
| `NTKRopeOp` | `no_verifier` |
| `NativeSparseAttnFusedOp` | `real` |
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
| `PowerAttnOp` | `real` |
| `QROp` | `real` |
| `QuantizeFP4Op` | `no_verifier` |
| `QuantizeFP8Op` | `no_verifier` |
| `RFFTOp` | `no_verifier` |
| `RLCISPOPolicyLossOp` | `real` |
| `RLGRPOPolicyLossOp` | `real` |
| `RLNormalizeGroupAdvantagesOp` | `real` |
| `RLPPOPolicyLossOp` | `real` |
| `RMSNormSafeOp` | `no_verifier` |
| `ReduceOp` | `no_verifier` |
| `ReduceScatterOp` | `no_verifier` |
| `ReluOp` | `no_verifier` |
| `ReshapeOp` | `real` |
| `RetentionOp` | `real` |
| `RingCreateOp` | `trivial_stub` |
| `RmsNormOp` | `no_verifier` |
| `RopeMergeOp` | `no_verifier` |
| `RopeOp` | `real` |
| `RopeSplitOp` | `no_verifier` |
| `SVDOp` | `real` |
| `SelectOp` | `no_verifier` |
| `SigmoidOp` | `no_verifier` |
| `SiluMulOp` | `no_verifier` |
| `SiluOp` | `no_verifier` |
| `SinOp` | `no_verifier` |
| `SoftmaxOp` | `real` |
| `SoftmaxSafeOp` | `no_verifier` |
| `SoftplusOp` | `no_verifier` |
| `SpectralConvOp` | `no_verifier` |
| `SubOp` | `no_verifier` |
| `SwigluFusedOp` | `no_verifier` |
| `TanhOp` | `no_verifier` |
| `TransposeOp` | `real` |
| `TriSolveOp` | `real` |
| `WriteRowOp` | `no_verifier` |

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
