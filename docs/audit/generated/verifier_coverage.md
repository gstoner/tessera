# MLIR Verifier Coverage Dashboard

Human-readable view. The canonical machine-readable artifact is `verifier_coverage.csv` in this directory — that CSV is what the drift gate compares. Don't edit either by hand; run `python -m tessera.compiler.audit verifier_coverage --write` (or `scripts/check_generated_docs.sh --write`) to refresh both. Drift is gated by `tests/unit/test_verifier_coverage.py`.

## Summary

| Status | Count | Meaning |
|--------|-------|---------|
| `real` | 123 | `hasVerifier = 1;` + substantive `verify()` body. |
| `trivial_stub` | 0 | `hasVerifier = 1;` + trivial `return success();` stub. |
| `absent` | 0 | `hasVerifier = 1;` but no `verify()` body (build error risk). |
| `no_verifier` | 49 | No verifier declared.  TD constraints suffice — fine for many ops. |
| **Total** | 172 | |

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
| `ArchGumbelSoftmaxOp` | `real` |
| `ArchHardConcreteOp` | `real` |
| `ArchMixedOp` | `real` |
| `ArchParameterOp` | `real` |
| `ArchSTEOneHotOp` | `no_verifier` |
| `ArchSwitchOp` | `real` |
| `ArchWeightedSumOp` | `real` |
| `AttnCompressedBlocksOp` | `real` |
| `AttnLocalWindow2DOp` | `real` |
| `AttnSlidingWindowOp` | `real` |
| `AttnTopKBlocksOp` | `real` |
| `BatchedGemmOp` | `real` |
| `BroadcastOp` | `real` |
| `CacheCommitOp` | `no_verifier` |
| `CachePageLookupOp` | `no_verifier` |
| `CacheRollbackOp` | `no_verifier` |
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
| `ControlForOp` | `real` |
| `ControlIfOp` | `real` |
| `ControlScanOp` | `real` |
| `ControlWhileOp` | `real` |
| `Conv2DNHWCOp` | `real` |
| `CustomAdjointCallOp` | `no_verifier` |
| `DCTOp` | `real` |
| `DeepSeekSparseAttentionOp` | `real` |
| `DequantGroupedGemmOp` | `real` |
| `DequantMatmulOp` | `real` |
| `DequantizeFP4Op` | `real` |
| `DequantizeFP8Op` | `real` |
| `DiffusionBlockStepOp` | `real` |
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
| `ExpandOp` | `real` |
| `FFTOp` | `real` |
| `FlashAttnOp` | `real` |
| `FlattenOp` | `real` |
| `FusedEpilogueOp` | `real` |
| `GQAAttentionOp` | `real` |
| `GatedAttentionOp` | `real` |
| `GatedDeltaNetOp` | `real` |
| `GeluOp` | `no_verifier` |
| `GroupedGemmOp` | `real` |
| `GuidedDenoiseRegionOp` | `real` |
| `HybridAttentionOp` | `real` |
| `IFFTOp` | `real` |
| `IRFFTOp` | `real` |
| `KVCacheAppendOp` | `real` |
| `KVCacheCreateOp` | `real` |
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
| `LogSoftmaxOp` | `real` |
| `LookaheadSparseAttentionOp` | `real` |
| `MLADecodeFusedOp` | `real` |
| `MLADecodeOp` | `real` |
| `MQAAttentionOp` | `real` |
| `MSAIndexScoresOp` | `real` |
| `MSASelectBlocksOp` | `real` |
| `MSASparseAttentionOp` | `real` |
| `MaskedFillOp` | `no_verifier` |
| `MatmulOp` | `real` |
| `ModifiedDeltaAttentionOp` | `real` |
| `MoeCombineOp` | `real` |
| `MoeDispatchOp` | `real` |
| `MoeSwigluBlockOp` | `real` |
| `MomentumOp` | `no_verifier` |
| `MorPartitionOp` | `real` |
| `MorRouterOp` | `real` |
| `MorScatterOp` | `real` |
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
| `PermuteOp` | `real` |
| `PowerAttnOp` | `real` |
| `QROp` | `real` |
| `QuantizeFP4Op` | `real` |
| `QuantizeFP8Op` | `real` |
| `RFFTOp` | `real` |
| `RLCISPOPolicyLossOp` | `real` |
| `RLGRPOPolicyLossOp` | `real` |
| `RLNormalizeGroupAdvantagesOp` | `real` |
| `RLPPOPolicyLossOp` | `real` |
| `RMSNormSafeOp` | `real` |
| `ReduceOp` | `no_verifier` |
| `ReduceScatterOp` | `no_verifier` |
| `ReluOp` | `no_verifier` |
| `ReshapeOp` | `real` |
| `RetentionOp` | `real` |
| `RingCreateOp` | `real` |
| `RmsNormOp` | `real` |
| `RopeMergeOp` | `no_verifier` |
| `RopeOp` | `real` |
| `RopeSplitOp` | `no_verifier` |
| `SVDOp` | `real` |
| `ScoreCombineOp` | `real` |
| `SelectOp` | `no_verifier` |
| `SelectiveSsmOp` | `real` |
| `SigmoidOp` | `no_verifier` |
| `SiluMulOp` | `no_verifier` |
| `SiluOp` | `no_verifier` |
| `SinOp` | `no_verifier` |
| `SoftmaxOp` | `real` |
| `SoftmaxSafeOp` | `real` |
| `SoftplusOp` | `no_verifier` |
| `SpecAcceptOp` | `real` |
| `SpecAcceptSampleOp` | `real` |
| `SpectralConvOp` | `no_verifier` |
| `SqueezeOp` | `real` |
| `SubOp` | `no_verifier` |
| `SwigluFusedOp` | `no_verifier` |
| `TanhOp` | `no_verifier` |
| `TransposeOp` | `real` |
| `TriSolveOp` | `real` |
| `UnsqueezeOp` | `real` |
| `VarlenSdpaOp` | `real` |
| `ViewOp` | `real` |
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
