# MLIR Verifier Coverage Dashboard

Human-readable view. The canonical machine-readable artifact is `verifier_coverage.csv` in this directory — that CSV is what the drift gate compares. Don't edit either by hand; run `python -m tessera.compiler.audit verifier_coverage --write` (or `scripts/check_generated_docs.sh --write`) to refresh both. Drift is gated by `tests/unit/test_verifier_coverage.py`.

## Summary

| Status | Count | Meaning |
|--------|-------|---------|
| `real` | 174 | `hasVerifier = 1;` + substantive `verify()` body. |
| `trivial_stub` | 0 | `hasVerifier = 1;` + trivial `return success();` stub. |
| `absent` | 0 | `hasVerifier = 1;` but no `verify()` body (build error risk). |
| `no_verifier` | 0 | No verifier declared.  TD constraints suffice — fine for many ops. |
| **Total** | 174 | |

## Per-dialect details

### `src/compiler/ir/TesseraOps.td`

| Op | Status |
|----|--------|
| `ALiBiOp` | `real` |
| `AdafactorOp` | `real` |
| `AdamOp` | `real` |
| `AdamWOp` | `real` |
| `AddOp` | `real` |
| `AllGatherOp` | `real` |
| `AllReduceOp` | `real` |
| `ArchGumbelSoftmaxOp` | `real` |
| `ArchHardConcreteOp` | `real` |
| `ArchMixedOp` | `real` |
| `ArchParameterOp` | `real` |
| `ArchSTEOneHotOp` | `real` |
| `ArchSwitchOp` | `real` |
| `ArchWeightedSumOp` | `real` |
| `AttnCompressedBlocksOp` | `real` |
| `AttnLocalWindow2DOp` | `real` |
| `AttnSlidingWindowOp` | `real` |
| `AttnTopKBlocksOp` | `real` |
| `BatchedGemmOp` | `real` |
| `BroadcastOp` | `real` |
| `CacheCommitOp` | `real` |
| `CachePageLookupOp` | `real` |
| `CacheRollbackOp` | `real` |
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
| `CustomAdjointCallOp` | `real` |
| `DCTOp` | `real` |
| `DeepSeekSparseAttentionOp` | `real` |
| `DequantGroupedGemmOp` | `real` |
| `DequantMatmulOp` | `real` |
| `DequantizeFP4Op` | `real` |
| `DequantizeFP8Op` | `real` |
| `DiffusionBlockStepOp` | `real` |
| `DivOp` | `real` |
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
| `GeluOp` | `real` |
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
| `LionOp` | `real` |
| `LogSoftmaxOp` | `real` |
| `LookaheadSparseAttentionOp` | `real` |
| `MLADecodeFusedOp` | `real` |
| `MLADecodeOp` | `real` |
| `MQAAttentionOp` | `real` |
| `MSAIndexScoresOp` | `real` |
| `MSASelectBlocksOp` | `real` |
| `MSASparseAttentionOp` | `real` |
| `MaskedFillOp` | `real` |
| `MatmulOp` | `real` |
| `ModifiedDeltaAttentionOp` | `real` |
| `MoeCombineOp` | `real` |
| `MoeDispatchOp` | `real` |
| `MoeSwigluBlockOp` | `real` |
| `MomentumOp` | `real` |
| `MorPartitionOp` | `real` |
| `MorRouterOp` | `real` |
| `MorScatterOp` | `real` |
| `MulOp` | `real` |
| `MultiHeadAttentionOp` | `real` |
| `NTKRopeOp` | `real` |
| `NativeSparseAttnFusedOp` | `real` |
| `NeighborsHaloExchangeOp` | `real` |
| `NeighborsHaloPackOp` | `real` |
| `NeighborsHaloRegionOp` | `real` |
| `NeighborsHaloTransportOp` | `real` |
| `NeighborsHaloUnpackOp` | `real` |
| `NeighborsNeighborReadOp` | `real` |
| `NeighborsPipelineConfigOp` | `real` |
| `NeighborsStencilApplyOp` | `real` |
| `NeighborsStencilDefineOp` | `real` |
| `NeighborsTopologyCreateOp` | `real` |
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
| `ReduceOp` | `real` |
| `ReduceScatterOp` | `real` |
| `ReluOp` | `real` |
| `ReshapeOp` | `real` |
| `RetentionOp` | `real` |
| `RingCreateOp` | `real` |
| `RmsNormOp` | `real` |
| `RopeMergeOp` | `real` |
| `RopeOp` | `real` |
| `RopeSplitOp` | `real` |
| `SVDOp` | `real` |
| `ScoreCombineOp` | `real` |
| `SelectOp` | `real` |
| `SelectiveSsmOp` | `real` |
| `SigmoidOp` | `real` |
| `SiluMulOp` | `real` |
| `SiluOp` | `real` |
| `SinOp` | `real` |
| `SoftmaxOp` | `real` |
| `SoftmaxSafeOp` | `real` |
| `SoftplusOp` | `real` |
| `SpecAcceptOp` | `real` |
| `SpecAcceptSampleOp` | `real` |
| `SpecAcceptTreeSampleOp` | `real` |
| `SpectralConvOp` | `real` |
| `SqueezeOp` | `real` |
| `SubOp` | `real` |
| `SwigluFusedOp` | `real` |
| `TanhOp` | `real` |
| `TargetVerifyOp` | `real` |
| `TransposeOp` | `real` |
| `TriSolveOp` | `real` |
| `UnsqueezeOp` | `real` |
| `VarlenSdpaOp` | `real` |
| `ViewOp` | `real` |
| `WriteRowOp` | `real` |

### `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td`

| Op | Status |
|----|--------|
| `CausalMaskOp` | `real` |
| `DropoutMaskOp` | `real` |
| `LseAccumulateOp` | `real` |
| `LseLoadOp` | `real` |
| `LseSaveOp` | `real` |
| `OnlineSoftmaxOp` | `real` |
| `ScaledDotProductOp` | `real` |

### `src/compiler/tile_opt_fa4/include/tessera/Dialect/Queue/Queue.td`

| Op | Status |
|----|--------|
| `CreateOp` | `real` |
| `PopOp` | `real` |
| `PushOp` | `real` |
