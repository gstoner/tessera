
set(LLVM_LINK_COMPONENTS Support)

add_library(TesseraPasses
  # Phase 1
  CanonicalizeTesseraIR.cpp
  VerifyTesseraIR.cpp
  MigrateTesseraIR.cpp
  Passes.cpp

  # Phase 2 lowering chain
  DistributionLoweringPass.cpp
  EffectAnnotationPass.cpp
  TilingPass.cpp
  TileToX86Pass.cpp

  # Phase 0 production spine — Graph IR → upstream linalg
  # (docs/spec/PRODUCTION_COMPILER_PLAN.md).
  TesseraToLinalgPass.cpp

  # CF0 — control-flow target guard (CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN)
  ControlFlowTargetGuardPass.cpp
  # CF2 — control-flow → scf lowering (control_for → scf.for)
  LowerControlFlowToSCFPass.cpp
  # CF4a — decode the control_for op-list payload into a real @body func
  MaterializeControlPayloadPass.cpp

  # Phase 3 GPU / FA-4 Tile IR lowering
  TileIRLoweringPass.cpp

  # Phase 4 distributed training
  GPUCollectiveInsertionPass.cpp
  PipelineStageInsertionPass.cpp

  # Phase F4 autodiff (reverse-mode via AdjointInterface).
  # ODS scaffold lives at src/compiler/ir/include/Tessera/AdjointInterface.td.
  AutodiffPass.cpp
  AutodiffPairedPass.cpp

  # Phase F5 — adjoint collective insertion (runs after AutodiffPass).
  AdjointCollectiveInsertionPass.cpp

  # Phase F2 (IR form) — activation rematerialization (runs after AutodiffPass).
  ActivationRematerializationPass.cpp

  # Phase 8.4.8 — SwiGLU fusion (Stage 2b of SwiGLU Performance Plan).
  SwigluFusionPass.cpp

  # attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion.
  MLAFusionPass.cpp

  # attention_variants_plan, NSA-4 — DeepSeek NSA fusion.
  NativeSparseAttnFusionPass.cpp

  # attention-family plan — Lightning / Delta / hybrid pass slots.
  AttentionFamilyPasses.cpp

  # Stage 13 — RL policy-loss compiler visibility / decomposition markers.
  RLLossDecomposePass.cpp
  VarlenSdpaDecomposePass.cpp

  # Sprint V2 (2026-05-22) — layout legality skeleton (SHAPE_SYSTEM §11.2 closure).
  LayoutLegalityPass.cpp
  LayoutAssignmentPass.cpp
  NVIDIAGraphLayoutMaterializationPass.cpp
  # 2026-07-08 — Tile IR global buffer assignment/reuse (Workstream H / W3).
  TileBufferReusePass.cpp
  # 2026-07-08 — realize the reuse plan into a concrete SMEM/TMEM arena (consumer).
  TileBufferArenaPass.cpp
  # 2026-06-19 — dtype / aliasing / buffer-binding contracts (Decision #15a).
  IRContractLegalityPass.cpp
  # 2026-06-23 — C2: barriers as a layout-reuse correctness property (TIRx review).
  TileBarrierReuseLegalityPass.cpp
  # 2026-06-23 — C3: typed barriers + PipelineState cross-op legality (TIRx review).
  TilePipelineLegalityPass.cpp
  # 2026-06-23 — C4: compute/storage dtype legalize split (Decision #15a, TIRx review).
  DtypeLegalizePass.cpp
  # 2026-06-23 — C6: warp-spec structural diagnostics (TIRx debugging appendix).
  WarpSpecLegalityPass.cpp
  # 2026-06-23 — pipeline-parallel: real stage partitioning + 1F1B schedule proof.
  PipelineStagePartitionPass.cpp
  PipelineScheduleLegalityPass.cpp

  # Sprint V5 (2026-05-22) — symbolic dim equality verifier (SHAPE_SYSTEM §11.2 closure).
  SymbolicDimEqualityPass.cpp

  # Phase 6 diagnostics
  ${CMAKE_SOURCE_DIR}/src/compiler/diagnostics/ShapeInferencePass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/diagnostics/ErrorReporter.cpp

  ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/lib/WarpSpecializationPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/lib/AsyncCopyLoweringPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVWGMMALoweringPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVTMADescriptorPass.cpp
  ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/NVFlashAttnKernelEmitter.cpp
)

target_include_directories(TesseraPasses
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/ir/include
    ${CMAKE_CURRENT_BINARY_DIR}/../../compiler/ir
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/tile_opt_fa4/include
)

# Phase F4 — pass body needs the AdjointInterface generated header.
add_dependencies(TesseraPasses TesseraAdjointInterfaceTableGen)

target_link_libraries(TesseraPasses
  PUBLIC
    TesseraIR
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    MLIRTransforms
    # Phase 2 dialect dependencies
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    # Phase 3 dialect dependencies
    TesseraAttnDialect
    TesseraQueueDialect
    MLIRNVVMDialect
    MLIRGPUDialect
)

mlir_check_all_link_libraries(TesseraPasses)
