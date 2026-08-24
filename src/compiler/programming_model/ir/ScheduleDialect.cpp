//===- ScheduleDialect.cpp — Schedule IR registration and verification ---===//

#include "tessera/ProgrammingModel/ScheduleDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace tessera::schedule;

#include "ScheduleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ScheduleMeshPipelineOps.cpp.inc"

void ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ScheduleMeshPipelineOps.cpp.inc"
      >();
}

LogicalResult MeshDefineOp::verify() {
  if (getDims().empty())
    return emitOpError("requires at least one mesh dimension");
  if (getDims().size() != getAxisNames().size())
    return emitOpError("requires one axis name per mesh dimension");
  for (Attribute attr : getDims()) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getInt() <= 0)
      return emitOpError("mesh dimensions must be positive integers");
  }
  for (Attribute attr : getAxisNames())
    if (!isa<StringAttr>(attr))
      return emitOpError("mesh axis names must be strings");
  return success();
}

LogicalResult MeshRegionOp::verify() {
  if (getAxis().empty())
    return emitOpError("requires a non-empty mesh axis");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one body block");
  Operation *terminator = getBody().front().getTerminator();
  if (!terminator || terminator->getName().getStringRef() != "schedule.yield")
    return emitOpError("body must terminate with schedule.yield");
  // Legacy marker-only regions may yield an informational value while exposing
  // no SSA result. Preserve those until their owning passes migrate; once the
  // region declares results, enforce the real value contract exactly.
  if (getNumResults() == 0)
    return success();
  if (terminator->getNumOperands() != getNumResults())
    return emitOpError("yield operand count must match region result count");
  for (auto [yielded, result] :
       llvm::zip_equal(terminator->getOperands(), getResults()))
    if (yielded.getType() != result.getType())
      return emitOpError("yield operand types must match region result types");
  return success();
}

LogicalResult PipelineRegionOp::verify() {
  if (getSchedule().empty())
    return emitOpError("requires a non-empty pipeline schedule");
  if (getMicroBatches() < 1)
    return emitOpError("requires micro_batches >= 1");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  return success();
}

LogicalResult StageOp::verify() {
  if (getDevices().empty())
    return emitOpError("requires at least one device");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  return success();
}

LogicalResult TileOp::verify() {
  if (getSource().empty() || getResult().empty())
    return emitOpError("requires non-empty source and result identities");
  if (getOrdinalAttr().getInt() < 0)
    return emitOpError("requires ordinal >= 0");
  for (StringRef name : {"tile_m", "tile_n", "tile_k", "tile_h", "tile_w",
                         "tile_c"}) {
    if (auto value = (*this)->getAttrOfType<IntegerAttr>(name);
        value && value.getInt() <= 0)
      return emitOpError("optional tile dimensions must be positive");
  }
  return success();
}

LogicalResult MatmulOp::verify() {
  if (getSubject().getType() != getScheduled().getType())
    return emitOpError("must preserve the scheduled Graph value type");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getArch().empty())
    return emitOpError("requires a non-empty architecture");
  if (getTileM() <= 0 || getTileN() <= 0 || getTileK() <= 0)
    return emitOpError("tile dimensions must be positive");
  if (getMacroTileM() < getTileM() || getMacroTileN() < getTileN() ||
      getMacroTileM() % getTileM() != 0 ||
      getMacroTileN() % getTileN() != 0)
    return emitOpError(
        "macro tile dimensions must be positive multiples of tile_m/tile_n");
  if (getWarps() != 1 && getWarps() != 4)
    return emitOpError("warps must be 1 or 4");
  if (getPipelineDepth() <= 0)
    return emitOpError("pipeline_depth must be positive");
  if (getStorage().empty() || getAccum().empty())
    return emitOpError("requires explicit storage and accumulation types");
  if (!llvm::is_contained({"none", "relu", "gelu", "silu"},
                          getActivation()))
    return emitOpError("requires a supported pointwise activation");
  if (getOutput() != "f32" && getOutput() != "f16")
    return emitOpError("requires f32 or f16 output storage");
  if (getALayout() != "row_major" || getBLayout() != "col_major")
    return emitOpError("initial matmul contract requires row/col layouts");
  if (getRasterOrder() != "row_major")
    return emitOpError("initial matmul contract requires row-major raster order");
  if (getRasterGroup() <= 0)
    return emitOpError("raster_group must be positive");
  return success();
}

static LogicalResult verifyContentAddressedKernel(Operation *op, Value subject,
                                                  Value scheduled,
                                                  StringRef artifactHash,
                                                  StringRef arch,
                                                  StringRef storage,
                                                  StringRef accum,
                                                  int64_t workgroupSize) {
  if (subject.getType() != scheduled.getType())
    return op->emitOpError("must preserve the scheduled Graph value type");
  if (artifactHash.size() != 64 ||
      !llvm::all_of(artifactHash, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return op->emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (arch.empty() || storage.empty() || accum.empty())
    return op->emitOpError("requires explicit architecture and numeric types");
  if (workgroupSize <= 0)
    return op->emitOpError("requires workgroup_size > 0");
  return success();
}

LogicalResult SoftmaxOp::verify() {
  if (failed(verifyContentAddressedKernel(
          *this, getSubject(), getScheduled(), getArtifactHash(), getArch(),
          getStorage(), getAccum(), getWorkgroupSize())))
    return failure();
  if (getAxisAttr().getInt() != -1)
    return emitOpError("initial softmax contract requires the last axis");
  if (getExpMode() != "accurate")
    return emitOpError("initial softmax contract requires accurate exp");
  return success();
}

LogicalResult ReduceOp::verify() {
  if (failed(verifyContentAddressedKernel(
          *this, getSubject(), getScheduled(), getArtifactHash(), getArch(),
          getStorage(), getAccum(), getWorkgroupSize())))
    return failure();
  if (getAxisAttr().getInt() < 0)
    return emitOpError("requires a normalized non-negative axis");
  if (getKind() != "sum" && getKind() != "mean" && getKind() != "max")
    return emitOpError("kind must be sum, mean, or max");
  if (getSchedule() != "serial" || getNanMode() != "propagate")
    return emitOpError("initial reduction contract requires serial/propagate policy");
  return success();
}

LogicalResult FFTOp::verify() {
  if (failed(verifyContentAddressedKernel(
          *this, getSubject(), getScheduled(), getArtifactHash(), getArch(),
          getStorage(), getAccum(), getWorkgroupSize())))
    return failure();
  if (getMode() != "c2c" && getMode() != "r2c" && getMode() != "c2r")
    return emitOpError("mode must be c2c, r2c, or c2r");
  auto physicalLength =
      getOperation()->getAttrOfType<IntegerAttr>("physical_length");
  auto realPolicy =
      getOperation()->getAttrOfType<StringAttr>("real_transform_policy");
  auto hermitianLayout =
      getOperation()->getAttrOfType<StringAttr>("hermitian_layout");
  if (getAxisAttr().getInt() < 0 || getLengthAttr().getInt() <= 0 ||
      getBatchAttr().getInt() <= 0 || !physicalLength ||
      physicalLength.getInt() <= 0 || !realPolicy || !hermitianLayout)
    return emitOpError("requires normalized axis and positive length/batch");
  if (getNormalization() != "backward" && getNormalization() != "forward" &&
      getNormalization() != "ortho")
    return emitOpError("normalization must be backward, forward, or ortho");
  auto hermitianWeight =
      getOperation()->getAttrOfType<StringAttr>("hermitian_weight");
  if (!hermitianWeight ||
      (hermitianWeight.getValue() != "none" &&
       hermitianWeight.getValue() != "half_interior" &&
       hermitianWeight.getValue() != "double_interior"))
    return emitOpError("requires an explicit Hermitian weighting policy");
  if (!getScale().isFinite() || getScale().convertToDouble() <= 0.0)
    return emitOpError("requires a finite positive normalization scale");
  if (getStorage() != "complex64_interleaved_f32" || getAccum() != "f32")
    return emitOpError("requires interleaved complex64 storage and f32 accumulation");
  if (getRadixPolicy() != "radix2" && getRadixPolicy() != "mixed_radix")
    return emitOpError("radix_policy must be radix2 or mixed_radix");
  if (getStrategy() != "radix2" && getStrategy() != "mixed_radix" &&
      getStrategy() != "dft" && getStrategy() != "bluestein")
    return emitOpError("requires a known FFT strategy");
  if ((getStrategy() == "radix2" || getStrategy() == "mixed_radix") &&
      physicalLength.getInt() > 1 && getRadixSequence().empty())
    return emitOpError("staged FFT strategy requires a radix sequence");
  int64_t product = 1;
  for (int64_t radix : getRadixSequence()) {
    if (radix < 2 || radix > 17)
      return emitOpError("radix sequence entries must be in [2, 17]");
    product *= radix;
  }
  if (!getRadixSequence().empty() &&
      product != physicalLength.getInt())
    return emitOpError("radix sequence product must equal physical transform length");
  bool bluestein = getStrategy() == "bluestein";
  const int64_t logicalLength = static_cast<int64_t>(getLength());
  if (bluestein != (getBluesteinM() > 0) ||
      (bluestein && static_cast<int64_t>(getBluesteinM()) <
                        2 * physicalLength.getInt() - 1))
    return emitOpError("Bluestein strategy requires a sufficient padded length");
  const bool realMode = getMode() == "r2c" || getMode() == "c2r";
  if (!realMode && hermitianWeight.getValue() != "none")
    return emitOpError("full-complex FFT cannot apply Hermitian weighting");
  if (getMode() == "r2c" && hermitianWeight.getValue() == "half_interior")
    return emitOpError("r2c cannot apply input-side half-interior weighting");
  if (getMode() == "c2r" && hermitianWeight.getValue() == "double_interior")
    return emitOpError("c2r cannot apply output-side double-interior weighting");
  if (realPolicy.getValue() == "packed_even_n2_hermitian_v1" &&
      (!realMode || logicalLength % 2 != 0 ||
       physicalLength.getInt() != logicalLength / 2))
    return emitOpError("packed-real policy requires an even N/2 physical transform");
  if ((realMode && hermitianLayout.getValue() !=
                       "half_spectrum_nyquist_explicit") ||
      (!realMode && hermitianLayout.getValue() != "full_complex"))
    return emitOpError("Hermitian layout does not match FFT mode");
  if ((getAlgorithm() != "stockham_autosort" &&
       getAlgorithm() != "cooley_tukey_dit") ||
      getWorkspacePolicy().empty() || getResidency().empty() ||
      getTwiddlePolicy().empty() || getWorkspaceElemsAttr().getInt() < 0 ||
      getTwiddleLayout() != "interleaved_f32" ||
      !getDeterministic() || getKernelFamily().empty())
    return emitOpError("requires explicit deterministic workspace/twiddle/kernel policy");
  return success();
}

LogicalResult SpectralProgramOp::verify() {
  if (getSubjects().empty() || getSubjects().size() > 2)
    return emitOpError("requires one or two input subjects");
  if (getSubjects().front().getType() != getScheduled().getType() &&
      getKind() == "tessera.spectral_filter")
    return emitOpError("spectral_filter must preserve its complex tensor type");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getTarget() != "x86" && getTarget() != "rocm")
    return emitOpError("target must be x86 or rocm");
  if ((getTarget() == "x86" && getArch() != "zen5-avx512") ||
      (getTarget() == "rocm" && getArch() != "gfx1151"))
    return emitOpError("architecture must match the exact target profile");
  if (getKind() != "tessera.spectral_filter" && getKind() != "tessera.dct" &&
      getKind() != "tessera.spectral_conv" && getKind() != "tessera.stft" &&
      getKind() != "tessera.istft")
    return emitOpError("requires a known compound spectral kind");
  unsigned expectedInputs =
      (getKind() == "tessera.dct") ? 1u : 2u;
  if (getSubjects().size() != expectedInputs)
    return emitOpError("input count does not match the spectral kind");
  if (getOutputShape().empty() ||
      llvm::any_of(getOutputShape(), [](int64_t dim) { return dim <= 0; }) ||
      getAxisAttr().getInt() < 0 || getPadding().size() != 2 ||
      getCrop().size() != 2 || getWindowLengthAttr().getInt() < 0 ||
      getHopAttr().getInt() < 0 || getFramesAttr().getInt() < 0 ||
      getWorkspaceBytesAttr().getInt() <= 0 || getWorkgroupSize() <= 0)
    return emitOpError("requires complete positive shape, launch, and workspace policy");
  if ((getNormalization() != "backward" && getNormalization() != "forward" &&
       getNormalization() != "ortho") ||
      getComplexLayout() != "interleaved_f32x2" ||
      (getShapePolicy() != "exact_runtime_specialization_v1" &&
       getShapePolicy() != "bounded_runtime_specialization_v1") ||
      (getStorage() != "f32" && getStorage() != "f16" &&
       getStorage() != "bf16") ||
      getAbiStorage() != "f32" ||
      (getStorageConversion() != "native_f32" &&
       getStorageConversion() !=
           "native_package_cast_f32_accumulate_cast_output_v1") ||
      (getAxisPacking() != "none_contiguous" &&
       getAxisPacking() != "native_package_host_pack_v1") ||
      getWorkspacePolicy() != "persistent_artifact_workspace" ||
      getFusionTopology().empty() ||
      getMutationLineage() != "inputs_immutable_output_fresh_v1" ||
      getNativeEntry().empty() || getInputShapes().empty() ||
      getInputSignature().empty() || getShapeBounds().empty() ||
      getTemplateDigest().size() != 64)
    return emitOpError("requires the canonical spectral numeric/workspace/lineage policy");
  auto dctType = (*this)->getAttrOfType<IntegerAttr>("dct_type");
  const bool directDct = getKind() == "tessera.dct" && dctType &&
                         dctType.getInt() != 2;
  if (getKind() != "tessera.spectral_filter" && !directDct &&
      getChildFftDigests().empty())
    return emitOpError("FFT-based spectral programs require child digests");
  return success();
}

LogicalResult SpectralBackwardOp::verify() {
  if (getSubjects().empty() || getGradients().size() != getSubjects().size())
    return emitOpError(
        "requires one retained Graph result per scheduled gradient");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if ((getTarget() == "x86" && getArch() != "zen5-avx512") ||
      (getTarget() == "rocm" && getArch() != "gfx1151") ||
      (getTarget() != "x86" && getTarget() != "rocm"))
    return emitOpError("requires an exact x86/Zen 5 or ROCm/gfx1151 profile");
  if (getKind() != "tessera.stft" && getKind() != "tessera.istft" &&
      getKind() != "tessera.spectral_filter" &&
      getKind() != "tessera.spectral_conv")
    return emitOpError("requires a registered compound spectral kind");
  if (getNormalization() != "backward" && getNormalization() != "forward" &&
      getNormalization() != "ortho")
    return emitOpError("requires a supported normalization");
  if (getPadMode() != "constant" && getPadMode() != "reflect")
    return emitOpError("requires pad_mode in {constant, reflect}");
  if (auto length = (*this)->getAttrOfType<IntegerAttr>("output_length");
      length && length.getInt() < 0)
    return emitOpError("requires output_length >= 0");
  if (getInputSignature().empty() || getOutputSignature().empty() ||
      getMutationLineage() != "inputs_immutable_outputs_fresh_v1")
    return emitOpError("requires output identity and immutable-input lineage");
  for (auto [subject, gradient] : llvm::zip(getSubjects(), getGradients()))
    if (subject.getType() != gradient.getType())
      return emitOpError("gradient types must match retained forward operands");
  return success();
}

LogicalResult AttentionOp::verify() {
  if (failed(verifyContentAddressedKernel(
          *this, getSubject(), getScheduled(), getArtifactHash(), getArch(),
          getStorage(), getAccum(), getWorkgroupSize())))
    return failure();
  if (!getScale().isFinite() || getScale().convertToDouble() <= 0.0)
    return emitOpError("requires a finite positive scale");
  if (static_cast<int64_t>(getWindowLeft()) < -1 ||
      static_cast<int64_t>(getWindowRight()) < -1)
    return emitOpError("requires window_left/window_right >= -1");
  if (!getSoftcap().isFinite() || getSoftcap().convertToDouble() < 0.0)
    return emitOpError("requires finite softcap >= 0");
  if (!getDropoutP().isFinite() || getDropoutP().convertToDouble() < 0.0 ||
      getDropoutP().convertToDouble() >= 1.0)
    return emitOpError("requires finite dropout_p in [0, 1)");
  if (getTileQ() <= 0 || getTileKv() <= 0)
    return emitOpError("requires positive tile_q/tile_kv");
  if (getRecurrence() != "rank4_batch_query_head_kv_online_softmax_v1")
    return emitOpError("requires the canonical rank-4 streaming recurrence");
  // Architecture-owned backward LSE policies.  This is a closed allowlist by
  // design: a backend must declare its saved/recompute identity here rather
  // than inheriting a sibling's.  `apple7_recompute` records that Apple's
  // backward recomputes m/l per query row and its ABI takes no LSE buffer
  // (APPLE-ATTN-STREAM-1).
  if (getBackwardLsePolicy() != "save_lse" &&
      getBackwardLsePolicy() != "gfx1151_auto_128" &&
      getBackwardLsePolicy() != "apple7_recompute")
    return emitOpError("requires an architecture-owned backward LSE policy");
  if (getBackwardLseSelection() != "saved" &&
      getBackwardLseSelection() != "recompute")
    return emitOpError("requires a concrete backward LSE selection");
  return success();
}

LogicalResult DepthAttentionOp::verify() {
  if (failed(verifyContentAddressedKernel(
          *this, getSubject(), getScheduled(), getArtifactHash(), getArch(),
          getStorage(), getAccum(), getWorkgroupSize())))
    return failure();
  if (getStorage() != "f32" || getSoftmax() != "f32" ||
      getAccum() != "f32")
    return emitOpError("initial physical boundary requires f32 storage, softmax, and accumulation");
  if (!getEps().isFinite() || getEps().convertToDouble() <= 0.0)
    return emitOpError("requires finite eps > 0");
  if (getSourceCount() <= 0 || getRows() <= 0 || getWidth() <= 0 ||
      getSourceTile() <= 0)
    return emitOpError("requires positive source_count, rows, width, and source_tile");
  if (getStatisticsRecurrence() != "rms_key_online_softmax_stats_v1" ||
      getMergeRecurrence() != "max_shifted_pairwise_merge_v1")
    return emitOpError("requires canonical depth-attention statistics and merge recurrences");
  if (!getReassociable())
    return emitOpError("requires the proven reassociable merge contract");
  return success();
}

LogicalResult AttentionBackwardOp::verify() {
  if (getDqSubject().getType() != getDqScheduled().getType() ||
      getDkSubject().getType() != getDkScheduled().getType() ||
      getDvSubject().getType() != getDvScheduled().getType())
    return emitOpError("must preserve all three gradient result types");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getArch().empty() || getStorage().empty() || getAccum() != "f32")
    return emitOpError("requires architecture, storage, and f32 accumulation");
  if (!getScale().isFinite() || getScale().convertToDouble() <= 0.0)
    return emitOpError("requires a finite positive scale");
  if (static_cast<int64_t>(getWindowLeft()) < -1 ||
      static_cast<int64_t>(getWindowRight()) < -1)
    return emitOpError("requires window_left/window_right >= -1");
  if (!getSoftcap().isFinite() || getSoftcap().convertToDouble() < 0.0 ||
      !getDropoutP().isFinite() || getDropoutP().convertToDouble() < 0.0 ||
      getDropoutP().convertToDouble() >= 1.0)
    return emitOpError("requires finite softcap and dropout policy");
  if (getQueryBlock() <= 0 || getKeyBlock() <= 0 || getSplitCount() < 2 ||
      getWorkspaceBytes() <= 0 || getWorkgroupSize() <= 0)
    return emitOpError("requires positive block/workspace policy and at least two splits");
  if (getReductionOrder().size() != getSplitCount())
    return emitOpError("requires one reduction-order entry per split");
  for (auto [index, value] : llvm::enumerate(getReductionOrder()))
    if (value != static_cast<int64_t>(index))
      return emitOpError("requires ascending fixed-order split reduction");
  if (getRecurrence() != "tensor_dq_split_dkdv_fixed_reduce_v1")
    return emitOpError("requires the canonical tensor-valued VJP recurrence");
  // Same closed allowlist as the forward op: a backend declares its own
  // saved/recompute identity rather than inheriting a sibling's.  Apple's
  // backward recomputes m/l per query row and its ABI takes no LSE buffer.
  if (getLseCheckpointPolicy() != "save_lse" &&
      getLseCheckpointPolicy() != "gfx1151_auto_128" &&
      getLseCheckpointPolicy() != "apple7_recompute")
    return emitOpError("requires an architecture-owned LSE policy");
  if (getLseCheckpointSelection() != "saved" &&
      getLseCheckpointSelection() != "recompute")
    return emitOpError("requires a concrete LSE checkpoint selection");
  return success();
}

LogicalResult LionVJPOp::verify() {
  Type expected = getParameter().getType();
  for (Value input : {getGradient(), getMoment(), getDparameter(), getDmoment()})
    if (input.getType() != expected)
      return emitOpError("requires all five inputs to have one tensor type");
  for (Value result : {getDparameterScheduled(), getDgradientScheduled(),
                       getDmomentScheduled()})
    if (result.getType() != expected)
      return emitOpError("requires all three VJP results to preserve the input type");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getLineagePayload().empty() || getArch().empty())
    return emitOpError("requires lineage payload and architecture identity");
  if (!getLearningRate().isFinite() || !getBeta2().isFinite() ||
      !getWeightDecay().isFinite())
    return emitOpError("requires finite Lion coefficients");
  if (getDerivativePolicy() != "stop_gradient_through_sign")
    return emitOpError("requires the canonical stop-sign derivative policy");
  if (getMutationMode() != "functional" ||
      getAliasPolicy() != "no_input_output_alias" ||
      getStateTransition() != "m@0-read-only;d_m@1-fresh")
    return emitOpError("requires the functional no-alias Lion state transition");
  if (getOrderedWrites().size() != 3 || getOrderedWrites()[0] != 0 ||
      getOrderedWrites()[1] != 1 || getOrderedWrites()[2] != 2)
    return emitOpError("requires fixed d_p/d_g/d_m write order");
  if (getWorkgroupSize() <= 0)
    return emitOpError("requires workgroup_size > 0");
  return success();
}

LogicalResult OptimizerVJPOp::verify() {
  StringRef optimizer = getOptimizer();
  unsigned inputCount = 0, resultCount = 0;
  if (optimizer == "sgd") {
    inputCount = 3;
    resultCount = 2;
  } else if (optimizer == "momentum" || optimizer == "nesterov") {
    inputCount = 5;
    resultCount = 3;
  } else if (optimizer == "adam" || optimizer == "adamw") {
    inputCount = 7;
    resultCount = 4;
  } else {
    return emitOpError("requires sgd, momentum, nesterov, adam, or adamw");
  }
  if (getInputs().size() != inputCount || getResults().size() != resultCount)
    return emitOpError("operand/result count disagrees with optimizer ABI");
  auto tensor = dyn_cast<RankedTensorType>(getInputs().front().getType());
  if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32())
    return emitOpError("initial optimizer VJP requires a static f32 tensor");
  for (Value value : llvm::concat<Value>(getInputs(), getResults()))
    if (value.getType() != tensor)
      return emitOpError("requires one tensor type for state and cotangents");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getLineagePayload().empty() ||
      (getArch() != "zen5-avx512" && getArch() != "gfx1151"))
    return emitOpError("requires lineage and a promoted architecture identity");
  for (const APFloat &value : {getLearningRate(), getBeta1(), getBeta2(),
                               getEpsilon(), getMomentum(), getWeightDecay()})
    if (!value.isFinite())
      return emitOpError("requires finite optimizer coefficients");
  if ((optimizer == "nesterov") != getNesterov())
    return emitOpError("nesterov identity disagrees with optimizer family");
  if (getMutationMode() != "functional" ||
      getAliasPolicy() != "no_input_output_alias" ||
      getStateTransition() != "state@0-read-only;d_state@1-fresh")
    return emitOpError("requires the functional no-alias VJP transition");
  if (getOrderedWrites().size() != resultCount)
    return emitOpError("requires one ordered write per cotangent result");
  for (auto [index, value] : llvm::enumerate(getOrderedWrites()))
    if (value != static_cast<int64_t>(index))
      return emitOpError("requires ascending deterministic write order");
  if (getWorkgroupSize() <= 0)
    return emitOpError("requires workgroup_size > 0");
  return success();
}

LogicalResult SolverIFTOp::verify() {
  Type expected = getParameter().getType();
  if (getSolution().getType() != expected ||
      getCotangent().getType() != expected)
    return emitOpError("requires parameter, solution, and cotangent to have one type");
  for (Value result : {getResidual(), getLinearSolution(),
                       getParameterCotangent()})
    if (result.getType() != expected)
      return emitOpError("requires all phase results to preserve the input type");
  auto tensor = dyn_cast<RankedTensorType>(expected);
  if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32())
    return emitOpError("initial physical IFT contract requires a static f32 tensor");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getLineagePayload().empty() || getResidualDigest().size() != 64)
    return emitOpError("requires lineage payload and residual digest");
  if (getArch() != "avx512" && getArch() != "gfx1151")
    return emitOpError("physical IFT is promoted only for avx512 and gfx1151");
  if (getResidualModel() != "diagonal_sqrt_v1" ||
      getLinearSolver() != "diagonal_matrix_free_v1" ||
      getWrt() != "parameter" || getAdjointScale().convertToDouble() != -1.0)
    return emitOpError("requires the canonical diagonal-sqrt IFT chain");
  if ((getProductMode() == "vjp" && !getTranspose()) ||
      (getProductMode() == "jvp" && getTranspose()) ||
      (getProductMode() != "vjp" && getProductMode() != "jvp"))
    return emitOpError("requires vjp/transpose or jvp/non-transpose solver products");
  if (getStorage() != "f32" || getAccum() != "f32" ||
      getWorkgroupSize() <= 0)
    return emitOpError("requires f32 storage/accumulation and workgroup_size > 0");
  return success();
}

LogicalResult ESLowRankCorrectionOp::verify() {
  if (getSubject().getType() != getScheduled().getType())
    return emitOpError("must preserve the Graph correction type");
  auto tensor = dyn_cast<RankedTensorType>(getSubject().getType());
  if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32())
    return emitOpError("initial physical ES contract requires a static f32 tensor");
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getLineagePayload().empty() ||
      (getArch() != "gfx1151" && getArch() != "zen5-avx512"))
    return emitOpError(
        "requires lineage payload and exact gfx1151 or Zen 5 AVX-512 identity");
  if (getPopulation() <= 0 || getRowsPerMember() <= 0 || getInDim() <= 0 ||
      getOutDim() <= 0 || getRank() != 1 || getEpoch() < 0)
    return emitOpError("requires positive static dimensions, epoch >= 0, and rank = 1");
  if (!getSigma().isFinite() || getSigma().convertToDouble() <= 0.0 ||
      getScore() != "gaussian" ||
      getRngAlgorithm() != "splitmix64-philox4x32-boxmuller" ||
      getRngVersion() != 1)
    return emitOpError("requires the version-1 Gaussian member RNG contract");
  if (getStorage() != "f32" || getAccum() != "f32" ||
      getWorkgroupSize() <= 0)
    return emitOpError("requires f32 storage/accumulation and workgroup_size > 0");
  return success();
}

LogicalResult AdafactorVJPOp::verify() {
  if (getArtifactHash().size() != 64 ||
      !llvm::all_of(getArtifactHash(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return emitOpError("requires a lowercase SHA-256 artifact_hash");
  if (getLineagePayload().empty() || getArch().empty())
    return emitOpError("requires lineage payload and architecture identity");
  bool factored = getTopology() == "factored";
  if (!factored && getTopology() != "full")
    return emitOpError("topology must be factored or full");
  if (getInputs().size() != (factored ? 5u : 4u) ||
      getResults().size() != (factored ? 4u : 3u))
    return emitOpError("operand/result count disagrees with state topology");
  auto parameter = dyn_cast<RankedTensorType>(getInputs()[0].getType());
  if (!parameter || !parameter.hasStaticShape() ||
      !parameter.getElementType().isF32())
    return emitOpError("requires a static f32 parameter tensor");
  if (getInputs()[1].getType() != parameter ||
      getInputs().back().getType() != parameter ||
      getResults()[0].getType() != parameter ||
      getResults()[1].getType() != parameter)
    return emitOpError("parameter, gradient, cotangent, dP, and dG must match");
  if (factored) {
    if (parameter.getRank() < 2 ||
        getResults()[2].getType() != getInputs()[2].getType() ||
        getResults()[3].getType() != getInputs()[3].getType())
      return emitOpError("factored row/column state result types must match");
  } else if (parameter.getRank() >= 2 ||
             getInputs()[2].getType() != parameter ||
             getResults()[2].getType() != parameter) {
    return emitOpError("full-moment Adafactor requires matching rank-0/1 state");
  }
  if (!getLearningRate().isFinite() || !getBeta2().isFinite() ||
      !getEpsilon().isFinite())
    return emitOpError("requires finite Adafactor coefficients");
  if (getMutationMode() != "functional" ||
      getAliasPolicy() != "no_input_output_alias" ||
      getStateTransition() != "state@0-read-only;d_state@1-fresh")
    return emitOpError("requires functional no-alias Adafactor state transition");
  size_t outputs = factored ? 4 : 3;
  if (getOrderedWrites().size() != outputs)
    return emitOpError("requires one ordered write per output");
  for (auto [index, value] : llvm::enumerate(getOrderedWrites()))
    if (value != static_cast<int64_t>(index))
      return emitOpError("requires ascending output write order");
  if (getWorkgroupSize() <= 0)
    return emitOpError("requires workgroup_size > 0");
  return success();
}

LogicalResult SequenceMixerBackwardOp::verify() {
  if (getArtifactHash().size() != 64 || getLineagePayload().empty() ||
      getArch().empty())
    return emitOpError("requires content-addressed lineage and architecture");
  if (getFamily() != "gated_deltanet" &&
      getFamily() != "kimi_delta_attention" &&
      getFamily() != "modified_delta_attention")
    return emitOpError("requires a proven DeltaNet sequence-mixer family");
  auto q = dyn_cast<RankedTensorType>(getQ().getType());
  auto v = dyn_cast<RankedTensorType>(getV().getType());
  auto scalar = dyn_cast<RankedTensorType>(getBeta().getType());
  if (!q || !v || !scalar || !q.hasStaticShape() || !v.hasStaticShape() ||
      !scalar.hasStaticShape() || !q.getElementType().isF32() ||
      !v.getElementType().isF32() || !scalar.getElementType().isF32() ||
      q.getRank() != 4 || v.getRank() != 4 || scalar.getRank() != 3)
    return emitOpError("requires static rank-4 f32 Q/K/V and rank-3 scalar state");
  if (getK().getType() != q || getDq().getType() != q ||
      getDk().getType() != q || getGate().getType() != v ||
      getDy().getType() != v || getDv().getType() != v ||
      getDgate().getType() != v || getDecay().getType() != scalar ||
      getDbeta().getType() != scalar || getDdecay().getType() != scalar)
    return emitOpError("sequence-mixer input/output tensor identities disagree");
  if (q.getShape().take_front(3) != v.getShape().take_front(3) ||
      scalar.getShape() != q.getShape().take_front(3))
    return emitOpError("sequence-mixer batch/head/sequence shapes disagree");
  if (getChunkSize() <= 0 || getWorkgroupSize() <= 0)
    return emitOpError("requires positive chunk and workgroup sizes");
  if (getMutationMode() != "functional" ||
      getAliasPolicy() != "no_input_output_alias" ||
      getWorkspaceOwner() != "program_launch")
    return emitOpError("requires functional outputs and launch-owned workspace");
  static constexpr StringLiteral phases[] = {
      "checkpoint", "chunk_summary", "chunk_prefix", "chunk_fill", "reverse"};
  if (getPhaseOrder().size() != std::size(phases))
    return emitOpError("requires the five ordered backward phases");
  for (auto [attribute, expected] : llvm::zip_equal(getPhaseOrder(), phases)) {
    auto value = dyn_cast<StringAttr>(attribute);
    if (!value || value.getValue() != expected)
      return emitOpError("requires canonical checkpoint-to-reverse phase order");
  }
  return success();
}

LogicalResult WarpOp::verify() {
  if (getRole().empty())
    return emitOpError("requires a non-empty warp role");
  if (auto count = getCountAttr(); count && count.getInt() <= 0)
    return emitOpError("warp count must be positive");
  if (getBody().empty())
    return emitOpError("requires a non-empty body");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one body block");
  Operation *terminator = getBody().front().getTerminator();
  if (!terminator || terminator->getName().getStringRef() != "schedule.yield")
    return emitOpError("body must terminate with schedule.yield");
  if (terminator->getNumOperands() != getNumResults())
    return emitOpError("yield operand count must match region result count");
  for (auto [yielded, result] :
       llvm::zip_equal(terminator->getOperands(), getResults()))
    if (yielded.getType() != result.getType())
      return emitOpError("yield operand types must match region result types");
  return success();
}

LogicalResult OptimizerShardOp::verify() {
  if (getAxis().empty())
    return emitOpError("requires a non-empty shard axis");
  if (getSubject().getType() != getSharded().getType())
    return emitOpError("must preserve the subject type");
  if (auto partitions = getPartitionsAttr();
      partitions && partitions.getInt() <= 0)
    return emitOpError("partitions must be positive");
  return success();
}

LogicalResult PrefetchOp::verify() {
  if (getSource().getType() != getStaged().getType())
    return emitOpError("must preserve the source type");
  if (getInto().empty())
    return emitOpError("requires a destination memory space");
  StringRef overlap = getOverlap();
  if (overlap != "none" && overlap != "compute" && overlap != "collective")
    return emitOpError("overlap must be none, compute, or collective");
  return success();
}

LogicalResult AsyncCopyOp::verify() {
  if (getSrcSpace().empty() || getDstSpace().empty())
    return emitOpError("requires source and destination memory spaces");
  if (getSrcSpace() == getDstSpace())
    return emitOpError("source and destination memory spaces must differ");
  if (getStageAttr().getInt() < 0)
    return emitOpError("requires stage >= 0");
  StringRef overlap = getOverlap();
  if (overlap != "none" && overlap != "compute" && overlap != "collective")
    return emitOpError("overlap must be none, compute, or collective");
  return success();
}

LogicalResult AwaitMovementOp::verify() { return success(); }

LogicalResult ArtifactOp::verify() {
  if (getHash().empty() || getArch().empty() || getShapeKey().empty())
    return emitOpError("requires non-empty hash, arch, and shape_key");
  return success();
}

LogicalResult KnobOp::verify() {
  if (getName().empty())
    return emitOpError("requires a non-empty knob name");
  if (getChoices().empty())
    return emitOpError("requires at least one choice");
  if (auto logits = getLogitsAttr();
      logits && logits.size() != getChoices().size())
    return emitOpError("logits and choices must have equal length");
  if (getSubject().getType() != getSelected().getType())
    return emitOpError("must preserve the subject type");
  return success();
}
