//===- Passes.h - Tessera Apple Silicon Backend Passes -------*- C++ -*-===//
//
// Pass + pipeline registration for the Apple Silicon Target IR backend.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_PASSES_H
#define TESSERA_TARGET_APPLE_PASSES_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
} // namespace mlir

namespace tessera {
namespace apple {

/// Table-driven runtime fusion lowering used by the production Apple pipeline.
std::unique_ptr<::mlir::Pass> createLowerDeclarativeFusionsToAppleGPUPass();

/// Tile IR → Apple CPU Target IR. `valueMode = false` (default) emits the
/// attribute-only artifact/inspection ops; `valueMode = true` emits
/// value-producing `tessera_apple.cpu.call` ops (the `-full` pipeline).
std::unique_ptr<::mlir::Pass> createLowerTileToAppleCPUPass(bool valueMode = false);

/// Tile IR → Apple GPU Target IR. `valueMode = true` emits value-producing
/// `tessera_apple.gpu.kernel_call` ops for executable ops (and a named
/// diagnostic otherwise); default emits artifact ops.
std::unique_ptr<::mlir::Pass> createLowerTileToAppleGPUPass(bool valueMode = false);

/// Phase-G G-B: tessera.control_for (bounded loop) → tessera_apple.gpu.control_loop
/// Target-IR op (value-preserving; records the run_graph_loop runtime symbol).
std::unique_ptr<::mlir::Pass> createLowerControlForToAppleGPUPass();

/// Phase-G close-out C: tessera.control_if (divergent if/else) →
/// tessera_apple.gpu.control_if Target-IR op (value-preserving; records the
/// run_graph_cond runtime symbol).
std::unique_ptr<::mlir::Pass> createLowerControlIfToAppleGPUPass();

/// Phase-G close-out D: tessera.control_while (bounded while) →
/// tessera_apple.gpu.control_while Target-IR op (value-preserving; records the
/// run_graph_while runtime symbol).
std::unique_ptr<::mlir::Pass> createLowerControlWhileToAppleGPUPass();

/// tessera.matmul (rank-2, f32) → func.call into the Apple CPU runtime
/// shim. Phase 8.2 — the executable counterpart to the artifact-only
/// `tessera-lower-to-apple_cpu` pipeline.
std::unique_ptr<::mlir::Pass> createLowerMatmulToAppleCPUPass();

/// tessera.matmul (rank-2, f32) → func.call into the Apple GPU runtime
/// shim (MTLDevice + MPSMatrixMultiplication). Phase 8.3 — the executable
/// counterpart to the artifact-only `tessera-lower-to-apple_gpu` pipeline.
std::unique_ptr<::mlir::Pass> createLowerMatmulToAppleGPUPass();

/// tessera.rope (rank-2, f32, x.shape == theta.shape) → func.call into the
/// Apple GPU runtime shim's custom-MSL rope kernel. Phase 8.4 — the first
/// concrete custom kernel emitted via the gpu.msl_kernel op contract.
std::unique_ptr<::mlir::Pass> createLowerRopeToAppleGPUPass();

/// tessera.flash_attn (rank-3, f32, head_dim <= 256) → func.call into the
/// Apple GPU runtime shim's custom-MSL flash-attention kernel. Phase 8.4.1.
std::unique_ptr<::mlir::Pass> createLowerFlashAttnToAppleGPUPass();

/// tessera.softmax (rank-2, f32, axis=-1) → func.call into the Apple GPU
/// runtime shim's custom-MSL softmax kernel. Phase 8.4.2.
std::unique_ptr<::mlir::Pass> createLowerSoftmaxToAppleGPUPass();

/// tessera.gelu (rank-2, f32) → func.call into the Apple GPU runtime shim's
/// custom-MSL gelu kernel. Phase 8.4.2.
std::unique_ptr<::mlir::Pass> createLowerGeluToAppleGPUPass();

/// Tier-1 unary activations (silu/relu/sigmoid/tanh/softplus/exp/log/sqrt/
/// rsqrt/neg/abs, f32/f16) → op-coded MPSGraph unary runtime call. 2026-05-29.
std::unique_ptr<::mlir::Pass> createLowerUnaryToAppleGPUPass();

/// tessera.silu_mul (SwiGLU gate, f32) → MPSGraph binary runtime call.
std::unique_ptr<::mlir::Pass> createLowerSiluMulToAppleGPUPass();

/// Tier-1 last-axis row ops (layer_norm/rmsnorm/rmsnorm_safe/log_softmax,
/// f32/f16) → MPSGraph row-op runtime calls (unweighted norms). 2026-05-29.
std::unique_ptr<::mlir::Pass> createLowerRowOpToAppleGPUPass();

/// Fused tessera.matmul → tessera.softmax (rank-2, f32, axis=-1, N <= 256)
/// → func.call into the Apple GPU runtime shim's fused custom-MSL kernel.
/// Phase 8.4.3 — first multi-op fusion. Must run before the single-op
/// matmul/softmax passes so the chain rewrite wins.
std::unique_ptr<::mlir::Pass> createLowerMatmulSoftmaxFusionToAppleGPUPass();

/// Fused tessera.matmul → tessera.softmax → tessera.matmul (rank-2,
/// f32/f16/bf16, axis=-1, N <= 256, P <= 256) → func.call into the
/// Apple GPU runtime shim's fused 3-op custom-MSL kernel — the full
/// attention block: O = softmax(A @ B) @ C. Phase 8.4.5. Must run before
/// the 2-op fusion so the longer chain wins.
std::unique_ptr<::mlir::Pass> createLowerMatmulSoftmaxMatmulFusionToAppleGPUPass();

/// Fused tessera.matmul → tessera.gelu (rank-2, f32, N <= 256) → func.call
/// into the Apple GPU runtime shim's matmul_gelu MSL kernel. Phase 8.4.7
/// — MLP block activation pattern.
std::unique_ptr<::mlir::Pass> createLowerMatmulGeluFusionToAppleGPUPass();

/// Fused tessera.matmul → tessera.rmsnorm (rank-2, f32, N <= 256) →
/// func.call into the Apple GPU runtime shim's matmul_rmsnorm MSL kernel.
/// Phase 8.4.7 — transformer normalization pattern.
std::unique_ptr<::mlir::Pass> createLowerMatmulRMSNormFusionToAppleGPUPass();

/// Fused tessera.swiglu_fused (rank-2, f32/f16/bf16, H ≤ 256, Kout ≤ 256)
/// → func.call into the Apple GPU runtime shim's swiglu MSL kernel —
/// the SwiGLU MLP block as a single GPU dispatch. Phase 8.4.8 (Stage 3
/// of the SwiGLU Performance Plan in `docs/CANONICAL_API.md`). Must run
/// before per-op matmul lowering so the longest fusion wins.
std::unique_ptr<::mlir::Pass> createLowerSwigluFusionToAppleGPUPass();

/// tessera.linear_attn (rank-4, f32, D_qk*D_v ≤ 256, causal) → func.call
/// into the Apple GPU runtime shim's linear-attn MSL kernel. attention
/// _variants_plan, LA-2 — linear / kernel-feature attention as a single
/// GPU dispatch. Out-of-envelope inputs fall through to the host
/// reference path.
std::unique_ptr<::mlir::Pass> createLowerLinearAttnToAppleGPUPass();

/// tessera.attn_local_window_2d (rank-5, f32, patch*D ≤ 1024) → func.call
/// into the Apple GPU runtime shim's 2D local-window attention MSL kernel.
/// Sub-2 (2026-05-20) — single-device tiled lowering for the
/// CorrDiff-style spatial-attention surface.  Out-of-envelope inputs
/// (non-f32, dynamic shape, oversized patch) fall through to the host
/// reference path.
std::unique_ptr<::mlir::Pass> createLowerAttnLocalWindow2DToAppleGPUPass();

/// tessera.mla_decode_fused (rank-3 Q + rank-2 weights, f32) →
/// func.call into the Apple GPU runtime shim. attention_variants_plan,
/// MLA-2 — DeepSeek MLA decode as a single GPU dispatch. Today the
/// runtime materializes the latent + expanded K/V on the host (the
/// memory-saving cache lives in the LatentKVCacheHandle); the
/// absorb-K speed win lands as a follow-up MSL kernel.
std::unique_ptr<::mlir::Pass> createLowerMLADecodeFusionToAppleGPUPass();

/// tessera.native_sparse_attn_fused (rank-4, f32) → func.call into the
/// Apple GPU runtime shim. attention_variants_plan, NSA-5 — DeepSeek
/// NSA fused kernel. Host-reference path today; a fully fused MSL
/// kernel that does all three branches in a single dispatch is a
/// follow-up.
std::unique_ptr<::mlir::Pass> createLowerNSAFusionToAppleGPUPass();

/// Register the Apple Silicon dialect into a DialectRegistry. Convenience
/// wrapper that forwards to registerAppleDialect from TesseraAppleDialect.h.
void registerTesseraAppleBackendDialects(::mlir::DialectRegistry &registry);

/// Register the Apple Silicon passes (and dialects) for use in tessera-opt.
/// This is the entry point tessera-opt should call.
void registerTesseraAppleBackendPasses(::mlir::DialectRegistry &registry);

/// Force-link the canonical pipelines:
///   - tessera-lower-to-apple_cpu
///   - tessera-lower-to-apple_gpu
/// Pipelines self-register at static init time; calling this from main()
/// keeps the linker from dropping the registration object in static builds.
void registerTesseraAppleBackendPipelines();

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_PASSES_H
