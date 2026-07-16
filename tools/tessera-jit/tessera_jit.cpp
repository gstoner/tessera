//===- tessera_jit.cpp ----------------------------------------------------===//
// Production-lane CPU JIT plumbing (docs/spec/PRODUCTION_COMPILER_PLAN.md;
// RUNTIME_ABI_SPEC.md §12). EXPERIMENTAL — NOT "runtime v2".
//
// Phase 0 (landed): boundary proof on a single hardcoded `tessera_jit_add`
// symbol with a typed C dispatcher. Phase 1 (this file): generalized to any
// MLIR function. The C ABI exposes three primitives:
//
//   tessera_jit_compile(mlir_text) -> handle
//   tessera_jit_invoke(handle, name, void** packed_args, int nargs) -> int
//   tessera_jit_destroy(handle)
//
// plus tessera_jit_last_error() and tessera_jit_invocation_count() for
// proof-of-execution. `invokePacked` from mlir::ExecutionEngine handles the
// c-iface dispatch for any function signature, so adding a new op needs zero
// changes here — only an MLIR lowering pattern + a Python helper.
//
// The whole module is run through the same pipeline:
//   tessera-to-linalg -> empty-tensor-to-alloc-tensor
//                     -> one-shot-bufferize (identity boundary layout)
//                     -> [walk] DPS rewrite: single-memref result -> trailing out-param
//                     -> linalg-to-loops -> scf-to-cf -> arith/cf/memref/func to LLVM
//                     -> reconcile-unrealized-casts
// `_mlir_ciface_<name>` wrappers are emitted on every function in the module.
//===----------------------------------------------------------------------===//

#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"  // createCanonicalizerPass / CSE
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
// Phase 4 linalg→vector GEMM lane (opt-in via TESSERA_JIT_VECTORIZE). Tiling +
// vectorization is driven by the TRANSFORM INTERPRETER (a proven path — the
// direct scf::tileUsingSCF C++ call null-derefs; see COMPILER_AUDIT Phase 4),
// then the resulting vector ops are lowered to LLVM.
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

// libffi: dynamic-arity call of the c-iface wrapper. Header path differs by
// platform (macOS SDK ships <ffi/ffi.h>; Linux ships <ffi.h>).
#if __has_include(<ffi.h>)
#include <ffi.h>
#elif __has_include(<ffi/ffi.h>)
#include <ffi/ffi.h>
#else
#error "libffi header (ffi.h) not found"
#endif

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using namespace mlir;

namespace {

thread_local std::string g_lastError;

// Proof-of-execution counter (RUNTIME_ABI_SPEC §12 guardrail): every successful
// JIT invoke increments this. A numerically-correct result without an increment
// would mean a silent fallback — the oracle tests assert the counter advanced.
std::atomic<int64_t> g_invocations{0};

// Compile counter: increments once per successful tessera_jit_compile. The
// Python compilation cache asserts that repeated same-shape calls do NOT
// re-increment this (cache hit ⇒ no recompile).
std::atomic<int64_t> g_compiles{0};

void setError(const std::string &msg) { g_lastError = msg; }

struct JitModule {
  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ExecutionEngine> engine;
};

void ensureNativeTargetInit() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

void maybeTrace(PassManager &pm) {
  if (::getenv("TESSERA_JIT_TRACE"))
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
        /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
        /*printModuleScope=*/true, /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
}

// DPS rewrite (RUNTIME_ABI_SPEC §12.3): for every function whose sole result is
// a memref, append the result as a trailing caller-allocated out-param and turn
// the function into a void return. Phase 0 hardcoded one name; Phase 1 walks.
// Functions that already return void / multiple results / non-memref are left
// untouched (a non-applicable function isn't an error).
//
// We copy the produced buffer into the out-param rather than redirect-and-erase
// the producer. The redirect trick only works when the producer writes through
// a retargetable `outs` operand; it silently destroys control flow (an scf.for
// whose result is replaced becomes dead and is erased, losing the loop). The
// copy is correct for ANY producer, and for our identity-layout C-contiguous
// boundary (§12.4) `memref.copy` lowers to a `memcpy` intrinsic — no runtime
// symbol, negligible cost.
LogicalResult rewriteResultsToOutParams(ModuleOp module) {
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (fn.getNumResults() == 0 || fn.isExternal())
      continue;
    // Every result must be a memref to apply DPS; otherwise leave the function.
    bool allMemref = llvm::all_of(fn.getResultTypes(), [](Type t) {
      return isa<MemRefType>(t);
    });
    if (!allMemref)
      continue;

    Block &entry = fn.getBody().front();
    auto retOp = cast<func::ReturnOp>(entry.getTerminator());
    SmallVector<Value> rets(retOp.getOperands());

    OpBuilder b(retOp);
    // Append one out-param per result (in result order, after the inputs) and
    // copy each result into it. c-iface order becomes (inputs..., out0, out1...).
    for (Value ret : rets) {
      auto memrefTy = cast<MemRefType>(ret.getType());
      BlockArgument outArg = entry.addArgument(memrefTy, fn.getLoc());
      memref::CopyOp::create(b, retOp.getLoc(), ret, outArg);
    }
    func::ReturnOp::create(b, retOp.getLoc());
    retOp.erase();
    fn.setType(
        FunctionType::get(fn.getContext(), entry.getArgumentTypes(), {}));
  }
  return success();
}

// Mark every non-external function for C-interface wrapper emission, and mark
// its tensor arguments read-only. The c-iface (`_mlir_ciface_<name>`) is what we
// look up in the ExecutionEngine. The read-only marking is an ABI guarantee:
// inputs must not be mutated (DPS — inputs read, outputs written). Without it,
// one-shot-bufferize may write in-place into a caller's input buffer (e.g.
// tensor.insert_slice for write_row), silently corrupting it; `writable = false`
// forces a copy instead.
void markCInterface(ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  auto unit = UnitAttr::get(ctx);
  auto notWritable = BoolAttr::get(ctx, false);
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (fn.isExternal())
      continue;
    fn->setAttr("llvm.emit_c_interface", unit);
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
      if (isa<RankedTensorType>(fn.getArgument(i).getType()))
        fn.setArgAttr(i, "bufferization.writable", notWritable);
  }
}

// Phase 4 (2026-06-16) — opt-in linalg→vector GEMM lane via the transform
// interpreter. Tiling each `linalg.matmul` to small static tiles makes the
// vectorizer emit a `vector.contract` whose K-reduction accumulates in a VECTOR
// REGISTER (the scf.for tensor iter_arg) rather than the memref C[i,j] reloaded
// every k-iteration — the memory accumulator that blocked LLVM's loop vectorizer
// (scalar ConvertLinalgToLoops ran ~2 GFLOP/s, ~50x off Accelerate). The direct
// scf::tileUsingSCF C++ call null-derefs in this context; the transform
// interpreter is the proven path (it tiles the identical op cleanly under
// mlir-opt). Runs on TENSORS before bufferization. Best-effort: a transform
// failure leaves the matmul as linalg → the scalar loop lowering (always
// correct).
static const char *kTileVectorizeTransform = R"MLIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [8, 16, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // ALSO tile the 2-D elementwise/fill ops (the matmul-output `add` + the C
    // init) — otherwise vectorize_children materializes a giant vector<MxN> for
    // them that LLVM unrolls into M·N scalar ops (compile time blows up: ~22s at
    // 256, unbounded beyond). Tiling them too keeps every vector bounded.
    %ew = transform.structured.match ops{["linalg.generic", "linalg.fill", "linalg.elementwise"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %te, %e0, %e1 = transform.structured.tile_using_for %ew tile_sizes [8, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // vectorize_children_and_apply_patterns produces the contract→outerproduct
    // (efficient fma) form (~7x the multi_reduction `vectorize` gives). Must
    // target an isolated-from-above op (the func), not a loop — fine now that
    // every linalg op is tiled to bounded sizes.
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)MLIR";

// Engage the lane only when every linalg op's static dims are within this bound.
// The earlier large-N runtime crash was the untiled elementwise ops blowing up
// into giant unrolled vectors; that's fixed (the transform tiles those too), so
// this is now purely a compile-time safety valve — a very large matmul has many
// tiles and a long (but finite) compile. 2048 covers typical transformer layer
// dims; override via TESSERA_JIT_VECTORIZE_MAXDIM for larger.
static int64_t vectorizeMaxDim() {
  if (const char *e = ::getenv("TESSERA_JIT_VECTORIZE_MAXDIM"))
    return std::strtoll(e, nullptr, 10);
  return 2048;
}

static bool withinVectorizeEnvelope(ModuleOp module) {
  int64_t maxDim = vectorizeMaxDim();
  bool ok = true;
  module.walk([&](linalg::LinalgOp op) {
    for (Value v : op->getOperands()) {
      if (auto t = dyn_cast<RankedTensorType>(v.getType()))
        for (int64_t d : t.getShape())
          if (ShapedType::isStatic(d) && d > maxDim)
            ok = false;
    }
  });
  return ok;
}

static LogicalResult tileAndVectorizeLinalg(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  // Parse the transform sequence in the payload's context (so the transform
  // dialect + extensions resolve against the same registry).
  OwningOpRef<ModuleOp> transformModule =
      parseSourceString<ModuleOp>(kTileVectorizeTransform, ctx);
  if (!transformModule)
    return failure();
  Operation *transformRoot =
      transform::detail::findTransformEntryPoint(module, *transformModule);
  if (!transformRoot)
    return failure();
  transform::TransformOptions options;
  return transform::applyTransformNamedSequence(module, transformRoot,
                                                *transformModule, options);
}

// Lower the vector.contract (→ outerproduct/fma) + transfer ops emitted by the
// vectorizer. Run AFTER bufferization so the transfers are memref-based (lowering
// them pre-bufferize on tensor values leaves unrealized_conversion_casts that
// fail LLVM translation).
static LogicalResult lowerVectorOps(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  // RAISE the vectorizer's multiply + multi_reduction back to vector.contract,
  // then lower contract → outerproduct → fma (the efficient form). Without the
  // raise, multi_reduction lowers to many scalar reduces (~5 GFLOP/s); with it,
  // the outerproduct fma path is ~7x faster.
  vector::populateVectorReductionToContractPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(
      patterns, vector::VectorContractLowering::OuterProduct);
  // Any multi_reduction the raise didn't catch still needs lowering.
  vector::populateVectorMultiReductionReorderPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  vector::populateVectorMultiReductionFlatteningPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  vector::populateVectorMultiReductionUnrollingPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  // NB: do NOT run populateVectorTransferLoweringPatterns — it rewrites
  // transfer_read→vector.load on the *strided* tile subview, which then can't
  // lower to LLVM. Leave transfers for ConvertVectorToSCF (pm2), which loops over
  // the strides cleanly.
  // multi_reduction lowering lifts to 2-D via vector.shape_cast — lower those
  // (and any vector.transpose) so only 1-D vector ops reach ConvertVectorToLLVM.
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransposeLowering::EltWise);
  return applyPatternsGreedily(module, std::move(patterns));
}

LogicalResult buildAndRunPipeline(ModuleOp module) {
  // Stage 1a: tessera → linalg (tensors).
  PassManager pm1a(module->getContext());
  maybeTrace(pm1a);
  // Phase 1 (front-to-back closure plan): canonicalize the Tessera dialect
  // *before* lowering, so per-op folders/canonicalizers (identity cast,
  // transpose-of-transpose, …) + CSE bite on the executed CPU path. This is
  // what makes the Graph-IR optimizations observable end-to-end through the JIT.
  pm1a.addPass(createCanonicalizerPass());
  pm1a.addPass(createCSEPass());
  pm1a.nest<func::FuncOp>().addPass(tessera::createTesseraToLinalgPass());
  if (failed(pm1a.run(module)))
    return failure();

  // Stage 1b (opt-in): tile + vectorize on tensors, before bufferization. Only
  // within the safe size envelope — larger programs stay on the scalar lane.
  bool vectorized = false;
  if (::getenv("TESSERA_JIT_VECTORIZE") && withinVectorizeEnvelope(module)) {
    if (failed(tileAndVectorizeLinalg(module)))
      return failure();
    vectorized = true;
  }

  // Stage 1c: tensor.empty → alloc_tensor, then one-shot bufferize.
  PassManager pm1(module->getContext());
  maybeTrace(pm1);
  // tensor.empty (DPS init) has no buffer semantics on its own; convert to
  // alloc_tensor so one-shot-bufferize can place it.
  pm1.nest<func::FuncOp>().addPass(
      bufferization::createEmptyTensorToAllocTensorPass());
  bufferization::OneShotBufferizePassOptions bopts;
  bopts.bufferizeFunctionBoundaries = true;
  // Identity layout at the boundary == the ABI's row-major descriptor contract.
  bopts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  // Phase 2 control flow: an scf.for body that yields a freshly-allocated tensor
  // (e.g. acc = acc + x) is not buffer-equivalent to its iter_arg. Permit the
  // loop to carry a new allocation rather than erroring on non-equivalence.
  bopts.allowReturnAllocsFromLoops = true;
  pm1.addPass(bufferization::createOneShotBufferizePass(bopts));
  if (failed(pm1.run(module)))
    return failure();

  // Stage 1.5: explicit DPS rewrite (every function in the module).
  if (failed(rewriteResultsToOutParams(module)))
    return failure();

  // Stage 1.6 (opt-in lane): lower vector.contract/transfer now that bufferize
  // has made the transfers memref-based. Only when the lane actually engaged.
  if (vectorized) {
    if (failed(lowerVectorOps(module)))
      return failure();
  }

  // Stage 2a: linalg → scalar scf loops (this is where the matmul's
  // arith.mulf/addf reduction body is created).
  PassManager pm2a(module->getContext());
  maybeTrace(pm2a);
  pm2a.nest<func::FuncOp>().addPass(createConvertLinalgToLoopsPass());
  if (failed(pm2a.run(module)))
    return failure();

  // Phase 4 (2026-06-16): stamp fast-math on float arith ops so LLVM may
  // vectorize the matmul/reduction inner loop. A float reduction (`acc += a*b`)
  // is NOT auto-vectorized without `reassoc` — reordering the additions changes
  // the result — so the loops stayed scalar (~2 GFLOP/s, ~50x off Accelerate).
  // Tessera's GEMM is fast-math by contract (f32 accumulate, rtol≈1e-4), so
  // `fast` is the intended numerics; it's the difference between a scalar and a
  // SIMD inner loop on NEON.
  auto fmFast = arith::FastMathFlagsAttr::get(module.getContext(),
                                              arith::FastMathFlags::fast);
  module.walk([&](Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
            arith::MinNumFOp, arith::NegFOp>(op))
      op->setAttr("fastmath", fmFast);
  });

  // Stage 2b: memref/loops/vector → LLVM dialect.
  PassManager pm2(module->getContext());
  maybeTrace(pm2);
  // Vector ops (from the opt-in linalg→vector lane) → LLVM. No-op when the lane
  // is off (no vector ops present).
  // Remaining vector.transfer ops (broadcast/permutation forms the pattern
  // lowering left) → scf loops + simple loads. Safety net before VectorToLLVM.
  // Expand strided metadata FIRST so the tile subviews (memref<..., strided<...>>
  // from the tiling's extract_slice) become plain base+offset arith — otherwise
  // vector.load/store on a strided memref can't lower and leaves casts.
  pm2.addPass(memref::createExpandStridedMetadataPass());
  // Remaining vector.transfer ops (broadcast/permutation forms the pattern
  // lowering left) → scf loops + simple loads.
  pm2.addPass(createConvertVectorToSCFPass());
  pm2.addPass(createLowerAffinePass());  // VectorToSCF emits affine.apply/min
  pm2.addPass(createConvertVectorToLLVMPass());
  // Vectorization emits `ub.poison` for padding lanes → lower to LLVM poison.
  pm2.addPass(createUBToLLVMConversionPass());
  pm2.addPass(createSCFToControlFlowPass());
  pm2.addPass(createConvertMathToLLVMPass());
  pm2.addPass(createArithToLLVMConversionPass());
  pm2.addPass(createConvertControlFlowToLLVMPass());
  pm2.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm2.addPass(createConvertFuncToLLVMPass());
  pm2.addPass(createReconcileUnrealizedCastsPass());
  return pm2.run(module);
}

} // namespace

extern "C" {

const char *tessera_jit_last_error(void) { return g_lastError.c_str(); }

// Compile any MLIR module. Every non-external function is marked for c-iface
// emission and has DPS applied when its sole result is a memref. Returns an
// opaque handle on success, nullptr on failure (see tessera_jit_last_error()).
void *tessera_jit_compile(const char *mlir_text) {
  ensureNativeTargetInit();
  auto jm = std::make_unique<JitModule>();

  DialectRegistry registry;
  tessera::registerTesseraDialects(registry);
  registry.insert<func::FuncDialect, arith::ArithDialect, scf::SCFDialect,
                  tensor::TensorDialect, linalg::LinalgDialect,
                  math::MathDialect, memref::MemRefDialect,
                  bufferization::BufferizationDialect, cf::ControlFlowDialect,
                  vector::VectorDialect, transform::TransformDialect,
                  ub::UBDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  // BufferizableOpInterface external models — without these, one-shot-bufferize
  // reports "op was not bufferized".
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  // Transform-dialect extension: the linalg/structured transform ops
  // (transform.structured.tile_using_for / vectorize / match) used by the opt-in
  // linalg→vector lane.
  linalg::registerTransformDialectExtension(registry);
  // TilingInterface external models on the PAYLOAD ops — the transform extension
  // provides the transform *ops*, but tile_using_for needs linalg.matmul (and the
  // tensor slice ops) to *implement* TilingInterface, else it errors "only ops
  // implementing TilingInterface are supported" and the lane falls back to numpy.
  linalg::registerTilingInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);

  jm->ctx = std::make_unique<MLIRContext>(registry);
  jm->ctx->disableMultithreading();

  jm->module = parseSourceString<ModuleOp>(mlir_text, jm->ctx.get());
  if (!jm->module) {
    setError("tessera_jit: failed to parse MLIR module");
    return nullptr;
  }

  markCInterface(*jm->module);

  if (failed(buildAndRunPipeline(*jm->module))) {
    setError("tessera_jit: lowering pipeline failed");
    return nullptr;
  }

  if (::getenv("TESSERA_JIT_DUMP"))
    jm->module->dump();

  ExecutionEngineOptions opts;
  // Phase 4 (2026-06-16): build a host-targeted TargetMachine so the LLVM
  // optimizer (the transformer) is target-aware. With targetMachine=nullptr the
  // vectorizer has no NEON cost model and the linalg-lowered loops stay scalar
  // (measured ~2 GFLOP/s GEMM, ~50-110x off numpy/Accelerate). detectHost() pins
  // the native CPU (apple-m1…) + features (NEON/FMA) so -O3 vectorizes for the
  // host. The TM must outlive ExecutionEngine::create (the transformer runs
  // synchronously inside it); this local does.
  std::unique_ptr<llvm::TargetMachine> hostTM;
  if (auto tmb = llvm::orc::JITTargetMachineBuilder::detectHost()) {
    if (auto tmOrErr = tmb->createTargetMachine())
      hostTM = std::move(*tmOrErr);
    else
      llvm::consumeError(tmOrErr.takeError());
  } else {
    llvm::consumeError(tmb.takeError());
  }
  opts.transformer = makeOptimizingTransformer(/*optLevel=*/3,
                                               /*sizeLevel=*/0,
                                               /*targetMachine=*/hostTM.get());
  // The opt-in vectorize lane's DPS out-param copy can lower memref.copy to the
  // generic `memrefCopy` runtime helper (between different-layout memrefs). Load
  // MLIR's C runner utils so that symbol resolves. Default to the Homebrew LLVM
  // path (the build pin); overridable via TESSERA_MLIR_RUNNER_UTILS. Only loaded
  // when the lane is on, so normal compiles are unaffected.
  static const std::string kRunnerUtils = [] {
    if (const char *e = ::getenv("TESSERA_MLIR_RUNNER_UTILS"))
      return std::string(e);
    return std::string("/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib");
  }();
  SmallVector<StringRef> sharedLibs;
  if (::getenv("TESSERA_JIT_VECTORIZE"))
    sharedLibs.push_back(kRunnerUtils);
  opts.sharedLibPaths = sharedLibs;
  auto expectedEngine = ExecutionEngine::create(*jm->module, opts);
  if (!expectedEngine) {
    setError("tessera_jit: ExecutionEngine::create failed");
    return nullptr;
  }
  jm->engine = std::move(*expectedEngine);
  g_compiles.fetch_add(1, std::memory_order_relaxed);
  return jm.release();
}

int64_t tessera_jit_compile_count(void) {
  return g_compiles.load(std::memory_order_relaxed);
}

// Generic invoke: dispatch any compiled function by name. Looks up
// `_mlir_ciface_<name>` (the stable C-interface wrapper MLIR emits when
// `llvm.emit_c_interface` is set) and calls it directly. The c-iface ABI for
// our DPS memref functions is `void(Desc*, Desc*, ..., Desc*)`, so
// `packed_args[i]` is the i-th memref descriptor pointer — one level of
// indirection, period. Avoids `invokePacked`'s wrapper-symbol semantics
// (which were brittle on this toolchain).
//
// Returns 0 on success, 1 on failure. The execution counter advances only on
// successful dispatch, so a numpy fallback masquerading as a JIT call is
// impossible — proof-of-execution survives generalization.
int tessera_jit_invoke(void *handle, const char *name, void **packed_args,
                       int nargs) {
  auto *jm = static_cast<JitModule *>(handle);
  if (!jm || !jm->engine) {
    setError("tessera_jit: null/invalid handle");
    return 1;
  }
  std::string sym = std::string("_mlir_ciface_") + name;
  auto expectedFn = jm->engine->lookup(sym);
  if (!expectedFn) {
    llvm::consumeError(expectedFn.takeError());
    setError("tessera_jit: symbol not found: " + sym);
    return 1;
  }
  void *fn = reinterpret_cast<void *>(*expectedFn);

  // Every c-iface argument is a `void*` (a memref descriptor pointer) and the
  // function returns void, so a single libffi call handles ANY arity — no
  // hand-written per-arity dispatch, no cap. ffi wants `avalues[i]` to point to
  // the storage of argument i; argument i's value is `packed_args[i]`, so we
  // pass `&packed_args[i]`.
  if (nargs < 0) {
    setError("tessera_jit: negative nargs");
    return 1;
  }
  std::vector<ffi_type *> atypes(static_cast<size_t>(nargs), &ffi_type_pointer);
  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, static_cast<unsigned>(nargs),
                   &ffi_type_void, atypes.data()) != FFI_OK) {
    setError("tessera_jit: ffi_prep_cif failed");
    return 1;
  }
  std::vector<void *> avalues(static_cast<size_t>(nargs));
  for (int i = 0; i < nargs; ++i)
    avalues[i] = &packed_args[i];
  ffi_call(&cif, FFI_FN(fn), /*rvalue=*/nullptr, avalues.data());

  g_invocations.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

int64_t tessera_jit_invocation_count(void) {
  return g_invocations.load(std::memory_order_relaxed);
}

void tessera_jit_destroy(void *handle) {
  delete static_cast<JitModule *>(handle);
}

} // extern "C"
