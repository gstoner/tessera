//===- tessera_jit.cpp ----------------------------------------------------===//
// Phase 0 production-lane plumbing (docs/spec/PRODUCTION_COMPILER_PLAN.md;
// RUNTIME_ABI_SPEC.md §12). EXPERIMENTAL — NOT "runtime v2".
//
// A standalone CPU JIT: take an MLIR module containing `func @tessera_jit_add`
// (in tessera+std form), run the production lowering pipeline
//   tessera-to-linalg -> one-shot-bufferize -> buffer-results-to-out-params
//                     -> linalg-to-loops -> scf-to-cf -> {arith,memref,cf,func}
//                     -> reconcile-unrealized-casts
// and JIT it with mlir::ExecutionEngine. Invocation is via the C-interface
// wrapper `_mlir_ciface_tessera_jit_add` (we set `llvm.emit_c_interface`).
//
// Phase-0 scope (ratified guardrails): rank-2 f32 only; one hardcoded descriptor
// struct; destination-passing (caller allocates `out`, fn returns void); failures
// surface as a nonzero return + last-error string (Python raises).
//===----------------------------------------------------------------------===//

#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

using namespace mlir;

namespace {

thread_local std::string g_lastError;

// Proof-of-execution counter (RUNTIME_ABI_SPEC §12 / Phase-0 guardrail): every
// successful JIT invoke increments this. A numerically-correct result without an
// increment would mean a silent fallback — the oracle test asserts count+1 to
// make that impossible to hide.
std::atomic<int64_t> g_invocations{0};

void setError(const std::string &msg) { g_lastError = msg; }

// rank-2, row-major, f32 — the only descriptor Phase 0 admits (guardrail).
struct MemRef2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

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

// Destination-passing rewrite (RUNTIME_ABI_SPEC §12.3): turn a single-memref-
// returning function into one that takes the destination as a trailing
// caller-allocated out-param and returns void. We do this explicitly rather than
// via -buffer-results-to-out-params, which silently declines to convert a
// freshly-allocated identity-layout result on this toolchain. Deterministic and
// exactly the ABI we want for the Phase-0 total-op class.
LogicalResult rewriteResultToOutParam(ModuleOp module, StringRef name) {
  auto fn = module.lookupSymbol<func::FuncOp>(name);
  if (!fn || fn.getNumResults() != 1)
    return failure();
  auto memrefTy = dyn_cast<MemRefType>(fn.getResultTypes()[0]);
  if (!memrefTy)
    return failure();
  Block &entry = fn.getBody().front();
  auto retOp = cast<func::ReturnOp>(entry.getTerminator());
  Value ret = retOp.getOperand(0);

  BlockArgument outArg = entry.addArgument(memrefTy, fn.getLoc());
  // Redirect the producer (e.g. linalg.generic outs) and the return to write the
  // caller's buffer; the original alloc becomes dead.
  ret.replaceAllUsesWith(outArg);
  if (Operation *def = ret.getDefiningOp())
    if (def->use_empty())
      def->erase();

  OpBuilder b(retOp);
  func::ReturnOp::create(b, retOp.getLoc());
  retOp.erase();
  fn.setType(FunctionType::get(fn.getContext(), entry.getArgumentTypes(), {}));
  return success();
}

// The Phase-0 lowering pipeline. Brittle by nature (pass ordering churn) — kept
// explicit so a failure points at the exact stage.
LogicalResult buildAndRunPipeline(ModuleOp module) {
  // Stage 1: tessera → linalg → bufferized memref form, identity boundary layout.
  PassManager pm1(module->getContext());
  maybeTrace(pm1);
  pm1.nest<func::FuncOp>().addPass(tessera::createTesseraToLinalgPass());
  // tensor.empty (the DPS init) has no buffer semantics; convert to alloc_tensor
  // so one-shot-bufferize can place it ("op was not bufferized" otherwise).
  pm1.nest<func::FuncOp>().addPass(
      bufferization::createEmptyTensorToAllocTensorPass());
  bufferization::OneShotBufferizePassOptions bopts;
  bopts.bufferizeFunctionBoundaries = true;
  // Identity layout at the boundary == the ABI's row-major descriptor contract
  // (RUNTIME_ABI_SPEC §12.4); dynamic strided layout would break descriptor packing.
  bopts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm1.addPass(bufferization::createOneShotBufferizePass(bopts));
  if (failed(pm1.run(module)))
    return failure();

  // Stage 1.5: explicit destination-passing rewrite.
  if (failed(rewriteResultToOutParam(module, "tessera_jit_add")))
    return failure();

  // Stage 2: memref/loops → LLVM dialect.
  PassManager pm2(module->getContext());
  maybeTrace(pm2);
  pm2.nest<func::FuncOp>().addPass(createConvertLinalgToLoopsPass());
  pm2.addPass(createSCFToControlFlowPass());
  pm2.addPass(memref::createExpandStridedMetadataPass());
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

// Compile `mlir_text` (must contain `func @tessera_jit_add`). Returns an opaque
// handle on success, nullptr on failure (see tessera_jit_last_error()).
void *tessera_jit_compile(const char *mlir_text) {
  ensureNativeTargetInit();
  auto jm = std::make_unique<JitModule>();

  DialectRegistry registry;
  tessera::registerTesseraDialects(registry);
  registry.insert<func::FuncDialect, arith::ArithDialect, scf::SCFDialect,
                  tensor::TensorDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  // One-shot-bufferize dispatches through BufferizableOpInterface; each dialect
  // ships its implementation as an *external model* that must be registered on
  // the registry. Without these, bufferization reports "op was not bufferized".
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  jm->ctx = std::make_unique<MLIRContext>(registry);
  // Single-shot compiles: multithreading buys nothing and blocks IR-print tracing.
  jm->ctx->disableMultithreading();
  // Do NOT loadAllAvailableDialects(): the parser loads dialects on demand and
  // the PassManager pre-loads each pass's declared dependent dialects. Force-
  // constructing every registered dialect here is both unnecessary and (on this
  // toolchain) crash-prone.

  jm->module = parseSourceString<ModuleOp>(mlir_text, jm->ctx.get());
  if (!jm->module) {
    setError("tessera_jit: failed to parse MLIR module");
    return nullptr;
  }

  // Mark the entry for C-interface wrapper emission (_mlir_ciface_*).
  if (auto fn = jm->module->lookupSymbol<func::FuncOp>("tessera_jit_add"))
    fn->setAttr("llvm.emit_c_interface", UnitAttr::get(jm->ctx.get()));
  else {
    setError("tessera_jit: module has no func @tessera_jit_add");
    return nullptr;
  }

  if (failed(buildAndRunPipeline(*jm->module))) {
    setError("tessera_jit: lowering pipeline failed");
    return nullptr;
  }

  if (::getenv("TESSERA_JIT_DUMP"))
    jm->module->dump();

  ExecutionEngineOptions opts;
  opts.transformer = makeOptimizingTransformer(/*optLevel=*/2,
                                               /*sizeLevel=*/0,
                                               /*targetMachine=*/nullptr);
  auto expectedEngine = ExecutionEngine::create(*jm->module, opts);
  if (!expectedEngine) {
    setError("tessera_jit: ExecutionEngine::create failed");
    return nullptr;
  }
  jm->engine = std::move(*expectedEngine);
  return jm.release();
}

// rank-2 f32 add. Caller allocates `out`. Returns 0 on success, 1 on failure.
int tessera_jit_add_2d_f32(void *handle, const float *a, const float *b,
                           float *out, int64_t d0, int64_t d1) {
  auto *jm = static_cast<JitModule *>(handle);
  if (!jm || !jm->engine) {
    setError("tessera_jit: null/invalid handle");
    return 1;
  }
  auto fill = [](MemRef2DF32 &d, const float *p, int64_t r, int64_t c) {
    d.allocated = const_cast<float *>(p);
    d.aligned = const_cast<float *>(p);
    d.offset = 0;
    d.sizes[0] = r;
    d.sizes[1] = c;
    d.strides[0] = c;
    d.strides[1] = 1;
  };
  MemRef2DF32 da, db, dout;
  fill(da, a, d0, d1);
  fill(db, b, d0, d1);
  fill(dout, out, d0, d1);

  auto expectedFn = jm->engine->lookup("_mlir_ciface_tessera_jit_add");
  if (!expectedFn) {
    setError("tessera_jit: _mlir_ciface_tessera_jit_add not found");
    return 1;
  }
  // C-iface passes each memref as a pointer to its descriptor; out-param last.
  using Fn = void (*)(void *, void *, void *);
  reinterpret_cast<Fn>(*expectedFn)(&da, &db, &dout);
  g_invocations.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

// Number of successful JIT invocations since process start. Proof-of-execution
// for the production lane (see g_invocations comment).
int64_t tessera_jit_invocation_count(void) {
  return g_invocations.load(std::memory_order_relaxed);
}

void tessera_jit_destroy(void *handle) {
  delete static_cast<JitModule *>(handle);
}

} // extern "C"
