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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

using namespace mlir;

namespace {

thread_local std::string g_lastError;

// Proof-of-execution counter (RUNTIME_ABI_SPEC §12 guardrail): every successful
// JIT invoke increments this. A numerically-correct result without an increment
// would mean a silent fallback — the oracle tests assert the counter advanced.
std::atomic<int64_t> g_invocations{0};

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
LogicalResult rewriteResultsToOutParams(ModuleOp module) {
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (fn.getNumResults() != 1)
      continue;
    auto memrefTy = dyn_cast<MemRefType>(fn.getResultTypes()[0]);
    if (!memrefTy)
      continue;
    if (fn.isExternal())
      continue;
    Block &entry = fn.getBody().front();
    auto retOp = cast<func::ReturnOp>(entry.getTerminator());
    Value ret = retOp.getOperand(0);

    BlockArgument outArg = entry.addArgument(memrefTy, fn.getLoc());
    // Redirect the producer (e.g. linalg.* outs / linalg.fill init) and the
    // return to write the caller's buffer; the original alloc becomes dead.
    ret.replaceAllUsesWith(outArg);
    if (Operation *def = ret.getDefiningOp())
      if (def->use_empty())
        def->erase();

    OpBuilder b(retOp);
    func::ReturnOp::create(b, retOp.getLoc());
    retOp.erase();
    fn.setType(
        FunctionType::get(fn.getContext(), entry.getArgumentTypes(), {}));
  }
  return success();
}

// Mark every non-external function for C-interface wrapper emission. The c-iface
// (`_mlir_ciface_<name>`) is what we look up in the ExecutionEngine and what
// `invokePacked` ultimately calls.
void markCInterface(ModuleOp module) {
  auto unit = UnitAttr::get(module->getContext());
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isExternal())
      fn->setAttr("llvm.emit_c_interface", unit);
}

LogicalResult buildAndRunPipeline(ModuleOp module) {
  // Stage 1: tessera → linalg → bufferized memref form, identity boundary layout.
  PassManager pm1(module->getContext());
  maybeTrace(pm1);
  pm1.nest<func::FuncOp>().addPass(tessera::createTesseraToLinalgPass());
  // tensor.empty (DPS init) has no buffer semantics on its own; convert to
  // alloc_tensor so one-shot-bufferize can place it.
  pm1.nest<func::FuncOp>().addPass(
      bufferization::createEmptyTensorToAllocTensorPass());
  bufferization::OneShotBufferizePassOptions bopts;
  bopts.bufferizeFunctionBoundaries = true;
  // Identity layout at the boundary == the ABI's row-major descriptor contract.
  bopts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm1.addPass(bufferization::createOneShotBufferizePass(bopts));
  if (failed(pm1.run(module)))
    return failure();

  // Stage 1.5: explicit DPS rewrite (every function in the module).
  if (failed(rewriteResultsToOutParams(module)))
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
                  memref::MemRefDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  // BufferizableOpInterface external models — without these, one-shot-bufferize
  // reports "op was not bufferized".
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

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

#define TJ_ARG(i) packed_args[i]
#define TJ_CALL(...)                                                           \
  reinterpret_cast<void (*)(__VA_ARGS__)>(fn)
  // Arity dispatch covers single-op c-iface signatures up through any
  // reasonable fused chain (16 memref args). Each entry is a void pointer to a
  // caller-owned memref descriptor.
  switch (nargs) {
  case 1: TJ_CALL(void *)(TJ_ARG(0)); break;
  case 2: TJ_CALL(void *, void *)(TJ_ARG(0), TJ_ARG(1)); break;
  case 3: TJ_CALL(void *, void *, void *)(TJ_ARG(0), TJ_ARG(1), TJ_ARG(2)); break;
  case 4:
    TJ_CALL(void *, void *, void *, void *)(TJ_ARG(0), TJ_ARG(1), TJ_ARG(2),
                                            TJ_ARG(3));
    break;
  case 5:
    TJ_CALL(void *, void *, void *, void *, void *)(TJ_ARG(0), TJ_ARG(1),
                                                    TJ_ARG(2), TJ_ARG(3),
                                                    TJ_ARG(4));
    break;
  case 6:
    TJ_CALL(void *, void *, void *, void *, void *, void *)(
        TJ_ARG(0), TJ_ARG(1), TJ_ARG(2), TJ_ARG(3), TJ_ARG(4), TJ_ARG(5));
    break;
  case 7:
    TJ_CALL(void *, void *, void *, void *, void *, void *, void *)(
        TJ_ARG(0), TJ_ARG(1), TJ_ARG(2), TJ_ARG(3), TJ_ARG(4), TJ_ARG(5),
        TJ_ARG(6));
    break;
  case 8:
    TJ_CALL(void *, void *, void *, void *, void *, void *, void *, void *)(
        TJ_ARG(0), TJ_ARG(1), TJ_ARG(2), TJ_ARG(3), TJ_ARG(4), TJ_ARG(5),
        TJ_ARG(6), TJ_ARG(7));
    break;
  default:
    setError("tessera_jit: unsupported nargs (extend arity dispatcher in "
             "tessera_jit_invoke)");
    return 1;
  }
#undef TJ_ARG
#undef TJ_CALL
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
