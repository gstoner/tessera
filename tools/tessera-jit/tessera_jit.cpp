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
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
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
#include "llvm/Config/llvm-config.h"
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
#include <cstdio>
#include <cstring>
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

// Boundary signature captured at compile time (RUNTIME_ABI_SPEC §12.6).
// The lowered module no longer carries tensor types, so the shapes each
// function was compiled against are recorded here, from the parsed module,
// before the pipeline runs. `tessera_jit_invoke` validates descriptors
// against this before dispatch: an argument-count or static-extent mismatch
// is a caller bug that would otherwise read/write out of bounds inside the
// generated code (the identity-layout ABI bakes static extents into the
// indexing math — the descriptor's sizes are NOT consulted for them).
struct ArgSig {
  bool isRankedTensor = false;
  SmallVector<int64_t> dims;  // ShapedType::kDynamic for '?'
  std::string typeText;       // element type for tensors, full type otherwise
};

struct FuncSig {
  // c-iface argument order after the DPS rewrite: inputs..., results... .
  SmallVector<ArgSig> cifaceArgs;
  unsigned numInputs = 0;
  unsigned numResults = 0;
  // All results are ranked tensors (or there are none) — the only shape the
  // DPS rewrite + void-return ffi dispatch in tessera_jit_invoke supports.
  bool dpsCompatible = true;
  std::string rendered;  // "in;in|out" — see renderSignatures()
};

struct JitModule {
  std::unique_ptr<MLIRContext> ctx;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ExecutionEngine> engine;
  llvm::StringMap<FuncSig> signatures;
};

// Capture every non-external function's boundary signature from the parsed
// (pre-lowering) module. Must run before the pipeline: bufferization/DPS
// erase the tensor-level function type.
void captureSignatures(ModuleOp module, llvm::StringMap<FuncSig> &sigs) {
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (fn.isExternal())
      continue;
    FuncSig sig;
    sig.numInputs = fn.getNumArguments();
    sig.numResults = fn.getNumResults();
    sig.dpsCompatible = llvm::all_of(fn.getResultTypes(), [](Type t) {
      return isa<RankedTensorType>(t);
    });
    auto addArg = [&](Type t) {
      ArgSig a;
      llvm::raw_string_ostream os(a.typeText);
      if (auto rt = dyn_cast<RankedTensorType>(t)) {
        a.isRankedTensor = true;
        a.dims.assign(rt.getShape().begin(), rt.getShape().end());
        os << rt.getElementType();
      } else {
        os << t;
      }
      sig.cifaceArgs.push_back(std::move(a));
    };
    for (Type t : fn.getArgumentTypes())
      addArg(t);
    if (sig.dpsCompatible)
      for (Type t : fn.getResultTypes())
        addArg(t);
    // Rendered form for the tessera_jit_signature ABI:
    //   inputs ';'-joined, '|', results ';'-joined
    // with each ranked tensor as tensor<AxBx...xELEM> ('?' for dynamic) and
    // any other type verbatim. ';' cannot appear in these type spellings.
    std::string txt;
    llvm::raw_string_ostream os(txt);
    auto render = [&](const ArgSig &a) {
      if (!a.isRankedTensor) {
        os << a.typeText;
        return;
      }
      os << "tensor<";
      for (int64_t d : a.dims) {
        if (ShapedType::isDynamic(d))
          os << "?";
        else
          os << d;
        os << "x";
      }
      os << a.typeText << ">";
    };
    for (unsigned i = 0; i < sig.numInputs; ++i) {
      if (i)
        os << ";";
      render(sig.cifaceArgs[i]);
    }
    os << "|";
    if (!sig.dpsCompatible) {
      // Function has non-tensor results: the DPS rewrite leaves it returning
      // values, which the void-return invoke dispatch cannot call. Mark it so
      // the Python layer fails closed instead of mis-parsing "no results".
      os << "!nondps";
    } else {
      for (unsigned i = sig.numInputs; i < sig.cifaceArgs.size(); ++i) {
        if (i != sig.numInputs)
          os << ";";
        render(sig.cifaceArgs[i]);
      }
    }
    sig.rendered = std::move(txt);
    sigs[fn.getSymName()] = std::move(sig);
  }
}

// In-process implementation of the `memrefCopy` C runner-utils helper
// (CRunnerUtils ABI: elemSize + two unranked-memref pointers, each
// {rank, descriptor*} with descriptor {allocated, aligned, offset,
// sizes[rank], strides[rank]}). Generic strided element-wise copy; the
// source authority is mlir/lib/ExecutionEngine/CRunnerUtils.cpp. Kept here
// so the engine never loads libmlir_c_runner_utils.so — see the
// registerSymbols call in tessera_jit_compile for why that dlopen is
// process-fatal alongside HIP.
struct TesseraUnrankedMemRef {
  int64_t rank;
  void *descriptor;
};

void tesseraMemrefCopy(int64_t elemSize, TesseraUnrankedMemRef *srcU,
                       TesseraUnrankedMemRef *dstU) {
  struct DescHead {
    char *allocated;
    char *aligned;
    int64_t offset;
    // int64_t sizes[rank]; int64_t strides[rank];
  };
  const int64_t rank = srcU->rank;
  auto *src = static_cast<DescHead *>(srcU->descriptor);
  auto *dst = static_cast<DescHead *>(dstU->descriptor);
  const int64_t *srcSizes = reinterpret_cast<const int64_t *>(src + 1);
  const int64_t *srcStrides = srcSizes + rank;
  const int64_t *dstSizes = reinterpret_cast<const int64_t *>(dst + 1);
  const int64_t *dstStrides = dstSizes + rank;
  (void)dstSizes;  // shapes must match; iteration uses the source's.
  char *srcBase = src->aligned + src->offset * elemSize;
  char *dstBase = dst->aligned + dst->offset * elemSize;
  if (rank == 0) {
    std::memcpy(dstBase, srcBase, static_cast<size_t>(elemSize));
    return;
  }
  int64_t total = 1;
  for (int64_t r = 0; r < rank; ++r)
    total *= srcSizes[r];
  if (total == 0)
    return;
  SmallVector<int64_t, 6> idx(static_cast<size_t>(rank), 0);
  for (int64_t e = 0; e < total; ++e) {
    int64_t srcOff = 0, dstOff = 0;
    for (int64_t r = 0; r < rank; ++r) {
      srcOff += idx[static_cast<size_t>(r)] * srcStrides[r];
      dstOff += idx[static_cast<size_t>(r)] * dstStrides[r];
    }
    std::memcpy(dstBase + dstOff * elemSize, srcBase + srcOff * elemSize,
                static_cast<size_t>(elemSize));
    for (int64_t r = rank - 1; r >= 0; --r) {
      if (++idx[static_cast<size_t>(r)] < srcSizes[r])
        break;
      idx[static_cast<size_t>(r)] = 0;
    }
  }
}

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
// Cache-level blocking above the register tiles (2026-08-23 perf loop).
// The original single-level [8,16,16] register tiling re-streams the B
// panels from L3/DRAM once the matrices outgrow L2 — measured 106.6
// GFLOP/s at n=256 decaying to 44.4 at n=1024 on Zen 5 (48 KB L1D, 1 MB
// L2, single-threaded lane). An outer tile_using_for keeps an
// (MC x KC) A-block + (KC x NC) B-block + (MC x NC) C-block L2-resident
// while the register kernel walks them. Sizes are tunable via
// TESSERA_JIT_CACHE_TILES="MC,NC,KC" (0,0,0 disables the outer level);
// the default was picked by an on-host sweep. The two-level script can
// fail where the single-level one succeeds (non-divisible extents make
// the second tiling's slices dynamic), so tileAndVectorizeLinalg tries
// two-level first and falls back to the single-level script, then to
// scalar — never a compile failure.
struct CacheTileSizes {
  int64_t mc, nc, kc;
  // tile_using_for treats a 0 tile size as "leave this dim untiled", so any
  // nonzero component makes an outer level meaningful (K-only / MN-only
  // blocking are valid configs).
  bool enabled() const { return mc > 0 || nc > 0 || kc > 0; }
};

static CacheTileSizes cacheTileSizes() {
  // Default picked by on-host sweep (Strix Halo / Zen 5, 2026-08-23), full
  // matrix in the x86 todo under JIT-CACHE-BLOCK-2026-08-23. K-only
  // chunking at the register k-tile (16) won decisively: it hoists the
  // k-chunk loop outermost, so a (16 x N) B row-panel stays cache-resident
  // across the whole (i,j) tile sweep — measured 161/144/139 GFLOP/s at
  // n=512/1024/2048 vs 77/45/43 single-level, flat instead of decaying.
  // Tiling M or N at the cache level made things WORSE (~18-35 GFLOP/s):
  // the strided cache-tile views defeat the inner kernel's vector loads.
  CacheTileSizes t{0, 0, 16};
  if (const char *e = ::getenv("TESSERA_JIT_CACHE_TILES")) {
    long long mc = 0, nc = 0, kc = 0;
    if (std::sscanf(e, "%lld,%lld,%lld", &mc, &nc, &kc) == 3) {
      t.mc = mc;
      t.nc = nc;
      t.kc = kc;
    }
  }
  return t;
}

static std::string tileVectorizeTransform(bool withCacheLevel) {
  CacheTileSizes ct = cacheTileSizes();
  std::string cacheStage;
  std::string regSource = "%mm";
  if (withCacheLevel && ct.enabled()) {
    // tile_using_for produces one loop result per NONZERO tile size; the
    // result list must match or the transform module fails to verify (and
    // the lane silently falls back to single-level).
    int nLoops = (ct.mc > 0) + (ct.nc > 0) + (ct.kc > 0);
    std::string results = "%mmc";
    std::string types = "!transform.any_op";
    for (int i = 0; i < nLoops; ++i) {
      results += ", %cl" + std::to_string(i);
      types += ", !transform.any_op";
    }
    cacheStage = "    " + results +
                 " = transform.structured.tile_using_for %mm tile_sizes [" +
                 std::to_string(ct.mc) + ", " + std::to_string(ct.nc) + ", " +
                 std::to_string(ct.kc) + "]\n        : (!transform.any_op) -> (" +
                 types + ")\n";
    regSource = "%mmc";
  }
  return std::string(R"MLIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
)MLIR") + cacheStage + std::string("    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for ") + regSource + std::string(R"MLIR( tile_sizes [8, 16, 16]
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
)MLIR");
}

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

// The LLVM_VERSION_MAJOR == 23 carve-out that used to live here (recorded
// reason: "one-shot bufferization aborts while querying
// SubsetInsertionOpInterface on the transform-vectorized tensor IR") was
// removed 2026-08-23: the abort no longer reproduces on LLVM 23 on this
// codebase — verified on the AVX-512 host across the full JIT packet and the
// matmul(+add) program family, with the tensor SubsetOpInterface external
// models now registered at engine setup (their absence is MLIR's
// abort-not-failure path when bufferization queries an insert_slice). Note
// the gate had never been re-checked against a real vectorized run: every
// fleet box is LLVM 23, so the gate itself kept the vectorized path
// unexercised everywhere (the Decision #19 standing-lesson pattern). The
// lane is additionally fail-safe now: the transform runs on a clone and any
// failure falls back to the scalar pipeline (see stage 1b).

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

// Undo the vectorizer's whole-tile rewrite of the CACHE-LEVEL
// tensor.insert_slice. vectorize_children_and_apply_patterns unconditionally
// vectorizes insert_slice into a full-tile transfer_read + transfer_write
// pair; at the (MC x NC) cache-tile size that materializes a 64 KB
// vector<128x128xf32> SSA value, which ConvertVectorToSCF stages through an
// `array<128 x vector<128xf32>>` alloca inside the outer loops — the
// two-level lane crashed at n>=512 on the resulting stack growth, and even
// when it survives, the pair forces a real per-iteration tile copy that
// tensor.insert_slice + one-shot-bufferize would have made a no-op
// (in-place). Restore the insert_slice form for any full-tile pair at or
// above 8 KB; the register-level 8x16 pairs (512 B) stay vectorized, where
// they fold with the inner kernel's transfers.
static void demoteLargeFullTileTransfers(ModuleOp module) {
  SmallVector<vector::TransferWriteOp> writes;
  module.walk([&](vector::TransferWriteOp w) { writes.push_back(w); });
  for (vector::TransferWriteOp w : writes) {
    auto r = w.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!r || !r->hasOneUse())
      continue;
    auto srcTy = dyn_cast<RankedTensorType>(r.getBase().getType());
    auto dstTy = dyn_cast<RankedTensorType>(w.getBase().getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape())
      continue;
    VectorType vecTy = r.getVectorType();
    if (vecTy.getShape() != srcTy.getShape())
      continue;
    int64_t bytes =
        vecTy.getNumElements() * vecTy.getElementTypeBitWidth() / 8;
    if (bytes < 8192)
      continue;
    if (!r.getPermutationMap().isIdentity() ||
        !w.getPermutationMap().isIdentity())
      continue;
    // A masked transfer suppresses lanes the insert_slice form would copy
    // — never demote those. And require in_bounds to be EXPLICIT for every
    // dimension and all-true: an absent in_bounds yields an empty
    // getInBoundsValues(), which a bare any_of(!b) accepts vacuously
    // without proving anything about the write.
    if (r.getMask() || w.getMask())
      continue;
    auto allExplicitlyInBounds = [](auto op, int64_t rank) {
      SmallVector<bool> ib = op.getInBoundsValues();
      return static_cast<int64_t>(ib.size()) == rank &&
             llvm::all_of(ib, [](bool b) { return b; });
    };
    if (!allExplicitlyInBounds(r, vecTy.getRank()) ||
        !allExplicitlyInBounds(w, vecTy.getRank()))
      continue;
    bool zeroReadIdx = llvm::all_of(r.getIndices(), [](Value v) {
      auto c = v.getDefiningOp<arith::ConstantIndexOp>();
      return c && c.value() == 0;
    });
    if (!zeroReadIdx)
      continue;
    OpBuilder b(w);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (Value idx : w.getIndices())
      offsets.push_back(idx);
    for (int64_t d : srcTy.getShape()) {
      sizes.push_back(b.getIndexAttr(d));
      strides.push_back(b.getIndexAttr(1));
    }
    // strides was filled once per dim above alongside sizes.
    auto ins = tensor::InsertSliceOp::create(
        b, w.getLoc(), r.getBase(), w.getBase(), offsets, sizes, strides);
    w.getOperation()->replaceAllUsesWith(
        ValueRange{ins.getResult()});
    w.erase();
    if (r->use_empty())
      r->erase();
  }
}

static LogicalResult tileAndVectorizeLinalg(ModuleOp module,
                                            bool withCacheLevel) {
  MLIRContext *ctx = module.getContext();
  // Parse the transform sequence in the payload's context (so the transform
  // dialect + extensions resolve against the same registry).
  OwningOpRef<ModuleOp> transformModule = parseSourceString<ModuleOp>(
      tileVectorizeTransform(withCacheLevel), ctx);
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

// One-shot bufferization cannot consume tensor-valued pointwise operations
// directly. The module-scoped elementwise-to-linalg pass above is intended to
// close that entire mathematical family (arith add/sub/mul/div, comparisons,
// select, min/max, casts, and tensor math), including operations nested in
// control flow. Keep an explicit postcondition here so a newly introduced or
// newly unsupported elementwise op fails at the owning boundary, before the
// much less actionable generic bufferization diagnostic.
static LogicalResult rejectResidualTensorElementwiseOps(ModuleOp module) {
  Operation *residual = nullptr;
  module.walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::Elementwise>())
      return WalkResult::advance();
    if (!llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); }))
      return WalkResult::advance();
    residual = op;
    return WalkResult::interrupt();
  });
  if (!residual)
    return success();

  std::string message =
      "tessera_jit: unsupported residual tensor elementwise operation '" +
      residual->getName().getStringRef().str() +
      "' after elementwise-to-linalg conversion";
  setError(message);
  residual->emitError(message);
  return failure();
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
  // Elementwise arith/math ops ON TENSORS (e.g. the paired autodiff pass's
  // cotangent accumulation `arith.addf : tensor<...>`) have no bufferization
  // interface of their own; rewrite them to linalg.generic first so
  // one-shot-bufferize can consume them (W4 x86 state-machine row).
  // This is a module pass (it rewrites elementwise ops in nested control-flow
  // regions too), so adding it under func silently prevents it from running.
  // In particular, paired state-machine backward functions carry tensor
  // arith.addf and arith.select under scf.if/scf.for; both must become
  // linalg.generic before one-shot bufferization.
  pm1a.addPass(createConvertElementwiseToLinalgPass());
  if (failed(pm1a.run(module)))
    return failure();
  if (failed(rejectResidualTensorElementwiseOps(module)))
    return failure();

  // Stage 1b (opt-in): tile + vectorize on tensors, before bufferization.
  // Engages only for modules that contain a linalg.matmul (this is the GEMM
  // lane; the transform script's tile_using_for errors on an empty matmul
  // match, which used to FAIL the whole compile for every non-matmul module
  // when the env var was set — measured 114 packet failures). Best-effort by
  // construction: the transform runs on a CLONE, and only a fully successful
  // transform replaces the module — any failure (unsupported op mix, dynamic
  // shapes the vectorizer rejects, …) falls back to the always-correct
  // scalar pipeline instead of failing or leaving half-transformed IR.
  bool vectorized = false;
  if (::getenv("TESSERA_JIT_VECTORIZE") && withinVectorizeEnvelope(module)) {
    bool hasMatmul = false;
    module.walk([&](linalg::MatmulOp) { hasMatmul = true; });
    if (hasMatmul) {
      // Two-level (cache + register) first; where its second tiling cannot
      // apply (non-divisible extents produce dynamic slices the vectorizer
      // rejects), retry with register tiles only — yesterday's behavior.
      for (bool withCacheLevel : {true, false}) {
        OwningOpRef<ModuleOp> candidate(cast<ModuleOp>(module->clone()));
        if (failed(tileAndVectorizeLinalg(*candidate, withCacheLevel)))
          continue;
        // Acceptance check: vectorize_children_and_apply_patterns succeeds
        // even when it vectorized NOTHING (it only fails on catastrophic
        // pattern breakage), so a config whose tiling produced dynamic
        // slices (e.g. a non-dividing KC) "succeeds" with the matmul left
        // as scalar linalg — measured ~1 GFLOP/s. A candidate counts only
        // if no linalg.matmul survived.
        bool residualMatmul = false;
        candidate->walk([&](linalg::MatmulOp) { residualMatmul = true; });
        if (residualMatmul)
          continue;
        demoteLargeFullTileTransfers(*candidate);
        module.getBodyRegion().takeBody(candidate->getBodyRegion());
        vectorized = true;
        break;
      }
    }
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
  //
  // Narrowed 2026-08-23 (x86 math-correctness audit): `fast` also carried
  // nnan|ninf|nsz|arcp|afn, which are SEMANTIC bits, and the stamp applies to
  // every float op in the module — including user-visible elementwise lanes
  // with IEEE expectations, not just the GEMM reduction body. Two measured
  // consequences on AVX-512: `arcp` rewrote elementwise x/y into
  // x*(1/y) (1-ulp divergence from correctly-rounded division vs numpy), and
  // nnan/ninf made NaN/Inf inputs (e.g. -inf attention-mask biases) poison —
  // latent today, legal to break tomorrow. reassoc|contract is exactly what
  // reduction vectorization + FMA formation need; GEMM throughput is
  // unchanged (measured 256/512 f32 on Strix Halo, scalar lane) and the
  // accumulation-order tolerance (rtol≈1e-4) is still the GEMM contract.
  auto fmVec = arith::FastMathFlagsAttr::get(
      module.getContext(),
      arith::FastMathFlags::reassoc | arith::FastMathFlags::contract);
  module.walk([&](Operation *op) {
    // Do not stamp maximum/minimum/maxnum/minnum: unlike GEMM arithmetic,
    // their NaN-propagation and signed-zero choice is the operation contract.
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::NegFOp>(op))
      op->setAttr("fastmath", fmVec);
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
  {
    // full-unroll: lower each (small, <8KB — see demoteLargeFullTileTransfers)
    // n-D transfer into unrolled rank-1 transfers on vector values instead of
    // staging through a memref.alloca. The staging allocas land INSIDE the
    // loop nest (scf.for is not an AutomaticAllocationScope), so after
    // scf-to-cf they become per-iteration llvm.alloca — the two-level cache
    // tiling executed ~65k inner iterations at n=512 and overflowed the 8 MB
    // stack at INVOKE time (the single-level lane had the same leak below
    // crash threshold). Unrolled form also avoids the alloca round-trip on
    // the strided cache-tile views, which was the 106 -> 23 GFLOP/s cliff.
    VectorTransferToSCFOptions vopts;
    vopts.enableFullUnroll(true);
    pm2.addPass(createConvertVectorToSCFPass(vopts));
  }
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
  g_lastError.clear();
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
  // SubsetOpInterface external models on tensor.insert_slice & friends —
  // one-shot-bufferize queries SubsetInsertionOpInterface on the
  // tile+vectorize lane's tensor IR, and an unregistered model is a hard
  // abort (not a pass failure). This missing registration was the actual
  // cause of the "MLIR 23 aborts in one-shot bufferization" toolchain gate
  // on the vectorize lane (root-caused 2026-08-23).
  tensor::registerSubsetOpInterfaceExternalModels(registry);
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

  // Record boundary signatures from the tensor-level module — the pipeline
  // below erases them, and invoke validation needs them (§12.6).
  captureSignatures(*jm->module, jm->signatures);

  markCInterface(*jm->module);

  if (failed(buildAndRunPipeline(*jm->module))) {
    if (g_lastError.empty())
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
  // ExecutionEngineOptions stores this as a non-owning function_ref in MLIR
  // 23. Keep the owning std::function alive through ExecutionEngine::create;
  // assigning the temporary directly leaves a dangling callback and can crash
  // on the first JIT compilation.
  auto optimizingTransformer = makeOptimizingTransformer(/*optLevel=*/3,
                                                          /*sizeLevel=*/0,
                                                          /*targetMachine=*/hostTM.get());
  opts.transformer = optimizingTransformer;
  auto expectedEngine = ExecutionEngine::create(*jm->module, opts);
  if (!expectedEngine) {
    setError("tessera_jit: ExecutionEngine::create failed");
    return nullptr;
  }
  jm->engine = std::move(*expectedEngine);
  // The vectorize lane's DPS out-param copy can lower memref.copy to the
  // generic `memrefCopy` runtime helper (between different-layout memrefs).
  // Resolve it to the IN-PROCESS implementation below instead of dlopening
  // libmlir_c_runner_utils: that shared library links the DYNAMIC
  // libLLVM.so, and loading it beside this library's statically linked
  // LLVM made a later dlopen of libamdhip64 (whose comgr embeds a third
  // LLVM) segfault in constructor/interposition cross-talk — the ROCm
  // state-machine tests died the moment they loaded HIP after a vectorized
  // compile (root-caused 2026-08-23). Registered unconditionally: it is
  // dead weight when no module references it, and removes the old
  // Homebrew-pathed TESSERA_MLIR_RUNNER_UTILS dlopen entirely.
  jm->engine->registerSymbols(
      [](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap map;
        map[interner("memrefCopy")] = {
            llvm::orc::ExecutorAddr::fromPtr(&tesseraMemrefCopy),
            llvm::JITSymbolFlags::Exported};
        return map;
      });
  g_compiles.fetch_add(1, std::memory_order_relaxed);
  return jm.release();
}

int64_t tessera_jit_compile_count(void) {
  return g_compiles.load(std::memory_order_relaxed);
}

// Boundary signature query (§12.6): returns the signature the named function
// was compiled against, as `inputs|results` with ';'-separated MLIR types
// (dynamic extents spelled '?'; a non-DPS-callable function renders its
// results section as `!nondps`). Returns nullptr when the handle is invalid
// or the function does not exist in the compiled module. The returned pointer
// is valid until the next tessera_jit_* call on this thread.
const char *tessera_jit_signature(void *handle, const char *name) {
  thread_local std::string storage;
  auto *jm = static_cast<JitModule *>(handle);
  if (!jm || !name)
    return nullptr;
  auto it = jm->signatures.find(name);
  if (it == jm->signatures.end())
    return nullptr;
  storage = it->second.rendered;
  return storage.c_str();
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

  // Boundary validation (§12.6): the generated code bakes static extents into
  // its indexing math and never consults the descriptor's sizes for them, so
  // an arity or extent mismatch here is guaranteed out-of-bounds access —
  // fail closed before dispatch. The Python layer performs the richer check
  // (rank + dtype); this is the memory-safety backstop for ALL callers.
  auto sigIt = jm->signatures.find(name);
  if (sigIt != jm->signatures.end()) {
    const FuncSig &sig = sigIt->second;
    if (!sig.dpsCompatible) {
      setError(std::string("tessera_jit: function '") + name +
               "' has non-tensor results; not callable through the DPS "
               "void-return invoke ABI");
      return 1;
    }
    if (static_cast<size_t>(nargs) != sig.cifaceArgs.size()) {
      setError(std::string("tessera_jit: function '") + name + "' expects " +
               std::to_string(sig.cifaceArgs.size()) +
               " arguments (inputs + DPS outs), got " + std::to_string(nargs));
      return 1;
    }
    for (int i = 0; i < nargs; ++i) {
      const ArgSig &a = sig.cifaceArgs[static_cast<size_t>(i)];
      if (!a.isRankedTensor)
        continue;
      if (!packed_args[i]) {
        setError(std::string("tessera_jit: function '") + name + "' argument " +
                 std::to_string(i) + " is null");
        return 1;
      }
      // Standard memref descriptor: {T* allocated; T* aligned; i64 offset;
      // i64 sizes[rank]; i64 strides[rank]}. Rank comes from the compiled
      // signature; the Python layer guarantees the caller built a descriptor
      // of that rank (a foreign caller passing a lower-rank descriptor gets a
      // detectable size mismatch rather than an out-of-bounds kernel write).
      const auto *sizes = reinterpret_cast<const int64_t *>(
          static_cast<const char *>(packed_args[i]) + 2 * sizeof(void *) +
          sizeof(int64_t));
      for (size_t r = 0; r < a.dims.size(); ++r) {
        int64_t expected = a.dims[r];
        if (ShapedType::isDynamic(expected))
          continue;
        if (sizes[r] != expected) {
          setError(std::string("tessera_jit: function '") + name +
                   "' argument " + std::to_string(i) + " dim " +
                   std::to_string(r) + " expects extent " +
                   std::to_string(expected) + ", got " +
                   std::to_string(sizes[r]) +
                   " (compiled signature: " + sig.rendered + ")");
          return 1;
        }
      }
    }
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
