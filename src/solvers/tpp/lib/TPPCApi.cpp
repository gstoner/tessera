//===- TPPCApi.cpp - In-process C ABI for the TPP pass pipeline ----------===//
//
// Glass-jaw #2 (2026-06-01) — kill the subprocess. This exposes the
// `tpp-space-time` pipeline (and any caller-supplied pass pipeline) as a
// plain C entry point so Python can run TPP lowering IN-PROCESS via
// ctypes — no `tessera-opt` subprocess, no PATH dependency.
//
// Mirrors the established Tessera pattern (the Apple GPU runtime is a
// ctypes-loaded shared library). The dialect set registered here matches
// what `tools/tessera-opt/tessera-opt.cpp` links, so any MLIR a TPP
// fixture parses under `tessera-opt` parses here too.
//
//===----------------------------------------------------------------------===//

#include "tpp/InitTPP.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// Register the TPP passes + the `tpp-space-time` alias exactly once per
// process — pass/pipeline registration is global state, so re-running it
// would assert on a duplicate registration.
void ensurePassesRegistered() {
  static std::once_flag once;
  std::call_once(once, [] {
    ::tessera::tpp::registerTPPPasses();
    ::tessera::tpp::registerTPPPipelines();
  });
}

char *dupString(const std::string &s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (out)
    std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

} // namespace

extern "C" {

// Run a pass pipeline over `mlir_in` in-process.
//
//   mlir_in  — NUL-terminated MLIR module text.
//   pipeline — pass-pipeline string (e.g.
//              "builtin.module(tpp-space-time)"); when null/empty the
//              canonical TPP pipeline is used.
//   mlir_out — on return, set to a malloc'd NUL-terminated string with
//              either the printed result module (rc==0) or an error
//              message (rc!=0). Caller frees via `tessera_tpp_free`.
//
// Returns 0 on success, 1 on a parse/pipeline/run failure, 2 on a bad
// argument.
int tessera_tpp_run_pipeline(const char *mlir_in, const char *pipeline,
                             char **mlir_out) {
  if (!mlir_in || !mlir_out)
    return 2;
  *mlir_out = nullptr;

  ensurePassesRegistered();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect>();
  ::tessera::tpp::registerTPPDialect(registry);

  mlir::MLIRContext ctx(registry);
  // TPP fixtures may carry target-IR ops outside the registered set once
  // lowered; allow them so printing the result never hard-fails.
  ctx.allowUnregisteredDialects(true);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_in, &ctx);
  if (!module) {
    *mlir_out = dupString("error: failed to parse input MLIR module");
    return 1;
  }

  mlir::PassManager pm(&ctx);
  std::string pipe =
      (pipeline && *pipeline) ? std::string(pipeline)
                              : std::string("builtin.module(tpp-space-time)");
  std::string errMsg;
  llvm::raw_string_ostream errStream(errMsg);
  if (mlir::failed(mlir::parsePassPipeline(pipe, pm, errStream))) {
    errStream.flush();
    *mlir_out = dupString("error: failed to parse pass pipeline: " + errMsg);
    return 1;
  }

  if (mlir::failed(pm.run(*module))) {
    *mlir_out = dupString("error: TPP pass pipeline failed");
    return 1;
  }

  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os);
  os.flush();
  *mlir_out = dupString(out);
  return 0;
}

// Free a string returned by `tessera_tpp_run_pipeline`.
void tessera_tpp_free(char *p) {
  if (p)
    std::free(p);
}

// Probe: returns 1 — lets the Python loader confirm the symbol resolves
// (and therefore the embedded pipeline is available) without running a
// pipeline.
int tessera_tpp_capi_available(void) { return 1; }

} // extern "C"
