#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"

#include <vector>
#include <string>

static bool which(const char* exe){
  std::string path = llvm::sys::FindProgramByName(exe).str();
  return !path.empty();
}

static int run(const std::vector<std::string>& cmd){
  llvm::SmallVector<llvm::StringRef, 16> args;
  for (auto &s : cmd) args.push_back(s);
  std::string err;
  int rc = llvm::sys::ExecuteAndWait(cmd[0], args, std::nullopt, {}, 0, 0, &err);
  if (rc != 0) {
    llvm::errs() << "Command failed: ";
    for (auto &s : cmd) llvm::errs() << s << " ";
    llvm::errs() << "\n" << err << "\n";
  }
  return rc;
}

static void writeMetadata(const std::string& path, mlir::ModuleOp m){
  llvm::json::Array kernels;
  for (auto fn : m.getOps<mlir::func::FuncOp>()) {
    if (!fn->hasAttr("tessera_rocm.kernel")) continue;
    llvm::json::Object k;
    k["name"] = fn.getName().str();
    auto wg = fn->getAttrOfType<mlir::ArrayAttr>("amdgpu-flat-work-group-size");
    if (wg && wg.size()==2) {
      k["wg_min"] = wg[0].cast<mlir::IntegerAttr>().getInt();
      k["wg_max"] = wg[1].cast<mlir::IntegerAttr>().getInt();
    }
    if (auto lds = fn->getAttrOfType<mlir::IntegerAttr>("amdgpu-lds-size"))
      k["lds_bytes"] = (int64_t)lds.getInt();
    kernels.push_back(std::move(k));
  }
  llvm::json::Object root;
  root["kernels"] = std::move(kernels);
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) return;
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
}

int main(int argc, char **argv){
  if (argc < 3) {
    llvm::errs() << "usage: tessera-rocm-emit <in.mlir> <out.hsaco> [--mcpu=gfx90a]\n";
    return 1;
  }
  std::string inMlir = argv[1], outHsaco = argv[2];
  std::string mcpu = "gfx90a";
  for (int i=3;i<argc;++i){ std::string a=argv[i]; if (a.rfind("--mcpu=",0)==0) mcpu=a.substr(7); }

  // Parse module for metadata
  llvm::SourceMgr sm;
  auto fileOrErr = mlir::openInputFile(inMlir);
  if (!fileOrErr){ llvm::errs() << "cannot open " << inMlir << "\n"; return 1; }
  sm.AddNewSourceBuffer(std::move(fileOrErr), llvm::SMLoc());
  mlir::MLIRContext ctx;
  auto module = parseSourceFile<mlir::ModuleOp>(sm, &ctx);
  if (!module){ llvm::errs() << "parse failed\n"; return 1; }

  // Derive intermediates
  std::string ll = outHsaco + ".ll";
  std::string obj = outHsaco + ".o";
  bool haveTranslate = which("mlir-translate");
  bool haveLLC = which("llc");
  bool haveLLD = which("ld.lld");
  bool haveClang = which("clang");

  if (haveTranslate && haveLLC && haveLLD) {
    if (run({"mlir-translate","--mlir-to-llvmir",inMlir,"-o",ll})) return 1;
    if (run({"llc","-filetype=obj","-mtriple=amdgcn-amd-amdhsa","-mcpu="+mcpu,ll,"-o",obj})) return 1;
    if (run({"ld.lld","-shared","-o",outHsaco,obj})) return 1;
  } else if (haveTranslate && haveClang) {
    if (run({"mlir-translate","--mlir-to-llvmir",inMlir,"-o",ll})) return 1;
    if (run({"clang","-target","amdgcn-amd-amdhsa","-mcpu="+mcpu,"-x","ir",ll,"-o",outHsaco})) return 1;
  } else {
    llvm::errs() << "No suitable toolchain found (need mlir-translate+llc+ld.lld or clang)\n";
    return 1;
  }

  // Write metadata.json alongside HSACO
  writeMetadata(outHsaco + ".metadata.json", *module);
  llvm::outs() << "Wrote " << outHsaco << " and metadata\n";
  return 0;
}
