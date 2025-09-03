#include "loader.h"
#include <iostream>
#ifdef __has_include
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    define TESSERA_HAS_HIP 1
#  endif
#endif
namespace tessera::rocm {
Loader::~Loader() {}
bool Loader::loadFile(const std::string &hsacoPath) {
#ifdef TESSERA_HAS_HIP
  hipModule_t mod; if (hipModuleLoad(&mod, hsacoPath.c_str()) != hipSuccess) return false;
  hipModule = (void*)mod; return true;
#else
  std::cout << "[stub] load " << hsacoPath << "\n"; return true;
#endif
}
bool Loader::getKernel(const std::string &name) {
#ifdef TESSERA_HAS_HIP
  hipFunction_t fn; if (hipModuleGetFunction(&fn, (hipModule_t)hipModule, name.c_str()) != hipSuccess) return false;
  hipModule = (void*)fn; return true;
#else
  std::cout << "[stub] get kernel " << name << "\n"; return true;
#endif
}
bool Loader::launch(const std::string &name, dim3 grid, dim3 block, std::vector<KernelArg> &args, size_t sharedBytes) {
#ifdef TESSERA_HAS_HIP
  hipFunction_t fn = (hipFunction_t)hipModule;
  std::vector<void*> argPtrs; argPtrs.reserve(args.size());
  for (auto &a:args) argPtrs.push_back(a.ptr);
  auto rc = hipModuleLaunchKernel(fn, grid.x,grid.y,grid.z, block.x,block.y,block.z, sharedBytes, nullptr, argPtrs.data(), nullptr);
  return rc==hipSuccess;
#else
  std::cout << "[stub] launch " << name << " grid=("<<grid.x<<","<<grid.y<<","<<grid.z<<") block=("<<block.x<<","<<block.y<<","<<block.z<<")\n";
  (void)args; return true;
#endif
}
} // namespace tessera::rocm
