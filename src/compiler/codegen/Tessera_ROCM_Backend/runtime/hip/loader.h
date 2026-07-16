#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__has_include) && __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define TESSERA_HAS_HIP 1
#else
struct dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};
#endif

namespace tessera::rocm {
struct KernelArg { void *ptr; size_t size; };
class Loader {
public:
  Loader() = default; ~Loader();
  bool loadFile(const std::string &hsacoPath);
  bool getKernel(const std::string &name);
  bool launch(const std::string &name, dim3 grid, dim3 block, std::vector<KernelArg> &args, size_t sharedBytes = 0);
private: void *hipModule = nullptr;
};
} // namespace tessera::rocm
