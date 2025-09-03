#pragma once
#include <cstdint>
#include <string>
#include <vector>
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
