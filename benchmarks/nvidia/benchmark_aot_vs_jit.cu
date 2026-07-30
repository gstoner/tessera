// Cold NVRTC-versus-cubin module-load benchmark for NVIDIA-AOT-1.
//
// Build/run on the NVIDIA host:
//   nvcc -std=c++17 -O2 benchmarks/nvidia/benchmark_aot_vs_jit.cu -lnvrtc -lcuda -o /tmp/nvidia-aot
//   /tmp/nvidia-aot 15
//
// Every sample has a unique entry name.  This prevents NVRTC, the CUDA driver,
// and the filesystem from serving a previously compiled module, while keeping
// JIT and AOT kernels semantically identical.  Offline nvcc build time is
// reported separately and excluded from the AOT load-and-launch interval.

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const char *what, int status) {
  throw std::runtime_error(std::string(what) + " failed: " + std::to_string(status));
}
void check(CUresult status, const char *what) { if (status != CUDA_SUCCESS) fail(what, status); }
void check(nvrtcResult status, const char *what) { if (status != NVRTC_SUCCESS) fail(what, status); }

std::string source_for(int sample) {
  return "extern \"C\" __global__ void tessera_aot_probe_" + std::to_string(sample) +
         "(int* out) { if (threadIdx.x == 0) *out = 1; }\n";
}
std::string entry_for(int sample) { return "tessera_aot_probe_" + std::to_string(sample); }

double millis(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

std::vector<char> compile_nvrtc(const std::string &source, const std::string &entry) {
  nvrtcProgram program{};
  check(nvrtcCreateProgram(&program, source.c_str(), (entry + ".cu").c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");
  const char *options[] = {"--gpu-architecture=compute_120", "--std=c++17"};
  nvrtcResult result = nvrtcCompileProgram(program, 2, options);
  if (result != NVRTC_SUCCESS) {
    size_t size = 0; nvrtcGetProgramLogSize(program, &size);
    std::string log(size, '\0'); nvrtcGetProgramLog(program, log.data());
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("NVRTC compile log: " + log);
  }
  size_t size = 0; check(nvrtcGetPTXSize(program, &size), "nvrtcGetPTXSize");
  std::vector<char> ptx(size); check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
  check(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
  return ptx;
}

void load_launch_unload(const void *image, const std::string &entry) {
  CUmodule module{}; CUfunction function{};
  CUdeviceptr output{};
  check(cuModuleLoadData(&module, image), "cuModuleLoadData");
  check(cuModuleGetFunction(&function, module, entry.c_str()), "cuModuleGetFunction");
  check(cuMemAlloc(&output, sizeof(int)), "cuMemAlloc");
  void *args[] = {&output};
  check(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  int host = 0; check(cuMemcpyDtoH(&host, output, sizeof(host)), "cuMemcpyDtoH");
  check(cuMemFree(output), "cuMemFree");
  if (host != 1) throw std::runtime_error("kernel output did not match the shared oracle");
  check(cuModuleUnload(module), "cuModuleUnload");
}

int main(int argc, char **argv) try {
  const int samples = argc > 1 ? std::stoi(argv[1]) : 15;
  if (samples < 3) throw std::invalid_argument("samples must be at least 3");
  check(cuInit(0), "cuInit");
  CUdevice device{}; check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context{}; check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");

  const fs::path root = fs::temp_directory_path() / "tessera_nvidia_aot_probe";
  fs::create_directories(root);
  std::vector<double> jit_ms, aot_ms, offline_ms;
  for (int sample = 0; sample < samples; ++sample) {
    const auto source = source_for(sample), entry = entry_for(sample);
    auto begin = Clock::now();
    const auto ptx = compile_nvrtc(source, entry);
    load_launch_unload(ptx.data(), entry);
    jit_ms.push_back(millis(begin, Clock::now()));

    const fs::path cu = root / (entry + ".cu");
    const fs::path cubin = root / (entry + ".cubin");
    { std::ofstream file(cu); file << source; }
    begin = Clock::now();
    const std::string cmd = "/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120a -cubin " +
        cu.string() + " -o " + cubin.string();
    if (std::system(cmd.c_str()) != 0) throw std::runtime_error("offline nvcc build failed");
    offline_ms.push_back(millis(begin, Clock::now()));
    std::ifstream file(cubin, std::ios::binary);
    std::vector<char> image((std::istreambuf_iterator<char>(file)), {});
    begin = Clock::now();
    load_launch_unload(image.data(), entry);
    aot_ms.push_back(millis(begin, Clock::now()));
  }
  std::cout << std::fixed << std::setprecision(3)
            << "{\"samples\":" << samples
            << ",\"jit_compile_load_launch_ms\":" << median(jit_ms)
            << ",\"aot_load_launch_ms\":" << median(aot_ms)
            << ",\"aot_offline_build_ms\":" << median(offline_ms)
            << ",\"aot_saved_ms\":" << median(jit_ms) - median(aot_ms) << "}\n";
  fs::remove_all(root);
  cuDevicePrimaryCtxRelease(device);
  return 0;
} catch (const std::exception &error) {
  std::cerr << "benchmark failed: " << error.what() << '\n';
  return 1;
}
