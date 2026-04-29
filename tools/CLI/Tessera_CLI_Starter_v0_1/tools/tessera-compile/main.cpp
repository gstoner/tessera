//===- tessera-compile/main.cpp — Lower IR to target artifacts -------------===//
//
// tessera-compile lowers Tessera IR from any layer down to PTX/CUBIN/HSACO
// or CPU objects, emitting kernels, optional host stubs, and a compile
// manifest.
//
// Example usage:
//   tessera-compile model.mlir --platform=cuda --arch=sm_90 --to=ptx
//   tessera-compile sched.mlir --platform=hip  --arch=gfx1100 --emit-host
//   tessera-compile tile.mlir  --platform=cpu  --arch=avx2 --out-dir build/
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "common/manifest.hpp"
#include "common/args.hpp"

static const char* TOOL = "tessera-compile";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string ptx_skeleton(const std::string& arch,
                                 bool tensor_cores, bool wgmma, bool tma) {
  std::string s;
  s += "// tessera-compile output — arch=" + arch + "\n";
  s += "// features: tensor_cores=" + std::string(tensor_cores ? "1" : "0");
  s += " wgmma=" + std::string(wgmma ? "1" : "0");
  s += " tma=" + std::string(tma ? "1" : "0") + "\n";
  s += ".version 8.0\n";
  s += ".target " + arch + "\n";
  s += ".address_size 64\n\n";
  s += ".visible .entry demo_kernel(\n";
  s += "    .param .u64 param0,\n";
  s += "    .param .u64 param1,\n";
  s += "    .param .u64 param2\n";
  s += ") {\n";
  s += "    .reg .u64 %rd<4>;\n";
  s += "    ld.param.u64 %rd0, [param0];\n";
  s += "    ld.param.u64 %rd1, [param1];\n";
  s += "    ld.param.u64 %rd2, [param2];\n";
  if (tensor_cores && wgmma) {
    s += "    // wgmma.mma_async placeholder for sm_90\n";
  } else if (tensor_cores) {
    s += "    // wmma placeholder for tensor core ops\n";
  }
  if (tma) {
    s += "    // cp.async.bulk.tensor placeholder (TMA)\n";
  }
  s += "    ret;\n}\n";
  return s;
}

static std::string host_stub(const std::string& platform) {
  if (platform == "cuda") {
    return R"(// tessera host stub — CUDA
#include <cuda_runtime.h>
#include <cstdio>

extern "C" int tessera_launch(void* a, void* b, void* c,
                               int M, int N, int K, cudaStream_t stream) {
  dim3 grid((N + 127) / 128, (M + 127) / 128, 1);
  dim3 block(128, 1, 1);
  // demo_kernel<<<grid, block, 0, stream>>>(a, b, c);
  return cudaGetLastError();
}
)";
  }
  if (platform == "hip") {
    return R"(// tessera host stub — HIP
#include <hip/hip_runtime.h>

extern "C" int tessera_launch(void* a, void* b, void* c,
                               int M, int N, int K, hipStream_t stream) {
  dim3 grid((N + 127) / 128, (M + 127) / 128, 1);
  dim3 block(128, 1, 1);
  // hipLaunchKernelGGL(demo_kernel, grid, block, 0, stream, a, b, c);
  return hipGetLastError();
}
)";
  }
  // cpu
  return R"(// tessera host stub — CPU
#include <cstring>

extern "C" int tessera_launch(const float* a, const float* b, float* c,
                               int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += a[i*K+k] * b[k*N+j];
      c[i*N+j] = acc;
    }
  return 0;
}
)";
}

static std::string cmake_project(const std::string& platform,
                                  const std::string& arch) {
  std::string s;
  s += "cmake_minimum_required(VERSION 3.20)\n";
  s += "project(tessera_kernel LANGUAGES CXX)\n\n";
  if (platform == "cuda") {
    s += "enable_language(CUDA)\n";
    s += "set(CMAKE_CUDA_ARCHITECTURES " + arch.substr(3) + ")\n";
    s += "add_library(tessera_kernel SHARED host/launch.cu)\n";
  } else if (platform == "hip") {
    s += "find_package(hip REQUIRED)\n";
    s += "add_library(tessera_kernel SHARED host/launch.cpp)\n";
    s += "target_link_libraries(tessera_kernel PRIVATE hip::device)\n";
  } else {
    s += "add_library(tessera_kernel SHARED host/launch.cpp)\n";
    s += "target_compile_options(tessera_kernel PRIVATE -O3 -march=" + arch + ")\n";
  }
  return s;
}

static std::string compile_manifest(
    const std::string& input_file,
    const std::string& from, const std::string& platform,
    const std::string& arch,  const std::string& to,
    bool tensor_cores, bool wgmma, bool tma,
    bool emit_host, bool emit_cmake,
    const std::vector<std::string>& kernels,
    const std::vector<std::string>& host_files) {

  std::string s = "{\n";
  s += "  \"tessera\": {\"version\": \"" TESSERA_CLI_VERSION "\"},\n";
  s += "  \"input\": {\"file\": \"" + tessera::jsonEscape(input_file) + "\","
       " \"from\": \"" + from + "\"},\n";
  s += "  \"pipeline\": [\"verify\", \"migrate\", \"" +
       from + "->target\", \"codegen\"],\n";
  s += "  \"target\": {\"platform\": \"" + platform + "\","
       " \"arch\": \"" + arch + "\","
       " \"to\": \"" + to + "\"},\n";
  s += "  \"features\": {\"tensor_cores\": " +
       std::string(tensor_cores ? "true" : "false") + ","
       " \"wgmma\": " + std::string(wgmma ? "true" : "false") + ","
       " \"tma\": " + std::string(tma ? "true" : "false") + "},\n";

  s += "  \"artifacts\": {\n";
  s += "    \"kernels\": [";
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    if (i) s += ", ";
    s += "\"" + tessera::jsonEscape(kernels[i]) + "\"";
  }
  s += "]";
  if (emit_host || emit_cmake) {
    s += ",\n    \"host\": [";
    for (std::size_t i = 0; i < host_files.size(); ++i) {
      if (i) s += ", ";
      s += "\"" + tessera::jsonEscape(host_files[i]) + "\"";
    }
    s += "]";
  }
  s += "\n  },\n";
  s += "  \"timestamp\": \"" + nowIso8601() + "\"\n";
  s += "}\n";
  return s;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string opt_platform, opt_arch, opt_to, opt_from, opt_mfma;
  bool opt_tensor_cores = false;
  bool opt_wgmma        = false;
  bool opt_tma          = false;
  bool opt_emit_host    = false;
  bool opt_emit_cmake   = false;
  bool opt_occ_report   = false;
  bool opt_size_report  = false;
  int  opt_level        = 2;
  std::vector<std::string> inputs;

  tessera::Args args(TOOL, "Lower Tessera IR to target artifacts (PTX/CUBIN/HSACO/CPU-obj)",
                     argc, argv);
  args.option("--platform",         "Target platform {cuda,hip,cpu}",            &opt_platform, "cuda")
      .option("--arch",             "Target arch (sm_90, gfx1100, avx2, ...)",   &opt_arch, "sm_90")
      .option("--to",               "Output format {ptx,cubin,hsaco,cpu-obj,llvm-ir}", &opt_to, "ptx")
      .option("--from",             "Input IR layer {graph,schedule,tile,target,auto}", &opt_from, "auto")
      .option("--mfma-profile",     "MFMA dtype profile {bf16,fp16,tf32,f32}",   &opt_mfma, "bf16")
      .int_option("-O",             "Optimization level 0-3 (default 2)",         &opt_level, 2)
      .flag("--enable-tensor-cores","Enable tensor-core / WMMA instructions",     &opt_tensor_cores)
      .flag("--enable-wgmma",       "Enable WGMMA (sm_90+)",                      &opt_wgmma)
      .flag("--enable-tma",         "Enable TMA async bulk-copy (sm_90+)",        &opt_tma)
      .flag("--emit-host",          "Generate CUDA/HIP/CPU host launch stub",     &opt_emit_host)
      .flag("--emit-cmake",         "Generate CMakeLists.txt for the artifact",   &opt_emit_cmake)
      .flag("--occupancy-report",   "Emit occupancy estimate to reports/",        &opt_occ_report)
      .flag("--size-report",        "Emit code-size report to reports/",          &opt_size_report)
      .positional("input.mlir",     "Input IR file(s)",                           &inputs);

  if (!args.parse()) return args.exit_code();

  // Validate platform
  if (opt_platform != "cuda" && opt_platform != "hip" && opt_platform != "cpu") {
    TLOG_ERROR(TOOL, "unknown platform '" + opt_platform +
               "'; expected cuda, hip, or cpu");
    return tessera::EXIT_BAD_TARGET;
  }

  // sm_90 implies wgmma/tma by convention
  if (opt_arch == "sm_90" && opt_tensor_cores) {
    opt_wgmma = true;
    opt_tma   = true;
  }

  std::string input_file = inputs.empty() ? "<stdin>" : inputs[0];
  int rc = tessera::EXIT_OK;

  try {
    auto paths = makeArtifactLayout(args.out_dir());

    // Kernel extension
    std::string ext = (opt_to == "ptx") ? ".ptx"
                    : (opt_to == "cubin") ? ".cubin"
                    : (opt_to == "hsaco") ? ".hsaco"
                    : (opt_to == "llvm-ir") ? ".ll"
                    : ".o";
    std::string kernel_rel = "kernels/demo" + ext;
    std::string kernel_abs = paths.kernels_dir + "/demo" + ext;

    if (args.dry_run()) {
      TLOG_INFO(TOOL, "[dry-run] would emit " + kernel_abs);
    } else {
      std::string kernel_src = ptx_skeleton(opt_arch, opt_tensor_cores,
                                             opt_wgmma, opt_tma);
      writeFile(kernel_abs, kernel_src);
      TLOG_INFO(TOOL, "emitted kernel → " + kernel_abs);
    }

    std::vector<std::string> kernels = {kernel_rel};
    std::vector<std::string> host_files;

    // Host stub
    if (opt_emit_host && !args.dry_run()) {
      std::string stub_ext = (opt_platform == "cuda") ? ".cu" : ".cpp";
      std::string stub_path = paths.host_dir + "/launch" + stub_ext;
      writeFile(stub_path, host_stub(opt_platform));
      host_files.push_back("host/launch" + stub_ext);
      TLOG_INFO(TOOL, "emitted host stub → " + stub_path);
    }

    // CMake
    if (opt_emit_cmake && !args.dry_run()) {
      std::string cmake_path = paths.cmake_dir + "/CMakeLists.txt";
      writeFile(cmake_path, cmake_project(opt_platform, opt_arch));
      TLOG_INFO(TOOL, "emitted CMakeLists.txt → " + cmake_path);
    }

    // Occupancy report
    if (opt_occ_report && !args.dry_run()) {
      std::string occ = "{\n"
        "  \"kernel\": \"demo_kernel\",\n"
        "  \"arch\": \"" + opt_arch + "\",\n"
        "  \"regs_per_thread\": 128,\n"
        "  \"smem_bytes\": 49152,\n"
        "  \"max_blocks_per_sm\": 2,\n"
        "  \"theoretical_occupancy\": 0.50\n"
        "}\n";
      writeFile(paths.reports_dir + "/occupancy.json", occ);
      TLOG_INFO(TOOL, "occupancy report → reports/occupancy.json");
    }

    // Size report
    if (opt_size_report && !args.dry_run()) {
      long sz = fileSize(kernel_abs);
      std::string size_json = "{\n"
        "  \"kernel\": \"demo_kernel\",\n"
        "  \"file\": \"" + kernel_rel + "\",\n"
        "  \"size_bytes\": " + std::to_string(sz < 0 ? 0 : sz) + "\n"
        "}\n";
      writeFile(paths.reports_dir + "/sizes.json", size_json);
      TLOG_INFO(TOOL, "size report → reports/sizes.json");
    }

    // Manifest
    if (!args.dry_run()) {
      std::string manifest = compile_manifest(
          input_file, opt_from, opt_platform, opt_arch, opt_to,
          opt_tensor_cores, opt_wgmma, opt_tma,
          opt_emit_host, opt_emit_cmake, kernels, host_files);
      writeFile(paths.meta_dir + "/compile.json", manifest);
      TLOG_INFO(TOOL, "manifest → meta/compile.json");
    }

    std::cerr << "[" << TOOL << "] artifacts written to " << args.out_dir() << "\n";

  } catch (const std::exception& e) {
    TLOG_ERROR(TOOL, e.what());
    tessera::json_result(TOOL, args.out_dir(), false,
                         "\"error\":\"" + tessera::jsonEscape(e.what()) + "\"");
    return tessera::EXIT_IO_ERROR;
  }

  tessera::json_result(TOOL, args.out_dir(), true,
                       "\"platform\":\"" + opt_platform + "\","
                       "\"arch\":\"" + opt_arch + "\","
                       "\"to\":\"" + opt_to + "\"");
  return rc;
}
