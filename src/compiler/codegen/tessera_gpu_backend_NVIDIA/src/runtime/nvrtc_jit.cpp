#include <nvrtc.h>
#include <cuda.h>
#include <string>
#include <vector>
#include <cstdio>

static bool jitToPtx(const char* src, const char* name, std::string& ptx, const char* arch="compute_90") {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src, name, 0, nullptr, nullptr);
    std::vector<const char*> opts = {"--std=c++17", "--fmad=true"};
    std::string gpu = std::string("--gpu-architecture=") + arch;
    opts.push_back(gpu.c_str());
    nvrtcResult rc = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    size_t logSize=0; nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(prog, log.data());
    if (rc != NVRTC_SUCCESS) {
        std::fprintf(stderr, "NVRTC compile failed: %s\n%s\n", nvrtcGetErrorString(rc), log.data());
        nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t ptxSize=0; nvrtcGetPTXSize(prog, &ptxSize);
    ptx.resize(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return true;
}
