
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>\n#ifdef USE_NVTX\n#include <nvToolsExt.h>\n#endif

using json = nlohmann::json;

int main(int argc, char** argv){
    size_t N = 1<<26; // elements
    int repeat = 3;
    for (int i=1;i<argc;i++){
        if (std::string(argv[i]).rfind("--size=",0)==0) N = std::stoull(std::string(argv[i]).substr(7));
        if (std::string(argv[i]).rfind("--repeat=",0)==0) repeat = std::stoi(std::string(argv[i]).substr(9));
    }
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 0.0f);
    double best_bw = 0.0;
    double last_ms = 0.0;

    for (int r=0;r<repeat;r++){
        #ifdef USE_NVTX\nnvtxRangePushA("bench");\n#endif\n    auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i=0;i<N;i++) C[i] = A[i] + 2.0f*B[i]; // triad-ish
        auto t1 = std::chrono::high_resolution_clock::now();\n#ifdef USE_NVTX\nnvtxRangePop();\n#endif
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        last_ms = ms;
        // bytes moved: read A,B and write C
        double bytes = (double)N * sizeof(float) * 3.0;
        double bw = bytes / (ms/1000.0);
        if (bw > best_bw) best_bw = bw;
    }

    json row;
    row["bytes_per_s"] = best_bw;
    row["latency_ms"] = last_ms;
    std::cout << row.dump() << std::endl;
    return 0;
}
