
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>\n#ifdef USE_NVTX\n#include <nvToolsExt.h>\n#endif

using json = nlohmann::json;

int main(int argc, char** argv){
    size_t N = 1<<24;
    int repeat = 5;
    for (int i=1;i<argc;i++){
        if (std::string(argv[i]).rfind("--size=",0)==0) N = std::stoull(std::string(argv[i]).substr(7));
        if (std::string(argv[i]).rfind("--repeat=",0)==0) repeat = std::stoi(std::string(argv[i]).substr(9));
    }

    std::vector<size_t> next(N);
    for (size_t i=0;i<N;i++) next[i] = (i+1)%N;

    volatile size_t sum = 0;
    double best_ns = 1e9;

    for (int r=0;r<repeat;r++){
        #ifdef USE_NVTX\nnvtxRangePushA("bench");\n#endif\n    auto t0 = std::chrono::high_resolution_clock::now();
        size_t idx = 0;
        for (size_t i=0;i<N;i++) idx = next[idx];
        auto t1 = std::chrono::high_resolution_clock::now();\n#ifdef USE_NVTX\nnvtxRangePop();\n#endif
        double ns = std::chrono::duration<double, std::nano>(t1-t0).count() / N;
        if (ns < best_ns) best_ns = ns;
        sum += idx;
    }

    json row;
    row["latency_ns"] = best_ns;
    std::cout << row.dump() << std::endl;
    return 0;
}
