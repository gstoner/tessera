
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>\n#ifdef USE_NVTX\n#include <nvToolsExt.h>\n#endif

using json = nlohmann::json;

int main(int argc, char** argv){
    int M=512,N=512,K=512,repeat=3;
    for (int i=1;i<argc;i++){
        std::string s(argv[i]);
        if (s.rfind("--m=",0)==0) M = std::stoi(s.substr(4));
        else if (s.rfind("--n=",0)==0) N = std::stoi(s.substr(4));
        else if (s.rfind("--k=",0)==0) K = std::stoi(s.substr(4));
        else if (s.rfind("--repeat=",0)==0) repeat = std::stoi(s.substr(9));
    }

    std::vector<float> A((size_t)M*K), B((size_t)K*N), C((size_t)M*N);
    for (size_t i=0;i<A.size();++i) A[i] = (float)(i%13)/13.0f;
    for (size_t i=0;i<B.size();++i) B[i] = (float)(i%17)/17.0f;

    double best_flops = 0.0;
    double last_ms = 0.0;

    for (int r=0;r<repeat;r++){
        #ifdef USE_NVTX\nnvtxRangePushA("bench");\n#endif\n    auto t0 = std::chrono::high_resolution_clock::now();
        for (int i=0;i<M;i++){
            for (int j=0;j<N;j++){
                float acc = 0.0f;
                for (int k=0;k<K;k++) acc += A[i*(size_t)K + k] * B[k*(size_t)N + j];
                C[i*(size_t)N + j] = acc;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();\n#ifdef USE_NVTX\nnvtxRangePop();\n#endif
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        last_ms = ms;
        double flops = 2.0 * (double)M * (double)N * (double)K / (ms/1000.0);
        if (flops > best_flops) best_flops = flops;
    }

    json row;
    row["throughput_flops"] = best_flops;
    row["latency_ms"] = last_ms;
    std::cout << row.dump() << std::endl;
    return 0;
}
