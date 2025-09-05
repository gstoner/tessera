// SPDX-License-Identifier: MIT
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

static double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Args {
    int N = 1<<20; // 1M points
    int repeats = 50;
    int warmup = 10;
};

static void parse_args(int argc, char** argv, Args& a) {
    for (int i=1;i<argc;i++) {
        std::string s(argv[i]);
        if (s=="--help" || s=="-h") {
            std::cout << "spectral_fft_bench [--N <int>] [--repeats <int>] [--warmup <int>]\n";
            std::exit(0);
        }
        if (s=="--N" && i+1<argc) { a.N = std::atoi(argv[++i]); continue; }
        if (s=="--repeats" && i+1<argc) { a.repeats = std::atoi(argv[++i]); continue; }
        if (s=="--warmup" && i+1<argc) { a.warmup = std::atoi(argv[++i]); continue; }
    }
}

int main(int argc, char** argv) {
    Args args; parse_args(argc, argv, args);
    std::cout << "Tessera C++ spectral bench\n";
    std::cout << "N="<<args.N<<" repeats="<<args.repeats<<" warmup="<<args.warmup<<"\n";

#if defined(TESSERA_HAVE_FFTW)
    std::cout << "FFTW path enabled (CPU)\n";
#else
    std::cout << "FFTW not found.\n";
#endif
#if defined(TESSERA_HAVE_CUFFT)
    std::cout << "cuFFT path enabled (CUDA)\n";
#else
    std::cout << "cuFFT not found.\n";
#endif
#if defined(TESSERA_HAVE_ROCFFT)
    std::cout << "rocFFT path enabled (ROCm)\n";
#else
    std::cout << "rocFFT not found.\n";
#endif

    // Placeholder loop so the binary produces output even without FFT libs.
    std::vector<double> x(args.N);
    for (int i=0;i<args.N;i++) x[i] = std::sin(0.001*i);

    auto work = [&](){
        double acc=0.0;
        for (int i=0;i<args.N;i++) acc += x[i]*0.5 + std::cos(x[i]);
        return acc;
    };

    for (int i=0;i<args.warmup;i++) (void)work();

    double t0 = now_seconds();
    double acc=0.0;
    for (int i=0;i<args.repeats;i++) acc += work();
    double t1 = now_seconds();

    double avg_ms = (t1 - t0) * 1000.0 / args.repeats;
    std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
    return 0;
}
