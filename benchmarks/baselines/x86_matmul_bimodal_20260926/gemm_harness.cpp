// X86-MATMUL-BIMODAL-1 C harness: times tessera_x86_avx512_gemm_f32 (256^3)
// with each operand at a chosen byte offset from a 2 MiB-aligned base
// (aligned_alloc(2 MiB), so the offset is also the offset within its 4 KiB
// page and cache line), or (offset < 0) from plain malloc. Each operand gets
// madvise(MADV_HUGEPAGE) when hugepage=1, else MADV_NOHUGEPAGE; B's mapping's
// AnonHugePages (kB, from /proc/self/smaps) is printed so a granted THP
// request is visible. Prints one line per process.
// usage: harness offA offB offC [hugepage(0/1)]
// build: g++ -O2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -std=gnu++17 \
//   -I <repo>/src/compiler/layout_algebra/include gemm_harness.cpp \
//   <repo>/src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_gemm_f32.cpp \
//   -o gemm_harness
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <sched.h>
#include <sys/mman.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>

extern "C" void tessera_x86_avx512_gemm_f32(const float*, const float*, int64_t,
                                            int64_t, int64_t, float*);

static double now_ns() {
  timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float* place(size_t bytes, long off, int huge) {
  if (off < 0) return (float*)malloc(bytes);
  size_t total = bytes + 4096 + (size_t)off + (2u << 20);
  char* raw = (char*)aligned_alloc(2u << 20, (total + (2u << 20) - 1) & ~((2ul << 20) - 1));
  if (huge) madvise(raw, total, MADV_HUGEPAGE); else madvise(raw, total, MADV_NOHUGEPAGE);
  return (float*)(raw + off);
}

// AnonHugePages (kB) of the smaps entry that contains addr; -1 if not found.
static long anon_huge_kb(const void* addr) {
  std::ifstream in("/proc/self/smaps");
  std::string line;
  uintptr_t a = (uintptr_t)addr;
  bool inside = false;
  while (std::getline(in, line)) {
    unsigned long lo, hi;
    if (sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2 && line.find(':') > line.find(' ')) {
      inside = lo <= a && a < hi;
      continue;
    }
    long kb;
    if (inside && sscanf(line.c_str(), "AnonHugePages: %ld kB", &kb) == 1) return kb;
  }
  return -1;
}

// dependent integer add chain: ~1 cycle per add, so ns/add -> 1/GHz
static double ghz_probe() {
  volatile uint64_t sink;
  uint64_t x = 1; const long n = 200000000;
  double t0 = now_ns();
  for (long i = 0; i < n; ++i) { asm volatile("add $1, %0" : "+r"(x)); }
  double t1 = now_ns(); sink = x; (void)sink;
  return n / (t1 - t0);
}

int main(int argc, char** argv) {
  long oa = argc > 1 ? atol(argv[1]) : -1, ob = argc > 2 ? atol(argv[2]) : -1,
       oc = argc > 3 ? atol(argv[3]) : -1;
  int huge = argc > 4 ? atoi(argv[4]) : 0;
  const int64_t M = 256, N = 256, K = 256;
  size_t bytes = M * N * sizeof(float);
  float *A = place(bytes, oa, huge), *B = place(bytes, ob, huge), *C = place(bytes, oc, huge);
  srand(20260926);
  for (int64_t i = 0; i < M * K; ++i) A[i] = (float)rand() / RAND_MAX - 0.5f;
  for (int64_t i = 0; i < K * N; ++i) B[i] = (float)rand() / RAND_MAX - 0.5f;
  memset(C, 0, bytes);
  double ghz0 = ghz_probe();
  tessera_x86_avx512_gemm_f32(A, B, M, N, K, C);
  std::vector<double> per;
  int cpu0 = sched_getcpu(), cpu1 = cpu0;
  for (int s = 0; s < 15; ++s) {
    double t0 = now_ns();
    for (int i = 0; i < 100; ++i) tessera_x86_avx512_gemm_f32(A, B, M, N, K, C);
    per.push_back((now_ns() - t0) / 100);
    cpu1 = sched_getcpu();
  }
  double ghz1 = ghz_probe();
  std::sort(per.begin(), per.end());
  printf("median_us=%.1f min_us=%.1f max_us=%.1f A%%4096=%lu B%%4096=%lu C%%4096=%lu "
         "A%%64=%lu B%%64=%lu C%%64=%lu cpu=%d->%d ghz=%.2f/%.2f huge_req=%d B_AnonHugePages_kB=%ld\n",
         per[7] / 1e3, per[0] / 1e3, per[14] / 1e3,
         (uintptr_t)A % 4096, (uintptr_t)B % 4096, (uintptr_t)C % 4096,
         (uintptr_t)A % 64, (uintptr_t)B % 64, (uintptr_t)C % 64, cpu0, cpu1, ghz0, ghz1, huge, anon_huge_kb(B));
  return 0;
}
