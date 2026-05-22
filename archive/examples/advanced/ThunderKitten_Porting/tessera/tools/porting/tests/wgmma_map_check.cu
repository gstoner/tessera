
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "../microbench/microbench_main.cu"  // reuse kernels/paths (or provide proper headers in real repo)

int main(){
  // This is a placeholder harness entry. In a real repo, we'd expose a clean API to invoke both paths.
  // Here we simply print a note: run microbench with --compute_path wgmma and then with wmma and compare JSON outputs offline.
  printf("Run the microbench twice with identical shapes/knobs:\n");
  printf("  1) --compute_path wgmma --epilogue none\n");
  printf("  2) --compute_path wmma  --epilogue none\n");
  printf("Compare correctness/perf as needed.\n");
  return 0;
}
