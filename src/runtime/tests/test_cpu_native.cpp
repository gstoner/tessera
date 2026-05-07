#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../include/tessera/tessera_runtime.h"

struct BarrierCtx {
  int* data;
};

static void kernel_barrier_reuse(void* user_ctx, const tsrTileCoord* tile, const tsrThreadCoord* thr) {
  tsrKernelCtx* kctx = (tsrKernelCtx*)user_ctx;
  BarrierCtx* ctx = (BarrierCtx*)kctx->user;
  int* shared = (int*)tsr_shared_mem(kctx);
  shared[thr->tx] = (int)thr->tx;
  tsr_tile_barrier(kctx);
  if (thr->tx == 0) {
    int sum = 0;
    for (uint32_t i = 0; i < kctx->shared_bytes / sizeof(int); ++i) sum += shared[i];
    ctx->data[tile->bx] = sum;
  }
}

int main() {
  assert(tsrInit() == TSR_STATUS_SUCCESS);
  tsrDevice dev = nullptr;
  assert(tsrGetDevice(0, &dev) == TSR_STATUS_SUCCESS);

  const int M = 3;
  const int N = 4;
  const int K = 5;
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);
  std::vector<float> want(M * N, 0.0f);
  for (int i = 0; i < M * K; ++i) a[i] = (float)((i % 7) - 3);
  for (int i = 0; i < K * N; ++i) b[i] = (float)((i % 5) + 1) * 0.25f;
  for (int i = 0; i < M; ++i)
  for (int j = 0; j < N; ++j)
  for (int p = 0; p < K; ++p) want[i * N + j] += a[i * K + p] * b[p * N + j];

  assert(tsrNativeGemmF32(dev, a.data(), b.data(), c.data(), M, N, K) == TSR_STATUS_SUCCESS);
  for (int i = 0; i < M * N; ++i) {
    if (std::fabs(c[i] - want[i]) > 1e-5f) {
      std::fprintf(stderr, "GEMM mismatch at %d: got %f want %f\n", i, c[i], want[i]);
      return 1;
    }
  }

  uint32_t before = 0;
  uint32_t after_first = 0;
  uint32_t after_second = 0;
  assert(tsrGetWorkerThreadCount(dev, &before) == TSR_STATUS_SUCCESS);

  constexpr uint32_t TILE = 64;
  constexpr uint32_t NTILES = 3;
  std::vector<int> reductions(NTILES, 0);
  BarrierCtx ctx{reductions.data()};
  tsrLaunchParams lp{};
  lp.grid = {NTILES, 1, 1};
  lp.tile = {TILE, 1, 1};
  lp.shared_mem_bytes = TILE * sizeof(int);

  assert(tsrLaunchHostTileKernelSync(dev, &lp, kernel_barrier_reuse, &ctx) == TSR_STATUS_SUCCESS);
  assert(tsrGetWorkerThreadCount(dev, &after_first) == TSR_STATUS_SUCCESS);
  assert(tsrLaunchHostTileKernelSync(dev, &lp, kernel_barrier_reuse, &ctx) == TSR_STATUS_SUCCESS);
  assert(tsrGetWorkerThreadCount(dev, &after_second) == TSR_STATUS_SUCCESS);

  const int expected = (int)((TILE - 1) * TILE / 2);
  for (uint32_t i = 0; i < NTILES; ++i) {
    if (reductions[i] != expected) {
      std::fprintf(stderr, "Barrier reduction mismatch at tile %u: got %d want %d\n", i, reductions[i], expected);
      return 1;
    }
  }
  assert(after_first >= TILE);
  assert(after_second == after_first);
  assert(after_second >= before);

  assert(tsrShutdown() == TSR_STATUS_SUCCESS);
  std::printf("CPU native runtime test passed.\n");
  return 0;
}
