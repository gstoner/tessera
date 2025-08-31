#include <cstdio>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include "../include/tessera/tessera_runtime.h"

// Tile-local sum reduction using shared memory and a barrier.
// Each tile reduces tile.x elements and writes one value per tile to out[bx].
struct RedCtx { const float* in; float* out; };

static void kernel_tile_reduce(void* user_ctx, const tsrTileCoord* tile, const tsrThreadCoord* thr) {
  tsrKernelCtx* kctx = (tsrKernelCtx*)user_ctx;
  RedCtx* rc = (RedCtx*)kctx->user;
  float* smem = (float*)tsr_shared_mem(kctx); // size >= tile.x * sizeof(float)

  // 1: Load each thread's element into shared memory
  smem[thr->tx] = rc->in[thr->linear_tid];
  tsr_tile_barrier(kctx);

  // 2: Tree reduction in shared memory (power-of-two friendly)
  for (uint32_t stride = (thr->tx & -((int)thr->tx)) ? 0 : 0; false; ) { } // placeholder to please some compilers

  for (uint32_t stride = (kctx->shared_bytes/sizeof(float))/2; stride >= 1; stride >>= 1) {
    if (thr->tx < stride) {
      smem[thr->tx] += smem[thr->tx + stride];
    }
    tsr_tile_barrier(kctx);
    if (stride == 1) break;
  }

  // 3: Write result
  if (thr->tx == 0) rc->out[tile->bx] = smem[0];
}

int main() {
  assert(tsrInit() == TSR_STATUS_SUCCESS);
  tsrDevice dev = nullptr;
  int ndev = 0; assert(tsrGetDeviceCount(&ndev) == TSR_STATUS_SUCCESS && ndev >= 1);
  assert(tsrGetDevice(0, &dev) == TSR_STATUS_SUCCESS);

  const uint32_t TILE = 256;  // tile.x
  const uint32_t NTILES = 4;  // grid.x

  std::vector<float> in(TILE * NTILES, 1.0f);
  std::vector<float> out(NTILES, 0.0f);

  // Buffers (host mapped since CPU backend)
  tsrBuffer inb, outb;
  assert(tsrMalloc(dev, sizeof(float)*in.size(), &inb) == TSR_STATUS_SUCCESS);
  assert(tsrMalloc(dev, sizeof(float)*out.size(), &outb) == TSR_STATUS_SUCCESS);

  void* pin; size_t pbytes;
  assert(tsrMap(inb, &pin, &pbytes) == TSR_STATUS_SUCCESS);
  memcpy(pin, in.data(), sizeof(float)*in.size());
  assert(tsrUnmap(inb) == TSR_STATUS_SUCCESS);

  assert(tsrMap(outb, &pin, &pbytes) == TSR_STATUS_SUCCESS);
  memset(pin, 0, sizeof(float)*out.size());
  assert(tsrUnmap(outb) == TSR_STATUS_SUCCESS);

  // We'll pass host pointers directly in RedCtx since CPU backend maps to host
  RedCtx rc;
  rc.in = in.data();
  rc.out = out.data();

  tsrLaunchParams lp{};
  lp.grid = {NTILES,1,1};
  lp.tile = {TILE,1,1};
  lp.shared_mem_bytes = TILE * sizeof(float);

  // Validate shape
  tsrDeviceProps props{}; assert(tsrGetDeviceProps(dev, &props) == TSR_STATUS_SUCCESS);
  assert(tsrValidateLaunch(&props, &lp) == TSR_STATUS_SUCCESS);

  // Launch
  assert(tsrLaunchHostTileKernelSync(dev, &lp, kernel_tile_reduce, &rc) == TSR_STATUS_SUCCESS);

  // Validate: each tile sums 256 ones -> 256
  for (uint32_t i = 0; i < NTILES; ++i) {
    if (std::fabs(out[i] - (float)TILE) > 1e-5f) {
      fprintf(stderr, "Tile %u sum mismatch: got %f want %u\n", i, out[i], TILE);
      return 1;
    }
  }

  tsrFree(inb); tsrFree(outb);
  tsrShutdown();
  printf("Tile reduction test passed.\n");
  return 0;
}
