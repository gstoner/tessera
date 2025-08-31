#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstring>
#include "../include/tessera/tessera_runtime.h"

// Simple increment kernel using the kernel context.
static void kernel_increment(void* user_ctx, const tsrTileCoord* tile, const tsrThreadCoord* thr) {
  (void)tile;
  tsrKernelCtx* kctx = (tsrKernelCtx*)user_ctx;
  struct Ctx { int* data; int n; }* ctx = (Ctx*)kctx->user;
  const int idx = (int)thr->linear_tid;
  if (idx < ctx->n) ctx->data[idx] += 1;
}

int main() {
  int M,m,p; tsrGetVersion(&M,&m,&p);
  assert(M >= 0 && m >= 0 && p >= 0);
  assert(tsrInit() == TSR_STATUS_SUCCESS);

  int count = 0;
  assert(tsrGetDeviceCount(&count) == TSR_STATUS_SUCCESS);
  assert(count >= 1);
  tsrDevice dev = nullptr;
  assert(tsrGetDevice(0, &dev) == TSR_STATUS_SUCCESS);

  tsrDeviceProps props{};
  assert(tsrGetDeviceProps(dev, &props) == TSR_STATUS_SUCCESS);

  const size_t N = 256;
  tsrBuffer buf;
  assert(tsrMalloc(dev, sizeof(int)*N, &buf) == TSR_STATUS_SUCCESS);
  assert(tsrMemset(buf, 0, sizeof(int)*N) == TSR_STATUS_SUCCESS);

  void* host = nullptr; size_t bytes = 0;
  assert(tsrMap(buf, &host, &bytes) == TSR_STATUS_SUCCESS);
  assert(bytes >= sizeof(int)*N);
  int* data = reinterpret_cast<int*>(host);

  tsrLaunchParams lp{};
  lp.grid = {1,1,1};
  lp.tile = { (uint32_t)N, 1, 1 };
  lp.shared_mem_bytes = 0;
  lp.flags = 0;

  struct Ctx { int* data; int n; } ctx{data, (int)N};
  assert(tsrLaunchHostTileKernelSync(dev, &lp, kernel_increment, &ctx) == TSR_STATUS_SUCCESS);

  for (size_t i = 0; i < N; ++i) {
    if (data[i] != 1) {
      fprintf(stderr, "Validation failed at %zu: got %d, want 1\n", i, data[i]);
      return 1;
    }
  }

  assert(tsrUnmap(buf) == TSR_STATUS_SUCCESS);
  assert(tsrFree(buf) == TSR_STATUS_SUCCESS);

  // Stream, event, profiling
  tsrStream s; assert(tsrCreateStream(dev, &s) == TSR_STATUS_SUCCESS);
  tsrEvent e;  assert(tsrCreateEvent(dev, &e) == TSR_STATUS_SUCCESS);
  assert(tsrRecordEvent(e, s) == TSR_STATUS_SUCCESS);
  uint64_t ts = 0; assert(tsrEventGetTimestamp(e, &ts) == TSR_STATUS_SUCCESS);
  assert(ts > 0);
  assert(tsrWaitEvent(e, s) == TSR_STATUS_SUCCESS);
  assert(tsrEventSynchronize(e) == TSR_STATUS_SUCCESS);
  assert(tsrDestroyEvent(e) == TSR_STATUS_SUCCESS);
  assert(tsrDestroyStream(s) == TSR_STATUS_SUCCESS);

  assert(tsrShutdown() == TSR_STATUS_SUCCESS);
  printf("Basic runtime test passed.\n");
  return 0;
}
