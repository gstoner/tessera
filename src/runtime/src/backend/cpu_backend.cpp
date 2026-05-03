#include "base_backend.h"
#include <cstring>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include <vector>
#include <chrono>
#include <condition_variable>

namespace tsr {

static uint64_t NowNs() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  return duration_cast<nanoseconds>(steady_clock::now() - t0).count();
}

class CpuBackend final : public Backend {
 public:
  CpuBackend() : pool_(std::max(1u, std::thread::hardware_concurrency())) {}

  DeviceProps props() const override {
    DeviceProps p;
    p.kind = TSR_DEVICE_CPU;
    p.name = "Tessera CPU Backend";
    p.logical_tile_threads_max = 2048;
    p.concurrent_tiles_hint = std::max(1u, std::thread::hardware_concurrency());
    return p;
  }

  Buffer* malloc(size_t bytes) override {
    Buffer* b = new Buffer();
    size_t aligned = ((bytes + 63) / 64) * 64;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, aligned ? aligned : 64) != 0) {
      delete b;
      return nullptr;
    }
    b->ptr = ptr;
    if (!b->ptr) { delete b; return nullptr; }
    b->bytes = bytes;
    return b;
  }

  void free(Buffer* b) override {
    if (!b) return;
    std::free(b->ptr);
    delete b;
  }

  void memset(Buffer* b, int value, size_t bytes) override {
    std::memset(b->ptr, value, std::min(bytes, b->bytes));
  }

  void memcpy(Buffer* dst, const Buffer* src, size_t bytes, TsrMemcpyKind) override {
    std::memcpy(dst->ptr, src->ptr, std::min({bytes, dst->bytes, src->bytes}));
  }

  void* map(Buffer* b) override { return b->ptr; }
  void unmap(Buffer* ) override {}

  Stream* createStream() override { return new Stream(&pool_); }
  void destroyStream(Stream* s) override { delete s; }
  void streamSync(Stream* s) override { (void)s; pool_.WaitIdle(); }

  Event* createEvent() override { return new Event(); }
  void destroyEvent(Event* e) override { delete e; }
  void recordEvent(Event* e, Stream* ) override {
    std::lock_guard<std::mutex> lk(e->mu);
    e->signaled = true;
    e->timestamp_ns = NowNs();
  }
  void waitEvent(Event* e, Stream* ) override {
    for (;;) {
      std::lock_guard<std::mutex> lk(e->mu);
      if (e->signaled) return;
      std::this_thread::yield();
    }
  }
  void eventSync(Event* e) override { waitEvent(e, nullptr); }

  class TileBarrier {
   public:
    explicit TileBarrier(uint32_t expected) : expected_(std::max(1u, expected)) {}

    void wait() {
      std::unique_lock<std::mutex> lock(mu_);
      const uint64_t phase = phase_;
      if (++arrived_ == expected_) {
        arrived_ = 0;
        ++phase_;
        cv_.notify_all();
        return;
      }
      cv_.wait(lock, [&]{ return phase_ != phase; });
    }

   private:
    uint32_t expected_;
    uint32_t arrived_ = 0;
    uint64_t phase_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
  };

  struct TileInternals {
    TileBarrier* bar;
  };

  static void cpu_tile_barrier(tsrKernelCtx* kctx) {
    auto* ti = reinterpret_cast<TileInternals*>(kctx->_impl);
    ti->bar->wait();
  }

  void launchHostKernel(Stream* s,
                        const tsrLaunchParams* params,
                        tsrHostKernelFn kernel,
                        void* user_payload) override {
    (void)s;
    // Execute one tile at a time. Inside each tile, spawn per-thread workers to
    // preserve logical thread and barrier semantics for the CPU validation spine.
    for (uint32_t bz = 0; bz < params->grid.z; ++bz)
    for (uint32_t by = 0; by < params->grid.y; ++by)
    for (uint32_t bx = 0; bx < params->grid.x; ++bx) {
      const uint32_t txN = params->tile.x;
      const uint32_t tyN = params->tile.y;
      const uint32_t tzN = params->tile.z;
      const uint32_t nthreads = txN * tyN * tzN;
      std::vector<uint8_t> shared(params->shared_mem_bytes, 0);

      TileBarrier bar(nthreads);
      TileInternals ti{&bar};
      tsrTileCoord tile{bx, by, bz};

      std::vector<std::thread> workers;
      workers.reserve(nthreads);
      uint32_t linear = 0;
      for (uint32_t tz = 0; tz < tzN; ++tz)
      for (uint32_t ty = 0; ty < tyN; ++ty)
      for (uint32_t tx = 0; tx < txN; ++tx) {
        const uint32_t ltid = linear++;
        workers.emplace_back([&, tx, ty, tz, ltid](){
          tsrThreadCoord thr{tx, ty, tz, ltid};
          tsrKernelCtx kctx;
          kctx.user = user_payload;
          kctx.shared_mem = shared.data();
          kctx.shared_bytes = shared.size();
          kctx._tile_barrier = &cpu_tile_barrier;
          kctx._impl = &ti;
          kernel(&kctx, &tile, &thr);
        });
      }
      for (auto& t : workers) t.join();
    }
  }

 private:
  ThreadPool pool_;
};

std::unique_ptr<Backend> CreateCpuBackend() { return std::make_unique<CpuBackend>(); }

// Stub factories for CUDA/HIP (return nullptr here; real impls in separate files if enabled)
std::unique_ptr<Backend> CreateCudaBackend() { return nullptr; }
std::unique_ptr<Backend> CreateHipBackend()  { return nullptr; }

} // namespace tsr
