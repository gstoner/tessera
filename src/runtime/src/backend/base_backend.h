#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include "../scheduler/tile_scheduler.h"
#include "../../include/tessera/tsr_types.h"
#include "../../include/tessera/tsr_status.h"
#include "../../include/tessera/tsr_kernel.h"

namespace tsr {

struct Buffer { void* ptr=nullptr; size_t bytes=0; };

struct Stream { explicit Stream(ThreadPool* p):pool(p){} ThreadPool* pool; };

struct Event {
  std::mutex mu;
  bool signaled=false;
  uint64_t timestamp_ns=0; // time of recordEvent (profiling)
};

struct DeviceProps {
  TsrDeviceKind kind;
  std::string name;
  uint32_t logical_tile_threads_max;
  uint32_t concurrent_tiles_hint;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual DeviceProps props() const = 0;

  virtual Buffer* malloc(size_t bytes) = 0;
  virtual void free(Buffer* b) = 0;
  virtual void memset(Buffer* b, int value, size_t bytes) = 0;
  virtual void memcpy(Buffer* dst, const Buffer* src, size_t bytes, TsrMemcpyKind kind) = 0;
  virtual void* map(Buffer* b) = 0;
  virtual void unmap(Buffer* b) = 0;

  virtual Stream* createStream() = 0;
  virtual void destroyStream(Stream* s) = 0;
  virtual void streamSync(Stream* s) = 0;

  virtual Event* createEvent() = 0;
  virtual void destroyEvent(Event* e) = 0;
  virtual void recordEvent(Event* e, Stream* s) = 0;
  virtual void waitEvent(Event* e, Stream* s) = 0;
  virtual void eventSync(Event* e) = 0;

  virtual void launchHostKernel(Stream* s,
                                const tsrLaunchParams* params,
                                tsrHostKernelFn kernel,
                                void* user_payload) = 0;
};

// Factories
std::unique_ptr<Backend> CreateCpuBackend();
std::unique_ptr<Backend> CreateCudaBackend(); // may return nullptr if not built
std::unique_ptr<Backend> CreateHipBackend();  // may return nullptr if not built

} // namespace tsr
