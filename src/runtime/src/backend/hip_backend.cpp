#include "base_backend.h"
#ifdef TESSERA_ENABLE_HIP
// Stub HIP backend
namespace tsr {
class HipBackend final : public Backend {
 public:
  DeviceProps props() const override {
    return DeviceProps{TSR_DEVICE_HIP, "Tessera HIP Backend (stub)", 1024, 1};
  }
  Buffer* malloc(size_t) override { return nullptr; }
  void free(Buffer*) override {}
  void memset(Buffer*, int, size_t) override {}
  void memcpy(Buffer*, const Buffer*, size_t, TsrMemcpyKind) override {}
  void* map(Buffer*) override { return nullptr; }
  void unmap(Buffer*) override {}
  Stream* createStream() override { return nullptr; }
  void destroyStream(Stream*) override {}
  void streamSync(Stream*) override {}
  Event* createEvent() override { return nullptr; }
  void destroyEvent(Event*) override {}
  void recordEvent(Event*, Stream*) override {}
  void waitEvent(Event*, Stream*) override {}
  void eventSync(Event*) override {}
  void launchHostKernel(Stream*, const tsrLaunchParams*, tsrHostKernelFn, void*) override {}
};
std::unique_ptr<Backend> CreateHipBackend() { return std::make_unique<HipBackend>(); }
} // namespace tsr
#else
namespace tsr { std::unique_ptr<Backend> CreateHipBackend() { return nullptr; } }
#endif
