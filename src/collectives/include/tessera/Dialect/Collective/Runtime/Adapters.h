#pragma once
#include <functional>
#include <cstddef>
#include <cstdint>

namespace tessera { namespace collective {

using Callback = std::function<void()>;

struct NCCLAdapter {
  bool enabled() const { return false; }
  void submitChunkAsync(const void* buf, size_t bytes, int device, int stream, Callback cb) {
    (void)buf; (void)bytes; (void)device; (void)stream; if (cb) cb();
  }
};

struct RCCLAdapter {
  bool enabled() const { return false; }
  void submitChunkAsync(const void* buf, size_t bytes, int device, int stream, Callback cb) {
    (void)buf; (void)bytes; (void)device; (void)stream; if (cb) cb();
  }
};

}} // ns
