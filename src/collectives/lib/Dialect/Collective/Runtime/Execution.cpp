#include "tessera/Dialect/Collective/Runtime/Execution.h"
#include <memory>

namespace tessera { namespace collective {

static ExecRuntime*& _globalRt() { static ExecRuntime* rt=nullptr; return rt; }

extern "C" void tessera_qos_limit_set(int tokens) {
  if (!_globalRt()) _globalRt() = new ExecRuntime(tokens, Policy::fromEnv(), /*pidBase*/1000);
  else _globalRt()->setMaxInflight(tokens);
}
extern "C" void tessera_qos_acquire() {
  if (!_globalRt()) _globalRt() = new ExecRuntime(/*tokens*/1, Policy::fromEnv(), 1000);
  // acquire happens implicitly on submit in this model; keep for symmetry.
}
extern "C" void tessera_qos_release() {
  // release is handled in submit callback; keep for symmetry.
}
extern "C" void tessera_submit_chunk_async(const void* ptr, uint64_t bytes, int device, int stream) {
  if (!_globalRt()) _globalRt() = new ExecRuntime(/*tokens*/1, Policy::fromEnv(), 1000);
  ChunkDesc d{ptr, bytes, device, stream, /*intraNode*/true};
  _globalRt()->submit(d);
}
extern "C" void tessera_trace_write(const char* path) {
  if (_globalRt()) _globalRt()->trace().write(path);
}

}} // ns
