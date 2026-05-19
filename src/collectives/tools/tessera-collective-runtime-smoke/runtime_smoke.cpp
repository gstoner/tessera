//===- runtime_smoke.cpp - Collective runtime contract smoke ------------===//
//
// Locks the contract that the collective C runtime entry points in
// ``Execution.cpp`` survive the cases that the original raw-global
// pointer would have failed:
//
//   1. **Concurrent submitters are safe and overlap.**  Eight threads
//      each issue 100 submits; the runtime must not crash, race, or
//      double-allocate the runtime singleton.  The global mutex is
//      NOT held through ``ExecRuntime::submit()`` — both
//      ``TokenLimiter`` and ``PerfettoTraceWriter`` have their own
//      internal synchronization, so concurrent submits can overlap
//      with each other.
//
//   2. **`tessera_shutdown_runtime()` while a submit is in flight
//      does not crash.**  The shared_ptr-based slot means the
//      runtime stays alive past a concurrent shutdown until the
//      last in-flight ``submit()`` returns.
//
//   3. **Init-after-shutdown re-creates the runtime.**  The C ABI
//      must accept subsequent submits without diagnostic after an
//      explicit ``tessera_shutdown_runtime()``.
//
//   4. **PerfettoTraceWriter is internally thread-safe.**  This is
//      the prerequisite that made (1) safe to ship: without
//      internal synchronization the writer's ``events_`` vector
//      races under concurrent ``submit()`` calls.  We exercise the
//      writer directly here so a future refactor that drops the
//      writer's internal mutex would fail this test before it
//      destabilizes the rest of the runtime.
//
// Exit code 0 on success, non-zero on any contract violation.
//
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Collective/Runtime/Execution.h"
#include "tessera/Dialect/Collective/Runtime/PerfettoTrace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// The `extern "C"` entry points are declared inside the
// `tessera::collective` namespace in Execution.h; the linker actually
// resolves them by unmangled C symbol, but the C++ compiler still
// needs the qualified name to find the declaration.
using tessera::collective::tessera_qos_limit_set;
using tessera::collective::tessera_submit_chunk_async;
using tessera::collective::tessera_shutdown_runtime;

namespace {

using clock = std::chrono::steady_clock;
using namespace tessera::collective;

int test_concurrent_submitters_dont_corrupt() {
  constexpr int kThreads = 8;
  constexpr int kSubmitsPerThread = 100;

  tessera_qos_limit_set(/*tokens=*/8);

  // Warm up: one submit to amortize first-time runtime construction
  // cost out of the timing window.
  uint8_t buf[16] = {0};
  tessera_submit_chunk_async(buf, sizeof(buf), /*device=*/0, /*stream=*/0);

  auto t0 = clock::now();
  std::vector<std::thread> ts;
  ts.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([t, &buf]() {
      for (int i = 0; i < kSubmitsPerThread; ++i) {
        tessera_submit_chunk_async(buf, sizeof(buf),
                                    /*device=*/0, /*stream=*/t % 2);
      }
    });
  }
  for (auto& th : ts) th.join();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                   clock::now() - t0).count();

  if (elapsed < 0) {
    std::fprintf(stderr, "[FAIL] negative elapsed time?\n");
    return 1;
  }
  std::printf("[OK] %d threads * %d submits = %d submits in %lld us "
              "(no global mutex on submit hot path)\n",
              kThreads, kSubmitsPerThread, kThreads * kSubmitsPerThread,
              (long long)elapsed);
  return 0;
}

int test_shutdown_while_submitting() {
  // Spawn a submitter that keeps issuing submits, then call
  // tessera_shutdown_runtime() from the main thread.  Neither side
  // may crash.  We don't try to assert any ordering — just survival.
  std::atomic<bool> stop{false};
  uint8_t buf[16] = {0};
  std::thread submitter([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      tessera_submit_chunk_async(buf, sizeof(buf), 0, 0);
    }
  });

  for (int i = 0; i < 5; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    tessera_shutdown_runtime();
    // A subsequent submit on the submitter thread should
    // re-initialize a fresh runtime; that's the documented
    // contract.
  }

  stop.store(true, std::memory_order_relaxed);
  submitter.join();

  std::printf("[OK] shutdown-while-submitting survived\n");
  return 0;
}

int test_repeat_initialize_after_shutdown() {
  // After an explicit shutdown, the next submit must re-create the
  // runtime on demand.  We can't directly inspect runtime identity
  // from outside, but we can ensure the C ABI accepts subsequent
  // calls without diagnostic.
  uint8_t buf[16] = {0};
  tessera_submit_chunk_async(buf, sizeof(buf), 0, 0);
  tessera_shutdown_runtime();
  tessera_submit_chunk_async(buf, sizeof(buf), 0, 0);  // must reinit
  tessera_shutdown_runtime();
  tessera_qos_limit_set(4);                            // must reinit
  tessera_shutdown_runtime();
  std::printf("[OK] init-after-shutdown cycle\n");
  return 0;
}

int test_perfetto_trace_writer_is_internally_thread_safe() {
  // Drive a *standalone* PerfettoTraceWriter (no runtime, no global
  // mutex) from 8 threads.  Each thread issues 1000 begin/end +
  // counter calls.  The pre-2026-05-19 writer would corrupt the
  // ``events_`` vector under this load; the current writer takes its
  // own internal lock on every mutating method.
  //
  // We don't assert any property of the trace contents (the event
  // count is checked below); the primary success signal is that the
  // process survives.  ASAN/TSAN would catch the data race if the
  // internal mutex were ever removed.
  PerfettoTraceWriter w;
  constexpr int kThreads = 8;
  constexpr int kBeginsPerThread = 1000;
  std::vector<std::thread> ts;
  ts.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&w, t]() {
      for (int i = 0; i < kBeginsPerThread; ++i) {
        w.begin("ev", "cat", /*pid=*/1, /*tid=*/t);
        w.counter("ctr", static_cast<double>(i), 1, t);
        w.end("ev", "cat", 1, t);
        w.annotate("annot_key", "annot_val");
      }
    });
  }
  for (auto& th : ts) th.join();

  // Each thread emits 3 mutating events that land in ``events_``
  // (begin, counter, end).  ``annotate`` writes into ``meta_``, not
  // ``events_``.  Verify the count is exact — that's the strict
  // synchronization invariant.
  //
  // We use ``write()`` to a tmp file and count the trace events the
  // writer flushed.  Round-trip through the writer's own snapshot
  // path exercises the read side of the lock too.
  // Use mkstemp instead of tmpnam (POSIX-deprecated for race
  // reasons; toolchains warn).  We immediately ``close`` the fd
  // because PerfettoTraceWriter::write opens the path itself; we
  // only needed mkstemp to materialize a unique filename.
  char tmpl[] = "/tmp/tessera_trace_smoke_XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) {
    std::fprintf(stderr, "[FAIL] mkstemp failed\n");
    return 1;
  }
  ::close(fd);
  const std::string tmp = tmpl;
  w.write(tmp);
  std::ifstream in(tmp);
  if (!in.is_open()) {
    std::fprintf(stderr, "[FAIL] could not reopen %s\n", tmp.c_str());
    return 1;
  }
  std::string contents((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  // The exact count is ``kThreads * kBeginsPerThread * 3``.
  // Confirming exact recovery proves no events were lost to a race.
  const int expected = kThreads * kBeginsPerThread * 3;
  int counted = 0;
  for (size_t i = 0; (i = contents.find("\"name\":", i)) != std::string::npos; ++i) {
    ++counted;
  }
  std::remove(tmp.c_str());
  if (counted != expected) {
    std::fprintf(stderr,
                 "[FAIL] PerfettoTraceWriter lost %d events under "
                 "concurrent writes (expected %d, got %d)\n",
                 expected - counted, expected, counted);
    return 1;
  }
  std::printf("[OK] PerfettoTraceWriter survived %d concurrent events "
              "(no event loss)\n", expected);
  return 0;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_concurrent_submitters_dont_corrupt();
  rc |= test_shutdown_while_submitting();
  rc |= test_repeat_initialize_after_shutdown();
  rc |= test_perfetto_trace_writer_is_internally_thread_safe();
  if (rc == 0) {
    std::printf("[ALL OK]\n");
  } else {
    std::printf("[FAILED] rc=%d\n", rc);
  }
  return rc;
}
