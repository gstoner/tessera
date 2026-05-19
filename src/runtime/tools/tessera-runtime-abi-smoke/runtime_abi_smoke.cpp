//===- runtime_abi_smoke.cpp - tsr* C ABI lifetime smoke ----------------===//
//
// Exercises the public ``tsr*`` C ABI under ASAN/TSAN/UBSAN so any
// regression in handle ownership, init/shutdown lifecycle, or
// destroy-side counter accounting shows up as a sanitizer report
// instead of silent memory corruption.
//
// The smoke specifically covers the lanes the 2026-05-19 audit
// flagged:
//
//   1. Happy-path counter symmetry.  Every ``tsrCreateStream`` /
//      ``tsrCreateEvent`` / ``tsrMalloc`` must be matched by a
//      ``tsrDestroy*`` / ``tsrFree`` so the live-handle counters
//      return to zero — otherwise ``tsrShutdown`` refuses.
//
//   2. Live-handle refusal.  ``tsrShutdown`` with outstanding handles
//      must return ``TSR_STATUS_INVALID_ARGUMENT`` and emit a
//      diagnostic naming the live counts.  This is the runtime side
//      of the use-after-free shape the ratchet was added to prevent;
//      without ASAN the test still passes by status-code, but ASAN is
//      what catches a future regression where someone reorders the
//      counter decrement past the ``delete``.
//
//   3. Re-initialize after clean shutdown.  ``tsrInit`` → handles →
//      destroy → ``tsrShutdown`` → ``tsrInit`` again must work
//      (the old ``std::once_flag`` design would silently leave
//      ``g_devices`` empty on the second init).
//
//   4. Idempotent init.  ``tsrInit`` then ``tsrInit`` is benign.
//
//   5. Memcpy + memset round-trip.  Confirms ``tsrMemcpy`` /
//      ``tsrMemset`` don't smash buffer boundaries (ASAN red-zone
//      catches off-by-ones).
//
//   6. Multi-device handle isolation.  Each handle carries its
//      device back-pointer; mixing them across two devices on
//      destroy must not corrupt the per-device counters.
//
// Exit code 0 on success; non-zero on any contract violation.
// Sanitizer reports turn any ASAN/TSAN/UBSAN finding into abort().
//
//===----------------------------------------------------------------------===//

#include "tessera/tessera_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

#define REQUIRE_OK(expr) do {                                              \
  TsrStatus _s = (expr);                                                   \
  if (_s != TSR_STATUS_SUCCESS) {                                          \
    std::fprintf(stderr,                                                   \
      "[FAIL] %s:%d: %s -> %s (last_error: %s)\n",                         \
      __FILE__, __LINE__, #expr, tsrStatusString(_s), tsrGetLastError());  \
    return 1;                                                              \
  }                                                                        \
} while (0)

#define REQUIRE_STATUS(expr, expected) do {                                \
  TsrStatus _s = (expr);                                                   \
  if (_s != (expected)) {                                                  \
    std::fprintf(stderr,                                                   \
      "[FAIL] %s:%d: %s -> %s (expected %s; last_error: %s)\n",            \
      __FILE__, __LINE__, #expr, tsrStatusString(_s),                      \
      tsrStatusString(expected), tsrGetLastError());                       \
    return 1;                                                              \
  }                                                                        \
} while (0)


int test_init_is_idempotent_and_returns_devices() {
  REQUIRE_OK(tsrInit());
  REQUIRE_OK(tsrInit());            // idempotent — second call is benign
  int initialized = 0;
  REQUIRE_OK(tsrIsInitialized(&initialized));
  if (!initialized) {
    std::fprintf(stderr, "[FAIL] tsrIsInitialized returned 0 after init\n");
    return 1;
  }
  int count = 0;
  REQUIRE_OK(tsrGetDeviceCount(&count));
  if (count < 1) {
    std::fprintf(stderr, "[FAIL] no devices after tsrInit\n");
    return 1;
  }
  std::printf("[OK] init idempotent; %d device(s) registered\n", count);
  return 0;
}


int test_buffer_create_destroy_balances_counters() {
  tsrDevice dev = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &dev));

  // Allocate a buffer, write to its full extent, free.  ASAN's
  // red-zone catches any off-by-one on either side; the destroy
  // path's counter decrement must run.
  tsrBuffer b = nullptr;
  REQUIRE_OK(tsrMalloc(dev, 256, &b));
  REQUIRE_OK(tsrMemset(b, 0xAB, 256));

  // Map + read back to confirm the write landed; this also pokes
  // the map/unmap lifecycle, which the previous swallowing-errors
  // bug could have masked.
  void* host_ptr = nullptr;
  size_t bytes = 0;
  REQUIRE_OK(tsrMap(b, &host_ptr, &bytes));
  if (bytes != 256) {
    std::fprintf(stderr,
      "[FAIL] mapped size mismatch: got %zu, expected 256\n", bytes);
    tsrUnmap(b); tsrFree(b);
    return 1;
  }
  for (size_t i = 0; i < bytes; ++i) {
    if (static_cast<uint8_t*>(host_ptr)[i] != 0xAB) {
      std::fprintf(stderr,
        "[FAIL] memset signature lost at byte %zu\n", i);
      tsrUnmap(b); tsrFree(b);
      return 1;
    }
  }
  REQUIRE_OK(tsrUnmap(b));
  REQUIRE_OK(tsrFree(b));
  std::printf("[OK] malloc/memset/map/free round-trip (counters balanced)\n");
  return 0;
}


int test_memcpy_intra_device_round_trip() {
  tsrDevice dev = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &dev));

  tsrBuffer src = nullptr, dst = nullptr;
  REQUIRE_OK(tsrMalloc(dev, 1024, &src));
  REQUIRE_OK(tsrMalloc(dev, 1024, &dst));
  REQUIRE_OK(tsrMemset(src, 0xCD, 1024));
  REQUIRE_OK(tsrMemset(dst, 0x00, 1024));
  REQUIRE_OK(tsrMemcpy(dst, src, 1024, TSR_MEMCPY_DEVICE_TO_DEVICE));

  void* host = nullptr;
  size_t n = 0;
  REQUIRE_OK(tsrMap(dst, &host, &n));
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<uint8_t*>(host)[i] != 0xCD) {
      std::fprintf(stderr,
        "[FAIL] memcpy lost data at byte %zu (got 0x%02X)\n",
        i, static_cast<unsigned>(static_cast<uint8_t*>(host)[i]));
      tsrUnmap(dst); tsrFree(dst); tsrFree(src);
      return 1;
    }
  }
  REQUIRE_OK(tsrUnmap(dst));
  REQUIRE_OK(tsrFree(dst));
  REQUIRE_OK(tsrFree(src));
  std::printf("[OK] memcpy intra-device round-trip\n");
  return 0;
}


int test_stream_event_create_destroy_balances() {
  tsrDevice dev = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &dev));

  // 16 streams + 16 events.  Counter accounting (g_live_streams,
  // g_live_events) must net to zero by the end of the loop.
  std::vector<tsrStream> streams(16);
  std::vector<tsrEvent>  events(16);
  for (int i = 0; i < 16; ++i) {
    REQUIRE_OK(tsrCreateStream(dev, &streams[i]));
    REQUIRE_OK(tsrCreateEvent(dev, &events[i]));
  }
  // Drive each to exercise the recordEvent / waitEvent / sync paths.
  for (int i = 0; i < 16; ++i) {
    REQUIRE_OK(tsrRecordEvent(events[i], streams[i]));
    REQUIRE_OK(tsrEventSynchronize(events[i]));
    REQUIRE_OK(tsrStreamSynchronize(streams[i]));
  }
  for (int i = 0; i < 16; ++i) {
    REQUIRE_OK(tsrDestroyEvent(events[i]));
    REQUIRE_OK(tsrDestroyStream(streams[i]));
  }
  std::printf("[OK] 16x stream+event create/record/sync/destroy cycle\n");
  return 0;
}


int test_shutdown_with_live_handles_refuses() {
  // Create one of each handle category, then try to shutdown.  The
  // ratchet must refuse with INVALID_ARGUMENT and the diagnostic
  // must name the live counts.  Without the ratchet, ``tsrShutdown``
  // would silently delete devices and the subsequent ``tsrFree`` /
  // ``tsrDestroyStream`` / ``tsrDestroyEvent`` would deref freed
  // memory — exactly the shape ASAN catches.
  tsrDevice dev = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &dev));
  tsrStream s = nullptr;
  tsrEvent  e = nullptr;
  tsrBuffer b = nullptr;
  REQUIRE_OK(tsrCreateStream(dev, &s));
  REQUIRE_OK(tsrCreateEvent(dev, &e));
  REQUIRE_OK(tsrMalloc(dev, 64, &b));

  REQUIRE_STATUS(tsrShutdown(), TSR_STATUS_INVALID_ARGUMENT);
  const char* err = tsrGetLastError();
  if (!err || !std::strstr(err, "live handles") ||
      !std::strstr(err, "streams=1") ||
      !std::strstr(err, "events=1") ||
      !std::strstr(err, "buffers=1")) {
    std::fprintf(stderr,
      "[FAIL] tsrShutdown refusal diagnostic missing live counts: %s\n",
      err ? err : "(null)");
    // Clean up so a later run starts fresh.
    tsrFree(b); tsrDestroyEvent(e); tsrDestroyStream(s);
    return 1;
  }

  // Tear down properly — counters must return to zero.
  REQUIRE_OK(tsrFree(b));
  REQUIRE_OK(tsrDestroyEvent(e));
  REQUIRE_OK(tsrDestroyStream(s));

  // Now shutdown must succeed.
  REQUIRE_OK(tsrShutdown());
  int initialized = 0;
  REQUIRE_OK(tsrIsInitialized(&initialized));
  if (initialized) {
    std::fprintf(stderr,
      "[FAIL] tsrIsInitialized still true after clean shutdown\n");
    return 1;
  }
  std::printf("[OK] tsrShutdown refuses live handles; clean shutdown then succeeds\n");
  return 0;
}


int test_reinitialize_after_clean_shutdown() {
  // After a clean shutdown, init must repopulate the device list.
  // The pre-2026-05-19 `std::call_once` design failed silently here.
  REQUIRE_OK(tsrInit());
  int count = 0;
  REQUIRE_OK(tsrGetDeviceCount(&count));
  if (count < 1) {
    std::fprintf(stderr,
      "[FAIL] reinit didn't repopulate devices: count=%d\n", count);
    return 1;
  }
  // Create and destroy one of each handle category to confirm the
  // counters are fresh (no leftover state from the previous lifecycle).
  tsrDevice dev = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &dev));
  tsrStream s = nullptr;
  tsrBuffer b = nullptr;
  REQUIRE_OK(tsrCreateStream(dev, &s));
  REQUIRE_OK(tsrMalloc(dev, 32, &b));
  REQUIRE_OK(tsrFree(b));
  REQUIRE_OK(tsrDestroyStream(s));
  // And shutdown cleanly again.
  REQUIRE_OK(tsrShutdown());
  std::printf("[OK] init→handles→shutdown→init→handles→shutdown cycle clean\n");
  return 0;
}


int test_multi_device_handle_isolation_when_more_than_one() {
  // Honor whatever the runtime exposes — CPU-only builds report 1
  // device and skip this scenario; CUDA/HIP builds wire ≥ 2.
  REQUIRE_OK(tsrInit());
  int count = 0;
  REQUIRE_OK(tsrGetDeviceCount(&count));
  if (count < 2) {
    REQUIRE_OK(tsrShutdown());
    std::printf("[OK] multi-device isolation skipped (count=%d)\n", count);
    return 0;
  }
  tsrDevice d0 = nullptr, d1 = nullptr;
  REQUIRE_OK(tsrGetDevice(0, &d0));
  REQUIRE_OK(tsrGetDevice(1, &d1));

  tsrBuffer b0 = nullptr, b1 = nullptr;
  REQUIRE_OK(tsrMalloc(d0, 64, &b0));
  REQUIRE_OK(tsrMalloc(d1, 64, &b1));

  // Free in reverse order — counters must not under/overshoot.
  REQUIRE_OK(tsrFree(b1));
  REQUIRE_OK(tsrFree(b0));
  REQUIRE_OK(tsrShutdown());
  std::printf("[OK] multi-device handle isolation\n");
  return 0;
}

}  // namespace


int main() {
  int rc = 0;
  rc |= test_init_is_idempotent_and_returns_devices();
  rc |= test_buffer_create_destroy_balances_counters();
  rc |= test_memcpy_intra_device_round_trip();
  rc |= test_stream_event_create_destroy_balances();
  rc |= test_shutdown_with_live_handles_refuses();
  rc |= test_reinitialize_after_clean_shutdown();
  rc |= test_multi_device_handle_isolation_when_more_than_one();
  if (rc == 0) {
    std::printf("[ALL OK]\n");
  } else {
    std::printf("[FAILED] rc=%d\n", rc);
  }
  return rc;
}
