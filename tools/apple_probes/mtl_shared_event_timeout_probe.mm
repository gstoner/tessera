// Does -[MTLSharedEvent waitUntilSignaledValue:timeoutMS:] honour its timeout?
//
// Isolates the API from every line of Tessera code. Three cases:
//   A. never signalled            -> must return NO after ~timeout
//   B. signalled by the CPU       -> must return YES promptly
//   C. signalled by a GPU encode  -> must return YES promptly (the real shape)
//
// Case A is the one that matters: `commit_mpsgraph_and_wait_with_timeout`
// passes 30000 and was observed blocked for 70 minutes.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>

static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("no Metal device\n"); return 1; }
    printf("device: %s\n\n", [[dev name] UTF8String]);

    // --- A. never signalled -------------------------------------------------
    for (uint64_t timeout : {(uint64_t)250, (uint64_t)1000, (uint64_t)3000}) {
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      auto t0 = std::chrono::steady_clock::now();
      BOOL ok = [ev waitUntilSignaledValue:1 timeoutMS:timeout];
      double elapsed = ms_since(t0);
      printf("A timeout=%5llums -> returned %s after %8.1f ms  (%s)\n",
             (unsigned long long)timeout, ok ? "YES" : "NO ", elapsed,
             (!ok && elapsed >= timeout * 0.5 && elapsed <= timeout * 3.0)
                 ? "HONOURED" : "NOT honoured");
    }

    // --- B. signalled by the CPU -------------------------------------------
    {
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
                     dispatch_get_global_queue(0, 0), ^{ ev.signaledValue = 7; });
      auto t0 = std::chrono::steady_clock::now();
      BOOL ok = [ev waitUntilSignaledValue:7 timeoutMS:5000];
      printf("\nB cpu-signal        -> returned %s after %8.1f ms\n",
             ok ? "YES" : "NO ", ms_since(t0));
    }

    // --- C. signalled by a GPU command buffer ------------------------------
    {
      id<MTLCommandQueue> q = [dev newCommandQueue];
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      [cb encodeSignalEvent:ev value:11];
      auto t0 = std::chrono::steady_clock::now();
      [cb commit];
      BOOL ok = [ev waitUntilSignaledValue:11 timeoutMS:5000];
      printf("C gpu-signal        -> returned %s after %8.1f ms\n",
             ok ? "YES" : "NO ", ms_since(t0));
    }

    // --- D. committed buffer that signals a DIFFERENT value ----------------
    // The shape of the hang: work completes, but the awaited value never
    // arrives. If the timeout is honoured here, a wedged GPU is the only
    // remaining explanation for the 70-minute block.
    {
      id<MTLCommandQueue> q = [dev newCommandQueue];
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      [cb encodeSignalEvent:ev value:1];
      [cb commit];
      auto t0 = std::chrono::steady_clock::now();
      BOOL ok = [ev waitUntilSignaledValue:999 timeoutMS:1000];
      double elapsed = ms_since(t0);
      printf("D awaits unreachable-> returned %s after %8.1f ms  (%s)\n",
             ok ? "YES" : "NO ", elapsed,
             (!ok && elapsed <= 3000) ? "HONOURED" : "NOT honoured");
    }
  }
  return 0;
}
