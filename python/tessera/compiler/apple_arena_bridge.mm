// Explicit native-arena binding. Uses Tessera's existing Metal device/queue.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
struct ArenaPipeline {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  std::mutex mutex;
  bool timedOut = false;
  double gpuStart = 0, gpuEnd = 0;
};
static int fail(char *error, const char *message, int code = 1) {
  snprintf(error, 1024, "%s", message); return code;
}
extern "C" void *tessera_arena_create(void *device, void *queue, const char *msl,
                                     const char *entry, char *error) { @autoreleasepool {
  id<MTLDevice> d = (__bridge id<MTLDevice>)device;
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
  if (!d || !q || q.device != d) { fail(error, "Metal device/queue mismatch"); return nullptr; }
  NSError *e = nil;
  MTLCompileOptions *options = [MTLCompileOptions new]; options.fastMathEnabled = NO;
  id<MTLLibrary> library = [d newLibraryWithSource:@(msl) options:options error:&e];
  id<MTLFunction> fn = [library newFunctionWithName:@(entry)];
  id<MTLComputePipelineState> pipeline = fn ? [d newComputePipelineStateWithFunction:fn error:&e] : nil;
  if (!pipeline) { fail(error, e ? e.description.UTF8String : "Metal arena entry missing"); return nullptr; }
  return new ArenaPipeline{d, q, pipeline};
} }
extern "C" void tessera_arena_destroy(void *handle) { delete static_cast<ArenaPipeline *>(handle); }
// kinds: 0 is MTLBuffer; 1 is signed index. lengths are required bytes for buffers.
extern "C" int tessera_arena_launch(void *handle, const uint8_t *kinds,
    const uint64_t *values, const uint64_t *lengths, unsigned count, int64_t bytes,
    const uint64_t *grid, const uint64_t *block, char *error) { @autoreleasepool {
  auto *p = static_cast<ArenaPipeline *>(handle);
  std::lock_guard<std::mutex> guard(p->mutex);
  p->gpuStart = p->gpuEnd = 0;
  if (p->timedOut) return fail(error, "Metal arena binding disabled after timeout");
  if (count > 31 || bytes < 0 || bytes % 16 ||
      uint64_t(bytes) > p->device.maxThreadgroupMemoryLength ||
      p->pipeline.staticThreadgroupMemoryLength > p->device.maxThreadgroupMemoryLength - uint64_t(bytes))
    return fail(error, "Metal arena size exceeds device limit");
  uint64_t threads = 1;
  MTLSize limit = p->device.maxThreadsPerThreadgroup;
  uint64_t limits[3] = {limit.width, limit.height, limit.depth};
  for (unsigned i = 0; i < 3; ++i) {
    if (!grid[i] || grid[i] > UINT32_MAX || !block[i] || block[i] > limits[i] ||
        block[i] > p->pipeline.maxTotalThreadsPerThreadgroup / threads)
      return fail(error, "Metal arena launch geometry exceeds device limit");
    threads *= block[i];
  }
  // Retain and validate every buffer before creating the command encoder.
  NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray new];
  for (unsigned i = 0; i < count; ++i) {
    if (kinds[i] > 1) return fail(error, "Metal arena ABI kind invalid");
    if (!kinds[i]) {
      id<MTLBuffer> b = (__bridge id<MTLBuffer>)(void *)values[i];
      if (!b || b.device != p->device || !lengths[i] || lengths[i] > b.length)
        return fail(error, "Metal arena buffer extent/device disagrees");
      [buffers addObject:b];
    }
  }
  id<MTLCommandBuffer> command = [p->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!command || !encoder) return fail(error, "Metal arena command allocation failed");
  [encoder setComputePipelineState:p->pipeline];
  unsigned bufferIndex = 0;
  for (unsigned i = 0; i < count; ++i) {
    if (!kinds[i]) [encoder setBuffer:buffers[bufferIndex++] offset:0 atIndex:i];
    else [encoder setBytes:&values[i] length:8 atIndex:i];
  }
  [encoder setThreadgroupMemoryLength:bytes atIndex:0];
  [encoder dispatchThreadgroups:MTLSizeMake(grid[0],grid[1],grid[2])
          threadsPerThreadgroup:MTLSizeMake(block[0],block[1],block[2])];
  [encoder endEncoding];
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  // A timeout must not release storage still referenced by submitted work.
  [command addCompletedHandler:^(id<MTLCommandBuffer> _) {
    (void)buffers.count; dispatch_semaphore_signal(done);
  }];
  [command commit];
  if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC))) {
    p->timedOut = true; return fail(error, "Metal arena device wait timed out", 2);
  }
  if (command.status != MTLCommandBufferStatusCompleted)
    return fail(error, command.error ? command.error.description.UTF8String : "Metal arena execution failed");
  p->gpuStart = command.GPUStartTime;
  p->gpuEnd = command.GPUEndTime;
  return 0;
} }
extern "C" void tessera_arena_last_gpu_interval(void *handle, double *start, double *end) {
  auto *p = static_cast<ArenaPipeline *>(handle);
  std::lock_guard<std::mutex> guard(p->mutex);
  *start = p->gpuStart; *end = p->gpuEnd;
}

// Record the producer device explicitly: shared events need not expose one.
struct ArenaFence { id<MTLDevice> device; id<MTLSharedEvent> event; };
extern "C" void tessera_arena_fence_release(void *fence) {
  delete static_cast<ArenaFence *>(fence);
}
// One shared event per fence avoids generation reuse across independent queues.
extern "C" void *tessera_arena_queue_create(void *device) { @autoreleasepool {
  id<MTLDevice> d = (__bridge id<MTLDevice>)device;
  return (__bridge_retained void *)[d newCommandQueue];
} }
extern "C" void tessera_arena_object_release(void *object) {
  if (object) { id released = CFBridgingRelease(object); (void)released; }
}
extern "C" void *tessera_arena_queue_signal(void *queue, char *error) { @autoreleasepool {
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
  id<MTLSharedEvent> event = [q.device newSharedEvent];
  id<MTLCommandBuffer> command = [q commandBuffer];
  if (!event || !command) { fail(error, "Metal queue fence allocation failed"); return nullptr; }
  [command encodeSignalEvent:event value:1];
  [command commit];
  return new ArenaFence{q.device, event};
} }
extern "C" int tessera_arena_queue_wait(void *queue, void *fence, char *error) { @autoreleasepool {
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
  auto *f = static_cast<ArenaFence *>(fence);
  if (!q || !f || q.device != f->device) return fail(error, "Metal fence device mismatch");
  id<MTLCommandBuffer> command = [q commandBuffer];
  if (!command) return fail(error, "Metal queue wait allocation failed");
  [command encodeWaitForEvent:f->event value:1];
  [command commit];
  return 0;
} }
extern "C" int tessera_arena_queue_fill(void *queue, void *buffer, uint64_t bytes,
                                       uint8_t value, char *error) { @autoreleasepool {
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer;
  if (!q || !b || q.device != b.device || !bytes || bytes > b.length)
    return fail(error, "Metal fill buffer extent/device mismatch");
  id<MTLCommandBuffer> command = [q commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
  if (!command || !encoder) return fail(error, "Metal fill allocation failed");
  [encoder fillBuffer:b range:NSMakeRange(0, bytes) value:value];
  [encoder endEncoding];
  [command commit];
  return 0;
} }
