// Owning-host probe for compiler-generated MSL + native LLVM sizing companion.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <initializer_list>

int main(int argc, const char **argv) { @autoreleasepool {
  if (argc != 4) return 2;
  NSError *error = nil;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { fprintf(stderr, "Metal device unavailable\n"); return 3; }
  NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.fastMathEnabled = NO;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 4; }
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"scratch"] error:&error];
  if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 5; }
  void *host = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!host) { fprintf(stderr, "%s\n", dlerror()); return 6; }
  using Sizer = int64_t (*)(void *, int64_t, int64_t);
  auto size = reinterpret_cast<Sizer>(dlsym(host, argv[3]));
  if (!size) return 7;
  id<MTLCommandQueue> queue = [device newCommandQueue];
  printf("{\"device\":\"%s\",\"rows\":[", device.name.UTF8String);
  bool first = true;
  for (int64_t width : {1, 17, 32, 64, 128, 256}) {
    int64_t rounds = 17, blocks = 32;
    int64_t bytes = size(nullptr, width, rounds);
    if (bytes <= 0 || bytes % 16 || bytes + pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength || width > pipeline.maxTotalThreadsPerThreadgroup) return 8;
    id<MTLBuffer> output = [device newBufferWithLength:blocks*width*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&width length:sizeof(width) atIndex:1];
    [encoder setBytes:&rounds length:sizeof(rounds) atIndex:2];
    [encoder setThreadgroupMemoryLength:bytes atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(completed); }];
    [command commit];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC))) return 9;
    if (command.status != MTLCommandBufferStatusCompleted) { fprintf(stderr,"%s\n",command.error.description.UTF8String); return 10; }
    const float *values = static_cast<const float *>(output.contents);
    for (int64_t i = 0; i < blocks*width; ++i) {
      float expected = rounds * ((i % width + 1) % width) + rounds*(rounds-1)/2;
      if (values[i] != expected) { fprintf(stderr,"mismatch %lld: %f != %f\n",i,values[i],expected); return 11; }
    }
    printf("%s{\"width\":%lld,\"rounds\":%lld,\"native_bytes\":%lld,\"oracle\":\"exact\"}",first?"":",",width,rounds,bytes);
    first=false;
  }
  if (size(nullptr, INT64_MAX, 17) != -1) return 12;
  printf("],\"oversize_rejected\":true,\"selector_promotion\":false}\n");
  dlclose(host);
  return 0;
} }
