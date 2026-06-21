#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

extern "C" bool tprof_metal_command_buffer_probe_compiled() {
  return true;
}

extern "C" bool tprof_metal_capture_command_buffer_timestamp(const char* label,
                                                             uint64_t command_buffer_id,
                                                             double* start_us,
                                                             double* end_us,
                                                             const char** error) {
  (void)command_buffer_id;
  if (start_us == nullptr || end_us == nullptr) {
    if (error != nullptr) *error = "timestamp output pointers are required";
    return false;
  }
  *start_us = 0.0;
  *end_us = 0.0;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      if (error != nullptr) *error = "MTLCreateSystemDefaultDevice returned nil";
      return false;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      if (error != nullptr) *error = "newCommandQueue returned nil";
      return false;
    }
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
      if (error != nullptr) *error = "commandBuffer returned nil";
      return false;
    }
    if (label != nullptr) {
      commandBuffer.label = [NSString stringWithUTF8String:label];
    }

    id<MTLBuffer> source = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> destination = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    if (source == nil || destination == nil) {
      if (error != nullptr) *error = "small Metal shared-buffer allocation failed";
      return false;
    }
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit == nil) {
      if (error != nullptr) *error = "blitCommandEncoder returned nil";
      return false;
    }
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:4];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
      if (error != nullptr) *error = "Metal command buffer did not complete";
      return false;
    }
    *start_us = commandBuffer.GPUStartTime * 1000000.0;
    *end_us = commandBuffer.GPUEndTime * 1000000.0;
    if (*end_us < *start_us) {
      if (error != nullptr) *error = "Metal command-buffer timestamps were not monotonic";
      return false;
    }
    if (error != nullptr) *error = nullptr;
    return true;
  }
}

extern "C" bool tprof_metal_discover_counter_sets(uint64_t* counter_set_count,
                                                  const char** first_counter_set,
                                                  const char** error) {
  if (counter_set_count == nullptr) {
    if (error != nullptr) *error = "counter_set_count output pointer is required";
    return false;
  }
  *counter_set_count = 0;
  if (first_counter_set != nullptr) *first_counter_set = nullptr;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      if (error != nullptr) *error = "MTLCreateSystemDefaultDevice returned nil";
      return false;
    }
    if (![device respondsToSelector:@selector(counterSets)]) {
      if (error != nullptr) *error = "MTLDevice counterSets selector is unavailable";
      return false;
    }
    NSArray<id<MTLCounterSet>>* sets = device.counterSets;
    *counter_set_count = static_cast<uint64_t>(sets.count);
    static std::string first_name;
    if (sets.count > 0 && first_counter_set != nullptr) {
      first_name = std::string(sets[0].name.UTF8String ? sets[0].name.UTF8String : "");
      *first_counter_set = first_name.c_str();
    }
    if (error != nullptr) *error = nullptr;
    return true;
  }
}
