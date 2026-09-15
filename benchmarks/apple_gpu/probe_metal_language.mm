// Probe language admission separately from GPU-family and arithmetic execution.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "No Metal device\n"); return 2; }
    NSMutableArray *rows = [NSMutableArray array];
    for (NSNumber *version in @[@((4<<16)), @((4<<16)+1)]) {
      for (NSString *type in @[@"float", @"half", @"bfloat"]) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        // The numeric encoding permits a runtime probe with a pre-4.1 SDK.
        // It does not claim the installed offline compiler supports that SDK.
        options.languageVersion = (MTLLanguageVersion)version.unsignedIntegerValue;
        options.fastMathEnabled = NO;
        NSString *source = [NSString stringWithFormat:
          @"#include <metal_stdlib>\nusing namespace metal;\n"
           "kernel void probe(device const float *in [[buffer(0)]], device float *out [[buffer(1)]]) {"
           "%@ a=%@(in[0]), b=%@(in[1]); out[0]=float(a+b); out[1]=float(a*b); out[2]=float(a/b); }", type,type,type];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        NSMutableDictionary *row = [@{@"language_raw":version, @"type":type,
          @"library_compiled":@(library != nil), @"executed":@NO} mutableCopy];
        if (!library) { row[@"error"] = error.localizedDescription ?: @"unknown"; [rows addObject:row]; continue; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
        if (!pipeline) { row[@"error"] = error.localizedDescription ?: @"unknown"; [rows addObject:row]; continue; }
        float input[2] = {1.5f, 2.0f};
        id<MTLBuffer> in = [device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:3*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:in offset:0 atIndex:0]; [encoder setBuffer:out offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        row[@"command_status"] = @(command.status);
        if (command.status == MTLCommandBufferStatusCompleted) {
          float *values = (float *)out.contents;
          row[@"executed"] = @YES;
          row[@"values"] = @[@(values[0]), @(values[1]), @(values[2])];
          row[@"exact_arithmetic"] = @(values[0]==3.5f && values[1]==3.f && values[2]==.75f);
        } else row[@"error"] = command.error.localizedDescription ?: @"command failed";
        [rows addObject:row];
      }
    }
    NSDictionary *report = @{@"device":device.name, @"registry_id":@(device.registryID),
      @"apple7":@([device supportsFamily:MTLGPUFamilyApple7]),
      @"metal4":@([device supportsFamily:MTLGPUFamilyMetal4]), @"rows":rows};
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    fwrite(json.bytes,1,json.length,stdout); puts("");
  }
}
