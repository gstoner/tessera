// Test-only observer shares the exact production metadata planner.
// This translation unit substitutes for matmul_prepared.cpp only in tests.
#include "../../../src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp"

// Test-only transport observer; never exported by the shipped provider.
extern "C" int tessera_nvidia_native_frame_test_metadata_v1(
    const TesseraNvidiaMatmulHostView *buffers,const uint8_t *roles,size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps,size_t stageCount,
    int64_t *firstScalar,uint64_t *arenaBytes) {
  error.clear();
  if (firstScalar) *firstScalar=0;
  if (arenaBytes) *arenaBytes=0;
  if (!firstScalar || !arenaBytes || getpid()!=process)
    return bad("invalid native frame metadata observer output/process");
  std::lock_guard<std::mutex> lock(mutex);
  try {
    NativeFrameOwner plan;
    if (int status=validateNativeFrameMetadata(
            buffers,roles,count,steps,stageCount,plan)) return status;
    if (!plan.stages.front().scalarCount)
      return bad("native frame metadata observer requires a scalar");
    *firstScalar=plan.stages.front().scalars.front();
    *arenaBytes=plan.bytes;
    return 0;
  } catch (...) {return bad("native frame metadata observer failed");}
}
