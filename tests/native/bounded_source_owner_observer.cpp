// Test-only observer compiled instead of attention_jvp_prepared.cpp.
// Reads immutable allocation bindings and active value geometry under the same
// mutex. No symbols from this file belong to the shipped provider ABI.
#include "../../src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp"

extern "C" int tessera_test_source_owner_state(uint64_t handle,uint64_t *out,size_t count) {
  if(getpid()!=ownerProcess || !out || count!=71)return 1;
  std::lock_guard<std::mutex> lock(ownerMutex);
  auto found=owners.find(handle);
  if(found==owners.end() || !found->second->sourceBounded)return 1;
  const auto &s=*found->second;
  size_t index=0;
  out[index++]=s.arena;out[index++]=s.prefixOutput;out[index++]=s.sourceGradient;
  out[index++]=s.arenaBytes;out[index++]=reinterpret_cast<uintptr_t>(s.hostArena);
  for(auto value:s.buffers)out[index++]=value;
  for(auto value:s.hostOffsets)out[index++]=value;
  for(auto value:s.capacityBytes)out[index++]=value;
  for(auto value:s.dims)out[index++]=uint64_t(value);
  for(auto value:s.bytes)out[index++]=value;
  out[index++]=uint64_t(s.prefixRows);out[index++]=uint64_t(s.prefixColumns);
  out[index++]=s.forwardGrid;out[index++]=s.reverseGrid;out[index++]=s.rowGrid;
  out[index++]=s.prefixGrid;out[index++]=s.sourcePullbackGrid;
  return index==count?0:1;
}
