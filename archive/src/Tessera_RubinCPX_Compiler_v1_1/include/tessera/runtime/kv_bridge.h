
#pragma once
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { void* impl; } tessera_kv_channel;
typedef enum { TESSERA_KV_POLICY_PCIE_CX9 = 0, TESSERA_KV_POLICY_NVLINK = 1 } tessera_kv_policy;

tessera_kv_channel* tessera_kv_open(tessera_kv_policy policy, int device_src, int device_dst,
                                    uint64_t chunk_bytes, int num_inflight);
void tessera_kv_close(tessera_kv_channel*);
int tessera_kv_send(tessera_kv_channel*, const void* src, uint64_t bytes, uint64_t tag);
int tessera_kv_recv(tessera_kv_channel*, void* dst, uint64_t bytes, uint64_t* out_tag);
void tessera_kv_set_prefetch_distance(tessera_kv_channel*, int distance);

#ifdef __cplusplus
}
#endif
