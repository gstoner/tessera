#ifndef TESSERA_EXCEPTION_HEAP_H
#define TESSERA_EXCEPTION_HEAP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Host-only bounded native heap. Handle 0 is null; recycled slots get a new
 * generation. No borrowed pointers escape: read copies payload to caller memory.
 * 0 success, 1 invalid argument/stale handle, 2 exhausted, 3 allocation failure.
 * Callers must stop all API users before destroy. Collection follows roots;
 * unrooted handles must be rooted before explicit collection if still needed. */
typedef struct tsr_exception_heap tsr_exception_heap;
int tsr_exception_heap_create(uint32_t nodes, uint32_t bytes, tsr_exception_heap **out);
void tsr_exception_heap_destroy(tsr_exception_heap *heap);
int tsr_exception_heap_alloc(tsr_exception_heap *heap, uint32_t kind,
    const void *payload, uint32_t size, uint64_t cause, uint64_t context,
    int rooted, uint64_t *out);
int tsr_exception_heap_edges(tsr_exception_heap *heap, uint64_t handle,
    uint64_t cause, uint64_t context);
int tsr_exception_heap_root(tsr_exception_heap *heap, uint64_t handle, int rooted);
int tsr_exception_heap_collect(tsr_exception_heap *heap, uint32_t *collected);
int tsr_exception_heap_read(tsr_exception_heap *heap, uint64_t handle,
    uint32_t *kind, uint64_t *cause, uint64_t *context, void *payload,
    uint32_t capacity, uint32_t *size);
#ifdef __cplusplus
}
#endif
#endif
