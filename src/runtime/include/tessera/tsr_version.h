#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TESSERA_VERSION_MAJOR 0
#define TESSERA_VERSION_MINOR 2
#define TESSERA_VERSION_PATCH 0

void tsrGetVersion(int* major, int* minor, int* patch);

// Profiling control and timing API
void tsrEnableProfiling(int enable);
uint64_t tsrTimestampNowNs(void);

#ifdef __cplusplus
} // extern "C"
#endif
