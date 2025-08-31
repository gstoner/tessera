#pragma once
#include <stdint.h>
#include <stddef.h>
#include "tsr_types.h"
#include "tsr_status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates that launch params are reasonable for the given device props.
// Returns TSR_STATUS_SUCCESS if ok; otherwise returns INVALID_ARGUMENT and
// puts a human-readable reason into an internal last-error string.
TsrStatus tsrValidateLaunch(const tsrDeviceProps* props, const tsrLaunchParams* p);

// Suggests a tile size that fits within device limits for a target logical_threads.
// Simple heuristic: tx = min(logical_threads, max_threads), others = 1.
void tsrSuggestTile(const tsrDeviceProps* props, uint32_t logical_threads,
                    tsrDim3* out_tile);

#ifdef __cplusplus
} // extern "C"
#endif
