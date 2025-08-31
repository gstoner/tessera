#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TSR_STATUS_SUCCESS = 0,
  TSR_STATUS_INVALID_ARGUMENT = 1,
  TSR_STATUS_NOT_FOUND = 2,
  TSR_STATUS_ALREADY_EXISTS = 3,
  TSR_STATUS_OUT_OF_MEMORY = 4,
  TSR_STATUS_UNIMPLEMENTED = 5,
  TSR_STATUS_INTERNAL = 6,
  TSR_STATUS_DEVICE_ERROR = 7
} TsrStatus;

const char* tsrStatusString(TsrStatus status);

// Returns a thread-local human-readable description of the last error (if any).
// The returned pointer is valid until the next tsr call on this thread.
const char* tsrGetLastError(void);
void tsrClearLastError(void);

#ifdef __cplusplus
} // extern "C"
#endif
