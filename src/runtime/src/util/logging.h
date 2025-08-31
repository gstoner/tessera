#pragma once
#include <stdio.h>
#ifndef TSR_LOG_LEVEL
#define TSR_LOG_LEVEL 1
#endif
#define TSR_LOGE(...) do { if (TSR_LOG_LEVEL >= 1) { fprintf(stderr, "[TSR][E] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while(0)
#define TSR_LOGW(...) do { if (TSR_LOG_LEVEL >= 2) { fprintf(stderr, "[TSR][W] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while(0)
#define TSR_LOGI(...) do { if (TSR_LOG_LEVEL >= 3) { fprintf(stderr, "[TSR][I] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while(0)
