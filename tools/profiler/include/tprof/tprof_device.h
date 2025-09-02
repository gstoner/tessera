#pragma once
#ifdef __cplusplus
#include "tprof/tprof_runtime.h"
#define TSPR_RANGE_PUSH(name)  ::tprof::push(name)
#define TSPR_RANGE_POP()       ::tprof::pop()
#define TSPR_MARKER(name)      ::tprof::marker(name)
#define TSPR_COUNTER_ADD(k,v)  ::tprof::counter_add(k, static_cast<double>(v))
#else
#define TSPR_RANGE_PUSH(name)  ((void)0)
#define TSPR_RANGE_POP()       ((void)0)
#define TSPR_MARKER(name)      ((void)0)
#define TSPR_COUNTER_ADD(k,v)  ((void)0)
#endif
