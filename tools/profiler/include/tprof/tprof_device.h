#pragma once
#ifdef __cplusplus
#include "tprof/tprof_runtime.h"
#define TSPR_RANGE_PUSH(name)  ::tprof::push(name)
#define TSPR_RANGE_POP()       ::tprof::pop()
#define TSPR_MARKER(name)      ::tprof::marker(name)
#define TSPR_COUNTER_ADD(k,v)  ::tprof::counter_add(k, static_cast<double>(v))
#define TSPR_RUNTIME_API(name,args) ::tprof::runtime_api(name, args)
#define TSPR_DEVICE_ACTIVITY(name,duration_us,args) ::tprof::device_activity(name, static_cast<double>(duration_us), args)
#define TSPR_INTRA_KERNEL_SAMPLE(name,v,args) ::tprof::intra_kernel_sample(name, static_cast<double>(v), args)
#else
#define TSPR_RANGE_PUSH(name)  ((void)0)
#define TSPR_RANGE_POP()       ((void)0)
#define TSPR_MARKER(name)      ((void)0)
#define TSPR_COUNTER_ADD(k,v)  ((void)0)
#define TSPR_RUNTIME_API(name,args) ((void)0)
#define TSPR_DEVICE_ACTIVITY(name,duration_us,args) ((void)0)
#define TSPR_INTRA_KERNEL_SAMPLE(name,v,args) ((void)0)
#endif
