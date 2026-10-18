// Additive completed-call ABI for the native source pullback outline.
#pragma once
#include "tessera_nvidia_active_source_program.h"
// Exactly five typed buffers, three readonly roots, one private half edge and
// one half result; two compiler-authored stages. Step schema=1, kinds=3/4.
// Invocation binds three roots plus one host output. Non-null streams mean
// three resident CUDA roots; completion precedes output commit and reuse.
extern "C" int tessera_nvidia_source_reverse_prepare_v1(
    const TesseraNvidiaMatmulHostView *,size_t,const int64_t *,
    const TesseraNvidiaActiveSourceProgramStep *,size_t,uint64_t *);
extern "C" int tessera_nvidia_source_reverse_invoke_v1(uint64_t,
    const TesseraNvidiaMatmulHostView *,size_t,const uint64_t *,size_t);
extern "C" int tessera_nvidia_source_reverse_profile_v1(uint64_t,
    const TesseraNvidiaMatmulHostView *,size_t,const uint64_t *,size_t,
    int,int,float *,size_t,float *);
extern "C" int tessera_nvidia_source_reverse_close_v1(uint64_t);
