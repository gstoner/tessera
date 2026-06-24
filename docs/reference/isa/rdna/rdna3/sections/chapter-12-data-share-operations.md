# Chapter 12. Data Share Operations

> RDNA3 ISA — pages 127–127

Chapter 12. Data Share Operations
Local data share (LDS) is a low-latency, RAM scratchpad for temporary data storage and for sharing data
between threads within a work-group. Accessing data through LDS may be significantly lower latency and
higher bandwidth than going through memory.

For compute workloads, it allows a simple method to pass data between threads in different waves within the
same work-group. For graphics, it is also used to hold vertex parameters for pixel shaders.

LDS space is allocated per work-group or wave (when work-groups not used) and recorded in dedicated LDS-
base/size (allocation) registers that are not writable by the shader. These restrict all LDS accesses to the space
owned by the work-group or wave.

12.1. Overview
The figure below shows how the LDS fits into the memory hierarchy of the GPU.

                                       Figure 3. High-Level Memory Configuration

There are 128kB of memory per work-group processor split up into 64 banks of DWORD-wide RAMs. These 64
banks are further sub-divided into two sets of 32-banks each where 32 of the banks are affiliated with a pair of
SIMD32’s, and the other 32 banks are affiliated with the other pair of SIMD32’s within the WGP. Each bank is a
512x32 two-port RAM (1R/1W per clock cycle). DWORDs are placed in the banks serially, but all banks can
execute a store or load simultaneously. One work-group can request up to 64kB memory.

The high bandwidth of the LDS memory is achieved not only through its proximity to the ALUs, but also
through simultaneous access to its memory banks. Thus, it is possible to concurrently execute 32 store or load
instructions, each nominally 32-bits; extended instructions, load_2addr/store_2addr, can be 64-bits each. If,
