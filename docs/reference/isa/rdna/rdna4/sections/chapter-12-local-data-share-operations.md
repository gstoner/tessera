# Chapter 12. Local Data Share Operations

> RDNA4 ISA — pages 154–154

Chapter 12. Local Data Share Operations
Local data store (LDS) is a low-latency, RAM scratchpad for temporary data storage and for sharing data
between threads within a work-group. Accessing data through LDS may be significantly lower latency and
higher bandwidth than going through memory.

For compute workloads, it allows a simple method to pass data between threads in different waves within the
same work-group. For graphics, it is also used to hold vertex parameters for pixel shaders.

The high bandwidth of the LDS memory is achieved not only through its proximity to the ALUs, but also
through simultaneous access to its memory banks. Thus, it is possible to concurrently execute many store or
load operations simultaneously. If, however, more than one access attempt is made to the same bank at the
same time, a bank conflict occurs. In this case, for indexed and atomic operations, the hardware is designed to
prevent the attempted concurrent accesses to the same bank by turning them into serial accesses. This can
decrease the effective bandwidth of the LDS. For increased throughput (optimal efficiency), therefore, it is
important to avoid bank conflicts. A knowledge of request scheduling and address mapping can be key to help
achieving this.

Data can be loaded into LDS either by transferring it from VGPRs to LDS using "DS" instructions, or by loading
in from memory. When loading from memory, the data may be loaded into VGPRs first or for some types of
loads it may be loaded directly into LDS from memory. To store data from LDS to global memory, data is read
from LDS and placed into the work-item’s VGPRs, then written out to global memory. To make effective use of
the LDS, a shader program must perform many operations on what is transferred between global memory and
LDS.

LDS space is allocated per work-group or wave (when work-groups not used) and recorded in dedicated LDS-
base/size (allocation) registers that are not writable by the shader. These restrict all LDS accesses to the space
owned by the work-group or wave.

12.1. Overview
There are 128kB of memory per work-group processor split up into 64 banks of DWORD-wide RAMs. These 64
banks are further sub-divided into two sets of 32-banks each where 32 of the banks are affiliated with a pair of
SIMD32’s, and the other 32 banks are affiliated with the other pair of SIMD32’s within the WGP. Each bank is a
512x32 two-port RAM (1R/1W per clock cycle). DWORDs are placed in the banks serially, but all banks can
execute a store or load simultaneously. One work-group can request up to 64kB memory.

LDS atomics are performed in the LDS hardware. Although ALUs are not directly used for these operations,
latency is incurred by the LDS executing this function.

12.1.1. LDS Modes and Allocation: CU vs. WGP Mode
Work-groups of waves are dispatched in one of two modes: CU or WGP. See this section for details: WGP and
CU Mode
