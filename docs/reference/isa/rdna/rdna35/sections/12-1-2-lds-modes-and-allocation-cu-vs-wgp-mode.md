# 12.1.2. LDS Modes and Allocation: CU vs. WGP Mode

> RDNA3.5 ISA — pages 130–130

however, more than one access attempt is made to the same bank at the same time, a bank conflict occurs. In
this case, for indexed and atomic operations, the hardware is designed to prevent the attempted concurrent
accesses to the same bank by turning them into serial accesses. This can decrease the effective bandwidth of
the LDS. For increased throughput (optimal efficiency), therefore, it is important to avoid bank conflicts. A
knowledge of request scheduling and address mapping can be key to help achieving this.

12.1.1. Dataflow in Memory Hierarchy
The figure below is a conceptual diagram of the dataflow within the memory structure.

Data can be loaded into LDS either by transferring it from VGPRs to LDS using "DS" instructions, or by loading
in from memory. When loading from memory, the data may be loaded into VGPRs first or for some types of
loads it may be loaded directly into LDS from memory. To store data from LDS to global memory, data is read
from LDS and placed into the work-item’s VGPRs, then written out to global memory. To help make effective
use of the LDS, a shader program must perform many operations on what is transferred between global
memory and LDS.

LDS atomics are performed in the LDS hardware. Although ALUs are not directly used for these operations,
latency is incurred by the LDS executing this function.

12.1.2. LDS Modes and Allocation: CU vs. WGP Mode
Work-groups of waves are dispatched in one of two modes: CU or WGP.

See this section for details: WGP and CU Mode

12.1.3. LDS Access Methods
There are 3 forms of Local Data Share access:
