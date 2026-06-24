# 1.2.2. Cache System Hierarchy

> RDNA4 ISA — pages 17–17

When it receives a request, the WGP processor pipeline loads instructions and data from memory, begins
execution, and continues until the end of the kernel. As kernels are running, the processor array hardware
automatically fetches instructions from memory into on-chip caches; software plays no role in this. Kernels
can load data from off-chip memory into on-chip general-purpose registers (GPRs) and caches.

The RDNA4 devices can detect floating point exceptions in hardware and can generate interrupts to the host.
These exceptions can be recorded for post-execution analysis.

The RDNA4 hides memory latency by keeping track of potentially hundreds of work-items in various stages of
execution, and by overlapping compute operations with memory-access operations.

1.2.2. Cache System Hierarchy
The memory system is divided into a number of levels of hierarchy. The figure below shows the memory
hierarchy that is available to each work-item. The actual number of GPRs may differ from what is shown in the
image below.

                                       Figure 2. Shared Memory Hierarchy

"R/W" = read/write cache.

1.2.2.1. Local Data Share (LDS)
Work-items within a work-group may share data with other work-items in the same work-group through the
cache-memory system, or through the local shared memory (LDS).

Each work-group processor (WGP) has a 128kB memory space that enables low-latency communication
between work-items within a work-group, or the work-items within a wave; this is the local data share (LDS).
This memory is configured with 64 banks, each with 512 entries of 4 bytes. The shared memory contains 64
integer atomic units to enable fast, unordered atomic operations. This memory can be used as a software cache
for predictable re-use of data, a data exchange machine for the work-items of a work-group, or as a cooperative
