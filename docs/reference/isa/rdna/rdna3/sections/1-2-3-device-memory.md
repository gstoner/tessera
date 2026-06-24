# 1.2.3. Device Memory

> RDNA3 ISA — pages 16–17

                                       Figure 2. Shared Memory Hierarchy

1.2.2.1. Local Data Share (LDS)
Each work-group processor (WGP) has a 128kB memory space that enables low-latency communication
between work-items within a work-group, or the work-items within a wave; this is the local data share (LDS).
This memory is configured with 64 banks, each with 512 entries of 4 bytes. The shared memory contains 64
integer atomic units to enable fast, unordered atomic operations. This memory can be used as a software cache
for predictable re-use of data, a data exchange machine for the work-items of a work-group, or as a cooperative
way to enable efficient access to off-chip memory. A single work-group may allocate up to 64kB of LDS space.

1.2.2.2. Global Data Share (GDS)
The AMD RDNA3 devices use a 4kB global data share (GDS) memory that can be used by waves of a kernel on
all WGPs. This memory provides 128 bytes per cycle of memory access to all the processing elements. It
provides full access to any location for any processor. The shared memory contains 2 integer atomic units to
enable fast, unordered atomic operations. This memory can be used as a software cache to store important
control data for compute kernels, reduction operations, or a small global shared surface. Data can be
preloaded from memory prior to kernel launch and written to memory after kernel completion. The GDS block
contains support logic for unordered append/consume and domain launch ordered append/consume
operations to buffers in memory. These dedicated circuits enable fast compaction of data or the creation of
complex data structures in memory.

1.2.3. Device Memory
The AMD RDNA3 devices offer several methods for access to off-chip memory from the processing elements

(PE) within each WGP. On the primary read path, the device consists of multiple channels of L2 cache that
provides data to read-only L1 caches, and finally to L0 caches per WGP. Specific cache-less load instructions
can force data to be retrieved from device memory during an execution of a load clause. Load requests that
overlap within the clause are cached with respect to each other. The output cache is formed by two levels of
cache: the first for write-combining cache (collect scatter and store operations and combine them to provide
good access patterns to memory); the second is a read/write cache with atomic units that lets each processing
element complete unordered atomic accesses that return the initial value. Each processing element provides
the destination address on which the atomic operation acts, the data to be used in the atomic operation, and a
return address for the read/write atomic unit to store the pre-op value in memory. Each store or atomic
operation can be set up to return an acknowledgment to the requesting PE upon write confirmation of the
return value (pre-atomic op value at destination) being stored to device memory.

This acknowledgment has two purposes:
  • enabling a PE to recover the pre-op value from an atomic operation by performing a cache-less load from
    its return address after receipt of the write confirmation acknowledgment, and
  • enabling the system to maintain a relaxed consistency model.

Each scatter write from a given PE to a given memory channel maintains order. The acknowledgment enables
one processing element to implement a fence to maintain serial consistency by ensuring all writes have been
posted to memory prior to completing a subsequent write. In this manner, the system can maintain a relaxed
consistency model between all parallel work-items operating on the system.
