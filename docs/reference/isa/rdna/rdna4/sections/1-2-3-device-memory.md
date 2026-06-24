# 1.2.3. Device Memory

> RDNA4 ISA — pages 18–18

way to enable efficient access to off-chip memory. A single work-group may allocate up to 64kB of LDS space.

1.2.3. Device Memory
The AMD RDNA4 devices offer several methods for access to off-chip memory from the processing elements
(PE) within each WGP. On the primary read path, the device consists of multiple channels of L2 cache that
provides data to read-only L1 caches, and finally to L0 caches per WGP. The memory cache is formed by two
levels of cache: the first-level (GL1) for write-combining cache (collect scatter and store operations and
combine them to provide good access patterns to memory); the second (L2) is a read/write cache with atomic
units that lets each processing element complete unordered atomic accesses that return the initial value.
Specific cache-less load instructions can force data to be retrieved from device memory during an execution of
a load clause.

The instruction cache provides shader instructions to each SIMD, and the constant cache provides access to
scalar constants. Both of these caches are read-only.

Each processing element provides the destination address on which the atomic operation acts, the data to be
used in the atomic operation, and a return address for the read/write atomic unit to store the pre-op value in
memory.

Each store or atomic operation can receive an acknowledgment from the cache system to the requesting PE
upon completing the write at the requested cache scope, or for atomics-with-return value, upon receiving the
"pre-op" value from memory. This acknowledgment enables one processing element to implement a fence to
maintain serial consistency by ensuring all writes have been posted to memory prior to completing a
subsequent write. In this manner, the system can maintain a relaxed consistency model between all parallel
work-items operating on the system. Each scatter write from a given PE to a given memory channel maintains
order.
