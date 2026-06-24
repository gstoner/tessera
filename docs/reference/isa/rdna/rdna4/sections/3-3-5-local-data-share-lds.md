# 3.3.5. Local Data Share (LDS)

> RDNA4 ISA — pages 30–30

3.3.4. Memory Alignment and Out-of-Range Behavior
This section defines the behavior when a source or destination GPR or memory address is outside the legal
range for a wave. Except where noted, these rules apply to LDS, buffer, global, flat and scratch memory
accesses.

Memory, LDS: Loads and Atomics with return:
  • If any source VGPR or SGPR is out-of-range, the data value is undefined.
  • If any destination VGPR is out-of-range, the operation is nullified by issuing the instruction as if the EXEC
    mask were cleared to 0.
       ◦ This out-of-range test checks all VGPRs that could be returned (e.g. VDST to VDST+3 for a
         BUFFER_LOAD_B128)
       ◦ This check also includes the extra PRT (partially resident texture) VGPR and nullifies the fetch if this
         VGPR would be out of range no matter whether the texture system actually returns this value or not.
       ◦ Atomic operations with out-of-range destination VGPRs are nullified: issued, but with EXEC mask of
         zero.
  • Image loads and stores consider DMASK bits when making an out-of-bounds determination.
  • Note: VDST is only checked for lds/mem-atomic that actually return a value.

VMEM memory alignment rules are defined using the config register: SH_MEM_CONFIG.alignment_mode.
This setting also affects LDS, Flat/Scratch/Global operations.

  DWORD              Automatic alignment to multiple of the smaller of element size or a DWORD.

  UNALIGNED          No alignment requirements, except for atomics.

Formatted ops such as BUFFER_LOAD_FORMAT_* must be aligned as follows:
  • 1-byte formats require 1-byte alignment
  • 2-byte formats require 2-byte alignment
  • 4-byte and larger formats require 4-byte alignment

Atomics must be aligned to the data size, or they trigger a MEMVIOL.

3.3.5. Local Data Share (LDS)
LDS is a scratch-pad memory allocated to waves or workgroups (in which case it is sometimes referred to as
"shared memory"). Waves may be allocated LDS memory, and waves in a work-group all share the same LDS
memory allocation. All accesses to LDS are restricted to the space allocated to that wave/work-group.

A wave may have 0 - 64kbyte of LDS space allocated, and it is allocated in blocks of 1024 bytes.

Internally LDS is composed of two blocks of memory of 64kB each. Each one of these two blocks is affiliated
with one CU or the other: byte addresses 0-65535 with CU0, 65536-131071 with CU1. Allocations of LDS space to
a wave or work-group do not wrap around: the allocation starting address is less than the ending address.

In CU mode, a wave’s entire LDS allocation resides in the same "side" of LDS as the wave is loaded. No access is
allowed to cross over or wrap around to the other side.

In WGP mode, a wave’s LDS allocation may be entirely in either the CU0 or CU1 part of LDS, or it may straddle
