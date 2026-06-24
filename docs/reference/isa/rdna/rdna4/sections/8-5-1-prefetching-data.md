# 8.5.1. Prefetching Data

> RDNA4 ISA — pages 111–111

8.5. Scalar Prefetch Instructions
The shader can request that instructions or data be prefetched into the first-level cache via shader instructions.
These do not use KMcnt.

The cache reuse policies are controlled through the SCOPE and TH bits and apply to data-prefetch but not
instruction prefetch.

These instructions are skipped (treated as S_NOP) if MODE.SCALAR_PREFETCH_EN == 0.

Prefetch Length: SOFFSET holds an SGPR/M0 (int) with the length, or set to NULL for zero, and SDATA carries
a literal constant value (set to 0 to ignore). The sum is 0-31 (modulo 32 - upper bits dropped) representing a
prefetch size of 1-32 chunks of 128Bytes.

All components of the address (base, offset, IOFFSET, M0) are in bytes, but the two LSBs of each component
are ignored and treated as if they were zero.

8.5.1. Prefetching Data
These instructions prefetch data into the constant cache using either a simple address, a buffer resource (V#),
or PC-relative address.

 S_PREFETCH_DATA     SBASE(addr)    IOFFSET   SOFFSET(length)   SDATA(as immediate value)
           MemAddr = (SBASE[63:0].u64 + IOFFSET.i24) & ~0x07f            Forced to 128B alignment
           Length    = SOFFSET + #SDATA : SGPR/M0 + immediate, limit to 1-32 chunks, units of 128B
 S_BUFFER_PREFETCH_DATA     SBASE(V#)    IOFFSET   SOFFSET(length)
           MemAddr = (SBASE[47:0].u64 + IOFFSET.i24) & ~0x07f
           Length    = SOFFSET + #SDATA : SGPR/M0 + immediate, limit to 1-32 cacheline-pairs, units of 128B
           Note: S_BUFFER_PREFETCH_DATA does not perform bounds checking on the prefetch address.
 S_PREFETCH_DATA_PC_REL    IOFFSET    SOFFSET(length)
           Addr     = (PC + IOFFSET.i24) & ~0x07f.      Forced to 128B alignment.
           Length = SOFFSET + #SDATA : SGPR/M0 + immediate, limit to 1-32 chunks, units of 128B
           "PC" is pointing to the instruction after this one when the address is computed.

Note: S_BUFFER_PREFETCH_DATA does not perform bounds-checking on the prefetch address.
S_BUFFER_PREFETCH_DATA does not support negative IOFFSET, but does not return MEMVIOL if IOFFSET is
negative (and the prefetch request is dropped).

8.5.2. Prefetching Instructions
These instructions prefetch instructions into the instruction cache using either a simple address, or PC-relative
address.

 S_PREFETCH_INST     SBASE(addr)    IOFFSET   SOFFSET(length)   SDATA(as immediate value)
           MemAddr = (SBASE[63:0].u64 + IOFFSET.i24) & ~0x07f            Forced to 128B alignment
           Length    = SOFFSET + #SDATA : SGPR/M0 + immediate, limit to 1-32 chunks, units of 128B
 S_PREFETCH_INST_PC_REL    IOFFSET    SOFFSET(length)    SDATA(as immediate value)
           Addr     = (PC + IOFFSET.i24) & ~0x07f.       Forced to 128B alignment.   SBASE is ignored.
           Length = SOFFSET + #SDATA : SGPR/M0 + immediate, limit to 1-32 chunks, units of 128B
           "PC" is pointing to the instruction after this one when the address is computed.
