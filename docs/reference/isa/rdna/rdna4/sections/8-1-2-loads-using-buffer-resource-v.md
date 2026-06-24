# 8.1.2. Loads using Buffer Resource (V#)

> RDNA4 ISA — pages 108–108

Opcode # Name                          Opcode # Name
5          S_LOAD_B96                  26       S_BUFFER_LOAD_I16
8          S_LOAD_I8                   27       S_BUFFER_LOAD_U16
9          S_LOAD_U8                   33       S_DCACHE_INV
10         S_LOAD_I16                  36       S_PREFETCH_INST
11         S_LOAD_U16                  37       S_PREFETCH_INST_PC_REL
16         S_BUFFER_LOAD_B32           38       S_PREFETCH_DATA
17         S_BUFFER_LOAD_B64           39       S_BUFFER_PREFETCH_DATA
18         S_BUFFER_LOAD_B128          40       S_PREFETCH_DATA_PC_REL

These instructions load 1-16 DWORDs from memory. The SDATA field indicates which SGPRs to load data into,
and the address is composed of the SBASE, OFFSET, and SOFFSET fields.

SMEM loads may use NULL as a destination SGPR in order to achieve a "prefetch data with acknowledge".

8.1.1. Scalar Memory Addressing
Non-buffer S_LOAD instructions use the following formula to calculate the memory address:

     ADDR = SGPR[base] + IOFFSET + { M0 or SGPR[offset] or zero }

All components of the address (base, offset, IOFFSET, M0) are in bytes, but the two LSBs of each component
are ignored and treated as if they were zero except for 8- and 16-bit loads. For 16-bit loads the one LSB of the
each component is ignored and treated as zero. For 8-bit loads the full byte address of each component is used.

For S_LOAD (non-buffer) instructions, it is illegal and undefined if (IOFFSET + (M0, SGPR[offset], or zero)) is
negative.

For S_BUFFER_LOAD, it is illegal and undefined for IOFFSET to be negative.

8.1.2. Loads using Buffer Resource (V#)
S_BUFFER_LOAD instructions use a similar formula, but the base address comes from the buffer resource’s
base_address field.

Buffer resource fields used: base_address, stride, num_records; other fields are ignored.

Scalar memory load does not support "swizzled" buffers. Stride is used only for memory address bounds
checking, not for computing the address to access. Stride and buffer size must be a multiple of 4.

The SMEM supplies only a SBASE address (byte) and an offset (byte). Any "index * stride" must be calculated
manually in shader code and added to the offset prior to the SMEM. IOFFSET must be non-negative - a negative
value of IOFFSET results in a MEMVIOL.

The two LSBs of V#.base are ignored and treated as if they were zero except for 8- and 16-bit loads. For 16-bit
loads the one LSB of V#.base is ignored and treated as zero. For 8-bit loads the full byte address of V#.base is
used.
