# 15.10. Flat, Global and Scratch Formats

> RDNA4 ISA — pages 212–212

15.10. Flat, Global and Scratch Formats
                                    Flat memory instructions come in three versions:
FLAT
     memory address (per work-item) may be in global memory, scratch (private) memory or shared memory
     (LDS)

GLOBAL
     same as FLAT, but assumes all memory addresses are global memory.

SCRATCH
     same as FLAT, but assumes all memory addresses are scratch (private) memory.

The microcode format is identical for each, and only the value of the SEG (segment) field differs.

  Description       FLAT Memory Access

                                    Table 117. FLAT, GLOBAL and SCRATCH Fields
Field Name       Bits          Format or Description
SADDR            [6:0]         SGPR that provides an address or offset (signed). Set this field to NULL to disable use.
                               Meaning of this field is different for Scratch and Global:
                               FLAT: NULL=Normal addressing; non-NULL=GVS addressing.
                               Scratch: use an SGPR for the address instead of a VGPR
                               Global: use the SGPR to provide a base address and the VGPR provides a 32-bit byte offset.
OP               [20:13]       Opcode. See tables below for FLAT, SCRATCH and GLOBAL opcodes.
ENCODING         [31:24]       Flat='b11101100, Global='b11101110, Scratch='b11101101
VDST             [39:32]       Destination VGPR for data returned from memory to VGPRs.
SVE              [49]          Scratch VGPR Enable. 1 = scratch address includes a VGPR to provide an offset; 0 = no
                               VGPR used.
SCOPE            [51:50]       Memory Scope
TH               [54:52]       Memory Temporal Hint
VSRC             [62:55]       Source VGPR for data sent from VGPRs to memory.
VADDR            [71:64]       VGPR that holds address or offset. For 64-bit addresses, ADDR has the LSBs and ADDR+1
                               has the MSBs. For offset a single VGPR has a 32 bit unsigned offset.
                               For FLAT_*: specifies an address.
                               For GLOBAL_* and SCRATCH_* when SADDR is NULL or 0x7f: specifies an address.
                               For GLOBAL_* and SCRATCH_* when SADDR is not NULL or 0x7f: specifies an offset.
