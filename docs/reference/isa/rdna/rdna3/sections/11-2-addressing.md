# 11.2. Addressing

> RDNA3 ISA — pages 124–124

Since these instructions do not access LDS, only VMcnt (or VScnt) is used, not LGKMcnt. It is not possible for a
scratch instruction to access LDS, and so no error checking is done (and no aperture check is performed).

11.2. Addressing
Global, Flat and Scratch each have their own addressing modes. Flat addressing is a subset of the global and
scratch modes. 64-bit addresses are stored with the LSB’s in the VGPR at ADDR, and the MSBs in the VGPR at
ADDR+1.

There are 4 distinct shader instructions:
  • GLOBAL
  • SCRATCH
  • LDS
  • FLAT - based on per-thread address (VGPR), can load/store: global memory, LDS or scratch memory.

Global Addressing
GV          mem_addr = VGPRU64 + INST_OFFSETI13
GVS         mem_addr = SGPRU64 + VGPRU32 + INST_OFFSETI13
GT          mem_addr = SGPRU64 + INST_OFFSETI13 + ThreadID*4

LDS Addressing (DS ops)
LDS         LDS_ADDR = VGPR_addrU32 + INST_OFFSETU16
            LDS address is relative to the LDS space allocated to this wave.

Scratch Addressing
SV          mem_addr = SCRATCH_BASEU64 + SWIZZLE(VGPR_offsetU32 + INST_OFFSETI13, ThreadID)
SS          mem_addr = SCRATCH_BASEU64 + SWIZZLE(SGPR_offsetU32 + INST_OFFSETI13, ThreadID)
SVS         mem_addr = SCRATCH_BASEU64 + SWIZZLE(SGPR_offsetU32 + VGPR_offsetU32 + INST_OFFSETI13, ThreadID)
ST          mem_addr = SCRATCH_BASEU64 + SWIZZLE(INST_OFFSETI13, ThreadID)
            SGPR_offset and VGPR_offset are 32 bits unsigned byte offsets.

The combined offsets inside SWIZZLE() must result in a non-negative number.
The value from an SGPR and VGPR are unsigned 32-bit byte offsets.

Flat Addressing
                   Aperture test on the address-VGPR value determines: Global/LDS/Scratch per thread (ignores
                   INST_OFFSET).
                   Use one of the 3 address equations per lane depending on which memory it maps to:
GLOBAL (GV)        mem_addr = VGPRU64 + INST_OFFSETI13
SCRATCH (SV)       mem_addr = SCRATCH_BASE(sgpr:U64) + SWIZZLE(VGPR_offset + INST_OFFSETI13, ThreadID)
LDS                LDS_ADDR = VGPR(addr) + INST_OFFSET - sharedApertureBase
                   If the address falls into LDS space, it is checked against the range: [0, LDS_allocated_size-1 ]

There is no range checking on this address.

Scratch Addressing Equation

     "SWIZZLE(offset,TID)" is hard coded based on wave size (32 or 64)
        Swizzle for Scratch is hard-coded to: elem_size=4bytes, const_index_stride=32 (wave32) or 64
        (wave64).
