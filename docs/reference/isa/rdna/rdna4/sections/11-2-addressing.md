# 11.2. Addressing

> RDNA4 ISA — pages 149–149

Since these instructions do not access LDS, only LOADcnt (or STOREcnt) is used, not DScnt. It is not possible
for a scratch instruction to access LDS, and so no error checking is done (and no aperture check is performed).

11.2. Addressing
Global, Flat and Scratch each have their own addressing modes. Flat addressing is a subset of the global and
scratch modes. 64-bit addresses are stored with the LSB’s in the VGPR at ADDR, and the MSBs in the VGPR at
ADDR+1.

There are 4 distinct shader instructions:
  • GLOBAL
  • SCRATCH
  • LDS
  • FLAT - based on per-thread address (VGPR), can load/store: global memory, LDS or scratch memory.

                         Table 68. Selecting Addressing Mode
Instruction Type           Instruction Modes           SVE                    SADDR
Scratch                    SV                          1                      NULL
                           SS                          0                      ! NULL
                           ST                          0                      NULL
                           SVS                         1                      ! NULL
Flat and Global            GV                          ignored                NULL
                           GT                          Global only: Indicated by opcode
LDS                        LDS                         Indicated by opcode

Global Addressing
GV         mem_addr = VGPRU64 + IOFFSETI24
GVS        mem_addr = SGPRU64 + VGPRU32 + IOFFSETI24
GT         mem_addr = SGPRU64 + IOFFSETI24 + ThreadID*4

LDS Addressing (DS ops)
LDS        LDS_ADDR = VGPR_addrU32 + IOFFSETU16
           LDS address is relative to the LDS space allocated to this wave.

Scratch Addressing
SV         mem_addr = SCRATCH_BASEU64 + SWIZZLE(VGPR_offsetI32 + IOFFSETI24, ThreadID)
SS         mem_addr = SCRATCH_BASEU64 + SWIZZLE(SGPR_offsetI32 + IOFFSETI24, ThreadID)
SVS        mem_addr = SCRATCH_BASEU64 + SWIZZLE(SGPR_offsetI32 + VGPR_offsetI32 + IOFFSETI24, ThreadID)
ST         mem_addr = SCRATCH_BASEU64 + SWIZZLE(IOFFSETI24, ThreadID)

The combined offsets inside SWIZZLE() must result in a non-negative number.
The offset value from SGPRs and VGPRs are signed 32-bit byte offsets.
IOFFSET: In Scratch ST mode, the IOFFSET must be aligned to the payload size: 4 byte aligned for 1-DWORD,
16-byte aligned for 4-DWORD.

For reference, BUFFER_LOAD when using swizzled-addressing is:
