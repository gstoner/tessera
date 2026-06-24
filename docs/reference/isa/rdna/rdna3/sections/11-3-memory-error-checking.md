# 11.3. Memory Error Checking

> RDNA3 ISA — pages 125–125

         Addr = SCRATCH_BASE + (offset / 4) * 4 * const_index_stride + (offset % 4) + TID*4
         where "offset" = either "INST_OFFSET + SGPR_offset" or "INST_OFFSET + VGPR_offset".

Restrictions:
  • Inst_offset :
      ◦ Flat and Scratch-ST mode: must not be negative
      ◦ Global and Scratch-SS and -SV modes: can be negative
      ◦ In Scratch SS mode, the inst_offset must be aligned to the payload size: 4 byte aligned for 1-DWORD,
        16-byte aligned for 4-DWORD.
         ▪ Also (SADDR + INST_OFFSET) must be at least DWORD-aligned

                               SADDR               SVE                    MODE
                               ==NULL              0                      ST
                               !=NULL              0                      SS
                               ==NULL              1                      SV
                               !=NULL              1                      SVS

Scratch Instruction Modes                                                                                 Indicated by SVE
                                                                                                          / SADDR
SV          Addr =   FLAT_SCRATCH                  + swizzle(Voff + Ioff, TID)                            1 / NULL
SS          Addr =   FLAT_SCRATCH                  + swizzle(Soff + Ioff, TID)                            0 / !NULL
ST          Addr =   FLAT_SCRATCH                  + swizzle(0 + Ioff, TID)                               0 / NULL
SVS         Addr =   FLAT_SCRATCH                  + swizzle(Soff + Voff + Ioff, TID)                     1 / !NULL
BUFFER_ Addr =       T#.base            + Soff     + swizzle( (Vidx + TID) * stride + Ioff + Voff)
+ LOAD
Global Instruction Modes
GV          Addr =   Vaddr64                       + Ioff                                                 x / NULL
GVS         Addr =   Saddr64            + Voff32   + Ioff                                                 x / !=NULL
GT          Addr =   Saddr64                       + Ioff + TID*4                                         x/x instruction
LDS Instruction Modes
LDS         Addr =   Vaddr                         + Ioff                                                 x/x instruction
Flat Instruction Modes
Scratch     Addr =   FLAT_SCRATCH                  swizzle (Voff + Ioff -privApertureBase, TID) // "SV"   x / NULL
LDS         Addr =   Vaddr                         + Ioff - sharedApertureBase // "LDS"                   x / NULL
Global      Addr =   Vaddr                         + Ioff // "GV"                                         x / NULL

  • Scratch: Voff and Soff are 32 bits, unsigned bytes.
  • Global: Addresses are 64 bits, offset is 32bits.
  • FLAT_SCRATCH is an SGPR-pair 64-bit address.
  • "Ioff" is the offset from the instruction field.
  • "x" = don’t care (either value works)

11.3. Memory Error Checking
Both Texture and LDS can report that an error occurred due to a bad address. This can occur due to:
  • Invalid address (outside any aperture)
  • Write to read-only global memory address
