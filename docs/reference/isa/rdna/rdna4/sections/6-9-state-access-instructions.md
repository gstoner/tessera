# 6.9. State Access Instructions

> RDNA4 ISA — pages 74–75

S_FLOOR_F16
S_TRUNC_F16
S_RNDNE_F16

The S_CMP ops set SCC as usual (SCC=1 indicates comparison condition is met); other ops do not set SCC.

S_CVT_HI_F32_F16 does not have an associated VALU counterpart instruction - it is a variant of
S_CVT_F32_F16 to convert the upper 16 bits of the SGPR source from F16 to F32.

Rounding and denormal handling follow the MODE.round and MODE.denorm settings.

These scalar floating point arithmetic instructions can trigger floating point exceptions. These exceptions are
handled in the same manner as exceptions occurring in the VALU pipe.

Scalar F16 instructions do not support encoding half SGPRs in their source/destination operand fields. All
scalar F16 instructions operate on the low part (bit[15:0]) of the SGPR specified, and set the high part
(bit[31:16]) of its SGPR destination to 0.

6.9. State Access Instructions
These instructions access hardware internal registers.

                                            Table 33. Hardware Internal Registers
Instruction                      Encoding     Sets       Operation
                                              SCC?
S_GETREG_B32                     SOPK         No         Read a hardware register into the LSBs of SDST.
S_SETREG_B32                     SOPK         No         Write the LSBs of SDST into a hardware register. (Note that SDST is
                                                         used as a source SGPR).
S_SETREG_IMM32_B32               SOPK         No         S_SETREG where 32-bit data comes from a literal constant (so this is
                                                         a 64-bit instruction format).
                                 GETREG/SETREG : #SIMM16 = { Size[4:0], Offset[4:0], hwRegId[5:0] }
                                  Offset is 0..31. Size is 1..32.

For hardware register index values, see Hardware Registers .

Note that S_SETREG should only write entire register fields, not partial fields. _Each bit of MODE.FP_ROUND
and FP_DENORM may be set individually.

6.10. Memory Aperture Query
Shaders can query the memory aperture base and size for shared and private space through scalar operands:
  • PRIVATE_BASE
  • PRIVATE_LIMIT
  • SHARED_BASE
  • SHARED_LIMIT

These values originate from the SH_MEM_BASES register ("SMB"), and are used primarily with FLAT memory
instructions. Setting Shared Base or Private (scratch) Base to zero disables that aperture. The table below shows

that starting address ("base") and ending address ("limit") of each of: memory segment0, memory segment1,
and the "hole" in the address map.

                                              Limit        0xFFFF_FFFF_FFFF_FFFF
                                       Seg1
                                              Base           0xFFFF_8000_0000_0000

                                              Limit         0xFFFF_7FFF_FFFF_FFFF
                                       Hole
                                              Base            0x0000_8000_0000_0000

                                              Limit         0x0000_7FFF_FFFF_FFFF
                                       Seg0
                                              Base            0x0000_0000_0000_0000

These constants can be used by SALU and VALU ops, and are 64-bit unsigned integers:

      SHARED_BASE    = {SMB.shared_base[15:0],        48'h000000000000}
      SHARED_LIMIT   = {SMB.shared_base[15:0],        48'h0000FFFFFFFF}
      PRIVATE_BASE   = {SMB.private_base[15:0], 48'h000000000000}
      PRIVATE_LIMIT = {SMB.private_base[15:0], 48'h0000FFFFFFFF}

      "Hole" = (addr[63:47] != all zeros or all ones and not in the shared or private aperture)   and is the
  illegal address section of memory
