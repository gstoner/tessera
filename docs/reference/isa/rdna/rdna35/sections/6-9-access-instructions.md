# 6.9. Access Instructions

> RDNA3.5 ISA — pages 63–64

S_RNDNE_F16

Note: S_CVT_HI_F32_F16 does not have an associated VALU counterpart instruction - it is a variant of
S_CVT_F32_F16 to convert the upper 16 bits of the SGPR source from F16 to F32.

These scalar floating point arithmetic instructions can trigger IEEE float exceptions. These exceptions are
handled in the same manner as exceptions occurring in the VALU pipe.

Scalar F16 instructions do not support encoding half SGPRs in their source/destination operand fields. All
scalar F16 instructions operate on the low part (bit[15:0]) of the SGPR specified, and set the high part
(bit[31:16]) of its SGPR destination to 0.

These instructions have a longer latency through the SALU than previous instructions: floating point ops take 4
cycles, while the other ops take 2 cycles. The SALU preserves instruction order and forwarding and stall as
needed to preserve correct results.

6.9. Access Instructions
These instructions access hardware internal registers.

                                         Table 25. Hardware Internal Registers
Instruction                  Encoding      Sets      Operation
                                           SCC?
S_GETREG_B32                 SOPK          No        Read a hardware register into the LSBs of SDST.
S_SETREG_B32                 SOPK          No        Write the LSBs of SDST into a hardware register. (Note that SDST is
                                                     used as a source SGPR).
S_SETREG_IMM32_B32           SOPK          No        S_SETREG where 32-bit data comes from a literal constant (so this is
                                                     a 64-bit instruction format).
                             GETREG/SETREG : #SIMM16 = { Size[4:0], Offset[4:0], hwRegId[5:0] }
                              Offset is 0..31. Size is 1..32.
S_ROUND_MODE                 SOPP          No        Set the round mode from an immediate: simm16[3:0]
S_DENORM_MODE                SOPP          No        Set the denorm mode from an immediate: simm16[3:0]

For hardware register index values, see Hardware Registers .

6.10. Memory Aperture Query
Shaders can query the memory aperture base and size for shared and private space through scalar operands:
  • PRIVATE_BASE
  • PRIVATE_LIMIT
  • SHARED_BASE
  • SHARED_LIMIT

These values originate from the SH_MEM_BASES register ("SMB"), and are used primarily with FLAT memory
instructions. Setting Shared Base or Private Base to zero disables that aperture.

"PTR32" is short for "Address mode is 32bit", and "SMB" is short for "SH_MEM_BASES". These constants can be
used by SALU and VALU ops, and are 64-bit unsigned integers:

SHARED_BASE = ptr32 ? {32’h0, SMB.shared_base[15:0], 16’h0000} : {SMB.shared_base[15:0], 48’h000000000000}
SHARED_LIMIT = ptr32 ? {32’h0, SMB.shared_base[15:0], 16’hFFFF} : {SMB.shared_base[15:0], 48’h0000FFFFFFFF}
PRIVATE_BASE = ptr32 ? {32’h0, SMB.private_base[15:0], 16’h0000} : {SMB.private_base[15:0], 48’h000000000000}
PRIVATE_LIMIT =ptr32 ? {32’h0, SMB.private_base[15:0], 16’hFFFF} : {SMB.private_base[15:0], 48’h0000FFFFFFFF}

   "Hole" = (addr[63:47] != all zeros or all ones) and is the illegal address section of memory
