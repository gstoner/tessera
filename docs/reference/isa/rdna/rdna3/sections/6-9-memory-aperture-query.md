# 6.9. Memory Aperture Query

> RDNA3 ISA — pages 61–62

Instruction                               Encoding     Sets SCC? Operation
S_{and, or, xor, and_not0,                SOP1         D!=0       Save the EXEC mask, then apply a bit-wise operation
and_not1,or_not0, or_not1, nand, nor,                             to it.
xnor}_SAVEEXEC_{B32,B64}                                          D = EXEC
                                                                  EXEC = S0 <op> EXEC
                                                                  SCC = (EXEC != 0)
                                                                  ("not1" version inverts EXEC)
                                                                  ("not0" version inverts SGPR)
S_{AND_NOT{0,1}_WREXEC_B{32,64}           SOP1         D!=0       NOT0: EXEC, D = ~S0 & EXEC
                                                                  NOT1: EXEC, D = S0 & ~EXEC
                                                                  Both D and EXEC get the same result. SCC = (result !=
                                                                  0). D cannot be EXEC.
S_MOVRELS_{B32,B64}                       SOP1         No         Move a value into an SGPR relative to the value in M0.
S_MOVRELD_{B32,B64}                                               MOVRELS: D = SGPR[S0+M0]
                                                                  MOVRELD: SGPR[D+M0] = S0
                                                                  Index must be even for B64. M0 is an unsigned index.

6.8. Access Instructions
These instructions access hardware internal registers.

                                        Table 25. Hardware Internal Registers
Instruction                  Encoding     Sets       Operation
                                          SCC?
S_GETREG_B32                 SOPK         No         Read a hardware register into the LSBs of SDST.
S_SETREG_B32                 SOPK         No         Write the LSBs of SDST into a hardware register. (Note that SDST is
                                                     used as a source SGPR).
S_SETREG_IMM32_B32           SOPK         No         S_SETREG where 32-bit data comes from a literal constant (so this is
                                                     a 64-bit instruction format).
                             GETREG/SETREG : #SIMM16 = { Size[4:0], Offset[4:0], hwRegId[5:0] }
                              Offset is 0..31. Size is 1..32.
S_ROUND_MODE                 SOPP         No         Set the round mode from an immediate: simm16[3:0]
S_DENORM_MODE                SOPP         No         Set the denorm mode from an immediate: simm16[3:0]

For hardware register index values, see Hardware Registers .

6.9. Memory Aperture Query
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
