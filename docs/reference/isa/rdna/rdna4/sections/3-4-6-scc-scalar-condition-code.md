# 3.4.6. SCC: Scalar Condition Code

> RDNA4 ISA — pages 35–35

Operation                   M0 Contents                      Notes
DS_PARAM_LOAD               { 1’b0, new_prim_mask[15:1],     Offset is in bytes and offset[6:0] must be zero.
                            parameter_offset[15:0] }         Wave32: new_prim_mask is {8’b0, mask[7:1] }
DS_DIRECT_LOAD              { 13’b0, DataType[2:0],          address is in bytes
                            LDS_address[15:0] }
EXPORT                      Row number for mesh shader POS   See Export chapter
                            & Param exports
S_SENDMSG / _RTN            varies                           sendmsg data. See [SendMessageTypes]
Various                     Temporary data[31:0]             can be used as general temporary data storage

M0 can only be written by the scalar ALU.

3.4.5. NULL
NULL is a scalar source and destination. Reading NULL returns zero, writing to NULL has no effect (write data
is discarded).

NULL may be used anywhere scalar sources can normally be used:
  • When NULL is used as the destination of an SALU instruction, the instruction executes: SDST is not written
    but SCC is updated (if the instruction normally updates SCC).
    Instructions like S_SWAP_PC with NULL as a DEST still execute, loading zero into the PC and not writing
    any other result.
  • NULL may not be used as an S#, V# or T#.

3.4.6. SCC: Scalar Condition Code
Many scalar ALU instructions set the Scalar Condition Code (SCC) bit, indicating the result of the operation.

  Compare operations: 1 = true
  Arithmetic operations: 1 = carry out
  Bit/logical operations: 1 = result was not zero
  Move: does not alter SCC

The SCC can be used as the carry-in for extended-precision integer arithmetic, as well as the selector for
conditional moves and branches.

3.4.7. Vector Compares: VCC and VCCZ
Vector ALU comparison instructions (V_CMP) compare two values and return a bit-mask of the result, where
each bit represents one lane (work-item) where: 1= pass, 0 = fail. This result mask is the Vector Condition Code
(VCC). VCC is also set for selected integer ALU operations (carry-out).

These instructions write this mask either to VCC, an SGPR or to EXEC, but do not write to both EXEC and
SGPRs. Wave64 writes 64-bits of VCC, EXEC or an aligned pair of SGPRs; Wave32 writes only the low 32 bits of
VCC, EXEC or a single SGPR.
