# 3.4.4. NULL

> RDNA3 ISA — pages 32–32

Field                       Bit    Description
                            Pos
FP16_OVFL                   23     If set, an overflowed FP16 VALU result is clamped to +/- MAX_FP16 regardless of round
                                   mode, while still preserving true INF values. (Inputs which are infinity may result in infinity,
                                   as does divide-by-zero).
DISABLE_PERF                27     1 = disable performance counting for this wave.

3.4.3. M0 : Miscellaneous Register
There is one 32-bit M0 register per wave and is it used for:

                                                    Table 7. M0 Register Fields
Operation                    M0 Contents                             Notes
LDS_PARAM_LOAD               { 1’b0, new_prim_mask[15:1],            Offset is in bytes and offset[6:0] must be zero.
                             parameter_offset[15:0] }                Wave32: new_prim_mask is {8’b0, mask[7:1] }
LDS_DIRECT_LOAD              { 13’b0, DataType[2:0],                 address is in bytes
                             LDS_address[15:0] }
LDS ADDTID                   { 16’h0, lds_offset[15:0] }             offset is in bytes, must be 4-byte aligned
Global Data Share            { base[15:0] , size[15:0] }             base and size are in bytes
GDS Ordered Count            { base[15:0], 3’h0,                     used for deferred attribute shading (split-GS)
                             logical_wave_id[12:0] }
Global Wave Sync             various uses                            see instruction definition
S/V_MOVREL                   GPR index                               See S_MOVREL and V_MOVREL instructions
S_SENDMSG / _RTN             varies                                  sendmsg data. See [Send_Message_Types]
EXPORT                       Row number for mesh shader POS          See Export chapter
                             & Param exports
SMEM                         address_offset[31:0]                    see SMEM section
Temporary                    data[31:0]                              can be used as general temporary data storage

M0 can only be written by the scalar ALU.

3.4.4. NULL
NULL is a scalar source and destination. Reading NULL returns zero, writing to NULL has no effect (write data
is discarded).

NULL may be used anywhere scalar sources can normally be used:
  • When NULL is used as the destination of an SALU instruction, the instruction executes: SDST is not written
    but SCC is updated (if the instruction normally updates SCC).
  • NULL may not be used as an S#, V# or T#.

3.4.5. SCC: Scalar Condition Code
Many scalar ALU instructions set the Scalar Condition Code (SCC) bit, indicating the result of the operation.

  Compare operations: 1 = true
