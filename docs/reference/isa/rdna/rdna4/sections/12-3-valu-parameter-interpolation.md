# 12.3. VALU Parameter Interpolation

> RDNA4 ISA — pages 158–158

and prevents the hazard..

12.3. VALU Parameter Interpolation
Parameter interpolation is performed using an FMA operation that includes a built-in DPP operation to unpack
the per-quad P0/P10/P20 values into per-lane values. Because this instruction reads data from neighboring
lanes, the implicit DPP acts as if "fetch invalid = 1", so that the instruction can read data from neighboring lanes
that have EXEC==0, rather than getting the value 0 from those. Standard interpolation is calculating:

     Per-Pixel-Parameter = P0 + I * P10 + J * P20        // I, J are per-pixel; P0/P10/P20 are per-primitive

This parameter interpolation is realized using a pair of instructions:

     V_INTERP_P10_F32   V5, V0, V1, v2    // tmp = P0 + I*P10      uses DPP8=1,1,1,1,5,5,5,5; Src2(P0) uses
  DPP8=0,0,0,0,4,4,4,4
     V_INTERP_P20_F32   V5, V3, V4, V5    // dst = J*P20 + tmp      uses DPP8=2,2,2,2,6,6,6,6

                                  Table 70. Parameter Interpolation Instruction Fields
Field    Size Description
OP       5    Instruction Opcode:

                    V_INTERP_P10_F32                 // tmp = P0 + I*P10. hardcoded DPP8 on 2 sources
                    V_INTERP_P2_F32                  // D = tmp + J*P20. hardcoded DPP8 on 1 source
                    V_INTERP_P10_F16_F32             // tmp = P0 + I*P10. hardcoded DPP8 on 2 sources
                    V_INTERP_P2_F16_F32              // D = tmp + J*P20. hardcoded DPP8 on 1 source
                    V_INTERP_RTZ_P10_F16_F32         // same as above, but round-toward-zero
                    V_INTERP_RTZ_P2_F16_F32          // same as above, but round-toward-zero
SRC0     9    First argument VGPR: Parameter data (P0 or P20) from LDS stored in a VGPR.
SRC1     9    Second argument VGPR: I or J barycentric
SRC2     9    Third argument VGPR: "P10" ops holds P10 data; "P2" ops holds partial result from "P10" op.
VDST     8    Destination VGPR
NEG      3    Negate the input (invert sign bit).
              bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
              For 16-bit interpolation this applies to both low and high halves.
WaitEXP 3     Wait for EXPcnt to be less than or equal to this value before issuing this instruction.
              Used to wait for a specific previous DS_PARAM_LOAD to have completed.
OPSEL    4    Operation select for 16-bit math: 1=select high half, 0=select low half
              [0]=src0, [1]=src1, [2]=src2, [3]=dest
              For dest=0, dest_vgpr[31:0] = {prev_dst_vgpr[31:16], result[15:0] }
              For dest=1, dest_vgpr[31:0] = {result[15:0], prev_dst_vgpr[15:0] }
              OPSEL may only be used for 16-bit operands, and must be zero for any other operands/results.
CM       1    Clamp result to [0, 1.0]
