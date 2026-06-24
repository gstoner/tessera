# 7.2.3. Output Operands

> RDNA4 ISA — pages 82–82

V_LSHRREV_B16                 V_MIN*16                     V_MAX*16                     V_MED3_*16
V_AND_B16                     V_NOT_B16                    V_OR_B16                     V_XOR_B16
V_MOV_B16                     V_SWAP_B16                   V_CNDMASK_B16
V_SAT_PK4_I4_I8               V_SAT_PK4_U4_U8              V_SAT_PK_U8_I16              V_LDEXP_F16
V_RNDNE_F16                   V_TRUNC_F16                  V_CEIL_F16                   V_DIV_FIXUP_F16
V_FLOOR_F16                   V_FRACT_F16                  V_FREXP_EXP_I16_F16          V_FREXP_MANT_F16
V_COS_F16                     V_SIN_F16                    V_EXP_F16                    V_LOG_F16
V_RCP_F16                     V_RSQ_F16                    V_SQRT_F16
V_CVT_F16_*                   V_CVT_*_{F16, U16}           V_CVT_PK_*_{F16, BF8, FP8}   V_CVT_PK_{F16, BF8, FP8}_*
V_INTERP_P10_RTZ_F16_F32      V_INTERP_P2_RTZ_F16_F32      V_INTERP_P2_F16_F32          V_INTERP_P10_F16_F32

7.2.3. Output Operands
VALU instructions typically write their results to VGPRs specified in the VDST field of the microcode word. A
thread only writes a result if the associated bit in the EXEC mask is set to 1.

V_CMPX instructions write the result of their comparison (one bit per thread) to the EXEC mask. Both V_CMP
and V_CMPX write a full mask of results (32-bits for wave32 and 64-bits for wave64 ) regardless of the value of
EXEC. Inactive lanes write zero into the result mask.

Instructions producing a carry-out (integer add and subtract) write their result to VCC when used in the VOP2
form, and to an arbitrary SGPR-pair when used in the VOP3 form. Inactive lanes write zero into this carry-out.

When the VOP3 form is used, instructions with a floating-point result may apply an output modifier (OMOD
field) that multiplies the result by: 0.5, 2.0, or 4.0. Optionally, the result can be clamped (CLAMP field) to the
min and max representable range (see next section).

7.2.3.1. Output Operand Modifiers
Output modifiers (OMOD) apply to half, single and double floating point results only and scale the result by :
0.5, 2.0, 4.0 or do not scale. Integer and packed float 16 results ignore the OMOD setting.

OMOD is allowed regardless of MODE.denorm setting, but flushes output denorms when OMOD!=0.
It has the result of: -0 * OMOD = +0.
Instructions with OMOD!=0 report exceptions, but not: underflow, or inexact. It reports: invalid, input-
denormal, divide-by-zero, overflow.

A few opcodes force output denorms to be disabled. See individual instruction descriptions.

Output Modifiers are not supported for:
  • V_SWAP_*
  • V_PERMLANE
  • DOT2_F16_F16
  • DOT2_BF16_BF16
  • Integer and bitwise ops
  • V_CMP*
  • V_CVT*F32{BF8,FP8}
  • DOT4 with BF8/FP8
