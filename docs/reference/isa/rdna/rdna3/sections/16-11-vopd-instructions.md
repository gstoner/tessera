# 16.11. VOPD Instructions

> RDNA3 ISA — pages 369–370

16.11. VOPD Instructions

The VOPD encoded describes two VALU opcodes that are executed in parallel.

For instruction definitions, refer to the VOP1, VOP2 and VOP3 sections.

16.11.1. VOPD X-Instructions
V_DUAL_FMAC_F32                                                                                                      0

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply-
add.

V_DUAL_FMAAK_F32                                                                                                     1

Multiply two single-precision floats and add a literal constant using fused multiply-add.

V_DUAL_FMAMK_F32                                                                                                     2

Multiply a single-precision float with a literal constant and add a second single-precision float using fused
multiply-add.

V_DUAL_MUL_F32                                                                                                       3

Multiply two floating point inputs and store the result into a vector register.

V_DUAL_ADD_F32                                                                                                       4

Add two floating point inputs and store the result into a vector register.

V_DUAL_SUB_F32                                                                                                       5

Subtract the second floating point input from the first input and store the result into a vector register.

V_DUAL_SUBREV_F32                                                                                                    6

Subtract the first floating point input from the second input and store the result into a vector register.

V_DUAL_MUL_DX9_ZERO_F32                                                                                              7

Multiply two floating point inputs and store the result in a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

V_DUAL_MOV_B32                                                                                                       8

Move data from a vector input into a vector register.

V_DUAL_CNDMASK_B32                                                                                                   9

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

V_DUAL_MAX_F32                                                                                                      10

Select the maximum of two floating point inputs and store the result into a vector register.

V_DUAL_MIN_F32                                                                                                      11

Select the minimum of two floating point inputs and store the result into a vector register.

V_DUAL_DOT2ACC_F32_F16                                                                                              12

Dot product of packed FP16 values, accumulate with destination. The initial value in D is used as S2.

V_DUAL_DOT2ACC_F32_BF16                                                                                             13

Dot product of packed brain-float values, accumulate with destination. The initial value in D is used as S2.
