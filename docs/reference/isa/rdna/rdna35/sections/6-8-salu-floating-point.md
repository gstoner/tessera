# 6.8. SALU Floating Point

> RDNA3.5 ISA — pages 62–62

Instruction                                Encoding     Sets SCC? Operation
S_{and, or, xor, and_not0,                 SOP1         D!=0      Save the EXEC mask, then apply a bit-wise operation
and_not1,or_not0, or_not1, nand, nor,                             to it.
xnor}_SAVEEXEC_{B32,B64}                                          D = EXEC
                                                                  EXEC = S0 <op> EXEC
                                                                  SCC = (EXEC != 0)
                                                                  ("not1" version inverts EXEC)
                                                                  ("not0" version inverts SGPR)
S_{AND_NOT{0,1}_WREXEC_B{32,64}            SOP1         D!=0      NOT0: EXEC, D = ~S0 & EXEC
                                                                  NOT1: EXEC, D = S0 & ~EXEC
                                                                  Both D and EXEC get the same result. SCC = (result !=
                                                                  0). D cannot be EXEC.
S_MOVRELS_{B32,B64}                        SOP1         No        Move a value into an SGPR relative to the value in M0.
S_MOVRELD_{B32,B64}                                               MOVRELS: D = SGPR[S0+M0]
                                                                  MOVRELD: SGPR[D+M0] = S0
                                                                  Index must be even for B64. M0 is an unsigned index.

6.8. SALU Floating Point
The SALU supports a set of floating point operations to offload uniform value calculation from the VALU pipe.
The table below shows the scalar float instructions. These scalar instructions produce identical results with
their VALU counterparts but with some limitations: The scalar instructions do not support operand modifiers.
The compiler can emulate such modifiers with additional instructions.

Scalar F32 Arithmetic Instructions       Scalar F32 Compare Instructions        Scalar Float Conversion Instructions
S_ADD_F32                                S_CMP_LT_F32                           S_CVT_F32_I32
S_SUB_F32                                S_CMP_EQ_F32                           S_CVT_F32_U32
S_MIN_F32                                S_CMP_LE_F32                           S_CVT_I32_F32
S_MAX_F32                                S_CMP_GT_F32                           S_CVT_U32_F32
S_MUL_F32                                S_CMP_LG_F32                           S_CVT_F16_F32
S_FMAAK_F32                              S_CMP_GE_F32                           S_CVT_PK_RTZ_F16_F32
S_FMAMK_F32                              S_CMP_O_F32                            S_CVT_F32_F16
S_FMAC_F32                               S_CMP_U_F32                            S_CVT_HI_F32_F16
S_CEIL_F32                               S_CMP_NGE_F32
S_FLOOR_F32                              S_CMP_NLG_F32
S_TRUNC_F32                              S_CMP_NGT_F32
S_RNDNE_F32                              S_CMP_NLE_F32
                                         S_CMP_NEQ_F32
                                         S_CMP_NLT_F32
Scalar F16 Arithmetic Instructions       Scalar F16 Compare Instructions
S_ADD_F16                                S_CMP_LT_F16                           S_CMP_U_F16
S_SUB_F16                                S_CMP_EQ_F16                           S_CMP_NGE_F16
S_MIN_F16                                S_CMP_LE_F16                           S_CMP_NLG_F16
S_MAX_F16                                S_CMP_GT_F16                           S_CMP_NGT_F16
S_MUL_F16                                S_CMP_LG_F16                           S_CMP_NLE_F16
S_FMAC_F16                               S_CMP_GE_F16                           S_CMP_NEQ_F16
S_CEIL_F16                               S_CMP_O_F16                            S_CMP_NLT_F16
S_FLOOR_F16
S_TRUNC_F16
