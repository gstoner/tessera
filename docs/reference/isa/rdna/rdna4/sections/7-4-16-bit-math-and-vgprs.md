# 7.4. 16-bit Math and VGPRs

> RDNA4 ISA — pages 87–87

VOP3                             VOP3 - 2 operands             VOP2                          VOP1
V_MQSAD_U32_U8                                                                               V_READFIRSTLANE_B32
V_MSAD_U8                                                                                    V_RNDNE_F16
V_MULLIT_F32                                                                                 V_RNDNE_F32
V_OR3_B32                                                                                    V_RNDNE_F64
V_PERMLANE16_B32                                                                             V_RSQ_F16
V_PERMLANEX16_B32                                                                            V_RSQ_F32
V_PERM_B32                                                                                   V_RSQ_F64
V_QSAD_PK_U16_U8                                                                             V_SAT_PK_U8_I16
V_SAD_HI_U8                                                                                  V_SIN_F16
V_SAD_U16                                                                                    V_SIN_F32
V_SAD_U32                                                                                    V_SQRT_F16
V_SAD_U8                                                                                     V_SQRT_F32
V_XAD_U32                                                                                    V_SQRT_F64
V_XOR3_B32                                                                                   V_SWAPREL_B32
                                                                                             V_SWAP_B16
                                                                                             V_SWAP_B32
                                                                                             V_TRUNC_F16
                                                                                             V_TRUNC_F32
                                                                                             V_TRUNC_F64

                                                  VOPC - Compare Ops
                                VOPC writes to VCC, VOP3 writes compare result to any SGPR
V_CMP                                                                                                        write VCC
                     I16, I32, I64, U16, U32, U64 F, LT, EQ, LE, GT, LG, GE, T
V_CMPX                                                                                                       write exec
V_CMP                F16, F32, F64               F, LT, EQ, LE, GT, LG, GE, T,                             write VCC
                                                 O, U, NGE, NLG, NGT, NLE, NEQ, NLT
V_CMPX                                           (T = True, F = False, O = total order, U = unordered, "N" write exec
                                                 = Not (inverse) compare)
V_CMP_CLASS          F16, F32, F64               Test for any combination of: signaling-NaN, quiet-NaN, write VCC
V_CMPX_CLASS                                     positive or negative: infinity, normal, subnormal, zero. write exec

7.4. 16-bit Math and VGPRs
VALU instructions that operate on 16-bit data (non-packed) can separately address the two halves of a 32-bit
VGPR.

16-bit VGPR-pairs are packed into a 32-bit VGPRs: the 32-bit VGPR "V0" contains two 16-bit VGPRs: "V0.L"
representing V0[15:0] and "V0.H" representing V0[31:16].

How this addressing is encoded in the ISA varies by the instruction encoding: The 16-bit instructions can be
encoded using VOP1/2/C as well as VOP3 / VOP3P.

16bit VGPR Naming
   The 32-bit VGPR is "V0". The two halves are called "V0.L" and "V0.H".

VOP1, VOP2, VOPC Encoding of 16-bit VGPRs
   SRC/DST[6:0] = 32-bit VGPR address;
   SRC/DST[7] = (1=hi, 0=lo half)
   In this encoding, only 256 16-bit VGPRs can be addressed.
