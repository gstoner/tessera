# 7.3. Instructions

> RDNA3.5 ISA — pages 74–75

7.3. Instructions
The table below lists the complete VALU instruction set by microcode encoding, except for VOP3P instructions
which are listed in a later section.

VOP3                             VOP3 - 2 operands       VOP2                    VOP1
V_ADD3_U32                       V_ADD_CO_U32            V_ADD_CO_CI_U32         V_BFREV_B32
V_ADD_LSHL_U32                   V_ADD_F64               V_ADD_F16               V_CEIL_F16
V_ALIGNBIT_B32                   V_ADD_NC_I16            V_ADD_F32               V_CEIL_F32
V_ALIGNBYTE_B32                  V_ADD_NC_I32            V_ADD_NC_U32            V_CEIL_F64
V_AND_OR_B32                     V_ADD_NC_U16            V_AND_B32               V_CLS_I32
V_BFE_I32                        V_AND_B16               V_ASHRREV_I32           V_CLZ_I32_U32
V_BFE_U32                        V_ASHRREV_I16           V_CNDMASK_B32           V_COS_F16
V_BFI_B32                        V_ASHRREV_I64           V_CVT_PK_RTZ_F16_F32    V_COS_F32
V_CNDMASK_B16                    V_BCNT_U32_B32          V_DOT2ACC_F32_F16       V_CTZ_I32_B32
V_CUBEID_F32                     V_BFM_B32               V_FMAAK_F16             V_CVT_F16_F32
V_CUBEMA_F32                     V_CVT_PK_I16_F32        V_FMAAK_F32             V_CVT_F16_I16
V_CUBESC_F32                     V_CVT_PK_I16_I32        V_FMAC_DX9_ZERO_F32     V_CVT_F16_U16
V_CUBETC_F32                     V_CVT_PK_NORM_I16_F16   V_FMAC_F16              V_CVT_F32_F16
V_CVT_PK_U8_F32                  V_CVT_PK_NORM_I16_F32   V_FMAC_F32              V_CVT_F32_F64
V_DIV_FIXUP_F16                  V_CVT_PK_NORM_U16_F16   V_FMAMK_F16             V_CVT_F32_I32
V_DIV_FIXUP_F32                  V_CVT_PK_NORM_U16_F32   V_FMAMK_F32             V_CVT_F32_U32
V_DIV_FIXUP_F64                  V_CVT_PK_U16_F32        V_LDEXP_F16             V_CVT_F32_UBYTE0
V_DIV_FMAS_F32                   V_CVT_PK_U16_U32        V_LSHLREV_B32           V_CVT_F32_UBYTE1
V_DIV_FMAS_F64                   V_LDEXP_F32             V_LSHRREV_B32           V_CVT_F32_UBYTE2
V_DIV_SCALE_F32                  V_LDEXP_F64             V_MAX_F16               V_CVT_F32_UBYTE3
V_DIV_SCALE_F64                  V_LSHLREV_B16           V_MAX_F32               V_CVT_F64_F32
V_DOT2_BF16_BF16                 V_LSHLREV_B64           V_MAX_I32               V_CVT_F64_I32
V_DOT2_F16_F16                   V_LSHRREV_B16           V_MAX_U32               V_CVT_F64_U32
V_FMA_DX9_ZERO_F32               V_LSHRREV_B64           V_MIN_F16               V_CVT_FLOOR_I32_F32
V_FMA_F16                        V_MAX_F64               V_MIN_F32               V_CVT_I16_F16
V_FMA_F32                        V_MAX_I16               V_MIN_I32               V_CVT_I32_F32
V_FMA_F64                        V_MAX_U16               V_MIN_U32               V_CVT_I32_F64
V_LERP_U8                        V_MBCNT_HI_U32_B32      V_MUL_DX9_ZERO_F32      V_CVT_I32_I16
V_LSHL_ADD_U32                   V_MBCNT_LO_U32_B32      V_MUL_F16               V_CVT_NEAREST_I32_F32
V_LSHL_OR_B32                    V_MIN_F64               V_MUL_F32               V_CVT_NORM_I16_F16
V_MAD_I16                        V_MIN_I16               V_MUL_HI_I32_I24        V_CVT_NORM_U16_F16
V_MAD_I32_I16                    V_MIN_U16               V_MUL_HI_U32_U24        V_CVT_OFF_F32_I4
V_MAD_I32_I24                    V_MUL_F64               V_MUL_I32_I24           V_CVT_U16_F16
V_MAD_I64_I32                    V_MUL_HI_I32            V_MUL_U32_U24           V_CVT_U32_F32
V_MAD_U16                        V_MUL_HI_U32            V_OR_B32                V_CVT_U32_F64
V_MAD_U32_U16                    V_MUL_LO_U16            V_PK_FMAC_F16           V_CVT_U32_U16
V_MAD_U32_U24                    V_MUL_LO_U32            V_SUBREV_CO_CI_U32      V_EXP_F16
V_MAD_U64_U32                    V_OR_B16                V_SUBREV_F16            V_EXP_F32
V_MAX3_F16                       V_PACK_B32_F16          V_SUBREV_F32            V_FLOOR_F16
V_MAX3_F32                       V_READLANE_B32          V_SUBREV_NC_U32         V_FLOOR_F32
V_MAX3_I16                       V_SUBREV_CO_U32         V_SUB_CO_CI_U32         V_FLOOR_F64
V_MAX3_I32                       V_SUB_CO_U32            V_SUB_F16               V_FRACT_F16
V_MAX3_U16                       V_SUB_NC_I16            V_SUB_F32               V_FRACT_F32
V_MAX3_U32                       V_SUB_NC_I32            V_SUB_NC_U32            V_FRACT_F64
V_MAXMIN_F16                     V_SUB_NC_U16            V_XNOR_B32              V_FREXP_EXP_I16_F16
V_MAXMIN_F32                     V_TRIG_PREOP_F64        V_XOR_B32               V_FREXP_EXP_I32_F32
V_MAXMIN_I32                     V_WRITELANE_B32                                 V_FREXP_EXP_I32_F64

VOP3                             VOP3 - 2 operands             VOP2                          VOP1
V_MAXMIN_U32                     V_XOR_B16                                                   V_FREXP_MANT_F16
V_MED3_F16                                                                                   V_FREXP_MANT_F32
V_MED3_F32                                                                                   V_FREXP_MANT_F64
V_MED3_I16                                                                                   V_LOG_F16
V_MED3_I32                                                                                   V_LOG_F32
V_MED3_U16                                                                                   V_MOVRELD_B32
V_MED3_U32                                                                                   V_MOVRELSD_2_B32
V_MIN3_F16                                                                                   V_MOVRELSD_B32
V_MIN3_F32                                                                                   V_MOVRELS_B32
V_MIN3_I16                                                                                   V_MOV_B16
V_MIN3_I32                                                                                   V_MOV_B32
V_MIN3_U16                                                                                   V_NOP
V_MIN3_U32                                                                                   V_NOT_B16
V_MINMAX_F16                                                                                 V_NOT_B32
V_MINMAX_F32                                                                                 V_PERMLANE64_B32
V_MINMAX_I32                                                                                 V_PIPEFLUSH
V_MINMAX_U32                                                                                 V_RCP_F16
V_MQSAD_PK_U16_U8                                                                            V_RCP_F32
V_MQSAD_U32_U8                                                                               V_RCP_F64
V_MSAD_U8                                                                                    V_RCP_IFLAG_F32
V_MULLIT_F32                                                                                 V_READFIRSTLANE_B32
V_OR3_B32                                                                                    V_RNDNE_F16
V_PERMLANE16_B32                                                                             V_RNDNE_F32
V_PERMLANEX16_B32                                                                            V_RNDNE_F64
V_PERM_B32                                                                                   V_RSQ_F16
V_QSAD_PK_U16_U8                                                                             V_RSQ_F32
V_SAD_HI_U8                                                                                  V_RSQ_F64
V_SAD_U16                                                                                    V_SAT_PK_U8_I16
V_SAD_U32                                                                                    V_SIN_F16
V_SAD_U8                                                                                     V_SIN_F32
V_XAD_U32                                                                                    V_SQRT_F16
V_XOR3_B32                                                                                   V_SQRT_F32
                                                                                             V_SQRT_F64
                                                                                             V_SWAPREL_B32
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
