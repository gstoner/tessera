# 7.2.7. Out-of-Range GPRs

> RDNA4 ISA — pages 85–86

7.2.6. Wave64 use of SGPRs as Input Operands
VALU instructions may use SGPRs as a uniform input, shared by all work-items. If the value is used as simple
data value, then the same SGPR is distributed to all 64 work-items. If, on the other hand, the data value
represents a mask (e.g. carry-in, mask for CNDMASK), then each work-item receives a separate value, and two
consecutive SGPRs are read.

7.2.7. Out-of-Range GPRs
When a source VGPR is out-of-range, it acts as if the instruction indicated VGPR0.

When the destination GPR is out-of-range, the instruction issues to the VALU but does not write the results or
report exceptions.

See VGPR Out Of Range Behavior for more information.

7.2.8. PERMLANE Specific Rules
V_PERMLANE* may not occur immediately after a V_CMPX. To prevent this, any other VALU opcode may be
inserted (e.g. V_NOP).

7.3. Instructions
The table below lists the complete VALU instruction set by microcode encoding, except for VOP3P instructions
which are listed in a later section.

VOP3                             VOP3 - 2 operands       VOP2                        VOP1
V_ADD3_U32                       V_ADD_CO_U32            V_ADD_CO_CI_U32             V_BFREV_B32
V_ADD_LSHL_U32                   V_ADD_NC_I16            V_ADD_F16                   V_CEIL_F16
V_ALIGNBIT_B32                   V_ADD_NC_I32            V_ADD_F32                   V_CEIL_F32
V_ALIGNBYTE_B32                  V_ADD_NC_U16            V_ADD_F64                   V_CEIL_F64
V_AND_OR_B32                     V_AND_B16               V_ADD_NC_U32                V_CLS_I32
V_BFE_I32                        V_ASHRREV_I16           V_AND_B32                   V_CLZ_I32_U32
V_BFE_U32                        V_ASHRREV_I64           V_ASHRREV_I32               V_COS_F16
V_BFI_B32                        V_BCNT_U32_B32          V_CNDMASK_B32               V_COS_F32
V_CNDMASK_B16                    V_BFM_B32               V_CVT_PK_RTZ_F16_F32        V_CTZ_I32_B32
V_CUBEID_F32                     V_CVT_PK_BF8_F32        V_FMAAK_F16                 V_CVT_F16_F32
V_CUBEMA_F32                     V_CVT_PK_FP8_F32        V_FMAAK_F32                 V_CVT_F16_I16
V_CUBESC_F32                     V_CVT_PK_I16_F32        V_FMAC_F16                  V_CVT_F16_U16
V_CUBETC_F32                     V_CVT_PK_I16_I32        V_FMAC_F32                  V_CVT_F32_BF8
V_CVT_PK_U8_F32                  V_CVT_PK_NORM_I16_F16   V_FMAMK_F16                 V_CVT_F32_F16
V_DIV_FIXUP_F16                  V_CVT_PK_NORM_I16_F32   V_FMAMK_F32                 V_CVT_F32_F64
V_DIV_FIXUP_F32                  V_CVT_PK_NORM_U16_F16   V_LDEXP_F16                 V_CVT_F32_FP8
V_DIV_FIXUP_F64                  V_CVT_PK_NORM_U16_F32   V_LSHLREV_B32               V_CVT_F32_I32
V_DIV_FMAS_F32                   V_CVT_PK_U16_F32        V_LSHLREV_B64               V_CVT_F32_U32
V_DIV_FMAS_F64                   V_CVT_PK_U16_U32        V_LSHRREV_B32               V_CVT_F32_UBYTE0
V_DIV_SCALE_F32                  V_CVT_SR_BF8_F32        V_MAX_I32                   V_CVT_F32_UBYTE1
V_DIV_SCALE_F64                  V_CVT_SR_FP8_F32        V_MAX_NUM_F16               V_CVT_F32_UBYTE2

VOP3                             VOP3 - 2 operands       VOP2                 VOP1
V_DOT2_BF16_BF16                 V_LDEXP_F32             V_MAX_NUM_F32        V_CVT_F32_UBYTE3
V_DOT2_F16_F16                   V_LDEXP_F64             V_MAX_NUM_F64        V_CVT_F64_F32
V_FMA_DX9_ZERO_F32               V_LSHLREV_B16           V_MAX_U32            V_CVT_F64_I32
V_FMA_F16                        V_LSHRREV_B16           V_MIN_I32            V_CVT_F64_U32
V_FMA_F32                        V_LSHRREV_B64           V_MIN_NUM_F16        V_CVT_FLOOR_I32_F32
V_FMA_F64                        V_MAXIMUM_F16           V_MIN_NUM_F32        V_CVT_I16_F16
V_LERP_U8                        V_MAXIMUM_F32           V_MIN_NUM_F64        V_CVT_I32_F32
V_LSHL_ADD_U32                   V_MAXIMUM_F64           V_MIN_U32            V_CVT_I32_F64
V_LSHL_OR_B32                    V_MAX_I16               V_MUL_DX9_ZERO_F32   V_CVT_I32_I16
V_MAD_CO_I64_I32                 V_MAX_U16               V_MUL_F16            V_CVT_NEAREST_I32_F32
V_MAD_CO_U64_U32                 V_MBCNT_HI_U32_B32      V_MUL_F32            V_CVT_NORM_I16_F16
V_MAD_I16                        V_MBCNT_LO_U32_B32      V_MUL_F64            V_CVT_NORM_U16_F16
V_MAD_I32_I16                    V_MINIMUM_F16           V_MUL_HI_I32_I24     V_CVT_OFF_F32_I4
V_MAD_I32_I24                    V_MINIMUM_F32           V_MUL_HI_U32_U24     V_CVT_PK_F32_BF8
V_MAD_U16                        V_MINIMUM_F64           V_MUL_I32_I24        V_CVT_PK_F32_FP8
V_MAD_U32_U16                    V_MIN_I16               V_MUL_U32_U24        V_CVT_U16_F16
V_MAD_U32_U24                    V_MIN_U16               V_OR_B32             V_CVT_U32_F32
V_MAX3_I16                       V_MUL_HI_I32            V_PK_FMAC_F16        V_CVT_U32_F64
V_MAX3_I32                       V_MUL_HI_U32            V_SUBREV_CO_CI_U32   V_CVT_U32_U16
V_MAX3_NUM_F16                   V_MUL_LO_U16            V_SUBREV_F16         V_EXP_F16
V_MAX3_NUM_F32                   V_MUL_LO_U32            V_SUBREV_F32         V_EXP_F32
V_MAX3_U16                       V_OR_B16                V_SUBREV_NC_U32      V_FLOOR_F16
V_MAX3_U32                       V_PACK_B32_F16          V_SUB_CO_CI_U32      V_FLOOR_F32
V_MAXIMUM3_F16                   V_PERMLANE16_VAR_B32    V_SUB_F16            V_FLOOR_F64
V_MAXIMUM3_F32                   V_PERMLANEX16_VAR_B32   V_SUB_F32            V_FRACT_F16
V_MAXIMUMMINIMUM_F16             V_READLANE_B32          V_SUB_NC_U32         V_FRACT_F32
V_MAXIMUMMINIMUM_F32             V_SUBREV_CO_U32         V_XNOR_B32           V_FRACT_F64
V_MAXMIN_I32                     V_SUB_CO_U32            V_XOR_B32            V_FREXP_EXP_I16_F16
V_MAXMIN_NUM_F16                 V_SUB_NC_I16                                 V_FREXP_EXP_I32_F32
V_MAXMIN_NUM_F32                 V_SUB_NC_I32                                 V_FREXP_EXP_I32_F64
V_MAXMIN_U32                     V_SUB_NC_U16                                 V_FREXP_MANT_F16
V_MED3_I16                       V_TRIG_PREOP_F64                             V_FREXP_MANT_F32
V_MED3_I32                       V_WRITELANE_B32                              V_FREXP_MANT_F64
V_MED3_NUM_F16                   V_XOR_B16                                    V_LOG_F16
V_MED3_NUM_F32                                                                V_LOG_F32
V_MED3_U16                                                                    V_MOVRELD_B32
V_MED3_U32                       One Operand:                                 V_MOVRELSD_2_B32

V_MIN3_I16                       V_S_EXP_F16                                  V_MOVRELSD_B32
V_MIN3_I32                       V_S_EXP_F32                                  V_MOVRELS_B32
V_MIN3_NUM_F16                   V_S_LOG_F16                                  V_MOV_B16
V_MIN3_NUM_F32                   V_S_LOG_F32                                  V_MOV_B32
V_MIN3_U16                       V_S_RCP_F16
V_MIN3_U32                       V_S_RCP_F32
V_MINIMUM3_F16                   V_S_RSQ_F16                                  V_NOP
V_MINIMUM3_F32                   V_S_RSQ_F32                                  V_NOT_B16
V_MINIMUMMAXIMUM_F16             V_S_SQRT_F16                                 V_NOT_B32
V_MINIMUMMAXIMUM_F32             V_S_SQRT_F32                                 V_PERMLANE64_B32
V_MINMAX_I32                                                                  V_PIPEFLUSH
V_MINMAX_NUM_F16                                                              V_RCP_F16
V_MINMAX_NUM_F32                                                              V_RCP_F32
V_MINMAX_U32                                                                  V_RCP_F64
V_MQSAD_PK_U16_U8                                                             V_RCP_IFLAG_F32
