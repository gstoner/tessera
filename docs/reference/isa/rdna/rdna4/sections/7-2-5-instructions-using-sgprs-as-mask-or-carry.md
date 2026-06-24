# 7.2.5. Instructions using SGPRs as Mask or Carry

> RDNA4 ISA — pages 84–84

Field           Bit Position       Description
FP_ROUND        3:0                [1:0] Single-precision round mode.
                                   [3:2] Double and Half-precision (FP16) round mode.
                                   Round Modes:
                                     0=nearest even
                                     1= +infinity
                                     2= -infinity
                                     3= toward zero
FP_DENORM       7:4                [5:4] Single-precision denormal mode.
                                   [7:6] Double and Half-precision (FP16) denormal mode.
                                   Denormal modes:
                                     0 = Flush input and output denorms
                                     1 = Allow input denorms, flush output denorms
                                     2 = Flush input denorms, allow output denorms
                                     3 = Allow input and output denorms

These mode bits do not affect rounding and denormal handling of float global memory atomics.

DOT2_F16_F16 and DOT2_BF16_BF16 support round-to-nearest-even rounding.
DOT2_F16_F16 supports denorms, and DOT2_BF16_BF16 disables all denorms.
These opcodes flush input and output denorms: FMA_DX9_ZERO_F32, CUBExx_F32, EXP_F32, LOG_F32,
RCP_F32, RCP_IFLAG_F32, RSQ_F32, SQRT_F32, COS_F32.

7.2.5. Instructions using SGPRs as Mask or Carry
Every VALU instruction can use SGPRs as a constant, but the following can read or write SGPRs as masks or
carry:

Read Mask or Carry in          Write Carry out              Implicitly Reads VCC            Implicitly Writes VCC
V_CNDMASK_B32                  V_CMP*                       V_DIV_FMAS_F32                  V_CMP (not V_CMPX)
V_CNDMASK_B16                  V_ADD_CO_CI_U32              V_DIV_FMAS_F64
V_ADD_CO_CI_U32                V_SUB_CO_CI_U32              (fmas reads 3 operands + VCC)
V_SUB_CO_CI_U32                V_SUBREV_CO_CI_U32           V_CNDMASK in VOP2
V_SUBREV_CO_CI_U32             V_ADD_CO_U32                 V_{ADD,SUB,SUBREV}_CO_CI_U
                                                            32 in VOP2
                               V_SUB_CO_U32
                               V_SUBREV_CO_U32
                               V_MAD_U64_U32
                               V_MAD_I64_I32
                               Write Data out (not carry)                                   Write Carry out / mask
                               V_READLANE                                                   V_DIV_SCALE_F32
                               V_READFIRSTLANE                                              V_DIV_SCALE_F64

"VCC" in the above table refers to VCC in a VOP2 or VOPC encoding.

V_CMPX is the only VALU instruction that writes EXEC.
