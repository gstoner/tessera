# 7.5. Packed Math

> RDNA3 ISA — pages 74–75

7.4. 16-bit Math and VGPRs
VALU instructions that operate on 16-bit data (non-packed) can separately address the two halves of a 32-bit
VGPR.

16-bit VGPR-pairs are packed into a 32-bit VGPRs: the 32-bit VGPR "V0" contains two 16-bit VGPRs: "V0.L"
representing V0[15:0] and "V0.H" representing V0[31:16].

How this addressing is encoded in the ISA varies by the instruction encoding: The 16-bit instructions can be
encoded using VOP1/2/C as well as VOP3/VOP3P/VINTERP.

16bit VGPR Naming
   The 32-bit VGPR is "V0". The two halves are called "V0.L" and "V0.H".

VOP1, VOP2, VOPC Encoding
   16-bit VGPRs are encoded as:
   SRC/DST[6:0] = 32-bit VGPR address;
   SRC/DST[7] = (1=hi, 0=lo half)
   In this encoding, only 256 16-bit VGPRs can be addressed.

VOP3, VOP3P, VINTERP
   16-bit VGPRs are encoded as:
   SRC/DST[7:0] = 32-bit VGPR address, OPSEL = high/low.
   In this encoding, a wave can address 512 16-bit VGPRs.

The packing shown below allows reading or writing in one cycle:
  • 32 lanes of one 32-bit VGPR: V0
  • 64 lanes of one 16-bit VGPR: V0.L
  • 32 lanes of two 16-bit VGPRs (a pair, as used by packed math): V0.L and V0.H

7.5. Packed Math
Packed math is a form of operation that accelerates arithmetic on two values packed into the same VGPR. It
performs operations on two 16-bit values within a DWORD as if they were separate threads. For example, a
packed add of V0=V1+V2 is really two separate adds: adding the low 16 bits of each DWORD and storing the
result in the low 16 bits of V0, and adding the high halves and storing the result in the high 16 bits of V0.

Packed math uses the instructions below and the microcode format "VOP3P". This format has OPSEL and NEG
fields for both the low and high operands, and does not have ABS and OMOD.

                                         Table 29. Packed Math Opcodes:
                                                Packed Math ops
V_PK_MUL_F16                           V_PK_FMA_F16            V_PK_MIN_F16
V_PK_ADD_F16                           V_PK_FMAC_F16           V_PK_MAX_F16
V_PK_ADD_I16                           V_PK_MAD_I16            V_PK_MIN_I16            V_PK_LSHLREV_B16

                                                      Packed Math ops
V_PK_ADD_U16                              V_PK_MAD_U16                V_PK_MIN_U16                 V_PK_LSHRREV_B16
V_PK_SUB_I16                              V_PK_MUL_LO_U16             V_PK_MAX_I16                 V_PK_ASHRREV_I16
V_PK_SUB_U16                                                          V_PK_MAX_U16
V_FMA_MIX_F32                             V_FMA_MIXLO_F16             V_FMA_MIXHI_F16
V_WMMA_F32_16X16X16_F16                                               V_DOT2_F32_BF16
V_WMMA_F32_16X16X16_BF16                                              V_DOT2_F32_F16
V_WMMA_F16_16X16X16_F16                                               V_DOT4_I32_IU8
V_WMMA_BF16_16X16X16_BF16                                             V_DOT4_U32_U8
V_WMMA_I32_16X16X16_IU8                                               V_DOT8_I32_IU4
V_WMMA_I32_16X16X16_IU4                                               V_DOT8_U32_U4

                   V_FMA_MIX_* and WMMA instructions are not packed math, but perform a single MAD
                  operation on a mixture of 16- and 32-bit inputs. They are listed here because they use the
                   VOP3P encoding.

VOP3P Instruction Fields

Field        Size      Description
OP           7         instruction opcode
SRC0         9         first instruction argument. May come from: vgpr, sgpr, VCC, M0, exec or a constant
                       WMMA: must be a VGPR
SRC1         9         second instruction argument. May come from: vgpr, sgpr, VCC, M0, exec or a constant
                       WMMA: must be a VGPR
SRC2         9         third instruction argument. May come from: vgpr, sgpr, VCC, M0, exec or a constant
VDST         8         vgpr that takes the result.
                       For V_READLANE, indicates the SGPR that receives the result.
NEG          3         negate the input (invert sign bit) for the lower-16bit operand. float inputs only.
                       bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
                       For V_FMA_MIX_* opcodes, this modifies all inputs.
                       For DOT…IU… and WMMA…IU… NEG[1:0] = signed(1)/unsigned(0) for src0 and src1,
                       and Neg[2] behavior is undefined.
NEG_HI       3         negate the input (invert sign bit) for the higher-16bit operand. float inputs only.
                       bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
                       For V_FMA_MIX_* opcodes, this acts as an ABS (absolute value) modifier.
                       For DOT…IU… and WMMA…IU… NEG_HI behavior is undefined.
OPSEL        3         Select the high (1) or low (0) operand as input to the operation that results in the lower-half of the
[13:11]                destination. [0] = src0, [1] = src1, [2] = src2
                       If either the source operand or destination operand is 32bits, the corresponding OPSEL bit must set
                       to zero. This rule does not apply to MIX instructions, which have a unique interpretation of OPSEL. See
                       notes below. OPSEL works for 16-bit VGPR, SGPR and literal-constant sources; for inline constant
                       sources OPSEL must be zero (value only exists in lower 16 bits).
                       OPSEL[0] and [1] are unused for WMMA ops, and OPSEL[2] is used only with WMMA ops with 16-bit
                       output to control whether the C matrix is read from upper or lower bits in the VGPR, and whether
                       the D matrix is stored into upper or lower bits.
