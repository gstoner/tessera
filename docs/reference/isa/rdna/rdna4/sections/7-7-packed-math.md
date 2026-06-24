# 7.7. Packed Math

> RDNA4 ISA — pages 90–91

7.7. Packed Math
Packed math is a form of operation that accelerates arithmetic on two values packed into the same VGPR. It
performs operations on two 16-bit values within a DWORD as if they were separate threads. For example, a
packed add of V0=V1+V2 is really two separate adds: adding the low 16 bits of each DWORD and storing the
result in the low 16 bits of V0, and adding the high halves and storing the result in the high 16 bits of V0.

Packed math uses the instructions below and the microcode format "VOP3P". This format has OPSEL and NEG
fields for both the low and high operands, and does not have ABS and OMOD.

                                             Table 37. Packed Math Opcodes:
                                                     Packed Math ops
V_PK_MUL_F16                                V_PK_MINIMUM_F16                         V_PK_FMA_F16
V_PK_ADD_F16                                V_PK_MAXIMUM_F16                         V_PK_FMAC_F16
V_PK_MIN_NUM_F16                            V_PK_MAX_NUM_F16
V_PK_ADD_I16                                V_PK_MIN_I16                             V_PK_MAD_I16
V_PK_ADD_U16                                V_PK_MIN_U16                             V_PK_MAD_U16
V_PK_SUB_I16                                V_PK_MAX_I16                             V_PK_MUL_LO_U16
V_PK_SUB_U16                                V_PK_MAX_U16
V_FMA_MIX_F32                               V_FMA_MIXHI_F16                          V_FMA_MIXLO_F16
V_PK_LSHLREV_B16                            V_PK_LSHRREV_B16                         V_PK_ASHRREV_I16
V_DOT2_F32_BF16                             V_DOT2_F32_F16                           V_DOT4_F32_FP8_BF8
V_DOT4_F32_BF8_FP8                          V_DOT4_F32_FP8_FP8                       V_DOT4_F32_BF8_BF8
V_DOT4_I32_IU8                              V_DOT4_U32_U8                            V_DOT8_I32_IU4
V_DOT8_U32_U4

                   V_FMA_MIX* and WMMA instructions are not packed math, but perform a single MAD
                  operation on a mixture of 16- and 32-bit inputs. They are listed here because they use the
                   VOP3P encoding. Matrix multiply operations are described in a later section.

VOP3P Instruction Fields

Field        Size      Description
OP           7         instruction opcode
SRC0         9         first instruction argument. May come from: VGPR, SGPR, VCC, M0, exec or a constant
SRC1         9         second instruction argument. May come from: VGPR, SGPR, VCC, M0, exec or a constant
SRC2         9         third instruction argument. May come from: VGPR, SGPR, VCC, M0, exec or a constant
VDST         8         VGPR that takes the result.
NEG          3         negate the input (invert sign bit) for the lower-16bit operand. float inputs only.
                       bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
                       For V_FMA_MIX* opcodes, this modifies all inputs.
                       For DOT…IU… and WMMA…IU… NEG[1:0] = signed(1)/unsigned(0) for src0 and src1,
                       and NEG[2] behavior is undefined.

Field        Size    Description
NEG_HI       3       negate the input (invert sign bit) for the higher-16bit operand. float inputs only.
                     bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
                     For V_FMA_MIX* opcodes, this acts as an ABS (absolute value) modifier.
                     For DOT…IU… and WMMA…IU… NEG_HI behavior is undefined.
OPSEL        3       Select the high (1) or low (0) operand as input to the operation that results in the lower-half of the
                     destination. [0] = src0, [1] = src1, [2] = src2
                     If either the source operand or destination operand is 32bits, the corresponding OPSEL bit must set
                     to zero. This rule does not apply to MIX instructions, which have a unique interpretation of OPSEL. See
                     notes below.
                     OPSEL works for 16-bit VGPR, SGPR and literal-constant sources; for inline constant sources OPSEL
                     must be zero (value only exists in lower 16 bits). (only wave64 uses OPSEL[1]), and OPSEL[2] is
                     unused.
OPSEL_HI     3       Select the high (1) or low (0) operand as input to the operation that results in the upper-half of the
                     destination. [0] = src0, [1] = src1, [2] = src2. Concatenation of ISA fields {OPSLH2, OPSLH}. If either
                     the source operand or destination operand is 32bits or is a constant, the corresponding OPSEL_HI
                     bit must set to zero. This rule does not apply to MIX instructions, which have a unique interpretation of
                     OPSEL. See notes below.
CM           1       clamp result.
                     Float arithmetic: clamp result to [0, 1.0]; -0 is clamped to +0.
                     Signed integer arithmetic: clamp result to [INT_MIN, INT_MAX]
                     Unsigned integer arithmetic: clamp result to [0, UINT_MAX]
                     Where "INT_MIN" and "INT_MAX" are the largest negative and positive representable integers for
                     the size of integer being used (16, 32 or 64 bit). "UINT_MAX" is the largest unsigned int.

OPSEL for MIX instructions

   MIX, MIXLO and MIXHI interpret OPSEL and OPSEL_HI as three 2-bit fields, one per source operand:
   { OPSEL_HI[0], OPSEL[0] } controls source0;
   { OPSEL_HI[1], OPSEL[1] } controls source1;
   { OPSEL_HI[2], OPSEL[2] } controls source2.

   These 2-bit fields control source-selection for each of the 3 source operands:
        2’b00: Src[31:0] as FP32
        2’b01: Src[31:0] as FP32
        2’b10: Src[15:0] as FP16
        2’b11: Src[31:16] as FP16

V_WMMA…IU… and V_DOT4…IU… with NEG:

   These instructions use the NEG[1:0] bits to indicate signed (0=unsigned, 1=signed) per input source
   instead of meaning "negate". NEG[2] should be set to zero (behavior is undefined). NEG_HI must be zero.

DOT4_F32_{FP8,BF8}_{FP8,BF8}

   OPSEL must be zero, and OPSEL_HI must be 7. Only SRC2 can apply NEG/NEG_HI (NEG_HI acts as the
   ABS modifier for these ops). OMOD and CLAMP are not supported, and round-mode is round-to-nearest-
   even. Exceptions are not reported.

WMMA_*_{FP8,BF8}_{FP8,BF8}

   No OPSEL (has other uses), ABS, NEG, OMOD, DPP, FP16_OVFL or clamp support. Matrix sources A and
   B must come from VGPRs; C may be a VGPR or inline constant. Exceptions are not reported.
