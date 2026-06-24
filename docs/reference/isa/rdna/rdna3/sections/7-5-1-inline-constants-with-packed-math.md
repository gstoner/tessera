# 7.5.1. Inline Constants with Packed Math

> RDNA3 ISA — pages 76–76

Field        Size    Description
OPSEL_HI 3           Select the high (1) or low (0) operand as input to the operation that results in the upper-half of the
{[60:59],[14]}       destination. [0] = src0, [1] = src1, [2] = src2. Concatenation of ISA fields { OPSLH, OPSLH0 }. If either
                     the source operand or destination operand is 32bits or is a constant, the corresponding OPSEL_HI
                     bit must set to zero. This rule does not apply to MIX instructions, which have a unique interpretation of
                     OPSEL. See notes below.
CLMP         1       clamp result.
                     Float arithmetic: clamp result to [0, 1.0]; -0 is clamped to +0.
                     Signed integer arithmetic: clamp result to [min_int, +max_int]
                     Unsigned integer arithmetic: clamp result to [0, +max_uint]
                     Where "min_int" and "max_int" are the largest negative and positive representable integers for the
                     size of integer being used (16, 32 or 64 bit). "max_uint" is the largest unsigned int.

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

V_WMMA…IU… and V_DOT4…IU… with NEG::

   These instructions use the NEG[1:0] bits to indicate signed (0=unsigned, 1=signed) per input source
   instead of meaning "negate". NEG[2] should be set to zero (behavior is undefined). NEG_HI must be zero.

7.5.1. Inline Constants with Packed Math
Inline constants may be used with packed math, but they require the use of OPSEL. Inline constants produce a
value in only the low 16-bits of the 32-bit constant value. Inline constants used with float 16-bit sources produce
an F16 constant value. Without using OPSEL, only the lower half of the source would contain the constant. To
use the inline constant in both halves, use OPSEL to select the lower input for both low and high sources.

BF16 uses 32-bit float constants and then the BF16 operand selects the upper 16 bits of the FP32 constant
(matches the definition of BF16).

For the WMMA_F16_F16_16x16x16 or VOPD DOT2_F32_F16, hardware automatically selects the low 16 bits of
the constant.

Any packed math instructions that use data sizes less than 16 bits do not work with inline constants, other than
the DOT instructions below:

Opcode                         inline                                                        OPSEL
DOT4_I32_IU8                   use 32bit inline src0/1 (ignore OPSEL)                        OPSEL/OPSEL_HI on src0/1
DOT8_I32_IU4                   use 32bit inline src0/1 (ignore OPSEL)                        OPSEL/OPSEL_HI on src0/1
DOT4_U32_U8                    use 32bit inline src0/1 (ignore OPSEL)                        OPSEL/OPSEL_HI on src0/1
