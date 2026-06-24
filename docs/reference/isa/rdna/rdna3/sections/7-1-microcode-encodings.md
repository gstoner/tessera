# 7.1. Microcode Encodings

> RDNA3 ISA — pages 63–64

Chapter 7. Vector ALU Operations
Vector ALU instructions (VALU) perform an arithmetic or logical operations on data for each of 32 or 64
threads and write results back to VGPRs, SGPRs or the EXEC mask.

Parameter interpolation is a two step process involving an LDS instruction followed by a VALU instruction and
is described in: Parameter Interpolation

Vector ALU (VALU) instructions control the SIMD32’s math unit and operate on 32 work-items of data at a time.
Each instruction may take input from either VGPRs, SGPRs or constants and typically returns results to VGPRs.
Mask results and carry-out are returned to SGPRs. The ALU provides operations that work on 16, 32 and 64-bit
data of both integer and float types. The ALU also supports "packed" data types that pack 2 16-bit values into
one VGPR, or 4 8-bit values into a VGPR.

7.1. Microcode Encodings
VALU instructions are encoded in one of these ways:

Name           Size         Function                                                Modifiers
VOP1           32 bit       VALU op with 1 input                                    -
VOP2           32 bit       VALU op with 2 inputs                                   -
VOP3           64 bit       VALU op with 3 inputs, or a VOP1,2,C instruction        abs, neg, omod, clamp
VOP3SD         64 bit       VALU op with 3 inputs and SDST                          neg, omod, clamp
VOPC           32 bit       VALU compare op with 2 inputs, writes to VCC/EXEC       -
VOP3P          64 bit       VALU op with 3 inputs using packed math                 neg, clamp
VOPD           64 bit       VALU dual opcode : 2 operations in one instruction      -

Many VALU instructions are available in two encodings: VOP3 that uses 64-bits of instruction, and one of three
32-bit encodings that offer a restricted set of capabilities but smaller code size. Some instructions are only
available in the VOP3 encoding. When an instruction is available in two microcode formats, it is up to the user
to decide which to use. It is recommended to use the 32-bit encoding whenever possible. VOP2 can also be used

for "ACCUM" type ops where the third input is implied to be the same as the dest.

Advantages of using VOP3 include:
  • More flexibility in source addressing (all source fields are 9 bits)
  • NEG, ABS, and OMOD fields (for floating point only)
  • CLAMP field for output range limiting
  • Ability to select alternate source and destination registers for VCC (carry in and out)

The following VOP1 and VOP2 instructions may not be promoted to VOP3:
  • swap and swaprel
  • fmamk, fmaak, pk_fmac

The VOP3 encoding has two variants:
  • VOP3 - used for most instructions including V_CMP*; has OPSEL and ABS fields
  • VOP3SD - has an SDST field instead of OPSEL and ABS. This encoding is used only for:
     ◦ V_{ADD,SUB,SUBREV}_CO_CI_U32, V_{ADD,SUB,SUBREV}_CO_U32 (adds with carry-out)
        ◦ V_DIV_SCALE_{F32, F64}, V_MAD_U64_U32, V_MAD_I64_I32.
        ◦ V_DOT2ACC_F32_F16
        ◦ VOP3SD is not used for V_CMP*.

Any of the VALU microcode formats may use a 32-bit literal constant, as well VOP3. Note however that VOP3
plus a literal makes a 96-bit instruction and excessive use of this combination may reduce performance.

VOP3P is for instructions that use "packed math": instructions that performs an operation on a pair of input
values that are packed into the high and low 16-bits of each operand; the two 16-bit results are written to a
single VGPR as two packed values.

Field       Size     Description
OP          varies   instruction opcode
SRC0        9        first instruction argument. May come from: vgpr, sgpr, VCC, M0, EXEC, SCC, or a constant
SRC1        9        second instruction argument. May come from: vgpr, sgpr, VCC, M0, EXEC, SCC, or a constant
VSRC1       8        second instruction argument. May come from: vgpr only
SRC2        9        third instruction argument. May come from: vgpr, sgpr, VCC, M0, EXEC, SCC, or a constant
VDST        8        VGPR that takes the result.
                     For V_READLANE and V_CMP, indicates the SGPR that receives the result. This cannot be M0 or EXEC.
SDST        8        SGPR that takes the result of operations that produce a scalar output. Can’t be M0 or EXEC. Supports
                     NULL to not write any SDST.
                     Used for: V_{ADD,SUB,SUBREV}_CO_U32, V_{ADD,SUB,SUBREV}_CO_CI_U32, V_DIV_SCALE*; not
                     used for V_CMP.
OMOD        2        output modifier. for float results only.
                     0 = no modifier, 1=multiply result by 2, 2=multiply result by 4, 3=divide result by 2
NEG         3        negate the input (invert sign bit). float inputs only.
                     bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
ABS         3        apply absolute value on input. float inputs only. applied before 'neg'.
                     bit 0 is for src0, bit 1 is for src1 and bit 2 is for src2.
