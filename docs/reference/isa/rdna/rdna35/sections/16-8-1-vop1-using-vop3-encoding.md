# 16.8.1. VOP1 using VOP3 encoding

> RDNA3.5 ISA — pages 321–321

Input and output modifiers not supported.

V_CVT_I32_I16                                                                                                    106

Convert from a signed 16-bit integer input to a signed 32-bit integer value using sign extension and store the
result into a vector register.

  D0.i32 = 32'I(signext(S0.i16))

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CVT_U32_U16                                                                                                    107

Convert from an unsigned 16-bit integer input to an unsigned 32-bit integer value using zero extension and
store the result into a vector register.

  D0 = { 16'0, S0.u16 }

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

16.8.1. VOP1 using VOP3 encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x180.
