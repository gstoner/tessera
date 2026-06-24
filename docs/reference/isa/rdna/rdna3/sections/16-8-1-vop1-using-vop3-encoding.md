# 16.8.1. VOP1 using VOP3 encoding

> RDNA3 ISA — pages 303–303

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CVT_U32_U16                                                                                               107

Convert from an 16-bit unsigned integer to a 32-bit unsigned integer, zero extending as needed.

  D0 = { 16'0, S0.u16 }

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

16.8.1. VOP1 using VOP3 encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x180.
