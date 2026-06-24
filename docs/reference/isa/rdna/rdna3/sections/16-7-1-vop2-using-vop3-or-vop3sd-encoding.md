# 16.7.1. VOP2 using VOP3 or VOP3SD encoding

> RDNA3 ISA — pages 273–273

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register. Compare with the ldexp() function in C.

  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))

V_PK_FMAC_F16                                                                                                     60

Multiply packed FP16 values and accumulate with destination.

  D0[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, D0[31 : 16].f16);
  D0[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, D0[15 : 0].f16)

Notes

VOP2 version of V_PK_FMA_F16 with third source VGPR address is the destination.

16.7.1. VOP2 using VOP3 or VOP3SD encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x100.
