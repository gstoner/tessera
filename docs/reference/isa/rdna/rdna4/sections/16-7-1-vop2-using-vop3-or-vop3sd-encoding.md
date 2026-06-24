# 16.7.1. VOP2 using VOP3 or VOP3SD encoding

> RDNA4 ISA — pages 314–314

  D0.f16 = fma(S0.f16, S1.f16, SIMM32.f16)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_LDEXP_F16                                                                                                       59

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))

Notes

Compare with the ldexp() function in C.

V_PK_FMAC_F16                                                                                                     60

Multiply two packed half-precision float inputs component-wise and accumulate the result into the destination
register using fused multiply add.

  D0[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, D0[31 : 16].f16);
  D0[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, D0[15 : 0].f16)

Notes

VOP2 version of V_PK_FMA_F16 with third source VGPR address is the destination.

16.7.1. VOP2 using VOP3 or VOP3SD encoding
Instructions in this format may also be encoded as VOP3. VOP3 allows access to the extra control bits (e.g. ABS,
OMOD) at the expense of a larger instruction word. The VOP3 opcode is: VOP2 opcode + 0x100.
