# 16.7.1. VOP2 using VOP3 or VOP3SD encoding

> RDNA3.5 ISA — pages 290–291

        else
            D0.f16 = S1.f16
        endif
  endif;
  // Inequalities in the above pseudocode behave differently from IEEE
  // when both inputs are +-0.

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input
denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so
output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can
be a denorm and this operation outputs as a denorm regardless of output denorm mode.

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
