# 16.13. VINTERP Instructions

> RDNA3 ISA — pages 507–509

16.13. VINTERP Instructions
Parameter interpolation VALU instructions.

V_INTERP_P10_F32                                                                                             0

Parameter interpolation, first pass.

  D0.f = S0[lane.i % 4 + 1].f * S1.f + S2[lane.i % 4].f

Notes

Performs a V_FMA_F32 operation using fixed DPP8 settings. S0 and S2 refer to a VGPR previously loaded with
LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate.

S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}.

S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}.

Example usage:

  s_mov_b32 m0, s0              // assume s0 contains newprim mask
  lds_param_load v0, attr0      // v0 is a temporary register
  v_interp_p10_f32 v3, v0, v1, v0 // v1 contains i coordinate
  v_interp_p2_f32 v3, v0, v2, v3       // v2 contains j coordinate

V_INTERP_P2_F32                                                                                              1

Parameter interpolation, second pass.

  D0.f = fma(S0[lane.i % 4 + 2].f, S1.f, S2.f)

Notes

Performs a V_FMA_F32 operation using fixed DPP8 settings. S0 refers to a VGPR previously loaded with
LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S2 is the result of a
previous V_INTERP_P10_F32 instruction.

S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.

V_INTERP_P10_F16_F32                                                                                           2

Parameter interpolation, first pass.

  D0.f = 32'F(S0[lane.i % 4 + 1].f16) * S1.f + 32'F(S2[lane.i % 4].f16)

Notes

Performs a hybrid 16/32-bit multiply-add operation using fixed DPP8 settings. S0 and S2 refer to a VGPR
previously loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate.

S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}.

S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}.

OPSEL is allowed for S0 and S2 to specify which half of the register to read from.

Note that the I/J coordinate is 32-bit and the destination is also 32-bit.

V_INTERP_P2_F16_F32                                                                                            3

Parameter interpolation, second pass.

  D0.f16 = 16'F(32'F(S0[lane.i % 4 + 2].f16) * S1.f + S2.f)

Notes

Performs a hybrid 16/32-bit multiply-add operation using fixed DPP8 settings. S0 refers to a VGPR previously
loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S2 is the
result of a previous V_INTERP_P10_F16_F32 instruction.

S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.

OPSEL is allowed for D and S0 to specify which half of the register to write to/read from.

Note that the I/J coordinate is 32-bit and the accumulator input is also 32-bit.

V_INTERP_P10_RTZ_F16_F32                                                                                       4

Same as V_INTERP_P10_F16_F32 except rounding mode is overridden to round toward zero.

V_INTERP_P2_RTZ_F16_F32                                                                                        5

Same as V_INTERP_P2_F16_F32 except rounding mode is overridden to round toward zero.
