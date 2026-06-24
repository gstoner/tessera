# 16.13. VINTERP Instructions

> RDNA3.5 ISA — pages 538–542

16.13. VINTERP Instructions
Parameter interpolation VALU instructions.

V_INTERP_P10_F32                                                                                                         0

Given the P10 parameter of an attribute, the I coordinate and the P0 parameter as single-precision float inputs,
compute the first part of parameter interpolation and store the intermediate result into a vector register. Use
V_INTERP_P2_F32 to complete the operation.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f32 = fma(VGPR[(laneId.u32 & 0xfffffffcU) + 1U][SRC0.u32].f32, S1.f32, VGPR[laneId.u32 &
  0xfffffffcU][SRC2.u32].f32)

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an
LDS_PARAM_LOAD instruction.

This operation performs a V_FMA_F32 operation using fixed DPP8 settings. S0 and S2 refer to a VGPR that
contains packed interpolation data. S1 is the I coordinate.

S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}.

S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}.

Example usage:

  s_mov_b32 m0, s0               // assume s0 contains newprim mask
  lds_param_load v0, attr0       // v0 is a temporary register
  v_interp_p10_f32 v3, v0, v1, v0 // v1 contains i coordinate
  v_interp_p2_f32 v3, v0, v2, v3      // v2 contains j coordinate

V_INTERP_P2_F32                                                                                                          1

Given the P20 parameter of an attribute, the J coordinate and the result of a prior V_INTERP_P10_F32
instruction as single-precision float inputs, compute the second part of parameter interpolation and store the
final result into a vector register.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f32 = fma(VGPR[(laneId.u32 & 0xfffffffcU) + 2U][SRC0.u32].f32, S1.f32, S2.f32)

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an
LDS_PARAM_LOAD instruction.

This operation performs a V_FMA_F32 operation using fixed DPP8 settings. S0 refers to a VGPR that contains
packed interpolation data. S1 is the J coordinate. S2 is the result of a previous V_INTERP_P10_F32 instruction.

S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.

V_INTERP_P10_F16_F32                                                                                                     2

Given a half-precision float P10 parameter of an attribute, a single-precision float I coordinate and a half-
precision float P0 parameter as inputs, compute the first part of parameter interpolation and store the
intermediate result in single-precision float format into a vector register. Use V_INTERP_P2_F16_F32 to
complete the operation.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f32 = fma(32'F(VGPR[(laneId.u32 & 0xfffffffcU) + 1U][SRC0.u32].f16), S1.f32, 32'F(VGPR[laneId.u32 &
  0xfffffffcU][SRC2.u32].f16))

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an
LDS_PARAM_LOAD instruction.

This operation performs a hybrid 16/32-bit fused multiply add operation using fixed DPP8 settings. S0 and S2

refer to a VGPR that contains packed interpolation data. S1 is the I coordinate.

S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}.

S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}.

OPSEL is used to specify which half of S0 and S2 to read from.

Note the I coordinate is 32-bit and the destination is also 32-bit.

V_INTERP_P2_F16_F32                                                                                                      3

Given a half-precision float P20 parameter of an attribute, a single-precision float J coordinate and the result of
a prior V_INTERP_P10_F16_F32 instruction as inputs, compute the second part of parameter interpolation and
store the final result into a vector register.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f16 = 16'F(fma(32'F(VGPR[(laneId.u32 & 0xfffffffcU) + 2U][SRC0.u32].f16), S1.f32, S2.f32))

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an
LDS_PARAM_LOAD instruction.

This operation performs a hybrid 16/32-bit fused multiply add operation using fixed DPP8 settings. S0 refers to
a VGPR that contains packed interpolation data. S1 is the J coordinate. S2 is the result of a previous
V_INTERP_P10_F16_F32 instruction.

S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.

OPSEL is used to specify which half of S0 to read from and which half of D0 to write to.

Note the J coordinate is 32-bit.

V_INTERP_P10_RTZ_F16_F32                                                                                                 4

Given a half-precision float P10 parameter of an attribute, a single-precision float I coordinate and a half-
precision float P0 parameter as inputs, compute the first part of parameter interpolation using round toward
zero semantics and store the intermediate result in single-precision float format into a vector register. Use
V_INTERP_P2_RTZ_F16_F32 to complete the operation.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f32 = fma(32'F(VGPR[(laneId.u32 & 0xfffffffcU) + 1U][SRC0.u32].f16), S1.f32, 32'F(VGPR[laneId.u32 &
  0xfffffffcU][SRC2.u32].f16))

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an
LDS_PARAM_LOAD instruction.

This operation performs a hybrid 16/32-bit fused multiply add operation using fixed DPP8 settings. S0 and S2
refer to a VGPR that contains packed interpolation data. S1 is the I coordinate.

S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}.

S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}.

OPSEL is used to specify which half of S0 and S2 to read from.

Note the I coordinate is 32-bit and the destination is also 32-bit.

Rounding mode is overridden to round toward zero.

V_INTERP_P2_RTZ_F16_F32                                                                                                  5

Given a half-precision float P20 parameter of an attribute, a single-precision float J coordinate and the result of
a prior V_INTERP_P10_RTZ_F16_F32 instruction as inputs, compute the second part of parameter
interpolation using round toward zero semantics and store the final result into a vector register.

The overall calculation is:

\[ + \begin{aligned} + R &= \left( P\sb{0} + P\sb{10} \cdot I \right) + P\sb{20} \cdot J \\ + P\sb{i0} &= P\sb{i} - P\sb{0}
\quad i \in \{1, 2\} \\ + P\sb{i} &= \text{attribute value at vertex $\harpoon{v}\sb{i}$} \quad i \in \{0, 1, 2\} \\ +
\unitvector{\imath} + &= \harpoon{v}\sb{1} - \harpoon{v}\sb{0} \quad\text{Basis vector for $I$ coordinate} \\ +
\unitvector{\jmath} + &= \harpoon{v}\sb{2} - \harpoon{v}\sb{0} \quad\text{Basis vector for $J$ coordinate} \\ +
\end{aligned} + \]

  D0.f32 = fma(32'F(VGPR[(laneId.u32 & 0xfffffffcU) + 2U][SRC0.u32].f16), S1.f32, S2.f32)

Notes

This operation is designed for use in pixel shaders where attribute data has previously been loaded with an

LDS_PARAM_LOAD instruction.

This operation performs a hybrid 16/32-bit fused multiply add operation using fixed DPP8 settings. S0 refers to
a VGPR that contains packed interpolation data. S1 is the J coordinate. S2 is the result of a previous
V_INTERP_P10_F16_F32 instruction.

S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.

OPSEL is used to specify which half of S0 to read from and which half of D0 to write to.

Note the J coordinate is 32-bit.

Rounding mode is overridden to round toward zero.
