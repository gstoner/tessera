# 16.10. VOP3P Instructions

> RDNA3 ISA — pages 360–368

16.10. VOP3P Instructions

V_PK_MAD_I16                                                                                    0

Packed multiply-add on signed shorts.

  D0[31 : 16].i16 = S0[31 : 16].i16 * S1[31 : 16].i16 + S2[31 : 16].i16;
  D0[15 : 0].i16 = S0[15 : 0].i16 * S1[15 : 0].i16 + S2[15 : 0].i16

V_PK_MUL_LO_U16                                                                                 1

Packed multiply on unsigned shorts.

  D0[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16

V_PK_ADD_I16                                                                                    2

Packed addition on signed shorts.

  D0[31 : 16].i16 = S0[31 : 16].i16 + S1[31 : 16].i16;
  D0[15 : 0].i16 = S0[15 : 0].i16 + S1[15 : 0].i16

V_PK_SUB_I16                                                                                    3

Packed subtraction on signed shorts. The second operand is subtracted from the first.

  D0[31 : 16].i16 = S0[31 : 16].i16 - S1[31 : 16].i16;
  D0[15 : 0].i16 = S0[15 : 0].i16 - S1[15 : 0].i16

V_PK_LSHLREV_B16                                                                                4

Packed logical shift left. The shift count is in the first operand.

  D0[31 : 16].u16 = (S1[31 : 16].u16 << S0.u[19 : 16].u);
  D0[15 : 0].u16 = (S1[15 : 0].u16 << S0.u[3 : 0].u)

V_PK_LSHRREV_B16                                                                                      5

Packed logical shift right. The shift count is in the first operand.

  D0[31 : 16].u16 = (S1[31 : 16].u16 >> S0.u[19 : 16].u);
  D0[15 : 0].u16 = (S1[15 : 0].u16 >> S0.u[3 : 0].u)

V_PK_ASHRREV_I16                                                                                      6

Packed arithmetic shift right (preserve sign bit). The shift count is in the first operand.

  D0[31 : 16].i16 = (S1[31 : 16].i16 >> S0.u[19 : 16].u);
  D0[15 : 0].i16 = (S1[15 : 0].i16 >> S0.u[3 : 0].u)

V_PK_MAX_I16                                                                                          7

Packed maximum of signed shorts.

  D0[31 : 16].i16 = S0[31 : 16].i16 >= S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  D0[15 : 0].i16 = S0[15 : 0].i16 >= S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16

V_PK_MIN_I16                                                                                          8

Packed minimum of signed shorts.

  D0[31 : 16].i16 = S0[31 : 16].i16 < S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  D0[15 : 0].i16 = S0[15 : 0].i16 < S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16

V_PK_MAD_U16                                                                                          9

Packed multiply-add on unsigned shorts.

  D0[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16 + S2[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16 + S2[15 : 0].u16

V_PK_ADD_U16                                                                                         10

Packed addition on unsigned shorts.

  D0[31 : 16].u16 = S0[31 : 16].u16 + S1[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 + S1[15 : 0].u16

V_PK_SUB_U16                                                                                         11

Packed subtraction on unsigned shorts. The second operand is subtracted from the first.

  D0[31 : 16].u16 = S0[31 : 16].u16 - S1[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 - S1[15 : 0].u16

V_PK_MAX_U16                                                                                         12

Packed maximum of unsigned shorts.

  D0[31 : 16].u16 = S0[31 : 16].u16 >= S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 >= S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16

V_PK_MIN_U16                                                                                         13

Packed minimum of unsigned shorts.

  D0[31 : 16].u16 = S0[31 : 16].u16 < S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  D0[15 : 0].u16 = S0[15 : 0].u16 < S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16

V_PK_FMA_F16                                                                                         14

Packed fused-multiply-add of FP16 values.

  D0[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, S2[31 : 16].f16);
  D0[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, S2[15 : 0].f16)

V_PK_ADD_F16                                                                         15

Packed addition of FP16 values.

  D0[31 : 16].f16 = S0[31 : 16].f16 + S1[31 : 16].f16;
  D0[15 : 0].f16 = S0[15 : 0].f16 + S1[15 : 0].f16

V_PK_MUL_F16                                                                         16

Packed multiply of FP16 values.

  D0[31 : 16].f16 = S0[31 : 16].f16 * S1[31 : 16].f16;
  D0[15 : 0].f16 = S0[15 : 0].f16 * S1[15 : 0].f16

V_PK_MIN_F16                                                                         17

Packed minimum of FP16 values.

  D0[31 : 16].f16 = v_min_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  D0[15 : 0].f16 = v_min_f16(S0[15 : 0].f16, S1[15 : 0].f16)

V_PK_MAX_F16                                                                         18

Packed maximum of FP16 values.

  D0[31 : 16].f16 = v_max_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  D0[15 : 0].f16 = v_max_f16(S0[15 : 0].f16, S1[15 : 0].f16)

V_DOT2_F32_F16                                                                       19

Dot product of packed FP16 values.

  tmp = 32'F(S0[15 : 0].f16) * 32'F(S1[15 : 0].f16);
  tmp += 32'F(S0[31 : 16].f16) * 32'F(S1[31 : 16].f16);
  tmp += S2.f;
  D0.f = tmp

V_DOT4_I32_IU8                                                                                 22

Dot product of signed or unsigned bytes.

  declare A : 32'I[4];
  declare B : 32'I[4];
  // Figure out whether inputs are signed/unsigned.
  for i in 0 : 3 do
        A8 = S0[i * 8 + 7 : i * 8];
        B8 = S1[i * 8 + 7 : i * 8];
        A[i] = NEG[0].u1 ? 32'I(signext(A8.i8)) : 32'I(32'U(A8.u8));
        B[i] = NEG[1].u1 ? 32'I(signext(B8.i8)) : 32'I(32'U(B8.u8))
  endfor;
  C = S2.i;
  // Signed multiplier/adder. Extend unsigned inputs with leading 0.
  D0.i = A[0] * B[0];
  D0.i += A[1] * B[1];
  D0.i += A[2] * B[2];
  D0.i += A[3] * B[3];
  D0.i += C

Notes

This opcode does not depend on the inference or deep learning features being enabled.

V_DOT4_U32_U8                                                                                  23

Dot product of unsigned bytes.

  tmp = 32'U(S0[7 : 0].u8) * 32'U(S1[7 : 0].u8);
  tmp += 32'U(S0[15 : 8].u8) * 32'U(S1[15 : 8].u8);
  tmp += 32'U(S0[23 : 16].u8) * 32'U(S1[23 : 16].u8);
  tmp += 32'U(S0[31 : 24].u8) * 32'U(S1[31 : 24].u8);
  tmp += S2.u;
  D0.u = tmp

Notes

This opcode does not depend on the inference or deep learning features being enabled.

V_DOT8_I32_IU4                                                                24

Dot product of signed or unsigned nibbles.

  declare A : 32'I[8];
  declare B : 32'I[8];
  // Figure out whether inputs are signed/unsigned.
  for i in 0 : 7 do
      A4 = S0[i * 4 + 3 : i * 4];
      B4 = S1[i * 4 + 3 : i * 4];
      A[i] = NEG[0].u1 ? 32'I(signext(A4.i4)) : 32'I(32'U(A4.u4));
      B[i] = NEG[1].u1 ? 32'I(signext(B4.i4)) : 32'I(32'U(B4.u4))
  endfor;
  C = S2.i;
  // Signed multiplier/adder. Extend unsigned inputs with leading 0.
  D0.i = A[0] * B[0];
  D0.i += A[1] * B[1];
  D0.i += A[2] * B[2];
  D0.i += A[3] * B[3];
  D0.i += A[4] * B[4];
  D0.i += A[5] * B[5];
  D0.i += A[6] * B[6];
  D0.i += A[7] * B[7];
  D0.i += C

V_DOT8_U32_U4                                                                 25

Dot product of unsigned nibbles.

  tmp = 32'U(S0[3 : 0].u4) * 32'U(S1[3 : 0].u4);
  tmp += 32'U(S0[7 : 4].u4) * 32'U(S1[7 : 4].u4);
  tmp += 32'U(S0[11 : 8].u4) * 32'U(S1[11 : 8].u4);
  tmp += 32'U(S0[15 : 12].u4) * 32'U(S1[15 : 12].u4);
  tmp += 32'U(S0[19 : 16].u4) * 32'U(S1[19 : 16].u4);
  tmp += 32'U(S0[23 : 20].u4) * 32'U(S1[23 : 20].u4);
  tmp += 32'U(S0[27 : 24].u4) * 32'U(S1[27 : 24].u4);
  tmp += 32'U(S0[31 : 28].u4) * 32'U(S1[31 : 28].u4);
  tmp += S2.u;
  D0.u = tmp

V_DOT2_F32_BF16                                                               26

Dot product of packed brain-float values.

  tmp = 32'F(S0[15 : 0].bf16) * 32'F(S1[15 : 0].bf16);
  tmp += 32'F(S0[31 : 16].bf16) * 32'F(S1[31 : 16].bf16);
  tmp += S2.f;

  D0.f = tmp

V_FMA_MIX_F32                                                                                                    32

Fused-multiply-add of single-precision values with MIX encoding.

Size and location of S0, S1 and S2 controlled by OPSEL: 0=src[31:0], 1=src[31:0], 2=src[15:0], 3=src[31:16]. Also,
for FMA_MIX, the NEG_HI field acts instead as an absolute-value modifier.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
      if !OPSEL_HI.u3[i] then
            in[i] = S[i].f
      elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
      else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
      endif
  endfor;
  D0[31 : 0].f = fma(in[0], in[1], in[2])

V_FMA_MIXLO_F16                                                                                                  33

Fused-multiply-add of FP16 values with MIX encoding, result stored in low 16 bits of destination.

Size and location of S0, S1 and S2 controlled by OPSEL: 0=src[31:0], 1=src[31:0], 2=src[15:0], 3=src[31:16]. Also,
for FMA_MIX, the NEG_HI field acts instead as an absolute-value modifier.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
      if !OPSEL_HI.u3[i] then
            in[i] = S[i].f
      elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
      else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
      endif
  endfor;
  D0[15 : 0].f16 = f32_to_f16(fma(in[0], in[1], in[2]))

V_FMA_MIXHI_F16                                                                                                  34

Fused-multiply-add of FP16 values with MIX encoding, result stored in HIGH 16 bits of destination.

Size and location of S0, S1 and S2 controlled by OPSEL: 0=src[31:0], 1=src[31:0], 2=src[15:0], 3=src[31:16]. Also,
for FMA_MIX, the NEG_HI field acts instead as an absolute-value modifier.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
      if !OPSEL_HI.u3[i] then
            in[i] = S[i].f
      elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
      else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
      endif
  endfor;
  D0[31 : 16].f16 = f32_to_f16(fma(in[0], in[1], in[2]))

V_WMMA_F32_16X16X16_F16                                                                                          64

WMMA matrix multiplication with F16 multiplicands and single precision result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f32(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f32(16x16)";
  EXEC = saved_exec

V_WMMA_F32_16X16X16_BF16                                                                                         65

WMMA matrix multiplication with brain float multiplicands and single precision result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f32(16x16) = S0.bf16(16x16) * S1.bf16(16x16) + S2.f32(16x16)";
  EXEC = saved_exec

V_WMMA_F16_16X16X16_F16                                                                                          66

WMMA matrix multiplication with F16 multiplicands and F16 result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f16(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f16(16x16)";
  EXEC = saved_exec

V_WMMA_BF16_16X16X16_BF16                                                                              67

WMMA matrix multiplication with brain float multiplicands and brain float result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.bf16(16x16) = S0.bf16(16x16) * S1.bf16(16x16) + S2.bf16(16x16)";
  EXEC = saved_exec

V_WMMA_I32_16X16X16_IU8                                                                                68

WMMA matrix multiplication with 8-bit integer multiplicands and signed 32-bit integer result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.i32(16x16) = S0.iu8(16x16) * S1.iu8(16x16) + S2.i32(16x16)";
  EXEC = saved_exec

V_WMMA_I32_16X16X16_IU4                                                                                69

WMMA matrix multiplication with 4-bit integer multiplicands and signed 32-bit integer result.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.i32(16x16) = S0.iu4(16x16) * S1.iu4(16x16) + S2.i32(16x16)";
  EXEC = saved_exec
