# 16.10. VOP3P Instructions

> RDNA3.5 ISA — pages 382–392

16.10. VOP3P Instructions

V_PK_MAD_I16                                                                                                  0

Multiply two packed signed 16-bit integer inputs component-wise, add a packed signed 16-bit integer value
from a third input component-wise, and store the result into a vector register.

  tmp[31 : 16].i16 = S0[31 : 16].i16 * S1[31 : 16].i16 + S2[31 : 16].i16;
  tmp[15 : 0].i16 = S0[15 : 0].i16 * S1[15 : 0].i16 + S2[15 : 0].i16;
  D0.b32 = tmp.b32

V_PK_MUL_LO_U16                                                                                               1

Multiply two packed unsigned 16-bit integer inputs component-wise and store the low bits of each resulting
component into a vector register.

  tmp[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_ADD_I16                                                                                                  2

Add two packed signed 16-bit integer inputs component-wise and store the result into a vector register. No
carry-in or carry-out support.

  tmp[31 : 16].i16 = S0[31 : 16].i16 + S1[31 : 16].i16;
  tmp[15 : 0].i16 = S0[15 : 0].i16 + S1[15 : 0].i16;
  D0.b32 = tmp.b32

V_PK_SUB_I16                                                                                                  3

Subtract the second packed signed 16-bit integer input from the first input component-wise and store the result
into a vector register. No carry-in or carry-out support.

  tmp[31 : 16].i16 = S0[31 : 16].i16 - S1[31 : 16].i16;
  tmp[15 : 0].i16 = S0[15 : 0].i16 - S1[15 : 0].i16;

  D0.b32 = tmp.b32

V_PK_LSHLREV_B16                                                                                                      4

Given a packed shift count in the first vector input, calculate the component-wise logical shift left of the second
packed vector input and store the result into a vector register.

  tmp[31 : 16].u16 = (S1[31 : 16].u16 << S0.u32[19 : 16].u32);
  tmp[15 : 0].u16 = (S1[15 : 0].u16 << S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32

V_PK_LSHRREV_B16                                                                                                      5

Given a packed shift count in the first vector input, calculate the component-wise logical shift right of the
second packed vector input and store the result into a vector register.

  tmp[31 : 16].u16 = (S1[31 : 16].u16 >> S0.u32[19 : 16].u32);
  tmp[15 : 0].u16 = (S1[15 : 0].u16 >> S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32

V_PK_ASHRREV_I16                                                                                                      6

Given a packed shift count in the first vector input, calculate the component-wise arithmetic shift right
(preserving sign bit) of the second packed vector input and store the result into a vector register.

  tmp[31 : 16].i16 = (S1[31 : 16].i16 >> S0.u32[19 : 16].u32);
  tmp[15 : 0].i16 = (S1[15 : 0].i16 >> S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32

V_PK_MAX_I16                                                                                                          7

Select the component-wise maximum of two packed signed 16-bit integer inputs and store the selected values
into a vector register.

  tmp[31 : 16].i16 = S0[31 : 16].i16 >= S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  tmp[15 : 0].i16 = S0[15 : 0].i16 >= S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16;
  D0.b32 = tmp.b32

V_PK_MIN_I16                                                                                                   8

Select the component-wise minimum of two packed signed 16-bit integer inputs and store the selected values
into a vector register.

  tmp[31 : 16].i16 = S0[31 : 16].i16 < S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  tmp[15 : 0].i16 = S0[15 : 0].i16 < S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16;
  D0.b32 = tmp.b32

V_PK_MAD_U16                                                                                                   9

Multiply two packed unsigned 16-bit integer inputs component-wise, add a packed unsigned 16-bit integer
value from a third input component-wise, and store the result into a vector register.

  tmp[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16 + S2[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16 + S2[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_ADD_U16                                                                                                 10

Add two packed unsigned 16-bit integer inputs component-wise and store the result into a vector register. No
carry-in or carry-out support.

  tmp[31 : 16].u16 = S0[31 : 16].u16 + S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 + S1[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_SUB_U16                                                                                                 11

Subtract the second packed unsigned 16-bit integer input from the first input component-wise and store the
result into a vector register. No carry-in or carry-out support.

  tmp[31 : 16].u16 = S0[31 : 16].u16 - S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 - S1[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_MAX_U16                                                                                                 12

Select the component-wise maximum of two packed unsigned 16-bit integer inputs and store the selected

values into a vector register.

  tmp[31 : 16].u16 = S0[31 : 16].u16 >= S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 >= S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_MIN_U16                                                                                                  13

Select the component-wise minimum of two packed unsigned 16-bit integer inputs and store the selected
values into a vector register.

  tmp[31 : 16].u16 = S0[31 : 16].u16 < S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 < S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16;
  D0.b32 = tmp.b32

V_PK_FMA_F16                                                                                                  14

Multiply two packed half-precision float inputs component-wise and add a third input component-wise using
fused multiply add, and store the result into a vector register.

  declare tmp : 32'B;
  tmp[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, S2[31 : 16].f16);
  tmp[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, S2[15 : 0].f16);
  D0.b32 = tmp

V_PK_ADD_F16                                                                                                  15

Add two packed half-precision float inputs component-wise and store the result into a vector register. No carry-
in or carry-out support.

  tmp[31 : 16].f16 = S0[31 : 16].f16 + S1[31 : 16].f16;
  tmp[15 : 0].f16 = S0[15 : 0].f16 + S1[15 : 0].f16;
  D0.b32 = tmp.b32

V_PK_MUL_F16                                                                                                  16

Multiply two packed half-precision float inputs component-wise and store the result into a vector register.

  tmp[31 : 16].f16 = S0[31 : 16].f16 * S1[31 : 16].f16;

  tmp[15 : 0].f16 = S0[15 : 0].f16 * S1[15 : 0].f16;
  D0.b32 = tmp.b32

V_PK_MIN_F16                                                                                                 17

Select the component-wise minimum of two packed half-precision float inputs and store the result into a vector
register.

  tmp[31 : 16].f16 = v_min_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  tmp[15 : 0].f16 = v_min_f16(S0[15 : 0].f16, S1[15 : 0].f16);
  D0.b32 = tmp.b32

V_PK_MAX_F16                                                                                                 18

Select the component-wise maximum of two packed half-precision float inputs and store the result into a
vector register.

  tmp[31 : 16].f16 = v_max_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  tmp[15 : 0].f16 = v_max_f16(S0[15 : 0].f16, S1[15 : 0].f16);
  D0.b32 = tmp.b32

V_DOT2_F32_F16                                                                                               19

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain, add
a single-precision float value from the third input and store the result into a vector register.

  tmp = S2.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp

V_DOT4_I32_IU8                                                                                               22

Compute the dot product of two packed 4-D unsigned 8-bit integer inputs in the signed 32-bit integer domain,
add a signed 32-bit integer value from the third input and store the result into a vector register.

The NEG modifier is used to specify whether each input is signed or unsigned: 0=unsigned input, 1=signed
input.

  declare A : 32'I[4];

  declare B : 32'I[4];
  // Figure out whether inputs are signed/unsigned.
  for i in 0 : 3 do
        A8 = S0[i * 8 + 7 : i * 8];
        B8 = S1[i * 8 + 7 : i * 8];
        A[i] = NEG[0].u1 ? 32'I(signext(A8.i8)) : 32'I(32'U(A8.u8));
        B[i] = NEG[1].u1 ? 32'I(signext(B8.i8)) : 32'I(32'U(B8.u8))
  endfor;
  C = S2.i32;
  // Signed multiplier/adder. Extend unsigned inputs with leading 0.
  tmp = C.i32;
  tmp += A[0] * B[0];
  tmp += A[1] * B[1];
  tmp += A[2] * B[2];
  tmp += A[3] * B[3];
  D0.i32 = tmp

Notes

This opcode does not depend on the inference or deep learning features being enabled.

V_DOT4_U32_U8                                                                                                    23

Compute the dot product of two packed 4-D unsigned 8-bit integer inputs in the unsigned 32-bit integer
domain, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

  tmp = S2.u32;
  tmp += u8_to_u32(S0[7 : 0].u8) * u8_to_u32(S1[7 : 0].u8);
  tmp += u8_to_u32(S0[15 : 8].u8) * u8_to_u32(S1[15 : 8].u8);
  tmp += u8_to_u32(S0[23 : 16].u8) * u8_to_u32(S1[23 : 16].u8);
  tmp += u8_to_u32(S0[31 : 24].u8) * u8_to_u32(S1[31 : 24].u8);
  D0.u32 = tmp

Notes

This opcode does not depend on the inference or deep learning features being enabled.

V_DOT8_I32_IU4                                                                                                   24

Compute the dot product of two packed 8-D unsigned 4-bit integer inputs in the signed 32-bit integer domain,
add a signed 32-bit integer value from the third input and store the result into a vector register.

The NEG modifier is used to specify whether each input is signed or unsigned: 0=unsigned input, 1=signed
input.

  declare A : 32'I[8];
  declare B : 32'I[8];
  // Figure out whether inputs are signed/unsigned.

  for i in 0 : 7 do
      A4 = S0[i * 4 + 3 : i * 4];
      B4 = S1[i * 4 + 3 : i * 4];
      A[i] = NEG[0].u1 ? 32'I(signext(A4.i4)) : 32'I(32'U(A4.u4));
      B[i] = NEG[1].u1 ? 32'I(signext(B4.i4)) : 32'I(32'U(B4.u4))
  endfor;
  C = S2.i32;
  // Signed multiplier/adder. Extend unsigned inputs with leading 0.
  tmp = C.i32;
  tmp += A[0] * B[0];
  tmp += A[1] * B[1];
  tmp += A[2] * B[2];
  tmp += A[3] * B[3];
  tmp += A[4] * B[4];
  tmp += A[5] * B[5];
  tmp += A[6] * B[6];
  tmp += A[7] * B[7];
  D0.i32 = tmp

V_DOT8_U32_U4                                                                                                    25

Compute the dot product of two packed 8-D unsigned 4-bit integer inputs in the unsigned 32-bit integer
domain, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

  tmp = S2.u32;
  tmp += u4_to_u32(S0[3 : 0].u4) * u4_to_u32(S1[3 : 0].u4);
  tmp += u4_to_u32(S0[7 : 4].u4) * u4_to_u32(S1[7 : 4].u4);
  tmp += u4_to_u32(S0[11 : 8].u4) * u4_to_u32(S1[11 : 8].u4);
  tmp += u4_to_u32(S0[15 : 12].u4) * u4_to_u32(S1[15 : 12].u4);
  tmp += u4_to_u32(S0[19 : 16].u4) * u4_to_u32(S1[19 : 16].u4);
  tmp += u4_to_u32(S0[23 : 20].u4) * u4_to_u32(S1[23 : 20].u4);
  tmp += u4_to_u32(S0[27 : 24].u4) * u4_to_u32(S1[27 : 24].u4);
  tmp += u4_to_u32(S0[31 : 28].u4) * u4_to_u32(S1[31 : 28].u4);
  D0.u32 = tmp

V_DOT2_F32_BF16                                                                                                  26

Compute the dot product of two packed 2-D BF16 float inputs in the single-precision float domain, add a single-
precision float value from the third input and store the result into a vector register.

  tmp = S2.f32;
  tmp += bf16_to_f32(S0[15 : 0].bf16) * bf16_to_f32(S1[15 : 0].bf16);
  tmp += bf16_to_f32(S0[31 : 16].bf16) * bf16_to_f32(S1[31 : 16].bf16);
  D0.f32 = tmp

V_FMA_MIX_F32                                                                                                    32

Multiply two inputs and add a third input using fused multiply add where the inputs are a mix of half-precision
float and single-precision float values. Store the result into a vector register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0], 2=src[15:0],
3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier for the three

inputs.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
       if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
       elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
       else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
       endif
  endfor;
  D0[31 : 0].f32 = fma(in[0], in[1], in[2])

V_FMA_MIXLO_F16                                                                                                             33

Multiply two inputs and add a third input using fused multiply add where the inputs are a mix of half-precision
float and single-precision float values. Convert the result to a half-precision float. Store the result into the low
bits of a vector register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0], 2=src[15:0],
3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier for the three

inputs.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
       if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
       elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
       else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
       endif
  endfor;
  D0[15 : 0].f16 = f32_to_f16(fma(in[0], in[1], in[2]))

V_FMA_MIXHI_F16                                                                                                             34

Multiply two inputs and add a third input using fused multiply add where the inputs are a mix of half-precision
float and single-precision float values. Convert the result to a half-precision float. Store the result into the high
bits of a vector register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0], 2=src[15:0],
3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier for the three

inputs.

  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
       if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
       elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
       else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
       endif
  endfor;
  D0[31 : 16].f16 = f32_to_f16(fma(in[0], in[1], in[2]))

V_WMMA_F32_16X16X16_F16                                                                                                     64

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f32(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f32(16x16)";
  EXEC = saved_exec

V_WMMA_F32_16X16X16_BF16                                                                                                    65

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher

performance.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f32(16x16) = S0.bf16(16x16) * S1.bf16(16x16) + S2.f32(16x16)";
  EXEC = saved_exec

V_WMMA_F16_16X16X16_F16                                                                                       66

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are half-precision float format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.f16(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f16(16x16)";
  EXEC = saved_exec

V_WMMA_BF16_16X16X16_BF16                                                                                     67

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are BF16 float format. Matrices C and D are BF16 float format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.bf16(16x16) = S0.bf16(16x16) * S1.bf16(16x16) + S2.bf16(16x16)";

  EXEC = saved_exec

V_WMMA_I32_16X16X16_IU8                                                                                         68

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are unsigned 8-bit integer format. Matrices C and D are signed 32-bit integer format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.i32(16x16) = S0.iu8(16x16) * S1.iu8(16x16) + S2.i32(16x16)";
  EXEC = saved_exec

V_WMMA_I32_16X16X16_IU4                                                                                         69

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

  D = A (16x16) * B (16x16) + C (16x16)

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are unsigned 4-bit integer format. Matrices C and D are signed 32-bit integer format.

  saved_exec = EXEC;
  EXEC = 64'B(-1);
  eval "D0.i32(16x16) = S0.iu4(16x16) * S1.iu4(16x16) + S2.i32(16x16)";
  EXEC = saved_exec
