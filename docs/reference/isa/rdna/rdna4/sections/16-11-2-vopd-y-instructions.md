# 16.11.2. VOPD Y-Instructions

> RDNA4 ISA — pages 425–431

  elsif ((S0.f32 < S1.f32) || ((abs(S0.f32) == 0.0F) && (abs(S1.f32) == 0.0F) && sign(S0.f32) &&
  !sign(S1.f32))) then
        // NOTE: -0<+0 is TRUE in this comparison
        D0.f32 = S0.f32
  else
        D0.f32 = S1.f32
  endif

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input
denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so
output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can
be a denorm and this operation outputs as a denorm regardless of output denorm mode.

V_DUAL_DOT2ACC_F32_F16                                                                                          12

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and
accumulate the resulting single-precision float value into the destination vector register. The initial value in D
is used as S2.

  tmp = D0.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp

V_DUAL_DOT2ACC_F32_BF16                                                                                         13

Dot product of packed brain-float values, accumulate with destination. The initial value in D is used as S2.

  tmp = D0.f32;
  tmp += bf16_to_f32(S0[15 : 0].bf16) * bf16_to_f32(S1[15 : 0].bf16);
  tmp += bf16_to_f32(S0[31 : 16].bf16) * bf16_to_f32(S1[31 : 16].bf16);
  D0.f32 = tmp

16.11.2. VOPD Y-Instructions
V_DUAL_FMAC_F32                                                                                                   0

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

  D0.f32 = fma(S0.f32, S1.f32, D0.f32)

V_DUAL_FMAAK_F32                                                                                                 1

Multiply two single-precision float inputs and add a literal constant using fused multiply add, and store the
result into a vector register.

  D0.f32 = fma(S0.f32, S1.f32, SIMM32.f32)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_DUAL_FMAMK_F32                                                                                                 2

Multiply a single-precision float input with a literal constant and add a second single-precision float input using
fused multiply add, and store the result into a vector register.

  D0.f32 = fma(S0.f32, SIMM32.f32, S1.f32)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_DUAL_MUL_F32                                                                                                   3

Multiply two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 * S1.f32

Notes

0.5ULP precision, denormals are supported.

V_DUAL_ADD_F32                                                                                                   4

Add two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 + S1.f32

Notes

0.5ULP precision, denormals are supported.

V_DUAL_SUB_F32                                                                                                       5

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f32 = S0.f32 - S1.f32

Notes

0.5ULP precision, denormals are supported.

V_DUAL_SUBREV_F32                                                                                                    6

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f32 = S1.f32 - S0.f32

Notes

0.5ULP precision, denormals are supported.

V_DUAL_MUL_DX9_ZERO_F32                                                                                              7

Multiply two floating point inputs and store the result into a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = 0.0F
  else
        D0.f32 = S0.f32 * S1.f32
  endif

V_DUAL_MOV_B32                                                                                                       8

Move 32-bit data from a vector input into a vector register.

  D0.b32 = S0.b32

Notes

Floating-point modifiers are valid for this instruction if S0 is a 32-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

Functional examples:

        v_mov_b32 v0, v1      // Move into v0 from v1
        v_mov_b32 v0, -v1     // Set v0 to the negation of v1
        v_mov_b32 v0, abs(v1)    // Set v0 to the absolute value of v1

V_DUAL_CNDMASK_B32                                                                                                     9

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

  D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

V_DUAL_MAX_NUM_F32                                                                                                    10

Select the IEEE maximumNumber() of two single-precision float inputs and store the result into a vector
register.

A numeric argument is favoured over NaN when determining which argument to return.

  if (isSignalNAN(64'F(S0.f32)) || isSignalNAN(64'F(S1.f32))) then
        TRAPSTS.INVALID = 1
  endif;
  if (isNAN(64'F(S0.f32)) && isNAN(64'F(S1.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
  elsif isNAN(64'F(S0.f32)) then
        D0.f32 = S1.f32
  elsif isNAN(64'F(S1.f32)) then
        D0.f32 = S0.f32

  elsif ((S0.f32 > S1.f32) || ((abs(S0.f32) == 0.0F) && (abs(S1.f32) == 0.0F) && !sign(S0.f32) &&
  sign(S1.f32))) then
        // NOTE: +0>-0 is TRUE in this comparison
        D0.f32 = S0.f32
  else
        D0.f32 = S1.f32
  endif

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input
denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so
output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can
be a denorm and this operation outputs as a denorm regardless of output denorm mode.

V_DUAL_MIN_NUM_F32                                                                                          11

Select the IEEE minimumNumber() of two single-precision float inputs and store the result into a vector
register.

A numeric argument is favoured over NaN when determining which argument to return.

  if (isSignalNAN(64'F(S0.f32)) || isSignalNAN(64'F(S1.f32))) then
        TRAPSTS.INVALID = 1
  endif;
  if (isNAN(64'F(S0.f32)) && isNAN(64'F(S1.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
  elsif isNAN(64'F(S0.f32)) then
        D0.f32 = S1.f32
  elsif isNAN(64'F(S1.f32)) then
        D0.f32 = S0.f32
  elsif ((S0.f32 < S1.f32) || ((abs(S0.f32) == 0.0F) && (abs(S1.f32) == 0.0F) && sign(S0.f32) &&
  !sign(S1.f32))) then
        // NOTE: -0<+0 is TRUE in this comparison
        D0.f32 = S0.f32
  else
        D0.f32 = S1.f32
  endif

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input
denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so
output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can
be a denorm and this operation outputs as a denorm regardless of output denorm mode.

V_DUAL_DOT2ACC_F32_F16                                                                                               12

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and
accumulate the resulting single-precision float value into the destination vector register. The initial value in D
is used as S2.

  tmp = D0.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp

V_DUAL_DOT2ACC_F32_BF16                                                                                              13

Dot product of packed brain-float values, accumulate with destination. The initial value in D is used as S2.

  tmp = D0.f32;
  tmp += bf16_to_f32(S0[15 : 0].bf16) * bf16_to_f32(S1[15 : 0].bf16);
  tmp += bf16_to_f32(S0[31 : 16].bf16) * bf16_to_f32(S1[31 : 16].bf16);
  D0.f32 = tmp

V_DUAL_ADD_NC_U32                                                                                                    16

Add two unsigned 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

  D0.u32 = S0.u32 + S1.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_DUAL_LSHLREV_B32                                                                                                   17

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u32 = (S1.u32 << S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_DUAL_AND_B32                                                                                   18

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 & S1.u32)

Notes

Input and output modifiers not supported.
