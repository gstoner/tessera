# 16.7. VOP2 Instructions

> RDNA3.5 ISA — pages 276–289

16.7. VOP2 Instructions

Instructions in this format may use a 32-bit literal constant or DPP that occurs immediately after the
instruction.

V_CNDMASK_B32                                                                                                         1

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

  D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

V_DOT2ACC_F32_F16                                                                                                     2

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and
accumulate the resulting single-precision float value into the destination vector register.

  tmp = D0.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp

V_ADD_F32                                                                                                             3

Add two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 + S1.f32

Notes

0.5ULP precision, denormals are supported.

V_SUB_F32                                                                                                            4

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f32 = S0.f32 - S1.f32

Notes

0.5ULP precision, denormals are supported.

V_SUBREV_F32                                                                                                         5

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f32 = S1.f32 - S0.f32

Notes

0.5ULP precision, denormals are supported.

V_FMAC_DX9_ZERO_F32                                                                                                  6

Multiply two single-precision values and accumulate the result with the destination. Follows DX9 rules where
0.0 times anything produces 0.0.

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = S2.f32
  else
        D0.f32 = fma(S0.f32, S1.f32, D0.f32)
  endif

V_MUL_DX9_ZERO_F32                                                                                                   7

Multiply two floating point inputs and store the result into a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = 0.0F
  else
        D0.f32 = S0.f32 * S1.f32

  endif

V_MUL_F32                                                                                                            8

Multiply two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 * S1.f32

Notes

0.5ULP precision, denormals are supported.

V_MUL_I32_I24                                                                                                        9

Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register.

  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_I32_I24.

V_MUL_HI_I32_I24                                                                                                    10

Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a
vector register.

  D0.i32 = 32'I((64'I(S0.i24) * 64'I(S1.i24)) >> 32U)

Notes

See also V_MUL_I32_I24.

V_MUL_U32_U24                                                                                                       11

Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector
register.

  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_U32_U24.

V_MUL_HI_U32_U24                                                                                                 12

Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer
into a vector register.

  D0.u32 = 32'U((64'U(S0.u24) * 64'U(S1.u24)) >> 32U)

Notes

See also V_MUL_U32_U24.

V_MIN_F32                                                                                                        15

Select the minimum of two single-precision float inputs and store the result into a vector register.

  LT_NEG_ZERO = lambda(a, b) (
        ((a < b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && sign(a) && !sign(b))));
  // Version of comparison where -0.0 < +0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f32)) then
            D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
        elsif isSignalNAN(64'F(S1.f32)) then
            D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
        elsif isQuietNAN(64'F(S1.f32)) then
            D0.f32 = S0.f32
        elsif isQuietNAN(64'F(S0.f32)) then
            D0.f32 = S1.f32
        elsif LT_NEG_ZERO(S0.f32, S1.f32) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f32 = S0.f32
        else
            D0.f32 = S1.f32
        endif
  else
        if isNAN(64'F(S1.f32)) then
            D0.f32 = S0.f32
        elsif isNAN(64'F(S0.f32)) then
            D0.f32 = S1.f32
        elsif LT_NEG_ZERO(S0.f32, S1.f32) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f32 = S0.f32

        else
            D0.f32 = S1.f32
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

V_MAX_F32                                                                                                     16

Select the maximum of two single-precision float inputs and store the result into a vector register.

  GT_NEG_ZERO = lambda(a, b) (
        ((a > b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && !sign(a) && sign(b))));
  // Version of comparison where +0.0 > -0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f32)) then
            D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
        elsif isSignalNAN(64'F(S1.f32)) then
            D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
        elsif isQuietNAN(64'F(S1.f32)) then
            D0.f32 = S0.f32
        elsif isQuietNAN(64'F(S0.f32)) then
            D0.f32 = S1.f32
        elsif GT_NEG_ZERO(S0.f32, S1.f32) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f32 = S0.f32
        else
            D0.f32 = S1.f32
        endif
  else
        if isNAN(64'F(S1.f32)) then
            D0.f32 = S0.f32
        elsif isNAN(64'F(S0.f32)) then
            D0.f32 = S1.f32
        elsif GT_NEG_ZERO(S0.f32, S1.f32) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f32 = S0.f32
        else
            D0.f32 = S1.f32
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

V_MIN_I32                                                                                                            17

Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = S0.i32 < S1.i32 ? S0.i32 : S1.i32

V_MAX_I32                                                                                                            18

Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = S0.i32 >= S1.i32 ? S0.i32 : S1.i32

V_MIN_U32                                                                                                            19

Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = S0.u32 < S1.u32 ? S0.u32 : S1.u32

V_MAX_U32                                                                                                            20

Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = S0.u32 >= S1.u32 ? S0.u32 : S1.u32

V_LSHLREV_B32                                                                                                        24

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u32 = (S1.u32 << S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_LSHRREV_B32                                                                                                       25

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u32 = (S1.u32 >> S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_ASHRREV_I32                                                                                                       26

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i32 = (S1.i32 >> S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_AND_B32                                                                                                           27

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 & S1.u32)

Notes

Input and output modifiers not supported.

V_OR_B32                                                                                                            28

Calculate bitwise OR on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 | S1.u32)

Notes

Input and output modifiers not supported.

V_XOR_B32                                                                                                      29

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 ^ S1.u32)

Notes

Input and output modifiers not supported.

V_XNOR_B32                                                                                                     30

Calculate bitwise XNOR on two vector inputs and store the result into a vector register.

  D0.u32 = ~(S0.u32 ^ S1.u32)

Notes

Input and output modifiers not supported.

V_ADD_CO_CI_U32                                                                                                32

Add two unsigned 32-bit integer inputs and a bit from a carry-in mask, store the result into a vector register and
store the carry-out mask into a scalar register.

  tmp = 64'U(S0.u32) + 64'U(S1.u32) + VCC.u64[laneId].u64;
  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_ADD_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at

S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_SUB_CO_CI_U32                                                                                                  33

Subtract the second unsigned 32-bit integer input from the first input, subtract a bit from the carry-in mask,
store the result into a vector register and store the carry-out mask into a scalar register.

  tmp = S0.u32 - S1.u32 - VCC.u64[laneId].u32;
  VCC.u64[laneId] = 64'U(S1.u32) + VCC.u64[laneId].u64 > 64'U(S0.u32) ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_CO_CI_U32                                                                                               34

Subtract the first unsigned 32-bit integer input from the second input, subtract a bit from the carry-in mask,
store the result into a vector register and store the carry-out mask into a scalar register.

  tmp = S1.u32 - S0.u32 - VCC.u64[laneId].u32;
  VCC.u64[laneId] = 64'U(S0.u32) + VCC.u64[laneId].u64 > 64'U(S1.u32) ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_ADD_NC_U32                                                                                                     37

Add two unsigned 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

  D0.u32 = S0.u32 + S1.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUB_NC_U32                                                                                                      38

Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

  D0.u32 = S0.u32 - S1.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_NC_U32                                                                                                   39

Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

  D0.u32 = S1.u32 - S0.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_FMAC_F32                                                                                                        43

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

  D0.f32 = fma(S0.f32, S1.f32, D0.f32)

V_FMAMK_F32                                                                                                       44

Multiply a single-precision float input with a literal constant and add a second single-precision float input using
fused multiply add, and store the result into a vector register.

  D0.f32 = fma(S0.f32, SIMM32.f32, S1.f32)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_FMAAK_F32                                                                                                     45

Multiply two single-precision float inputs and add a literal constant using fused multiply add, and store the
result into a vector register.

  D0.f32 = fma(S0.f32, S1.f32, SIMM32.f32)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_CVT_PK_RTZ_F16_F32                                                                                            47

Convert two single-precision float inputs to a packed half-precision float value using round toward zero
semantics (ignore the current rounding mode), and store the result into a vector register.

  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_TOWARD_ZERO;
  tmp[15 : 0].f16 = f32_to_f16(S0.f32);
  tmp[31 : 16].f16 = f32_to_f16(S1.f32);
  D0 = tmp.b32;
  ROUND_MODE = prev_mode;
  // Round-toward-zero regardless of current round mode setting in hardware.

Notes

V_ADD_F16                                                                                                       50

Add two floating point inputs and store the result into a vector register.

  D0.f16 = S0.f16 + S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_SUB_F16                                                                                                           51

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f16 = S0.f16 - S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_SUBREV_F16                                                                                                        52

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f16 = S1.f16 - S0.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_MUL_F16                                                                                                           53

Multiply two floating point inputs and store the result into a vector register.

  D0.f16 = S0.f16 * S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_FMAC_F16                                                                                                          54

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

  D0.f16 = fma(S0.f16, S1.f16, D0.f16)

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_FMAMK_F16                                                                                                       55

Multiply a half-precision float input with a literal constant and add a second half-precision float input using
fused multiply add, and store the result into a vector register.

  D0.f16 = fma(S0.f16, SIMM32.f16, S1.f16)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_FMAAK_F16                                                                                                       56

Multiply two half-precision float inputs and add a literal constant using fused multiply add, and store the result
into a vector register.

  D0.f16 = fma(S0.f16, S1.f16, SIMM32.f16)

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

V_MAX_F16                                                                                                         57

Select the maximum of two half-precision float inputs and store the result into a vector register.

  GT_NEG_ZERO = lambda(a, b) (
        ((a > b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && !sign(a) && sign(b))));
  // Version of comparison where +0.0 > -0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f16)) then
            D0.f16 = 16'F(cvtToQuietNAN(64'F(S0.f16)))
        elsif isSignalNAN(64'F(S1.f16)) then
            D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
        elsif isQuietNAN(64'F(S1.f16)) then
            D0.f16 = S0.f16
        elsif isQuietNAN(64'F(S0.f16)) then
            D0.f16 = S1.f16
        elsif GT_NEG_ZERO(S0.f16, S1.f16) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f16 = S0.f16
        else
            D0.f16 = S1.f16
        endif
  else

        if isNAN(64'F(S1.f16)) then
            D0.f16 = S0.f16
        elsif isNAN(64'F(S0.f16)) then
            D0.f16 = S1.f16
        elsif GT_NEG_ZERO(S0.f16, S1.f16) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f16 = S0.f16
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

V_MIN_F16                                                                                                   58

Select the minimum of two half-precision float inputs and store the result into a vector register.

  LT_NEG_ZERO = lambda(a, b) (
        ((a < b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && sign(a) && !sign(b))));
  // Version of comparison where -0.0 < +0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f16)) then
            D0.f16 = 16'F(cvtToQuietNAN(64'F(S0.f16)))
        elsif isSignalNAN(64'F(S1.f16)) then
            D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
        elsif isQuietNAN(64'F(S1.f16)) then
            D0.f16 = S0.f16
        elsif isQuietNAN(64'F(S0.f16)) then
            D0.f16 = S1.f16
        elsif LT_NEG_ZERO(S0.f16, S1.f16) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f16 = S0.f16
        else
            D0.f16 = S1.f16
        endif
  else
        if isNAN(64'F(S1.f16)) then
            D0.f16 = S0.f16
        elsif isNAN(64'F(S0.f16)) then
            D0.f16 = S1.f16
        elsif LT_NEG_ZERO(S0.f16, S1.f16) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f16 = S0.f16
