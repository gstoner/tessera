# 16.1. SOP2 Instructions

> RDNA3.5 ISA — pages 199–215

16.1. SOP2 Instructions

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

S_ADD_U32                                                                                                             0

Add two unsigned 32-bit integer inputs, store the result into a scalar register and store the carry-out bit into
SCC.

  tmp = 64'U(S0.u32) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_ADDC_U32.
  D0.u32 = tmp.u32

S_SUB_U32                                                                                                             1

Subtract the second unsigned 32-bit integer input from the first input, store the result into a scalar register and
store the carry-out bit into SCC.

  tmp = S0.u32 - S1.u32;
  SCC = S1.u32 > S0.u32 ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_SUBB_U32.
  D0.u32 = tmp.u32

S_ADD_I32                                                                                                             2

Add two signed 32-bit integer inputs, store the result into a scalar register and store the carry-out bit into SCC.

  tmp = S0.i32 + S1.i32;
  SCC = ((S0.u32[31] == S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));
  // signed overflow.
  D0.i32 = tmp.i32

Notes

This opcode is not suitable for use with S_ADDC_U32 for implementing 64-bit operations.

S_SUB_I32                                                                                                             3

Subtract the second signed 32-bit integer input from the first input, store the result into a scalar register and
store the carry-out bit into SCC.

  tmp = S0.i32 - S1.i32;
  SCC = ((S0.u32[31] != S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));
  // signed overflow.
  D0.i32 = tmp.i32

Notes

This opcode is not suitable for use with S_SUBB_U32 for implementing 64-bit operations.

S_ADDC_U32                                                                                                          4

Add two unsigned 32-bit integer inputs and a carry-in bit from SCC, store the result into a scalar register and
store the carry-out bit into SCC.

  tmp = 64'U(S0.u32) + 64'U(S1.u32) + SCC.u64;
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_ADDC_U32.
  D0.u32 = tmp.u32

S_SUBB_U32                                                                                                          5

Subtract the second unsigned 32-bit integer input from the first input, subtract the carry-in bit, store the result
into a scalar register and store the carry-out bit into SCC.

  tmp = S0.u32 - S1.u32 - SCC.u32;
  SCC = 64'U(S1.u32) + SCC.u64 > 64'U(S0.u32) ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_SUBB_U32.
  D0.u32 = tmp.u32

S_ABSDIFF_I32                                                                                                       6

Calculate the absolute value of difference between two scalar inputs, store the result into a scalar register and
set SCC iff the result is nonzero.

  D0.i32 = S0.i32 - S1.i32;
  if D0.i32 < 0 then
        D0.i32 = -D0.i32
  endif;
  SCC = D0.i32 != 0

Notes

Functional examples:

  S_ABSDIFF_I32(0x00000002, 0x00000005) => 0x00000003
  S_ABSDIFF_I32(0xffffffff, 0x00000000) => 0x00000001
  S_ABSDIFF_I32(0x80000000, 0x00000000) => 0x80000000          // Note: result is negative!
  S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff
  S_ABSDIFF_I32(0x80000000, 0xffffffff) => 0x7fffffff
  S_ABSDIFF_I32(0x80000000, 0xfffffffe) => 0x7ffffffe

S_LSHL_B32                                                                                                               8

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u32 = (S0.u32 << S1[4 : 0].u32);
  SCC = D0.u32 != 0U

S_LSHL_B64                                                                                                               9

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = (S0.u64 << S1[5 : 0].u32);
  SCC = D0.u64 != 0ULL

S_LSHR_B32                                                                                                              10

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u32 = (S0.u32 >> S1[4 : 0].u32);
  SCC = D0.u32 != 0U

S_LSHR_B64                                                                                                              11

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = (S0.u64 >> S1[5 : 0].u32);
  SCC = D0.u64 != 0ULL

S_ASHR_I32                                                                                                            12

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

  D0.i32 = 32'I(signext(S0.i32) >> S1[4 : 0].u32);
  SCC = D0.i32 != 0

S_ASHR_I64                                                                                                            13

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

  D0.i64 = (signext(S0.i64) >> S1[5 : 0].u32);
  SCC = D0.i64 != 0LL

S_LSHL1_ADD_U32                                                                                                       14

Calculate the logical shift left of the first input by 1, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u32) << 1U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32

S_LSHL2_ADD_U32                                                                                                       15

Calculate the logical shift left of the first input by 2, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u32) << 2U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32

S_LSHL3_ADD_U32                                                                                                       16

Calculate the logical shift left of the first input by 3, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u32) << 3U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32

S_LSHL4_ADD_U32                                                                                                       17

Calculate the logical shift left of the first input by 4, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u32) << 4U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32

S_MIN_I32                                                                                                             18

Select the minimum of two signed 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

  SCC = S0.i32 < S1.i32;
  D0.i32 = SCC ? S0.i32 : S1.i32

S_MIN_U32                                                                                                             19

Select the minimum of two unsigned 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

  SCC = S0.u32 < S1.u32;
  D0.u32 = SCC ? S0.u32 : S1.u32

S_MAX_I32                                                                                                             20

Select the maximum of two signed 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

  SCC = S0.i32 >= S1.i32;
  D0.i32 = SCC ? S0.i32 : S1.i32

S_MAX_U32                                                                                                           21

Select the maximum of two unsigned 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

  SCC = S0.u32 >= S1.u32;
  D0.u32 = SCC ? S0.u32 : S1.u32

S_AND_B32                                                                                                           22

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u32 = (S0.u32 & S1.u32);
  SCC = D0.u32 != 0U

S_AND_B64                                                                                                           23

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL

S_OR_B32                                                                                                            24

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u32 = (S0.u32 | S1.u32);
  SCC = D0.u32 != 0U

S_OR_B64                                                                                                            25

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL

S_XOR_B32                                                                                                           26

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u32 = (S0.u32 ^ S1.u32);
  SCC = D0.u32 != 0U

S_XOR_B64                                                                                                           27

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 ^ S1.u64);
  SCC = D0.u64 != 0ULL

S_NAND_B32                                                                                                          28

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u32 = ~(S0.u32 & S1.u32);
  SCC = D0.u32 != 0U

S_NAND_B64                                                                                                          29

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL

S_NOR_B32                                                                                                           30

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u32 = ~(S0.u32 | S1.u32);
  SCC = D0.u32 != 0U

S_NOR_B64                                                                                                           31

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL

S_XNOR_B32                                                                                                          32

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u32 = ~(S0.u32 ^ S1.u32);
  SCC = D0.u32 != 0U

S_XNOR_B64                                                                                                          33

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 ^ S1.u64);
  SCC = D0.u64 != 0ULL

S_AND_NOT1_B32                                                                                                      34

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u32 = (S0.u32 & ~S1.u32);
  SCC = D0.u32 != 0U

S_AND_NOT1_B64                                                                                                     35

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u64 = (S0.u64 & ~S1.u64);
  SCC = D0.u64 != 0ULL

S_OR_NOT1_B32                                                                                                      36

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u32 = (S0.u32 | ~S1.u32);
  SCC = D0.u32 != 0U

S_OR_NOT1_B64                                                                                                      37

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u64 = (S0.u64 | ~S1.u64);
  SCC = D0.u64 != 0ULL

S_BFE_U32                                                                                                          38

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

  D0.u32 = ((S0.u32 >> S1[4 : 0].u32) & ((1U << S1[22 : 16].u32) - 1U));
  SCC = D0.u32 != 0U

S_BFE_I32                                                                                                           39

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

  tmp.i32 = ((S0.i32 >> S1[4 : 0].u32) & ((1 << S1[22 : 16].u32) - 1));
  D0.i32 = signext_from_bit(tmp.i32, S1[22 : 16].u32);
  SCC = D0.i32 != 0

S_BFE_U64                                                                                                           40

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = ((S0.u64 >> S1[5 : 0].u32) & ((1ULL << S1[22 : 16].u32) - 1ULL));
  SCC = D0.u64 != 0ULL

S_BFE_I64                                                                                                           41

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

  tmp.i64 = ((S0.i64 >> S1[5 : 0].u32) & ((1LL << S1[22 : 16].u32) - 1LL));
  D0.i64 = signext_from_bit(tmp.i64, S1[22 : 16].u32);
  SCC = D0.i64 != 0LL

S_BFM_B32                                                                                                           42

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

  D0.u32 = (((1U << S0[4 : 0].u32) - 1U) << S1[4 : 0].u32)

S_BFM_B64                                                                                                           43

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

  D0.u64 = (((1ULL << S0[5 : 0].u32) - 1ULL) << S1[5 : 0].u32)

S_MUL_I32                                                                                                          44

Multiply two signed 32-bit integer inputs and store the result into a scalar register.

  D0.i32 = S0.i32 * S1.i32

S_MUL_HI_U32                                                                                                       45

Multiply two unsigned integers and store the high 32 bits of the result into a scalar register.

  D0.u32 = 32'U((64'U(S0.u32) * 64'U(S1.u32)) >> 32U)

S_MUL_HI_I32                                                                                                       46

Multiply two signed integers and store the high 32 bits of the result into a scalar register.

  D0.i32 = 32'I((64'I(S0.i32) * 64'I(S1.i32)) >> 32U)

S_CSELECT_B32                                                                                                      48

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

  D0.u32 = SCC ? S0.u32 : S1.u32

S_CSELECT_B64                                                                                                      49

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

  D0.u64 = SCC ? S0.u64 : S1.u64

S_PACK_LL_B32_B16                                                                                                  50

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[15 : 0].u16, S0[15 : 0].u16 }

S_PACK_LH_B32_B16                                                                                                 51

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[31 : 16].u16, S0[15 : 0].u16 }

S_PACK_HH_B32_B16                                                                                                 52

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[31 : 16].u16, S0[31 : 16].u16 }

S_PACK_HL_B32_B16                                                                                                 53

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[15 : 0].u16, S0[31 : 16].u16 }

S_ADD_F32                                                                                                         64

Add two floating point inputs and store the result into a scalar register.

  D0.f32 = S0.f32 + S1.f32

S_SUB_F32                                                                                                         65

Subtract the second floating point input from the first input and store the result in a scalar register.

  D0.f32 = S0.f32 - S1.f32

S_MIN_F32                                                                                                     66

Select the minimum of two single-precision float inputs and store the result into a scalar register.

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

S_MAX_F32                                                                                                     67

Select the maximum of two single-precision float inputs and store the result into a scalar register.

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

S_MUL_F32                                                                                                       68

Multiply two floating point inputs and store the result into a scalar register.

  D0.f32 = S0.f32 * S1.f32

S_FMAAK_F32                                                                                                     69

Multiply two floating point inputs and add a literal constant using fused multiply add, and store the result into
a scalar register.

  D0.f32 = fma(S0.f32, S1.f32, SIMM32.f32)

S_FMAMK_F32                                                                                                     70

Multiply a floating point input with a literal constant and add a second floating point input using fused multiply
add, and store the result into a scalar register.

  D0.f32 = fma(S0.f32, SIMM32.f32, S1.f32)

S_FMAC_F32                                                                                                        71

Compute the fused multiply add of floating point inputs and accumulate with the destination operand, and
store the result into the destination.

  D0.f32 = fma(S0.f32, S1.f32, D0.f32)

S_CVT_PK_RTZ_F16_F32                                                                                              72

Convert two single-precision float inputs into a packed half-precision float result using round toward zero
semantics (ignore the current rounding mode), and store the result into a scalar register.

  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_TOWARD_ZERO;
  tmp[15 : 0].f16 = f32_to_f16(S0.f32);
  tmp[31 : 16].f16 = f32_to_f16(S1.f32);
  D0 = tmp.b32;
  ROUND_MODE = prev_mode;
  // Round-toward-zero regardless of current round mode setting in hardware.

S_ADD_F16                                                                                                         73

Add two floating point inputs and store the result into a scalar register.

  D0.f16 = S0.f16 + S1.f16

S_SUB_F16                                                                                                         74

Subtract the second floating point input from the first input and store the result in a scalar register.

  D0.f16 = S0.f16 - S1.f16

S_MIN_F16                                                                                                         75

Select the minimum of two half-precision float inputs and store the result into a scalar register.

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
      else
           D0.f16 = S1.f16
      endif
  endif;
  // Inequalities in the above pseudocode behave differently from IEEE
  // when both inputs are +-0.

S_MAX_F16                                                                                                   76

Select the maximum of two half-precision float inputs and store the result into a scalar register.

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

S_MUL_F16                                                                                                  77

Multiply two floating point inputs and store the result into a scalar register.

  D0.f16 = S0.f16 * S1.f16

S_FMAC_F16                                                                                                 78

Compute the fused multiply add of floating point inputs and accumulate with the destination operand, and
store the result into the destination.

  D0.f16 = fma(S0.f16, S1.f16, D0.f16)
