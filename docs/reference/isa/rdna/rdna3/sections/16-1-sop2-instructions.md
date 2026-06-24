# 16.1. SOP2 Instructions

> RDNA3 ISA — pages 197–208

16.1. SOP2 Instructions

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

S_ADD_U32                                                                                                         0

Add two unsigned inputs, store the result into a scalar register and store the carry-out bit into SCC.

  tmp = 64'U(S0.u) + 64'U(S1.u);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_ADDC_U32.
  D0.u = tmp.u

S_SUB_U32                                                                                                         1

Subtract the second unsigned input from the first input, store the result into a scalar register and store the
carry-out bit into SCC.

  tmp = S0.u - S1.u;
  SCC = S1.u > S0.u ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_SUBB_U32.
  D0.u = tmp.u

S_ADD_I32                                                                                                         2

Add two signed inputs, store the result into a scalar register and store the carry-out bit into SCC.

  tmp = S0.i + S1.i;
  SCC = ((S0.u[31] == S1.u[31]) && (S0.u[31] != tmp.u[31]));
  // signed overflow.
  D0.i = tmp.i

Notes

This opcode is not suitable for use with S_ADDC_U32 for implementing 64-bit operations.

S_SUB_I32                                                                                                         3

Subtract the second signed input from the first input, store the result into a scalar register and store the carry-
out bit into SCC.

  tmp = S0.i - S1.i;
  SCC = ((S0.u[31] != S1.u[31]) && (S0.u[31] != tmp.u[31]));
  // signed overflow.
  D0.i = tmp.i

Notes

This opcode is not suitable for use with S_SUBB_U32 for implementing 64-bit operations.

S_ADDC_U32                                                                                                            4

Add two unsigned inputs and a carry-in bit, store the result into a scalar register and store the carry-out bit into
SCC.

  tmp = 64'U(S0.u) + 64'U(S1.u) + SCC.u64;
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_SUBB_U32                                                                                                            5

Subtract the second unsigned input from the first input, subtract the carry-in bit, store the result into a scalar
register and store the carry-out bit into SCC.

  tmp = S0.u - S1.u - SCC.u;
  SCC = 64'U(S1.u) + SCC.u64 > 64'U(S0.u) ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_ABSDIFF_I32                                                                                                         6

Calculate the absolute value of difference between two scalar inputs, store the result into a scalar register and
set SCC iff the result is nonzero.

  D0.i = S0.i - S1.i;
  if D0.i < 0 then
        D0.i = -D0.i
  endif;
  SCC = D0.i != 0

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

  D0.u = (S0.u << S1.u[4 : 0].u);
  SCC = D0.u != 0U

S_LSHL_B64                                                                                                               9

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = (S0.u64 << S1.u[5 : 0].u);
  SCC = D0.u64 != 0ULL

S_LSHR_B32                                                                                                              10

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u = (S0.u >> S1.u[4 : 0].u);
  SCC = D0.u != 0U

S_LSHR_B64                                                                                                              11

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = (S0.u64 >> S1.u[5 : 0].u);
  SCC = D0.u64 != 0ULL

S_ASHR_I32                                                                                                            12

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

  D0.i = 32'I(signext(S0.i) >> S1.u[4 : 0].u);
  SCC = D0.i != 0

S_ASHR_I64                                                                                                            13

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

  D0.i64 = (signext(S0.i64) >> S1.u[5 : 0].u);
  SCC = D0.i64 != 0LL

S_LSHL1_ADD_U32                                                                                                       14

Calculate the logical shift left of the first input by 1, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u) << 1U) + 64'U(S1.u);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_LSHL2_ADD_U32                                                                                                       15

Calculate the logical shift left of the first input by 2, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u) << 2U) + 64'U(S1.u);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_LSHL3_ADD_U32                                                                                                       16

Calculate the logical shift left of the first input by 3, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u) << 3U) + 64'U(S1.u);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_LSHL4_ADD_U32                                                                                                       17

Calculate the logical shift left of the first input by 4, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

  tmp = (64'U(S0.u) << 4U) + 64'U(S1.u);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u = tmp.u

S_MIN_I32                                                                                                             18

Select the minimum of two signed integers, store the selected value into a scalar register and set SCC iff the
first value is selected.

  SCC = S0.i < S1.i;
  D0.i = SCC ? S0.i : S1.i

S_MIN_U32                                                                                                             19

Select the minimum of two unsigned integers, store the selected value into a scalar register and set SCC iff the
first value is selected.

  SCC = S0.u < S1.u;
  D0.u = SCC ? S0.u : S1.u

S_MAX_I32                                                                                                             20

Select the maximum of two signed integers, store the selected value into a scalar register and set SCC iff the
first value is selected.

  SCC = S0.i > S1.i;
  D0.i = SCC ? S0.i : S1.i

S_MAX_U32                                                                                                           21

Select the maximum of two unsigned integers, store the selected value into a scalar register and set SCC iff the
first value is selected.

  SCC = S0.u > S1.u;
  D0.u = SCC ? S0.u : S1.u

S_AND_B32                                                                                                           22

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u = (S0.u & S1.u);
  SCC = D0.u != 0U

S_AND_B64                                                                                                           23

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL

S_OR_B32                                                                                                            24

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u = (S0.u | S1.u);
  SCC = D0.u != 0U

S_OR_B64                                                                                                            25

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL

S_XOR_B32                                                                                                           26

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u = (S0.u ^ S1.u);
  SCC = D0.u != 0U

S_XOR_B64                                                                                                           27

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

  D0.u64 = (S0.u64 ^ S1.u64);
  SCC = D0.u64 != 0ULL

S_NAND_B32                                                                                                          28

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u = ~(S0.u & S1.u);
  SCC = D0.u != 0U

S_NAND_B64                                                                                                          29

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL

S_NOR_B32                                                                                                           30

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u = ~(S0.u | S1.u);
  SCC = D0.u != 0U

S_NOR_B64                                                                                                           31

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL

S_XNOR_B32                                                                                                          32

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u = ~(S0.u ^ S1.u);
  SCC = D0.u != 0U

S_XNOR_B64                                                                                                          33

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

  D0.u64 = ~(S0.u64 ^ S1.u64);
  SCC = D0.u64 != 0ULL

S_AND_NOT1_B32                                                                                                      34

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u = (S0.u & ~S1.u);
  SCC = D0.u != 0U

S_AND_NOT1_B64                                                                                                     35

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u64 = (S0.u64 & ~S1.u64);
  SCC = D0.u64 != 0ULL

S_OR_NOT1_B32                                                                                                      36

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u = (S0.u | ~S1.u);
  SCC = D0.u != 0U

S_OR_NOT1_B64                                                                                                      37

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

  D0.u64 = (S0.u64 | ~S1.u64);
  SCC = D0.u64 != 0ULL

S_BFE_U32                                                                                                          38

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

  D0.u = ((S0.u >> S1.u[4 : 0].u) & ((1U << S1.u[22 : 16].u) - 1U));
  SCC = D0.u != 0U

S_BFE_I32                                                                                                           39

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

  tmp = ((S0.i >> S1.u[4 : 0].u) & ((1 << S1.u[22 : 16].u) - 1));
  D0.i = 32'I(signextFromBit(tmp, S1.i[22 : 16].i));
  SCC = D0.i != 0

S_BFE_U64                                                                                                           40

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

  D0.u64 = ((S0.u64 >> S1.u[5 : 0].u) & ((1ULL << S1.u[22 : 16].u) - 1ULL));
  SCC = D0.u64 != 0ULL

S_BFE_I64                                                                                                           41

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

  tmp = ((S0.i64 >> S1.u[5 : 0].u) & ((1LL << S1.u[22 : 16].u) - 1LL));
  D0.i64 = signextFromBit(tmp, S1.i[22 : 16].i64);
  SCC = D0.i64 != 0LL

S_BFM_B32                                                                                                           42

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

  D0.u = (((1U << S0.u[4 : 0].u) - 1U) << S1.u[4 : 0].u)

S_BFM_B64                                                                                                           43

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

  D0.u64 = (((1ULL << S0.u[5 : 0].u) - 1ULL) << S1.u[5 : 0].u)

S_MUL_I32                                                                                                          44

Multiply two signed integers and store the result into a scalar register.

  D0.i = S0.i * S1.i

S_MUL_HI_U32                                                                                                       45

Multiply two unsigned integers and store the high 32 bits of the result into a scalar register.

  D0.u = 32'U((64'U(S0.u) * 64'U(S1.u)) >> 32U)

S_MUL_HI_I32                                                                                                       46

Multiply two signed integers and store the high 32 bits of the result into a scalar register.

  D0.i = 32'I((64'I(S0.i) * 64'I(S1.i)) >> 32U)

S_CSELECT_B32                                                                                                      48

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

  D0.u = SCC ? S0.u : S1.u

S_CSELECT_B64                                                                                                      49

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

  D0.u64 = SCC ? S0.u64 : S1.u64

S_PACK_LL_B32_B16                                                                                                  50

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[15 : 0].u16, S0[15 : 0].u16 }

S_PACK_LH_B32_B16                                              51

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[31 : 16].u16, S0[15 : 0].u16 }

S_PACK_HH_B32_B16                                              52

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[31 : 16].u16, S0[31 : 16].u16 }

S_PACK_HL_B32_B16                                              53

Pack two 16-bit scalar values into a scalar register.

  D0 = { S1[15 : 0].u16, S0[31 : 16].u16 }
