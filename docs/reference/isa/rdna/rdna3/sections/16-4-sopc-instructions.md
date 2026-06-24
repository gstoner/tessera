# 16.4. SOPC Instructions

> RDNA3 ISA — pages 238–241

16.4. SOPC Instructions

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

S_CMP_EQ_I32                                                                                                     0

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.i == S1.i

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_LG_I32                                                                                                     1

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.i <> S1.i

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_GT_I32                                                                                                     2

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.i > S1.i

S_CMP_GE_I32                                                                                                     3

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.i >= S1.i

S_CMP_LT_I32                                                                                            4

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.i < S1.i

S_CMP_LE_I32                                                                                            5

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.i <= S1.i

S_CMP_EQ_U32                                                                                            6

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.u == S1.u

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_LG_U32                                                                                            7

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.u <> S1.u

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_GT_U32                                                                                            8

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.u > S1.u

S_CMP_GE_U32                                                                                                        9

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.u >= S1.u

S_CMP_LT_U32                                                                                                       10

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.u < S1.u

S_CMP_LE_U32                                                                                                       11

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.u <= S1.u

S_BITCMP0_B32                                                                                                      12

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

  SCC = S0.u[S1.u[4 : 0]] == 1'0U

S_BITCMP1_B32                                                                                                      13

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

  SCC = S0.u[S1.u[4 : 0]] == 1'1U

S_BITCMP0_B64                                                                                                      14

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

  SCC = S0.u64[S1.u[5 : 0]] == 1'0U

S_BITCMP1_B64                                                                                                      15

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

  SCC = S0.u64[S1.u[5 : 0]] == 1'1U

S_CMP_EQ_U64                                                                                                       16

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.u64 == S1.u64

S_CMP_LG_U64                                                                                                       17

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.u64 <> S1.u64
