# 16.4. SOPC Instructions

> RDNA4 ISA — pages 267–275

16.4. SOPC Instructions

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

S_CMP_EQ_I32                                                                                                     0

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.i32 == S1.i32

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_LG_I32                                                                                                     1

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.i32 <> S1.i32

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_GT_I32                                                                                                     2

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.i32 > S1.i32

S_CMP_GE_I32                                                                                                     3

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.i32 >= S1.i32

S_CMP_LT_I32                                                                                            4

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.i32 < S1.i32

S_CMP_LE_I32                                                                                            5

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.i32 <= S1.i32

S_CMP_EQ_U32                                                                                            6

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.u32 == S1.u32

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_LG_U32                                                                                            7

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.u32 <> S1.u32

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

S_CMP_GT_U32                                                                                            8

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.u32 > S1.u32

S_CMP_GE_U32                                                                                                        9

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.u32 >= S1.u32

S_CMP_LT_U32                                                                                                       10

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.u32 < S1.u32

S_CMP_LE_U32                                                                                                       11

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.u32 <= S1.u32

S_BITCMP0_B32                                                                                                      12

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

  SCC = S0.u32[S1.u32[4 : 0]] == 1'0U

S_BITCMP1_B32                                                                                                      13

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

  SCC = S0.u32[S1.u32[4 : 0]] == 1'1U

S_BITCMP0_B64                                                                                                      14

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

  SCC = S0.u64[S1.u32[5 : 0]] == 1'0U

S_BITCMP1_B64                                                                                                      15

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

  SCC = S0.u64[S1.u32[5 : 0]] == 1'1U

S_CMP_EQ_U64                                                                                                       16

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.u64 == S1.u64

S_CMP_LG_U64                                                                                                       17

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.u64 <> S1.u64

S_CMP_LT_F32                                                                                                       65

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.f32 < S1.f32

S_CMP_LT_F16                                                                                                       81

Set SCC to 1 iff the first scalar input is less than the second scalar input.

  SCC = S0.f16 < S1.f16

S_CMP_EQ_F32                                                                                       66

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.f32 == S1.f32

S_CMP_EQ_F16                                                                                       82

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

  SCC = S0.f16 == S1.f16

S_CMP_LE_F32                                                                                       67

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.f32 <= S1.f32

S_CMP_LE_F16                                                                                       83

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

  SCC = S0.f16 <= S1.f16

S_CMP_GT_F32                                                                                       68

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.f32 > S1.f32

S_CMP_GT_F16                                                                                           84

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

  SCC = S0.f16 > S1.f16

S_CMP_LG_F32                                                                                           69

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.f32 <> S1.f32

S_CMP_LG_F16                                                                                           85

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

  SCC = S0.f16 <> S1.f16

S_CMP_GE_F32                                                                                           70

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.f32 >= S1.f32

S_CMP_GE_F16                                                                                           86

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

  SCC = S0.f16 >= S1.f16

S_CMP_O_F32                                                                                            71

Set SCC to 1 iff the first scalar input is orderable to the second scalar input.

  SCC = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)))

S_CMP_O_F16                                                                                               87

Set SCC to 1 iff the first scalar input is orderable to the second scalar input.

  SCC = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)))

S_CMP_U_F32                                                                                               72

Set SCC to 1 iff the first scalar input is not orderable to the second scalar input.

  SCC = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)))

S_CMP_U_F16                                                                                               88

Set SCC to 1 iff the first scalar input is not orderable to the second scalar input.

  SCC = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)))

S_CMP_NGE_F32                                                                                             73

Set SCC to 1 iff the first scalar input is not greater than or equal to the second scalar input.

  SCC = !(S0.f32 >= S1.f32);
  // With NAN inputs this is not the same operation as <

S_CMP_NGE_F16                                                                                             89

Set SCC to 1 iff the first scalar input is not greater than or equal to the second scalar input.

  SCC = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <

S_CMP_NLG_F32                                                                                              74

Set SCC to 1 iff the first scalar input is not less than or greater than the second scalar input.

  SCC = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==

S_CMP_NLG_F16                                                                                              90

Set SCC to 1 iff the first scalar input is not less than or greater than the second scalar input.

  SCC = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==

S_CMP_NGT_F32                                                                                              75

Set SCC to 1 iff the first scalar input is not greater than the second scalar input.

  SCC = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=

S_CMP_NGT_F16                                                                                              91

Set SCC to 1 iff the first scalar input is not greater than the second scalar input.

  SCC = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=

S_CMP_NLE_F32                                                                                              76

Set SCC to 1 iff the first scalar input is not less than or equal to the second scalar input.

  SCC = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >

S_CMP_NLE_F16                                                                                          92

Set SCC to 1 iff the first scalar input is not less than or equal to the second scalar input.

  SCC = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >

S_CMP_NEQ_F32                                                                                          77

Set SCC to 1 iff the first scalar input is not equal to the second scalar input.

  SCC = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=

S_CMP_NEQ_F16                                                                                          93

Set SCC to 1 iff the first scalar input is not equal to the second scalar input.

  SCC = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=

S_CMP_NLT_F32                                                                                          78

Set SCC to 1 iff the first scalar input is not less than the second scalar input.

  SCC = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=

S_CMP_NLT_F16                                                                                          94

Set SCC to 1 iff the first scalar input is not less than the second scalar input.

  SCC = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=
