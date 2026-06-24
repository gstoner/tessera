# 16.9. VOPC Instructions

> RDNA3.5 ISA — pages 322–379

16.9. VOPC Instructions
The bitfield map for VOPC is:

        SRC0   = First operand for instruction.
        VSRC1 = Second operand for instruction.
        OP     = Instruction opcode.
        All VOPC instructions can alternatively be encoded in the VOP3 format.

Compare instructions perform the same compare operation on each lane (work-Item or thread) using that
lane’s private data, and producing a 1 bit result per lane into VCC or EXEC.

Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction.

Most compare instructions fall into one of two categories:

    • Those which can use one of 16 compare operations (floating point types). "{COMPF}"
    • Those which can use one of 8 compare operations (integer types). "{COMPI}"

The opcode number is such that for these the opcode number can be calculated from a base opcode number
for the data type, plus an offset for the specific compare operation.

                    Table 112. Float Compare Operations
Compare Operation         Opcode Offset      Description
F                         0                  D.u = 0
LT                        1                  D.u = (S0 < S1)
EQ                        2                  D.u = (S0 == S1)
LE                        3                  D.u = (S0 <= S1)
GT                        4                  D.u = (S0 > S1)
LG                        5                  D.u = (S0 <> S1)
GE                        6                  D.u = (S0 >= S1)
O                         7                  D.u = (!isNaN(S0) && !isNaN(S1))
U                         8                  D.u = (!isNaN(S0) || !isNaN(S1))
NGE                       9                  D.u = !(S0 >= S1)
NLG                       10                 D.u = !(S0 <> S1)
NGT                       11                 D.u = !(S0 > S1)
NLE                       12                 D.u = !(S0 <= S1)
NEQ                       13                 D.u = !(S0 == S1)
NLT                       14                 D.u = !(S0 < S1)
TRU                       15                 D.u = 1

               Table 113. Instructions with Sixteen Compare Operations
Instruction                    Description                                      Hex Range
V_CMP_{COMPF}_F16              16-bit float compare. Writes VCC/SGPR.           0x20 to 0x2F
V_CMPX_{COMPF}_F16             16-bit float compare. Writes EXEC.               0x30 to 0x3F
V_CMP_{COMPF}_F32              32-bit float compare. Writes VCC/SGPR.           0x40 to 0x4F

Instruction                   Description                                Hex Range
V_CMPX_{COMPF}_F32            32-bit float compare. Writes EXEC.         0x50 to 0x5F
V_CMP_{COMPF}_F64             64-bit float compare. Writes VCC/SGPR.     0x60 to 0x6F
V_CMPX_{COMPF}_F64            64-bit float compare. Writes EXEC.         0x70 to 0x7F

         Table 114. Integer Compare Operations
Compare Operation         Opcode Offset     Description
F                         0                 D.u = 0
LT                        1                 D.u = (S0 < S1)
EQ                        2                 D.u = (S0 == S1)
LE                        3                 D.u = (S0 <= S1)
GT                        4                 D.u = (S0 > S1)
LG                        5                 D.u = (S0 <> S1)
GE                        6                 D.u = (S0 >= S1)
TRU                       7                 D.u = 1

                   Table 115. Instructions with Eight Compare Operations
Instruction                   Description                                          Hex Range
V_CMP_{COMPI}_I16             16-bit signed integer compare. Writes VCC/SGPR.      0xA0 - 0xA7
V_CMP_{COMPI}_U16             16-bit signed integer compare. Writes VCC/SGPR.      0xA8 - 0xAF
V_CMPX_{COMPI}_I16            16-bit unsigned integer compare. Writes EXEC.        0xB0 - 0xB7
V_CMPX_{COMPI}_U16            16-bit unsigned integer compare. Writes EXEC.        0xB8 - 0xBF
V_CMP_{COMPI}_I32             32-bit signed integer compare. Writes VCC/SGPR.      0xC0 - 0xC7
V_CMP_{COMPI}_U32             32-bit signed integer compare. Writes VCC/SGPR.      0xC8 - 0xCF
V_CMPX_{COMPI}_I32            32-bit unsigned integer compare. Writes EXEC.        0xD0 - 0xD7
V_CMPX_{COMPI}_U32            32-bit unsigned integer compare. Writes EXEC.        0xD8 - 0xDF
V_CMP_{COMPI}_I64             64-bit signed integer compare. Writes VCC/SGPR.      0xE0 - 0xE7
V_CMP_{COMPI}_U64             64-bit signed integer compare. Writes VCC/SGPR.      0xE8 - 0xEF
V_CMPX_{COMPI}_I64            64-bit unsigned integer compare. Writes EXEC.        0xF0 - 0xF7
V_CMPX_{COMPI}_U64            64-bit unsigned integer compare. Writes EXEC.        0xF8 - 0xFF

V_CMP_F_F16                                                                                                         0

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

    D0.u64[laneId] = 1'0U;
    // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F16                                                                                                        1

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a

scalar register.

  D0.u64[laneId] = S0.f16 < S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F16                                                                                                           2

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.f16 == S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F16                                                                                                           3

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.f16 <= S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F16                                                                                                           4

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.f16 > S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F16                                                                                                        5

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f16 <> S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F16                                                                                                        6

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f16 >= S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F16                                                                                                         7

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F16                                                                                                         8

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F16                                                                                                          9

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F16                                                                                                      10

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F16                                                                                                      11

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = !(S0.f16 > S1.f16);

  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F16                                                                                                       12

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F16                                                                                                       13

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F16                                                                                                       14

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F16                                                                                                        15

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_F32                                                                                                        16

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F32                                                                                                       17

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.f32 < S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F32                                                                                                       18

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a

scalar register.

  D0.u64[laneId] = S0.f32 == S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F32                                                                                                           19

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.f32 <= S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F32                                                                                                           20

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.f32 > S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F32                                                                                                           21

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f32 <> S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F32                                                                                                       22

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f32 >= S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F32                                                                                                        23

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F32                                                                                                        24

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F32                                                                                                      25

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f32 >= S1.f32);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F32                                                                                                       26

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F32                                                                                                       27

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F32                                                                                                       28

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F32                                                                                                      29

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F32                                                                                                      30

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F32                                                                                                        31

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_F64                                                                                                            32

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F64                                                                                                           33

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.f64 < S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F64                                                                                                           34

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.f64 == S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F64                                                                                                           35

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.f64 <= S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F64                                                                                                        36

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.f64 > S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F64                                                                                                        37

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f64 <> S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F64                                                                                                        38

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f64 >= S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F64                                                                                                        39

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F64                                                                                                        40

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F64                                                                                                      41

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f64 >= S1.f64);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F64                                                                                                      42

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F64                                                                                                       43

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

  D0.u64[laneId] = !(S0.f64 > S1.f64);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F64                                                                                                       44

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F64                                                                                                       45

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F64                                                                                                      46

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F64                                                                                                        47

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I16                                                                                                       49

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i16 < S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I16                                                                                                           50

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i16 == S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I16                                                                                                           51

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.i16 <= S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I16                                                                                                           52

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.i16 > S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I16                                                                                                           53

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC

or a scalar register.

  D0.u64[laneId] = S0.i16 <> S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I16                                                                                                       54

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.i16 >= S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U16                                                                                                       57

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u16 < S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U16                                                                                                       58

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u16 == S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U16                                                                                                           59

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.u16 <= S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U16                                                                                                           60

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u16 > S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U16                                                                                                           61

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u16 <> S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U16                                                                                                           62

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.u16 >= S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_I32                                                                                                        64

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I32                                                                                                       65

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i32 < S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I32                                                                                                       66

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i32 == S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I32                                                                                                           67

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.i32 <= S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I32                                                                                                           68

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.i32 > S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I32                                                                                                           69

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.i32 <> S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I32                                                                                                           70

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.i32 >= S1.i32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_I32                                                                                                        71

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_U32                                                                                                        72

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U32                                                                                                       73

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u32 < S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U32                                                                                                           74

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u32 == S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U32                                                                                                           75

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.u32 <= S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U32                                                                                                           76

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u32 > S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U32                                                                                                           77

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC

or a scalar register.

  D0.u64[laneId] = S0.u32 <> S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U32                                                                                                       78

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.u32 >= S1.u32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_U32                                                                                                        79

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_I64                                                                                                        80

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I64                                                                                                           81

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i64 < S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I64                                                                                                           82

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.i64 == S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I64                                                                                                           83

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.i64 <= S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I64                                                                                                           84

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC

or a scalar register.

  D0.u64[laneId] = S0.i64 > S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I64                                                                                                       85

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.i64 <> S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I64                                                                                                       86

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.i64 >= S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_I64                                                                                                        87

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_U64                                                                                                            88

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U64                                                                                                           89

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u64 < S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U64                                                                                                           90

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u64 == S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U64                                                                                                           91

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

  D0.u64[laneId] = S0.u64 <= S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U64                                                                                                       92

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u64 > S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U64                                                                                                       93

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u64 <> S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U64                                                                                                       94

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.u64 >= S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_U64                                                                                                       95

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F16                                                                                                  125

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
half-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f16)) then
        result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f16)) then
        result = S1.u32[1]
  elsif exponent(S0.f16) == 31 then
        // +-INF
        result = S1.u32[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f16) ? 4 : 7]
  else

        // +-0.0
        result = S1.u32[sign(S0.f16) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F32                                                                                                    126

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
single-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f32)) then
        result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f32)) then
        result = S1.u32[1]
  elsif exponent(S0.f32) == 255 then
        // +-INF
        result = S1.u32[sign(S0.f32) ? 2 : 9]
  elsif exponent(S0.f32) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f32) ? 3 : 8]
  elsif 64'F(abs(S0.f32)) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f32) ? 4 : 7]
  else
        // +-0.0
        result = S1.u32[sign(S0.f32) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F64                                                                                                 127

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
double-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(S0.f64) then
        result = S1.u32[0]
  elsif isQuietNAN(S0.f64) then
        result = S1.u32[1]
  elsif exponent(S0.f64) == 2047 then
        // +-INF
        result = S1.u32[sign(S0.f64) ? 2 : 9]
  elsif exponent(S0.f64) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f64) ? 4 : 7]
  else
        // +-0.0
        result = S1.u32[sign(S0.f64) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F16                                                                                                        128

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F16                                                                                                       129

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f16 < S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F16                                                                                                       130

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.f16 == S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F16                                                                                                       131

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.f16 <= S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F16                                                                                                       132

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f16 > S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F16                                                                                                       133

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f16 <> S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F16                                                                                                       134

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f16 >= S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F16                                                                                                        135

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F16                                                                                                      136

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F16                                                                                                    137

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F16                                                                                                    138

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F16                                                                                                      139

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F16                                                                                                      140

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F16                                                                                                      141

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F16                                                                                                      142

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F16                                                                                                        143

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F32                                                                                                        144

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F32                                                                                                       145

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f32 < S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F32                                                                                                       146

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.f32 == S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F32                                                                                                       147

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.f32 <= S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F32                                                                                                       148

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f32 > S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F32                                                                                                       149

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f32 <> S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F32                                                                                                      150

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f32 >= S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F32                                                                                                       151

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F32                                                                                                       152

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F32                                                                                                     153

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 >= S1.f32);

  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F32                                                                                                      154

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F32                                                                                                      155

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F32                                                                                                      156

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F32                                                                                                    157

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F32                                                                                                    158

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F32                                                                                                      159

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F64                                                                                                      160

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F64                                                                                                       161

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f64 < S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F64                                                                                                       162

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.f64 == S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F64                                                                                                       163

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.f64 <= S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F64                                                                                                       164

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.f64 > S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F64                                                                                                       165

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f64 <> S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F64                                                                                                       166

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f64 >= S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F64                                                                                                        167

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F64                                                                                                      168

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F64                                                                                                    169

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 >= S1.f64);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F64                                                                                                    170

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F64                                                                                                    171

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 > S1.f64);

  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F64                                                                                                      172

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F64                                                                                                      173

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F64                                                                                                      174

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F64                                                                                                        175

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I16                                                                                                       177

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i16 < S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I16                                                                                                       178

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.i16 == S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I16                                                                                                       179

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.i16 <= S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I16                                                                                                       180

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i16 > S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I16                                                                                                       181

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i16 <> S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I16                                                                                                       182

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.i16 >= S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U16                                                                                                       185

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u16 < S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U16                                                                                                     186

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.u16 == S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U16                                                                                                     187

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.u16 <= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U16                                                                                                     188

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u16 > S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U16                                                                                                       189

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u16 <> S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U16                                                                                                       190

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.u16 >= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_I32                                                                                                        192

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I32                                                                                                       193

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i32 < S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I32                                                                                                     194

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.i32 == S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I32                                                                                                     195

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.i32 <= S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I32                                                                                                     196

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i32 > S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I32                                                                                                     197

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i32 <> S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I32                                                                                                      198

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.i32 >= S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_I32                                                                                                       199

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_U32                                                                                                       200

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U32                                                                                                      201

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u32 < S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U32                                                                                                       202

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.u32 == S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U32                                                                                                       203

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.u32 <= S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U32                                                                                                       204

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u32 > S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U32                                                                                                      205

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u32 <> S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U32                                                                                                      206

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.u32 >= S1.u32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_U32                                                                                                       207

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_I64                                                                                                       208

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I64                                                                                                       209

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i64 < S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I64                                                                                                       210

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.i64 == S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I64                                                                                                       211

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.i64 <= S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I64                                                                                                       212

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i64 > S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I64                                                                                                      213

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i64 <> S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I64                                                                                                      214

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.i64 >= S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_I64                                                                                                       215

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_U64                                                                                                       216

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U64                                                                                                       217

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u64 < S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U64                                                                                                       218

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.u64 == S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U64                                                                                                       219

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.u64 <= S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U64                                                                                                      220

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u64 > S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U64                                                                                                      221

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u64 <> S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U64                                                                                                      222

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.u64 >= S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_U64                                                                                                       223

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F16                                                                                                  253

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
half-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f16)) then
        result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f16)) then
        result = S1.u32[1]
  elsif exponent(S0.f16) == 31 then
        // +-INF
        result = S1.u32[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f16) ? 4 : 7]
  else
        // +-0.0
        result = S1.u32[sign(S0.f16) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F32                                                                                                  254

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
single-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f32)) then
        result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f32)) then
        result = S1.u32[1]
  elsif exponent(S0.f32) == 255 then
        // +-INF
        result = S1.u32[sign(S0.f32) ? 2 : 9]
  elsif exponent(S0.f32) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f32) ? 3 : 8]
  elsif 64'F(abs(S0.f32)) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f32) ? 4 : 7]
  else
        // +-0.0
        result = S1.u32[sign(S0.f32) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F64                                                                                                255

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
double-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
