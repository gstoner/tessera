# 16.9. VOPC Instructions

> RDNA3 ISA — pages 304–358

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

V_CMP_F_F16                                                                                              0

Return 0.

    D0.u64[laneId] = 1'0U;
    // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F16                                                                                             1

Return 1 iff A less than B.

  D0.u64[laneId] = S0.f16 < S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F16                                                              2

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.f16 == S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F16                                                              3

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.f16 <= S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F16                                                              4

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.f16 > S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F16                                                                 5

Return 1 iff A less than or greater than B.

  D0.u64[laneId] = S0.f16 <> S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F16                                                                 6

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.f16 >= S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F16                                                                  7

Return 1 iff A orderable with B.

  D0.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F16                                                                  8

Return 1 iff A not orderable with B.

  D0.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F16                                                             9

Return 1 iff A not greater than or equal to B.

  D0.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F16                                                            10

Return 1 iff A not less than or greater than B.

  D0.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F16                                                            11

Return 1 iff A not greater than B.

  D0.u64[laneId] = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F16                                                            12

Return 1 iff A not less than or equal to B.

  D0.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F16                                                            13

Return 1 iff A not equal to B.

  D0.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F16                                                            14

Return 1 iff A not less than B.

  D0.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F16                                                              15

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_F32                                                              16

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F32                                                             17

Return 1 iff A less than B.

  D0.u64[laneId] = S0.f < S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F32                                                             18

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.f == S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F32                                                             19

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.f <= S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F32                                                             20

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.f > S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F32                                                             21

Return 1 iff A less than or greater than B.

  D0.u64[laneId] = S0.f <> S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F32                                                             22

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.f >= S1.f;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F32                                                              23

Return 1 iff A orderable with B.

  D0.u64[laneId] = (!isNAN(64'F(S0.f)) && !isNAN(64'F(S1.f)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F32                                                              24

Return 1 iff A not orderable with B.

  D0.u64[laneId] = (isNAN(64'F(S0.f)) || isNAN(64'F(S1.f)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F32                                                            25

Return 1 iff A not greater than or equal to B.

  D0.u64[laneId] = !(S0.f >= S1.f);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F32                                                            26

Return 1 iff A not less than or greater than B.

  D0.u64[laneId] = !(S0.f <> S1.f);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F32                                                            27

Return 1 iff A not greater than B.

  D0.u64[laneId] = !(S0.f > S1.f);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F32                                                            28

Return 1 iff A not less than or equal to B.

  D0.u64[laneId] = !(S0.f <= S1.f);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F32                                                            29

Return 1 iff A not equal to B.

  D0.u64[laneId] = !(S0.f == S1.f);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F32                                                            30

Return 1 iff A not less than B.

  D0.u64[laneId] = !(S0.f < S1.f);
  // With NAN inputs this is not the same operation as >=

  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F32                                                              31

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_F64                                                              32

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F64                                                             33

Return 1 iff A less than B.

  D0.u64[laneId] = S0.f64 < S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F64                                                             34

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.f64 == S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_F64                                                             35

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.f64 <= S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_F64                                                             36

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.f64 > S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F64                                                             37

Return 1 iff A less than or greater than B.

  D0.u64[laneId] = S0.f64 <> S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F64                                                             38

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.f64 >= S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F64                                                              39

Return 1 iff A orderable with B.

  D0.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F64                                                              40

Return 1 iff A not orderable with B.

  D0.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGE_F64                                                            41

Return 1 iff A not greater than or equal to B.

  D0.u64[laneId] = !(S0.f64 >= S1.f64);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLG_F64                                                            42

Return 1 iff A not less than or greater than B.

  D0.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F64                                                            43

Return 1 iff A not greater than B.

  D0.u64[laneId] = !(S0.f64 > S1.f64);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLE_F64                                                            44

Return 1 iff A not less than or equal to B.

  D0.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NEQ_F64                                                            45

Return 1 iff A not equal to B.

  D0.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F64                                                            46

Return 1 iff A not less than B.

  D0.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_F64                                                              47

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I16                                                             49

Return 1 iff A less than B.

  D0.u64[laneId] = S0.i16 < S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I16                                                             50

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.i16 == S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I16                                                             51

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.i16 <= S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I16                                                             52

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.i16 > S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I16                                                             53

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.i16 <> S1.i16;

  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I16                                                             54

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.i16 >= S1.i16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U16                                                             57

Return 1 iff A less than B.

  D0.u64[laneId] = S0.u16 < S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U16                                                             58

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.u16 == S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U16                                                             59

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.u16 <= S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U16                                                             60

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.u16 > S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U16                                                             61

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.u16 <> S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U16                                                             62

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.u16 >= S1.u16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_I32                                                              64

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I32                                                             65

Return 1 iff A less than B.

  D0.u64[laneId] = S0.i < S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I32                                                             66

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.i == S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I32                                                             67

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.i <= S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I32                                                             68

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.i > S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I32                                                             69

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.i <> S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I32                                                             70

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.i >= S1.i;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_I32                                                              71

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_U32                                                              72

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U32                                                             73

Return 1 iff A less than B.

  D0.u64[laneId] = S0.u < S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U32                                                             74

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.u == S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U32                                                             75

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.u <= S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U32                                                             76

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.u > S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U32                                                             77

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.u <> S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U32                                                             78

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.u >= S1.u;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_U32                                                              79

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_I64                                                              80

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_I64                                                             81

Return 1 iff A less than B.

  D0.u64[laneId] = S0.i64 < S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_I64                                                             82

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.i64 == S1.i64;

  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_I64                                                             83

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.i64 <= S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_I64                                                             84

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.i64 > S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_I64                                                             85

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.i64 <> S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_I64                                                             86

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.i64 >= S1.i64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_I64                                                              87

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_F_U64                                                              88

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U64                                                             89

Return 1 iff A less than B.

  D0.u64[laneId] = S0.u64 < S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U64                                                             90

Return 1 iff A equal to B.

  D0.u64[laneId] = S0.u64 == S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LE_U64                                                             91

Return 1 iff A less than or equal to B.

  D0.u64[laneId] = S0.u64 <= S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GT_U64                                                             92

Return 1 iff A greater than B.

  D0.u64[laneId] = S0.u64 > S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U64                                                             93

Return 1 iff A not equal to B.

  D0.u64[laneId] = S0.u64 <> S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U64                                                                                                    94

Return 1 iff A greater than or equal to B.

  D0.u64[laneId] = S0.u64 >= S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_U64                                                                                                     95

Return 1.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F16                                                                                                125

IEEE numeric class function specified in S1.u, performed on S0.f16.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f16)) then
        result = S1.u[0]
  elsif isQuietNAN(64'F(S0.f16)) then
        result = S1.u[1]
  elsif exponent(S0.f16) == 31 then
        // +-INF
        result = S1.u[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f16) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f16) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F32                                                                                                126

IEEE numeric class function specified in S1.u, performed on S0.f.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f)) then
        result = S1.u[0]
  elsif isQuietNAN(64'F(S0.f)) then
        result = S1.u[1]
  elsif exponent(S0.f) == 255 then
        // +-INF
        result = S1.u[sign(S0.f) ? 2 : 9]

  elsif exponent(S0.f) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f) ? 3 : 8]
  elsif 64'F(abs(S0.f)) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F64                                                                                                127

IEEE numeric class function specified in S1.u, performed on S0.f64.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(S0.f64) then
        result = S1.u[0]
  elsif isQuietNAN(S0.f64) then
        result = S1.u[1]
  elsif exponent(S0.f64) == 1023 then
        // +-INF
        result = S1.u[sign(S0.f64) ? 2 : 9]
  elsif exponent(S0.f64) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f64) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f64) ? 5 : 6]

  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F16                                                                                               128

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F16                                                                                              129

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.f16 < S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F16                                                                                              130

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.f16 == S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F16                                                                                              131

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.f16 <= S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F16                                                                                              132

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.f16 > S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F16                                                                                              133

Return 1 iff A less than or greater than B.

  EXEC.u64[laneId] = S0.f16 <> S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F16                                                                                              134

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.f16 >= S1.f16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F16                                                                                               135

Return 1 iff A orderable with B.

  EXEC.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F16                                                                                               136

Return 1 iff A not orderable with B.

  EXEC.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F16                                                                                             137

Return 1 iff A not greater than or equal to B.

  EXEC.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F16                                                                                             138

Return 1 iff A not less than or greater than B.

  EXEC.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F16                                                                                             139

Return 1 iff A not greater than B.

  EXEC.u64[laneId] = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F16                                                                                             140

Return 1 iff A not less than or equal to B.

  EXEC.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F16                                                                                             141

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F16                                                                                             142

Return 1 iff A not less than B.

  EXEC.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F16                                                                                               143

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F32                                                                                               144

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F32                                                                                              145

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.f < S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F32                                                                                              146

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.f == S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F32                                                                                              147

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.f <= S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F32                                                                                              148

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.f > S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F32                                                                                              149

Return 1 iff A less than or greater than B.

  EXEC.u64[laneId] = S0.f <> S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F32                                                                                              150

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.f >= S1.f

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F32                                                                                               151

Return 1 iff A orderable with B.

  EXEC.u64[laneId] = (!isNAN(64'F(S0.f)) && !isNAN(64'F(S1.f)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F32                                                                                               152

Return 1 iff A not orderable with B.

  EXEC.u64[laneId] = (isNAN(64'F(S0.f)) || isNAN(64'F(S1.f)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F32                                                                                             153

Return 1 iff A not greater than or equal to B.

  EXEC.u64[laneId] = !(S0.f >= S1.f);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F32                                                                                             154

Return 1 iff A not less than or greater than B.

  EXEC.u64[laneId] = !(S0.f <> S1.f);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F32                                                                                             155

Return 1 iff A not greater than B.

  EXEC.u64[laneId] = !(S0.f > S1.f);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F32                                                                                             156

Return 1 iff A not less than or equal to B.

  EXEC.u64[laneId] = !(S0.f <= S1.f);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F32                                                                                             157

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = !(S0.f == S1.f);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F32                                                                                             158

Return 1 iff A not less than B.

  EXEC.u64[laneId] = !(S0.f < S1.f);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F32                                                                                               159

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F64                                                                                               160

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F64                                                                                              161

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.f64 < S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_F64                                                                                              162

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.f64 == S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_F64                                                                                              163

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.f64 <= S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_F64                                                                                              164

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.f64 > S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LG_F64                                                                                              165

Return 1 iff A less than or greater than B.

  EXEC.u64[laneId] = S0.f64 <> S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_F64                                                                                              166

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.f64 >= S1.f64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F64                                                                                               167

Return 1 iff A orderable with B.

  EXEC.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F64                                                                                               168

Return 1 iff A not orderable with B.

  EXEC.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F64                                                                                             169

Return 1 iff A not greater than or equal to B.

  EXEC.u64[laneId] = !(S0.f64 >= S1.f64);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F64                                                                                             170

Return 1 iff A not less than or greater than B.

  EXEC.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F64                                                                                             171

Return 1 iff A not greater than B.

  EXEC.u64[laneId] = !(S0.f64 > S1.f64);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F64                                                                                             172

Return 1 iff A not less than or equal to B.

  EXEC.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F64                                                                                             173

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F64                                                                                             174

Return 1 iff A not less than B.

  EXEC.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F64                                                                                               175

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I16                                                                                              177

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.i16 < S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I16                                                                                              178

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.i16 == S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I16                                                                                              179

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.i16 <= S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I16                                                                                              180

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.i16 > S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I16                                                                                              181

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.i16 <> S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I16                                                                                              182

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.i16 >= S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U16                                                                                              185

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.u16 < S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U16                                                                                              186

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.u16 == S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U16                                                                                              187

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.u16 <= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U16                                                                                              188

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.u16 > S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U16                                                                                              189

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.u16 <> S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U16                                                                                              190

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.u16 >= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_I32                                                                                               192

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I32                                                                                              193

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.i < S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I32                                                                                              194

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.i == S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I32                                                                                              195

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.i <= S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I32                                                                                              196

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.i > S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I32                                                                                              197

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.i <> S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I32                                                                                              198

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.i >= S1.i

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_I32                                                                                               199

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_U32                                                                                               200

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U32                                                                                              201

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.u < S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U32                                                                                              202

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.u == S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U32                                                                                              203

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.u <= S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U32                                                                                              204

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.u > S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U32                                                                                              205

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.u <> S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U32                                                                                              206

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.u >= S1.u

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_U32                                                                                               207

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_I64                                                                                               208

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I64                                                                                              209

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.i64 < S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I64                                                                                              210

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.i64 == S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I64                                                                                              211

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.i64 <= S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I64                                                                                              212

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.i64 > S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I64                                                                                              213

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.i64 <> S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I64                                                                                              214

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.i64 >= S1.i64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_I64                                                                                               215

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_U64                                                                                               216

Return 0.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_U64                                                                                              217

Return 1 iff A less than B.

  EXEC.u64[laneId] = S0.u64 < S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_U64                                                                                              218

Return 1 iff A equal to B.

  EXEC.u64[laneId] = S0.u64 == S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U64                                                                                              219

Return 1 iff A less than or equal to B.

  EXEC.u64[laneId] = S0.u64 <= S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U64                                                                                              220

Return 1 iff A greater than B.

  EXEC.u64[laneId] = S0.u64 > S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U64                                                                                              221

Return 1 iff A not equal to B.

  EXEC.u64[laneId] = S0.u64 <> S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U64                                                                                                  222

Return 1 iff A greater than or equal to B.

  EXEC.u64[laneId] = S0.u64 >= S1.u64

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_U64                                                                                                   223

Return 1.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F16                                                                                               253

IEEE numeric class function specified in S1.u, performed on S0.f16.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;

  if isSignalNAN(64'F(S0.f16)) then
        result = S1.u[0]
  elsif isQuietNAN(64'F(S0.f16)) then
        result = S1.u[1]
  elsif exponent(S0.f16) == 31 then
        // +-INF
        result = S1.u[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f16) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f16) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F32                                                                                               254

IEEE numeric class function specified in S1.u, performed on S0.f.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(64'F(S0.f)) then
        result = S1.u[0]
  elsif isQuietNAN(64'F(S0.f)) then
        result = S1.u[1]
  elsif exponent(S0.f) == 255 then
        // +-INF
        result = S1.u[sign(S0.f) ? 2 : 9]
  elsif exponent(S0.f) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f) ? 3 : 8]

  elsif 64'F(abs(S0.f)) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F64                                                                                               255

IEEE numeric class function specified in S1.u, performed on S0.f64.

The function reports true if the floating point value is any of the numeric types selected in S1.u according to the
following list:

S1.u[0] -- value is a signaling NAN.
S1.u[1] -- value is a quiet NAN.
S1.u[2] -- value is negative infinity.
S1.u[3] -- value is a negative normal value.
S1.u[4] -- value is a negative denormal value.
S1.u[5] -- value is negative zero.
S1.u[6] -- value is positive zero.
S1.u[7] -- value is a positive denormal value.
S1.u[8] -- value is a positive normal value.
S1.u[9] -- value is positive infinity.

  declare result : 1'U;
  if isSignalNAN(S0.f64) then
        result = S1.u[0]
  elsif isQuietNAN(S0.f64) then
        result = S1.u[1]
  elsif exponent(S0.f64) == 1023 then
        // +-INF
        result = S1.u[sign(S0.f64) ? 2 : 9]
  elsif exponent(S0.f64) > 0 then
        // +-normal value
        result = S1.u[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
        // +-denormal value
        result = S1.u[sign(S0.f64) ? 4 : 7]
  else
        // +-0.0
        result = S1.u[sign(S0.f64) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = result
