# 16.8. VOP1 Instructions

> RDNA3.5 ISA — pages 292–320

16.8. VOP1 Instructions

Instructions in this format may use a 32-bit literal constant or DPP that occurs immediately after the
instruction.

V_NOP                                                                                                                 0

Do nothing.

V_MOV_B32                                                                                                             1

Move 32-bit data from a vector input into a vector register.

  D0.b32 = S0.b32

Notes

Floating-point modifiers are valid for this instruction if S0 is a 32-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

Functional examples:

        v_mov_b32 v0, v1    // Move into v0 from v1
        v_mov_b32 v0, -v1   // Set v0 to the negation of v1
        v_mov_b32 v0, abs(v1)    // Set v0 to the absolute value of v1

V_READFIRSTLANE_B32                                                                                                   2

Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register.

  declare lane : 32'U;
  if WAVE64 then
        // 64 lanes
        if EXEC == 0x0LL then
            lane = 0U;
            // Force lane 0 if all lanes are disabled
        else
            lane = 32'U(s_ff1_i32_b64(EXEC));
            // Lowest active lane
        endif
  else

        // 32 lanes
        if EXEC_LO.i32 == 0 then
            lane = 0U;
            // Force lane 0 if all lanes are disabled
        else
            lane = 32'U(s_ff1_i32_b32(EXEC_LO));
            // Lowest active lane
        endif
  endif;
  D0.b32 = VGPR[lane][SRC0.u32]

Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped
operation.

V_CVT_I32_F64                                                                                                     3

Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector
register.

  D0.i32 = f64_to_i32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_I32                                                                                                     4

Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector
register.

  D0.f64 = i32_to_f64(S0.i32)

Notes

0ULP accuracy.

V_CVT_F32_I32                                                                                                     5

Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector

register.

  D0.f32 = i32_to_f32(S0.i32)

Notes

0.5ULP accuracy.

V_CVT_F32_U32                                                                                                     6

Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0.u32)

Notes

0.5ULP accuracy.

V_CVT_U32_F32                                                                                                     7

Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector
register.

  D0.u32 = f32_to_u32(S0.f32)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_I32_F32                                                                                                     8

Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector
register.

  D0.i32 = f32_to_i32(S0.f32)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F16_F32                                                                                                    10

Convert from a single-precision float input to a half-precision float value and store the result into a vector
register.

  D0.f16 = f32_to_f16(S0.f32)

Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on
output if specified based on DP denorm mode. Output rounding based on DP rounding mode.

V_CVT_F32_F16                                                                                                    11

Convert from a half-precision float input to a single-precision float value and store the result into a vector
register.

  D0.f32 = f16_to_f32(S0.f16)

Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm
mode.

V_CVT_NEAREST_I32_F32                                                                                            12

Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer
semantics (ignore the default rounding mode) and store the result into a vector register.

  D0.i32 = f32_to_i32(floor(S0.f32 + 0.5F))

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_FLOOR_I32_F32                                                                                                13

Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore
the default rounding mode) and store the result into a vector register.

  D0.i32 = f32_to_i32(floor(S0.f32))

Notes

1ULP accuracy, denormals are supported.

V_CVT_OFF_F32_I4                                                                                                   14

Convert from a signed 4-bit integer input to a single-precision float value using an offset table and store the
result into a vector register.

Used for interpolation in shader. Lookup table on S0[3:0]:

S0 binary Result
1000 -0.5000f
1001 -0.4375f
1010 -0.3750f
1011 -0.3125f
1100 -0.2500f
1101 -0.1875f
1110 -0.1250f
1111 -0.0625f
0000 +0.0000f
0001 +0.0625f
0010 +0.1250f
0011 +0.1875f
0100 +0.2500f
0101 +0.3125f
0110 +0.3750f
0111 +0.4375f

  declare CVT_OFF_TABLE : 32'F[16];
  D0.f32 = CVT_OFF_TABLE[S0.u32[3 : 0]]

V_CVT_F32_F64                                                                                                      15

Convert from a double-precision float input to a single-precision float value and store the result into a vector
register.

  D0.f32 = f64_to_f32(S0.f64)

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_F64_F32                                                                                                      16

Convert from a single-precision float input to a double-precision float value and store the result into a vector
register.

  D0.f64 = f32_to_f64(S0.f32)

Notes

0ULP accuracy, denormals are supported.

V_CVT_F32_UBYTE0                                                                                                   17

Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[7 : 0].u32)

V_CVT_F32_UBYTE1                                                                                                   18

Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[15 : 8].u32)

V_CVT_F32_UBYTE2                                                                                                   19

Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[23 : 16].u32)

V_CVT_F32_UBYTE3                                                                                                 20

Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[31 : 24].u32)

V_CVT_U32_F64                                                                                                    21

Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a
vector register.

  D0.u32 = f64_to_u32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_U32                                                                                                    22

Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a
vector register.

  D0.f64 = u32_to_f64(S0.u32)

Notes

0ULP accuracy.

V_TRUNC_F64                                                                                                      23

Compute the integer part of a double-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

  D0.f64 = trunc(S0.f64)

V_CEIL_F64                                                                                                       24

Round the double-precision float input up to next integer and store the result in floating point format into a
vector register.

  D0.f64 = trunc(S0.f64);
  if ((S0.f64 > 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += 1.0
  endif

V_RNDNE_F64                                                                                                      25

Round the double-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f64 = floor(S0.f64 + 0.5);
  if (isEven(floor(S0.f64)) && (fract(S0.f64) == 0.5)) then
      D0.f64 -= 1.0
  endif

V_FLOOR_F64                                                                                                      26

Round the double-precision float input down to previous integer and store the result in floating point format
into a vector register.

  D0.f64 = trunc(S0.f64);
  if ((S0.f64 < 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += -1.0
  endif

V_PIPEFLUSH                                                                                                      27

Flush the vector ALU pipeline through the destination cache.

V_MOV_B16                                                                                                        28

Move 16-bit data from a vector input into a vector register.

  D0.b16 = S0.b16

Notes

Floating-point modifiers are valid for this instruction if S0 is a 16-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

V_FRACT_F32                                                                                                           32

Compute the fractional portion of a single-precision float input and store the result in floating point format into
a vector register.

  D0.f32 = S0.f32 + -floor(S0.f32)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3f7fffff.

V_TRUNC_F32                                                                                                           33

Compute the integer part of a single-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

  D0.f32 = trunc(S0.f32)

V_CEIL_F32                                                                                                            34

Round the single-precision float input up to next integer and store the result in floating point format into a
vector register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 > 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += 1.0F
  endif

V_RNDNE_F32                                                                                                           35

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f32 = floor(S0.f32 + 0.5F);
  if (isEven(64'F(floor(S0.f32))) && (fract(S0.f32) == 0.5F)) then
        D0.f32 -= 1.0F
  endif

V_FLOOR_F32                                                                                                        36

Round the single-precision float input down to previous integer and store the result in floating point format
into a vector register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 < 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += -1.0F
  endif

V_EXP_F32                                                                                                          37

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

  D0.f32 = pow(2.0F, S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_EXP_F32(0xff800000) => 0x00000000        // exp(-INF) = 0
  V_EXP_F32(0x80000000) => 0x3f800000        // exp(-0.0) = 1
  V_EXP_F32(0x7f800000) => 0x7f800000        // exp(+INF) = +INF

V_LOG_F32                                                                                                          39

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

  D0.f32 = log2(S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_LOG_F32(0xff800000) => 0xffc00000       // log(-INF) = NAN
  V_LOG_F32(0xbf800000) => 0xffc00000       // log(-1.0) = NAN
  V_LOG_F32(0x80000000) => 0xff800000       // log(-0.0) = -INF
  V_LOG_F32(0x00000000) => 0xff800000       // log(+0.0) = -INF
  V_LOG_F32(0x3f800000) => 0x00000000       // log(+1.0) = 0
  V_LOG_F32(0x7f800000) => 0x7f800000       // log(+INF) = +INF

V_RCP_F32                                                                                                           42

Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector
register.

  D0.f32 = 1.0F / S0.f32

Notes

1ULP accuracy. Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA
operations. Denormals are flushed.

Functional examples:

  V_RCP_F32(0xff800000) => 0x80000000       // rcp(-INF) = -0
  V_RCP_F32(0xc0000000) => 0xbf000000       // rcp(-2.0) = -0.5
  V_RCP_F32(0x80000000) => 0xff800000       // rcp(-0.0) = -INF
  V_RCP_F32(0x00000000) => 0x7f800000       // rcp(+0.0) = +INF
  V_RCP_F32(0x7f800000) => 0x00000000       // rcp(+INF) = +0

V_RCP_IFLAG_F32                                                                                                     43

Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result
into a vector register. This opcode is intended for use as part of an integer division macro.

  D0.f32 = 1.0F / S0.f32;
  // Can only raise integer DIV_BY_ZERO exception

Notes

Can raise integer DIV_BY_ZERO exception but cannot raise floating-point exceptions. To be used in an integer

reciprocal macro by the compiler with one of the sequences listed below (depending on signed or unsigned
operation).

Unsigned usage:
CVT_F32_U32
RCP_IFLAG_F32
MUL_F32 (2**32 - 1)
CVT_U32_F32

Signed usage:
CVT_F32_I32
RCP_IFLAG_F32
MUL_F32 (2**31 - 1)
CVT_I32_F32

V_RSQ_F32                                                                                                          46

Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the
result into a vector register.

  D0.f32 = 1.0F / sqrt(S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_RSQ_F32(0xff800000) => 0xffc00000       // rsq(-INF) = NAN
  V_RSQ_F32(0x80000000) => 0xff800000       // rsq(-0.0) = -INF
  V_RSQ_F32(0x00000000) => 0x7f800000       // rsq(+0.0) = +INF
  V_RSQ_F32(0x40800000) => 0x3f000000       // rsq(+4.0) = +0.5
  V_RSQ_F32(0x7f800000) => 0x00000000       // rsq(+INF) = +0

V_RCP_F64                                                                                                          47

Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector
register.

  D0.f64 = 1.0 / S0.f64

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_RSQ_F64                                                                                                           49

Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the
result into a vector register.

  D0.f64 = 1.0 / sqrt(S0.f64)

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_SQRT_F32                                                                                                          51

Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector
register.

  D0.f32 = sqrt(S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_SQRT_F32(0xff800000) => 0xffc00000       // sqrt(-INF) = NAN
  V_SQRT_F32(0x80000000) => 0x80000000       // sqrt(-0.0) = -0
  V_SQRT_F32(0x00000000) => 0x00000000       // sqrt(+0.0) = +0
  V_SQRT_F32(0x40800000) => 0x40000000       // sqrt(+4.0) = +2.0
  V_SQRT_F32(0x7f800000) => 0x7f800000       // sqrt(+INF) = +INF

V_SQRT_F64                                                                                                          52

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector
register.

  D0.f64 = sqrt(S0.f64)

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_SIN_F32                                                                                                         53

Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f32 = sin(S0.f32 * 32'F(PI * 2.0))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_SIN_F32(0xff800000) => 0xffc00000       // sin(-INF) = NAN
  V_SIN_F32(0xff7fffff) => 0x00000000       // -MaxFloat, finite
  V_SIN_F32(0x80000000) => 0x80000000       // sin(-0.0) = -0
  V_SIN_F32(0x3e800000) => 0x3f800000       // sin(0.25) = 1
  V_SIN_F32(0x7f800000) => 0xffc00000       // sin(+INF) = NAN

V_COS_F32                                                                                                         54

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f32 = cos(S0.f32 * 32'F(PI * 2.0))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_COS_F32(0xff800000) => 0xffc00000       // cos(-INF) = NAN
  V_COS_F32(0xff7fffff) => 0x3f800000       // -MaxFloat, finite
  V_COS_F32(0x80000000) => 0x3f800000       // cos(-0.0) = 1
  V_COS_F32(0x3e800000) => 0x00000000       // cos(0.25) = 0
  V_COS_F32(0x7f800000) => 0xffc00000       // cos(+INF) = NAN

V_NOT_B32                                                                                                         55

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u32 = ~S0.u32

Notes

Input and output modifiers not supported.

V_BFREV_B32                                                                                                       56

Reverse the order of bits in a vector input and store the result into a vector register.

  D0.u32[31 : 0] = S0.u32[0 : 31]

Notes

Input and output modifiers not supported.

V_CLZ_I32_U32                                                                                                     57

Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits.

  D0.i32 = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from MSB
        if S0.u32[31 - i] == 1'1U then
            D0.i32 = i;
            break
        endif
  endfor

Notes

Compare with S_CLZ_I32_U32, which performs the equivalent operation in the scalar ALU.

Functional examples:

  V_CLZ_I32_U32(0x00000000) => 0xffffffff
  V_CLZ_I32_U32(0x800000ff) => 0
  V_CLZ_I32_U32(0x100000ff) => 3
  V_CLZ_I32_U32(0x0000ffff) => 16
  V_CLZ_I32_U32(0x00000001) => 31

V_CTZ_I32_B32                                                                                                      58

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits in the input.

  D0.i32 = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from LSB
        if S0.u32[i] == 1'1U then
            D0.i32 = i;
            break
        endif
  endfor

Notes

Compare with S_CTZ_I32_B32, which performs the equivalent operation in the scalar ALU.

Functional examples:

  V_CTZ_I32_B32(0x00000000) => 0xffffffff
  V_CTZ_I32_B32(0xff000001) => 0
  V_CTZ_I32_B32(0xff000008) => 3
  V_CTZ_I32_B32(0xffff0000) => 16
  V_CTZ_I32_B32(0x80000000) => 31

V_CLS_I32                                                                                                          59

Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a
vector register. Store -1 if all input bits are the same.

  D0.i32 = -1;
  // Set if all bits are the same
  for i in 1 : 31 do
        // Search from MSB
        if S0.i32[31 - i] != S0.i32[31] then
            D0.i32 = i;
            break
        endif
  endfor

Notes

Compare with S_CLS_I32, which performs the equivalent operation in the scalar ALU.

Functional examples:

  V_CLS_I32(0x00000000) => 0xffffffff
  V_CLS_I32(0x40000000) => 1
  V_CLS_I32(0x80000000) => 1
  V_CLS_I32(0x0fffffff) => 4
  V_CLS_I32(0xffff0000) => 16
  V_CLS_I32(0xfffffffe) => 31
  V_CLS_I32(0xffffffff) => 0xffffffff

V_FREXP_EXP_I32_F64                                                                                                 60

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a
vector register.

  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f64) - 1023 + 1
  endif

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64,
which returns the significand. See the C library function frexp() for more information.

V_FREXP_MANT_F64                                                                                                    61

Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double-
precision float into a vector register.

  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.f64 = S0.f64
  else
        D0.f64 = mantissa(S0.f64)
  endif

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I32_F64, which returns integer exponent. See the C library function
frexp() for more information.

V_FRACT_F64                                                                                                         62

Compute the fractional portion of a double-precision float input and store the result in floating point format

into a vector register.

  D0.f64 = S0.f64 + -floor(S0.f64)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3fefffffffffffff.

V_FREXP_EXP_I32_F32                                                                                                 63

Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector
register.

  if ((64'F(S0.f32) == +INF) || (64'F(S0.f32) == -INF) || isNAN(64'F(S0.f32))) then
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f32) - 127 + 1
  endif

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32,
which returns the significand. See the C library function frexp() for more information.

V_FREXP_MANT_F32                                                                                                    64

Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single-
precision float into a vector register.

  if ((64'F(S0.f32) == +INF) || (64'F(S0.f32) == -INF) || isNAN(64'F(S0.f32))) then
        D0.f32 = S0.f32
  else
        D0.f32 = mantissa(S0.f32)
  endif

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I32_F32, which returns integer exponent. See the C library function

frexp()   for more information.

V_MOVRELD_B32                                                                                                 66

Move data from a vector input into a relatively-indexed vector register.

   addr = DST.u32;
   // Raw value from instruction
   addr += M0.u32[31 : 0];
   VGPR[laneId][addr].b32 = S0.b32

Notes

Example: The following instruction sequence performs the move v15 <= v7:

          s_mov_b32 m0, 10
          v_movreld_b32 v5, v7

V_MOVRELS_B32                                                                                                 67

Move data from a relatively-indexed vector register into another vector register.

   addr = SRC0.u32;
   // Raw value from instruction
   addr += M0.u32[31 : 0];
   D0.b32 = VGPR[laneId][addr].b32

Notes

Example: The following instruction sequence performs the move v5 <= v17:

          s_mov_b32 m0, 10
          v_movrels_b32 v5, v7

V_MOVRELSD_B32                                                                                                68

Move data from a relatively-indexed vector register into another relatively-indexed vector register.

   addrs = SRC0.u32;
   // Raw value from instruction
   addrd = DST.u32;

  // Raw value from instruction
  addrs += M0.u32[31 : 0];
  addrd += M0.u32[31 : 0];
  VGPR[laneId][addrd].b32 = VGPR[laneId][addrs].b32

Notes

Example: The following instruction sequence performs the move v15 <= v17:

        s_mov_b32 m0, 10
        v_movrelsd_b32 v5, v7

V_MOVRELSD_2_B32                                                                                                   72

Move data from a relatively-indexed vector register into another relatively-indexed vector register, using
different offsets for each index.

  addrs = SRC0.u32;
  // Raw value from instruction
  addrd = DST.u32;
  // Raw value from instruction
  addrs += M0.u32[9 : 0].u32;
  addrd += M0.u32[25 : 16].u32;
  VGPR[laneId][addrd].b32 = VGPR[laneId][addrs].b32

Notes

Example: The following instruction sequence performs the move v25 <= v17:

        s_mov_b32 m0, ((20 << 16) | 10)
        v_movrelsd_2_b32 v5, v7

V_CVT_F16_U16                                                                                                      80

Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector
register.

  D0.f16 = u16_to_f16(S0.u16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_F16_I16                                                                                                      81

Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector
register.

  D0.f16 = i16_to_f16(S0.i16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_U16_F16                                                                                                      82

Convert from a half-precision float input to an unsigned 16-bit integer value and store the result into a vector
register.

  D0.u16 = f16_to_u16(S0.f16)

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_I16_F16                                                                                                      83

Convert from a half-precision float input to a signed 16-bit integer value and store the result into a vector
register.

  D0.i16 = f16_to_i16(S0.f16)

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_RCP_F16                                                                                                         84

Calculate the reciprocal of the half-precision float input using IEEE rules and store the result into a vector
register.

  D0.f16 = 16'1.0 / S0.f16

Notes

0.51ULP accuracy.

Functional examples:

  V_RCP_F16(0xfc00) => 0x8000        // rcp(-INF) = -0
  V_RCP_F16(0xc000) => 0xb800        // rcp(-2.0) = -0.5
  V_RCP_F16(0x8000) => 0xfc00        // rcp(-0.0) = -INF
  V_RCP_F16(0x0000) => 0x7c00        // rcp(+0.0) = +INF
  V_RCP_F16(0x7c00) => 0x0000        // rcp(+INF) = +0

V_SQRT_F16                                                                                                        85

Calculate the square root of the half-precision float input using IEEE rules and store the result into a vector
register.

  D0.f16 = sqrt(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_SQRT_F16(0xfc00) => 0xfe00           // sqrt(-INF) = NAN
  V_SQRT_F16(0x8000) => 0x8000           // sqrt(-0.0) = -0
  V_SQRT_F16(0x0000) => 0x0000           // sqrt(+0.0) = +0
  V_SQRT_F16(0x4400) => 0x4000           // sqrt(+4.0) = +2.0
  V_SQRT_F16(0x7c00) => 0x7c00           // sqrt(+INF) = +INF

V_RSQ_F16                                                                                                         86

Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result
into a vector register.

  D0.f16 = 16'1.0 / sqrt(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_RSQ_F16(0xfc00) => 0xfe00        // rsq(-INF) = NAN
  V_RSQ_F16(0x8000) => 0xfc00        // rsq(-0.0) = -INF
  V_RSQ_F16(0x0000) => 0x7c00        // rsq(+0.0) = +INF
  V_RSQ_F16(0x4400) => 0x3800        // rsq(+4.0) = +0.5
  V_RSQ_F16(0x7c00) => 0x0000        // rsq(+INF) = +0

V_LOG_F16                                                                                                         87

Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register.

  D0.f16 = log2(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_LOG_F16(0xfc00) => 0xfe00        // log(-INF) = NAN
  V_LOG_F16(0xbc00) => 0xfe00        // log(-1.0) = NAN
  V_LOG_F16(0x8000) => 0xfc00        // log(-0.0) = -INF
  V_LOG_F16(0x0000) => 0xfc00        // log(+0.0) = -INF
  V_LOG_F16(0x3c00) => 0x0000        // log(+1.0) = 0
  V_LOG_F16(0x7c00) => 0x7c00        // log(+INF) = +INF

V_EXP_F16                                                                                                         88

Calculate 2 raised to the power of the half-precision float input and store the result into a vector register.

  D0.f16 = pow(16'2.0, S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_EXP_F16(0xfc00) => 0x0000        // exp(-INF) = 0
  V_EXP_F16(0x8000) => 0x3c00        // exp(-0.0) = 1
  V_EXP_F16(0x7c00) => 0x7c00        // exp(+INF) = +INF

V_FREXP_MANT_F16                                                                                                     89

Extract the binary significand, or mantissa, of a half-precision float input and store the result as a half-
precision float into a vector register.

  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.f16 = S0.f16
  else
        D0.f16 = mantissa(S0.f16)
  endif

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I16_F16, which returns integer exponent. See the C library function
frexp() for more information.

V_FREXP_EXP_I16_F16                                                                                                  90

Extract the exponent of a half-precision float input and store the result as a signed 16-bit integer into a vector
register.

  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.i16 = 16'0
  else
        D0.i16 = 16'I(exponent(S0.f16) - 15 + 1)
  endif

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). See also V_FREXP_MANT_F16,
which returns the significand. See the C library function frexp() for more information.

V_FLOOR_F16                                                                                                          91

Round the half-precision float input down to previous integer and store the result in floating point format into
a vector register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 < 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += -16'1.0
  endif

V_CEIL_F16                                                                                                       92

Round the half-precision float input up to next integer and store the result in floating point format into a vector
register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 > 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += 16'1.0
  endif

V_TRUNC_F16                                                                                                      93

Compute the integer part of a half-precision float input using round toward zero semantics and store the result
in floating point format into a vector register.

  D0.f16 = trunc(S0.f16)

V_RNDNE_F16                                                                                                      94

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
      D0.f16 -= 16'1.0
  endif

V_FRACT_F16                                                                                                      95

Compute the fractional portion of a half-precision float input and store the result in floating point format into a
vector register.

  D0.f16 = S0.f16 + -floor(S0.f16)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

V_SIN_F16                                                                                                         96

Calculate the trigonometric sine of a half-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f16 = sin(S0.f16 * 16'F(PI * 2.0))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_SIN_F16(0xfc00) => 0xfe00        // sin(-INF) = NAN
  V_SIN_F16(0xfbff) => 0x0000        // Most negative finite FP16
  V_SIN_F16(0x8000) => 0x8000        // sin(-0.0) = -0
  V_SIN_F16(0x3400) => 0x3c00        // sin(0.25) = 1
  V_SIN_F16(0x7bff) => 0x0000        // Most positive finite FP16
  V_SIN_F16(0x7c00) => 0xfe00        // sin(+INF) = NAN

V_COS_F16                                                                                                         97

Calculate the trigonometric cosine of a half-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f16 = cos(S0.f16 * 16'F(PI * 2.0))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_COS_F16(0xfc00) => 0xfe00        // cos(-INF) = NAN
  V_COS_F16(0xfbff) => 0x3c00        // Most negative finite FP16
  V_COS_F16(0x8000) => 0x3c00        // cos(-0.0) = 1
  V_COS_F16(0x3400) => 0x0000        // cos(0.25) = 0

  V_COS_F16(0x7bff) => 0x3c00        // Most positive finite FP16
  V_COS_F16(0x7c00) => 0xfe00        // cos(+INF) = NAN

V_SAT_PK_U8_I16                                                                                                 98

Given two 16-bit signed integer inputs, saturate each input over an 8-bit unsigned range, pack the resulting
values into a 16-bit word and store the result into a vector register.

  SAT8 = lambda(n) (
        if n.i32 <= 0 then
            return 8'0U
        elsif n >= 16'I(0xff) then
            return 8'255U
        else
            return n[7 : 0].u8
        endif);
  D0.b16 = { SAT8(S0[31 : 16].i16), SAT8(S0[15 : 0].i16) }

Notes

Used for 4x16bit data packed as 4x8bit data.

V_CVT_NORM_I16_F16                                                                                              99

Convert from a half-precision float input to a signed normalized short and store the result into a vector
register.

  D0.i16 = f16_to_snorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_CVT_NORM_U16_F16                                                                                             100

Convert from a half-precision float input to an unsigned normalized short and store the result into a vector
register.

  D0.u16 = f16_to_unorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_SWAP_B32                                                                                               101

Swap the values in two vector registers.

  tmp = D0.b32;
  D0.b32 = S0.b32;
  S0.b32 = tmp

Notes

Input and output modifiers not supported; this is an untyped operation.

V_SWAP_B16                                                                                               102

Swap the values in two vector registers.

  tmp = D0.b16;
  D0.b16 = S0.b16;
  S0.b16 = tmp

Notes

Input and output modifiers not supported; this is an untyped operation.

V_PERMLANE64_B32                                                                                         103

Perform a specific permutation across lanes where the high half and low half of a wave64 are swapped.
Performs no operation in wave32 mode.

  declare tmp : 32'B[64];
  declare lane : 32'U;
  if WAVE32 then
        // Supported in wave64 ONLY; treated as scalar NOP in wave32
        s_nop(16'0U)
  else
        for lane in 0U : 63U do
            // Copy original S0 in case D==S0
            tmp[lane] = VGPR[lane][SRC0.u32]
        endfor;
        for lane in 0U : 63U do
            altlane = { ~lane[5], lane[4 : 0] };
            // 0<->32, ..., 31<->63
            if EXEC[lane].u1 then
                  VGPR[lane][VDST.u32] = tmp[altlane]
            endif

        endfor
  endif

Notes

In wave32 mode this opcode is translated to V_NOP and performs no writes.

In wave64 the EXEC mask of the destination lane is used as the read mask for the alternate lane; as a result this
opcode may read values from disabled lanes.

The source must be a VGPR and SVGPRs are not allowed for this opcode.

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

V_SWAPREL_B32                                                                                                104

Swap the values in two relatively-indexed vector registers.

  addrs = SRC0.u32;
  // Raw value from instruction
  addrd = DST.u32;
  // Raw value from instruction
  addrs += M0.u32[9 : 0].u32;
  addrd += M0.u32[25 : 16].u32;
  tmp = VGPR[laneId][addrd].b32;
  VGPR[laneId][addrd].b32 = VGPR[laneId][addrs].b32;
  VGPR[laneId][addrs].b32 = tmp

Notes

Input and output modifiers not supported; this is an untyped operation.

Example: The following instruction sequence swaps v25 and v17:

        s_mov_b32 m0, ((20 << 16) | 10)
        v_swaprel_b32 v5, v7

V_NOT_B16                                                                                                    105

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u16 = ~S0.u16

Notes
