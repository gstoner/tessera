# 16.12. VOP3 & VOP3SD Instructions

> RDNA3 ISA — pages 374–506

16.12. VOP3 & VOP3SD Instructions
VOP3 instructions use one of two encodings:

  VOP3SD        this encoding allows specifying a unique scalar destination, and is used only for:
                V_ADD_CO_U32
                V_SUB_CO_U32
                V_SUBREV_CO_U32
                V_ADDC_CO_U32
                V_SUBB_CO_U32
                V_SUBBREV_CO_U32
                V_DIV_SCALE_F32
                V_DIV_SCALE_F64
                V_MAD_U64_U32
                V_MAD_I64_I32

  VOP3          all other VALU instructions use this encoding

V_NOP                                                                                                              384

Do nothing.

V_MOV_B32                                                                                                          385

Move data from a vector input into a vector register.

  D0.b = S0.b

Notes

Floating-point modifiers are valid for this instruction if S0.u is a 32-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

Functional examples:

        v_mov_b32 v0, v1    // Move v1 to v0
        v_mov_b32 v0, -v1   // Set v1 to the negation of v0
        v_mov_b32 v0, abs(v1)   // Set v1 to the absolute value of v0

V_READFIRSTLANE_B32                                                                                                 386

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
        if EXEC_LO.i == 0 then
            lane = 0U;
            // Force lane 0 if all lanes are disabled
        else
            lane = 32'U(s_ff1_i32_b32(EXEC_LO));
            // Lowest active lane
        endif
  endif;
  D0.b = VGPR[lane][SRC0.u]

Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped
operation.

V_CVT_I32_F64                                                                                                       387

Convert from a double-precision float input to a signed 32-bit integer and store the result into a vector register.

  D0.i = f64_to_i32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_I32                                                                                                       388

Convert from a signed 32-bit integer input to a double-precision float and store the result into a vector register.

  D0.f64 = i32_to_f64(S0.i)

Notes

0ULP accuracy.

V_CVT_F32_I32                                                                                                   389

Convert from a signed 32-bit integer input to a single-precision float and store the result into a vector register.

  D0.f = i32_to_f32(S0.i)

Notes

0.5ULP accuracy.

V_CVT_F32_U32                                                                                                   390

Convert from an unsigned 32-bit integer input to a single-precision float and store the result into a vector
register.

  D0.f = u32_to_f32(S0.u)

Notes

0.5ULP accuracy.

V_CVT_U32_F32                                                                                                   391

Convert from a single-precision float input to an unsigned 32-bit integer and store the result into a vector
register.

  D0.u = f32_to_u32(S0.f)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this

conversion iff CLAMP == 1.

V_CVT_I32_F32                                                                                                    392

Convert from a single-precision float input to a signed 32-bit integer and store the result into a vector register.

  D0.i = f32_to_i32(S0.f)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F16_F32                                                                                                    394

Convert from a single-precision float input to an FP16 float and store the result into a vector register.

  D0.f16 = f32_to_f16(S0.f)

Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on
output if specified based on DP denorm mode. Output rounding based on DP rounding mode.

V_CVT_F32_F16                                                                                                    395

Convert from an FP16 float input to a single-precision float and store the result into a vector register.

  D0.f = f16_to_f32(S0.f16)

Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm
mode.

V_CVT_NEAREST_I32_F32                                                                                            396

Convert from a single-precision float input to a signed 32-bit integer using round-to-nearest-integer semantics
(ignore the default rounding mode) and store the result into a vector register.

  D0.i = f32_to_i32(floor(S0.f + 0.5F))

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_FLOOR_I32_F32                                                                                                 397

Convert from a single-precision float input to a signed 32-bit integer using round-down semantics (ignore the
default rounding mode) and store the result into a vector register.

  D0.i = f32_to_i32(floor(S0.f))

Notes

1ULP accuracy, denormals are supported.

V_CVT_OFF_F32_I4                                                                                                    398

Convert from a signed 4-bit integer to a single-precision float using an offset table and store the result into a
vector register.

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
  D0.f = CVT_OFF_TABLE[S0.u[3 : 0]]

V_CVT_F32_F64                                                                                                    399

Convert from a double-precision float input to a single-precision float and store the result into a vector register.

  D0.f = f64_to_f32(S0.f64)

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_F64_F32                                                                                                    400

Convert from a single-precision float input to a double-precision float and store the result into a vector register.

  D0.f64 = f32_to_f64(S0.f)

Notes

0ULP accuracy, denormals are supported.

V_CVT_F32_UBYTE0                                                                                                 401

Convert an unsigned byte in byte 0 of the input to a single-precision float and store the result into a vector
register.

  D0.f = u32_to_f32(S0.u[7 : 0].u)

V_CVT_F32_UBYTE1                                                                                                 402

Convert an unsigned byte in byte 1 of the input to a single-precision float and store the result into a vector
register.

  D0.f = u32_to_f32(S0.u[15 : 8].u)

V_CVT_F32_UBYTE2                                                                                                 403

Convert an unsigned byte in byte 2 of the input to a single-precision float and store the result into a vector

register.

  D0.f = u32_to_f32(S0.u[23 : 16].u)

V_CVT_F32_UBYTE3                                                                                                 404

Convert an unsigned byte in byte 3 of the input to a single-precision float and store the result into a vector
register.

  D0.f = u32_to_f32(S0.u[31 : 24].u)

V_CVT_U32_F64                                                                                                    405

Convert from a double-precision float input to an unsigned 32-bit integer and store the result into a vector
register.

  D0.u = f64_to_u32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_U32                                                                                                    406

Convert from an unsigned 32-bit integer input to a double-precision float and store the result into a vector
register.

  D0.f64 = u32_to_f64(S0.u)

Notes

0ULP accuracy.

V_TRUNC_F64                                                                                                      407

Compute the integer part of a double-precision float input with round-toward-zero semantics and store the

result in floating point format into a vector register.

  D0.f64 = trunc(S0.f64)

V_CEIL_F64                                                                                                       408

Round the double-precision float input up to next integer and store the result in floating point format into a
vector register.

  D0.f64 = trunc(S0.f64);
  if ((S0.f64 > 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += 1.0
  endif

V_RNDNE_F64                                                                                                      409

Round the double-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f64 = floor(S0.f64 + 0.5);
  if (isEven(floor(S0.f64)) && (fract(S0.f64) == 0.5)) then
      D0.f64 -= 1.0
  endif

V_FLOOR_F64                                                                                                      410

Round the double-precision float input down to previous integer and store the result in floating point format
into a vector register.

  D0.f64 = trunc(S0.f64);
  if ((S0.f64 < 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += -1.0
  endif

V_PIPEFLUSH                                                                                                      411

Flush the VALU destination cache.

V_MOV_B16                                                                                                          412

Move data to a VGPR.

  D0.b16 = S0.b16

Notes

Floating-point modifiers are valid for this instruction if S0.u16 is a 16-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

V_FRACT_F32                                                                                                        416

Compute the fractional portion of a single-precision float input and store the result in floating point format into
a vector register.

  D0.f = S0.f + -floor(S0.f)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3f7fffff.

V_TRUNC_F32                                                                                                        417

Compute the integer part of a single-precision float input with round-toward-zero semantics and store the
result in floating point format into a vector register.

  D0.f = trunc(S0.f)

V_CEIL_F32                                                                                                         418

Round the single-precision float input up to next integer and store the result in floating point format into a
vector register.

  D0.f = trunc(S0.f);
  if ((S0.f > 0.0F) && (S0.f != D0.f)) then

        D0.f += 1.0F
  endif

V_RNDNE_F32                                                                                                        419

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f = floor(S0.f + 0.5F);
  if (isEven(64'F(floor(S0.f))) && (fract(S0.f) == 0.5F)) then
        D0.f -= 1.0F
  endif

V_FLOOR_F32                                                                                                        420

Round the single-precision float input down to previous integer and store the result in floating point format
into a vector register.

  D0.f = trunc(S0.f);
  if ((S0.f < 0.0F) && (S0.f != D0.f)) then
        D0.f += -1.0F
  endif

V_EXP_F32                                                                                                          421

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

  D0.f = pow(2.0F, S0.f)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_EXP_F32(0xff800000) => 0x00000000        // exp(-INF) = 0
  V_EXP_F32(0x80000000) => 0x3f800000        // exp(-0.0) = 1
  V_EXP_F32(0x7f800000) => 0x7f800000        // exp(+INF) = +INF

V_LOG_F32                                                                                                          423

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

  D0.f = log2(S0.f)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_LOG_F32(0xff800000) => 0xffc00000       // log(-INF) = NAN
  V_LOG_F32(0xbf800000) => 0xffc00000       // log(-1.0) = NAN
  V_LOG_F32(0x80000000) => 0xff800000       // log(-0.0) = -INF
  V_LOG_F32(0x00000000) => 0xff800000       // log(+0.0) = -INF
  V_LOG_F32(0x3f800000) => 0x00000000       // log(+1.0) = 0
  V_LOG_F32(0x7f800000) => 0x7f800000       // log(+INF) = +INF

V_RCP_F32                                                                                                          426

Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector
register.

  D0.f = 1.0F / S0.f

Notes

1ULP accuracy. Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA
operations. Denormals are flushed.

Functional examples:

  V_RCP_F32(0xff800000) => 0x80000000       // rcp(-INF) = -0
  V_RCP_F32(0xc0000000) => 0xbf000000       // rcp(-2.0) = -0.5
  V_RCP_F32(0x80000000) => 0xff800000       // rcp(-0.0) = -INF
  V_RCP_F32(0x00000000) => 0x7f800000       // rcp(+0.0) = +INF
  V_RCP_F32(0x7f800000) => 0x00000000       // rcp(+INF) = +0

V_RCP_IFLAG_F32                                                                                                    427

Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result
into a vector register. This opcode is intended for use as part of an integer division macro.

  D0.f = 1.0F / S0.f;
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

V_RSQ_F32                                                                                                        430

Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the
result into a vector register.

  D0.f = 1.0F / sqrt(S0.f)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_RSQ_F32(0xff800000) => 0xffc00000       // rsq(-INF) = NAN
  V_RSQ_F32(0x80000000) => 0xff800000       // rsq(-0.0) = -INF
  V_RSQ_F32(0x00000000) => 0x7f800000       // rsq(+0.0) = +INF
  V_RSQ_F32(0x40800000) => 0x3f000000       // rsq(+4.0) = +0.5
  V_RSQ_F32(0x7f800000) => 0x00000000       // rsq(+INF) = +0

V_RCP_F64                                                                                                        431

Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector
register.

  D0.f64 = 1.0 / S0.f64

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_RSQ_F64                                                                                                        433

Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the
result into a vector register.

  D0.f64 = 1.0 / sqrt(S0.f64)

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_SQRT_F32                                                                                                       435

Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector
register.

  D0.f = sqrt(S0.f)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_SQRT_F32(0xff800000) => 0xffc00000       // sqrt(-INF) = NAN
  V_SQRT_F32(0x80000000) => 0x80000000       // sqrt(-0.0) = -0
  V_SQRT_F32(0x00000000) => 0x00000000       // sqrt(+0.0) = +0
  V_SQRT_F32(0x40800000) => 0x40000000       // sqrt(+4.0) = +2.0
  V_SQRT_F32(0x7f800000) => 0x7f800000       // sqrt(+INF) = +INF

V_SQRT_F64                                                                                                       436

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector
register.

  D0.f64 = sqrt(S0.f64)

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_SIN_F32                                                                                                         437

Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f = 32'F(sin(64'F(S0.f) * 2.0 * PI))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_SIN_F32(0xff800000) => 0xffc00000       // sin(-INF) = NAN
  V_SIN_F32(0xff7fffff) => 0x00000000       // -MaxFloat, finite
  V_SIN_F32(0x80000000) => 0x80000000       // sin(-0.0) = -0
  V_SIN_F32(0x3e800000) => 0x3f800000       // sin(0.25) = 1
  V_SIN_F32(0x7f800000) => 0xffc00000       // sin(+INF) = NAN

V_COS_F32                                                                                                         438

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f = 32'F(cos(64'F(S0.f) * 2.0 * PI))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_COS_F32(0xff800000) => 0xffc00000       // cos(-INF) = NAN
  V_COS_F32(0xff7fffff) => 0x3f800000       // -MaxFloat, finite
  V_COS_F32(0x80000000) => 0x3f800000       // cos(-0.0) = 1
  V_COS_F32(0x3e800000) => 0x00000000       // cos(0.25) = 0

  V_COS_F32(0x7f800000) => 0xffc00000        // cos(+INF) = NAN

V_NOT_B32                                                                                                        439

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u = ~S0.u

Notes

Input and output modifiers not supported.

V_BFREV_B32                                                                                                      440

Reverse the order of bits in a vector input and store the result into a vector register.

  D0.u[31 : 0] = S0.u[0 : 31]

Notes

Input and output modifiers not supported.

V_CLZ_I32_U32                                                                                                    441

Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits.

  D0.i = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from MSB
        if S0.u[31 - i] == 1'1U then
            D0.i = i;
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

V_CTZ_I32_B32                                                                                                     442

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits in the input.

  D0.i = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from LSB
        if S0.u[i] == 1'1U then
            D0.i = i;
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

V_CLS_I32                                                                                                         443

Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a
vector register. Store -1 if all input bits are the same.

  D0.i = -1;
  // Set if all bits are the same
  for i in 1 : 31 do
        // Search from MSB
        if S0.i[31 - i] != S0.i[31] then
            D0.i = i;
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

V_FREXP_EXP_I32_F64                                                                                             444

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a
vector register.

  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.i = 0
  else
        D0.i = exponent(S0.f64) - 1023 + 1
  endif

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64,
which returns the significand. See the C library function frexp() for more information.

V_FREXP_MANT_F64                                                                                                445

Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double-
precision float into a vector register.

  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.f64 = S0.f64
  else
        D0.f64 = mantissa(S0.f64)
  endif

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I_F64, which returns integer exponent. See the C library function
frexp() for more information.

V_FRACT_F64                                                                                                        446

Compute the fractional portion of a double-precision float input and store the result in floating point format
into a vector register.

  D0.f64 = S0.f64 + -floor(S0.f64)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3fefffffffffffff.

V_FREXP_EXP_I32_F32                                                                                                447

Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector
register.

  if ((64'F(S0.f) == +INF) || (64'F(S0.f) == -INF) || isNAN(64'F(S0.f))) then
        D0.i = 0
  else
        D0.i = exponent(S0.f) - 127 + 1
  endif

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32,
which returns the significand. See the C library function frexp() for more information.

V_FREXP_MANT_F32                                                                                                   448

Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single-
precision float into a vector register.

  if ((64'F(S0.f) == +INF) || (64'F(S0.f) == -INF) || isNAN(64'F(S0.f))) then
        D0.f = S0.f

  else
        D0.f = mantissa(S0.f)
  endif

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I_F32, which returns integer exponent. See the C library function
frexp() for more information.

V_MOVRELD_B32                                                                                                      450

Move to a relative destination address.

  addr = DST.u;
  // Raw value from instruction
  addr += M0.u[31 : 0];
  VGPR[laneId][addr].b = S0.b

Notes

Example: The following instruction sequence performs the move v15 <= v7:

        s_mov_b32 m0, 10
        v_movreld_b32 v5, v7

V_MOVRELS_B32                                                                                                      451

Move from a relative source address.

  addr = SRC0.u;
  // Raw value from instruction
  addr += M0.u[31 : 0];
  D0.b = VGPR[laneId][addr].b

Notes

Example: The following instruction sequence performs the move v5 <= v17:

        s_mov_b32 m0, 10
        v_movrels_b32 v5, v7

V_MOVRELSD_B32                                                                                                 452

Move from a relative source address to a relative destination address.

  addrs = SRC0.u;
  // Raw value from instruction
  addrd = DST.u;
  // Raw value from instruction
  addrs += M0.u[31 : 0];
  addrd += M0.u[31 : 0];
  VGPR[laneId][addrd].b = VGPR[laneId][addrs].b

Notes

Example: The following instruction sequence performs the move v15 <= v17:

        s_mov_b32 m0, 10
        v_movrelsd_b32 v5, v7

V_MOVRELSD_2_B32                                                                                               456

Move from a relative source address to a relative destination address, with different relative offsets.

  addrs = SRC0.u;
  // Raw value from instruction
  addrd = DST.u;
  // Raw value from instruction
  addrs += M0.u[9 : 0].u;
  addrd += M0.u[25 : 16].u;
  VGPR[laneId][addrd].b = VGPR[laneId][addrs].b

Notes

Example: The following instruction sequence performs the move v25 <= v17:

        s_mov_b32 m0, ((20 << 16) | 10)
        v_movrelsd_2_b32 v5, v7

V_CVT_F16_U16                                                                                                  464

Convert from an unsigned 16-bit integer input to an FP16 float and store the result into a vector register.

  D0.f16 = u16_to_f16(S0.u16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_F16_I16                                                                                                   465

Convert from a signed 16-bit integer input to an FP16 float and store the result into a vector register.

  D0.f16 = i16_to_f16(S0.i16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_U16_F16                                                                                                   466

Convert from an FP16 float input to an unsigned 16-bit integer and store the result into a vector register.

  D0.u16 = f16_to_u16(S0.f16)

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_I16_F16                                                                                                   467

Convert from an FP16 float input to a signed 16-bit integer and store the result into a vector register.

  D0.i16 = f16_to_i16(S0.f16)

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_RCP_F16                                                                                                         468

Calculate the reciprocal of the half-precision float input using IEEE rules and store the result into a vector
register.

  D0.f16 = 16'1.0 / S0.f16

Notes

0.51ULP accuracy.

Functional examples:

  V_RCP_F16(0xfc00) => 0x8000          // rcp(-INF) = -0
  V_RCP_F16(0xc000) => 0xb800          // rcp(-2.0) = -0.5
  V_RCP_F16(0x8000) => 0xfc00          // rcp(-0.0) = -INF
  V_RCP_F16(0x0000) => 0x7c00          // rcp(+0.0) = +INF
  V_RCP_F16(0x7c00) => 0x0000          // rcp(+INF) = +0

V_SQRT_F16                                                                                                        469

Calculate the square root of the half-precision float input using IEEE rules and store the result into a vector
register.

  D0.f16 = sqrt(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_SQRT_F16(0xfc00) => 0xfe00         // sqrt(-INF) = NAN
  V_SQRT_F16(0x8000) => 0x8000         // sqrt(-0.0) = -0
  V_SQRT_F16(0x0000) => 0x0000         // sqrt(+0.0) = +0
  V_SQRT_F16(0x4400) => 0x4000         // sqrt(+4.0) = +2.0
  V_SQRT_F16(0x7c00) => 0x7c00         // sqrt(+INF) = +INF

V_RSQ_F16                                                                                                         470

Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result
into a vector register.

  D0.f16 = 16'1.0 / sqrt(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_RSQ_F16(0xfc00) => 0xfe00          // rsq(-INF) = NAN
  V_RSQ_F16(0x8000) => 0xfc00          // rsq(-0.0) = -INF
  V_RSQ_F16(0x0000) => 0x7c00          // rsq(+0.0) = +INF
  V_RSQ_F16(0x4400) => 0x3800          // rsq(+4.0) = +0.5
  V_RSQ_F16(0x7c00) => 0x0000          // rsq(+INF) = +0

V_LOG_F16                                                                                                        471

Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register.

  D0.f16 = log2(S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_LOG_F16(0xfc00) => 0xfe00          // log(-INF) = NAN
  V_LOG_F16(0xbc00) => 0xfe00          // log(-1.0) = NAN
  V_LOG_F16(0x8000) => 0xfc00          // log(-0.0) = -INF
  V_LOG_F16(0x0000) => 0xfc00          // log(+0.0) = -INF
  V_LOG_F16(0x3c00) => 0x0000          // log(+1.0) = 0
  V_LOG_F16(0x7c00) => 0x7c00          // log(+INF) = +INF

V_EXP_F16                                                                                                        472

Calculate 2 raised to the power of the half-precision float input and store the result into a vector register.

  D0.f16 = pow(16'2.0, S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_EXP_F16(0xfc00) => 0x0000          // exp(-INF) = 0
  V_EXP_F16(0x8000) => 0x3c00          // exp(-0.0) = 1
  V_EXP_F16(0x7c00) => 0x7c00          // exp(+INF) = +INF

V_FREXP_MANT_F16                                                                                                   473

Extract the binary significand, or mantissa, of an FP16 float input and store the result as an FP16 float into a
vector register.

  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.f16 = S0.f16
  else
        D0.f16 = mantissa(S0.f16)
  endif

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I_F16, which returns integer exponent. See the C library function
frexp() for more information.

V_FREXP_EXP_I16_F16                                                                                                474

Extract the exponent of an FP16 float input and store the result as a signed 16-bit integer into a vector register.

  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.i16 = 16'0
  else
        D0.i16 = 16'I(exponent(S0.f16) - 15 + 1)
  endif

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). See also V_FREXP_MANT_F16,
which returns the significand. See the C library function frexp() for more information.

V_FLOOR_F16                                                                                                        475

Round the half-precision float input down to previous integer and store the result in floating point format into
a vector register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 < 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += -16'1.0
  endif

V_CEIL_F16                                                                                                       476

Round the half-precision float input up to next integer and store the result in floating point format into a vector
register.

  D0.f16 = trunc(S0.f16);
  if ((S0.f16 > 16'0.0) && (S0.f16 != D0.f16)) then
      D0.f16 += 16'1.0
  endif

V_TRUNC_F16                                                                                                      477

Compute the integer part of an FP16 float input with round-toward-zero semantics and store the result in
floating point format into a vector register.

  D0.f16 = trunc(S0.f16)

V_RNDNE_F16                                                                                                      478

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
      D0.f16 -= 16'1.0
  endif

V_FRACT_F16                                                                                                      479

Compute the fractional portion of an FP16 float input and store the result in floating point format into a vector
register.

  D0.f16 = S0.f16 + -floor(S0.f16)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

V_SIN_F16                                                                                                       480

Calculate the trigonometric sine of an FP16 float value using IEEE rules and store the result into a vector
register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f16 = 16'F(sin(64'F(S0.f16) * 2.0 * PI))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_SIN_F16(0xfc00) => 0xfe00          // sin(-INF) = NAN
  V_SIN_F16(0xfbff) => 0x0000          // Most negative finite FP16
  V_SIN_F16(0x8000) => 0x8000          // sin(-0.0) = -0
  V_SIN_F16(0x3400) => 0x3c00          // sin(0.25) = 1
  V_SIN_F16(0x7bff) => 0x0000          // Most positive finite FP16
  V_SIN_F16(0x7c00) => 0xfe00          // sin(+INF) = NAN

V_COS_F16                                                                                                       481

Calculate the trigonometric cosine of an FP16 float value using IEEE rules and store the result into a vector
register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f16 = 16'F(cos(64'F(S0.f16) * 2.0 * PI))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_COS_F16(0xfc00) => 0xfe00          // cos(-INF) = NAN
  V_COS_F16(0xfbff) => 0x3c00          // Most negative finite FP16
  V_COS_F16(0x8000) => 0x3c00          // cos(-0.0) = 1
  V_COS_F16(0x3400) => 0x0000          // cos(0.25) = 0

  V_COS_F16(0x7bff) => 0x3c00          // Most positive finite FP16
  V_COS_F16(0x7c00) => 0xfe00          // cos(+INF) = NAN

V_SAT_PK_U8_I16                                                                                              482

Given two 16-bit unsigned integer inputs, saturate each input over an 8-bit unsigned range, pack the resulting
values into a 16-bit word and store the result into a vector register.

  SAT8 = lambda(n) (
        if n.i <= 0 then
            return 8'0U
        elsif n >= 16'I(0xff) then
            return 8'255U
        else
            return n[7 : 0].u8
        endif);
  D0.b16 = { SAT8(S0[31 : 16].i16), SAT8(S0[15 : 0].i16) }

Notes

Used for 4x16bit data packed as 4x8bit data.

V_CVT_NORM_I16_F16                                                                                           483

Convert from an FP16 float input to a signed normalized short and store the result into a vector register.

  D0.i16 = f16_to_snorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_CVT_NORM_U16_F16                                                                                           484

Convert from an FP16 float input to an unsigned normalized short and store the result into a vector register.

  D0.u16 = f16_to_unorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_NOT_B16                                                                                                          489

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u16 = ~S0.u16

Notes

Input and output modifiers not supported.

V_CVT_I32_I16                                                                                                      490

Convert from an 16-bit signed integer to a 32-bit signed integer, sign extending as needed.

  D0.i = 32'I(signext(S0.i16))

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CVT_U32_U16                                                                                                      491

Convert from an 16-bit unsigned integer to a 32-bit unsigned integer, zero extending as needed.

  D0 = { 16'0, S0.u16 }

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CNDMASK_B32                                                                                                      257

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

  D0.u = VCC.u64[laneId] ? S1.u : S0.u

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This

instruction is suitable for negating or taking the absolute value of a floating-point value.

V_ADD_F32                                                                                                         259

Add two floating point inputs and store the result into a vector register.

  D0.f = S0.f + S1.f

Notes

0.5ULP precision, denormals are supported.

V_SUB_F32                                                                                                         260

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f = S0.f - S1.f

Notes

0.5ULP precision, denormals are supported.

V_SUBREV_F32                                                                                                      261

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f = S1.f - S0.f

Notes

0.5ULP precision, denormals are supported.

V_FMAC_DX9_ZERO_F32                                                                                               262

Multiply two single-precision values and accumulate the result with the destination. Follows DX9 rules where
0.0 times anything produces 0.0 (this is not IEEE compliant).

  if ((64'F(S0.f) == 0.0) || (64'F(S1.f) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f = S2.f

  else
        D0.f = fma(S0.f, S1.f, D0.f)
  endif

V_MUL_DX9_ZERO_F32                                                                                                  263

Multiply two floating point inputs and store the result in a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

  if ((64'F(S0.f) == 0.0) || (64'F(S1.f) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f = 0.0F
  else
        D0.f = S0.f * S1.f
  endif

V_MUL_F32                                                                                                           264

Multiply two floating point inputs and store the result into a vector register.

  D0.f = S0.f * S1.f

Notes

0.5ULP precision, denormals are supported.

V_MUL_I32_I24                                                                                                       265

Multiply two signed 24 bit integer inputs and store the result as a signed 32 bit integer into a vector register.

  D0.i = 32'I(S0.i24) * 32'I(S1.i24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_I32_I24.

V_MUL_HI_I32_I24                                                                                                    266

Multiply two signed 24 bit integer inputs and store the high 32 bits of the result as a signed 32 bit integer into a

vector register.

  D0.i = 32'I((64'I(S0.i24) * 64'I(S1.i24)) >> 32U)

Notes

See also V_MUL_I32_I24.

V_MUL_U32_U24                                                                                                   267

Multiply two unsigned 24 bit integer inputs and store the result as a unsigned 32 bit integer into a vector
register.

  D0.u = 32'U(S0.u24) * 32'U(S1.u24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_U32_U24.

V_MUL_HI_U32_U24                                                                                                268

Multiply two unsigned 24 bit integer inputs and store the high 32 bits of the result as a unsigned 32 bit integer
into a vector register.

  D0.u = 32'U((64'U(S0.u24) * 64'U(S1.u24)) >> 32U)

Notes

See also V_MUL_U32_U24.

V_MIN_F32                                                                                                       271

Select the minimum of two floating point inputs and store the result into a vector register.

  LT_NEG_ZERO = lambda(a, b) (
        ((a < b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && sign(a) && !sign(b))));
  // Version of comparison where -0.0 < +0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f)) then
            D0.f = 32'F(cvtToQuietNAN(64'F(S0.f)))
        elsif isSignalNAN(64'F(S1.f)) then

            D0.f = 32'F(cvtToQuietNAN(64'F(S1.f)))
        elsif isQuietNAN(64'F(S1.f)) then
            D0.f = S0.f
        elsif isQuietNAN(64'F(S0.f)) then
            D0.f = S1.f
        elsif LT_NEG_ZERO(S0.f, S1.f) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f = S0.f
        else
            D0.f = S1.f
        endif
  else
        if isNAN(64'F(S1.f)) then
            D0.f = S0.f
        elsif isNAN(64'F(S0.f)) then
            D0.f = S1.f
        elsif LT_NEG_ZERO(S0.f, S1.f) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f = S0.f
        else
            D0.f = S1.f
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

V_MAX_F32                                                                                               272

Select the maximum of two floating point inputs and store the result into a vector register.

  GT_NEG_ZERO = lambda(a, b) (
        ((a > b) || ((64'F(abs(a)) == 0.0) && (64'F(abs(b)) == 0.0) && !sign(a) && sign(b))));
  // Version of comparison where +0.0 > -0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(64'F(S0.f)) then
            D0.f = 32'F(cvtToQuietNAN(64'F(S0.f)))
        elsif isSignalNAN(64'F(S1.f)) then
            D0.f = 32'F(cvtToQuietNAN(64'F(S1.f)))
        elsif isQuietNAN(64'F(S1.f)) then
            D0.f = S0.f
        elsif isQuietNAN(64'F(S0.f)) then
            D0.f = S1.f
        elsif GT_NEG_ZERO(S0.f, S1.f) then
            // NOTE: +0>-0 is TRUE in this comparison

            D0.f = S0.f
        else
            D0.f = S1.f
        endif
  else
        if isNAN(64'F(S1.f)) then
            D0.f = S0.f
        elsif isNAN(64'F(S0.f)) then
            D0.f = S1.f
        elsif GT_NEG_ZERO(S0.f, S1.f) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f = S0.f
        else
            D0.f = S1.f
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

V_MIN_I32                                                                                               273

Select the minimum of two signed integers and store the selected value into a vector register.

  D0.i = S0.i < S1.i ? S0.i : S1.i

V_MAX_I32                                                                                               274

Select the maximum of two signed integers and store the selected value into a vector register.

  D0.i = S0.i >= S1.i ? S0.i : S1.i

V_MIN_U32                                                                                               275

Select the minimum of two unsigned integers and store the selected value into a vector register.

  D0.u = S0.u < S1.u ? S0.u : S1.u

V_MAX_U32                                                                                                          276

Select the maximum of two unsigned integers and store the selected value into a vector register.

  D0.u = S0.u >= S1.u ? S0.u : S1.u

V_LSHLREV_B32                                                                                                      280

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u = (S1.u << S0[4 : 0].u)

V_LSHRREV_B32                                                                                                      281

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u = (S1.u >> S0[4 : 0].u)

V_ASHRREV_I32                                                                                                      282

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i = (S1.i >> S0[4 : 0].u)

V_AND_B32                                                                                                          283

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u = (S0.u & S1.u)

Notes

Input and output modifiers not supported.

V_OR_B32                                                                                                        284

Calculate bitwise OR on two vector inputs and store the result into a vector register.

  D0.u = (S0.u | S1.u)

Notes

Input and output modifiers not supported.

V_XOR_B32                                                                                                       285

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

  D0.u = (S0.u ^ S1.u)

Notes

Input and output modifiers not supported.

V_XNOR_B32                                                                                                      286

Calculate bitwise XNOR on two vector inputs and store the result into a vector register.

  D0.u = ~(S0.u ^ S1.u)

Notes

Input and output modifiers not supported.

V_ADD_CO_CI_U32                                                                                                 288

Add two unsigned inputs and a bit from a carry-in mask, store the result into a vector register and store the
carry-out mask into a scalar register.

  tmp = 64'U(S0.u) + 64'U(S1.u) + VCC.u64[laneId].u64;

  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_SUB_CO_CI_U32                                                                                                289

Subtract the second unsigned input from the first input, subtract a bit from the carry-in mask, store the result
into a vector register and store the carry-out mask to a scalar register.

  tmp = S0.u - S1.u - VCC.u64[laneId].u;
  VCC.u64[laneId] = 64'U(S1.u) + VCC.u64[laneId].u64 > 64'U(S0.u) ? 1'1U : 1'0U;
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_CO_CI_U32                                                                                             290

Subtract the first unsigned input from the second input, subtract a bit from the carry-in mask, store the result
into a vector register and store the carry-out mask to a scalar register.

  tmp = S1.u - S0.u - VCC.u64[laneId].u;
  VCC.u64[laneId] = 64'U(S1.u) + VCC.u64[laneId].u64 > 64'U(S0.u) ? 1'1U : 1'0U;
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

V_ADD_NC_U32                                                                                                   293

Add two unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.u = S0.u + S1.u

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUB_NC_U32                                                                                                    294

Subtract the second unsigned input from the first input and store the result into a vector register. No carry-in
or carry-out support.

  D0.u = S0.u - S1.u

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_NC_U32                                                                                                 295

Subtract the first unsigned input from the second input and store the result into a vector register. No carry-in or
carry-out support.

  D0.u = S1.u - S0.u

Notes

Supports saturation (unsigned 32-bit integer domain).

V_FMAC_F32                                                                                                      299

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply-
add.

  D0.f = fma(S0.f, S1.f, D0.f)

V_CVT_PK_RTZ_F16_F32                                                                                            303

Convert two single-precision float inputs into a packed FP16 result with round toward zero semantics (ignore
the current rounding mode), and store the result into a vector register.

  D0[15 : 0].f16 = f32_to_f16(S0.f);
  D0[31 : 16].f16 = f32_to_f16(S1.f);
  // Round-toward-zero regardless of current round mode setting in hardware.

Notes

This opcode is intended for use with 16-bit compressed exports. See V_CVT_F16_F32 for a version that respects
the current rounding mode.

V_ADD_F16                                                                                                         306

Add two floating point inputs and store the result into a vector register.

  D0.f16 = S0.f16 + S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_SUB_F16                                                                                                         307

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f16 = S0.f16 - S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_SUBREV_F16                                                                                                      308

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f16 = S1.f16 - S0.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_MUL_F16                                                                                                    309

Multiply two floating point inputs and store the result into a vector register.

  D0.f16 = S0.f16 * S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_FMAC_F16                                                                                                   310

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply-
add.

  D0.f16 = fma(S0.f16, S1.f16, D0.f16)

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_MAX_F16                                                                                                    313

Select the maximum of two floating point inputs and store the result into a vector register.

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

V_MIN_F16                                                                                               314

Select the minimum of two floating point inputs and store the result into a vector register.

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

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

Denorm flushing for this operation is effectively controlled by the input denorm mode control: If input
denorm mode is disabling denorm, the internal result of a min/max operation cannot be a denorm value, so
output denorm mode is irrelevant. If input denorm mode is enabling denorm, the internal min/max result can
be a denorm and this operation outputs as a denorm regardless of output denorm mode.

V_LDEXP_F16                                                                                                     315

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register. Compare with the ldexp() function in C.

  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))

V_FMA_DX9_ZERO_F32                                                                                              521

Multiply and add single-precision values. Follows DX9 rules where 0.0 times anything produces 0.0 (this is not
IEEE compliant).

  if ((64'F(S0.f) == 0.0) || (64'F(S1.f) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f = S2.f
  else
        D0.f = fma(S0.f, S1.f, S2.f)
  endif

V_MAD_I32_I24                                                                                                   522

Multiply two signed 24-bit integers, add a signed 32-bit integer and store the result as a signed 32-bit integer.

  D0.i = 32'I(S0.i24) * 32'I(S1.i24) + S2.i

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier.

V_MAD_U32_U24                                                                                                   523

Multiply two unsigned 24-bit integers, add an unsigned 32-bit integer and store the result as an unsigned 32-bit
integer.

  D0.u = 32'U(S0.u24) * 32'U(S1.u24) + S2.u

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier.

V_CUBEID_F32                                                                                                    524

Cubemap Face ID determination. Result is a floating point face ID.

  // Set D0.f = cubemap face ID ({0.0, 1.0, ..., 5.0}).
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f) >= abs(S0.f)) && (abs(S2.f) >= abs(S1.f))) then
        if S2.f < 0.0F then
            D0.f = 5.0F
        else
            D0.f = 4.0F
        endif
  elsif abs(S1.f) >= abs(S0.f) then
        if S1.f < 0.0F then
            D0.f = 3.0F
        else
            D0.f = 2.0F
        endif
  else
        if S0.f < 0.0F then
            D0.f = 1.0F
        else
            D0.f = 0.0F
        endif
  endif

V_CUBESC_F32                                                                                                    525

Cubemap S coordinate.

  // D0.f = cubemap S coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f) >= abs(S0.f)) && (abs(S2.f) >= abs(S1.f))) then
      if S2.f < 0.0F then
           D0.f = -S0.f
      else
           D0.f = S0.f
      endif
  elsif abs(S1.f) >= abs(S0.f) then
      D0.f = S0.f
  else
      if S0.f < 0.0F then
           D0.f = S2.f
      else
           D0.f = -S2.f
      endif
  endif

V_CUBETC_F32                                                            526

Cubemap T coordinate.

  // D0.f = cubemap T coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f) >= abs(S0.f)) && (abs(S2.f) >= abs(S1.f))) then
      D0.f = -S1.f
  elsif abs(S1.f) >= abs(S0.f) then
      if S1.f < 0.0F then
           D0.f = -S2.f
      else
           D0.f = S2.f
      endif
  else
      D0.f = -S1.f
  endif

V_CUBEMA_F32                                                            527

Determine cubemap major axis.

  // D0.f = 2.0 * cubemap major axis.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f) >= abs(S0.f)) && (abs(S2.f) >= abs(S1.f))) then
      D0.f = S2.f * 2.0F
  elsif abs(S1.f) >= abs(S0.f) then
      D0.f = S1.f * 2.0F
  else
      D0.f = S0.f * 2.0F
  endif

V_BFE_U32                                                                                                       528

Bitfield extract. Extract unsigned bitfield from first operand using field offset in second operand and field size
in third operand.

  D0.u = ((S0.u >> S1.u[4 : 0].u) & 32'U((1 << S2.u[4 : 0].u) - 1))

V_BFE_I32                                                                                                       529

Bitfield extract. Extract signed bitfield from first operand using field offset in second operand and field size in
third operand.

  tmp = ((S0.i >> S1.u[4 : 0].u) & ((1 << S2.u[4 : 0].u) - 1));
  D0.i = 32'I(signextFromBit(tmp.i, S2.i[4 : 0].i))

V_BFI_B32                                                                                                       530

Bitfield insert. Using a bitmask from the first operand, merge bitfield in second operand with packed value in
third operand.

  D0.u = ((S0.u & S1.u) | (~S0.u & S2.u))

V_FMA_F32                                                                                                       531

Fused single precision multiply add.

  D0.f = fma(S0.f, S1.f, S2.f)

Notes

0.5ULP accuracy, denormals are supported.

V_FMA_F64                                                                                                   532

Fused double precision multiply add.

  D0.f64 = fma(S0.f64, S1.f64, S2.f64)

Notes

0.5ULP precision, denormals are supported.

V_LERP_U8                                                                                                   533

Unsigned 8-bit pixel average on packed unsigned bytes (linear interpolation).

Each byte in S2 acts as a round mode; if the LSB is set then 0.5 rounds up, otherwise 0.5 truncates.

  D0.u = 32'U((S0.u[31 : 24] + S1.u[31 : 24] + S2.u[24].u8) >> 1U << 24U);
  D0.u += 32'U((S0.u[23 : 16] + S1.u[23 : 16] + S2.u[16].u8) >> 1U << 16U);
  D0.u += 32'U((S0.u[15 : 8] + S1.u[15 : 8] + S2.u[8].u8) >> 1U << 8U);
  D0.u += 32'U((S0.u[7 : 0] + S1.u[7 : 0] + S2.u[0].u8) >> 1U)

V_ALIGNBIT_B32                                                                                              534

Align a value to the specified bit position.

  D0.u = 32'U(({ S0.u, S1.u } >> S2.u[4 : 0].u) & 0xffffffffLL)

Notes

                S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

V_ALIGNBYTE_B32                                                                                             535

Align a value to the specified byte position.

  D0.u = 32'U(({ S0.u, S1.u } >> (S2.u[1 : 0].u * 8U)) & 0xffffffffLL)

Notes

                S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

V_MULLIT_F32                                                                                            536

Multiply for lighting. Specific rules apply: 0.0 * x = 0.0; specific INF, NAN, overflow rules.

  if ((S1.f == -MAX_FLOAT_F32) || (64'F(S1.f) == -INF) || isNAN(64'F(S1.f)) || (S2.f <= 0.0F) ||
  isNAN(64'F(S2.f))) then
        D0.f = -MAX_FLOAT_F32
  else
        D0.f = S0.f * S1.f
  endif

Notes

V_MIN3_F32                                                                                              537

Return minimum single-precision value of three inputs.

  D0.f = v_min_f32(v_min_f32(S0.f, S1.f), S2.f)

V_MIN3_I32                                                                                              538

Return minimum signed integer value of three inputs.

  D0.i = v_min_i32(v_min_i32(S0.i, S1.i), S2.i)

V_MIN3_U32                                                                                              539

Return minimum unsigned integer value of three inputs.

  D0.u = v_min_u32(v_min_u32(S0.u, S1.u), S2.u)

V_MAX3_F32                                                                     540

Return maximum single precision value of three inputs.

  D0.f = v_max_f32(v_max_f32(S0.f, S1.f), S2.f)

V_MAX3_I32                                                                     541

Return maximum signed integer value of three inputs.

  D0.i = v_max_i32(v_max_i32(S0.i, S1.i), S2.i)

V_MAX3_U32                                                                     542

Return maximum unsigned integer value of three inputs.

  D0.u = v_max_u32(v_max_u32(S0.u, S1.u), S2.u)

V_MED3_F32                                                                     543

Return median single precision value of three inputs.

  if (isNAN(64'F(S0.f)) || isNAN(64'F(S1.f)) || isNAN(64'F(S2.f))) then
      D0.f = v_min3_f32(S0.f, S1.f, S2.f)
  elsif v_max3_f32(S0.f, S1.f, S2.f) == S0.f then
      D0.f = v_max_f32(S1.f, S2.f)
  elsif v_max3_f32(S0.f, S1.f, S2.f) == S1.f then
      D0.f = v_max_f32(S0.f, S2.f)
  else
      D0.f = v_max_f32(S0.f, S1.f)
  endif

V_MED3_I32                                                                     544

Return median signed integer value of three inputs.

  if v_max3_i32(S0.i, S1.i, S2.i) == S0.i then
      D0.i = v_max_i32(S1.i, S2.i)
  elsif v_max3_i32(S0.i, S1.i, S2.i) == S1.i then
      D0.i = v_max_i32(S0.i, S2.i)
  else
      D0.i = v_max_i32(S0.i, S1.i)
  endif

V_MED3_U32                                                                                                 545

Return median unsigned integer value of three inputs.

  if v_max3_u32(S0.u, S1.u, S2.u) == S0.u then
      D0.u = v_max_u32(S1.u, S2.u)
  elsif v_max3_u32(S0.u, S1.u, S2.u) == S1.u then
      D0.u = v_max_u32(S0.u, S2.u)
  else
      D0.u = v_max_u32(S0.u, S1.u)
  endif

V_SAD_U8                                                                                                   546

Sum of absolute differences with accumulation, overflow into upper bits is allowed.

  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u = S2.u;
  D0.u += 32'U(ABSDIFF(S0.u[31 : 24], S1.u[31 : 24]));
  D0.u += 32'U(ABSDIFF(S0.u[23 : 16], S1.u[23 : 16]));
  D0.u += 32'U(ABSDIFF(S0.u[15 : 8], S1.u[15 : 8]));
  D0.u += 32'U(ABSDIFF(S0.u[7 : 0], S1.u[7 : 0]))

V_SAD_HI_U8                                                                                                547

Sum of absolute differences with accumulation, accumulate from the higher-order bits of the third source
operand.

  D0.u = (32'U(v_sad_u8(S0, S1, 0U)) << 16U) + S2.u

V_SAD_U16                                                                                                      548

Short SAD with accumulation.

  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u = S2.u;
  D0.u += ABSDIFF(S0[31 : 16].u16, S1[31 : 16].u16);
  D0.u += ABSDIFF(S0[15 : 0].u16, S1[15 : 0].u16)

V_SAD_U32                                                                                                      549

Dword SAD with accumulation.

  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u = ABSDIFF(S0.u, S1.u) + S2.u

V_CVT_PK_U8_F32                                                                                                550

Packed float to byte conversion.

Convert floating point value S0 to 8-bit unsigned integer and pack the result into byte S1 of dword S2.

  D0.u = (S2.u & 32'U(~(0xff << (S1.u[1 : 0].u * 8U))));
  D0.u = (D0.u | ((32'U(f32_to_u8(S0.f)) & 255U) << (S1.u[1 : 0].u * 8U)))

V_DIV_FIXUP_F32                                                                                                551

Single precision division fixup.

S0 = Quotient, S1 = Denominator, S2 = Numerator.

Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case
numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by
zero exceptions caused by the division.

  sign_out = (sign(S1.f) ^ sign(S2.f));
  if isNAN(64'F(S2.f)) then
      D0.f = 32'F(cvtToQuietNAN(64'F(S2.f)))
  elsif isNAN(64'F(S1.f)) then

      D0.f = 32'F(cvtToQuietNAN(64'F(S1.f)))
  elsif ((64'F(S1.f) == 0.0) && (64'F(S2.f) == 0.0)) then
      // 0/0
      D0.f = 32'F(0xffc00000)
  elsif ((64'F(abs(S1.f)) == +INF) && (64'F(abs(S2.f)) == +INF)) then
      // inf/inf
      D0.f = 32'F(0xffc00000)
  elsif ((64'F(S1.f) == 0.0) || (64'F(abs(S2.f)) == +INF)) then
      // x/0, or inf/y
      D0.f = sign_out ? -INF.f : +INF.f
  elsif ((64'F(abs(S1.f)) == +INF) || (64'F(S2.f) == 0.0)) then
      // x/inf, 0/y
      D0.f = sign_out ? -0.0F : 0.0F
  elsif exponent(S2.f) - exponent(S1.f) < -150 then
      D0.f = sign_out ? -UNDERFLOW_F32 : UNDERFLOW_F32
  elsif exponent(S1.f) == 255 then
      D0.f = sign_out ? -OVERFLOW_F32 : OVERFLOW_F32
  else
      D0.f = sign_out ? -abs(S0.f) : abs(S0.f)
  endif

V_DIV_FIXUP_F64                                                                                             552

Double precision division fixup.

S0 = Quotient, S1 = Denominator, S2 = Numerator.

Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case
numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by
zero exceptions caused by the division.

  sign_out = (sign(S1.f64) ^ sign(S2.f64));
  if isNAN(S2.f64) then
      D0.f64 = cvtToQuietNAN(S2.f64)
  elsif isNAN(S1.f64) then
      D0.f64 = cvtToQuietNAN(S1.f64)
  elsif ((S1.f64 == 0.0) && (S2.f64 == 0.0)) then
      // 0/0
      D0.f64 = 64'F(0xfff8000000000000LL)
  elsif ((abs(S1.f64) == +INF) && (abs(S2.f64) == +INF)) then
      // inf/inf
      D0.f64 = 64'F(0xfff8000000000000LL)
  elsif ((S1.f64 == 0.0) || (abs(S2.f64) == +INF)) then
      // x/0, or inf/y
      D0.f64 = sign_out ? -INF : +INF
  elsif ((abs(S1.f64) == +INF) || (S2.f64 == 0.0)) then
      // x/inf, 0/y
      D0.f64 = sign_out ? -0.0 : 0.0
  elsif exponent(S2.f64) - exponent(S1.f64) < -1075 then
      D0.f64 = sign_out ? -UNDERFLOW_F64 : UNDERFLOW_F64
  elsif exponent(S1.f64) == 2047 then
      D0.f64 = sign_out ? -OVERFLOW_F64 : OVERFLOW_F64
  else

        D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)
  endif

V_DIV_FMAS_F32                                                                                          567

Single precision FMA with fused scale.

This opcode performs a standard Fused Multiply-Add operation and conditionally scales the resulting exponent
if VCC is set.

  if VCC.u64[laneId] then
        D0.f = 2.0F ** 32 * fma(S0.f, S1.f, S2.f)
  else
        D0.f = fma(S0.f, S1.f, S2.f)
  endif

Notes

Input denormals are not flushed, but output flushing is allowed.

V_DIV_FMAS_F64                                                                                          568

Double precision FMA with fused scale.

This opcode performs a standard Fused Multiply-Add operation and conditionally scales the resulting exponent
if VCC is set.

  if VCC.u64[laneId] then
        D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)
  else
        D0.f64 = fma(S0.f64, S1.f64, S2.f64)
  endif

Notes

Input denormals are not flushed, but output flushing is allowed.

V_MSAD_U8                                                                                               569

Masked sum of absolute differences with accumulation, overflow into upper bits is allowed.

Components where the reference value in S1 is zero are not included in the sum.

  ABSDIFF = lambda(x, y) (

      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u = S2.u;
  D0.u += S1.u[31 : 24] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u[31 : 24], S1.u[31 : 24]));
  D0.u += S1.u[23 : 16] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u[23 : 16], S1.u[23 : 16]));
  D0.u += S1.u[15 : 8] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u[15 : 8], S1.u[15 : 8]));
  D0.u += S1.u[7 : 0] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u[7 : 0], S1.u[7 : 0]))

V_QSAD_PK_U16_U8                                                                                     570

Quad-byte SAD with 16-bit packed accumulation.

  D0[63 : 48] = 16'B(v_sad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u));
  D0[47 : 32] = 16'B(v_sad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u));
  D0[31 : 16] = 16'B(v_sad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u));
  D0[15 : 0] = 16'B(v_sad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u))

V_MQSAD_PK_U16_U8                                                                                    571

Quad-byte masked SAD with 16-bit packed accumulation.

  D0[63 : 48] = 16'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u));
  D0[47 : 32] = 16'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u));
  D0[31 : 16] = 16'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u));
  D0[15 : 0] = 16'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u))

V_MQSAD_U32_U8                                                                                       573

Quad-byte masked SAD with 32-bit packed accumulation.

  D0[127 : 96] = 32'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[127 : 96].u));
  D0[95 : 64] = 32'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[95 : 64].u));
  D0[63 : 32] = 32'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[63 : 32].u));
  D0[31 : 0] = 32'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[31 : 0].u))

V_XOR3_B32                                                                                           576

Calculate the bitwise XOR of three vector inputs and store the result into a vector register.

  D0.u = (S0.u ^ S1.u ^ S2.u)

Notes

Input and output modifiers not supported.

V_MAD_U16                                                                                                       577

Multiply and add three unsigned short values.

  D0.u16 = S0.u16 * S1.u16 + S2.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_PERM_B32                                                                                                      580

Permute a 64-bit value constructed from two vector inputs using a per-lane selector and store the result into a
vector register.

  BYTE_PERMUTE = lambda(data, sel) (
        declare in : 8'B[8];
        for i in 0 : 7 do
            in[i] = data[i * 8 + 7 : i * 8].b8
        endfor;
        if sel.u >= 13U then
            return 8'0xff
        elsif sel.u == 12U then
            return 8'0x0
        elsif sel.u == 11U then
            return in[7][7].b8 * 8'0xff
        elsif sel.u == 10U then
            return in[5][7].b8 * 8'0xff
        elsif sel.u == 9U then
            return in[3][7].b8 * 8'0xff
        elsif sel.u == 8U then
            return in[1][7].b8 * 8'0xff
        else
            return in[sel]
        endif);
  D0[31 : 24] = BYTE_PERMUTE({ S0.u, S1.u }, S2.u[31 : 24]);
  D0[23 : 16] = BYTE_PERMUTE({ S0.u, S1.u }, S2.u[23 : 16]);
  D0[15 : 8] = BYTE_PERMUTE({ S0.u, S1.u }, S2.u[15 : 8]);
  D0[7 : 0] = BYTE_PERMUTE({ S0.u, S1.u }, S2.u[7 : 0])

Notes

Selects 8 through 11 are useful in modeling sign extension of a smaller-precision signed integer to a larger-
precision result.

Note the MSBs of the 64-bit value being selected are stored in S0. This is counterintuitive for a little-endian
architecture.

V_XAD_U32                                                                                                          581

Calculate bitwise XOR of the first two vector inputs, then add the third vector input to the intermediate result,
then store the result into a vector register.

  D0.u = (S0.u ^ S1.u) + S2.u

Notes

No carryin/carryout and no saturation. This opcode is designed to help accelerate the SHA256 hash algorithm.

V_LSHL_ADD_U32                                                                                                     582

Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input
to the intermediate result, then store the final result into a vector register.

  D0.u = (S0.u << S1.u[4 : 0].u) + S2.u

V_ADD_LSHL_U32                                                                                                     583

Add the first two integer inputs, then given a shift count in the third input, calculate the logical shift left of the
intermediate result, then store the final result into a vector register.

  D0.u = ((S0.u + S1.u) << S2.u[4 : 0].u)

V_FMA_F16                                                                                                          584

Fused half precision multiply add.

  D0.f16 = fma(S0.f16, S1.f16, S2.f16)

Notes

0.5ULP accuracy, denormals are supported.

V_MIN3_F16                                                     585

Return minimum FP16 value of three inputs.

  D0.f16 = v_min_f16(v_min_f16(S0.f16, S1.f16), S2.f16)

V_MIN3_I16                                                     586

Return minimum signed short value of three inputs.

  D0.i16 = v_min_i16(v_min_i16(S0.i16, S1.i16), S2.i16)

V_MIN3_U16                                                     587

Return minimum unsigned short value of three inputs.

  D0.u16 = v_min_u16(v_min_u16(S0.u16, S1.u16), S2.u16)

V_MAX3_F16                                                     588

Return maximum FP16 value of three inputs.

  D0.f16 = v_max_f16(v_max_f16(S0.f16, S1.f16), S2.f16)

V_MAX3_I16                                                     589

Return maximum signed short value of three inputs.

  D0.i16 = v_max_i16(v_max_i16(S0.i16, S1.i16), S2.i16)

V_MAX3_U16                                                     590

Return maximum unsigned short value of three inputs.

  D0.u16 = v_max_u16(v_max_u16(S0.u16, S1.u16), S2.u16)

V_MED3_F16                                                                           591

Return median FP16 value of three inputs.

  if (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)) || isNAN(64'F(S2.f16))) then
      D0.f16 = v_min3_f16(S0.f16, S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S0.f16 then
      D0.f16 = v_max_f16(S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S1.f16 then
      D0.f16 = v_max_f16(S0.f16, S2.f16)
  else
      D0.f16 = v_max_f16(S0.f16, S1.f16)
  endif

V_MED3_I16                                                                           592

Return median signed short value of three inputs.

  if v_max3_i16(S0.i16, S1.i16, S2.i16) == S0.i16 then
      D0.i16 = v_max_i16(S1.i16, S2.i16)
  elsif v_max3_i16(S0.i16, S1.i16, S2.i16) == S1.i16 then
      D0.i16 = v_max_i16(S0.i16, S2.i16)
  else
      D0.i16 = v_max_i16(S0.i16, S1.i16)
  endif

V_MED3_U16                                                                           593

Return median unsigned short value of three inputs.

  if v_max3_u16(S0.u16, S1.u16, S2.u16) == S0.u16 then
      D0.u16 = v_max_u16(S1.u16, S2.u16)
  elsif v_max3_u16(S0.u16, S1.u16, S2.u16) == S1.u16 then
      D0.u16 = v_max_u16(S0.u16, S2.u16)
  else
      D0.u16 = v_max_u16(S0.u16, S1.u16)
  endif

V_MAD_I16                                                                            595

Multiply and add three signed short values.

  D0.i16 = S0.i16 * S1.i16 + S2.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_DIV_FIXUP_F16                                                                                             596

Half precision division fixup.

S0 = Quotient, S1 = Denominator, S2 = Numerator.

Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case
numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by
zero exceptions caused by the division.

  sign_out = (sign(S1.f16) ^ sign(S2.f16));
  if isNAN(64'F(S2.f16)) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S2.f16)))
  elsif isNAN(64'F(S1.f16)) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
  elsif ((64'F(S1.f16) == 0.0) && (64'F(S2.f16) == 0.0)) then
        // 0/0
        D0.f16 = 16'F(0xfe00)
  elsif ((64'F(abs(S1.f16)) == +INF) && (64'F(abs(S2.f16)) == +INF)) then
        // inf/inf
        D0.f16 = 16'F(0xfe00)
  elsif ((64'F(S1.f16) == 0.0) || (64'F(abs(S2.f16)) == +INF)) then
        // x/0, or inf/y
        D0.f16 = sign_out ? -INF.f16 : +INF.f16
  elsif ((64'F(abs(S1.f16)) == +INF) || (64'F(S2.f16) == 0.0)) then
        // x/inf, 0/y
        D0.f16 = sign_out ? -16'0.0 : 16'0.0
  else
        D0.f16 = sign_out ? -abs(S0.f16) : abs(S0.f16)
  endif

V_ADD3_U32                                                                                                  597

Add three unsigned integers.

  D0.u = S0.u + S1.u + S2.u

V_LSHL_OR_B32                                                                                                      598

Given a shift count in the second input, calculate the logical shift left of the first input, then calculate the
bitwise OR of the intermediate result and the third input, then store the final result into a vector register.

  D0.u = ((S0.u << S1.u[4 : 0].u) | S2.u)

V_AND_OR_B32                                                                                                       599

Calculate bitwise AND on the first two vector inputs, then compute the bitwise OR of the intermediate result
and the third vector input, then store the result into a vector register.

  D0.u = ((S0.u & S1.u) | S2.u)

Notes

Input and output modifiers not supported.

V_OR3_B32                                                                                                          600

Calculate the bitwise OR of three vector inputs and store the result into a vector register.

  D0.u = (S0.u | S1.u | S2.u)

Notes

Input and output modifiers not supported.

V_MAD_U32_U16                                                                                                      601

Multiply and add unsigned values.

  D0.u = 32'U(S0.u16) * 32'U(S1.u16) + S2.u

V_MAD_I32_I16                                                                                                      602

Multiply and add signed values.

  D0.i = 32'I(S0.i16) * 32'I(S1.i16) + S2.i

V_PERMLANE16_B32                                                                                          603

Perform arbitrary gather-style operation within a row (16 contiguous lanes).

The first source must be a VGPR and the second and third sources must be scalar values; the second and third
source are combined into a single 64-bit value representing lane selects used to swizzle within each row.

OPSEL is not used in its typical manner for this instruction. For this instruction OPSEL[0] is overloaded to
represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit.
The remaining OPSEL bits are reserved for this instruction.

Compare with V_PERMLANEX16_B32.

  declare tmp : 32'B[64];
  lanesel = { S2.u, S1.u };
  // Concatenate lane select bits
  for i in 0 : WAVE32 ? 31 : 63 do
        // Copy original S0 in case D==S0
        tmp[i] = VGPR[i][SRC0.u]
  endfor;
  for row in 0 : WAVE32 ? 1 : 3 do
        // Implement arbitrary swizzle within each row
        for i in 0 : 15 do
            if EXEC[row * 16 + i].u1 then
                 VGPR[row * 16 + i][VDST.u] = tmp[64'B(row * 16) + lanesel[i * 4 + 3 : i * 4]]
            endif
        endfor
  endfor

Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

Example implementing a rotation within each row:

  v_mov_b32 s0, 0x87654321;
  v_mov_b32 s1, 0x0fedcba9;
  v_permlane16_b32 v1, v0, s0, s1;
  // ROW 0:
  // v1.lane[0] <- v0.lane[1]
  // v1.lane[1] <- v0.lane[2]
  // ...
  // v1.lane[14] <- v0.lane[15]
  // v1.lane[15] <- v0.lane[0]
  //
  // ROW 1:
  // v1.lane[16] <- v0.lane[17]
  // v1.lane[17] <- v0.lane[18]

  // ...
  // v1.lane[30] <- v0.lane[31]
  // v1.lane[31] <- v0.lane[16]

V_PERMLANEX16_B32                                                                                          604

Perform arbitrary gather-style operation across two rows (each row is 16 contiguous lanes).

The first source must be a VGPR and the second and third sources must be scalar values; the second and third
source are combined into a single 64-bit value representing lane selects used to swizzle within each row.

OPSEL is not used in its typical manner for this instruction. For this instruction OPSEL[0] is overloaded to
represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit.
The remaining OPSEL bits are reserved for this instruction.

Compare with V_PERMLANE16_B32.

  declare tmp : 32'B[64];
  lanesel = { S2.u, S1.u };
  // Concatenate lane select bits
  for i in 0 : WAVE32 ? 31 : 63 do
        // Copy original S0 in case D==S0
        tmp[i] = VGPR[i][SRC0.u]
  endfor;
  for row in 0 : WAVE32 ? 1 : 3 do
        // Implement arbitrary swizzle across two rows
        altrow = { row[1], ~row[0] };
        // 1<->0, 3<->2
        for i in 0 : 15 do
            if EXEC[row * 16 + i].u1 then
                 VGPR[row * 16 + i][VDST.u] = tmp[64'B(altrow.i * 16) + lanesel[i * 4 + 3 : i * 4]]
            endif
        endfor
  endfor

Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

Example implementing a rotation across an entire wave32 wavefront:

  // Note for this to work, source and destination VGPRs must be different.
  // For this rotation, lane 15 gets data from lane 16, lane 31 gets data from lane 0.
  // These are the only two lanes that need to use v_permlanex16_b32.

   // Enable only the threads that get data from their own row.
  v_mov_b32 exec_lo, 0x7fff7fff; // Lanes getting data from their own row
  v_mov_b32 s0, 0x87654321;
  v_mov_b32 s1, 0x0fedcba9;
  v_permlane16_b32 v1, v0, s0, s1 fi; // FI bit needed for lanes 14 and 30
  // ROW 0:

  // v1.lane[0] <- v0.lane[1]
  // v1.lane[1] <- v0.lane[2]
  // ...
  // v1.lane[14] <- v0.lane[15] (needs FI to read)
  // v1.lane[15] unset
  //
  // ROW 1:
  // v1.lane[16] <- v0.lane[17]
  // v1.lane[17] <- v0.lane[18]
  // ...
  // v1.lane[30] <- v0.lane[31] (needs FI to read)
  // v1.lane[31] unset

  // Enable only the threads that get data from the other row.
  v_mov_b32 exec_lo, 0x80008000; // Lanes getting data from the other row
  v_permlanex16_b32 v1, v0, s0, s1 fi; // FI bit needed for lanes 15 and 31
  // v1.lane[15] <- v0.lane[16]
  // v1.lane[31] <- v0.lane[0]

V_CNDMASK_B16                                                                                                      605

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

  D0.u16 = VCC.u64[laneId] ? S1.u16 : S0.u16

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 16-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

V_MAXMIN_F32                                                                                                       606

Compute maximum of first two operands, followed by minimum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.f = v_min_f32(v_max_f32(S0.f, S1.f), S2.f)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MINMAX_F32                                                                                               607

Compute minimum of first two operands, followed by maximum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's minBound > maxBound.

  D0.f = v_max_f32(v_min_f32(S0.f, S1.f), S2.f)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MAXMIN_F16                                                                                               608

Compute maximum of first two operands, followed by minimum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.f16 = v_min_f16(v_max_f16(S0.f16, S1.f16), S2.f16)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MINMAX_F16                                                                                               609

Compute minimum of first two operands, followed by maximum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.f16 = v_max_f16(v_min_f16(S0.f16, S1.f16), S2.f16)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MAXMIN_U32                                                                                               610

Compute maximum of first two operands, followed by minimum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.i = 32'I(v_min_u32(v_max_u32(S0.u, S1.u), S2.u))

V_MINMAX_U32                                                                                               611

Compute minimum of first two operands, followed by maximum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.i = 32'I(v_max_u32(v_min_u32(S0.u, S1.u), S2.u))

V_MAXMIN_I32                                                                                               612

Compute maximum of first two operands, followed by minimum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.i = v_min_i32(v_max_i32(S0.i, S1.i), S2.i)

V_MINMAX_I32                                                                                               613

Compute minimum of first two operands, followed by maximum of that result and the third operand.

This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the
clamp's maxBound < minBound.

  D0.i = v_max_i32(v_min_i32(S0.i, S1.i), S2.i)

V_DOT2_F16_F16                                                                                             614

Dot product of packed FP16 values.

  tmp = S0[15 : 0].f16 * S1[15 : 0].f16;
  tmp += S0[31 : 16].f16 * S1[31 : 16].f16;

  tmp += S2.f16;
  D0.f16 = tmp

Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are
ignored.

V_DOT2_BF16_BF16                                                                                              615

Dot product of packed brain-float values.

  tmp = S0[15 : 0].bf16 * S1[15 : 0].bf16;
  tmp += S0[31 : 16].bf16 * S1[31 : 16].bf16;
  tmp += S2.bf16;
  D0.bf16 = tmp

Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are
ignored.

V_DIV_SCALE_F32                                                                                               764

Single precision division pre-scale.

S0 = Input to scale (either denominator or numerator), S1 = Denominator, S2 = Numerator.

Given a numerator and denominator, this opcode appropriately scales inputs for division to avoid subnormal
terms during Newton-Raphson correction method. S0 must be the same value as either S1 or S2.

This opcode produces a VCC flag for post-scaling of the quotient (using V_DIV_FMAS_F32).

  VCC = 0x0LL;
  if ((64'F(S2.f) == 0.0) || (64'F(S1.f) == 0.0)) then
        D0.f = NAN.f
  elsif exponent(S2.f) - exponent(S1.f) >= 96 then
        // N/D near MAX_FLOAT_F32
        VCC = 0x1LL;
        if S0.f == S1.f then
            // Only scale the denominator
            D0.f = ldexp(S0.f, 64)
        endif
  elsif S1.f == DENORM.f then
        D0.f = ldexp(S0.f, 64)
  elsif ((1.0 / 64'F(S1.f) == DENORM.f64) && (S2.f / S1.f == DENORM.f)) then
        VCC = 0x1LL;
        if S0.f == S1.f then

           // Only scale the denominator
           D0.f = ldexp(S0.f, 64)
      endif
  elsif 1.0 / 64'F(S1.f) == DENORM.f64 then
      D0.f = ldexp(S0.f, -64)
  elsif S2.f / S1.f == DENORM.f then
      VCC = 0x1LL;
      if S0.f == S2.f then
           // Only scale the numerator
           D0.f = ldexp(S0.f, 64)
      endif
  elsif exponent(S2.f) <= 23 then
      // Numerator is tiny
      D0.f = ldexp(S0.f, 64)
  endif

V_DIV_SCALE_F64                                                                                         765

Double precision division pre-scale.

S0 = Input to scale (either denominator or numerator), S1 = Denominator, S2 = Numerator.

Given a numerator and denominator, this opcode appropriately scales inputs for division to avoid subnormal
terms during Newton-Raphson correction method. S0 must be the same value as either S1 or S2.

This opcode produces a VCC flag for post-scaling of the quotient (using V_DIV_FMAS_F64).

  VCC = 0x0LL;
  if ((S2.f64 == 0.0) || (S1.f64 == 0.0)) then
      D0.f64 = NAN.f64
  elsif exponent(S2.f64) - exponent(S1.f64) >= 768 then
      // N/D near MAX_FLOAT_F64
      VCC = 0x1LL;
      if S0.f64 == S1.f64 then
           // Only scale the denominator
           D0.f64 = ldexp(S0.f64, 128)
      endif
  elsif S1.f64 == DENORM.f64 then
      D0.f64 = ldexp(S0.f64, 128)
  elsif ((1.0 / S1.f64 == DENORM.f64) && (S2.f64 / S1.f64 == DENORM.f64)) then
      VCC = 0x1LL;
      if S0.f64 == S1.f64 then
           // Only scale the denominator
           D0.f64 = ldexp(S0.f64, 128)
      endif
  elsif 1.0 / S1.f64 == DENORM.f64 then
      D0.f64 = ldexp(S0.f64, -128)
  elsif S2.f64 / S1.f64 == DENORM.f64 then
      VCC = 0x1LL;
      if S0.f64 == S2.f64 then
           // Only scale the numerator
           D0.f64 = ldexp(S0.f64, 128)
      endif

  elsif exponent(S2.f64) <= 53 then
        // Numerator is tiny
        D0.f64 = ldexp(S0.f64, 128)
  endif

V_MAD_U64_U32                                                                                                   766

Multiply two unsigned integer inputs, add a third unsigned integer input, store the result into a 64-bit vector
register and store the overflow/carryout into a scalar mask register.

  { D1.u1, D0.u64 } = 65'B(65'U(S0.u) * 65'U(S1.u) + 65'U(S2.u64))

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

V_MAD_I64_I32                                                                                                   767

Multiply two signed integer inputs, add a third signed integer input, store the result into a 64-bit vector register
and store the overflow/carryout into a scalar mask register.

  { D1.i1, D0.i64 } = 65'B(65'I(S0.i) * 65'I(S1.i) + 65'I(S2.i64))

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

V_ADD_CO_U32                                                                                                    768

Add two unsigned inputs, store the result into a vector register and store the carry-out mask into a scalar
register.

  tmp = 64'U(S0.u) + 64'U(S1.u);
  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_ADD_CO_CI_U32.
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_SUB_CO_U32                                                                                                     769

Subtract the second unsigned input from the first input, store the result into a vector register and store the
carry-out mask into a scalar register.

  tmp = S0.u - S1.u;
  VCC.u64[laneId] = S1.u > S0.u ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_CO_U32                                                                                                  770

Subtract the first unsigned input from the second input, store the result into a vector register and store the
carry-out mask into a scalar register.

  tmp = S1.u - S0.u;
  VCC.u64[laneId] = S0.u > S1.u ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u = tmp.u

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_ADD_NC_U16                                                                                                     771

Add two unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.u16 = S0.u16 + S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_SUB_NC_U16                                                                                                  772

Subtract the second unsigned input from the first input and store the result into a vector register. No carry-in
or carry-out support.

  D0.u16 = S0.u16 - S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_MUL_LO_U16                                                                                                  773

Multiply two unsigned inputs and store the low bits of the result into a vector register.

  D0.u16 = S0.u16 * S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_CVT_PK_I16_F32                                                                                              774

Convert two single-precision floats into a packed value of signed words.

  D0[31 : 16] = 16'B(v_cvt_i16_f32(S1.f));
  D0[15 : 0] = 16'B(v_cvt_i16_f32(S0.f))

V_CVT_PK_U16_F32                                                                                              775

Convert two single-precision floats into a packed value of unsigned words.

  D0[31 : 16] = 16'B(v_cvt_u16_f32(S1.f));
  D0[15 : 0] = 16'B(v_cvt_u16_f32(S0.f))

V_MAX_U16                                                                                                     777

Select the maximum of two unsigned integers and store the selected value into a vector register.

  D0.u16 = S0.u16 >= S1.u16 ? S0.u16 : S1.u16

V_MAX_I16                                                                                                      778

Select the maximum of two signed integers and store the selected value into a vector register.

  D0.i16 = S0.i16 >= S1.i16 ? S0.i16 : S1.i16

V_MIN_U16                                                                                                      779

Select the minimum of two unsigned integers and store the selected value into a vector register.

  D0.u16 = S0.u16 < S1.u16 ? S0.u16 : S1.u16

V_MIN_I16                                                                                                      780

Select the minimum of two signed integers and store the selected value into a vector register.

  D0.i16 = S0.i16 < S1.i16 ? S0.i16 : S1.i16

V_ADD_NC_I16                                                                                                   781

Add two signed inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.i16 = S0.i16 + S1.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_SUB_NC_I16                                                                                                   782

Subtract the second signed input from the first input and store the result into a vector register. No carry-in or
carry-out support.

  D0.i16 = S0.i16 - S1.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_PACK_B32_F16                                                                                                  785

Pack two FP16 values together.

  D0[31 : 16].f16 = S1.f16;
  D0[15 : 0].f16 = S0.f16

V_CVT_PK_NORM_I16_F16                                                                                           786

Convert two FP16 values into packed signed normalized shorts.

  D0[15 : 0].i16 = f16_to_snorm(S0[15 : 0].f16);
  D0[31 : 16].i16 = f16_to_snorm(S1[15 : 0].f16)

V_CVT_PK_NORM_U16_F16                                                                                           787

Convert two FP16 values into packed unsigned normalized shorts.

  D0[15 : 0].u16 = f16_to_unorm(S0[15 : 0].f16);
  D0[31 : 16].u16 = f16_to_unorm(S1[15 : 0].f16)

V_LDEXP_F32                                                                                                     796

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register. Compare with the ldexp() function in C.

  D0.f = S0.f * 2.0F ** S1.i

V_BFM_B32                                                                                                       797

Bitfield modify.

S0 is the bitfield width and S1 is the bitfield offset.

  D0.u = 32'U(((1 << S0[4 : 0].u) - 1) << S1[4 : 0].u)

V_BCNT_U32_B32                                                                                         798

Count the number of "1" bits in the vector input and store the result into a vector register.

  D0.u = S1.u;
  for i in 0 : 31 do
       D0.u += S0[i].u;
       // count i'th bit
  endfor

V_MBCNT_LO_U32_B32                                                                                     799

Masked bit count.

laneId is the position of this thread in the wavefront (in 0..63). See also V_MBCNT_HI_U32_B32.

  ThreadMask = (1LL << laneId.u) - 1LL;
  MaskedValue = (S0.u & ThreadMask[31 : 0].u);
  D0.u = S1.u;
  for i in 0 : 31 do
       D0.u += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor

V_MBCNT_HI_U32_B32                                                                                     800

Masked bit count, high pass.

laneId is the position of this thread in the wavefront (in 0..63). See also V_MBCNT_LO_U32_B32.

  ThreadMask = (1LL << laneId.u) - 1LL;
  MaskedValue = (S0.u & ThreadMask[63 : 32].u);
  D0.u = S1.u;
  for i in 0 : 31 do
       D0.u += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor

Notes

Note that in Wave32 mode ThreadMask[63:32] == 0 and this instruction simply performs a move from S1 to D.

Example to compute each thread's position in 0..63:

        v_mbcnt_lo_u32_b32 v0, -1, 0
        v_mbcnt_hi_u32_b32 v0, -1, v0
        // v0 now contains laneId

V_CVT_PK_NORM_I16_F32                                                                                  801

Convert two single-precision floats into a packed signed normalized value.

  D0[15 : 0].i16 = f32_to_snorm(S0.f);
  D0[31 : 16].i16 = f32_to_snorm(S1.f)

V_CVT_PK_NORM_U16_F32                                                                                  802

Convert two single-precision floats into a packed unsigned normalized value.

  D0[15 : 0].u16 = f32_to_unorm(S0.f);
  D0[31 : 16].u16 = f32_to_unorm(S1.f)

V_CVT_PK_U16_U32                                                                                       803

Convert two unsigned integers into a packed unsigned short.

  D0[15 : 0].u16 = u32_to_u16(S0.u);
  D0[31 : 16].u16 = u32_to_u16(S1.u)

V_CVT_PK_I16_I32                                                                                       804

Convert two signed integers into a packed signed short.

  D0[15 : 0].i16 = i32_to_i16(S0.i);
  D0[31 : 16].i16 = i32_to_i16(S1.i)

V_SUB_NC_I32                                                                                                   805

Subtract the second signed input from the first input and store the result into a vector register. No carry-in or
carry-out support.

  D0.i = S0.i - S1.i

Notes

Supports saturation (signed 32-bit integer domain).

V_ADD_NC_I32                                                                                                   806

Add two signed inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.i = S0.i + S1.i

Notes

Supports saturation (signed 32-bit integer domain).

V_ADD_F64                                                                                                      807

Add two floating point inputs and store the result into a vector register.

  D0.f64 = S0.f64 + S1.f64

Notes

0.5ULP precision, denormals are supported.

V_MUL_F64                                                                                                      808

Multiply two floating point inputs and store the result into a vector register.

  D0.f64 = S0.f64 * S1.f64

Notes

0.5ULP precision, denormals are supported.

V_MIN_F64                                                                                               809

Select the minimum of two floating point inputs and store the result into a vector register.

  LT_NEG_ZERO = lambda(a, b) (
        ((a < b) || ((abs(a) == 0.0) && (abs(b) == 0.0) && sign(a) && !sign(b))));
  // Version of comparison where -0.0 < +0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(S0.f64) then
            D0.f64 = cvtToQuietNAN(S0.f64)
        elsif isSignalNAN(S1.f64) then
            D0.f64 = cvtToQuietNAN(S1.f64)
        elsif isQuietNAN(S1.f64) then
            D0.f64 = S0.f64
        elsif isQuietNAN(S0.f64) then
            D0.f64 = S1.f64
        elsif LT_NEG_ZERO(S0.f64, S1.f64) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f64 = S0.f64
        else
            D0.f64 = S1.f64
        endif
  else
        if isNAN(S1.f64) then
            D0.f64 = S0.f64
        elsif isNAN(S0.f64) then
            D0.f64 = S1.f64
        elsif LT_NEG_ZERO(S0.f64, S1.f64) then
            // NOTE: -0<+0 is TRUE in this comparison
            D0.f64 = S0.f64
        else
            D0.f64 = S1.f64
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

V_MAX_F64                                                                                               810

Select the maximum of two floating point inputs and store the result into a vector register.

  GT_NEG_ZERO = lambda(a, b) (
        ((a > b) || ((abs(a) == 0.0) && (abs(b) == 0.0) && !sign(a) && sign(b))));
  // Version of comparison where +0.0 > -0.0, differs from IEEE
  if WAVE_MODE.IEEE then
        if isSignalNAN(S0.f64) then
            D0.f64 = cvtToQuietNAN(S0.f64)
        elsif isSignalNAN(S1.f64) then
            D0.f64 = cvtToQuietNAN(S1.f64)
        elsif isQuietNAN(S1.f64) then
            D0.f64 = S0.f64
        elsif isQuietNAN(S0.f64) then
            D0.f64 = S1.f64
        elsif GT_NEG_ZERO(S0.f64, S1.f64) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f64 = S0.f64
        else
            D0.f64 = S1.f64
        endif
  else
        if isNAN(S1.f64) then
            D0.f64 = S0.f64
        elsif isNAN(S0.f64) then
            D0.f64 = S1.f64
        elsif GT_NEG_ZERO(S0.f64, S1.f64) then
            // NOTE: +0>-0 is TRUE in this comparison
            D0.f64 = S0.f64
        else
            D0.f64 = S1.f64
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

V_LDEXP_F64                                                                                                     811

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register. Compare with the ldexp() function in C.

  D0.f64 = S0.f64 * 2.0 ** S1.i

V_MUL_LO_U32                                                                                                 812

Multiply two unsigned integers.

  D0.u = S0.u * S1.u

Notes

To multiply integers with small magnitudes consider V_MUL_U32_U24, which is intended to be a more
efficient implementation.

V_MUL_HI_U32                                                                                                 813

Multiply two unsigned integers and store the high 32 bits of the result.

  D0.u = 32'U((64'U(S0.u) * 64'U(S1.u)) >> 32U)

Notes

To multiply integers with small magnitudes consider V_MUL_HI_U32_U24, which is intended to be a more
efficient implementation.

V_MUL_HI_I32                                                                                                 814

Multiply two signed integers and store the high 32 bits of the result.

  D0.i = 32'I((64'I(S0.i) * 64'I(S1.i)) >> 32U)

Notes

To multiply integers with small magnitudes consider V_MUL_HI_I32_I24, which is intended to be a more
efficient implementation.

V_TRIG_PREOP_F64                                                                                             815

Look Up 2/PI (S0.f64) with segment select S1.u[4:0].

This operation returns an aligned, double precision segment of 2/PI needed to do range reduction on S0.f64
(double-precision value). Multiple segments can be specified through S1.u[4:0]. Rounding is round-to-zero.
Large inputs (exp > 1968) are scaled to avoid loss of precision through denormalization.

  shift = 32'I(S1[4 : 0].u) * 53;

  if exponent(S0.f64) > 1077 then
       shift += exponent(S0.f64) - 1077
  endif;
  // (2.0/PI) == 0.{b_1200, b_1199, b_1198, ..., b_1, b_0}
  // b_1200 is the MSB of the fractional part of 2.0/PI
  // Left shift operation indicates which bits are brought
  // into the whole part of the number.
  // Only whole part of result is kept.
  result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u) & 1201'0x1fffffffffffff);
  scale = -53 - shift;
  if exponent(S0.f64) >= 1968 then
       scale += 128
  endif;
  D0.f64 = ldexp(result, scale)

V_LSHLREV_B16                                                                                                      824

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u[15 : 0] = (S1.u[15 : 0] << S0[3 : 0].u)

V_LSHRREV_B16                                                                                                      825

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u[15 : 0] = (S1.u[15 : 0] >> S0[3 : 0].u)

V_ASHRREV_I16                                                                                                      826

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i[15 : 0] = (S1.i[15 : 0] >> S0[3 : 0].u)

V_LSHLREV_B64                                                                                                      828

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u64 = (S1.u64 << S0[5 : 0].u)

Notes

Only one scalar broadcast constant is allowed.

V_LSHRREV_B64                                                                                                      829

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u64 = (S1.u64 >> S0[5 : 0].u)

Notes

Only one scalar broadcast constant is allowed.

V_ASHRREV_I64                                                                                                      830

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i64 = (S1.i64 >> S0[5 : 0].u)

Notes

Only one scalar broadcast constant is allowed.

V_READLANE_B32                                                                                                     864

Copy one VGPR value from a single lane to one SGPR.

  declare lane : 32'U;
  if WAVE32 then
        lane = S1.u[4 : 0].u;
        // Lane select for wave32
  else
        lane = S1.u[5 : 0].u;
        // Lane select for wave64
  endif;
  D0.b = VGPR[lane][SRC0.u]

Notes

Ignores EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped operation.

V_WRITELANE_B32                                                                                           865

Write scalar value into one VGPR in one lane.

  declare lane : 32'U;
  if WAVE32 then
        lane = S1.u[4 : 0].u;
        // Lane select for wave32
  else
        lane = S1.u[5 : 0].u;
        // Lane select for wave64
  endif;
  VGPR[lane][VDST.u] = S0.b

Notes

Ignores EXEC mask for the VGPR write. Input and output modifiers not supported; this is an untyped
operation.

V_AND_B16                                                                                                 866

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 & S1.u16)

Notes

Input and output modifiers not supported.

V_OR_B16                                                                                                  867

Calculate bitwise OR on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 | S1.u16)

Notes

Input and output modifiers not supported.

V_XOR_B16                                                                                      868

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 ^ S1.u16)

Notes

Input and output modifiers not supported.

V_CMP_F_F16                                                                                       0

Return 0.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F16                                                                                      1

Return 1 iff A less than B.

  D0.u64[laneId] = S0.f16 < S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F16                                                                                      2

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

V_CMP_LG_F16                                                              5

Return 1 iff A less than or greater than B.

  D0.u64[laneId] = S0.f16 <> S1.f16;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F16                                                              6

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

V_CMP_NGE_F16                                                                9

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

V_CMP_GE_U64                                                             94

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

V_CMPX_F_F16                                                                                                   128

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

V_CMPX_GE_U64                                                                                              222

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

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
