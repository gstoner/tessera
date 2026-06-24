# 16.12. VOP3 & VOP3SD Instructions

> RDNA3.5 ISA — pages 398–537

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

V_NOP                                                                                                                 384

Do nothing.

V_MOV_B32                                                                                                             385

Move 32-bit data from a vector input into a vector register.

  D0.b32 = S0.b32

Notes

Floating-point modifiers are valid for this instruction if S0 is a 32-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

Functional examples:

        v_mov_b32 v0, v1    // Move into v0 from v1
        v_mov_b32 v0, -v1   // Set v0 to the negation of v1
        v_mov_b32 v0, abs(v1)    // Set v0 to the absolute value of v1

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

V_CVT_I32_F64                                                                                                       387

Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector
register.

  D0.i32 = f64_to_i32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_I32                                                                                                       388

Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector
register.

  D0.f64 = i32_to_f64(S0.i32)

Notes

0ULP accuracy.

V_CVT_F32_I32                                                                                                     389

Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector
register.

  D0.f32 = i32_to_f32(S0.i32)

Notes

0.5ULP accuracy.

V_CVT_F32_U32                                                                                                     390

Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0.u32)

Notes

0.5ULP accuracy.

V_CVT_U32_F32                                                                                                     391

Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector
register.

  D0.u32 = f32_to_u32(S0.f32)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_I32_F32                                                                                                     392

Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector
register.

  D0.i32 = f32_to_i32(S0.f32)

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F16_F32                                                                                                     394

Convert from a single-precision float input to a half-precision float value and store the result into a vector
register.

  D0.f16 = f32_to_f16(S0.f32)

Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on
output if specified based on DP denorm mode. Output rounding based on DP rounding mode.

V_CVT_F32_F16                                                                                                     395

Convert from a half-precision float input to a single-precision float value and store the result into a vector
register.

  D0.f32 = f16_to_f32(S0.f16)

Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm
mode.

V_CVT_NEAREST_I32_F32                                                                                             396

Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer
semantics (ignore the default rounding mode) and store the result into a vector register.

  D0.i32 = f32_to_i32(floor(S0.f32 + 0.5F))

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_FLOOR_I32_F32                                                                                               397

Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore
the default rounding mode) and store the result into a vector register.

  D0.i32 = f32_to_i32(floor(S0.f32))

Notes

1ULP accuracy, denormals are supported.

V_CVT_OFF_F32_I4                                                                                                  398

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

V_CVT_F32_F64                                                                                                  399

Convert from a double-precision float input to a single-precision float value and store the result into a vector
register.

  D0.f32 = f64_to_f32(S0.f64)

Notes

0.5ULP accuracy, denormals are supported.

V_CVT_F64_F32                                                                                                  400

Convert from a single-precision float input to a double-precision float value and store the result into a vector
register.

  D0.f64 = f32_to_f64(S0.f32)

Notes

0ULP accuracy, denormals are supported.

V_CVT_F32_UBYTE0                                                                                               401

Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[7 : 0].u32)

V_CVT_F32_UBYTE1                                                                                               402

Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[15 : 8].u32)

V_CVT_F32_UBYTE2                                                                                               403

Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[23 : 16].u32)

V_CVT_F32_UBYTE3                                                                                               404

Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector
register.

  D0.f32 = u32_to_f32(S0[31 : 24].u32)

V_CVT_U32_F64                                                                                                  405

Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a
vector register.

  D0.u32 = f64_to_u32(S0.f64)

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_F64_U32                                                                                                  406

Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a
vector register.

  D0.f64 = u32_to_f64(S0.u32)

Notes

0ULP accuracy.

V_TRUNC_F64                                                                                                      407

Compute the integer part of a double-precision float input using round toward zero semantics and store the
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

V_PIPEFLUSH                                                                                                           411

Flush the vector ALU pipeline through the destination cache.

V_MOV_B16                                                                                                             412

Move 16-bit data from a vector input into a vector register.

  D0.b16 = S0.b16

Notes

Floating-point modifiers are valid for this instruction if S0 is a 16-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

V_FRACT_F32                                                                                                           416

Compute the fractional portion of a single-precision float input and store the result in floating point format into
a vector register.

  D0.f32 = S0.f32 + -floor(S0.f32)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3f7fffff.

V_TRUNC_F32                                                                                                           417

Compute the integer part of a single-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

  D0.f32 = trunc(S0.f32)

V_CEIL_F32                                                                                                         418

Round the single-precision float input up to next integer and store the result in floating point format into a
vector register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 > 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += 1.0F
  endif

V_RNDNE_F32                                                                                                        419

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f32 = floor(S0.f32 + 0.5F);
  if (isEven(64'F(floor(S0.f32))) && (fract(S0.f32) == 0.5F)) then
        D0.f32 -= 1.0F
  endif

V_FLOOR_F32                                                                                                        420

Round the single-precision float input down to previous integer and store the result in floating point format
into a vector register.

  D0.f32 = trunc(S0.f32);
  if ((S0.f32 < 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += -1.0F
  endif

V_EXP_F32                                                                                                          421

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

  D0.f32 = pow(2.0F, S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_EXP_F32(0xff800000) => 0x00000000       // exp(-INF) = 0
  V_EXP_F32(0x80000000) => 0x3f800000       // exp(-0.0) = 1
  V_EXP_F32(0x7f800000) => 0x7f800000       // exp(+INF) = +INF

V_LOG_F32                                                                                                          423

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

V_RCP_F32                                                                                                          426

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

V_RCP_IFLAG_F32                                                                                                  427

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

V_RSQ_F32                                                                                                        430

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

  D0.f32 = sqrt(S0.f32)

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

  V_SQRT_F32(0xff800000) => 0xffc00000       // sqrt(-INF) = NAN
  V_SQRT_F32(0x80000000) => 0x80000000       // sqrt(-0.0) = -0
  V_SQRT_F32(0x00000000) => 0x00000000       // sqrt(+0.0) = +0
  V_SQRT_F32(0x40800000) => 0x40000000       // sqrt(+4.0) = +2.0
  V_SQRT_F32(0x7f800000) => 0x7f800000       // sqrt(+INF) = +INF

V_SQRT_F64                                                                                                        436

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector
register.

  D0.f64 = sqrt(S0.f64)

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

V_SIN_F32                                                                                                         437

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

V_COS_F32                                                                                                         438

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

  D0.f32 = cos(S0.f32 * 32'F(PI * 2.0))

Notes

Denormals are supported. Full range input is supported.

Functional examples:

  V_COS_F32(0xff800000) => 0xffc00000        // cos(-INF) = NAN
  V_COS_F32(0xff7fffff) => 0x3f800000        // -MaxFloat, finite
  V_COS_F32(0x80000000) => 0x3f800000        // cos(-0.0) = 1
  V_COS_F32(0x3e800000) => 0x00000000        // cos(0.25) = 0
  V_COS_F32(0x7f800000) => 0xffc00000        // cos(+INF) = NAN

V_NOT_B32                                                                                                        439

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u32 = ~S0.u32

Notes

Input and output modifiers not supported.

V_BFREV_B32                                                                                                      440

Reverse the order of bits in a vector input and store the result into a vector register.

  D0.u32[31 : 0] = S0.u32[0 : 31]

Notes

Input and output modifiers not supported.

V_CLZ_I32_U32                                                                                                    441

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

V_CTZ_I32_B32                                                                                                     442

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

V_CLS_I32                                                                                                         443

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

V_FREXP_EXP_I32_F64                                                                                             444

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
in normal cases. See also V_FREXP_EXP_I32_F64, which returns integer exponent. See the C library function
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

  if ((64'F(S0.f32) == +INF) || (64'F(S0.f32) == -INF) || isNAN(64'F(S0.f32))) then
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f32) - 127 + 1
  endif

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32,
which returns the significand. See the C library function frexp() for more information.

V_FREXP_MANT_F32                                                                                                   448

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
frexp() for more information.

V_MOVRELD_B32                                                                                                      450

Move data from a vector input into a relatively-indexed vector register.

  addr = DST.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  VGPR[laneId][addr].b32 = S0.b32

Notes

Example: The following instruction sequence performs the move v15 <= v7:

        s_mov_b32 m0, 10
        v_movreld_b32 v5, v7

V_MOVRELS_B32                                                                                                      451

Move data from a relatively-indexed vector register into another vector register.

  addr = SRC0.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  D0.b32 = VGPR[laneId][addr].b32

Notes

Example: The following instruction sequence performs the move v5 <= v17:

        s_mov_b32 m0, 10
        v_movrels_b32 v5, v7

V_MOVRELSD_B32                                                                                               452

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

V_MOVRELSD_2_B32                                                                                             456

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

V_CVT_F16_U16                                                                                                   464

Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector
register.

  D0.f16 = u16_to_f16(S0.u16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_F16_I16                                                                                                   465

Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector
register.

  D0.f16 = i16_to_f16(S0.i16)

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

V_CVT_U16_F16                                                                                                   466

Convert from a half-precision float input to an unsigned 16-bit integer value and store the result into a vector
register.

  D0.u16 = f16_to_u16(S0.f16)

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

V_CVT_I16_F16                                                                                                   467

Convert from a half-precision float input to a signed 16-bit integer value and store the result into a vector
register.

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

  V_RCP_F16(0xfc00) => 0x8000        // rcp(-INF) = -0
  V_RCP_F16(0xc000) => 0xb800        // rcp(-2.0) = -0.5
  V_RCP_F16(0x8000) => 0xfc00        // rcp(-0.0) = -INF
  V_RCP_F16(0x0000) => 0x7c00        // rcp(+0.0) = +INF
  V_RCP_F16(0x7c00) => 0x0000        // rcp(+INF) = +0

V_SQRT_F16                                                                                                        469

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

V_RSQ_F16                                                                                                       470

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

V_LOG_F16                                                                                                       471

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

V_EXP_F16                                                                                                          472

Calculate 2 raised to the power of the half-precision float input and store the result into a vector register.

  D0.f16 = pow(16'2.0, S0.f16)

Notes

0.51ULP accuracy, denormals are supported.

Functional examples:

  V_EXP_F16(0xfc00) => 0x0000        // exp(-INF) = 0
  V_EXP_F16(0x8000) => 0x3c00        // exp(-0.0) = 1
  V_EXP_F16(0x7c00) => 0x7c00        // exp(+INF) = +INF

V_FREXP_MANT_F16                                                                                                   473

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

V_FREXP_EXP_I16_F16                                                                                                474

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

V_FLOOR_F16                                                                                                      475

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

Compute the integer part of a half-precision float input using round toward zero semantics and store the result
in floating point format into a vector register.

  D0.f16 = trunc(S0.f16)

V_RNDNE_F16                                                                                                      478

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
        D0.f16 -= 16'1.0

  endif

V_FRACT_F16                                                                                                       479

Compute the fractional portion of a half-precision float input and store the result in floating point format into a
vector register.

  D0.f16 = S0.f16 + -floor(S0.f16)

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

V_SIN_F16                                                                                                         480

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

V_COS_F16                                                                                                         481

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

V_SAT_PK_U8_I16                                                                                                482

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

V_CVT_NORM_I16_F16                                                                                             483

Convert from a half-precision float input to a signed normalized short and store the result into a vector
register.

  D0.i16 = f16_to_snorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_CVT_NORM_U16_F16                                                                                               484

Convert from a half-precision float input to an unsigned normalized short and store the result into a vector
register.

  D0.u16 = f16_to_unorm(S0.f16)

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

V_NOT_B16                                                                                                        489

Calculate bitwise negation on a vector input and store the result into a vector register.

  D0.u16 = ~S0.u16

Notes

Input and output modifiers not supported.

V_CVT_I32_I16                                                                                                    490

Convert from a signed 16-bit integer input to a signed 32-bit integer value using sign extension and store the
result into a vector register.

  D0.i32 = 32'I(signext(S0.i16))

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CVT_U32_U16                                                                                                    491

Convert from an unsigned 16-bit integer input to an unsigned 32-bit integer value using zero extension and
store the result into a vector register.

  D0 = { 16'0, S0.u16 }

Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.

V_CNDMASK_B32                                                                                                      257

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

  D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

V_ADD_F32                                                                                                          259

Add two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 + S1.f32

Notes

0.5ULP precision, denormals are supported.

V_SUB_F32                                                                                                          260

Subtract the second floating point input from the first input and store the result into a vector register.

  D0.f32 = S0.f32 - S1.f32

Notes

0.5ULP precision, denormals are supported.

V_SUBREV_F32                                                                                                       261

Subtract the first floating point input from the second input and store the result into a vector register.

  D0.f32 = S1.f32 - S0.f32

Notes

0.5ULP precision, denormals are supported.

V_FMAC_DX9_ZERO_F32                                                                                           262

Multiply two single-precision values and accumulate the result with the destination. Follows DX9 rules where
0.0 times anything produces 0.0.

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = S2.f32
  else
        D0.f32 = fma(S0.f32, S1.f32, D0.f32)
  endif

V_MUL_DX9_ZERO_F32                                                                                            263

Multiply two floating point inputs and store the result into a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = 0.0F
  else
        D0.f32 = S0.f32 * S1.f32
  endif

V_MUL_F32                                                                                                     264

Multiply two floating point inputs and store the result into a vector register.

  D0.f32 = S0.f32 * S1.f32

Notes

0.5ULP precision, denormals are supported.

V_MUL_I32_I24                                                                                                       265

Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register.

  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_I32_I24.

V_MUL_HI_I32_I24                                                                                                    266

Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a
vector register.

  D0.i32 = 32'I((64'I(S0.i24) * 64'I(S1.i24)) >> 32U)

Notes

See also V_MUL_I32_I24.

V_MUL_U32_U24                                                                                                       267

Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector
register.

  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24)

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_U32_U24.

V_MUL_HI_U32_U24                                                                                                    268

Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer
into a vector register.

  D0.u32 = 32'U((64'U(S0.u24) * 64'U(S1.u24)) >> 32U)

Notes

See also V_MUL_U32_U24.

V_MIN_F32                                                                                                   271

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

V_MAX_F32                                                                                                     272

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

V_MIN_I32                                                                                                     273

Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = S0.i32 < S1.i32 ? S0.i32 : S1.i32

V_MAX_I32                                                                                                          274

Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = S0.i32 >= S1.i32 ? S0.i32 : S1.i32

V_MIN_U32                                                                                                          275

Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = S0.u32 < S1.u32 ? S0.u32 : S1.u32

V_MAX_U32                                                                                                          276

Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = S0.u32 >= S1.u32 ? S0.u32 : S1.u32

V_LSHLREV_B32                                                                                                      280

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u32 = (S1.u32 << S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_LSHRREV_B32                                                                                                      281

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u32 = (S1.u32 >> S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_ASHRREV_I32                                                                                                      282

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i32 = (S1.i32 >> S0[4 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_AND_B32                                                                                                          283

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 & S1.u32)

Notes

Input and output modifiers not supported.

V_OR_B32                                                                                                           284

Calculate bitwise OR on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 | S1.u32)

Notes

Input and output modifiers not supported.

V_XOR_B32                                                                                                          285

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 ^ S1.u32)

Notes

Input and output modifiers not supported.

V_XNOR_B32                                                                                                    286

Calculate bitwise XNOR on two vector inputs and store the result into a vector register.

  D0.u32 = ~(S0.u32 ^ S1.u32)

Notes

Input and output modifiers not supported.

V_ADD_CO_CI_U32                                                                                               288

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

V_SUB_CO_CI_U32                                                                                               289

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

V_SUBREV_CO_CI_U32                                                                                               290

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

V_ADD_NC_U32                                                                                                     293

Add two unsigned 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

  D0.u32 = S0.u32 + S1.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUB_NC_U32                                                                                                     294

Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

  D0.u32 = S0.u32 - S1.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_NC_U32                                                                                                 295

Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

  D0.u32 = S1.u32 - S0.u32

Notes

Supports saturation (unsigned 32-bit integer domain).

V_FMAC_F32                                                                                                      299

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

  D0.f32 = fma(S0.f32, S1.f32, D0.f32)

V_CVT_PK_RTZ_F16_F32                                                                                            303

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

V_ADD_F16                                                                                                       306

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

V_MUL_F16                                                                                                         309

Multiply two floating point inputs and store the result into a vector register.

  D0.f16 = S0.f16 * S1.f16

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_FMAC_F16                                                                                                        310

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

  D0.f16 = fma(S0.f16, S1.f16, D0.f16)

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

V_MAX_F16                                                                                                 313

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

V_MIN_F16                                                                                                       314

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
integer value, and store the floating point result into a vector register.

  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))

Notes

Compare with the ldexp() function in C.

V_FMA_DX9_ZERO_F32                                                                                              521

Multiply and add single-precision values. Follows DX9 rules where 0.0 times anything produces 0.0.

  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = S2.f32
  else
        D0.f32 = fma(S0.f32, S1.f32, S2.f32)
  endif

V_MAD_I32_I24                                                                                                   522

Multiply two signed 24-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value
from a third input, and store the result as a signed 32-bit integer into a vector register.

  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24) + S2.i32

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier.

V_MAD_U32_U24                                                                                                   523

Multiply two unsigned 24-bit integer inputs in the unsigned 32-bit integer domain, add a unsigned 32-bit
integer value from a third input, and store the result as an unsigned 32-bit integer into a vector register.

  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24) + S2.u32

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier.

V_CUBEID_F32                                                                                                 524

Compute the cubemap face ID of a 3D coordinate specified as three single-precision float inputs. Store the
result in single-precision float format into a vector register.

  // Set D0.f = cubemap face ID ({0.0, 1.0, ..., 5.0}).
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      if S2.f32 < 0.0F then
           D0.f32 = 5.0F
      else
           D0.f32 = 4.0F
      endif
  elsif abs(S1.f32) >= abs(S0.f32) then
      if S1.f32 < 0.0F then
           D0.f32 = 3.0F
      else
           D0.f32 = 2.0F
      endif
  else
      if S0.f32 < 0.0F then
           D0.f32 = 1.0F
      else
           D0.f32 = 0.0F
      endif
  endif

V_CUBESC_F32                                                                                                 525

Compute the cubemap S coordinate of a 3D coordinate specified as three single-precision float inputs. Store the
result in single-precision float format into a vector register.

  // D0.f = cubemap S coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      if S2.f32 < 0.0F then
           D0.f32 = -S0.f32
      else
           D0.f32 = S0.f32
      endif
  elsif abs(S1.f32) >= abs(S0.f32) then
      D0.f32 = S0.f32
  else
      if S0.f32 < 0.0F then
           D0.f32 = S2.f32

      else
           D0.f32 = -S2.f32
      endif
  endif

V_CUBETC_F32                                                                                              526

Compute the cubemap T coordinate of a 3D coordinate specified as three single-precision float inputs. Store
the result in single-precision float format into a vector register.

  // D0.f = cubemap T coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      D0.f32 = -S1.f32
  elsif abs(S1.f32) >= abs(S0.f32) then
      if S1.f32 < 0.0F then
           D0.f32 = -S2.f32
      else
           D0.f32 = S2.f32
      endif
  else
      D0.f32 = -S1.f32
  endif

V_CUBEMA_F32                                                                                              527

Compute the cubemap major axis coordinate of a 3D coordinate specified as three single-precision float inputs.
Store the result in single-precision float format into a vector register.

  // D0.f = 2.0 * cubemap major axis.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      D0.f32 = S2.f32 * 2.0F
  elsif abs(S1.f32) >= abs(S0.f32) then
      D0.f32 = S1.f32 * 2.0F
  else
      D0.f32 = S0.f32 * 2.0F
  endif

V_BFE_U32                                                                                                 528

Extract an unsigned bitfield from the first input using field offset from the second input and size from the third
input, then store the result into a vector register.

  D0.u32 = ((S0.u32 >> S1[4 : 0].u32) & ((1U << S2[4 : 0].u32) - 1U))

V_BFE_I32                                                                                                         529

Extract a signed bitfield from the first input using field offset from the second input and size from the third
input, then store the result into a vector register.

  tmp.i32 = ((S0.i32 >> S1[4 : 0].u32) & ((1 << S2[4 : 0].u32) - 1));
  D0.i32 = signext_from_bit(tmp.i32, S2[4 : 0].u32)

V_BFI_B32                                                                                                         530

Overwrite a bitfield in the third input with a bitfield from the second input using a mask from the first input,
then store the result into a vector register.

  D0.u32 = ((S0.u32 & S1.u32) | (~S0.u32 & S2.u32))

V_FMA_F32                                                                                                         531

Multiply two single-precision float inputs and add a third input using fused multiply add, and store the result
into a vector register.

  D0.f32 = fma(S0.f32, S1.f32, S2.f32)

Notes

0.5ULP accuracy, denormals are supported.

V_FMA_F64                                                                                                         532

Multiply two double-precision float inputs and add a third input using fused multiply add, and store the result
into a vector register.

  D0.f64 = fma(S0.f64, S1.f64, S2.f64)

Notes

0.5ULP accuracy, denormals are supported.

V_LERP_U8                                                                                                        533

Average two 4-D vectors stored as packed bytes in the first two inputs with rounding control provided by the
third input, then store the result into a vector register. Each byte in the third input acts as a rounding mode for
the corresponding element; if the LSB is set then 0.5 rounds up, otherwise 0.5 truncates.

  tmp = ((S0.u32[31 : 24] + S1.u32[31 : 24] + S2.u32[24].u8) >> 1U << 24U);
  tmp += ((S0.u32[23 : 16] + S1.u32[23 : 16] + S2.u32[16].u8) >> 1U << 16U);
  tmp += ((S0.u32[15 : 8] + S1.u32[15 : 8] + S2.u32[8].u8) >> 1U << 8U);
  tmp += ((S0.u32[7 : 0] + S1.u32[7 : 0] + S2.u32[0].u8) >> 1U);
  D0.u32 = tmp.u32

V_ALIGNBIT_B32                                                                                                   534

Align a 64-bit value encoded in the first two inputs to a bit position specified in the third input, then store the
result into a 32-bit vector register.

  D0.u32 = 32'U(({ S0.u32, S1.u32 } >> S2.u32[4 : 0].u32) & 0xffffffffLL)

Notes

                S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

V_ALIGNBYTE_B32                                                                                                  535

Align a 64-bit value encoded in the first two inputs to a byte position specified in the third input, then store the
result into a 32-bit vector register.

  D0.u32 = 32'U(({ S0.u32, S1.u32 } >> (S2.u32[1 : 0].u32 * 8U)) & 0xffffffffLL)

Notes

                S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

V_MULLIT_F32                                                                                                     536

Multiply two floating point inputs and store the result into a vector register. Specific rules apply to
accommodate lighting calculations: 0.0 * x = 0.0 and alternate INF, NAN, overflow rules apply.

  if ((S1.f32 == -MAX_FLOAT_F32) || (64'F(S1.f32) == -INF) || isNAN(64'F(S1.f32)) || (S2.f32 <= 0.0F) ||
  isNAN(64'F(S2.f32))) then
        D0.f32 = -MAX_FLOAT_F32
  else
        D0.f32 = S0.f32 * S1.f32
  endif

Notes

V_MIN3_F32                                                                                                       537

Select the minimum of three single-precision float inputs and store the selected value into a vector register.

  D0.f32 = v_min_f32(v_min_f32(S0.f32, S1.f32), S2.f32)

V_MIN3_I32                                                                                                       538

Select the minimum of three signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = v_min_i32(v_min_i32(S0.i32, S1.i32), S2.i32)

V_MIN3_U32                                                                                                       539

Select the minimum of three unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = v_min_u32(v_min_u32(S0.u32, S1.u32), S2.u32)

V_MAX3_F32                                                                                                       540

Select the maximum of three single-precision float inputs and store the selected value into a vector register.

  D0.f32 = v_max_f32(v_max_f32(S0.f32, S1.f32), S2.f32)

V_MAX3_I32                                                                                                      541

Select the maximum of three signed 32-bit integer inputs and store the selected value into a vector register.

  D0.i32 = v_max_i32(v_max_i32(S0.i32, S1.i32), S2.i32)

V_MAX3_U32                                                                                                      542

Select the maximum of three unsigned 32-bit integer inputs and store the selected value into a vector register.

  D0.u32 = v_max_u32(v_max_u32(S0.u32, S1.u32), S2.u32)

V_MED3_F32                                                                                                      543

Select the median of three single-precision float values and store the selected value into a vector register.

  if (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)) || isNAN(64'F(S2.f32))) then
      D0.f32 = v_min3_f32(S0.f32, S1.f32, S2.f32)
  elsif v_max3_f32(S0.f32, S1.f32, S2.f32) == S0.f32 then
      D0.f32 = v_max_f32(S1.f32, S2.f32)
  elsif v_max3_f32(S0.f32, S1.f32, S2.f32) == S1.f32 then
      D0.f32 = v_max_f32(S0.f32, S2.f32)
  else
      D0.f32 = v_max_f32(S0.f32, S1.f32)
  endif

V_MED3_I32                                                                                                      544

Select the median of three signed 32-bit integer values and store the selected value into a vector register.

  if v_max3_i32(S0.i32, S1.i32, S2.i32) == S0.i32 then
      D0.i32 = v_max_i32(S1.i32, S2.i32)
  elsif v_max3_i32(S0.i32, S1.i32, S2.i32) == S1.i32 then
      D0.i32 = v_max_i32(S0.i32, S2.i32)
  else
      D0.i32 = v_max_i32(S0.i32, S1.i32)
  endif

V_MED3_U32                                                                                                      545

Select the median of three unsigned 32-bit integer values and store the selected value into a vector register.

  if v_max3_u32(S0.u32, S1.u32, S2.u32) == S0.u32 then
        D0.u32 = v_max_u32(S1.u32, S2.u32)
  elsif v_max3_u32(S0.u32, S1.u32, S2.u32) == S1.u32 then
        D0.u32 = v_max_u32(S0.u32, S2.u32)
  else
        D0.u32 = v_max_u32(S0.u32, S1.u32)
  endif

V_SAD_U8                                                                                                         546

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer
inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

  ABSDIFF = lambda(x, y) (
        x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += 32'U(ABSDIFF(S0.u32[7 : 0], S1.u32[7 : 0]));
  tmp += 32'U(ABSDIFF(S0.u32[15 : 8], S1.u32[15 : 8]));
  tmp += 32'U(ABSDIFF(S0.u32[23 : 16], S1.u32[23 : 16]));
  tmp += 32'U(ABSDIFF(S0.u32[31 : 24], S1.u32[31 : 24]));
  D0.u32 = tmp

Notes

Overflow into the upper bits is allowed.

V_SAD_HI_U8                                                                                                      547

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer
inputs, shift the sum left by 16 bits, add an unsigned 32-bit integer value from the third input and store the
result into a vector register.

  D0.u32 = (32'U(v_sad_u8(S0, S1, 0U)) << 16U) + S2.u32

Notes

Overflow into the upper bits is allowed.

V_SAD_U16                                                                                                        548

Calculate the sum of absolute differences of elements in two packed 2-component unsigned 16-bit integer

inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += ABSDIFF(S0[15 : 0].u16, S1[15 : 0].u16);
  tmp += ABSDIFF(S0[31 : 16].u16, S1[31 : 16].u16);
  D0.u32 = tmp

V_SAD_U32                                                                                                        549

Calculate the absolute difference of two unsigned 32-bit integer inputs, add an unsigned 32-bit integer value
from the third input and store the result into a vector register.

  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u32 = ABSDIFF(S0.u32, S1.u32) + S2.u32

V_CVT_PK_U8_F32                                                                                                  550

Convert a single-precision float value from the first input to an unsigned 8-bit integer value and pack the result
into one byte of the third input using the second input as a byte select. Store the result into a vector register.

  tmp = (S2.u32 & 32'U(~(0xff << (S1.u32[1 : 0].u32 * 8U))));
  tmp = (tmp | ((32'U(f32_to_u8(S0.f32)) & 255U) << (S1.u32[1 : 0].u32 * 8U)));
  D0.u32 = tmp

V_DIV_FIXUP_F32                                                                                                  551

Given a single-precision float quotient in the first input, a denominator in the second input and a numerator in
the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

  sign_out = (sign(S1.f32) ^ sign(S2.f32));
  if isNAN(64'F(S2.f32)) then
      D0.f32 = 32'F(cvtToQuietNAN(64'F(S2.f32)))

  elsif isNAN(64'F(S1.f32)) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
  elsif ((64'F(S1.f32) == 0.0) && (64'F(S2.f32) == 0.0)) then
        // 0/0
        D0.f32 = 32'F(0xffc00000)
  elsif ((64'F(abs(S1.f32)) == +INF) && (64'F(abs(S2.f32)) == +INF)) then
        // inf/inf
        D0.f32 = 32'F(0xffc00000)
  elsif ((64'F(S1.f32) == 0.0) || (64'F(abs(S2.f32)) == +INF)) then
        // x/0, or inf/y
        D0.f32 = sign_out ? -INF.f32 : +INF.f32
  elsif ((64'F(abs(S1.f32)) == +INF) || (64'F(S2.f32) == 0.0)) then
        // x/inf, 0/y
        D0.f32 = sign_out ? -0.0F : 0.0F
  elsif exponent(S2.f32) - exponent(S1.f32) < -150 then
        D0.f32 = sign_out ? -UNDERFLOW_F32 : UNDERFLOW_F32
  elsif exponent(S1.f32) == 255 then
        D0.f32 = sign_out ? -OVERFLOW_F32 : OVERFLOW_F32
  else
        D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)
  endif

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

V_DIV_FIXUP_F64                                                                                               552

Given a double-precision float quotient in the first input, a denominator in the second input and a numerator
in the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

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

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

V_DIV_FMAS_F32                                                                                                    567

Multiply two single-precision float inputs and add a third input using fused multiply add, then scale the
exponent of the result by a fixed factor if the vector condition code is set. Store the result into a vector register.

This operation is designed for use in floating point division macros and relies on V_DIV_SCALE_F32 to set the
vector condition code iff the quotient requires post-scaling.

  if VCC.u64[laneId] then
        D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)
  else
        D0.f32 = fma(S0.f32, S1.f32, S2.f32)
  endif

Notes

Input denormals are not flushed but output flushing is allowed.

V_DIV_SCALE_F32, V_DIV_FMAS_F32 and V_DIV_FIXUP_F32 are all designed for use in a high precision
division macro that utilizes V_RCP_F32 and V_MUL_F32 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

V_DIV_FMAS_F64                                                                                                    568

Multiply two double-precision float inputs and add a third input using fused multiply add, then scale the
exponent of the result by a fixed factor if the vector condition code is set. Store the result into a vector register.

This operation is designed for use in floating point division macros and relies on V_DIV_SCALE_F64 to set the
vector condition code iff the quotient requires post-scaling.

  if VCC.u64[laneId] then
        D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)

  else
        D0.f64 = fma(S0.f64, S1.f64, S2.f64)
  endif

Notes

Input denormals are not flushed but output flushing is allowed.

V_DIV_SCALE_F64, V_DIV_FMAS_F64 and V_DIV_FIXUP_F64 are all designed for use in a high precision
division macro that utilizes V_RCP_F64 and V_MUL_F64 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

V_MSAD_U8                                                                                                      569

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer
inputs, except that elements where the second input (known as the reference input) is zero are not included in
the sum. Add an unsigned 32-bit integer value from the third input and store the result into a vector register.

  ABSDIFF = lambda(x, y) (
        x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += S1.u32[7 : 0] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[7 : 0], S1.u32[7 : 0]));
  tmp += S1.u32[15 : 8] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[15 : 8], S1.u32[15 : 8]));
  tmp += S1.u32[23 : 16] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[23 : 16], S1.u32[23 : 16]));
  tmp += S1.u32[31 : 24] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[31 : 24], S1.u32[31 : 24]));
  D0.u32 = tmp

Notes

Overflow into the upper bits is allowed.

V_QSAD_PK_U16_U8                                                                                               570

Perform the V_SAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Truncate each result to 16 bits, pack the values into a 4-entry array and
store the array into a vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second
input is a 4-entry array of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 16-bit
integers.

  tmp[63 : 48] = 16'B(v_sad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u32));
  tmp[47 : 32] = 16'B(v_sad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u32));
  tmp[31 : 16] = 16'B(v_sad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u32));
  tmp[15 : 0] = 16'B(v_sad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u32));

  D0.b64 = tmp.b64

V_MQSAD_PK_U16_U8                                                                                                571

Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Truncate each result to 16 bits, pack the values into a 4-entry array and
store the array into a vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second
input is a 4-entry array of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 16-bit
integers.

  tmp[63 : 48] = 16'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u32));
  tmp[47 : 32] = 16'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u32));
  tmp[31 : 16] = 16'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u32));
  tmp[15 : 0] = 16'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u32));
  D0.b64 = tmp.b64

V_MQSAD_U32_U8                                                                                                   573

Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Pack each 32-bit value into a 4-entry array and store the array into a
vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second input is a 4-entry array
of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 32-bit integers.

  tmp[127 : 96] = 32'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[127 : 96].u32));
  tmp[95 : 64] = 32'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[95 : 64].u32));
  tmp[63 : 32] = 32'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[63 : 32].u32));
  tmp[31 : 0] = 32'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[31 : 0].u32));
  D0.b128 = tmp.b128

V_XOR3_B32                                                                                                       576

Calculate the bitwise XOR of three vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 ^ S1.u32 ^ S2.u32)

Notes

Input and output modifiers not supported.

V_MAD_U16                                                                                                        577

Multiply two unsigned 16-bit integer inputs, add an unsigned 16-bit integer value from a third input, and store
the result into a vector register.

  D0.u16 = S0.u16 * S1.u16 + S2.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_PERM_B32                                                                                                        580

Permute a 64-bit value constructed from two vector inputs (most significant bits come from the first input)
using a per-lane selector from the third input. The lane selector allows each byte of the result to choose from
any of the 8 input bytes, perform sign extension or pad with 0/1 bits. Store the result into a vector register.

  BYTE_PERMUTE = lambda(data, sel) (
        declare in : 8'B[8];
        for i in 0 : 7 do
            in[i] = data[i * 8 + 7 : i * 8].b8
        endfor;
        if sel.u32 >= 13U then
            return 8'0xff
        elsif sel.u32 == 12U then
            return 8'0x0
        elsif sel.u32 == 11U then
            return in[7][7].b8 * 8'0xff
        elsif sel.u32 == 10U then
            return in[5][7].b8 * 8'0xff
        elsif sel.u32 == 9U then
            return in[3][7].b8 * 8'0xff
        elsif sel.u32 == 8U then
            return in[1][7].b8 * 8'0xff
        else
            return in[sel]
        endif);
  D0[31 : 24] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[31 : 24]);
  D0[23 : 16] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[23 : 16]);
  D0[15 : 8] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[15 : 8]);
  D0[7 : 0] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[7 : 0])

Notes

Selects 0 through 7 select the corresponding byte of the 64-bit input value.

Selects 8 through 11 are useful in modeling sign extension of a smaller-precision signed integer to a larger-
precision result by replicating the leading bit of a selected byte.

Selects 12 and 13 return padding values of 0 and 1 bits respectively.

Note the MSBs of the 64-bit value being selected are stored in S0. This is counterintuitive for a little-endian

architecture.

V_XAD_U32                                                                                                          581

Calculate bitwise XOR of the first two vector inputs, then add the third vector input to the intermediate result,
then store the final result into a vector register.

  D0.u32 = (S0.u32 ^ S1.u32) + S2.u32

Notes

No carryin/carryout and no saturation. This opcode is designed to help accelerate the SHA256 hash algorithm.

V_LSHL_ADD_U32                                                                                                     582

Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input
to the intermediate result, then store the final result into a vector register.

  D0.u32 = (S0.u32 << S1.u32[4 : 0].u32) + S2.u32

V_ADD_LSHL_U32                                                                                                     583

Add the first two integer inputs, then given a shift count in the third input, calculate the logical shift left of the
intermediate result, then store the final result into a vector register.

  D0.u32 = ((S0.u32 + S1.u32) << S2.u32[4 : 0].u32)

V_FMA_F16                                                                                                          584

Multiply two half-precision float inputs and add a third input using fused multiply add, and store the result into
a vector register.

  D0.f16 = fma(S0.f16, S1.f16, S2.f16)

Notes

0.5ULP accuracy, denormals are supported.

V_MIN3_F16                                                                                                      585

Select the minimum of three half-precision float inputs and store the selected value into a vector register.

  D0.f16 = v_min_f16(v_min_f16(S0.f16, S1.f16), S2.f16)

V_MIN3_I16                                                                                                      586

Select the minimum of three signed 16-bit integer inputs and store the selected value into a vector register.

  D0.i16 = v_min_i16(v_min_i16(S0.i16, S1.i16), S2.i16)

V_MIN3_U16                                                                                                      587

Select the minimum of three unsigned 16-bit integer inputs and store the selected value into a vector register.

  D0.u16 = v_min_u16(v_min_u16(S0.u16, S1.u16), S2.u16)

V_MAX3_F16                                                                                                      588

Select the maximum of three half-precision float inputs and store the selected value into a vector register.

  D0.f16 = v_max_f16(v_max_f16(S0.f16, S1.f16), S2.f16)

V_MAX3_I16                                                                                                      589

Select the maximum of three signed 16-bit integer inputs and store the selected value into a vector register.

  D0.i16 = v_max_i16(v_max_i16(S0.i16, S1.i16), S2.i16)

V_MAX3_U16                                                                                                      590

Select the maximum of three unsigned 16-bit integer inputs and store the selected value into a vector register.

  D0.u16 = v_max_u16(v_max_u16(S0.u16, S1.u16), S2.u16)

V_MED3_F16                                                                                                       591

Select the median of three half-precision float values and store the selected value into a vector register.

  if (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)) || isNAN(64'F(S2.f16))) then
      D0.f16 = v_min3_f16(S0.f16, S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S0.f16 then
      D0.f16 = v_max_f16(S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S1.f16 then
      D0.f16 = v_max_f16(S0.f16, S2.f16)
  else
      D0.f16 = v_max_f16(S0.f16, S1.f16)
  endif

V_MED3_I16                                                                                                       592

Select the median of three signed 16-bit integer values and store the selected value into a vector register.

  if v_max3_i16(S0.i16, S1.i16, S2.i16) == S0.i16 then
      D0.i16 = v_max_i16(S1.i16, S2.i16)
  elsif v_max3_i16(S0.i16, S1.i16, S2.i16) == S1.i16 then
      D0.i16 = v_max_i16(S0.i16, S2.i16)
  else
      D0.i16 = v_max_i16(S0.i16, S1.i16)
  endif

V_MED3_U16                                                                                                       593

Select the median of three unsigned 16-bit integer values and store the selected value into a vector register.

  if v_max3_u16(S0.u16, S1.u16, S2.u16) == S0.u16 then
      D0.u16 = v_max_u16(S1.u16, S2.u16)
  elsif v_max3_u16(S0.u16, S1.u16, S2.u16) == S1.u16 then
      D0.u16 = v_max_u16(S0.u16, S2.u16)
  else
      D0.u16 = v_max_u16(S0.u16, S1.u16)
  endif

V_MAD_I16                                                                                                        595

Multiply two signed 16-bit integer inputs, add a signed 16-bit integer value from a third input, and store the
result into a vector register.

  D0.i16 = S0.i16 * S1.i16 + S2.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_DIV_FIXUP_F16                                                                                                  596

Given a half-precision float quotient in the first input, a denominator in the second input and a numerator in
the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

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

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

V_ADD3_U32                                                                                                       597

Add three unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.u32 = S0.u32 + S1.u32 + S2.u32

V_LSHL_OR_B32                                                                                                      598

Given a shift count in the second input, calculate the logical shift left of the first input, then calculate the
bitwise OR of the intermediate result and the third input, then store the final result into a vector register.

  D0.u32 = ((S0.u32 << S1.u32[4 : 0].u32) | S2.u32)

V_AND_OR_B32                                                                                                       599

Calculate bitwise AND on the first two vector inputs, then compute the bitwise OR of the intermediate result
and the third vector input, then store the final result into a vector register.

  D0.u32 = ((S0.u32 & S1.u32) | S2.u32)

Notes

Input and output modifiers not supported.

V_OR3_B32                                                                                                          600

Calculate the bitwise OR of three vector inputs and store the result into a vector register.

  D0.u32 = (S0.u32 | S1.u32 | S2.u32)

Notes

Input and output modifiers not supported.

V_MAD_U32_U16                                                                                                      601

Multiply two unsigned 16-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit
integer value from a third input, and store the result as an unsigned 32-bit integer into a vector register.

  D0.u32 = 32'U(S0.u16) * 32'U(S1.u16) + S2.u32

V_MAD_I32_I16                                                                                                 602

Multiply two signed 16-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value
from a third input, and store the result as a signed 32-bit integer into a vector register.

  D0.i32 = 32'I(S0.i16) * 32'I(S1.i16) + S2.i32

V_PERMLANE16_B32                                                                                              603

Perform arbitrary gather-style operation within a row (16 contiguous lanes).

The first source must be a VGPR and the second and third sources must be scalar values; the second and third
source are combined into a single 64-bit value representing lane selects used to swizzle within each row.

OPSEL is not used in its typical manner for this instruction. For this instruction OPSEL[0] is overloaded to
represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit.
The remaining OPSEL bits are reserved for this instruction.

Compare with V_PERMLANEX16_B32.

  declare tmp : 32'B[64];
  lanesel = { S2.u32, S1.u32 };
  // Concatenate lane select bits
  for i in 0 : WAVE32 ? 31 : 63 do
        // Copy original S0 in case D==S0
        tmp[i] = VGPR[i][SRC0.u32]
  endfor;
  for row in 0 : WAVE32 ? 1 : 3 do
        // Implement arbitrary swizzle within each row
        for i in 0 : 15 do
            if EXEC[row * 16 + i].u1 then
                 VGPR[row * 16 + i][VDST.u32] = tmp[64'B(row * 16) + lanesel[i * 4 + 3 : i * 4]]
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

V_PERMLANEX16_B32                                                                                              604

Perform arbitrary gather-style operation across two rows (each row is 16 contiguous lanes).

The first source must be a VGPR and the second and third sources must be scalar values; the second and third
source are combined into a single 64-bit value representing lane selects used to swizzle within each row.

OPSEL is not used in its typical manner for this instruction. For this instruction OPSEL[0] is overloaded to
represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit.
The remaining OPSEL bits are reserved for this instruction.

Compare with V_PERMLANE16_B32.

  declare tmp : 32'B[64];
  lanesel = { S2.u32, S1.u32 };
  // Concatenate lane select bits
  for i in 0 : WAVE32 ? 31 : 63 do
        // Copy original S0 in case D==S0
        tmp[i] = VGPR[i][SRC0.u32]
  endfor;
  for row in 0 : WAVE32 ? 1 : 3 do
        // Implement arbitrary swizzle across two rows
        altrow = { row[1], ~row[0] };
        // 1<->0, 3<->2
        for i in 0 : 15 do
            if EXEC[row * 16 + i].u1 then
                 VGPR[row * 16 + i][VDST.u32] = tmp[64'B(altrow.i32 * 16) + lanesel[i * 4 + 3 : i * 4]]
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

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

  D0.u16 = VCC.u64[laneId] ? S1.u16 : S0.u16

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 16-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

V_MAXMIN_F32                                                                                                       606

Select the maximum of the first two single-precision float inputs and then select the minimum of that result
and third single-precision float input. Store the final result into a vector register.

  D0.f32 = v_min_f32(v_max_f32(S0.f32, S1.f32), S2.f32)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MINMAX_F32                                                                                                   607

Select the minimum of the first two single-precision float inputs and then select the maximum of that result
and third single-precision float input. Store the final result into a vector register.

  D0.f32 = v_max_f32(v_min_f32(S0.f32, S1.f32), S2.f32)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MAXMIN_F16                                                                                                   608

Select the maximum of the first two half-precision float inputs and then select the minimum of that result and
third half-precision float input. Store the final result into a vector register.

  D0.f16 = v_min_f16(v_max_f16(S0.f16, S1.f16), S2.f16)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MINMAX_F16                                                                                                   609

Select the minimum of the first two half-precision float inputs and then select the maximum of that result and
third half-precision float input. Store the final result into a vector register.

  D0.f16 = v_max_f16(v_min_f16(S0.f16, S1.f16), S2.f16)

Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.

V_MAXMIN_U32                                                                                                   610

Select the maximum of the first two unsigned 32-bit integer inputs and then select the minimum of that result
and third unsigned 32-bit integer input. Store the final result into a vector register.

  D0.u32 = v_min_u32(v_max_u32(S0.u32, S1.u32), S2.u32)

V_MINMAX_U32                                                                                                  611

Select the minimum of the first two unsigned 32-bit integer inputs and then select the maximum of that result
and third unsigned 32-bit integer input. Store the final result into a vector register.

  D0.u32 = v_max_u32(v_min_u32(S0.u32, S1.u32), S2.u32)

V_MAXMIN_I32                                                                                                  612

Select the maximum of the first two signed 32-bit integer inputs and then select the minimum of that result and
third signed 32-bit integer input. Store the final result into a vector register.

  D0.i32 = v_min_i32(v_max_i32(S0.i32, S1.i32), S2.i32)

V_MINMAX_I32                                                                                                  613

Select the minimum of the first two signed 32-bit integer inputs and then select the maximum of that result and
third signed 32-bit integer input. Store the final result into a vector register.

  D0.i32 = v_max_i32(v_min_i32(S0.i32, S1.i32), S2.i32)

V_DOT2_F16_F16                                                                                                614

Compute the dot product of two packed 2-D half-precision float inputs, add the third input and store the result
into a vector register.

  tmp = S2.f16;
  tmp += S0[15 : 0].f16 * S1[15 : 0].f16;
  tmp += S0[31 : 16].f16 * S1[31 : 16].f16;
  D0.f16 = tmp

Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are
ignored.

V_DOT2_BF16_BF16                                                                                               615

Compute the dot product of two packed 2-D BF16 float inputs, add the third input and store the result into a
vector register.

  tmp = S2.bf16;
  tmp += S0[15 : 0].bf16 * S1[15 : 0].bf16;
  tmp += S0[31 : 16].bf16 * S1[31 : 16].bf16;
  D0.bf16 = tmp

Notes

OPSEL[2] controls which half of S2 is read and OPSEL[3] controls which half of D is written; OPSEL[1:0] are
ignored.

V_DIV_SCALE_F32                                                                                                764

Given a single-precision float value to scale in the first input, a denominator in the second input and a
numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing
during application of the Newton-Raphson correction method. Store the scaled result into a vector register and
set the vector condition code iff post-scaling is required.

This operation is designed for use in a high precision division macro. The first input should be the same value
as either the second or third input; other scale values produce predictable results but may not be
mathematically useful. The vector condition code is used by V_DIV_FMAS_F32 to determine if the quotient
requires post-scaling.

  VCC = 0x0LL;
  if ((64'F(S2.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        D0.f32 = NAN.f32
  elsif exponent(S2.f32) - exponent(S1.f32) >= 96 then
        // N/D near MAX_FLOAT_F32
        VCC = 0x1LL;
        if S0.f32 == S1.f32 then
            // Only scale the denominator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif S1.f32 == DENORM.f32 then
        D0.f32 = ldexp(S0.f32, 64)
  elsif ((1.0 / 64'F(S1.f32) == DENORM.f64) && (S2.f32 / S1.f32 == DENORM.f32)) then
        VCC = 0x1LL;
        if S0.f32 == S1.f32 then
            // Only scale the denominator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif 1.0 / 64'F(S1.f32) == DENORM.f64 then
        D0.f32 = ldexp(S0.f32, -64)
  elsif S2.f32 / S1.f32 == DENORM.f32 then
        VCC = 0x1LL;
        if S0.f32 == S2.f32 then

            // Only scale the numerator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif exponent(S2.f32) <= 23 then
        // Numerator is tiny
        D0.f32 = ldexp(S0.f32, 64)
  endif

Notes

V_DIV_SCALE_F32, V_DIV_FMAS_F32 and V_DIV_FIXUP_F32 are all designed for use in a high precision
division macro that utilizes V_RCP_F32 and V_MUL_F32 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

V_DIV_SCALE_F64                                                                                               765

Given a double-precision float value to scale in the first input, a denominator in the second input and a
numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing
during application of the Newton-Raphson correction method. Store the scaled result into a vector register and
set the vector condition code iff post-scaling is required.

This operation is designed for use in a high precision division macro. The first input should be the same value
as either the second or third input; other scale values produce predictable results but may not be
mathematically useful. The vector condition code is used by V_DIV_FMAS_F64 to determine if the quotient
requires post-scaling.

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

Notes

V_DIV_SCALE_F64, V_DIV_FMAS_F64 and V_DIV_FIXUP_F64 are all designed for use in a high precision
division macro that utilizes V_RCP_F64 and V_MUL_F64 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

V_MAD_U64_U32                                                                                                   766

Multiply two unsigned integer inputs, add a third unsigned integer input, store the result into a 64-bit vector
register and store the overflow/carryout into a scalar mask register.

  { D1.u1, D0.u64 } = 65'B(65'U(S0.u32) * 65'U(S1.u32) + 65'U(S2.u64))

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

V_MAD_I64_I32                                                                                                   767

Multiply two signed integer inputs, add a third signed integer input, store the result into a 64-bit vector register
and store the overflow/carryout into a scalar mask register.

  { D1.i1, D0.i64 } = 65'B(65'I(S0.i32) * 65'I(S1.i32) + 65'I(S2.i64))

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

V_ADD_CO_U32                                                                                                    768

Add two unsigned 32-bit integer inputs, store the result into a vector register and store the carry-out mask into
a scalar register.

  tmp = 64'U(S0.u32) + 64'U(S1.u32);

  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_ADD_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_SUB_CO_U32                                                                                                    769

Subtract the second unsigned 32-bit integer input from the first input, store the result into a vector register and
store the carry-out mask into a scalar register.

  tmp = S0.u32 - S1.u32;
  VCC.u64[laneId] = S1.u32 > S0.u32 ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_SUBREV_CO_U32                                                                                                 770

Subtract the first unsigned 32-bit integer input from the second input, store the result into a vector register and
store the carry-out mask into a scalar register.

  tmp = S1.u32 - S0.u32;
  VCC.u64[laneId] = S0.u32 > S1.u32 ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUB_CO_CI_U32.
  D0.u32 = tmp.u32

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

V_ADD_NC_U16                                                                                                    771

Add two unsigned 16-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

  D0.u16 = S0.u16 + S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_SUB_NC_U16                                                                                                      772

Subtract the second unsigned 16-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

  D0.u16 = S0.u16 - S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_MUL_LO_U16                                                                                                      773

Multiply two unsigned 16-bit integer inputs and store the low bits of the result into a vector register.

  D0.u16 = S0.u16 * S1.u16

Notes

Supports saturation (unsigned 16-bit integer domain).

V_CVT_PK_I16_F32                                                                                                  774

Convert two single-precision float inputs into a packed signed 16-bit integer value and store the result into a
vector register.

  declare tmp : 32'B;
  tmp[31 : 16] = 16'B(v_cvt_i16_f32(S1.f32));
  tmp[15 : 0] = 16'B(v_cvt_i16_f32(S0.f32));
  D0 = tmp.b32

V_CVT_PK_U16_F32                                                                                              775

Convert two single-precision float inputs into a packed unsigned 16-bit integer value and store the result into a
vector register.

  declare tmp : 32'B;
  tmp[31 : 16] = 16'B(v_cvt_u16_f32(S1.f32));
  tmp[15 : 0] = 16'B(v_cvt_u16_f32(S0.f32));
  D0 = tmp.b32

V_MAX_U16                                                                                                     777

Select the maximum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

  D0.u16 = S0.u16 >= S1.u16 ? S0.u16 : S1.u16

V_MAX_I16                                                                                                     778

Select the maximum of two signed 16-bit integer inputs and store the selected value into a vector register.

  D0.i16 = S0.i16 >= S1.i16 ? S0.i16 : S1.i16

V_MIN_U16                                                                                                     779

Select the minimum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

  D0.u16 = S0.u16 < S1.u16 ? S0.u16 : S1.u16

V_MIN_I16                                                                                                     780

Select the minimum of two signed 16-bit integer inputs and store the selected value into a vector register.

  D0.i16 = S0.i16 < S1.i16 ? S0.i16 : S1.i16

V_ADD_NC_I16                                                                                                  781

Add two signed 16-bit integer inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.i16 = S0.i16 + S1.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_SUB_NC_I16                                                                                                        782

Subtract the second signed 16-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

  D0.i16 = S0.i16 - S1.i16

Notes

Supports saturation (signed 16-bit integer domain).

V_PACK_B32_F16                                                                                                      785

Pack two half-precision float values into a single 32-bit value and store the result into a vector register.

  D0[31 : 16].f16 = S1.f16;
  D0[15 : 0].f16 = S0.f16

V_CVT_PK_NORM_I16_F16                                                                                               786

Convert from two half-precision float inputs to a packed signed normalized short and store the result into a
vector register.

  declare tmp : 32'B;
  tmp[15 : 0].i16 = f16_to_snorm(S0.f16);
  tmp[31 : 16].i16 = f16_to_snorm(S1.f16);
  D0 = tmp.b32

V_CVT_PK_NORM_U16_F16                                                                                               787

Convert from two half-precision float inputs to a packed unsigned normalized short and store the result into a

vector register.

  declare tmp : 32'B;
  tmp[15 : 0].u16 = f16_to_unorm(S0.f16);
  tmp[31 : 16].u16 = f16_to_unorm(S1.f16);
  D0 = tmp.b32

V_LDEXP_F32                                                                                                         796

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

  D0.f32 = S0.f32 * 2.0F ** S1.i32

Notes

Compare with the ldexp() function in C.

V_BFM_B32                                                                                                           797

Calculate a bitfield mask given a field offset and size and store the result into a vector register.

  D0.u32 = (((1U << S0[4 : 0].u32) - 1U) << S1[4 : 0].u32)

V_BCNT_U32_B32                                                                                                      798

Count the number of "1" bits in the vector input and store the result into a vector register.

  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += S0[i].u32;
        // count i'th bit
  endfor;
  D0.u32 = tmp

V_MBCNT_LO_U32_B32                                                                                                  799

For each lane 0 <= N < 32, examine the N least significant bits of the first input and count how many of those
bits are "1". For each lane 32 <= N < 64, all "1" bits in the first input are counted. Add this count to the value in
the second input and store the result into a vector register.

In conjunction with V_MBCNT_HI_U32_B32 and with a vector condition code as input, this counts the number
of lanes at or below the current lane number that have set their vector condition code bit.

  ThreadMask = (1LL << laneId.u32) - 1LL;
  MaskedValue = (S0.u32 & ThreadMask[31 : 0].u32);
  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor;
  D0.u32 = tmp

Notes

See also V_MBCNT_HI_U32_B32.

V_MBCNT_HI_U32_B32                                                                                            800

For each lane 32 <= N < 64, examine the N least significant bits of the first input and count how many of those
bits are "1". For lane positions 0 <= N < 32 no bits are examined and the count is zero. Add this count to the
value in the second input and store the result into a vector register.

In conjunction with V_MBCNT_LO_U32_B32 and with a vector condition code as input, this counts the number
of lanes at or below the current lane number that have set their vector condition code bit.

  ThreadMask = (1LL << laneId.u32) - 1LL;
  MaskedValue = (S0.u32 & ThreadMask[63 : 32].u32);
  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor;
  D0.u32 = tmp

Notes

Example to compute each lane's position in 0..63:

        v_mbcnt_lo_u32_b32 v0, -1, 0
        v_mbcnt_hi_u32_b32 v0, -1, v0
        // v0 now contains laneId

Example to compute each lane's position in a list of all lanes whose VCC bits are set, where the first lane with
VCC set is assigned position 1, the second lane with VCC set is assigned position 2, etc.:

        v_mbcnt_lo_u32_b32 v0, vcc_lo, 0
        v_mbcnt_hi_u32_b32 v0, vcc_hi, v0 // Note vcc_hi is passed in for second instruction
        // v0 now contains position among lanes with VCC=1

See also V_MBCNT_LO_U32_B32.

V_CVT_PK_NORM_I16_F32                                                                                         801

Convert from two single-precision float inputs to a packed signed normalized short and store the result into a
vector register.

  declare tmp : 32'B;
  tmp[15 : 0].i16 = f32_to_snorm(S0.f32);
  tmp[31 : 16].i16 = f32_to_snorm(S1.f32);
  D0 = tmp.b32

V_CVT_PK_NORM_U16_F32                                                                                         802

Convert from two single-precision float inputs to a packed unsigned normalized short and store the result into
a vector register.

  declare tmp : 32'B;
  tmp[15 : 0].u16 = f32_to_unorm(S0.f32);
  tmp[31 : 16].u16 = f32_to_unorm(S1.f32);
  D0 = tmp.b32

V_CVT_PK_U16_U32                                                                                              803

Convert from two unsigned 32-bit integer inputs to a packed unsigned 16-bit integer value and store the result
into a vector register.

  declare tmp : 32'B;
  tmp[15 : 0].u16 = u32_to_u16(S0.u32);
  tmp[31 : 16].u16 = u32_to_u16(S1.u32);
  D0 = tmp.b32

V_CVT_PK_I16_I32                                                                                              804

Convert from two signed 32-bit integer inputs to a packed signed 16-bit integer value and store the result into a
vector register.

  declare tmp : 32'B;
  tmp[15 : 0].i16 = i32_to_i16(S0.i32);
  tmp[31 : 16].i16 = i32_to_i16(S1.i32);

  D0 = tmp.b32

V_SUB_NC_I32                                                                                                    805

Subtract the second signed 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

  D0.i32 = S0.i32 - S1.i32

Notes

Supports saturation (signed 32-bit integer domain).

V_ADD_NC_I32                                                                                                    806

Add two signed 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out support.

  D0.i32 = S0.i32 + S1.i32

Notes

Supports saturation (signed 32-bit integer domain).

V_ADD_F64                                                                                                       807

Add two floating point inputs and store the result into a vector register.

  D0.f64 = S0.f64 + S1.f64

Notes

0.5ULP precision, denormals are supported.

V_MUL_F64                                                                                                       808

Multiply two floating point inputs and store the result into a vector register.

  D0.f64 = S0.f64 * S1.f64

Notes

0.5ULP precision, denormals are supported.

V_MIN_F64                                                                                                   809

Select the minimum of two double-precision float inputs and store the result into a vector register.

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

V_MAX_F64                                                                                                       810

Select the maximum of two double-precision float inputs and store the result into a vector register.

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
integer value, and store the floating point result into a vector register.

  D0.f64 = S0.f64 * 2.0 ** S1.i32

Notes

Compare with the ldexp() function in C.

V_MUL_LO_U32                                                                                                      812

Multiply two unsigned 32-bit integer inputs and store the result into a vector register.

  D0.u32 = S0.u32 * S1.u32

Notes

To multiply integers with small magnitudes consider V_MUL_U32_U24, which is intended to be a more
efficient implementation.

V_MUL_HI_U32                                                                                                      813

Multiply two unsigned 32-bit integer inputs and store the high 32 bits of the result into a vector register.

  D0.u32 = 32'U((64'U(S0.u32) * 64'U(S1.u32)) >> 32U)

Notes

To multiply integers with small magnitudes consider V_MUL_HI_U32_U24, which is intended to be a more
efficient implementation.

V_MUL_HI_I32                                                                                                      814

Multiply two signed 32-bit integer inputs and store the high 32 bits of the result into a vector register.

  D0.i32 = 32'I((64'I(S0.i32) * 64'I(S1.i32)) >> 32U)

Notes

To multiply integers with small magnitudes consider V_MUL_HI_I32_I24, which is intended to be a more
efficient implementation.

V_TRIG_PREOP_F64                                                                                                   815

Look up a 53-bit segment of 2/PI using an integer segment select in the second input. Scale the intermediate
result by the exponent from the first double-precision float input and store the double-precision float result
into a vector register.

This operation returns an aligned, double precision segment of 2/PI needed to do trigonometric argument
reduction on the floating point input. Multiple segments can be accessed using the first input. Rounding is
toward zero. Large floating point inputs (with an exponent > 1968) are scaled to avoid loss of precision through
denormalization.

  shift = 32'I(S1[4 : 0].u32) * 53;
  if exponent(S0.f64) > 1077 then
        shift += exponent(S0.f64) - 1077
  endif;
  // (2.0/PI) == 0.{b_1200, b_1199, b_1198, ..., b_1, b_0}
  // b_1200 is the MSB of the fractional part of 2.0/PI
  // Left shift operation indicates which bits are brought
  // into the whole part of the number.
  // Only whole part of result is kept.
  result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff);
  scale = -53 - shift;
  if exponent(S0.f64) >= 1968 then
        scale += 128
  endif;
  D0.f64 = ldexp(result, scale)

Notes

For a more complete treatment of trigonometric argument reduction refer to Argument Reduction for Huge
Arguments: Good to the Last Bit, K. C. Ng et.al., March 1992, available online.

V_LSHLREV_B16                                                                                                      824

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u16 = (S1.u16 << S0[3 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_LSHRREV_B16                                                                                                      825

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u16 = (S1.u16 >> S0[3 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_ASHRREV_I16                                                                                                      826

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i16 = (S1.i16 >> S0[3 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted.

V_LSHLREV_B64                                                                                                      828

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

  D0.u64 = (S1.u64 << S0[5 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted. Only one scalar broadcast constant is allowed.

V_LSHRREV_B64                                                                                                      829

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

  D0.u64 = (S1.u64 >> S0[5 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted. Only one scalar broadcast constant is allowed.

V_ASHRREV_I64                                                                                                      830

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

  D0.i64 = (S1.i64 >> S0[5 : 0].u32)

Notes

DPP operates on the shift count, not the data being shifted. Only one scalar broadcast constant is allowed.

V_READLANE_B32                                                                                                     864

Read the scalar value in the specified lane of the first input where the lane select is in the second input. Store
the result into a scalar register.

  declare lane : 32'U;
  if WAVE32 then
        lane = S1.u32[4 : 0].u32;
        // Lane select for wave32
  else
        lane = S1.u32[5 : 0].u32;
        // Lane select for wave64
  endif;
  D0.b32 = VGPR[lane][SRC0.u32]

Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped
operation.

V_WRITELANE_B32                                                                                                    865

Write the scalar value in the first input into the specified lane of a vector register where the lane select is in the
second input.

  declare lane : 32'U;
  if WAVE32 then
        lane = S1.u32[4 : 0].u32;
        // Lane select for wave32
  else
        lane = S1.u32[5 : 0].u32;
        // Lane select for wave64
  endif;
  VGPR[lane][VDST.u32] = S0.b32

Notes

Overrides EXEC mask for the VGPR write. Input and output modifiers not supported; this is an untyped
operation.

V_AND_B16                                                                                               866

Calculate bitwise AND on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 & S1.u16)

Notes

Input and output modifiers not supported.

V_OR_B16                                                                                                867

Calculate bitwise OR on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 | S1.u16)

Notes

Input and output modifiers not supported.

V_XOR_B16                                                                                               868

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

  D0.u16 = (S0.u16 ^ S1.u16)

Notes

Input and output modifiers not supported.

V_CMP_F_F16                                                                                                0

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;

  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F16                                                                                                           1

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

V_CMP_GT_F16                                                                                                        4

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

V_CMP_U_F16                                                                                                            8

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

V_CMP_NGT_F16                                                                                                       11

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

V_CMP_NLT_F16                                                                                                      14

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

V_CMP_EQ_F32                                                                                                           18

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

V_CMP_LG_F32                                                                                                        21

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f32 <> S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F32                                                                                                        22

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f32 >= S1.f32;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_O_F32                                                                                                         23

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_U_F32                                                                                                         24

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

V_CMP_NLG_F32                                                                                                      26

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NGT_F32                                                                                                      27

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

V_CMP_NEQ_F32                                                                                                       29

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NLT_F32                                                                                                       30

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

V_CMP_F_F64                                                                                                        32

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_F64                                                                                                       33

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.f64 < S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_F64                                                                                                       34

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

V_CMP_GT_F64                                                                                                           36

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.f64 > S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LG_F64                                                                                                           37

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.f64 <> S1.f64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_F64                                                                                                       38

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

V_CMP_NLG_F64                                                                                                       42

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

V_CMP_NEQ_F64                                                                                                      45

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

V_CMP_LT_I16                                                                                                           49

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

V_CMP_NE_I16                                                                                                       53

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

V_CMP_EQ_U16                                                                                                           58

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

V_CMP_GE_U16                                                                                                       62

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

V_CMP_EQ_I32                                                                                                           66

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

V_CMP_GE_I32                                                                                                       70

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

V_CMP_LT_U32                                                                                                           73

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

V_CMP_NE_U32                                                                                                       77

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

V_CMP_F_I64                                                                                                            80

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

V_CMP_GT_I64                                                                                                       84

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

V_CMP_F_U64                                                                                                        88

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_LT_U64                                                                                                       89

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

  D0.u64[laneId] = S0.u64 < S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_EQ_U64                                                                                                       90

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

V_CMP_GT_U64                                                                                                           92

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u64 > S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_NE_U64                                                                                                           93

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

  D0.u64[laneId] = S0.u64 <> S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_GE_U64                                                                                                        94

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

  D0.u64[laneId] = S0.u64 >= S1.u64;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_T_U64                                                                                                         95

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.

Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMP_CLASS_F16                                                                                                    125

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

V_CMPX_O_F16                                                                                                      135

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

V_CMPX_NLG_F16                                                                                                      138

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

V_CMPX_NLT_F16                                                                                                    142

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F16                                                                                                      143

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F32                                                                                                      144

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_F32                                                                                                     145

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

V_CMPX_GE_F32                                                                                                       150

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.f32 >= S1.f32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_O_F32                                                                                                        151

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_U_F32                                                                                                        152

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)))

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGE_F32                                                                                                    153

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 >= S1.f32);
  // With NAN inputs this is not the same operation as <

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLG_F32                                                                                                    154

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NGT_F32                                                                                                    155

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLE_F32                                                                                                    156

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NEQ_F32                                                                                                      157

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NLT_F32                                                                                                      158

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_T_F32                                                                                                        159

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_F64                                                                                                        160

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

V_CMPX_O_F64                                                                                                      167

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

V_CMPX_NGT_F64                                                                                                      171

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

V_CMPX_LE_I16                                                                                                      179

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.i16 <= S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I16                                                                                                      180

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i16 > S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I16                                                                                                      181

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i16 <> S1.i16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_I16                                                                                                      182

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

V_CMPX_EQ_U16                                                                                                       186

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.u16 == S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_U16                                                                                                       187

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.u16 <= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_U16                                                                                                       188

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u16 > S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_U16                                                                                                      189

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.u16 <> S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GE_U16                                                                                                      190

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask.

  EXEC.u64[laneId] = S0.u16 >= S1.u16

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_F_I32                                                                                                       192

Set the per-lane condition code to 0. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'0U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LT_I32                                                                                                      193

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i32 < S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_EQ_I32                                                                                                       194

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask.

  EXEC.u64[laneId] = S0.i32 == S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_LE_I32                                                                                                       195

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask.

  EXEC.u64[laneId] = S0.i32 <= S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_GT_I32                                                                                                       196

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask.

  EXEC.u64[laneId] = S0.i32 > S1.i32

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_NE_I32                                                                                                      197

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

V_CMPX_LT_U32                                                                                                       201

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

V_CMPX_GT_I64                                                                                                      212

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

V_CMPX_F_U64                                                                                                        216

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

V_CMPX_T_U64                                                                                                    223

Set the per-lane condition code to 1. Store the result into the EXEC mask.

  EXEC.u64[laneId] = 1'1U

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.

V_CMPX_CLASS_F16                                                                                                253

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

V_CMPX_CLASS_F64                                                                                                  255

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
double-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask.

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
  EXEC.u64[laneId] = result

Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
