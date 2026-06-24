# 13.3. NaN Handling

> RDNA3.5 ISA — pages 144–144

The floating point atomic instructions (ds_{min,max,cmpst}_f32) have the option of passing denormal values
through, or flushing them to zero. This is controlled with the MODE.fp_denorm bits that also control VALU
denormal behavior. There is no separate input and output denormal control: only bit 0 of sp_denorm or bit 0 of
dp_denorm is considered. The rest of the denormal rules are identical to LDS.
Float atomic add is hardwired to flush input denormals - it does not use the MODE.fp_denorm bits.

13.3. NaN Handling
Not A Number ("NaN") is a IEEE-754 value representing a result that cannot be computed.

There two types of NaN: quiet and signaling
  • Quiet NaN Exponent=0xFF, Mantissa MSB=1
  • Signaling NaN Exponent=0xFF, Mantissa MSB=0 and at least one other mantissa bit ==1

The LDS does not produce any exception or "signal" due to a signaling NaN.

DS_ADD_F32 can create a quiet NaN, or propagate NaN from its inputs: if either input is a NaN, the output is
that same NaN, and if both inputs are NaN, the NaN from the first input is selected as the output. Signaling NaN
is converted to Quiet NaN.

Floating point atomics (CMPSWAP, MIN, MAX) flush input denormals only when
MODE (allow_input_denorm)=0, otherwise values are passed through without modification. When flushing,
denorms are flushed before the operation (i.e. before the comparison).

FP Max Selection Rules:

      if          (src0 == SNaN) result = QNaN (src0)         // bits of SRC0 are preserved but is a QNaN
      else if (src1 == SNaN) result = QNaN (src1)
      else                      result = larger of (src0, src1)
      "Larger" order from smallest to largest: QNaN, -inf, -float, -denorm, -0, +0, +denorm, +float, +inf

FP Min Selection Rules:

      if          (src0 == SNaN) result = QNaN (src0)
      else if (src1 == SNaN) result = QNaN (src1)
      else                      result = smaller of (src0, src1)
      "Smaller" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf, QNaN

FP Compare Swap: only swap if the compare condition (==) is true, treating +0 and -0 as equal

      doSwap = (src0 != NaN) && (src1 != NaN) && (src0 == src1) // allow +0 == -0

Float Add rules:
 1. -INF + INF = QNAN (mantissa is all zeros except MSB)
 2. +/-INF + NAN = QNAN (NAN input is copied to output but made quiet NAN)
 3. -INF + INF, or INF - INF = -QNAN
 4. -0 + 0 = +0
