# 13.3. NaN Handling

> RDNA4 ISA — pages 167–168

Flat instructions that are processed by LDS do not flush denorms, regardless of the MODE.denorm setting.

CompareStore ("compare swap") flushes the result when input denormal flushing occurs.

13.3. NaN Handling
Not A Number ("NaN") is a IEEE-754 value representing a result that cannot be computed.

There two types of NaN: quiet and signaling
  • Quiet NaN Exponent=0xFF, Mantissa MSB=1
  • Signaling NaN Exponent=0xFF, Mantissa MSB=0 and at least one other mantissa bit ==1

The LDS does not produce any exception or "signal" due to a signaling NaN.

DS_ADD_F32 can create a quiet NaN, or propagate NaN from its inputs: if either input is a NaN, the output is
that same NaN, and if both inputs are NaN, the NaN from the first input is selected as the output. Signaling NaN
is converted to Quiet NaN.

When denormals are flushed (see table in previous section), they are flushed before the operation (i.e. before
the comparison).

FP MAX / MIN Selection Rules

      if       (Src0==NaN)     result = quiet(Src0)     // either SNaN or QNaN
      else if (Src1==NaN)      result = quiet(Src1)
      else if (Src0>Src1)      larger_of (src0, src1)   // or smaller_of   for Minimum

      "Larger_of" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf
      -0 < +0.   Preserves -0.   If any input is sNaN, signal invalid exception.

      "Smaller_of" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf
      -0 < +0.   Preserves -0.   If any input is sNaN, signal invalid exception.

FP MAXNUM / MINNUM Selection Rules

      if       (src0 == SNaN or QNaN)   && (src1 == SNaN or QNaN) result = QNaN (src0)
      else if (src0 == SNaN or QNaN)    result = src1
      else if (src1 == SNaN or QNaN)    result = src0
      else                              result = larger_of (src0, src1)    // or smaller_of for minimumNumber

      "Larger_of" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf
      -0 < +0.   Preserves -0.   If any input is sNaN, signal invalid exception.

      "Smaller_of" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf
      -0 < +0.   Preserves -0.   If any input is sNaN, signal invalid exception.

                Memory atomics use the MINNUM / MAXNUM style.

For VALU ops, when CLAMP=1 any NaN result is clamped to zero, and exceptions are reported before applying
the CLAMP.

Float Add rules:
 1. if SRC0 == NaN: result = QNaN of SRC0 (preserve sign)
 2. else if SRC1 == NaN: result = QNaN of SRC1 (preserve sign)
 3. else if SRC0 ==INF and SRC1==INF and signs differ: -QNAN (FP32: 0xFFC00000)
 4. else if SRC0 == INF: result = SRC0
 5. else if SRC1 == INF: result = SRC1
