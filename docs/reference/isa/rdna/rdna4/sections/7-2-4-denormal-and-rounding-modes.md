# 7.2.4. Denormal and Rounding Modes

> RDNA4 ISA — pages 83–83

The clamp bit has multiple uses. For V_CMP instructions, setting the clamp bit to 1 indicates that the compare
signals if a floating point exception occurs. For integer operations, it clamps the result to the largest and
smallest representable value. For floating point operations, it clamps the result to the range: [0.0, 1.0].

When CLAMP==1, any NaN result is clamped to zero, and exceptions are reported on the result before CLAMP
is applied.

All operands that can produce exceptions quiet SNaNs. E.g. V_ADD_F32 quiets SNaN but V_MOV_B32 does not.

Output Clamping: The clamp instruction bit applies to the following operations and data types:
  • Float clamp to [0.0, 1.0]
  • Signed Int [INT_MIN, INT_MAX]
  • Unsigned int [0, UINT_MAX]
  • Bool (V_CMP) enables signaling compare when clamp==1

The clamp bit is not supported for (ignored):

                V_PERMLANE*                V_PERM_B32                   Float DOT instructions
                V_SWAP and V_SWAPREL       WMMA ops                     V_ADD3_NC
                V_MOV                      V_CVT*F32_{BF8,FP8}
                V_DOT4_I32_IU8             V_DOT4_U32_U8
                V_ADD_LSHL                 V_ALIGN*                     Bitwise ops
                V_CMP*_CLASS               V_CMP on integers
                V_READLANE                 V_READFIRSTLANE              V_WRITELANE

7.2.3.2. Wave64 Destination Restrictions on SGPRs
When a VALU instruction is issued from a wave64, it may issue twice as two wave32 instructions. While in most
cases the programmer need not be aware of this, it does impose a prohibition on wave64 VALU instructions
that both write and read the same SGPR value. Doing this may lead to unpredictable results. Specifically, the first
pass of a wave64 VALU instruction may not overwrite a scalar value used by the second half.

7.2.4. Denormal and Rounding Modes
The shader program has explicit control over the rounding mode applied and the handling of denormalized
inputs and results. The MODE register is set using the S_SETREG instruction; it has separate bits for controlling
the behavior of single and double-precision floating-point numbers.

Round and denormal modes can also be set using S_ROUND_MODE and S_DENORM_MODE.

16-bit floats support denormals, infinity and NaN.

                                  Table 36. Round and Denormal Modes
