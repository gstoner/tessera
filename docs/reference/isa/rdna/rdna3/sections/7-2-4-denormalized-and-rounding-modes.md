# 7.2.4. Denormalized and Rounding Modes

> RDNA3 ISA — pages 70–70

The clamp bit is not supported for (ignored):

                V_PERMLANE*                 V_PERM_B32                    Float DOT instructions
                V_SWAP and V_SWAPREL        WMMA ops                      V_ADD3
                V_ADD_LSHL                  V_ALIGN*                      Bitwise ops
                V_CMP*_CLASS                V_CMP on integers
                V_READLANE                  V_READFIRSTLANE               V_WRITELANE

7.2.3.2. Wave64 Destination Restrictions
When a VALU instruction is issued from a wave64, it may issue twice as two wave32 instructions. While in most
cases the programmer need not be aware of this, it does impose a prohibition on wave64 VALU instructions
that both write and read the same SGPR value. Doing this may lead to unpredictable results. Specifically, the first
pass of a wave64 VALU instruction may not overwrite a scalar value used by the second half.

7.2.4. Denormalized and Rounding Modes
The shader program has explicit control over the rounding mode applied and the handling of denormalized
inputs and results. The MODE register is set using the S_SETREG instruction; it has separate bits for controlling
the behavior of single and double-precision floating-point numbers.

Round and denormal modes can also be set using S_ROUND_MODE and S_DENORM_MODE which is the
preferred method over using S_SETREG.

16-bit floats support denormals, infinity and NaN.

                                  Table 28. Round and Denormal Modes
Field            Bit Position    Description
FP_ROUND         3:0             [1:0] Single-precision round mode.
                                 [3:2] Double and Half-precision (FP16) round mode.
                                 Round Modes:
                                   0=nearest even
                                   1= +infinity
                                   2= -infinity
                                   3= toward zero
FP_DENORM        7:4             [5:4] Single-precision denormal mode.
                                 [7:6] Double and Half-precision (FP16) denormal mode.
                                 Denormal modes:
                                   0 = Flush input and output denorms
                                   1 = Allow input denorms, flush output denorms
                                   2 = Flush input denorms, allow output denorms
                                   3 = Allow input and output denorms

These mode bits do not affect rounding and denormal handling of F32 global memory atomics.

DOT2_F16_F16 and DOT2_BF16_BF16 support round-to-nearest-even rounding. DOT2_F16_F16 supports
denorms, and DOT2_BF16_BF16 disables all denorms.
