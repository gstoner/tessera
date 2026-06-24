# 7.6.2. OPSEL with FP8 and BF8 Conversions

> RDNA4 ISA — pages 89–89

7.6.1. Data Convert Instruction Types
  Normal / Single            Convert one value in a source register into one value in a destination register

  Packed ("PK")              Convert two consecutive values in source register(s) into two values in destination
                             register(s)

  Stochastic Rounded         add a pseudo-random value (unique per lane) to each source value, then truncate

Stochastic rounding adds bits from a source VGPR supplying a random value to the input data value and then
truncates to the smaller precision result type. Due to the small range of F8, the result of conversion from FP32
to F8 has an option to be clamped +/-MAX normal value instead of +/-INF.

Convert with Stochastic Round:
  • one source has the number to convert and the other has a random number used in rounding
  • These ops add a random value from the specified VGPR and then truncate to the smaller result data type

7.6.2. OPSEL with FP8 and BF8 Conversions
CVT_PK_FP8_F32           16-bits of data written represents two packed 8-bit values.
CVT_PK_BF8_F32           case OPSEL[3]:
                           0: write DST[15:0], preserve DST[31:16]
                           1: write DST[31:16], preserve DST[15:0]
CVT_SR_FP8_F32           case OPSEL[3:2]:
CVT_SR_BF8_F32             0: write DST[7:0], preserve other DST bits
                           1: write DST[16:8], preserve other DST bits
                           2: write DST[24:16], preserve other DST bits
                           3: write DST[31:24], preserve other DST bits
CVT_PK_F32_FP8           SourceData[15:0] = (SRC0 >> (16 * OPSEL[0])) & 0xFFFF
CVT_PK_F32_BF8
CVT_F32_FP8              SourceData[7:0] = (SRC0 >> (8 * OPSEL[0:1])) & 0xFF
CVT_F32_BF8

These instructions do not support ABS, NEG, OMOD or CLAMP. CVT_PK_*_F32 uses round-to-nearest-even;
others do not need rounding (except CVT_SR opcodes: uses truncate). CVT_PK_F16_* does not support DPP.

The FP16_OVFL flag is applied to data conversions from F32 to FP8/BF8 formats. The overflow behaviour is
specified in the table below:

                                                                          Destination Value
                                                           FP8                                   BF8
             Source Value               FP16_OVFL=1           FP16_OVFL=0          FP16_OVFL=1    FP16_OVFL=0
NaN                                     NaN                   NaN                  NaN            NaN
±Inf                                    ±max_E4M3             NaN                  ±max_E5M2      ±Inf
Greater than max FP8 magnitude          ±max_E4M3             NaN                  ±max_E5M2      ±Inf
