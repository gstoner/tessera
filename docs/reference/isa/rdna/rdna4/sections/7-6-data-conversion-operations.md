# 7.6. Data Conversion Operations

> RDNA4 ISA — pages 88–88

VOP3, VOP3P, VINTERP Encoding of 16-bit VGPRs
     SRC/DST[7:0] = 32-bit VGPR address, OPSEL = high/low.
     In this encoding, a wave can address 512 16-bit VGPRs.

The packing shown below allows reading or writing in one cycle:
    • 32 lanes of one 32-bit VGPR: V0
    • 64 lanes of one 16-bit VGPR: V0.L
    • 32 lanes of two 16-bit VGPRs (a pair, as used by packed math): V0.L and V0.H

7.5. 8-bit Math
8-bit floating point values can be represented in one of two forms:
    • FP8 : { Sign, Exp4, Mant3 }; ExpBias = 7
    • BF8 : { Sign, Exp5, Mant2 }; ExpBias = 15

These formats are supported in DOT, WMMA and CVT operations.
The ISA does not allow directly addressing individual 8-bit quantities in VGPRs.
The following table provides the numeric ranges based on the data format and bias selection.

F8_       Fmt     Sign-Exp-   Bias      +0        INF,        NaN,         Max       Min           Min (denorm)
Mode              Mant                  -0        -INF        -NaN                   (norm)
x         FP16    E5M10       15        0x0000    0x7C00      (normal)     65504     6.10352E-05   5.96046E-08
                                        0x8000    0xFC00
1         BF8     E5M2        16        0x00      0x80        0x80         57344     3.05176E-05   7.62939E-06
                                        0x00
0         BF8     E5M2        15        0x00      0x7C        +: 0x7D-7F   57344     6.10352E-05   1.52588E-05
                                        0x80      0xFC        -: 0xFD-FF
1         FP8     E4M3        8         0x00      0x80        0x80         240       0.0078125     0.000976563
                                        0x00
0         FP8     E4M3        7         0x00      N/A         +: 0x7F      448       0.0156250     0.001953125
                                        0x80                  -: 0xFF

7.6. Data Conversion Operations
Common Restrictions on data conversion operations:

    • Input modifiers (neg, abs) are not supported
    • Rounding is RNE (except for "SR" ops, which truncate)
    • OMOD is not used
    • CLAMP is not supported for conversions from BF8/FP8 or smaller floats
    • FP16_OVFL applies when the destination is FP16 or BF16
    • OPSEL is not supported for packed or stochastic converts
