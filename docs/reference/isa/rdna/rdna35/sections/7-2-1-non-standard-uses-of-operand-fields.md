# 7.2.1. Non-Standard Uses of Operand Fields

> RDNA3.5 ISA — pages 68–68

                                Code        Meaning
Vector    Scalar    Scalar      0-105       SGPR 0 .. 105         SGPRs. One DWORD each.
Source    Source (8 Dest (7     106         VCC_LO                VCC[31:0]
(when 9   bits)     bits)       107         VCC_HI                VCC[63:32]
bits)
                                108-123     ttmp0 .. ttmp15       Trap handler temporary SGPRs (privileged)
                                124         NULL                  Reads return zero, writes are ignored. When used
                                                                  as a destination, nullifies the instruction.
                                125         M0                    Temporary register, use for a variety of functions
                                126         EXEC_LO               EXEC[31:0]
                                127         EXEC_HI               EXEC[63:32]
                     Integer   128          0                     Inline constant zero
                     Inline    129-192      int 1 .. 64           Integer inline constants
                     Constants 193-208      int -1 .. -16
                                209-232     Reserved              Reserved
                                233         DPP8                  8-lane DPP (only valid as SRC0)
                                234         DPP8FI                8-lane DPP with Fetch-Invalid (only valid as SRC0)
                                235         SHARED_BASE           Memory Aperture Definition
                                236         SHARED_LIMIT
                                237         PRIVATE_BASE
                                238         PRIVATE_LIMIT
                                239         Reserved              Reserved
                     Float     240          0.5                   Inline floating point constants. Can be used in 16,
                     Inline    241          -0.5                  32 and 64 bit floating point math. They may be
                     Constants 242          1.0                   used with non-float instructions but the value
                                                                  remains a float.
                                243         -1.0
                                244         2.0                   1/(2*PI) is 0.15915494. The hex values are:
                                245         -2.0                  half: 0x3118
                                246         4.0                   single: 0x3e22f983
                                247         -4.0                  double: 0x3fc45f306dc9c882
                                248         1.0 / (2 * PI)
                                249         Reserved              Reserved
                                250         DPP16                 data parallel primitive
                                251         Reserved              Reserved
                                252         Reserved              Reserved
                                253         SCC                   { 31’b0, SCC }
                                254         Reserved              Reserved
                                255         Literal constant      32 bit constant from instruction stream
          Vector Src/Dst        256 - 511   VGPR 0 .. 255         Vector GPRs. One DWORD each.
          (8 bits)

7.2.1. Non-Standard Uses of Operand Fields
A few instructions use the operand fields in non-standard ways:
