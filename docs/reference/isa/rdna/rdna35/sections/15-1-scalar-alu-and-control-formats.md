# 15.1. Scalar ALU and Control Formats

> RDNA3.5 ISA — pages 155–155

15.1. Scalar ALU and Control Formats

15.1.1. SOP2

  Description       This is a scalar instruction with two inputs and one output. Can be followed by a 32-bit
                    literal constant.

                                                   Table 65. SOP2 Fields
Field Name               Bits            Format or Description
SSRC0                    [7:0]           Source 0. First operand for the instruction.
                         0-105           SGPR0 - SGPR105: Scalar general-purpose registers.
                         106             VCC_LO: VCC[31:0].
                         107             VCC_HI: VCC[63:32].
                         108-123         TTMP0 - TTMP15: Trap handler temporary register.
                         124             NULL
                         125             M0. Misc register 0.
                         126             EXEC_LO: EXEC[31:0].
                         127             EXEC_HI: EXEC[63:32].
                         128             0.
                         129-192         Signed integer 1 to 64.
                         193-208         Signed integer -1 to -16.
                         209-234         Reserved.
                         235             SHARED_BASE (Memory Aperture definition).
                         236             SHARED_LIMIT (Memory Aperture definition).
                         237             PRIVATE_BASE (Memory Aperture definition).
                         238             PRIVATE_LIMIT (Memory Aperture definition).
                         239             Reserved.
                         240             0.5.
                         241             -0.5.
                         242             1.0.
                         243             -1.0.
                         244             2.0.
                         245             -2.0.
                         246             4.0.
                         247             -4.0.
                         248             1/(2*PI).
                         249 - 252       Reserved.
                         253             SCC.
                         254             Reserved.
                         255             Literal constant.
SSRC1                    [15:8]          Second scalar source operand.
                                         Same codes as SSRC0, above.
SDST                     [22:16]         Scalar destination.
                                         Same codes as SSRC0, above except only codes 0-127 are valid.
OP                       [29:23]         See Opcode table below.
ENCODING                 [31:30]         'b10

                        Table 66. SOP2 Opcodes
