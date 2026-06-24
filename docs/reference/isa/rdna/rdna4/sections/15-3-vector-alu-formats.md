# 15.3. Vector ALU Formats

> RDNA4 ISA — pages 182–182

15.3. Vector ALU Formats

15.3.1. VOP2

  Description       Vector ALU format with two input operands. Can be followed by a 32-bit literal constant
                    or DPP instruction DWORD when the instruction allows it.

                                                 Table 86. VOP2 Fields
Field Name                 Bits        Format or Description
SRC0                       [8:0]       Source 0. First operand for the instruction.
                           0-105       SGPR0 - SGPR105: Scalar general-purpose registers.
                           106         VCC_LO: VCC[31:0].
                           107         VCC_HI: VCC[63:32].
                           108-123     TTMP0 - TTMP15: Trap handler temporary register.
                           124         NULL
                           125         M0. Misc register 0.
                           126         EXEC_LO: EXEC[31:0].
                           127         EXEC_HI: EXEC[63:32].
                           128         0.
                           129-192     Signed integer 1 to 64.
                           193-208     Signed integer -1 to -16.
                           209-232     Reserved.
                           233         DPP8 (only valid as SRC0)
                           234         DPP8FI (only valid as SRC0)
                           235         SHARED_BASE (Memory Aperture definition).
                           236         SHARED_LIMIT (Memory Aperture definition).
                           237         PRIVATE_BASE (Memory Aperture definition).
                           238         PRIVATE_LIMIT (Memory Aperture definition).
                           239         Reserved.
                           240         0.5.
                           241         -0.5.
                           242         1.0.
                           243         -1.0.
                           244         2.0.
                           245         -2.0.
                           246         4.0.
                           247         -4.0.
                           248         1/(2*PI).
                           250         DPP16 (only valid as SRC0)
                           253         SCC.
                           254         64-bit Literal constant.
                           255         32-bit Literal constant.
                           256 - 511   VGPR 0 - 255
VSRC1                      [16:9]      VGPR that provides the second operand.
VDST                       [24:17]     Destination VGPR.
OP                         [30:25]     See Opcode table below.
ENCODING                   [31]        'b0

                             Table 87. VOP2 Opcodes
