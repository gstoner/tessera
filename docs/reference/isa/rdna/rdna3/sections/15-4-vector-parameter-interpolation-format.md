# 15.4. Vector Parameter Interpolation Format

> RDNA3 ISA — pages 181–181

15.4. Vector Parameter Interpolation Format

15.4.1. VINTERP

  Description       Vector Parameter Interpolation.
                    These opcodes perform parameter interpolation using vertex data in pixel shaders.

                                                Table 96. VINTERP Fields
Field Name               Bits          Format or Description
VDST                     [7:0]         Destination VGPR
WAITEXP                  [10:8]        Wait for EXPcnt to be less-than or equal-to this value before issuing instruction.
OPSEL                    [14:11]       Select low or high for low sources 0=[11], 1=[12], 2=[13], dst=[14].
CLMP                     [15]          1 = clamp result.
OP                       [22:16]       Opcode. see next table.
ENCODING                 [31:26]       'b11001101
SRC0                     [40:32]       Source 0. First operand for the instruction: VGPR 0-255.
