# 15.4. Vector Parameter Interpolation Format

> RDNA4 ISA — pages 202–203

15.4. Vector Parameter Interpolation Format

15.4.1. VINTERP

    Description     Vector Parameter Interpolation.
                    These opcodes perform parameter interpolation using vertex data in pixel shaders.

                                                  Table 105. VINTERP Fields
Field Name                  Bits          Format or Description
VDST                        [7:0]         Destination VGPR
WAITEXP                     [10:8]        Wait for EXPcnt to be less-than or equal-to this value before issuing instruction.
OPSEL                       [14:11]       Select low or high for low sources 0=[11], 1=[12], 2=[13], dst=[14].
CM                          [15]          1 = clamp result.
OP                          [20:16]       Opcode. see next table.
ENCODING                    [31:26]       'b11001101
SRC0                        [40:32]       Source 0. First operand for the instruction.
                            256 - 511     VGPR 0 - 255
SRC1                        [49:41]       Second input operand. Same options as SRC0.
SRC2                        [58:50]       Third input operand. Same options as SRC0.
NEG                         [63:61]       Negate input for low 16-bits of sources. [61] = src0, [62] = src1, [63] = src2

                              Table 106. VINTERP Opcodes
Opcode # Name                               Opcode # Name
0          V_INTERP_P10_F32                 3            V_INTERP_P2_F16_F32
1          V_INTERP_P2_F32                  4            V_INTERP_P10_RTZ_F16_F32
2          V_INTERP_P10_F16_F32             5            V_INTERP_P2_RTZ_F16_F32

15.5. Parameter and Direct Load from LDS

15.5.1. VDSDIR

    Description     LDS Direct and Parameter Load.
                    These opcodes read either pixel parameter data or individual DWORDs from LDS into
                    VGPRs.

                                                   Table 107. VDSDIR Fields
Field Name          Bits              Format or Description
VDST                [7:0]             Destination VGPR
ATTR_CHAN           [9:8]             Attribute channel: 0=X, 1=Y, 2=Z, 3=W

Field Name          Bits         Format or Description
ATTR                [15:10]      Attribute number: 0 - 32.
WAIT_VA             [19:16]      Wait for previous VALU instructions to complete to resolve data dependency. Value is
                                 the max number of VALU ops still outstanding when issuing this instruction.
OP                  [21:20]      Opcode:
                                 0: DS_PARAM_LOAD
                                 1: DS_DIRECT_LOAD
                                 2, 3: Reserved.
WAIT_VMVSRC         [23]         When set to 0, wait for all previously issued VMEM ops (including LDS) to have finished
                                 reading source VGPRs before issuing this instruction; when set to one issue as usual (no
                                 extra waiting).
ENCODING            [31:24]      'b11001110

                  Table 108. VDSDIR Opcodes
Opcode # Name                    Opcode # Name
0          DS_PARAM_LOAD         1            DS_DIRECT_LOAD
