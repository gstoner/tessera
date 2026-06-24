# 15.5. Parameter and Direct Load from LDS

> RDNA3 ISA — pages 182–183

Field Name               Bits          Format or Description
SRC0                     [40:32]       Source 0. First operand for the instruction.
                         0-105         SGPR0 - SGPR105: Scalar general-purpose registers.
                         106           VCC_LO: VCC[31:0].
                         107           VCC_HI: VCC[63:32].
                         108-123       TTMP0 - TTMP15: Trap handler temporary register.
                         124           NULL
                         125           M0. Misc register 0.
                         126           EXEC_LO: EXEC[31:0].
                         127           EXEC_HI: EXEC[63:32].
                         128           0.
                         129-192       Signed integer 1 to 64.
                         193-208       Signed integer -1 to -16.
                         209-232       Reserved.
                         233           DPP8
                         234           DPP8FI
                         235           SHARED_BASE (Memory Aperture definition).
                         236           SHARED_LIMIT (Memory Aperture definition).
                         237           PRIVATE_BASE (Memory Aperture definition).
                         238           PRIVATE_LIMIT (Memory Aperture definition).
                         239           Reserved.
                         240           0.5.
                         241           -0.5.
                         242           1.0.
                         243           -1.0.
                         244           2.0.
                         245           -2.0.
                         246           4.0.
                         247           -4.0.
                         248           1/(2*PI).
                         250           DPP16
                         253           SCC.
                         254           Reserved.
                         255           Literal constant.
                         256 - 511     VGPR 0 - 255
SRC1                     [49:41]       Second input operand. Same options as SRC0.
SRC2                     [58:50]       Third input operand. Same options as SRC0.
NEG                      [63:61]       Negate input for low 16-bits of sources. [61] = src0, [62] = src1, [63] = src2

                           Table 97. VINTERP Opcodes
Opcode # Name                            Opcode # Name
0          V_INTERP_P10_F32              3           V_INTERP_P2_F16_F32
1          V_INTERP_P2_F32               4           V_INTERP_P10_RTZ_F16_F32
2          V_INTERP_P10_F16_F32          5           V_INTERP_P2_RTZ_F16_F32

15.5. Parameter and Direct Load from LDS

15.5.1. LDSDIR

  Description       LDS Direct and Parameter Load.
                    These opcodes read either pixel parameter data or individual DWORDs from LDS into
                    VGPRs.

                                                Table 98. LDSDIR Fields
Field Name               Bits          Format or Description
VDST                     [7:0]         Destination VGPR
ATTR_CHAN                [9:8]         Attribute channel: 0=X, 1=Y, 2=Z, 3=W
ATTR                     [15:10]       Attribute number: 0 - 32.
WAIT_VA                  [19:16]       Wait for previous VALU instructions to complete to resolve data dependency. Value
                                       is the max number of VALU ops still outstanding when issuing this instruction.
OP                       [21:20]       Opcode:
                                       0: LDS_DIRECT_LOAD
                                       1: LDS_PARAM_LOAD
                                       2, 3: Reserved.
ENCODING                 [31:24]       'b11001110
