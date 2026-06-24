# 15.11.1. VEXPORT

> RDNA4 ISA — pages 215–215

15.11. Export Format

15.11.1. VEXPORT

  Description       EXPORT instructions

The export format has only a single opcode, "EXPORT".

                                               Table 121. VEXPORT Fields
Field Name      Bits          Format or Description
EN              [3:0]         VGPR Enables: [0] enables VSRC0, … [3] enables VSRC3.
TARGET          [9:4]          Export destination:
                                0..7     MRT 0..7
                                8        Z
                                12-16    Position 0-4
                                20       Primitive data
                                21       Dual Source Blend Left
                                22       Dual Source Blend Right
DONE            [11]          Indicates that this is the last export from the shader. Used only for Position and Pixel/color
                              data.
ROW             [13]          Row to export
ENCODING        [31:26]       'b111110
VSRC0           [39:32]       VGPR for source 0.
VSRC1           [47:40]       VGPR for source 1.
VSRC2           [55:48]       VGPR for source 2.
VSRC3           [63:56]       VGPR for source 3.
