# Chapter 14. Export: Position, Color/MRT

> RDNA3.5 ISA — pages 150–150

Chapter 14. Export: Position, Color/MRT
"Export" is the act of copying data from a VGPR to the one of the export buffers (position, color or Z). Exports
use the EXEC mask and only output the enabled pixels or vertices. A shader may export to each target only
once. The last export from a pixel shader, or the last position export of a vertex shader must indicate "done" -
there are no more pixel shader exports or vertex position exports. This allows the values to be consumed by the
Render back-end and Primitive Assembler respectively.

Exports can transfer 32-bit or 16-bit data per element. 16-bit exports occurs in pairs: 32-bits transferred from
one VGPR that holds two 16-bit values. The export instruction does not know or care about the difference
between the two - it just moves 32-bits of data per lane. 16-bit exports are a contract between the shader
program that is responsible for converting and packing 16-bit data, and the receiving hardware in
configuration registers that declare the exported data type. 16-bit data is packed into a VGPR, with the first
component in the lower 16 bits.

Instruction Fields

Field             Size       Description
Done              1          Indicates this is the last export from the shader.Used only for Pixel, Position and Primitive
                             data. Must be set for primitive export.
Target            6          Export Target:

                                           0-7          MRT 0-7
                                           8            Z
                                           12-16        Position 0-4 (Pos4 is for stereo rendering)
                                           20           NGG Primitive data (connectivity data)
                                           21           Dual source blend Left
                                           22           Dual source blend Right
EN                4          16-bit components: export half-DWORD enable. Valid values are: 0x0,1,3
                               [0] enables VSRC0 : R,G from one VGPR (R in low bits, G high)
                               [1] enables VSRC1 : B,A from one VGPR (B in low bits, A high)
                             32-bit components: [0-3] = enables for VSRC0-3.
VSRC0             8          VGPR to read data from.
VSRC1             8          Pos: vsrc0=X, 1=Y, 2=Z, 3=W
VSRC2             8          MRT: vsrc0=R, 1=G, 2=B, 3=A
VSRC3             8
ROW_EN            1          0 = normal mode; 1 = use M0 to provide the row number for mesh shader’s POS and PRIM
                             exports.
(M0)              8          Row number for mesh shader POS and PRIM exports

32-bit components        EN[0]                VSRC0            Red/X/ …
                         EN[1]                VSRC1            Green/Y/…
                         EN[2]                VSRC2            Blue/Z/…
                         EN[3]                VSRC3            Alpha/W/…
16-bit components        EN[0]                VSRC0            {green, red} / { y, x}
                         EN[1]                VSRC1            {alpha, blue} / {w,z}
                         EN[2], EN[3]         ignored          unused
