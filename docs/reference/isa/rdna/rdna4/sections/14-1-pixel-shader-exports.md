# 14.1. Pixel Shader Exports

> RDNA4 ISA — pages 170–170

14.1. Pixel Shader Exports
Pixel Exports
     Export instructions copy color data to the MRTs. Data has up to four components (R, G, B, A).
     Optionally, export instructions also output depth (Z) data.
     Every pixel shader must have at least one export instruction.
     The last export instruction executed must have the DONE bit set to one.
     The EXEC mask is applied to all exports. Only pixels with the corresponding EXEC bit set to 1 export data to
     the output buffer.
     Each export target must be exported to only once.

The shader program is responsible for conversion of data from 32b to 16b for 16-bit exports.
The shader program is responsible for alpha-test. If the shader can kill a pixel, it must be determined before
the first export. Each pixel shader export must have the same pixel-valid mask (EXEC) and this must be the
final mask value (subsequent exports cannot have a different mask value).

Pixel exports do not execute until export-buffer space has been allocated to the wave. If
STATUS.SKIP_EXPORT==1, the shader program ignores export instructions.

OREO - Opaque Random Order Export of Pixel Data
     OREO allows opaque surfaces to export out of order. Without this, all exports must occur in-order between
     waves. OREO has a mechanism that detects conflicts (waves that write to the same pixel) and enforces
     ordering. If a wave has a 'conflict', exports for that wave may be at reduced rate while the conflict exists. No
     software is needed to support this.

All data that can affect the sample mask must be sent on the first export from the shader. If depth is being
exported, it must be exported first. If alpha to mask is being exported, MRT0 must be exported first, unless
depth is also enabled, in which case, MRT0’s alpha value must be written to the depth export’s alpha value. If
alpha to mask and coverage to mask are both enabled, then the depth export’s alpha value is be set to the
minimum of the alpha to mask value (alpha of MRT0) and the coverage to mask value (alpha of what would
have been in the depth export).

Pixel Shader Dual-Source Blend
     In this mode, alternating lanes (threads) hold MRT0 and MRT1, not all threads going to one MRT. There are
     two instructions to complete a dual-source blend export. It is required that exports to 21 and 22 be back-to-
     back, with no other export types in between them.

Export target       EXEC mask                                MRT          Lane 0   Lane 1     Lane 2
                                                             Exported
21                  exec_mask =                              0            Pix0,    Pix0       Pix2 MRT0
                    (exec_mask & 0x5555_5555) |                           MRT0     MRT1
                    ((exec_mask <<1) & 0xAAAA_AAAA)
22                  exec_mask =                              1            Pix1,    Pix1,      Pix3 MRT0
                    (exec_mask & 0xAAAA_AAAA) |                           MRT0     MRT1
                    ((exec_mask >>1) & 0x5555_5555)

14.2. Primitive Shader Exports (From GS shader stage)
The GS shader uses export instructions to output vertex position data, and memory stores for vertex parameter
