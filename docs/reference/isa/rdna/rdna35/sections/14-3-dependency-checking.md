# 14.3. Dependency Checking

> RDNA3.5 ISA — pages 151–152

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
The shader program is responsible for alpha-test.

All data that can affect the sample mask must be sent on the first export from the shader. This means if depth
is being exported, it must be exported first. If alpha to mask is enabled, MRT0 must be exported first, unless
depth is also enabled, in which case, MRT0’s alpha value must be written to the depth export’s alpha value. If
alpha to mask and coverage to mask are both enabled, then the depth export’s alpha value is set to the
minimum of the alpha to mask value (alpha of MRT0) and the coverage to mask value (alpha of what would
have been in the depth export). If the shader can kill a pixel, it must be determined before the first export.

Pixel Shader Dual-Source Blend
     In this mode, alternating lanes (threads) hold MRT0 and MRT1, not all threads going to one MRT. There are
     two instructions to complete a dual-source blend export. It is required that exports to 21 and 22 be back-to-
     back, with no other export types in between them.

Export target       EXEC mask                               MRT         Lane 0    Lane 1    Lane 2
                                                            Exported
21                  exec_mask =                             0           Pix0,     Pix0      Pix2 MRT0
                    (exec_mask & 0x5555_5555) |                         MRT0      MRT1
                    ((exec_mask <<1) & 0xAAAA_AAAA)
22                  exec_mask =                             1           Pix1,     Pix1,     Pix3 MRT0
                    (exec_mask & 0xAAAA_AAAA) |                         MRT0      MRT1
                    ((exec_mask >>1) & 0x5555_5555)

14.2. Primitive Shader Exports (From GS shader stage)
The GS shader uses export instructions to output vertex position data, and memory stores for vertex parameter
data. This data is passed on to subsequent pixel shaders.

Every vertex shader must output at least one position vector (x, y, z; w is optional) to the POS0 target. The last
position export must have the DONE bit set to 1. For optimized performance, it is recommended to output all
position data as early as possible in the vertex shader.

14.3. Dependency Checking
Export instructions are executed by the hardware in two phases. First, the instruction is selected to be
executed, and EXPCNT is incremented by 1. At this time, the wave has made a request to export data, but the

data has not been exported yet. Later, when the export actually occurs the EXEC mask and VGPR data is read
and the data is exported, and finally EXPcnt is decremented.

Use S_WAITCNT on EXPcnt to prevent the shader program from overwriting EXEC or the VGPRs holding the
data to be exported before the export operation has completed.

Multiple export instructions can be outstanding at one time. Exports of the same type (for example: position)
are completed in order, but exports of different types can be completed out of order. If the STATUS register’s
SKIP_EXPORT bit is set to one, the hardware treats all EXPORT instructions as if they were NOPs.
