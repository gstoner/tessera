# 10.4. VGPR Usage

> RDNA4 ISA — pages 133–133

Opcode                      a16[0] acnt type               VGPRn[31:0]      VGPRn+1[31:0]   VGPRn+2[31:0]     VGPRn+3[31:0]
Gather "_L"                 0         2    2D              s                t               lod
                                      3    Cube(Array) s                    t               face              lod
                                      3    2D Array        s                t               slice             lod
                            1         2    2D              t, s             -, lod
                                      3    Cube(Array) t, s                 lod, face
                                      3    2D Array        t, s             lod, slice
Gather "_CL"                0         2    2D              s                t               clamp
                                      3    Cube(Array) s                    t               face              clamp
                                      3    2D Array        s                t               slice             clamp
                            1         2    2D              t, s             -, clamp
                                      3    Cube(Array) t, s                 clamp, face
                                      3    2D Array        t, s             clamp, slice
MSAA_LOAD                   0         2    2D MSAA         s                t               fragid
                                      3    2D Array        s                t               slice             fragid
                                           MSAA
                            1         2    2D MSAA         t, s             -, fragid
                                      3    2D Array        t, s             fragid, slice
                                           MSAA

The table below lists and briefly describes the legal suffixes for image instructions:

                                           Table 59. Sample Instruction Suffix Key
Suffix   Meaning        Extra Addresses         Description
_L       LOD            -                       LOD is used instead of computed LOD.
_B       LOD BIAS       1: lod bias             Add this BIAS to the computed LOD.
_CL      LOD CLAMP      -                       Clamp the computed LOD to be no larger than this value.
_D       Derivative     2,4 or 6: slopes        Send dx/dv, dx/dy, etc. slopes to be used in LOD computation.
_LZ      Level 0        -                       Force use of MIP level 0.
_C       PCF            1: z-comp               Percentage closer filtering.
_O       Offset         1: offsets              Send X, Y, Z integer offsets (packed into 1 DWORD) to offset XYZ address.
_G16     Gradient 16b   -                       Gradients are 16-bits instead of 32-bits, packed 2 gradients per VGPR (dX in
                                                low 16bits, dY in high 16bits).

10.4. VGPR Usage
Address: The address consists of up to 5 parts: { offset } { bias } { z-compare } { derivative } { body }

These are all in VGPRs, and can consist of up to 12 values.
  • Offset: SAMPLE*O*, GATHER*O*
    1 DWORD of 'offset_xyz' . The offsets are 6-bit signed integers: X=[5:0], Y=[13:8], Z=[21:16]
  • Bias: SAMPLE*B*, GATHER*B*. 1 DWORD float.
  • Z-compare: SAMPLE*C*, GATHER*C*. 1 DWORD.
  • Derivatives (SAMPLE_D): 2,4 or 6 DWORDS - these packed 1 DWORD per derivative as shown below (F32).
  • Body: One to four DWORDs, as defined by the table: Image Opcodes with a Sampler
    Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc.
    The number of components in "body" is the value of the ACNT field in the table, plus one.
