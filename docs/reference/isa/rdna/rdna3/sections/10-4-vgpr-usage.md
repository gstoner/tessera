# 10.4. VGPR Usage

> RDNA3 ISA — pages 110–111

Opcode                      a16[0] acnt type               VGPRn[31:0]      VGPRn+1[31:0]   VGPRn+2[31:0]     VGPRn+3[31:0]
Gather                      0         1    2D              s                t
                                      2    Cube(Array) s                    t               face
                                      2    2D Array        s                t               slice
                            1         1    2D              t, s
                                      2    Cube(Array) t, s                 -, face
                                      2    2D Array        t, s             -, slice
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

The table below lists and briefly describes the legal suffixes for image instructions:

                                           Table 49. Sample Instruction Suffix Key
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

These are all packed into consecutive VGPRs, (may be non-consecutive if "NSA" is used), and can consist of up to
12 values.
  • Offset: SAMPLE*O*, GATHER*O*
    1 DWORD of 'offset_xyz' . The offsets are 6-bit signed integers: X=[5:0], Y=[13:8], Z=[21:16]
  • Bias: SAMPLE*B*, GATHER*B*. 1 DWORD float.
  • Z-compare: SAMPLE*C*, GATHER*C*. 1 DWORD.
  • Derivatives (SAMPLE_D): 2,4 or 6 DWORDS - these packed 1 DWORD per derivative as shown below (F32).
  • Body: One to four DWORDs, as defined by the table: Image Opcodes with a Sampler
    Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc.

    The number of components in "body" is the value of the ACNT field in the table, plus one.

Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc.

Note: Bias and Derivatives are mutually exclusive - the shader can use one or the other, but not both.

32-bit derivatives:

                   Image Dim           VGPR N      N+1       N+2       N+3       N+4         N+5
                   1D                  dx/dh       dx/dv     -         -         -           -
                   2D/cube             dx/dh       dy/dh     dx/dv     dy/dv     -           —
                   3D                  dx/dh       dy/dh     dz/dh     dx/dv     dy/dv       dz/dv

16-bit derivatives:

                   Image Type                   VGPR_D       VGPR_D+1      VGPR_D+2      VGPR_D+3
                   1 (1D, 1D Array)             16’hx, dx/dh 16’hx dx/dv   -             -
                   2 (2D, 2D Array, Cubemap)    dy/dh, dx/dh dy/dv, dx/dv -              -
                   3 (3D)                       dy/dh, dx/dh 16’hx, dz/dh dy/dv, dx/dv 16’hx, dz/dv

The "A16" instruction bit specifies that address components are 16 bits instead of the usual 32 bits.

Data :
   data is stored from or returned to 1-4 consecutive VGPRs. The amount of data loaded or stored is completely
   determined by the DMASK field of the instruction.

Loads
   DMASK specifies which elements of the resource are returned to consecutive VGPRs. The texture system
   loads data from memory and based on the data format expands it to a canonical RGBA form, filling in
   values for missing components based on T#.dst_sel. Then DMASK is applied and only those components
   selected are returned to the shader.

Stores
   When writing an image object, it is only possible to write an entire element (all components) - not only
   individual components. The components come from consecutive VGPRs and the texture system fill in the
   value zero for any missing components of the image’s data format, and ignore any values that are not part
   of the stored data format. For example if the DMASK=1001, the shader sends Red from VGPR_N and Alpha from
   VGPR_N+1 to the texture unit. If the image object is RGB, the texel is overwritten with Red from the VGPR_N,
   Green and Blue set to zero, and Alpha from the shader ignored. For D16=1, the DMASK has 1 bit set per 16-bits of
   data to be written from VGPRs to memory. The position of the bits in DMASK is irrelevant, only the number
   of bits set to 1.

"D16" instructions
   Load and store instructions also come in a "d16" variant. For stores, each 32bit VGPR holds two 16bit data
   elements that are passed to the texture unit which in turn, converts to the texture format before writing to
   memory. For loads, data returned from the texture unit is converted to 16 bits and a pair of data are stored
   in each 32bit VGPR (LSBs first, then MSBs). If there is only one component, the data goes into the lower half
   of the VGPR unless the "HI" instruction variant is used in which case the high-half of the VGPR is loaded
   with data.
