# 10.3. Image Opcodes with a Sampler

> RDNA3 ISA — pages 109–109

instruction’s DIM field and the opcode.

10.3. Image Opcodes with a Sampler
Opcodes with a sampler: all VGPR address values are taken as FLOAT except for Texel-offset which are UINT.
For cubemaps, face_id = slice * 8 + face.
(Note that the "*8" differs from the non-sampler case which is "*6").
Certain sample and gather opcodes require additional values from VGPRs beyond what is shown in the table
below. These values are: offset, bias, z-compare and gradients. Please see the next section for details. MSAA
surfaces do not support sample or gather4 operations.

Opcode                     a16[0] acnt type           VGPRn[31:0]   VGPRn+1[31:0]   VGPRn+2[31:0]   VGPRn+3[31:0]
Sample                     0         0   1D           s
GetLod                               1   2D           s             t
                                     2   3D           s             t               r
                                     2   Cube(Array) s              t               face
                                     1   1D Array     s             slice
                                     2   2D Array     s             t               slice
                           1         0   1D           -, s
                                     1   2D           t, s
                                     2   3D           t, s          -, r
                                     2   Cube(Array) t, s           -, face
                                     1   1D Array     slice, s
                                     2   2D Array     t, s          -, slice
Sample "_L":               0         1   1D           s             lod
                                     2   2D           s             t               lod
                                     3   3D           s             t               r               lod
                                     3   Cube(Array) s              t               face            lod
                                     2   1D Array     s             slice           lod
                                     3   2D Array     s             t               slice           lod
                           1         1   1D           lod, s
                                     2   2D           t, s          -, lod
                                     3   3D           t, s          lod, r
                                     3   Cube(Array) t, s           lod, face
                                     2   1D Array     slice, s      -, lod
                                     3   2D Array     t, s          lod, slice
Sample "_CL":              0         1   1D           s             clamp
                                     2   2D           s             t               clamp
                                     3   3D           s             t               r               clamp
                                     3   Cube(Array) s              t               face            clamp
                                     2   1D Array     s             slice           clamp
                                     3   2D Array     s             t               slice           clamp
                           1         1   1D           clamp, s
                                     2   2D           t, s          -, clamp
                                     3   3D           t, s          clamp, r
                                     3   Cube(Array) t, s           clamp, face
                                     2   1D Array     slice, s      -, clamp
                                     3   2D Array     t, s          clamp, slice
