# 10.3. Image Opcodes with a Sampler

> RDNA4 ISA — pages 131–132

Opcode                 a16[0] type            acnt VGPRn[31:0]   VGPRn+1[31:0]   VGPRn+2[31:0]   VGPRn+3[31:0]
ATOMIC                 0      1D              0    s
                              2D              1    s             t
                              3D              2    s             t               r
                              1D Array        1    s             slice
                              2D Array        2    s             t               slice
                              2D MSAA         2    s             t               fragid
                              2D Array MSAA   3    s             t               slice           fragid
                       1      1D              0    -, s
                              2D              1    t, s
                              3D              2    t, s          -, r
                              1D Array        1    slice, s
                              2D Array        2    t, s          -, slice
                              2D MSAA         2    t, s          -, fragid
                              2D Array MSAA   3    t, s          fragid, slice
LOAD_MIP         0            1D              1    s             mipid
LOAD_MIP_PCK                  2D              2    s             t               mipid
LOAD_MIP_PCK_SGN              3D              3    s             t               r               mipid
STORE_MIP
                              Cube/Cube Array 3    s             t               face            mipid
STORE_MIP_PCK
                              1D Array        2    s             slice           mipid
                              2D Array        3    s             t               slice           mipid
                       1      1D              1    mipid, s
                              2D              2    t, s          -, mipid
                              3D              3    t, s          mipid, r
                              Cube/Cube Array 3    t, s          mipid, face
                              1D Array        2    slice, s      -, mipid
                              2D Array        3    t, s          mipid, slice

  • Image_Load : image_load, image_load_mip, image_load_{pck, pck_sgn, mip_pck, mip_pck_sgn}
  • Image_Store: image_store, image_store_mip
  • Image_Atomic_*: swap, cmpswap, add, sub, {u,s}{min,max}, and, or, xor, inc, dec, add_f32, min_f32,
    max_f32.

"ACNT" is the Address Count: the number of VGPRs that supply the "body" of the address, derived from the
instruction’s DIM field and the opcode.

10.3. Image Opcodes with a Sampler
Opcodes with a sampler: all VGPR address values are taken as FLOAT except for Texel-offset, which are UINT.
For cubemaps, face_id = slice * 8 + face.
(Note that the "*8" differs from the non-sampler case which is "*6").
Certain sample and gather opcodes require additional values from VGPRs beyond what is shown in the table
below. These values are: offset, bias, z-compare and gradients. Please see the next section for details. MSAA
surfaces do not support sample or gather4 operations. MSAA_LOAD does not use a sampler, but uses the
VSAMPLE instruction encoding (users should set SAMP = NULL).

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
Gather                     0         1   2D           s             t
                                     2   Cube(Array) s              t               face
                                     2   2D Array     s             t               slice
                           1         1   2D           t, s
                                     2   Cube(Array) t, s           -, face
                                     2   2D Array     t, s          -, slice
