# 10.1.3. A16 Instructions

> RDNA4 ISA — pages 130–130

VGPR.

10.1.3. A16 Instructions
The A16 instruction bit indicates that the address components are 16 bits instead of the usual 32 bits.
Components are packed such that the first address component goes into the low 16 bits ([15:0]), and the next
into the high 16 bits ([31:16]).

10.1.4. G16 Instructions
The instructions with "G16" in the name mean the user provided derivatives are 16 bits instead of the usual 32
bits. Derivatives are packed such that the first derivative goes into the low 16 bits ([15:0]), and the next into the
high 16 bits ([31:16]).

10.2. Image Opcodes with No Sampler
For image opcodes with no sampler, all VGPR address values are taken as uint.
For cubemaps, face_id = slice * 6 + face.

MSAA surfaces support only load, store and atomics; not load-mip or store-mip.

The table below shows the contents of address VGPRs for the various image opcodes.

Opcode                 a16[0] type              acnt VGPRn[31:0]     VGPRn+1[31:0]    VGPRn+2[31:0]   VGPRn+3[31:0]
GET_RESINFO            x      Any               0     mipid
LOAD                   0      1D                0     s
LOAD_PCK                      2D                1     s              t
LOAD_PCK_SGN                  3D                2     s              t                r
STORE
                              Cube/Cube Array 2       s              t                face
STORE_PCK
                              1D Array          1     s              slice
                              2D Array          2     s              t                slice
                              2D MSAA           2     s              t                fragid
                              2D Array MSAA     3     s              t                slice           fragid
                       1      1D                0     -, s
                              2D                1     t, s
                              3D                2     t, s           -, r
                              Cube/Cube Array 2       t, s           -, face
                              1D Array          1     slice, s
                              2D Array          2     t, s           -, slice
                              2D MSAA           2     t, s           -, fragid
                              2D Array MSAA     3     t, s           fragid, slice
