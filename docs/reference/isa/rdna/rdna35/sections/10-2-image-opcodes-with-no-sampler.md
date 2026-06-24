# 10.2. Image Opcodes with No Sampler

> RDNA3.5 ISA — pages 109–110

DWORDs long. The first address goes in the VADDR field, and subsequent addresses go into ADDR1-4. This 3
DWORD form of the instruction can supply up to 5 addresses.

NSA allows an image instruction to specify up to 5 unique address VGPRs. These are the rules for how
instructions requiring more than 5 addresses are handled with NSA. It is permissible to use non-NSA mode
where all addresses are in sequential VGPRs.

  • VADDR provides the first address component
  • ADDR1 provides the second address component
  • ADDR2 provides the third address component
  • ADDR3 provides the fourth address component
  • ADDR4 provides all additional components in sequential VGPRs: VADDR4, VADDR4+1, etc.

When using 16-bit addresses, each VGPR holds a pair of addresses and these cannot be located in different
VGPRs. The lower numbered 16-bit value is in the LSBs of the VGPR.

For Ray Tracing, the VGPRs are divided up into 5 groups of VGPRs. The VGPRs within each group must be
contiguous, but the groups can be scattered. The packing is different when A16=1 because RayDir.Z and
RayInvDir.x are in the same DWORD. In A16 mode, the RayDir and RayInvDir are merged into 3 VGPRs but in a
different order: RayDir and RayInvDir per component share a VGPR.

10.2. Image Opcodes with No Sampler
For image opcodes with no sampler, all VGPR address values are taken as uint.
For cubemaps, face_id = slice * 6 + face.

MSAA surfaces support only load, store and atomics; not load-mip or store-mip.

The table below shows the contents of address VGPRs for the various image opcodes.

Opcode                 a16[0] type            acnt VGPRn[31:0]   VGPRn+1[31:0]   VGPRn+2[31:0]   VGPRn+3[31:0]
GET_RESINFO            x      Any             0    mipid
MSAA_LOAD              0      2D MSAA         2    s             t               fragid
                              2D Array MSAA   3    s             t               slice           fragid
                       1      2D MSAA         2    t, s          -, fragid
                              2D Array MSAA   3    t, s          fragid, slice

Opcode                 a16[0] type            acnt VGPRn[31:0]   VGPRn+1[31:0]   VGPRn+2[31:0]   VGPRn+3[31:0]
LOAD                   0      1D              0    s
LOAD_PCK                      2D              1    s             t
LOAD_PCK_SGN                  3D              2    s             t               r
STORE
                              Cube/Cube Array 2    s             t               face
STORE_PCK
                              1D Array        1    s             slice
                              2D Array        2    s             t               slice
                              2D MSAA         2    s             t               fragid
                              2D Array MSAA   3    s             t               slice           fragid
                       1      1D              0    -, s
                              2D              1    t, s
                              3D              2    t, s          -, r
                              Cube/Cube Array 2    t, s          -, face
                              1D Array        1    slice, s
                              2D Array        2    t, s          -, slice
                              2D MSAA         2    t, s          -, fragid
                              2D Array MSAA   3    t, s          fragid, slice
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
  • Image_Atomic_*: swap, cmpswap, add, sub, {u,s}{min,max}, and, or, xor, inc, dec.

"ACNT" is the Address Count: the number of VGPRs that supply the "body" of the address, derived from the
