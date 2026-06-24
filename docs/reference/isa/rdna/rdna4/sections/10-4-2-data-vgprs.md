# 10.4.2. Data VGPRs

> RDNA4 ISA — pages 134–134

Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc.

Note: Bias and Derivatives are mutually exclusive - the shader can use one or the other, but not both.

32-bit derivatives:

                   Image Dim           VGPR N      N+1       N+2       N+3       N+4         N+5
                   1D                  dx/dh       dx/dv     -         -         -           -
                   2D/cube             dx/dh       dy/dh     dx/dv     dy/dv     -           —
                   3D                  dx/dh       dy/dh     dz/dh     dx/dv     dy/dv       dz/dv

16-bit derivatives:

                   Image Type                   VGPR_N       VGPR_N+1      VGPR_N+2      VGPR_N+3
                   1 (1D, 1D Array)             16’hx, dx/dh 16’hx dx/dv   -             -
                   2 (2D, 2D Array, Cubemap)    dy/dh, dx/dh dy/dv, dx/dv -              -
                   3 (3D)                       dy/dh, dx/dh 16’hx, dz/dh dy/dv, dx/dv 16’hx, dz/dv

10.4.1. Address VGPRs
Image and Sample instructions support multiple address VGPRs to allow the shader to perform a gather
operation of address components. This allows an image instruction to specify up to 5 unique address VGPRs.

  • VADDR provides the first address component
  • VADDR1 provides the second address component
  • VADDR2 provides the third address component
  • VADDR3 (for VIMAGE) provides the fourth address component for VIMAGE, or for
    VSAMPLE provides all additional components in sequential VGPRs: VADDR3, VADDR3+1, etc.
  • VADDR4 (for VIMAGE) provides all additional components in sequential VGPRs: VADDR4, VADDR4+1, etc.

The "A16" instruction bit specifies that address components are 16 bits instead of the usual 32 bits. When using
16-bit addresses, each VGPR holds a pair of addresses and these cannot be located in different VGPRs. The
lower numbered 16-bit value is in the LSBs of the VGPR.

For Ray Tracing, the VGPRs are divided up into 5 groups of VGPRs. The VGPRs within each group must be
contiguous, but the groups can be scattered. The packing is different when A16=1 because RayDir.Z and
RayInvDir.x are in the same DWORD. In A16 mode, the RayDir and RayInvDir are merged into 3 VGPRs but in a
different order: RayDir and RayInvDir per component share a VGPR.

10.4.2. Data VGPRs
Data :
   data is stored from or returned to 1-4 consecutive VGPRs. The amount of data loaded or stored is completely
   determined by the DMASK field of the instruction.

Loads
   DMASK specifies which elements of the resource are returned to consecutive VGPRs. The texture system
   loads data from memory and based on the data format expands it to a canonical RGBA form, filling in
