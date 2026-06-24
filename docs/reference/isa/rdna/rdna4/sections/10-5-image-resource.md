# 10.5. Image Resource

> RDNA4 ISA — pages 136–136

        FORMAT                         VGPR data format                               If D16==1
        others                         32-bit float                                   16 bit float
        Atomics                        depends on opcode: uint or float               -

10.5. Image Resource
The image resource (also referred to as T#) defines the location of the image buffer in memory, its dimensions,
tiling, and data format. These resources are stored in four or eight consecutive SGPRs and are read by image
instructions. All undefined or reserved bit must be set to zero unless otherwise specified.

                                            Table 60. Image Resource Definition
Bits       Size   Name                 Comments
128-bit Resource: 1D-tex, 2d-tex, 2d-msaa (multi-sample anti-aliasing)
39:0       40     base address         256-byte aligned (represents bits 47:8).
48:44      5      max mip              MSAA resources: holds Log2(number of samples); others holds maximum mip level:
                                       MipLevels-1. This describes the resource, not the resource view (like
                                       base_level/last_level).
56:49      8      format               Memory Data format
61:57      5      base level           largest mip level in the resource view. For MSAA, this should be set to 0.
77:62      16     width                width-1 of mip 0 in texels
93:78      16     height               height-1 of mip 0 in texels
98:96      3      dst_sel_x            0 = 0, 1 = 1, 4 = R, 5 = G, 6 = B, 7 = A.
101:99     3      dst_sel_y
104:102    3      dst_sel_z
107:105    3      dst_sel_w
115:111    5      last level           smallest mip level in resource view. For MSAA, holds log2(number of samples).
123:121    3      BC Swizzle           Specifies channel ordering for border color data independent of the T# dst_sel_*s.
                                       Internal xyzw channels get the following border color channels as stored in memory.
                                       0=xyzw, 1=xwyz, 2=wzyx, 3=wxyz, 4=zyxw, 5=yxwz
127:124    4      type                 0 = buf, 8 = 1d, 9 = 2d, 10 = 3d, 11 = cube, 12 = 1d-array, 13 = 2d-array, 14 = 2d-msaa, 15
                                       = 2d-msaa-array. 1-7 are reserved.
256-bit Resource: 1d-array, 2d-array, 3d, cubemap, MSAA
141:128    14     depth                3D resources: (depth-1) of mip 0.
                                       1D or 2D Array, cube resources: last array slice (see type table below). Max 8k - bit 14
                                       not used.
                                       1D or 2D, MSAA resources: LSBs of pitch-1. i.e. (pitch-1)[13:0] of mip 0, if pitch >
                                       width. The pitch_msb field contains the MSBs of the pitch field.
                                       Other resources: Must be zero
143:142    2      pitch_msb            1D or 2D, MSAA resources: MSBs of pitch-1. i.e. (pitch-1)[15:14]] of mip 0, if pitch >
                                       width.
                                       Other resources: Must be zero
156:144    13     base array           First slice in array of the resource view.
164        1      UAV3D                3D resources: bit 0 indicates SRV or UAV:
                                       0: SRV (base_array ignored, depth w.r.t. base map)
                                       1: UAV (base_array and depth are first and last layer in view, and w.r.t. mip level
                                       specified)
                                       Other resources: Not used
177:165    13     min_lod_warn         feedback trigger for LOD, u5.8 format
