# 10.6. Image Sampler

> RDNA3.5 ISA — pages 115–116

Bits          Size        Name                 Comments
91:78         14          height               height-1 of mip 0 in texels
98:96         3           dst_sel_x            0 = 0, 1 = 1, 4 = R, 5 = G, 6 = B, 7 = A.
101:99        3           dst_sel_y
104:102       3           dst_sel_z
107:105       3           dst_sel_w
111:108       4           base level           largest mip level in the resource view. For MSAA, this should be set to 0
115:112       4           last level           smallest mip level in resource view. For MSAA, holds log2(number of
                                               samples).
123:121       3           BC Swizzle           Specifies channel ordering for border color data independent of the T#
                                               dst_sel_*s. Internal xyzw channels get the following border color
                                               channels as stored in memory. 0=xyzw, 1=xwyz, 2=wzyx, 3=wxyz, 4=zyxw,
                                               5=yxwz
127:124       4           type                 0 = buf, 8 = 1d, 9 = 2d, 10 = 3d, 11 = cube, 12 = 1d-array, 13 = 2d-array, 14 =
                                               2d-msaa, 15 = 2d-msaa-array. 1-7 are reserved.
256-bit Resource: 1d-array, 2d-array, 3d, cubemap, MSAA
140:128       13          depth                Depth-1 of Mip0 for a 3D map; last array slice for a 2D-array or 1D-array
                                               or cube-map; (pitch-1)[12:0] of mip0 for 1D, 2D, 2D-MSAA resources if
                                               pitch > width.
141           1           Pitch[13]            (pitch-1)[13] of mip0 for 1D, 2D and 2D-MSAA.
156:144       13          base array           First slice in array of the resource view.
163:160       4           array pitch          For 3D, bit 0 indicates SRV or UAV:
                                               0: SRV (base_array ignored, depth w.r.t. base map)
                                               1: UAV (base_array and depth are first and last layer in view, and w.r.t.
                                               mip level specified)
179:168       12          min lod warn         feedback trigger for LOD, u4.8 format
183           1           corner samples mod Describes how texels were generated in the resource. 0=center sampled,
                                             1 = corner sampled.
198:187       12          min_lod              smallest LOD allowed for PRTs, U4.8 format
198:187       12          min LOD              smallest LOD allowed for PRTs, u4.8 format.
202           1           Iterate 256          Indicates that compressed tiles in this surface have been flushed out to
                                               every 256B of the tile. Applies only to MSAA depth surfaces.
211           1           Meta Pipe Aligned    Maintains pipe alignment in metadata addressing (DCC and tiling)
213           1           Compression Enable enable delta color compression (DCC)
214           1           Alpha is on MSB      Set to 1 if the surface’s component swap is not reversed (DCC)
215           1           Color Transform      Auto=0, none=1 (DCC)
255:216       40          Meta Data Address    Upper bits of meta-data address (DCC) [47:8]

A resource that is all zeros is treated as 'unbound': it returns all zeros and not generate a memory transaction.
The "resource-level" field is ignored when checking for "all zeros".

10.6. Image Sampler
The sampler resource (also referred to as S#) defines what operations to perform on texture map data loaded
by sample instructions. These are primarily address clamping and filter options. Sampler resources are
defined in four consecutive SGPRs and are supplied to the texture cache with every sample instruction.

                                         Table 51. Image Sampler Definition

Bits       Size       Name                 Description
2:0        3          clamp x              Clamp/wrap mode:
                                           0: Wrap
                                           1: Mirror
5:3        3          clamp y              2: ClampLastTexel
                                           3: MirrorOnceLastTexel
                                           4: ClampHalfBorder
8:6        3          clamp z              5: MirrorOnceHalfBorder
                                           6: ClampBorder
                                           7: MirrorOnceBorder
11:9       3          max aniso ratio      0 = 1:1
                                           1 = 2:1
                                           2 = 4:1
                                           3 = 8:1
                                           4 = 16:1
14:12      3          depth compare func   0: Never
                                           1: Less
                                           2: Equal
                                           3: Less than or equal
                                           4: Greater
                                           5: Not equal
                                           6: Greater than or equal
                                           7: Always
15         1          force unnormalized   Force address cords to be unorm: 0 = address coordinates are
                                           normalized, in [0,1); 1 = address coordinates are unnormalized in the
                                           range [0,dim).
18:16      3          aniso threshold      threshold under which floor(aniso ratio) determines number of samples
                                           and step size
19         1          mc coord trunc       enables bilinear blend fraction truncation to 1 bit for motion
                                           compensation
20         1          force degamma        force format to srgb if data_format allows
26:21      6          aniso bias           6 bits, in u1.5 format.
27         1          trunc coord          selects texel coordinate rounding or truncation.
28         1          disable cube wrap    disables seamless DX10 cubemaps, allows cubemaps to clamp according
                                           to clamp_x and clamp_y fields
30:29      2          filter_mode          0 = Blend (lerp); 1 = min, 2 = max.
31         1          skip degamma         disabled degamma (sRGB→Linear) conversion.
43:32      12         min lod              minimum LOD ins resource view space (0.0 = T#.base_level) u4.8.
55:44      12         max lod              maximum LOD ins resource view space
77:64      14         lod bias             LOD bias s6.8.
83:78      6          lod bias sec         bias (s2.4) added to computed LOD
85:84      2          xy mag filter        Magnification filter: 0=point, 1=bilinear, 2=aniso-point, 3=aniso-linear
87:86      2          xy min filter        Minification filter: 0=point, 1=bilinear, 2=aniso-point, 3=aniso-linear
89:88      2          z filter             Volume Filter: 0=none (use XY min/mag filter), 1=point, 2=linear
91:90      2          mip filter           Mip level filter: 0=none (disable mipmapping,use base-leve), 1=point,
                                           2=linear
94         1          Blend PRT            For PRT fetches, bled the PRT_default valu for non-resident levels
107:96     12         border color ptr
