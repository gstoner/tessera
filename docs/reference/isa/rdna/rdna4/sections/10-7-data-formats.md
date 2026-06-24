# 10.7. Data Formats

> RDNA4 ISA — pages 138–138

Bits       Size      Name                         Description
31         1         skip degamma                 disabled degamma (sRGB→Linear) conversion.
44:32      13        min lod                      minimum LOD ins resource view space (0.0 = T#.base_level) u5.8.
57:45      13        max lod                      maximum LOD ins resource view space
77:64      14        lod bias                     LOD bias s6.8.
83:78      6         lod bias sec                 bias (s2.4) added to computed LOD
85:84      2         xy mag filter                Magnification filter: 0=point, 1=bilinear, 2=aniso-point, 3=aniso-linear
87:86      2         xy min filter                Minification filter: 0=point, 1=bilinear, 2=aniso-point, 3=aniso-linear
89:88      2         z filter                     Volume Filter: 0=none (use XY min/mag filter), 1=point, 2=linear
91:90      2         mip filter                   Mip level filter: 0=none (disable mipmapping,use base-level), 1=point,
                                                  2=linear
125:114    12        border color ptr             index to border color space
127:126    2         border color type            Opaque-black, transparent-black, white, use border color ptr.
                                                  0: Transparent Black
                                                  1: Opaque Black
                                                  2: Opaque White
                                                  3: Register (User border color, pointed to by border_color_ptr)"

10.7. Data Formats
The table below details all the data formats that can be used by image and buffer resources.

                                         Table 62. Buffer and Image Data Formats
#          Format                          #         Format                          #           Format
0          INVALID                         32        10_10_10_2_UNORM                64          8_SRGB
1          8_UNORM                         33        10_10_10_2_SNORM                65          8_8_SRGB
2          8_SNORM                         34        10_10_10_2_UINT                 66          8_8_8_8_SRGB
3          8_USCALED                       35        10_10_10_2_SINT                 67          5_9_9_9_FLOAT
4          8_SSCALED                       36        2_10_10_10_UNORM                68          5_6_5_UNORM
5          8_UINT                          37        2_10_10_10_SNORM                69          1_5_5_5_UNORM
6          8_SINT                          38        2_10_10_10_USCALED              70          5_5_5_1_UNORM
7          16_UNORM                        39        2_10_10_10_SSCALED              71          4_4_4_4_UNORM
8          16_SNORM                        40        2_10_10_10_UINT                 72          4_4_UNORM
9          16_USCALED                      41        2_10_10_10_SINT                 73          1_UNORM
10         16_SSCALED                      42        8_8_8_8_UNORM                   74          1_REVERSED_UNORM
11         16_UINT                         43        8_8_8_8_SNORM                   75          32_FLOAT_CLAMP
12         16_SINT                         44        8_8_8_8_USCALED                 76          8_24_UNORM
13         16_FLOAT                        45        8_8_8_8_SSCALED                 77          8_24_UINT
14         8_8_UNORM                       46        8_8_8_8_UINT                    78          24_8_UNORM
15         8_8_SNORM                       47        8_8_8_8_SINT                    79          24_8_UINT
16         8_8_USCALED                     48        32_32_UINT                      80          X24_8_32_UINT
17         8_8_SSCALED                     49        32_32_SINT                      81          X24_8_32_FLOAT
18         8_8_UINT                        50        32_32_FLOAT                     82          GB_GR_UNORM
19         8_8_SINT                        51        16_16_16_16_UNORM               83          GB_GR_SNORM
20         32_UINT                         52        16_16_16_16_SNORM               84          GB_GR_UINT
21         32_SINT                         53        16_16_16_16_USCALED             85          GB_GR_SRGB
22         32_FLOAT                        54        16_16_16_16_SSCALED             86          BG_RG_UNORM
23         16_16_UNORM                     55        16_16_16_16_UINT                87          BG_RG_SNORM
