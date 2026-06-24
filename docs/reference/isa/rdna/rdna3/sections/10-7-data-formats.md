# 10.7. Data Formats

> RDNA3 ISA — pages 115–115

Bits       Size      Name                         Description
127:126    2         border color type            Opaque-black, transparent-black, white, use border color ptr.
                                                  0: Transparent Black
                                                  1: Opaque Black
                                                  2: Opaque White
                                                  3: Register (User border color, pointed to by border_color_ptr)"

10.7. Data Formats
The table below details all the data formats that can be used by image and buffer resources.

                                         Table 52. Buffer and Image Data Formats
#          Format                          #         Format                         #           Format
0          INVALID                         31        11_11_10_FLOAT                 64          8_SRGB
1          8_UNORM                         32        10_10_10_2_UNORM               65          8_8_SRGB
2          8_SNORM                         33        10_10_10_2_SNORM               66          8_8_8_8_SRGB
3          8_USCALED                       34        10_10_10_2_UINT                67          5_9_9_9_FLOAT
4          8_SSCALED                       35        10_10_10_2_SINT                68          5_6_5_UNORM
5          8_UINT                          36        2_10_10_10_UNORM               69          1_5_5_5_UNORM
6          8_SINT                          37        2_10_10_10_SNORM               70          5_5_5_1_UNORM
7          16_UNORM                        38        2_10_10_10_USCALED             71          4_4_4_4_UNORM
8          16_SNORM                        39        2_10_10_10_SSCALED             72          4_4_UNORM
9          16_USCALED                      40        2_10_10_10_UINT                73          1_UNORM
10         16_SSCALED                      41        2_10_10_10_SINT                74          1_REVERSED_UNORM
11         16_UINT                         42        8_8_8_8_UNORM                  75          32_FLOAT_CLAMP
12         16_SINT                         43        8_8_8_8_SNORM                  76          8_24_UNORM
13         16_FLOAT                        44        8_8_8_8_USCALED                77          8_24_UINT
14         8_8_UNORM                       45        8_8_8_8_SSCALED                78          24_8_UNORM
15         8_8_SNORM                       46        8_8_8_8_UINT                   79          24_8_UINT
16         8_8_USCALED                     47        8_8_8_8_SINT                   80          X24_8_32_UINT
17         8_8_SSCALED                     48        32_32_UINT                     81          X24_8_32_FLOAT
18         8_8_UINT                        49        32_32_SINT                     82          GB_GR_UNORM
19         8_8_SINT                        50        32_32_FLOAT                    83          GB_GR_SNORM
20         32_UINT                         51        16_16_16_16_UNORM              84          GB_GR_UINT
21         32_SINT                         52        16_16_16_16_SNORM              85          GB_GR_SRGB
22         32_FLOAT                        53        16_16_16_16_USCALED            86          BG_RG_UNORM
23         16_16_UNORM                     54        16_16_16_16_SSCALED            87          BG_RG_SNORM
24         16_16_SNORM                     55        16_16_16_16_UINT               88          BG_RG_UINT
25         16_16_USCALED                   56        16_16_16_16_SINT               89          BG_RG_SRGB
26         16_16_SSCALED                   57        16_16_16_16_FLOAT
27         16_16_UINT                      58        32_32_32_UINT                              Compressed Formats
28         16_16_SINT                      59        32_32_32_SINT                  109         BC1_UNORM
29         16_16_FLOAT                     60        32_32_32_FLOAT                 110         BC1_SRGB
30         10_11_11_FLOAT                  61        32_32_32_32_UINT               111         BC2_UNORM
                                           62        32_32_32_32_SINT               112         BC2_SRGB
                                           63        32_32_32_32_FLOAT              113         BC3_UNORM
                                                                                    114         BC3_SRGB
                                                                                    115         BC4_UNORM
