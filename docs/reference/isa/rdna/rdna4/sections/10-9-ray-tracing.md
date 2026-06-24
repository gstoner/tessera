# 10.9. Ray Tracing

> RDNA4 ISA — pages 139–139

#          Format                      #            Format                       #          Format
24         16_16_SNORM                 56           16_16_16_16_SINT             88         BG_RG_UINT
25         16_16_USCALED               57           16_16_16_16_FLOAT            89         BG_RG_SRGB
26         16_16_SSCALED               58           32_32_32_UINT
27         16_16_UINT                  59           32_32_32_SINT                           Compressed Formats
28         16_16_SINT                  60           32_32_32_FLOAT               109        BC1_UNORM
29         16_16_FLOAT                 61           32_32_32_32_UINT             110        BC1_SRGB
30         10_11_11_FLOAT              62           32_32_32_32_SINT             111        BC2_UNORM
31         11_11_10_FLOAT              63           32_32_32_32_FLOAT            112        BC2_SRGB
                                                                                 113        BC3_UNORM
                                                                                 114        BC3_SRGB
                                                                                 115        BC4_UNORM
                                                                                 116        BC4_SNORM
                                                                                 117        BC5_UNORM
                                                                                 118        BC5_SNORM
                                                                                 119        BC6_UFLOAT
                                                                                 120        BC6_SFLOAT
                                                                                 121        BC7_UNORM
                                                                                 122        BC7_SRGB
                                                                                 205        YCBCR_UNORM
                                                                                 206        YCBCR_SRGB
                                                                                 227        6E4_FLOAT

10.8. Vector Memory Instruction Data Dependencies
When a VM instruction is issued, it schedules the reads of address and store-data from VGPRs to be sent to the
texture unit. Any ALU instruction that attempts to write this data before it has been sent to the texture unit is
stalled.

The shader developer’s responsibility to avoid data hazards associated with VMEM instructions include waiting
for VMEM read instruction completion before reading data fetched from the TC (LOADcnt and STOREcnt).
Ray-tracing Image BVH instructions are tracked with BVHcnt.

This is explained in the section: Data Dependency Resolution

10.9. Ray Tracing
Ray Tracing support includes the following instructions:

            IMAGE_BVH_INTERSECT_RAY                      tests a single QBVH node referenced by a 32 bit
                                                         node pointer per lane
            IMAGE_BVH64_INTERSECT_RAY                    tests a single QBVH node referenced by a 64 bit
                                                         node pointer per lane
            IMAGE_BVH_DUAL_INTERSECT_RAY                 tests two QBVH nodes referenced by a 64 bit BVH
                                                         base and two 32 bit node offsets
            IMAGE_BVH8_INTERSECT_RAY                     tests a BVH8 node referenced by a 64 bit BVH
                                                         base and a 32 bit node offset
