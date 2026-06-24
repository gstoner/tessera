# 10.8. Vector Memory Instruction Data Dependencies

> RDNA3.5 ISA — pages 118–118

#           Format                       #          Format                    #          Format
                                                                              116        BC4_SNORM
                                                                              117        BC5_UNORM
                                                                              118        BC5_SNORM
                                                                              119        BC6_UFLOAT
                                                                              120        BC6_SFLOAT
                                                                              121        BC7_UNORM
                                                                              122        BC7_SRGB
                                                                              205        YCBCR_UNORM
                                                                              206        YCBCR_SRGB

10.8. Vector Memory Instruction Data Dependencies
When a VM instruction is issued, it schedules the reads of address and store-data from VGPRs to be sent to the
texture unit. Any ALU instruction that attempts to write this data before it has been sent to the texture unit is
stalled.

The shader developer’s responsibility to avoid data hazards associated with VMEM instructions include waiting
for VMEM load instruction completion before reading data fetched from the cache (VMCNT and VSCNT).

This is explained in the section: Data Dependency Resolution

10.9. Ray Tracing
Ray Tracing support includes the following instructions:
    • IMAGE_BVH_INTERSECT_RAY
    • IMAGE_BVH64_INTERSECT_RAY

These instructions receive ray data from the VGPRs and fetch BVH (Bounding Volume Hierarchy) from
memory.
    • Box BVH nodes perform 4x Ray/Box intersection, sorts the 4 children based on intersection distance and
      returns the child pointers and hit status.
    • Triangle nodes perform 1 Ray/Triangle intersection test and returns the intersection point and triangle ID.

The two instructions are identical, except that the "64" version supports a 64-bit address while the normal
version supports only a 32bit address. Both instructions can use the "A16" instruction field to reduce some (but
not all) of the address components to 16 bits (from 32). These addresses are: ray_dir and ray_inv_dir.

10.9.1. Instruction definition and fields

    image_bvh_intersect_ray vgpr_d[4], vgpr_a[11], sgpr_r[4]
    image_bvh_intersect_ray vgpr_d[4], vgpr_a[8], sgpr_r[4] A16=1
    image_bvh64_intersect_ray vgpr_d[4], vgpr_a[12], sgpr_r[4]
    image_bvh64_intersect_ray vgpr_d[4], vgpr_a[9], sgpr_r[4]    A16=1
