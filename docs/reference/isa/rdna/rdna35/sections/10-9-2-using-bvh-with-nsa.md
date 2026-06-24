# 10.9.2. Using BVH with NSA

> RDNA3.5 ISA — pages 119–119

                                          Table 53. Ray Tracing VGPR Contents
VGPR_ BVH A16=0                  BVH A16=1                       BVH64 A16=0                  BVH64 A16=1
A
0         node_pointer (u32)     node_pointer (u32)              node_pointer [31:0] (u32)    node_pointer [31:0] (u32)
1         ray_extent (f32)       ray_extent (f32)                node_pointer [63:32] (u32)   node_pointer [63:32] (u32)
2         ray_origin.x (f32)     ray_origin.x (f32)              ray_extent (f32)             ray_extent (f32)
3         ray_origin.y (f32)     ray_origin.y (f32)              ray_origin.x (f32)           ray_origin.x (f32)
4         ray_origin.z (f32)     ray_origin.z (f32)              ray_origin.y (f32)           ray_origin.y (f32)
5         ray_dir.x (f32)        [15:0] = ray_dir.x (f16)        ray_origin.z (f32)           ray_origin.z (f32)
                                 [31:16] = ray_inv_dir.x (f16)
6         ray_dir.y (f32)        [15:0] = ray_dir.y (f16)        ray_dir.x (f32)              [15:0] = ray_dir.x (f16)
                                 [31:16] = ray_inv_dir.y(f16)                                 [31:16] = ray_inv_dir.x (f16)
7         ray_dir.z (f32)        [15:0] = ray_dir.z (f16)        ray_dir.y (f32)              [15:0] = ray_dir.y (f16)
                                 [31:16] = ray_inv_dir.z (f16)                                [31:16] = ray_inv_dir.y(f16)
8         ray_inv_dir.x (f32)    unused                          ray_dir.z (f32)              [15:0] = ray_dir.z (f16)
                                                                                              [31:16] = ray_inv_dir.z (f16)
9         ray_inv_dir.y (f32)    unused                          ray_inv_dir.x (f32)          unused
10        ray_inv_dir.z (f32)    unused                          ray_inv_dir.y (f32)          unused
11        unused                 unused                          ray_inv_dir.z (f32)          unused

Vgpr_d[4] are the destination VGPRs of the results of intersection testing. The values returned here are
different depending on the type of BVH node that was fetched. For box nodes the results contain the 4 pointers
of the children boxes in intersection time sorted order. For triangle BVH nodes the results contain the
intersection time and triangle ID of the triangle tested.

Sgpr_r[4] is the texture descriptor for the operation. The instruction is encoded with use_128bit_resource=1.

Restrictions on image_bvh instructions
    • DMASK must be set to 0xf (instruction returns all four DWORDs)
    • D16 must be set to 0 (16 bit return data is not supported)
    • R128 must be set to 1 (256 bit T#s are not supported)
    • UNRM must be set to 1 (only unnormalized coordinates are supported)
    • DIM must be set to 0 (BVH textures are 1D)
    • LWE must be set to 0 (LOD warn is not supported)
    • TFE must be set to 0 (no support for writing out the extra DWORD for the PRT hit status)
    • SSAMP must be set to 0 (just a placeholder, since samplers are not used by the instruction)

The return order settings of the BVH ops are ignored instead they use the in-order load return queue.

10.9.2. Using BVH with NSA
When using the BVH instruction with Non-Sequential Address, the BVH components fall into 5 groups each of
which is specified by a NSA address VGPR.
    • node pointer : 1 vgpr
    • ray extent : 1 vgpr
    • ray origin : 3 consecutive vgprs
    • ray dir : 3 consecutive vgprs
    • ray inv dir : 3 consecutive vgprs (paired with ray-dir for 16-bit addresses)
