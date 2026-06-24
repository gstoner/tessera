# 10.9.2. VGPR_A Field Organization

> RDNA4 ISA — pages 142–143

tested. For Instance nodes, the data consists of the 64 bit BVH base, the node pointer offset in the BVH, a 24 bit
user data field, and 8 bit instance mask. See section "Intersection Engine Return Data" for more information.

The last two DWORDs contain the ShapeID/GeoID of each triangle tested.

IMAGE_BVH8_INTERSECT_RAY:

vgpr_d[10] are the destination VGPRs of the results of intersection testing. The values returned here are
different depending on the type of BVH node that was fetched.

For box nodes the results contain the 8 pointers of the children boxes in intersection time sorted order. When
wide sorting is disabled, the first 4 values correspond to the results from boxes 0-3, while the second 4 values
correspond to the results of boxes 4-7. If wide sorting is enabled, the 8 values are fully intermixed and contain
results for boxes 0-7 (as dictated by the 8 wide sort). For triangle BVH nodes the results contain the intersection
time and triangle ID or barycentrics of both the triangles tested. For Instance nodes, the data consists of the 64
bit BVH base, the node pointer offset in the BVH, a 24 bit user data field, and 8 bit instance mask. See section
"Intersection Engine Return Data" for more information.

The last two DWORDs contain the ShapeID/GeoID of each triangle tested.

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

10.9.2. VGPR_A Field Organization
The VIMAGE instruction encoding specifies 5 VADDR fields that are used with BVH instructions as follows:

node pointer                      VADDR0: 1 vgpr
ray extent                        VADDR1: 1 vgpr
ray origin                        VADDR2: 3 consecutive vgprs
ray dir                           VADDR3: 3 consecutive vgprs
ray inv dir                       VADDR4: 3 consecutive vgprs (paired with ray-dir for 16-bit addresses)

When using A16=1 mode, ray-dir and ray-inv-dir share the same vgprs and ADDR4 is unused.

10.9.3. BVH Texture Resource Definition
The T# used with these BVH instructions is different from other image instructions.

                                                Table 65. BVH Resource Definition
Bits    Size     Field                       Description
39:0    40       base_address[47:8]          Base address of the BVH texture, 256 byte aligned.
51:40   12       Reserved                    Must be zero.
52      1        sort_triangles_first        0: Pointers to triangle nodes are not treated differently during child sorting.
                                             1: Pointers to triangle nodes (type 0 and 1 for image_bvh8, type 0,1,2,3 for all other
                                             image_bvh ops) are sorted before valid box nodes.
54:53   2        box_sorting_heuristic       Specifies which heuristic should be utilized for sorting children when box sorting
                                             is enabled.
                                             0: Closest Traversal is ordered to enter the children that intersect the ray closer to
                                             the ray origin first.
                                             1: LargestFirst Traversal is ordered to enter the children that have the largest
                                             interval where the box intersects the ray first.
                                             2: ClosestMidpoint Traversal is ordered to enter the children that have a midpoint
                                             in the interval, where the box intersects that has the lowest intersection time
                                             before clamping.
                                             3: Undefined Reserved
62:55   8        box_grow_value              UINT — used to extend the MAX plane of the box intersection
63      1        box_sort_en                 boolean to enable sorting the box intersect results
105:64 42        size[47:6]                  In units of 64 bytes. Represents the number of nodes in BVH texture minus 1. Used
                                             for bounds checking.
115:106 10       Reserved                    Must be zero.
116     1        box_node_64B                0: node type 4 is FP16 box node
                                             1: node type 4 is 64B high precision box node
117     1        wide_sort_en                0: sort across 4 box children
                                             1: sort across 8 box children
118     1        instance_en                 0: node 6 is user node
                                             1: node 6 is instance node
119     1        pointer_flags               0: Do not use pointer flags or features supported by pointer flags.
                                             1: Utilize pointer flags to enable HW winding, back face cull, opaque/non-opaque
                                             culling and primitive type based culling.
120     1        triangle_return_mode        0: return hit/miss with triangle test result dword[3:0] = {t_num, t_denom,
                                             triangle_id, hit_status}
                                             1: return barycentrics with triangle test result dword[3:0] = {t_num, t_denom,
                                             I_nim, J_num}
123:121 3        Reserved                    Must be zero.
127:124 4        type                        Must be set to 0x08.

Barycentrics
The ray-tracing hardware is designed to support computation of barycentric coordinates directly in hardware.
This uses the "triangle_return_mode" in the table in the previous section (T# descriptor).

                                               Table 66. Ray Tracing Return Mode
DWORD        Return Mode =0                                                  Return Mode = 1
             Field Name            Type                                      Field Name            Type
0            t_num                 float32                                   t_num                 float32
1            t_denom               float32                                   t_denom               float32
2            triangle_id           uint32                                    I_num                 float32
3            hit_status            uint32 (boolean value)                    J_num                 float32
