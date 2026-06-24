# 10.9.1. Instruction definition and fields

> RDNA4 ISA — pages 140–141

Ray tracing instruction support also includes BVH stack operations in LDS. See Ray Tracing Stack Ops.

These instructions receive ray data from the VGPRs and fetch BVH (Bounding Volume Hierarchy) from
memory.
    • Box BVH nodes perform 4x Ray/Box intersection, sorts the 4 children based on intersection distance and
      returns the child pointers and hit status.
    • Triangle nodes perform 1 Ray/Triangle intersection test and returns the intersection point and triangle ID.

The first two instructions are identical, except that the "64" version supports a 64-bit address while the normal
version supports only a 32bit address. These instructions can use the "A16" instruction field to reduce some
(but not all) of the address components to 16 bits (from 32), except for image_bvh_dual_intersect_ray and
image_bvh8_intersect_ray that do not support A16=1. The addresses that can be 16-bits are: ray_dir and
ray_inv_dir.

10.9.1. Instruction definition and fields

        image_bvh_intersect_ray          vgpr_d[4], vgpr_a[11], sgpr_r[4]
        image_bvh_intersect_ray          vgpr_d[4], vgpr_a[8], sgpr_r[4]     A16=1
        image_bvh64_intersect_ray        vgpr_d[4], vgpr_a[12], sgpr_r[4]
        image_bvh64_intersect_ray        vgpr_d[4], vgpr_a[9], sgpr_r[4]     A16=1
        image_bvh_dual_intersect_ray     vgpr_d[10], vgpr_a[12], sgpr_r[4]
        image_bvh8_intersect_ray         vgpr_d[10], vgpr_a[11], sgpr_r[4]

When issued these instructions perform the following operations for every active lane in the wave:
 1. Receive data that describes the ray to be tested from the SP (stored in vgpr_a)
 2. Receives the BVH node pointers that should be fetched and tested from the SP (stored in vgpr_a)
 3. Receives a BVH resource descriptor from the Shader (stored in sgpr_r)
 4. Calculates the BVH node type, data size and address for the BVH node that is being tested
 5. Fetches the BVH node from memory
 6. Performs intersection testing based on the BVH node type
 7. Returns intersection results to the shader (where they are stored in vgpr_d)
 8. Updates the ray origin and direction if an instance node is tested (updating values in vgpr_a)

Every lane in the wave can test a different ray against a different BVH node, even mixing multiple types of BVH
node tests within a single wave in one instruction issue. Additionally, image_bvh instructions can be pipelined
with other image, buffer and flat memory instructions using the typical s_waitcnt based synchronization.

However, hardware does not do any recursion or looping internally before returning control to the shader. It
only tests the BVH nodes against each ray and returns - the shader must implement the traversal loop required
to implement a full BVH traversal.

           Table 63. Ray Tracing VGPR Contents for image_bvh_intersect_ray and image_bvh64_intersect_ray
VGPR_ BVH A16=0                   BVH A16=1                  BVH64 A16=0                  BVH64 A16=1
A
0         node_pointer (u32)      node_pointer (u32)         node_pointer [31:0] (u32)    node_pointer [31:0] (u32)
1         ray_extent (f32)        ray_extent (f32)           node_pointer [63:32] (u32)   node_pointer [63:32] (u32)
2         ray_origin.x (f32)      ray_origin.x (f32)         ray_extent (f32)             ray_extent (f32)

VGPR_ BVH A16=0                 BVH A16=1                       BVH64 A16=0                    BVH64 A16=1
A
3        ray_origin.y (f32)     ray_origin.y (f32)              ray_origin.x (f32)             ray_origin.x (f32)
4        ray_origin.z (f32)     ray_origin.z (f32)              ray_origin.y (f32)             ray_origin.y (f32)
5        ray_dir.x (f32)        [15:0] = ray_dir.x (f16)        ray_origin.z (f32)             ray_origin.z (f32)
                                [31:16] = ray_dir.y (f16)
6        ray_dir.y (f32)        [15:0] = ray_dir.z (f16)        ray_dir.x (f32)                [15:0] = ray_dir.x (f16)
                                [31:16] = ray_inv_dir.x(f16)                                   [31:16] = ray_dir.y (f16)
7        ray_dir.z (f32)        [15:0] = ray_inv_dir.y (f16)    ray_dir.y (f32)                [15:0] = ray_dir.z (f16)
                                [31:16] = ray_inv_dir.z (f16)                                  [31:16] = ray_inv_dir.x(f16)
8        ray_inv_dir.x (f32)    unused                          ray_dir.z (f32)                [15:0] = ray_inv_dir.y (f16)
                                                                                               [31:16] = ray_inv_dir.z (f16)
9        ray_inv_dir.y (f32)    unused                          ray_inv_dir.x (f32)            unused
10       ray_inv_dir.z (f32)    unused                          ray_inv_dir.y (f32)            unused
11       unused                 unused                          ray_inv_dir.z (f32)            unused

Vgpr_d[4] are the destination VGPRs of the results of intersection testing. The values returned here are
different depending on the type of BVH node that was fetched. For box nodes the results contain the 4 pointers
of the children boxes in intersection time sorted order. For triangle BVH nodes the results contain the
intersection time and triangle ID of the triangle tested.

Sgpr_r[4] is the texture descriptor for the operation. The instruction is encoded with use_128bit_resource=1.

IMAGE_BVH_DUAL_INTERSECT_RAY:

       Table 64. Ray Tracing VGPR Contents for image_bvh_dual_intersect_ray and image_bvh8_intersect_ray
VGPR Group VGPR in Group       Contents
0             0                bvh_base[31:0] (first part of uint64, input, bits[2:0] must be 0)
0             1                bvh_base[63:32] (second part of uint64,input)
1             0                ray_extent (float32,input)
1             1                instance_mask (8bit uint,input)
2             0                ray_origin.x (float32, input and output parameter)
2             1                ray_origin.y (float32, input and output parameter)
2             2                ray_origin.z (float32, input and output parameter)
3             0                ray_dir.x (float32, input and output parameter)
3             1                ray_dir.y (float32, input and output parameter)
3             2                ray_dir.z (float32, input and output parameter)
4             0                image_bvh_dual_intersect_ray: node_pointer0 (offset of first node to test, uint32,input)
                               image_bvh8_intersect_ray: node_pointer (offset of BVH8 node to test, uint32)
4             1                image_bvh_dual_intersect_ray: node_pointer1 (offset of second node to test,
                               uint32,input)
                               image_bvh8_intersect_ray: unused

vgpr_d[10] are the destination VGPRs of the results of intersection testing. The first 4 values correspond to the
results from the first node, while the second 4 values correspond to the results of the second node. If both
nodes are box nodes and wide sorting is enabled, the 8 values are intermixed between both nodes tested (as
dictated by the 8 wide sort). The values returned here are different depending on the type of BVH node that
was fetched.

For box nodes the results contain the 4 pointers of the children boxes in intersection time sorted order. For
triangle BVH nodes the results contain the intersection time and triangle ID or barycentrics of the triangle
