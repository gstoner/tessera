# 16.17. VIMAGE Instructions

> RDNA4 ISA — pages 650–661

16.17. VIMAGE Instructions
The bitfield map of the VIMAGE format is:

IMAGE_LOAD                                                                                                           0

Load a texel from the largest miplevel in an image surface and store the result into a vector register. Perform
the format conversion specified by the resource descriptor. No sampling is performed.

IMAGE_LOAD_MIP                                                                                                       1

Load a texel from a user-specified miplevel in an image surface and store the result into a vector register.
Perform the format conversion specified by the resource descriptor. No sampling is performed.

IMAGE_LOAD_PCK                                                                                                       2

Load a texel from the largest miplevel in an image surface and store the result into a vector register. 8- and 16-
bit components are zero-extended. The format specified in the resource descriptor is ignored. No sampling is
performed.

IMAGE_LOAD_PCK_SGN                                                                                                   3

Load a texel from the largest miplevel in an image surface and store the result into a vector register. 8- and 16-
bit components are sign-extended. The format specified in the resource descriptor is ignored. No sampling is
performed.

IMAGE_LOAD_MIP_PCK                                                                                                   4

Load a texel from a user-specified miplevel in an image surface and store the result into a vector register. 8-
and 16-bit components are zero-extended. The format specified in the resource descriptor is ignored. No
sampling is performed.

IMAGE_LOAD_MIP_PCK_SGN                                                                                               5

Load a texel from a user-specified miplevel in an image surface and store the result into a vector register. 8-
and 16-bit components are sign-extended. The format specified in the resource descriptor is ignored. No

sampling is performed.

IMAGE_STORE                                                                                                         6

Store a texel from a vector register to the largest miplevel in an image surface. The texel data is converted using
the format conversion specified by the resource descriptor prior to storage.

IMAGE_STORE_MIP                                                                                                     7

Store a texel from a vector register to a user-specified miplevel in an image surface. The texel data is converted
using the format conversion specified by the resource descriptor prior to storage.

IMAGE_STORE_PCK                                                                                                     8

Store a texel from a vector register to the largest miplevel in an image surface. The texel data is already packed
and the format specified in the resource descriptor is ignored.

IMAGE_STORE_MIP_PCK                                                                                                 9

Store a texel from a vector register to a user-specified miplevel in an image surface. The texel data is already
packed and the format specified in the resource descriptor is ignored.

IMAGE_ATOMIC_SWAP                                                                                                  10

Swap an unsigned 32-bit integer value in the data register with a location in an image surface. Store the original
value from image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_CMPSWAP                                                                                               11

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in an image
surface. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);

  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_ADD_UINT                                                                                           12

Add two unsigned 32-bit integer values stored in the data register and a location in an image surface. Store the
original value from image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_SUB_UINT                                                                                           13

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in an
image surface. Store the original value from image surface into a vector register iff the temporal hint enables
atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_MIN_INT                                                                                            14

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Update the image surface with the selected value. Store the original value from
image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

IMAGE_ATOMIC_MIN_UINT                                                                                           15

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Update the image surface with the selected value. Store the original value from
image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_MAX_INT                                                                                            16

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Update the image surface with the selected value. Store the original value from
image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

IMAGE_ATOMIC_MAX_UINT                                                                                           17

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Update the image surface with the selected value. Store the original value from
image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_AND                                                                                                18

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the temporal hint enables
atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);

  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_OR                                                                                                  19

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the temporal hint enables
atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_XOR                                                                                                 20

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the temporal hint enables
atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_INC_UINT                                                                                            21

Increment an unsigned 32-bit integer value from a location in an image surface with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from image surface into a vector register iff
the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_DEC_UINT                                                                                            22

Decrement an unsigned 32-bit integer value from a location in an image surface with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from image surface into a

vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

IMAGE_GET_RESINFO                                                                                                 23

Gather resource information for a given miplevel provided in the address register. Returns 4 integer values into
registers 3:0 as { num_mip_levels, depth, height, width }. No memory access is performed.

IMAGE_BVH_INTERSECT_RAY                                                                                           25

Test the intersection of rays with either box nodes or triangle nodes within a bounded volume hierarchy using
32 bit node pointers. Store the results of the test into a vector register. This instruction does not take a sampler
constant.

DATA:

The destination VGPRs contain the results of intersection testing. The values returned here are different
depending on the type of BVH node that was fetched.

For box nodes the results contain the 4 pointers of the children boxes in intersection time sorted order.

For triangle BVH nodes the results contain the intersection time and triangle ID of the triangle tested.

The address GPR packing varies based on addressing mode (A16).

ADDR (A16 = 0):

11 address VGPRs contain the ray data and BVH node pointer for the intersection test. The data is laid out as
follows:

  • Dword Register Value
    0 VADDR0[0] = node_pointer (uint32)
    1 VADDR1[0] = ray_extent (float32)
    2 VADDR2[0] = ray_origin.x (float32)
    3 VADDR2[1] = ray_origin.y (float32)
    4 VADDR2[2] = ray_origin.z (float32)
    5 VADDR3[0] = ray_dir.x (float32)
    6 VADDR3[1] = ray_dir.y (float32)
    7 VADDR3[2] = ray_dir.z (float32)
    8 VADDR4[0] = ray_inv_dir.x (float32)
    9 VADDR4[1] = ray_inv_dir.y (float32)
    10 VADDR4[2] = ray_inv_dir.z (float32)

ADDR (A16 = 1):

For performance and power optimization, the instruction can be encoded to use 16 bit floats for ray_dir and
ray_inv_dir by setting A16 to 1. When the instruction is encoded with 16 bit addresses only 8 address VGPRs are
used as follows:

  • Dword Register Value
    0 VADDR0[0] = node_pointer (uint32)
    1 VADDR1[0] = ray_extent (float32)
    2 VADDR2[0] = ray_origin.x (float32)
    3 VADDR2[1] = ray_origin.y (float32)
    4 VADDR2[2] = ray_origin.z (float32)
    5 VADDR3[0] = {ray_dir.y, ray_dir.x} (2x float16, Y in upper bits)
    6 VADDR3[1] = {ray_inv_dir.x, ray_inv.z} (2x float16, X in upper bits)
    7 VADDR3[2] = {ray_inv_dir.z, ray_inv_dir.y} (2x float16, Z in upper bits)

RSRC:

The resource is the texture descriptor for the operation. The instruction must be encoded with R128=1.

RESTRICTIONS:

The image_bvh_intersect_ray and image_bvh64_intersect_ray opcode do not support all of the features of a
standard image instruction. This puts some restrictions on how the instruction is encoded:

  • DMASK must be set to 0xf (instruction returns all four DWORDs)
  • D16 must be set to 0 (16 bit return data is not supported)
  • R128 must be set to 1 (256 bit T#s are not supported)
  • UNRM must be set to 1 (only unnormalized coordinates are supported)
  • DIM must be set to 0 (BVH textures are 1D)
  • LWE must be set to 0 (LOD warn is not supported)
  • TFE must be set to 0 (no support for writing out the extra DWORD for the PRT hit status)

These restrictions must be respected by the SW/compiler, and are not enforced by HW. HW is allowed to
assume that these values are encoded according to the above restrictions, and ignore improper values, or do
any other undefined behavior, if the above fields do not match their specified values for these instructions.

The HW also has some additional restrictions on the BVH instructions when they are issued:

  • The HW ignores the return order settings of the BVH ops and schedules them in the in-order read return
    queue when fetching data from the texture pipe.

Notes

This instruction optimizes ray tracing by efficiently determining which parts of a scene a ray intersects with.

IMAGE_BVH64_INTERSECT_RAY                                                                                         26

Test the intersection of rays with either box nodes or triangle nodes within a bounded volume hierarchy using
64 bit node pointers. Store the results of the test into a vector register. This instruction does not take a sampler

constant.

This instruction allows support for very large BVHs (larger than 32 GBs) that may occur in workstation
workloads. See IMAGE_BVH_INTERSECT_RAY for basic information including restrictions. Only differences
are described here.

ADDR (A16 = 0):

12 address VGPRs contain the ray data and BVH node pointer for the intersection test. The data is laid out as
follows:

  • Dword Register Value
    0 VADDR0[0] = node_pointer[31:0] (uint32)
    1 VADDR0[1] = node_pointer[63:32] (uint32)
    2 VADDR1[0] = ray_extent (float32)
    3 VADDR2[0] = ray_origin.x (float32)
    4 VADDR2[1] = ray_origin.y (float32)
    5 VADDR2[2] = ray_origin.z (float32)
    6 VADDR3[0] = ray_dir.x (float32)
    7 VADDR3[1] = ray_dir.y (float32)
    8 VADDR3[2] = ray_dir.z (float32)
    9 VADDR4[0] = ray_inv_dir.x (float32)
    10 VADDR4[1] = ray_inv_dir.y (float32)
    11 VADDR4[2] = ray_inv_dir.z (float32)

ADDR (A16 = 1):

When the instruction is encoded with 16 bit addresses only 9 address VGPRs are used as follows:

  • Dword Register Value
    0 VADDR0[0] = node_pointer[31:0] (uint32)
    1 VADDR0[1] = node_pointer[63:32] (uint32)
    2 VADDR1[0] = ray_extent (float32)
    3 VADDR2[0] = ray_origin.x (float32)
    4 VADDR2[1] = ray_origin.y (float32)
    5 VADDR2[2] = ray_origin.z (float32)
    6 VADDR3[0] = {ray_dir.y, ray_dir.x} (2x float16, Y in upper bits)
    7 VADDR3[1] = {ray_inv_dir.x, ray_inv.z} (2x float16, X in upper bits)
    8 VADDR3[2] = {ray_inv_dir.z, ray_inv_dir.y} (2x float16, Z in upper bits)

RSRC:

The resource is the texture descriptor for the operation. The instruction must be encoded with R128=1.

Notes

This instruction optimizes ray tracing by efficiently determining which parts of a scene a ray intersects with.

IMAGE_BVH_DUAL_INTERSECT_RAY                                                                                  128

This instruction supports testing two QBVH nodes against the same ray per lane using both intersection
engines. It is typically used to implement the BVH4x2 traversal algorithm.

DATA:

DATA is the first of 10 destination VGPRs of the results of intersection testing. The first 4 values correspond to
the results from the first node, while the second 4 values correspond to the results of the second node. If both
nodes are box nodes and wide sorting is enabled, the 8 values are intermixed between both nodes tested (as
dictated by the 8 wide sort). The values returned here are different depending on the type of BVH node that
was fetched.

For box nodes the results contain the 4 pointers of the children boxes in intersection time sorted order. For
triangle BVH nodes the results contain the intersection time and triangle ID or barycentrics of the triangle
tested. For Instance nodes, the data consists of the updated 64 bit BVH base, the updated node pointer offset in
the BVH, a 24 bit user data field, and 8 bit instance mask.

The last two DWORDs contain the ShapeID/GeoID of each triangle tested.

ADDR:

The first of 12 VGPRs that contain the ray data and BVH node pointer for the intersection test. The data is laid
out as follows:

  • Dword Register Value
    0 VADDR0[0] = bvh_base[31:0] (first part of uint64, input)
    1 VADDR0[1] = bvh_base[63:32] (second part of uint64, input)
    2 VADDR1[0] = ray_extent (float32, input)
    3 VADDR1[1] = instance_mask (uint8, input)
    4 VADDR2[0] = ray_origin.x (float32, input and output parameter)
    5 VADDR2[1] = ray_origin.y (float32, input and output parameter)
    6 VADDR2[2] = ray_origin.z (float32, input and output parameter)
    7 VADDR3[0] = ray_dir.x (float32, input and output parameter)
    8 VADDR3[1] = ray_dir.y (float32, input and output parameter)
    9 VADDR3[2] = ray_dir.z (float32, input and output parameter)
    10 VADDR4[0] = node_pointer0 (offset of first node to test, uint32,input)
    11 VADDR4[1] = node_pointer1 (offset of second node to test, uint32,input)

A16 is not supported.

If an instance node is tested then the ray origin and ray direction components of vgpr_a are modified.

RSRC:

Carries the texture descriptor for the operation. The instruction must be encoded with R128=1.

IMAGE_BVH8_INTERSECT_RAY                                                                                       129

This instruction supports testing one BVH8 node against one ray per lane using both intersection engines.

DATA:

DATA is the first of 10 destination VGPRs of the results of intersection testing. The values returned here are
different depending on the type of BVH node that was fetched.

For box nodes the results contain the 8 pointers of the children boxes in intersection time sorted order. When
wide sorting is disabled, the first 4 values correspond to the results from boxes 0-3, while the second 4 values
correspond to the results of boxes 4-7. If wide sorting is enabled, the 8 values are fully intermixed and contain
results for boxes 0-7 (as dictated by the 8 wide sort). For triangle BVH nodes the results contain the intersection
time and triangle ID or barycentrics of both the triangles tested. For Instance nodes, the data consists of the
updated 64 bit BVH base, the updated node pointer offset in the BVH, a 24 bit user data field, and 8 bit instance
mask.

The last two DWORDs contain the ShapeID/GeoID of each triangle tested.

ADDR:

ADDR contains the first of 11 VGPRs contain the ray data and BVH node pointer for the intersection test. The
data is laid out as follows:

  • Dword Register Value
    0 VADDR0[0] = bvh_base[31:0] (first part of uint64)
    1 VADDR0[1] = bvh_base[63:32] (second part of uint64)
    2 VADDR1[0] = ray_extent (float32)
    3 VADDR1[1] = instance_mask (uint8)
    4 VADDR2[0] = ray_origin.x (float32, input and output parameter)
    5 VADDR2[1] = ray_origin.y (float32, input and output parameter)
    6 VADDR2[2] = ray_origin.z (float32, input and output parameter)
    7 VADDR3[0] = ray_dir.x (float32, input and output parameter)
    8 VADDR3[1] = ray_dir.y (float32, input and output parameter)
    9 VADDR3[2] = ray_dir.z (float32, input and output parameter)
    10 VADDR4[0] = node_pointer (offset of BVH8 node to test, uint32)

A16 is not supported.

If an instance node is tested then the ray origin and ray direction components of vgpr_a are modified.

RSRC:

Contains the texture descriptor for the operation. The instruction must be encoded with R128=1.

IMAGE_ATOMIC_ADD_FLT                                                                                             131

Add two single-precision float values stored in the data register and a location in an image surface. Store the
original value from image surface into a vector register iff the temporal hint enables atomic return.

  addr = CalcImageAddr(v_addr.b128);
  tmp = MEM[addr].f32;
  MEM[addr].f32 += DATA.f32;
  RETURN_DATA.f32 = tmp

Notes

Floating-point addition handles NAN/INF/denorm.

IMAGE_ATOMIC_MIN_FLT                                                                                         132

Select the IEEE minimumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in an image surface. Update the image surface with the selected value. Store the original
value from image surface into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
        MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
        MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
        MEM[ADDR].f32 = src.f32
  elsif ((src.f32 < tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && sign(src.f32) &&
  !sign(tmp.f32))) then
        // NOTE: -0<+0 is TRUE in this comparison
        MEM[ADDR].f32 = src.f32
  else
        MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

IMAGE_ATOMIC_MAX_FLT                                                                                         133

Select the IEEE maximumNumber() of two single-precision float inputs, given two values stored in the data
register and a location in an image surface. Update the image surface with the selected value. Store the original
value from image surface into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].f32;
  src = DATA.f32;
  if (isNAN(64'F(src.f32)) && isNAN(64'F(tmp.f32))) then
        MEM[ADDR].f32 = 32'F(cvtToQuietNAN(64'F(src.f32)))
  elsif isNAN(64'F(src.f32)) then
        MEM[ADDR].f32 = tmp.f32
  elsif isNAN(64'F(tmp.f32)) then
        MEM[ADDR].f32 = src.f32
  elsif ((src.f32 > tmp.f32) || ((abs(src.f32) == 0.0F) && (abs(tmp.f32) == 0.0F) && !sign(src.f32) &&
  sign(tmp.f32))) then
        // NOTE: +0>-0 is TRUE in this comparison
        MEM[ADDR].f32 = src.f32
  else
        MEM[ADDR].f32 = tmp.f32
  endif;
  RETURN_DATA.f32 = tmp

IMAGE_ATOMIC_PK_ADD_F16                                                                                      134

Add a packed 2-component half-precision float value from the data register to a location in an image surface.
Store the original value from image surface into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].f16 = src[15 : 0].f16 + tmp[15 : 0].f16;
  dst[31 : 16].f16 = src[31 : 16].f16 + tmp[31 : 16].f16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32

IMAGE_ATOMIC_PK_ADD_BF16                                                                                     135

Add a packed 2-component BF16 float value from the data register to a location in an image surface. Store the
original value from image surface into a vector register iff the temporal hint enables atomic return.

  tmp = MEM[ADDR].b32;
  src = DATA.b32;
  dst[15 : 0].bf16 = src[15 : 0].bf16 + tmp[15 : 0].bf16;
  dst[31 : 16].bf16 = src[31 : 16].bf16 + tmp[31 : 16].bf16;
  MEM[ADDR].b32 = dst.b32;
  RETURN_DATA.b32 = tmp.b32
