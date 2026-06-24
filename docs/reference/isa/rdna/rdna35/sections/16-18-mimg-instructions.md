# 16.18. MIMG Instructions

> RDNA3.5 ISA — pages 605–621

16.18. MIMG Instructions
The bitfield map of the MIMG format is:

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
value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_CMPSWAP                                                                                               11

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in an image
surface. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA[31 : 0].u32;

  cmp = DATA[63 : 32].u32;
  MEM[ADDR].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_ADD                                                                                                 12

Add two unsigned 32-bit integer values stored in the data register and a location in an image surface. Store the
original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  MEM[ADDR].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_SUB                                                                                                 13

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in an
image surface. Store the original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  MEM[ADDR].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_SMIN                                                                                                14

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Store the original value from image surface into a vector register iff the GLC bit is
set.

  tmp = MEM[ADDR].i32;
  src = DATA.i32;
  MEM[ADDR].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

IMAGE_ATOMIC_UMIN                                                                                                15

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Store the original value from image surface into a vector register iff the GLC bit is
set.

  tmp = MEM[ADDR].u32;

  src = DATA.u32;
  MEM[ADDR].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_SMAX                                                                                                16

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Store the original value from image surface into a vector register iff the GLC bit is
set.

  tmp = MEM[ADDR].i32;
  src = DATA.i32;
  MEM[ADDR].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp

IMAGE_ATOMIC_UMAX                                                                                                17

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in an image surface. Store the original value from image surface into a vector register iff the GLC bit is
set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_AND                                                                                                 18

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_OR                                                                                                  19

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_XOR                                                                                                 20

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in an
image surface. Store the original value from image surface into a vector register iff the GLC bit is set.

  tmp = MEM[ADDR].b32;
  MEM[ADDR].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp

IMAGE_ATOMIC_INC                                                                                                 21

Increment an unsigned 32-bit integer value from a location in an image surface with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from image surface into a vector register iff
the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp

IMAGE_ATOMIC_DEC                                                                                                 22

Decrement an unsigned 32-bit integer value from a location in an image surface with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from image surface into a
vector register iff the GLC bit is set.

  tmp = MEM[ADDR].u32;
  src = DATA.u32;
  MEM[ADDR].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp

IMAGE_GET_RESINFO                                                                                                23

Gather resource information for a given miplevel provided in the address register. Returns 4 integer values into
registers 3:0 as { num_mip_levels, depth, height, width }. No memory access is performed.

IMAGE_MSAA_LOAD                                                                                                   24

Load up to 4 samples of 1 component from an MSAA resource with a user-specified fragment ID. No sampling
is performed.

IMAGE_BVH_INTERSECT_RAY                                                                                           25

Test the intersection of rays with either box nodes or triangle nodes within a bounded volume hierarchy using
32 bit node pointers. Store the results of the test into a vector register. This instruction does not take a sampler
constant.

DATA:

The destination VGPRs contain the results of intersection testing. The values returned here are different
depending on the type of BVH node that was fetched.

For box nodes the results contain the 4 pointers of the children boxes in intersection time sorted order.

For triangle BVH nodes the results contain the intersection time and triangle ID of the triangle tested.

The address GPR packing varies based on addressing mode (A16) and NSA mode.

ADDR (A16 = 0):

11 address VGPRs contain the ray data and BVH node pointer for the intersection test. The data is laid out as
follows (dependent on NSA mode):

  • NSA=0 NSA=1 Value
    VADDR[0] VADDR[0] = node_pointer (uint32)
    VADDR[1] VADDRA[0] = ray_extent (float32)
    VADDR[2] VADDRB[0] = ray_origin.x (float32)
    VADDR[3] VADDRB[1] = ray_origin.y (float32)
    VADDR[4] VADDRB[2] = ray_origin.z (float32)
    VADDR[5] VADDRC[0] = ray_dir.x (float32)
    VADDR[6] VADDRC[1] = ray_dir.y (float32)
    VADDR[7] VADDRC[2] = ray_dir.z (float32)
    VADDR[8] VADDRD[0] = ray_inv_dir.x (float32)
    VADDR[9] VADDRD[1] = ray_inv_dir.y (float32)
    VADDR[10] VADDRD[2] = ray_inv_dir.z (float32)

ADDR (A16 = 1):

For performance and power optimization, the instruction can be encoded to use 16 bit floats for ray_dir and
ray_inv_dir by setting A16 to 1. When the instruction is encoded with 16 bit addresses only 8 address VGPRs are
used as follows (dependent on NSA mode):

  • NSA=0 NSA=1 Value
    VADDR[0] VADDR[0] = node_pointer (uint32)
    VADDR[1] VADDRA[0] = ray_extent (float32)

    VADDR[2] VADDRB[0] = ray_origin.x (float32)
    VADDR[3] VADDRB[1] = ray_origin.y (float32)
    VADDR[4] VADDRB[2] = ray_origin.z (float32)
    VADDR[5] VADDRC[0] = {ray_inv_dir.x, ray_dir.x} (2x float16)
    VADDR[6] VADDRC[1] = {ray_inv_dir.y, ray_dir.y} (2x float16)
    VADDR[7] VADDRC[2] = {ray_inv_dir.z, ray_dir.z} (2x float16)

RSRC:

The resource is the texture descriptor for the operation. The instruction must be encoded with r128=1.

RESTRICTIONS:

The image_bvh_intersect_ray and image_bvh64_intersect_ray opcode do not support all of the features of a
standard MIMG instruction. This puts some restrictions on how the instruction is encoded:

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

  • The HW ignores the return order settings of the BVH ops and schedules them in the in order read return
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

follows (dependent on NSA mode):

  • NSA=0 NSA=1 Value
    VADDR[0] VADDR[0] = node_pointer[31:0] (uint32)
    VADDR[1] VADDR[1] = node_pointer[63:32] (uint32)
    VADDR[2] VADDRA[0] = ray_extent (float32)
    VADDR[3] VADDRB[0] = ray_origin.x (float32)
    VADDR[4] VADDRB[1] = ray_origin.y (float32)
    VADDR[5] VADDRB[2] = ray_origin.z (float32)
    VADDR[6] VADDRC[0] = ray_dir.x (float32)
    VADDR[7] VADDRC[1] = ray_dir.y (float32)
    VADDR[8] VADDRC[2] = ray_dir.z (float32)
    VADDR[9] VADDRD[0] = ray_inv_dir.x (float32)
    VADDR[10] VADDRD[1] = ray_inv_dir.y (float32)
    VADDR[11] VADDRD[2] = ray_inv_dir.z (float32)

ADDR (A16 = 1):

When the instruction is encoded with 16 bit addresses only 9 address VGPRs are used as follows (dependent on
NSA mode):

  • NSA=0 NSA=1 Value
    VADDR[0] VADDR[0] = node_pointer[31:0] (uint32)
    VADDR[1] VADDR[1] = node_pointer[63:32] (uint32)
    VADDR[2] VADDRA[0] = ray_extent (float32)
    VADDR[3] VADDRB[0] = ray_origin.x (float32)
    VADDR[4] VADDRB[1] = ray_origin.y (float32)
    VADDR[5] VADDRB[2] = ray_origin.z (float32)
    VADDR[6] VADDRC[0] = {ray_inv_dir.x, ray_dir.x} (2x float16)
    VADDR[7] VADDRC[1] = {ray_inv_dir.y, ray_dir.y} (2x float16)
    VADDR[8] VADDRC[2] = {ray_inv_dir.z, ray_dir.z} (2x float16)

Notes

This instruction optimizes ray tracing by efficiently determining which parts of a scene a ray intersects with.

IMAGE_SAMPLE                                                                                                      27

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers.

IMAGE_SAMPLE_D                                                                                                    28

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives are provided by the address registers.

IMAGE_SAMPLE_L                                                                                                   29

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD are provided by the address registers.

IMAGE_SAMPLE_B                                                                                                   30

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD bias are provided by the address registers.

IMAGE_SAMPLE_LZ                                                                                                  31

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for are provided by the address registers. Mipmap level is set to
zero.

IMAGE_SAMPLE_C                                                                                                   32

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF are provided by the address registers.

IMAGE_SAMPLE_C_D                                                                                                 33

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user derivatives are provided by the address registers.

IMAGE_SAMPLE_C_L                                                                                                 34

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD are provided by the address registers.

IMAGE_SAMPLE_C_B                                                                                                 35

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD bias are provided by the address registers.

IMAGE_SAMPLE_C_LZ                                                                                                36

Sample texels from an image surface using texel coordinates provided by the address input registers and store

the result into vector registers. Additional data for PCF are provided by the address registers. Mipmap level is
set to zero.

IMAGE_SAMPLE_O                                                                                                   37

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user offsets are provided by the address registers.

IMAGE_SAMPLE_D_O                                                                                                 38

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives, user offsets are provided by the address
registers.

IMAGE_SAMPLE_L_O                                                                                                 39

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD, user offsets are provided by the address registers.

IMAGE_SAMPLE_B_O                                                                                                 40

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD bias, user offsets are provided by the address registers.

IMAGE_SAMPLE_LZ_O                                                                                                41

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user offsets are provided by the address registers. Mipmap
level is set to zero.

IMAGE_SAMPLE_C_O                                                                                                 42

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user offsets are provided by the address registers.

IMAGE_SAMPLE_C_D_O                                                                                               43

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user derivatives, user offsets are provided by the

address registers.

IMAGE_SAMPLE_C_L_O                                                                                              44

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD, user offsets are provided by the address registers.

IMAGE_SAMPLE_C_B_O                                                                                              45

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD bias, user offsets are provided by the address
registers.

IMAGE_SAMPLE_C_LZ_O                                                                                             46

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user offsets are provided by the address registers.
Mipmap level is set to zero.

IMAGE_GATHER4                                                                                                   47

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1.

IMAGE_GATHER4_L                                                                                                 48

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD are provided by the address registers.

IMAGE_GATHER4_B                                                                                                 49

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD bias are provided by the address registers.

IMAGE_GATHER4_LZ                                                                                                50

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.

The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_C                                                                                                 51

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF are provided by the address registers.

IMAGE_GATHER4_C_LZ                                                                                              52

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_O                                                                                                 53

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for user offsets are provided by the address registers.

IMAGE_GATHER4_LZ_O                                                                                              54

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for user offsets are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_C_LZ_O                                                                                            55

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, user offsets are provided by the address registers. Mipmap level is set to zero.

IMAGE_GET_LOD                                                                                                   56

Return the calculated level of detail (LOD) for the provided input as two single-precision float values. No
memory access is performed.

  VDATA[0] = clampedLOD;

  VDATA[1] = rawLOD.

IMAGE_SAMPLE_D_G16                                                                                               57

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for 16-bit derivatives are provided by the address registers.

IMAGE_SAMPLE_C_D_G16                                                                                             58

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, 16-bit derivatives are provided by the address registers.

IMAGE_SAMPLE_D_O_G16                                                                                             59

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user offsets, 16-bit derivatives are provided by the address
registers.

IMAGE_SAMPLE_C_D_O_G16                                                                                           60

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user offsets, 16-bit derivatives are provided by the
address registers.

IMAGE_SAMPLE_CL                                                                                                  64

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp are provided by the address registers.

IMAGE_SAMPLE_D_CL                                                                                                65

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives, LOD clamp are provided by the address
registers.

IMAGE_SAMPLE_B_CL                                                                                                66

Sample texels from an image surface using texel coordinates provided by the address input registers and store

the result into vector registers. Additional data for LOD bias, LOD clamp are provided by the address registers.

IMAGE_SAMPLE_C_CL                                                                                              67

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD clamp are provided by the address registers.

IMAGE_SAMPLE_C_D_CL                                                                                            68

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user derivatives, LOD clamp are provided by the
address registers.

IMAGE_SAMPLE_C_B_CL                                                                                            69

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD bias, LOD clamp are provided by the address
registers.

IMAGE_SAMPLE_CL_O                                                                                              70

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp, user offsets are provided by the address
registers.

IMAGE_SAMPLE_D_CL_O                                                                                            71

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives, LOD clamp, user offsets are provided by
the address registers.

IMAGE_SAMPLE_B_CL_O                                                                                            72

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD bias, LOD clamp, user offsets are provided by the
address registers.

IMAGE_SAMPLE_C_CL_O                                                                                            73

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD clamp, user offsets are provided by the address
registers.

IMAGE_SAMPLE_C_D_CL_O                                                                                           74

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user derivatives, LOD clamp, user offsets are provided
by the address registers.

IMAGE_SAMPLE_C_B_CL_O                                                                                           75

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD bias, LOD clamp, user offsets are provided by the
address registers.

IMAGE_SAMPLE_C_D_CL_G16                                                                                         84

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD clamp, 16-bit derivatives are provided by the
address registers.

IMAGE_SAMPLE_D_CL_O_G16                                                                                         85

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp, user offsets, 16-bit derivatives are provided by
the address registers.

IMAGE_SAMPLE_C_D_CL_O_G16                                                                                       86

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD clamp, user offsets, 16-bit derivatives are provided
by the address registers.

IMAGE_SAMPLE_D_CL_G16                                                                                           95

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp, 16-bit derivatives are provided by the address
registers.

IMAGE_GATHER4_CL                                                                                             96

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD clamp are provided by the address registers.

IMAGE_GATHER4_B_CL                                                                                           97

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD bias, LOD clamp are provided by the address registers.

IMAGE_GATHER4_C_CL                                                                                           98

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, LOD clamp are provided by the address registers.

IMAGE_GATHER4_C_L                                                                                            99

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, LOD are provided by the address registers.

IMAGE_GATHER4_C_B                                                                                           100

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, LOD bias are provided by the address registers.

IMAGE_GATHER4_C_B_CL                                                                                        101

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, LOD bias, LOD clamp are provided by the address registers.

IMAGE_GATHER4H                                                                                              144

Gather 4 single-component texels from a 4x1 row vector on an image surface. Store the result into vector
registers. The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1.
