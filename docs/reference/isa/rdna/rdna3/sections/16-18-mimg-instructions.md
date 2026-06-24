# 16.18. MIMG Instructions

> RDNA3 ISA — pages 567–580

16.18. MIMG Instructions
The bitfield map of the MIMG format is:

IMAGE_LOAD                                                                                                   0

Load element from largest miplevel in resource view, with format conversion specified in the resource
constant. No sampler.

IMAGE_LOAD_MIP                                                                                               1

Load element from user-specified miplevel in resource view, with format conversion specified in the resource
constant. No sampler.

IMAGE_LOAD_PCK                                                                                               2

Load element from largest miplevel in resource view, without format conversion. 8- and 16-bit elements are
not sign-extended. No sampler.

IMAGE_LOAD_PCK_SGN                                                                                           3

Load element from largest miplevel in resource view, without format conversion. 8- and 16-bit elements are
sign-extended. No sampler.

IMAGE_LOAD_MIP_PCK                                                                                           4

Load element from user-supplied miplevel in resource view, without format conversion. 8- and 16-bit elements
are not sign-extended. No sampler.

IMAGE_LOAD_MIP_PCK_SGN                                                                                       5

Load element from user-supplied miplevel in resource view, without format conversion. 8- and 16-bit elements
are sign-extended. No sampler.

IMAGE_STORE                                                                                                     6

Store element to largest miplevel in resource view, with format conversion specified in resource constant. No
sampler.

IMAGE_STORE_MIP                                                                                                 7

Store element to user-specified miplevel in resource view, with format conversion specified in resource
constant. No sampler.

IMAGE_STORE_PCK                                                                                                 8

Store element to largest miplevel in resource view, without format conversion. No sampler.

IMAGE_STORE_MIP_PCK                                                                                             9

Store element to user-specified miplevel in resource view, without format conversion. No sampler.

IMAGE_ATOMIC_SWAP                                                                                            10

Swap values in data register and memory.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = DATA.b;
  RETURN_DATA.b = tmp

IMAGE_ATOMIC_CMPSWAP                                                                                         11

Compare and swap with memory value.

  tmp = MEM[ADDR].b;
  src = DATA[31 : 0].b;
  cmp = DATA[63 : 32].b;
  MEM[ADDR].b = tmp == cmp ? src : tmp;
  RETURN_DATA.b = tmp

IMAGE_ATOMIC_ADD                                                                                             12

Add data register to memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u += DATA.u;
  RETURN_DATA.u = tmp

IMAGE_ATOMIC_SUB                                   13

Subtract data register from memory value.

  tmp = MEM[ADDR].u;
  MEM[ADDR].u -= DATA.u;
  RETURN_DATA.u = tmp

IMAGE_ATOMIC_SMIN                                  14

Minimum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src < tmp ? src : tmp;
  RETURN_DATA.i = tmp

IMAGE_ATOMIC_UMIN                                  15

Minimum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src < tmp ? src : tmp;
  RETURN_DATA.u = tmp

IMAGE_ATOMIC_SMAX                                  16

Maximum of two signed integer values.

  tmp = MEM[ADDR].i;
  src = DATA.i;
  MEM[ADDR].i = src > tmp ? src : tmp;

  RETURN_DATA.i = tmp

IMAGE_ATOMIC_UMAX                                        17

Maximum of two unsigned integer values.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = src > tmp ? src : tmp;
  RETURN_DATA.u = tmp

IMAGE_ATOMIC_AND                                         18

Bitwise AND of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp & DATA.b);
  RETURN_DATA.b = tmp

IMAGE_ATOMIC_OR                                          19

Bitwise OR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp | DATA.b);
  RETURN_DATA.b = tmp

IMAGE_ATOMIC_XOR                                         20

Bitwise XOR of register value and memory value.

  tmp = MEM[ADDR].b;
  MEM[ADDR].b = (tmp ^ DATA.b);
  RETURN_DATA.b = tmp

IMAGE_ATOMIC_INC                                         21

Increment memory value with wraparound to zero when incremented to register value.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u = tmp

IMAGE_ATOMIC_DEC                                                                                                  22

Decrement memory value with wraparound to register value when decremented below zero.

  tmp = MEM[ADDR].u;
  src = DATA.u;
  MEM[ADDR].u = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u = tmp

IMAGE_GET_RESINFO                                                                                                 23

Return resource info for a given mip level specified in the address vgpr. No sampler. Returns 4 integer values
into VGPRs 3-0: {num_mip_levels, depth, height, width}.

IMAGE_MSAA_LOAD                                                                                                   24

Load up to 4 samples of 1 component from an MSAA resource with a user-specified fragment ID. No sampler.

IMAGE_BVH_INTERSECT_RAY                                                                                           25

Intersection test on bound volume hierarchy nodes for ray tracing acceleration. 32-bit node pointer. No
sampler.

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

IMAGE_BVH64_INTERSECT_RAY                                                                                       26

Intersection test on bound volume hierarchy nodes for ray tracing acceleration. 64-bit node pointer. No
sampler.

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

IMAGE_SAMPLE                                        27

Sample texture map.

IMAGE_SAMPLE_D                                      28

Sample texture map, with user derivatives.

IMAGE_SAMPLE_L                                      29

Sample texture map, with user LOD.

IMAGE_SAMPLE_B                                      30

Sample texture map, with lod bias.

IMAGE_SAMPLE_LZ                                     31

Sample texture map, from level 0.

IMAGE_SAMPLE_C                                      32

Sample texture map, with PCF.

IMAGE_SAMPLE_C_D                                    33

SAMPLE_C, with user derivatives.

IMAGE_SAMPLE_C_L                                    34

SAMPLE_C, with user LOD.

IMAGE_SAMPLE_C_B                                    35

SAMPLE_C, with lod bias.

IMAGE_SAMPLE_C_LZ                               36

SAMPLE_C, from level 0.

IMAGE_SAMPLE_O                                  37

Sample texture map, with user offsets.

IMAGE_SAMPLE_D_O                                38

SAMPLE_O, with user derivatives.

IMAGE_SAMPLE_L_O                                39

SAMPLE_O, with user LOD.

IMAGE_SAMPLE_B_O                                40

SAMPLE_O, with lod bias.

IMAGE_SAMPLE_LZ_O                               41

SAMPLE_O, from level 0.

IMAGE_SAMPLE_C_O                                42

SAMPLE_C with user specified offsets.

IMAGE_SAMPLE_C_D_O                              43

SAMPLE_C_O, with user derivatives.

IMAGE_SAMPLE_C_L_O                              44

SAMPLE_C_O, with user LOD.

IMAGE_SAMPLE_C_B_O                                                      45

SAMPLE_C_O, with lod bias.

IMAGE_SAMPLE_C_LZ_O                                                     46

SAMPLE_C_O, from level 0.

IMAGE_GATHER4                                                           47

Gather 4 single component elements (2x2).

IMAGE_GATHER4_L                                                         48

Gather 4 single component elements (2x2) with user LOD.

IMAGE_GATHER4_B                                                         49

Gather 4 single component elements (2x2) with user bias.

IMAGE_GATHER4_LZ                                                        50

Gather 4 single component elements (2x2) at level 0.

IMAGE_GATHER4_C                                                         51

Gather 4 single component elements (2x2) with PCF.

IMAGE_GATHER4_C_LZ                                                      52

Gather 4 single component elements (2x2) at level 0, with PCF.

IMAGE_GATHER4_O                                                         53

GATHER4, with user offsets.

IMAGE_GATHER4_LZ_O                                                        54

GATHER4_LZ, with user offsets.

IMAGE_GATHER4_C_LZ_O                                                      55

GATHER4_C_LZ, with user offsets.

IMAGE_GET_LOD                                                             56

Return calculated LOD as two 32-bit floating point values.

  VDATA[0] = clampedLOD;
  VDATA[1] = rawLOD.

IMAGE_SAMPLE_D_G16                                                        57

SAMPLE_D with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_C_D_G16                                                      58

SAMPLE_C_D with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_D_O_G16                                                      59

SAMPLE_D_O with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_C_D_O_G16                                                    60

SAMPLE_C_D_O with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_CL                                                           64

Sample texture map, with LOD clamp specified in shader.

IMAGE_SAMPLE_D_CL                                                                       65

Sample texture map, with LOD clamp specified in shader, with user derivatives.

IMAGE_SAMPLE_B_CL                                                                       66

Sample texture map, with LOD clamp specified in shader, with lod bias.

IMAGE_SAMPLE_C_CL                                                                       67

SAMPLE_C, with LOD clamp specified in shader.

IMAGE_SAMPLE_C_D_CL                                                                     68

SAMPLE_C, with LOD clamp specified in shader, with user derivatives.

IMAGE_SAMPLE_C_B_CL                                                                     69

SAMPLE_C, with LOD clamp specified in shader, with lod bias.

IMAGE_SAMPLE_CL_O                                                                       70

SAMPLE_O with LOD clamp specified in shader.

IMAGE_SAMPLE_D_CL_O                                                                     71

SAMPLE_O, with LOD clamp specified in shader, with user derivatives.

IMAGE_SAMPLE_B_CL_O                                                                     72

SAMPLE_O, with LOD clamp specified in shader, with lod bias.

IMAGE_SAMPLE_C_CL_O                                                                     73

SAMPLE_C_O, with LOD clamp specified in shader.

IMAGE_SAMPLE_C_D_CL_O                                                           74

SAMPLE_C_O, with LOD clamp specified in shader, with user derivatives.

IMAGE_SAMPLE_C_B_CL_O                                                           75

SAMPLE_C_O, with LOD clamp specified in shader, with lod bias.

IMAGE_SAMPLE_C_D_CL_G16                                                         84

SAMPLE_C_D_CL with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_D_CL_O_G16                                                         85

SAMPLE_D_CL_O with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_C_D_CL_O_G16                                                       86

SAMPLE_C_D_CL_O with 16-bit floating point derivatives (gradients).

IMAGE_SAMPLE_D_CL_G16                                                           95

SAMPLE_D_CL with 16-bit floating point derivatives (gradients).

IMAGE_GATHER4_CL                                                                96

Gather 4 single component elements (2x2) with user LOD clamp.

IMAGE_GATHER4_B_CL                                                              97

Gather 4 single component elements (2x2) with user bias and clamp.

IMAGE_GATHER4_C_CL                                                              98

Gather 4 single component elements (2x2) with user LOD clamp and PCF.

IMAGE_GATHER4_C_L                                                                                        99

Gather 4 single component elements (2x2) with user LOD and PCF.

IMAGE_GATHER4_C_B                                                                                       100

Gather 4 single component elements (2x2) with user bias and PCF.

IMAGE_GATHER4_C_B_CL                                                                                    101

Gather 4 single component elements (2x2) with user bias, clamp and PCF.

IMAGE_GATHER4H                                                                                          144

Fetch 1 component per texel from 4x1 texels. DMASK selects which component to read (R,G,B,A) and must
have only one bit set to 1.
