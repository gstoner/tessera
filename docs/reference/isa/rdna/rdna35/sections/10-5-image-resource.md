# 10.5. Image Resource

> RDNA3.5 ISA — pages 114–114

Atomics
     Image atomic operations are supported only on 32- and 64-bit-per-pixel surfaces. The surface data format is
     specified in the resource constant. Atomic operations treat the element as a single component of 32- or 64-
     bits. For atomic operations, DMASK is set to the number of VGPRs (DWORDs) to send to the texture unit.
     DMASK legal values for atomic image operations: All other values of DMASK are illegal.

        • 0x1 = 32bit atomics except cmpswap
        • 0x3 = 32bit atomic cmpswap
        • 0x3 = 64bit atomics except cmpswap
        • 0xf = 64bit atomic cmpswap
        • Atomics with Return: Data is read out of the VGPR(s), starting at VDATA, to supply to the atomic
          operation. If the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at
          VDATA.

The DMASK must be compatible with the resource’s data format.

Denormals in Floats
     Sample ops flush denormals, and loads do not modify denormals.

10.4.1. Data format in VGPRs
Data in VGPRs sent to texture (stores) or returned from texture (loads) is in one of a few standard formats, and
the texture unit converts to/from the memory format.

        FORMAT                         VGPR data format                             If D16==1
        SINT                           signed 32-bit integer                        16 bit signed int
        UINT                           unsigned 32-bit integer                      16 bit unsigned int
        others                         32-bit float                                 16 bit float
        Atomics                        depends on opcode: uint or float             -
        ASTC data formats              32-bit float                                 -

10.5. Image Resource
The image resource (also referred to as T#) defines the location of the image buffer in memory, its dimensions,
tiling, and data format. These resources are stored in four or eight consecutive SGPRs and are read by MIMG
instructions. All undefined or reserved bit must be set to zero unless otherwise specified.

                                           Table 50. Image Resource Definition
Bits              Size      Name                  Comments
128-bit Resource: 1D-tex, 2d-tex, 2d-msaa (multi-sample anti-aliasing)
39:0              40        base address          256-byte aligned (represents bits 47:8).
47                1         Big Page              0 = No page size override, 1 = coalesce page translation requests to 64kB
                                                  granularity. Use only when entire resource uses pages 64kB or greater.
51:48             4         max mip               MSAA resources: holds Log2(number of samples); others holds:
                                                  MipLevels-1. This describes the resource, not the resource view.
59:52             8         format                Memory Data format
75:62             14        width                 width-1 of mip 0 in texels
