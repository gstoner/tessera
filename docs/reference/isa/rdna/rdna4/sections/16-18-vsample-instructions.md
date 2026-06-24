# 16.18. VSAMPLE Instructions

> RDNA4 ISA — pages 662–669

16.18. VSAMPLE Instructions
The bitfield map of the VSAMPLE format is:

IMAGE_MSAA_LOAD                                                                                                  24

Load up to 4 samples of 1 component from an MSAA resource with a user-specified fragment ID. No sampling
is performed.

IMAGE_SAMPLE                                                                                                     27

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers.

IMAGE_SAMPLE_D                                                                                                   28

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

IMAGE_SAMPLE_C_L_O                                                                                               44

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD, user offsets are provided by the address registers.

IMAGE_SAMPLE_C_B_O                                                                                               45

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, LOD bias, user offsets are provided by the address
registers.

IMAGE_SAMPLE_C_LZ_O                                                                                              46

Sample texels from an image surface using texel coordinates provided by the address input registers and store

the result into vector registers. Additional data for PCF, user offsets are provided by the address registers.
Mipmap level is set to zero.

IMAGE_GATHER4                                                                                                    47

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1.

IMAGE_GATHER4_L                                                                                                  48

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD are provided by the address registers.

IMAGE_GATHER4_B                                                                                                  49

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD bias are provided by the address registers.

IMAGE_GATHER4_LZ                                                                                                 50

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_C                                                                                                  51

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF are provided by the address registers.

IMAGE_GATHER4_C_LZ                                                                                               52

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_O                                                                                                  53

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for user offsets are provided by the address registers.

IMAGE_GATHER4_LZ_O                                                                                               54

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for user offsets are provided by the address registers. Mipmap level is set to zero.

IMAGE_GATHER4_C_LZ_O                                                                                             55

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, user offsets are provided by the address registers. Mipmap level is set to zero.

IMAGE_GET_LOD                                                                                                    56

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

IMAGE_SAMPLE_C_D_O_G16                                                                                         60

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for PCF, user offsets, 16-bit derivatives are provided by the
address registers.

IMAGE_SAMPLE_CL                                                                                                64

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp are provided by the address registers.

IMAGE_SAMPLE_D_CL                                                                                              65

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives, LOD clamp are provided by the address
registers.

IMAGE_SAMPLE_B_CL                                                                                              66

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

IMAGE_SAMPLE_CL_O                                                                                               70

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD clamp, user offsets are provided by the address
registers.

IMAGE_SAMPLE_D_CL_O                                                                                             71

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for user derivatives, LOD clamp, user offsets are provided by
the address registers.

IMAGE_SAMPLE_B_CL_O                                                                                             72

Sample texels from an image surface using texel coordinates provided by the address input registers and store
the result into vector registers. Additional data for LOD bias, LOD clamp, user offsets are provided by the
address registers.

IMAGE_SAMPLE_C_CL_O                                                                                             73

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

IMAGE_GATHER4_CL                                                                                                96

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD clamp are provided by the address registers.

IMAGE_GATHER4_B_CL                                                                                              97

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for LOD bias, LOD clamp are provided by the address registers.

IMAGE_GATHER4_C_CL                                                                                              98

Gather 4 single-component texels from a 2x2 matrix on an image surface. Store the result into vector registers.
The DMASK selects which channel to read from (R, G, B, A) and must only have one bit set to 1. Additional data
for PCF, LOD clamp are provided by the address registers.

IMAGE_GATHER4_C_L                                                                                               99
