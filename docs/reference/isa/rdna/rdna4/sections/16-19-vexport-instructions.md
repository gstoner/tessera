# 16.19. VEXPORT Instructions

> RDNA4 ISA — pages 670–670

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

16.19. VEXPORT Instructions
Transfer vertex position, vertex parameter, pixel color, or pixel depth information to the output buffer. Every
pixel shader must do at least one export to a color, depth or NULL target with the VM bit set to 1. This
communicates the pixel-valid mask to the color and depth buffers. Every pixel does only one of the above
export types with the DONE bit set to 1. Vertex shaders must do one or more position exports, and at least one
parameter export. The final position export must have the DONE bit set to 1.
