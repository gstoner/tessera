# 16.19. EXPORT Instructions

> RDNA3.5 ISA — pages 622–622

16.19. EXPORT Instructions
Transfer vertex position, vertex parameter, pixel color, or pixel depth information to the output buffer. Every
pixel shader must do at least one export to a color, depth or NULL target with the VM bit set to 1. This
communicates the pixel-valid mask to the color and depth buffers. Every pixel does only one of the above
export types with the DONE bit set to 1. Vertex shaders must do one or more position exports, and at least one
parameter export. The final position export must have the DONE bit set to 1.
