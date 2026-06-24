# Chapter 10. Vector Memory Image Instructions

> RDNA3.5 ISA — pages 105–106

Chapter 10. Vector Memory Image Instructions
Vector Memory (VMEM) Image operations transfer data between the VGPRs and memory through the texture
cache. Image operations support access to image objects such as texture maps and typed surfaces. Sample
operations read multiple elements from a surface and combine them to produce a single result per lane.

Image objects are accessed using from one to four dimensional addresses; they are composed of homogeneous
samples, each sample containing one to four elements. These image objects are read from, or written to, using
IMAGE_* or SAMPLE_* instructions, all of which use the MIMG instruction format. IMAGE_LOAD instructions
load an element from the image buffer directly into VGPRS, and SAMPLE instructions use sampler constants
(S#) and apply filtering to the data after it is read. IMAGE_ATOMIC instructions combine data from VGPRs with
data already in memory, and optionally return the value that was in memory before the operation.

VMEM image operations use an image resource constant (T#) that is a 128-bit or 256-bit value in SGPRs. This
constant is sent to the texture cache when the instruction is executed. This constant defines the address, data
format, and characteristics of the surface in memory. Some image instructions also use a sampler constant that
is a 128-bit constant in SGPRs. Typically, these constants are fetched from memory using scalar memory loads
prior to executing VM instructions, but these constants can also be generated within the shader.

Texture fetch instructions have a data mask (DMASK) field. DMASK specifies how many data components it
receives. If DMASK is less than the number of components in the texture, the texture unit only sends DMASK
components, starting with R, then G, B, and A. if DMASK specifies more than the texture format specifies, the
shader receives data based on T#.DST_SEL for the missing components. Image ops do not generate MemViol -
instead they apply clamp modes if the address goes out of range.

Memory operations of different types (e.g. loads, stores and samples) can complete out of order with respect to
each other.

10.1. Image Instructions
This section describes the image instruction set, and the microcode fields available to those instructions.

MIMG Instructions
IMAGE_SAMPLE                             Load and filter data from a image object
IMAGE_SAMPLE_G16                         Sample with 16-bit gradients
IMAGE_GATHER4                            Load and return samples from 4 texels for software filtering. Returns a single
                                         component, starting with the lower-left texel and in counter-clockwise order.
IMAGE_GATHER4H                           4H: fetch 1 component per texel from 4x1 texels
                                         "DMASK" selects which component to load (R,G,B,A) and must have only one bit
                                         set to 1.
IMAGE_LOAD_{-, PCK, PCK_SGN}      Load data from an image object
IMAGE_LOAD_MIP_{-, PCK, PCK_SGN } Load data from an image object from a specified mip level.
IMAGE_MSAA_LOAD                          Load up to 4 samples of 1 component from an MSAA resource with a user-
                                         specified fragment ID.
                                         Uses DMASK as component select - it behaves like gather4 ops and returns 4
                                         VGPR (2 if D16=1).
IMAGE_STORE_{-, PCK }                    Store data to an image object to a specific mipmap level
IMAGE_STORE_MIP_{-, PCK }

MIMG Instructions
IMAGE_ATOMIC_{SWAP, CMPSWAP,      Image atomic operations
ADD, SUB, SMIN, UMIN, SMAX, UMAX,
AND, OR, XOR, INC, DEC }
IMAGE_GET_RESINFO                          Return resource info into 4 VGPRs for the MIP level specified. These are 32bit
                                           integer values:
                                           VDATA3-0 = { #mipLevels, depth, height, width }
                                           For cubemaps, depth = 6 * Number_of_array_faces.
                                           (DX expects the # of cubes, but gets # of faces instead)
IMAGE_GET_LOD                              Return the calculated LOD. Treated as a Sample instruction.
                                           Returns the "raw" LOD and the "clamped" LOD into VDATA as two 32 bit floats:
                                           First VGPR = clampLOD
                                           Second VGPR = rawLOD

                                                 Table 48. Instruction Fields
Instruction Fields
Field          Size        Description
OP             8           Opcode
VADDR          8           Address of VGPR to supply first component of address.
VDATA          8           Address of VGPR to supply first component of store-data or receive first component of load-data.
SSAMP          5           SGPR to supply S# (sampler constant) in 4 consecutive SGPRs.
                           missing 2 LSB’s of SGPR-address since must be aligned to 4.
SRSRC          5           SGPR to supply T# (resource constant) in 8 consecutive SGPRs.
                           missing 2 LSB’s of SGPR-address since must be aligned to 4.
UNRM           1           Force address to be un-normalized. Must be set to 1 for Image stores & atomics.
                           0: for image ops with samplers, S,T,R from [0.0, 1.0] span the entire texture map;
                           1: for image ops with samplers, S,T,R from [0.0 to N] span the texture map, where N is width,
                           height or depth. Array/cube slice, lod, bias etc. are not affected. Image ops without sampler are
                           not affected. UINT inputs are "unnormalized".
                           This bit is logically OR’d with the S#.force_unnormalized bit.
R128           1           Texture Resource Size: 1 = 128bits, 0 = 256bits
A16            1           Address components are 16-bits (instead of the usual 32 bits).
                           When set, all address components are 16 bits (packed into 2 per DWORD), except:
                           Texel offsets (3 6bit UINT packed into 1 DWORD)
                           PCF reference (for "_C" instructions)
                           Address components are 16b uint for image ops without sampler; 16b float with sampler.
DIM            3           Surface Dimension:

                                                    0: 1D                    4: 1d array
                                                    1: 2D                    5: 2d array
                                                    2: 3D                    6: 2d msaa
                                                    3: cube                  7: 2d msaa array
