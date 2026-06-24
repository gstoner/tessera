# 10.1. Image Instructions

> RDNA4 ISA — pages 126–128

Chapter 10. Vector Memory Image Instructions
Vector Memory (VMEM) Image operations transfer data between the VGPRs and memory through the texture
cache. Image operations support access to image objects such as texture maps and typed surfaces. Image-
Sample operations read multiple elements from a surface and combine them to produce a single result per
lane.

Image objects are accessed using from one to four dimensional addresses; they are composed of homogeneous
samples, each sample containing one to four elements. These image objects are read from, or written to, using
IMAGE_* or SAMPLE_* instructions, all of which use the VIMAGE or VSAMPLE instruction formats.
IMAGE_LOAD instructions load an element from the image buffer directly into VGPRS, and SAMPLE
instructions use sampler constants (S#) and apply filtering to the data after it is read. IMAGE_ATOMIC
instructions combine data from VGPRs with data already in memory, and optionally return the value that was
in memory before the operation.

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
Image instruction are divided into two categories based on instruction encoding: VIMAGE and VSAMPLE.

VIMAGE Instructions
IMAGE_LOAD_{-, PCK, PCK_SGN}      Load data from an image object
IMAGE_LOAD_MIP_{-, PCK, PCK_SGN } Load data from an image object from a specified mip level.
IMAGE_STORE                            Store data to an image object to a specific mipmap level
IMAGE_STORE_PCK
IMAGE_STORE_MIP
IMAGE_STORE_MIP_PCK

VIMAGE Instructions
IMAGE_ATOMIC_SWAP                      Image atomic operations
IMAGE_ATOMIC_CMPSWAP
IMAGE_ATOMIC_ADD
IMAGE_ATOMIC_SUB
IMAGE_ATOMIC_SMIN
IMAGE_ATOMIC_UMIN
IMAGE_ATOMIC_SMAX
IMAGE_ATOMIC_UMAX
IMAGE_ATOMIC_AND
IMAGE_ATOMIC_OR
IMAGE_ATOMIC_XOR
IMAGE_ATOMIC_INC
IMAGE_ATOMIC_DEC
IMAGE_ATOMIC_ADD_F32
IMAGE_ATOMIC_MIN_F32
IMAGE_ATOMIC_MAX_F32
IMAGE_ATOMIC_PK_ADD_F16
IMAGE_ATOMIC_PK_ADD_BF16
IMAGE_GET_RESINFO                      Return resource info into 4 VGPRs for the MIP level specified. These are 32bit
                                       integer values:
                                       VDATA3-0 = { #mipLevels, depth, height, width }
                                       For cubemaps, depth = 6 * Number_of_array_faces.
                                       (DX expects the # of cubes, but gets # of faces instead)
BVH ops                                All Ray tracing BVH instructions.
                                       See: Ray Tracing Instructions

VSAMPLE Instructions
IMAGE_SAMPLE_*                         Load and filter data from a image object
IMAGE_SAMPLE_*_G16                     Sample with 16-bit gradients
IMAGE_GATHER4_*                        Load and return samples from 4 texels for software filtering. Returns a single
                                       component, starting with the lower-left texel and in counter-clockwise order.
IMAGE_GATHER4H                         4H: fetch 1 component per texel from 4x1 texels
                                       "DMASK" selects which component to load (R,G,B,A) and must have only one bit
                                       set to 1.
IMAGE_MSAA_LOAD                        Load up to 4 samples of 1 component from an MSAA resource with a user-
                                       specified fragment ID.
                                       Uses DMASK as component select - it behaves like gather4 ops and returns 4
                                       VGPR (2 if D16=1).
                                       SAMP should be set to NULL since this operation does not use a sampler.
IMAGE_GET_LOD                          Return the calculated LOD. Treated as a Sample instruction.
                                       Returns the "raw" LOD and the "clamped" LOD into VDATA as two 32 bit floats:
                                       First VGPR = clampLOD
                                       Second VGPR = rawLOD

                                             Table 58. Instruction Fields

Instruction Fields
Field          Size        Description
OP             8           Opcode
DIM            3           Surface Dimension:

                                                    0: 1D                    4: 1d array
                                                    1: 2D                    5: 2d array
                                                    2: 3D                    6: 2d MSAA
                                                    3: cube                  7: 2d MSAA array
DMASK          4           Data VGPR enable mask: 1 .. 4 consecutive VGPRs
                           Loads: defines which components are returned: 0=red,1=green,2=blue,3=alpha
                           Stores: defines which components are written with data from VGPRs (missing components get
                           the value of the X component).
                           Enabled components come from consecutive VGPRs.
                           E.G. DMASK=1001 : Red is in VGPRn and alpha in VGPRn+1.

                           For D16 loads, DMASK indicates which components to return;
                           For D16 stores, the DMASK the mask indicates which components to store but has restrictions:
                           Data is read out of consecutive VGPRs: LSB’s of VDATA, then MSB’s of VDATA then LSB’s
                           of VDATA+1 and last if needed MSB’s of VDATA+1. This is regardless of which DMASK bits
                           are set, only how many bits are set. The position of the DMASK bits controls which components
                           are written in memory.
                           If DMASK==0, the TA overrides DMASK=1 and puts zeros in VGPR followed by LWE status if exists. TFE
                           status is not generated since the fetch is dropped.
                           For IMAGE_GATHER4* instructions, DMASK indicates which component (RGBA), and the
                           number of VGPRs to use is determined automatically by hardware (4 VGPRs when D16=0, and 2
                           VGPRs when D16=1).
R128           1           Texture Resource Size: 1 = 128bits, 0 = 256bits
A16            1           Address components are 16-bits (instead of the usual 32 bits).
                           When set, all address components are 16 bits (packed into 2 per DWORD), except:
                           Texel offsets (3 6bit UINT packed into 1 DWORD)
                           PCF reference (for "_C" instructions)
                           Address components are 16b uint for image ops without sampler; 16b float with sampler.
D16            1           VGPR-Data-16bit. On loads, convert data in memory to 16-bit format before storing it in VGPRs.
                           For stores, convert 16-bit data in VGPRs to the memory format before going to memory. Whether
                           the data is treated as float or int is decided by NFMT. Allowed only with these opcodes:

                             • IMAGE_SAMPLE*
                             • IMAGE_GATHER4
                             • IMAGE_MSAA_LOAD
                             • IMAGE_LOAD
                             • IMAGE_LOAD_MIP
                             • IMAGE_STORE
                             • IMAGE_STORE_MIP
VDATA          8           Address of VGPR to supply first component of store-data or receive first component of load-data.
                           0-255.
RSRC           9           Specifies which SGPR supplies T# (resource constant) in four consecutive SGPRs. Must be a
                           multiple of 4, in the range 0-120.
SCOPE          2           Memory Scope
TH             3           Memory Temporal Hint
TFE            1           Texel Fault Enable for PRT (Partially Resident Textures). When set, fetch may return a NACK that
                           causes a VGPR write into DST+1 (first GPR after all fetch-dest gprs).
                                               Fields Available only in VIMAGE
