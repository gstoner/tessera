# Chapter 9. Vector Memory Buffer Instructions

> RDNA3 ISA — pages 89–92

Chapter 9. Vector Memory Buffer Instructions
Vector-memory (VM) buffer operations transfer data between the VGPRs and buffer objects in memory
through the texture cache (TC). Vector means that one or more piece of data is transferred uniquely for every
thread in the wave, in contrast to scalar memory loads that transfer only one value that is shared by all threads
in the wave.

The instruction defines which VGPR(s) supply the addresses for the operation, which VGPRs supply or receive
data from the operation, and a series of SGPRs that contain the memory buffer descriptor (V#). Buffer atomics
have the option of returning the pre-op memory value to VGPRs.

Examples of buffer objects are vertex buffers, raw buffers, stream-out buffers, and structured buffers.

Buffer objects support both homogeneous and heterogeneous data, but no filtering of load-data (no samplers).
Buffer instructions are divided into two groups:

MUBUF: Untyped buffer objects
      • Data format is specified in the resource constant.
      • Load, store, atomic operations, with or without data format conversion.
MTBUF: Typed buffer objects
      • Data format is specified in the instruction.
      • The only operations are Load and Store, both with data format conversion.

All buffer operations use a buffer resource constant (V#) that is a 128-bit value in SGPRs. This constant is sent
to the texture cache when the instruction is executed. This constant defines the address and characteristics of
the buffer in memory. Typically, these constants are fetched from memory using scalar memory loads prior to
executing VM instructions, but these constants also can be generated within the shader.

Memory operations of different types (loads, stores) can complete out of order with respect to each other.

Simplified view of buffer addressing

The equation below shows how the memory address is calculated for a buffer access:

Memory instructions return MEMVIOL for any misaligned access when the alignment mode does not allow it.

9.1. Buffer Instructions
Buffer instructions (MTBUF and MUBUF) allow the shader program to load from, and store to, linear buffers in
memory. These operations can operate on data as small as one byte, and up to four DWORDs per work-item.
Atomic operations take data from VGPRs and combine them arithmetically with data already in memory.
Optionally, the value that was in memory before the operation took place can be returned to the shader.

The D16 instruction variants of buffer ops convert the results to and from packed 16-bit values. For example,
BUFFER_LOAD_D16_FORMAT_XYZW stores two VGPRs with 4 16-bit values.

                                               Table 36. Buffer Instructions
MTBUF Instructions
TBUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}                    Load from or store to a Typed buffer object.
TBUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}
TBUFFER_LOAD_D16_FORMAT_{x,xy,xyz,xyzw}                Convert data to 16-bits before loading into VGPRs.
TBUFFER_STORE_D16_FORMAT_{x,xy,xyz,xyzw}               Convert data from 16-bits to tex-format before storing to memory

MUBUF Instructions
BUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}                     Load from or store to an Untyped Buffer object
BUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}                    <size> = I8, U8, I16, U16, B32, B64, B96, B128
BUFFER_LOAD_D16_FORMAT_{x,xy,xyz,xyzw}
BUFFER_STORE_D16_FORMAT_{x,xy,xyz,xyzw}
BUFFER_LOAD_<size> BUFFER_STORE_<size>
BUFFER_{LOAD,STORE}_D16_FORMAT_X
BUFFER_{LOAD,STORE}_D16_HI_FORMAT_X
BUFFER_ATOMIC_<op>                                     Buffer object atomic operation. Automatically globally coherent.
                                                       Operates on 32bit or 64bit values.
BUFFER_GL{0,1}_INV                                     Cache invalidate: either L0 or L1 cache for the CU (L0) and Shader
                                                       Array (L1) associated with this wave.

                                              Table 37. Microcode Formats
Field     Bit Size Description
OP        4        MTBUF: Opcode for Typed buffer instructions.
          8        MUBUF: Opcode for Untyped buffer instructions.
VADDR     8        Address of VGPR to supply first component of address (offset or index). When both index and offset are
                   used, index is in the first VGPR, offset in the second.
VDATA     8        Address of VGPR to supply first component of store data or receive first component of load-data.
SOFFSET 8          SGPR to supply unsigned byte offset. SGPR, M0, NULL, or inline constant.
SRSRC     5        Specifies which SGPR supplies V# (resource constant) in four consecutive SGPRs. This field is missing
                   the two LSBs of the SGPR address, since this address is be aligned to a multiple of four SGPRs.
FORMA 7            Data Format of data in memory buffer. See: Buffer Image Format Table
T
OFFSET 12          Unsigned byte offset.
OFFEN     1        1 = Supply an offset from VGPR (VADDR). 0 = Do not (offset = 0).
IDXEN     1        1 = Supply an index from VGPR (VADDR). 0 = Do not (index = 0).
GLC       1        Globally Coherent. Controls how loads and stores are handled by the L0 texture cache.
                   ATOMIC
                   GLC = 0 Previous data value is not returned.
                   GLC = 1 Previous data value is returned.
DLC       1        Device Level Coherent.
SLC       1        System Level Coherent.

Field     Bit Size Description
TFE       1        Texel Fault Enable for PRT (partially resident textures). When set to 1 and fetch returns a NACK, status
                   is written to the VGPR after the last fetch-dest VGPR.

                                             Table 38. MTBUF Instructions
Opcode                                         Description - all address components for buffer ops are uint
TBUFFER_LOAD_FORMAT_X                          load X component w/ format convert
TBUFFER_LOAD_FORMAT_XY                         load XY components w/ format convert
TBUFFER_LOAD_FORMAT_XYZ                        load XYZ components w/ format convert
TBUFFER_LOAD_FORMAT_XYZW                       load XYZW components w/ format convert
TBUFFER_STORE_FORMAT_X                         store X component w/ format convert
TBUFFER_STORE_FORMAT_XY                        store XY components w/ format convert
TBUFFER_STORE_FORMAT_XYZ                       store XYZ components w/ format convert
TBUFFER_STORE_FORMAT_XYZW                      store XYZW components w/ format convert
TBUFFER_LOAD_D16_FORMAT_X                      load X component w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XY                     load XY components w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XYZ                    load XYZ components w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XYZW                   load XYZW components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_X                     store X component w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XY                    store XY components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XYZ                   store XYZ components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XYZW                  store XYZW components w/ format convert, 16bit

  • TBUFFER*_FORMAT instructions include a data-format conversion specified in the instruction.

                                             Table 39. MUBUF Instructions
Opcode                                    Description - all address components for buffer ops are uint
BUFFER_LOAD_U8                            load unsigned byte (extend 0’s to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_U8                        load unsigned byte into VGPR[15:0]
BUFFER_LOAD_D16_HI_U8                     load unsigned byte into VGPR[31:16]
BUFFER_LOAD_I8                            load signed byte (sign extend to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_I8                        load signed byte into VGPR[15:0]
BUFFER_LOAD_D16_HI_I8                     load signed byte into VGPR[31:16]
BUFFER_LOAD_U16                           load unsigned short (extend 0’s to MSB’s of DWORD VGPR)
BUFFER_LOAD_I16                           load signed short (sign extend to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_B16                       load short into VGPR[15:0]
BUFFER_LOAD_D16_HI_B16                    load short into VGPR[31:16]
BUFFER_LOAD_B32                           load DWORD
BUFFER_LOAD_B64                           load 2 DWORD per element
BUFFER_LOAD_B96                           load 3 DWORD per element
BUFFER_LOAD_B128                          load 4 DWORD per element
BUFFER_LOAD_FORMAT_X                      load X component w/ format convert
BUFFER_LOAD_FORMAT_XY                     load XY components w/ format convert
BUFFER_LOAD_FORMAT_XYZ                    load XYZ components w/ format convert
BUFFER_LOAD_FORMAT_XYZW                   load XYZW components w/ format convert
BUFFER_LOAD_D16_FORMAT_X                  load X component w/ format convert, 16b
BUFFER_LOAD_D16_HI_FORMAT_X               load X component w/ format convert, 16b
BUFFER_LOAD_D16_FORMAT_XY                 load XY components w/ format convert, 16b

Opcode                                 Description - all address components for buffer ops are uint
BUFFER_LOAD_D16_FORMAT_XYZ             load XYZ components w/ format convert, 16b
BUFFER_LOAD_D16_FORMAT_XYZW            load XYZW components w/ format convert, 16b
BUFFER_STORE_B8                        store byte (ignore MSB’s of DWORD VGPR)
BUFFER_STORE_D16_HI_B8                 store byte from VGPR bits [23:16]
BUFFER_STORE_B16                       store short (ignore MSB’s of DWORD VGPR)
BUFFER_STORE_D16_HI_B16                store short from VGPR bits [32:16]
BUFFER_STORE_B32                       store DWORD
BUFFER_STORE_B64                       store 2 DWORD per element
BUFFER_STORE_B96                       store 3 DWORD per element
BUFFER_STORE_B128                      store 4 DWORD per element
BUFFER_STORE_FORMAT_X                  store X component w/ format convert
BUFFER_STORE_FORMAT_XY                 store XY components w/ format convert
BUFFER_STORE_FORMAT_XYZ                store XYZ components w/ format convert
BUFFER_STORE_FORMAT_XYZW               store XYZW components w/ format convert
BUFFER_STORE_D16_FORMAT_X              store X component w/ format convert, 16b
BUFFER_STORE_D16_HI_FORMAT_X           store X component w/ format convert, 16b
BUFFER_STORE_D16_FORMAT_XY             store XY components w/ format convert, 16b
BUFFER_STORE_D16_FORMAT_XYZ            store XYZ components w/ format convert, 16b
BUFFER_STORE_D16_FORMAT_XYZW           store XYZW components w/ format convert, 16b
BUFFER_ATOMIC_ADD_U32                  32b , dst += src, returns previous value if glc==1
BUFFER_ATOMIC_ADD_F32                  32b , dst += src, returns previous value if glc==1
BUFFER_ATOMIC_ADD_U64                  64b , dst += src, returns previous value if glc==1
BUFFER_ATOMIC_AND_B32                  32b , dst &= src, returns previous value if glc==1
BUFFER_ATOMIC_AND_B64                  64b , dst &= src, returns previous value if glc==1
BUFFER_ATOMIC_CMPSWAP_B32              32b , dst = (dst == cmp) ? src : dst, returns previous value if glc==1. Src is from
                                       vdata, cmp from vdata+1
BUFFER_ATOMIC_CMPSWAP_B64              64b , dst = (dst == cmp) ? src : dst, returns previous value if glc==1
BUFFER_ATOMIC_CSUB_U32                 32b , dst = if (src > dst) ? 0 : dst - src, returns previous . GLC must be set to 1.
BUFFER_ATOMIC_DEC_U32                  32b , dst = dst == 0) | (dst > src ? src : dst-1, returns previous value if glc==1
BUFFER_ATOMIC_DEC_U64                  64b , dst = dst == 0) | (dst > src ? src : dst-1, returns previous value if glc==1
BUFFER_ATOMIC_CMPSWAP_F32              32b , dst = (dst == cmp) ? src : dst, returns previous value if glc==1. Src is from
                                       vdata, cmp from vdata+1
BUFFER_ATOMIC_MAX_F32                  32b , dst = (src > dst) ? src : dst, (float) returns previous value if glc==1
BUFFER_ATOMIC_MIN_F32                  32b , dst = (src < dst) ? src : dst, (float) returns previous value if glc==1
BUFFER_ATOMIC_INC_U32                  32b , dst = (dst >= src) ? 0 : dst+1, returns previous value if glc==1
BUFFER_ATOMIC_INC_U64                  64b , dst = (dst >= src) ? 0 : dst+1, returns previous value if glc==1
BUFFER_ATOMIC_OR_B32                   32b , dst |= src, returns previous value if glc==1
BUFFER_ATOMIC_OR_B64                   64b , dst |= src, returns previous value if glc==1
BUFFER_ATOMIC_MAX_I32                  32b , dst = (src > dst) ? src : dst, (signed) returns previous value if glc==1
BUFFER_ATOMIC_MAX_I64                  64b , dst = (src > dst) ? src : dst, (signed) returns previous value if glc==1
BUFFER_ATOMIC_MIN_I32                  32b , dst = (src < dst) ? src : dst, (signed) returns previous value if glc==1
BUFFER_ATOMIC_MIN_I64                  64b , dst = (src < dst) ? src : dst, (signed) returns previous value if glc==1
BUFFER_ATOMIC_SUB_U32                  32b , dst -= src, returns previous value if glc==1
BUFFER_ATOMIC_SUB_U64                  64b , dst -= src, returns previous value if glc==1
BUFFER_ATOMIC_SWAP_B32                 32b , dst = src, returns previous value of dst if glc==1
BUFFER_ATOMIC_SWAP_B64                 64b , dst = src, returns previous value of dst if glc==1
BUFFER_ATOMIC_MAX_U32                  32b , dst = (src > dst) ? src : dst, (unsigned) returns previous value if glc==1
BUFFER_ATOMIC_MAX_U64                  64b , dst = (src > dst) ? src : dst, (unsigned) returns previous value if glc==1
