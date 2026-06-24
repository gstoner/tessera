# Chapter 9. Vector Memory Buffer Instructions

> RDNA4 ISA — pages 112–115

Chapter 9. Vector Memory Buffer Instructions
Vector-memory (VM) buffer operations transfer data between the VGPRs and buffer objects in memory
through the L0 cache. Vector means that one or more piece of data is transferred uniquely for every thread in
the wave, in contrast to scalar memory loads that transfer only one value that is shared by all threads in the
wave.

The instruction defines which VGPR(s) supply the addresses for the operation, which VGPRs supply or receive
data from the operation, and a series of SGPRs that contain the memory buffer descriptor (V#). Buffer atomics
have the option of returning the pre-op memory value to VGPRs.

Examples of buffer objects are vertex buffers, raw buffers, stream-out buffers, and structured buffers.

Buffer objects support both homogeneous and heterogeneous data, but no filtering of load-data. Buffer
instructions are divided into two groups:

Untyped buffer objects
      • Data format is specified in the resource constant.
      • Load, store, atomic operations, with or without data format conversion.
Typed buffer objects
      • Data format is specified in the instruction.
      • The only operations are Load and Store, both with data format conversion.

All buffer operations use a buffer resource constant (V#) that is a 128-bit value in SGPRs. This constant is sent
when the instruction is executed. This constant defines the address and characteristics of the buffer in
memory. Typically, these constants are fetched from memory using scalar memory loads prior to executing
VM instructions, but these constants also can be generated within the shader.

Memory operations of different types (loads, stores) can complete out of order with respect to each other.

Simplified view of buffer addressing

The equation below shows how the memory address is calculated for a buffer access:

Memory instructions return MEMVIOL for any misaligned access when the alignment mode does not allow it.

9.1. Buffer Instructions
Buffer instructions allow the shader program to load from, and store to, linear buffers in memory. These
operations can operate on data as small as one byte, and up to four DWORDs per work-item. Atomic operations
take data from VGPRs and combine them arithmetically with data already in memory. Optionally, the value
that was in memory before the operation took place can be returned to the shader.

The D16 instruction variants of buffer ops convert the results to and from packed 16-bit values. For example,
BUFFER_LOAD_D16_FORMAT_XYZW stores two VGPRs with 4 16-bit values.

                                              Table 46. Buffer Instructions
Untyped Buffer Instructions
BUFFER_LOAD_<size>                                     Load from or store to an Untyped Buffer object
BUFFER_STORE_<size>                                    <size> = I8, U8, I16, U16, B32, B64, B96, B128
BUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}
BUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}
BUFFER_LOAD_D16_FORMAT_{x,xy,xyz,xyzw}
BUFFER_STORE_D16_FORMAT_{x,xy,xyz,xyzw}
BUFFER_LOAD_D16_FORMAT_X
BUFFER_STORE_D16_FORMAT_X
BUFFER_LOAD_D16_HI_FORMAT_X
BUFFER_STORE_D16_HI_FORMAT_X
BUFFER_ATOMIC_<op>                                     Buffer object atomic operation. Automatically globally coherent.
                                                       Operates on 32bit or 64bit values.

Typed Buffer Instructions
TBUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}                    Load from or store to a Typed buffer object.
TBUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}
TBUFFER_LOAD_D16_FORMAT_{x,xy,xyz,xyzw}                Convert data to 16-bits before loading into VGPRs.
TBUFFER_STORE_D16_FORMAT_{x,xy,xyz,xyzw}               Convert data from 16-bits to tex-format before storing to memory

                                              Table 47. Microcode Formats
Field              Bit Size   Description
OP                 8          Opcode
SOFFSET            7          SGPR to supply unsigned byte offset. SGPR, M0, NULL
VADDR              8          Address of VGPR to supply first component of address (offset or index). When both index
                              and offset are used, index is in the first VGPR, offset in the second.
VDATA              8          Address of VGPR to supply first component of store data or receive first component of load-
                              data.
IOFFSET            24         signed 24-bit byte offset, must be non-negative.
RSRC               9          Specifies which SGPR supplies V# (resource constant) in four consecutive SGPRs. Must be a
                              multiple of 4, in the range 0-120.
FORMAT             7          Data Format of data in memory buffer for Typed-Buffer instructions. See: Buffer Image
                              Format Table
OFFEN              1          1 = Supply an offset from VGPR (VADDR). 0 = Do not (offset = 0).
IDXEN              1          1 = Supply an index from VGPR (VADDR). 0 = Do not (index = 0).
SCOPE              2          Memory Scope
TH                 3          Memory Temporal Hint. For atomics, indicates whether or not to return the pre-op value.
TFE                1          Texel Fault Enable for PRT (partially resident textures). When set to 1 and fetch returns a
                              NACK, status is written to the VGPR after the last fetch-dest VGPR.

TBUFFER*_FORMAT instructions (shown below) include a data-format conversion specified in the instruction.
The instruction specifies the format of data in memory, and it is expanded to 32-bits per component in VGPRs
except for "D16" instructions - they expand data to 16-bits per component.

                                        Table 48. Typed Buffer Instructions
Opcode                                      Description - all address components for buffer ops are uint
TBUFFER_LOAD_FORMAT_X                       load X component w/ format convert
TBUFFER_LOAD_FORMAT_XY                      load XY components w/ format convert
TBUFFER_LOAD_FORMAT_XYZ                     load XYZ components w/ format convert
TBUFFER_LOAD_FORMAT_XYZW                    load XYZW components w/ format convert
TBUFFER_STORE_FORMAT_X                      store X component w/ format convert
TBUFFER_STORE_FORMAT_XY                     store XY components w/ format convert
TBUFFER_STORE_FORMAT_XYZ                    store XYZ components w/ format convert
TBUFFER_STORE_FORMAT_XYZW                   store XYZW components w/ format convert
TBUFFER_LOAD_D16_FORMAT_X                   load X component w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XY                  load XY components w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XYZ                 load XYZ components w/ format convert, 16bit
TBUFFER_LOAD_D16_FORMAT_XYZW                load XYZW components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_X                  store X component w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XY                 store XY components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XYZ                store XYZ components w/ format convert, 16bit
TBUFFER_STORE_D16_FORMAT_XYZW               store XYZW components w/ format convert, 16bit

BUFFER*_FORMAT instructions (shown below) include a data-format conversion specified in the resource
constant (V#).

In the table below, "D16" means the data in the VGPR is 16-bits, not the usual 32 bits.
"D16_HI" means that the upper 16-bits of the VGPR is used instead of "D16" that uses the lower 16 bits.

                                       Table 49. Untyped Buffer Instructions
Opcode                                  Description - all address components for buffer ops are uint
BUFFER_LOAD_U8                          load unsigned byte (extend 0’s to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_U8                      load unsigned byte into VGPR[15:0]
BUFFER_LOAD_D16_HI_U8                   load unsigned byte into VGPR[31:16]
BUFFER_LOAD_I8                          load signed byte (sign extend to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_I8                      load signed byte into VGPR[15:0]
BUFFER_LOAD_D16_HI_I8                   load signed byte into VGPR[31:16]
BUFFER_LOAD_U16                         load unsigned short (extend 0’s to MSB’s of DWORD VGPR)
BUFFER_LOAD_I16                         load signed short (sign extend to MSB’s of DWORD VGPR)
BUFFER_LOAD_D16_B16                     load short into VGPR[15:0]
BUFFER_LOAD_D16_HI_B16                  load short into VGPR[31:16]
BUFFER_LOAD_B32                         load DWORD
BUFFER_LOAD_B64                         load 2 DWORD per element
BUFFER_LOAD_B96                         load 3 DWORD per element
BUFFER_LOAD_B128                        load 4 DWORD per element
BUFFER_LOAD_FORMAT_X                    load X component w/ format convert
BUFFER_LOAD_FORMAT_XY                   load XY components w/ format convert
BUFFER_LOAD_FORMAT_XYZ                  load XYZ components w/ format convert
BUFFER_LOAD_FORMAT_XYZW                 load XYZW components w/ format convert
BUFFER_LOAD_D16_FORMAT_X                load X component w/ format convert, 16b
BUFFER_LOAD_D16_HI_FORMAT_X             load X component w/ format convert, 16b into upper 16-bits of VGPR
BUFFER_LOAD_D16_FORMAT_XY               load XY components w/ format convert, 16b

Opcode                                 Description - all address components for buffer ops are uint
BUFFER_LOAD_D16_FORMAT_XYZ             load XYZ components w/ format convert, 16b
BUFFER_LOAD_D16_FORMAT_XYZW            load XYZW components w/ format convert, 16b
BUFFER_STORE_B8                        store byte (ignore MSB’s of DWORD VGPR)
BUFFER_STORE_D16_HI_B8                 store byte from VGPR bits [23:16]
BUFFER_STORE_B16                       store short (ignore MSB’s of DWORD VGPR)
BUFFER_STORE_D16_HI_B16                store short from VGPR bits [31:16]
BUFFER_STORE_B32                       store DWORD
BUFFER_STORE_B64                       store 2 DWORD per element
BUFFER_STORE_B96                       store 3 DWORD per element
BUFFER_STORE_B128                      store 4 DWORD per element
BUFFER_STORE_FORMAT_X                  store X component w/ format convert
BUFFER_STORE_FORMAT_XY                 store XY components w/ format convert
BUFFER_STORE_FORMAT_XYZ                store XYZ components w/ format convert
BUFFER_STORE_FORMAT_XYZW               store XYZW components w/ format convert
BUFFER_STORE_D16_FORMAT_X              store X component w/ format convert, 16b
BUFFER_STORE_D16_HI_FORMAT_X           store X component w/ format convert, 16b from upper 16-bits of VGPR
BUFFER_STORE_D16_FORMAT_XY             store XY components w/ format convert, 16b
BUFFER_STORE_D16_FORMAT_XYZ            store XYZ components w/ format convert, 16b
BUFFER_STORE_D16_FORMAT_XYZW           store XYZW components w/ format convert, 16b
BUFFER_ATOMIC_ADD_U32                  32b , dst += src, returns previous value if TH==RET
BUFFER_ATOMIC_ADD_F32                  32b , dst += src, returns previous value if TH==RET
BUFFER_ATOMIC_PK_ADD_F16               32b , dst[15:0] += src[15:0]; dst[31:16] += src[31:16]. returns previous value if
                                       TH==RET
BUFFER_ATOMIC_ADD_U64                  64b , dst += src, returns previous value if TH==RET
BUFFER_ATOMIC_AND_B32                  32b , dst &= src, returns previous value if TH==RET
BUFFER_ATOMIC_AND_B64                  64b , dst &= src, returns previous value if TH==RET
BUFFER_ATOMIC_CMPSWAP_B32              32b , dst = (dst == cmp) ? src : dst, returns previous value if TH==RET. Src is from
                                       vdata, cmp from vdata+1
BUFFER_ATOMIC_CMPSWAP_B64              64b , dst = (dst == cmp) ? src : dst, returns previous value if TH==RET
BUFFER_ATOMIC_DEC_U32                  32b , dst = ( (dst == 0) | (dst > src) ) ? src : dst-1, returns previous value if TH==RET
BUFFER_ATOMIC_DEC_U64                  64b , dst = ( (dst == 0) | (dst > src) ) ? src : dst-1, returns previous value if TH==RET
BUFFER_ATOMIC_MAX_NUM_F32              32b , dst = (src > dst) ? src : dst, (float) returns previous value if TH==RET
BUFFER_ATOMIC_MIN_NUM_F32              32b , dst = (src < dst) ? src : dst, (float) returns previous value if TH==RET
BUFFER_ATOMIC_INC_U32                  32b , dst = (dst >= src) ? 0 : dst+1, returns previous value if TH==RET
BUFFER_ATOMIC_INC_U64                  64b , dst = (dst >= src) ? 0 : dst+1, returns previous value if TH==RET
BUFFER_ATOMIC_OR_B32                   32b , dst |= src, returns previous value if TH==RET
BUFFER_ATOMIC_OR_B64                   64b , dst |= src, returns previous value if TH==RET
BUFFER_ATOMIC_MAX_I32                  32b , dst = (src > dst) ? src : dst, (signed) returns previous value if TH==RET
BUFFER_ATOMIC_MAX_I64                  64b , dst = (src > dst) ? src : dst, (signed) returns previous value if TH==RET
BUFFER_ATOMIC_MIN_I32                  32b , dst = (src < dst) ? src : dst, (signed) returns previous value if TH==RET
BUFFER_ATOMIC_MIN_I64                  64b , dst = (src < dst) ? src : dst, (signed) returns previous value if TH==RET
BUFFER_ATOMIC_SUB_U32                  32b , dst -= src, returns previous value if TH==RET
BUFFER_ATOMIC_SUB_U64                  64b , dst -= src, returns previous value if TH==RET
BUFFER_ATOMIC_SWAP_B32                 32b , dst = src, returns previous value of dst if TH==RET
BUFFER_ATOMIC_SWAP_B64                 64b , dst = src, returns previous value of dst if TH==RET
BUFFER_ATOMIC_MAX_U32                  32b , dst = (src > dst) ? src : dst, (unsigned) returns previous value if TH==RET
BUFFER_ATOMIC_MAX_U64                  64b , dst = (src > dst) ? src : dst, (unsigned) returns previous value if TH==RET
BUFFER_ATOMIC_MIN_U32                  32b , dst = (src < dst) ? src : dst, (unsigned) returns previous value if TH==RET
