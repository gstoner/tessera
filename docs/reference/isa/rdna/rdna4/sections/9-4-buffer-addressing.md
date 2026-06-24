# 9.4. Buffer Addressing

> RDNA4 ISA — pages 119–119

  • D16_HI loads data into or stores data from the upper 16 bits of a VGPR.

For example, BUFFER_LOAD_D16_U8 loads a byte per work-item from memory, converts it to a 16-bit integer,
then loads it into the lower 16 bits of the data VGPR.

9.3.2. LOAD/STORE_FORMAT and DATA-FORMAT mismatches
The "format" instructions specify a number of elements (x, xy, xyz or xyzw) and this could mismatch with the
number of elements in the data format specified in the instruction’s or resource’s data-format field. When that
happens.

  • buffer_load_format_x and dfmt is "32_32_32_32" : load 4 DWORDs from memory, but only load first into
    the shader.
  • buffer_store_format_x and dfmt is "32_32_32_32" : stores "x" to all non-constant channels in memory if "x"
    is sent from shader, stores 0 otherwise.
  • buffer_load_format_xyzw and dfmt is "32" : load 1 DWORD from memory, return 4 to shader (dst_sel).
  • buffer_store_format_xyzw and dfmt is "32" : store 1 DWORD (X) to memory, ignore YZW.

9.4. Buffer Addressing
A buffer is a data structure in memory that is addressed with an index and an offset. The index points to a
particular record of size stride bytes, and the offset is the byte-offset within the record. The stride comes from
the resource, the index from a VGPR (or zero), and the offset from an SGPR or VGPR and also from the
instruction itself.

                                    Table 53. BUFFER Instruction Fields for Addressing
Field    Size Description
IOFFSET 24     Literal byte offset from the instruction.
IDXEN    1     Boolean: get per-lane index from VGPR when true, or no index when false.
OFFEN    1     Boolean: get per-lane offset from VGPR when true, or no offset when false. Note that IOFFSET is present
               regardless of this bit.

The "element size" for a buffer instruction is the amount of data the instruction transfers. It is determined by
the FORMAT field for typed-buffer instructions, or from the opcode for untyped-buffer instructions, and is: 1,
2, 4, 8, 12 or 16 bytes. For example, format "16_16" has an element size of 4-bytes.

                                  Table 54. Buffer Resource Constant Fields for Addressing
Field                        Size      Description
const_base                   48        Base address of the buffer resource, in bytes.
const_stride                 14        Stride of the record in bytes, then multiplied by stride_scale.
const_num_records            32        Number of records in the buffer. In units of:
                                       Bytes if: const_stride ⇐ 1, otherwise, in units of "stride".
const_add_tid_enable         1         Boolean. Add thread_ID within the wave to the index when true.
const_swizzle_enable         2         Swizzle AOS (Array of Structures) according to stride, index_stride and element_size:
                                       0: disabled
                                       1: enabled with element_size = 4-byte
                                       2: Reserved
                                       3: enabled with element_size = 16-byte
