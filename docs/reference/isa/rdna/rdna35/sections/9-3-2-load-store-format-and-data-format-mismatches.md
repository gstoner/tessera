# 9.3.2. LOAD/STORE_FORMAT and DATA-FORMAT mismatches

> RDNA3.5 ISA — pages 98–98

9.3.1. D16 Instructions
Load-format and store-format instructions also come in a "D16" variant. The D16 buffer instructions allow a
shader program to load or store just 16 bits per work-item between VGPRs and memory. For stores, each 32bit
VGPR holds two 16bit data elements that are passed to the texture unit which in turn, converts to the texture
format before writing to memory. For loads, data returned from the texture unit is converted to 16 bits and a
pair of data are stored in each 32bit VGPR (LSBs first, then MSBs). Control over int vs. float is controlled by
FORMAT. Conversion of float32 to float16 uses truncation; conversion of other input data formats uses round-
to-nearest-even.

There are two variants of these instructions:
  • D16 loads data into or stores data from the lower 16 bits of a VGPR.
  • D16_HI loads data into or stores data from the upper 16 bits of a VGPR.

For example, BUFFER_LOAD_D16_U8 loads a byte per work-item from memory, converts it to a 16-bit integer,
then loads it into the lower 16 bits of the data VGPR.

9.3.2. LOAD/STORE_FORMAT and DATA-FORMAT mismatches
The "format" instructions specify a number of elements (x, xy, xyz or xyzw) and this could mismatch with the
number of elements in the data format specified in the instruction’s or resource’s data-format field. When that
happens.

  • buffer_load_format_x and dfmt is "32_32_32_32" : load 4 DWORDs from memory, but only load first into
    the shader
  • buffer_store_format_x and dfmt is "32_32_32_32" : stores 4 DWORDs to memory based on dst_sel
  • buffer_load_format_xyzw and dfmt is "32" : load 1 DWORD from memory, return 4 to shader (dst_sel)
  • buffer_store_format_xyzw and dfmt is "32" : store 1 DWORD (X) to memory, ignore YZW.

9.4. Buffer Addressing
A buffer is a data structure in memory that is addressed with an index and an offset. The index points to a
particular record of size stride bytes, and the offset is the byte-offset within the record. The stride comes from
the resource, the index from a VGPR (or zero), and the offset from an SGPR or VGPR and also from the
instruction itself.

                                   Table 43. BUFFER Instruction Fields for Addressing
Field      Size Description
inst_offset 12   Literal byte offset from the instruction.
inst_idxen 1     Boolean: get per-lane index from VGPR when true, or no index when false.
inst_offen 1     Boolean: get per-lane offset from VGPR when true, or no offset when false. Note that inst_offset is present
                 regardless of this bit.

The "element size" for a buffer instruction is the amount of data the instruction transfers in bytes. It is
determined by the FORMAT field for MTBUF instructions, or from the opcode for MUBUF instructions, and is:
1, 2, 4, 8, 12 or 16 bytes. For example, format "16_16" has an element size of 4-bytes.
