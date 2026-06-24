# 9.3.1. D16 Instructions

> RDNA4 ISA — pages 118–118

9.3. Buffer Data
The amount and type of data that is loaded or stored is controlled by the following: the resource format field,
destination-component-selects (dst_sel), and the opcode.

Data-format can come from the resource, instruction fields, or the opcode itself. Typed buffer ops derive data-
format from the instruction’s FORMAT field, untyped "format" instructions use FORMAT from the resource,
and other buffer opcodes derive data-format from the instruction itself. DST_SEL comes from the resource, but
is ignored for many operations.

                                             Table 52. Buffer Instructions
                               Instruction                      Data Format   DST SEL
                               TBUFFER_LOAD_FORMAT_*            instruction   identity
                               TBUFFER_STORE_FORMAT_*           instruction   identity
                               BUFFER_LOAD_FORMAT_*             resource      resource
                               BUFFER_STORE_FORMAT_*            resource      resource
                               BUFFER_LOAD_<type>               derived       identity
                               BUFFER_STORE_<type>              derived       identity
                               BUFFER_ATOMIC_*                  derived       identity

Instruction : The instruction’s format field is used instead of the resource’s fields.

Data format derived : The data format is derived from the opcode and ignores the resource definition. For
example, BUFFER_LOAD_U8 sets the data-format to uint-8.

                   The resource’s data format must not be INVALID; that format has specific meaning
                  (unbound resource), and for that case the data format is not replaced by the instruction’s
                   implied data format.

DST_SEL identity : Depending on the number of components in the data-format, this is: X000, XY00, XYZ0, or
XYZW.

When the shader provides fewer components than the surface format, the first component is replicated to fill
in the missing ones. E.g. if the surface has 4 components and the shader only supplies X and Y, the result is
written with XYXX.

9.3.1. D16 Instructions
Load and store instructions also come in a "D16" variant. The D16 buffer instructions allow a shader program to
load or store just 16 bits per work-item between VGPRs and memory. For stores, each 32bit VGPR holds two
16bit data elements that are passed to the texture unit which in turn, converts to the buffer format before
writing to memory. For loads, data returned from the texture unit is converted to 16 bits and a pair of data are
stored in each 32bit VGPR (LSBs first, then MSBs). Control over int vs. float is controlled by FORMAT.
Conversion of float32 to float16 uses truncation; conversion of other input data formats uses round-to-nearest-
even.

There are two variants of these instructions:
  • D16 loads data into or stores data from the lower 16 bits of a VGPR.
