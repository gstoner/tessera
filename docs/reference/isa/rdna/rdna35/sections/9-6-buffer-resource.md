# 9.6. Buffer Resource

> RDNA3.5 ISA — pages 103–104

 0. : DWORD - hardware automatically aligns request to the smaller of: element-size or DWORD.
    For DWORD or larger loads or stores of non-formatted ops (such as BUFFER_LOAD_DWORD), the two
    LSBs of the byte-address are ignored, thus forcing DWORD alignment.
 1. : DWORD_STRICT - must be aligned to the smaller of: element-size or DWORD.
 2. : STRICT - access must be aligned to data size
 3. : UNALIGNED - any alignment is allowed

Options 1 and 2 report MEMVIOL if a request is made with incorrect address alignment. In options 1 and 2,
loads that are misaligned return zero, and stores that are misaligned are discarded. Note that in this context
"element-size" refers to the size of the data transfer indicated by the instruction, not const_element_size.

9.6. Buffer Resource
The buffer resource (V#) describes the location of a buffer in memory and the format of the data in the buffer.
It is specified in four consecutive SGPRs (4-SGPR aligned) and sent to the texture cache with each buffer
instruction.

The table below details the fields that make up the buffer resource descriptor.

                                           Table 47. Buffer Resource Descriptor
Bits          Size        Name                   Description
47:0          48          Base address           Byte address.
61:48         14          Stride                 Bytes 0 to 16383
63:62         2           swizzle Enable         Swizzle AOS according to stride, index_stride and element_size;
                                                 otherwise linear.
                                                 0: disabled
                                                 1: enabled with element_size = 4byte
                                                 2: Reserved
                                                 3: enabled with element_size = 16byte
95:64         32          Num_records            In units of stride if (stride >=1), else in bytes.
98:96         3           Dst_sel_x              Destination channel select:
101:99        3           Dst_sel_y              0=0, 1=1, 4=R, 5=G, 6=B, 7=A
104:102       3           Dst_sel_z
107:105       3           Dst_sel_w
113:108       6           Format                 Memory data type.
118:117       2           Index stride           0:8, 1:16, 2:32, or 3:64. Used for swizzled buffer addressing.
119           1           Add tid enable         Add thread ID to the index for to calculate the address.
123:122       2           Reserved               Set to zero.
125:124       2           OOB_SELECT             Out of bounds select.
127:126       2           Type                   Value == 0 for buffer. Overlaps upper two bits of four-bit TYPE field in
                                                 128-bit V# resource.

Unbound Resources

Setting the resource constant to all zeros has the effect of forcing any loads to return zero, and stores to be
ignored. This is keyed off the "data-format" being set to zero (INVALID), and for MUBUF the "add_tid_en =
false".

Resource - Instruction mismatch

If the resource type and instruction mismatch (e.g. a buffer constant with an image instruction, or an image
resource with a buffer instruction), the instruction is ignored (loads return nothing and stores do not alter
memory).
