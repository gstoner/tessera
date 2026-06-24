# 9.6. Buffer Resource

> RDNA4 ISA — pages 124–125

      For DWORD or larger loads or stores of non-formatted ops (such as BUFFER_LOAD_DWORD), the two
      LSBs of the byte-address are ignored, thus forcing DWORD alignment. For ATOMICs this forces the
      required alignment by ignoring the LSBs until the atomic is payload aligned, this means 8-byte ATOMIC
      operations are forced to a greater alignment than DWORD.
 1. : DWORD_STRICT - must be aligned to the smaller of: element-size or DWORD.
 2. : STRICT - access must be aligned to data size
 3. : UNALIGNED - any alignment is allowed (but atomics must still be aligned)

9.6. Buffer Resource
The buffer resource (V#) describes the location of a buffer in memory and the format of the data in the buffer.
It is specified in four consecutive SGPRs (4-SGPR aligned) and sent to the texture cache with each buffer
instruction.

The table below details the fields that make up the buffer resource descriptor.

                                          Table 57. Buffer Resource Descriptor
Bits           Size      Name                  Description
47:0           48        Base address          Byte address.
61:48          14        Stride                Bytes 0 to 16383 (modified by Stride Scale)
63:62          2         swizzle Enable        Swizzle AOS according to stride, index_stride and element_size; otherwise
                                               linear.
                                               0: disabled
                                               1: enabled with element_size = 4byte
                                               2: Reserved
                                               3: enabled with element_size = 16byte
95:64          32        Num_records           In units of stride if (stride >=1), else in bytes.
98:96          3         Dst_sel_x             Destination channel select:
101:99         3         Dst_sel_y             0=0, 1=1, 4=R, 5=G, 6=B, 7=A
104:102        3         Dst_sel_z
107:105        3         Dst_sel_w
113:108        6         Format                Memory data type. Used only by Untyped-buffer "FORMAT" instructions.
115:114        2         Stride Scale          Multiply the stride field by: 0: x1; 1: x4; 2: x8; 3: x32.
118:117        2         Index stride          0:8, 1:16, 2:32, or 3:64. Used for swizzled buffer addressing.
119            1         Add tid enable        Add thread ID to the index for to calculate the address.
120            1         Write Compression     1 = enable write compression, 0 = disabled
                         Enable
121            1         Compression Enable 0 = bypass compression (resource is not compressible);
                                            1 = don’t bypass compression
123:122        2         Compression Access 0 = normal
                         Mode               1 = force existing data to compress
                                            2 = compressed data access
                                            3 = metadata access
125:124        2         OOB_SELECT            Out of bounds select.
127:126        2         Type                  Value == 0 for buffer. Overlaps upper two bits of four-bit TYPE field in
                                               128-bit V# resource.

Unbound Resources

This is keyed off the "data-format" being set to zero (INVALID), and for untyped-buffer the "add_tid_en = false".

Resource - Instruction mismatch

If the resource type and instruction mismatch (e.g. a buffer resource with an image instruction, or an image
resource with a buffer instruction), the instruction is ignored (loads return nothing and stores do not alter
memory).
