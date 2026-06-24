# 9.4.2. Swizzled Buffer Addressing

> RDNA4 ISA — pages 122–122

9.4.1.4. Scalar Memory
Scalar memory does the following, that works with RAW buffers and unswizzled structured buffers:

  Addr =    Base    +    offset
               V#         SGPR or Inst

Address Out-of-Range if: offset >= ( (stride==0 ? 1 : stride) * num_records).

Notes
 1. Loads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1).
    Stores that are out of range do not store anything.
 2. Atomics and Load/store-format-* instruction are range-checked "all or nothing" - either entirely in or out.
 3. Load/store-DWORD-x{2,3,4} perform range-check per component.

9.4.2. Swizzled Buffer Addressing
Swizzled addressing rearranges the data in the buffer that may improve cache locality for arrays of structures.
A single fetch instruction must not fetch a unit larger than const_element_size. The buffer’s STRIDE must be a
multiple of const_element_size.

const_element_size is either 4 or 16 bytes, depending on the setting of V#.swizzle_enable

  Index          = (IDXEN ? vgpr_index : 0) + (const_add_tid_enable ? thread_id[5:0] : 0)
  index_msb         = index / const_index_stride
  index_lsb         = index % const_index_stride

  total_offset      = ( (OFFEN ? vgpr_offset : 0) + IOFFSET + sgpr_offset) & 32'hffffffff
  offset_msb        = total_offset / const_element_size
  offset_lsb        = total_offset % const_element_size

  buffer_offset     = (index_msb * const_stride + offset_msb * const_element_size) * const_index_stride +
                         index_lsb * const_element_size + offset_lsb

  Final Address = const_base + buffer_offset

Restrictions and behavior with swizzled buffers:
  • 16-byte elements are not supported with const_swizzle_en==3 (16byte)
  • For 16-byte elements that are only DWORD-aligned, addresses to higher dwords are offset linearly from the
    address of the first dword

                                              Example of Buffer Swizzling
