# 9.4.2. Swizzled Buffer Addressing

> RDNA3.5 ISA — pages 101–101

9.4.1.3. Scratch Buffer
The address calculation for swizzle_en = 0 is…(unswizzled scratch buffer)

     ADDR = Base      + baseOffset + Ioff +   Stride * TID +    (OffEn ? Voff : 0)
              V#         SGPR          INST      V#     0..63      INST    VGPR

Swizzle of scratch buffer is also supported (and is typical). The MSBs of the TID (TID / 64) is folded into
baseOffset. No range checking (using OOB mode 2).

9.4.1.4. Scalar Memory
Scalar memory does the following, that works with RAW buffers and unswizzled structured buffers:

  Addr =     Base    +    offset
               V#          SGPR or Inst

Address Out-of-Range if: offset >= ( (stride==0 ? 1 : stride) * num_records).

Notes
 1. Loads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1).
    Stores that are out of range do not write anything.
 2. Load/store-format-* instruction and atomics are range-checked "all or nothing" - either entirely in or out.
 3. Load/store-DWORD-x{2,3,4} perform range-check per component.

9.4.2. Swizzled Buffer Addressing
Swizzled addressing rearranges the data in the buffer that may improve cache locality for arrays of structures.
Swizzled addressing also requires DWORD-aligned accesses. A single fetch instruction must not fetch a unit
larger than const_element_size. The buffer’s STRIDE must be a multiple of const_element_size.

const_element_size is either 4 or 16 bytes, depending on the setting of V#.swizzle_enable

  Index            = (inst_idxen ? vgpr_index : 0) + (const_add_tid_enable ? thread_id[5:0] : 0)
  Offset           = (inst_offen ? vgpr_offset : 0) + inst_offset

  index_msb          = index / const_index_stride
  index_lsb          = index % const_index_stride
  offset_msb         = offset / const_element_size
  offset_lsb         = offset % const_element_size

  buffer_offset      = (index_msb * const_stride + offset_msb * const_element_size) * const_index_stride +
                         index_lsb * const_element_size + offset_lsb

  Final Address = const_base + sgpr_offset + buffer_offset
        The "sgpr_offset" is not a part of the "offset" term in the above equations - it's in the "base".
