# 9.4.1. Range Checking

> RDNA4 ISA — pages 120–121

Field                        Size      Description
const_index_stride           2         Used only when const_swizzle_en = true. Number of contiguous indices for a single
                                       element (of const_element_size=4 or 16 bytes) before switching to the next element.
                                       8, 16, 32 or 64 indices.

                                 Table 55. Address Components from GPRs
Field           Size Description
SGPR_offset     32   An unsigned byte-offset to the address. Comes from an SGPR or M0.
VGPR_offset     32   An optional unsigned byte-offset. It is per-thread, and comes from a VGPR.
VGPR_index      32   An optional index value. It is per-thread and comes from a VGPR.

The final buffer memory address is composed of three parts:
    • the base address from the buffer resource (V#),
    • the offset from the SGPR, and
    • a buffer-offset that is calculated differently, depending on whether the buffer is linearly addressed (a
      simple Array-of-Structures calculation) or is swizzled.

                                          Address Calculation for a Linear Buffer

9.4.1. Range Checking
Buffer addresses are checked against the size of the memory buffer. Loads that are out of range return zero,
and stores and atomics are dropped. Range checking is per-component for non-formatted loads and stores that
are larger than one DWORD. Note that load/store_B64, B96 and B128 are considered "2-DWORD/3-DWORD/4-
DWORD load/store", and each DWORD is bounds checked separately. The method of clamping is controlled by
a 2-bit field in the buffer resource: OOB_SELECT (Out of Bounds select).

Payload below is the number of bytes the instruction transfers.

                                         Table 56. Buffer Out Of Bounds Selection
OOB          Out of Bounds if:                                                          Description or use
SELECT
0            (index >= NumRecords) || (offset+payload > stride)                         structured buffers
1            (index >= NumRecords)                                                      Raw buffers
2            (NumRecords == 0)                                                          do not check bounds (except
                                                                                        empty buffer)

OOB        Out of Bounds if:                                                             Description or use
SELECT
3          Bounds check:                                                                 Raw
                                                                                         In this mode, "num_records" is
                                                                                         reduced by "sgpr_offset"
              if (swizzle_en && const_stride != 0x0)
                  OOB = (index >= NumRecords || (offset+payload > stride))
              else
                  OOB = (offset+payload > NumRecords)

Notes:
 1. Loads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1).
 2. Stores that are out-of-range do not store anything.
 3. Load/store-format-* instruction and atomics are range-checked "all or nothing" - either entirely in or out.
 4. Load/store-B{64,96,128} and range-check per component.
    For typed-buffer, if any component of the thread is out of bounds, the whole thread is considered out of
    bounds and returns zero. For untyped-buffer, only the components that are out of bounds return zero.

9.4.1.1. Structured Buffer
The address calculation for swizzle_en==0 is: (unswizzled structured buffer)

     ADDR = Base     + baseOff + Ioff +    Stride * Vidx       + (OffEn ? Voff : 0)
             V#          SGPR    INST         V#        VGPR       INST      VGPR

NumRecords for structured buffer is in units of stride.

9.4.1.2. Raw Buffer

     ADDR = Base     + baseOff + Ioff +    (OffEn ? Voff : 0)
             V#          SGPR    INST         INST      VGPR

NumRecords for raw buffer is in units of bytes. This is an exact range check, meaning it includes the payload
and handles multi-DWORD and unaligned correctly. The stride field is ignored.

9.4.1.3. Scratch Buffer
The address calculation for swizzle_en = 0 is…(unswizzled scratch buffer)

     ADDR = Base     + baseOffset + Ioff +     Stride * TID +       (OffEn ? Voff : 0)
             V#          SGPR          INST        V#      0..63      INST      VGPR

Swizzle of scratch buffer is also supported (and is typical). The MSBs of the TID (TID / 64) is folded into
baseOffset. No range checking (using OOB mode 2).
