# 8.1.2. Loads using Buffer Constant

> RDNA3.5 ISA — pages 88–88

These instructions load 1-16 DWORDs from memory. The data in SGPRs is specified in SDATA, and the address
is composed of the SBASE, OFFSET, and SOFFSET fields.

8.1.1. Scalar Memory Addressing
Non-buffer S_LOAD instructions use the following formula to calculate the memory address:

  ADDR = SGPR[base] + inst_offset + { M0 or SGPR[offset] or zero }

All components of the address (base, offset, inst_offset, M0) are in bytes, but the two LSBs are ignored and
treated as if they were zero.

It is illegal and undefined for the inst_offset to be negative if the resulting
(inst_offset + (M0, SGPR[offset], or zero)) is negative.

8.1.2. Loads using Buffer Constant
S_BUFFER_LOAD instructions use a similar formula, but the base address comes from the buffer constant’s
base_address field.

Buffer constant fields used: base_address, stride, num_records. Other fields are ignored.

Scalar memory load does not support "swizzled" buffers. Stride is used only for memory address bounds
checking, not for computing the address to access.

The SMEM supplies only a SBASE address (byte) and an offset (byte or DWORD). Any "index * stride" must be
calculated manually in shader code and added to the offset prior to the SMEM. Inst_offset must be non-
negative - a negative value of inst_offset results in a MEMVIOL.

The two LSBs of V#.base and of the final address are ignored to force DWORD alignment.

  "m_*" components come from the buffer constant (V#):
    offset      = OFFSET + SOFFSET (M0, SGPR or zero)
    m_base      = { SGPR[SBASE * 2 +1][15:0], SGPR[SBASE*2] }
    m_stride    = SGPR[SBASE * 2 +1][31:16]
    m_num_records = SGPR[SBASE * 2 + 2]
    m_size      = (m_stride == 0 ? 1 : m_stride) * m_num_records
    addr        = (m_base & ~3) + (offset & ~0x3)
    SGPR[SDST] = load_dword_from_dcache(addr, m_size)

    If more than 1 DWORD is being loaded, it is returned to SDST+1, SDST+2, etc,
    and the offset is incremented by 4 bytes per DWORD.

8.1.3. S_DCACHE_INV and S_GL1_INV
This instruction invalidates the entire scalar cache or L1 cache. It does not return anything to SDST.
