# 8.1.3. Loads of 8 and 16-bit Data

> RDNA4 ISA — pages 109–109

  "m_*" components come from the buffer resource (V#):
    def alignToSize(addr): return (dataSize == 8bit ? addr :      dataSize == 16bit ? addr & ~1 :   addr & ~3)

    resource   = { SGPR[SBASE * 2 +3], SGPR[SBASE * 2 +2], SGPR[SBASE * 2 +1], SGPR[SBASE * 2] }
    m_base         = alignToSize(resource[47:0], dataSize)
    m_stride       = resource[61:48]
    m_num_records = resource[95:64]
    offset      = alignToSize(IOFFSET) + alignToSize(SOFFSET)
    m_size      = alignToSize((m_stride == 0 ? 1 : m_stride) * m_num_records)
    addr        = m_base + offset
    SGPR[SDST] = load_dword_from_dcache(addr, m_size)

    If more than 1 DWORD is being loaded, it is returned to SDST+1, SDST+2, etc,
    and the offset is incremented by 4 bytes per DWORD.
    Note: the V#.stride_scale field is ignored.

8.1.3. Loads of 8 and 16-bit Data
  • S_LOAD_U8: load 8-bit unsigned data, zero-extend to 32 bits
  • S_LOAD_I8: load 8-bit signed data, sign-extend to 32 bits
  • S_LOAD_U16: load 16-bit unsigned data, zero-extend to 32 bits
  • S_LOAD_I16: load 16-bit signed data, sign-extend to 32 bits

Same for S_BUFFER_LOAD.
16-bit loads must be 16-bit aligned in memory.

8.1.4. S_DCACHE_INV
These instructions invalidate the entire scalar constant cache. It does not return anything to SDST.
S_DCACHE_INV does not have any address or data arguments.

8.2. Dependency Checking
Scalar memory loads can return data out-of-order from how they were issued; they can return partial results at
different times when the load crosses two cache lines. The shader program uses the KMcnt counter to
determine when the data has been returned to the SDST SGPRs. This is done as follows.

  • KMcnt is incremented by 1 for every fetch of a single DWORD, or cache invalidates.
  • KMcnt is incremented by 2 for every fetch of two or more DWORDs.
  • KMcnt is decremented by an equal amount when each instruction completes.

Because the instructions can return out-of-order, the only sensible way to use this counter is to implement
"S_WAIT_KMCNT 0"; this imposes a wait for all data to return from previous SMEMs before continuing.

Cache invalidate instructions are not known to have completed until the shader waits for KMcnt==0.
