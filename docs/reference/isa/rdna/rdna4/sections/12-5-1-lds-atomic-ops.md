# 12.5.1. LDS Atomic Ops

> RDNA4 ISA — pages 162–162

Single Address Instructions

  LDS_Addr = LDS_BASE + VGPR[ADDR] + {OFFSET1,OFFSET0}

Double Address Instructions

  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + OFFSET0*ADJ +
  LDS_Addr1 = LDS_BASE + VGPR[ADDR] + OFFSET1*ADJ +
     Where ADJ = 4 for 8, 16 and 32-bit data types; and ADJ = 8 for 64-bit.

The double address instructions are: LOAD_2ADDR*, STORE_2ADDR*, and STOREXCHG_2ADDR_*. The
address comes from VGPR, and both VGPR[ADDR] and OFFSET are byte addresses. At the time of wave
creation, LDS_BASE is assigned to the physical LDS region owned by this wave or work-group.

DS_{LOAD,STORE}_ADDTID Addressing

  LDS_Addr = LDS_BASE + {OFFSET1, OFFSET0} + TID(0..63)*4 + M0
    Note: no part of the address comes from a VGPR.           M0 must be DWORD-aligned.

The "ADDTID" (add thread-id) is a separate form where the base address for the instruction is common to all
threads, but then each thread has a fixed offset added in based on its thread-ID within the wave. This allows a
convenient way to quickly transfer data between VGPRs and LDS without having to use a VGPR to supply an
address.

12.5.1. LDS Atomic Ops
Atomic ops combine data from a VGPR with data in LDS, write the result back to LDS memory and optionally
return the "pre-op" value from LDS memory back to a VGPR. When multiple lanes in a wave access the same
LDS location there it is not specified in which order the lanes perform their operations, only that each lane
performs the complete read-modify-write operation before another lane operates on the data.

  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + {OFFSET1,OFFSET0}

VGPR[ADDR] is a byte address. VGPRs 0,1 and dst are double-GPRs for doubles data. VGPR data sources can
only be VGPRs or constant values, not SGPRs. Floating point atomic ops use the MODE register to control
denormal flushing behavior.

Atomic Opcodes
Instruction Fields: op, offset0, offset1, vdst, addr, data0, data1
32-bit no return          32-bit with return                     64-bit no return     64-bit with return
ds_add_u32                ds_add_rtn_u32                         ds_add_u64           ds_add_rtn_u64
ds_sub_u32                ds_sub_rtn_u32                         ds_sub_u64           ds_rsub_rtn_u64
ds_rsub_u32               ds_rsub_rtn_u32                        ds_rsub_u64          ds_rsub_rtn_u64
ds_inc_u32                ds_inc_rtn_u32                         ds_inc_u64           ds_inc_rtn_u64
ds_dec_u32                ds_dec_rtn_u32                         ds_dec_u64           ds_dec_rtn_u64
ds_min_{u32,i32}          ds_min_rtn_{u32,i32}                   ds_min_{u64,i64}     ds_min_rtn_{u64,i64}
