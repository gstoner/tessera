# 12.5.2. LDS Lane-permute Ops

> RDNA3.5 ISA — pages 139–139

12.5.1. LDS Atomic Ops
Atomic ops combine data from a VGPR with data in LDS, write the result back to LDS memory and optionally
return the "pre-op" value from LDS memory back to a VGPR. When multiple lanes in a wave access the same
LDS location there it is not specified in which order the lanes perform their operations, only that each lane
performs the complete read-modify-write operation before another lane operates on the data.

  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + {InstOffset1,InstOffset0}

VGPR[ADDR] is a byte address. VGPRs 0,1 and dst are double-GPRs for doubles data. VGPR data sources can
only be VGPRs or constant values, not SGPRs. Floating point atomic ops use the MODE register to control
denormal flushing behavior.

LDS & GDS Atomic Opcodes
Instruction Fields: op, gds, offset0, offset1, vdst, addr, data0, data1
32-bit no return               32-bit with return                 64-bit no return       64-bit with return
ds_add_u32                     ds_add_rtn_u32                     ds_add_u64             ds_add_rtn_u64
ds_sub_u32                     ds_sub_rtn_u32                     ds_sub_u64             ds_rsub_rtn_u64
ds_rsub_u32                    ds_rsub_rtn_u32                    ds_rsub_u64            ds_rsub_rtn_u64
ds_inc_u32                     ds_inc_rtn_u32                     ds_inc_u64             ds_inc_rtn_u64
ds_dec_u32                     ds_dec_rtn_u32                     ds_dec_u64             ds_dec_rtn_u64
ds_min_{u32,i32,f32}           ds_min_rtn_{u32,i32,f32}           ds_min_{u64,i64,f64}   ds_min_rtn_{u64,i64,f64}
ds_max_{u32,i32,f32}           ds_max_rtn_{u32,i32,f32}           ds_max_{u64,i64,f64}   ds_max_rtn_{u64,i64,f64}
ds_and_b32                     ds_and_rtn_b32                     ds_and_b64             ds_and_rtn_b64
ds_or_b32                      ds_or_rtn_b32                      ds_or_b64              ds_or_rtn_b64
ds_xor_b32                     ds_xor_rtn_b32                     ds_xor_b64             ds_xor_rtn_b64
ds_mskor_b32                   ds_mskor_rtn_b32                   ds_mskor_b64           ds_mskor_rtn_b64
ds_cmpstore_b32                ds_cmpstore_rtn_b32                ds_cmpstore_b64        ds_cmpstore_rtn_b64
ds_cmpstore_f32                ds_cmpstore_rtn_f32                ds_cmpstore_f64        ds_cmpstore_rtn_f64
ds_add_f32                     ds_add_rtn_f32
                               ds_storexchg_rtn_b32                                      ds_storexchg_rtn_b64
                               ds_storexchg_2addr_rtn_b32                                ds_storexchg_2addr_rtn_b64
                               ds_storexchg_2addr_stride64_rt                            ds_storexchg_2addr_stride64_rt
                               n_b32                                                     n_b64

12.5.2. LDS Lane-permute Ops
DS_PERMUTE instructions allow data to be swizzled arbitrarily across 32 lanes. Two versions of the instruction
are provided: forward (scatter) and backward (gather). These exist in LDS only, not GDS.

Note that in wave64 mode the permute operates only across 32 lanes at a time on each half of a wave64. In
other words, it executes as if were two independent wave32’s. Each half-wave can use indices in the range 0-31
to reference lanes in that same half-wave.

These instructions use the LDS hardware but do not use any memory storage, and may be used by waves that
have not allocated any LDS space. The instructions supply a data value from VGPRs and an index value per
lane.
