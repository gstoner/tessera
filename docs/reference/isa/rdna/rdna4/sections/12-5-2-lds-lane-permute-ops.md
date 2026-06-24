# 12.5.2. LDS Lane-permute Ops

> RDNA4 ISA — pages 163–163

Atomic Opcodes
ds_max_{u32,i32}        ds_max_rtn_{u32,i32}                ds_max_{u64,i64}     ds_max_rtn_{u64,i64}
ds_min_num_f32          ds_min_num_rtn_f32                  ds_min_num_f64       ds_min_num_rtn_f64
ds_max_num_f32          ds_max_num_rtn_f32                  ds_max_num_f64       ds_max_num_rtn_f64
ds_and_b32              ds_and_rtn_b32                      ds_and_b64           ds_and_rtn_b64
ds_or_b32               ds_or_rtn_b32                       ds_or_b64            ds_or_rtn_b64
ds_xor_b32              ds_xor_rtn_b32                      ds_xor_b64           ds_xor_rtn_b64
ds_mskor_b32            ds_mskor_rtn_b32                    ds_mskor_b64         ds_mskor_rtn_b64
ds_cmpstore_b32         ds_cmpstore_rtn_b32                 ds_cmpstore_b64      ds_cmpstore_rtn_b64
ds_add_f32              ds_add_rtn_f32
ds_pk_add_f16           ds_pk_add_rtn_f16
ds_pk_add_bf16          ds_pk_add_rtn_bf16
                        ds_storexchg_rtn_b32                                     ds_storexchg_rtn_b64
                        ds_storexchg_2addr_rtn_b32                               ds_storexchg_2addr_rtn_b64
                        ds_storexchg_2addr_stride64_rtn_b                        ds_storexchg_2addr_stride64_rtn_b
                        32                                                       64
                        ds_condxchg32_rtn_b64
ds_clampsub_u32         ds_clampsub_rtn_u32
ds_condsub_u32          ds_condsub_rtn_u32

12.5.2. LDS Lane-permute Ops
DS_PERMUTE instructions allow data to be swizzled arbitrarily across 32 lanes for wave32, or across all 64
lanes of a wave64. Two versions of the instruction are provided: forward (scatter) and backward (gather).

These instructions use the LDS hardware but do not use any memory storage, and may be used by waves that
have not allocated any LDS space. The instructions supply a data value from VGPRs and an index value per
lane.

  • ds_permute_b32 : Dst[index[0..31]] = src[0..31]    Where [0..31] is the lane number
  • ds_bpermute_b32 : Dst[0..31] = src[index[0..31]]
      ◦ For wave64, replace "31" with "63" above.

The EXEC mask is honored for both reading the source and writing the destination, except for
DS_BPERMUTE_FI_B32 which reads all lanes and EXEC applies only to which lanes to write. Index values out
of range wrap around: for wave32, only index bits [6:2] are used, the other bits of the index are ignored; for
wave64 index bits [7:2] are used. Reading from disabled lanes returns zero.

In the instruction word: VDST is the dest VGPR, ADDR is the index VGPR, and DATA0 is the source data VGPR.
Note that index values are in bytes (so multiply by 4), and have the 'offset0' field added to them before use.
