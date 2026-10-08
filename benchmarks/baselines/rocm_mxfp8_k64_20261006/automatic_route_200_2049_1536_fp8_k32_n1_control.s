
/tmp/tmpdxh5h2q8.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b04: f4002100 f80000d8
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b0c: f4004500 f80000c8
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b14: 320a0081
	s_mov_b32 s8, ttmp7                                        // 000000001b18: be880073
	s_ashr_i32 s9, ttmp7, 31                                   // 000000001b1c: 86099f73
	s_mov_b32 s2, ttmp9                                        // 000000001b20: be820075
	s_lshl_b64 s[12:13], s[8:9], 7                             // 000000001b24: 848c8708
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[14:15], s[0:1], 0x8                           // 000000001b2c: f4002380 f8000008
	s_load_b64 s[16:17], s[0:1], 0x30                          // 000000001b34: f4002400 f8000030
	s_load_b64 s[10:11], s[0:1], 0x58                          // 000000001b3c: f4002280 f8000058
	s_load_b64 s[6:7], s[0:1], 0x80                            // 000000001b44: f4002180 f8000080
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b4c: 86039f75
	v_or_b32_e32 v3, s12, v5                                   // 000000001b50: 38060a0c
	v_dual_mov_b32 v4, s13 :: v_dual_lshlrev_b32 v11, 4, v0    // 000000001b54: ca22000d 040a0084
	s_lshl_b64 s[18:19], s[2:3], 6                             // 000000001b5c: 84928602
	v_dual_mov_b32 v18, 0 :: v_dual_and_b32 v23, 32, v0        // 000000001b60: ca240080 121600a0
	v_add_co_u32 v1, s2, s18, v5                               // 000000001b68: d7000201 02020a12
	s_delay_alu instid0(valu_dep_1)                            // 000000001b70: bf870001
	v_add_co_ci_u32_e64 v2, null, s19, 0, s2                   // 000000001b74: d5207c02 00090013
	v_and_b32_e32 v6, 0x60, v5                                 // 000000001b7c: 360c0aff 00000060
	v_and_b32_e32 v17, 8, v5                                   // 000000001b84: 36220a88
	s_wait_kmcnt 0x0                                           // 000000001b88: bfc70000
	s_add_nc_u64 s[24:25], s[20:21], -1                        // 000000001b8c: a998c114
	s_add_nc_u64 s[8:9], s[22:23], -1                          // 000000001b90: a988c116
	v_cmp_gt_u64_e64 s2, s[24:25], v[3:4]                      // 000000001b94: d45c0002 02020618
	v_cmp_gt_u64_e32 vcc_lo, s[8:9], v[1:2]                    // 000000001b9c: 7cb80208
	v_and_b32_e32 v22, 15, v0                                  // 000000001ba0: 362c008f
	v_cmp_gt_u32_e64 s3, 0x80, v0                              // 000000001ba4: d44c0003 020200ff 00000080
	v_and_b32_e32 v0, 47, v0                                   // 000000001bb0: 360000af
	v_or_b32_e32 v25, 1, v17                                   // 000000001bb4: 38322281
	s_wait_alu depctr_va_sdst(0)                               // 000000001bb8: bf88f19f
	v_cndmask_b32_e64 v3, s24, v3, s2                          // 000000001bbc: d5010003 000a0618
	v_cndmask_b32_e64 v4, s25, v4, s2                          // 000000001bc4: d5010004 000a0819
	v_cndmask_b32_e32 v9, s9, v2, vcc_lo                       // 000000001bcc: 02120409
	v_dual_cndmask_b32 v10, s8, v1 :: v_dual_and_b32 v11, 16, v11// 000000001bd0: ca640208 0a0a1690
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000001bd8: bf870234
	v_mul_lo_u32 v12, v3, s5                                   // 000000001bdc: d72c000c 02000b03
	v_mad_co_u64_u32 v[1:2], null, v3, s4, s[14:15]            // 000000001be4: d6fe7c01 00380903
	v_mul_lo_u32 v13, v4, s4                                   // 000000001bec: d72c000d 02000904
	v_mul_lo_u32 v15, v10, s5                                  // 000000001bf4: d72c000f 02000b0a
	v_mul_lo_u32 v9, v9, s4                                    // 000000001bfc: d72c0009 02000909
	v_mad_co_u64_u32 v[3:4], null, v10, s4, s[16:17]           // 000000001c04: d6fe7c03 0040090a
	v_or_b32_e32 v8, v22, v23                                  // 000000001c0c: 38102f16
	v_mul_u32_u24_e32 v14, 48, v5                              // 000000001c10: 161c0ab0
	v_or_b32_e32 v29, 2, v17                                   // 000000001c14: 383a2282
	v_add_co_u32 v20, vcc_lo, v1, v11                          // 000000001c18: d7006a14 02021701
	v_add3_u32 v2, v13, v2, v12                                // 000000001c20: d6550002 0432050d
	v_mov_b32_e32 v13, s13                                     // 000000001c28: 7e1a020d
	v_add3_u32 v1, v9, v4, v15                                 // 000000001c2c: d6550001 043e0909
	v_or_b32_e32 v24, 16, v8                                   // 000000001c34: 38301090
	v_or_b32_e32 v16, s12, v6                                  // 000000001c38: 38200c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000001c3c: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, 0, v2, vcc_lo               // 000000001c40: d5207c15 01aa0480
	v_or_b32_e32 v2, v6, v22                                   // 000000001c48: 38042d06
	v_add_co_u32 v42, vcc_lo, v3, v11                          // 000000001c4c: d7006a2a 02021703
	s_wait_alu depctr_va_vcc(0)                                // 000000001c54: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, 0, v1, vcc_lo               // 000000001c58: d5207c2b 01aa0280
	s_delay_alu instid0(valu_dep_3)                            // 000000001c60: bf870003
	v_mul_u32_u24_e32 v1, 48, v2                               // 000000001c64: 160204b0
	v_or_b32_e32 v7, 16, v6                                    // 000000001c68: 380e0c90
	v_or_b32_e32 v12, v16, v17                                 // 000000001c6c: 38182310
	s_lshr_b64 s[8:9], s[4:5], 5                               // 000000001c70: 85888504
	v_or_b32_e32 v8, s18, v8                                   // 000000001c74: 38101012
	v_or_b32_e32 v44, v1, v17                                  // 000000001c78: 38582301
	v_mul_u32_u24_e32 v1, 48, v24                              // 000000001c7c: 160230b0
	v_or_b32_e32 v4, v7, v22                                   // 000000001c80: 38082d07
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[12:13]                // 000000001c84: 7ca81814
	v_add_nc_u32_e32 v19, v14, v11                             // 000000001c88: 4a26170e
	v_or_b32_e32 v26, s12, v7                                  // 000000001c8c: 38340e0c
	v_or_b32_e32 v28, v1, v17                                  // 000000001c90: 38382301
	v_mov_b32_e32 v1, s13                                      // 000000001c94: 7e02020d
	v_mul_u32_u24_e32 v0, 48, v0                               // 000000001c98: 160000b0
	v_mul_u32_u24_e32 v2, 48, v4                               // 000000001c9c: 160408b0
	s_wait_alu depctr_va_vcc(0)                                // 000000001ca0: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v12, vcc_lo                       // 000000001ca4: 02061880
	s_lshr_b32 s12, s5, 5                                      // 000000001ca8: 850c8505
	v_mov_b32_e32 v11, s13                                     // 000000001cac: 7e16020d
	v_or_b32_e32 v27, v17, v0                                  // 000000001cb0: 38360111
	v_or_b32_e32 v0, v25, v16                                  // 000000001cb4: 38002119
	v_or_b32_e32 v45, v2, v17                                  // 000000001cb8: 385a2302
	v_dual_cndmask_b32 v2, 0, v13 :: v_dual_mov_b32 v75, 0     // 000000001cbc: ca501a80 024a0080
	v_mov_b32_e32 v40, 0                                       // 000000001cc4: 7e500280
	s_delay_alu instid0(valu_dep_4)                            // 000000001cc8: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001ccc: 7ca80014
	v_dual_mov_b32 v46, 0 :: v_dual_add_nc_u32 v91, 0x1800, v27// 000000001cd0: ca200080 2e5a36ff 00001800
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cdc: bf88ff9e
	v_mul_lo_u32 v10, s8, v2                                   // 000000001ce0: d72c000a 02020408
	v_mov_b32_e32 v69, 0                                       // 000000001ce8: 7e8a0280
	v_mov_b32_e32 v61, 0                                       // 000000001cec: 7e7a0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001cf0: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v0, vcc_lo                        // 000000001cf4: 020e0080
	v_or_b32_e32 v0, v29, v16                                  // 000000001cf8: 3800211d
	v_cndmask_b32_e32 v6, 0, v1, vcc_lo                        // 000000001cfc: 020c0280
	v_mul_lo_u32 v4, s12, v3                                   // 000000001d00: d72c0004 0202060c
	v_mad_co_u64_u32 v[2:3], null, s8, v3, 0                   // 000000001d08: d6fe7c02 02020608
	v_mul_lo_u32 v14, s12, v7                                  // 000000001d10: d72c000e 02020e0c
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001d18: 7ca80014
	v_mul_lo_u32 v15, s8, v6                                   // 000000001d1c: d72c000f 02020c08
	v_mad_co_u64_u32 v[6:7], null, s8, v7, 0                   // 000000001d24: d6fe7c06 02020e08
	v_mov_b32_e32 v39, 0                                       // 000000001d2c: 7e4e0280
	v_mov_b32_e32 v53, 0                                       // 000000001d30: 7e6a0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001d34: bf88ff9d
	v_dual_mov_b32 v49, 0 :: v_dual_cndmask_b32 v32, 0, v0     // 000000001d38: ca120080 31200080
	v_or_b32_e32 v30, 3, v17                                   // 000000001d40: 383c2283
	v_add3_u32 v3, v3, v10, v4                                 // 000000001d44: d6550003 04121503
	v_cndmask_b32_e32 v31, 0, v1, vcc_lo                       // 000000001d4c: 023e0280
	v_add3_u32 v7, v7, v15, v14                                // 000000001d50: d6550007 043a1f07
	v_mul_lo_u32 v14, s12, v32                                 // 000000001d58: d72c000e 0202400c
	v_or_b32_e32 v10, v30, v16                                 // 000000001d60: 3814211e
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001d64: 3e000482
	v_mul_lo_u32 v15, s8, v31                                  // 000000001d68: d72c000f 02023e08
	v_or_b32_e32 v31, 4, v17                                   // 000000001d70: 383e2284
	v_mad_co_u64_u32 v[2:3], null, s8, v32, 0                  // 000000001d74: d6fe7c02 02024008
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001d7c: 7ca81414
	v_mov_b32_e32 v9, s19                                      // 000000001d80: 7e120213
	v_or_b32_e32 v37, 7, v17                                   // 000000001d84: 384a2287
	v_mov_b32_e32 v47, 0                                       // 000000001d88: 7e5e0280
	v_mov_b32_e32 v41, 0                                       // 000000001d8c: 7e520280
	v_mov_b32_e32 v27, 0                                       // 000000001d90: 7e360280
	s_wait_alu depctr_va_vcc(0)                                // 000000001d94: bf88ff9d
	v_cndmask_b32_e32 v33, 0, v10, vcc_lo                      // 000000001d98: 02421480
	v_or_b32_e32 v10, v31, v16                                 // 000000001d9c: 3814211f
	v_cndmask_b32_e32 v32, 0, v11, vcc_lo                      // 000000001da0: 02401680
	v_add_co_u32 v51, vcc_lo, s10, v0                          // 000000001da4: d7006a33 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001dac: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, s11, v1, vcc_lo             // 000000001db0: d5207c34 01aa020b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001db8: 7ca81414
	v_add3_u32 v3, v3, v15, v14                                // 000000001dbc: d6550003 043a1f03
	v_mul_lo_u32 v15, s8, v32                                  // 000000001dc4: d72c000f 02024008
	v_or_b32_e32 v32, 5, v17                                   // 000000001dcc: 38402285
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001dd0: 3e000c82
	v_mul_lo_u32 v14, s12, v33                                 // 000000001dd4: d72c000e 0202420c
	v_mad_co_u64_u32 v[6:7], null, s8, v33, 0                  // 000000001ddc: d6fe7c06 02024208
	s_wait_alu depctr_va_vcc(0)                                // 000000001de4: bf88ff9d
	v_cndmask_b32_e32 v34, 0, v10, vcc_lo                      // 000000001de8: 02441480
	v_or_b32_e32 v10, v32, v16                                 // 000000001dec: 38142120
	v_cndmask_b32_e32 v33, 0, v11, vcc_lo                      // 000000001df0: 02421680
	v_add_co_u32 v54, vcc_lo, s10, v0                          // 000000001df4: d7006a36 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001dfc: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s11, v1, vcc_lo             // 000000001e00: d5207c37 01aa020b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001e08: 7ca81414
	v_add3_u32 v7, v7, v15, v14                                // 000000001e0c: d6550007 043a1f07
	v_mul_lo_u32 v15, s8, v33                                  // 000000001e14: d72c000f 02024208
	v_or_b32_e32 v33, 6, v17                                   // 000000001e1c: 38422286
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001e20: 3e000482
	v_mul_lo_u32 v14, s12, v34                                 // 000000001e24: d72c000e 0202440c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e2c: bf88ff9d
	v_cndmask_b32_e32 v35, 0, v10, vcc_lo                      // 000000001e30: 02461480
	v_mad_co_u64_u32 v[2:3], null, s8, v34, 0                  // 000000001e34: d6fe7c02 02024408
	v_or_b32_e32 v10, v33, v16                                 // 000000001e3c: 38142121
	v_cndmask_b32_e32 v34, 0, v11, vcc_lo                      // 000000001e40: 02441680
	v_add_co_u32 v56, vcc_lo, s10, v0                          // 000000001e44: d7006a38 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e4c: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s11, v1, vcc_lo             // 000000001e50: d5207c39 01aa020b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001e58: 7ca81414
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e5c: 3e000c82
	v_add3_u32 v3, v3, v15, v14                                // 000000001e60: d6550003 043a1f03
	v_mul_lo_u32 v36, s12, v35                                 // 000000001e68: d72c0024 0202460c
	v_mul_lo_u32 v34, s8, v34                                  // 000000001e70: d72c0022 02024408
	s_wait_alu depctr_va_vcc(0)                                // 000000001e78: bf88ff9d
	v_dual_mov_b32 v7, s13 :: v_dual_cndmask_b32 v14, 0, v11   // 000000001e7c: ca12000d 070e1680
	v_dual_cndmask_b32 v15, 0, v10 :: v_dual_add_nc_u32 v92, 0x1800, v28// 000000001e84: ca601480 0f5c38ff 00001800
	v_or_b32_e32 v6, v37, v16                                  // 000000001e90: 380c2125
	v_mad_co_u64_u32 v[10:11], null, s8, v35, 0                // 000000001e94: d6fe7c0a 02024608
	v_add_co_u32 v59, s4, s10, v0                              // 000000001e9c: d700043b 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 000000001ea4: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s11, v1, s4                 // 000000001ea8: d5207c3c 0012020b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001eb0: 7ca80c14
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001eb4: 3e000482
	v_mul_lo_u32 v16, s12, v15                                 // 000000001eb8: d72c0010 02021e0c
	v_mul_lo_u32 v35, s8, v14                                  // 000000001ec0: d72c0023 02021c08
	v_mad_co_u64_u32 v[14:15], null, s8, v15, 0                // 000000001ec8: d6fe7c0e 02021e08
	v_add3_u32 v11, v11, v34, v36                              // 000000001ed0: d655000b 0492450b
	s_wait_alu depctr_va_vcc(0)                                // 000000001ed8: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v7 :: v_dual_mov_b32 v78, 0      // 000000001edc: ca500e80 074e0080
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo                        // 000000001ee4: 020c0c80
	v_add_co_u32 v62, vcc_lo, s10, v0                          // 000000001ee8: d7006a3e 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef0: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s11, v1, vcc_lo             // 000000001ef4: d5207c3f 01aa020b
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]                      // 000000001efc: 3e001482
	v_add3_u32 v15, v15, v35, v16                              // 000000001f00: d655000f 0442470f
	v_dual_mov_b32 v11, s13 :: v_dual_mov_b32 v72, 0           // 000000001f08: ca10000d 0b480080
	v_or_b32_e32 v10, v26, v17                                 // 000000001f10: 3814231a
	v_mov_b32_e32 v66, 0                                       // 000000001f14: 7e840280
	s_delay_alu instid0(valu_dep_4)                            // 000000001f18: bf870004
	v_lshlrev_b64_e32 v[2:3], 2, v[14:15]                      // 000000001f1c: 3e041c82
	v_mul_lo_u32 v14, s12, v6                                  // 000000001f20: d72c000e 02020c0c
	v_mul_lo_u32 v15, s8, v7                                   // 000000001f28: d72c000f 02020e08
	v_mad_co_u64_u32 v[6:7], null, s8, v6, 0                   // 000000001f30: d6fe7c06 02020c08
	v_add_co_u32 v64, vcc_lo, s10, v0                          // 000000001f38: d7006a40 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s11, v1, vcc_lo             // 000000001f44: d5207c41 01aa020b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001f4c: 7ca81414
	v_mov_b32_e32 v1, s13                                      // 000000001f50: 7e02020d
	v_or_b32_e32 v0, v26, v25                                  // 000000001f54: 3800331a
	v_add3_u32 v7, v7, v15, v14                                // 000000001f58: d6550007 043a1f07
	v_add_co_u32 v67, s4, s10, v2                              // 000000001f60: d7000443 0202040a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f68: bf88ff9d
	v_dual_cndmask_b32 v14, 0, v11 :: v_dual_cndmask_b32 v15, 0, v10// 000000001f6c: ca521680 0e0e1480
	v_mov_b32_e32 v58, 0                                       // 000000001f74: 7e740280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001f78: 7ca80014
	s_wait_alu depctr_va_sdst(0)                               // 000000001f7c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s11, v3, s4                 // 000000001f80: d5207c44 0012060b
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001f88: 3e040c82
	v_mul_lo_u32 v16, s12, v15                                 // 000000001f8c: d72c0010 02021e0c
	v_mul_lo_u32 v17, s8, v14                                  // 000000001f94: d72c0011 02021c08
	v_mad_co_u64_u32 v[14:15], null, s8, v15, 0                // 000000001f9c: d6fe7c0e 02021e08
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa4: bf88ff9d
	v_dual_cndmask_b32 v25, 0, v0 :: v_dual_mov_b32 v36, 0     // 000000001fa8: ca500080 19240080
	v_or_b32_e32 v0, v26, v29                                  // 000000001fb0: 38003b1a
	v_or_b32_e32 v6, s18, v24                                  // 000000001fb4: 380c3012
	v_cndmask_b32_e32 v24, 0, v1, vcc_lo                       // 000000001fb8: 02300280
	v_add_co_u32 v70, vcc_lo, s10, v2                          // 000000001fbc: d7006a46 0202040a
	s_delay_alu instid0(valu_dep_4)                            // 000000001fc4: bf870004
	v_cmp_gt_i64_e64 s4, s[20:21], v[0:1]                      // 000000001fc8: d4540004 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000001fd0: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s11, v3, vcc_lo             // 000000001fd4: d5207c47 01aa060b
	v_add3_u32 v15, v15, v17, v16                              // 000000001fdc: d655000f 0442230f
	v_mul_lo_u32 v16, s12, v25                                 // 000000001fe4: d72c0010 0202320c
	v_mul_lo_u32 v17, s8, v24                                  // 000000001fec: d72c0011 02023008
	v_mad_co_u64_u32 v[2:3], null, s8, v25, 0                  // 000000001ff4: d6fe7c02 02023208
	s_wait_alu depctr_va_sdst(0)                               // 000000001ffc: bf88f19f
	v_cndmask_b32_e64 v24, 0, v1, s4                           // 000000002000: d5010018 00120280
	v_cndmask_b32_e64 v25, 0, v0, s4                           // 000000002008: d5010019 00120080
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 000000002010: 3e001c82
	v_mov_b32_e32 v15, s13                                     // 000000002014: 7e1e020d
	v_or_b32_e32 v14, v26, v30                                 // 000000002018: 381c3d1a
	v_mul_lo_u32 v24, s8, v24                                  // 00000000201c: d72c0018 02023008
	v_mul_lo_u32 v29, s12, v25                                 // 000000002024: d72c001d 0202320c
	v_add3_u32 v3, v3, v17, v16                                // 00000000202c: d6550003 04422303
	v_add_co_u32 v73, s5, s10, v0                              // 000000002034: d7000549 0202000a
	v_cmp_gt_i64_e64 s4, s[20:21], v[14:15]                    // 00000000203c: d4540004 02021c14
	s_wait_alu depctr_va_sdst(0)                               // 000000002044: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s11, v1, s5                 // 000000002048: d5207c4a 0016020b
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000002050: 3e000482
	v_dual_mov_b32 v3, s13 :: v_dual_mov_b32 v50, 0            // 000000002054: ca10000d 03320080
	v_or_b32_e32 v2, v26, v31                                  // 00000000205c: 38043f1a
	v_mad_co_u64_u32 v[16:17], null, s8, v25, 0                // 000000002060: d6fe7c10 02023208
	v_cndmask_b32_e64 v15, 0, v15, s4                          // 000000002068: d501000f 00121e80
	v_cndmask_b32_e64 v14, 0, v14, s4                          // 000000002070: d501000e 00121c80
	v_add_co_u32 v76, s5, s10, v0                              // 000000002078: d700054c 0202000a
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 000000002080: d4540004 02020414
	s_delay_alu instid0(valu_dep_4)                            // 000000002088: bf870004
	v_mul_lo_u32 v25, s8, v15                                  // 00000000208c: d72c0019 02021e08
	s_wait_alu depctr_va_sdst(0)                               // 000000002094: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s11, v1, s5                 // 000000002098: d5207c4d 0016020b
	v_add3_u32 v17, v17, v24, v29                              // 0000000020a0: d6550011 04763111
	v_mul_lo_u32 v24, s12, v14                                 // 0000000020a8: d72c0018 02021c0c
	v_mad_co_u64_u32 v[14:15], null, s8, v14, 0                // 0000000020b0: d6fe7c0e 02021c08
	v_cndmask_b32_e64 v30, 0, v2, s4                           // 0000000020b8: d501001e 00120480
	v_or_b32_e32 v2, v26, v32                                  // 0000000020c0: 3804411a
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 0000000020c4: 3e002082
	v_cndmask_b32_e64 v29, 0, v3, s4                           // 0000000020c8: d501001d 00120680
	v_mov_b32_e32 v48, 0                                       // 0000000020d0: 7e600280
	v_mul_lo_u32 v31, s12, v30                                 // 0000000020d4: d72c001f 02023c0c
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 0000000020dc: d4540004 02020414
	v_add3_u32 v15, v15, v25, v24                              // 0000000020e4: d655000f 0462330f
	v_mov_b32_e32 v25, s13                                     // 0000000020ec: 7e32020d
	v_or_b32_e32 v24, v26, v33                                 // 0000000020f0: 3830431a
	v_add_co_u32 v79, s5, s10, v0                              // 0000000020f4: d700054f 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020fc: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s11, v1, s5                 // 000000002100: d5207c50 0016020b
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 000000002108: 3e001c82
	v_cndmask_b32_e64 v15, 0, v2, s4                           // 00000000210c: d501000f 00120480
	v_or_b32_e32 v2, v26, v37                                  // 000000002114: 38044b1a
	v_mul_lo_u32 v29, s8, v29                                  // 000000002118: d72c001d 02023a08
	v_mad_co_u64_u32 v[16:17], null, s8, v30, 0                // 000000002120: d6fe7c10 02023c08
	v_cmp_gt_i64_e64 s5, s[20:21], v[24:25]                    // 000000002128: d4540005 02023014
	v_cndmask_b32_e64 v14, 0, v3, s4                           // 000000002130: d501000e 00120680
	v_cmp_gt_i64_e64 s4, s[20:21], v[2:3]                      // 000000002138: d4540004 02020414
	v_mul_lo_u32 v26, s12, v15                                 // 000000002140: d72c001a 02021e0c
	v_dual_mov_b32 v7, s19 :: v_dual_mov_b32 v38, 0            // 000000002148: ca100013 07260080
	s_wait_alu depctr_va_sdst(0)                               // 000000002150: bf88f19f
	v_cndmask_b32_e64 v25, 0, v25, s5                          // 000000002154: d5010019 00163280
	v_cndmask_b32_e64 v24, 0, v24, s5                          // 00000000215c: d5010018 00163080
	v_add3_u32 v17, v17, v29, v31                              // 000000002164: d6550011 047e3b11
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 00000000216c: d5010003 00120680
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000002174: d5010002 00120480
	v_mul_lo_u32 v29, s8, v14                                  // 00000000217c: d72c001d 02021c08
	v_mad_co_u64_u32 v[14:15], null, s8, v15, 0                // 000000002184: d6fe7c0e 02021e08
	v_mul_lo_u32 v30, s12, v24                                 // 00000000218c: d72c001e 0202300c
	v_mul_lo_u32 v31, s8, v25                                  // 000000002194: d72c001f 02023208
	v_mad_co_u64_u32 v[24:25], null, s8, v24, 0                // 00000000219c: d6fe7c18 02023008
	v_add_co_u32 v81, s4, s10, v0                              // 0000000021a4: d7000451 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021ac: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s11, v1, s4                 // 0000000021b0: d5207c52 0012020b
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 0000000021b8: 3e002082
	v_mul_lo_u32 v16, s12, v2                                  // 0000000021bc: d72c0010 0202040c
	v_mul_lo_u32 v17, s8, v3                                   // 0000000021c4: d72c0011 02020608
	v_mad_co_u64_u32 v[2:3], null, s8, v2, 0                   // 0000000021cc: d6fe7c02 02020408
	v_add3_u32 v15, v15, v29, v26                              // 0000000021d4: d655000f 046a3b0f
	v_add3_u32 v25, v25, v31, v30                              // 0000000021dc: d6550019 047a3f19
	v_cmp_gt_i64_e64 s2, s[22:23], v[8:9]                      // 0000000021e4: d4540002 02021016
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[6:7]                  // 0000000021ec: 7ca80c16
	v_add_co_u32 v83, s4, s10, v0                              // 0000000021f0: d7000453 0202000a
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]                    // 0000000021f8: 3e1c1c82
	v_add3_u32 v3, v3, v17, v16                                // 0000000021fc: d6550003 04422303
	s_wait_alu depctr_va_sdst(0)                               // 000000002204: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s11, v1, s4                 // 000000002208: d5207c54 0012020b
	v_lshlrev_b64_e32 v[0:1], 2, v[24:25]                      // 000000002210: 3e003082
	v_cndmask_b32_e64 v5, 0, v9, s2                            // 000000002214: d5010005 000a1280
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 00000000221c: 3e040482
	v_cndmask_b32_e64 v4, 0, v8, s2                            // 000000002220: d5010004 000a1080
	s_wait_alu depctr_va_vcc(0)                                // 000000002228: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v7 :: v_dual_mov_b32 v34, 0      // 00000000222c: ca500e80 07220080
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo                        // 000000002234: 020c0c80
	v_add_co_u32 v85, s4, s10, v14                             // 000000002238: d7000455 02021c0a
	s_wait_alu depctr_va_sdst(0)                               // 000000002240: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s11, v15, s4                // 000000002244: d5207c56 00121e0b
	v_add_co_u32 v87, s4, s10, v0                              // 00000000224c: d7000457 0202000a
	s_wait_alu depctr_va_sdst(0)                               // 000000002254: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s11, v1, s4                 // 000000002258: d5207c58 0012020b
	v_add_co_u32 v89, s4, s10, v2                              // 000000002260: d7000459 0202040a
	v_lshlrev_b64_e32 v[14:15], 2, v[4:5]                      // 000000002268: 3e1c0882
	v_lshlrev_b64_e32 v[16:17], 2, v[6:7]                      // 00000000226c: 3e200c82
	s_wait_alu depctr_va_sdst(0)                               // 000000002270: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s11, v3, s4                 // 000000002274: d5207c5a 0012060b
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v24, 0             // 00000000227c: ca100080 25180080
	v_mov_b32_e32 v35, 0                                       // 000000002284: 7e460280
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0             // 000000002288: ca100080 21200080
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0             // 000000002290: ca100080 1f1e0080
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0             // 000000002298: ca100080 1d1c0080
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0             // 0000000022a0: ca100080 1a180080
	s_mov_b64 s[10:11], 0                                      // 0000000022a8: be8a0180
	s_branch 302                                               // 0000000022ac: bfa0012e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xc68>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000022b4: 8c7e047e
	s_mul_u64 s[14:15], s[10:11], s[22:23]                     // 0000000022b8: aa8e160a
	s_lshl_b64 s[12:13], s[10:11], 2                           // 0000000022bc: 848c820a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022c0: bf88ff9e
	s_lshl_b64 s[14:15], s[14:15], 2                           // 0000000022c4: 848e820e
	v_add_co_u32 v0, s4, v51, s12                              // 0000000022c8: d7000400 02001933
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	s_add_nc_u64 s[14:15], s[6:7], s[14:15]                    // 0000000022d4: a98e0e06
	v_add_co_ci_u32_e64 v1, null, s13, v52, s4                 // 0000000022d8: d5207c01 0012680d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022e0: bf88ff9e
	v_add_co_u32 v2, s4, s14, v14                              // 0000000022e4: d7000402 02021c0e
	s_wait_alu depctr_va_sdst(0)                               // 0000000022ec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v15, s4                 // 0000000022f0: d5207c03 00121e0f
	v_add_co_u32 v4, s4, v54, s12                              // 0000000022f8: d7000404 02001936
	s_wait_alu depctr_va_sdst(0)                               // 000000002300: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v55, s4                 // 000000002304: d5207c05 00126e0d
	s_wait_dscnt 0x0                                           // 00000000230c: bfc60000
	s_barrier_signal -1                                        // 000000002310: be804ec1
	s_barrier_wait 0xffff                                      // 000000002314: bf94ffff
	global_load_b32 v131, v[0:1], off                          // 000000002318: ee05007c 00000083 00000000
	v_add_co_u32 v0, s4, v56, s12                              // 000000002324: d7000400 02001938
	global_load_b32 v132, v[2:3], off                          // 00000000232c: ee05007c 00000084 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002338: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v57, s4                 // 00000000233c: d5207c01 0012720d
	v_add_co_u32 v2, s4, v59, s12                              // 000000002344: d7000402 0200193b
	global_load_b32 v133, v[4:5], off                          // 00000000234c: ee05007c 00000085 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002358: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v60, s4                 // 00000000235c: d5207c03 0012780d
	v_add_co_u32 v4, s4, v62, s12                              // 000000002364: d7000404 0200193e
	s_wait_alu depctr_va_sdst(0)                               // 00000000236c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v63, s4                 // 000000002370: d5207c05 00127e0d
	v_add_co_u32 v6, s4, v64, s12                              // 000000002378: d7000406 02001940
	s_wait_alu depctr_va_sdst(0)                               // 000000002380: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v65, s4                 // 000000002384: d5207c07 0012820d
	v_add_co_u32 v93, s4, v67, s12                             // 00000000238c: d700045d 02001943
	s_wait_alu depctr_va_sdst(0)                               // 000000002394: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v68, s4                // 000000002398: d5207c5e 0012880d
	global_load_b32 v134, v[0:1], off                          // 0000000023a0: ee05007c 00000086 00000000
	v_add_co_u32 v0, s4, v70, s12                              // 0000000023ac: d7000400 02001946
	global_load_b32 v135, v[2:3], off                          // 0000000023b4: ee05007c 00000087 00000002
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v71, s4                 // 0000000023c4: d5207c01 00128e0d
	v_add_co_u32 v2, s4, s14, v16                              // 0000000023cc: d7000402 0202200e
	global_load_b32 v136, v[4:5], off                          // 0000000023d4: ee05007c 00000088 00000004
	s_wait_alu depctr_va_sdst(0)                               // 0000000023e0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v17, s4                 // 0000000023e4: d5207c03 0012220f
	v_add_co_u32 v4, s4, v73, s12                              // 0000000023ec: d7000404 02001949
	global_load_b32 v137, v[6:7], off                          // 0000000023f4: ee05007c 00000089 00000006
	s_wait_alu depctr_va_sdst(0)                               // 000000002400: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v74, s4                 // 000000002404: d5207c05 0012940d
	v_add_co_u32 v6, s4, v76, s12                              // 00000000240c: d7000406 0200194c
	global_load_b32 v138, v[93:94], off                        // 000000002414: ee05007c 0000008a 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 000000002420: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v77, s4                 // 000000002424: d5207c07 00129a0d
	v_add_co_u32 v93, s4, v79, s12                             // 00000000242c: d700045d 0200194f
	s_wait_alu depctr_va_sdst(0)                               // 000000002434: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v80, s4                // 000000002438: d5207c5e 0012a00d
	global_load_b32 v139, v[0:1], off                          // 000000002440: ee05007c 0000008b 00000000
	v_add_co_u32 v0, s4, v81, s12                              // 00000000244c: d7000400 02001951
	global_load_b32 v140, v[2:3], off                          // 000000002454: ee05007c 0000008c 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002460: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v82, s4                 // 000000002464: d5207c01 0012a40d
	v_add_co_u32 v2, s4, v83, s12                              // 00000000246c: d7000402 02001953
	global_load_b32 v141, v[4:5], off                          // 000000002474: ee05007c 0000008d 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002480: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v84, s4                 // 000000002484: d5207c03 0012a80d
	v_add_co_u32 v4, s4, v85, s12                              // 00000000248c: d7000404 02001955
	global_load_b32 v142, v[6:7], off                          // 000000002494: ee05007c 0000008e 00000006
	s_wait_alu depctr_va_sdst(0)                               // 0000000024a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v86, s4                 // 0000000024a4: d5207c05 0012ac0d
	v_add_co_u32 v6, s4, v87, s12                              // 0000000024ac: d7000406 02001957
	global_load_b32 v143, v[93:94], off                        // 0000000024b4: ee05007c 0000008f 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v88, s4                 // 0000000024c4: d5207c07 0012b00d
	v_add_co_u32 v93, s4, v89, s12                             // 0000000024cc: d700045d 02001959
	s_wait_alu depctr_va_sdst(0)                               // 0000000024d4: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s13, v90, s4                // 0000000024d8: d5207c5e 0012b40d
	s_clause 0x4                                               // 0000000024e0: bf850004
	global_load_b32 v144, v[0:1], off                          // 0000000024e4: ee05007c 00000090 00000000
	global_load_b32 v145, v[2:3], off                          // 0000000024f0: ee05007c 00000091 00000002
	global_load_b32 v146, v[4:5], off                          // 0000000024fc: ee05007c 00000092 00000004
	global_load_b32 v147, v[6:7], off                          // 000000002508: ee05007c 00000093 00000006
	global_load_b32 v148, v[93:94], off                        // 000000002514: ee05007c 00000094 0000005d
	ds_load_2addr_b64 v[115:118], v44 offset1:2                // 000000002520: d9dc0200 7300002c
	ds_load_2addr_b64 v[119:122], v91 offset1:2                // 000000002528: d9dc0200 7700005b
	ds_load_2addr_b64 v[123:126], v92 offset1:2                // 000000002530: d9dc0200 7b00005c
	ds_load_2addr_b64 v[127:130], v45 offset1:2                // 000000002538: d9dc0200 7f00002d
	s_add_nc_u64 s[10:11], s[10:11], 1                         // 000000002540: a98a810a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002544: bf88ff9e
	s_cmp_lg_u64 s[10:11], s[8:9]                              // 000000002548: bf11080a
	s_wait_dscnt 0x2                                           // 00000000254c: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[119:120], 0// 000000002550: cc464000 1a02ef73
	s_wait_dscnt 0x1                                           // 000000002558: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[115:116], v[123:124], 0// 00000000255c: cc46405d 1a02f773
	s_wait_dscnt 0x0                                           // 000000002564: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[127:128], v[119:120], 0// 000000002568: cc464065 1a02ef7f
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[127:128], v[123:124], 0// 000000002570: cc46406d 1a02f77f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[117:118], v[121:122], v[0:7]// 000000002578: cc464000 1c02f375
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[117:118], v[125:126], v[93:100]// 000000002580: cc46405d 1d76fb75
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002588: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[129:130], v[121:122], v[101:108]// 00000000258c: cc464065 1d96f381
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[129:130], v[125:126], v[109:116]// 000000002594: cc46406d 1db6fb81
	s_wait_loadcnt 0xf                                         // 00000000259c: bfc0000f
	v_dual_mul_f32 v117, v131, v132 :: v_dual_mul_f32 v118, v132, v133// 0000000025a0: c8c70983 75770b84
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000025a8: bf870091
	v_dual_mul_f32 v0, v0, v117 :: v_dual_mul_f32 v1, v1, v118 // 0000000025ac: c8c6eb00 0000ed01
	v_add_f32_e32 v18, v18, v0                                 // 0000000025b4: 06240112
	s_wait_loadcnt 0xe                                         // 0000000025b8: bfc0000e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 0000000025bc: bf870142
	v_dual_add_f32 v78, v78, v1 :: v_dual_mul_f32 v119, v132, v134// 0000000025c0: c906034e 4e770d84
	s_wait_loadcnt 0xd                                         // 0000000025c8: bfc0000d
	v_mul_f32_e32 v120, v132, v135                             // 0000000025cc: 10f10f84
	s_wait_loadcnt 0xc                                         // 0000000025d0: bfc0000c
	v_dual_mul_f32 v2, v2, v119 :: v_dual_mul_f32 v121, v132, v136// 0000000025d4: c8c6ef02 02791184
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025dc: bf870122
	v_mul_f32_e32 v3, v3, v120                                 // 0000000025e0: 1006f103
	s_wait_loadcnt 0xb                                         // 0000000025e4: bfc0000b
	v_dual_add_f32 v75, v75, v2 :: v_dual_mul_f32 v122, v132, v137// 0000000025e8: c906054b 4b7b1384
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000025f0: bf870193
	v_mul_f32_e32 v4, v4, v121                                 // 0000000025f4: 1008f304
	v_add_f32_e32 v72, v72, v3                                 // 0000000025f8: 06900748
	s_wait_loadcnt 0xa                                         // 0000000025fc: bfc0000a
	v_mul_f32_e32 v123, v132, v138                             // 000000002600: 10f71584
	v_mul_f32_e32 v5, v5, v122                                 // 000000002604: 100af505
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002608: bf870112
	v_dual_add_f32 v69, v69, v4 :: v_dual_mul_f32 v6, v6, v123 // 00000000260c: c9060945 4506f706
	v_add_f32_e32 v66, v66, v5                                 // 000000002614: 06840b42
	s_wait_loadcnt 0x8                                         // 000000002618: bfc00008
	v_dual_mul_f32 v124, v132, v139 :: v_dual_mul_f32 v125, v131, v140// 00000000261c: c8c71784 7c7d1983
	v_dual_mul_f32 v126, v133, v140 :: v_dual_mul_f32 v127, v134, v140// 000000002624: c8c71985 7e7f1986
	v_dual_mul_f32 v128, v135, v140 :: v_dual_mul_f32 v129, v136, v140// 00000000262c: c8c71987 80811988
	v_dual_mul_f32 v130, v137, v140 :: v_dual_mul_f32 v131, v138, v140// 000000002634: c8c71989 8283198a
	s_wait_loadcnt 0x7                                         // 00000000263c: bfc00007
	v_dual_mul_f32 v133, v139, v140 :: v_dual_mul_f32 v134, v132, v141// 000000002640: c8c7198b 85871b84
	v_mul_f32_e32 v141, v140, v141                             // 000000002648: 111b1b8c
	s_wait_loadcnt 0x6                                         // 00000000264c: bfc00006
	v_mul_f32_e32 v135, v132, v142                             // 000000002650: 110f1d84
	v_dual_mul_f32 v142, v140, v142 :: v_dual_mul_f32 v7, v7, v124// 000000002654: c8c71d8c 8e06f907
	v_dual_mul_f32 v93, v93, v125 :: v_dual_mul_f32 v94, v94, v126// 00000000265c: c8c6fb5d 5d5efd5e
	s_wait_loadcnt 0x5                                         // 000000002664: bfc00005
	v_mul_f32_e32 v136, v132, v143                             // 000000002668: 11111f84
	v_mul_f32_e32 v143, v140, v143                             // 00000000266c: 111f1f8c
	v_dual_mul_f32 v95, v95, v127 :: v_dual_mul_f32 v96, v96, v128// 000000002670: c8c6ff5f 5f610160
	v_dual_mul_f32 v97, v97, v129 :: v_dual_mul_f32 v98, v98, v130// 000000002678: c8c70361 61630562
	v_mul_f32_e32 v99, v99, v131                               // 000000002680: 10c70763
	s_wait_loadcnt 0x3                                         // 000000002684: bfc00003
	v_dual_mul_f32 v137, v132, v144 :: v_dual_mul_f32 v138, v132, v145// 000000002688: c8c72184 898b2384
	s_wait_loadcnt 0x2                                         // 000000002690: bfc00002
	v_mul_f32_e32 v139, v132, v146                             // 000000002694: 11172584
	s_wait_loadcnt 0x0                                         // 000000002698: bfc00000
	v_dual_mul_f32 v149, v132, v147 :: v_dual_mul_f32 v132, v132, v148// 00000000269c: c8c72784 95852984
	v_dual_mul_f32 v144, v140, v144 :: v_dual_mul_f32 v145, v140, v145// 0000000026a4: c8c7218c 9091238c
	v_dual_mul_f32 v146, v140, v146 :: v_dual_mul_f32 v147, v140, v147// 0000000026ac: c8c7258c 9293278c
	v_mul_f32_e32 v140, v140, v148                             // 0000000026b4: 1119298c
	v_dual_mul_f32 v100, v100, v133 :: v_dual_mul_f32 v101, v101, v134// 0000000026b8: c8c70b64 64650d65
	v_dual_mul_f32 v102, v102, v135 :: v_dual_mul_f32 v103, v103, v136// 0000000026c0: c8c70f66 66671167
	v_dual_mul_f32 v104, v104, v137 :: v_dual_mul_f32 v105, v105, v138// 0000000026c8: c8c71368 68691569
	v_dual_mul_f32 v106, v106, v139 :: v_dual_mul_f32 v107, v107, v149// 0000000026d0: c8c7176a 6a6b2b6b
	v_dual_mul_f32 v108, v108, v132 :: v_dual_mul_f32 v109, v109, v141// 0000000026d8: c8c7096c 6c6d1b6d
	v_dual_mul_f32 v110, v110, v142 :: v_dual_mul_f32 v111, v111, v143// 0000000026e0: c8c71d6e 6e6f1f6f
	v_dual_mul_f32 v112, v112, v144 :: v_dual_mul_f32 v113, v113, v145// 0000000026e8: c8c72170 70712371
	v_dual_mul_f32 v114, v114, v146 :: v_dual_mul_f32 v115, v115, v147// 0000000026f0: c8c72572 72732773
	v_dual_mul_f32 v116, v116, v140 :: v_dual_add_f32 v61, v61, v6// 0000000026f8: c8c91974 743c0d3d
	v_dual_add_f32 v58, v58, v7 :: v_dual_add_f32 v39, v39, v93// 000000002700: c9080f3a 3a26bb27
	v_dual_add_f32 v38, v38, v94 :: v_dual_add_f32 v37, v37, v95// 000000002708: c908bd26 2624bf25
	v_dual_add_f32 v36, v36, v96 :: v_dual_add_f32 v35, v35, v97// 000000002710: c908c124 2422c323
	v_dual_add_f32 v34, v34, v98 :: v_dual_add_f32 v33, v33, v99// 000000002718: c908c522 2220c721
	v_dual_add_f32 v32, v32, v100 :: v_dual_add_f32 v53, v53, v101// 000000002720: c908c920 2034cb35
	v_dual_add_f32 v50, v50, v102 :: v_dual_add_f32 v49, v49, v103// 000000002728: c908cd32 3230cf31
	v_dual_add_f32 v48, v48, v104 :: v_dual_add_f32 v47, v47, v105// 000000002730: c908d130 302ed32f
	v_dual_add_f32 v46, v46, v106 :: v_dual_add_f32 v41, v41, v107// 000000002738: c908d52e 2e28d729
	v_dual_add_f32 v40, v40, v108 :: v_dual_add_f32 v31, v31, v109// 000000002740: c908d928 281edb1f
	v_dual_add_f32 v30, v30, v110 :: v_dual_add_f32 v29, v29, v111// 000000002748: c908dd1e 1e1cdf1d
	v_dual_add_f32 v28, v28, v112 :: v_dual_add_f32 v27, v27, v113// 000000002750: c908e11c 1c1ae31b
	v_dual_add_f32 v26, v26, v114 :: v_dual_add_f32 v25, v25, v115// 000000002758: c908e51a 1a18e719
	v_add_f32_e32 v24, v24, v116                               // 000000002760: 0630e918
	s_cbranch_scc0 37                                          // 000000002764: bfa10025 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xcfc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002768: bf88ff9e
	s_lshl_b64 s[12:13], s[10:11], 5                           // 00000000276c: 848c850a
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0               // 000000002770: ca100080 02020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002778: bf88ff9e
	v_add_co_u32 v0, s4, v20, s12                              // 00000000277c: d7000400 02001914
	s_wait_alu depctr_va_sdst(0)                               // 000000002784: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v21, s4                 // 000000002788: d5207c01 00122a0d
	global_load_b128 v[4:7], v[0:1], off                       // 000000002790: ee05c07c 00000004 00000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 00000000279c: ca100080 00000080
	s_and_saveexec_b32 s5, s3                                  // 0000000027a4: be852003
	s_cbranch_execz 8                                          // 0000000027a8: bfa50008 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xccc>
	v_add_co_u32 v0, s4, v42, s12                              // 0000000027ac: d7000400 0200192a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v43, s4                 // 0000000027b8: d5207c01 0012560d
	global_load_b128 v[0:3], v[0:1], off                       // 0000000027c0: ee05c07c 00000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000027d0: 8c7e057e
	s_barrier_signal -1                                        // 0000000027d4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000027d8: bf94ffff
	s_wait_loadcnt 0x0                                         // 0000000027dc: bfc00000
	ds_store_b128 v19, v[4:7]                                  // 0000000027e0: db7c0000 00000413
	s_and_saveexec_b32 s4, s3                                  // 0000000027e8: be842003
	s_cbranch_execz 65200                                      // 0000000027ec: bfa5feb0 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x7b0>
	ds_store_b128 v19, v[0:3] offset:6144                      // 0000000027f0: db7c1800 00000013
	s_branch 65197                                             // 0000000027f8: bfa0fead <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x7b0>
	s_load_b64 s[24:25], s[0:1], 0xa8                          // 0000000027fc: f4002600 f80000a8
	v_mul_lo_u32 v4, s23, v12                                  // 000000002804: d72c0004 02021817
	v_mul_lo_u32 v5, s22, v13                                  // 00000000280c: d72c0005 02021a16
	v_mad_co_u64_u32 v[2:3], null, s22, v12, 0                 // 000000002814: d6fe7c02 02021816
	v_sub_co_u32 v0, s0, s20, v12                              // 00000000281c: d7010000 02021814
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002824: bf870191
	v_sub_co_ci_u32_e64 v1, null, s21, v13, s0                 // 000000002828: d5217c01 00021a15
	v_add3_u32 v3, v3, v5, v4                                  // 000000002830: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002838: bf8701a2
	v_cmp_lt_i64_e64 s15, 0, v[0:1]                            // 00000000283c: d451000f 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 000000002844: 3e081081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002848: 3e040481
	s_and_b32 s0, s15, s2                                      // 00000000284c: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002850: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002854: be812000
	s_cbranch_execz 28                                         // 000000002858: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xdcc>
	v_bfe_u32 v6, v18, 16, 1                                   // 00000000285c: d6100006 02052112
	s_wait_kmcnt 0x0                                           // 000000002864: bfc70000
	v_add_co_u32 v7, s0, s24, v2                               // 000000002868: d7000007 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002870: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s25, v3, s0                 // 000000002874: d5207c0c 00020619
	v_add3_u32 v13, v6, v18, 0x7fff                            // 00000000287c: d655000d 03fe2506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002888: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 00000000288c: d7000006 02020907
	v_or_b32_e32 v14, 0x400000, v18                            // 000000002894: 381c24ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000289c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v12, v5, s0                  // 0000000028a0: d5207c07 00020b0c
	v_cmp_u_f32_e64 s0, v18, v18                               // 0000000028a8: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 0000000028b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000028b4: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 0000000028b8: d501000c 00021d0d
	global_store_d16_hi_b16 v[6:7], v12, off                   // 0000000028c0: ee09407c 06000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000028d0: 8c7e017e
	v_add_co_u32 v6, s0, s22, v8                               // 0000000028d4: d7000006 02021016
	s_wait_alu depctr_va_sdst(0)                               // 0000000028dc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s23, v9, s0                  // 0000000028e0: d5207c07 00021217
	v_cmp_lt_i64_e64 s16, 1, v[0:1]                            // 0000000028e8: d4510010 02020081
	s_delay_alu instid0(valu_dep_2)                            // 0000000028f0: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 0000000028f4: 3e0c0c81
	s_and_b32 s0, s16, s2                                      // 0000000028f8: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028fc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002900: be812000
	s_cbranch_execz 28                                         // 000000002904: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xe78>
	v_bfe_u32 v12, v78, 16, 1                                  // 000000002908: d610000c 0205214e
	s_wait_kmcnt 0x0                                           // 000000002910: bfc70000
	v_add_co_u32 v13, s0, s24, v2                              // 000000002914: d700000d 02020418
	s_wait_alu depctr_va_sdst(0)                               // 00000000291c: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s25, v3, s0                 // 000000002920: d5207c0e 00020619
	v_add3_u32 v15, v12, v78, 0x7fff                           // 000000002928: d655000f 03fe9d0c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002934: bf870003
	v_add_co_u32 v12, s0, v13, v6                              // 000000002938: d700000c 02020d0d
	v_or_b32_e32 v16, 0x400000, v78                            // 000000002940: 38209cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002948: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v7, s0                 // 00000000294c: d5207c0d 00020f0e
	v_cmp_u_f32_e64 s0, v78, v78                               // 000000002954: d4180000 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 00000000295c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002960: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002964: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 00000000296c: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002978: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000297c: 8c7e017e
	s_lshl_b64 s[38:39], s[22:23], 1                           // 000000002980: 84a68116
	v_cmp_lt_i64_e64 s14, 2, v[0:1]                            // 000000002984: d451000e 02020082
	v_add_co_u32 v12, s0, s38, v8                              // 00000000298c: d700000c 02021026
	s_wait_alu depctr_va_sdst(0)                               // 000000002994: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v9, s0                 // 000000002998: d5207c0d 00021227
	s_and_b32 s0, s14, s2                                      // 0000000029a0: 8b00020e
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 0000000029a4: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029a8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000029ac: be812000
	s_cbranch_execz 28                                         // 0000000029b0: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xf24>
	v_bfe_u32 v14, v75, 16, 1                                  // 0000000029b4: d610000e 0205214b
	s_wait_kmcnt 0x0                                           // 0000000029bc: bfc70000
	v_add_co_u32 v15, s0, s24, v2                              // 0000000029c0: d700000f 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000029c8: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s25, v3, s0                 // 0000000029cc: d5207c10 00020619
	v_add3_u32 v17, v14, v75, 0x7fff                           // 0000000029d4: d6550011 03fe970e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000029e0: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 0000000029e4: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v75                            // 0000000029ec: 382496ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f4: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 0000000029f8: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v75, v75                               // 000000002a00: d4180000 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 000000002a08: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002a0c: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002a10: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002a18: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002a28: 8c7e017e
	s_mul_u64 s[36:37], s[22:23], 3                            // 000000002a2c: aaa48316
	v_cmp_lt_i64_e64 s13, 3, v[0:1]                            // 000000002a30: d451000d 02020083
	v_add_co_u32 v14, s0, s36, v8                              // 000000002a38: d700000e 02021024
	s_wait_alu depctr_va_sdst(0)                               // 000000002a40: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v9, s0                 // 000000002a44: d5207c0f 00021225
	s_and_b32 s0, s13, s2                                      // 000000002a4c: 8b00020d
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002a50: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a54: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002a58: be812000
	s_cbranch_execz 28                                         // 000000002a5c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0xfd0>
	v_bfe_u32 v16, v72, 16, 1                                  // 000000002a60: d6100010 02052148
	s_wait_kmcnt 0x0                                           // 000000002a68: bfc70000
	v_add_co_u32 v17, s0, s24, v2                              // 000000002a6c: d7000011 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002a74: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s25, v3, s0                 // 000000002a78: d5207c12 00020619
	v_add3_u32 v19, v16, v72, 0x7fff                           // 000000002a80: d6550013 03fe9110 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002a8c: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002a90: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v72                            // 000000002a98: 382890ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002aa0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002aa4: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v72, v72                               // 000000002aac: d4180000 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ab8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000002abc: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002ac4: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ad4: 8c7e017e
	s_lshl_b64 s[34:35], s[22:23], 2                           // 000000002ad8: 84a28216
	v_cmp_lt_i64_e64 s12, 4, v[0:1]                            // 000000002adc: d451000c 02020084
	v_add_co_u32 v16, s0, s34, v8                              // 000000002ae4: d7000010 02021022
	s_wait_alu depctr_va_sdst(0)                               // 000000002aec: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v9, s0                 // 000000002af0: d5207c11 00021223
	s_and_b32 s0, s12, s2                                      // 000000002af8: 8b00020c
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002afc: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b00: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002b04: be812000
	s_cbranch_execz 28                                         // 000000002b08: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x107c>
	v_bfe_u32 v18, v69, 16, 1                                  // 000000002b0c: d6100012 02052145
	s_wait_kmcnt 0x0                                           // 000000002b14: bfc70000
	v_add_co_u32 v19, s0, s24, v2                              // 000000002b18: d7000013 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002b20: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s25, v3, s0                 // 000000002b24: d5207c14 00020619
	v_add3_u32 v21, v18, v69, 0x7fff                           // 000000002b2c: d6550015 03fe8b12 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b38: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002b3c: d7000012 02022113
	v_or_b32_e32 v42, 0x400000, v69                            // 000000002b44: 38548aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b4c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 000000002b50: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v69, v69                               // 000000002b58: d4180000 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000002b60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b64: bf870001
	v_cndmask_b32_e64 v20, v21, v42, s0                        // 000000002b68: d5010014 00025515
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002b70: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002b80: 8c7e017e
	s_mul_u64 s[30:31], s[22:23], 5                            // 000000002b84: aa9e8516
	v_cmp_lt_i64_e64 s11, 5, v[0:1]                            // 000000002b88: d451000b 02020085
	v_add_co_u32 v18, s0, s30, v8                              // 000000002b90: d7000012 0202101e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b98: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s31, v9, s0                 // 000000002b9c: d5207c13 0002121f
	s_and_b32 s0, s11, s2                                      // 000000002ba4: 8b00020b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002ba8: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bac: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002bb0: be812000
	s_cbranch_execz 28                                         // 000000002bb4: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1128>
	v_bfe_u32 v20, v66, 16, 1                                  // 000000002bb8: d6100014 02052142
	s_wait_kmcnt 0x0                                           // 000000002bc0: bfc70000
	v_add_co_u32 v21, s0, s24, v2                              // 000000002bc4: d7000015 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002bcc: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v3, s0                 // 000000002bd0: d5207c2a 00020619
	v_add3_u32 v43, v20, v66, 0x7fff                           // 000000002bd8: d655002b 03fe8514 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002be4: bf870003
	v_add_co_u32 v20, s0, v21, v18                             // 000000002be8: d7000014 02022515
	v_or_b32_e32 v44, 0x400000, v66                            // 000000002bf0: 385884ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002bf8: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v42, v19, s0                // 000000002bfc: d5207c15 0002272a
	v_cmp_u_f32_e64 s0, v66, v66                               // 000000002c04: d4180000 02028542
	s_wait_alu depctr_va_sdst(0)                               // 000000002c0c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c10: bf870001
	v_cndmask_b32_e64 v42, v43, v44, s0                        // 000000002c14: d501002a 0002592b
	global_store_d16_hi_b16 v[20:21], v42, off                 // 000000002c1c: ee09407c 15000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c2c: 8c7e017e
	s_mul_u64 s[28:29], s[22:23], 6                            // 000000002c30: aa9c8616
	v_cmp_lt_i64_e64 s9, 6, v[0:1]                             // 000000002c34: d4510009 02020086
	v_add_co_u32 v20, s0, s28, v8                              // 000000002c3c: d7000014 0202101c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c44: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s29, v9, s0                 // 000000002c48: d5207c15 0002121d
	s_and_b32 s0, s9, s2                                       // 000000002c50: 8b000209
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000002c54: 3e282881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c58: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c5c: be812000
	s_cbranch_execz 28                                         // 000000002c60: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x11d4>
	v_bfe_u32 v42, v61, 16, 1                                  // 000000002c64: d610002a 0205213d
	s_wait_kmcnt 0x0                                           // 000000002c6c: bfc70000
	v_add_co_u32 v43, s0, s24, v2                              // 000000002c70: d700002b 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002c78: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s25, v3, s0                 // 000000002c7c: d5207c2c 00020619
	v_add3_u32 v45, v42, v61, 0x7fff                           // 000000002c84: d655002d 03fe7b2a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c90: bf870003
	v_add_co_u32 v42, s0, v43, v20                             // 000000002c94: d700002a 0202292b
	v_or_b32_e32 v51, 0x400000, v61                            // 000000002c9c: 38667aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca4: bf88f19f
	v_add_co_ci_u32_e64 v43, null, v44, v21, s0                // 000000002ca8: d5207c2b 00022b2c
	v_cmp_u_f32_e64 s0, v61, v61                               // 000000002cb0: d4180000 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000002cb8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002cbc: bf870001
	v_cndmask_b32_e64 v44, v45, v51, s0                        // 000000002cc0: d501002c 0002672d
	global_store_d16_hi_b16 v[42:43], v44, off                 // 000000002cc8: ee09407c 16000000 0000002a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002cd8: 8c7e017e
	s_mul_u64 s[26:27], s[22:23], 7                            // 000000002cdc: aa9a8716
	v_cmp_lt_i64_e64 s8, 7, v[0:1]                             // 000000002ce0: d4510008 02020087
	v_add_co_u32 v8, s0, s26, v8                               // 000000002ce8: d7000008 0202101a
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s27, v9, s0                  // 000000002cf4: d5207c09 0002121b
	s_and_b32 s0, s8, s2                                       // 000000002cfc: 8b000208
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002d00: 3e101081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d04: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002d08: be812000
	s_cbranch_execz 28                                         // 000000002d0c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1280>
	v_bfe_u32 v0, v58, 16, 1                                   // 000000002d10: d6100000 0205213a
	s_wait_kmcnt 0x0                                           // 000000002d18: bfc70000
	v_add_co_u32 v1, s0, s24, v2                               // 000000002d1c: d7000001 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002d24: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v3, s0                 // 000000002d28: d5207c2a 00020619
	v_add3_u32 v43, v0, v58, 0x7fff                            // 000000002d30: d655002b 03fe7500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d3c: bf870003
	v_add_co_u32 v0, s0, v1, v8                                // 000000002d40: d7000000 02021101
	v_or_b32_e32 v44, 0x400000, v58                            // 000000002d48: 385874ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d50: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v42, v9, s0                  // 000000002d54: d5207c01 0002132a
	v_cmp_u_f32_e64 s0, v58, v58                               // 000000002d5c: d4180000 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000002d64: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d68: bf870001
	v_cndmask_b32_e64 v42, v43, v44, s0                        // 000000002d6c: d501002a 0002592b
	global_store_d16_hi_b16 v[0:1], v42, off                   // 000000002d74: ee09407c 15000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d84: 8c7e017e
	v_mul_lo_u32 v42, s23, v10                                 // 000000002d88: d72c002a 02021417
	v_mul_lo_u32 v43, s22, v11                                 // 000000002d90: d72c002b 02021616
	v_mad_co_u64_u32 v[0:1], null, s22, v10, 0                 // 000000002d98: d6fe7c00 02021416
	v_sub_co_u32 v10, s0, s20, v10                             // 000000002da0: d701000a 02021414
	s_wait_alu depctr_va_sdst(0)                               // 000000002da8: bf88f19f
	v_sub_co_ci_u32_e64 v11, null, s21, v11, s0                // 000000002dac: d5217c0b 00021615
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000002db4: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[10:11]                          // 000000002db8: d451000a 02021480
	v_add3_u32 v1, v1, v43, v42                                // 000000002dc0: d6550001 04aa5701
	s_delay_alu instid0(valu_dep_1)                            // 000000002dc8: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002dcc: 3e000081
	s_and_b32 s0, s10, s2                                      // 000000002dd0: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dd4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002dd8: be812000
	s_cbranch_execz 28                                         // 000000002ddc: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1350>
	s_wait_kmcnt 0x0                                           // 000000002de0: bfc70000
	v_add_co_u32 v43, s0, s24, v0                              // 000000002de4: d700002b 02020018
	v_bfe_u32 v42, v53, 16, 1                                  // 000000002dec: d610002a 02052135
	s_wait_alu depctr_va_sdst(0)                               // 000000002df4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s25, v1, s0                 // 000000002df8: d5207c2c 00020219
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002e00: bf870193
	v_add_co_u32 v4, s0, v43, v4                               // 000000002e04: d7000004 0202092b
	v_add3_u32 v42, v42, v53, 0x7fff                           // 000000002e0c: d655002a 03fe6b2a 00007fff
	v_or_b32_e32 v45, 0x400000, v53                            // 000000002e18: 385a6aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e20: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v44, v5, s0                  // 000000002e24: d5207c05 00020b2c
	v_cmp_u_f32_e64 s0, v53, v53                               // 000000002e2c: d4180000 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000002e34: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e38: bf870001
	v_cndmask_b32_e64 v42, v42, v45, s0                        // 000000002e3c: d501002a 00025b2a
	global_store_d16_hi_b16 v[4:5], v42, off                   // 000000002e44: ee09407c 15000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e54: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[10:11]                           // 000000002e58: d4510007 02021481
	s_and_b32 s0, s7, s2                                       // 000000002e60: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e64: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e68: be812000
	s_cbranch_execz 28                                         // 000000002e6c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x13e0>
	v_bfe_u32 v4, v50, 16, 1                                   // 000000002e70: d6100004 02052132
	s_wait_kmcnt 0x0                                           // 000000002e78: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002e7c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002e84: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s25, v1, s0                 // 000000002e88: d5207c2a 00020219
	v_add3_u32 v43, v4, v50, 0x7fff                            // 000000002e90: d655002b 03fe6504 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e9c: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 000000002ea0: d7000004 02020d05
	v_or_b32_e32 v44, 0x400000, v50                            // 000000002ea8: 385864ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v42, v7, s0                  // 000000002eb4: d5207c05 00020f2a
	v_cmp_u_f32_e64 s0, v50, v50                               // 000000002ebc: d4180000 02026532
	s_wait_alu depctr_va_sdst(0)                               // 000000002ec4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ec8: bf870001
	v_cndmask_b32_e64 v6, v43, v44, s0                         // 000000002ecc: d5010006 0002592b
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002ed4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ee4: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[10:11]                           // 000000002ee8: d4510006 02021482
	s_and_b32 s0, s6, s2                                       // 000000002ef0: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ef4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ef8: be812000
	s_cbranch_execz 28                                         // 000000002efc: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1470>
	v_bfe_u32 v4, v49, 16, 1                                   // 000000002f00: d6100004 02052131
	s_wait_kmcnt 0x0                                           // 000000002f08: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002f0c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002f14: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000002f18: d5207c06 00020219
	v_add3_u32 v7, v4, v49, 0x7fff                             // 000000002f20: d6550007 03fe6304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f2c: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000002f30: d7000004 02021905
	v_or_b32_e32 v42, 0x400000, v49                            // 000000002f38: 385462ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f40: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 000000002f44: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v49, v49                               // 000000002f4c: d4180000 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000002f54: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f58: bf870001
	v_cndmask_b32_e64 v6, v7, v42, s0                          // 000000002f5c: d5010006 00025507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002f64: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f74: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[10:11]                           // 000000002f78: d4510005 02021483
	s_and_b32 s0, s5, s2                                       // 000000002f80: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f84: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f88: be812000
	s_cbranch_execz 28                                         // 000000002f8c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1500>
	v_bfe_u32 v4, v48, 16, 1                                   // 000000002f90: d6100004 02052130
	s_wait_kmcnt 0x0                                           // 000000002f98: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 000000002f9c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000002fa4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000002fa8: d5207c06 00020219
	v_add3_u32 v7, v4, v48, 0x7fff                             // 000000002fb0: d6550007 03fe6104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002fbc: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 000000002fc0: d7000004 02021d05
	v_or_b32_e32 v12, 0x400000, v48                            // 000000002fc8: 381860ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 000000002fd4: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v48, v48                               // 000000002fdc: d4180000 02026130
	s_wait_alu depctr_va_sdst(0)                               // 000000002fe4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fe8: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000002fec: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000002ff4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003000: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003004: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[10:11]                           // 000000003008: d4510004 02021484
	s_and_b32 s0, s4, s2                                       // 000000003010: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003014: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003018: be812000
	s_cbranch_execz 28                                         // 00000000301c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1590>
	v_bfe_u32 v4, v47, 16, 1                                   // 000000003020: d6100004 0205212f
	s_wait_kmcnt 0x0                                           // 000000003028: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 00000000302c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000003038: d5207c06 00020219
	v_add3_u32 v7, v4, v47, 0x7fff                             // 000000003040: d6550007 03fe5f04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000304c: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003050: d7000004 02022105
	v_or_b32_e32 v12, 0x400000, v47                            // 000000003058: 38185eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003060: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 000000003064: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v47, v47                               // 00000000306c: d4180000 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003074: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003078: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 00000000307c: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003084: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003090: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003094: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[10:11]                           // 000000003098: d4510003 02021485
	s_and_b32 s0, s3, s2                                       // 0000000030a0: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030a8: be812000
	s_cbranch_execz 28                                         // 0000000030ac: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1620>
	v_bfe_u32 v4, v46, 16, 1                                   // 0000000030b0: d6100004 0205212e
	s_wait_kmcnt 0x0                                           // 0000000030b8: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 0000000030bc: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 0000000030c8: d5207c06 00020219
	v_add3_u32 v7, v4, v46, 0x7fff                             // 0000000030d0: d6550007 03fe5d04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030dc: bf870003
	v_add_co_u32 v4, s0, v5, v18                               // 0000000030e0: d7000004 02022505
	v_or_b32_e32 v12, 0x400000, v46                            // 0000000030e8: 38185cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s0                  // 0000000030f4: d5207c05 00022706
	v_cmp_u_f32_e64 s0, v46, v46                               // 0000000030fc: d4180000 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003104: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003108: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 00000000310c: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003114: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003120: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003124: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[10:11]                           // 000000003128: d4510001 02021486
	s_and_b32 s0, s1, s2                                       // 000000003130: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 000000003134: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 000000003138: be912000
	s_cbranch_execz 28                                         // 00000000313c: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x16b0>
	v_bfe_u32 v4, v41, 16, 1                                   // 000000003140: d6100004 02052129
	s_wait_kmcnt 0x0                                           // 000000003148: bfc70000
	v_add_co_u32 v5, s0, s24, v0                               // 00000000314c: d7000005 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003154: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s0                  // 000000003158: d5207c06 00020219
	v_add3_u32 v7, v4, v41, 0x7fff                             // 000000003160: d6550007 03fe5304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000316c: bf870003
	v_add_co_u32 v4, s0, v5, v20                               // 000000003170: d7000004 02022905
	v_or_b32_e32 v12, 0x400000, v41                            // 000000003178: 381852ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003180: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s0                  // 000000003184: d5207c05 00022b06
	v_cmp_u_f32_e64 s0, v41, v41                               // 00000000318c: d4180000 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000003194: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003198: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 00000000319c: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000031a4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000031b4: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[10:11]                           // 0000000031b8: d4510000 02021487
	s_and_b32 s2, s0, s2                                       // 0000000031c0: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031c4: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 0000000031c8: be912002
	s_cbranch_execz 28                                         // 0000000031cc: bfa5001c <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1740>
	v_bfe_u32 v4, v40, 16, 1                                   // 0000000031d0: d6100004 02052128
	s_wait_kmcnt 0x0                                           // 0000000031d8: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000031dc: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000031e8: d5207c06 000a0219
	v_add3_u32 v7, v4, v40, 0x7fff                             // 0000000031f0: d6550007 03fe5104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031fc: bf870003
	v_add_co_u32 v4, s2, v5, v8                                // 000000003200: d7000204 02021105
	v_or_b32_e32 v10, 0x400000, v40                            // 000000003208: 381450ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003210: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s2                   // 000000003214: d5207c05 000a1306
	v_cmp_u_f32_e64 s2, v40, v40                               // 00000000321c: d4180002 02025128
	s_wait_alu depctr_va_sdst(0)                               // 000000003224: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003228: bf870001
	v_cndmask_b32_e64 v6, v7, v10, s2                          // 00000000322c: d5010006 000a1507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003234: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003240: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003244: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 000000003248: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000324c: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003250: be8f2002
	s_cbranch_execz 40                                         // 000000003254: bfa50028 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x17f8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003258: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003260: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003264: d5207c05 00082680
	v_bfe_u32 v6, v39, 16, 1                                   // 00000000326c: d6100006 02052127
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003274: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003278: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003280: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003284: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 00000000328c: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003290: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003298: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 00000000329c: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000032a4: 3e080881
	v_add3_u32 v6, v6, v39, 0x7fff                             // 0000000032a8: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 0000000032b4: 38124eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000032bc: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 0000000032c0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000032c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000032cc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v39, v39                               // 0000000032d4: d4180002 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 0000000032dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000032e0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000032e4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000032ec: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000032fc: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 000000003300: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000003304: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003308: be8f2002
	s_cbranch_execz 46                                         // 00000000330c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x18c8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003310: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000331c: d5207c05 00082680
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003324: d6100006 02052126
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000332c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003330: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003338: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000333c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v38                             // 000000003344: 38124cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000334c: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 000000003350: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000003358: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 00000000335c: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 000000003364: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003368: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003370: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003374: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000337c: 3e080881
	v_add3_u32 v6, v6, v38, 0x7fff                             // 000000003380: d6550006 03fe4d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000338c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003390: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003398: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000339c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v38, v38                               // 0000000033a4: d4180002 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 0000000033ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033b0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000033b4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000033bc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000033cc: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 0000000033d0: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d4: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 0000000033d8: be8e2002
	s_cbranch_execz 46                                         // 0000000033dc: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1998>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000033e0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000033ec: d5207c05 00082680
	v_bfe_u32 v6, v37, 16, 1                                   // 0000000033f4: d6100006 02052125
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000033fc: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003400: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003408: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000340c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v37                             // 000000003414: 38124aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000341c: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 000000003420: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003428: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 00000000342c: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 000000003434: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003438: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003440: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003444: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000344c: 3e080881
	v_add3_u32 v6, v6, v37, 0x7fff                             // 000000003450: d6550006 03fe4b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000345c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003460: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003468: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000346c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v37, v37                               // 000000003474: d4180002 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 00000000347c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003480: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003484: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000348c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003498: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 00000000349c: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 0000000034a0: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034a4: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 0000000034a8: be8d2002
	s_cbranch_execz 46                                         // 0000000034ac: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1a68>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000034b0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000034b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000034bc: d5207c05 00082680
	v_bfe_u32 v6, v36, 16, 1                                   // 0000000034c4: d6100006 02052124
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034cc: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000034d0: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000034d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000034dc: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v36                             // 0000000034e4: 381248ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034ec: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 0000000034f0: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 0000000034fc: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 000000003504: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003508: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003510: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003514: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000351c: 3e080881
	v_add3_u32 v6, v6, v36, 0x7fff                             // 000000003520: d6550006 03fe4906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000352c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003530: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003538: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000353c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v36, v36                               // 000000003544: d4180002 02024924
	s_wait_alu depctr_va_sdst(0)                               // 00000000354c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003550: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003554: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000355c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003568: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 00000000356c: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003570: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003574: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 000000003578: be8c2002
	s_cbranch_execz 46                                         // 00000000357c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1b38>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003580: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003588: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000358c: d5207c05 00082680
	v_bfe_u32 v6, v35, 16, 1                                   // 000000003594: d6100006 02052123
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000359c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000035a0: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000035a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000035ac: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v35                             // 0000000035b4: 381246ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035bc: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000035c0: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000035cc: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000035d4: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000035d8: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000035e0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000035e4: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000035ec: 3e080881
	v_add3_u32 v6, v6, v35, 0x7fff                             // 0000000035f0: d6550006 03fe4706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035fc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003600: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003608: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000360c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v35, v35                               // 000000003614: d4180002 02024723
	s_wait_alu depctr_va_sdst(0)                               // 00000000361c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003620: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003624: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000362c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003638: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 00000000363c: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 000000003640: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003644: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 000000003648: be8b2002
	s_cbranch_execz 46                                         // 00000000364c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1c08>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003650: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003658: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000365c: d5207c05 00082680
	v_bfe_u32 v6, v34, 16, 1                                   // 000000003664: d6100006 02052122
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000366c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003670: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003678: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000367c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v34                             // 000000003684: 381244ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000368c: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 000000003690: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000003698: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 00000000369c: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 0000000036a4: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000036a8: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000036b4: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000036bc: 3e080881
	v_add3_u32 v6, v6, v34, 0x7fff                             // 0000000036c0: d6550006 03fe4506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036cc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000036d0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000036d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000036dc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v34, v34                               // 0000000036e4: d4180002 02024522
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036f0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000036f4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000036fc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003708: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 00000000370c: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 000000003710: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000003714: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 000000003718: be892002
	s_cbranch_execz 46                                         // 00000000371c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1cd8>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003720: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003728: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000372c: d5207c05 00082680
	v_bfe_u32 v6, v33, 16, 1                                   // 000000003734: d6100006 02052121
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000373c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003740: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003748: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000374c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v33                             // 000000003754: 381242ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000375c: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003760: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003768: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 00000000376c: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000003774: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003778: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003780: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003784: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000378c: 3e080881
	v_add3_u32 v6, v6, v33, 0x7fff                             // 000000003790: d6550006 03fe4306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000379c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000037a0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000037a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000037ac: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v33, v33                               // 0000000037b4: d4180002 02024321
	s_wait_alu depctr_va_sdst(0)                               // 0000000037bc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037c0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000037c4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000037cc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000037dc: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 0000000037e0: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037e4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000037e8: be882002
	s_cbranch_execz 46                                         // 0000000037ec: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1da8>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000037f0: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000037fc: d5207c05 00082680
	v_bfe_u32 v6, v32, 16, 1                                   // 000000003804: d6100006 02052120
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000380c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003810: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003818: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000381c: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003824: 380e40ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000382c: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003830: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003838: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 00000000383c: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000003844: bfc70000
	v_add_co_u32 v2, s2, s24, v2                               // 000000003848: d7000202 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003850: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s2                  // 000000003854: d5207c03 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000385c: 3e080881
	v_add3_u32 v6, v6, v32, 0x7fff                             // 000000003860: d6550006 03fe4106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000386c: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000003870: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000003878: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 00000000387c: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v32, v32                               // 000000003884: d4180002 02024120
	s_wait_alu depctr_va_sdst(0)                               // 00000000388c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003890: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000003894: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 00000000389c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000038ac: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 0000000038b0: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038b4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000038b8: be882002
	s_cbranch_execz 40                                         // 0000000038bc: bfa50028 <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1e60>
	v_add_co_u32 v2, s2, v23, s18                              // 0000000038c0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000038c8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 0000000038cc: d5207c03 00082680
	v_bfe_u32 v4, v31, 16, 1                                   // 0000000038d4: d6100004 0205211f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038dc: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 0000000038e0: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000038ec: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 0000000038f4: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000038f8: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003900: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003904: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 00000000390c: 3e040481
	v_add3_u32 v4, v4, v31, 0x7fff                             // 000000003910: d6550004 03fe3f04 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 00000000391c: 380e3eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003924: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000003928: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003930: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003934: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v31, v31                               // 00000000393c: d4180002 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000003944: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003948: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000394c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003954: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003960: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003964: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000003968: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 00000000396c: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000003970: be872002
	s_cbranch_execz 46                                         // 000000003974: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1f30>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003978: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003980: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003984: d5207c03 00082680
	v_bfe_u32 v4, v30, 16, 1                                   // 00000000398c: d6100004 0205211e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003994: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003998: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000039a4: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v30                             // 0000000039ac: 380e3cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039b4: bf8701a3
	v_add_co_u32 v2, s2, s22, v2                               // 0000000039b8: d7000202 02020416
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 0000000039c4: d5207c03 000a0617
	s_wait_kmcnt 0x0                                           // 0000000039cc: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000039d0: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000039dc: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000039e4: 3e040481
	v_add3_u32 v4, v4, v30, 0x7fff                             // 0000000039e8: d6550004 03fe3d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039f4: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000039f8: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003a00: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003a04: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v30, v30                               // 000000003a0c: d4180002 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a14: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a18: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003a1c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003a24: ee09407c 02000000 00002002
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003a30: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000003a34: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a38: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000003a3c: be862002
	s_cbranch_execz 46                                         // 000000003a40: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x1ffc>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003a44: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003a4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003a50: d5207c03 00082680
	v_bfe_u32 v4, v29, 16, 1                                   // 000000003a58: d6100004 0205211d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a60: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003a64: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003a6c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003a70: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v29                             // 000000003a78: 380e3aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a80: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000003a84: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000003a8c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000003a90: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000003a98: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003a9c: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003aa8: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003ab0: 3e040481
	v_add3_u32 v4, v4, v29, 0x7fff                             // 000000003ab4: d6550004 03fe3b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ac0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003ac4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003acc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003ad0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v29, v29                               // 000000003ad8: d4180002 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000003ae0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ae4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003ae8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003af0: ee09407c 02000000 00002002
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003afc: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 000000003b00: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b04: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000003b08: be852002
	s_cbranch_execz 46                                         // 000000003b0c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x20c8>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003b10: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003b18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003b1c: d5207c03 00082680
	v_bfe_u32 v4, v28, 16, 1                                   // 000000003b24: d6100004 0205211c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b2c: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003b30: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003b38: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003b3c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v28                             // 000000003b44: 380e38ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b4c: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000003b50: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000003b58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000003b5c: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000003b64: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003b68: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003b70: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003b74: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003b7c: 3e040481
	v_add3_u32 v4, v4, v28, 0x7fff                             // 000000003b80: d6550004 03fe3904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b8c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003b90: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003b98: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003b9c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v28, v28                               // 000000003ba4: d4180002 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000003bac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003bb0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003bb4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003bbc: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003bcc: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 000000003bd0: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bd4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003bd8: be842002
	s_cbranch_execz 46                                         // 000000003bdc: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x2198>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003be0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003be8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003bec: d5207c03 00082680
	v_bfe_u32 v4, v27, 16, 1                                   // 000000003bf4: d6100004 0205211b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bfc: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003c00: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003c08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003c0c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v27                             // 000000003c14: 380e36ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c1c: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000003c20: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000003c28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000003c2c: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000003c34: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003c38: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003c40: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003c44: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003c4c: 3e040481
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000003c50: d6550004 03fe3704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c5c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003c60: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003c68: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003c6c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v27, v27                               // 000000003c74: d4180002 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000003c7c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c80: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003c84: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c8c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003c9c: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000003ca0: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ca4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003ca8: be832002
	s_cbranch_execz 46                                         // 000000003cac: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x2268>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003cb0: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003cbc: d5207c03 00082680
	v_bfe_u32 v4, v26, 16, 1                                   // 000000003cc4: d6100004 0205211a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ccc: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003cd0: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003cd8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003cdc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v26                             // 000000003ce4: 380e34ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cec: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000003cf0: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000003cfc: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000003d04: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003d08: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003d10: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003d14: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003d1c: 3e040481
	v_add3_u32 v4, v4, v26, 0x7fff                             // 000000003d20: d6550004 03fe3504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d2c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003d30: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003d38: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003d3c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v26, v26                               // 000000003d44: d4180002 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000003d4c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d50: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003d54: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d5c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003d6c: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000003d70: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d74: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000003d78: be822001
	s_cbranch_execz 46                                         // 000000003d7c: bfa5002e <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x2338>
	v_add_co_u32 v2, s1, v23, s18                              // 000000003d80: d7000102 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003d88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s1                   // 000000003d8c: d5207c03 00042680
	v_bfe_u32 v4, v25, 16, 1                                   // 000000003d94: d6100004 02052119
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d9c: bf8701a3
	v_add_co_u32 v2, s1, v2, v22                               // 000000003da0: d7000102 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003da8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000003dac: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v25                             // 000000003db4: 380e32ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dbc: bf8701a3
	v_add_co_u32 v2, s1, s28, v2                               // 000000003dc0: d7000102 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s1                  // 000000003dcc: d5207c03 0006061d
	s_wait_kmcnt 0x0                                           // 000000003dd4: bfc70000
	v_add_co_u32 v5, s1, s24, v0                               // 000000003dd8: d7000105 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003de0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s1                  // 000000003de4: d5207c06 00060219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003dec: 3e040481
	v_add3_u32 v4, v4, v25, 0x7fff                             // 000000003df0: d6550004 03fe3304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dfc: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000003e00: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003e08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000003e0c: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v25, v25                               // 000000003e14: d4180001 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000003e1c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e20: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000003e24: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003e2c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e38: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003e3c: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000003e40: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e44: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003e48: be812000
	s_cbranch_execz 43                                         // 000000003e4c: bfa5002b <tessera_rocm_scaled_matmul_lds_cce0eb0c281cf52d+0x23fc>
	v_add_co_u32 v2, s0, v23, s18                              // 000000003e50: d7000002 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003e58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s0                   // 000000003e5c: d5207c03 00002680
	v_bfe_u32 v4, v24, 16, 1                                   // 000000003e64: d6100004 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e6c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v22                           // 000000003e70: d7006a02 02022d02
	s_wait_alu depctr_va_vcc(0)                                // 000000003e78: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000003e7c: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v24                             // 000000003e84: 380a30ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e8c: bf8701a3
	v_add_co_u32 v2, vcc_lo, s26, v2                           // 000000003e90: d7006a02 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 000000003e98: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s27, v3, vcc_lo              // 000000003e9c: d5207c03 01aa061b
	s_wait_kmcnt 0x0                                           // 000000003ea4: bfc70000
	v_add_co_u32 v0, vcc_lo, s24, v0                           // 000000003ea8: d7006a00 02020018
	s_wait_alu depctr_va_vcc(0)                                // 000000003eb0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s25, v1, vcc_lo              // 000000003eb4: d5207c01 01aa0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003ebc: 3e040481
	v_add3_u32 v4, v4, v24, 0x7fff                             // 000000003ec0: d6550004 03fe3104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ecc: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000003ed0: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000003edc: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000003ee4: 7c303118
	s_wait_alu depctr_va_vcc(0)                                // 000000003ee8: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000003eec: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000003ef0: ee09407c 01000000 00002000
	s_nop 0                                                    // 000000003efc: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000003f00: bfb60003
	s_endpgm                                                   // 000000003f04: bfb00000
	s_code_end                                                 // 000000003f08: bf9f0000
	s_code_end                                                 // 000000003f0c: bf9f0000
	s_code_end                                                 // 000000003f10: bf9f0000
	s_code_end                                                 // 000000003f14: bf9f0000
	s_code_end                                                 // 000000003f18: bf9f0000
	s_code_end                                                 // 000000003f1c: bf9f0000
	s_code_end                                                 // 000000003f20: bf9f0000
	s_code_end                                                 // 000000003f24: bf9f0000
	s_code_end                                                 // 000000003f28: bf9f0000
	s_code_end                                                 // 000000003f2c: bf9f0000
	s_code_end                                                 // 000000003f30: bf9f0000
	s_code_end                                                 // 000000003f34: bf9f0000
	s_code_end                                                 // 000000003f38: bf9f0000
	s_code_end                                                 // 000000003f3c: bf9f0000
	s_code_end                                                 // 000000003f40: bf9f0000
	s_code_end                                                 // 000000003f44: bf9f0000
	s_code_end                                                 // 000000003f48: bf9f0000
	s_code_end                                                 // 000000003f4c: bf9f0000
	s_code_end                                                 // 000000003f50: bf9f0000
	s_code_end                                                 // 000000003f54: bf9f0000
	s_code_end                                                 // 000000003f58: bf9f0000
	s_code_end                                                 // 000000003f5c: bf9f0000
	s_code_end                                                 // 000000003f60: bf9f0000
	s_code_end                                                 // 000000003f64: bf9f0000
	s_code_end                                                 // 000000003f68: bf9f0000
	s_code_end                                                 // 000000003f6c: bf9f0000
	s_code_end                                                 // 000000003f70: bf9f0000
	s_code_end                                                 // 000000003f74: bf9f0000
	s_code_end                                                 // 000000003f78: bf9f0000
	s_code_end                                                 // 000000003f7c: bf9f0000
	s_code_end                                                 // 000000003f80: bf9f0000
	s_code_end                                                 // 000000003f84: bf9f0000
	s_code_end                                                 // 000000003f88: bf9f0000
	s_code_end                                                 // 000000003f8c: bf9f0000
	s_code_end                                                 // 000000003f90: bf9f0000
	s_code_end                                                 // 000000003f94: bf9f0000
	s_code_end                                                 // 000000003f98: bf9f0000
	s_code_end                                                 // 000000003f9c: bf9f0000
	s_code_end                                                 // 000000003fa0: bf9f0000
	s_code_end                                                 // 000000003fa4: bf9f0000
	s_code_end                                                 // 000000003fa8: bf9f0000
	s_code_end                                                 // 000000003fac: bf9f0000
	s_code_end                                                 // 000000003fb0: bf9f0000
	s_code_end                                                 // 000000003fb4: bf9f0000
	s_code_end                                                 // 000000003fb8: bf9f0000
	s_code_end                                                 // 000000003fbc: bf9f0000
	s_code_end                                                 // 000000003fc0: bf9f0000
	s_code_end                                                 // 000000003fc4: bf9f0000
	s_code_end                                                 // 000000003fc8: bf9f0000
	s_code_end                                                 // 000000003fcc: bf9f0000
	s_code_end                                                 // 000000003fd0: bf9f0000
	s_code_end                                                 // 000000003fd4: bf9f0000
	s_code_end                                                 // 000000003fd8: bf9f0000
	s_code_end                                                 // 000000003fdc: bf9f0000
	s_code_end                                                 // 000000003fe0: bf9f0000
	s_code_end                                                 // 000000003fe4: bf9f0000
	s_code_end                                                 // 000000003fe8: bf9f0000
	s_code_end                                                 // 000000003fec: bf9f0000
	s_code_end                                                 // 000000003ff0: bf9f0000
	s_code_end                                                 // 000000003ff4: bf9f0000
	s_code_end                                                 // 000000003ff8: bf9f0000
	s_code_end                                                 // 000000003ffc: bf9f0000
	s_code_end                                                 // 000000004000: bf9f0000
	s_code_end                                                 // 000000004004: bf9f0000
	s_code_end                                                 // 000000004008: bf9f0000
	s_code_end                                                 // 00000000400c: bf9f0000
	s_code_end                                                 // 000000004010: bf9f0000
	s_code_end                                                 // 000000004014: bf9f0000
	s_code_end                                                 // 000000004018: bf9f0000
	s_code_end                                                 // 00000000401c: bf9f0000
	s_code_end                                                 // 000000004020: bf9f0000
	s_code_end                                                 // 000000004024: bf9f0000
	s_code_end                                                 // 000000004028: bf9f0000
	s_code_end                                                 // 00000000402c: bf9f0000
	s_code_end                                                 // 000000004030: bf9f0000
	s_code_end                                                 // 000000004034: bf9f0000
	s_code_end                                                 // 000000004038: bf9f0000
	s_code_end                                                 // 00000000403c: bf9f0000
	s_code_end                                                 // 000000004040: bf9f0000
	s_code_end                                                 // 000000004044: bf9f0000
	s_code_end                                                 // 000000004048: bf9f0000
	s_code_end                                                 // 00000000404c: bf9f0000
	s_code_end                                                 // 000000004050: bf9f0000
	s_code_end                                                 // 000000004054: bf9f0000
	s_code_end                                                 // 000000004058: bf9f0000
	s_code_end                                                 // 00000000405c: bf9f0000
	s_code_end                                                 // 000000004060: bf9f0000
	s_code_end                                                 // 000000004064: bf9f0000
	s_code_end                                                 // 000000004068: bf9f0000
	s_code_end                                                 // 00000000406c: bf9f0000
	s_code_end                                                 // 000000004070: bf9f0000
	s_code_end                                                 // 000000004074: bf9f0000
	s_code_end                                                 // 000000004078: bf9f0000
	s_code_end                                                 // 00000000407c: bf9f0000
	s_code_end                                                 // 000000004080: bf9f0000
	s_code_end                                                 // 000000004084: bf9f0000
	s_code_end                                                 // 000000004088: bf9f0000
	s_code_end                                                 // 00000000408c: bf9f0000
	s_code_end                                                 // 000000004090: bf9f0000
	s_code_end                                                 // 000000004094: bf9f0000
	s_code_end                                                 // 000000004098: bf9f0000
	s_code_end                                                 // 00000000409c: bf9f0000
	s_code_end                                                 // 0000000040a0: bf9f0000
	s_code_end                                                 // 0000000040a4: bf9f0000
	s_code_end                                                 // 0000000040a8: bf9f0000
	s_code_end                                                 // 0000000040ac: bf9f0000
	s_code_end                                                 // 0000000040b0: bf9f0000
	s_code_end                                                 // 0000000040b4: bf9f0000
	s_code_end                                                 // 0000000040b8: bf9f0000
	s_code_end                                                 // 0000000040bc: bf9f0000
	s_code_end                                                 // 0000000040c0: bf9f0000
	s_code_end                                                 // 0000000040c4: bf9f0000
	s_code_end                                                 // 0000000040c8: bf9f0000
	s_code_end                                                 // 0000000040cc: bf9f0000
	s_code_end                                                 // 0000000040d0: bf9f0000
	s_code_end                                                 // 0000000040d4: bf9f0000
	s_code_end                                                 // 0000000040d8: bf9f0000
	s_code_end                                                 // 0000000040dc: bf9f0000
	s_code_end                                                 // 0000000040e0: bf9f0000
	s_code_end                                                 // 0000000040e4: bf9f0000
	s_code_end                                                 // 0000000040e8: bf9f0000
	s_code_end                                                 // 0000000040ec: bf9f0000
	s_code_end                                                 // 0000000040f0: bf9f0000
	s_code_end                                                 // 0000000040f4: bf9f0000
	s_code_end                                                 // 0000000040f8: bf9f0000
	s_code_end                                                 // 0000000040fc: bf9f0000
