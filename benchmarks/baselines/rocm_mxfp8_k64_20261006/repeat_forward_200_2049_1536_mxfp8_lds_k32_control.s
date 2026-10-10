
/tmp/tmpz9tjovbb.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b04: f4002100 f80000d8
	s_load_b128 s[24:27], s[0:1], 0xc8                         // 000000001b0c: f4004600 f80000c8
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b14: 320a0081
	s_mov_b32 s2, ttmp9                                        // 000000001b18: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b1c: 86039f75
	s_mov_b32 s8, ttmp7                                        // 000000001b20: be880073
	s_ashr_i32 s9, ttmp7, 31                                   // 000000001b24: 86099f73
	s_lshl_b64 s[22:23], s[2:3], 6                             // 000000001b28: 84968602
	s_lshl_b64 s[8:9], s[8:9], 7                               // 000000001b2c: 84888708
	v_add_co_u32 v1, s2, s22, v5                               // 000000001b30: d7000201 02020a16
	s_clause 0x3                                               // 000000001b38: bf850003
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001b3c: f4002280 f8000008
	s_load_b64 s[12:13], s[0:1], 0x30                          // 000000001b44: f4002300 f8000030
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000001b4c: f4002180 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b54: f4002700 f8000080
	v_add_co_ci_u32_e64 v2, null, s23, 0, s2                   // 000000001b5c: d5207c02 00090017
	v_or_b32_e32 v3, s8, v5                                    // 000000001b64: 38060a08
	v_mov_b32_e32 v4, s9                                       // 000000001b68: 7e080209
	v_mul_u32_u24_e32 v11, 48, v5                              // 000000001b6c: 16160ab0
	v_and_b32_e32 v6, 0x60, v5                                 // 000000001b70: 360c0aff 00000060
	v_dual_mov_b32 v64, 0 :: v_dual_and_b32 v47, 32, v0        // 000000001b78: ca240080 402e00a0
	v_cmp_gt_u32_e64 s3, 0x80, v0                              // 000000001b80: d44c0003 020200ff 00000080
	s_wait_kmcnt 0x0                                           // 000000001b8c: bfc70000
	s_add_nc_u64 s[14:15], s[26:27], -1                        // 000000001b90: a98ec11a
	s_add_nc_u64 s[16:17], s[24:25], -1                        // 000000001b94: a990c118
	v_cmp_gt_u64_e32 vcc_lo, s[14:15], v[1:2]                  // 000000001b98: 7cb8020e
	v_cmp_gt_u64_e64 s2, s[16:17], v[3:4]                      // 000000001b9c: d45c0002 02020610
	v_and_b32_e32 v46, 15, v0                                  // 000000001ba4: 365c008f
	s_lshr_b64 s[30:31], s[4:5], 5                             // 000000001ba8: 859e8504
	v_mov_b32_e32 v89, 0                                       // 000000001bac: 7eb20280
	v_mov_b32_e32 v87, 0                                       // 000000001bb0: 7eae0280
	v_dual_cndmask_b32 v9, s15, v2 :: v_dual_lshlrev_b32 v2, 4, v0// 000000001bb4: ca62040f 09020084
	s_wait_alu depctr_va_sdst(0)                               // 000000001bbc: bf88f19f
	v_cndmask_b32_e64 v3, s16, v3, s2                          // 000000001bc0: d5010003 000a0610
	v_cndmask_b32_e64 v4, s17, v4, s2                          // 000000001bc8: d5010004 000a0811
	v_cndmask_b32_e32 v10, s14, v1, vcc_lo                     // 000000001bd0: 0214020e
	v_mul_lo_u32 v9, v9, s4                                    // 000000001bd4: d72c0009 02000909
	v_and_b32_e32 v12, 16, v2                                  // 000000001bdc: 36180490
	v_mul_lo_u32 v13, v3, s5                                   // 000000001be0: d72c000d 02000b03
	v_mad_co_u64_u32 v[1:2], null, v3, s4, s[10:11]            // 000000001be8: d6fe7c01 00280903
	v_mul_lo_u32 v14, v4, s4                                   // 000000001bf0: d72c000e 02000904
	v_mad_co_u64_u32 v[3:4], null, v10, s4, s[12:13]           // 000000001bf8: d6fe7c03 0030090a
	v_add_nc_u32_e32 v68, v11, v12                             // 000000001c00: 4a88190b
	v_mul_lo_u32 v11, v10, s5                                  // 000000001c04: d72c000b 02000b0a
	v_and_b32_e32 v10, 8, v5                                   // 000000001c0c: 36140a88
	v_and_b32_e32 v0, 47, v0                                   // 000000001c10: 360000af
	v_or_b32_e32 v8, v46, v47                                  // 000000001c14: 38105f2e
	v_add_co_u32 v70, vcc_lo, v1, v12                          // 000000001c18: d7006a46 02021901
	v_add3_u32 v2, v14, v2, v13                                // 000000001c20: d6550002 0436050e
	v_mov_b32_e32 v13, s9                                      // 000000001c28: 7e1a0209
	v_or_b32_e32 v22, s8, v6                                   // 000000001c2c: 382c0c08
	v_add3_u32 v1, v9, v4, v11                                 // 000000001c30: d6550001 042e0909
	v_or_b32_e32 v7, 16, v6                                    // 000000001c38: 380e0c90
	s_wait_alu depctr_va_vcc(0)                                // 000000001c3c: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, 0, v2, vcc_lo               // 000000001c40: d5207c47 01aa0480
	v_add_co_u32 v72, vcc_lo, v3, v12                          // 000000001c48: d7006a48 02021903
	v_or_b32_e32 v12, v22, v10                                 // 000000001c50: 38181516
	s_wait_alu depctr_va_vcc(0)                                // 000000001c54: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, 0, v1, vcc_lo               // 000000001c58: d5207c49 01aa0280
	v_or_b32_e32 v38, s8, v7                                   // 000000001c60: 384c0e08
	v_or_b32_e32 v7, v7, v46                                   // 000000001c64: 380e5d07
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001c68: 7ca81818
	v_mul_u32_u24_e32 v0, 48, v0                               // 000000001c6c: 160000b0
	v_or_b32_e32 v6, v6, v46                                   // 000000001c70: 380c5d06
	v_mov_b32_e32 v1, s9                                       // 000000001c74: 7e020209
	v_mul_u32_u24_e32 v4, 48, v7                               // 000000001c78: 16080eb0
	s_lshr_b32 s8, s5, 5                                       // 000000001c7c: 85088505
	s_wait_alu depctr_va_vcc(0)                                // 000000001c80: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v13, vcc_lo                       // 000000001c84: 02061a80
	v_mul_u32_u24_e32 v2, 48, v6                               // 000000001c88: 16040cb0
	v_or_b32_e32 v6, 16, v8                                    // 000000001c8c: 380c1090
	v_or_b32_e32 v76, v4, v10                                  // 000000001c90: 38981504
	v_mov_b32_e32 v9, s23                                      // 000000001c94: 7e120217
	v_mul_lo_u32 v4, s30, v3                                   // 000000001c98: d72c0004 0202061e
	v_mov_b32_e32 v3, s9                                       // 000000001ca0: 7e060209
	v_or_b32_e32 v7, 1, v10                                    // 000000001ca4: 380e1481
	v_or_b32_e32 v48, v10, v0                                  // 000000001ca8: 3860010a
	v_or_b32_e32 v74, v2, v10                                  // 000000001cac: 38941502
	v_mul_u32_u24_e32 v2, 48, v6                               // 000000001cb0: 16040cb0
	v_or_b32_e32 v30, 2, v10                                   // 000000001cb4: 383c1482
	v_or_b32_e32 v0, v7, v22                                   // 000000001cb8: 38002d07
	v_or_b32_e32 v32, 3, v10                                   // 000000001cbc: 38401483
	v_or_b32_e32 v33, 4, v10                                   // 000000001cc0: 38421484
	v_or_b32_e32 v49, v2, v10                                  // 000000001cc4: 38621502
	v_cndmask_b32_e32 v2, 0, v12, vcc_lo                       // 000000001cc8: 02041880
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000001ccc: d4540004 02020018
	v_or_b32_e32 v36, 5, v10                                   // 000000001cd4: 38481485
	v_or_b32_e32 v39, 6, v10                                   // 000000001cd8: 384e1486
	v_or_b32_e32 v40, 7, v10                                   // 000000001cdc: 38501487
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ce0: bf88ff9e
	v_mul_lo_u32 v5, s8, v2                                    // 000000001ce4: d72c0005 02020408
	v_mad_co_u64_u32 v[14:15], null, s30, v2, s[6:7]           // 000000001cec: d6fe7c0e 001a041e
	s_wait_alu depctr_va_sdst(0)                               // 000000001cf4: bf88f19f
	v_cndmask_b32_e64 v0, 0, v0, s4                            // 000000001cf8: d5010000 00120080
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 000000001d00: d5010001 00120280
	v_or_b32_e32 v2, v32, v22                                  // 000000001d08: 38042d20
	v_or_b32_e32 v10, v38, v10                                 // 000000001d0c: 38141526
	v_or_b32_e32 v8, s22, v8                                   // 000000001d10: 38101016
	v_mul_lo_u32 v18, s8, v0                                   // 000000001d14: d72c0012 02020008
	v_mul_lo_u32 v11, s30, v1                                  // 000000001d1c: d72c000b 0202021e
	v_mad_co_u64_u32 v[16:17], null, s30, v0, s[6:7]           // 000000001d24: d6fe7c10 001a001e
	v_mov_b32_e32 v1, s9                                       // 000000001d2c: 7e020209
	v_or_b32_e32 v0, v30, v22                                  // 000000001d30: 38002d1e
	v_add3_u32 v15, v5, v15, v4                                // 000000001d34: d655000f 04121f05
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 000000001d3c: d4540004 02020418
	v_cmp_gt_i64_e64 s2, s[26:27], v[8:9]                      // 000000001d44: d4540002 0202101a
	v_dual_mov_b32 v60, 0 :: v_dual_add_nc_u32 v91, 0x1800, v48// 000000001d4c: ca200080 3c5a60ff 00001800
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001d58: 7ca80018
	v_add3_u32 v17, v18, v17, v11                              // 000000001d5c: d6550011 042e2312
	s_wait_alu depctr_va_sdst(0)                               // 000000001d64: bf88f19f
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000001d68: d5010003 00120680
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000001d70: d5010002 00120480
	v_cndmask_b32_e64 v80, 0, v9, s2                           // 000000001d78: d5010050 000a1280
	v_cndmask_b32_e64 v81, 0, v8, s2                           // 000000001d80: d5010051 000a1080
	s_wait_alu depctr_va_vcc(0)                                // 000000001d88: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v0, vcc_lo                        // 000000001d8c: 02080080
	v_or_b32_e32 v0, v33, v22                                  // 000000001d90: 38002d21
	v_cndmask_b32_e32 v5, 0, v1, vcc_lo                        // 000000001d94: 020a0280
	v_mul_lo_u32 v29, s30, v3                                  // 000000001d98: d72c001d 0202061e
	v_mul_lo_u32 v31, s8, v2                                   // 000000001da0: d72c001f 02020408
	v_mad_co_u64_u32 v[20:21], null, s30, v2, s[6:7]           // 000000001da8: d6fe7c14 001a041e
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001db0: 7ca80018
	v_or_b32_e32 v2, v39, v22                                  // 000000001db4: 38042d27
	v_mul_lo_u32 v11, s30, v5                                  // 000000001db8: d72c000b 02020a1e
	v_mov_b32_e32 v5, s9                                       // 000000001dc0: 7e0a0209
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v78, 0             // 000000001dc4: ca100080 554e0080
	s_wait_alu depctr_va_vcc(0)                                // 000000001dcc: bf88ff9d
	v_cndmask_b32_e32 v23, 0, v0, vcc_lo                       // 000000001dd0: 022e0080
	v_or_b32_e32 v0, v36, v22                                  // 000000001dd4: 38002d24
	v_cndmask_b32_e32 v3, 0, v1, vcc_lo                        // 000000001dd8: 02060280
	v_mul_lo_u32 v28, s8, v4                                   // 000000001ddc: d72c001c 02020808
	v_mad_co_u64_u32 v[18:19], null, s30, v4, s[6:7]           // 000000001de4: d6fe7c12 001a081e
	v_or_b32_e32 v4, v40, v22                                  // 000000001dec: 38082d28
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001df0: 7ca80018
	v_mul_lo_u32 v34, s30, v3                                  // 000000001df4: d72c0022 0202061e
	v_mov_b32_e32 v3, s9                                       // 000000001dfc: 7e060209
	v_add3_u32 v21, v31, v21, v29                              // 000000001e00: d6550015 04762b1f
	v_add_nc_u32_e32 v92, 0x1800, v49                          // 000000001e08: 4ab862ff 00001800
	v_mov_b32_e32 v90, 0                                       // 000000001e10: 7eb40280
	s_wait_alu depctr_va_vcc(0)                                // 000000001e14: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001e18: 02000080
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 000000001e1c: d4540004 02020418
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001e24: 02020280
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001e28: 7ca80818
	v_add3_u32 v19, v28, v19, v11                              // 000000001e2c: d6550013 042e271c
	v_mul_lo_u32 v37, s8, v0                                   // 000000001e34: d72c0025 02020008
	v_mad_co_u64_u32 v[24:25], null, s30, v0, s[6:7]           // 000000001e3c: d6fe7c18 001a001e
	s_wait_alu depctr_va_sdst(0)                               // 000000001e44: bf88f19f
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000001e48: d5010002 00120480
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000001e50: d5010003 00120680
	v_mul_lo_u32 v1, s30, v1                                   // 000000001e58: d72c0001 0202021e
	v_mov_b32_e32 v84, 0                                       // 000000001e60: 7ea80280
	v_mul_lo_u32 v35, s8, v23                                  // 000000001e64: d72c0023 02022e08
	v_mad_co_u64_u32 v[26:27], null, s30, v2, s[6:7]           // 000000001e6c: d6fe7c1a 001a041e
	v_mul_lo_u32 v0, s30, v3                                   // 000000001e74: d72c0000 0202061e
	s_wait_alu depctr_va_vcc(0)                                // 000000001e7c: bf88ff9d
	v_dual_cndmask_b32 v3, 0, v4 :: v_dual_cndmask_b32 v4, 0, v5// 000000001e80: ca520880 03040a80
	v_mul_lo_u32 v5, s8, v2                                    // 000000001e88: d72c0005 02020408
	v_add3_u32 v25, v37, v25, v1                               // 000000001e90: d6550019 04063325
	v_mov_b32_e32 v1, s23                                      // 000000001e98: 7e020217
	s_delay_alu instid0(valu_dep_4)                            // 000000001e9c: bf870004
	v_mad_co_u64_u32 v[28:29], null, s30, v3, s[6:7]           // 000000001ea0: d6fe7c1c 001a061e
	v_mul_lo_u32 v2, s30, v4                                   // 000000001ea8: d72c0002 0202081e
	v_mul_lo_u32 v4, s8, v3                                    // 000000001eb0: d72c0004 02020608
	v_mov_b32_e32 v3, s9                                       // 000000001eb8: 7e060209
	v_mad_co_u64_u32 v[22:23], null, s30, v23, s[6:7]          // 000000001ebc: d6fe7c16 001a2e1e
	v_add3_u32 v27, v5, v27, v0                                // 000000001ec4: d655001b 04023705
	v_or_b32_e32 v0, s22, v6                                   // 000000001ecc: 38000c16
	v_dual_mov_b32 v5, s9 :: v_dual_mov_b32 v82, 0             // 000000001ed0: ca100009 05520080
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v66, 0             // 000000001ed8: ca100080 53420080
	s_delay_alu instid0(valu_dep_3)                            // 000000001ee0: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[0:1]                  // 000000001ee4: 7ca8001a
	v_mov_b32_e32 v11, s9                                      // 000000001ee8: 7e160209
	v_add3_u32 v29, v4, v29, v2                                // 000000001eec: d655001d 040a3b04
	v_or_b32_e32 v2, v38, v7                                   // 000000001ef4: 38040f26
	v_or_b32_e32 v4, v38, v30                                  // 000000001ef8: 38083d26
	v_add3_u32 v23, v35, v23, v34                              // 000000001efc: d6550017 048a2f23
	s_wait_alu depctr_va_vcc(0)                                // 000000001f04: bf88ff9d
	v_cndmask_b32_e32 v86, 0, v1, vcc_lo                       // 000000001f08: 02ac0280
	v_cmp_gt_i64_e64 s4, s[24:25], v[10:11]                    // 000000001f0c: d4540004 02021418
	v_cmp_gt_i64_e64 s5, s[24:25], v[2:3]                      // 000000001f14: d4540005 02020418
	v_cndmask_b32_e32 v88, 0, v0, vcc_lo                       // 000000001f1c: 02b00080
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0             // 000000001f20: ca100080 3e3e0080
	v_mov_b32_e32 v61, 0                                       // 000000001f28: 7e7a0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001f2c: bf88f19f
	v_cndmask_b32_e64 v1, 0, v10, s4                           // 000000001f30: d5010001 00121480
	v_cndmask_b32_e64 v0, 0, v3, s5                            // 000000001f38: d5010000 00160680
	v_cndmask_b32_e64 v6, 0, v11, s4                           // 000000001f40: d5010006 00121680
	v_cmp_gt_i64_e64 s4, s[24:25], v[4:5]                      // 000000001f48: d4540004 02020818
	v_cndmask_b32_e64 v7, 0, v2, s5                            // 000000001f50: d5010007 00160480
	v_mul_lo_u32 v50, s8, v1                                   // 000000001f58: d72c0032 02020208
	v_mad_co_u64_u32 v[30:31], null, s30, v1, s[6:7]           // 000000001f60: d6fe7c1e 001a021e
	v_mul_lo_u32 v51, s30, v0                                  // 000000001f68: d72c0033 0202001e
	v_mov_b32_e32 v1, s9                                       // 000000001f70: 7e020209
	v_or_b32_e32 v0, v38, v32                                  // 000000001f74: 38004126
	v_or_b32_e32 v2, v38, v33                                  // 000000001f78: 38044326
	s_wait_alu depctr_va_sdst(0)                               // 000000001f7c: bf88f19f
	v_cndmask_b32_e64 v4, 0, v4, s4                            // 000000001f80: d5010004 00120880
	v_cndmask_b32_e64 v5, 0, v5, s4                            // 000000001f88: d5010005 00120a80
	v_mul_lo_u32 v52, s8, v7                                   // 000000001f90: d72c0034 02020e08
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000001f98: d4540004 02020018
	v_cmp_gt_i64_e64 s5, s[24:25], v[2:3]                      // 000000001fa0: d4540005 02020418
	v_mul_lo_u32 v53, s8, v4                                   // 000000001fa8: d72c0035 02020808
	v_mad_co_u64_u32 v[34:35], null, s30, v4, s[6:7]           // 000000001fb0: d6fe7c22 001a081e
	v_mad_co_u64_u32 v[32:33], null, s30, v7, s[6:7]           // 000000001fb8: d6fe7c20 001a0e1e
	v_mul_lo_u32 v7, s30, v5                                   // 000000001fc0: d72c0007 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000001fc8: bf88f19f
	v_cndmask_b32_e64 v4, 0, v0, s4                            // 000000001fcc: d5010004 00120080
	v_or_b32_e32 v0, v38, v36                                  // 000000001fd4: 38004926
	v_cndmask_b32_e64 v5, 0, v1, s4                            // 000000001fd8: d5010005 00120280
	v_cndmask_b32_e64 v41, 0, v2, s5                           // 000000001fe0: d5010029 00160480
	v_cndmask_b32_e64 v2, 0, v3, s5                            // 000000001fe8: d5010002 00160680
	v_mul_lo_u32 v55, s8, v4                                   // 000000001ff0: d72c0037 02020808
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000001ff8: d4540004 02020018
	v_mul_lo_u32 v54, s30, v5                                  // 000000002000: d72c0036 02020a1e
	v_mad_co_u64_u32 v[36:37], null, s30, v4, s[6:7]           // 000000002008: d6fe7c24 001a081e
	v_mul_lo_u32 v56, s30, v2                                  // 000000002010: d72c0038 0202041e
	v_or_b32_e32 v2, v38, v39                                  // 000000002018: 38044f26
	v_mov_b32_e32 v5, s9                                       // 00000000201c: 7e0a0209
	v_or_b32_e32 v4, v38, v40                                  // 000000002020: 38085126
	s_wait_alu depctr_va_sdst(0)                               // 000000002024: bf88f19f
	v_cndmask_b32_e64 v0, 0, v0, s4                            // 000000002028: d5010000 00120080
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 000000002030: d5010001 00120280
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 000000002038: d4540004 02020418
	v_mul_lo_u32 v6, s30, v6                                   // 000000002040: d72c0006 02020c1e
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 000000002048: d4540005 02020818
	v_mul_lo_u32 v57, s8, v41                                  // 000000002050: d72c0039 02025208
	v_mad_co_u64_u32 v[38:39], null, s30, v41, s[6:7]          // 000000002058: d6fe7c26 001a521e
	v_mul_lo_u32 v1, s30, v1                                   // 000000002060: d72c0001 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002068: bf88f19f
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 00000000206c: d5010002 00120480
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000002074: d5010003 00120680
	v_cndmask_b32_e64 v4, 0, v4, s5                            // 00000000207c: d5010004 00160880
	v_cndmask_b32_e64 v5, 0, v5, s5                            // 000000002084: d5010005 00160a80
	v_mul_lo_u32 v58, s8, v0                                   // 00000000208c: d72c003a 02020008
	v_mad_co_u64_u32 v[40:41], null, s30, v0, s[6:7]           // 000000002094: d6fe7c28 001a001e
	v_mul_lo_u32 v0, s30, v3                                   // 00000000209c: d72c0000 0202061e
	v_mul_lo_u32 v3, s8, v2                                    // 0000000020a4: d72c0003 02020408
	v_mad_co_u64_u32 v[42:43], null, s30, v2, s[6:7]           // 0000000020ac: d6fe7c2a 001a041e
	v_mul_lo_u32 v2, s30, v5                                   // 0000000020b4: d72c0002 02020a1e
	v_mul_lo_u32 v5, s8, v4                                    // 0000000020bc: d72c0005 02020808
	v_mad_co_u64_u32 v[44:45], null, s30, v4, s[6:7]           // 0000000020c4: d6fe7c2c 001a081e
	v_add3_u32 v31, v50, v31, v6                               // 0000000020cc: d655001f 041a3f32
	v_add3_u32 v33, v52, v33, v51                              // 0000000020d4: d6550021 04ce4334
	v_add3_u32 v35, v53, v35, v7                               // 0000000020dc: d6550023 041e4735
	v_add3_u32 v37, v55, v37, v54                              // 0000000020e4: d6550025 04da4b37
	v_add3_u32 v39, v57, v39, v56                              // 0000000020ec: d6550027 04e24f39
	v_add3_u32 v41, v58, v41, v1                               // 0000000020f4: d6550029 0406533a
	v_add3_u32 v43, v3, v43, v0                                // 0000000020fc: d655002b 04025703
	v_add3_u32 v45, v5, v45, v2                                // 000000002104: d655002d 040a5b05
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v50, 0             // 00000000210c: ca100080 3b320080
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0             // 000000002114: ca100080 3a380080
	v_mov_b32_e32 v48, 0                                       // 00000000211c: 7e600280
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v79, 0             // 000000002120: ca100080 384e0080
	v_mov_b32_e32 v77, 0                                       // 000000002128: 7e9a0280
	v_mov_b32_e32 v75, 0                                       // 00000000212c: 7e960280
	v_mov_b32_e32 v69, 0                                       // 000000002130: 7e8a0280
	v_mov_b32_e32 v67, 0                                       // 000000002134: 7e860280
	v_mov_b32_e32 v65, 0                                       // 000000002138: 7e820280
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0             // 00000000213c: ca100080 37360080
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0             // 000000002144: ca100080 35340080
	v_mov_b32_e32 v51, 0                                       // 00000000214c: 7e660280
	v_mov_b32_e32 v49, 0                                       // 000000002150: 7e620280
	s_mov_b64 s[34:35], 0                                      // 000000002154: bea20180
	s_branch 516                                               // 000000002158: bfa00204 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0xe6c>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000215c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002160: 8c7e047e
	v_add_co_u32 v0, s4, v14, s34                              // 000000002164: d7000400 0200450e
	s_wait_alu depctr_va_sdst(0)                               // 00000000216c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v15, s4                 // 000000002170: d5207c01 00121e23
	s_mul_u64 s[4:5], s[34:35], s[26:27]                       // 000000002178: aa841a22
	s_wait_dscnt 0x0                                           // 00000000217c: bfc60000
	s_barrier_signal -1                                        // 000000002180: be804ec1
	s_barrier_wait 0xffff                                      // 000000002184: bf94ffff
	global_load_u8 v131, v[0:1], off                           // 000000002188: ee04007c 00000083 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002194: bf88ff9e
	s_add_nc_u64 s[6:7], s[28:29], s[4:5]                      // 000000002198: a986041c
	v_add_co_u32 v0, s4, v16, s34                              // 00000000219c: d7000400 02004510
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v17, s4                 // 0000000021a8: d5207c01 00122223
	v_add_co_u32 v2, s4, v18, s34                              // 0000000021b0: d7000402 02004512
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v19, s4                 // 0000000021bc: d5207c03 00122623
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021c4: bf88ff9e
	v_add_co_u32 v4, s4, s6, v81                               // 0000000021c8: d7000404 0202a206
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s7, v80, s4                  // 0000000021d4: d5207c05 0012a007
	global_load_u8 v132, v[0:1], off                           // 0000000021dc: ee04007c 00000084 00000000
	v_add_co_u32 v0, s4, v20, s34                              // 0000000021e8: d7000400 02004514
	global_load_u8 v133, v[2:3], off                           // 0000000021f0: ee04007c 00000085 00000002
	s_wait_alu depctr_va_sdst(0)                               // 0000000021fc: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v21, s4                 // 000000002200: d5207c01 00122a23
	v_add_co_u32 v2, s4, v22, s34                              // 000000002208: d7000402 02004516
	s_wait_alu depctr_va_sdst(0)                               // 000000002210: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v23, s4                 // 000000002214: d5207c03 00122e23
	v_add_co_u32 v6, s4, v24, s34                              // 00000000221c: d7000406 02004518
	s_wait_alu depctr_va_sdst(0)                               // 000000002224: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v25, s4                 // 000000002228: d5207c07 00123223
	v_add_co_u32 v93, s4, v26, s34                             // 000000002230: d700045d 0200451a
	s_wait_alu depctr_va_sdst(0)                               // 000000002238: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v27, s4                // 00000000223c: d5207c5e 00123623
	v_add_co_u32 v95, s4, v28, s34                             // 000000002244: d700045f 0200451c
	global_load_u8 v135, v[2:3], off                           // 00000000224c: ee04007c 00000087 00000002
	v_add_co_u32 v2, s5, v30, s34                              // 000000002258: d7000502 0200451e
	s_wait_alu depctr_va_sdst(0)                               // 000000002260: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v29, s4                // 000000002264: d5207c60 00123a23
	global_load_u8 v136, v[6:7], off                           // 00000000226c: ee04007c 00000088 00000006
	v_add_co_ci_u32_e64 v3, null, s35, v31, s5                 // 000000002278: d5207c03 00163e23
	v_add_co_u32 v6, s5, v32, s34                              // 000000002280: d7000506 02004520
	global_load_u8 v137, v[93:94], off                         // 000000002288: ee04007c 00000089 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 000000002294: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v33, s5                 // 000000002298: d5207c07 00164223
	v_add_co_u32 v93, s5, v34, s34                             // 0000000022a0: d700055d 02004522
	global_load_u8 v134, v[0:1], off                           // 0000000022a8: ee04007c 00000086 00000000
	v_add_co_u32 v0, s4, s6, v88                               // 0000000022b4: d7000400 0202b006
	global_load_u8 v138, v[95:96], off                         // 0000000022bc: ee04007c 0000008a 0000005f
	s_wait_alu depctr_va_sdst(0)                               // 0000000022c8: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v35, s5                // 0000000022cc: d5207c5e 00164623
	v_add_co_u32 v95, s5, v36, s34                             // 0000000022d4: d700055f 02004524
	v_add_co_ci_u32_e64 v1, null, s7, v86, s4                  // 0000000022dc: d5207c01 0012ac07
	global_load_u8 v139, v[2:3], off                           // 0000000022e4: ee04007c 0000008b 00000002
	v_add_co_u32 v2, s4, v38, s34                              // 0000000022f0: d7000402 02004526
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f8: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v37, s5                // 0000000022fc: d5207c60 00164a23
	global_load_u8 v140, v[6:7], off                           // 000000002304: ee04007c 0000008c 00000006
	v_add_co_ci_u32_e64 v3, null, s35, v39, s4                 // 000000002310: d5207c03 00124e23
	v_add_co_u32 v6, s4, v40, s34                              // 000000002318: d7000406 02004528
	global_load_u8 v141, v[93:94], off                         // 000000002320: ee04007c 0000008d 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 00000000232c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v41, s4                 // 000000002330: d5207c07 00125223
	v_add_co_u32 v93, s4, v42, s34                             // 000000002338: d700045d 0200452a
	global_load_u8 v142, v[95:96], off                         // 000000002340: ee04007c 0000008e 0000005f
	s_wait_alu depctr_va_sdst(0)                               // 00000000234c: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v43, s4                // 000000002350: d5207c5e 00125623
	v_add_co_u32 v95, s4, v44, s34                             // 000000002358: d700045f 0200452c
	s_wait_alu depctr_va_sdst(0)                               // 000000002360: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v45, s4                // 000000002364: d5207c60 00125a23
	s_clause 0x1                                               // 00000000236c: bf850001
	global_load_u8 v146, v[4:5], off                           // 000000002370: ee04007c 00000092 00000004
	global_load_u8 v148, v[0:1], off                           // 00000000237c: ee04007c 00000094 00000000
	s_clause 0x3                                               // 000000002388: bf850003
	global_load_u8 v143, v[2:3], off                           // 00000000238c: ee04007c 0000008f 00000002
	global_load_u8 v144, v[6:7], off                           // 000000002398: ee04007c 00000090 00000006
	global_load_u8 v145, v[93:94], off                         // 0000000023a4: ee04007c 00000091 0000005d
	global_load_u8 v147, v[95:96], off                         // 0000000023b0: ee04007c 00000093 0000005f
	ds_load_2addr_b64 v[115:118], v74 offset1:2                // 0000000023bc: d9dc0200 7300004a
	ds_load_2addr_b64 v[119:122], v91 offset1:2                // 0000000023c4: d9dc0200 7700005b
	ds_load_2addr_b64 v[123:126], v92 offset1:2                // 0000000023cc: d9dc0200 7b00005c
	ds_load_2addr_b64 v[127:130], v76 offset1:2                // 0000000023d4: d9dc0200 7f00004c
	s_add_nc_u64 s[34:35], s[34:35], 1                         // 0000000023dc: a9a28122
	s_wait_dscnt 0x2                                           // 0000000023e0: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[119:120], 0// 0000000023e4: cc464000 1a02ef73
	s_wait_dscnt 0x1                                           // 0000000023ec: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[115:116], v[123:124], 0// 0000000023f0: cc46405d 1a02f773
	s_wait_dscnt 0x0                                           // 0000000023f8: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[127:128], v[119:120], 0// 0000000023fc: cc464065 1a02ef7f
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[127:128], v[123:124], 0// 000000002404: cc46406d 1a02f77f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[117:118], v[121:122], v[0:7]// 00000000240c: cc464000 1c02f375
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[117:118], v[125:126], v[93:100]// 000000002414: cc46405d 1d76fb75
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000241c: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[129:130], v[121:122], v[101:108]// 000000002420: cc464065 1d96f381
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[129:130], v[125:126], v[109:116]// 000000002428: cc46406d 1db6fb81
	s_wait_loadcnt 0x11                                        // 000000002430: bfc00011
	v_cmp_eq_u32_e64 s4, 0xff, v131                            // 000000002434: d44a0004 020306ff 000000ff
	s_wait_loadcnt 0x10                                        // 000000002440: bfc00010
	v_cmp_eq_u32_e64 s5, 0xff, v132                            // 000000002444: d44a0005 020308ff 000000ff
	s_wait_loadcnt 0xf                                         // 000000002450: bfc0000f
	v_cmp_eq_u32_e64 s6, 0xff, v133                            // 000000002454: d44a0006 02030aff 000000ff
	s_wait_loadcnt 0xe                                         // 000000002460: bfc0000e
	v_cmp_eq_u32_e64 s8, 0xff, v135                            // 000000002464: d44a0008 02030eff 000000ff
	s_wait_loadcnt 0xd                                         // 000000002470: bfc0000d
	v_cmp_eq_u32_e64 s9, 0xff, v136                            // 000000002474: d44a0009 020310ff 000000ff
	s_wait_loadcnt 0xc                                         // 000000002480: bfc0000c
	v_cmp_eq_u32_e64 s10, 0xff, v137                           // 000000002484: d44a000a 020312ff 000000ff
	s_wait_loadcnt 0xb                                         // 000000002490: bfc0000b
	v_cmp_eq_u32_e64 s7, 0xff, v134                            // 000000002494: d44a0007 02030cff 000000ff
	s_wait_loadcnt 0xa                                         // 0000000024a0: bfc0000a
	v_cmp_eq_u32_e64 s11, 0xff, v138                           // 0000000024a4: d44a000b 020314ff 000000ff
	s_wait_loadcnt 0x9                                         // 0000000024b0: bfc00009
	v_cmp_eq_u32_e64 s12, 0xff, v139                           // 0000000024b4: d44a000c 020316ff 000000ff
	s_wait_loadcnt 0x8                                         // 0000000024c0: bfc00008
	v_cmp_eq_u32_e64 s13, 0xff, v140                           // 0000000024c4: d44a000d 020318ff 000000ff
	s_wait_loadcnt 0x7                                         // 0000000024d0: bfc00007
	v_cmp_eq_u32_e64 s14, 0xff, v141                           // 0000000024d4: d44a000e 02031aff 000000ff
	s_wait_loadcnt 0x6                                         // 0000000024e0: bfc00006
	v_cmp_eq_u32_e64 s15, 0xff, v142                           // 0000000024e4: d44a000f 02031cff 000000ff
	s_wait_loadcnt 0x5                                         // 0000000024f0: bfc00005
	v_add_nc_u32_e32 v117, 0xffffff02, v146                    // 0000000024f4: 4aeb24ff ffffff02
	s_wait_loadcnt 0x4                                         // 0000000024fc: bfc00004
	v_add_nc_u32_e32 v118, 0xffffff02, v148                    // 000000002500: 4aed28ff ffffff02
	s_wait_loadcnt 0x3                                         // 000000002508: bfc00003
	v_cmp_eq_u32_e64 s16, 0xff, v143                           // 00000000250c: d44a0010 02031eff 000000ff
	s_wait_loadcnt 0x2                                         // 000000002518: bfc00002
	v_cmp_eq_u32_e64 s17, 0xff, v144                           // 00000000251c: d44a0011 020320ff 000000ff
	s_wait_loadcnt 0x1                                         // 000000002528: bfc00001
	v_cmp_eq_u32_e64 s18, 0xff, v145                           // 00000000252c: d44a0012 020322ff 000000ff
	v_cmp_eq_u32_e64 s20, 0xff, v146                           // 000000002538: d44a0014 020324ff 000000ff
	v_cmp_eq_u32_e64 s21, 0xff, v148                           // 000000002544: d44a0015 020328ff 000000ff
	v_add_nc_u32_e32 v119, v117, v131                          // 000000002550: 4aef0775
	v_add_nc_u32_e32 v120, v117, v132                          // 000000002554: 4af10975
	v_add_nc_u32_e32 v121, v117, v133                          // 000000002558: 4af30b75
	v_add_nc_u32_e32 v122, v117, v134                          // 00000000255c: 4af50d75
	v_add_nc_u32_e32 v123, v117, v135                          // 000000002560: 4af70f75
	v_add_nc_u32_e32 v124, v117, v136                          // 000000002564: 4af91175
	v_add_nc_u32_e32 v125, v117, v137                          // 000000002568: 4afb1375
	v_add_nc_u32_e32 v126, v117, v138                          // 00000000256c: 4afd1575
	v_add_nc_u32_e32 v127, v118, v131                          // 000000002570: 4aff0776
	v_add_nc_u32_e32 v128, v118, v132                          // 000000002574: 4b010976
	v_add_nc_u32_e32 v129, v118, v133                          // 000000002578: 4b030b76
	v_add_nc_u32_e32 v130, v118, v134                          // 00000000257c: 4b050d76
	v_add_nc_u32_e32 v131, v118, v135                          // 000000002580: 4b070f76
	v_add_nc_u32_e32 v132, v118, v136                          // 000000002584: 4b091176
	v_add_nc_u32_e32 v133, v118, v137                          // 000000002588: 4b0b1376
	v_add_nc_u32_e32 v134, v118, v138                          // 00000000258c: 4b0d1576
	v_add_nc_u32_e32 v135, v117, v139                          // 000000002590: 4b0f1775
	v_add_nc_u32_e32 v136, v117, v140                          // 000000002594: 4b111975
	v_add_nc_u32_e32 v137, v117, v141                          // 000000002598: 4b131b75
	v_add_nc_u32_e32 v138, v117, v142                          // 00000000259c: 4b151d75
	v_add_nc_u32_e32 v146, v117, v143                          // 0000000025a0: 4b251f75
	v_add_nc_u32_e32 v148, v117, v144                          // 0000000025a4: 4b292175
	v_add_nc_u32_e32 v149, v117, v145                          // 0000000025a8: 4b2b2375
	s_wait_loadcnt 0x0                                         // 0000000025ac: bfc00000
	v_add_nc_u32_e32 v117, v117, v147                          // 0000000025b0: 4aeb2775
	v_add_nc_u32_e32 v139, v118, v139                          // 0000000025b4: 4b171776
	v_add_nc_u32_e32 v140, v118, v140                          // 0000000025b8: 4b191976
	v_add_nc_u32_e32 v141, v118, v141                          // 0000000025bc: 4b1b1b76
	v_add_nc_u32_e32 v142, v118, v142                          // 0000000025c0: 4b1d1d76
	v_add_nc_u32_e32 v143, v118, v143                          // 0000000025c4: 4b1f1f76
	v_add_nc_u32_e32 v144, v118, v144                          // 0000000025c8: 4b212176
	v_add_nc_u32_e32 v145, v118, v145                          // 0000000025cc: 4b232376
	v_add_nc_u32_e32 v118, v118, v147                          // 0000000025d0: 4aed2776
	v_cmp_eq_u32_e64 s19, 0xff, v147                           // 0000000025d4: d44a0013 020326ff 000000ff
	v_ldexp_f32 v0, v0, v119                                   // 0000000025e0: d71c0000 0202ef00
	v_ldexp_f32 v1, v1, v120                                   // 0000000025e8: d71c0001 0202f101
	v_ldexp_f32 v2, v2, v121                                   // 0000000025f0: d71c0002 0202f302
	v_ldexp_f32 v3, v3, v122                                   // 0000000025f8: d71c0003 0202f503
	v_ldexp_f32 v4, v4, v123                                   // 000000002600: d71c0004 0202f704
	v_ldexp_f32 v5, v5, v124                                   // 000000002608: d71c0005 0202f905
	v_ldexp_f32 v6, v6, v125                                   // 000000002610: d71c0006 0202fb06
	v_ldexp_f32 v7, v7, v126                                   // 000000002618: d71c0007 0202fd07
	v_ldexp_f32 v93, v93, v127                                 // 000000002620: d71c005d 0202ff5d
	v_ldexp_f32 v94, v94, v128                                 // 000000002628: d71c005e 0203015e
	v_ldexp_f32 v95, v95, v129                                 // 000000002630: d71c005f 0203035f
	v_ldexp_f32 v96, v96, v130                                 // 000000002638: d71c0060 02030560
	v_ldexp_f32 v97, v97, v131                                 // 000000002640: d71c0061 02030761
	v_ldexp_f32 v98, v98, v132                                 // 000000002648: d71c0062 02030962
	v_ldexp_f32 v99, v99, v133                                 // 000000002650: d71c0063 02030b63
	v_ldexp_f32 v100, v100, v134                               // 000000002658: d71c0064 02030d64
	v_ldexp_f32 v101, v101, v135                               // 000000002660: d71c0065 02030f65
	v_ldexp_f32 v102, v102, v136                               // 000000002668: d71c0066 02031166
	v_ldexp_f32 v103, v103, v137                               // 000000002670: d71c0067 02031367
	v_ldexp_f32 v104, v104, v138                               // 000000002678: d71c0068 02031568
	v_ldexp_f32 v105, v105, v146                               // 000000002680: d71c0069 02032569
	v_ldexp_f32 v106, v106, v148                               // 000000002688: d71c006a 0203296a
	v_ldexp_f32 v107, v107, v149                               // 000000002690: d71c006b 02032b6b
	v_ldexp_f32 v108, v108, v117                               // 000000002698: d71c006c 0202eb6c
	v_ldexp_f32 v109, v109, v139                               // 0000000026a0: d71c006d 0203176d
	v_ldexp_f32 v110, v110, v140                               // 0000000026a8: d71c006e 0203196e
	v_ldexp_f32 v111, v111, v141                               // 0000000026b0: d71c006f 02031b6f
	v_ldexp_f32 v112, v112, v142                               // 0000000026b8: d71c0070 02031d70
	v_ldexp_f32 v113, v113, v143                               // 0000000026c0: d71c0071 02031f71
	v_ldexp_f32 v114, v114, v144                               // 0000000026c8: d71c0072 02032172
	v_ldexp_f32 v115, v115, v145                               // 0000000026d0: d71c0073 02032373
	v_ldexp_f32 v116, v116, v118                               // 0000000026d8: d71c0074 0202ed74
	s_or_b32 s33, s4, s20                                      // 0000000026e0: 8c211404
	s_or_b32 s36, s20, s5                                      // 0000000026e4: 8c240514
	s_or_b32 s37, s20, s6                                      // 0000000026e8: 8c250614
	s_or_b32 s38, s20, s7                                      // 0000000026ec: 8c260714
	s_or_b32 s39, s20, s8                                      // 0000000026f0: 8c270814
	s_or_b32 s40, s20, s9                                      // 0000000026f4: 8c280914
	s_or_b32 s41, s20, s10                                     // 0000000026f8: 8c290a14
	s_or_b32 s42, s20, s11                                     // 0000000026fc: 8c2a0b14
	s_or_b32 s4, s4, s21                                       // 000000002700: 8c041504
	s_or_b32 s5, s5, s21                                       // 000000002704: 8c051505
	s_or_b32 s6, s6, s21                                       // 000000002708: 8c061506
	s_or_b32 s7, s7, s21                                       // 00000000270c: 8c071507
	s_or_b32 s8, s8, s21                                       // 000000002710: 8c081508
	s_or_b32 s9, s9, s21                                       // 000000002714: 8c091509
	s_or_b32 s10, s10, s21                                     // 000000002718: 8c0a150a
	s_or_b32 s11, s11, s21                                     // 00000000271c: 8c0b150b
	s_or_b32 s43, s20, s12                                     // 000000002720: 8c2b0c14
	s_or_b32 s44, s20, s13                                     // 000000002724: 8c2c0d14
	s_or_b32 s45, s20, s14                                     // 000000002728: 8c2d0e14
	s_or_b32 s46, s20, s15                                     // 00000000272c: 8c2e0f14
	s_or_b32 s47, s20, s16                                     // 000000002730: 8c2f1014
	s_or_b32 s48, s20, s17                                     // 000000002734: 8c301114
	s_or_b32 s49, s20, s18                                     // 000000002738: 8c311214
	s_or_b32 s20, s20, s19                                     // 00000000273c: 8c141314
	s_or_b32 s12, s21, s12                                     // 000000002740: 8c0c0c15
	s_or_b32 s13, s21, s13                                     // 000000002744: 8c0d0d15
	s_or_b32 s14, s21, s14                                     // 000000002748: 8c0e0e15
	s_or_b32 s15, s21, s15                                     // 00000000274c: 8c0f0f15
	s_or_b32 s16, s21, s16                                     // 000000002750: 8c101015
	s_or_b32 s17, s21, s17                                     // 000000002754: 8c111115
	s_or_b32 s18, s21, s18                                     // 000000002758: 8c121215
	s_or_b32 s19, s21, s19                                     // 00000000275c: 8c131315
	s_wait_alu depctr_sa_sdst(0)                               // 000000002760: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s33                  // 000000002764: d5010000 0085ff00 7fc00000
	v_cndmask_b32_e64 v1, v1, 0x7fc00000, s36                  // 000000002770: d5010001 0091ff01 7fc00000
	v_cndmask_b32_e64 v2, v2, 0x7fc00000, s37                  // 00000000277c: d5010002 0095ff02 7fc00000
	v_cndmask_b32_e64 v3, v3, 0x7fc00000, s38                  // 000000002788: d5010003 0099ff03 7fc00000
	v_cndmask_b32_e64 v4, v4, 0x7fc00000, s39                  // 000000002794: d5010004 009dff04 7fc00000
	v_cndmask_b32_e64 v5, v5, 0x7fc00000, s40                  // 0000000027a0: d5010005 00a1ff05 7fc00000
	v_cndmask_b32_e64 v6, v6, 0x7fc00000, s41                  // 0000000027ac: d5010006 00a5ff06 7fc00000
	v_cndmask_b32_e64 v7, v7, 0x7fc00000, s42                  // 0000000027b8: d5010007 00a9ff07 7fc00000
	v_cndmask_b32_e64 v93, v93, 0x7fc00000, s4                 // 0000000027c4: d501005d 0011ff5d 7fc00000
	v_cndmask_b32_e64 v94, v94, 0x7fc00000, s5                 // 0000000027d0: d501005e 0015ff5e 7fc00000
	v_cndmask_b32_e64 v95, v95, 0x7fc00000, s6                 // 0000000027dc: d501005f 0019ff5f 7fc00000
	v_cndmask_b32_e64 v96, v96, 0x7fc00000, s7                 // 0000000027e8: d5010060 001dff60 7fc00000
	v_cndmask_b32_e64 v97, v97, 0x7fc00000, s8                 // 0000000027f4: d5010061 0021ff61 7fc00000
	v_cndmask_b32_e64 v98, v98, 0x7fc00000, s9                 // 000000002800: d5010062 0025ff62 7fc00000
	v_cndmask_b32_e64 v99, v99, 0x7fc00000, s10                // 00000000280c: d5010063 0029ff63 7fc00000
	v_cndmask_b32_e64 v100, v100, 0x7fc00000, s11              // 000000002818: d5010064 002dff64 7fc00000
	v_cndmask_b32_e64 v101, v101, 0x7fc00000, s43              // 000000002824: d5010065 00adff65 7fc00000
	v_cndmask_b32_e64 v102, v102, 0x7fc00000, s44              // 000000002830: d5010066 00b1ff66 7fc00000
	v_cndmask_b32_e64 v103, v103, 0x7fc00000, s45              // 00000000283c: d5010067 00b5ff67 7fc00000
	v_cndmask_b32_e64 v104, v104, 0x7fc00000, s46              // 000000002848: d5010068 00b9ff68 7fc00000
	v_cndmask_b32_e64 v105, v105, 0x7fc00000, s47              // 000000002854: d5010069 00bdff69 7fc00000
	v_cndmask_b32_e64 v106, v106, 0x7fc00000, s48              // 000000002860: d501006a 00c1ff6a 7fc00000
	v_cndmask_b32_e64 v107, v107, 0x7fc00000, s49              // 00000000286c: d501006b 00c5ff6b 7fc00000
	v_cndmask_b32_e64 v108, v108, 0x7fc00000, s20              // 000000002878: d501006c 0051ff6c 7fc00000
	v_cndmask_b32_e64 v109, v109, 0x7fc00000, s12              // 000000002884: d501006d 0031ff6d 7fc00000
	v_cndmask_b32_e64 v110, v110, 0x7fc00000, s13              // 000000002890: d501006e 0035ff6e 7fc00000
	v_cndmask_b32_e64 v111, v111, 0x7fc00000, s14              // 00000000289c: d501006f 0039ff6f 7fc00000
	v_cndmask_b32_e64 v112, v112, 0x7fc00000, s15              // 0000000028a8: d5010070 003dff70 7fc00000
	v_cndmask_b32_e64 v113, v113, 0x7fc00000, s16              // 0000000028b4: d5010071 0041ff71 7fc00000
	v_cndmask_b32_e64 v114, v114, 0x7fc00000, s17              // 0000000028c0: d5010072 0045ff72 7fc00000
	v_cndmask_b32_e64 v115, v115, 0x7fc00000, s18              // 0000000028cc: d5010073 0049ff73 7fc00000
	v_cndmask_b32_e64 v116, v116, 0x7fc00000, s19              // 0000000028d8: d5010074 004dff74 7fc00000
	v_add_f32_e32 v64, v64, v0                                 // 0000000028e4: 06800140
	v_dual_add_f32 v90, v90, v1 :: v_dual_add_f32 v89, v89, v2 // 0000000028e8: c908035a 5a580559
	v_add_f32_e32 v87, v87, v3                                 // 0000000028f0: 06ae0757
	v_dual_add_f32 v85, v85, v4 :: v_dual_add_f32 v84, v84, v5 // 0000000028f4: c9080955 55540b54
	v_dual_add_f32 v83, v83, v6 :: v_dual_add_f32 v82, v82, v7 // 0000000028fc: c9080d53 53520f52
	v_dual_add_f32 v63, v63, v93 :: v_dual_add_f32 v62, v62, v94// 000000002904: c908bb3f 3f3ebd3e
	v_dual_add_f32 v61, v61, v95 :: v_dual_add_f32 v60, v60, v96// 00000000290c: c908bf3d 3d3cc13c
	v_dual_add_f32 v59, v59, v97 :: v_dual_add_f32 v58, v58, v98// 000000002914: c908c33b 3b3ac53a
	v_dual_add_f32 v57, v57, v99 :: v_dual_add_f32 v56, v56, v100// 00000000291c: c908c739 3938c938
	v_dual_add_f32 v79, v79, v101 :: v_dual_add_f32 v78, v78, v102// 000000002924: c908cb4f 4f4ecd4e
	v_add_f32_e32 v77, v77, v103                               // 00000000292c: 069acf4d
	v_add_f32_e32 v75, v75, v104                               // 000000002930: 0696d14b
	v_add_f32_e32 v69, v69, v105                               // 000000002934: 068ad345
	v_dual_add_f32 v67, v67, v106 :: v_dual_add_f32 v66, v66, v107// 000000002938: c908d543 4342d742
	v_add_f32_e32 v65, v65, v108                               // 000000002940: 0682d941
	v_dual_add_f32 v55, v55, v109 :: v_dual_add_f32 v54, v54, v110// 000000002944: c908db37 3736dd36
	v_dual_add_f32 v53, v53, v111 :: v_dual_add_f32 v52, v52, v112// 00000000294c: c908df35 3534e134
	v_dual_add_f32 v51, v51, v113 :: v_dual_add_f32 v50, v50, v114// 000000002954: c908e333 3332e532
	v_dual_add_f32 v49, v49, v115 :: v_dual_add_f32 v48, v48, v116// 00000000295c: c908e731 3130e930
	s_cmp_lg_u64 s[34:35], s[30:31]                            // 000000002964: bf111e22
	s_cbranch_scc0 36                                          // 000000002968: bfa10024 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0xefc>
	s_lshl_b64 s[6:7], s[34:35], 5                             // 00000000296c: 84868522
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0               // 000000002970: ca100080 02020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002978: bf88ff9e
	v_add_co_u32 v0, s4, v70, s6                               // 00000000297c: d7000400 02000d46
	s_wait_alu depctr_va_sdst(0)                               // 000000002984: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v71, s4                  // 000000002988: d5207c01 00128e07
	global_load_b128 v[4:7], v[0:1], off                       // 000000002990: ee05c07c 00000004 00000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 00000000299c: ca100080 00000080
	s_and_saveexec_b32 s5, s3                                  // 0000000029a4: be852003
	s_cbranch_execz 8                                          // 0000000029a8: bfa50008 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0xecc>
	v_add_co_u32 v0, s4, v72, s6                               // 0000000029ac: d7000400 02000d48
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v73, s4                  // 0000000029b8: d5207c01 00129207
	global_load_b128 v[0:3], v[0:1], off                       // 0000000029c0: ee05c07c 00000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000029d0: 8c7e057e
	s_barrier_signal -1                                        // 0000000029d4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000029d8: bf94ffff
	s_wait_loadcnt 0x0                                         // 0000000029dc: bfc00000
	ds_store_b128 v68, v[4:7]                                  // 0000000029e0: db7c0000 00000444
	s_and_saveexec_b32 s4, s3                                  // 0000000029e8: be842003
	s_cbranch_execz 64987                                      // 0000000029ec: bfa5fddb <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x65c>
	ds_store_b128 v68, v[0:3] offset:6144                      // 0000000029f0: db7c1800 00000044
	s_branch 64984                                             // 0000000029f8: bfa0fdd8 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x65c>
	s_load_b64 s[18:19], s[0:1], 0xa8                          // 0000000029fc: f4002480 f80000a8
	v_mul_lo_u32 v4, s27, v12                                  // 000000002a04: d72c0004 0202181b
	v_mul_lo_u32 v5, s26, v13                                  // 000000002a0c: d72c0005 02021a1a
	v_mad_co_u64_u32 v[2:3], null, s26, v12, 0                 // 000000002a14: d6fe7c02 0202181a
	v_sub_co_u32 v0, s0, s24, v12                              // 000000002a1c: d7010000 02021818
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002a24: bf870191
	v_sub_co_ci_u32_e64 v1, null, s25, v13, s0                 // 000000002a28: d5217c01 00021a19
	v_add3_u32 v3, v3, v5, v4                                  // 000000002a30: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a38: bf8701a2
	v_cmp_lt_i64_e64 s15, 0, v[0:1]                            // 000000002a3c: d451000f 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 000000002a44: 3e081081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002a48: 3e040481
	s_and_b32 s0, s15, s2                                      // 000000002a4c: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a50: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002a54: be812000
	s_cbranch_execz 28                                         // 000000002a58: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0xfcc>
	v_bfe_u32 v6, v64, 16, 1                                   // 000000002a5c: d6100006 02052140
	s_wait_kmcnt 0x0                                           // 000000002a64: bfc70000
	v_add_co_u32 v7, s0, s18, v2                               // 000000002a68: d7000007 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002a70: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s19, v3, s0                 // 000000002a74: d5207c0c 00020613
	v_add3_u32 v13, v6, v64, 0x7fff                            // 000000002a7c: d655000d 03fe8106 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002a88: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 000000002a8c: d7000006 02020907
	v_or_b32_e32 v14, 0x400000, v64                            // 000000002a94: 381c80ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002a9c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v12, v5, s0                  // 000000002aa0: d5207c07 00020b0c
	v_cmp_u_f32_e64 s0, v64, v64                               // 000000002aa8: d4180000 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ab4: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 000000002ab8: d501000c 00021d0d
	global_store_d16_hi_b16 v[6:7], v12, off                   // 000000002ac0: ee09407c 06000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000002acc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ad0: 8c7e017e
	v_add_co_u32 v6, s0, s26, v8                               // 000000002ad4: d7000006 0202101a
	s_wait_alu depctr_va_sdst(0)                               // 000000002adc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v9, s0                  // 000000002ae0: d5207c07 0002121b
	v_cmp_lt_i64_e64 s16, 1, v[0:1]                            // 000000002ae8: d4510010 02020081
	s_delay_alu instid0(valu_dep_2)                            // 000000002af0: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000002af4: 3e0c0c81
	s_and_b32 s0, s16, s2                                      // 000000002af8: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 000000002afc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002b00: be812000
	s_cbranch_execz 28                                         // 000000002b04: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1078>
	v_bfe_u32 v12, v90, 16, 1                                  // 000000002b08: d610000c 0205215a
	s_wait_kmcnt 0x0                                           // 000000002b10: bfc70000
	v_add_co_u32 v13, s0, s18, v2                              // 000000002b14: d700000d 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002b1c: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s19, v3, s0                 // 000000002b20: d5207c0e 00020613
	v_add3_u32 v15, v12, v90, 0x7fff                           // 000000002b28: d655000f 03feb50c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b34: bf870003
	v_add_co_u32 v12, s0, v13, v6                              // 000000002b38: d700000c 02020d0d
	v_or_b32_e32 v16, 0x400000, v90                            // 000000002b40: 3820b4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b48: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v7, s0                 // 000000002b4c: d5207c0d 00020f0e
	v_cmp_u_f32_e64 s0, v90, v90                               // 000000002b54: d4180000 0202b55a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b5c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b60: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002b64: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 000000002b6c: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002b7c: 8c7e017e
	s_lshl_b64 s[38:39], s[26:27], 1                           // 000000002b80: 84a6811a
	v_cmp_lt_i64_e64 s14, 2, v[0:1]                            // 000000002b84: d451000e 02020082
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b8c: bf88ff9e
	v_add_co_u32 v12, s0, s38, v8                              // 000000002b90: d700000c 02021026
	s_wait_alu depctr_va_sdst(0)                               // 000000002b98: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v9, s0                 // 000000002b9c: d5207c0d 00021227
	s_and_b32 s0, s14, s2                                      // 000000002ba4: 8b00020e
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002ba8: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bac: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002bb0: be812000
	s_cbranch_execz 28                                         // 000000002bb4: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1128>
	v_bfe_u32 v14, v89, 16, 1                                  // 000000002bb8: d610000e 02052159
	s_wait_kmcnt 0x0                                           // 000000002bc0: bfc70000
	v_add_co_u32 v15, s0, s18, v2                              // 000000002bc4: d700000f 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002bcc: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s19, v3, s0                 // 000000002bd0: d5207c10 00020613
	v_add3_u32 v17, v14, v89, 0x7fff                           // 000000002bd8: d6550011 03feb30e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002be4: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000002be8: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v89                            // 000000002bf0: 3824b2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002bf8: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000002bfc: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v89, v89                               // 000000002c04: d4180000 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 000000002c0c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c10: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002c14: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002c1c: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c2c: 8c7e017e
	s_mul_u64 s[36:37], s[26:27], 3                            // 000000002c30: aaa4831a
	v_cmp_lt_i64_e64 s13, 3, v[0:1]                            // 000000002c34: d451000d 02020083
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c3c: bf88ff9e
	v_add_co_u32 v14, s0, s36, v8                              // 000000002c40: d700000e 02021024
	s_wait_alu depctr_va_sdst(0)                               // 000000002c48: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v9, s0                 // 000000002c4c: d5207c0f 00021225
	s_and_b32 s0, s13, s2                                      // 000000002c54: 8b00020d
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002c58: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c5c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c60: be812000
	s_cbranch_execz 28                                         // 000000002c64: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x11d8>
	v_bfe_u32 v16, v87, 16, 1                                  // 000000002c68: d6100010 02052157
	s_wait_kmcnt 0x0                                           // 000000002c70: bfc70000
	v_add_co_u32 v17, s0, s18, v2                              // 000000002c74: d7000011 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002c7c: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v3, s0                 // 000000002c80: d5207c12 00020613
	v_add3_u32 v19, v16, v87, 0x7fff                           // 000000002c88: d6550013 03feaf10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c94: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002c98: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v87                            // 000000002ca0: 3828aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca8: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002cac: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v87, v87                               // 000000002cb4: d4180000 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 000000002cbc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002cc0: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000002cc4: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002ccc: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cd8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002cdc: 8c7e017e
	s_lshl_b64 s[34:35], s[26:27], 2                           // 000000002ce0: 84a2821a
	v_cmp_lt_i64_e64 s12, 4, v[0:1]                            // 000000002ce4: d451000c 02020084
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cec: bf88ff9e
	v_add_co_u32 v16, s0, s34, v8                              // 000000002cf0: d7000010 02021022
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf8: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v9, s0                 // 000000002cfc: d5207c11 00021223
	s_and_b32 s0, s12, s2                                      // 000000002d04: 8b00020c
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002d08: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d0c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002d10: be812000
	s_cbranch_execz 28                                         // 000000002d14: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1288>
	v_bfe_u32 v18, v85, 16, 1                                  // 000000002d18: d6100012 02052155
	s_wait_kmcnt 0x0                                           // 000000002d20: bfc70000
	v_add_co_u32 v19, s0, s18, v2                              // 000000002d24: d7000013 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002d2c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v3, s0                 // 000000002d30: d5207c14 00020613
	v_add3_u32 v21, v18, v85, 0x7fff                           // 000000002d38: d6550015 03feab12 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d44: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002d48: d7000012 02022113
	v_or_b32_e32 v22, 0x400000, v85                            // 000000002d50: 382caaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d58: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 000000002d5c: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v85, v85                               // 000000002d64: d4180000 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 000000002d6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d70: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 000000002d74: d5010014 00022d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002d7c: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d8c: 8c7e017e
	s_mul_u64 s[30:31], s[26:27], 5                            // 000000002d90: aa9e851a
	v_cmp_lt_i64_e64 s11, 5, v[0:1]                            // 000000002d94: d451000b 02020085
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d9c: bf88ff9e
	v_add_co_u32 v18, s0, s30, v8                              // 000000002da0: d7000012 0202101e
	s_wait_alu depctr_va_sdst(0)                               // 000000002da8: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s31, v9, s0                 // 000000002dac: d5207c13 0002121f
	s_and_b32 s0, s11, s2                                      // 000000002db4: 8b00020b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002db8: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dbc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002dc0: be812000
	s_cbranch_execz 28                                         // 000000002dc4: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1338>
	v_bfe_u32 v20, v84, 16, 1                                  // 000000002dc8: d6100014 02052154
	s_wait_kmcnt 0x0                                           // 000000002dd0: bfc70000
	v_add_co_u32 v21, s0, s18, v2                              // 000000002dd4: d7000015 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002ddc: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v3, s0                 // 000000002de0: d5207c16 00020613
	v_add3_u32 v23, v20, v84, 0x7fff                           // 000000002de8: d6550017 03fea914 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002df4: bf870003
	v_add_co_u32 v20, s0, v21, v18                             // 000000002df8: d7000014 02022515
	v_or_b32_e32 v24, 0x400000, v84                            // 000000002e00: 3830a8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e08: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v19, s0                // 000000002e0c: d5207c15 00022716
	v_cmp_u_f32_e64 s0, v84, v84                               // 000000002e14: d4180000 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 000000002e1c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e20: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 000000002e24: d5010016 00023117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000002e2c: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e38: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e3c: 8c7e017e
	s_mul_u64 s[28:29], s[26:27], 6                            // 000000002e40: aa9c861a
	v_cmp_lt_i64_e64 s9, 6, v[0:1]                             // 000000002e44: d4510009 02020086
	v_add_co_u32 v20, s0, s28, v8                              // 000000002e4c: d7000014 0202101c
	s_wait_alu depctr_va_sdst(0)                               // 000000002e54: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s29, v9, s0                 // 000000002e58: d5207c15 0002121d
	s_and_b32 s0, s9, s2                                       // 000000002e60: 8b000209
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000002e64: 3e282881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e68: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e6c: be812000
	s_cbranch_execz 28                                         // 000000002e70: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x13e4>
	v_bfe_u32 v22, v83, 16, 1                                  // 000000002e74: d6100016 02052153
	s_wait_kmcnt 0x0                                           // 000000002e7c: bfc70000
	v_add_co_u32 v23, s0, s18, v2                              // 000000002e80: d7000017 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002e88: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v3, s0                 // 000000002e8c: d5207c18 00020613
	v_add3_u32 v25, v22, v83, 0x7fff                           // 000000002e94: d6550019 03fea716 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002ea0: bf870003
	v_add_co_u32 v22, s0, v23, v20                             // 000000002ea4: d7000016 02022917
	v_or_b32_e32 v26, 0x400000, v83                            // 000000002eac: 3834a6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb4: bf88f19f
	v_add_co_ci_u32_e64 v23, null, v24, v21, s0                // 000000002eb8: d5207c17 00022b18
	v_cmp_u_f32_e64 s0, v83, v83                               // 000000002ec0: d4180000 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000002ec8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ecc: bf870001
	v_cndmask_b32_e64 v24, v25, v26, s0                        // 000000002ed0: d5010018 00023519
	global_store_d16_hi_b16 v[22:23], v24, off                 // 000000002ed8: ee09407c 0c000000 00000016
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ee8: 8c7e017e
	s_mul_u64 s[20:21], s[26:27], 7                            // 000000002eec: aa94871a
	v_cmp_lt_i64_e64 s8, 7, v[0:1]                             // 000000002ef0: d4510008 02020087
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ef8: bf88ff9e
	v_add_co_u32 v8, s0, s20, v8                               // 000000002efc: d7000008 02021014
	s_wait_alu depctr_va_sdst(0)                               // 000000002f04: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s21, v9, s0                  // 000000002f08: d5207c09 00021215
	s_and_b32 s0, s8, s2                                       // 000000002f10: 8b000208
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002f14: 3e101081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f18: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f1c: be812000
	s_cbranch_execz 28                                         // 000000002f20: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1494>
	v_bfe_u32 v0, v82, 16, 1                                   // 000000002f24: d6100000 02052152
	s_wait_kmcnt 0x0                                           // 000000002f2c: bfc70000
	v_add_co_u32 v1, s0, s18, v2                               // 000000002f30: d7000001 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002f38: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v3, s0                 // 000000002f3c: d5207c16 00020613
	v_add3_u32 v23, v0, v82, 0x7fff                            // 000000002f44: d6550017 03fea500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f50: bf870003
	v_add_co_u32 v0, s0, v1, v8                                // 000000002f54: d7000000 02021101
	v_or_b32_e32 v24, 0x400000, v82                            // 000000002f5c: 3830a4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f64: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v22, v9, s0                  // 000000002f68: d5207c01 00021316
	v_cmp_u_f32_e64 s0, v82, v82                               // 000000002f70: d4180000 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000002f78: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f7c: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 000000002f80: d5010016 00023117
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000002f88: ee09407c 0b000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f98: 8c7e017e
	v_mul_lo_u32 v22, s27, v10                                 // 000000002f9c: d72c0016 0202141b
	v_mul_lo_u32 v23, s26, v11                                 // 000000002fa4: d72c0017 0202161a
	v_mad_co_u64_u32 v[0:1], null, s26, v10, 0                 // 000000002fac: d6fe7c00 0202141a
	v_sub_co_u32 v10, s0, s24, v10                             // 000000002fb4: d701000a 02021418
	s_wait_alu depctr_va_sdst(0)                               // 000000002fbc: bf88f19f
	v_sub_co_ci_u32_e64 v11, null, s25, v11, s0                // 000000002fc0: d5217c0b 00021619
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000002fc8: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[10:11]                          // 000000002fcc: d451000a 02021480
	v_add3_u32 v1, v1, v23, v22                                // 000000002fd4: d6550001 045a2f01
	s_delay_alu instid0(valu_dep_1)                            // 000000002fdc: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002fe0: 3e000081
	s_and_b32 s0, s10, s2                                      // 000000002fe4: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002fec: be812000
	s_cbranch_execz 28                                         // 000000002ff0: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1564>
	s_wait_kmcnt 0x0                                           // 000000002ff4: bfc70000
	v_add_co_u32 v23, s0, s18, v0                              // 000000002ff8: d7000017 02020012
	v_bfe_u32 v22, v79, 16, 1                                  // 000000003000: d6100016 0205214f
	s_wait_alu depctr_va_sdst(0)                               // 000000003008: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v1, s0                 // 00000000300c: d5207c18 00020213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003014: bf870193
	v_add_co_u32 v4, s0, v23, v4                               // 000000003018: d7000004 02020917
	v_add3_u32 v22, v22, v79, 0x7fff                           // 000000003020: d6550016 03fe9f16 00007fff
	v_or_b32_e32 v25, 0x400000, v79                            // 00000000302c: 38329eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v24, v5, s0                  // 000000003038: d5207c05 00020b18
	v_cmp_u_f32_e64 s0, v79, v79                               // 000000003040: d4180000 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 000000003048: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000304c: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s0                        // 000000003050: d5010016 00023316
	global_store_d16_hi_b16 v[4:5], v22, off                   // 000000003058: ee09407c 0b000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003064: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003068: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[10:11]                           // 00000000306c: d4510007 02021481
	s_and_b32 s0, s7, s2                                       // 000000003074: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 000000003078: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000307c: be812000
	s_cbranch_execz 28                                         // 000000003080: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x15f4>
	v_bfe_u32 v4, v78, 16, 1                                   // 000000003084: d6100004 0205214e
	s_wait_kmcnt 0x0                                           // 00000000308c: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003090: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003098: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v1, s0                 // 00000000309c: d5207c16 00020213
	v_add3_u32 v23, v4, v78, 0x7fff                            // 0000000030a4: d6550017 03fe9d04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030b0: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 0000000030b4: d7000004 02020d05
	v_or_b32_e32 v24, 0x400000, v78                            // 0000000030bc: 38309cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v22, v7, s0                  // 0000000030c8: d5207c05 00020f16
	v_cmp_u_f32_e64 s0, v78, v78                               // 0000000030d0: d4180000 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030dc: bf870001
	v_cndmask_b32_e64 v6, v23, v24, s0                         // 0000000030e0: d5010006 00023117
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000030e8: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030f4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030f8: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[10:11]                           // 0000000030fc: d4510006 02021482
	s_and_b32 s0, s6, s2                                       // 000000003104: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 000000003108: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000310c: be812000
	s_cbranch_execz 28                                         // 000000003110: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1684>
	v_bfe_u32 v4, v77, 16, 1                                   // 000000003114: d6100004 0205214d
	s_wait_kmcnt 0x0                                           // 00000000311c: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003120: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003128: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 00000000312c: d5207c06 00020213
	v_add3_u32 v7, v4, v77, 0x7fff                             // 000000003134: d6550007 03fe9b04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003140: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000003144: d7000004 02021905
	v_or_b32_e32 v22, 0x400000, v77                            // 00000000314c: 382c9aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003154: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 000000003158: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v77, v77                               // 000000003160: d4180000 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000003168: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000316c: bf870001
	v_cndmask_b32_e64 v6, v7, v22, s0                          // 000000003170: d5010006 00022d07
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003178: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003184: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003188: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[10:11]                           // 00000000318c: d4510005 02021483
	s_and_b32 s0, s5, s2                                       // 000000003194: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 000000003198: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000319c: be812000
	s_cbranch_execz 28                                         // 0000000031a0: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1714>
	v_bfe_u32 v4, v75, 16, 1                                   // 0000000031a4: d6100004 0205214b
	s_wait_kmcnt 0x0                                           // 0000000031ac: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 0000000031b0: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 0000000031bc: d5207c06 00020213
	v_add3_u32 v7, v4, v75, 0x7fff                             // 0000000031c4: d6550007 03fe9704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031d0: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 0000000031d4: d7000004 02021d05
	v_or_b32_e32 v12, 0x400000, v75                            // 0000000031dc: 381896ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 0000000031e8: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v75, v75                               // 0000000031f0: d4180000 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031fc: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003200: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003208: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003218: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[10:11]                           // 00000000321c: d4510004 02021484
	s_and_b32 s0, s4, s2                                       // 000000003224: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003228: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000322c: be812000
	s_cbranch_execz 28                                         // 000000003230: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x17a4>
	v_bfe_u32 v4, v69, 16, 1                                   // 000000003234: d6100004 02052145
	s_wait_kmcnt 0x0                                           // 00000000323c: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003240: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003248: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 00000000324c: d5207c06 00020213
	v_add3_u32 v7, v4, v69, 0x7fff                             // 000000003254: d6550007 03fe8b04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003260: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003264: d7000004 02022105
	v_or_b32_e32 v12, 0x400000, v69                            // 00000000326c: 38188aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003274: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 000000003278: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v69, v69                               // 000000003280: d4180000 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000003288: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000328c: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003290: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003298: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032a8: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[10:11]                           // 0000000032ac: d4510003 02021485
	s_and_b32 s0, s3, s2                                       // 0000000032b4: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032b8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032bc: be812000
	s_cbranch_execz 28                                         // 0000000032c0: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1834>
	v_bfe_u32 v4, v67, 16, 1                                   // 0000000032c4: d6100004 02052143
	s_wait_kmcnt 0x0                                           // 0000000032cc: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 0000000032d0: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000032d8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 0000000032dc: d5207c06 00020213
	v_add3_u32 v7, v4, v67, 0x7fff                             // 0000000032e4: d6550007 03fe8704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000032f0: bf870003
	v_add_co_u32 v4, s0, v5, v18                               // 0000000032f4: d7000004 02022505
	v_or_b32_e32 v12, 0x400000, v67                            // 0000000032fc: 381886ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003304: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s0                  // 000000003308: d5207c05 00022706
	v_cmp_u_f32_e64 s0, v67, v67                               // 000000003310: d4180000 02028743
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000331c: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003320: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003328: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003334: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003338: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[10:11]                           // 00000000333c: d4510001 02021486
	s_and_b32 s0, s1, s2                                       // 000000003344: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 000000003348: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 00000000334c: be912000
	s_cbranch_execz 28                                         // 000000003350: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x18c4>
	v_bfe_u32 v4, v66, 16, 1                                   // 000000003354: d6100004 02052142
	s_wait_kmcnt 0x0                                           // 00000000335c: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003360: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003368: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 00000000336c: d5207c06 00020213
	v_add3_u32 v7, v4, v66, 0x7fff                             // 000000003374: d6550007 03fe8504 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003380: bf870003
	v_add_co_u32 v4, s0, v5, v20                               // 000000003384: d7000004 02022905
	v_or_b32_e32 v12, 0x400000, v66                            // 00000000338c: 381884ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003394: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s0                  // 000000003398: d5207c05 00022b06
	v_cmp_u_f32_e64 s0, v66, v66                               // 0000000033a0: d4180000 02028542
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033ac: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 0000000033b0: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000033b8: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000033c8: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[10:11]                           // 0000000033cc: d4510000 02021487
	s_and_b32 s2, s0, s2                                       // 0000000033d4: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d8: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 0000000033dc: be912002
	s_cbranch_execz 28                                         // 0000000033e0: bfa5001c <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1954>
	v_bfe_u32 v4, v65, 16, 1                                   // 0000000033e4: d6100004 02052141
	s_wait_kmcnt 0x0                                           // 0000000033ec: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 0000000033f0: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 0000000033fc: d5207c06 000a0213
	v_add3_u32 v7, v4, v65, 0x7fff                             // 000000003404: d6550007 03fe8304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003410: bf870003
	v_add_co_u32 v4, s2, v5, v8                                // 000000003414: d7000204 02021105
	v_or_b32_e32 v10, 0x400000, v65                            // 00000000341c: 381482ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003424: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s2                   // 000000003428: d5207c05 000a1306
	v_cmp_u_f32_e64 s2, v65, v65                               // 000000003430: d4180002 02028341
	s_wait_alu depctr_va_sdst(0)                               // 000000003438: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000343c: bf870001
	v_cndmask_b32_e64 v6, v7, v10, s2                          // 000000003440: d5010006 000a1507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003448: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003454: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003458: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 00000000345c: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003460: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003464: be8f2002
	s_cbranch_execz 40                                         // 000000003468: bfa50028 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1a0c>
	v_add_co_u32 v4, s2, v47, s22                              // 00000000346c: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003474: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003478: d5207c05 00082e80
	v_bfe_u32 v6, v63, 16, 1                                   // 000000003480: d6100006 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003488: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 00000000348c: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003494: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003498: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 0000000034a0: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000034a4: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ac: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000034b0: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000034b8: 3e080881
	v_add3_u32 v6, v6, v63, 0x7fff                             // 0000000034bc: d6550006 03fe7f06 00007fff
	v_or_b32_e32 v9, 0x400000, v63                             // 0000000034c8: 38127eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000034d0: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 0000000034d4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000034dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000034e0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v63, v63                               // 0000000034e8: d4180002 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034f4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000034f8: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003500: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000350c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000003510: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 000000003514: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000003518: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 00000000351c: be8f2002
	s_cbranch_execz 46                                         // 000000003520: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1adc>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003524: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000352c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003530: d5207c05 00082e80
	v_bfe_u32 v6, v62, 16, 1                                   // 000000003538: d6100006 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003540: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003544: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000354c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003550: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v62                             // 000000003558: 38127cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003560: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003564: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 00000000356c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000003570: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000003578: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 00000000357c: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003584: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003588: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003590: 3e080881
	v_add3_u32 v6, v6, v62, 0x7fff                             // 000000003594: d6550006 03fe7d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035a0: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000035a4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000035ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000035b0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v62, v62                               // 0000000035b8: d4180002 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035c4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000035c8: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000035d0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000035e0: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 0000000035e4: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e8: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 0000000035ec: be8e2002
	s_cbranch_execz 46                                         // 0000000035f0: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1bac>
	v_add_co_u32 v4, s2, v47, s22                              // 0000000035f4: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000035fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003600: d5207c05 00082e80
	v_bfe_u32 v6, v61, 16, 1                                   // 000000003608: d6100006 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003610: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003614: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000361c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003620: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v61                             // 000000003628: 38127aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003630: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 000000003634: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 00000000363c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 000000003640: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 000000003648: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 00000000364c: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003654: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003658: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003660: 3e080881
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000003664: d6550006 03fe7b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003670: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003674: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000367c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003680: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v61, v61                               // 000000003688: d4180002 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000003690: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003694: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003698: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000036a0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 0000000036b0: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 0000000036b4: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036b8: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 0000000036bc: be8d2002
	s_cbranch_execz 46                                         // 0000000036c0: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1c7c>
	v_add_co_u32 v4, s2, v47, s22                              // 0000000036c4: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000036cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000036d0: d5207c05 00082e80
	v_bfe_u32 v6, v60, 16, 1                                   // 0000000036d8: d6100006 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036e0: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 0000000036e4: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000036f0: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v60                             // 0000000036f8: 381278ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003700: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 000000003704: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 00000000370c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 000000003710: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 000000003718: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 00000000371c: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003724: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003728: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003730: 3e080881
	v_add3_u32 v6, v6, v60, 0x7fff                             // 000000003734: d6550006 03fe7906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003740: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003744: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000374c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003750: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v60, v60                               // 000000003758: d4180002 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 000000003760: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003764: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003768: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003770: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000377c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003780: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003784: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003788: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 00000000378c: be8c2002
	s_cbranch_execz 46                                         // 000000003790: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1d4c>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003794: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000379c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000037a0: d5207c05 00082e80
	v_bfe_u32 v6, v59, 16, 1                                   // 0000000037a8: d6100006 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037b0: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 0000000037b4: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000037bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000037c0: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v59                             // 0000000037c8: 381276ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037d0: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000037d4: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000037dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000037e0: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000037e8: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000037ec: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000037f8: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003800: 3e080881
	v_add3_u32 v6, v6, v59, 0x7fff                             // 000000003804: d6550006 03fe7706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003810: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003814: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000381c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003820: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v59, v59                               // 000000003828: d4180002 0202773b
	s_wait_alu depctr_va_sdst(0)                               // 000000003830: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003834: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003838: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003840: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000384c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003850: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 000000003854: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003858: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 00000000385c: be8b2002
	s_cbranch_execz 46                                         // 000000003860: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1e1c>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003864: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000386c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003870: d5207c05 00082e80
	v_bfe_u32 v6, v58, 16, 1                                   // 000000003878: d6100006 0205213a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003880: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003884: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000388c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003890: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v58                             // 000000003898: 381274ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038a0: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 0000000038a4: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000038ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 0000000038b0: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 0000000038b8: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000038bc: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000038c4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000038c8: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000038d0: 3e080881
	v_add3_u32 v6, v6, v58, 0x7fff                             // 0000000038d4: d6550006 03fe7506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038e0: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000038e4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000038ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000038f0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v58, v58                               // 0000000038f8: d4180002 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000003900: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003904: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003908: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003910: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000391c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 000000003920: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 000000003924: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000003928: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 00000000392c: be892002
	s_cbranch_execz 46                                         // 000000003930: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1eec>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003934: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000393c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003940: d5207c05 00082e80
	v_bfe_u32 v6, v57, 16, 1                                   // 000000003948: d6100006 02052139
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003950: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003954: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000395c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003960: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v57                             // 000000003968: 381272ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003970: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003974: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000003980: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000003988: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 00000000398c: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003994: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003998: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000039a0: 3e080881
	v_add3_u32 v6, v6, v57, 0x7fff                             // 0000000039a4: d6550006 03fe7306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039b0: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000039b4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000039bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000039c0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v57, v57                               // 0000000039c8: d4180002 02027339
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039d4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000039d8: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000039e0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000039f0: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 0000000039f4: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f8: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000039fc: be882002
	s_cbranch_execz 46                                         // 000000003a00: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x1fbc>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003a04: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a0c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003a10: d5207c05 00082e80
	v_bfe_u32 v6, v56, 16, 1                                   // 000000003a18: d6100006 02052138
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a20: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003a24: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003a2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003a30: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v56                             // 000000003a38: 380e70ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a40: bf8701a3
	v_add_co_u32 v4, s2, s20, v4                               // 000000003a44: d7000204 02020814
	s_wait_alu depctr_va_sdst(0)                               // 000000003a4c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v5, s2                  // 000000003a50: d5207c05 000a0a15
	s_wait_kmcnt 0x0                                           // 000000003a58: bfc70000
	v_add_co_u32 v2, s2, s18, v2                               // 000000003a5c: d7000202 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003a64: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s19, v3, s2                  // 000000003a68: d5207c03 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003a70: 3e080881
	v_add3_u32 v6, v6, v56, 0x7fff                             // 000000003a74: d6550006 03fe7106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a80: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000003a84: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000003a8c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 000000003a90: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v56, v56                               // 000000003a98: d4180002 02027138
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003aa4: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000003aa8: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ab0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003abc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003ac0: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 000000003ac4: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ac8: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 000000003acc: be882002
	s_cbranch_execz 40                                         // 000000003ad0: bfa50028 <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2074>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003ad4: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003adc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003ae0: d5207c03 00082e80
	v_bfe_u32 v4, v55, 16, 1                                   // 000000003ae8: d6100004 02052137
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003af0: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003af4: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003afc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003b00: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 000000003b08: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003b0c: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003b14: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003b18: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003b20: 3e040481
	v_add3_u32 v4, v4, v55, 0x7fff                             // 000000003b24: d6550004 03fe6f04 00007fff
	v_or_b32_e32 v7, 0x400000, v55                             // 000000003b30: 380e6eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003b38: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000003b3c: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003b44: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003b48: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v55, v55                               // 000000003b50: d4180002 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000003b58: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b5c: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003b60: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b68: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003b78: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000003b7c: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b80: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000003b84: be872002
	s_cbranch_execz 46                                         // 000000003b88: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2144>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003b8c: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003b94: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003b98: d5207c03 00082e80
	v_bfe_u32 v4, v54, 16, 1                                   // 000000003ba0: d6100004 02052136
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ba8: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003bac: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003bb8: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v54                             // 000000003bc0: 380e6cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bc8: bf8701a3
	v_add_co_u32 v2, s2, s26, v2                               // 000000003bcc: d7000202 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s2                  // 000000003bd8: d5207c03 000a061b
	s_wait_kmcnt 0x0                                           // 000000003be0: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003be4: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003bec: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003bf0: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003bf8: 3e040481
	v_add3_u32 v4, v4, v54, 0x7fff                             // 000000003bfc: d6550004 03fe6d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c08: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003c0c: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003c14: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003c18: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v54, v54                               // 000000003c20: d4180002 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003c28: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c2c: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003c30: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c38: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003c48: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000003c4c: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c50: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000003c54: be862002
	s_cbranch_execz 46                                         // 000000003c58: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2214>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003c5c: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003c64: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003c68: d5207c03 00082e80
	v_bfe_u32 v4, v53, 16, 1                                   // 000000003c70: d6100004 02052135
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c78: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003c7c: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003c84: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003c88: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v53                             // 000000003c90: 380e6aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c98: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000003c9c: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000003ca8: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000003cb0: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003cb4: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003cbc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003cc0: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003cc8: 3e040481
	v_add3_u32 v4, v4, v53, 0x7fff                             // 000000003ccc: d6550004 03fe6b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cd8: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003cdc: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003ce8: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v53, v53                               // 000000003cf0: d4180002 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003cfc: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003d00: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d08: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003d18: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 000000003d1c: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d20: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000003d24: be852002
	s_cbranch_execz 46                                         // 000000003d28: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x22e4>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003d2c: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003d34: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003d38: d5207c03 00082e80
	v_bfe_u32 v4, v52, 16, 1                                   // 000000003d40: d6100004 02052134
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d48: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003d4c: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003d54: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003d58: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v52                             // 000000003d60: 380e68ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d68: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000003d6c: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000003d74: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000003d78: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000003d80: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003d84: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003d8c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003d90: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003d98: 3e040481
	v_add3_u32 v4, v4, v52, 0x7fff                             // 000000003d9c: d6550004 03fe6904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003da8: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003dac: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003db4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003db8: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v52, v52                               // 000000003dc0: d4180002 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003dcc: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003dd0: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003dd8: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003de8: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 000000003dec: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003df0: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003df4: be842002
	s_cbranch_execz 46                                         // 000000003df8: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x23b4>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003dfc: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003e04: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003e08: d5207c03 00082e80
	v_bfe_u32 v4, v51, 16, 1                                   // 000000003e10: d6100004 02052133
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e18: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003e1c: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003e24: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003e28: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v51                             // 000000003e30: 380e66ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e38: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000003e3c: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000003e44: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000003e48: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000003e50: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003e54: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003e5c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003e60: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003e68: 3e040481
	v_add3_u32 v4, v4, v51, 0x7fff                             // 000000003e6c: d6550004 03fe6704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e78: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003e7c: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003e84: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003e88: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v51, v51                               // 000000003e90: d4180002 02026733
	s_wait_alu depctr_va_sdst(0)                               // 000000003e98: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e9c: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003ea0: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ea8: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003eb8: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000003ebc: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ec0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003ec4: be832002
	s_cbranch_execz 46                                         // 000000003ec8: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2484>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003ecc: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003ed4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003ed8: d5207c03 00082e80
	v_bfe_u32 v4, v50, 16, 1                                   // 000000003ee0: d6100004 02052132
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ee8: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003eec: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003ef4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003ef8: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v50                             // 000000003f00: 380e64ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f08: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000003f0c: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000003f14: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000003f18: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000003f20: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003f24: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003f2c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003f30: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003f38: 3e040481
	v_add3_u32 v4, v4, v50, 0x7fff                             // 000000003f3c: d6550004 03fe6504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f48: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003f4c: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003f54: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003f58: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v50, v50                               // 000000003f60: d4180002 02026532
	s_wait_alu depctr_va_sdst(0)                               // 000000003f68: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003f6c: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003f70: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003f78: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f84: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003f88: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000003f8c: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f90: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000003f94: be822001
	s_cbranch_execz 46                                         // 000000003f98: bfa5002e <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2554>
	v_add_co_u32 v2, s1, v47, s22                              // 000000003f9c: d7000102 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s1                   // 000000003fa8: d5207c03 00042e80
	v_bfe_u32 v4, v49, 16, 1                                   // 000000003fb0: d6100004 02052131
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fb8: bf8701a3
	v_add_co_u32 v2, s1, v2, v46                               // 000000003fbc: d7000102 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003fc4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000003fc8: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v49                             // 000000003fd0: 380e62ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fd8: bf8701a3
	v_add_co_u32 v2, s1, s28, v2                               // 000000003fdc: d7000102 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s1                  // 000000003fe8: d5207c03 0006061d
	s_wait_kmcnt 0x0                                           // 000000003ff0: bfc70000
	v_add_co_u32 v5, s1, s18, v0                               // 000000003ff4: d7000105 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003ffc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s1                  // 000000004000: d5207c06 00060213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004008: 3e040481
	v_add3_u32 v4, v4, v49, 0x7fff                             // 00000000400c: d6550004 03fe6304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004018: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 00000000401c: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004024: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000004028: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v49, v49                               // 000000004030: d4180001 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000004038: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000403c: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000004040: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004048: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004054: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004058: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 00000000405c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000004060: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004064: be812000
	s_cbranch_execz 43                                         // 000000004068: bfa5002b <tessera_rocm_scaled_matmul_lds_92651cf3b1e389ba+0x2618>
	v_add_co_u32 v2, s0, v47, s22                              // 00000000406c: d7000002 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000004074: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s0                   // 000000004078: d5207c03 00002e80
	v_bfe_u32 v4, v48, 16, 1                                   // 000000004080: d6100004 02052130
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004088: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v46                           // 00000000408c: d7006a02 02025d02
	s_wait_alu depctr_va_vcc(0)                                // 000000004094: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000004098: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v48                             // 0000000040a0: 380a60ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040a8: bf8701a3
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 0000000040ac: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000040b4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 0000000040b8: d5207c03 01aa0615
	s_wait_kmcnt 0x0                                           // 0000000040c0: bfc70000
	v_add_co_u32 v0, vcc_lo, s18, v0                           // 0000000040c4: d7006a00 02020012
	s_wait_alu depctr_va_vcc(0)                                // 0000000040cc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s19, v1, vcc_lo              // 0000000040d0: d5207c01 01aa0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000040d8: 3e040481
	v_add3_u32 v4, v4, v48, 0x7fff                             // 0000000040dc: d6550004 03fe6104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040e8: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000040ec: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000040f4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000040f8: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000004100: 7c306130
	s_wait_alu depctr_va_vcc(0)                                // 000000004104: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000004108: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000410c: ee09407c 01000000 00002000
	s_nop 0                                                    // 000000004118: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 00000000411c: bfb60003
	s_endpgm                                                   // 000000004120: bfb00000
	s_code_end                                                 // 000000004124: bf9f0000
	s_code_end                                                 // 000000004128: bf9f0000
	s_code_end                                                 // 00000000412c: bf9f0000
	s_code_end                                                 // 000000004130: bf9f0000
	s_code_end                                                 // 000000004134: bf9f0000
	s_code_end                                                 // 000000004138: bf9f0000
	s_code_end                                                 // 00000000413c: bf9f0000
	s_code_end                                                 // 000000004140: bf9f0000
	s_code_end                                                 // 000000004144: bf9f0000
	s_code_end                                                 // 000000004148: bf9f0000
	s_code_end                                                 // 00000000414c: bf9f0000
	s_code_end                                                 // 000000004150: bf9f0000
	s_code_end                                                 // 000000004154: bf9f0000
	s_code_end                                                 // 000000004158: bf9f0000
	s_code_end                                                 // 00000000415c: bf9f0000
	s_code_end                                                 // 000000004160: bf9f0000
	s_code_end                                                 // 000000004164: bf9f0000
	s_code_end                                                 // 000000004168: bf9f0000
	s_code_end                                                 // 00000000416c: bf9f0000
	s_code_end                                                 // 000000004170: bf9f0000
	s_code_end                                                 // 000000004174: bf9f0000
	s_code_end                                                 // 000000004178: bf9f0000
	s_code_end                                                 // 00000000417c: bf9f0000
	s_code_end                                                 // 000000004180: bf9f0000
	s_code_end                                                 // 000000004184: bf9f0000
	s_code_end                                                 // 000000004188: bf9f0000
	s_code_end                                                 // 00000000418c: bf9f0000
	s_code_end                                                 // 000000004190: bf9f0000
	s_code_end                                                 // 000000004194: bf9f0000
	s_code_end                                                 // 000000004198: bf9f0000
	s_code_end                                                 // 00000000419c: bf9f0000
	s_code_end                                                 // 0000000041a0: bf9f0000
	s_code_end                                                 // 0000000041a4: bf9f0000
	s_code_end                                                 // 0000000041a8: bf9f0000
	s_code_end                                                 // 0000000041ac: bf9f0000
	s_code_end                                                 // 0000000041b0: bf9f0000
	s_code_end                                                 // 0000000041b4: bf9f0000
	s_code_end                                                 // 0000000041b8: bf9f0000
	s_code_end                                                 // 0000000041bc: bf9f0000
	s_code_end                                                 // 0000000041c0: bf9f0000
	s_code_end                                                 // 0000000041c4: bf9f0000
	s_code_end                                                 // 0000000041c8: bf9f0000
	s_code_end                                                 // 0000000041cc: bf9f0000
	s_code_end                                                 // 0000000041d0: bf9f0000
	s_code_end                                                 // 0000000041d4: bf9f0000
	s_code_end                                                 // 0000000041d8: bf9f0000
	s_code_end                                                 // 0000000041dc: bf9f0000
	s_code_end                                                 // 0000000041e0: bf9f0000
	s_code_end                                                 // 0000000041e4: bf9f0000
	s_code_end                                                 // 0000000041e8: bf9f0000
	s_code_end                                                 // 0000000041ec: bf9f0000
	s_code_end                                                 // 0000000041f0: bf9f0000
	s_code_end                                                 // 0000000041f4: bf9f0000
	s_code_end                                                 // 0000000041f8: bf9f0000
	s_code_end                                                 // 0000000041fc: bf9f0000
	s_code_end                                                 // 000000004200: bf9f0000
	s_code_end                                                 // 000000004204: bf9f0000
	s_code_end                                                 // 000000004208: bf9f0000
	s_code_end                                                 // 00000000420c: bf9f0000
	s_code_end                                                 // 000000004210: bf9f0000
	s_code_end                                                 // 000000004214: bf9f0000
	s_code_end                                                 // 000000004218: bf9f0000
	s_code_end                                                 // 00000000421c: bf9f0000
	s_code_end                                                 // 000000004220: bf9f0000
	s_code_end                                                 // 000000004224: bf9f0000
	s_code_end                                                 // 000000004228: bf9f0000
	s_code_end                                                 // 00000000422c: bf9f0000
	s_code_end                                                 // 000000004230: bf9f0000
	s_code_end                                                 // 000000004234: bf9f0000
	s_code_end                                                 // 000000004238: bf9f0000
	s_code_end                                                 // 00000000423c: bf9f0000
	s_code_end                                                 // 000000004240: bf9f0000
	s_code_end                                                 // 000000004244: bf9f0000
	s_code_end                                                 // 000000004248: bf9f0000
	s_code_end                                                 // 00000000424c: bf9f0000
	s_code_end                                                 // 000000004250: bf9f0000
	s_code_end                                                 // 000000004254: bf9f0000
	s_code_end                                                 // 000000004258: bf9f0000
	s_code_end                                                 // 00000000425c: bf9f0000
	s_code_end                                                 // 000000004260: bf9f0000
	s_code_end                                                 // 000000004264: bf9f0000
	s_code_end                                                 // 000000004268: bf9f0000
	s_code_end                                                 // 00000000426c: bf9f0000
	s_code_end                                                 // 000000004270: bf9f0000
	s_code_end                                                 // 000000004274: bf9f0000
	s_code_end                                                 // 000000004278: bf9f0000
	s_code_end                                                 // 00000000427c: bf9f0000
	s_code_end                                                 // 000000004280: bf9f0000
	s_code_end                                                 // 000000004284: bf9f0000
	s_code_end                                                 // 000000004288: bf9f0000
	s_code_end                                                 // 00000000428c: bf9f0000
	s_code_end                                                 // 000000004290: bf9f0000
	s_code_end                                                 // 000000004294: bf9f0000
	s_code_end                                                 // 000000004298: bf9f0000
	s_code_end                                                 // 00000000429c: bf9f0000
	s_code_end                                                 // 0000000042a0: bf9f0000
	s_code_end                                                 // 0000000042a4: bf9f0000
	s_code_end                                                 // 0000000042a8: bf9f0000
	s_code_end                                                 // 0000000042ac: bf9f0000
	s_code_end                                                 // 0000000042b0: bf9f0000
	s_code_end                                                 // 0000000042b4: bf9f0000
	s_code_end                                                 // 0000000042b8: bf9f0000
	s_code_end                                                 // 0000000042bc: bf9f0000
	s_code_end                                                 // 0000000042c0: bf9f0000
	s_code_end                                                 // 0000000042c4: bf9f0000
	s_code_end                                                 // 0000000042c8: bf9f0000
	s_code_end                                                 // 0000000042cc: bf9f0000
	s_code_end                                                 // 0000000042d0: bf9f0000
	s_code_end                                                 // 0000000042d4: bf9f0000
	s_code_end                                                 // 0000000042d8: bf9f0000
	s_code_end                                                 // 0000000042dc: bf9f0000
	s_code_end                                                 // 0000000042e0: bf9f0000
	s_code_end                                                 // 0000000042e4: bf9f0000
	s_code_end                                                 // 0000000042e8: bf9f0000
	s_code_end                                                 // 0000000042ec: bf9f0000
	s_code_end                                                 // 0000000042f0: bf9f0000
	s_code_end                                                 // 0000000042f4: bf9f0000
	s_code_end                                                 // 0000000042f8: bf9f0000
	s_code_end                                                 // 0000000042fc: bf9f0000
