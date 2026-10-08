
/tmp/tmp4vw7w7kk.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[24:27], s[0:1], 0xc8                         // 000000001b04: f4004600 f80000c8
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b0c: f4002100 f80000d8
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b14: 320a0081
	s_mov_b32 s8, ttmp7                                        // 000000001b18: be880073
	s_ashr_i32 s9, ttmp7, 31                                   // 000000001b1c: 86099f73
	s_clause 0x3                                               // 000000001b20: bf850003
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001b24: f4002280 f8000008
	s_load_b64 s[12:13], s[0:1], 0x30                          // 000000001b2c: f4002300 f8000030
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000001b34: f4002180 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b3c: f4002700 f8000080
	s_lshl_b64 s[8:9], s[8:9], 7                               // 000000001b44: 84888708
	s_mov_b32 s2, ttmp9                                        // 000000001b48: be820075
	v_or_b32_e32 v1, s8, v5                                    // 000000001b4c: 38020a08
	v_dual_mov_b32 v2, s9 :: v_dual_lshlrev_b32 v3, 4, v0      // 000000001b50: ca220009 02020084
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b58: 86039f75
	v_and_b32_e32 v6, 0x60, v5                                 // 000000001b5c: 360c0aff 00000060
	s_lshl_b64 s[22:23], s[2:3], 6                             // 000000001b64: 84968602
	v_mul_u32_u24_e32 v11, 48, v5                              // 000000001b68: 16160ab0
	v_and_b32_e32 v12, 16, v3                                  // 000000001b6c: 36180690
	v_dual_mov_b32 v64, 0 :: v_dual_and_b32 v47, 32, v0        // 000000001b70: ca240080 402e00a0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v78, 0             // 000000001b78: ca100080 594e0080
	s_wait_kmcnt 0x0                                           // 000000001b80: bfc70000
	s_add_nc_u64 s[2:3], s[24:25], -1                          // 000000001b84: a982c118
	v_add_nc_u32_e32 v68, v11, v12                             // 000000001b88: 4a88190b
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[1:2]                    // 000000001b8c: 7cb80202
	v_and_b32_e32 v46, 15, v0                                  // 000000001b90: 365c008f
	s_lshr_b64 s[30:31], s[4:5], 5                             // 000000001b94: 859e8504
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v66, 0             // 000000001b98: ca100080 57420080
	v_mov_b32_e32 v85, 0                                       // 000000001ba0: 7eaa0280
	v_cndmask_b32_e32 v1, s2, v1, vcc_lo                       // 000000001ba4: 02020202
	v_cndmask_b32_e32 v10, s3, v2, vcc_lo                      // 000000001ba8: 02140403
	v_add_co_u32 v4, s2, s22, v5                               // 000000001bac: d7000204 02020a16
	s_wait_alu depctr_va_sdst(0)                               // 000000001bb4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s23, 0, s2                   // 000000001bb8: d5207c09 00090017
	v_mul_lo_u32 v13, v1, s5                                   // 000000001bc0: d72c000d 02000b01
	v_mad_co_u64_u32 v[1:2], null, v1, s4, s[10:11]            // 000000001bc8: d6fe7c01 00280901
	v_mul_lo_u32 v10, v10, s4                                  // 000000001bd0: d72c000a 0200090a
	s_delay_alu instid0(valu_dep_4)                            // 000000001bd8: bf870004
	v_mul_lo_u32 v9, s4, v9                                    // 000000001bdc: d72c0009 02021204
	v_mul_lo_u32 v11, s5, v4                                   // 000000001be4: d72c000b 02020805
	v_mad_co_u64_u32 v[3:4], null, s4, v4, s[12:13]            // 000000001bec: d6fe7c03 00320804
	v_cmp_gt_u32_e64 s3, 0x80, v0                              // 000000001bf4: d44c0003 020200ff 00000080
	v_and_b32_e32 v0, 47, v0                                   // 000000001c00: 360000af
	v_or_b32_e32 v8, v46, v47                                  // 000000001c04: 38105f2e
	v_add_co_u32 v70, vcc_lo, v1, v12                          // 000000001c08: d7006a46 02021901
	v_add3_u32 v2, v10, v2, v13                                // 000000001c10: d6550002 0436050a
	v_mov_b32_e32 v13, s9                                      // 000000001c18: 7e1a0209
	v_or_b32_e32 v7, 16, v6                                    // 000000001c1c: 380e0c90
	v_and_b32_e32 v10, 8, v5                                   // 000000001c20: 36140a88
	v_or_b32_e32 v22, s8, v6                                   // 000000001c24: 382c0c08
	v_add3_u32 v1, v11, v4, v9                                 // 000000001c28: d6550001 0426090b
	v_mul_u32_u24_e32 v0, 48, v0                               // 000000001c30: 160000b0
	v_or_b32_e32 v38, s8, v7                                   // 000000001c34: 384c0e08
	v_or_b32_e32 v7, v7, v46                                   // 000000001c38: 380e5d07
	s_wait_alu depctr_va_vcc(0)                                // 000000001c3c: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, 0, v2, vcc_lo               // 000000001c40: d5207c47 01aa0480
	v_add_co_u32 v72, vcc_lo, v3, v12                          // 000000001c48: d7006a48 02021903
	s_delay_alu instid0(valu_dep_3)                            // 000000001c50: bf870003
	v_mul_u32_u24_e32 v4, 48, v7                               // 000000001c54: 16080eb0
	v_or_b32_e32 v7, 1, v10                                    // 000000001c58: 380e1481
	s_wait_alu depctr_va_vcc(0)                                // 000000001c5c: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, 0, v1, vcc_lo               // 000000001c60: d5207c49 01aa0280
	v_or_b32_e32 v48, v10, v0                                  // 000000001c68: 3860010a
	v_mov_b32_e32 v1, s9                                       // 000000001c6c: 7e020209
	v_or_b32_e32 v0, v7, v22                                   // 000000001c70: 38002d07
	v_or_b32_e32 v6, v6, v46                                   // 000000001c74: 380c5d06
	s_lshr_b32 s8, s5, 5                                       // 000000001c78: 85088505
	v_mov_b32_e32 v9, s23                                      // 000000001c7c: 7e120217
	v_add_nc_u32_e32 v91, 0x1800, v48                          // 000000001c80: 4ab660ff 00001800
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000001c88: d4540004 02020018
	v_mul_u32_u24_e32 v2, 48, v6                               // 000000001c90: 16040cb0
	v_or_b32_e32 v6, 16, v8                                    // 000000001c94: 380c1090
	v_or_b32_e32 v8, s22, v8                                   // 000000001c98: 38101016
	v_mov_b32_e32 v83, 0                                       // 000000001c9c: 7ea60280
	v_mov_b32_e32 v63, 0                                       // 000000001ca0: 7e7e0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001ca4: bf88f19f
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 000000001ca8: d5010001 00120280
	v_cndmask_b32_e64 v0, 0, v0, s4                            // 000000001cb0: d5010000 00120080
	v_cmp_gt_i64_e64 s2, s[26:27], v[8:9]                      // 000000001cb8: d4540002 0202101a
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v48, 0             // 000000001cc0: ca100080 3d300080
	s_delay_alu instid0(valu_dep_4)                            // 000000001cc8: bf870004
	v_mul_lo_u32 v11, s30, v1                                  // 000000001ccc: d72c000b 0202021e
	v_mov_b32_e32 v1, s9                                       // 000000001cd4: 7e020209
	v_or_b32_e32 v12, v22, v10                                 // 000000001cd8: 38181516
	v_or_b32_e32 v74, v2, v10                                  // 000000001cdc: 38941502
	v_mul_u32_u24_e32 v2, 48, v6                               // 000000001ce0: 16040cb0
	v_or_b32_e32 v30, 2, v10                                   // 000000001ce4: 383c1482
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ce8: bf88ff9e
	v_mul_lo_u32 v18, s8, v0                                   // 000000001cec: d72c0012 02020008
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[12:13]                // 000000001cf4: 7ca81818
	v_mad_co_u64_u32 v[16:17], null, s30, v0, s[6:7]           // 000000001cf8: d6fe7c10 001a001e
	v_or_b32_e32 v49, v2, v10                                  // 000000001d00: 38621502
	v_or_b32_e32 v0, v30, v22                                  // 000000001d04: 38002d1e
	v_or_b32_e32 v76, v4, v10                                  // 000000001d08: 38981504
	v_or_b32_e32 v32, 3, v10                                   // 000000001d0c: 38401483
	s_wait_alu depctr_va_vcc(0)                                // 000000001d10: bf88ff9d
	v_dual_cndmask_b32 v2, 0, v12 :: v_dual_cndmask_b32 v3, 0, v13// 000000001d14: ca521880 02021a80
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001d1c: 7ca80018
	v_or_b32_e32 v33, 4, v10                                   // 000000001d20: 38421484
	v_or_b32_e32 v36, 5, v10                                   // 000000001d24: 38481485
	s_delay_alu instid0(valu_dep_4)                            // 000000001d28: bf870004
	v_mul_lo_u32 v5, s8, v2                                    // 000000001d2c: d72c0005 02020408
	v_mul_lo_u32 v4, s30, v3                                   // 000000001d34: d72c0004 0202061e
	v_mad_co_u64_u32 v[14:15], null, s30, v2, s[6:7]           // 000000001d3c: d6fe7c0e 001a041e
	v_mov_b32_e32 v3, s9                                       // 000000001d44: 7e060209
	v_or_b32_e32 v2, v32, v22                                  // 000000001d48: 38042d20
	v_or_b32_e32 v40, 7, v10                                   // 000000001d4c: 38501487
	v_add3_u32 v17, v18, v17, v11                              // 000000001d50: d6550011 042e2312
	v_or_b32_e32 v39, 6, v10                                   // 000000001d58: 384e1486
	v_or_b32_e32 v10, v38, v10                                 // 000000001d5c: 38141526
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 000000001d60: d4540004 02020418
	v_add3_u32 v15, v5, v15, v4                                // 000000001d68: d655000f 04121f05
	s_wait_alu depctr_va_vcc(0)                                // 000000001d70: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v0, vcc_lo                        // 000000001d74: 02080080
	v_or_b32_e32 v0, v33, v22                                  // 000000001d78: 38002d21
	v_cndmask_b32_e32 v5, 0, v1, vcc_lo                        // 000000001d7c: 020a0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001d80: bf88f19f
	v_cndmask_b32_e64 v80, 0, v9, s2                           // 000000001d84: d5010050 000a1280
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000001d8c: d5010003 00120680
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000001d94: d5010002 00120480
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001d9c: 7ca80018
	v_mul_lo_u32 v11, s30, v5                                  // 000000001da0: d72c000b 02020a1e
	v_mov_b32_e32 v84, 0                                       // 000000001da8: 7ea80280
	v_mul_lo_u32 v29, s30, v3                                  // 000000001dac: d72c001d 0202061e
	v_mul_lo_u32 v31, s8, v2                                   // 000000001db4: d72c001f 02020408
	v_mad_co_u64_u32 v[20:21], null, s30, v2, s[6:7]           // 000000001dbc: d6fe7c14 001a041e
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc4: bf88ff9d
	v_cndmask_b32_e32 v23, 0, v0, vcc_lo                       // 000000001dc8: 022e0080
	v_or_b32_e32 v0, v36, v22                                  // 000000001dcc: 38002d24
	v_cndmask_b32_e32 v3, 0, v1, vcc_lo                        // 000000001dd0: 02060280
	v_mul_lo_u32 v28, s8, v4                                   // 000000001dd4: d72c001c 02020808
	v_mad_co_u64_u32 v[18:19], null, s30, v4, s[6:7]           // 000000001ddc: d6fe7c12 001a081e
	v_or_b32_e32 v4, v40, v22                                  // 000000001de4: 38082d28
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[0:1]                  // 000000001de8: 7ca80018
	v_or_b32_e32 v2, v39, v22                                  // 000000001dec: 38042d27
	v_add3_u32 v21, v31, v21, v29                              // 000000001df0: d6550015 04762b1f
	v_mul_lo_u32 v35, s8, v23                                  // 000000001df8: d72c0023 02022e08
	v_mov_b32_e32 v90, 0                                       // 000000001e00: 7eb40280
	v_mad_co_u64_u32 v[22:23], null, s30, v23, s[6:7]          // 000000001e04: d6fe7c16 001a2e1e
	s_wait_alu depctr_va_vcc(0)                                // 000000001e0c: bf88ff9d
	v_dual_cndmask_b32 v0, 0, v0 :: v_dual_cndmask_b32 v1, 0, v1// 000000001e10: ca520080 00000280
	v_mul_lo_u32 v34, s30, v3                                  // 000000001e18: d72c0022 0202061e
	v_add3_u32 v19, v28, v19, v11                              // 000000001e20: d6550013 042e271c
	v_mov_b32_e32 v11, s9                                      // 000000001e28: 7e160209
	s_delay_alu instid0(valu_dep_4)                            // 000000001e2c: bf870004
	v_mul_lo_u32 v37, s8, v0                                   // 000000001e30: d72c0025 02020008
	v_mul_lo_u32 v1, s30, v1                                   // 000000001e38: d72c0001 0202021e
	v_mad_co_u64_u32 v[24:25], null, s30, v0, s[6:7]           // 000000001e40: d6fe7c18 001a001e
	v_cndmask_b32_e64 v81, 0, v8, s2                           // 000000001e48: d5010051 000a1080
	v_mov_b32_e32 v59, 0                                       // 000000001e50: 7e760280
	v_add3_u32 v23, v35, v23, v34                              // 000000001e54: d6550017 048a2f23
	v_mov_b32_e32 v79, 0                                       // 000000001e5c: 7e9e0280
	v_mov_b32_e32 v77, 0                                       // 000000001e60: 7e9a0280
	v_mov_b32_e32 v75, 0                                       // 000000001e64: 7e960280
	v_mov_b32_e32 v69, 0                                       // 000000001e68: 7e8a0280
	v_add3_u32 v25, v37, v25, v1                               // 000000001e6c: d6550019 04063325
	v_mov_b32_e32 v1, s23                                      // 000000001e74: 7e020217
	v_mov_b32_e32 v5, s9                                       // 000000001e78: 7e0a0209
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v92, 0x1800, v49// 000000001e7c: ca200080 435c62ff 00001800
	v_mov_b32_e32 v65, 0                                       // 000000001e88: 7e820280
	v_mov_b32_e32 v49, 0                                       // 000000001e8c: 7e620280
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000001e90: bf8701d4
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001e94: 7ca80818
	v_dual_mov_b32 v3, s9 :: v_dual_mov_b32 v82, 0             // 000000001e98: ca100009 03520080
	s_mov_b64 s[34:35], 0                                      // 000000001ea0: bea20180
	v_mov_b32_e32 v62, 0                                       // 000000001ea4: 7e7c0280
	v_mov_b32_e32 v60, 0                                       // 000000001ea8: 7e780280
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 000000001eac: d4540004 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000001eb4: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000001eb8: bf8701b1
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 000000001ebc: d5010003 00120680
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 000000001ec4: d5010002 00120480
	v_cmp_gt_i64_e64 s4, s[24:25], v[10:11]                    // 000000001ecc: d4540004 02021418
	v_mul_lo_u32 v0, s30, v3                                   // 000000001ed4: d72c0000 0202061e
	s_wait_alu depctr_va_vcc(0)                                // 000000001edc: bf88ff9d
	v_dual_cndmask_b32 v3, 0, v4 :: v_dual_cndmask_b32 v4, 0, v5// 000000001ee0: ca520880 03040a80
	v_mul_lo_u32 v5, s8, v2                                    // 000000001ee8: d72c0005 02020408
	v_mad_co_u64_u32 v[26:27], null, s30, v2, s[6:7]           // 000000001ef0: d6fe7c1a 001a041e
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_4)// 000000001ef8: bf870213
	v_mad_co_u64_u32 v[28:29], null, s30, v3, s[6:7]           // 000000001efc: d6fe7c1c 001a061e
	v_mul_lo_u32 v2, s30, v4                                   // 000000001f04: d72c0002 0202081e
	v_mul_lo_u32 v4, s8, v3                                    // 000000001f0c: d72c0004 02020608
	v_mov_b32_e32 v3, s9                                       // 000000001f14: 7e060209
	v_add3_u32 v27, v5, v27, v0                                // 000000001f18: d655001b 04023705
	v_or_b32_e32 v0, s22, v6                                   // 000000001f20: 38000c16
	v_mov_b32_e32 v5, s9                                       // 000000001f24: 7e0a0209
	s_wait_alu depctr_va_sdst(0)                               // 000000001f28: bf88f19f
	v_cndmask_b32_e64 v6, 0, v11, s4                           // 000000001f2c: d5010006 00121680
	v_add3_u32 v29, v4, v29, v2                                // 000000001f34: d655001d 040a3b04
	v_or_b32_e32 v2, v38, v7                                   // 000000001f3c: 38040f26
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[0:1]                  // 000000001f40: 7ca8001a
	v_or_b32_e32 v4, v38, v30                                  // 000000001f44: 38083d26
	v_mul_lo_u32 v6, s30, v6                                   // 000000001f48: d72c0006 02020c1e
	s_delay_alu instid0(valu_dep_4)                            // 000000001f50: bf870004
	v_cmp_gt_i64_e64 s5, s[24:25], v[2:3]                      // 000000001f54: d4540005 02020418
	s_wait_alu depctr_va_vcc(0)                                // 000000001f5c: bf88ff9d
	v_cndmask_b32_e32 v86, 0, v1, vcc_lo                       // 000000001f60: 02ac0280
	v_cndmask_b32_e64 v1, 0, v10, s4                           // 000000001f64: d5010001 00121480
	v_cndmask_b32_e32 v88, 0, v0, vcc_lo                       // 000000001f6c: 02b00080
	v_cmp_gt_i64_e64 s4, s[24:25], v[4:5]                      // 000000001f70: d4540004 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000001f78: bf88f19f
	v_cndmask_b32_e64 v0, 0, v3, s5                            // 000000001f7c: d5010000 00160680
	v_cndmask_b32_e64 v7, 0, v2, s5                            // 000000001f84: d5010007 00160480
	v_mul_lo_u32 v50, s8, v1                                   // 000000001f8c: d72c0032 02020208
	v_mad_co_u64_u32 v[30:31], null, s30, v1, s[6:7]           // 000000001f94: d6fe7c1e 001a021e
	v_mov_b32_e32 v1, s9                                       // 000000001f9c: 7e020209
	v_mul_lo_u32 v51, s30, v0                                  // 000000001fa0: d72c0033 0202001e
	v_or_b32_e32 v0, v38, v32                                  // 000000001fa8: 38004126
	v_cndmask_b32_e64 v4, 0, v4, s4                            // 000000001fac: d5010004 00120880
	v_cndmask_b32_e64 v5, 0, v5, s4                            // 000000001fb4: d5010005 00120a80
	v_or_b32_e32 v2, v38, v33                                  // 000000001fbc: 38044326
	v_mul_lo_u32 v52, s8, v7                                   // 000000001fc0: d72c0034 02020e08
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000001fc8: d4540004 02020018
	v_mul_lo_u32 v53, s8, v4                                   // 000000001fd0: d72c0035 02020808
	v_mad_co_u64_u32 v[34:35], null, s30, v4, s[6:7]           // 000000001fd8: d6fe7c22 001a081e
	v_mad_co_u64_u32 v[32:33], null, s30, v7, s[6:7]           // 000000001fe0: d6fe7c20 001a0e1e
	v_mul_lo_u32 v7, s30, v5                                   // 000000001fe8: d72c0007 02020a1e
	v_cmp_gt_i64_e64 s5, s[24:25], v[2:3]                      // 000000001ff0: d4540005 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff8: bf88f19f
	v_cndmask_b32_e64 v4, 0, v0, s4                            // 000000001ffc: d5010004 00120080
	v_or_b32_e32 v0, v38, v36                                  // 000000002004: 38004926
	v_cndmask_b32_e64 v5, 0, v1, s4                            // 000000002008: d5010005 00120280
	v_add3_u32 v31, v50, v31, v6                               // 000000002010: d655001f 041a3f32
	v_mov_b32_e32 v50, 0                                       // 000000002018: 7e640280
	v_cndmask_b32_e64 v41, 0, v2, s5                           // 00000000201c: d5010029 00160480
	v_cmp_gt_i64_e64 s4, s[24:25], v[0:1]                      // 000000002024: d4540004 02020018
	v_cndmask_b32_e64 v2, 0, v3, s5                            // 00000000202c: d5010002 00160680
	v_mul_lo_u32 v55, s8, v4                                   // 000000002034: d72c0037 02020808
	v_mad_co_u64_u32 v[36:37], null, s30, v4, s[6:7]           // 00000000203c: d6fe7c24 001a081e
	v_or_b32_e32 v4, v38, v40                                  // 000000002044: 38085126
	v_mul_lo_u32 v57, s8, v41                                  // 000000002048: d72c0039 02025208
	s_wait_alu depctr_va_sdst(0)                               // 000000002050: bf88f19f
	v_cndmask_b32_e64 v0, 0, v0, s4                            // 000000002054: d5010000 00120080
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 00000000205c: d5010001 00120280
	v_mul_lo_u32 v56, s30, v2                                  // 000000002064: d72c0038 0202041e
	v_or_b32_e32 v2, v38, v39                                  // 00000000206c: 38044f26
	v_mad_co_u64_u32 v[38:39], null, s30, v41, s[6:7]          // 000000002070: d6fe7c26 001a521e
	v_mul_lo_u32 v58, s8, v0                                   // 000000002078: d72c003a 02020008
	v_mul_lo_u32 v1, s30, v1                                   // 000000002080: d72c0001 0202021e
	v_mad_co_u64_u32 v[40:41], null, s30, v0, s[6:7]           // 000000002088: d6fe7c28 001a001e
	v_mul_lo_u32 v54, s30, v5                                  // 000000002090: d72c0036 02020a1e
	v_mov_b32_e32 v5, s9                                       // 000000002098: 7e0a0209
	v_cmp_gt_i64_e64 s4, s[24:25], v[2:3]                      // 00000000209c: d4540004 02020418
	v_add3_u32 v33, v52, v33, v51                              // 0000000020a4: d6550021 04ce4334
	v_add3_u32 v35, v53, v35, v7                               // 0000000020ac: d6550023 041e4735
	v_add3_u32 v39, v57, v39, v56                              // 0000000020b4: d6550027 04e24f39
	v_mov_b32_e32 v57, 0                                       // 0000000020bc: 7e720280
	v_add3_u32 v41, v58, v41, v1                               // 0000000020c0: d6550029 0406533a
	v_mov_b32_e32 v58, 0                                       // 0000000020c8: 7e740280
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 0000000020cc: d4540005 02020818
	s_wait_alu depctr_va_sdst(0)                               // 0000000020d4: bf88f19f
	v_cndmask_b32_e64 v2, 0, v2, s4                            // 0000000020d8: d5010002 00120480
	v_cndmask_b32_e64 v3, 0, v3, s4                            // 0000000020e0: d5010003 00120680
	v_add3_u32 v37, v55, v37, v54                              // 0000000020e8: d6550025 04da4b37
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0             // 0000000020f0: ca100080 38360080
	v_cndmask_b32_e64 v4, 0, v4, s5                            // 0000000020f8: d5010004 00160880
	v_cndmask_b32_e64 v5, 0, v5, s5                            // 000000002100: d5010005 00160a80
	v_mul_lo_u32 v0, s30, v3                                   // 000000002108: d72c0000 0202061e
	v_mul_lo_u32 v3, s8, v2                                    // 000000002110: d72c0003 02020408
	v_mad_co_u64_u32 v[42:43], null, s30, v2, s[6:7]           // 000000002118: d6fe7c2a 001a041e
	v_mad_co_u64_u32 v[44:45], null, s30, v4, s[6:7]           // 000000002120: d6fe7c2c 001a081e
	v_mul_lo_u32 v2, s30, v5                                   // 000000002128: d72c0002 02020a1e
	v_mul_lo_u32 v5, s8, v4                                    // 000000002130: d72c0005 02020808
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0             // 000000002138: ca100080 36340080
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0             // 000000002140: ca100080 34320080
	v_add3_u32 v43, v3, v43, v0                                // 000000002148: d655002b 04025703
	s_delay_alu instid0(valu_dep_4)                            // 000000002150: bf870004
	v_add3_u32 v45, v5, v45, v2                                // 000000002154: d655002d 040a5b05
	s_branch 516                                               // 00000000215c: bfa00204 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0xe70>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002160: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002164: 8c7e047e
	v_add_co_u32 v0, s4, v14, s34                              // 000000002168: d7000400 0200450e
	s_wait_alu depctr_va_sdst(0)                               // 000000002170: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v15, s4                 // 000000002174: d5207c01 00121e23
	s_mul_u64 s[4:5], s[34:35], s[26:27]                       // 00000000217c: aa841a22
	s_wait_dscnt 0x0                                           // 000000002180: bfc60000
	s_barrier_signal -1                                        // 000000002184: be804ec1
	s_barrier_wait 0xffff                                      // 000000002188: bf94ffff
	global_load_u8 v131, v[0:1], off                           // 00000000218c: ee04007c 00000083 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002198: bf88ff9e
	s_add_nc_u64 s[6:7], s[28:29], s[4:5]                      // 00000000219c: a986041c
	v_add_co_u32 v0, s4, v16, s34                              // 0000000021a0: d7000400 02004510
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v17, s4                 // 0000000021ac: d5207c01 00122223
	v_add_co_u32 v2, s4, v18, s34                              // 0000000021b4: d7000402 02004512
	s_wait_alu depctr_va_sdst(0)                               // 0000000021bc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v19, s4                 // 0000000021c0: d5207c03 00122623
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021c8: bf88ff9e
	v_add_co_u32 v4, s4, s6, v81                               // 0000000021cc: d7000404 0202a206
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s7, v80, s4                  // 0000000021d8: d5207c05 0012a007
	global_load_u8 v132, v[0:1], off                           // 0000000021e0: ee04007c 00000084 00000000
	v_add_co_u32 v0, s4, v20, s34                              // 0000000021ec: d7000400 02004514
	global_load_u8 v133, v[2:3], off                           // 0000000021f4: ee04007c 00000085 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002200: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s35, v21, s4                 // 000000002204: d5207c01 00122a23
	v_add_co_u32 v2, s4, v22, s34                              // 00000000220c: d7000402 02004516
	s_wait_alu depctr_va_sdst(0)                               // 000000002214: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v23, s4                 // 000000002218: d5207c03 00122e23
	v_add_co_u32 v6, s4, v24, s34                              // 000000002220: d7000406 02004518
	s_wait_alu depctr_va_sdst(0)                               // 000000002228: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v25, s4                 // 00000000222c: d5207c07 00123223
	v_add_co_u32 v93, s4, v26, s34                             // 000000002234: d700045d 0200451a
	s_wait_alu depctr_va_sdst(0)                               // 00000000223c: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v27, s4                // 000000002240: d5207c5e 00123623
	v_add_co_u32 v95, s4, v28, s34                             // 000000002248: d700045f 0200451c
	global_load_u8 v135, v[2:3], off                           // 000000002250: ee04007c 00000087 00000002
	v_add_co_u32 v2, s5, v30, s34                              // 00000000225c: d7000502 0200451e
	s_wait_alu depctr_va_sdst(0)                               // 000000002264: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v29, s4                // 000000002268: d5207c60 00123a23
	global_load_u8 v136, v[6:7], off                           // 000000002270: ee04007c 00000088 00000006
	v_add_co_ci_u32_e64 v3, null, s35, v31, s5                 // 00000000227c: d5207c03 00163e23
	v_add_co_u32 v6, s5, v32, s34                              // 000000002284: d7000506 02004520
	global_load_u8 v137, v[93:94], off                         // 00000000228c: ee04007c 00000089 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 000000002298: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v33, s5                 // 00000000229c: d5207c07 00164223
	v_add_co_u32 v93, s5, v34, s34                             // 0000000022a4: d700055d 02004522
	global_load_u8 v134, v[0:1], off                           // 0000000022ac: ee04007c 00000086 00000000
	v_add_co_u32 v0, s4, s6, v88                               // 0000000022b8: d7000400 0202b006
	global_load_u8 v138, v[95:96], off                         // 0000000022c0: ee04007c 0000008a 0000005f
	s_wait_alu depctr_va_sdst(0)                               // 0000000022cc: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v35, s5                // 0000000022d0: d5207c5e 00164623
	v_add_co_u32 v95, s5, v36, s34                             // 0000000022d8: d700055f 02004524
	v_add_co_ci_u32_e64 v1, null, s7, v86, s4                  // 0000000022e0: d5207c01 0012ac07
	global_load_u8 v139, v[2:3], off                           // 0000000022e8: ee04007c 0000008b 00000002
	v_add_co_u32 v2, s4, v38, s34                              // 0000000022f4: d7000402 02004526
	s_wait_alu depctr_va_sdst(0)                               // 0000000022fc: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v37, s5                // 000000002300: d5207c60 00164a23
	global_load_u8 v140, v[6:7], off                           // 000000002308: ee04007c 0000008c 00000006
	v_add_co_ci_u32_e64 v3, null, s35, v39, s4                 // 000000002314: d5207c03 00124e23
	v_add_co_u32 v6, s4, v40, s34                              // 00000000231c: d7000406 02004528
	global_load_u8 v141, v[93:94], off                         // 000000002324: ee04007c 0000008d 0000005d
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v41, s4                 // 000000002334: d5207c07 00125223
	v_add_co_u32 v93, s4, v42, s34                             // 00000000233c: d700045d 0200452a
	global_load_u8 v142, v[95:96], off                         // 000000002344: ee04007c 0000008e 0000005f
	s_wait_alu depctr_va_sdst(0)                               // 000000002350: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s35, v43, s4                // 000000002354: d5207c5e 00125623
	v_add_co_u32 v95, s4, v44, s34                             // 00000000235c: d700045f 0200452c
	s_wait_alu depctr_va_sdst(0)                               // 000000002364: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s35, v45, s4                // 000000002368: d5207c60 00125a23
	s_clause 0x1                                               // 000000002370: bf850001
	global_load_u8 v146, v[4:5], off                           // 000000002374: ee04007c 00000092 00000004
	global_load_u8 v148, v[0:1], off                           // 000000002380: ee04007c 00000094 00000000
	s_clause 0x3                                               // 00000000238c: bf850003
	global_load_u8 v143, v[2:3], off                           // 000000002390: ee04007c 0000008f 00000002
	global_load_u8 v144, v[6:7], off                           // 00000000239c: ee04007c 00000090 00000006
	global_load_u8 v145, v[93:94], off                         // 0000000023a8: ee04007c 00000091 0000005d
	global_load_u8 v147, v[95:96], off                         // 0000000023b4: ee04007c 00000093 0000005f
	ds_load_2addr_b64 v[115:118], v74 offset1:2                // 0000000023c0: d9dc0200 7300004a
	ds_load_2addr_b64 v[119:122], v91 offset1:2                // 0000000023c8: d9dc0200 7700005b
	ds_load_2addr_b64 v[123:126], v92 offset1:2                // 0000000023d0: d9dc0200 7b00005c
	ds_load_2addr_b64 v[127:130], v76 offset1:2                // 0000000023d8: d9dc0200 7f00004c
	s_add_nc_u64 s[34:35], s[34:35], 1                         // 0000000023e0: a9a28122
	s_wait_dscnt 0x2                                           // 0000000023e4: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[119:120], 0// 0000000023e8: cc464000 1a02ef73
	s_wait_dscnt 0x1                                           // 0000000023f0: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[115:116], v[123:124], 0// 0000000023f4: cc46405d 1a02f773
	s_wait_dscnt 0x0                                           // 0000000023fc: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[127:128], v[119:120], 0// 000000002400: cc464065 1a02ef7f
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[127:128], v[123:124], 0// 000000002408: cc46406d 1a02f77f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[117:118], v[121:122], v[0:7]// 000000002410: cc464000 1c02f375
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[117:118], v[125:126], v[93:100]// 000000002418: cc46405d 1d76fb75
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002420: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[129:130], v[121:122], v[101:108]// 000000002424: cc464065 1d96f381
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[129:130], v[125:126], v[109:116]// 00000000242c: cc46406d 1db6fb81
	s_wait_loadcnt 0x11                                        // 000000002434: bfc00011
	v_cmp_eq_u32_e64 s4, 0xff, v131                            // 000000002438: d44a0004 020306ff 000000ff
	s_wait_loadcnt 0x10                                        // 000000002444: bfc00010
	v_cmp_eq_u32_e64 s5, 0xff, v132                            // 000000002448: d44a0005 020308ff 000000ff
	s_wait_loadcnt 0xf                                         // 000000002454: bfc0000f
	v_cmp_eq_u32_e64 s6, 0xff, v133                            // 000000002458: d44a0006 02030aff 000000ff
	s_wait_loadcnt 0xe                                         // 000000002464: bfc0000e
	v_cmp_eq_u32_e64 s8, 0xff, v135                            // 000000002468: d44a0008 02030eff 000000ff
	s_wait_loadcnt 0xd                                         // 000000002474: bfc0000d
	v_cmp_eq_u32_e64 s9, 0xff, v136                            // 000000002478: d44a0009 020310ff 000000ff
	s_wait_loadcnt 0xc                                         // 000000002484: bfc0000c
	v_cmp_eq_u32_e64 s10, 0xff, v137                           // 000000002488: d44a000a 020312ff 000000ff
	s_wait_loadcnt 0xb                                         // 000000002494: bfc0000b
	v_cmp_eq_u32_e64 s7, 0xff, v134                            // 000000002498: d44a0007 02030cff 000000ff
	s_wait_loadcnt 0xa                                         // 0000000024a4: bfc0000a
	v_cmp_eq_u32_e64 s11, 0xff, v138                           // 0000000024a8: d44a000b 020314ff 000000ff
	s_wait_loadcnt 0x9                                         // 0000000024b4: bfc00009
	v_cmp_eq_u32_e64 s12, 0xff, v139                           // 0000000024b8: d44a000c 020316ff 000000ff
	s_wait_loadcnt 0x8                                         // 0000000024c4: bfc00008
	v_cmp_eq_u32_e64 s13, 0xff, v140                           // 0000000024c8: d44a000d 020318ff 000000ff
	s_wait_loadcnt 0x7                                         // 0000000024d4: bfc00007
	v_cmp_eq_u32_e64 s14, 0xff, v141                           // 0000000024d8: d44a000e 02031aff 000000ff
	s_wait_loadcnt 0x6                                         // 0000000024e4: bfc00006
	v_cmp_eq_u32_e64 s15, 0xff, v142                           // 0000000024e8: d44a000f 02031cff 000000ff
	s_wait_loadcnt 0x5                                         // 0000000024f4: bfc00005
	v_add_nc_u32_e32 v117, 0xffffff02, v146                    // 0000000024f8: 4aeb24ff ffffff02
	s_wait_loadcnt 0x4                                         // 000000002500: bfc00004
	v_add_nc_u32_e32 v118, 0xffffff02, v148                    // 000000002504: 4aed28ff ffffff02
	s_wait_loadcnt 0x3                                         // 00000000250c: bfc00003
	v_cmp_eq_u32_e64 s16, 0xff, v143                           // 000000002510: d44a0010 02031eff 000000ff
	s_wait_loadcnt 0x2                                         // 00000000251c: bfc00002
	v_cmp_eq_u32_e64 s17, 0xff, v144                           // 000000002520: d44a0011 020320ff 000000ff
	s_wait_loadcnt 0x1                                         // 00000000252c: bfc00001
	v_cmp_eq_u32_e64 s18, 0xff, v145                           // 000000002530: d44a0012 020322ff 000000ff
	v_cmp_eq_u32_e64 s20, 0xff, v146                           // 00000000253c: d44a0014 020324ff 000000ff
	v_cmp_eq_u32_e64 s21, 0xff, v148                           // 000000002548: d44a0015 020328ff 000000ff
	v_add_nc_u32_e32 v119, v117, v131                          // 000000002554: 4aef0775
	v_add_nc_u32_e32 v120, v117, v132                          // 000000002558: 4af10975
	v_add_nc_u32_e32 v121, v117, v133                          // 00000000255c: 4af30b75
	v_add_nc_u32_e32 v122, v117, v134                          // 000000002560: 4af50d75
	v_add_nc_u32_e32 v123, v117, v135                          // 000000002564: 4af70f75
	v_add_nc_u32_e32 v124, v117, v136                          // 000000002568: 4af91175
	v_add_nc_u32_e32 v125, v117, v137                          // 00000000256c: 4afb1375
	v_add_nc_u32_e32 v126, v117, v138                          // 000000002570: 4afd1575
	v_add_nc_u32_e32 v127, v118, v131                          // 000000002574: 4aff0776
	v_add_nc_u32_e32 v128, v118, v132                          // 000000002578: 4b010976
	v_add_nc_u32_e32 v129, v118, v133                          // 00000000257c: 4b030b76
	v_add_nc_u32_e32 v130, v118, v134                          // 000000002580: 4b050d76
	v_add_nc_u32_e32 v131, v118, v135                          // 000000002584: 4b070f76
	v_add_nc_u32_e32 v132, v118, v136                          // 000000002588: 4b091176
	v_add_nc_u32_e32 v133, v118, v137                          // 00000000258c: 4b0b1376
	v_add_nc_u32_e32 v134, v118, v138                          // 000000002590: 4b0d1576
	v_add_nc_u32_e32 v135, v117, v139                          // 000000002594: 4b0f1775
	v_add_nc_u32_e32 v136, v117, v140                          // 000000002598: 4b111975
	v_add_nc_u32_e32 v137, v117, v141                          // 00000000259c: 4b131b75
	v_add_nc_u32_e32 v138, v117, v142                          // 0000000025a0: 4b151d75
	v_add_nc_u32_e32 v146, v117, v143                          // 0000000025a4: 4b251f75
	v_add_nc_u32_e32 v148, v117, v144                          // 0000000025a8: 4b292175
	v_add_nc_u32_e32 v149, v117, v145                          // 0000000025ac: 4b2b2375
	s_wait_loadcnt 0x0                                         // 0000000025b0: bfc00000
	v_add_nc_u32_e32 v117, v117, v147                          // 0000000025b4: 4aeb2775
	v_add_nc_u32_e32 v139, v118, v139                          // 0000000025b8: 4b171776
	v_add_nc_u32_e32 v140, v118, v140                          // 0000000025bc: 4b191976
	v_add_nc_u32_e32 v141, v118, v141                          // 0000000025c0: 4b1b1b76
	v_add_nc_u32_e32 v142, v118, v142                          // 0000000025c4: 4b1d1d76
	v_add_nc_u32_e32 v143, v118, v143                          // 0000000025c8: 4b1f1f76
	v_add_nc_u32_e32 v144, v118, v144                          // 0000000025cc: 4b212176
	v_add_nc_u32_e32 v145, v118, v145                          // 0000000025d0: 4b232376
	v_add_nc_u32_e32 v118, v118, v147                          // 0000000025d4: 4aed2776
	v_cmp_eq_u32_e64 s19, 0xff, v147                           // 0000000025d8: d44a0013 020326ff 000000ff
	v_ldexp_f32 v0, v0, v119                                   // 0000000025e4: d71c0000 0202ef00
	v_ldexp_f32 v1, v1, v120                                   // 0000000025ec: d71c0001 0202f101
	v_ldexp_f32 v2, v2, v121                                   // 0000000025f4: d71c0002 0202f302
	v_ldexp_f32 v3, v3, v122                                   // 0000000025fc: d71c0003 0202f503
	v_ldexp_f32 v4, v4, v123                                   // 000000002604: d71c0004 0202f704
	v_ldexp_f32 v5, v5, v124                                   // 00000000260c: d71c0005 0202f905
	v_ldexp_f32 v6, v6, v125                                   // 000000002614: d71c0006 0202fb06
	v_ldexp_f32 v7, v7, v126                                   // 00000000261c: d71c0007 0202fd07
	v_ldexp_f32 v93, v93, v127                                 // 000000002624: d71c005d 0202ff5d
	v_ldexp_f32 v94, v94, v128                                 // 00000000262c: d71c005e 0203015e
	v_ldexp_f32 v95, v95, v129                                 // 000000002634: d71c005f 0203035f
	v_ldexp_f32 v96, v96, v130                                 // 00000000263c: d71c0060 02030560
	v_ldexp_f32 v97, v97, v131                                 // 000000002644: d71c0061 02030761
	v_ldexp_f32 v98, v98, v132                                 // 00000000264c: d71c0062 02030962
	v_ldexp_f32 v99, v99, v133                                 // 000000002654: d71c0063 02030b63
	v_ldexp_f32 v100, v100, v134                               // 00000000265c: d71c0064 02030d64
	v_ldexp_f32 v101, v101, v135                               // 000000002664: d71c0065 02030f65
	v_ldexp_f32 v102, v102, v136                               // 00000000266c: d71c0066 02031166
	v_ldexp_f32 v103, v103, v137                               // 000000002674: d71c0067 02031367
	v_ldexp_f32 v104, v104, v138                               // 00000000267c: d71c0068 02031568
	v_ldexp_f32 v105, v105, v146                               // 000000002684: d71c0069 02032569
	v_ldexp_f32 v106, v106, v148                               // 00000000268c: d71c006a 0203296a
	v_ldexp_f32 v107, v107, v149                               // 000000002694: d71c006b 02032b6b
	v_ldexp_f32 v108, v108, v117                               // 00000000269c: d71c006c 0202eb6c
	v_ldexp_f32 v109, v109, v139                               // 0000000026a4: d71c006d 0203176d
	v_ldexp_f32 v110, v110, v140                               // 0000000026ac: d71c006e 0203196e
	v_ldexp_f32 v111, v111, v141                               // 0000000026b4: d71c006f 02031b6f
	v_ldexp_f32 v112, v112, v142                               // 0000000026bc: d71c0070 02031d70
	v_ldexp_f32 v113, v113, v143                               // 0000000026c4: d71c0071 02031f71
	v_ldexp_f32 v114, v114, v144                               // 0000000026cc: d71c0072 02032172
	v_ldexp_f32 v115, v115, v145                               // 0000000026d4: d71c0073 02032373
	v_ldexp_f32 v116, v116, v118                               // 0000000026dc: d71c0074 0202ed74
	s_or_b32 s33, s4, s20                                      // 0000000026e4: 8c211404
	s_or_b32 s36, s20, s5                                      // 0000000026e8: 8c240514
	s_or_b32 s37, s20, s6                                      // 0000000026ec: 8c250614
	s_or_b32 s38, s20, s7                                      // 0000000026f0: 8c260714
	s_or_b32 s39, s20, s8                                      // 0000000026f4: 8c270814
	s_or_b32 s40, s20, s9                                      // 0000000026f8: 8c280914
	s_or_b32 s41, s20, s10                                     // 0000000026fc: 8c290a14
	s_or_b32 s42, s20, s11                                     // 000000002700: 8c2a0b14
	s_or_b32 s4, s4, s21                                       // 000000002704: 8c041504
	s_or_b32 s5, s5, s21                                       // 000000002708: 8c051505
	s_or_b32 s6, s6, s21                                       // 00000000270c: 8c061506
	s_or_b32 s7, s7, s21                                       // 000000002710: 8c071507
	s_or_b32 s8, s8, s21                                       // 000000002714: 8c081508
	s_or_b32 s9, s9, s21                                       // 000000002718: 8c091509
	s_or_b32 s10, s10, s21                                     // 00000000271c: 8c0a150a
	s_or_b32 s11, s11, s21                                     // 000000002720: 8c0b150b
	s_or_b32 s43, s20, s12                                     // 000000002724: 8c2b0c14
	s_or_b32 s44, s20, s13                                     // 000000002728: 8c2c0d14
	s_or_b32 s45, s20, s14                                     // 00000000272c: 8c2d0e14
	s_or_b32 s46, s20, s15                                     // 000000002730: 8c2e0f14
	s_or_b32 s47, s20, s16                                     // 000000002734: 8c2f1014
	s_or_b32 s48, s20, s17                                     // 000000002738: 8c301114
	s_or_b32 s49, s20, s18                                     // 00000000273c: 8c311214
	s_or_b32 s20, s20, s19                                     // 000000002740: 8c141314
	s_or_b32 s12, s21, s12                                     // 000000002744: 8c0c0c15
	s_or_b32 s13, s21, s13                                     // 000000002748: 8c0d0d15
	s_or_b32 s14, s21, s14                                     // 00000000274c: 8c0e0e15
	s_or_b32 s15, s21, s15                                     // 000000002750: 8c0f0f15
	s_or_b32 s16, s21, s16                                     // 000000002754: 8c101015
	s_or_b32 s17, s21, s17                                     // 000000002758: 8c111115
	s_or_b32 s18, s21, s18                                     // 00000000275c: 8c121215
	s_or_b32 s19, s21, s19                                     // 000000002760: 8c131315
	s_wait_alu depctr_sa_sdst(0)                               // 000000002764: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s33                  // 000000002768: d5010000 0085ff00 7fc00000
	v_cndmask_b32_e64 v1, v1, 0x7fc00000, s36                  // 000000002774: d5010001 0091ff01 7fc00000
	v_cndmask_b32_e64 v2, v2, 0x7fc00000, s37                  // 000000002780: d5010002 0095ff02 7fc00000
	v_cndmask_b32_e64 v3, v3, 0x7fc00000, s38                  // 00000000278c: d5010003 0099ff03 7fc00000
	v_cndmask_b32_e64 v4, v4, 0x7fc00000, s39                  // 000000002798: d5010004 009dff04 7fc00000
	v_cndmask_b32_e64 v5, v5, 0x7fc00000, s40                  // 0000000027a4: d5010005 00a1ff05 7fc00000
	v_cndmask_b32_e64 v6, v6, 0x7fc00000, s41                  // 0000000027b0: d5010006 00a5ff06 7fc00000
	v_cndmask_b32_e64 v7, v7, 0x7fc00000, s42                  // 0000000027bc: d5010007 00a9ff07 7fc00000
	v_cndmask_b32_e64 v93, v93, 0x7fc00000, s4                 // 0000000027c8: d501005d 0011ff5d 7fc00000
	v_cndmask_b32_e64 v94, v94, 0x7fc00000, s5                 // 0000000027d4: d501005e 0015ff5e 7fc00000
	v_cndmask_b32_e64 v95, v95, 0x7fc00000, s6                 // 0000000027e0: d501005f 0019ff5f 7fc00000
	v_cndmask_b32_e64 v96, v96, 0x7fc00000, s7                 // 0000000027ec: d5010060 001dff60 7fc00000
	v_cndmask_b32_e64 v97, v97, 0x7fc00000, s8                 // 0000000027f8: d5010061 0021ff61 7fc00000
	v_cndmask_b32_e64 v98, v98, 0x7fc00000, s9                 // 000000002804: d5010062 0025ff62 7fc00000
	v_cndmask_b32_e64 v99, v99, 0x7fc00000, s10                // 000000002810: d5010063 0029ff63 7fc00000
	v_cndmask_b32_e64 v100, v100, 0x7fc00000, s11              // 00000000281c: d5010064 002dff64 7fc00000
	v_cndmask_b32_e64 v101, v101, 0x7fc00000, s43              // 000000002828: d5010065 00adff65 7fc00000
	v_cndmask_b32_e64 v102, v102, 0x7fc00000, s44              // 000000002834: d5010066 00b1ff66 7fc00000
	v_cndmask_b32_e64 v103, v103, 0x7fc00000, s45              // 000000002840: d5010067 00b5ff67 7fc00000
	v_cndmask_b32_e64 v104, v104, 0x7fc00000, s46              // 00000000284c: d5010068 00b9ff68 7fc00000
	v_cndmask_b32_e64 v105, v105, 0x7fc00000, s47              // 000000002858: d5010069 00bdff69 7fc00000
	v_cndmask_b32_e64 v106, v106, 0x7fc00000, s48              // 000000002864: d501006a 00c1ff6a 7fc00000
	v_cndmask_b32_e64 v107, v107, 0x7fc00000, s49              // 000000002870: d501006b 00c5ff6b 7fc00000
	v_cndmask_b32_e64 v108, v108, 0x7fc00000, s20              // 00000000287c: d501006c 0051ff6c 7fc00000
	v_cndmask_b32_e64 v109, v109, 0x7fc00000, s12              // 000000002888: d501006d 0031ff6d 7fc00000
	v_cndmask_b32_e64 v110, v110, 0x7fc00000, s13              // 000000002894: d501006e 0035ff6e 7fc00000
	v_cndmask_b32_e64 v111, v111, 0x7fc00000, s14              // 0000000028a0: d501006f 0039ff6f 7fc00000
	v_cndmask_b32_e64 v112, v112, 0x7fc00000, s15              // 0000000028ac: d5010070 003dff70 7fc00000
	v_cndmask_b32_e64 v113, v113, 0x7fc00000, s16              // 0000000028b8: d5010071 0041ff71 7fc00000
	v_cndmask_b32_e64 v114, v114, 0x7fc00000, s17              // 0000000028c4: d5010072 0045ff72 7fc00000
	v_cndmask_b32_e64 v115, v115, 0x7fc00000, s18              // 0000000028d0: d5010073 0049ff73 7fc00000
	v_cndmask_b32_e64 v116, v116, 0x7fc00000, s19              // 0000000028dc: d5010074 004dff74 7fc00000
	v_add_f32_e32 v64, v64, v0                                 // 0000000028e8: 06800140
	v_dual_add_f32 v90, v90, v1 :: v_dual_add_f32 v89, v89, v2 // 0000000028ec: c908035a 5a580559
	v_add_f32_e32 v87, v87, v3                                 // 0000000028f4: 06ae0757
	v_dual_add_f32 v85, v85, v4 :: v_dual_add_f32 v84, v84, v5 // 0000000028f8: c9080955 55540b54
	v_dual_add_f32 v83, v83, v6 :: v_dual_add_f32 v82, v82, v7 // 000000002900: c9080d53 53520f52
	v_dual_add_f32 v63, v63, v93 :: v_dual_add_f32 v62, v62, v94// 000000002908: c908bb3f 3f3ebd3e
	v_dual_add_f32 v61, v61, v95 :: v_dual_add_f32 v60, v60, v96// 000000002910: c908bf3d 3d3cc13c
	v_dual_add_f32 v59, v59, v97 :: v_dual_add_f32 v58, v58, v98// 000000002918: c908c33b 3b3ac53a
	v_dual_add_f32 v57, v57, v99 :: v_dual_add_f32 v56, v56, v100// 000000002920: c908c739 3938c938
	v_dual_add_f32 v79, v79, v101 :: v_dual_add_f32 v78, v78, v102// 000000002928: c908cb4f 4f4ecd4e
	v_add_f32_e32 v77, v77, v103                               // 000000002930: 069acf4d
	v_add_f32_e32 v75, v75, v104                               // 000000002934: 0696d14b
	v_add_f32_e32 v69, v69, v105                               // 000000002938: 068ad345
	v_dual_add_f32 v67, v67, v106 :: v_dual_add_f32 v66, v66, v107// 00000000293c: c908d543 4342d742
	v_add_f32_e32 v65, v65, v108                               // 000000002944: 0682d941
	v_dual_add_f32 v55, v55, v109 :: v_dual_add_f32 v54, v54, v110// 000000002948: c908db37 3736dd36
	v_dual_add_f32 v53, v53, v111 :: v_dual_add_f32 v52, v52, v112// 000000002950: c908df35 3534e134
	v_dual_add_f32 v51, v51, v113 :: v_dual_add_f32 v50, v50, v114// 000000002958: c908e333 3332e532
	v_dual_add_f32 v49, v49, v115 :: v_dual_add_f32 v48, v48, v116// 000000002960: c908e731 3130e930
	s_cmp_lg_u64 s[34:35], s[30:31]                            // 000000002968: bf111e22
	s_cbranch_scc0 36                                          // 00000000296c: bfa10024 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0xf00>
	s_lshl_b64 s[6:7], s[34:35], 5                             // 000000002970: 84868522
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0               // 000000002974: ca100080 02020080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000297c: bf88ff9e
	v_add_co_u32 v0, s4, v70, s6                               // 000000002980: d7000400 02000d46
	s_wait_alu depctr_va_sdst(0)                               // 000000002988: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v71, s4                  // 00000000298c: d5207c01 00128e07
	global_load_b128 v[4:7], v[0:1], off                       // 000000002994: ee05c07c 00000004 00000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 0000000029a0: ca100080 00000080
	s_and_saveexec_b32 s5, s3                                  // 0000000029a8: be852003
	s_cbranch_execz 8                                          // 0000000029ac: bfa50008 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0xed0>
	v_add_co_u32 v0, s4, v72, s6                               // 0000000029b0: d7000400 02000d48
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v73, s4                  // 0000000029bc: d5207c01 00129207
	global_load_b128 v[0:3], v[0:1], off                       // 0000000029c4: ee05c07c 00000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000029d4: 8c7e057e
	s_barrier_signal -1                                        // 0000000029d8: be804ec1
	s_barrier_wait 0xffff                                      // 0000000029dc: bf94ffff
	s_wait_loadcnt 0x0                                         // 0000000029e0: bfc00000
	ds_store_b128 v68, v[4:7]                                  // 0000000029e4: db7c0000 00000444
	s_and_saveexec_b32 s4, s3                                  // 0000000029ec: be842003
	s_cbranch_execz 64987                                      // 0000000029f0: bfa5fddb <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x660>
	ds_store_b128 v68, v[0:3] offset:6144                      // 0000000029f4: db7c1800 00000044
	s_branch 64984                                             // 0000000029fc: bfa0fdd8 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x660>
	s_load_b64 s[18:19], s[0:1], 0xa8                          // 000000002a00: f4002480 f80000a8
	v_mul_lo_u32 v4, s27, v12                                  // 000000002a08: d72c0004 0202181b
	v_mul_lo_u32 v5, s26, v13                                  // 000000002a10: d72c0005 02021a1a
	v_mad_co_u64_u32 v[2:3], null, s26, v12, 0                 // 000000002a18: d6fe7c02 0202181a
	v_sub_co_u32 v0, s0, s24, v12                              // 000000002a20: d7010000 02021818
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002a28: bf870191
	v_sub_co_ci_u32_e64 v1, null, s25, v13, s0                 // 000000002a2c: d5217c01 00021a19
	v_add3_u32 v3, v3, v5, v4                                  // 000000002a34: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a3c: bf8701a2
	v_cmp_lt_i64_e64 s15, 0, v[0:1]                            // 000000002a40: d451000f 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 000000002a48: 3e081081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002a4c: 3e040481
	s_and_b32 s0, s15, s2                                      // 000000002a50: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a54: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002a58: be812000
	s_cbranch_execz 28                                         // 000000002a5c: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0xfd0>
	v_bfe_u32 v6, v64, 16, 1                                   // 000000002a60: d6100006 02052140
	s_wait_kmcnt 0x0                                           // 000000002a68: bfc70000
	v_add_co_u32 v7, s0, s18, v2                               // 000000002a6c: d7000007 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002a74: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s19, v3, s0                 // 000000002a78: d5207c0c 00020613
	v_add3_u32 v13, v6, v64, 0x7fff                            // 000000002a80: d655000d 03fe8106 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002a8c: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 000000002a90: d7000006 02020907
	v_or_b32_e32 v14, 0x400000, v64                            // 000000002a98: 381c80ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002aa0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v12, v5, s0                  // 000000002aa4: d5207c07 00020b0c
	v_cmp_u_f32_e64 s0, v64, v64                               // 000000002aac: d4180000 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ab8: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 000000002abc: d501000c 00021d0d
	global_store_d16_hi_b16 v[6:7], v12, off                   // 000000002ac4: ee09407c 06000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ad4: 8c7e017e
	v_add_co_u32 v6, s0, s26, v8                               // 000000002ad8: d7000006 0202101a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ae0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v9, s0                  // 000000002ae4: d5207c07 0002121b
	v_cmp_lt_i64_e64 s16, 1, v[0:1]                            // 000000002aec: d4510010 02020081
	s_delay_alu instid0(valu_dep_2)                            // 000000002af4: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000002af8: 3e0c0c81
	s_and_b32 s0, s16, s2                                      // 000000002afc: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b00: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002b04: be812000
	s_cbranch_execz 28                                         // 000000002b08: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x107c>
	v_bfe_u32 v12, v90, 16, 1                                  // 000000002b0c: d610000c 0205215a
	s_wait_kmcnt 0x0                                           // 000000002b14: bfc70000
	v_add_co_u32 v13, s0, s18, v2                              // 000000002b18: d700000d 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002b20: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s19, v3, s0                 // 000000002b24: d5207c0e 00020613
	v_add3_u32 v15, v12, v90, 0x7fff                           // 000000002b2c: d655000f 03feb50c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b38: bf870003
	v_add_co_u32 v12, s0, v13, v6                              // 000000002b3c: d700000c 02020d0d
	v_or_b32_e32 v16, 0x400000, v90                            // 000000002b44: 3820b4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b4c: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v7, s0                 // 000000002b50: d5207c0d 00020f0e
	v_cmp_u_f32_e64 s0, v90, v90                               // 000000002b58: d4180000 0202b55a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b64: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002b68: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 000000002b70: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002b80: 8c7e017e
	s_lshl_b64 s[38:39], s[26:27], 1                           // 000000002b84: 84a6811a
	v_cmp_lt_i64_e64 s14, 2, v[0:1]                            // 000000002b88: d451000e 02020082
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b90: bf88ff9e
	v_add_co_u32 v12, s0, s38, v8                              // 000000002b94: d700000c 02021026
	s_wait_alu depctr_va_sdst(0)                               // 000000002b9c: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v9, s0                 // 000000002ba0: d5207c0d 00021227
	s_and_b32 s0, s14, s2                                      // 000000002ba8: 8b00020e
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002bac: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bb0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002bb4: be812000
	s_cbranch_execz 28                                         // 000000002bb8: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x112c>
	v_bfe_u32 v14, v89, 16, 1                                  // 000000002bbc: d610000e 02052159
	s_wait_kmcnt 0x0                                           // 000000002bc4: bfc70000
	v_add_co_u32 v15, s0, s18, v2                              // 000000002bc8: d700000f 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd0: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s19, v3, s0                 // 000000002bd4: d5207c10 00020613
	v_add3_u32 v17, v14, v89, 0x7fff                           // 000000002bdc: d6550011 03feb30e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002be8: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000002bec: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v89                            // 000000002bf4: 3824b2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002bfc: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000002c00: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v89, v89                               // 000000002c08: d4180000 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 000000002c10: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c14: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002c18: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002c20: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c30: 8c7e017e
	s_mul_u64 s[36:37], s[26:27], 3                            // 000000002c34: aaa4831a
	v_cmp_lt_i64_e64 s13, 3, v[0:1]                            // 000000002c38: d451000d 02020083
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c40: bf88ff9e
	v_add_co_u32 v14, s0, s36, v8                              // 000000002c44: d700000e 02021024
	s_wait_alu depctr_va_sdst(0)                               // 000000002c4c: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v9, s0                 // 000000002c50: d5207c0f 00021225
	s_and_b32 s0, s13, s2                                      // 000000002c58: 8b00020d
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002c5c: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c60: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c64: be812000
	s_cbranch_execz 28                                         // 000000002c68: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x11dc>
	v_bfe_u32 v16, v87, 16, 1                                  // 000000002c6c: d6100010 02052157
	s_wait_kmcnt 0x0                                           // 000000002c74: bfc70000
	v_add_co_u32 v17, s0, s18, v2                              // 000000002c78: d7000011 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002c80: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v3, s0                 // 000000002c84: d5207c12 00020613
	v_add3_u32 v19, v16, v87, 0x7fff                           // 000000002c8c: d6550013 03feaf10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c98: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002c9c: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v87                            // 000000002ca4: 3828aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002cac: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002cb0: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v87, v87                               // 000000002cb8: d4180000 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002cc4: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000002cc8: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002cd0: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cdc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ce0: 8c7e017e
	s_lshl_b64 s[34:35], s[26:27], 2                           // 000000002ce4: 84a2821a
	v_cmp_lt_i64_e64 s12, 4, v[0:1]                            // 000000002ce8: d451000c 02020084
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cf0: bf88ff9e
	v_add_co_u32 v16, s0, s34, v8                              // 000000002cf4: d7000010 02021022
	s_wait_alu depctr_va_sdst(0)                               // 000000002cfc: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v9, s0                 // 000000002d00: d5207c11 00021223
	s_and_b32 s0, s12, s2                                      // 000000002d08: 8b00020c
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002d0c: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d10: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002d14: be812000
	s_cbranch_execz 28                                         // 000000002d18: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x128c>
	v_bfe_u32 v18, v85, 16, 1                                  // 000000002d1c: d6100012 02052155
	s_wait_kmcnt 0x0                                           // 000000002d24: bfc70000
	v_add_co_u32 v19, s0, s18, v2                              // 000000002d28: d7000013 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002d30: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v3, s0                 // 000000002d34: d5207c14 00020613
	v_add3_u32 v21, v18, v85, 0x7fff                           // 000000002d3c: d6550015 03feab12 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d48: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002d4c: d7000012 02022113
	v_or_b32_e32 v22, 0x400000, v85                            // 000000002d54: 382caaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d5c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 000000002d60: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v85, v85                               // 000000002d68: d4180000 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 000000002d70: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d74: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 000000002d78: d5010014 00022d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002d80: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d8c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d90: 8c7e017e
	s_mul_u64 s[30:31], s[26:27], 5                            // 000000002d94: aa9e851a
	v_cmp_lt_i64_e64 s11, 5, v[0:1]                            // 000000002d98: d451000b 02020085
	s_wait_alu depctr_sa_sdst(0)                               // 000000002da0: bf88ff9e
	v_add_co_u32 v18, s0, s30, v8                              // 000000002da4: d7000012 0202101e
	s_wait_alu depctr_va_sdst(0)                               // 000000002dac: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s31, v9, s0                 // 000000002db0: d5207c13 0002121f
	s_and_b32 s0, s11, s2                                      // 000000002db8: 8b00020b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002dbc: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dc0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002dc4: be812000
	s_cbranch_execz 28                                         // 000000002dc8: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x133c>
	v_bfe_u32 v20, v84, 16, 1                                  // 000000002dcc: d6100014 02052154
	s_wait_kmcnt 0x0                                           // 000000002dd4: bfc70000
	v_add_co_u32 v21, s0, s18, v2                              // 000000002dd8: d7000015 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002de0: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v3, s0                 // 000000002de4: d5207c16 00020613
	v_add3_u32 v23, v20, v84, 0x7fff                           // 000000002dec: d6550017 03fea914 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002df8: bf870003
	v_add_co_u32 v20, s0, v21, v18                             // 000000002dfc: d7000014 02022515
	v_or_b32_e32 v24, 0x400000, v84                            // 000000002e04: 3830a8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e0c: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v19, s0                // 000000002e10: d5207c15 00022716
	v_cmp_u_f32_e64 s0, v84, v84                               // 000000002e18: d4180000 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 000000002e20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e24: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 000000002e28: d5010016 00023117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000002e30: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e40: 8c7e017e
	s_mul_u64 s[28:29], s[26:27], 6                            // 000000002e44: aa9c861a
	v_cmp_lt_i64_e64 s9, 6, v[0:1]                             // 000000002e48: d4510009 02020086
	v_add_co_u32 v20, s0, s28, v8                              // 000000002e50: d7000014 0202101c
	s_wait_alu depctr_va_sdst(0)                               // 000000002e58: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s29, v9, s0                 // 000000002e5c: d5207c15 0002121d
	s_and_b32 s0, s9, s2                                       // 000000002e64: 8b000209
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000002e68: 3e282881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e6c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e70: be812000
	s_cbranch_execz 28                                         // 000000002e74: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x13e8>
	v_bfe_u32 v22, v83, 16, 1                                  // 000000002e78: d6100016 02052153
	s_wait_kmcnt 0x0                                           // 000000002e80: bfc70000
	v_add_co_u32 v23, s0, s18, v2                              // 000000002e84: d7000017 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002e8c: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v3, s0                 // 000000002e90: d5207c18 00020613
	v_add3_u32 v25, v22, v83, 0x7fff                           // 000000002e98: d6550019 03fea716 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002ea4: bf870003
	v_add_co_u32 v22, s0, v23, v20                             // 000000002ea8: d7000016 02022917
	v_or_b32_e32 v26, 0x400000, v83                            // 000000002eb0: 3834a6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb8: bf88f19f
	v_add_co_ci_u32_e64 v23, null, v24, v21, s0                // 000000002ebc: d5207c17 00022b18
	v_cmp_u_f32_e64 s0, v83, v83                               // 000000002ec4: d4180000 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000002ecc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ed0: bf870001
	v_cndmask_b32_e64 v24, v25, v26, s0                        // 000000002ed4: d5010018 00023519
	global_store_d16_hi_b16 v[22:23], v24, off                 // 000000002edc: ee09407c 0c000000 00000016
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002eec: 8c7e017e
	s_mul_u64 s[20:21], s[26:27], 7                            // 000000002ef0: aa94871a
	v_cmp_lt_i64_e64 s8, 7, v[0:1]                             // 000000002ef4: d4510008 02020087
	s_wait_alu depctr_sa_sdst(0)                               // 000000002efc: bf88ff9e
	v_add_co_u32 v8, s0, s20, v8                               // 000000002f00: d7000008 02021014
	s_wait_alu depctr_va_sdst(0)                               // 000000002f08: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s21, v9, s0                  // 000000002f0c: d5207c09 00021215
	s_and_b32 s0, s8, s2                                       // 000000002f14: 8b000208
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002f18: 3e101081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f1c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f20: be812000
	s_cbranch_execz 28                                         // 000000002f24: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1498>
	v_bfe_u32 v0, v82, 16, 1                                   // 000000002f28: d6100000 02052152
	s_wait_kmcnt 0x0                                           // 000000002f30: bfc70000
	v_add_co_u32 v1, s0, s18, v2                               // 000000002f34: d7000001 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000002f3c: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v3, s0                 // 000000002f40: d5207c16 00020613
	v_add3_u32 v23, v0, v82, 0x7fff                            // 000000002f48: d6550017 03fea500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f54: bf870003
	v_add_co_u32 v0, s0, v1, v8                                // 000000002f58: d7000000 02021101
	v_or_b32_e32 v24, 0x400000, v82                            // 000000002f60: 3830a4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f68: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v22, v9, s0                  // 000000002f6c: d5207c01 00021316
	v_cmp_u_f32_e64 s0, v82, v82                               // 000000002f74: d4180000 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000002f7c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f80: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 000000002f84: d5010016 00023117
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000002f8c: ee09407c 0b000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f9c: 8c7e017e
	v_mul_lo_u32 v22, s27, v10                                 // 000000002fa0: d72c0016 0202141b
	v_mul_lo_u32 v23, s26, v11                                 // 000000002fa8: d72c0017 0202161a
	v_mad_co_u64_u32 v[0:1], null, s26, v10, 0                 // 000000002fb0: d6fe7c00 0202141a
	v_sub_co_u32 v10, s0, s24, v10                             // 000000002fb8: d701000a 02021418
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc0: bf88f19f
	v_sub_co_ci_u32_e64 v11, null, s25, v11, s0                // 000000002fc4: d5217c0b 00021619
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000002fcc: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[10:11]                          // 000000002fd0: d451000a 02021480
	v_add3_u32 v1, v1, v23, v22                                // 000000002fd8: d6550001 045a2f01
	s_delay_alu instid0(valu_dep_1)                            // 000000002fe0: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002fe4: 3e000081
	s_and_b32 s0, s10, s2                                      // 000000002fe8: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002ff0: be812000
	s_cbranch_execz 28                                         // 000000002ff4: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1568>
	s_wait_kmcnt 0x0                                           // 000000002ff8: bfc70000
	v_add_co_u32 v23, s0, s18, v0                              // 000000002ffc: d7000017 02020012
	v_bfe_u32 v22, v79, 16, 1                                  // 000000003004: d6100016 0205214f
	s_wait_alu depctr_va_sdst(0)                               // 00000000300c: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v1, s0                 // 000000003010: d5207c18 00020213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003018: bf870193
	v_add_co_u32 v4, s0, v23, v4                               // 00000000301c: d7000004 02020917
	v_add3_u32 v22, v22, v79, 0x7fff                           // 000000003024: d6550016 03fe9f16 00007fff
	v_or_b32_e32 v25, 0x400000, v79                            // 000000003030: 38329eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003038: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v24, v5, s0                  // 00000000303c: d5207c05 00020b18
	v_cmp_u_f32_e64 s0, v79, v79                               // 000000003044: d4180000 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 00000000304c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003050: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s0                        // 000000003054: d5010016 00023316
	global_store_d16_hi_b16 v[4:5], v22, off                   // 00000000305c: ee09407c 0b000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003068: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000306c: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[10:11]                           // 000000003070: d4510007 02021481
	s_and_b32 s0, s7, s2                                       // 000000003078: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 00000000307c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003080: be812000
	s_cbranch_execz 28                                         // 000000003084: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x15f8>
	v_bfe_u32 v4, v78, 16, 1                                   // 000000003088: d6100004 0205214e
	s_wait_kmcnt 0x0                                           // 000000003090: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003094: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 00000000309c: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v1, s0                 // 0000000030a0: d5207c16 00020213
	v_add3_u32 v23, v4, v78, 0x7fff                            // 0000000030a8: d6550017 03fe9d04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030b4: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 0000000030b8: d7000004 02020d05
	v_or_b32_e32 v24, 0x400000, v78                            // 0000000030c0: 38309cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v22, v7, s0                  // 0000000030cc: d5207c05 00020f16
	v_cmp_u_f32_e64 s0, v78, v78                               // 0000000030d4: d4180000 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030e0: bf870001
	v_cndmask_b32_e64 v6, v23, v24, s0                         // 0000000030e4: d5010006 00023117
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000030ec: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030fc: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[10:11]                           // 000000003100: d4510006 02021482
	s_and_b32 s0, s6, s2                                       // 000000003108: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 00000000310c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003110: be812000
	s_cbranch_execz 28                                         // 000000003114: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1688>
	v_bfe_u32 v4, v77, 16, 1                                   // 000000003118: d6100004 0205214d
	s_wait_kmcnt 0x0                                           // 000000003120: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003124: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 00000000312c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003130: d5207c06 00020213
	v_add3_u32 v7, v4, v77, 0x7fff                             // 000000003138: d6550007 03fe9b04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003144: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000003148: d7000004 02021905
	v_or_b32_e32 v22, 0x400000, v77                            // 000000003150: 382c9aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003158: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 00000000315c: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v77, v77                               // 000000003164: d4180000 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 00000000316c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003170: bf870001
	v_cndmask_b32_e64 v6, v7, v22, s0                          // 000000003174: d5010006 00022d07
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000317c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003188: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000318c: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[10:11]                           // 000000003190: d4510005 02021483
	s_and_b32 s0, s5, s2                                       // 000000003198: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 00000000319c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000031a0: be812000
	s_cbranch_execz 28                                         // 0000000031a4: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1718>
	v_bfe_u32 v4, v75, 16, 1                                   // 0000000031a8: d6100004 0205214b
	s_wait_kmcnt 0x0                                           // 0000000031b0: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 0000000031b4: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000031bc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 0000000031c0: d5207c06 00020213
	v_add3_u32 v7, v4, v75, 0x7fff                             // 0000000031c8: d6550007 03fe9704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031d4: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 0000000031d8: d7000004 02021d05
	v_or_b32_e32 v12, 0x400000, v75                            // 0000000031e0: 381896ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 0000000031ec: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v75, v75                               // 0000000031f4: d4180000 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 0000000031fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003200: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003204: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000320c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003218: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000321c: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[10:11]                           // 000000003220: d4510004 02021484
	s_and_b32 s0, s4, s2                                       // 000000003228: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000322c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003230: be812000
	s_cbranch_execz 28                                         // 000000003234: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x17a8>
	v_bfe_u32 v4, v69, 16, 1                                   // 000000003238: d6100004 02052145
	s_wait_kmcnt 0x0                                           // 000000003240: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003244: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 00000000324c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003250: d5207c06 00020213
	v_add3_u32 v7, v4, v69, 0x7fff                             // 000000003258: d6550007 03fe8b04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003264: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003268: d7000004 02022105
	v_or_b32_e32 v12, 0x400000, v69                            // 000000003270: 38188aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003278: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 00000000327c: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v69, v69                               // 000000003284: d4180000 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 00000000328c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003290: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003294: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000329c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032ac: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[10:11]                           // 0000000032b0: d4510003 02021485
	s_and_b32 s0, s3, s2                                       // 0000000032b8: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032c0: be812000
	s_cbranch_execz 28                                         // 0000000032c4: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1838>
	v_bfe_u32 v4, v67, 16, 1                                   // 0000000032c8: d6100004 02052143
	s_wait_kmcnt 0x0                                           // 0000000032d0: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 0000000032d4: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000032dc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 0000000032e0: d5207c06 00020213
	v_add3_u32 v7, v4, v67, 0x7fff                             // 0000000032e8: d6550007 03fe8704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000032f4: bf870003
	v_add_co_u32 v4, s0, v5, v18                               // 0000000032f8: d7000004 02022505
	v_or_b32_e32 v12, 0x400000, v67                            // 000000003300: 381886ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003308: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s0                  // 00000000330c: d5207c05 00022706
	v_cmp_u_f32_e64 s0, v67, v67                               // 000000003314: d4180000 02028743
	s_wait_alu depctr_va_sdst(0)                               // 00000000331c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003320: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 000000003324: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000332c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003338: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000333c: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[10:11]                           // 000000003340: d4510001 02021486
	s_and_b32 s0, s1, s2                                       // 000000003348: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 00000000334c: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 000000003350: be912000
	s_cbranch_execz 28                                         // 000000003354: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x18c8>
	v_bfe_u32 v4, v66, 16, 1                                   // 000000003358: d6100004 02052142
	s_wait_kmcnt 0x0                                           // 000000003360: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003364: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 00000000336c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003370: d5207c06 00020213
	v_add3_u32 v7, v4, v66, 0x7fff                             // 000000003378: d6550007 03fe8504 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003384: bf870003
	v_add_co_u32 v4, s0, v5, v20                               // 000000003388: d7000004 02022905
	v_or_b32_e32 v12, 0x400000, v66                            // 000000003390: 381884ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003398: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s0                  // 00000000339c: d5207c05 00022b06
	v_cmp_u_f32_e64 s0, v66, v66                               // 0000000033a4: d4180000 02028542
	s_wait_alu depctr_va_sdst(0)                               // 0000000033ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033b0: bf870001
	v_cndmask_b32_e64 v6, v7, v12, s0                          // 0000000033b4: d5010006 00021907
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000033bc: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000033cc: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[10:11]                           // 0000000033d0: d4510000 02021487
	s_and_b32 s2, s0, s2                                       // 0000000033d8: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033dc: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 0000000033e0: be912002
	s_cbranch_execz 28                                         // 0000000033e4: bfa5001c <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1958>
	v_bfe_u32 v4, v65, 16, 1                                   // 0000000033e8: d6100004 02052141
	s_wait_kmcnt 0x0                                           // 0000000033f0: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 0000000033f4: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000033fc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003400: d5207c06 000a0213
	v_add3_u32 v7, v4, v65, 0x7fff                             // 000000003408: d6550007 03fe8304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003414: bf870003
	v_add_co_u32 v4, s2, v5, v8                                // 000000003418: d7000204 02021105
	v_or_b32_e32 v10, 0x400000, v65                            // 000000003420: 381482ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003428: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s2                   // 00000000342c: d5207c05 000a1306
	v_cmp_u_f32_e64 s2, v65, v65                               // 000000003434: d4180002 02028341
	s_wait_alu depctr_va_sdst(0)                               // 00000000343c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003440: bf870001
	v_cndmask_b32_e64 v6, v7, v10, s2                          // 000000003444: d5010006 000a1507
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000344c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003458: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 00000000345c: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 000000003460: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003464: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003468: be8f2002
	s_cbranch_execz 40                                         // 00000000346c: bfa50028 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1a10>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003470: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003478: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 00000000347c: d5207c05 00082e80
	v_bfe_u32 v6, v63, 16, 1                                   // 000000003484: d6100006 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000348c: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003490: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003498: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000349c: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 0000000034a4: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000034a8: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000034b0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000034b4: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000034bc: 3e080881
	v_add3_u32 v6, v6, v63, 0x7fff                             // 0000000034c0: d6550006 03fe7f06 00007fff
	v_or_b32_e32 v9, 0x400000, v63                             // 0000000034cc: 38127eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000034d4: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 0000000034d8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000034e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000034e4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v63, v63                               // 0000000034ec: d4180002 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034f8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000034fc: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003504: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003510: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000003514: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 000000003518: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 00000000351c: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003520: be8f2002
	s_cbranch_execz 46                                         // 000000003524: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1ae0>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003528: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003530: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003534: d5207c05 00082e80
	v_bfe_u32 v6, v62, 16, 1                                   // 00000000353c: d6100006 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003544: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003548: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003550: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003554: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v62                             // 00000000355c: 38127cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003564: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003568: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003570: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000003574: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 00000000357c: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003580: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003588: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 00000000358c: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003594: 3e080881
	v_add3_u32 v6, v6, v62, 0x7fff                             // 000000003598: d6550006 03fe7d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035a4: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000035a8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000035b4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v62, v62                               // 0000000035bc: d4180002 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035c8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000035cc: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000035d4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000035e4: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 0000000035e8: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035ec: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 0000000035f0: be8e2002
	s_cbranch_execz 46                                         // 0000000035f4: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1bb0>
	v_add_co_u32 v4, s2, v47, s22                              // 0000000035f8: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003600: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003604: d5207c05 00082e80
	v_bfe_u32 v6, v61, 16, 1                                   // 00000000360c: d6100006 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003614: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003618: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003620: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003624: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v61                             // 00000000362c: 38127aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003634: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 000000003638: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003640: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 000000003644: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 00000000364c: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003650: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003658: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 00000000365c: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003664: 3e080881
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000003668: d6550006 03fe7b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003674: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003678: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003680: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003684: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v61, v61                               // 00000000368c: d4180002 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000003694: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003698: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000369c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000036a4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 0000000036b4: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 0000000036b8: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036bc: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 0000000036c0: be8d2002
	s_cbranch_execz 46                                         // 0000000036c4: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1c80>
	v_add_co_u32 v4, s2, v47, s22                              // 0000000036c8: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000036d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000036d4: d5207c05 00082e80
	v_bfe_u32 v6, v60, 16, 1                                   // 0000000036dc: d6100006 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036e4: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 0000000036e8: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000036f4: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v60                             // 0000000036fc: 381278ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003704: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 000000003708: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000003710: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 000000003714: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 00000000371c: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003720: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003728: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 00000000372c: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003734: 3e080881
	v_add3_u32 v6, v6, v60, 0x7fff                             // 000000003738: d6550006 03fe7906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003744: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003748: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003750: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003754: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v60, v60                               // 00000000375c: d4180002 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 000000003764: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003768: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000376c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003774: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003780: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003784: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003788: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000378c: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 000000003790: be8c2002
	s_cbranch_execz 46                                         // 000000003794: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1d50>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003798: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000037a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000037a4: d5207c05 00082e80
	v_bfe_u32 v6, v59, 16, 1                                   // 0000000037ac: d6100006 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037b4: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 0000000037b8: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000037c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000037c4: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v59                             // 0000000037cc: 381276ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037d4: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000037d8: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000037e4: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000037ec: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000037f0: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000037fc: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003804: 3e080881
	v_add3_u32 v6, v6, v59, 0x7fff                             // 000000003808: d6550006 03fe7706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003814: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003818: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003820: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003824: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v59, v59                               // 00000000382c: d4180002 0202773b
	s_wait_alu depctr_va_sdst(0)                               // 000000003834: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003838: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000383c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003844: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003850: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003854: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 000000003858: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000385c: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 000000003860: be8b2002
	s_cbranch_execz 46                                         // 000000003864: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1e20>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003868: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003870: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003874: d5207c05 00082e80
	v_bfe_u32 v6, v58, 16, 1                                   // 00000000387c: d6100006 0205213a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003884: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003888: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003890: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003894: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v58                             // 00000000389c: 381274ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038a4: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 0000000038a8: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 0000000038b4: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 0000000038bc: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 0000000038c0: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000038c8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000038cc: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000038d4: 3e080881
	v_add3_u32 v6, v6, v58, 0x7fff                             // 0000000038d8: d6550006 03fe7506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038e4: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000038e8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000038f4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v58, v58                               // 0000000038fc: d4180002 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000003904: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003908: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000390c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003914: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003920: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 000000003924: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 000000003928: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 00000000392c: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 000000003930: be892002
	s_cbranch_execz 46                                         // 000000003934: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1ef0>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003938: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003944: d5207c05 00082e80
	v_bfe_u32 v6, v57, 16, 1                                   // 00000000394c: d6100006 02052139
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003954: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003958: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003960: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003964: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v57                             // 00000000396c: 381272ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003974: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003978: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003980: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000003984: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 00000000398c: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003990: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003998: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 00000000399c: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000039a4: 3e080881
	v_add3_u32 v6, v6, v57, 0x7fff                             // 0000000039a8: d6550006 03fe7306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039b4: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000039b8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000039c4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v57, v57                               // 0000000039cc: d4180002 02027339
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039d8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000039dc: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000039e4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000039f4: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 0000000039f8: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039fc: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 000000003a00: be882002
	s_cbranch_execz 46                                         // 000000003a04: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x1fc0>
	v_add_co_u32 v4, s2, v47, s22                              // 000000003a08: d7000204 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a10: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000003a14: d5207c05 00082e80
	v_bfe_u32 v6, v56, 16, 1                                   // 000000003a1c: d6100006 02052138
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a24: bf8701a3
	v_add_co_u32 v4, s2, v4, v46                               // 000000003a28: d7000204 02025d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003a30: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003a34: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v56                             // 000000003a3c: 380e70ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a44: bf8701a3
	v_add_co_u32 v4, s2, s20, v4                               // 000000003a48: d7000204 02020814
	s_wait_alu depctr_va_sdst(0)                               // 000000003a50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v5, s2                  // 000000003a54: d5207c05 000a0a15
	s_wait_kmcnt 0x0                                           // 000000003a5c: bfc70000
	v_add_co_u32 v2, s2, s18, v2                               // 000000003a60: d7000202 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003a68: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s19, v3, s2                  // 000000003a6c: d5207c03 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003a74: 3e080881
	v_add3_u32 v6, v6, v56, 0x7fff                             // 000000003a78: d6550006 03fe7106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a84: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000003a88: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000003a90: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 000000003a94: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v56, v56                               // 000000003a9c: d4180002 02027138
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003aa8: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000003aac: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ab4: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ac0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003ac4: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 000000003ac8: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003acc: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 000000003ad0: be882002
	s_cbranch_execz 40                                         // 000000003ad4: bfa50028 <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x2078>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003ad8: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003ae0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003ae4: d5207c03 00082e80
	v_bfe_u32 v4, v55, 16, 1                                   // 000000003aec: d6100004 02052137
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003af4: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003af8: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003b00: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003b04: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 000000003b0c: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003b10: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003b18: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003b1c: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003b24: 3e040481
	v_add3_u32 v4, v4, v55, 0x7fff                             // 000000003b28: d6550004 03fe6f04 00007fff
	v_or_b32_e32 v7, 0x400000, v55                             // 000000003b34: 380e6eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003b3c: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000003b40: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003b48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003b4c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v55, v55                               // 000000003b54: d4180002 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000003b5c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b60: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003b64: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b6c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003b7c: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000003b80: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b84: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000003b88: be872002
	s_cbranch_execz 46                                         // 000000003b8c: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x2148>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003b90: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003b98: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003b9c: d5207c03 00082e80
	v_bfe_u32 v4, v54, 16, 1                                   // 000000003ba4: d6100004 02052136
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bac: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003bb0: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003bbc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v54                             // 000000003bc4: 380e6cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bcc: bf8701a3
	v_add_co_u32 v2, s2, s26, v2                               // 000000003bd0: d7000202 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s2                  // 000000003bdc: d5207c03 000a061b
	s_wait_kmcnt 0x0                                           // 000000003be4: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003be8: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003bf0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003bf4: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003bfc: 3e040481
	v_add3_u32 v4, v4, v54, 0x7fff                             // 000000003c00: d6550004 03fe6d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c0c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003c10: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003c18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003c1c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v54, v54                               // 000000003c24: d4180002 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003c2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c30: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003c34: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c3c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003c4c: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000003c50: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c54: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000003c58: be862002
	s_cbranch_execz 46                                         // 000000003c5c: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x2218>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003c60: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003c68: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003c6c: d5207c03 00082e80
	v_bfe_u32 v4, v53, 16, 1                                   // 000000003c74: d6100004 02052135
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c7c: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003c80: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003c88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003c8c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v53                             // 000000003c94: 380e6aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c9c: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000003ca0: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000003cac: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000003cb4: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003cb8: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003cc4: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003ccc: 3e040481
	v_add3_u32 v4, v4, v53, 0x7fff                             // 000000003cd0: d6550004 03fe6b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cdc: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003ce0: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003cec: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v53, v53                               // 000000003cf4: d4180002 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000003cfc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d00: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003d04: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d0c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003d1c: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 000000003d20: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d24: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000003d28: be852002
	s_cbranch_execz 46                                         // 000000003d2c: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x22e8>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003d30: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003d38: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003d3c: d5207c03 00082e80
	v_bfe_u32 v4, v52, 16, 1                                   // 000000003d44: d6100004 02052134
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d4c: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003d50: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003d58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003d5c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v52                             // 000000003d64: 380e68ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d6c: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000003d70: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000003d78: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000003d7c: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000003d84: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003d88: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003d90: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003d94: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003d9c: 3e040481
	v_add3_u32 v4, v4, v52, 0x7fff                             // 000000003da0: d6550004 03fe6904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dac: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003db0: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003db8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003dbc: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v52, v52                               // 000000003dc4: d4180002 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000003dcc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003dd0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003dd4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ddc: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003dec: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 000000003df0: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003df4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003df8: be842002
	s_cbranch_execz 46                                         // 000000003dfc: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x23b8>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003e00: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003e08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003e0c: d5207c03 00082e80
	v_bfe_u32 v4, v51, 16, 1                                   // 000000003e14: d6100004 02052133
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e1c: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003e20: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003e28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003e2c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v51                             // 000000003e34: 380e66ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e3c: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000003e40: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000003e48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000003e4c: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000003e54: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003e58: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003e60: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003e64: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003e6c: 3e040481
	v_add3_u32 v4, v4, v51, 0x7fff                             // 000000003e70: d6550004 03fe6704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e7c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003e80: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003e88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003e8c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v51, v51                               // 000000003e94: d4180002 02026733
	s_wait_alu depctr_va_sdst(0)                               // 000000003e9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ea0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003ea4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003eac: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003ebc: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000003ec0: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ec4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003ec8: be832002
	s_cbranch_execz 46                                         // 000000003ecc: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x2488>
	v_add_co_u32 v2, s2, v47, s22                              // 000000003ed0: d7000202 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003ed8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000003edc: d5207c03 00082e80
	v_bfe_u32 v4, v50, 16, 1                                   // 000000003ee4: d6100004 02052132
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003eec: bf8701a3
	v_add_co_u32 v2, s2, v2, v46                               // 000000003ef0: d7000202 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003ef8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003efc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v50                             // 000000003f04: 380e64ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f0c: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000003f10: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000003f18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000003f1c: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000003f24: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003f28: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003f30: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003f34: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003f3c: 3e040481
	v_add3_u32 v4, v4, v50, 0x7fff                             // 000000003f40: d6550004 03fe6504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f4c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003f50: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003f58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003f5c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v50, v50                               // 000000003f64: d4180002 02026532
	s_wait_alu depctr_va_sdst(0)                               // 000000003f6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003f70: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003f74: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003f7c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003f8c: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000003f90: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f94: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000003f98: be822001
	s_cbranch_execz 46                                         // 000000003f9c: bfa5002e <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x2558>
	v_add_co_u32 v2, s1, v47, s22                              // 000000003fa0: d7000102 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s1                   // 000000003fac: d5207c03 00042e80
	v_bfe_u32 v4, v49, 16, 1                                   // 000000003fb4: d6100004 02052131
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fbc: bf8701a3
	v_add_co_u32 v2, s1, v2, v46                               // 000000003fc0: d7000102 02025d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003fc8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000003fcc: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v49                             // 000000003fd4: 380e62ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fdc: bf8701a3
	v_add_co_u32 v2, s1, s28, v2                               // 000000003fe0: d7000102 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s1                  // 000000003fec: d5207c03 0006061d
	s_wait_kmcnt 0x0                                           // 000000003ff4: bfc70000
	v_add_co_u32 v5, s1, s18, v0                               // 000000003ff8: d7000105 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000004000: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s1                  // 000000004004: d5207c06 00060213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 00000000400c: 3e040481
	v_add3_u32 v4, v4, v49, 0x7fff                             // 000000004010: d6550004 03fe6304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000401c: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000004020: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004028: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 00000000402c: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v49, v49                               // 000000004034: d4180001 02026331
	s_wait_alu depctr_va_sdst(0)                               // 00000000403c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004040: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000004044: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 00000000404c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004058: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 00000000405c: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000004060: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000004064: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004068: be812000
	s_cbranch_execz 43                                         // 00000000406c: bfa5002b <tessera_rocm_scaled_matmul_lds_cf3445be1476a993+0x261c>
	v_add_co_u32 v2, s0, v47, s22                              // 000000004070: d7000002 02002d2f
	s_wait_alu depctr_va_sdst(0)                               // 000000004078: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s0                   // 00000000407c: d5207c03 00002e80
	v_bfe_u32 v4, v48, 16, 1                                   // 000000004084: d6100004 02052130
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000408c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v46                           // 000000004090: d7006a02 02025d02
	s_wait_alu depctr_va_vcc(0)                                // 000000004098: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 00000000409c: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v48                             // 0000000040a4: 380a60ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040ac: bf8701a3
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 0000000040b0: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000040b8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 0000000040bc: d5207c03 01aa0615
	s_wait_kmcnt 0x0                                           // 0000000040c4: bfc70000
	v_add_co_u32 v0, vcc_lo, s18, v0                           // 0000000040c8: d7006a00 02020012
	s_wait_alu depctr_va_vcc(0)                                // 0000000040d0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s19, v1, vcc_lo              // 0000000040d4: d5207c01 01aa0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000040dc: 3e040481
	v_add3_u32 v4, v4, v48, 0x7fff                             // 0000000040e0: d6550004 03fe6104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040ec: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000040f0: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000040f8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000040fc: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000004104: 7c306130
	s_wait_alu depctr_va_vcc(0)                                // 000000004108: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 00000000410c: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004110: ee09407c 01000000 00002000
	s_nop 0                                                    // 00000000411c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000004120: bfb60003
	s_endpgm                                                   // 000000004124: bfb00000
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
