
/tmp/tmpwx4zzw8c.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_5f09ec8682bcb143>:
	s_clause 0x2                                               // 000000001b00: bf850002
	s_load_b64 s[26:27], s[0:1], 0xd8                          // 000000001b04: f4002680 f80000d8
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b0c: f4004300 f80000c8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b14: f4002400 f80000a8
	v_and_b32_e32 v2, 15, v0                                   // 000000001b1c: 3604008f
	s_mov_b32 s4, ttmp7                                        // 000000001b20: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b24: 86059f73
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[28:29], s[0:1], 0x8                           // 000000001b2c: f4002700 f8000008
	s_load_b64 s[30:31], s[0:1], 0x30                          // 000000001b34: f4002780 f8000030
	s_load_b64 s[24:25], s[0:1], 0x58                          // 000000001b3c: f4002600 f8000058
	s_load_b64 s[20:21], s[0:1], 0x80                          // 000000001b44: f4002500 f8000080
	s_lshl_b64 s[22:23], s[4:5], 4                             // 000000001b4c: 84968404
	s_mov_b32 s2, ttmp9                                        // 000000001b50: be820075
	v_or_b32_e32 v1, s22, v2                                   // 000000001b54: 38020416
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b58: 86039f75
	s_add_nc_u64 s[0:1], s[22:23], 16                          // 000000001b5c: a9809016
	s_lshl_b64 s[2:3], s[2:3], 4                               // 000000001b60: 84828402
	v_bfe_u32 v37, v0, 4, 1                                    // 000000001b64: d6100025 02050900
	s_add_nc_u64 s[4:5], s[2:3], 16                            // 000000001b6c: a9849002
	v_or_b32_e32 v11, s2, v2                                   // 000000001b70: 38160402
	v_mov_b32_e32 v12, s3                                      // 000000001b74: 7e180203
	s_wait_kmcnt 0x0                                           // 000000001b78: bfc70000
	v_mul_lo_u32 v3, s27, v1                                   // 000000001b7c: d72c0003 0202021b
	v_mad_co_u64_u32 v[13:14], null, s26, v1, 0                // 000000001b84: d6fe7c0d 0202021a
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b8c: d4540000 02001800
	v_cmp_gt_i64_e64 s1, s[4:5], s[14:15]                      // 000000001b94: d4540001 02001c04
	s_mul_i32 s2, s26, s23                                     // 000000001b9c: 9602171a
	v_cmp_lt_i64_e64 s11, s[26:27], 32                         // 000000001ba0: d451000b 0201401a
	s_and_b32 s18, s26, 0xffffffe0                             // 000000001ba8: 8b12ff1a ffffffe0
	s_mov_b32 s19, s27                                         // 000000001bb0: be93001b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	v_add3_u32 v36, v14, s2, v3                                // 000000001bb8: d6550024 040c050e
	s_or_b32 s0, s0, s1                                        // 000000001bc0: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc4: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bc8: 8b6a007e
	s_cbranch_vccz 10                                          // 000000001bcc: bfa3000a <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0xf8>
	s_and_b32 s0, s11, exec_lo                                 // 000000001bd0: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 000000001bd4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bd8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bdc: bf078100
	s_cbranch_scc1 8                                           // 000000001be0: bfa20008 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x104>
	v_lshl_or_b32 v14, v37, 3, s22                             // 000000001be4: d656000e 00590725
	v_mov_b32_e32 v15, s23                                     // 000000001bec: 7e1e0217
	s_mov_b32 s0, 0                                            // 000000001bf0: be800080
	s_branch 4                                                 // 000000001bf4: bfa00004 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x108>
	s_mov_b32 s0, 0                                            // 000000001bf8: be800080
	s_cbranch_execnz 1617                                      // 000000001bfc: bfa60651 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1a44>
	s_branch 2457                                              // 000000001c00: bfa00999 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2768>
	s_mov_b32 s0, -1                                           // 000000001c04: be8000c1
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v45, 0             // 000000001c08: ca100080 0a2c0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001c14: 8b007e00
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v47, 0             // 000000001c18: ca100080 262e0080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0             // 000000001c20: ca100080 28260080
	v_mov_b32_e32 v42, 0                                       // 000000001c28: 7e540280
	v_mov_b32_e32 v46, 0                                       // 000000001c2c: 7e5c0280
	s_cselect_b32 s0, 1, 0                                     // 000000001c30: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c34: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c38: bf078100
	s_cbranch_scc1 1272                                        // 000000001c3c: bfa204f8 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1520>
	v_dual_mov_b32 v2, s23 :: v_dual_lshlrev_b32 v41, 3, v37   // 000000001c40: ca220017 02284a83
	v_mov_b32_e32 v15, s23                                     // 000000001c48: 7e1e0217
	s_lshr_b64 s[4:5], s[26:27], 5                             // 000000001c4c: 8584851a
	s_lshr_b32 s3, s27, 5                                      // 000000001c50: 8503851b
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_1)// 000000001c54: bf8700a2
	v_or_b32_e32 v14, s22, v41                                 // 000000001c58: 381c5216
	v_add_co_u32 v43, vcc_lo, v13, v41                         // 000000001c5c: d7006a2b 0202530d
	v_add_co_ci_u32_e64 v44, null, 0, v36, vcc_lo              // 000000001c64: d5207c2c 01aa4880
	s_delay_alu instid0(valu_dep_3)                            // 000000001c6c: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[14:15]                // 000000001c70: 7ca81c0c
	v_cmp_gt_i64_e64 s0, s[12:13], v[1:2]                      // 000000001c74: d4540000 0202020c
	v_or_b32_e32 v0, 1, v14                                    // 000000001c7c: 38001c81
	v_mov_b32_e32 v1, s23                                      // 000000001c80: 7e020217
	v_mov_b32_e32 v39, 0                                       // 000000001c84: 7e4e0280
	v_mad_co_u64_u32 v[8:9], null, s14, v41, v[11:12]          // 000000001c88: d6fe7c08 042e520e
	s_wait_alu depctr_va_vcc(0)                                // 000000001c90: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v14, vcc_lo                       // 000000001c94: 02041c80
	v_cndmask_b32_e64 v3, 0, s23, vcc_lo                       // 000000001c98: d5010003 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001ca0: 7ca8000c
	v_or_b32_e32 v1, 2, v14                                    // 000000001ca4: 38021c82
	v_cmp_gt_i64_e64 s1, s[14:15], v[11:12]                    // 000000001ca8: d4540001 0202160e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cb0: bf88ff9e
	v_mul_lo_u32 v4, s3, v2                                    // 000000001cb4: d72c0004 02020403
	v_mad_co_u64_u32 v[16:17], null, s4, v2, s[24:25]          // 000000001cbc: d6fe7c10 00620404
	v_mov_b32_e32 v2, s23                                      // 000000001cc4: 7e040217
	v_mul_lo_u32 v3, s4, v3                                    // 000000001cc8: d72c0003 02020604
	s_wait_alu depctr_va_vcc(0)                                // 000000001cd0: bf88ff9d
	v_cndmask_b32_e32 v5, 0, v0, vcc_lo                        // 000000001cd4: 020a0080
	v_cndmask_b32_e64 v0, 0, s23, vcc_lo                       // 000000001cd8: d5010000 01a82e80
	v_mad_co_u64_u32 v[9:10], null, s15, v41, v[9:10]          // 000000001ce0: d6fe7c09 0426520f
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[1:2]                  // 000000001ce8: 7ca8020c
	v_or_b32_e32 v2, 4, v14                                    // 000000001cec: 38041c84
	s_wait_alu depctr_va_sdst(0)                               // 000000001cf0: bf88f19f
	v_cndmask_b32_e64 v7, 0, v11, s1                           // 000000001cf4: d5010007 00061680
	v_mul_lo_u32 v10, s4, v0                                   // 000000001cfc: d72c000a 02020004
	v_add3_u32 v17, v4, v17, v3                                // 000000001d04: d6550011 040e2304
	v_or_b32_e32 v0, 3, v14                                    // 000000001d0c: 38001c83
	s_wait_alu depctr_va_vcc(0)                                // 000000001d10: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v1 :: v_dual_mov_b32 v1, s23     // 000000001d14: ca500280 04000017
	v_cndmask_b32_e64 v20, 0, s23, vcc_lo                      // 000000001d1c: d5010014 01a82e80
	v_mov_b32_e32 v3, s23                                      // 000000001d24: 7e060217
	v_mul_lo_u32 v34, s3, v5                                   // 000000001d28: d72c0022 02020a03
	v_mad_co_u64_u32 v[18:19], null, s4, v5, s[24:25]          // 000000001d30: d6fe7c12 00620a04
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001d38: 7ca8000c
	v_mul_lo_u32 v5, s4, v20                                   // 000000001d3c: d72c0005 02022804
	v_mul_lo_u32 v35, s3, v4                                   // 000000001d44: d72c0023 02020803
	v_mad_co_u64_u32 v[20:21], null, s4, v4, s[24:25]          // 000000001d4c: d6fe7c14 00620804
	v_cndmask_b32_e64 v6, 0, v12, s1                           // 000000001d54: d5010006 00061880
	s_lshl_b64 s[34:35], s[14:15], 1                           // 000000001d5c: 84a2810e
	s_wait_alu depctr_va_vcc(0)                                // 000000001d60: bf88ff9d
	v_cndmask_b32_e64 v4, 0, s23, vcc_lo                       // 000000001d64: d5010004 01a82e80
	v_add3_u32 v19, v34, v19, v10                              // 000000001d6c: d6550013 042a2722
	v_mov_b32_e32 v10, 0                                       // 000000001d74: 7e140280
	s_mul_u64 s[36:37], s[14:15], 3                            // 000000001d78: aaa4830e
	s_lshl_b64 s[38:39], s[14:15], 2                           // 000000001d7c: 84a6820e
	v_mul_lo_u32 v38, s4, v4                                   // 000000001d80: d72c0026 02020804
	v_mov_b32_e32 v4, s23                                      // 000000001d88: 7e080217
	v_cmp_gt_i64_e64 s2, s[12:13], v[2:3]                      // 000000001d8c: d4540002 0202040c
	v_cndmask_b32_e32 v3, 0, v0, vcc_lo                        // 000000001d94: 02060080
	v_or_b32_e32 v0, 5, v14                                    // 000000001d98: 38001c85
	v_add3_u32 v21, v35, v21, v5                               // 000000001d9c: d6550015 04162b23
	s_mul_u64 s[40:41], s[14:15], 5                            // 000000001da4: aaa8850e
	s_mul_u64 s[42:43], s[14:15], 6                            // 000000001da8: aaaa860e
	v_cndmask_b32_e64 v24, 0, v2, s2                           // 000000001dac: d5010018 000a0480
	v_cndmask_b32_e64 v2, 0, s23, s2                           // 000000001db4: d5010002 00082e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001dbc: 7ca8000c
	v_mul_lo_u32 v40, s3, v3                                   // 000000001dc0: d72c0028 02020603
	v_mad_co_u64_u32 v[22:23], null, s4, v3, s[24:25]          // 000000001dc8: d6fe7c16 00620604
	v_or_b32_e32 v1, 6, v14                                    // 000000001dd0: 38021c86
	v_mul_lo_u32 v42, s4, v2                                   // 000000001dd4: d72c002a 02020404
	v_mov_b32_e32 v2, s23                                      // 000000001ddc: 7e040217
	v_or_b32_e32 v3, 7, v14                                    // 000000001de0: 38061c87
	s_wait_alu depctr_va_vcc(0)                                // 000000001de4: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001de8: 02000080
	v_cndmask_b32_e64 v26, 0, s23, vcc_lo                      // 000000001dec: d501001a 01a82e80
	s_mul_u64 s[44:45], s[14:15], 7                            // 000000001df4: aaac870e
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[1:2]                  // 000000001df8: 7ca8020c
	v_cmp_gt_i64_e64 s2, s[12:13], v[3:4]                      // 000000001dfc: d4540002 0202060c
	v_mul_lo_u32 v2, s3, v24                                   // 000000001e04: d72c0002 02023003
	v_mad_co_u64_u32 v[24:25], null, s4, v24, s[24:25]         // 000000001e0c: d6fe7c18 00623004
	v_mul_lo_u32 v4, s4, v26                                   // 000000001e14: d72c0004 02023404
	v_mul_lo_u32 v45, s3, v0                                   // 000000001e1c: d72c002d 02020003
	s_wait_alu depctr_va_vcc(0)                                // 000000001e24: bf88ff9d
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001e28: 02020280
	v_cndmask_b32_e64 v28, 0, s23, vcc_lo                      // 000000001e2c: d501001c 01a82e80
	s_wait_alu depctr_va_sdst(0)                               // 000000001e34: bf88f19f
	v_cndmask_b32_e64 v3, 0, v3, s2                            // 000000001e38: d5010003 000a0680
	v_cndmask_b32_e64 v30, 0, s23, s2                          // 000000001e40: d501001e 00082e80
	v_mad_co_u64_u32 v[26:27], null, s4, v0, s[24:25]          // 000000001e48: d6fe7c1a 00620004
	v_add3_u32 v25, v2, v25, v42                               // 000000001e50: d6550019 04aa3302
	v_mul_lo_u32 v0, s4, v28                                   // 000000001e58: d72c0000 02023804
	v_mul_lo_u32 v47, s3, v3                                   // 000000001e60: d72c002f 02020603
	v_mov_b32_e32 v42, 0                                       // 000000001e68: 7e540280
	v_mul_lo_u32 v46, s3, v1                                   // 000000001e6c: d72c002e 02020203
	v_mad_co_u64_u32 v[28:29], null, s4, v1, s[24:25]          // 000000001e74: d6fe7c1c 00620204
	v_mul_lo_u32 v1, s4, v30                                   // 000000001e7c: d72c0001 02023c04
	v_mad_co_u64_u32 v[30:31], null, s4, v3, s[24:25]          // 000000001e84: d6fe7c1e 00620604
	v_add_co_u32 v32, vcc_lo, s20, v7                          // 000000001e8c: d7006a20 02020e14
	s_lshl_b64 s[2:3], s[14:15], 4                             // 000000001e94: 8482840e
	s_wait_alu depctr_va_vcc(0)                                // 000000001e98: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s21, v6, vcc_lo             // 000000001e9c: d5207c21 01aa0c15
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ea4: bf88ff9e
	v_add_co_u32 v34, vcc_lo, s2, v8                           // 000000001ea8: d7006a22 02021002
	v_add3_u32 v23, v40, v23, v38                              // 000000001eb0: d6550017 049a2f28
	v_add3_u32 v27, v45, v27, v4                               // 000000001eb8: d655001b 0412372d
	v_add3_u32 v29, v46, v29, v0                               // 000000001ec0: d655001d 04023b2e
	v_add3_u32 v31, v47, v31, v1                               // 000000001ec8: d655001f 04063f2f
	s_wait_alu depctr_va_vcc(0)                                // 000000001ed0: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s3, v9, vcc_lo              // 000000001ed4: d5207c23 01aa1203
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v40, 0             // 000000001edc: ca100080 2f280080
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0             // 000000001ee4: ca100080 2e2c0080
	v_mov_b32_e32 v38, 0                                       // 000000001eec: 7e4c0280
	s_mov_b64 s[46:47], 0                                      // 000000001ef0: beae0180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000001ef4: bf8701d9
	v_dual_mov_b32 v5, s47 :: v_dual_mov_b32 v2, s47           // 000000001ef8: ca10002f 0502002f
	v_or_b32_e32 v4, s46, v41                                  // 000000001f00: 3808522e
	v_add_co_u32 v48, vcc_lo, v43, s46                         // 000000001f04: d7006a30 02005d2b
	s_wait_alu depctr_va_vcc(0)                                // 000000001f0c: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s47, v44, vcc_lo            // 000000001f10: d5207c31 01aa582f
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[4:5]                  // 000000001f18: 7ca8081a
	s_mul_i32 s33, s46, s15                                    // 000000001f1c: 96210f2e
	v_mov_b32_e32 v53, s47                                     // 000000001f20: 7e6a022f
	s_and_b32 s2, s0, vcc_lo                                   // 000000001f24: 8b026a00
	s_and_b32 vcc_lo, s1, vcc_lo                               // 000000001f28: 8b6a6a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f2c: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v48, s2                           // 000000001f30: d5010000 000a6080
	v_cndmask_b32_e64 v1, 0, v49, s2                           // 000000001f38: d5010001 000a6280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000001f40: bf870122
	v_add_co_u32 v0, s3, s28, v0                               // 000000001f44: d7000300 0202001c
	s_wait_alu depctr_va_sdst(0)                               // 000000001f4c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s29, v1, s3                  // 000000001f50: d5207c01 000e021d
	global_load_d16_u8 v0, v[0:1], off                         // 000000001f58: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v4                                     // 000000001f64: 38020881
	s_wait_loadcnt 0x0                                         // 000000001f68: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s2                            // 000000001f6c: d65d0000 000a0080
	v_add_co_u32 v3, s2, v48, 1                                // 000000001f74: d7000203 02010330
	s_wait_alu depctr_va_sdst(0)                               // 000000001f7c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v49, s2                   // 000000001f80: d5207c06 000a6280
	v_cmp_gt_i64_e64 s2, s[26:27], v[1:2]                      // 000000001f88: d4540002 0202021a
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000001f90: d7620000 020200ff 000000ff
	s_and_b32 s3, s0, s2                                       // 000000001f9c: 8b030200
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa0: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s3                            // 000000001fa4: d5010001 000e0680
	v_cndmask_b32_e64 v2, 0, v6, s3                            // 000000001fac: d5010002 000e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000001fb4: bf870122
	v_add_co_u32 v1, s4, s28, v1                               // 000000001fb8: d7000401 0202021c
	s_wait_alu depctr_va_sdst(0)                               // 000000001fc0: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s29, v2, s4                  // 000000001fc4: d5207c02 0012041d
	global_load_d16_hi_u8 v0, v[1:2], off                      // 000000001fcc: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v4                                     // 000000001fd8: 38020882
	v_mov_b32_e32 v2, s47                                      // 000000001fdc: 7e04022f
	s_wait_loadcnt 0x0                                         // 000000001fe0: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s3                            // 000000001fe4: d65d5000 000e0080
	v_add_co_u32 v3, s3, v48, 2                                // 000000001fec: d7000303 02010530
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v49, s3                   // 000000001ff8: d5207c06 000e6280
	v_cmp_gt_i64_e64 s3, s[26:27], v[1:2]                      // 000000002000: d4540003 0202021a
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 000000002008: d7385000 02020088
	s_and_b32 s4, s0, s3                                       // 000000002010: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002014: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s4                            // 000000002018: d5010001 00120680
	v_cndmask_b32_e64 v2, 0, v6, s4                            // 000000002020: d5010002 00120c80
	v_mov_b32_e32 v3, s47                                      // 000000002028: 7e06022f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000202c: bf8701a3
	v_add_co_u32 v1, s5, s28, v1                               // 000000002030: d7000501 0202021c
	s_wait_alu depctr_va_sdst(0)                               // 000000002038: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s29, v2, s5                  // 00000000203c: d5207c02 0016041d
	global_load_d16_u8 v1, v[1:2], off                         // 000000002044: ee07807c 00000001 00000001
	v_or_b32_e32 v2, 3, v4                                     // 000000002050: 38040883
	s_wait_loadcnt 0x0                                         // 000000002054: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s4                            // 000000002058: d65d0001 00120280
	v_add_co_u32 v6, s4, v48, 3                                // 000000002060: d7000406 02010730
	s_wait_alu depctr_va_sdst(0)                               // 000000002068: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s4                   // 00000000206c: d5207c07 00126280
	v_cmp_gt_i64_e64 s4, s[26:27], v[2:3]                      // 000000002074: d4540004 0202041a
	v_and_b16 v1.l, 0xff, v1.l                                 // 00000000207c: d7620001 020202ff 000000ff
	s_and_b32 s5, s0, s4                                       // 000000002088: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 00000000208c: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s5                            // 000000002090: d5010002 00160c80
	v_cndmask_b32_e64 v3, 0, v7, s5                            // 000000002098: d5010003 00160e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000020a0: bf870122
	v_add_co_u32 v2, s6, s28, v2                               // 0000000020a4: d7000602 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 0000000020ac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s6                  // 0000000020b0: d5207c03 001a061d
	global_load_d16_hi_u8 v1, v[2:3], off                      // 0000000020b8: ee08407c 00000001 00000002
	v_or_b32_e32 v2, 4, v4                                     // 0000000020c4: 38040884
	v_mov_b32_e32 v3, s47                                      // 0000000020c8: 7e06022f
	s_wait_loadcnt 0x0                                         // 0000000020cc: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s5                            // 0000000020d0: d65d5001 00160280
	v_add_co_u32 v6, s5, v48, 4                                // 0000000020d8: d7000506 02010930
	s_wait_alu depctr_va_sdst(0)                               // 0000000020e0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s5                   // 0000000020e4: d5207c07 00166280
	v_cmp_gt_i64_e64 s5, s[26:27], v[2:3]                      // 0000000020ec: d4540005 0202041a
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 0000000020f4: d7385001 02020288
	s_and_b32 s6, s0, s5                                       // 0000000020fc: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002100: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v6, s6                            // 000000002104: d5010002 001a0c80
	v_cndmask_b32_e64 v3, 0, v7, s6                            // 00000000210c: d5010003 001a0e80
	v_or_b32_e32 v6, 5, v4                                     // 000000002114: 380c0885
	v_mov_b32_e32 v7, s47                                      // 000000002118: 7e0e022f
	s_delay_alu instid0(valu_dep_4)                            // 00000000211c: bf870004
	v_add_co_u32 v2, s7, s28, v2                               // 000000002120: d7000702 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000002128: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s7                  // 00000000212c: d5207c03 001e061d
	global_load_d16_u8 v2, v[2:3], off                         // 000000002134: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000002140: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s6                            // 000000002144: d65d0002 001a0480
	v_add_co_u32 v3, s6, v48, 5                                // 00000000214c: d7000603 02010b30
	s_wait_alu depctr_va_sdst(0)                               // 000000002154: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v49, s6                  // 000000002158: d5207c32 001a6280
	v_cmp_gt_i64_e64 s6, s[26:27], v[6:7]                      // 000000002160: d4540006 02020c1a
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002168: d7620002 020204ff 000000ff
	s_and_b32 s7, s0, s6                                       // 000000002174: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002178: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s7                            // 00000000217c: d5010003 001e0680
	v_cndmask_b32_e64 v7, 0, v50, s7                           // 000000002184: d5010007 001e6480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000218c: bf870122
	v_add_co_u32 v6, s8, s28, v3                               // 000000002190: d7000806 0202061c
	s_wait_alu depctr_va_sdst(0)                               // 000000002198: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s29, v7, s8                  // 00000000219c: d5207c07 00220e1d
	global_load_d16_hi_u8 v2, v[6:7], off                      // 0000000021a4: ee08407c 00000002 00000006
	v_or_b32_e32 v6, 6, v4                                     // 0000000021b0: 380c0886
	v_mov_b32_e32 v7, s47                                      // 0000000021b4: 7e0e022f
	v_or_b32_e32 v4, 7, v4                                     // 0000000021b8: 38080887
	s_wait_loadcnt 0x0                                         // 0000000021bc: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s7                            // 0000000021c0: d65d5002 001e0480
	v_add_co_u32 v3, s7, v48, 6                                // 0000000021c8: d7000703 02010d30
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d0: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v49, s7                  // 0000000021d4: d5207c32 001e6280
	v_cmp_gt_i64_e64 s7, s[26:27], v[6:7]                      // 0000000021dc: d4540007 02020c1a
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000021e4: d7385002 02020488
	s_and_b32 s8, s0, s7                                       // 0000000021ec: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021f0: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s8                            // 0000000021f4: d5010003 00220680
	v_cndmask_b32_e64 v7, 0, v50, s8                           // 0000000021fc: d5010007 00226480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002204: bf870122
	v_add_co_u32 v6, s9, s28, v3                               // 000000002208: d7000906 0202061c
	s_wait_alu depctr_va_sdst(0)                               // 000000002210: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s29, v7, s9                  // 000000002214: d5207c07 00260e1d
	global_load_d16_u8 v3, v[6:7], off                         // 00000000221c: ee07807c 00000003 00000006
	s_wait_loadcnt 0x0                                         // 000000002228: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s8                            // 00000000222c: d65d0003 00220680
	v_add_co_u32 v6, s8, v48, 7                                // 000000002234: d7000806 02010f30
	s_wait_alu depctr_va_sdst(0)                               // 00000000223c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v49, s8                   // 000000002240: d5207c07 00226280
	v_cmp_gt_i64_e64 s8, s[26:27], v[4:5]                      // 000000002248: d4540008 0202081a
	v_or_b16 v48.l, v0.l, v0.h op_sel:[0,1,0]                  // 000000002250: d7631030 02020100
	v_or_b16 v48.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002258: d7635030 02020301
	v_or_b16 v49.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002260: d7631031 02020502
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002268: d7620003 020206ff 000000ff
	s_and_b32 s9, s0, s8                                       // 000000002274: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002278: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s9                            // 00000000227c: d5010004 00260c80
	v_cndmask_b32_e64 v5, 0, v7, s9                            // 000000002284: d5010005 00260e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000228c: bf870122
	v_add_co_u32 v4, s10, s28, v4                              // 000000002290: d7000a04 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000002298: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s10                 // 00000000229c: d5207c05 002a0a1d
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000022a4: ee08407c 00000003 00000004
	v_mad_co_u64_u32 v[4:5], null, s46, s14, v[8:9]            // 0000000022b0: d6fe7c04 04201c2e
	s_delay_alu instid0(valu_dep_1)                            // 0000000022b8: bf870001
	v_cndmask_b32_e32 v0, 0, v4, vcc_lo                        // 0000000022bc: 02000880
	s_wait_loadcnt 0x0                                         // 0000000022c0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s9                            // 0000000022c4: d65d5003 00260680
	s_mul_i32 s9, s47, s14                                     // 0000000022cc: 96090e2f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	s_add_co_i32 s33, s33, s9                                  // 0000000022d4: 81210921
	v_add_co_u32 v0, s9, s30, v0                               // 0000000022d8: d7000900 0202001e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022e0: bf88ff9e
	v_add_nc_u32_e32 v7, s33, v5                               // 0000000022e4: 4a0e0a21
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000022e8: d7385003 02020688
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000022f0: bf870112
	v_cndmask_b32_e32 v1, 0, v7, vcc_lo                        // 0000000022f4: 02020e80
	v_or_b16 v49.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000022f8: d7635031 02020703
	s_wait_alu depctr_va_sdst(0)                               // 000000002300: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002304: bf870002
	v_add_co_ci_u32_e64 v1, null, s31, v1, s9                  // 000000002308: d5207c01 0026021f
	global_load_d16_u8 v0, v[0:1], off                         // 000000002310: ee07807c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 00000000231c: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, vcc_lo                        // 000000002320: d65d0000 01aa0080
	v_add_co_u32 v1, vcc_lo, v4, s14                           // 000000002328: d7006a01 02001d04
	s_wait_alu depctr_va_vcc(0)                                // 000000002330: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s15, v7, vcc_lo              // 000000002334: d5207c02 01aa0e0f
	s_and_b32 vcc_lo, s1, s2                                   // 00000000233c: 8b6a0201
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002340: d7620000 020200ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000234c: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 000000002350: ca520280 01020480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002358: bf870121
	v_add_co_u32 v1, s2, s30, v1                               // 00000000235c: d7000201 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002364: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s31, v2, s2                  // 000000002368: d5207c02 000a041f
	global_load_d16_hi_u8 v0, v[1:2], off                      // 000000002370: ee08407c 00000000 00000001
	s_wait_loadcnt 0x0                                         // 00000000237c: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, vcc_lo                        // 000000002380: d65d5000 01aa0080
	v_add_co_u32 v1, vcc_lo, v4, s34                           // 000000002388: d7006a01 02004504
	s_wait_alu depctr_va_vcc(0)                                // 000000002390: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s35, v7, vcc_lo              // 000000002394: d5207c02 01aa0e23
	s_and_b32 vcc_lo, s1, s3                                   // 00000000239c: 8b6a0301
	v_lshlrev_b16 v0.h, 8, v0.h op_sel:[0,1,1]                 // 0000000023a0: d7385000 02020088
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023a8: bf88ff9e
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_cndmask_b32 v2, 0, v2// 0000000023ac: ca520280 01020480
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000023b4: bf870112
	v_or_b16 v50.l, v0.l, v0.h op_sel:[0,1,0]                  // 0000000023b8: d7631032 02020100
	v_add_co_u32 v1, s2, s30, v1                               // 0000000023c0: d7000201 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000023cc: bf870003
	v_add_co_ci_u32_e64 v2, null, s31, v2, s2                  // 0000000023d0: d5207c02 000a041f
	global_load_d16_u8 v1, v[1:2], off                         // 0000000023d8: ee07807c 00000001 00000001
	s_wait_loadcnt 0x0                                         // 0000000023e4: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, vcc_lo                        // 0000000023e8: d65d0001 01aa0280
	v_add_co_u32 v2, vcc_lo, v4, s36                           // 0000000023f0: d7006a02 02004904
	s_wait_alu depctr_va_vcc(0)                                // 0000000023f8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s37, v7, vcc_lo              // 0000000023fc: d5207c03 01aa0e25
	s_and_b32 vcc_lo, s1, s4                                   // 000000002404: 8b6a0401
	v_and_b16 v1.l, 0xff, v1.l                                 // 000000002408: d7620001 020202ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002414: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 000000002418: ca520480 02020680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002420: bf870121
	v_add_co_u32 v2, s2, s30, v2                               // 000000002424: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 00000000242c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000002430: d5207c03 000a061f
	global_load_d16_hi_u8 v1, v[2:3], off                      // 000000002438: ee08407c 00000001 00000002
	s_wait_loadcnt 0x0                                         // 000000002444: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, vcc_lo                        // 000000002448: d65d5001 01aa0280
	v_add_co_u32 v2, vcc_lo, v4, s38                           // 000000002450: d7006a02 02004d04
	s_wait_alu depctr_va_vcc(0)                                // 000000002458: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s39, v7, vcc_lo              // 00000000245c: d5207c03 01aa0e27
	s_and_b32 vcc_lo, s1, s5                                   // 000000002464: 8b6a0501
	v_lshlrev_b16 v1.h, 8, v1.h op_sel:[0,1,1]                 // 000000002468: d7385001 02020288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002470: bf88ff9e
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_cndmask_b32 v3, 0, v3// 000000002474: ca520480 02020680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000247c: bf870112
	v_or_b16 v50.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002480: d7635032 02020301
	v_add_co_u32 v2, s2, s30, v2                               // 000000002488: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002490: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002494: bf870003
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000002498: d5207c03 000a061f
	global_load_d16_u8 v2, v[2:3], off                         // 0000000024a0: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 0000000024ac: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 0000000024b0: d65d0002 01aa0480
	v_add_co_u32 v3, vcc_lo, v4, s40                           // 0000000024b8: d7006a03 02005104
	s_wait_alu depctr_va_vcc(0)                                // 0000000024c0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s41, v7, vcc_lo              // 0000000024c4: d5207c05 01aa0e29
	s_and_b32 vcc_lo, s1, s6                                   // 0000000024cc: 8b6a0601
	v_and_b16 v2.l, 0xff, v2.l                                 // 0000000024d0: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024dc: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v6, 0, v5// 0000000024e0: ca520680 03060a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024e8: bf870121
	v_add_co_u32 v5, s2, s30, v3                               // 0000000024ec: d7000205 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 0000000024f4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s2                  // 0000000024f8: d5207c06 000a0c1f
	global_load_d16_hi_u8 v2, v[5:6], off                      // 000000002500: ee08407c 00000002 00000005
	s_wait_loadcnt 0x0                                         // 00000000250c: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 000000002510: d65d5002 01aa0480
	v_add_co_u32 v3, vcc_lo, v4, s42                           // 000000002518: d7006a03 02005504
	s_wait_alu depctr_va_vcc(0)                                // 000000002520: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s43, v7, vcc_lo              // 000000002524: d5207c05 01aa0e2b
	s_and_b32 vcc_lo, s1, s7                                   // 00000000252c: 8b6a0701
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002530: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002538: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v6, 0, v5// 00000000253c: ca520680 03060a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002544: bf870112
	v_or_b16 v51.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002548: d7631033 02020502
	v_add_co_u32 v5, s2, s30, v3                               // 000000002550: d7000205 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002558: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000255c: bf870003
	v_add_co_ci_u32_e64 v6, null, s31, v6, s2                  // 000000002560: d5207c06 000a0c1f
	global_load_d16_u8 v3, v[5:6], off                         // 000000002568: ee07807c 00000003 00000005
	s_wait_loadcnt 0x0                                         // 000000002574: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 000000002578: d65d0003 01aa0680
	v_add_co_u32 v4, vcc_lo, v4, s44                           // 000000002580: d7006a04 02005904
	s_wait_alu depctr_va_vcc(0)                                // 000000002588: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s45, v7, vcc_lo              // 00000000258c: d5207c05 01aa0e2d
	s_and_b32 vcc_lo, s1, s8                                   // 000000002594: 8b6a0801
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002598: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025a4: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v5// 0000000025a8: ca520880 04040a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025b0: bf870121
	v_add_co_u32 v4, s2, s30, v4                               // 0000000025b4: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000025bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 0000000025c0: d5207c05 000a0a1f
	s_or_b32 s2, s46, 16                                       // 0000000025c8: 8c02902e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025cc: bf88ff9e
	v_or_b32_e32 v52, s2, v41                                  // 0000000025d0: 38685202
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000025d4: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000025e0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 0000000025e4: d65d5003 01aa0680
	v_add_co_u32 v56, vcc_lo, v43, s2                          // 0000000025ec: d7006a38 0200052b
	s_wait_alu depctr_va_vcc(0)                                // 0000000025f4: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s47, v44, vcc_lo            // 0000000025f8: d5207c39 01aa582f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002600: bf870123
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002604: d7385003 02020688
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[52:53]                // 00000000260c: 7ca8681a
	v_or_b16 v51.h, v3.l, v3.h op_sel:[0,1,1]                  // 000000002610: d7635033 02020703
	s_and_b32 s2, s0, vcc_lo                                   // 000000002618: 8b026a00
	s_and_b32 vcc_lo, s1, vcc_lo                               // 00000000261c: 8b6a6a01
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000002620: bf8701d1
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[48:49], v[50:51], 0  // 000000002624: cc464000 1a026530
	s_wait_alu depctr_sa_sdst(0)                               // 00000000262c: bf88ff9e
	v_cndmask_b32_e64 v48, 0, v56, s2                          // 000000002630: d5010030 000a7080
	v_cndmask_b32_e64 v49, 0, v57, s2                          // 000000002638: d5010031 000a7280
	v_mov_b32_e32 v50, s47                                     // 000000002640: 7e64022f
	v_add_co_u32 v48, s3, s28, v48                             // 000000002644: d7000330 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 00000000264c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002650: bf870003
	v_add_co_ci_u32_e64 v49, null, s29, v49, s3                // 000000002654: d5207c31 000e621d
	global_load_d16_u8 v48, v[48:49], off                      // 00000000265c: ee07807c 00000030 00000030
	v_or_b32_e32 v49, 1, v52                                   // 000000002668: 38626881
	s_wait_loadcnt 0x0                                         // 00000000266c: bfc00000
	v_cndmask_b16 v48.l, 0, v48.l, s2                          // 000000002670: d65d0030 000a6080
	v_add_co_u32 v51, s2, v56, 1                               // 000000002678: d7000233 02010338
	s_wait_alu depctr_va_sdst(0)                               // 000000002680: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s2                  // 000000002684: d5207c36 000a7280
	v_cmp_gt_i64_e64 s2, s[26:27], v[49:50]                    // 00000000268c: d4540002 0202621a
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002694: d7620030 020260ff 000000ff
	s_and_b32 s3, s0, s2                                       // 0000000026a0: 8b030200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026a4: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v51, s3                          // 0000000026a8: d5010031 000e6680
	v_cndmask_b32_e64 v50, 0, v54, s3                          // 0000000026b0: d5010032 000e6c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000026b8: bf870122
	v_add_co_u32 v49, s4, s28, v49                             // 0000000026bc: d7000431 0202621c
	s_wait_alu depctr_va_sdst(0)                               // 0000000026c4: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s29, v50, s4                // 0000000026c8: d5207c32 0012641d
	global_load_d16_hi_u8 v48, v[49:50], off                   // 0000000026d0: ee08407c 00000030 00000031
	v_or_b32_e32 v49, 2, v52                                   // 0000000026dc: 38626882
	v_mov_b32_e32 v50, s47                                     // 0000000026e0: 7e64022f
	s_wait_loadcnt 0x0                                         // 0000000026e4: bfc00000
	v_cndmask_b16 v48.h, 0, v48.h, s3                          // 0000000026e8: d65d5030 000e6080
	v_add_co_u32 v51, s3, v56, 2                               // 0000000026f0: d7000333 02010538
	s_wait_alu depctr_va_sdst(0)                               // 0000000026f8: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s3                  // 0000000026fc: d5207c36 000e7280
	v_cmp_gt_i64_e64 s3, s[26:27], v[49:50]                    // 000000002704: d4540003 0202621a
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 00000000270c: d7385030 02026088
	s_and_b32 s4, s0, s3                                       // 000000002714: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002718: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v51, s4                          // 00000000271c: d5010031 00126680
	v_cndmask_b32_e64 v50, 0, v54, s4                          // 000000002724: d5010032 00126c80
	v_mov_b32_e32 v51, s47                                     // 00000000272c: 7e66022f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002730: bf8701a3
	v_add_co_u32 v49, s5, s28, v49                             // 000000002734: d7000531 0202621c
	s_wait_alu depctr_va_sdst(0)                               // 00000000273c: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s29, v50, s5                // 000000002740: d5207c32 0016641d
	global_load_d16_u8 v49, v[49:50], off                      // 000000002748: ee07807c 00000031 00000031
	v_or_b32_e32 v50, 3, v52                                   // 000000002754: 38646883
	s_wait_loadcnt 0x0                                         // 000000002758: bfc00000
	v_cndmask_b16 v49.l, 0, v49.l, s4                          // 00000000275c: d65d0031 00126280
	v_add_co_u32 v54, s4, v56, 3                               // 000000002764: d7000436 02010738
	s_wait_alu depctr_va_sdst(0)                               // 00000000276c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s4                  // 000000002770: d5207c37 00127280
	v_cmp_gt_i64_e64 s4, s[26:27], v[50:51]                    // 000000002778: d4540004 0202641a
	v_and_b16 v49.l, 0xff, v49.l                               // 000000002780: d7620031 020262ff 000000ff
	s_and_b32 s5, s0, s4                                       // 00000000278c: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002790: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s5                          // 000000002794: d5010032 00166c80
	v_cndmask_b32_e64 v51, 0, v55, s5                          // 00000000279c: d5010033 00166e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000027a4: bf870122
	v_add_co_u32 v50, s6, s28, v50                             // 0000000027a8: d7000632 0202641c
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b0: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s6                // 0000000027b4: d5207c33 001a661d
	global_load_d16_hi_u8 v49, v[50:51], off                   // 0000000027bc: ee08407c 00000031 00000032
	v_or_b32_e32 v50, 4, v52                                   // 0000000027c8: 38646884
	v_mov_b32_e32 v51, s47                                     // 0000000027cc: 7e66022f
	s_wait_loadcnt 0x0                                         // 0000000027d0: bfc00000
	v_cndmask_b16 v49.h, 0, v49.h, s5                          // 0000000027d4: d65d5031 00166280
	v_add_co_u32 v54, s5, v56, 4                               // 0000000027dc: d7000536 02010938
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e4: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s5                  // 0000000027e8: d5207c37 00167280
	v_cmp_gt_i64_e64 s5, s[26:27], v[50:51]                    // 0000000027f0: d4540005 0202641a
	v_lshlrev_b16 v49.h, 8, v49.h op_sel:[0,1,1]               // 0000000027f8: d7385031 02026288
	s_and_b32 s6, s0, s5                                       // 000000002800: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002804: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s6                          // 000000002808: d5010032 001a6c80
	v_cndmask_b32_e64 v51, 0, v55, s6                          // 000000002810: d5010033 001a6e80
	v_or_b32_e32 v54, 5, v52                                   // 000000002818: 386c6885
	v_mov_b32_e32 v55, s47                                     // 00000000281c: 7e6e022f
	s_delay_alu instid0(valu_dep_4)                            // 000000002820: bf870004
	v_add_co_u32 v50, s7, s28, v50                             // 000000002824: d7000732 0202641c
	s_wait_alu depctr_va_sdst(0)                               // 00000000282c: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s7                // 000000002830: d5207c33 001e661d
	global_load_d16_u8 v50, v[50:51], off                      // 000000002838: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002844: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, s6                          // 000000002848: d65d0032 001a6480
	v_add_co_u32 v51, s6, v56, 5                               // 000000002850: d7000633 02010b38
	s_wait_alu depctr_va_sdst(0)                               // 000000002858: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v57, s6                  // 00000000285c: d5207c3a 001a7280
	v_cmp_gt_i64_e64 s6, s[26:27], v[54:55]                    // 000000002864: d4540006 02026c1a
	v_and_b16 v50.l, 0xff, v50.l                               // 00000000286c: d7620032 020264ff 000000ff
	s_and_b32 s7, s0, s6                                       // 000000002878: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 00000000287c: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s7                          // 000000002880: d5010033 001e6680
	v_cndmask_b32_e64 v55, 0, v58, s7                          // 000000002888: d5010037 001e7480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002890: bf870122
	v_add_co_u32 v54, s8, s28, v51                             // 000000002894: d7000836 0202661c
	s_wait_alu depctr_va_sdst(0)                               // 00000000289c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s8                // 0000000028a0: d5207c37 00226e1d
	global_load_d16_hi_u8 v50, v[54:55], off                   // 0000000028a8: ee08407c 00000032 00000036
	v_or_b32_e32 v54, 6, v52                                   // 0000000028b4: 386c6886
	v_mov_b32_e32 v55, s47                                     // 0000000028b8: 7e6e022f
	v_or_b32_e32 v52, 7, v52                                   // 0000000028bc: 38686887
	s_wait_loadcnt 0x0                                         // 0000000028c0: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, s7                          // 0000000028c4: d65d5032 001e6480
	v_add_co_u32 v51, s7, v56, 6                               // 0000000028cc: d7000733 02010d38
	s_wait_alu depctr_va_sdst(0)                               // 0000000028d4: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v57, s7                  // 0000000028d8: d5207c3a 001e7280
	v_cmp_gt_i64_e64 s7, s[26:27], v[54:55]                    // 0000000028e0: d4540007 02026c1a
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 0000000028e8: d7385032 02026488
	s_and_b32 s8, s0, s7                                       // 0000000028f0: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028f4: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s8                          // 0000000028f8: d5010033 00226680
	v_cndmask_b32_e64 v55, 0, v58, s8                          // 000000002900: d5010037 00227480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002908: bf870122
	v_add_co_u32 v54, s9, s28, v51                             // 00000000290c: d7000936 0202661c
	s_wait_alu depctr_va_sdst(0)                               // 000000002914: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s9                // 000000002918: d5207c37 00266e1d
	global_load_d16_u8 v51, v[54:55], off                      // 000000002920: ee07807c 00000033 00000036
	s_wait_loadcnt 0x0                                         // 00000000292c: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, s8                          // 000000002930: d65d0033 00226680
	v_add_co_u32 v54, s8, v56, 7                               // 000000002938: d7000836 02010f38
	s_wait_alu depctr_va_sdst(0)                               // 000000002940: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s8                  // 000000002944: d5207c37 00227280
	v_cmp_gt_i64_e64 s8, s[26:27], v[52:53]                    // 00000000294c: d4540008 0202681a
	v_and_b16 v51.l, 0xff, v51.l                               // 000000002954: d7620033 020266ff 000000ff
	s_and_b32 s9, s0, s8                                       // 000000002960: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002964: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v54, s9                          // 000000002968: d5010034 00266c80
	v_cndmask_b32_e64 v53, 0, v55, s9                          // 000000002970: d5010035 00266e80
	v_mad_co_u64_u32 v[54:55], null, s46, s14, v[34:35]        // 000000002978: d6fe7c36 04881c2e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002980: bf8701a3
	v_add_co_u32 v52, s10, s28, v52                            // 000000002984: d7000a34 0202681c
	s_wait_alu depctr_va_sdst(0)                               // 00000000298c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s29, v53, s10               // 000000002990: d5207c35 002a6a1d
	s_delay_alu instid0(valu_dep_3)                            // 000000002998: bf870003
	v_add_nc_u32_e32 v57, s33, v55                             // 00000000299c: 4a726e21
	global_load_d16_hi_u8 v51, v[52:53], off                   // 0000000029a0: ee08407c 00000033 00000034
	v_or_b16 v52.l, v48.l, v48.h op_sel:[0,1,0]                // 0000000029ac: d7631034 02026130
	v_cndmask_b32_e32 v48, 0, v54, vcc_lo                      // 0000000029b4: 02606c80
	v_or_b16 v52.h, v49.l, v49.h op_sel:[0,1,1]                // 0000000029b8: d7635034 02026331
	v_cndmask_b32_e32 v49, 0, v57, vcc_lo                      // 0000000029c0: 02627280
	v_or_b16 v53.l, v50.l, v50.h op_sel:[0,1,0]                // 0000000029c4: d7631035 02026532
	s_wait_loadcnt 0x0                                         // 0000000029cc: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, s9                          // 0000000029d0: d65d5033 00266680
	v_add_co_u32 v48, s9, s30, v48                             // 0000000029d8: d7000930 0202601e
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e0: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s31, v49, s9                // 0000000029e4: d5207c31 0026621f
	s_delay_alu instid0(valu_dep_3)                            // 0000000029ec: bf870003
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 0000000029f0: d7385033 02026688
	global_load_d16_u8 v48, v[48:49], off                      // 0000000029f8: ee07807c 00000030 00000030
	v_or_b16 v53.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002a04: d7635035 02026733
	s_wait_loadcnt 0x0                                         // 000000002a0c: bfc00000
	v_cndmask_b16 v48.l, 0, v48.l, vcc_lo                      // 000000002a10: d65d0030 01aa6080
	v_add_co_u32 v49, vcc_lo, v54, s14                         // 000000002a18: d7006a31 02001d36
	s_wait_alu depctr_va_vcc(0)                                // 000000002a20: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s15, v57, vcc_lo            // 000000002a24: d5207c32 01aa720f
	s_and_b32 vcc_lo, s1, s2                                   // 000000002a2c: 8b6a0201
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002a30: d7620030 020260ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a3c: bf88ff9e
	v_dual_cndmask_b32 v49, 0, v49 :: v_dual_cndmask_b32 v50, 0, v50// 000000002a40: ca526280 31326480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a48: bf870121
	v_add_co_u32 v49, s2, s30, v49                             // 000000002a4c: d7000231 0202621e
	s_wait_alu depctr_va_sdst(0)                               // 000000002a54: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s31, v50, s2                // 000000002a58: d5207c32 000a641f
	global_load_d16_hi_u8 v48, v[49:50], off                   // 000000002a60: ee08407c 00000030 00000031
	s_wait_loadcnt 0x0                                         // 000000002a6c: bfc00000
	v_cndmask_b16 v48.h, 0, v48.h, vcc_lo                      // 000000002a70: d65d5030 01aa6080
	v_add_co_u32 v49, vcc_lo, v54, s34                         // 000000002a78: d7006a31 02004536
	s_wait_alu depctr_va_vcc(0)                                // 000000002a80: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s35, v57, vcc_lo            // 000000002a84: d5207c32 01aa7223
	s_and_b32 vcc_lo, s1, s3                                   // 000000002a8c: 8b6a0301
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 000000002a90: d7385030 02026088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a98: bf88ff9e
	v_dual_cndmask_b32 v49, 0, v49 :: v_dual_cndmask_b32 v50, 0, v50// 000000002a9c: ca526280 31326480
	s_lshr_b32 s3, s47, 5                                      // 000000002aa4: 8503852f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002aa8: bf88ff9e
	s_mul_i32 s3, s3, s14                                      // 000000002aac: 96030e03
	s_delay_alu instid0(valu_dep_1)                            // 000000002ab0: bf870001
	v_add_co_u32 v49, s2, s30, v49                             // 000000002ab4: d7000231 0202621e
	s_wait_alu depctr_va_sdst(0)                               // 000000002abc: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s31, v50, s2                // 000000002ac0: d5207c32 000a641f
	global_load_d16_u8 v49, v[49:50], off                      // 000000002ac8: ee07807c 00000031 00000031
	s_wait_loadcnt 0x0                                         // 000000002ad4: bfc00000
	v_cndmask_b16 v49.l, 0, v49.l, vcc_lo                      // 000000002ad8: d65d0031 01aa6280
	v_add_co_u32 v50, vcc_lo, v54, s36                         // 000000002ae0: d7006a32 02004936
	s_wait_alu depctr_va_vcc(0)                                // 000000002ae8: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s37, v57, vcc_lo            // 000000002aec: d5207c33 01aa7225
	s_and_b32 vcc_lo, s1, s4                                   // 000000002af4: 8b6a0401
	v_and_b16 v49.l, 0xff, v49.l                               // 000000002af8: d7620031 020262ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b04: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002b08: ca526480 32326680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b10: bf870121
	v_add_co_u32 v50, s2, s30, v50                             // 000000002b14: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b1c: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002b20: d5207c33 000a661f
	global_load_d16_hi_u8 v49, v[50:51], off                   // 000000002b28: ee08407c 00000031 00000032
	s_wait_loadcnt 0x0                                         // 000000002b34: bfc00000
	v_cndmask_b16 v49.h, 0, v49.h, vcc_lo                      // 000000002b38: d65d5031 01aa6280
	v_add_co_u32 v50, vcc_lo, v54, s38                         // 000000002b40: d7006a32 02004d36
	s_wait_alu depctr_va_vcc(0)                                // 000000002b48: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s39, v57, vcc_lo            // 000000002b4c: d5207c33 01aa7227
	s_and_b32 vcc_lo, s1, s5                                   // 000000002b54: 8b6a0501
	v_lshlrev_b16 v49.h, 8, v49.h op_sel:[0,1,1]               // 000000002b58: d7385031 02026288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b60: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002b64: ca526480 32326680
	s_lshr_b64 s[4:5], s[46:47], 5                             // 000000002b6c: 8584852e
	s_add_nc_u64 s[46:47], s[46:47], 32                        // 000000002b70: a9aea02e
	s_delay_alu instid0(valu_dep_1)                            // 000000002b74: bf870001
	v_add_co_u32 v50, s2, s30, v50                             // 000000002b78: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b80: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002b84: d5207c33 000a661f
	global_load_d16_u8 v50, v[50:51], off                      // 000000002b8c: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002b98: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 000000002b9c: d65d0032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s40                         // 000000002ba4: d7006a33 02005136
	s_wait_alu depctr_va_vcc(0)                                // 000000002bac: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s41, v57, vcc_lo            // 000000002bb0: d5207c37 01aa7229
	s_and_b32 vcc_lo, s1, s6                                   // 000000002bb8: 8b6a0601
	v_and_b16 v50.l, 0xff, v50.l                               // 000000002bbc: d7620032 020264ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc8: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002bcc: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002bd0: 02706e80
	s_mul_i32 s6, s4, s15                                      // 000000002bd4: 96060f04
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bd8: bf870122
	v_add_co_u32 v55, s2, s30, v51                             // 000000002bdc: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002be4: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002be8: d5207c38 000a701f
	global_load_d16_hi_u8 v50, v[55:56], off                   // 000000002bf0: ee08407c 00000032 00000037
	s_wait_loadcnt 0x0                                         // 000000002bfc: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, vcc_lo                      // 000000002c00: d65d5032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s42                         // 000000002c08: d7006a33 02005536
	s_wait_alu depctr_va_vcc(0)                                // 000000002c10: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s43, v57, vcc_lo            // 000000002c14: d5207c37 01aa722b
	s_and_b32 vcc_lo, s1, s7                                   // 000000002c1c: 8b6a0701
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000002c20: d7385032 02026488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c28: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002c2c: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002c30: 02706e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c34: bf870122
	v_add_co_u32 v55, s2, s30, v51                             // 000000002c38: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c40: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002c44: d5207c38 000a701f
	global_load_d16_u8 v51, v[55:56], off                      // 000000002c4c: ee07807c 00000033 00000037
	s_wait_loadcnt 0x0                                         // 000000002c58: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, vcc_lo                      // 000000002c5c: d65d0033 01aa6680
	v_add_co_u32 v54, vcc_lo, v54, s44                         // 000000002c64: d7006a36 02005936
	s_wait_alu depctr_va_vcc(0)                                // 000000002c6c: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s45, v57, vcc_lo            // 000000002c70: d5207c37 01aa722d
	s_and_b32 vcc_lo, s1, s8                                   // 000000002c78: 8b6a0801
	v_and_b16 v51.l, 0xff, v51.l                               // 000000002c7c: d7620033 020266ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c88: bf88ff9e
	v_dual_cndmask_b32 v54, 0, v54 :: v_dual_cndmask_b32 v55, 0, v55// 000000002c8c: ca526c80 36366e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c94: bf870121
	v_add_co_u32 v54, s2, s30, v54                             // 000000002c98: d7000236 02026c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca0: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s31, v55, s2                // 000000002ca4: d5207c37 000a6e1f
	global_load_d16_hi_u8 v51, v[54:55], off                   // 000000002cac: ee08407c 00000033 00000036
	s_wait_loadcnt 0x0                                         // 000000002cb8: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, vcc_lo                      // 000000002cbc: d65d5033 01aa6680
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002cc4: bf870091
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000002cc8: d7385033 02026688
	v_or_b16 v51.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002cd0: d7635033 02026733
	v_or_b16 v51.l, v50.l, v50.h op_sel:[0,1,0]                // 000000002cd8: d7631033 02026532
	v_or_b16 v50.l, v48.l, v48.h op_sel:[0,1,0]                // 000000002ce0: d7631032 02026130
	v_add_co_u32 v48, vcc_lo, v16, s4                          // 000000002ce8: d7006a30 02000910
	v_or_b16 v50.h, v49.l, v49.h op_sel:[0,1,1]                // 000000002cf0: d7635032 02026331
	s_wait_alu depctr_va_vcc(0)                                // 000000002cf8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v17, vcc_lo             // 000000002cfc: d5207c31 01aa2205
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 000000002d04: bf8700b2
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[52:53], v[50:51], v[0:7]// 000000002d08: cc464000 1c026534
	global_load_u8 v50, v[48:49], off                          // 000000002d10: ee04007c 00000032 00000030
	v_mad_co_u64_u32 v[48:49], null, s4, s14, v[32:33]         // 000000002d1c: d6fe7c30 04801c04
	v_add3_u32 v49, s6, s3, v49                                // 000000002d24: d6550031 04c40606
	global_load_u8 v48, v[48:49], off                          // 000000002d2c: ee04007c 00000030 00000030
	s_wait_loadcnt 0x1                                         // 000000002d38: bfc00001
	v_cmp_eq_u32_e64 s2, 0xff, v50                             // 000000002d3c: d44a0002 020264ff 000000ff
	s_wait_loadcnt 0x0                                         // 000000002d48: bfc00000
	v_add_nc_u32_e32 v51, 0xffffff02, v48                      // 000000002d4c: 4a6660ff ffffff02
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v48                         // 000000002d54: 7c9460ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_1)// 000000002d5c: bf8700a2
	v_add_nc_u32_e32 v48, v51, v50                             // 000000002d60: 4a606533
	s_or_b32 s2, s2, vcc_lo                                    // 000000002d64: 8c026a02
	v_ldexp_f32 v0, v0, v48                                    // 000000002d68: d71c0000 02026100
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d70: bf88ff9e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000002d74: bf8701c1
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002d78: d5010000 0009ff00 7fc00000
	v_add_co_u32 v48, s2, v18, s4                              // 000000002d84: d7000230 02000912
	s_wait_alu depctr_va_sdst(0)                               // 000000002d8c: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s5, v19, s2                 // 000000002d90: d5207c31 000a2605
	v_add_f32_e32 v39, v39, v0                                 // 000000002d98: 064e0127
	global_load_u8 v0, v[48:49], off                           // 000000002d9c: ee04007c 00000000 00000030
	s_wait_loadcnt 0x0                                         // 000000002da8: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002dac: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002db8: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002dbc: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002dc0: bf8700a1
	v_ldexp_f32 v0, v1, v0                                     // 000000002dc4: d71c0000 02020101
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dcc: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002dd0: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002ddc: bf870001
	v_add_f32_e32 v47, v47, v0                                 // 000000002de0: 065e012f
	v_add_co_u32 v0, s2, v20, s4                               // 000000002de4: d7000200 02000914
	s_wait_alu depctr_va_sdst(0)                               // 000000002dec: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v21, s2                  // 000000002df0: d5207c01 000a2a05
	global_load_u8 v0, v[0:1], off                             // 000000002df8: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002e04: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002e08: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002e14: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002e18: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002e1c: bf8700a1
	v_ldexp_f32 v0, v2, v0                                     // 000000002e20: d71c0000 02020102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e28: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002e2c: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002e38: bf870001
	v_add_f32_e32 v46, v46, v0                                 // 000000002e3c: 065c012e
	v_add_co_u32 v0, s2, v22, s4                               // 000000002e40: d7000200 02000916
	s_wait_alu depctr_va_sdst(0)                               // 000000002e48: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v23, s2                  // 000000002e4c: d5207c01 000a2e05
	global_load_u8 v0, v[0:1], off                             // 000000002e54: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002e60: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002e64: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002e70: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002e74: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002e78: bf8700a1
	v_ldexp_f32 v0, v3, v0                                     // 000000002e7c: d71c0000 02020103
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e84: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002e88: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002e94: bf870001
	v_add_f32_e32 v45, v45, v0                                 // 000000002e98: 065a012d
	v_add_co_u32 v0, s2, v24, s4                               // 000000002e9c: d7000200 02000918
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v25, s2                  // 000000002ea8: d5207c01 000a3205
	global_load_u8 v0, v[0:1], off                             // 000000002eb0: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002ebc: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002ec0: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002ecc: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002ed0: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002ed4: bf8700a1
	v_ldexp_f32 v0, v4, v0                                     // 000000002ed8: d71c0000 02020104
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ee0: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002ee4: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002ef0: bf870001
	v_add_f32_e32 v42, v42, v0                                 // 000000002ef4: 0654012a
	v_add_co_u32 v0, s2, v26, s4                               // 000000002ef8: d7000200 0200091a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f00: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v27, s2                  // 000000002f04: d5207c01 000a3605
	global_load_u8 v0, v[0:1], off                             // 000000002f0c: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002f18: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002f1c: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002f28: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002f2c: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002f30: bf8700a1
	v_ldexp_f32 v0, v5, v0                                     // 000000002f34: d71c0000 02020105
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f3c: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002f40: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002f4c: bf870001
	v_add_f32_e32 v40, v40, v0                                 // 000000002f50: 06500128
	v_add_co_u32 v0, s2, v28, s4                               // 000000002f54: d7000200 0200091c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f5c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v29, s2                  // 000000002f60: d5207c01 000a3a05
	global_load_u8 v0, v[0:1], off                             // 000000002f68: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002f74: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002f78: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002f84: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002f88: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002f8c: bf8700a1
	v_ldexp_f32 v0, v6, v0                                     // 000000002f90: d71c0000 02020106
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f98: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002f9c: d5010000 0009ff00 7fc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000002fa8: bf870001
	v_add_f32_e32 v38, v38, v0                                 // 000000002fac: 064c0126
	v_add_co_u32 v0, s2, v30, s4                               // 000000002fb0: d7000200 0200091e
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s5, v31, s2                  // 000000002fbc: d5207c01 000a3e05
	global_load_u8 v0, v[0:1], off                             // 000000002fc4: ee04007c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002fd0: bfc00000
	v_cmp_eq_u32_e64 s2, 0xff, v0                              // 000000002fd4: d44a0002 020200ff 000000ff
	v_add_nc_u32_e32 v0, v51, v0                               // 000000002fe0: 4a000133
	s_or_b32 s2, vcc_lo, s2                                    // 000000002fe4: 8c02026a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002fe8: bf8700a1
	v_ldexp_f32 v0, v7, v0                                     // 000000002fec: d71c0000 02020107
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff4: bf88ff9e
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s2                   // 000000002ff8: d5010000 0009ff00 7fc00000
	v_cmp_lt_i64_e64 s2, s[46:47], s[18:19]                    // 000000003004: d4510002 0200242e
	s_delay_alu instid0(valu_dep_2)                            // 00000000300c: bf870002
	v_add_f32_e32 v10, v10, v0                                 // 000000003010: 0614010a
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000003014: 8b6a027e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003018: bf88ff9e
	s_cbranch_vccnz 64437                                      // 00000000301c: bfa4fbb5 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x3f4>
	v_mul_lo_u32 v4, s15, v14                                  // 000000003020: d72c0004 02021c0f
	v_mul_lo_u32 v5, s14, v15                                  // 000000003028: d72c0005 02021e0e
	v_mad_co_u64_u32 v[2:3], null, s14, v14, 0                 // 000000003030: d6fe7c02 02021c0e
	v_sub_co_u32 v0, vcc_lo, s12, v14                          // 000000003038: d7016a00 02021c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000003040: bf88ff9d
	v_sub_co_ci_u32_e64 v1, null, s13, v15, vcc_lo             // 000000003044: d5217c01 01aa1e0d
	v_cmp_gt_i64_e32 vcc_lo, s[14:15], v[11:12]                // 00000000304c: 7ca8160e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000003050: bf870194
	v_add3_u32 v3, v3, v5, v4                                  // 000000003054: d6550003 04120b03
	v_cmp_lt_i64_e64 s0, 0, v[0:1]                             // 00000000305c: d4510000 02020080
	s_delay_alu instid0(valu_dep_2)                            // 000000003064: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003068: 3e040481
	s_and_b32 s0, s0, vcc_lo                                   // 00000000306c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003070: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003074: be812000
	s_cbranch_execz 28                                         // 000000003078: bfa5001c <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x15ec>
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 00000000307c: 3e081681
	v_add_co_u32 v7, s0, s16, v2                               // 000000003080: d7000007 02020410
	v_bfe_u32 v6, v39, 16, 1                                   // 000000003088: d6100006 02052127
	s_wait_alu depctr_va_sdst(0)                               // 000000003090: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v3, s0                  // 000000003094: d5207c08 00020611
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000309c: bf870193
	v_add_co_u32 v4, s0, v7, v4                                // 0000000030a0: d7000004 02020907
	v_add3_u32 v6, v6, v39, 0x7fff                             // 0000000030a8: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 0000000030b4: 38124eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s0                   // 0000000030c0: d5207c05 00020b08
	v_cmp_u_f32_e64 s0, v39, v39                               // 0000000030c8: d4180000 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030d4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s0                           // 0000000030d8: d5010006 00021306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000030e0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030f0: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[0:1]                             // 0000000030f4: d4510000 02020081
	s_and_b32 s0, s0, vcc_lo                                   // 0000000030fc: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003100: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003104: be812000
	s_cbranch_execz 35                                         // 000000003108: bfa50023 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1698>
	v_add_co_u32 v6, s0, s16, v2                               // 00000000310c: d7000006 02020410
	s_wait_alu depctr_va_sdst(0)                               // 000000003114: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v3, s0                  // 000000003118: d5207c07 00020611
	s_lshl_b64 s[2:3], s[14:15], 1                             // 000000003120: 8482810e
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003124: 3e081681
	s_wait_alu depctr_sa_sdst(0)                               // 000000003128: bf88ff9e
	v_add_co_u32 v6, s0, v6, s2                                // 00000000312c: d7000006 02000506
	v_bfe_u32 v8, v47, 16, 1                                   // 000000003134: d6100008 0205212f
	s_wait_alu depctr_va_sdst(0)                               // 00000000313c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s3, v7, s0                   // 000000003140: d5207c07 00020e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003148: bf870193
	v_add_co_u32 v4, s0, v6, v4                                // 00000000314c: d7000004 02020906
	v_add3_u32 v8, v8, v47, 0x7fff                             // 000000003154: d6550008 03fe5f08 00007fff
	v_or_b32_e32 v9, 0x400000, v47                             // 000000003160: 38125eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003168: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s0                   // 00000000316c: d5207c05 00020b07
	v_cmp_u_f32_e64 s0, v47, v47                               // 000000003174: d4180000 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000317c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003180: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s0                           // 000000003184: d5010006 00021308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000318c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003198: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000319c: 8c7e017e
	v_cmp_lt_i64_e64 s0, 2, v[0:1]                             // 0000000031a0: d4510000 02020082
	s_and_b32 s0, s0, vcc_lo                                   // 0000000031a8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031ac: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000031b0: be812000
	s_cbranch_execz 34                                         // 0000000031b4: bfa50022 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1740>
	v_add_co_u32 v8, s0, s16, v2                               // 0000000031b8: d7000008 02020410
	v_bfe_u32 v4, v46, 16, 1                                   // 0000000031c0: d6100004 0205212e
	s_wait_alu depctr_va_sdst(0)                               // 0000000031c8: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s17, v3, s0                  // 0000000031cc: d5207c09 00020611
	s_lshl_b64 s[2:3], s[14:15], 2                             // 0000000031d4: 8482820e
	v_or_b32_e32 v6, 0x400000, v46                             // 0000000031d8: 380c5cff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031e0: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 0000000031e4: d7000008 02000508
	v_add3_u32 v7, v4, v46, 0x7fff                             // 0000000031ec: d6550007 03fe5d04 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 0000000031f8: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 0000000031fc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 000000003200: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v46, v46                               // 000000003208: d4180000 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003210: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003214: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 000000003218: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 000000003220: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 000000003228: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 00000000322c: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003234: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003240: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003244: 8c7e017e
	v_cmp_lt_i64_e64 s0, 3, v[0:1]                             // 000000003248: d4510000 02020083
	s_and_b32 s0, s0, vcc_lo                                   // 000000003250: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003254: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003258: be812000
	s_cbranch_execz 33                                         // 00000000325c: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x17e4>
	v_add_co_u32 v4, s0, s16, v2                               // 000000003260: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 000000003268: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 00000000326c: d5207c05 00020611
	v_bfe_u32 v6, v45, 16, 1                                   // 000000003274: d6100006 0205212d
	v_or_b32_e32 v8, 0x400000, v45                             // 00000000327c: 38105aff 00400000
	v_cmp_u_f32_e64 s0, v45, v45                               // 000000003284: d4180000 02025b2d
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000328c: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 6, v[4:5]              // 000000003290: d6fe7c04 04110c0e
	v_add3_u32 v9, v6, v45, 0x7fff                             // 000000003298: d6550009 03fe5b06 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 0000000032a4: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 0000000032a8: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 0000000032ac: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 6, v[5:6]              // 0000000032b4: d6fe7c05 04150c0f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 0000000032bc: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000032c0: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 0000000032c4: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000032cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 0000000032d0: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 0000000032d8: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032e8: 8c7e017e
	v_cmp_lt_i64_e64 s0, 4, v[0:1]                             // 0000000032ec: d4510000 02020084
	s_and_b32 s0, s0, vcc_lo                                   // 0000000032f4: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032fc: be812000
	s_cbranch_execz 34                                         // 000000003300: bfa50022 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x188c>
	v_add_co_u32 v8, s0, s16, v2                               // 000000003304: d7000008 02020410
	v_bfe_u32 v4, v42, 16, 1                                   // 00000000330c: d6100004 0205212a
	s_wait_alu depctr_va_sdst(0)                               // 000000003314: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s17, v3, s0                  // 000000003318: d5207c09 00020611
	s_lshl_b64 s[2:3], s[14:15], 3                             // 000000003320: 8482830e
	v_or_b32_e32 v6, 0x400000, v42                             // 000000003324: 380c54ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000332c: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 000000003330: d7000008 02000508
	v_add3_u32 v7, v4, v42, 0x7fff                             // 000000003338: d6550007 03fe5504 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003344: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 000000003348: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 00000000334c: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v42, v42                               // 000000003354: d4180000 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 00000000335c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003360: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 000000003364: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 00000000336c: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 000000003374: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 000000003378: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003380: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000338c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003390: 8c7e017e
	v_cmp_lt_i64_e64 s0, 5, v[0:1]                             // 000000003394: d4510000 02020085
	s_and_b32 s0, s0, vcc_lo                                   // 00000000339c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000033a4: be812000
	s_cbranch_execz 33                                         // 0000000033a8: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1930>
	v_add_co_u32 v4, s0, s16, v2                               // 0000000033ac: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 0000000033b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 0000000033b8: d5207c05 00020611
	v_bfe_u32 v6, v40, 16, 1                                   // 0000000033c0: d6100006 02052128
	v_or_b32_e32 v8, 0x400000, v40                             // 0000000033c8: 381050ff 00400000
	v_cmp_u_f32_e64 s0, v40, v40                               // 0000000033d0: d4180000 02025128
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000033d8: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 10, v[4:5]             // 0000000033dc: d6fe7c04 0411140e
	v_add3_u32 v9, v6, v40, 0x7fff                             // 0000000033e4: d6550009 03fe5106 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 0000000033f4: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 0000000033f8: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 10, v[5:6]             // 000000003400: d6fe7c05 0415140f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003408: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000340c: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 000000003410: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003418: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 00000000341c: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000003424: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003430: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003434: 8c7e017e
	v_cmp_lt_i64_e64 s0, 6, v[0:1]                             // 000000003438: d4510000 02020086
	s_and_b32 s0, s0, vcc_lo                                   // 000000003440: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003444: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003448: be812000
	s_cbranch_execz 33                                         // 00000000344c: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x19d4>
	v_add_co_u32 v4, s0, s16, v2                               // 000000003450: d7000004 02020410
	s_wait_alu depctr_va_sdst(0)                               // 000000003458: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v3, s0                  // 00000000345c: d5207c05 00020611
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003464: d6100006 02052126
	v_or_b32_e32 v8, 0x400000, v38                             // 00000000346c: 38104cff 00400000
	v_cmp_u_f32_e64 s0, v38, v38                               // 000000003474: d4180000 02024d26
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000347c: bf870214
	v_mad_co_u64_u32 v[4:5], null, s14, 12, v[4:5]             // 000000003480: d6fe7c04 0411180e
	v_add3_u32 v9, v6, v38, 0x7fff                             // 000000003488: d6550009 03fe4d06 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003494: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003498: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 00000000349c: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s15, 12, v[5:6]             // 0000000034a4: d6fe7c05 0415180f
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 0000000034ac: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000034b0: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 0000000034b4: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000034bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 0000000034c0: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 0000000034c8: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000034d8: 8c7e017e
	v_cmp_lt_i64_e64 s0, 7, v[0:1]                             // 0000000034dc: d4510000 02020087
	s_mov_b32 s1, 0                                            // 0000000034e4: be810080
	s_and_b32 s2, s0, vcc_lo                                   // 0000000034e8: 8b026a00
	s_mov_b32 s0, 0                                            // 0000000034ec: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000034f4: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f8: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 0000000034fc: 8d02037e
	v_add_co_u32 v0, vcc_lo, s16, v2                           // 000000003500: d7006a00 02020410
	s_wait_alu depctr_va_vcc(0)                                // 000000003508: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v3, vcc_lo              // 00000000350c: d5207c01 01aa0611
	s_mov_b32 s0, exec_lo                                      // 000000003514: be80007e
	v_mad_co_u64_u32 v[0:1], null, s14, 14, v[0:1]             // 000000003518: d6fe7c00 04011c0e
	s_delay_alu instid0(valu_dep_1)                            // 000000003520: bf870001
	v_mad_co_u64_u32 v[1:2], null, s15, 14, v[1:2]             // 000000003524: d6fe7c01 04051c0f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000352c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003530: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 000000003534: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000003538: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000353c: bf88ff9e
	s_cbranch_vccz 841                                         // 000000003540: bfa30349 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2768>
	s_and_b32 s0, s11, exec_lo                                 // 000000003544: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 000000003548: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000354c: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000003550: bf078100
	s_cbranch_scc1 19                                          // 000000003554: bfa20013 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1aa4>
	v_lshl_or_b32 v14, v37, 3, s22                             // 000000003558: d656000e 00590725
	v_mov_b32_e32 v15, s23                                     // 000000003560: 7e1e0217
	v_mov_b32_e32 v21, s23                                     // 000000003564: 7e2a0217
	v_mov_b32_e32 v17, s23                                     // 000000003568: 7e220217
	v_mov_b32_e32 v19, s23                                     // 00000000356c: 7e260217
	v_or_b32_e32 v20, 1, v14                                   // 000000003570: 38281c81
	v_or_b32_e32 v16, 2, v14                                   // 000000003574: 38201c82
	v_or_b32_e32 v18, 3, v14                                   // 000000003578: 38241c83
	v_or_b32_e32 v6, 4, v14                                    // 00000000357c: 380c1c84
	v_mov_b32_e32 v7, s23                                      // 000000003580: 7e0e0217
	v_or_b32_e32 v8, 5, v14                                    // 000000003584: 38101c85
	v_mov_b32_e32 v9, s23                                      // 000000003588: 7e120217
	v_or_b32_e32 v4, 6, v14                                    // 00000000358c: 38081c86
	v_mov_b32_e32 v5, s23                                      // 000000003590: 7e0a0217
	v_or_b32_e32 v2, 7, v14                                    // 000000003594: 38041c87
	v_mov_b32_e32 v3, s23                                      // 000000003598: 7e060217
	s_mov_b32 s0, 0                                            // 00000000359c: be800080
	s_branch 1                                                 // 0000000035a0: bfa00001 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1aa8>
	s_mov_b32 s0, -1                                           // 0000000035a4: be8000c1
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v41, 0             // 0000000035a8: ca100080 0a280080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035b0: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 0000000035b4: 8b007e00
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0             // 0000000035b8: ca100080 2a2a0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0             // 0000000035c0: ca100080 2d2e0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v40, 0             // 0000000035c8: ca100080 2f280080
	s_cselect_b32 s0, 1, 0                                     // 0000000035d0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035d4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000035d8: bf078100
	s_cbranch_scc1 554                                         // 0000000035dc: bfa2022a <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2388>
	v_dual_mov_b32 v15, s23 :: v_dual_lshlrev_b32 v2, 3, v37   // 0000000035e0: ca220017 0f024a83
	v_add_co_u32 v0, vcc_lo, s30, v11                          // 0000000035e8: d7006a00 0202161e
	s_wait_alu depctr_va_vcc(0)                                // 0000000035f0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s31, v12, vcc_lo             // 0000000035f4: d5207c01 01aa181f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 0000000035fc: bf8701c3
	v_or_b32_e32 v14, s22, v2                                  // 000000003600: 381c0416
	v_add_co_u32 v3, vcc_lo, s28, v13                          // 000000003604: d7006a03 02021a1c
	s_wait_alu depctr_va_vcc(0)                                // 00000000360c: bf88ff9d
	v_add_co_ci_u32_e64 v4, null, s29, v36, vcc_lo             // 000000003610: d5207c04 01aa481d
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[14:15]                // 000000003618: 7ca81c0c
	v_or_b32_e32 v20, 1, v14                                   // 00000000361c: 38281c81
	v_mov_b32_e32 v21, s23                                     // 000000003620: 7e2a0217
	v_add_co_u32 v13, s0, v3, v2                               // 000000003624: d700000d 02020503
	v_mad_co_u64_u32 v[0:1], null, s14, v2, v[0:1]             // 00000000362c: d6fe7c00 0402040e
	s_wait_alu depctr_va_vcc(0)                                // 000000003634: bf88ff9d
	v_dual_mov_b32 v40, 0 :: v_dual_cndmask_b32 v3, 0, v14     // 000000003638: ca120080 28021c80
	s_wait_alu depctr_va_sdst(0)                               // 000000003640: bf88f19f
	v_add_co_ci_u32_e64 v44, null, 0, v4, s0                   // 000000003644: d5207c2c 00020880
	v_cndmask_b32_e64 v4, 0, s23, vcc_lo                       // 00000000364c: d5010004 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[20:21]                // 000000003654: 7ca8280c
	v_or_b32_e32 v16, 2, v14                                   // 000000003658: 38201c82
	v_or_b32_e32 v18, 3, v14                                   // 00000000365c: 38241c83
	v_mov_b32_e32 v17, s23                                     // 000000003660: 7e220217
	s_lshr_b64 s[2:3], s[26:27], 5                             // 000000003664: 8582851a
	s_lshr_b32 s1, s27, 5                                      // 000000003668: 8501851b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000366c: bf88ff9e
	v_mul_lo_u32 v4, s2, v4                                    // 000000003670: d72c0004 02020802
	v_mul_lo_u32 v5, s1, v3                                    // 000000003678: d72c0005 02020601
	v_mad_co_u64_u32 v[22:23], null, s2, v3, s[24:25]          // 000000003680: d6fe7c16 00620602
	v_mad_co_u64_u32 v[1:2], null, s15, v2, v[1:2]             // 000000003688: d6fe7c01 0406040f
	v_cmp_gt_i64_e64 s0, s[14:15], v[11:12]                    // 000000003690: d4540000 0202160e
	s_wait_alu depctr_va_vcc(0)                                // 000000003698: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v20, vcc_lo                       // 00000000369c: 02042880
	v_cndmask_b32_e64 v3, 0, s23, vcc_lo                       // 0000000036a0: d5010003 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[16:17]                // 0000000036a8: 7ca8200c
	v_or_b32_e32 v6, 4, v14                                    // 0000000036ac: 380c1c84
	v_mov_b32_e32 v7, s23                                      // 0000000036b0: 7e0e0217
	v_or_b32_e32 v8, 5, v14                                    // 0000000036b4: 38101c85
	v_mov_b32_e32 v19, s23                                     // 0000000036b8: 7e260217
	s_wait_alu depctr_va_sdst(0)                               // 0000000036bc: bf88f19f
	v_cndmask_b32_e64 v10, 0, v12, s0                          // 0000000036c0: d501000a 00021880
	v_cndmask_b32_e64 v38, 0, v11, s0                          // 0000000036c8: d5010026 00021680
	v_add3_u32 v23, v5, v23, v4                                // 0000000036d0: d6550017 04122f05
	v_mul_lo_u32 v41, s2, v3                                   // 0000000036d8: d72c0029 02020602
	s_wait_alu depctr_va_vcc(0)                                // 0000000036e0: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v16, vcc_lo                       // 0000000036e4: 02062080
	v_cndmask_b32_e64 v4, 0, s23, vcc_lo                       // 0000000036e8: d5010004 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[18:19]                // 0000000036f0: 7ca8240c
	v_cmp_gt_i64_e64 s0, s[12:13], v[6:7]                      // 0000000036f4: d4540000 02020c0c
	v_mul_lo_u32 v42, s1, v2                                   // 0000000036fc: d72c002a 02020401
	v_mad_co_u64_u32 v[24:25], null, s2, v2, s[24:25]          // 000000003704: d6fe7c18 00620402
	v_mov_b32_e32 v9, s23                                      // 00000000370c: 7e120217
	v_mul_lo_u32 v43, s2, v4                                   // 000000003710: d72c002b 02020802
	v_mul_lo_u32 v45, s1, v3                                   // 000000003718: d72c002d 02020601
	v_mad_co_u64_u32 v[26:27], null, s2, v3, s[24:25]          // 000000003720: d6fe7c1a 00620602
	s_wait_alu depctr_va_vcc(0)                                // 000000003728: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v18, vcc_lo                       // 00000000372c: 02042480
	v_cndmask_b32_e64 v3, 0, s23, vcc_lo                       // 000000003730: d5010003 01a82e80
	s_wait_alu depctr_va_sdst(0)                               // 000000003738: bf88f19f
	v_cndmask_b32_e64 v4, 0, s23, s0                           // 00000000373c: d5010004 00002e80
	v_add3_u32 v25, v42, v25, v41                              // 000000003744: d6550019 04a6332a
	v_mov_b32_e32 v42, 0                                       // 00000000374c: 7e540280
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[8:9]                  // 000000003750: 7ca8100c
	v_mul_lo_u32 v46, s2, v3                                   // 000000003754: d72c002e 02020602
	v_mul_lo_u32 v47, s1, v2                                   // 00000000375c: d72c002f 02020401
	v_mad_co_u64_u32 v[28:29], null, s2, v2, s[24:25]          // 000000003764: d6fe7c1c 00620402
	v_mul_lo_u32 v48, s2, v4                                   // 00000000376c: d72c0030 02020802
	v_or_b32_e32 v4, 6, v14                                    // 000000003774: 38081c86
	v_mov_b32_e32 v5, s23                                      // 000000003778: 7e0a0217
	v_or_b32_e32 v2, 7, v14                                    // 00000000377c: 38041c87
	v_mov_b32_e32 v3, s23                                      // 000000003780: 7e060217
	v_cndmask_b32_e64 v30, 0, v6, s0                           // 000000003784: d501001e 00020c80
	s_wait_alu depctr_va_vcc(0)                                // 00000000378c: bf88ff9d
	v_cndmask_b32_e32 v32, 0, v8, vcc_lo                       // 000000003790: 02401080
	v_cndmask_b32_e64 v33, 0, s23, vcc_lo                      // 000000003794: d5010021 01a82e80
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[4:5]                  // 00000000379c: 7ca8080c
	v_cmp_gt_i64_e64 s0, s[12:13], v[2:3]                      // 0000000037a0: d4540000 0202040c
	v_mul_lo_u32 v49, s1, v30                                  // 0000000037a8: d72c0031 02023c01
	v_mad_co_u64_u32 v[30:31], null, s2, v30, s[24:25]         // 0000000037b0: d6fe7c1e 00623c02
	v_mul_lo_u32 v50, s2, v33                                  // 0000000037b8: d72c0032 02024202
	v_mul_lo_u32 v51, s1, v32                                  // 0000000037c0: d72c0033 02024001
	s_wait_alu depctr_va_vcc(0)                                // 0000000037c8: bf88ff9d
	v_cndmask_b32_e32 v34, 0, v4, vcc_lo                       // 0000000037cc: 02440880
	v_cndmask_b32_e64 v35, 0, s23, vcc_lo                      // 0000000037d0: d5010023 01a82e80
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d8: bf88f19f
	v_cndmask_b32_e64 v36, 0, v2, s0                           // 0000000037dc: d5010024 00020480
	v_cndmask_b32_e64 v37, 0, s23, s0                          // 0000000037e4: d5010025 00002e80
	v_mad_co_u64_u32 v[32:33], null, s2, v32, s[24:25]         // 0000000037ec: d6fe7c20 00624002
	v_mul_lo_u32 v53, s1, v34                                  // 0000000037f4: d72c0035 02024401
	v_mul_lo_u32 v52, s2, v35                                  // 0000000037fc: d72c0034 02024602
	v_mad_co_u64_u32 v[34:35], null, s2, v34, s[24:25]         // 000000003804: d6fe7c22 00624402
	v_mul_lo_u32 v54, s2, v37                                  // 00000000380c: d72c0036 02024a02
	v_mul_lo_u32 v55, s1, v36                                  // 000000003814: d72c0037 02024801
	v_mad_co_u64_u32 v[36:37], null, s2, v36, s[24:25]         // 00000000381c: d6fe7c24 00624802
	v_add_co_u32 v38, vcc_lo, s20, v38                         // 000000003824: d7006a26 02024c14
	s_wait_alu depctr_va_vcc(0)                                // 00000000382c: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v10, vcc_lo            // 000000003830: d5207c27 01aa1415
	v_add3_u32 v27, v45, v27, v43                              // 000000003838: d655001b 04ae372d
	v_add3_u32 v29, v47, v29, v46                              // 000000003840: d655001d 04ba3b2f
	v_add3_u32 v31, v49, v31, v48                              // 000000003848: d655001f 04c23f31
	v_add3_u32 v33, v51, v33, v50                              // 000000003850: d6550021 04ca4333
	v_add3_u32 v35, v53, v35, v52                              // 000000003858: d6550023 04d24735
	v_add3_u32 v37, v55, v37, v54                              // 000000003860: d6550025 04da4b37
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0             // 000000003868: ca100080 2f2e0080
	v_mov_b32_e32 v45, 0                                       // 000000003870: 7e5a0280
	v_mov_b32_e32 v43, 0                                       // 000000003874: 7e560280
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v10, 0             // 000000003878: ca100080 290a0080
	s_lshl_b64 s[10:11], s[14:15], 4                           // 000000003880: 848a840e
	s_mov_b64 s[12:13], 0                                      // 000000003884: be8c0180
	s_wait_alu depctr_sa_sdst(0)                               // 000000003888: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v13, s12                         // 00000000388c: d7006a30 0200190d
	s_lshr_b64 s[0:1], s[12:13], 5                             // 000000003894: 8580850c
	s_wait_alu depctr_va_vcc(0)                                // 000000003898: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s13, v44, vcc_lo            // 00000000389c: d5207c31 01aa580d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038a4: bf88ff9e
	v_add_co_u32 v52, vcc_lo, v22, s0                          // 0000000038a8: d7006a34 02000116
	s_wait_alu depctr_va_vcc(0)                                // 0000000038b0: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s1, v23, vcc_lo             // 0000000038b4: d5207c35 01aa2e01
	v_add_co_u32 v56, vcc_lo, v24, s0                          // 0000000038bc: d7006a38 02000118
	s_wait_alu depctr_va_vcc(0)                                // 0000000038c4: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s1, v25, vcc_lo             // 0000000038c8: d5207c39 01aa3201
	v_add_co_u32 v58, vcc_lo, v26, s0                          // 0000000038d0: d7006a3a 0200011a
	v_mad_co_u64_u32 v[50:51], null, s12, s14, v[0:1]          // 0000000038d8: d6fe7c32 04001c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000038e0: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s1, v27, vcc_lo             // 0000000038e4: d5207c3b 01aa3601
	v_add_co_u32 v60, vcc_lo, v28, s0                          // 0000000038ec: d7006a3c 0200011c
	s_wait_alu depctr_va_vcc(0)                                // 0000000038f4: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s1, v29, vcc_lo             // 0000000038f8: d5207c3d 01aa3a01
	v_add_co_u32 v62, vcc_lo, v30, s0                          // 000000003900: d7006a3e 0200011e
	s_wait_alu depctr_va_vcc(0)                                // 000000003908: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s1, v31, vcc_lo             // 00000000390c: d5207c3f 01aa3e01
	v_add_co_u32 v64, vcc_lo, v32, s0                          // 000000003914: d7006a40 02000120
	s_mul_i32 s2, s13, s14                                     // 00000000391c: 96020e0d
	s_mul_i32 s3, s12, s15                                     // 000000003920: 96030f0c
	s_wait_alu depctr_va_vcc(0)                                // 000000003924: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s1, v33, vcc_lo             // 000000003928: d5207c41 01aa4201
	v_add_co_u32 v66, vcc_lo, v34, s0                          // 000000003930: d7006a42 02000122
	s_wait_alu depctr_sa_sdst(0)                               // 000000003938: bf88ff9e
	v_add3_u32 v51, s3, s2, v51                                // 00000000393c: d6550033 04cc0403
	s_wait_alu depctr_va_vcc(0)                                // 000000003944: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s1, v35, vcc_lo             // 000000003948: d5207c43 01aa4601
	v_add_co_u32 v68, vcc_lo, v36, s0                          // 000000003950: d7006a44 02000124
	s_mul_i32 s5, s0, s15                                      // 000000003958: 96050f00
	v_mad_co_u64_u32 v[54:55], null, s0, s14, v[38:39]         // 00000000395c: d6fe7c36 04981c00
	s_wait_alu depctr_va_vcc(0)                                // 000000003964: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s1, v37, vcc_lo             // 000000003968: d5207c45 01aa4a01
	s_clause 0x1                                               // 000000003970: bf850001
	global_load_b64 v[70:71], v[48:49], off                    // 000000003974: ee05407c 00000046 00000030
	global_load_b64 v[72:73], v[48:49], off offset:16          // 000000003980: ee05407c 00000048 00001030
	s_clause 0x6                                               // 00000000398c: bf850006
	global_load_u8 v74, v[52:53], off                          // 000000003990: ee04007c 0000004a 00000034
	global_load_u8 v75, v[56:57], off                          // 00000000399c: ee04007c 0000004b 00000038
	global_load_u8 v58, v[58:59], off                          // 0000000039a8: ee04007c 0000003a 0000003a
	global_load_u8 v59, v[60:61], off                          // 0000000039b4: ee04007c 0000003b 0000003c
	global_load_u8 v60, v[62:63], off                          // 0000000039c0: ee04007c 0000003c 0000003e
	global_load_u8 v61, v[64:65], off                          // 0000000039cc: ee04007c 0000003d 00000040
	global_load_u8 v62, v[66:67], off                          // 0000000039d8: ee04007c 0000003e 00000042
	v_add_co_u32 v48, vcc_lo, v50, s14                         // 0000000039e4: d7006a30 02001d32
	v_add_co_u32 v52, s0, v50, s10                             // 0000000039ec: d7000034 02001532
	s_wait_alu depctr_va_vcc(0)                                // 0000000039f4: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s15, v51, vcc_lo            // 0000000039f8: d5207c31 01aa660f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a00: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s11, v51, s0                // 000000003a04: d5207c35 0002660b
	global_load_u8 v63, v[50:51], off                          // 000000003a0c: ee04007c 0000003f 00000032
	s_lshr_b32 s4, s13, 5                                      // 000000003a18: 8504850d
	v_add_co_u32 v56, vcc_lo, v48, s14                         // 000000003a1c: d7006a38 02001d30
	s_clause 0x1                                               // 000000003a24: bf850001
	global_load_u8 v66, v[52:53], off                          // 000000003a28: ee04007c 00000042 00000034
	global_load_u8 v65, v[48:49], off                          // 000000003a34: ee04007c 00000041 00000030
	v_add_co_u32 v50, s0, v52, s14                             // 000000003a40: d7000032 02001d34
	s_wait_alu depctr_va_sdst(0)                               // 000000003a48: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s15, v53, s0                // 000000003a4c: d5207c33 00026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a54: bf88ff9e
	s_mul_i32 s4, s4, s14                                      // 000000003a58: 96040e04
	s_wait_alu depctr_va_vcc(0)                                // 000000003a5c: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s15, v49, vcc_lo            // 000000003a60: d5207c39 01aa620f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a68: bf88ff9e
	v_add3_u32 v55, s5, s4, v55                                // 000000003a6c: d6550037 04dc0805
	v_add_co_u32 v48, vcc_lo, v50, s14                         // 000000003a74: d7006a30 02001d32
	s_wait_alu depctr_va_vcc(0)                                // 000000003a7c: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s15, v51, vcc_lo            // 000000003a80: d5207c31 01aa660f
	global_load_u8 v64, v[54:55], off                          // 000000003a88: ee04007c 00000040 00000036
	v_add_co_u32 v52, s0, v56, s14                             // 000000003a94: d7000034 02001d38
	v_add_co_u32 v54, vcc_lo, v48, s14                         // 000000003a9c: d7006a36 02001d30
	s_wait_alu depctr_va_sdst(0)                               // 000000003aa4: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s15, v57, s0                // 000000003aa8: d5207c35 0002720f
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab0: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s15, v49, vcc_lo            // 000000003ab4: d5207c37 01aa620f
	s_clause 0x4                                               // 000000003abc: bf850004
	global_load_u8 v56, v[56:57], off                          // 000000003ac0: ee04007c 00000038 00000038
	global_load_u8 v67, v[52:53], off                          // 000000003acc: ee04007c 00000043 00000034
	global_load_u8 v76, v[48:49], off                          // 000000003ad8: ee04007c 0000004c 00000030
	global_load_u8 v78, v[54:55], off                          // 000000003ae4: ee04007c 0000004e 00000036
	global_load_u8 v57, v[50:51], off                          // 000000003af0: ee04007c 00000039 00000032
	v_add_co_u32 v50, s0, v52, s14                             // 000000003afc: d7000032 02001d34
	s_wait_alu depctr_va_sdst(0)                               // 000000003b04: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s15, v53, s0                // 000000003b08: d5207c33 00026a0f
	v_add_co_u32 v52, vcc_lo, v54, s14                         // 000000003b10: d7006a34 02001d36
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b18: bf8701a3
	v_add_co_u32 v48, s0, v50, s14                             // 000000003b1c: d7000030 02001d32
	s_wait_alu depctr_va_sdst(0)                               // 000000003b24: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s15, v51, s0                // 000000003b28: d5207c31 0002660f
	s_wait_alu depctr_va_vcc(0)                                // 000000003b30: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s15, v55, vcc_lo            // 000000003b34: d5207c35 01aa6e0f
	global_load_u8 v77, v[50:51], off                          // 000000003b3c: ee04007c 0000004d 00000032
	s_add_nc_u64 s[12:13], s[12:13], 32                        // 000000003b48: a98ca00c
	s_clause 0x1                                               // 000000003b4c: bf850001
	global_load_u8 v80, v[52:53], off                          // 000000003b50: ee04007c 00000050 00000034
	global_load_u8 v79, v[48:49], off                          // 000000003b5c: ee04007c 0000004f 00000030
	v_add_co_u32 v50, vcc_lo, v52, s14                         // 000000003b68: d7006a32 02001d34
	s_wait_alu depctr_va_vcc(0)                                // 000000003b70: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s15, v53, vcc_lo            // 000000003b74: d5207c33 01aa6a0f
	v_add_co_u32 v54, s0, v48, s14                             // 000000003b7c: d7000036 02001d30
	s_wait_alu depctr_va_sdst(0)                               // 000000003b84: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s15, v49, s0                // 000000003b88: d5207c37 0002620f
	v_add_co_u32 v48, vcc_lo, v50, s14                         // 000000003b90: d7006a30 02001d32
	s_wait_alu depctr_va_vcc(0)                                // 000000003b98: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s15, v51, vcc_lo            // 000000003b9c: d5207c31 01aa660f
	v_add_co_u32 v52, s0, v54, s14                             // 000000003ba4: d7000034 02001d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003bac: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s15, v55, s0                // 000000003bb0: d5207c35 00026e0f
	s_clause 0x1                                               // 000000003bb8: bf850001
	global_load_u8 v54, v[54:55], off                          // 000000003bbc: ee04007c 00000036 00000036
	global_load_u8 v55, v[50:51], off                          // 000000003bc8: ee04007c 00000037 00000032
	v_add_co_u32 v50, vcc_lo, v48, s14                         // 000000003bd4: d7006a32 02001d30
	s_wait_alu depctr_va_vcc(0)                                // 000000003bdc: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s15, v49, vcc_lo            // 000000003be0: d5207c33 01aa620f
	s_clause 0x1                                               // 000000003be8: bf850001
	global_load_u8 v52, v[52:53], off                          // 000000003bec: ee04007c 00000034 00000034
	global_load_u8 v48, v[48:49], off                          // 000000003bf8: ee04007c 00000030 00000030
	global_load_u8 v49, v[50:51], off                          // 000000003c04: ee04007c 00000031 00000032
	global_load_u8 v50, v[68:69], off                          // 000000003c10: ee04007c 00000032 00000044
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c1c: bf88ff9e
	v_cmp_lt_i64_e64 s0, s[12:13], s[18:19]                    // 000000003c20: d4510000 0200240c
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000003c28: 8b6a007e
	s_wait_loadcnt 0x18                                        // 000000003c2c: bfc00018
	v_cmp_eq_u32_e64 s0, 0xff, v74                             // 000000003c30: d44a0000 020294ff 000000ff
	s_wait_loadcnt 0x17                                        // 000000003c3c: bfc00017
	v_cmp_eq_u32_e64 s1, 0xff, v75                             // 000000003c40: d44a0001 020296ff 000000ff
	s_wait_loadcnt 0x16                                        // 000000003c4c: bfc00016
	v_cmp_eq_u32_e64 s2, 0xff, v58                             // 000000003c50: d44a0002 020274ff 000000ff
	s_wait_loadcnt 0x15                                        // 000000003c5c: bfc00015
	v_cmp_eq_u32_e64 s3, 0xff, v59                             // 000000003c60: d44a0003 020276ff 000000ff
	s_wait_loadcnt 0x14                                        // 000000003c6c: bfc00014
	v_cmp_eq_u32_e64 s4, 0xff, v60                             // 000000003c70: d44a0004 020278ff 000000ff
	s_wait_loadcnt 0x13                                        // 000000003c7c: bfc00013
	v_cmp_eq_u32_e64 s5, 0xff, v61                             // 000000003c80: d44a0005 02027aff 000000ff
	s_wait_loadcnt 0x12                                        // 000000003c8c: bfc00012
	v_cmp_eq_u32_e64 s6, 0xff, v62                             // 000000003c90: d44a0006 02027cff 000000ff
	s_wait_loadcnt 0xf                                         // 000000003c9c: bfc0000f
	v_perm_b32 v53, v63, v65, 0xc0c0004                        // 000000003ca0: d6440035 03fe833f 0c0c0004
	s_wait_loadcnt 0xe                                         // 000000003cac: bfc0000e
	v_add_nc_u32_e32 v51, 0xffffff02, v64                      // 000000003cb0: 4a6680ff ffffff02
	v_cmp_eq_u32_e64 s8, 0xff, v64                             // 000000003cb8: d44a0008 020280ff 000000ff
	s_delay_alu instid0(valu_dep_2)                            // 000000003cc4: bf870002
	v_add_nc_u32_e32 v63, v51, v74                             // 000000003cc8: 4a7e9533
	v_add_nc_u32_e32 v64, v51, v75                             // 000000003ccc: 4a809733
	v_add_nc_u32_e32 v65, v51, v58                             // 000000003cd0: 4a827533
	v_add_nc_u32_e32 v68, v51, v59                             // 000000003cd4: 4a887733
	v_add_nc_u32_e32 v60, v51, v60                             // 000000003cd8: 4a787933
	v_add_nc_u32_e32 v61, v51, v61                             // 000000003cdc: 4a7a7b33
	v_add_nc_u32_e32 v62, v51, v62                             // 000000003ce0: 4a7c7d33
	s_or_b32 s3, s8, s3                                        // 000000003ce4: 8c030308
	s_or_b32 s5, s8, s5                                        // 000000003ce8: 8c050508
	s_or_b32 s6, s8, s6                                        // 000000003cec: 8c060608
	s_or_b32 s1, s8, s1                                        // 000000003cf0: 8c010108
	s_or_b32 s2, s8, s2                                        // 000000003cf4: 8c020208
	s_or_b32 s4, s8, s4                                        // 000000003cf8: 8c040408
	s_or_b32 s0, s0, s8                                        // 000000003cfc: 8c000800
	s_wait_loadcnt 0x4                                         // 000000003d00: bfc00004
	v_perm_b32 v59, v80, v55, 0xc0c0004                        // 000000003d04: d644003b 03fe6f50 0c0c0004
	s_wait_loadcnt 0x0                                         // 000000003d10: bfc00000
	v_cmp_eq_u32_e64 s7, 0xff, v50                             // 000000003d14: d44a0007 020264ff 000000ff
	v_add_nc_u32_e32 v69, v51, v50                             // 000000003d20: 4a8a6533
	v_perm_b32 v50, v66, v57, 0xc0c0004                        // 000000003d24: d6440032 03fe7342 0c0c0004
	v_perm_b32 v51, v56, v67, 0xc0c0004                        // 000000003d30: d6440033 03fe8738 0c0c0004
	v_perm_b32 v57, v76, v78, 0xc0c0004                        // 000000003d3c: d6440039 03fe9d4c 0c0c0004
	v_perm_b32 v66, v48, v49, 0xc0c0004                        // 000000003d48: d6440042 03fe6330 0c0c0004
	s_or_b32 s7, s8, s7                                        // 000000003d54: 8c070708
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_2)// 000000003d58: bf870153
	v_lshl_or_b32 v56, v51, 16, v53                            // 000000003d5c: d6560038 04d52133
	v_perm_b32 v51, v77, v79, 0xc0c0004                        // 000000003d64: d6440033 03fe9f4d 0c0c0004
	v_lshl_or_b32 v58, v57, 16, v50                            // 000000003d70: d656003a 04c92139
	v_perm_b32 v50, v54, v52, 0xc0c0004                        // 000000003d78: d6440032 03fe6936 0c0c0004
	v_lshl_or_b32 v59, v66, 16, v59                            // 000000003d84: d656003b 04ed2142
	v_lshl_or_b32 v57, v50, 16, v51                            // 000000003d8c: d6560039 04cd2132
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d94: bf870091
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[70:71], v[56:57], 0// 000000003d98: cc464030 1a027146
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[72:73], v[58:59], v[48:55]// 000000003da0: cc464030 1cc27548
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_2)// 000000003da8: bf870111
	v_ldexp_f32 v55, v55, v69                                  // 000000003dac: d71c0037 02028b37
	v_ldexp_f32 v51, v51, v68                                  // 000000003db4: d71c0033 02028933
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_4)// 000000003dbc: bf870213
	v_ldexp_f32 v54, v54, v62                                  // 000000003dc0: d71c0036 02027d36
	v_ldexp_f32 v48, v48, v63                                  // 000000003dc8: d71c0030 02027f30
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dd0: bf88ff9e
	v_cndmask_b32_e64 v55, v55, 0x7fc00000, s7                 // 000000003dd4: d5010037 001dff37 7fc00000
	v_cndmask_b32_e64 v51, v51, 0x7fc00000, s3                 // 000000003de0: d5010033 000dff33 7fc00000
	v_cndmask_b32_e64 v54, v54, 0x7fc00000, s6                 // 000000003dec: d5010036 0019ff36 7fc00000
	v_cndmask_b32_e64 v48, v48, 0x7fc00000, s0                 // 000000003df8: d5010030 0001ff30 7fc00000
	s_delay_alu instid0(valu_dep_4)                            // 000000003e04: bf870004
	v_add_f32_e32 v10, v10, v55                                // 000000003e08: 06146f0a
	v_ldexp_f32 v53, v53, v61                                  // 000000003e0c: d71c0035 02027b35
	v_add_f32_e32 v45, v45, v51                                // 000000003e14: 065a672d
	v_ldexp_f32 v52, v52, v60                                  // 000000003e18: d71c0034 02027934
	v_add_f32_e32 v41, v41, v54                                // 000000003e20: 06526d29
	v_ldexp_f32 v49, v49, v64                                  // 000000003e24: d71c0031 02028131
	v_cndmask_b32_e64 v53, v53, 0x7fc00000, s5                 // 000000003e2c: d5010035 0015ff35 7fc00000
	v_add_f32_e32 v40, v40, v48                                // 000000003e38: 06506128
	v_cndmask_b32_e64 v52, v52, 0x7fc00000, s4                 // 000000003e3c: d5010034 0011ff34 7fc00000
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003e48: bf870214
	v_cndmask_b32_e64 v49, v49, 0x7fc00000, s1                 // 000000003e4c: d5010031 0005ff31 7fc00000
	v_add_f32_e32 v42, v42, v53                                // 000000003e58: 06546b2a
	v_ldexp_f32 v50, v50, v65                                  // 000000003e5c: d71c0032 02028332
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003e64: bf870214
	v_add_f32_e32 v43, v43, v52                                // 000000003e68: 0656692b
	v_add_f32_e32 v47, v47, v49                                // 000000003e6c: 065e632f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003e70: bf870093
	v_cndmask_b32_e64 v50, v50, 0x7fc00000, s2                 // 000000003e74: d5010032 0009ff32 7fc00000
	v_add_f32_e32 v46, v46, v50                                // 000000003e80: 065c652e
	s_cbranch_vccnz 65152                                      // 000000003e84: bfa4fe80 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1d88>
	v_mul_lo_u32 v13, s15, v14                                 // 000000003e88: d72c000d 02021c0f
	v_mul_lo_u32 v15, s14, v15                                 // 000000003e90: d72c000f 02021e0e
	v_mad_co_u64_u32 v[0:1], null, s14, v14, 0                 // 000000003e98: d6fe7c00 02021c0e
	v_mul_lo_u32 v22, s15, v20                                 // 000000003ea0: d72c0016 0202280f
	v_mul_lo_u32 v23, s14, v21                                 // 000000003ea8: d72c0017 02022a0e
	v_or_b32_e32 v25, 0x400000, v40                            // 000000003eb0: 383250ff 00400000
	v_bfe_u32 v24, v47, 16, 1                                  // 000000003eb8: d6100018 0205212f
	v_mul_lo_u32 v17, s14, v17                                 // 000000003ec0: d72c0011 0202220e
	v_mul_lo_u32 v19, s14, v19                                 // 000000003ec8: d72c0013 0202260e
	s_mov_b32 s0, -1                                           // 000000003ed0: be8000c1
	v_add3_u32 v1, v1, v15, v13                                // 000000003ed4: d6550001 04361f01
	v_mad_co_u64_u32 v[13:14], null, s14, v20, 0               // 000000003edc: d6fe7c0d 0202280e
	v_bfe_u32 v15, v40, 16, 1                                  // 000000003ee4: d610000f 02052128
	v_lshlrev_b64_e32 v[20:21], 1, v[11:12]                    // 000000003eec: 3e281681
	v_add3_u32 v24, v24, v47, 0x7fff                           // 000000003ef0: d6550018 03fe5f18 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003efc: 3e000081
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000003f00: bf870234
	v_add3_u32 v15, v15, v40, 0x7fff                           // 000000003f04: d655000f 03fe510f 00007fff
	v_add3_u32 v14, v14, v23, v22                              // 000000003f10: d655000e 045a2f0e
	v_or_b32_e32 v23, 0x400000, v47                            // 000000003f18: 382e5eff 00400000
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000003f20: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003f28: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 000000003f2c: d5207c01 01aa0211
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000003f34: 7c305128
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000003f38: 3e1a1a81
	s_wait_alu depctr_va_vcc(0)                                // 000000003f3c: bf88ff9d
	v_cndmask_b32_e32 v22, v15, v25, vcc_lo                    // 000000003f40: 022c330f
	v_mul_lo_u32 v25, s15, v16                                 // 000000003f44: d72c0019 0202200f
	v_mad_co_u64_u32 v[15:16], null, s14, v16, 0               // 000000003f4c: d6fe7c0f 0202200e
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 000000003f54: d7006a00 02022900
	s_wait_alu depctr_va_vcc(0)                                // 000000003f5c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 000000003f60: d5207c01 01aa2b01
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000003f68: 7c305f2f
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000003f6c: ee09407c 0b000000 00000000
	v_add3_u32 v16, v16, v17, v25                              // 000000003f78: d6550010 04662310
	s_wait_alu depctr_va_vcc(0)                                // 000000003f80: bf88ff9d
	v_cndmask_b32_e32 v22, v24, v23, vcc_lo                    // 000000003f84: 022c2f18
	v_add_co_u32 v0, vcc_lo, s16, v13                          // 000000003f88: d7006a00 02021a10
	v_bfe_u32 v13, v46, 16, 1                                  // 000000003f90: d610000d 0205212e
	s_wait_alu depctr_va_vcc(0)                                // 000000003f98: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v14, vcc_lo             // 000000003f9c: d5207c01 01aa1c11
	v_mul_lo_u32 v24, s15, v18                                 // 000000003fa4: d72c0018 0202240f
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 000000003fac: d7006a00 02022900
	v_add3_u32 v17, v13, v46, 0x7fff                           // 000000003fb4: d6550011 03fe5d0d 00007fff
	v_lshlrev_b64_e32 v[13:14], 1, v[15:16]                    // 000000003fc0: 3e1a1e81
	v_mad_co_u64_u32 v[15:16], null, s14, v18, 0               // 000000003fc4: d6fe7c0f 0202240e
	s_wait_alu depctr_va_vcc(0)                                // 000000003fcc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 000000003fd0: d5207c01 01aa2b01
	v_or_b32_e32 v23, 0x400000, v46                            // 000000003fd8: 382e5cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000003fe0: 7c305d2e
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000003fe4: ee09407c 0b000000 00000000
	v_add3_u32 v16, v16, v19, v24                              // 000000003ff0: d6550010 04622710
	s_wait_alu depctr_va_vcc(0)                                // 000000003ff8: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v23, vcc_lo                    // 000000003ffc: 02222f11
	v_add_co_u32 v0, vcc_lo, s16, v13                          // 000000004000: d7006a00 02021a10
	v_bfe_u32 v13, v45, 16, 1                                  // 000000004008: d610000d 0205212d
	s_wait_alu depctr_va_vcc(0)                                // 000000004010: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v14, vcc_lo             // 000000004014: d5207c01 01aa1c11
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000401c: bf870193
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 000000004020: d7006a00 02022900
	v_add3_u32 v18, v13, v45, 0x7fff                           // 000000004028: d6550012 03fe5b0d 00007fff
	v_lshlrev_b64_e32 v[13:14], 1, v[15:16]                    // 000000004034: 3e1a1e81
	v_mul_lo_u32 v15, s15, v6                                  // 000000004038: d72c000f 02020c0f
	v_mul_lo_u32 v16, s14, v7                                  // 000000004040: d72c0010 02020e0e
	v_mad_co_u64_u32 v[6:7], null, s14, v6, 0                  // 000000004048: d6fe7c06 02020c0e
	s_wait_alu depctr_va_vcc(0)                                // 000000004050: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 000000004054: d5207c01 01aa2b01
	v_or_b32_e32 v19, 0x400000, v45                            // 00000000405c: 38265aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000004064: 7c305b2d
	global_store_d16_hi_b16 v[0:1], v17, off                   // 000000004068: ee09407c 08800000 00000000
	v_add3_u32 v7, v7, v16, v15                                // 000000004074: d6550007 043e2107
	s_wait_alu depctr_va_vcc(0)                                // 00000000407c: bf88ff9d
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo                    // 000000004080: 02222712
	v_add_co_u32 v0, vcc_lo, s16, v13                          // 000000004084: d7006a00 02021a10
	s_wait_alu depctr_va_vcc(0)                                // 00000000408c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v14, vcc_lo             // 000000004090: d5207c01 01aa1c11
	v_bfe_u32 v13, v43, 16, 1                                  // 000000004098: d610000d 0205212b
	s_delay_alu instid0(valu_dep_3)                            // 0000000040a0: bf870003
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 0000000040a4: d7006a00 02022900
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 0000000040ac: 3e0c0c81
	v_mul_lo_u32 v15, s15, v8                                  // 0000000040b0: d72c000f 0202100f
	v_mul_lo_u32 v16, s14, v9                                  // 0000000040b8: d72c0010 0202120e
	v_mad_co_u64_u32 v[8:9], null, s14, v8, 0                  // 0000000040c0: d6fe7c08 0202100e
	s_wait_alu depctr_va_vcc(0)                                // 0000000040c8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 0000000040cc: d5207c01 01aa2b01
	v_add3_u32 v13, v13, v43, 0x7fff                           // 0000000040d4: d655000d 03fe570d 00007fff
	v_or_b32_e32 v14, 0x400000, v43                            // 0000000040e0: 381c56ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 0000000040e8: 7c30572b
	global_store_d16_hi_b16 v[0:1], v17, off                   // 0000000040ec: ee09407c 08800000 00000000
	v_mul_lo_u32 v18, s14, v3                                  // 0000000040f8: d72c0012 0202060e
	v_add3_u32 v9, v9, v16, v15                                // 000000004100: d6550009 043e2109
	v_or_b32_e32 v15, 0x400000, v42                            // 000000004108: 381e54ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004110: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v14, vcc_lo                    // 000000004114: 021a1d0d
	v_add_co_u32 v0, vcc_lo, s16, v6                           // 000000004118: d7006a00 02020c10
	s_wait_alu depctr_va_vcc(0)                                // 000000004120: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v7, vcc_lo              // 000000004124: d5207c01 01aa0e11
	v_bfe_u32 v14, v42, 16, 1                                  // 00000000412c: d610000e 0205212a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004134: bf8701a3
	v_add_co_u32 v6, vcc_lo, v0, v20                           // 000000004138: d7006a06 02022900
	s_wait_alu depctr_va_vcc(0)                                // 000000004140: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v1, v21, vcc_lo              // 000000004144: d5207c07 01aa2b01
	v_lshlrev_b64_e32 v[0:1], 1, v[8:9]                        // 00000000414c: 3e001081
	v_mul_lo_u32 v8, s15, v4                                   // 000000004150: d72c0008 0202080f
	v_mul_lo_u32 v9, s14, v5                                   // 000000004158: d72c0009 02020a0e
	v_mad_co_u64_u32 v[4:5], null, s14, v4, 0                  // 000000004160: d6fe7c04 0202080e
	v_add3_u32 v14, v14, v42, 0x7fff                           // 000000004168: d655000e 03fe550e 00007fff
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000004174: 7c30552a
	s_wait_alu depctr_va_vcc(0)                                // 000000004178: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 00000000417c: bf870002
	v_cndmask_b32_e32 v14, v14, v15, vcc_lo                    // 000000004180: 021c1f0e
	v_bfe_u32 v15, v41, 16, 1                                  // 000000004184: d610000f 02052129
	v_add_co_u32 v16, vcc_lo, s16, v0                          // 00000000418c: d7006a10 02020010
	v_add3_u32 v5, v5, v9, v8                                  // 000000004194: d6550005 04221305
	s_wait_alu depctr_va_vcc(0)                                // 00000000419c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s17, v1, vcc_lo             // 0000000041a0: d5207c11 01aa0211
	v_add3_u32 v8, v15, v41, 0x7fff                            // 0000000041a8: d6550008 03fe530f 00007fff
	v_mul_lo_u32 v15, s15, v2                                  // 0000000041b4: d72c000f 0202040f
	v_mad_co_u64_u32 v[0:1], null, s14, v2, 0                  // 0000000041bc: d6fe7c00 0202040e
	v_lshlrev_b64_e32 v[2:3], 1, v[4:5]                        // 0000000041c4: 3e040881
	v_add_co_u32 v4, vcc_lo, v16, v20                          // 0000000041c8: d7006a04 02022910
	v_or_b32_e32 v9, 0x400000, v41                             // 0000000041d0: 381252ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000041d8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v17, v21, vcc_lo             // 0000000041dc: d5207c05 01aa2b11
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 0000000041e4: 7c305329
	v_add3_u32 v1, v1, v18, v15                                // 0000000041e8: d6550001 043e2501
	s_clause 0x1                                               // 0000000041f0: bf850001
	global_store_d16_hi_b16 v[6:7], v13, off                   // 0000000041f4: ee09407c 06800000 00000006
	global_store_d16_hi_b16 v[4:5], v14, off                   // 000000004200: ee09407c 07000000 00000004
	s_wait_alu depctr_va_vcc(0)                                // 00000000420c: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo                       // 000000004210: 02101308
	v_add_co_u32 v2, vcc_lo, s16, v2                           // 000000004214: d7006a02 02020410
	s_wait_alu depctr_va_vcc(0)                                // 00000000421c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo              // 000000004220: d5207c03 01aa0611
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004228: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000422c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v20                           // 000000004230: d7006a02 02022902
	s_wait_alu depctr_va_vcc(0)                                // 000000004238: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v21, vcc_lo              // 00000000423c: d5207c03 01aa2b03
	s_delay_alu instid0(valu_dep_3)                            // 000000004244: bf870003
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000004248: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004250: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 000000004254: d5207c01 01aa0211
	global_store_d16_hi_b16 v[2:3], v8, off                    // 00000000425c: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004268: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000426c: be812000
	s_cbranch_execnz 1                                         // 000000004270: bfa60001 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2778>
	s_endpgm                                                   // 000000004274: bfb00000
	v_bfe_u32 v2, v10, 16, 1                                   // 000000004278: d6100002 0205210a
	v_or_b32_e32 v4, 0x400000, v10                             // 000000004280: 380814ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v10, v10                           // 000000004288: 7c30150a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 00000000428c: bf870133
	v_add3_u32 v5, v2, v10, 0x7fff                             // 000000004290: d6550005 03fe1502 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[11:12]                      // 00000000429c: 3e041681
	s_wait_alu depctr_va_vcc(0)                                // 0000000042a0: bf88ff9d
	v_cndmask_b32_e32 v4, v5, v4, vcc_lo                       // 0000000042a4: 02080905
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042a8: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000042ac: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000042b4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000042b8: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 0000000042c0: ee09407c 02000000 00000000
	s_endpgm                                                   // 0000000042cc: bfb00000
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
	s_code_end                                                 // 000000004300: bf9f0000
	s_code_end                                                 // 000000004304: bf9f0000
	s_code_end                                                 // 000000004308: bf9f0000
	s_code_end                                                 // 00000000430c: bf9f0000
	s_code_end                                                 // 000000004310: bf9f0000
	s_code_end                                                 // 000000004314: bf9f0000
	s_code_end                                                 // 000000004318: bf9f0000
	s_code_end                                                 // 00000000431c: bf9f0000
	s_code_end                                                 // 000000004320: bf9f0000
	s_code_end                                                 // 000000004324: bf9f0000
	s_code_end                                                 // 000000004328: bf9f0000
	s_code_end                                                 // 00000000432c: bf9f0000
	s_code_end                                                 // 000000004330: bf9f0000
	s_code_end                                                 // 000000004334: bf9f0000
	s_code_end                                                 // 000000004338: bf9f0000
	s_code_end                                                 // 00000000433c: bf9f0000
	s_code_end                                                 // 000000004340: bf9f0000
	s_code_end                                                 // 000000004344: bf9f0000
	s_code_end                                                 // 000000004348: bf9f0000
	s_code_end                                                 // 00000000434c: bf9f0000
	s_code_end                                                 // 000000004350: bf9f0000
	s_code_end                                                 // 000000004354: bf9f0000
	s_code_end                                                 // 000000004358: bf9f0000
	s_code_end                                                 // 00000000435c: bf9f0000
	s_code_end                                                 // 000000004360: bf9f0000
	s_code_end                                                 // 000000004364: bf9f0000
	s_code_end                                                 // 000000004368: bf9f0000
	s_code_end                                                 // 00000000436c: bf9f0000
	s_code_end                                                 // 000000004370: bf9f0000
	s_code_end                                                 // 000000004374: bf9f0000
	s_code_end                                                 // 000000004378: bf9f0000
	s_code_end                                                 // 00000000437c: bf9f0000
	s_code_end                                                 // 000000004380: bf9f0000
	s_code_end                                                 // 000000004384: bf9f0000
	s_code_end                                                 // 000000004388: bf9f0000
	s_code_end                                                 // 00000000438c: bf9f0000
	s_code_end                                                 // 000000004390: bf9f0000
	s_code_end                                                 // 000000004394: bf9f0000
	s_code_end                                                 // 000000004398: bf9f0000
	s_code_end                                                 // 00000000439c: bf9f0000
	s_code_end                                                 // 0000000043a0: bf9f0000
	s_code_end                                                 // 0000000043a4: bf9f0000
	s_code_end                                                 // 0000000043a8: bf9f0000
	s_code_end                                                 // 0000000043ac: bf9f0000
	s_code_end                                                 // 0000000043b0: bf9f0000
	s_code_end                                                 // 0000000043b4: bf9f0000
	s_code_end                                                 // 0000000043b8: bf9f0000
	s_code_end                                                 // 0000000043bc: bf9f0000
	s_code_end                                                 // 0000000043c0: bf9f0000
	s_code_end                                                 // 0000000043c4: bf9f0000
	s_code_end                                                 // 0000000043c8: bf9f0000
	s_code_end                                                 // 0000000043cc: bf9f0000
	s_code_end                                                 // 0000000043d0: bf9f0000
	s_code_end                                                 // 0000000043d4: bf9f0000
	s_code_end                                                 // 0000000043d8: bf9f0000
	s_code_end                                                 // 0000000043dc: bf9f0000
	s_code_end                                                 // 0000000043e0: bf9f0000
	s_code_end                                                 // 0000000043e4: bf9f0000
	s_code_end                                                 // 0000000043e8: bf9f0000
	s_code_end                                                 // 0000000043ec: bf9f0000
	s_code_end                                                 // 0000000043f0: bf9f0000
	s_code_end                                                 // 0000000043f4: bf9f0000
	s_code_end                                                 // 0000000043f8: bf9f0000
	s_code_end                                                 // 0000000043fc: bf9f0000
	s_code_end                                                 // 000000004400: bf9f0000
	s_code_end                                                 // 000000004404: bf9f0000
	s_code_end                                                 // 000000004408: bf9f0000
	s_code_end                                                 // 00000000440c: bf9f0000
	s_code_end                                                 // 000000004410: bf9f0000
	s_code_end                                                 // 000000004414: bf9f0000
	s_code_end                                                 // 000000004418: bf9f0000
	s_code_end                                                 // 00000000441c: bf9f0000
	s_code_end                                                 // 000000004420: bf9f0000
	s_code_end                                                 // 000000004424: bf9f0000
	s_code_end                                                 // 000000004428: bf9f0000
	s_code_end                                                 // 00000000442c: bf9f0000
	s_code_end                                                 // 000000004430: bf9f0000
	s_code_end                                                 // 000000004434: bf9f0000
	s_code_end                                                 // 000000004438: bf9f0000
	s_code_end                                                 // 00000000443c: bf9f0000
	s_code_end                                                 // 000000004440: bf9f0000
	s_code_end                                                 // 000000004444: bf9f0000
	s_code_end                                                 // 000000004448: bf9f0000
	s_code_end                                                 // 00000000444c: bf9f0000
	s_code_end                                                 // 000000004450: bf9f0000
	s_code_end                                                 // 000000004454: bf9f0000
	s_code_end                                                 // 000000004458: bf9f0000
	s_code_end                                                 // 00000000445c: bf9f0000
	s_code_end                                                 // 000000004460: bf9f0000
	s_code_end                                                 // 000000004464: bf9f0000
	s_code_end                                                 // 000000004468: bf9f0000
	s_code_end                                                 // 00000000446c: bf9f0000
	s_code_end                                                 // 000000004470: bf9f0000
	s_code_end                                                 // 000000004474: bf9f0000
	s_code_end                                                 // 000000004478: bf9f0000
	s_code_end                                                 // 00000000447c: bf9f0000
