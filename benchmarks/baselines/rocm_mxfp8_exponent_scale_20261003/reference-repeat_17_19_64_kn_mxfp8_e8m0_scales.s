
/tmp/tmpjr3ec6dl.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_5f09ec8682bcb143>:
	s_clause 0x2                                               // 000000001b00: bf850002
	s_load_b64 s[26:27], s[0:1], 0xd8                          // 000000001b04: f4002680 f80000d8
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b0c: f4004400 f80000c8
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000001b14: f4002500 f80000a8
	v_and_b32_e32 v2, 15, v0                                   // 000000001b1c: 3604008f
	s_mov_b32 s4, ttmp7                                        // 000000001b20: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b24: 86059f73
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[28:29], s[0:1], 0x8                           // 000000001b2c: f4002700 f8000008
	s_load_b64 s[30:31], s[0:1], 0x30                          // 000000001b34: f4002780 f8000030
	s_load_b64 s[24:25], s[0:1], 0x58                          // 000000001b3c: f4002600 f8000058
	s_load_b64 s[12:13], s[0:1], 0x80                          // 000000001b44: f4002300 f8000080
	s_lshl_b64 s[14:15], s[4:5], 4                             // 000000001b4c: 848e8404
	s_mov_b32 s2, ttmp9                                        // 000000001b50: be820075
	v_or_b32_e32 v1, s14, v2                                   // 000000001b54: 3802040e
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b58: 86039f75
	s_add_nc_u64 s[0:1], s[14:15], 16                          // 000000001b5c: a980900e
	s_lshl_b64 s[2:3], s[2:3], 4                               // 000000001b60: 84828402
	v_bfe_u32 v39, v0, 4, 1                                    // 000000001b64: d6100027 02050900
	s_add_nc_u64 s[4:5], s[2:3], 16                            // 000000001b6c: a9849002
	v_or_b32_e32 v11, s2, v2                                   // 000000001b70: 38160402
	v_mov_b32_e32 v12, s3                                      // 000000001b74: 7e180203
	s_wait_kmcnt 0x0                                           // 000000001b78: bfc70000
	v_mul_lo_u32 v3, s27, v1                                   // 000000001b7c: d72c0003 0202021b
	v_mad_co_u64_u32 v[13:14], null, s26, v1, 0                // 000000001b84: d6fe7c0d 0202021a
	v_cmp_gt_i64_e64 s0, s[0:1], s[16:17]                      // 000000001b8c: d4540000 02002000
	v_cmp_gt_i64_e64 s1, s[4:5], s[18:19]                      // 000000001b94: d4540001 02002404
	s_mul_i32 s2, s26, s15                                     // 000000001b9c: 96020f1a
	v_cmp_lt_i64_e64 s11, s[26:27], 32                         // 000000001ba0: d451000b 0201401a
	s_and_b32 s22, s26, 0xffffffe0                             // 000000001ba8: 8b16ff1a ffffffe0
	s_mov_b32 s23, s27                                         // 000000001bb0: be97001b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	v_add3_u32 v38, v14, s2, v3                                // 000000001bb8: d6550026 040c050e
	s_or_b32 s0, s0, s1                                        // 000000001bc0: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc4: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bc8: 8b6a007e
	s_cbranch_vccz 10                                          // 000000001bcc: bfa3000a <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0xf8>
	s_and_b32 s0, s11, exec_lo                                 // 000000001bd0: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 000000001bd4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bd8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bdc: bf078100
	s_cbranch_scc1 8                                           // 000000001be0: bfa20008 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x104>
	v_lshl_or_b32 v14, v39, 3, s14                             // 000000001be4: d656000e 00390727
	v_mov_b32_e32 v15, s15                                     // 000000001bec: 7e1e020f
	s_mov_b32 s0, 0                                            // 000000001bf0: be800080
	s_branch 4                                                 // 000000001bf4: bfa00004 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x108>
	s_mov_b32 s0, 0                                            // 000000001bf8: be800080
	s_cbranch_execnz 1719                                      // 000000001bfc: bfa606b7 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1bdc>
	s_branch 2648                                              // 000000001c00: bfa00a58 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2a64>
	s_mov_b32 s0, -1                                           // 000000001c04: be8000c1
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v47, 0             // 000000001c08: ca100080 0a2e0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001c14: 8b007e00
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v49, 0             // 000000001c18: ca100080 28300080
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0             // 000000001c20: ca100080 2a280080
	v_mov_b32_e32 v44, 0                                       // 000000001c28: 7e580280
	v_mov_b32_e32 v48, 0                                       // 000000001c2c: 7e600280
	s_cselect_b32 s0, 1, 0                                     // 000000001c30: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c34: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c38: bf078100
	s_cbranch_scc1 1374                                        // 000000001c3c: bfa2055e <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x16b8>
	v_dual_mov_b32 v2, s15 :: v_dual_lshlrev_b32 v43, 3, v39   // 000000001c40: ca22000f 022a4e83
	v_mov_b32_e32 v15, s15                                     // 000000001c48: 7e1e020f
	s_lshr_b64 s[4:5], s[26:27], 5                             // 000000001c4c: 8584851a
	s_lshr_b32 s3, s27, 5                                      // 000000001c50: 8503851b
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_1)// 000000001c54: bf8700a2
	v_or_b32_e32 v14, s14, v43                                 // 000000001c58: 381c560e
	v_add_co_u32 v45, vcc_lo, v13, v43                         // 000000001c5c: d7006a2d 0202570d
	v_add_co_ci_u32_e64 v46, null, 0, v38, vcc_lo              // 000000001c64: d5207c2e 01aa4c80
	s_delay_alu instid0(valu_dep_3)                            // 000000001c6c: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[14:15]                // 000000001c70: 7ca81c10
	v_cmp_gt_i64_e64 s0, s[16:17], v[1:2]                      // 000000001c74: d4540000 02020210
	v_or_b32_e32 v0, 1, v14                                    // 000000001c7c: 38001c81
	v_mov_b32_e32 v1, s15                                      // 000000001c80: 7e02020f
	v_mov_b32_e32 v41, 0                                       // 000000001c84: 7e520280
	v_mad_co_u64_u32 v[8:9], null, s18, v43, v[11:12]          // 000000001c88: d6fe7c08 042e5612
	s_wait_alu depctr_va_vcc(0)                                // 000000001c90: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v14, vcc_lo                       // 000000001c94: 02041c80
	v_cndmask_b32_e64 v3, 0, s15, vcc_lo                       // 000000001c98: d5010003 01a81e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001ca0: 7ca80010
	v_or_b32_e32 v1, 2, v14                                    // 000000001ca4: 38021c82
	v_cmp_gt_i64_e64 s1, s[18:19], v[11:12]                    // 000000001ca8: d4540001 02021612
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cb0: bf88ff9e
	v_mul_lo_u32 v4, s3, v2                                    // 000000001cb4: d72c0004 02020403
	v_mad_co_u64_u32 v[16:17], null, s4, v2, s[24:25]          // 000000001cbc: d6fe7c10 00620404
	v_mov_b32_e32 v2, s15                                      // 000000001cc4: 7e04020f
	v_mul_lo_u32 v3, s4, v3                                    // 000000001cc8: d72c0003 02020604
	s_wait_alu depctr_va_vcc(0)                                // 000000001cd0: bf88ff9d
	v_cndmask_b32_e32 v5, 0, v0, vcc_lo                        // 000000001cd4: 020a0080
	v_cndmask_b32_e64 v0, 0, s15, vcc_lo                       // 000000001cd8: d5010000 01a81e80
	v_mad_co_u64_u32 v[9:10], null, s19, v43, v[9:10]          // 000000001ce0: d6fe7c09 04265613
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[1:2]                  // 000000001ce8: 7ca80210
	v_or_b32_e32 v2, 4, v14                                    // 000000001cec: 38041c84
	s_wait_alu depctr_va_sdst(0)                               // 000000001cf0: bf88f19f
	v_cndmask_b32_e64 v7, 0, v11, s1                           // 000000001cf4: d5010007 00061680
	v_mul_lo_u32 v10, s4, v0                                   // 000000001cfc: d72c000a 02020004
	v_add3_u32 v17, v4, v17, v3                                // 000000001d04: d6550011 040e2304
	v_or_b32_e32 v0, 3, v14                                    // 000000001d0c: 38001c83
	s_wait_alu depctr_va_vcc(0)                                // 000000001d10: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v1 :: v_dual_mov_b32 v1, s15     // 000000001d14: ca500280 0400000f
	v_cndmask_b32_e64 v20, 0, s15, vcc_lo                      // 000000001d1c: d5010014 01a81e80
	v_mov_b32_e32 v3, s15                                      // 000000001d24: 7e06020f
	v_mul_lo_u32 v34, s3, v5                                   // 000000001d28: d72c0022 02020a03
	v_mad_co_u64_u32 v[18:19], null, s4, v5, s[24:25]          // 000000001d30: d6fe7c12 00620a04
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001d38: 7ca80010
	v_mul_lo_u32 v5, s4, v20                                   // 000000001d3c: d72c0005 02022804
	v_mul_lo_u32 v35, s3, v4                                   // 000000001d44: d72c0023 02020803
	v_mad_co_u64_u32 v[20:21], null, s4, v4, s[24:25]          // 000000001d4c: d6fe7c14 00620804
	v_cndmask_b32_e64 v6, 0, v12, s1                           // 000000001d54: d5010006 00061880
	v_mov_b32_e32 v49, 0                                       // 000000001d5c: 7e620280
	s_wait_alu depctr_va_vcc(0)                                // 000000001d60: bf88ff9d
	v_cndmask_b32_e64 v4, 0, s15, vcc_lo                       // 000000001d64: d5010004 01a81e80
	v_add3_u32 v19, v34, v19, v10                              // 000000001d6c: d6550013 042a2722
	v_mov_b32_e32 v48, 0                                       // 000000001d74: 7e600280
	v_mov_b32_e32 v10, 0                                       // 000000001d78: 7e140280
	s_lshl_b64 s[34:35], s[18:19], 1                           // 000000001d7c: 84a28112
	v_mul_lo_u32 v36, s4, v4                                   // 000000001d80: d72c0024 02020804
	v_mov_b32_e32 v4, s15                                      // 000000001d88: 7e08020f
	v_cmp_gt_i64_e64 s2, s[16:17], v[2:3]                      // 000000001d8c: d4540002 02020410
	v_cndmask_b32_e32 v3, 0, v0, vcc_lo                        // 000000001d94: 02060080
	v_or_b32_e32 v0, 5, v14                                    // 000000001d98: 38001c85
	v_add3_u32 v21, v35, v21, v5                               // 000000001d9c: d6550015 04162b23
	s_mul_u64 s[36:37], s[18:19], 3                            // 000000001da4: aaa48312
	s_lshl_b64 s[38:39], s[18:19], 2                           // 000000001da8: 84a68212
	v_cndmask_b32_e64 v24, 0, v2, s2                           // 000000001dac: d5010018 000a0480
	v_cndmask_b32_e64 v2, 0, s15, s2                           // 000000001db4: d5010002 00081e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001dbc: 7ca80010
	v_mul_lo_u32 v37, s3, v3                                   // 000000001dc0: d72c0025 02020603
	v_mad_co_u64_u32 v[22:23], null, s4, v3, s[24:25]          // 000000001dc8: d6fe7c16 00620604
	v_or_b32_e32 v1, 6, v14                                    // 000000001dd0: 38021c86
	v_mul_lo_u32 v40, s4, v2                                   // 000000001dd4: d72c0028 02020404
	v_mov_b32_e32 v2, s15                                      // 000000001ddc: 7e04020f
	v_or_b32_e32 v3, 7, v14                                    // 000000001de0: 38061c87
	s_wait_alu depctr_va_vcc(0)                                // 000000001de4: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001de8: 02000080
	v_cndmask_b32_e64 v26, 0, s15, vcc_lo                      // 000000001dec: d501001a 01a81e80
	s_mul_u64 s[40:41], s[18:19], 5                            // 000000001df4: aaa88512
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[1:2]                  // 000000001df8: 7ca80210
	v_cmp_gt_i64_e64 s2, s[16:17], v[3:4]                      // 000000001dfc: d4540002 02020610
	v_mul_lo_u32 v2, s3, v24                                   // 000000001e04: d72c0002 02023003
	v_mad_co_u64_u32 v[24:25], null, s4, v24, s[24:25]         // 000000001e0c: d6fe7c18 00623004
	v_mul_lo_u32 v4, s4, v26                                   // 000000001e14: d72c0004 02023404
	v_mul_lo_u32 v42, s3, v0                                   // 000000001e1c: d72c002a 02020003
	s_wait_alu depctr_va_vcc(0)                                // 000000001e24: bf88ff9d
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001e28: 02020280
	v_cndmask_b32_e64 v28, 0, s15, vcc_lo                      // 000000001e2c: d501001c 01a81e80
	s_wait_alu depctr_va_sdst(0)                               // 000000001e34: bf88f19f
	v_cndmask_b32_e64 v3, 0, v3, s2                            // 000000001e38: d5010003 000a0680
	v_cndmask_b32_e64 v30, 0, s15, s2                          // 000000001e40: d501001e 00081e80
	v_mad_co_u64_u32 v[26:27], null, s4, v0, s[24:25]          // 000000001e48: d6fe7c1a 00620004
	v_mul_lo_u32 v44, s3, v1                                   // 000000001e50: d72c002c 02020203
	v_mul_lo_u32 v0, s4, v28                                   // 000000001e58: d72c0000 02023804
	v_mad_co_u64_u32 v[28:29], null, s4, v1, s[24:25]          // 000000001e60: d6fe7c1c 00620204
	v_mul_lo_u32 v1, s4, v30                                   // 000000001e68: d72c0001 02023c04
	v_mul_lo_u32 v47, s3, v3                                   // 000000001e70: d72c002f 02020603
	v_mad_co_u64_u32 v[30:31], null, s4, v3, s[24:25]          // 000000001e78: d6fe7c1e 00620604
	v_add_co_u32 v32, vcc_lo, s12, v7                          // 000000001e80: d7006a20 02020e0c
	s_lshl_b64 s[2:3], s[18:19], 4                             // 000000001e88: 84828412
	s_wait_alu depctr_va_vcc(0)                                // 000000001e8c: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s13, v6, vcc_lo             // 000000001e90: d5207c21 01aa0c0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e98: bf88ff9e
	v_add_co_u32 v34, vcc_lo, s2, v8                           // 000000001e9c: d7006a22 02021002
	v_add3_u32 v23, v37, v23, v36                              // 000000001ea4: d6550017 04922f25
	v_add3_u32 v25, v2, v25, v40                               // 000000001eac: d6550019 04a23302
	v_add3_u32 v27, v42, v27, v4                               // 000000001eb4: d655001b 0412372a
	v_add3_u32 v29, v44, v29, v0                               // 000000001ebc: d655001d 04023b2c
	v_add3_u32 v31, v47, v31, v1                               // 000000001ec4: d655001f 04063f2f
	s_wait_alu depctr_va_vcc(0)                                // 000000001ecc: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s3, v9, vcc_lo              // 000000001ed0: d5207c23 01aa1203
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v40, 0             // 000000001ed8: ca100080 2f280080
	v_mov_b32_e32 v44, 0                                       // 000000001ee0: 7e580280
	v_mov_b32_e32 v42, 0                                       // 000000001ee4: 7e540280
	s_mul_u64 s[42:43], s[18:19], 6                            // 000000001ee8: aaaa8612
	s_mul_u64 s[44:45], s[18:19], 7                            // 000000001eec: aaac8712
	s_mov_b64 s[46:47], 0                                      // 000000001ef0: beae0180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000001ef4: bf8701d9
	v_dual_mov_b32 v5, s47 :: v_dual_mov_b32 v2, s47           // 000000001ef8: ca10002f 0502002f
	v_or_b32_e32 v4, s46, v43                                  // 000000001f00: 3808562e
	v_add_co_u32 v36, vcc_lo, v45, s46                         // 000000001f04: d7006a24 02005d2d
	s_wait_alu depctr_va_vcc(0)                                // 000000001f0c: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s47, v46, vcc_lo            // 000000001f10: d5207c25 01aa5c2f
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[4:5]                  // 000000001f18: 7ca8081a
	s_mul_i32 s33, s46, s19                                    // 000000001f1c: 9621132e
	v_mov_b32_e32 v53, s47                                     // 000000001f20: 7e6a022f
	s_and_b32 s2, s0, vcc_lo                                   // 000000001f24: 8b026a00
	s_and_b32 vcc_lo, s1, vcc_lo                               // 000000001f28: 8b6a6a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f2c: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v36, s2                           // 000000001f30: d5010000 000a4880
	v_cndmask_b32_e64 v1, 0, v37, s2                           // 000000001f38: d5010001 000a4a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000001f40: bf870122
	v_add_co_u32 v0, s3, s28, v0                               // 000000001f44: d7000300 0202001c
	s_wait_alu depctr_va_sdst(0)                               // 000000001f4c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s29, v1, s3                  // 000000001f50: d5207c01 000e021d
	global_load_d16_u8 v0, v[0:1], off                         // 000000001f58: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v4                                     // 000000001f64: 38020881
	s_wait_loadcnt 0x0                                         // 000000001f68: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s2                            // 000000001f6c: d65d0000 000a0080
	v_add_co_u32 v3, s2, v36, 1                                // 000000001f74: d7000203 02010324
	s_wait_alu depctr_va_sdst(0)                               // 000000001f7c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v37, s2                   // 000000001f80: d5207c06 000a4a80
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
	v_add_co_u32 v3, s3, v36, 2                                // 000000001fec: d7000303 02010524
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v37, s3                   // 000000001ff8: d5207c06 000e4a80
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
	v_add_co_u32 v6, s4, v36, 3                                // 000000002060: d7000406 02010724
	s_wait_alu depctr_va_sdst(0)                               // 000000002068: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v37, s4                   // 00000000206c: d5207c07 00124a80
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
	v_add_co_u32 v6, s5, v36, 4                                // 0000000020d8: d7000506 02010924
	s_wait_alu depctr_va_sdst(0)                               // 0000000020e0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v37, s5                   // 0000000020e4: d5207c07 00164a80
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
	v_add_co_u32 v3, s6, v36, 5                                // 00000000214c: d7000603 02010b24
	s_wait_alu depctr_va_sdst(0)                               // 000000002154: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v37, s6                  // 000000002158: d5207c32 001a4a80
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
	v_add_co_u32 v3, s7, v36, 6                                // 0000000021c8: d7000703 02010d24
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d0: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v37, s7                  // 0000000021d4: d5207c32 001e4a80
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
	v_add_co_u32 v6, s8, v36, 7                                // 000000002234: d7000806 02010f24
	s_wait_alu depctr_va_sdst(0)                               // 00000000223c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v37, s8                   // 000000002240: d5207c07 00224a80
	v_cmp_gt_i64_e64 s8, s[26:27], v[4:5]                      // 000000002248: d4540008 0202081a
	v_or_b16 v36.l, v0.l, v0.h op_sel:[0,1,0]                  // 000000002250: d7631024 02020100
	v_or_b16 v36.h, v1.l, v1.h op_sel:[0,1,1]                  // 000000002258: d7635024 02020301
	v_or_b16 v37.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002260: d7631025 02020502
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
	v_mad_co_u64_u32 v[4:5], null, s46, s18, v[8:9]            // 0000000022b0: d6fe7c04 0420242e
	s_delay_alu instid0(valu_dep_1)                            // 0000000022b8: bf870001
	v_cndmask_b32_e32 v0, 0, v4, vcc_lo                        // 0000000022bc: 02000880
	s_wait_loadcnt 0x0                                         // 0000000022c0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s9                            // 0000000022c4: d65d5003 00260680
	s_mul_i32 s9, s47, s18                                     // 0000000022cc: 9609122f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	s_add_co_i32 s33, s33, s9                                  // 0000000022d4: 81210921
	v_add_co_u32 v0, s9, s30, v0                               // 0000000022d8: d7000900 0202001e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022e0: bf88ff9e
	v_add_nc_u32_e32 v7, s33, v5                               // 0000000022e4: 4a0e0a21
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000022e8: d7385003 02020688
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000022f0: bf870112
	v_cndmask_b32_e32 v1, 0, v7, vcc_lo                        // 0000000022f4: 02020e80
	v_or_b16 v37.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000022f8: d7635025 02020703
	s_wait_alu depctr_va_sdst(0)                               // 000000002300: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002304: bf870002
	v_add_co_ci_u32_e64 v1, null, s31, v1, s9                  // 000000002308: d5207c01 0026021f
	global_load_d16_u8 v0, v[0:1], off                         // 000000002310: ee07807c 00000000 00000000
	s_wait_loadcnt 0x0                                         // 00000000231c: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, vcc_lo                        // 000000002320: d65d0000 01aa0080
	v_add_co_u32 v1, vcc_lo, v4, s18                           // 000000002328: d7006a01 02002504
	s_wait_alu depctr_va_vcc(0)                                // 000000002330: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v7, vcc_lo              // 000000002334: d5207c02 01aa0e13
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
	v_or_b32_e32 v52, s2, v43                                  // 0000000025d0: 38685602
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000025d4: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000025e0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 0000000025e4: d65d5003 01aa0680
	v_add_co_u32 v56, vcc_lo, v45, s2                          // 0000000025ec: d7006a38 0200052d
	s_wait_alu depctr_va_vcc(0)                                // 0000000025f4: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s47, v46, vcc_lo            // 0000000025f8: d5207c39 01aa5c2f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002600: bf870123
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002604: d7385003 02020688
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[52:53]                // 00000000260c: 7ca8681a
	v_or_b16 v51.h, v3.l, v3.h op_sel:[0,1,1]                  // 000000002610: d7635033 02020703
	s_and_b32 s2, s0, vcc_lo                                   // 000000002618: 8b026a00
	s_and_b32 vcc_lo, s1, vcc_lo                               // 00000000261c: 8b6a6a01
	s_delay_alu instid0(valu_dep_1)                            // 000000002620: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[36:37], v[50:51], 0  // 000000002624: cc464000 1a026524
	s_wait_alu depctr_sa_sdst(0)                               // 00000000262c: bf88ff9e
	v_cndmask_b32_e64 v36, 0, v56, s2                          // 000000002630: d5010024 000a7080
	v_cndmask_b32_e64 v37, 0, v57, s2                          // 000000002638: d5010025 000a7280
	v_or_b32_e32 v50, 1, v52                                   // 000000002640: 38646881
	v_mov_b32_e32 v51, s47                                     // 000000002644: 7e66022f
	s_delay_alu instid0(valu_dep_4)                            // 000000002648: bf870004
	v_add_co_u32 v36, s3, s28, v36                             // 00000000264c: d7000324 0202481c
	s_wait_alu depctr_va_sdst(0)                               // 000000002654: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s29, v37, s3                // 000000002658: d5207c25 000e4a1d
	global_load_d16_u8 v36, v[36:37], off                      // 000000002660: ee07807c 00000024 00000024
	s_wait_loadcnt 0x0                                         // 00000000266c: bfc00000
	v_cndmask_b16 v36.l, 0, v36.l, s2                          // 000000002670: d65d0024 000a4880
	v_add_co_u32 v37, s2, v56, 1                               // 000000002678: d7000225 02010338
	s_wait_alu depctr_va_sdst(0)                               // 000000002680: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s2                  // 000000002684: d5207c36 000a7280
	v_cmp_gt_i64_e64 s2, s[26:27], v[50:51]                    // 00000000268c: d4540002 0202641a
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002694: d7620024 020248ff 000000ff
	s_and_b32 s3, s0, s2                                       // 0000000026a0: 8b030200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026a4: bf88ff9e
	v_cndmask_b32_e64 v37, 0, v37, s3                          // 0000000026a8: d5010025 000e4a80
	v_cndmask_b32_e64 v51, 0, v54, s3                          // 0000000026b0: d5010033 000e6c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000026b8: bf870122
	v_add_co_u32 v50, s4, s28, v37                             // 0000000026bc: d7000432 02024a1c
	s_wait_alu depctr_va_sdst(0)                               // 0000000026c4: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s4                // 0000000026c8: d5207c33 0012661d
	global_load_d16_hi_u8 v36, v[50:51], off                   // 0000000026d0: ee08407c 00000024 00000032
	v_or_b32_e32 v50, 2, v52                                   // 0000000026dc: 38646882
	v_mov_b32_e32 v51, s47                                     // 0000000026e0: 7e66022f
	s_wait_loadcnt 0x0                                         // 0000000026e4: bfc00000
	v_cndmask_b16 v36.h, 0, v36.h, s3                          // 0000000026e8: d65d5024 000e4880
	v_add_co_u32 v37, s3, v56, 2                               // 0000000026f0: d7000325 02010538
	s_wait_alu depctr_va_sdst(0)                               // 0000000026f8: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v57, s3                  // 0000000026fc: d5207c36 000e7280
	v_cmp_gt_i64_e64 s3, s[26:27], v[50:51]                    // 000000002704: d4540003 0202641a
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 00000000270c: d7385024 02024888
	s_and_b32 s4, s0, s3                                       // 000000002714: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002718: bf88ff9e
	v_cndmask_b32_e64 v37, 0, v37, s4                          // 00000000271c: d5010025 00124a80
	v_cndmask_b32_e64 v51, 0, v54, s4                          // 000000002724: d5010033 00126c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000272c: bf870122
	v_add_co_u32 v50, s5, s28, v37                             // 000000002730: d7000532 02024a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002738: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s5                // 00000000273c: d5207c33 0016661d
	global_load_d16_u8 v37, v[50:51], off                      // 000000002744: ee07807c 00000025 00000032
	v_or_b32_e32 v50, 3, v52                                   // 000000002750: 38646883
	v_mov_b32_e32 v51, s47                                     // 000000002754: 7e66022f
	s_wait_loadcnt 0x0                                         // 000000002758: bfc00000
	v_cndmask_b16 v37.l, 0, v37.l, s4                          // 00000000275c: d65d0025 00124a80
	v_add_co_u32 v54, s4, v56, 3                               // 000000002764: d7000436 02010738
	s_wait_alu depctr_va_sdst(0)                               // 00000000276c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s4                  // 000000002770: d5207c37 00127280
	v_cmp_gt_i64_e64 s4, s[26:27], v[50:51]                    // 000000002778: d4540004 0202641a
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002780: d7620025 02024aff 000000ff
	s_and_b32 s5, s0, s4                                       // 00000000278c: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002790: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s5                          // 000000002794: d5010032 00166c80
	v_cndmask_b32_e64 v51, 0, v55, s5                          // 00000000279c: d5010033 00166e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000027a4: bf870122
	v_add_co_u32 v50, s6, s28, v50                             // 0000000027a8: d7000632 0202641c
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b0: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s29, v51, s6                // 0000000027b4: d5207c33 001a661d
	global_load_d16_hi_u8 v37, v[50:51], off                   // 0000000027bc: ee08407c 00000025 00000032
	v_or_b32_e32 v50, 4, v52                                   // 0000000027c8: 38646884
	v_mov_b32_e32 v51, s47                                     // 0000000027cc: 7e66022f
	s_wait_loadcnt 0x0                                         // 0000000027d0: bfc00000
	v_cndmask_b16 v37.h, 0, v37.h, s5                          // 0000000027d4: d65d5025 00164a80
	v_add_co_u32 v54, s5, v56, 4                               // 0000000027dc: d7000536 02010938
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e4: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v57, s5                  // 0000000027e8: d5207c37 00167280
	v_cmp_gt_i64_e64 s5, s[26:27], v[50:51]                    // 0000000027f0: d4540005 0202641a
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 0000000027f8: d7385025 02024a88
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
	v_mad_co_u64_u32 v[54:55], null, s46, s18, v[34:35]        // 000000002978: d6fe7c36 0488242e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002980: bf8701a3
	v_add_co_u32 v52, s10, s28, v52                            // 000000002984: d7000a34 0202681c
	s_wait_alu depctr_va_sdst(0)                               // 00000000298c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s29, v53, s10               // 000000002990: d5207c35 002a6a1d
	s_delay_alu instid0(valu_dep_3)                            // 000000002998: bf870003
	v_add_nc_u32_e32 v57, s33, v55                             // 00000000299c: 4a726e21
	global_load_d16_hi_u8 v51, v[52:53], off                   // 0000000029a0: ee08407c 00000033 00000034
	v_or_b16 v52.l, v36.l, v36.h op_sel:[0,1,0]                // 0000000029ac: d7631034 02024924
	v_cndmask_b32_e32 v36, 0, v54, vcc_lo                      // 0000000029b4: 02486c80
	v_or_b16 v52.h, v37.l, v37.h op_sel:[0,1,1]                // 0000000029b8: d7635034 02024b25
	v_cndmask_b32_e32 v37, 0, v57, vcc_lo                      // 0000000029c0: 024a7280
	v_or_b16 v53.l, v50.l, v50.h op_sel:[0,1,0]                // 0000000029c4: d7631035 02026532
	s_wait_loadcnt 0x0                                         // 0000000029cc: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, s9                          // 0000000029d0: d65d5033 00266680
	v_add_co_u32 v36, s9, s30, v36                             // 0000000029d8: d7000924 0202481e
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e0: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s31, v37, s9                // 0000000029e4: d5207c25 00264a1f
	s_delay_alu instid0(valu_dep_3)                            // 0000000029ec: bf870003
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 0000000029f0: d7385033 02026688
	global_load_d16_u8 v36, v[36:37], off                      // 0000000029f8: ee07807c 00000024 00000024
	v_or_b16 v53.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002a04: d7635035 02026733
	s_wait_loadcnt 0x0                                         // 000000002a0c: bfc00000
	v_cndmask_b16 v36.l, 0, v36.l, vcc_lo                      // 000000002a10: d65d0024 01aa4880
	v_add_co_u32 v37, vcc_lo, v54, s18                         // 000000002a18: d7006a25 02002536
	s_wait_alu depctr_va_vcc(0)                                // 000000002a20: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s19, v57, vcc_lo            // 000000002a24: d5207c32 01aa7213
	s_and_b32 vcc_lo, s1, s2                                   // 000000002a2c: 8b6a0201
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002a30: d7620024 020248ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a3c: bf88ff9e
	v_cndmask_b32_e32 v37, 0, v37, vcc_lo                      // 000000002a40: 024a4a80
	v_cndmask_b32_e32 v51, 0, v50, vcc_lo                      // 000000002a44: 02666480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a48: bf870122
	v_add_co_u32 v50, s2, s30, v37                             // 000000002a4c: d7000232 02024a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002a54: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002a58: d5207c33 000a661f
	global_load_d16_hi_u8 v36, v[50:51], off                   // 000000002a60: ee08407c 00000024 00000032
	s_wait_loadcnt 0x0                                         // 000000002a6c: bfc00000
	v_cndmask_b16 v36.h, 0, v36.h, vcc_lo                      // 000000002a70: d65d5024 01aa4880
	v_add_co_u32 v37, vcc_lo, v54, s34                         // 000000002a78: d7006a25 02004536
	s_wait_alu depctr_va_vcc(0)                                // 000000002a80: bf88ff9d
	v_add_co_ci_u32_e64 v50, null, s35, v57, vcc_lo            // 000000002a84: d5207c32 01aa7223
	s_and_b32 vcc_lo, s1, s3                                   // 000000002a8c: 8b6a0301
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 000000002a90: d7385024 02024888
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a98: bf88ff9e
	v_cndmask_b32_e32 v37, 0, v37, vcc_lo                      // 000000002a9c: 024a4a80
	v_cndmask_b32_e32 v51, 0, v50, vcc_lo                      // 000000002aa0: 02666480
	s_lshr_b32 s3, s47, 5                                      // 000000002aa4: 8503852f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002aa8: bf88ff9e
	s_mul_i32 s3, s3, s18                                      // 000000002aac: 96031203
	v_add_co_u32 v50, s2, s30, v37                             // 000000002ab0: d7000232 02024a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab8: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002abc: d5207c33 000a661f
	global_load_d16_u8 v37, v[50:51], off                      // 000000002ac4: ee07807c 00000025 00000032
	s_wait_loadcnt 0x0                                         // 000000002ad0: bfc00000
	v_cndmask_b16 v37.l, 0, v37.l, vcc_lo                      // 000000002ad4: d65d0025 01aa4a80
	v_add_co_u32 v50, vcc_lo, v54, s36                         // 000000002adc: d7006a32 02004936
	s_wait_alu depctr_va_vcc(0)                                // 000000002ae4: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s37, v57, vcc_lo            // 000000002ae8: d5207c33 01aa7225
	s_and_b32 vcc_lo, s1, s4                                   // 000000002af0: 8b6a0401
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002af4: d7620025 02024aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b00: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002b04: ca526480 32326680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b0c: bf870121
	v_add_co_u32 v50, s2, s30, v50                             // 000000002b10: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b18: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002b1c: d5207c33 000a661f
	global_load_d16_hi_u8 v37, v[50:51], off                   // 000000002b24: ee08407c 00000025 00000032
	s_wait_loadcnt 0x0                                         // 000000002b30: bfc00000
	v_cndmask_b16 v37.h, 0, v37.h, vcc_lo                      // 000000002b34: d65d5025 01aa4a80
	v_add_co_u32 v50, vcc_lo, v54, s38                         // 000000002b3c: d7006a32 02004d36
	s_wait_alu depctr_va_vcc(0)                                // 000000002b44: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s39, v57, vcc_lo            // 000000002b48: d5207c33 01aa7227
	s_and_b32 vcc_lo, s1, s5                                   // 000000002b50: 8b6a0501
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002b54: d7385025 02024a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b5c: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v50 :: v_dual_cndmask_b32 v51, 0, v51// 000000002b60: ca526480 32326680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b68: bf870121
	v_add_co_u32 v50, s2, s30, v50                             // 000000002b6c: d7000232 0202641e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b74: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s31, v51, s2                // 000000002b78: d5207c33 000a661f
	global_load_d16_u8 v50, v[50:51], off                      // 000000002b80: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002b8c: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 000000002b90: d65d0032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s40                         // 000000002b98: d7006a33 02005136
	s_wait_alu depctr_va_vcc(0)                                // 000000002ba0: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s41, v57, vcc_lo            // 000000002ba4: d5207c37 01aa7229
	s_and_b32 vcc_lo, s1, s6                                   // 000000002bac: 8b6a0601
	v_and_b16 v50.l, 0xff, v50.l                               // 000000002bb0: d7620032 020264ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bbc: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002bc0: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002bc4: 02706e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bc8: bf870122
	v_add_co_u32 v55, s2, s30, v51                             // 000000002bcc: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd4: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002bd8: d5207c38 000a701f
	global_load_d16_hi_u8 v50, v[55:56], off                   // 000000002be0: ee08407c 00000032 00000037
	s_wait_loadcnt 0x0                                         // 000000002bec: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, vcc_lo                      // 000000002bf0: d65d5032 01aa6480
	v_add_co_u32 v51, vcc_lo, v54, s42                         // 000000002bf8: d7006a33 02005536
	s_wait_alu depctr_va_vcc(0)                                // 000000002c00: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s43, v57, vcc_lo            // 000000002c04: d5207c37 01aa722b
	s_and_b32 vcc_lo, s1, s7                                   // 000000002c0c: 8b6a0701
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000002c10: d7385032 02026488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c18: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002c1c: 02666680
	v_cndmask_b32_e32 v56, 0, v55, vcc_lo                      // 000000002c20: 02706e80
	s_lshr_b64 s[6:7], s[46:47], 5                             // 000000002c24: 8586852e
	s_add_nc_u64 s[46:47], s[46:47], 32                        // 000000002c28: a9aea02e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c2c: bf88ff9e
	s_mul_i32 s4, s6, s19                                      // 000000002c30: 96041306
	v_add_co_u32 v55, s2, s30, v51                             // 000000002c34: d7000237 0202661e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c3c: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s31, v56, s2                // 000000002c40: d5207c38 000a701f
	global_load_d16_u8 v51, v[55:56], off                      // 000000002c48: ee07807c 00000033 00000037
	s_wait_loadcnt 0x0                                         // 000000002c54: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, vcc_lo                      // 000000002c58: d65d0033 01aa6680
	v_add_co_u32 v54, vcc_lo, v54, s44                         // 000000002c60: d7006a36 02005936
	s_wait_alu depctr_va_vcc(0)                                // 000000002c68: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s45, v57, vcc_lo            // 000000002c6c: d5207c37 01aa722d
	s_and_b32 vcc_lo, s1, s8                                   // 000000002c74: 8b6a0801
	v_and_b16 v51.l, 0xff, v51.l                               // 000000002c78: d7620033 020266ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c84: bf88ff9e
	v_dual_cndmask_b32 v54, 0, v54 :: v_dual_cndmask_b32 v55, 0, v55// 000000002c88: ca526c80 36366e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c90: bf870121
	v_add_co_u32 v54, s2, s30, v54                             // 000000002c94: d7000236 02026c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c9c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s31, v55, s2                // 000000002ca0: d5207c37 000a6e1f
	global_load_d16_hi_u8 v51, v[54:55], off                   // 000000002ca8: ee08407c 00000033 00000036
	s_wait_loadcnt 0x0                                         // 000000002cb4: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, vcc_lo                      // 000000002cb8: d65d5033 01aa6680
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002cc0: bf870091
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000002cc4: d7385033 02026688
	v_or_b16 v51.h, v51.l, v51.h op_sel:[0,1,1]                // 000000002ccc: d7635033 02026733
	v_or_b16 v51.l, v50.l, v50.h op_sel:[0,1,0]                // 000000002cd4: d7631033 02026532
	v_or_b16 v50.l, v36.l, v36.h op_sel:[0,1,0]                // 000000002cdc: d7631032 02024924
	v_add_co_u32 v36, vcc_lo, v16, s6                          // 000000002ce4: d7006a24 02000d10
	v_or_b16 v50.h, v37.l, v37.h op_sel:[0,1,1]                // 000000002cec: d7635032 02024b25
	s_wait_alu depctr_va_vcc(0)                                // 000000002cf4: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s7, v17, vcc_lo             // 000000002cf8: d5207c25 01aa2207
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 000000002d00: bf870142
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[52:53], v[50:51], v[0:7]// 000000002d04: cc464000 1c026534
	global_load_u8 v52, v[36:37], off                          // 000000002d0c: ee04007c 00000034 00000024
	v_mad_co_u64_u32 v[36:37], null, s6, s18, v[32:33]         // 000000002d18: d6fe7c24 04802406
	v_cvt_f64_f32_e32 v[50:51], v0                             // 000000002d20: 7e642100
	v_add3_u32 v37, s4, s3, v37                                // 000000002d24: d6550025 04940604
	global_load_u8 v36, v[36:37], off                          // 000000002d2c: ee04007c 00000024 00000024
	s_wait_loadcnt 0x1                                         // 000000002d38: bfc00001
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v52                         // 000000002d3c: 7c9a68ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v52                                // 000000002d44: d44d0002 02026880
	s_wait_loadcnt 0x0                                         // 000000002d4c: bfc00000
	v_lshlrev_b32_e32 v0, 23, v36                              // 000000002d50: 30004897
	v_cmp_ne_u32_e64 s3, 0xff, v36                             // 000000002d54: d44d0003 020248ff 000000ff
	v_cmp_ne_u32_e64 s4, 0, v36                                // 000000002d60: d44d0004 02024880
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_1)// 000000002d68: bf8700a3
	v_cvt_f64_f32_e32 v[36:37], v0                             // 000000002d6c: 7e482100
	s_wait_alu depctr_va_sdst(0)                               // 000000002d70: bf88f19f
	v_cndmask_b32_e64 v0, 0x38000000, v37, s4                  // 000000002d74: d5010000 00124aff 38000000
	s_and_b32 s4, s3, s4                                       // 000000002d80: 8b040403
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d84: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002d88: bf870112
	v_cndmask_b32_e64 v36, 0, v36, s4                          // 000000002d8c: d5010024 00124880
	v_cndmask_b32_e64 v37, 0x7ff80000, v0, s3                  // 000000002d94: d5010025 000e00ff 7ff80000
	v_lshlrev_b32_e32 v0, 23, v52                              // 000000002da0: 30006897
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002da4: bf870091
	v_cvt_f64_f32_e32 v[52:53], v0                             // 000000002da8: 7e682100
	v_cndmask_b32_e64 v0, 0x38000000, v53, s2                  // 000000002dac: d5010000 000a6aff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000002db8: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dbc: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002dc0: bf870112
	v_cndmask_b32_e64 v52, 0, v52, s2                          // 000000002dc4: d5010034 000a6880
	v_cndmask_b32_e32 v53, 0x7ff80000, v0, vcc_lo              // 000000002dcc: 026a00ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002dd4: bf870091
	v_mul_f64_e32 v[50:51], v[52:53], v[50:51]                 // 000000002dd8: 0c646534
	v_mul_f64_e32 v[50:51], v[50:51], v[36:37]                 // 000000002ddc: 0c644932
	s_delay_alu instid0(valu_dep_1)                            // 000000002de0: bf870001
	v_cvt_f32_f64_e32 v0, v[50:51]                             // 000000002de4: 7e001f32
	v_add_co_u32 v50, vcc_lo, v18, s6                          // 000000002de8: d7006a32 02000d12
	s_wait_alu depctr_va_vcc(0)                                // 000000002df0: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s7, v19, vcc_lo             // 000000002df4: d5207c33 01aa2607
	global_load_u8 v50, v[50:51], off                          // 000000002dfc: ee04007c 00000032 00000032
	v_add_f32_e32 v41, v41, v0                                 // 000000002e08: 06520129
	v_cvt_f64_f32_e32 v[0:1], v1                               // 000000002e0c: 7e002101
	s_wait_loadcnt 0x0                                         // 000000002e10: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v50                         // 000000002e14: 7c9a64ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v50                                // 000000002e1c: d44d0002 02026480
	v_lshlrev_b32_e32 v50, 23, v50                             // 000000002e24: 30646497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002e28: bf8700a1
	v_cvt_f64_f32_e32 v[50:51], v50                            // 000000002e2c: 7e642132
	s_wait_alu depctr_va_sdst(0)                               // 000000002e30: bf88f19f
	v_cndmask_b32_e64 v51, 0x38000000, v51, s2                 // 000000002e34: d5010033 000a66ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000002e40: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e44: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002e48: bf870112
	v_cndmask_b32_e64 v50, 0, v50, s2                          // 000000002e4c: d5010032 000a6480
	v_cndmask_b32_e32 v51, 0x7ff80000, v51, vcc_lo             // 000000002e54: 026666ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e5c: bf870091
	v_mul_f64_e32 v[0:1], v[50:51], v[0:1]                     // 000000002e60: 0c000132
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 000000002e64: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e68: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 000000002e6c: 7e001f00
	v_add_f32_e32 v49, v49, v0                                 // 000000002e70: 06620131
	v_add_co_u32 v0, vcc_lo, v20, s6                           // 000000002e74: d7006a00 02000d14
	s_wait_alu depctr_va_vcc(0)                                // 000000002e7c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v21, vcc_lo              // 000000002e80: d5207c01 01aa2a07
	global_load_u8 v50, v[0:1], off                            // 000000002e88: ee04007c 00000032 00000000
	v_cvt_f64_f32_e32 v[0:1], v2                               // 000000002e94: 7e002102
	s_wait_loadcnt 0x0                                         // 000000002e98: bfc00000
	v_lshlrev_b32_e32 v2, 23, v50                              // 000000002e9c: 30046497
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v50                         // 000000002ea0: 7c9a64ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v50                                // 000000002ea8: d44d0002 02026480
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_1)// 000000002eb0: bf8700a3
	v_cvt_f64_f32_e32 v[50:51], v2                             // 000000002eb4: 7e642102
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb8: bf88f19f
	v_cndmask_b32_e64 v2, 0x38000000, v51, s2                  // 000000002ebc: d5010002 000a66ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000002ec8: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ecc: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002ed0: bf870112
	v_cndmask_b32_e64 v50, 0, v50, s2                          // 000000002ed4: d5010032 000a6480
	v_cndmask_b32_e32 v51, 0x7ff80000, v2, vcc_lo              // 000000002edc: 026604ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002ee4: bf870091
	v_mul_f64_e32 v[0:1], v[50:51], v[0:1]                     // 000000002ee8: 0c000132
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 000000002eec: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002ef0: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 000000002ef4: 7e001f00
	v_add_f32_e32 v48, v48, v0                                 // 000000002ef8: 06600130
	v_add_co_u32 v0, vcc_lo, v22, s6                           // 000000002efc: d7006a00 02000d16
	s_wait_alu depctr_va_vcc(0)                                // 000000002f04: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v23, vcc_lo              // 000000002f08: d5207c01 01aa2e07
	global_load_u8 v2, v[0:1], off                             // 000000002f10: ee04007c 00000002 00000000
	v_cvt_f64_f32_e32 v[0:1], v3                               // 000000002f1c: 7e002103
	s_wait_loadcnt 0x0                                         // 000000002f20: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v2                          // 000000002f24: 7c9a04ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 000000002f2c: d44d0002 02020480
	v_lshlrev_b32_e32 v2, 23, v2                               // 000000002f34: 30040497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002f38: bf8700a1
	v_cvt_f64_f32_e32 v[2:3], v2                               // 000000002f3c: 7e042102
	s_wait_alu depctr_va_sdst(0)                               // 000000002f40: bf88f19f
	v_cndmask_b32_e64 v3, 0x38000000, v3, s2                   // 000000002f44: d5010003 000a06ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000002f50: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f54: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002f58: bf870112
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 000000002f5c: d5010002 000a0480
	v_cndmask_b32_e32 v3, 0x7ff80000, v3, vcc_lo               // 000000002f64: 020606ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f6c: bf870091
	v_mul_f64_e32 v[0:1], v[2:3], v[0:1]                       // 000000002f70: 0c000102
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 000000002f74: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f78: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 000000002f7c: 7e001f00
	v_add_f32_e32 v47, v47, v0                                 // 000000002f80: 065e012f
	v_add_co_u32 v0, vcc_lo, v24, s6                           // 000000002f84: d7006a00 02000d18
	s_wait_alu depctr_va_vcc(0)                                // 000000002f8c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v25, vcc_lo              // 000000002f90: d5207c01 01aa3207
	global_load_u8 v2, v[0:1], off                             // 000000002f98: ee04007c 00000002 00000000
	v_cvt_f64_f32_e32 v[0:1], v4                               // 000000002fa4: 7e002104
	s_wait_loadcnt 0x0                                         // 000000002fa8: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v2                          // 000000002fac: 7c9a04ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 000000002fb4: d44d0002 02020480
	v_lshlrev_b32_e32 v2, 23, v2                               // 000000002fbc: 30040497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002fc0: bf8700a1
	v_cvt_f64_f32_e32 v[2:3], v2                               // 000000002fc4: 7e042102
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc8: bf88f19f
	v_cndmask_b32_e64 v3, 0x38000000, v3, s2                   // 000000002fcc: d5010003 000a06ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000002fd8: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fdc: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002fe0: bf870112
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 000000002fe4: d5010002 000a0480
	v_cndmask_b32_e32 v3, 0x7ff80000, v3, vcc_lo               // 000000002fec: 020606ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002ff4: bf870091
	v_mul_f64_e32 v[0:1], v[2:3], v[0:1]                       // 000000002ff8: 0c000102
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 000000002ffc: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003000: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 000000003004: 7e001f00
	v_add_f32_e32 v44, v44, v0                                 // 000000003008: 0658012c
	v_add_co_u32 v0, vcc_lo, v26, s6                           // 00000000300c: d7006a00 02000d1a
	s_wait_alu depctr_va_vcc(0)                                // 000000003014: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v27, vcc_lo              // 000000003018: d5207c01 01aa3607
	global_load_u8 v2, v[0:1], off                             // 000000003020: ee04007c 00000002 00000000
	v_cvt_f64_f32_e32 v[0:1], v5                               // 00000000302c: 7e002105
	s_wait_loadcnt 0x0                                         // 000000003030: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v2                          // 000000003034: 7c9a04ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 00000000303c: d44d0002 02020480
	v_lshlrev_b32_e32 v2, 23, v2                               // 000000003044: 30040497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003048: bf8700a1
	v_cvt_f64_f32_e32 v[2:3], v2                               // 00000000304c: 7e042102
	s_wait_alu depctr_va_sdst(0)                               // 000000003050: bf88f19f
	v_cndmask_b32_e64 v3, 0x38000000, v3, s2                   // 000000003054: d5010003 000a06ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000003060: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003064: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003068: bf870112
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 00000000306c: d5010002 000a0480
	v_cndmask_b32_e32 v3, 0x7ff80000, v3, vcc_lo               // 000000003074: 020606ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000307c: bf870091
	v_mul_f64_e32 v[0:1], v[2:3], v[0:1]                       // 000000003080: 0c000102
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 000000003084: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003088: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 00000000308c: 7e001f00
	v_add_f32_e32 v42, v42, v0                                 // 000000003090: 0654012a
	v_add_co_u32 v0, vcc_lo, v28, s6                           // 000000003094: d7006a00 02000d1c
	s_wait_alu depctr_va_vcc(0)                                // 00000000309c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v29, vcc_lo              // 0000000030a0: d5207c01 01aa3a07
	global_load_u8 v2, v[0:1], off                             // 0000000030a8: ee04007c 00000002 00000000
	v_cvt_f64_f32_e32 v[0:1], v6                               // 0000000030b4: 7e002106
	s_wait_loadcnt 0x0                                         // 0000000030b8: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v2                          // 0000000030bc: 7c9a04ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 0000000030c4: d44d0002 02020480
	v_lshlrev_b32_e32 v2, 23, v2                               // 0000000030cc: 30040497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000030d0: bf8700a1
	v_cvt_f64_f32_e32 v[2:3], v2                               // 0000000030d4: 7e042102
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d8: bf88f19f
	v_cndmask_b32_e64 v3, 0x38000000, v3, s2                   // 0000000030dc: d5010003 000a06ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 0000000030e8: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030ec: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000030f0: bf870112
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 0000000030f4: d5010002 000a0480
	v_cndmask_b32_e32 v3, 0x7ff80000, v3, vcc_lo               // 0000000030fc: 020606ff 7ff80000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003104: bf870091
	v_mul_f64_e32 v[0:1], v[2:3], v[0:1]                       // 000000003108: 0c000102
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 00000000310c: 0c000124
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003110: bf870091
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 000000003114: 7e001f00
	v_add_f32_e32 v40, v40, v0                                 // 000000003118: 06500128
	v_add_co_u32 v0, vcc_lo, v30, s6                           // 00000000311c: d7006a00 02000d1e
	s_wait_alu depctr_va_vcc(0)                                // 000000003124: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v31, vcc_lo              // 000000003128: d5207c01 01aa3e07
	global_load_u8 v2, v[0:1], off                             // 000000003130: ee04007c 00000002 00000000
	v_cvt_f64_f32_e32 v[0:1], v7                               // 00000000313c: 7e002107
	s_wait_loadcnt 0x0                                         // 000000003140: bfc00000
	v_cmp_ne_u32_e32 vcc_lo, 0xff, v2                          // 000000003144: 7c9a04ff 000000ff
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 00000000314c: d44d0002 02020480
	v_lshlrev_b32_e32 v2, 23, v2                               // 000000003154: 30040497
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003158: bf8700a1
	v_cvt_f64_f32_e32 v[2:3], v2                               // 00000000315c: 7e042102
	s_wait_alu depctr_va_sdst(0)                               // 000000003160: bf88f19f
	v_cndmask_b32_e64 v3, 0x38000000, v3, s2                   // 000000003164: d5010003 000a06ff 38000000
	s_and_b32 s2, vcc_lo, s2                                   // 000000003170: 8b02026a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003174: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 000000003178: bf8700b2
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 00000000317c: d5010002 000a0480
	v_cmp_lt_i64_e64 s2, s[46:47], s[22:23]                    // 000000003184: d4510002 02002c2e
	v_cndmask_b32_e32 v3, 0x7ff80000, v3, vcc_lo               // 00000000318c: 020606ff 7ff80000
	v_mul_f64_e32 v[0:1], v[2:3], v[0:1]                       // 000000003194: 0c000102
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000003198: 8b6a027e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000319c: bf870091
	v_mul_f64_e32 v[0:1], v[36:37], v[0:1]                     // 0000000031a0: 0c000124
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 0000000031a4: 7e001f00
	s_delay_alu instid0(valu_dep_1)                            // 0000000031a8: bf870001
	v_add_f32_e32 v10, v10, v0                                 // 0000000031ac: 0614010a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031b0: bf88ff9e
	s_cbranch_vccnz 64335                                      // 0000000031b4: bfa4fb4f <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x3f4>
	v_mul_lo_u32 v4, s19, v14                                  // 0000000031b8: d72c0004 02021c13
	v_mul_lo_u32 v5, s18, v15                                  // 0000000031c0: d72c0005 02021e12
	v_mad_co_u64_u32 v[2:3], null, s18, v14, 0                 // 0000000031c8: d6fe7c02 02021c12
	v_sub_co_u32 v0, vcc_lo, s16, v14                          // 0000000031d0: d7016a00 02021c10
	s_wait_alu depctr_va_vcc(0)                                // 0000000031d8: bf88ff9d
	v_sub_co_ci_u32_e64 v1, null, s17, v15, vcc_lo             // 0000000031dc: d5217c01 01aa1e11
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[11:12]                // 0000000031e4: 7ca81612
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 0000000031e8: bf870194
	v_add3_u32 v3, v3, v5, v4                                  // 0000000031ec: d6550003 04120b03
	v_cmp_lt_i64_e64 s0, 0, v[0:1]                             // 0000000031f4: d4510000 02020080
	s_delay_alu instid0(valu_dep_2)                            // 0000000031fc: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003200: 3e040481
	s_and_b32 s0, s0, vcc_lo                                   // 000000003204: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003208: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000320c: be812000
	s_cbranch_execz 28                                         // 000000003210: bfa5001c <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1784>
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003214: 3e081681
	v_add_co_u32 v7, s0, s20, v2                               // 000000003218: d7000007 02020414
	v_bfe_u32 v6, v41, 16, 1                                   // 000000003220: d6100006 02052129
	s_wait_alu depctr_va_sdst(0)                               // 000000003228: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s0                  // 00000000322c: d5207c08 00020615
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003234: bf870193
	v_add_co_u32 v4, s0, v7, v4                                // 000000003238: d7000004 02020907
	v_add3_u32 v6, v6, v41, 0x7fff                             // 000000003240: d6550006 03fe5306 00007fff
	v_or_b32_e32 v9, 0x400000, v41                             // 00000000324c: 381252ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003254: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s0                   // 000000003258: d5207c05 00020b08
	v_cmp_u_f32_e64 s0, v41, v41                               // 000000003260: d4180000 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000003268: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000326c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s0                           // 000000003270: d5010006 00021306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003278: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003284: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003288: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[0:1]                             // 00000000328c: d4510000 02020081
	s_and_b32 s0, s0, vcc_lo                                   // 000000003294: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003298: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000329c: be812000
	s_cbranch_execz 35                                         // 0000000032a0: bfa50023 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1830>
	v_add_co_u32 v6, s0, s20, v2                               // 0000000032a4: d7000006 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000032ac: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s21, v3, s0                  // 0000000032b0: d5207c07 00020615
	s_lshl_b64 s[2:3], s[18:19], 1                             // 0000000032b8: 84828112
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 0000000032bc: 3e081681
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032c0: bf88ff9e
	v_add_co_u32 v6, s0, v6, s2                                // 0000000032c4: d7000006 02000506
	v_bfe_u32 v8, v49, 16, 1                                   // 0000000032cc: d6100008 02052131
	s_wait_alu depctr_va_sdst(0)                               // 0000000032d4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s3, v7, s0                   // 0000000032d8: d5207c07 00020e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000032e0: bf870193
	v_add_co_u32 v4, s0, v6, v4                                // 0000000032e4: d7000004 02020906
	v_add3_u32 v8, v8, v49, 0x7fff                             // 0000000032ec: d6550008 03fe6308 00007fff
	v_or_b32_e32 v9, 0x400000, v49                             // 0000000032f8: 381262ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003300: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s0                   // 000000003304: d5207c05 00020b07
	v_cmp_u_f32_e64 s0, v49, v49                               // 00000000330c: d4180000 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000003314: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003318: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s0                           // 00000000331c: d5010006 00021308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003324: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003330: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003334: 8c7e017e
	v_cmp_lt_i64_e64 s0, 2, v[0:1]                             // 000000003338: d4510000 02020082
	s_and_b32 s0, s0, vcc_lo                                   // 000000003340: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003344: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003348: be812000
	s_cbranch_execz 34                                         // 00000000334c: bfa50022 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x18d8>
	v_add_co_u32 v8, s0, s20, v2                               // 000000003350: d7000008 02020414
	v_bfe_u32 v4, v48, 16, 1                                   // 000000003358: d6100004 02052130
	s_wait_alu depctr_va_sdst(0)                               // 000000003360: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s21, v3, s0                  // 000000003364: d5207c09 00020615
	s_lshl_b64 s[2:3], s[18:19], 2                             // 00000000336c: 84828212
	v_or_b32_e32 v6, 0x400000, v48                             // 000000003370: 380c60ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003378: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 00000000337c: d7000008 02000508
	v_add3_u32 v7, v4, v48, 0x7fff                             // 000000003384: d6550007 03fe6104 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 000000003390: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 000000003394: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 000000003398: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v48, v48                               // 0000000033a0: d4180000 02026130
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033ac: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 0000000033b0: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 0000000033b8: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 0000000033c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 0000000033c4: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000033cc: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000033dc: 8c7e017e
	v_cmp_lt_i64_e64 s0, 3, v[0:1]                             // 0000000033e0: d4510000 02020083
	s_and_b32 s0, s0, vcc_lo                                   // 0000000033e8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000033f0: be812000
	s_cbranch_execz 33                                         // 0000000033f4: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x197c>
	v_add_co_u32 v4, s0, s20, v2                               // 0000000033f8: d7000004 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003400: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v3, s0                  // 000000003404: d5207c05 00020615
	v_bfe_u32 v6, v47, 16, 1                                   // 00000000340c: d6100006 0205212f
	v_or_b32_e32 v8, 0x400000, v47                             // 000000003414: 38105eff 00400000
	v_cmp_u_f32_e64 s0, v47, v47                               // 00000000341c: d4180000 02025f2f
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003424: bf870214
	v_mad_co_u64_u32 v[4:5], null, s18, 6, v[4:5]              // 000000003428: d6fe7c04 04110c12
	v_add3_u32 v9, v6, v47, 0x7fff                             // 000000003430: d6550009 03fe5f06 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000343c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003440: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003444: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s19, 6, v[5:6]              // 00000000344c: d6fe7c05 04150c13
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003454: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003458: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 00000000345c: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003464: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 000000003468: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000003470: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000347c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003480: 8c7e017e
	v_cmp_lt_i64_e64 s0, 4, v[0:1]                             // 000000003484: d4510000 02020084
	s_and_b32 s0, s0, vcc_lo                                   // 00000000348c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003490: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003494: be812000
	s_cbranch_execz 34                                         // 000000003498: bfa50022 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1a24>
	v_add_co_u32 v8, s0, s20, v2                               // 00000000349c: d7000008 02020414
	v_bfe_u32 v4, v44, 16, 1                                   // 0000000034a4: d6100004 0205212c
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ac: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s21, v3, s0                  // 0000000034b0: d5207c09 00020615
	s_lshl_b64 s[2:3], s[18:19], 3                             // 0000000034b8: 84828312
	v_or_b32_e32 v6, 0x400000, v44                             // 0000000034bc: 380c58ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034c4: bf88ff9e
	v_add_co_u32 v8, s0, v8, s2                                // 0000000034c8: d7000008 02000508
	v_add3_u32 v7, v4, v44, 0x7fff                             // 0000000034d0: d6550007 03fe5904 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[11:12]                      // 0000000034dc: 3e081681
	s_wait_alu depctr_va_sdst(0)                               // 0000000034e0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s3, v9, s0                   // 0000000034e4: d5207c09 00021203
	v_cmp_u_f32_e64 s0, v44, v44                               // 0000000034ec: d4180000 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034f8: bf870001
	v_cndmask_b32_e64 v6, v7, v6, s0                           // 0000000034fc: d5010006 00020d07
	v_add_co_u32 v4, s0, v8, v4                                // 000000003504: d7000004 02020908
	s_wait_alu depctr_va_sdst(0)                               // 00000000350c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v9, v5, s0                   // 000000003510: d5207c05 00020b09
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003518: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003524: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003528: 8c7e017e
	v_cmp_lt_i64_e64 s0, 5, v[0:1]                             // 00000000352c: d4510000 02020085
	s_and_b32 s0, s0, vcc_lo                                   // 000000003534: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003538: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000353c: be812000
	s_cbranch_execz 33                                         // 000000003540: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1ac8>
	v_add_co_u32 v4, s0, s20, v2                               // 000000003544: d7000004 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000354c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v3, s0                  // 000000003550: d5207c05 00020615
	v_bfe_u32 v6, v42, 16, 1                                   // 000000003558: d6100006 0205212a
	v_or_b32_e32 v8, 0x400000, v42                             // 000000003560: 381054ff 00400000
	v_cmp_u_f32_e64 s0, v42, v42                               // 000000003568: d4180000 0202552a
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003570: bf870214
	v_mad_co_u64_u32 v[4:5], null, s18, 10, v[4:5]             // 000000003574: d6fe7c04 04111412
	v_add3_u32 v9, v6, v42, 0x7fff                             // 00000000357c: d6550009 03fe5506 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003588: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 00000000358c: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003590: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s19, 10, v[5:6]             // 000000003598: d6fe7c05 04151413
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 0000000035a0: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000035a4: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 0000000035a8: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 0000000035b4: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 0000000035bc: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000035cc: 8c7e017e
	v_cmp_lt_i64_e64 s0, 6, v[0:1]                             // 0000000035d0: d4510000 02020086
	s_and_b32 s0, s0, vcc_lo                                   // 0000000035d8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035dc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000035e0: be812000
	s_cbranch_execz 33                                         // 0000000035e4: bfa50021 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1b6c>
	v_add_co_u32 v4, s0, s20, v2                               // 0000000035e8: d7000004 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000035f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v3, s0                  // 0000000035f4: d5207c05 00020615
	v_bfe_u32 v6, v40, 16, 1                                   // 0000000035fc: d6100006 02052128
	v_or_b32_e32 v8, 0x400000, v40                             // 000000003604: 381050ff 00400000
	v_cmp_u_f32_e64 s0, v40, v40                               // 00000000360c: d4180000 02025128
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003614: bf870214
	v_mad_co_u64_u32 v[4:5], null, s18, 12, v[4:5]             // 000000003618: d6fe7c04 04111812
	v_add3_u32 v9, v6, v40, 0x7fff                             // 000000003620: d6550009 03fe5106 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000362c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003630: bf870191
	v_cndmask_b32_e64 v8, v9, v8, s0                           // 000000003634: d5010008 00021109
	v_mad_co_u64_u32 v[5:6], null, s19, 12, v[5:6]             // 00000000363c: d6fe7c05 04151813
	v_lshlrev_b64_e32 v[6:7], 1, v[11:12]                      // 000000003644: 3e0c1681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003648: bf870121
	v_add_co_u32 v4, s0, v4, v6                                // 00000000364c: d7000004 02020d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003654: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v5, v7, s0                   // 000000003658: d5207c05 00020f05
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000003660: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000366c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003670: 8c7e017e
	v_cmp_lt_i64_e64 s0, 7, v[0:1]                             // 000000003674: d4510000 02020087
	s_mov_b32 s1, 0                                            // 00000000367c: be810080
	s_and_b32 s2, s0, vcc_lo                                   // 000000003680: 8b026a00
	s_mov_b32 s0, 0                                            // 000000003684: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003688: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000368c: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003690: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 000000003694: 8d02037e
	v_add_co_u32 v0, vcc_lo, s20, v2                           // 000000003698: d7006a00 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000036a0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v3, vcc_lo              // 0000000036a4: d5207c01 01aa0615
	s_mov_b32 s0, exec_lo                                      // 0000000036ac: be80007e
	v_mad_co_u64_u32 v[0:1], null, s18, 14, v[0:1]             // 0000000036b0: d6fe7c00 04011c12
	s_delay_alu instid0(valu_dep_1)                            // 0000000036b8: bf870001
	v_mad_co_u64_u32 v[1:2], null, s19, 14, v[1:2]             // 0000000036bc: d6fe7c01 04051c13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000036c8: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 0000000036cc: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 0000000036d0: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036d4: bf88ff9e
	s_cbranch_vccz 930                                         // 0000000036d8: bfa303a2 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2a64>
	s_and_b32 s0, s11, exec_lo                                 // 0000000036dc: 8b007e0b
	s_cselect_b32 s0, 1, 0                                     // 0000000036e0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036e4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000036e8: bf078100
	s_cbranch_scc1 19                                          // 0000000036ec: bfa20013 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1c3c>
	v_lshl_or_b32 v14, v39, 3, s14                             // 0000000036f0: d656000e 00390727
	v_mov_b32_e32 v15, s15                                     // 0000000036f8: 7e1e020f
	v_mov_b32_e32 v21, s15                                     // 0000000036fc: 7e2a020f
	v_mov_b32_e32 v17, s15                                     // 000000003700: 7e22020f
	v_mov_b32_e32 v19, s15                                     // 000000003704: 7e26020f
	v_or_b32_e32 v20, 1, v14                                   // 000000003708: 38281c81
	v_or_b32_e32 v16, 2, v14                                   // 00000000370c: 38201c82
	v_or_b32_e32 v18, 3, v14                                   // 000000003710: 38241c83
	v_or_b32_e32 v6, 4, v14                                    // 000000003714: 380c1c84
	v_mov_b32_e32 v7, s15                                      // 000000003718: 7e0e020f
	v_or_b32_e32 v8, 5, v14                                    // 00000000371c: 38101c85
	v_mov_b32_e32 v9, s15                                      // 000000003720: 7e12020f
	v_or_b32_e32 v4, 6, v14                                    // 000000003724: 38081c86
	v_mov_b32_e32 v5, s15                                      // 000000003728: 7e0a020f
	v_or_b32_e32 v2, 7, v14                                    // 00000000372c: 38041c87
	v_mov_b32_e32 v3, s15                                      // 000000003730: 7e06020f
	s_mov_b32 s0, 0                                            // 000000003734: be800080
	s_branch 1                                                 // 000000003738: bfa00001 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1c40>
	s_mov_b32 s0, -1                                           // 00000000373c: be8000c1
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v41, 0             // 000000003740: ca100080 0a280080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003748: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 00000000374c: 8b007e00
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0             // 000000003750: ca100080 2a2a0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0             // 000000003758: ca100080 2d2e0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v40, 0             // 000000003760: ca100080 2f280080
	s_cselect_b32 s0, 1, 0                                     // 000000003768: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000376c: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000003770: bf078100
	s_cbranch_scc1 643                                         // 000000003774: bfa20283 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2684>
	v_dual_mov_b32 v15, s15 :: v_dual_lshlrev_b32 v2, 3, v39   // 000000003778: ca22000f 0f024e83
	v_add_co_u32 v0, vcc_lo, s30, v11                          // 000000003780: d7006a00 0202161e
	s_wait_alu depctr_va_vcc(0)                                // 000000003788: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s31, v12, vcc_lo             // 00000000378c: d5207c01 01aa181f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003794: bf8701c3
	v_or_b32_e32 v14, s14, v2                                  // 000000003798: 381c040e
	v_add_co_u32 v3, vcc_lo, s28, v13                          // 00000000379c: d7006a03 02021a1c
	s_wait_alu depctr_va_vcc(0)                                // 0000000037a4: bf88ff9d
	v_add_co_ci_u32_e64 v4, null, s29, v38, vcc_lo             // 0000000037a8: d5207c04 01aa4c1d
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[14:15]                // 0000000037b0: 7ca81c10
	v_or_b32_e32 v20, 1, v14                                   // 0000000037b4: 38281c81
	v_mov_b32_e32 v21, s15                                     // 0000000037b8: 7e2a020f
	v_add_co_u32 v13, s0, v3, v2                               // 0000000037bc: d700000d 02020503
	v_mad_co_u64_u32 v[0:1], null, s18, v2, v[0:1]             // 0000000037c4: d6fe7c00 04020412
	s_wait_alu depctr_va_vcc(0)                                // 0000000037cc: bf88ff9d
	v_dual_mov_b32 v40, 0 :: v_dual_cndmask_b32 v3, 0, v14     // 0000000037d0: ca120080 28021c80
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d8: bf88f19f
	v_add_co_ci_u32_e64 v44, null, 0, v4, s0                   // 0000000037dc: d5207c2c 00020880
	v_cndmask_b32_e64 v4, 0, s15, vcc_lo                       // 0000000037e4: d5010004 01a81e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[20:21]                // 0000000037ec: 7ca82810
	v_or_b32_e32 v16, 2, v14                                   // 0000000037f0: 38201c82
	v_or_b32_e32 v18, 3, v14                                   // 0000000037f4: 38241c83
	v_mov_b32_e32 v17, s15                                     // 0000000037f8: 7e22020f
	s_lshr_b64 s[2:3], s[26:27], 5                             // 0000000037fc: 8582851a
	s_lshr_b32 s1, s27, 5                                      // 000000003800: 8501851b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003804: bf88ff9e
	v_mul_lo_u32 v4, s2, v4                                    // 000000003808: d72c0004 02020802
	v_mul_lo_u32 v5, s1, v3                                    // 000000003810: d72c0005 02020601
	v_mad_co_u64_u32 v[22:23], null, s2, v3, s[24:25]          // 000000003818: d6fe7c16 00620602
	v_mad_co_u64_u32 v[1:2], null, s19, v2, v[1:2]             // 000000003820: d6fe7c01 04060413
	v_cmp_gt_i64_e64 s0, s[18:19], v[11:12]                    // 000000003828: d4540000 02021612
	s_wait_alu depctr_va_vcc(0)                                // 000000003830: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v20, vcc_lo                       // 000000003834: 02042880
	v_cndmask_b32_e64 v3, 0, s15, vcc_lo                       // 000000003838: d5010003 01a81e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[16:17]                // 000000003840: 7ca82010
	v_or_b32_e32 v6, 4, v14                                    // 000000003844: 380c1c84
	v_mov_b32_e32 v7, s15                                      // 000000003848: 7e0e020f
	v_or_b32_e32 v8, 5, v14                                    // 00000000384c: 38101c85
	v_mov_b32_e32 v19, s15                                     // 000000003850: 7e26020f
	s_wait_alu depctr_va_sdst(0)                               // 000000003854: bf88f19f
	v_cndmask_b32_e64 v10, 0, v12, s0                          // 000000003858: d501000a 00021880
	v_cndmask_b32_e64 v38, 0, v11, s0                          // 000000003860: d5010026 00021680
	v_add3_u32 v23, v5, v23, v4                                // 000000003868: d6550017 04122f05
	v_mul_lo_u32 v41, s2, v3                                   // 000000003870: d72c0029 02020602
	s_wait_alu depctr_va_vcc(0)                                // 000000003878: bf88ff9d
	v_cndmask_b32_e32 v3, 0, v16, vcc_lo                       // 00000000387c: 02062080
	v_cndmask_b32_e64 v4, 0, s15, vcc_lo                       // 000000003880: d5010004 01a81e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[18:19]                // 000000003888: 7ca82410
	v_cmp_gt_i64_e64 s0, s[16:17], v[6:7]                      // 00000000388c: d4540000 02020c10
	v_mul_lo_u32 v42, s1, v2                                   // 000000003894: d72c002a 02020401
	v_mad_co_u64_u32 v[24:25], null, s2, v2, s[24:25]          // 00000000389c: d6fe7c18 00620402
	v_mov_b32_e32 v9, s15                                      // 0000000038a4: 7e12020f
	v_mul_lo_u32 v43, s2, v4                                   // 0000000038a8: d72c002b 02020802
	v_mul_lo_u32 v45, s1, v3                                   // 0000000038b0: d72c002d 02020601
	v_mad_co_u64_u32 v[26:27], null, s2, v3, s[24:25]          // 0000000038b8: d6fe7c1a 00620602
	s_wait_alu depctr_va_vcc(0)                                // 0000000038c0: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v18, vcc_lo                       // 0000000038c4: 02042480
	v_cndmask_b32_e64 v3, 0, s15, vcc_lo                       // 0000000038c8: d5010003 01a81e80
	s_wait_alu depctr_va_sdst(0)                               // 0000000038d0: bf88f19f
	v_cndmask_b32_e64 v4, 0, s15, s0                           // 0000000038d4: d5010004 00001e80
	v_add3_u32 v25, v42, v25, v41                              // 0000000038dc: d6550019 04a6332a
	v_mov_b32_e32 v42, 0                                       // 0000000038e4: 7e540280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 0000000038e8: 7ca81010
	v_mul_lo_u32 v46, s2, v3                                   // 0000000038ec: d72c002e 02020602
	v_mul_lo_u32 v47, s1, v2                                   // 0000000038f4: d72c002f 02020401
	v_mad_co_u64_u32 v[28:29], null, s2, v2, s[24:25]          // 0000000038fc: d6fe7c1c 00620402
	v_mul_lo_u32 v48, s2, v4                                   // 000000003904: d72c0030 02020802
	v_or_b32_e32 v4, 6, v14                                    // 00000000390c: 38081c86
	v_mov_b32_e32 v5, s15                                      // 000000003910: 7e0a020f
	v_or_b32_e32 v2, 7, v14                                    // 000000003914: 38041c87
	v_mov_b32_e32 v3, s15                                      // 000000003918: 7e06020f
	v_cndmask_b32_e64 v30, 0, v6, s0                           // 00000000391c: d501001e 00020c80
	s_wait_alu depctr_va_vcc(0)                                // 000000003924: bf88ff9d
	v_cndmask_b32_e32 v32, 0, v8, vcc_lo                       // 000000003928: 02401080
	v_cndmask_b32_e64 v33, 0, s15, vcc_lo                      // 00000000392c: d5010021 01a81e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000003934: 7ca80810
	v_cmp_gt_i64_e64 s0, s[16:17], v[2:3]                      // 000000003938: d4540000 02020410
	v_mul_lo_u32 v49, s1, v30                                  // 000000003940: d72c0031 02023c01
	v_mad_co_u64_u32 v[30:31], null, s2, v30, s[24:25]         // 000000003948: d6fe7c1e 00623c02
	v_mul_lo_u32 v50, s2, v33                                  // 000000003950: d72c0032 02024202
	v_mul_lo_u32 v51, s1, v32                                  // 000000003958: d72c0033 02024001
	s_wait_alu depctr_va_vcc(0)                                // 000000003960: bf88ff9d
	v_cndmask_b32_e32 v34, 0, v4, vcc_lo                       // 000000003964: 02440880
	v_cndmask_b32_e64 v35, 0, s15, vcc_lo                      // 000000003968: d5010023 01a81e80
	s_wait_alu depctr_va_sdst(0)                               // 000000003970: bf88f19f
	v_cndmask_b32_e64 v36, 0, v2, s0                           // 000000003974: d5010024 00020480
	v_cndmask_b32_e64 v37, 0, s15, s0                          // 00000000397c: d5010025 00001e80
	v_mad_co_u64_u32 v[32:33], null, s2, v32, s[24:25]         // 000000003984: d6fe7c20 00624002
	v_mul_lo_u32 v53, s1, v34                                  // 00000000398c: d72c0035 02024401
	v_mul_lo_u32 v52, s2, v35                                  // 000000003994: d72c0034 02024602
	v_mad_co_u64_u32 v[34:35], null, s2, v34, s[24:25]         // 00000000399c: d6fe7c22 00624402
	v_mul_lo_u32 v54, s2, v37                                  // 0000000039a4: d72c0036 02024a02
	v_mul_lo_u32 v55, s1, v36                                  // 0000000039ac: d72c0037 02024801
	v_mad_co_u64_u32 v[36:37], null, s2, v36, s[24:25]         // 0000000039b4: d6fe7c24 00624802
	v_add_co_u32 v38, vcc_lo, s12, v38                         // 0000000039bc: d7006a26 02024c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000039c4: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s13, v10, vcc_lo            // 0000000039c8: d5207c27 01aa140d
	v_add3_u32 v27, v45, v27, v43                              // 0000000039d0: d655001b 04ae372d
	v_add3_u32 v29, v47, v29, v46                              // 0000000039d8: d655001d 04ba3b2f
	v_add3_u32 v31, v49, v31, v48                              // 0000000039e0: d655001f 04c23f31
	v_add3_u32 v33, v51, v33, v50                              // 0000000039e8: d6550021 04ca4333
	v_add3_u32 v35, v53, v35, v52                              // 0000000039f0: d6550023 04d24735
	v_add3_u32 v37, v55, v37, v54                              // 0000000039f8: d6550025 04da4b37
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0             // 000000003a00: ca100080 2f2e0080
	v_mov_b32_e32 v45, 0                                       // 000000003a08: 7e5a0280
	v_mov_b32_e32 v43, 0                                       // 000000003a0c: 7e560280
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v10, 0             // 000000003a10: ca100080 290a0080
	s_lshl_b64 s[16:17], s[18:19], 4                           // 000000003a18: 84908412
	s_mov_b64 s[24:25], 0                                      // 000000003a1c: be980180
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a20: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v13, s24                         // 000000003a24: d7006a30 0200310d
	s_lshr_b64 s[0:1], s[24:25], 5                             // 000000003a2c: 85808518
	s_wait_alu depctr_va_vcc(0)                                // 000000003a30: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s25, v44, vcc_lo            // 000000003a34: d5207c31 01aa5819
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a3c: bf88ff9e
	v_add_co_u32 v52, vcc_lo, v22, s0                          // 000000003a40: d7006a34 02000116
	s_wait_alu depctr_va_vcc(0)                                // 000000003a48: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s1, v23, vcc_lo             // 000000003a4c: d5207c35 01aa2e01
	v_add_co_u32 v56, vcc_lo, v24, s0                          // 000000003a54: d7006a38 02000118
	s_wait_alu depctr_va_vcc(0)                                // 000000003a5c: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s1, v25, vcc_lo             // 000000003a60: d5207c39 01aa3201
	v_add_co_u32 v58, vcc_lo, v26, s0                          // 000000003a68: d7006a3a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000003a70: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s1, v27, vcc_lo             // 000000003a74: d5207c3b 01aa3601
	v_add_co_u32 v60, vcc_lo, v28, s0                          // 000000003a7c: d7006a3c 0200011c
	s_wait_alu depctr_va_vcc(0)                                // 000000003a84: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s1, v29, vcc_lo             // 000000003a88: d5207c3d 01aa3a01
	v_add_co_u32 v62, vcc_lo, v30, s0                          // 000000003a90: d7006a3e 0200011e
	s_wait_alu depctr_va_vcc(0)                                // 000000003a98: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s1, v31, vcc_lo             // 000000003a9c: d5207c3f 01aa3e01
	v_add_co_u32 v64, vcc_lo, v32, s0                          // 000000003aa4: d7006a40 02000120
	s_wait_alu depctr_va_vcc(0)                                // 000000003aac: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s1, v33, vcc_lo             // 000000003ab0: d5207c41 01aa4201
	v_add_co_u32 v66, vcc_lo, v34, s0                          // 000000003ab8: d7006a42 02000122
	s_wait_alu depctr_va_vcc(0)                                // 000000003ac0: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s1, v35, vcc_lo             // 000000003ac4: d5207c43 01aa4601
	v_add_co_u32 v68, vcc_lo, v36, s0                          // 000000003acc: d7006a44 02000124
	s_wait_alu depctr_va_vcc(0)                                // 000000003ad4: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s1, v37, vcc_lo             // 000000003ad8: d5207c45 01aa4a01
	s_clause 0x1                                               // 000000003ae0: bf850001
	global_load_b64 v[70:71], v[48:49], off                    // 000000003ae4: ee05407c 00000046 00000030
	global_load_b64 v[72:73], v[48:49], off offset:16          // 000000003af0: ee05407c 00000048 00001030
	s_clause 0x7                                               // 000000003afc: bf850007
	global_load_u8 v79, v[52:53], off                          // 000000003b00: ee04007c 0000004f 00000034
	global_load_u8 v81, v[56:57], off                          // 000000003b0c: ee04007c 00000051 00000038
	global_load_u8 v82, v[58:59], off                          // 000000003b18: ee04007c 00000052 0000003a
	global_load_u8 v83, v[60:61], off                          // 000000003b24: ee04007c 00000053 0000003c
	global_load_u8 v84, v[62:63], off                          // 000000003b30: ee04007c 00000054 0000003e
	global_load_u8 v85, v[64:65], off                          // 000000003b3c: ee04007c 00000055 00000040
	global_load_u8 v86, v[66:67], off                          // 000000003b48: ee04007c 00000056 00000042
	global_load_u8 v87, v[68:69], off                          // 000000003b54: ee04007c 00000057 00000044
	v_mad_co_u64_u32 v[50:51], null, s24, s18, v[0:1]          // 000000003b60: d6fe7c32 04002418
	s_mul_i32 s2, s25, s18                                     // 000000003b68: 96021219
	s_mul_i32 s3, s24, s19                                     // 000000003b6c: 96031318
	s_mul_i32 s5, s0, s19                                      // 000000003b70: 96051300
	v_mad_co_u64_u32 v[54:55], null, s0, s18, v[38:39]         // 000000003b74: d6fe7c36 04982400
	s_lshr_b32 s4, s25, 5                                      // 000000003b7c: 85048519
	s_add_nc_u64 s[24:25], s[24:25], 32                        // 000000003b80: a998a018
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b84: bf88ff9e
	s_mul_i32 s4, s4, s18                                      // 000000003b88: 96041204
	v_add3_u32 v51, s3, s2, v51                                // 000000003b8c: d6550033 04cc0403
	v_add_co_u32 v48, vcc_lo, v50, s18                         // 000000003b94: d7006a30 02002532
	v_add_co_u32 v52, s0, v50, s16                             // 000000003b9c: d7000034 02002132
	s_wait_alu depctr_va_vcc(0)                                // 000000003ba4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003ba8: bf870003
	v_add_co_ci_u32_e64 v49, null, s19, v51, vcc_lo            // 000000003bac: d5207c31 01aa6613
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb4: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s17, v51, s0                // 000000003bb8: d5207c35 00026611
	v_add_co_u32 v56, s0, v48, s18                             // 000000003bc0: d7000038 02002530
	s_wait_alu depctr_va_sdst(0)                               // 000000003bc8: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s19, v49, s0                // 000000003bcc: d5207c39 00026213
	s_clause 0x1                                               // 000000003bd4: bf850001
	global_load_u8 v76, v[50:51], off                          // 000000003bd8: ee04007c 0000004c 00000032
	global_load_u8 v78, v[52:53], off                          // 000000003be4: ee04007c 0000004e 00000034
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bf0: bf88ff9e
	v_add3_u32 v55, s5, s4, v55                                // 000000003bf4: d6550037 04dc0805
	s_clause 0x1                                               // 000000003bfc: bf850001
	global_load_u8 v80, v[56:57], off                          // 000000003c00: ee04007c 00000050 00000038
	global_load_u8 v77, v[48:49], off                          // 000000003c0c: ee04007c 0000004d 00000030
	v_add_co_u32 v50, vcc_lo, v52, s18                         // 000000003c18: d7006a32 02002534
	s_wait_alu depctr_va_vcc(0)                                // 000000003c20: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s19, v53, vcc_lo            // 000000003c24: d5207c33 01aa6a13
	v_add_co_u32 v52, s0, v56, s18                             // 000000003c2c: d7000034 02002538
	s_delay_alu instid0(valu_dep_3)                            // 000000003c34: bf870003
	v_add_co_u32 v48, vcc_lo, v50, s18                         // 000000003c38: d7006a30 02002532
	s_wait_alu depctr_va_sdst(0)                               // 000000003c40: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s19, v57, s0                // 000000003c44: d5207c35 00027213
	s_wait_alu depctr_va_vcc(0)                                // 000000003c4c: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s19, v51, vcc_lo            // 000000003c50: d5207c31 01aa6613
	v_add_co_u32 v56, vcc_lo, v48, s18                         // 000000003c58: d7006a38 02002530
	s_clause 0x2                                               // 000000003c60: bf850002
	global_load_u8 v88, v[50:51], off                          // 000000003c64: ee04007c 00000058 00000032
	global_load_u8 v89, v[52:53], off                          // 000000003c70: ee04007c 00000059 00000034
	global_load_u8 v90, v[48:49], off                          // 000000003c7c: ee04007c 0000005a 00000030
	v_add_co_u32 v50, s0, v52, s18                             // 000000003c88: d7000032 02002534
	s_wait_alu depctr_va_sdst(0)                               // 000000003c90: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s19, v53, s0                // 000000003c94: d5207c33 00026a13
	s_wait_alu depctr_va_vcc(0)                                // 000000003c9c: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s19, v49, vcc_lo            // 000000003ca0: d5207c39 01aa6213
	v_add_co_u32 v52, vcc_lo, v56, s18                         // 000000003ca8: d7006a34 02002538
	v_add_co_u32 v48, s0, v50, s18                             // 000000003cb0: d7000030 02002532
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb8: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s19, v51, s0                // 000000003cbc: d5207c31 00026613
	s_wait_alu depctr_va_vcc(0)                                // 000000003cc4: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s19, v57, vcc_lo            // 000000003cc8: d5207c35 01aa7213
	s_clause 0x3                                               // 000000003cd0: bf850003
	global_load_u8 v91, v[50:51], off                          // 000000003cd4: ee04007c 0000005b 00000032
	global_load_u8 v92, v[56:57], off                          // 000000003ce0: ee04007c 0000005c 00000038
	global_load_u8 v94, v[52:53], off                          // 000000003cec: ee04007c 0000005e 00000034
	global_load_u8 v93, v[48:49], off                          // 000000003cf8: ee04007c 0000005d 00000030
	v_add_co_u32 v50, vcc_lo, v52, s18                         // 000000003d04: d7006a32 02002534
	s_wait_alu depctr_va_vcc(0)                                // 000000003d0c: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s19, v53, vcc_lo            // 000000003d10: d5207c33 01aa6a13
	global_load_u8 v96, v[50:51], off                          // 000000003d18: ee04007c 00000060 00000032
	v_add_co_u32 v56, s0, v48, s18                             // 000000003d24: d7000038 02002530
	s_wait_alu depctr_va_sdst(0)                               // 000000003d2c: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s19, v49, s0                // 000000003d30: d5207c39 00026213
	v_add_co_u32 v48, vcc_lo, v50, s18                         // 000000003d38: d7006a30 02002532
	s_wait_alu depctr_va_vcc(0)                                // 000000003d40: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s19, v51, vcc_lo            // 000000003d44: d5207c31 01aa6613
	v_add_co_u32 v52, s0, v56, s18                             // 000000003d4c: d7000034 02002538
	s_wait_alu depctr_va_sdst(0)                               // 000000003d54: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s19, v57, s0                // 000000003d58: d5207c35 00027213
	v_add_co_u32 v50, vcc_lo, v48, s18                         // 000000003d60: d7006a32 02002530
	s_wait_alu depctr_va_vcc(0)                                // 000000003d68: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s19, v49, vcc_lo            // 000000003d6c: d5207c33 01aa6213
	s_clause 0x1                                               // 000000003d74: bf850001
	global_load_u8 v52, v[52:53], off                          // 000000003d78: ee04007c 00000034 00000034
	global_load_u8 v48, v[48:49], off                          // 000000003d84: ee04007c 00000030 00000030
	global_load_u8 v49, v[50:51], off                          // 000000003d90: ee04007c 00000031 00000032
	global_load_u8 v97, v[54:55], off                          // 000000003d9c: ee04007c 00000061 00000036
	global_load_u8 v95, v[56:57], off                          // 000000003da8: ee04007c 0000005f 00000038
	s_wait_loadcnt 0x18                                        // 000000003db4: bfc00018
	v_cmp_ne_u32_e32 vcc_lo, 0, v79                            // 000000003db8: 7c9a9e80
	v_lshlrev_b32_e32 v50, 23, v79                             // 000000003dbc: 30649e97
	v_cmp_ne_u32_e64 s7, 0xff, v79                             // 000000003dc0: d44d0007 02029eff 000000ff
	s_wait_loadcnt 0x17                                        // 000000003dcc: bfc00017
	v_cmp_ne_u32_e64 s0, 0, v81                                // 000000003dd0: d44d0000 0202a280
	v_cmp_ne_u32_e64 s8, 0xff, v81                             // 000000003dd8: d44d0008 0202a2ff 000000ff
	s_wait_loadcnt 0x16                                        // 000000003de4: bfc00016
	v_cmp_ne_u32_e64 s1, 0, v82                                // 000000003de8: d44d0001 0202a480
	v_cvt_f64_f32_e32 v[56:57], v50                            // 000000003df0: 7e702132
	v_cmp_ne_u32_e64 s9, 0xff, v82                             // 000000003df4: d44d0009 0202a4ff 000000ff
	s_wait_loadcnt 0x15                                        // 000000003e00: bfc00015
	v_lshlrev_b32_e32 v54, 23, v83                             // 000000003e04: 306ca697
	s_wait_loadcnt 0x13                                        // 000000003e08: bfc00013
	v_lshlrev_b32_e32 v66, 23, v85                             // 000000003e0c: 3084aa97
	s_wait_loadcnt 0x12                                        // 000000003e10: bfc00012
	v_lshlrev_b32_e32 v68, 23, v86                             // 000000003e14: 3088ac97
	s_wait_loadcnt 0x11                                        // 000000003e18: bfc00011
	v_lshlrev_b32_e32 v74, 23, v87                             // 000000003e1c: 3094ae97
	v_cmp_ne_u32_e64 s2, 0, v83                                // 000000003e20: d44d0002 0202a680
	v_cvt_f64_f32_e32 v[62:63], v54                            // 000000003e28: 7e7c2136
	v_cvt_f64_f32_e32 v[66:67], v66                            // 000000003e2c: 7e842142
	v_cvt_f64_f32_e32 v[68:69], v68                            // 000000003e30: 7e882144
	v_cvt_f64_f32_e32 v[74:75], v74                            // 000000003e34: 7e94214a
	v_cmp_ne_u32_e64 s10, 0xff, v83                            // 000000003e38: d44d000a 0202a6ff 000000ff
	v_cmp_ne_u32_e64 s3, 0, v84                                // 000000003e44: d44d0003 0202a880
	v_cmp_ne_u32_e64 s11, 0xff, v84                            // 000000003e4c: d44d000b 0202a8ff 000000ff
	v_cmp_ne_u32_e64 s4, 0, v85                                // 000000003e58: d44d0004 0202aa80
	v_cmp_ne_u32_e64 s12, 0xff, v85                            // 000000003e60: d44d000c 0202aaff 000000ff
	v_cmp_ne_u32_e64 s5, 0, v86                                // 000000003e6c: d44d0005 0202ac80
	v_cmp_ne_u32_e64 s6, 0, v87                                // 000000003e74: d44d0006 0202ae80
	v_cmp_ne_u32_e64 s13, 0xff, v86                            // 000000003e7c: d44d000d 0202acff 000000ff
	v_cmp_ne_u32_e64 s14, 0xff, v87                            // 000000003e88: d44d000e 0202aeff 000000ff
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_cndmask_b32_e32 v57, 0x38000000, v57, vcc_lo             // 000000003e98: 027272ff 38000000
	s_and_b32 vcc_lo, s7, vcc_lo                               // 000000003ea0: 8b6a6a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ea4: bf88ff9e
	v_dual_cndmask_b32 v56, 0, v56 :: v_dual_lshlrev_b32 v51, 23, v81// 000000003ea8: ca627080 3832a297
	s_and_b32 vcc_lo, s8, s0                                   // 000000003eb0: 8b6a0008
	v_cndmask_b32_e64 v57, 0x7ff80000, v57, s7                 // 000000003eb4: d5010039 001e72ff 7ff80000
	s_delay_alu instid0(valu_dep_2)                            // 000000003ec0: bf870002
	v_cvt_f64_f32_e32 v[58:59], v51                            // 000000003ec4: 7e742133
	v_lshlrev_b32_e32 v55, 23, v84                             // 000000003ec8: 306ea897
	s_wait_loadcnt 0xd                                         // 000000003ecc: bfc0000d
	v_perm_b32 v51, v76, v77, 0xc0c0004                        // 000000003ed0: d6440033 03fe9b4c 0c0c0004
	v_cndmask_b32_e64 v63, 0x38000000, v63, s2                 // 000000003edc: d501003f 000a7eff 38000000
	v_cndmask_b32_e64 v67, 0x38000000, v67, s4                 // 000000003ee8: d5010043 001286ff 38000000
	v_cndmask_b32_e64 v69, 0x38000000, v69, s5                 // 000000003ef4: d5010045 00168aff 38000000
	v_cndmask_b32_e64 v75, 0x38000000, v75, s6                 // 000000003f00: d501004b 001a96ff 38000000
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003f0c: bf870214
	v_cndmask_b32_e64 v63, 0x7ff80000, v63, s10                // 000000003f10: d501003f 002a7eff 7ff80000
	v_cndmask_b32_e64 v67, 0x7ff80000, v67, s12                // 000000003f1c: d5010043 003286ff 7ff80000
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003f28: bf870214
	v_cndmask_b32_e64 v69, 0x7ff80000, v69, s13                // 000000003f2c: d5010045 00368aff 7ff80000
	v_cndmask_b32_e64 v75, 0x7ff80000, v75, s14                // 000000003f38: d501004b 003a96ff 7ff80000
	s_wait_loadcnt 0x8                                         // 000000003f44: bfc00008
	v_perm_b32 v54, v90, v92, 0xc0c0004                        // 000000003f48: d6440036 03feb95a 0c0c0004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f54: bf88ff9e
	v_cndmask_b32_e32 v58, 0, v58, vcc_lo                      // 000000003f58: 02747480
	s_and_b32 vcc_lo, s9, s1                                   // 000000003f5c: 8b6a0109
	v_lshlrev_b32_e32 v53, 23, v82                             // 000000003f60: 306aa497
	v_cvt_f64_f32_e32 v[64:65], v55                            // 000000003f64: 7e802137
	v_cndmask_b32_e64 v59, 0x38000000, v59, s0                 // 000000003f68: d501003b 000276ff 38000000
	s_wait_loadcnt 0x2                                         // 000000003f74: bfc00002
	v_perm_b32 v81, v48, v49, 0xc0c0004                        // 000000003f78: d6440051 03fe6330 0c0c0004
	s_wait_loadcnt 0x1                                         // 000000003f84: bfc00001
	v_lshlrev_b32_e32 v50, 23, v97                             // 000000003f88: 3064c297
	v_cvt_f64_f32_e32 v[60:61], v53                            // 000000003f8c: 7e782135
	v_perm_b32 v53, v80, v89, 0xc0c0004                        // 000000003f90: d6440035 03feb350 0c0c0004
	v_cndmask_b32_e64 v59, 0x7ff80000, v59, s8                 // 000000003f9c: d501003b 002276ff 7ff80000
	v_cmp_ne_u32_e64 s0, 0xff, v97                             // 000000003fa8: d44d0000 0202c2ff 000000ff
	v_cvt_f64_f32_e32 v[76:77], v50                            // 000000003fb4: 7e982132
	v_perm_b32 v50, v78, v88, 0xc0c0004                        // 000000003fb8: d6440032 03feb14e 0c0c0004
	v_lshl_or_b32 v78, v53, 16, v51                            // 000000003fc4: d656004e 04cd2135
	v_perm_b32 v51, v91, v93, 0xc0c0004                        // 000000003fcc: d6440033 03febb5b 0c0c0004
	v_perm_b32 v88, v94, v96, 0xc0c0004                        // 000000003fd8: d6440058 03fec15e 0c0c0004
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 000000003fe4: bf8701b4
	v_lshl_or_b32 v80, v54, 16, v50                            // 000000003fe8: d6560050 04c92136
	s_wait_loadcnt 0x0                                         // 000000003ff0: bfc00000
	v_perm_b32 v50, v95, v52, 0xc0c0004                        // 000000003ff4: d6440032 03fe695f 0c0c0004
	v_lshl_or_b32 v81, v81, 16, v88                            // 000000004000: d6560051 05612151
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000004008: bf870092
	v_lshl_or_b32 v79, v50, 16, v51                            // 00000000400c: d656004f 04cd2132
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[70:71], v[78:79], 0// 000000004014: cc464030 1a029d46
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000401c: bf870091
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[72:73], v[80:81], v[48:55]// 000000004020: cc464030 1cc2a148
	v_cvt_f64_f32_e32 v[70:71], v48                            // 000000004028: 7e8c2130
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_3)// 00000000402c: bf870192
	v_cvt_f64_f32_e32 v[48:49], v49                            // 000000004030: 7e602131
	v_cvt_f64_f32_e32 v[72:73], v50                            // 000000004034: 7e902132
	s_delay_alu instid0(valu_dep_4)                            // 000000004038: bf870004
	v_cvt_f64_f32_e32 v[50:51], v51                            // 00000000403c: 7e642133
	v_cvt_f64_f32_e32 v[78:79], v52                            // 000000004040: 7e9c2134
	v_cvt_f64_f32_e32 v[52:53], v53                            // 000000004044: 7e682135
	v_cvt_f64_f32_e32 v[80:81], v54                            // 000000004048: 7ea02136
	v_cvt_f64_f32_e32 v[54:55], v55                            // 00000000404c: 7e6c2137
	s_wait_alu depctr_sa_sdst(0)                               // 000000004050: bf88ff9e
	v_cndmask_b32_e32 v60, 0, v60, vcc_lo                      // 000000004054: 02787880
	s_and_b32 vcc_lo, s10, s2                                  // 000000004058: 8b6a020a
	v_cndmask_b32_e64 v61, 0x38000000, v61, s1                 // 00000000405c: d501003d 00067aff 38000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004068: bf88ff9e
	v_cndmask_b32_e32 v62, 0, v62, vcc_lo                      // 00000000406c: 027c7c80
	s_and_b32 vcc_lo, s11, s3                                  // 000000004070: 8b6a030b
	v_cndmask_b32_e64 v65, 0x38000000, v65, s3                 // 000000004074: d5010041 000e82ff 38000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004080: bf88ff9e
	v_cndmask_b32_e32 v64, 0, v64, vcc_lo                      // 000000004084: 02808080
	s_and_b32 vcc_lo, s12, s4                                  // 000000004088: 8b6a040c
	v_cndmask_b32_e64 v61, 0x7ff80000, v61, s9                 // 00000000408c: d501003d 00267aff 7ff80000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004098: bf88ff9e
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo                      // 00000000409c: 02848480
	s_and_b32 vcc_lo, s13, s5                                  // 0000000040a0: 8b6a050d
	v_cndmask_b32_e64 v65, 0x7ff80000, v65, s11                // 0000000040a4: d5010041 002e82ff 7ff80000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040b0: bf88ff9e
	v_cndmask_b32_e32 v68, 0, v68, vcc_lo                      // 0000000040b4: 02888880
	s_and_b32 vcc_lo, s14, s6                                  // 0000000040b8: 8b6a060e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040bc: bf88ff9e
	v_cndmask_b32_e32 v74, 0, v74, vcc_lo                      // 0000000040c0: 02949480
	v_cmp_ne_u32_e32 vcc_lo, 0, v97                            // 0000000040c4: 7c9ac280
	v_mul_f64_e32 v[56:57], v[56:57], v[70:71]                 // 0000000040c8: 0c708d38
	v_mul_f64_e32 v[48:49], v[58:59], v[48:49]                 // 0000000040cc: 0c60613a
	v_mul_f64_e32 v[58:59], v[60:61], v[72:73]                 // 0000000040d0: 0c74913c
	v_mul_f64_e32 v[50:51], v[62:63], v[50:51]                 // 0000000040d4: 0c64653e
	v_mul_f64_e32 v[60:61], v[64:65], v[78:79]                 // 0000000040d8: 0c789d40
	v_mul_f64_e32 v[52:53], v[66:67], v[52:53]                 // 0000000040dc: 0c686942
	v_mul_f64_e32 v[62:63], v[68:69], v[80:81]                 // 0000000040e0: 0c7ca144
	v_mul_f64_e32 v[54:55], v[74:75], v[54:55]                 // 0000000040e4: 0c6c6d4a
	s_wait_alu depctr_va_vcc(0)                                // 0000000040e8: bf88ff9d
	v_cndmask_b32_e32 v64, 0x38000000, v77, vcc_lo             // 0000000040ec: 02809aff 38000000
	s_and_b32 vcc_lo, s0, vcc_lo                               // 0000000040f4: 8b6a6a00
	s_delay_alu instid0(valu_dep_1)                            // 0000000040f8: bf870001
	v_cndmask_b32_e64 v65, 0x7ff80000, v64, s0                 // 0000000040fc: d5010041 000280ff 7ff80000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004108: bf88ff9e
	v_cndmask_b32_e32 v64, 0, v76, vcc_lo                      // 00000000410c: 02809880
	v_cmp_lt_i64_e64 s0, s[24:25], s[22:23]                    // 000000004110: d4510000 02002c18
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000004118: 8b6a007e
	v_mul_f64_e32 v[56:57], v[56:57], v[64:65]                 // 00000000411c: 0c708138
	v_mul_f64_e32 v[48:49], v[64:65], v[48:49]                 // 000000004120: 0c606140
	v_mul_f64_e32 v[58:59], v[64:65], v[58:59]                 // 000000004124: 0c747540
	v_mul_f64_e32 v[50:51], v[64:65], v[50:51]                 // 000000004128: 0c646540
	v_mul_f64_e32 v[60:61], v[64:65], v[60:61]                 // 00000000412c: 0c787940
	v_mul_f64_e32 v[52:53], v[64:65], v[52:53]                 // 000000004130: 0c686940
	v_mul_f64_e32 v[62:63], v[64:65], v[62:63]                 // 000000004134: 0c7c7d40
	v_mul_f64_e32 v[54:55], v[64:65], v[54:55]                 // 000000004138: 0c6c6d40
	v_cvt_f32_f64_e32 v56, v[56:57]                            // 00000000413c: 7e701f38
	v_cvt_f32_f64_e32 v48, v[48:49]                            // 000000004140: 7e601f30
	v_cvt_f32_f64_e32 v49, v[58:59]                            // 000000004144: 7e621f3a
	v_cvt_f32_f64_e32 v50, v[50:51]                            // 000000004148: 7e641f32
	v_cvt_f32_f64_e32 v51, v[60:61]                            // 00000000414c: 7e661f3c
	v_cvt_f32_f64_e32 v52, v[52:53]                            // 000000004150: 7e681f34
	v_cvt_f32_f64_e32 v53, v[62:63]                            // 000000004154: 7e6a1f3e
	v_cvt_f32_f64_e32 v54, v[54:55]                            // 000000004158: 7e6c1f36
	v_add_f32_e32 v40, v40, v56                                // 00000000415c: 06507128
	v_dual_add_f32 v47, v47, v48 :: v_dual_add_f32 v46, v46, v49// 000000004160: c908612f 2f2e632e
	v_add_f32_e32 v45, v45, v50                                // 000000004168: 065a652d
	v_dual_add_f32 v43, v43, v51 :: v_dual_add_f32 v42, v42, v52// 00000000416c: c908672b 2b2a692a
	v_dual_add_f32 v41, v41, v53 :: v_dual_add_f32 v10, v10, v54// 000000004174: c9086b29 290a6d0a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000417c: bf88ff9e
	s_cbranch_vccnz 65063                                      // 000000004180: bfa4fe27 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x1f20>
	v_mul_lo_u32 v13, s19, v14                                 // 000000004184: d72c000d 02021c13
	v_mul_lo_u32 v15, s18, v15                                 // 00000000418c: d72c000f 02021e12
	v_mad_co_u64_u32 v[0:1], null, s18, v14, 0                 // 000000004194: d6fe7c00 02021c12
	v_mul_lo_u32 v22, s19, v20                                 // 00000000419c: d72c0016 02022813
	v_mul_lo_u32 v23, s18, v21                                 // 0000000041a4: d72c0017 02022a12
	v_or_b32_e32 v25, 0x400000, v40                            // 0000000041ac: 383250ff 00400000
	v_bfe_u32 v24, v47, 16, 1                                  // 0000000041b4: d6100018 0205212f
	v_mul_lo_u32 v17, s18, v17                                 // 0000000041bc: d72c0011 02022212
	v_mul_lo_u32 v19, s18, v19                                 // 0000000041c4: d72c0013 02022612
	s_mov_b32 s0, -1                                           // 0000000041cc: be8000c1
	v_add3_u32 v1, v1, v15, v13                                // 0000000041d0: d6550001 04361f01
	v_mad_co_u64_u32 v[13:14], null, s18, v20, 0               // 0000000041d8: d6fe7c0d 02022812
	v_bfe_u32 v15, v40, 16, 1                                  // 0000000041e0: d610000f 02052128
	v_lshlrev_b64_e32 v[20:21], 1, v[11:12]                    // 0000000041e8: 3e281681
	v_add3_u32 v24, v24, v47, 0x7fff                           // 0000000041ec: d6550018 03fe5f18 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000041f8: 3e000081
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 0000000041fc: bf870234
	v_add3_u32 v15, v15, v40, 0x7fff                           // 000000004200: d655000f 03fe510f 00007fff
	v_add3_u32 v14, v14, v23, v22                              // 00000000420c: d655000e 045a2f0e
	v_or_b32_e32 v23, 0x400000, v47                            // 000000004214: 382e5eff 00400000
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 00000000421c: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000004224: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 000000004228: d5207c01 01aa0215
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000004230: 7c305128
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000004234: 3e1a1a81
	s_wait_alu depctr_va_vcc(0)                                // 000000004238: bf88ff9d
	v_cndmask_b32_e32 v22, v15, v25, vcc_lo                    // 00000000423c: 022c330f
	v_mul_lo_u32 v25, s19, v16                                 // 000000004240: d72c0019 02022013
	v_mad_co_u64_u32 v[15:16], null, s18, v16, 0               // 000000004248: d6fe7c0f 02022012
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 000000004250: d7006a00 02022900
	s_wait_alu depctr_va_vcc(0)                                // 000000004258: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 00000000425c: d5207c01 01aa2b01
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000004264: 7c305f2f
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000004268: ee09407c 0b000000 00000000
	v_add3_u32 v16, v16, v17, v25                              // 000000004274: d6550010 04662310
	s_wait_alu depctr_va_vcc(0)                                // 00000000427c: bf88ff9d
	v_cndmask_b32_e32 v22, v24, v23, vcc_lo                    // 000000004280: 022c2f18
	v_add_co_u32 v0, vcc_lo, s20, v13                          // 000000004284: d7006a00 02021a14
	v_bfe_u32 v13, v46, 16, 1                                  // 00000000428c: d610000d 0205212e
	s_wait_alu depctr_va_vcc(0)                                // 000000004294: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v14, vcc_lo             // 000000004298: d5207c01 01aa1c15
	v_mul_lo_u32 v24, s19, v18                                 // 0000000042a0: d72c0018 02022413
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 0000000042a8: d7006a00 02022900
	v_add3_u32 v17, v13, v46, 0x7fff                           // 0000000042b0: d6550011 03fe5d0d 00007fff
	v_lshlrev_b64_e32 v[13:14], 1, v[15:16]                    // 0000000042bc: 3e1a1e81
	v_mad_co_u64_u32 v[15:16], null, s18, v18, 0               // 0000000042c0: d6fe7c0f 02022412
	s_wait_alu depctr_va_vcc(0)                                // 0000000042c8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 0000000042cc: d5207c01 01aa2b01
	v_or_b32_e32 v23, 0x400000, v46                            // 0000000042d4: 382e5cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 0000000042dc: 7c305d2e
	global_store_d16_hi_b16 v[0:1], v22, off                   // 0000000042e0: ee09407c 0b000000 00000000
	v_add3_u32 v16, v16, v19, v24                              // 0000000042ec: d6550010 04622710
	s_wait_alu depctr_va_vcc(0)                                // 0000000042f4: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v23, vcc_lo                    // 0000000042f8: 02222f11
	v_add_co_u32 v0, vcc_lo, s20, v13                          // 0000000042fc: d7006a00 02021a14
	v_bfe_u32 v13, v45, 16, 1                                  // 000000004304: d610000d 0205212d
	s_wait_alu depctr_va_vcc(0)                                // 00000000430c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v14, vcc_lo             // 000000004310: d5207c01 01aa1c15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004318: bf870193
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 00000000431c: d7006a00 02022900
	v_add3_u32 v18, v13, v45, 0x7fff                           // 000000004324: d6550012 03fe5b0d 00007fff
	v_lshlrev_b64_e32 v[13:14], 1, v[15:16]                    // 000000004330: 3e1a1e81
	v_mul_lo_u32 v15, s19, v6                                  // 000000004334: d72c000f 02020c13
	v_mul_lo_u32 v16, s18, v7                                  // 00000000433c: d72c0010 02020e12
	v_mad_co_u64_u32 v[6:7], null, s18, v6, 0                  // 000000004344: d6fe7c06 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 00000000434c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 000000004350: d5207c01 01aa2b01
	v_or_b32_e32 v19, 0x400000, v45                            // 000000004358: 38265aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000004360: 7c305b2d
	global_store_d16_hi_b16 v[0:1], v17, off                   // 000000004364: ee09407c 08800000 00000000
	v_add3_u32 v7, v7, v16, v15                                // 000000004370: d6550007 043e2107
	s_wait_alu depctr_va_vcc(0)                                // 000000004378: bf88ff9d
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo                    // 00000000437c: 02222712
	v_add_co_u32 v0, vcc_lo, s20, v13                          // 000000004380: d7006a00 02021a14
	s_wait_alu depctr_va_vcc(0)                                // 000000004388: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v14, vcc_lo             // 00000000438c: d5207c01 01aa1c15
	v_bfe_u32 v13, v43, 16, 1                                  // 000000004394: d610000d 0205212b
	s_delay_alu instid0(valu_dep_3)                            // 00000000439c: bf870003
	v_add_co_u32 v0, vcc_lo, v0, v20                           // 0000000043a0: d7006a00 02022900
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 0000000043a8: 3e0c0c81
	v_mul_lo_u32 v15, s19, v8                                  // 0000000043ac: d72c000f 02021013
	v_mul_lo_u32 v16, s18, v9                                  // 0000000043b4: d72c0010 02021212
	v_mad_co_u64_u32 v[8:9], null, s18, v8, 0                  // 0000000043bc: d6fe7c08 02021012
	s_wait_alu depctr_va_vcc(0)                                // 0000000043c4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v21, vcc_lo              // 0000000043c8: d5207c01 01aa2b01
	v_add3_u32 v13, v13, v43, 0x7fff                           // 0000000043d0: d655000d 03fe570d 00007fff
	v_or_b32_e32 v14, 0x400000, v43                            // 0000000043dc: 381c56ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 0000000043e4: 7c30572b
	global_store_d16_hi_b16 v[0:1], v17, off                   // 0000000043e8: ee09407c 08800000 00000000
	v_mul_lo_u32 v18, s18, v3                                  // 0000000043f4: d72c0012 02020612
	v_add3_u32 v9, v9, v16, v15                                // 0000000043fc: d6550009 043e2109
	v_or_b32_e32 v15, 0x400000, v42                            // 000000004404: 381e54ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000440c: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v14, vcc_lo                    // 000000004410: 021a1d0d
	v_add_co_u32 v0, vcc_lo, s20, v6                           // 000000004414: d7006a00 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 00000000441c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v7, vcc_lo              // 000000004420: d5207c01 01aa0e15
	v_bfe_u32 v14, v42, 16, 1                                  // 000000004428: d610000e 0205212a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004430: bf8701a3
	v_add_co_u32 v6, vcc_lo, v0, v20                           // 000000004434: d7006a06 02022900
	s_wait_alu depctr_va_vcc(0)                                // 00000000443c: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v1, v21, vcc_lo              // 000000004440: d5207c07 01aa2b01
	v_lshlrev_b64_e32 v[0:1], 1, v[8:9]                        // 000000004448: 3e001081
	v_mul_lo_u32 v8, s19, v4                                   // 00000000444c: d72c0008 02020813
	v_mul_lo_u32 v9, s18, v5                                   // 000000004454: d72c0009 02020a12
	v_mad_co_u64_u32 v[4:5], null, s18, v4, 0                  // 00000000445c: d6fe7c04 02020812
	v_add3_u32 v14, v14, v42, 0x7fff                           // 000000004464: d655000e 03fe550e 00007fff
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000004470: 7c30552a
	s_wait_alu depctr_va_vcc(0)                                // 000000004474: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000004478: bf870002
	v_cndmask_b32_e32 v14, v14, v15, vcc_lo                    // 00000000447c: 021c1f0e
	v_bfe_u32 v15, v41, 16, 1                                  // 000000004480: d610000f 02052129
	v_add_co_u32 v16, vcc_lo, s20, v0                          // 000000004488: d7006a10 02020014
	v_add3_u32 v5, v5, v9, v8                                  // 000000004490: d6550005 04221305
	s_wait_alu depctr_va_vcc(0)                                // 000000004498: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s21, v1, vcc_lo             // 00000000449c: d5207c11 01aa0215
	v_add3_u32 v8, v15, v41, 0x7fff                            // 0000000044a4: d6550008 03fe530f 00007fff
	v_mul_lo_u32 v15, s19, v2                                  // 0000000044b0: d72c000f 02020413
	v_mad_co_u64_u32 v[0:1], null, s18, v2, 0                  // 0000000044b8: d6fe7c00 02020412
	v_lshlrev_b64_e32 v[2:3], 1, v[4:5]                        // 0000000044c0: 3e040881
	v_add_co_u32 v4, vcc_lo, v16, v20                          // 0000000044c4: d7006a04 02022910
	v_or_b32_e32 v9, 0x400000, v41                             // 0000000044cc: 381252ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000044d4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v17, v21, vcc_lo             // 0000000044d8: d5207c05 01aa2b11
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 0000000044e0: 7c305329
	v_add3_u32 v1, v1, v18, v15                                // 0000000044e4: d6550001 043e2501
	s_clause 0x1                                               // 0000000044ec: bf850001
	global_store_d16_hi_b16 v[6:7], v13, off                   // 0000000044f0: ee09407c 06800000 00000006
	global_store_d16_hi_b16 v[4:5], v14, off                   // 0000000044fc: ee09407c 07000000 00000004
	s_wait_alu depctr_va_vcc(0)                                // 000000004508: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo                       // 00000000450c: 02101308
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 000000004510: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000004518: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 00000000451c: d5207c03 01aa0615
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004524: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004528: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v20                           // 00000000452c: d7006a02 02022902
	s_wait_alu depctr_va_vcc(0)                                // 000000004534: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v21, vcc_lo              // 000000004538: d5207c03 01aa2b03
	s_delay_alu instid0(valu_dep_3)                            // 000000004540: bf870003
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 000000004544: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 00000000454c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 000000004550: d5207c01 01aa0215
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000004558: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004564: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004568: be812000
	s_cbranch_execnz 3                                         // 00000000456c: bfa60003 <tessera_rocm_scaled_matmul_5f09ec8682bcb143+0x2a7c>
	s_nop 0                                                    // 000000004570: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000004574: bfb60003
	s_endpgm                                                   // 000000004578: bfb00000
	v_bfe_u32 v2, v10, 16, 1                                   // 00000000457c: d6100002 0205210a
	v_or_b32_e32 v4, 0x400000, v10                             // 000000004584: 380814ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v10, v10                           // 00000000458c: 7c30150a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 000000004590: bf870133
	v_add3_u32 v5, v2, v10, 0x7fff                             // 000000004594: d6550005 03fe1502 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[11:12]                      // 0000000045a0: 3e041681
	s_wait_alu depctr_va_vcc(0)                                // 0000000045a4: bf88ff9d
	v_cndmask_b32_e32 v4, v5, v4, vcc_lo                       // 0000000045a8: 02080905
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045ac: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000045b0: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000045b8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000045bc: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 0000000045c4: ee09407c 02000000 00000000
	s_nop 0                                                    // 0000000045d0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000045d4: bfb60003
	s_endpgm                                                   // 0000000045d8: bfb00000
	s_code_end                                                 // 0000000045dc: bf9f0000
	s_code_end                                                 // 0000000045e0: bf9f0000
	s_code_end                                                 // 0000000045e4: bf9f0000
	s_code_end                                                 // 0000000045e8: bf9f0000
	s_code_end                                                 // 0000000045ec: bf9f0000
	s_code_end                                                 // 0000000045f0: bf9f0000
	s_code_end                                                 // 0000000045f4: bf9f0000
	s_code_end                                                 // 0000000045f8: bf9f0000
	s_code_end                                                 // 0000000045fc: bf9f0000
	s_code_end                                                 // 000000004600: bf9f0000
	s_code_end                                                 // 000000004604: bf9f0000
	s_code_end                                                 // 000000004608: bf9f0000
	s_code_end                                                 // 00000000460c: bf9f0000
	s_code_end                                                 // 000000004610: bf9f0000
	s_code_end                                                 // 000000004614: bf9f0000
	s_code_end                                                 // 000000004618: bf9f0000
	s_code_end                                                 // 00000000461c: bf9f0000
	s_code_end                                                 // 000000004620: bf9f0000
	s_code_end                                                 // 000000004624: bf9f0000
	s_code_end                                                 // 000000004628: bf9f0000
	s_code_end                                                 // 00000000462c: bf9f0000
	s_code_end                                                 // 000000004630: bf9f0000
	s_code_end                                                 // 000000004634: bf9f0000
	s_code_end                                                 // 000000004638: bf9f0000
	s_code_end                                                 // 00000000463c: bf9f0000
	s_code_end                                                 // 000000004640: bf9f0000
	s_code_end                                                 // 000000004644: bf9f0000
	s_code_end                                                 // 000000004648: bf9f0000
	s_code_end                                                 // 00000000464c: bf9f0000
	s_code_end                                                 // 000000004650: bf9f0000
	s_code_end                                                 // 000000004654: bf9f0000
	s_code_end                                                 // 000000004658: bf9f0000
	s_code_end                                                 // 00000000465c: bf9f0000
	s_code_end                                                 // 000000004660: bf9f0000
	s_code_end                                                 // 000000004664: bf9f0000
	s_code_end                                                 // 000000004668: bf9f0000
	s_code_end                                                 // 00000000466c: bf9f0000
	s_code_end                                                 // 000000004670: bf9f0000
	s_code_end                                                 // 000000004674: bf9f0000
	s_code_end                                                 // 000000004678: bf9f0000
	s_code_end                                                 // 00000000467c: bf9f0000
	s_code_end                                                 // 000000004680: bf9f0000
	s_code_end                                                 // 000000004684: bf9f0000
	s_code_end                                                 // 000000004688: bf9f0000
	s_code_end                                                 // 00000000468c: bf9f0000
	s_code_end                                                 // 000000004690: bf9f0000
	s_code_end                                                 // 000000004694: bf9f0000
	s_code_end                                                 // 000000004698: bf9f0000
	s_code_end                                                 // 00000000469c: bf9f0000
	s_code_end                                                 // 0000000046a0: bf9f0000
	s_code_end                                                 // 0000000046a4: bf9f0000
	s_code_end                                                 // 0000000046a8: bf9f0000
	s_code_end                                                 // 0000000046ac: bf9f0000
	s_code_end                                                 // 0000000046b0: bf9f0000
	s_code_end                                                 // 0000000046b4: bf9f0000
	s_code_end                                                 // 0000000046b8: bf9f0000
	s_code_end                                                 // 0000000046bc: bf9f0000
	s_code_end                                                 // 0000000046c0: bf9f0000
	s_code_end                                                 // 0000000046c4: bf9f0000
	s_code_end                                                 // 0000000046c8: bf9f0000
	s_code_end                                                 // 0000000046cc: bf9f0000
	s_code_end                                                 // 0000000046d0: bf9f0000
	s_code_end                                                 // 0000000046d4: bf9f0000
	s_code_end                                                 // 0000000046d8: bf9f0000
	s_code_end                                                 // 0000000046dc: bf9f0000
	s_code_end                                                 // 0000000046e0: bf9f0000
	s_code_end                                                 // 0000000046e4: bf9f0000
	s_code_end                                                 // 0000000046e8: bf9f0000
	s_code_end                                                 // 0000000046ec: bf9f0000
	s_code_end                                                 // 0000000046f0: bf9f0000
	s_code_end                                                 // 0000000046f4: bf9f0000
	s_code_end                                                 // 0000000046f8: bf9f0000
	s_code_end                                                 // 0000000046fc: bf9f0000
	s_code_end                                                 // 000000004700: bf9f0000
	s_code_end                                                 // 000000004704: bf9f0000
	s_code_end                                                 // 000000004708: bf9f0000
	s_code_end                                                 // 00000000470c: bf9f0000
	s_code_end                                                 // 000000004710: bf9f0000
	s_code_end                                                 // 000000004714: bf9f0000
	s_code_end                                                 // 000000004718: bf9f0000
	s_code_end                                                 // 00000000471c: bf9f0000
	s_code_end                                                 // 000000004720: bf9f0000
	s_code_end                                                 // 000000004724: bf9f0000
	s_code_end                                                 // 000000004728: bf9f0000
	s_code_end                                                 // 00000000472c: bf9f0000
	s_code_end                                                 // 000000004730: bf9f0000
	s_code_end                                                 // 000000004734: bf9f0000
	s_code_end                                                 // 000000004738: bf9f0000
	s_code_end                                                 // 00000000473c: bf9f0000
	s_code_end                                                 // 000000004740: bf9f0000
	s_code_end                                                 // 000000004744: bf9f0000
	s_code_end                                                 // 000000004748: bf9f0000
	s_code_end                                                 // 00000000474c: bf9f0000
	s_code_end                                                 // 000000004750: bf9f0000
	s_code_end                                                 // 000000004754: bf9f0000
	s_code_end                                                 // 000000004758: bf9f0000
	s_code_end                                                 // 00000000475c: bf9f0000
	s_code_end                                                 // 000000004760: bf9f0000
	s_code_end                                                 // 000000004764: bf9f0000
	s_code_end                                                 // 000000004768: bf9f0000
	s_code_end                                                 // 00000000476c: bf9f0000
	s_code_end                                                 // 000000004770: bf9f0000
	s_code_end                                                 // 000000004774: bf9f0000
	s_code_end                                                 // 000000004778: bf9f0000
	s_code_end                                                 // 00000000477c: bf9f0000
