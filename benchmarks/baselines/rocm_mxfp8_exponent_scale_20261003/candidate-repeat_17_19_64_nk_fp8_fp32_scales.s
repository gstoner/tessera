
/tmp/tmpe74s_0m_.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_c6867cc7267a885c>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[36:39], s[0:1], 0xc8                         // 000000001b04: f4004900 f80000c8
	s_load_b64 s[40:41], s[0:1], 0xa8                          // 000000001b0c: f4002a00 f80000a8
	s_load_b64 s[50:51], s[0:1], 0xd8                          // 000000001b14: f4002c80 f80000d8
	s_load_b64 s[46:47], s[0:1], 0x8                           // 000000001b1c: f4002b80 f8000008
	s_load_b64 s[48:49], s[0:1], 0x30                          // 000000001b24: f4002c00 f8000030
	s_load_b64 s[42:43], s[0:1], 0x58                          // 000000001b2c: f4002a80 f8000058
	s_load_b64 s[56:57], s[0:1], 0x80                          // 000000001b34: f4002e00 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_mov_b32 s4, ttmp7                                        // 000000001b40: be840073
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b44: 86039f75
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[54:55], s[2:3], 4                             // 000000001b4c: 84b68402
	s_lshl_b64 s[52:53], s[4:5], 4                             // 000000001b50: 84b48404
	s_add_nc_u64 s[2:3], s[54:55], 16                          // 000000001b54: a9829036
	s_add_nc_u64 s[0:1], s[52:53], 16                          // 000000001b58: a9809034
	v_dual_mov_b32 v4, s55 :: v_dual_and_b32 v23, 15, v0       // 000000001b5c: ca240037 0416008f
	v_bfe_u32 v21, v0, 4, 1                                    // 000000001b64: d6100015 02050900
	s_delay_alu instid0(valu_dep_2)                            // 000000001b6c: bf870002
	v_or_b32_e32 v3, s54, v23                                  // 000000001b70: 38062e36
	s_wait_kmcnt 0x0                                           // 000000001b74: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[36:37]                      // 000000001b78: d4540000 02004800
	v_cmp_gt_i64_e64 s1, s[2:3], s[38:39]                      // 000000001b80: d4540001 02004c02
	v_cmp_lt_i64_e64 s35, s[50:51], 32                         // 000000001b88: d4510023 02014032
	s_and_b32 s44, s50, 0xffffffe0                             // 000000001b90: 8b2cff32 ffffffe0
	s_mov_b32 s45, s51                                         // 000000001b98: bead0033
	s_or_b32 s0, s0, s1                                        // 000000001b9c: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ba0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001ba4: 8b6a007e
	s_cbranch_vccz 10                                          // 000000001ba8: bfa3000a <tessera_rocm_scaled_matmul_c6867cc7267a885c+0xd4>
	s_and_b32 s0, s35, exec_lo                                 // 000000001bac: 8b007e23
	s_cselect_b32 s0, 1, 0                                     // 000000001bb0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bb8: bf078100
	s_cbranch_scc1 8                                           // 000000001bbc: bfa20008 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0xe0>
	v_lshl_or_b32 v5, v21, 3, s52                              // 000000001bc0: d6560005 00d10715
	v_mov_b32_e32 v6, s53                                      // 000000001bc8: 7e0c0235
	s_mov_b32 s0, 0                                            // 000000001bcc: be800080
	s_branch 4                                                 // 000000001bd0: bfa00004 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0xe4>
	s_mov_b32 s0, 0                                            // 000000001bd4: be800080
	s_cbranch_execnz 1538                                      // 000000001bd8: bfa60602 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x18e4>
	s_branch 2161                                              // 000000001bdc: bfa00871 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x22a4>
	s_mov_b32 s0, -1                                           // 000000001be0: be8000c1
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v2, 0              // 000000001be4: ca100080 13020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bec: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001bf0: 8b007e00
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0             // 000000001bf4: ca100080 0b0c0080
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v16, 0             // 000000001bfc: ca100080 0d100080
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v1, 0              // 000000001c04: ca100080 14000080
	s_cselect_b32 s0, 1, 0                                     // 000000001c0c: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c14: bf078100
	s_cbranch_scc1 1195                                        // 000000001c18: bfa204ab <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x13c8>
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v0, 3, v21      // 000000001c1c: ca220080 01002a83
	v_dual_mov_b32 v6, s53 :: v_dual_mov_b32 v15, s53          // 000000001c24: ca100035 060e0035
	s_lshr_b64 s[4:5], s[50:51], 5                             // 000000001c2c: 85848532
	s_delay_alu instid0(valu_dep_2)                            // 000000001c30: bf870002
	v_or_b32_e32 v5, s52, v0                                   // 000000001c34: 380a0034
	s_lshr_b32 s3, s51, 5                                      // 000000001c38: 85038533
	v_or_b32_e32 v11, s52, v23                                 // 000000001c3c: 38162e34
	v_mul_lo_u32 v2, s50, v4                                   // 000000001c40: d72c0002 02020832
	v_mul_lo_u32 v20, s51, v3                                  // 000000001c48: d72c0014 02020633
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[5:6]                  // 000000001c50: 7ca80a24
	v_mov_b32_e32 v12, s53                                     // 000000001c54: 7e180235
	v_or_b32_e32 v14, 1, v5                                    // 000000001c58: 381c0a81
	v_mul_lo_u32 v22, s51, v11                                 // 000000001c5c: d72c0016 02021633
	v_mad_co_u64_u32 v[7:8], null, s50, v11, v[0:1]            // 000000001c64: d6fe7c07 04021632
	v_cmp_gt_i64_e64 s0, s[38:39], v[3:4]                      // 000000001c6c: d4540000 02020626
	v_cndmask_b32_e32 v9, 0, v5, vcc_lo                        // 000000001c74: 02120a80
	v_cndmask_b32_e64 v10, 0, s53, vcc_lo                      // 000000001c78: d501000a 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[14:15]                // 000000001c80: 7ca81c24
	v_mov_b32_e32 v19, s53                                     // 000000001c84: 7e260235
	v_cmp_gt_i64_e64 s1, s[36:37], v[11:12]                    // 000000001c88: d4540001 02021624
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c90: bf88ff9e
	v_mul_lo_u32 v24, s3, v9                                   // 000000001c94: d72c0018 02021203
	v_mad_co_u64_u32 v[16:17], null, s4, v9, 0                 // 000000001c9c: d6fe7c10 02021204
	v_mul_lo_u32 v25, s4, v10                                  // 000000001ca4: d72c0019 02021404
	v_mad_co_u64_u32 v[9:10], null, s50, v3, v[0:1]            // 000000001cac: d6fe7c09 04020632
	v_or_b32_e32 v18, 2, v5                                    // 000000001cb4: 38240a82
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb8: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v14 :: v_dual_mov_b32 v32, s53  // 000000001cbc: ca501c80 0b200035
	s_mul_i32 s2, s50, s53                                     // 000000001cc4: 96023532
	v_cndmask_b32_e64 v13, 0, v4, s0                           // 000000001cc8: d501000d 00020880
	v_cndmask_b32_e64 v12, 0, v3, s0                           // 000000001cd0: d501000c 00020680
	v_add3_u32 v17, v17, v25, v24                              // 000000001cd8: d6550011 04623311
	v_add3_u32 v10, v20, v10, v2                               // 000000001ce0: d655000a 040a1514
	v_cndmask_b32_e64 v2, 0, s53, vcc_lo                       // 000000001ce8: d5010002 01a86a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cf0: bf88ff9e
	v_add3_u32 v8, v22, v8, s2                                 // 000000001cf4: d6550008 000a1116
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[18:19]                // 000000001cfc: 7ca82424
	v_mul_lo_u32 v22, s3, v11                                  // 000000001d00: d72c0016 02021603
	v_mad_co_u64_u32 v[19:20], null, s4, v11, 0                // 000000001d08: d6fe7c13 02021604
	v_mul_lo_u32 v2, s4, v2                                    // 000000001d10: d72c0002 02020404
	v_lshlrev_b64_e32 v[14:15], 2, v[16:17]                    // 000000001d18: 3e1c2082
	v_lshlrev_b64_e32 v[11:12], 2, v[12:13]                    // 000000001d1c: 3e161882
	v_or_b32_e32 v16, 3, v5                                    // 000000001d20: 38200a83
	v_mov_b32_e32 v17, s53                                     // 000000001d24: 7e220235
	s_wait_alu depctr_va_vcc(0)                                // 000000001d28: bf88ff9d
	v_cndmask_b32_e32 v13, 0, v18, vcc_lo                      // 000000001d2c: 021a2480
	v_cndmask_b32_e64 v18, 0, s53, vcc_lo                      // 000000001d30: d5010012 01a86a80
	v_add_co_u32 v14, s2, s42, v14                             // 000000001d38: d700020e 02021c2a
	v_add3_u32 v20, v20, v2, v22                               // 000000001d40: d6550014 045a0514
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[16:17]                // 000000001d48: 7ca82024
	s_wait_alu depctr_va_sdst(0)                               // 000000001d4c: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s43, v15, s2                // 000000001d50: d5207c0f 000a1e2b
	v_mul_lo_u32 v2, s3, v13                                   // 000000001d58: d72c0002 02021a03
	v_mul_lo_u32 v22, s4, v18                                  // 000000001d60: d72c0016 02022404
	v_mad_co_u64_u32 v[25:26], null, s4, v13, 0                // 000000001d68: d6fe7c19 02021a04
	v_add_co_u32 v17, s2, s56, v11                             // 000000001d70: d7000211 02021638
	s_wait_alu depctr_va_sdst(0)                               // 000000001d78: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s57, v12, s2                // 000000001d7c: d5207c12 000a1839
	v_lshlrev_b64_e32 v[11:12], 2, v[19:20]                    // 000000001d84: 3e162682
	v_or_b32_e32 v19, 4, v5                                    // 000000001d88: 38260a84
	s_wait_alu depctr_va_vcc(0)                                // 000000001d8c: bf88ff9d
	v_dual_mov_b32 v20, s53 :: v_dual_cndmask_b32 v13, 0, v16  // 000000001d90: ca120035 140c2080
	v_cndmask_b32_e64 v16, 0, s53, vcc_lo                      // 000000001d98: d5010010 01a86a80
	v_add3_u32 v26, v26, v22, v2                               // 000000001da0: d655001a 040a2d1a
	v_add_co_u32 v22, s2, s42, v11                             // 000000001da8: d7000216 0202162a
	s_delay_alu instid0(valu_dep_4)                            // 000000001db0: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[19:20]                // 000000001db4: 7ca82624
	v_mul_lo_u32 v2, s3, v13                                   // 000000001db8: d72c0002 02021a03
	v_mul_lo_u32 v16, s4, v16                                  // 000000001dc0: d72c0010 02022004
	v_mad_co_u64_u32 v[27:28], null, s4, v13, 0                // 000000001dc8: d6fe7c1b 02021a04
	s_wait_alu depctr_va_sdst(0)                               // 000000001dd0: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s43, v12, s2                // 000000001dd4: d5207c18 000a182b
	v_lshlrev_b64_e32 v[11:12], 2, v[25:26]                    // 000000001ddc: 3e163282
	s_wait_alu depctr_va_vcc(0)                                // 000000001de0: bf88ff9d
	v_cndmask_b32_e32 v13, 0, v19, vcc_lo                      // 000000001de4: 021a2680
	v_cndmask_b32_e64 v25, 0, s53, vcc_lo                      // 000000001de8: d5010019 01a86a80
	v_or_b32_e32 v19, 5, v5                                    // 000000001df0: 38260a85
	v_or_b32_e32 v31, 6, v5                                    // 000000001df4: 383e0a86
	v_add3_u32 v28, v28, v16, v2                               // 000000001df8: d655001c 040a211c
	v_mul_lo_u32 v2, s3, v13                                   // 000000001e00: d72c0002 02021a03
	v_mul_lo_u32 v16, s4, v25                                  // 000000001e08: d72c0010 02023204
	v_mad_co_u64_u32 v[29:30], null, s4, v13, 0                // 000000001e10: d6fe7c1d 02021a04
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[19:20]                // 000000001e18: 7ca82624
	v_add_co_u32 v25, s2, s42, v11                             // 000000001e1c: d7000219 0202162a
	s_wait_alu depctr_va_sdst(0)                               // 000000001e24: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s43, v12, s2                // 000000001e28: d5207c1a 000a182b
	v_cmp_gt_i64_e64 s2, s[36:37], v[31:32]                    // 000000001e30: d4540002 02023e24
	v_lshlrev_b64_e32 v[11:12], 2, v[27:28]                    // 000000001e38: 3e163682
	v_add3_u32 v30, v30, v16, v2                               // 000000001e3c: d655001e 040a211e
	s_wait_alu depctr_va_vcc(0)                                // 000000001e44: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v19, vcc_lo                       // 000000001e48: 02042680
	v_or_b32_e32 v19, 7, v5                                    // 000000001e4c: 38260a87
	v_cndmask_b32_e64 v13, 0, s53, vcc_lo                      // 000000001e50: d501000d 01a86a80
	s_wait_alu depctr_va_sdst(0)                               // 000000001e58: bf88f19f
	v_cndmask_b32_e64 v27, 0, v31, s2                          // 000000001e5c: d501001b 000a3e80
	v_cndmask_b32_e64 v28, 0, s53, s2                          // 000000001e64: d501001c 00086a80
	v_mul_lo_u32 v16, s3, v2                                   // 000000001e6c: d72c0010 02020403
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[19:20]                // 000000001e74: 7ca82624
	v_mul_lo_u32 v13, s4, v13                                  // 000000001e78: d72c000d 02021a04
	v_mad_co_u64_u32 v[31:32], null, s4, v2, 0                 // 000000001e80: d6fe7c1f 02020404
	v_mul_lo_u32 v2, s3, v27                                   // 000000001e88: d72c0002 02023603
	v_mul_lo_u32 v35, s4, v28                                  // 000000001e90: d72c0023 02023804
	v_mad_co_u64_u32 v[33:34], null, s4, v27, 0                // 000000001e98: d6fe7c21 02023604
	s_wait_alu depctr_va_vcc(0)                                // 000000001ea0: bf88ff9d
	v_cndmask_b32_e32 v19, 0, v19, vcc_lo                      // 000000001ea4: 02262680
	v_cndmask_b32_e64 v20, 0, s53, vcc_lo                      // 000000001ea8: d5010014 01a86a80
	v_add_co_u32 v27, vcc_lo, s42, v11                         // 000000001eb0: d7006a1b 0202162a
	v_add3_u32 v32, v32, v13, v16                              // 000000001eb8: d6550020 04421b20
	s_wait_alu depctr_va_vcc(0)                                // 000000001ec0: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s43, v12, vcc_lo            // 000000001ec4: d5207c1c 01aa182b
	v_mul_lo_u32 v16, s4, v20                                  // 000000001ecc: d72c0010 02022804
	v_add3_u32 v34, v34, v35, v2                               // 000000001ed4: d6550022 040a4722
	v_mov_b32_e32 v2, v1                                       // 000000001edc: 7e040301
	v_mul_lo_u32 v13, s3, v19                                  // 000000001ee0: d72c000d 02022603
	v_mad_co_u64_u32 v[19:20], null, s4, v19, 0                // 000000001ee8: d6fe7c13 02022604
	v_lshlrev_b64_e32 v[11:12], 2, v[29:30]                    // 000000001ef0: 3e163a82
	v_lshlrev_b64_e32 v[31:32], 2, v[31:32]                    // 000000001ef4: 3e3e3e82
	s_mov_b64 s[58:59], 0                                      // 000000001ef8: beba0180
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_4)// 000000001efc: bf870212
	v_add_co_u32 v29, vcc_lo, s42, v11                         // 000000001f00: d7006a1d 0202162a
	v_add3_u32 v20, v20, v16, v13                              // 000000001f08: d6550014 04362114
	s_wait_alu depctr_va_vcc(0)                                // 000000001f10: bf88ff9d
	v_add_co_ci_u32_e64 v30, null, s43, v12, vcc_lo            // 000000001f14: d5207c1e 01aa182b
	v_lshlrev_b64_e32 v[11:12], 2, v[33:34]                    // 000000001f1c: 3e164282
	v_add_co_u32 v31, vcc_lo, s42, v31                         // 000000001f20: d7006a1f 02023e2a
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]                    // 000000001f28: 3e262682
	s_wait_alu depctr_va_vcc(0)                                // 000000001f2c: bf88ff9d
	v_add_co_ci_u32_e64 v32, null, s43, v32, vcc_lo            // 000000001f30: d5207c20 01aa402b
	s_delay_alu instid0(valu_dep_4)                            // 000000001f38: bf870004
	v_add_co_u32 v33, vcc_lo, s42, v11                         // 000000001f3c: d7006a21 0202162a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f44: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s43, v12, vcc_lo            // 000000001f48: d5207c22 01aa182b
	v_add_co_u32 v35, vcc_lo, s42, v19                         // 000000001f50: d7006a23 0202262a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v36, null, s43, v20, vcc_lo            // 000000001f5c: d5207c24 01aa282b
	v_mov_b32_e32 v20, v1                                      // 000000001f64: 7e280301
	v_dual_mov_b32 v16, v1 :: v_dual_mov_b32 v13, v1           // 000000001f68: ca100101 100c0101
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v11, v1           // 000000001f70: ca100101 0c0a0101
	v_mov_b32_e32 v19, v1                                      // 000000001f78: 7e260301
	v_or_b32_e32 v37, s58, v0                                  // 000000001f7c: 384a003a
	v_add_co_u32 v53, vcc_lo, v7, s58                          // 000000001f80: d7006a35 02007507
	s_wait_alu depctr_va_vcc(0)                                // 000000001f88: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, s59, v8, vcc_lo             // 000000001f8c: d5207c36 01aa103b
	v_mov_b32_e32 v38, s59                                     // 000000001f94: 7e4c023b
	v_or_b32_e32 v39, 1, v37                                   // 000000001f98: 384e4a81
	v_mov_b32_e32 v40, s59                                     // 000000001f9c: 7e50023b
	v_or_b32_e32 v45, 3, v37                                   // 000000001fa0: 385a4a83
	v_mov_b32_e32 v46, s59                                     // 000000001fa4: 7e5c023b
	v_cmp_gt_i64_e64 s9, s[50:51], v[37:38]                    // 000000001fa8: d4540009 02024a32
	v_or_b32_e32 v49, 5, v37                                   // 000000001fb0: 38624a85
	v_cmp_gt_i64_e64 s10, s[50:51], v[39:40]                   // 000000001fb4: d454000a 02024e32
	v_add_co_u32 v39, vcc_lo, v53, 1                           // 000000001fbc: d7006a27 02010335
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc4: bf88ff9d
	v_add_co_ci_u32_e64 v40, null, 0, v54, vcc_lo              // 000000001fc8: d5207c28 01aa6c80
	s_and_b32 vcc_lo, s1, s9                                   // 000000001fd0: 8b6a0901
	s_and_b32 s2, s1, s10                                      // 000000001fd4: 8b020a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fd8: bf88ff9e
	v_dual_cndmask_b32 v42, 0, v53 :: v_dual_cndmask_b32 v41, 0, v54// 000000001fdc: ca526a80 2a286c80
	v_cndmask_b32_e64 v44, 0, v39, s2                          // 000000001fe4: d501002c 000a4e80
	v_cndmask_b32_e64 v43, 0, v40, s2                          // 000000001fec: d501002b 000a5080
	v_cmp_gt_i64_e64 s13, s[50:51], v[45:46]                   // 000000001ff4: d454000d 02025a32
	s_delay_alu instid0(valu_dep_4)                            // 000000001ffc: bf870004
	v_add_co_u32 v39, s3, s46, v42                             // 000000002000: d7000327 0202542e
	s_wait_alu depctr_va_sdst(0)                               // 000000002008: bf88f19f
	v_add_co_ci_u32_e64 v40, null, s47, v41, s3                // 00000000200c: d5207c28 000e522f
	v_add_co_u32 v41, s3, s46, v44                             // 000000002014: d7000329 0202582e
	s_wait_alu depctr_va_sdst(0)                               // 00000000201c: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s47, v43, s3                // 000000002020: d5207c2a 000e562f
	v_or_b32_e32 v43, 2, v37                                   // 000000002028: 38564a82
	v_mov_b32_e32 v44, s59                                     // 00000000202c: 7e58023b
	v_add_co_u32 v47, s3, v53, 2                               // 000000002030: d700032f 02010535
	s_wait_alu depctr_va_sdst(0)                               // 000000002038: bf88f19f
	v_add_co_ci_u32_e64 v48, null, 0, v54, s3                  // 00000000203c: d5207c30 000e6c80
	s_delay_alu instid0(valu_dep_3)                            // 000000002044: bf870003
	v_cmp_gt_i64_e64 s11, s[50:51], v[43:44]                   // 000000002048: d454000b 02025632
	v_add_co_u32 v43, s3, v53, 3                               // 000000002050: d700032b 02010735
	s_wait_alu depctr_va_sdst(0)                               // 000000002058: bf88f19f
	v_add_co_ci_u32_e64 v44, null, 0, v54, s3                  // 00000000205c: d5207c2c 000e6c80
	s_and_b32 s3, s1, s13                                      // 000000002064: 8b030d01
	s_and_b32 s4, s1, s11                                      // 000000002068: 8b040b01
	v_mov_b32_e32 v50, s59                                     // 00000000206c: 7e64023b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002070: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v47, s4                          // 000000002074: d501002e 00125e80
	v_cndmask_b32_e64 v45, 0, v48, s4                          // 00000000207c: d501002d 00126080
	v_cndmask_b32_e64 v48, 0, v43, s3                          // 000000002084: d5010030 000e5680
	v_cndmask_b32_e64 v47, 0, v44, s3                          // 00000000208c: d501002f 000e5880
	v_cmp_gt_i64_e64 s15, s[50:51], v[49:50]                   // 000000002094: d454000f 02026232
	v_add_co_u32 v43, s5, s46, v46                             // 00000000209c: d700052b 02025c2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s47, v45, s5                // 0000000020a8: d5207c2c 00165a2f
	v_add_co_u32 v45, s5, s46, v48                             // 0000000020b0: d700052d 0202602e
	s_wait_alu depctr_va_sdst(0)                               // 0000000020b8: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s47, v47, s5                // 0000000020bc: d5207c2e 00165e2f
	v_or_b32_e32 v47, 4, v37                                   // 0000000020c4: 385e4a84
	v_mov_b32_e32 v48, s59                                     // 0000000020c8: 7e60023b
	v_add_co_u32 v51, s5, v53, 4                               // 0000000020cc: d7000533 02010935
	s_wait_alu depctr_va_sdst(0)                               // 0000000020d4: bf88f19f
	v_add_co_ci_u32_e64 v52, null, 0, v54, s5                  // 0000000020d8: d5207c34 00166c80
	s_delay_alu instid0(valu_dep_3)                            // 0000000020e0: bf870003
	v_cmp_gt_i64_e64 s14, s[50:51], v[47:48]                   // 0000000020e4: d454000e 02025e32
	v_add_co_u32 v47, s5, v53, 5                               // 0000000020ec: d700052f 02010b35
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f4: bf88f19f
	v_add_co_ci_u32_e64 v48, null, 0, v54, s5                  // 0000000020f8: d5207c30 00166c80
	s_and_b32 s6, s1, s15                                      // 000000002100: 8b060f01
	s_and_b32 s5, s1, s14                                      // 000000002104: 8b050e01
	s_and_b32 s9, s0, s9                                       // 000000002108: 8b090900
	s_wait_alu depctr_sa_sdst(0)                               // 00000000210c: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v51, s5                          // 000000002110: d5010032 00166680
	v_cndmask_b32_e64 v49, 0, v52, s5                          // 000000002118: d5010031 00166880
	v_cndmask_b32_e64 v52, 0, v47, s6                          // 000000002120: d5010034 001a5e80
	v_cndmask_b32_e64 v51, 0, v48, s6                          // 000000002128: d5010033 001a6080
	s_and_b32 s10, s0, s10                                     // 000000002130: 8b0a0a00
	v_add_co_u32 v47, s7, s46, v50                             // 000000002134: d700072f 0202642e
	s_wait_alu depctr_va_sdst(0)                               // 00000000213c: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s47, v49, s7                // 000000002140: d5207c30 001e622f
	v_add_co_u32 v49, s7, s46, v52                             // 000000002148: d7000731 0202682e
	s_wait_alu depctr_va_sdst(0)                               // 000000002150: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s47, v51, s7                // 000000002154: d5207c32 001e662f
	v_or_b32_e32 v51, 6, v37                                   // 00000000215c: 38664a86
	v_mov_b32_e32 v52, s59                                     // 000000002160: 7e68023b
	v_or_b32_e32 v37, 7, v37                                   // 000000002164: 384a4a87
	v_add_co_u32 v55, s7, v53, 6                               // 000000002168: d7000737 02010d35
	s_wait_alu depctr_va_sdst(0)                               // 000000002170: bf88f19f
	v_add_co_ci_u32_e64 v56, null, 0, v54, s7                  // 000000002174: d5207c38 001e6c80
	v_cmp_gt_i64_e64 s16, s[50:51], v[51:52]                   // 00000000217c: d4540010 02026632
	v_cmp_gt_i64_e64 s17, s[50:51], v[37:38]                   // 000000002184: d4540011 02024a32
	v_add_co_u32 v37, s7, v53, 7                               // 00000000218c: d7000725 02010f35
	s_wait_alu depctr_va_sdst(0)                               // 000000002194: bf88f19f
	v_add_co_ci_u32_e64 v38, null, 0, v54, s7                  // 000000002198: d5207c26 001e6c80
	s_and_b32 s7, s1, s16                                      // 0000000021a0: 8b071001
	s_and_b32 s8, s1, s17                                      // 0000000021a4: 8b081101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021a8: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v55, s7                          // 0000000021ac: d5010033 001e6e80
	v_cndmask_b32_e64 v52, 0, v56, s7                          // 0000000021b4: d5010034 001e7080
	v_cndmask_b32_e64 v37, 0, v37, s8                          // 0000000021bc: d5010025 00224a80
	v_cndmask_b32_e64 v38, 0, v38, s8                          // 0000000021c4: d5010026 00224c80
	s_or_b32 s60, s58, 16                                      // 0000000021cc: 8c3c903a
	v_add_co_u32 v51, s12, s46, v51                            // 0000000021d0: d7000c33 0202662e
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s47, v52, s12               // 0000000021dc: d5207c34 0032682f
	v_add_co_u32 v53, s12, s46, v37                            // 0000000021e4: d7000c35 02024a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000021ec: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s47, v38, s12               // 0000000021f0: d5207c36 00324c2f
	v_add_co_u32 v55, s12, v9, s58                             // 0000000021f8: d7000c37 02007509
	s_wait_alu depctr_va_sdst(0)                               // 000000002200: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s59, v10, s12               // 000000002204: d5207c38 0032143b
	s_clause 0x7                                               // 00000000220c: bf850007
	global_load_d16_u8 v37, v[39:40], off                      // 000000002210: ee07807c 00000025 00000027
	global_load_d16_hi_u8 v37, v[41:42], off                   // 00000000221c: ee08407c 00000025 00000029
	global_load_d16_u8 v38, v[43:44], off                      // 000000002228: ee07807c 00000026 0000002b
	global_load_d16_hi_u8 v38, v[45:46], off                   // 000000002234: ee08407c 00000026 0000002d
	global_load_d16_u8 v39, v[47:48], off                      // 000000002240: ee07807c 00000027 0000002f
	global_load_d16_hi_u8 v39, v[49:50], off                   // 00000000224c: ee08407c 00000027 00000031
	global_load_d16_u8 v40, v[51:52], off                      // 000000002258: ee07807c 00000028 00000033
	global_load_d16_hi_u8 v40, v[53:54], off                   // 000000002264: ee08407c 00000028 00000035
	v_add_co_u32 v41, s12, v55, 1                              // 000000002270: d7000c29 02010337
	s_wait_alu depctr_va_sdst(0)                               // 000000002278: bf88f19f
	v_add_co_ci_u32_e64 v42, null, 0, v56, s12                 // 00000000227c: d5207c2a 00327080
	v_cndmask_b32_e64 v44, 0, v55, s9                          // 000000002284: d501002c 00266e80
	v_cndmask_b32_e64 v43, 0, v56, s9                          // 00000000228c: d501002b 00267080
	v_cndmask_b32_e64 v46, 0, v41, s10                         // 000000002294: d501002e 002a5280
	s_delay_alu instid0(valu_dep_4)                            // 00000000229c: bf870004
	v_cndmask_b32_e64 v45, 0, v42, s10                         // 0000000022a0: d501002d 002a5480
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022a8: bf88ff9e
	v_or_b32_e32 v57, s60, v0                                  // 0000000022ac: 3872003c
	v_add_co_u32 v41, s12, s48, v44                            // 0000000022b0: d7000c29 02025830
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b8: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s49, v43, s12               // 0000000022bc: d5207c2a 00325631
	v_add_co_u32 v43, s12, s48, v46                            // 0000000022c4: d7000c2b 02025c30
	s_wait_alu depctr_va_sdst(0)                               // 0000000022cc: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s49, v45, s12               // 0000000022d0: d5207c2c 00325a31
	v_add_co_u32 v45, s12, v55, 2                              // 0000000022d8: d7000c2d 02010537
	s_wait_alu depctr_va_sdst(0)                               // 0000000022e0: bf88f19f
	v_add_co_ci_u32_e64 v46, null, 0, v56, s12                 // 0000000022e4: d5207c2e 00327080
	v_add_co_u32 v47, s12, v55, 3                              // 0000000022ec: d7000c2f 02010737
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f4: bf88f19f
	v_add_co_ci_u32_e64 v48, null, 0, v56, s12                 // 0000000022f8: d5207c30 00327080
	s_and_b32 s12, s0, s11                                     // 000000002300: 8b0c0b00
	s_and_b32 s11, s0, s13                                     // 000000002304: 8b0b0d00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002308: bf88ff9e
	v_cndmask_b32_e64 v45, 0, v45, s12                         // 00000000230c: d501002d 00325a80
	v_cndmask_b32_e64 v46, 0, v46, s12                         // 000000002314: d501002e 00325c80
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 00000000231c: d501002f 002e5e80
	v_cndmask_b32_e64 v48, 0, v48, s11                         // 000000002324: d5010030 002e6080
	v_mov_b32_e32 v58, s59                                     // 00000000232c: 7e74023b
	v_add_co_u32 v45, s13, s48, v45                            // 000000002330: d7000d2d 02025a30
	s_wait_alu depctr_va_sdst(0)                               // 000000002338: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s49, v46, s13               // 00000000233c: d5207c2e 00365c31
	v_add_co_u32 v47, s13, s48, v47                            // 000000002344: d7000d2f 02025e30
	s_wait_alu depctr_va_sdst(0)                               // 00000000234c: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s49, v48, s13               // 000000002350: d5207c30 00366031
	v_add_co_u32 v49, s13, v55, 4                              // 000000002358: d7000d31 02010937
	s_wait_alu depctr_va_sdst(0)                               // 000000002360: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v56, s13                 // 000000002364: d5207c32 00367080
	v_add_co_u32 v51, s13, v55, 5                              // 00000000236c: d7000d33 02010b37
	s_wait_alu depctr_va_sdst(0)                               // 000000002374: bf88f19f
	v_add_co_ci_u32_e64 v52, null, 0, v56, s13                 // 000000002378: d5207c34 00367080
	s_and_b32 s13, s0, s14                                     // 000000002380: 8b0d0e00
	s_and_b32 s14, s0, s15                                     // 000000002384: 8b0e0f00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002388: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v49, s13                         // 00000000238c: d5010031 00366280
	v_cndmask_b32_e64 v50, 0, v50, s13                         // 000000002394: d5010032 00366480
	v_cndmask_b32_e64 v51, 0, v51, s14                         // 00000000239c: d5010033 003a6680
	v_cndmask_b32_e64 v52, 0, v52, s14                         // 0000000023a4: d5010034 003a6880
	v_cmp_gt_i64_e64 s25, s[50:51], v[57:58]                   // 0000000023ac: d4540019 02027232
	v_add_co_u32 v49, s15, s48, v49                            // 0000000023b4: d7000f31 02026230
	s_wait_alu depctr_va_sdst(0)                               // 0000000023bc: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s49, v50, s15               // 0000000023c0: d5207c32 003e6431
	v_add_co_u32 v51, s15, s48, v51                            // 0000000023c8: d7000f33 02026630
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d0: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s49, v52, s15               // 0000000023d4: d5207c34 003e6831
	v_add_co_u32 v53, s15, v55, 6                              // 0000000023dc: d7000f35 02010d37
	s_wait_alu depctr_va_sdst(0)                               // 0000000023e4: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v56, s15                 // 0000000023e8: d5207c36 003e7080
	v_add_co_u32 v55, s15, v55, 7                              // 0000000023f0: d7000f37 02010f37
	s_wait_alu depctr_va_sdst(0)                               // 0000000023f8: bf88f19f
	v_add_co_ci_u32_e64 v56, null, 0, v56, s15                 // 0000000023fc: d5207c38 003e7080
	s_and_b32 s15, s0, s16                                     // 000000002404: 8b0f1000
	s_and_b32 s16, s0, s17                                     // 000000002408: 8b101100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000240c: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s15                         // 000000002410: d5010035 003e6a80
	v_cndmask_b32_e64 v54, 0, v54, s15                         // 000000002418: d5010036 003e6c80
	v_cndmask_b32_e64 v55, 0, v55, s16                         // 000000002420: d5010037 00426e80
	v_cndmask_b32_e64 v56, 0, v56, s16                         // 000000002428: d5010038 00427080
	s_lshr_b64 s[62:63], s[58:59], 3                           // 000000002430: 85be833a
	v_add_co_u32 v53, s17, s48, v53                            // 000000002434: d7001135 02026a30
	s_wait_alu depctr_va_sdst(0)                               // 00000000243c: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s49, v54, s17               // 000000002440: d5207c36 00466c31
	v_add_co_u32 v55, s17, s48, v55                            // 000000002448: d7001137 02026e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002450: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s49, v56, s17               // 000000002454: d5207c38 00467031
	s_clause 0x7                                               // 00000000245c: bf850007
	global_load_d16_u8 v41, v[41:42], off                      // 000000002460: ee07807c 00000029 00000029
	global_load_d16_hi_u8 v41, v[43:44], off                   // 00000000246c: ee08407c 00000029 0000002b
	global_load_d16_u8 v42, v[45:46], off                      // 000000002478: ee07807c 0000002a 0000002d
	global_load_d16_hi_u8 v42, v[47:48], off                   // 000000002484: ee08407c 0000002a 0000002f
	global_load_d16_u8 v43, v[49:50], off                      // 000000002490: ee07807c 0000002b 00000031
	global_load_d16_hi_u8 v43, v[51:52], off                   // 00000000249c: ee08407c 0000002b 00000033
	global_load_d16_u8 v44, v[53:54], off                      // 0000000024a8: ee07807c 0000002c 00000035
	global_load_d16_hi_u8 v44, v[55:56], off                   // 0000000024b4: ee08407c 0000002c 00000037
	v_or_b32_e32 v45, 1, v57                                   // 0000000024c0: 385a7281
	v_mov_b32_e32 v46, s59                                     // 0000000024c4: 7e5c023b
	v_add_co_u32 v61, s17, v7, s60                             // 0000000024c8: d700113d 02007907
	s_wait_alu depctr_va_sdst(0)                               // 0000000024d0: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s59, v8, s17                // 0000000024d4: d5207c3e 0046103b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000024dc: bf870193
	v_cmp_gt_i64_e64 s26, s[50:51], v[45:46]                   // 0000000024e0: d454001a 02025a32
	v_add_co_u32 v45, s17, v61, 1                              // 0000000024e8: d700112d 0201033d
	s_wait_alu depctr_va_sdst(0)                               // 0000000024f0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000024f4: bf870003
	v_add_co_ci_u32_e64 v46, null, 0, v62, s17                 // 0000000024f8: d5207c2e 00467c80
	s_and_b32 s17, s1, s25                                     // 000000002500: 8b111901
	s_and_b32 s18, s1, s26                                     // 000000002504: 8b121a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002508: bf88ff9e
	v_cndmask_b32_e64 v48, 0, v61, s17                         // 00000000250c: d5010030 00467a80
	v_cndmask_b32_e64 v47, 0, v62, s17                         // 000000002514: d501002f 00467c80
	v_cndmask_b32_e64 v50, 0, v45, s18                         // 00000000251c: d5010032 004a5a80
	v_cndmask_b32_e64 v49, 0, v46, s18                         // 000000002524: d5010031 004a5c80
	v_or_b32_e32 v51, 3, v57                                   // 00000000252c: 38667283
	v_add_co_u32 v45, s19, s46, v48                            // 000000002530: d700132d 0202602e
	s_wait_alu depctr_va_sdst(0)                               // 000000002538: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s47, v47, s19               // 00000000253c: d5207c2e 004e5e2f
	v_add_co_u32 v47, s19, s46, v50                            // 000000002544: d700132f 0202642e
	s_wait_alu depctr_va_sdst(0)                               // 00000000254c: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s47, v49, s19               // 000000002550: d5207c30 004e622f
	v_or_b32_e32 v49, 2, v57                                   // 000000002558: 38627282
	v_mov_b32_e32 v50, s59                                     // 00000000255c: 7e64023b
	v_mov_b32_e32 v52, s59                                     // 000000002560: 7e68023b
	v_add_co_u32 v53, s19, v61, 2                              // 000000002564: d7001335 0201053d
	s_wait_alu depctr_va_sdst(0)                               // 00000000256c: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v62, s19                 // 000000002570: d5207c36 004e7c80
	v_cmp_gt_i64_e64 s27, s[50:51], v[49:50]                   // 000000002578: d454001b 02026232
	v_cmp_gt_i64_e64 s29, s[50:51], v[51:52]                   // 000000002580: d454001d 02026632
	v_add_co_u32 v49, s19, v61, 3                              // 000000002588: d7001331 0201073d
	s_wait_alu depctr_va_sdst(0)                               // 000000002590: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v62, s19                 // 000000002594: d5207c32 004e7c80
	s_and_b32 s20, s1, s27                                     // 00000000259c: 8b141b01
	s_and_b32 s19, s1, s29                                     // 0000000025a0: 8b131d01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025a4: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v53, s20                         // 0000000025a8: d5010034 00526a80
	v_cndmask_b32_e64 v51, 0, v54, s20                         // 0000000025b0: d5010033 00526c80
	v_cndmask_b32_e64 v54, 0, v49, s19                         // 0000000025b8: d5010036 004e6280
	v_cndmask_b32_e64 v53, 0, v50, s19                         // 0000000025c0: d5010035 004e6480
	v_or_b32_e32 v55, 5, v57                                   // 0000000025c8: 386e7285
	v_add_co_u32 v49, s21, s46, v52                            // 0000000025cc: d7001531 0202682e
	s_wait_alu depctr_va_sdst(0)                               // 0000000025d4: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s47, v51, s21               // 0000000025d8: d5207c32 0056662f
	v_add_co_u32 v51, s21, s46, v54                            // 0000000025e0: d7001533 02026c2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000025e8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s47, v53, s21               // 0000000025ec: d5207c34 00566a2f
	v_or_b32_e32 v53, 4, v57                                   // 0000000025f4: 386a7284
	v_mov_b32_e32 v54, s59                                     // 0000000025f8: 7e6c023b
	v_mov_b32_e32 v56, s59                                     // 0000000025fc: 7e70023b
	v_add_co_u32 v59, s21, v61, 4                              // 000000002600: d700153b 0201093d
	s_wait_alu depctr_va_sdst(0)                               // 000000002608: bf88f19f
	v_add_co_ci_u32_e64 v60, null, 0, v62, s21                 // 00000000260c: d5207c3c 00567c80
	v_cmp_gt_i64_e64 s30, s[50:51], v[53:54]                   // 000000002614: d454001e 02026a32
	v_cmp_gt_i64_e64 s31, s[50:51], v[55:56]                   // 00000000261c: d454001f 02026e32
	v_add_co_u32 v53, s21, v61, 5                              // 000000002624: d7001535 02010b3d
	s_wait_alu depctr_va_sdst(0)                               // 00000000262c: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v62, s21                 // 000000002630: d5207c36 00567c80
	s_and_b32 s21, s1, s30                                     // 000000002638: 8b151e01
	s_and_b32 s22, s1, s31                                     // 00000000263c: 8b161f01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002640: bf88ff9e
	v_cndmask_b32_e64 v56, 0, v59, s21                         // 000000002644: d5010038 00567680
	v_cndmask_b32_e64 v55, 0, v60, s21                         // 00000000264c: d5010037 00567880
	v_cndmask_b32_e64 v60, 0, v53, s22                         // 000000002654: d501003c 005a6a80
	v_cndmask_b32_e64 v59, 0, v54, s22                         // 00000000265c: d501003b 005a6c80
	s_and_b32 s25, s0, s25                                     // 000000002664: 8b191900
	v_add_co_u32 v53, s23, s46, v56                            // 000000002668: d7001735 0202702e
	s_wait_alu depctr_va_sdst(0)                               // 000000002670: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s47, v55, s23               // 000000002674: d5207c36 005e6e2f
	v_add_co_u32 v55, s23, s46, v60                            // 00000000267c: d7001737 0202782e
	s_wait_alu depctr_va_sdst(0)                               // 000000002684: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s47, v59, s23               // 000000002688: d5207c38 005e762f
	v_or_b32_e32 v59, 6, v57                                   // 000000002690: 38767286
	v_mov_b32_e32 v60, s59                                     // 000000002694: 7e78023b
	v_or_b32_e32 v57, 7, v57                                   // 000000002698: 38727287
	v_add_co_u32 v63, s23, v61, 6                              // 00000000269c: d700173f 02010d3d
	s_wait_alu depctr_va_sdst(0)                               // 0000000026a4: bf88f19f
	v_add_co_ci_u32_e64 v64, null, 0, v62, s23                 // 0000000026a8: d5207c40 005e7c80
	v_cmp_gt_i64_e64 s33, s[50:51], v[59:60]                   // 0000000026b0: d4540021 02027632
	v_cmp_gt_i64_e64 s34, s[50:51], v[57:58]                   // 0000000026b8: d4540022 02027232
	v_add_co_u32 v57, s23, v61, 7                              // 0000000026c0: d7001739 02010f3d
	s_wait_alu depctr_va_sdst(0)                               // 0000000026c8: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v62, s23                 // 0000000026cc: d5207c3a 005e7c80
	s_and_b32 s23, s1, s33                                     // 0000000026d4: 8b172101
	s_and_b32 s24, s1, s34                                     // 0000000026d8: 8b182201
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026dc: bf88ff9e
	v_cndmask_b32_e64 v60, 0, v63, s23                         // 0000000026e0: d501003c 005e7e80
	v_cndmask_b32_e64 v59, 0, v64, s23                         // 0000000026e8: d501003b 005e8080
	v_cndmask_b32_e64 v62, 0, v57, s24                         // 0000000026f0: d501003e 00627280
	v_cndmask_b32_e64 v61, 0, v58, s24                         // 0000000026f8: d501003d 00627480
	s_and_b32 s26, s0, s26                                     // 000000002700: 8b1a1a00
	v_add_co_u32 v57, s28, s46, v60                            // 000000002704: d7001c39 0202782e
	s_wait_alu depctr_va_sdst(0)                               // 00000000270c: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s47, v59, s28               // 000000002710: d5207c3a 0072762f
	v_add_co_u32 v59, s28, s46, v62                            // 000000002718: d7001c3b 02027c2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002720: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s47, v61, s28               // 000000002724: d5207c3c 00727a2f
	v_add_co_u32 v61, s28, v9, s60                             // 00000000272c: d7001c3d 02007909
	s_wait_alu depctr_va_sdst(0)                               // 000000002734: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s59, v10, s28               // 000000002738: d5207c3e 0072143b
	s_clause 0x7                                               // 000000002740: bf850007
	global_load_d16_u8 v45, v[45:46], off                      // 000000002744: ee07807c 0000002d 0000002d
	global_load_d16_hi_u8 v45, v[47:48], off                   // 000000002750: ee08407c 0000002d 0000002f
	global_load_d16_u8 v46, v[49:50], off                      // 00000000275c: ee07807c 0000002e 00000031
	global_load_d16_hi_u8 v46, v[51:52], off                   // 000000002768: ee08407c 0000002e 00000033
	global_load_d16_u8 v47, v[53:54], off                      // 000000002774: ee07807c 0000002f 00000035
	global_load_d16_hi_u8 v47, v[55:56], off                   // 000000002780: ee08407c 0000002f 00000037
	global_load_d16_u8 v48, v[57:58], off                      // 00000000278c: ee07807c 00000030 00000039
	global_load_d16_hi_u8 v48, v[59:60], off                   // 000000002798: ee08407c 00000030 0000003b
	v_add_co_u32 v49, s28, v61, 1                              // 0000000027a4: d7001c31 0201033d
	s_wait_alu depctr_va_sdst(0)                               // 0000000027ac: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v62, s28                 // 0000000027b0: d5207c32 00727c80
	v_cndmask_b32_e64 v52, 0, v61, s25                         // 0000000027b8: d5010034 00667a80
	v_cndmask_b32_e64 v51, 0, v62, s25                         // 0000000027c0: d5010033 00667c80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027c8: bf88ff9e
	v_cndmask_b32_e64 v54, 0, v49, s26                         // 0000000027cc: d5010036 006a6280
	v_cndmask_b32_e64 v53, 0, v50, s26                         // 0000000027d4: d5010035 006a6480
	s_lshr_b64 s[60:61], s[58:59], 5                           // 0000000027dc: 85bc853a
	v_add_co_u32 v49, s28, s48, v52                            // 0000000027e0: d7001c31 02026830
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e8: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s49, v51, s28               // 0000000027ec: d5207c32 00726631
	v_add_co_u32 v51, s28, s48, v54                            // 0000000027f4: d7001c33 02026c30
	s_wait_alu depctr_va_sdst(0)                               // 0000000027fc: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s49, v53, s28               // 000000002800: d5207c34 00726a31
	v_add_co_u32 v53, s28, v61, 2                              // 000000002808: d7001c35 0201053d
	s_wait_alu depctr_va_sdst(0)                               // 000000002810: bf88f19f
	v_add_co_ci_u32_e64 v54, null, 0, v62, s28                 // 000000002814: d5207c36 00727c80
	v_add_co_u32 v55, s28, v61, 3                              // 00000000281c: d7001c37 0201073d
	s_wait_alu depctr_va_sdst(0)                               // 000000002824: bf88f19f
	v_add_co_ci_u32_e64 v56, null, 0, v62, s28                 // 000000002828: d5207c38 00727c80
	s_and_b32 s28, s0, s27                                     // 000000002830: 8b1c1b00
	s_and_b32 s27, s0, s29                                     // 000000002834: 8b1b1d00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002838: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s28                         // 00000000283c: d5010035 00726a80
	v_cndmask_b32_e64 v54, 0, v54, s28                         // 000000002844: d5010036 00726c80
	v_cndmask_b32_e64 v55, 0, v55, s27                         // 00000000284c: d5010037 006e6e80
	v_cndmask_b32_e64 v56, 0, v56, s27                         // 000000002854: d5010038 006e7080
	s_mul_u64 s[60:61], s[60:61], s[38:39]                     // 00000000285c: aabc263c
	v_add_co_u32 v53, s29, s48, v53                            // 000000002860: d7001d35 02026a30
	s_wait_alu depctr_va_sdst(0)                               // 000000002868: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s49, v54, s29               // 00000000286c: d5207c36 00766c31
	v_add_co_u32 v55, s29, s48, v55                            // 000000002874: d7001d37 02026e30
	s_wait_alu depctr_va_sdst(0)                               // 00000000287c: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s49, v56, s29               // 000000002880: d5207c38 00767031
	v_add_co_u32 v57, s29, v61, 4                              // 000000002888: d7001d39 0201093d
	s_wait_alu depctr_va_sdst(0)                               // 000000002890: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v62, s29                 // 000000002894: d5207c3a 00767c80
	v_add_co_u32 v59, s29, v61, 5                              // 00000000289c: d7001d3b 02010b3d
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a4: bf88f19f
	v_add_co_ci_u32_e64 v60, null, 0, v62, s29                 // 0000000028a8: d5207c3c 00767c80
	s_and_b32 s29, s0, s30                                     // 0000000028b0: 8b1d1e00
	s_and_b32 s30, s0, s31                                     // 0000000028b4: 8b1e1f00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028b8: bf88ff9e
	v_cndmask_b32_e64 v57, 0, v57, s29                         // 0000000028bc: d5010039 00767280
	v_cndmask_b32_e64 v58, 0, v58, s29                         // 0000000028c4: d501003a 00767480
	v_cndmask_b32_e64 v59, 0, v59, s30                         // 0000000028cc: d501003b 007a7680
	v_cndmask_b32_e64 v60, 0, v60, s30                         // 0000000028d4: d501003c 007a7880
	s_lshl_b64 s[60:61], s[60:61], 2                           // 0000000028dc: 84bc823c
	v_add_co_u32 v57, s31, s48, v57                            // 0000000028e0: d7001f39 02027230
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e8: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s49, v58, s31               // 0000000028ec: d5207c3a 007e7431
	v_add_co_u32 v59, s31, s48, v59                            // 0000000028f4: d7001f3b 02027630
	s_wait_alu depctr_va_sdst(0)                               // 0000000028fc: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s49, v60, s31               // 000000002900: d5207c3c 007e7831
	v_add_co_u32 v63, s31, v61, 6                              // 000000002908: d7001f3f 02010d3d
	s_wait_alu depctr_va_sdst(0)                               // 000000002910: bf88f19f
	v_add_co_ci_u32_e64 v64, null, 0, v62, s31                 // 000000002914: d5207c40 007e7c80
	v_add_co_u32 v61, s31, v61, 7                              // 00000000291c: d7001f3d 02010f3d
	s_wait_alu depctr_va_sdst(0)                               // 000000002924: bf88f19f
	v_add_co_ci_u32_e64 v62, null, 0, v62, s31                 // 000000002928: d5207c3e 007e7c80
	s_and_b32 s31, s0, s33                                     // 000000002930: 8b1f2100
	s_and_b32 s33, s0, s34                                     // 000000002934: 8b212200
	s_wait_alu depctr_sa_sdst(0)                               // 000000002938: bf88ff9e
	v_cndmask_b32_e64 v63, 0, v63, s31                         // 00000000293c: d501003f 007e7e80
	v_cndmask_b32_e64 v64, 0, v64, s31                         // 000000002944: d5010040 007e8080
	v_cndmask_b32_e64 v66, 0, v61, s33                         // 00000000294c: d5010042 00867a80
	v_cndmask_b32_e64 v65, 0, v62, s33                         // 000000002954: d5010041 00867c80
	s_add_nc_u64 s[58:59], s[58:59], 32                        // 00000000295c: a9baa03a
	v_add_co_u32 v61, s34, s48, v63                            // 000000002960: d700223d 02027e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002968: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s49, v64, s34               // 00000000296c: d5207c3e 008a8031
	v_add_co_u32 v63, s34, s48, v66                            // 000000002974: d700223f 02028430
	s_wait_alu depctr_va_sdst(0)                               // 00000000297c: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s49, v65, s34               // 000000002980: d5207c40 008a8231
	s_clause 0x7                                               // 000000002988: bf850007
	global_load_d16_u8 v49, v[49:50], off                      // 00000000298c: ee07807c 00000031 00000031
	global_load_d16_hi_u8 v49, v[51:52], off                   // 000000002998: ee08407c 00000031 00000033
	global_load_d16_u8 v50, v[53:54], off                      // 0000000029a4: ee07807c 00000032 00000035
	global_load_d16_hi_u8 v50, v[55:56], off                   // 0000000029b0: ee08407c 00000032 00000037
	global_load_d16_u8 v51, v[57:58], off                      // 0000000029bc: ee07807c 00000033 00000039
	global_load_d16_hi_u8 v51, v[59:60], off                   // 0000000029c8: ee08407c 00000033 0000003b
	global_load_d16_u8 v52, v[61:62], off                      // 0000000029d4: ee07807c 00000034 0000003d
	global_load_d16_hi_u8 v52, v[63:64], off                   // 0000000029e0: ee08407c 00000034 0000003f
	v_add_co_u32 v53, s34, v14, s62                            // 0000000029ec: d7002235 02007d0e
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f4: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s63, v15, s34               // 0000000029f8: d5207c36 008a1e3f
	v_add_co_u32 v55, s34, v17, s60                            // 000000002a00: d7002237 02007911
	s_wait_alu depctr_va_sdst(0)                               // 000000002a08: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s61, v18, s34               // 000000002a0c: d5207c38 008a243d
	v_add_co_u32 v57, s34, v22, s62                            // 000000002a14: d7002239 02007d16
	s_wait_alu depctr_va_sdst(0)                               // 000000002a1c: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s63, v24, s34               // 000000002a20: d5207c3a 008a303f
	v_add_co_u32 v59, s34, v25, s62                            // 000000002a28: d700223b 02007d19
	s_wait_alu depctr_va_sdst(0)                               // 000000002a30: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s63, v26, s34               // 000000002a34: d5207c3c 008a343f
	v_add_co_u32 v61, s34, v27, s62                            // 000000002a3c: d700223d 02007d1b
	s_wait_alu depctr_va_sdst(0)                               // 000000002a44: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s63, v28, s34               // 000000002a48: d5207c3e 008a383f
	global_load_b32 v63, v[53:54], off                         // 000000002a50: ee05007c 0000003f 00000035
	global_load_b32 v64, v[55:56], off                         // 000000002a5c: ee05007c 00000040 00000037
	s_clause 0x2                                               // 000000002a68: bf850002
	global_load_b32 v65, v[57:58], off                         // 000000002a6c: ee05007c 00000041 00000039
	global_load_b32 v66, v[59:60], off                         // 000000002a78: ee05007c 00000042 0000003b
	global_load_b32 v61, v[61:62], off                         // 000000002a84: ee05007c 0000003d 0000003d
	v_add_co_u32 v53, s34, v29, s62                            // 000000002a90: d7002235 02007d1d
	s_wait_alu depctr_va_sdst(0)                               // 000000002a98: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s63, v30, s34               // 000000002a9c: d5207c36 008a3c3f
	v_add_co_u32 v55, s34, v31, s62                            // 000000002aa4: d7002237 02007d1f
	s_wait_alu depctr_va_sdst(0)                               // 000000002aac: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s63, v32, s34               // 000000002ab0: d5207c38 008a403f
	v_add_co_u32 v57, s34, v33, s62                            // 000000002ab8: d7002239 02007d21
	s_wait_alu depctr_va_sdst(0)                               // 000000002ac0: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s63, v34, s34               // 000000002ac4: d5207c3a 008a443f
	v_add_co_u32 v59, s34, v35, s62                            // 000000002acc: d700223b 02007d23
	s_wait_alu depctr_va_sdst(0)                               // 000000002ad4: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s63, v36, s34               // 000000002ad8: d5207c3c 008a483f
	s_clause 0x3                                               // 000000002ae0: bf850003
	global_load_b32 v62, v[53:54], off                         // 000000002ae4: ee05007c 0000003e 00000035
	global_load_b32 v67, v[55:56], off                         // 000000002af0: ee05007c 00000043 00000037
	global_load_b32 v57, v[57:58], off                         // 000000002afc: ee05007c 00000039 00000039
	global_load_b32 v58, v[59:60], off                         // 000000002b08: ee05007c 0000003a 0000003b
	s_wait_loadcnt 0x25                                        // 000000002b14: bfc00025
	v_cndmask_b16 v38.l, 0, v38.l, s4                          // 000000002b18: d65d0026 00124c80
	s_wait_loadcnt 0x21                                        // 000000002b20: bfc00021
	v_cndmask_b16 v40.h, 0, v40.h, s8                          // 000000002b24: d65d5028 00225080
	v_cndmask_b16 v40.l, 0, v40.l, s7                          // 000000002b2c: d65d0028 001e5080
	v_cndmask_b16 v39.h, 0, v39.h, s6                          // 000000002b34: d65d5027 001a4e80
	v_cndmask_b16 v39.l, 0, v39.l, s5                          // 000000002b3c: d65d0027 00164e80
	v_cndmask_b16 v38.h, 0, v38.h, s3                          // 000000002b44: d65d5026 000e4c80
	v_lshlrev_b16 v40.h, 8, v40.h op_sel:[0,1,1]               // 000000002b4c: d7385028 02025088
	v_and_b16 v40.l, 0xff, v40.l                               // 000000002b54: d7620028 020250ff 000000ff
	v_lshlrev_b16 v39.h, 8, v39.h op_sel:[0,1,1]               // 000000002b60: d7385027 02024e88
	v_and_b16 v39.l, 0xff, v39.l                               // 000000002b68: d7620027 02024eff 000000ff
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002b74: d7385026 02024c88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002b7c: d7620026 02024cff 000000ff
	v_cndmask_b16 v37.h, 0, v37.h, s2                          // 000000002b88: d65d5025 000a4a80
	v_cndmask_b16 v37.l, 0, v37.l, vcc_lo                      // 000000002b90: d65d0025 01aa4a80
	v_or_b16 v54.h, v40.l, v40.h op_sel:[0,1,1]                // 000000002b98: d7635036 02025128
	v_or_b16 v54.l, v39.l, v39.h op_sel:[0,1,0]                // 000000002ba0: d7631036 02024f27
	v_or_b16 v53.h, v38.l, v38.h op_sel:[0,1,1]                // 000000002ba8: d7635035 02024d26
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002bb0: d7385025 02024a88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002bb8: d7620025 02024aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc4: bf88ff9e
	v_cmp_lt_i64_e64 s2, s[58:59], s[44:45]                    // 000000002bc8: d4510002 0200583a
	s_delay_alu instid0(valu_dep_2)                            // 000000002bd0: bf870002
	v_or_b16 v53.l, v37.l, v37.h op_sel:[0,1,0]                // 000000002bd4: d7631035 02024b25
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002bdc: 8b6a027e
	s_wait_loadcnt 0x1f                                        // 000000002be0: bfc0001f
	v_cndmask_b16 v38.l, 0, v41.l, s9                          // 000000002be4: d65d0026 00265280
	v_cndmask_b16 v38.h, 0, v41.h, s10                         // 000000002bec: d65d5026 002a5280
	s_wait_loadcnt 0x1d                                        // 000000002bf4: bfc0001d
	v_cndmask_b16 v39.l, 0, v42.l, s12                         // 000000002bf8: d65d0027 00325480
	v_cndmask_b16 v41.h, 0, v42.h, s11                         // 000000002c00: d65d5029 002e5480
	s_wait_loadcnt 0x1b                                        // 000000002c08: bfc0001b
	v_cndmask_b16 v41.l, 0, v43.l, s13                         // 000000002c0c: d65d0029 00365680
	v_cndmask_b16 v40.h, 0, v43.h, s14                         // 000000002c14: d65d5028 003a5680
	s_wait_loadcnt 0x19                                        // 000000002c1c: bfc00019
	v_cndmask_b16 v40.l, 0, v44.l, s15                         // 000000002c20: d65d0028 003e5880
	v_cndmask_b16 v39.h, 0, v44.h, s16                         // 000000002c28: d65d5027 00425880
	v_lshlrev_b16 v41.h, 8, v41.h op_sel:[0,1,1]               // 000000002c30: d7385029 02025288
	v_and_b16 v41.l, 0xff, v41.l                               // 000000002c38: d7620029 020252ff 000000ff
	v_lshlrev_b16 v40.h, 8, v40.h op_sel:[0,1,1]               // 000000002c44: d7385028 02025088
	v_and_b16 v40.l, 0xff, v40.l                               // 000000002c4c: d7620028 020250ff 000000ff
	v_lshlrev_b16 v39.h, 8, v39.h op_sel:[0,1,1]               // 000000002c58: d7385027 02024e88
	v_and_b16 v39.l, 0xff, v39.l                               // 000000002c60: d7620027 02024eff 000000ff
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002c6c: d7385026 02024c88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002c74: d7620026 02024cff 000000ff
	v_or_b16 v56.l, v41.l, v40.h op_sel:[0,1,0]                // 000000002c80: d7631038 02025129
	v_or_b16 v56.h, v40.l, v39.h op_sel:[0,1,1]                // 000000002c88: d7635038 02024f28
	v_or_b16 v55.h, v39.l, v41.h op_sel:[0,1,1]                // 000000002c90: d7635037 02025327
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000002c98: bf870094
	v_or_b16 v55.l, v38.l, v38.h op_sel:[0,1,0]                // 000000002c9c: d7631037 02024d26
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[53:54], v[55:56], 0// 000000002ca4: cc464025 1a026f35
	s_wait_loadcnt 0x17                                        // 000000002cac: bfc00017
	v_cndmask_b16 v45.l, 0, v45.l, s17                         // 000000002cb0: d65d002d 00465a80
	v_cndmask_b16 v45.h, 0, v45.h, s18                         // 000000002cb8: d65d502d 004a5a80
	s_wait_loadcnt 0x15                                        // 000000002cc0: bfc00015
	v_cndmask_b16 v46.l, 0, v46.l, s20                         // 000000002cc4: d65d002e 00525c80
	v_cndmask_b16 v46.h, 0, v46.h, s19                         // 000000002ccc: d65d502e 004e5c80
	s_wait_loadcnt 0x13                                        // 000000002cd4: bfc00013
	v_cndmask_b16 v47.l, 0, v47.l, s21                         // 000000002cd8: d65d002f 00565e80
	v_cndmask_b16 v47.h, 0, v47.h, s22                         // 000000002ce0: d65d502f 005a5e80
	s_wait_loadcnt 0x11                                        // 000000002ce8: bfc00011
	v_cndmask_b16 v48.l, 0, v48.l, s23                         // 000000002cec: d65d0030 005e6080
	v_cndmask_b16 v48.h, 0, v48.h, s24                         // 000000002cf4: d65d5030 00626080
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000002cfc: d738502e 02025c88
	v_and_b16 v47.l, 0xff, v47.l                               // 000000002d04: d762002f 02025eff 000000ff
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000002d10: d738502f 02025e88
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002d18: d7620030 020260ff 000000ff
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 000000002d24: d7385030 02026088
	v_and_b16 v46.l, 0xff, v46.l                               // 000000002d2c: d762002e 02025cff 000000ff
	v_lshlrev_b16 v45.h, 8, v45.h op_sel:[0,1,1]               // 000000002d38: d738502d 02025a88
	v_or_b16 v54.l, v47.l, v47.h op_sel:[0,1,0]                // 000000002d40: d7631036 02025f2f
	v_and_b16 v45.l, 0xff, v45.l                               // 000000002d48: d762002d 02025aff 000000ff
	v_or_b16 v54.h, v48.l, v48.h op_sel:[0,1,1]                // 000000002d54: d7635036 02026130
	v_or_b16 v53.h, v46.l, v46.h op_sel:[0,1,1]                // 000000002d5c: d7635035 02025d2e
	s_delay_alu instid0(valu_dep_3)                            // 000000002d64: bf870003
	v_or_b16 v53.l, v45.l, v45.h op_sel:[0,1,0]                // 000000002d68: d7631035 02025b2d
	s_wait_loadcnt 0xf                                         // 000000002d70: bfc0000f
	v_cndmask_b16 v46.l, 0, v49.l, s25                         // 000000002d74: d65d002e 00666280
	v_cndmask_b16 v46.h, 0, v49.h, s26                         // 000000002d7c: d65d502e 006a6280
	s_wait_loadcnt 0xd                                         // 000000002d84: bfc0000d
	v_cndmask_b16 v47.l, 0, v50.l, s28                         // 000000002d88: d65d002f 00726480
	v_cndmask_b16 v49.h, 0, v50.h, s27                         // 000000002d90: d65d5031 006e6480
	s_wait_loadcnt 0xb                                         // 000000002d98: bfc0000b
	v_cndmask_b16 v49.l, 0, v51.l, s29                         // 000000002d9c: d65d0031 00766680
	v_cndmask_b16 v48.h, 0, v51.h, s30                         // 000000002da4: d65d5030 007a6680
	s_wait_loadcnt 0x9                                         // 000000002dac: bfc00009
	v_cndmask_b16 v48.l, 0, v52.l, s31                         // 000000002db0: d65d0030 007e6880
	v_cndmask_b16 v47.h, 0, v52.h, s33                         // 000000002db8: d65d502f 00866880
	v_lshlrev_b16 v49.h, 8, v49.h op_sel:[0,1,1]               // 000000002dc0: d7385031 02026288
	v_and_b16 v49.l, 0xff, v49.l                               // 000000002dc8: d7620031 020262ff 000000ff
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 000000002dd4: d7385030 02026088
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002ddc: d7620030 020260ff 000000ff
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000002de8: d738502f 02025e88
	v_and_b16 v47.l, 0xff, v47.l                               // 000000002df0: d762002f 02025eff 000000ff
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000002dfc: d738502e 02025c88
	v_and_b16 v46.l, 0xff, v46.l                               // 000000002e04: d762002e 02025cff 000000ff
	v_or_b16 v50.l, v49.l, v48.h op_sel:[0,1,0]                // 000000002e10: d7631032 02026131
	v_or_b16 v50.h, v48.l, v47.h op_sel:[0,1,1]                // 000000002e18: d7635032 02025f30
	v_or_b16 v49.h, v47.l, v49.h op_sel:[0,1,1]                // 000000002e20: d7635031 0202632f
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000002e28: bf870094
	v_or_b16 v49.l, v46.l, v46.h op_sel:[0,1,0]                // 000000002e2c: d7631031 02025d2e
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[53:54], v[49:50], v[37:44]// 000000002e34: cc464025 1c966335
	s_wait_loadcnt 0x6                                         // 000000002e3c: bfc00006
	v_dual_mul_f32 v45, v63, v64 :: v_dual_mul_f32 v46, v64, v65// 000000002e40: c8c6813f 2d2e8340
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e48: bf870091
	v_dual_mul_f32 v37, v37, v45 :: v_dual_mul_f32 v38, v38, v46// 000000002e4c: c8c65b25 25265d26
	v_dual_add_f32 v1, v1, v37 :: v_dual_add_f32 v20, v20, v38 // 000000002e54: c9084b01 01144d14
	s_wait_loadcnt 0x3                                         // 000000002e5c: bfc00003
	v_dual_mul_f32 v37, v64, v61 :: v_dual_mul_f32 v38, v64, v62// 000000002e60: c8c67b40 25267d40
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000002e68: bf8701b1
	v_mul_f32_e32 v37, v40, v37                                // 000000002e6c: 104a4b28
	s_wait_loadcnt 0x0                                         // 000000002e70: bfc00000
	v_dual_mul_f32 v45, v64, v57 :: v_dual_mul_f32 v40, v64, v58// 000000002e74: c8c67340 2d287540
	v_mul_f32_e32 v38, v41, v38                                // 000000002e7c: 104c4d29
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002e80: bf870193
	v_add_f32_e32 v13, v13, v37                                // 000000002e84: 061a4b0d
	v_dual_mul_f32 v41, v43, v45 :: v_dual_mul_f32 v40, v44, v40// 000000002e88: c8c65b2b 2928512c
	v_mul_f32_e32 v47, v64, v66                                // 000000002e90: 105e8540
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002e94: bf870194
	v_add_f32_e32 v12, v12, v38                                // 000000002e98: 06184d0c
	v_dual_add_f32 v2, v2, v41 :: v_dual_add_f32 v19, v19, v40 // 000000002e9c: c9085302 02125113
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000002ea4: bf870093
	v_mul_f32_e32 v39, v39, v47                                // 000000002ea8: 104e5f27
	v_add_f32_e32 v16, v16, v39                                // 000000002eac: 06204f10
	v_mul_f32_e32 v39, v64, v67                                // 000000002eb0: 104e8740
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002eb4: bf870091
	v_mul_f32_e32 v39, v42, v39                                // 000000002eb8: 104e4f2a
	v_add_f32_e32 v11, v11, v39                                // 000000002ebc: 06164f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ec0: bf88ff9e
	s_cbranch_vccnz 64557                                      // 000000002ec4: bfa4fc2d <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x47c>
	v_mul_lo_u32 v0, s39, v5                                   // 000000002ec8: d72c0000 02020a27
	v_mul_lo_u32 v14, s38, v6                                  // 000000002ed0: d72c000e 02020c26
	v_mad_co_u64_u32 v[9:10], null, s38, v5, 0                 // 000000002ed8: d6fe7c09 02020a26
	v_sub_co_u32 v7, vcc_lo, s36, v5                           // 000000002ee0: d7016a07 02020a24
	s_wait_alu depctr_va_vcc(0)                                // 000000002ee8: bf88ff9d
	v_sub_co_ci_u32_e64 v8, null, s37, v6, vcc_lo              // 000000002eec: d5217c08 01aa0c25
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[3:4]                  // 000000002ef4: 7ca80626
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002ef8: bf870194
	v_add3_u32 v10, v10, v14, v0                               // 000000002efc: d655000a 04021d0a
	v_cmp_lt_i64_e64 s0, 0, v[7:8]                             // 000000002f04: d4510000 02020e80
	s_delay_alu instid0(valu_dep_2)                            // 000000002f0c: bf870002
	v_lshlrev_b64_e32 v[5:6], 1, v[9:10]                       // 000000002f10: 3e0a1281
	s_and_b32 s0, s0, vcc_lo                                   // 000000002f14: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f18: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f1c: be812000
	s_cbranch_execz 28                                         // 000000002f20: bfa5001c <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1494>
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002f24: 3e120681
	v_add_co_u32 v14, s0, s40, v5                              // 000000002f28: d700000e 02020a28
	v_bfe_u32 v0, v1, 16, 1                                    // 000000002f30: d6100000 02052101
	s_wait_alu depctr_va_sdst(0)                               // 000000002f38: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s41, v6, s0                 // 000000002f3c: d5207c0f 00020c29
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002f44: bf870193
	v_add_co_u32 v9, s0, v14, v9                               // 000000002f48: d7000009 0202130e
	v_add3_u32 v0, v0, v1, 0x7fff                              // 000000002f50: d6550000 03fe0300 00007fff
	v_or_b32_e32 v17, 0x400000, v1                             // 000000002f5c: 382202ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f64: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v15, v10, s0                // 000000002f68: d5207c0a 0002150f
	v_cmp_u_f32_e64 s0, v1, v1                                 // 000000002f70: d4180000 02020301
	s_wait_alu depctr_va_sdst(0)                               // 000000002f78: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f7c: bf870001
	v_cndmask_b32_e64 v0, v0, v17, s0                          // 000000002f80: d5010000 00022300
	global_store_d16_hi_b16 v[9:10], v0, off                   // 000000002f88: ee09407c 00000000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f98: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[7:8]                             // 000000002f9c: d4510000 02020e81
	s_and_b32 s0, s0, vcc_lo                                   // 000000002fa4: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fa8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002fac: be812000
	s_cbranch_execz 35                                         // 000000002fb0: bfa50023 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1540>
	v_add_co_u32 v9, s0, s40, v5                               // 000000002fb4: d7000009 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000002fbc: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s41, v6, s0                 // 000000002fc0: d5207c0a 00020c29
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000002fc8: 84828126
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 000000002fcc: 3e000681
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fd0: bf88ff9e
	v_add_co_u32 v9, s0, v9, s2                                // 000000002fd4: d7000009 02000509
	v_bfe_u32 v14, v20, 16, 1                                  // 000000002fdc: d610000e 02052114
	s_wait_alu depctr_va_sdst(0)                               // 000000002fe4: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s3, v10, s0                 // 000000002fe8: d5207c0a 00021403
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ff0: bf870193
	v_add_co_u32 v0, s0, v9, v0                                // 000000002ff4: d7000000 02020109
	v_add3_u32 v14, v14, v20, 0x7fff                           // 000000002ffc: d655000e 03fe290e 00007fff
	v_or_b32_e32 v15, 0x400000, v20                            // 000000003008: 381e28ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003010: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v10, v1, s0                  // 000000003014: d5207c01 0002030a
	v_cmp_u_f32_e64 s0, v20, v20                               // 00000000301c: d4180000 02022914
	s_wait_alu depctr_va_sdst(0)                               // 000000003024: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003028: bf870001
	v_cndmask_b32_e64 v9, v14, v15, s0                         // 00000000302c: d5010009 00021f0e
	global_store_d16_hi_b16 v[0:1], v9, off                    // 000000003034: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003040: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003044: 8c7e017e
	v_cmp_lt_i64_e64 s0, 2, v[7:8]                             // 000000003048: d4510000 02020e82
	s_and_b32 s0, s0, vcc_lo                                   // 000000003050: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003054: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003058: be812000
	s_cbranch_execz 34                                         // 00000000305c: bfa50022 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x15e8>
	v_add_co_u32 v14, s0, s40, v5                              // 000000003060: d700000e 02020a28
	v_bfe_u32 v0, v16, 16, 1                                   // 000000003068: d6100000 02052110
	s_wait_alu depctr_va_sdst(0)                               // 000000003070: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s41, v6, s0                 // 000000003074: d5207c0f 00020c29
	s_lshl_b64 s[2:3], s[38:39], 2                             // 00000000307c: 84828226
	v_or_b32_e32 v9, 0x400000, v16                             // 000000003080: 381220ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003088: bf88ff9e
	v_add_co_u32 v14, s0, v14, s2                              // 00000000308c: d700000e 0200050e
	v_add3_u32 v10, v0, v16, 0x7fff                            // 000000003094: d655000a 03fe2100 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 0000000030a0: 3e000681
	s_wait_alu depctr_va_sdst(0)                               // 0000000030a4: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s3, v15, s0                 // 0000000030a8: d5207c0f 00021e03
	v_cmp_u_f32_e64 s0, v16, v16                               // 0000000030b0: d4180000 02022110
	s_wait_alu depctr_va_sdst(0)                               // 0000000030b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030bc: bf870001
	v_cndmask_b32_e64 v9, v10, v9, s0                          // 0000000030c0: d5010009 0002130a
	v_add_co_u32 v0, s0, v14, v0                               // 0000000030c8: d7000000 0202010e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v15, v1, s0                  // 0000000030d4: d5207c01 0002030f
	global_store_d16_hi_b16 v[0:1], v9, off                    // 0000000030dc: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030ec: 8c7e017e
	v_cmp_lt_i64_e64 s0, 3, v[7:8]                             // 0000000030f0: d4510000 02020e83
	s_and_b32 s0, s0, vcc_lo                                   // 0000000030f8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030fc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003100: be812000
	s_cbranch_execz 32                                         // 000000003104: bfa50020 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1688>
	v_add_co_u32 v0, s0, s40, v5                               // 000000003108: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003110: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 000000003114: d5207c01 00020c29
	v_bfe_u32 v9, v13, 16, 1                                   // 00000000311c: d6100009 0205210d
	v_lshlrev_b64_e32 v[14:15], 1, v[3:4]                      // 000000003124: 3e1c0681
	v_or_b32_e32 v16, 0x400000, v13                            // 000000003128: 38201aff 00400000
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 000000003130: bf8701b4
	v_mad_co_u64_u32 v[0:1], null, s38, 6, v[0:1]              // 000000003134: d6fe7c00 04010c26
	v_cmp_u_f32_e64 s0, v13, v13                               // 00000000313c: d4180000 02021b0d
	v_add3_u32 v17, v9, v13, 0x7fff                            // 000000003144: d6550011 03fe1b09 00007fff
	v_mad_co_u64_u32 v[9:10], null, s39, 6, v[1:2]             // 000000003150: d6fe7c09 04050c27
	s_wait_alu depctr_va_sdst(0)                               // 000000003158: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 00000000315c: bf8701b2
	v_cndmask_b32_e64 v10, v17, v16, s0                        // 000000003160: d501000a 00022111
	v_add_co_u32 v0, s0, v0, v14                               // 000000003168: d7000000 02021d00
	s_wait_alu depctr_va_sdst(0)                               // 000000003170: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v15, s0                  // 000000003174: d5207c01 00021f09
	global_store_d16_hi_b16 v[0:1], v10, off                   // 00000000317c: ee09407c 05000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003188: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000318c: 8c7e017e
	v_cmp_lt_i64_e64 s0, 4, v[7:8]                             // 000000003190: d4510000 02020e84
	s_and_b32 s0, s0, vcc_lo                                   // 000000003198: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000319c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000031a0: be812000
	s_cbranch_execz 34                                         // 0000000031a4: bfa50022 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1730>
	v_add_co_u32 v13, s0, s40, v5                              // 0000000031a8: d700000d 02020a28
	v_bfe_u32 v0, v12, 16, 1                                   // 0000000031b0: d6100000 0205210c
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b8: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s41, v6, s0                 // 0000000031bc: d5207c0e 00020c29
	s_lshl_b64 s[2:3], s[38:39], 3                             // 0000000031c4: 84828326
	v_or_b32_e32 v9, 0x400000, v12                             // 0000000031c8: 381218ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031d0: bf88ff9e
	v_add_co_u32 v13, s0, v13, s2                              // 0000000031d4: d700000d 0200050d
	v_add3_u32 v10, v0, v12, 0x7fff                            // 0000000031dc: d655000a 03fe1900 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 0000000031e8: 3e000681
	s_wait_alu depctr_va_sdst(0)                               // 0000000031ec: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s3, v14, s0                 // 0000000031f0: d5207c0e 00021c03
	v_cmp_u_f32_e64 s0, v12, v12                               // 0000000031f8: d4180000 0202190c
	s_wait_alu depctr_va_sdst(0)                               // 000000003200: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003204: bf870001
	v_cndmask_b32_e64 v9, v10, v9, s0                          // 000000003208: d5010009 0002130a
	v_add_co_u32 v0, s0, v13, v0                               // 000000003210: d7000000 0202010d
	s_wait_alu depctr_va_sdst(0)                               // 000000003218: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v14, v1, s0                  // 00000000321c: d5207c01 0002030e
	global_store_d16_hi_b16 v[0:1], v9, off                    // 000000003224: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003230: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003234: 8c7e017e
	v_cmp_lt_i64_e64 s0, 5, v[7:8]                             // 000000003238: d4510000 02020e85
	s_and_b32 s0, s0, vcc_lo                                   // 000000003240: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003244: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003248: be812000
	s_cbranch_execz 32                                         // 00000000324c: bfa50020 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x17d0>
	v_add_co_u32 v0, s0, s40, v5                               // 000000003250: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003258: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 00000000325c: d5207c01 00020c29
	v_bfe_u32 v9, v11, 16, 1                                   // 000000003264: d6100009 0205210b
	v_lshlrev_b64_e32 v[12:13], 1, v[3:4]                      // 00000000326c: 3e180681
	v_or_b32_e32 v14, 0x400000, v11                            // 000000003270: 381c16ff 00400000
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 000000003278: bf8701b4
	v_mad_co_u64_u32 v[0:1], null, s38, 10, v[0:1]             // 00000000327c: d6fe7c00 04011426
	v_cmp_u_f32_e64 s0, v11, v11                               // 000000003284: d4180000 0202170b
	v_add3_u32 v15, v9, v11, 0x7fff                            // 00000000328c: d655000f 03fe1709 00007fff
	v_mad_co_u64_u32 v[9:10], null, s39, 10, v[1:2]            // 000000003298: d6fe7c09 04051427
	s_wait_alu depctr_va_sdst(0)                               // 0000000032a0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 0000000032a4: bf8701b2
	v_cndmask_b32_e64 v10, v15, v14, s0                        // 0000000032a8: d501000a 00021d0f
	v_add_co_u32 v0, s0, v0, v12                               // 0000000032b0: d7000000 02021900
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v13, s0                  // 0000000032bc: d5207c01 00021b09
	global_store_d16_hi_b16 v[0:1], v10, off                   // 0000000032c4: ee09407c 05000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032d4: 8c7e017e
	v_cmp_lt_i64_e64 s0, 6, v[7:8]                             // 0000000032d8: d4510000 02020e86
	s_and_b32 s0, s0, vcc_lo                                   // 0000000032e0: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032e8: be812000
	s_cbranch_execz 33                                         // 0000000032ec: bfa50021 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1874>
	v_add_co_u32 v0, s0, s40, v5                               // 0000000032f0: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 0000000032f8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 0000000032fc: d5207c01 00020c29
	v_bfe_u32 v9, v2, 16, 1                                    // 000000003304: d6100009 02052102
	v_or_b32_e32 v12, 0x400000, v2                             // 00000000330c: 381804ff 00400000
	v_cmp_u_f32_e64 s0, v2, v2                                 // 000000003314: d4180000 02020502
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000331c: bf870214
	v_mad_co_u64_u32 v[0:1], null, s38, 12, v[0:1]             // 000000003320: d6fe7c00 04011826
	v_add3_u32 v13, v9, v2, 0x7fff                             // 000000003328: d655000d 03fe0509 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000003334: bf8701b2
	v_mad_co_u64_u32 v[9:10], null, s39, 12, v[1:2]            // 000000003338: d6fe7c09 04051827
	v_lshlrev_b64_e32 v[10:11], 1, v[3:4]                      // 000000003340: 3e140681
	s_wait_alu depctr_va_sdst(0)                               // 000000003344: bf88f19f
	v_cndmask_b32_e64 v2, v13, v12, s0                         // 000000003348: d5010002 0002190d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003350: bf8701a2
	v_add_co_u32 v0, s0, v0, v10                               // 000000003354: d7000000 02021500
	s_wait_alu depctr_va_sdst(0)                               // 00000000335c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v11, s0                  // 000000003360: d5207c01 00021709
	global_store_d16_hi_b16 v[0:1], v2, off                    // 000000003368: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003374: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003378: 8c7e017e
	v_cmp_lt_i64_e64 s0, 7, v[7:8]                             // 00000000337c: d4510000 02020e87
	s_mov_b32 s1, 0                                            // 000000003384: be810080
	s_and_b32 s2, s0, vcc_lo                                   // 000000003388: 8b026a00
	s_mov_b32 s0, 0                                            // 00000000338c: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003390: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003394: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003398: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 00000000339c: 8d02037e
	v_add_co_u32 v0, vcc_lo, s40, v5                           // 0000000033a0: d7006a00 02020a28
	s_wait_alu depctr_va_vcc(0)                                // 0000000033a8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v6, vcc_lo              // 0000000033ac: d5207c01 01aa0c29
	s_mov_b32 s0, exec_lo                                      // 0000000033b4: be80007e
	v_mad_co_u64_u32 v[0:1], null, s38, 14, v[0:1]             // 0000000033b8: d6fe7c00 04011c26
	s_delay_alu instid0(valu_dep_1)                            // 0000000033c0: bf870001
	v_mad_co_u64_u32 v[1:2], null, s39, 14, v[1:2]             // 0000000033c4: d6fe7c01 04051c27
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000033d0: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 0000000033d4: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 0000000033d8: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033dc: bf88ff9e
	s_cbranch_vccz 624                                         // 0000000033e0: bfa30270 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x22a4>
	s_and_b32 s0, s35, exec_lo                                 // 0000000033e4: 8b007e23
	s_cselect_b32 s0, 1, 0                                     // 0000000033e8: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033ec: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000033f0: bf078100
	s_cbranch_scc1 20                                          // 0000000033f4: bfa20014 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1948>
	v_lshl_or_b32 v15, v21, 3, s52                             // 0000000033f8: d656000f 00d10715
	v_dual_mov_b32 v16, s53 :: v_dual_mov_b32 v1, s53          // 000000003400: ca100035 10000035
	v_mov_b32_e32 v18, s53                                     // 000000003408: 7e240235
	v_mov_b32_e32 v14, s53                                     // 00000000340c: 7e1c0235
	s_delay_alu instid0(valu_dep_4)                            // 000000003410: bf870004
	v_or_b32_e32 v17, 1, v15                                   // 000000003414: 38221e81
	v_or_b32_e32 v13, 2, v15                                   // 000000003418: 381a1e82
	v_or_b32_e32 v11, 3, v15                                   // 00000000341c: 38161e83
	v_mov_b32_e32 v12, s53                                     // 000000003420: 7e180235
	v_or_b32_e32 v9, 4, v15                                    // 000000003424: 38121e84
	v_mov_b32_e32 v10, s53                                     // 000000003428: 7e140235
	v_or_b32_e32 v5, 5, v15                                    // 00000000342c: 380a1e85
	v_mov_b32_e32 v6, s53                                      // 000000003430: 7e0c0235
	v_or_b32_e32 v7, 6, v15                                    // 000000003434: 380e1e86
	v_mov_b32_e32 v8, s53                                      // 000000003438: 7e100235
	v_or_b32_e32 v0, 7, v15                                    // 00000000343c: 38001e87
	s_mov_b32 s0, 0                                            // 000000003440: be800080
	s_branch 1                                                 // 000000003444: bfa00001 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x194c>
	s_mov_b32 s0, -1                                           // 000000003448: be8000c1
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v2, 0              // 00000000344c: ca100080 13020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003454: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000003458: 8b007e00
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0             // 00000000345c: ca100080 2b2c0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0             // 000000003464: ca100080 2d2e0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v20, 0             // 00000000346c: ca100080 2f140080
	s_cselect_b32 s0, 1, 0                                     // 000000003474: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003478: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 00000000347c: bf078100
	s_cbranch_scc1 337                                         // 000000003480: bfa20151 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1ec8>
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[3:4]                  // 000000003484: 7ca80626
	v_dual_mov_b32 v16, s53 :: v_dual_lshlrev_b32 v19, 3, v21  // 000000003488: ca220035 10122a83
	v_mov_b32_e32 v20, 0                                       // 000000003490: 7e280280
	v_mov_b32_e32 v10, s53                                     // 000000003494: 7e140235
	v_mov_b32_e32 v6, s53                                      // 000000003498: 7e0c0235
	s_delay_alu instid0(valu_dep_4)                            // 00000000349c: bf870004
	v_or_b32_e32 v15, s52, v19                                 // 0000000034a0: 381e2634
	s_wait_alu depctr_va_vcc(0)                                // 0000000034a4: bf88ff9d
	v_dual_cndmask_b32 v1, 0, v4 :: v_dual_cndmask_b32 v0, 0, v3// 0000000034a8: ca520880 01000680
	s_lshr_b64 s[2:3], s[50:51], 5                             // 0000000034b0: 85828532
	v_mov_b32_e32 v8, s53                                      // 0000000034b4: 7e100235
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[15:16]                // 0000000034b8: 7ca81e24
	v_or_b32_e32 v17, 1, v15                                   // 0000000034bc: 38221e81
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 0000000034c0: 3e000082
	v_or_b32_e32 v13, 2, v15                                   // 0000000034c4: 381a1e82
	v_or_b32_e32 v11, 3, v15                                   // 0000000034c8: 38161e83
	v_or_b32_e32 v9, 4, v15                                    // 0000000034cc: 38121e84
	s_wait_alu depctr_va_vcc(0)                                // 0000000034d0: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v15, vcc_lo                       // 0000000034d4: 02041e80
	v_cndmask_b32_e64 v30, 0, s53, vcc_lo                      // 0000000034d8: d501001e 01a86a80
	v_add_co_u32 v21, vcc_lo, s56, v0                          // 0000000034e0: d7006a15 02020038
	s_wait_alu depctr_va_vcc(0)                                // 0000000034e8: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s57, v1, vcc_lo             // 0000000034ec: d5207c16 01aa0239
	v_dual_mov_b32 v1, s53 :: v_dual_mov_b32 v18, s53          // 0000000034f4: ca100035 01120035
	v_mov_b32_e32 v14, s53                                     // 0000000034fc: 7e1c0235
	v_mov_b32_e32 v12, s53                                     // 000000003500: 7e180235
	v_or_b32_e32 v5, 5, v15                                    // 000000003504: 380a1e85
	v_or_b32_e32 v0, 7, v15                                    // 000000003508: 38001e87
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[17:18]                // 00000000350c: 7ca82224
	v_or_b32_e32 v7, 6, v15                                    // 000000003510: 380e1e86
	v_cmp_gt_i64_e64 s0, s[36:37], v[11:12]                    // 000000003514: d4540000 02021624
	s_wait_alu depctr_va_vcc(0)                                // 00000000351c: bf88ff9d
	v_cndmask_b32_e32 v32, 0, v17, vcc_lo                      // 000000003520: 02402280
	v_cndmask_b32_e64 v33, 0, s53, vcc_lo                      // 000000003524: d5010021 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[13:14]                // 00000000352c: 7ca81a24
	s_wait_alu depctr_va_sdst(0)                               // 000000003530: bf88f19f
	v_cndmask_b32_e64 v36, 0, v11, s0                          // 000000003534: d5010024 00021680
	v_cndmask_b32_e64 v37, 0, s53, s0                          // 00000000353c: d5010025 00006a80
	v_cmp_gt_i64_e64 s0, s[36:37], v[5:6]                      // 000000003544: d4540000 02020a24
	s_wait_alu depctr_va_vcc(0)                                // 00000000354c: bf88ff9d
	v_cndmask_b32_e32 v34, 0, v13, vcc_lo                      // 000000003550: 02441a80
	v_cndmask_b32_e64 v35, 0, s53, vcc_lo                      // 000000003554: d5010023 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[9:10]                 // 00000000355c: 7ca81224
	s_wait_alu depctr_va_sdst(0)                               // 000000003560: bf88f19f
	v_cndmask_b32_e64 v40, 0, v5, s0                           // 000000003564: d5010028 00020a80
	v_cndmask_b32_e64 v41, 0, s53, s0                          // 00000000356c: d5010029 00006a80
	v_add_co_u32 v26, s0, s54, v23                             // 000000003574: d700001a 02022e36
	s_wait_alu depctr_va_sdst(0)                               // 00000000357c: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s55, 0, s0                  // 000000003580: d5207c1b 00010037
	s_wait_alu depctr_va_vcc(0)                                // 000000003588: bf88ff9d
	v_cndmask_b32_e32 v38, 0, v9, vcc_lo                       // 00000000358c: 024c1280
	v_cndmask_b32_e64 v39, 0, s53, vcc_lo                      // 000000003590: d5010027 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[0:1]                  // 000000003598: 7ca80024
	s_lshr_b32 s0, s51, 5                                      // 00000000359c: 85008533
	v_mad_co_u64_u32 v[24:25], null, s50, v26, v[19:20]        // 0000000035a0: d6fe7c18 044e3432
	v_mul_lo_u32 v31, s50, v27                                 // 0000000035a8: d72c001f 02023632
	v_mul_lo_u32 v42, s51, v26                                 // 0000000035b0: d72c002a 02023433
	v_add_co_u32 v23, s1, s52, v23                             // 0000000035b8: d7000117 02022e34
	s_wait_alu depctr_va_vcc(0)                                // 0000000035c0: bf88ff9d
	v_cndmask_b32_e32 v28, 0, v0, vcc_lo                       // 0000000035c4: 02380080
	v_cndmask_b32_e64 v29, 0, s53, vcc_lo                      // 0000000035c8: d501001d 01a86a80
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d0: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s53, 0, s1                  // 0000000035d4: d5207c2d 00050035
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[7:8]                  // 0000000035dc: 7ca80e24
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e0: bf88ff9e
	v_mul_lo_u32 v43, v28, s0                                  // 0000000035e4: d72c002b 0200011c
	v_mul_lo_u32 v29, v29, s2                                  // 0000000035ec: d72c001d 0200051d
	v_mad_co_u64_u32 v[26:27], null, v28, s2, 0                // 0000000035f4: d6fe7c1a 0200051c
	v_add3_u32 v25, v42, v25, v31                              // 0000000035fc: d6550019 047e332a
	v_mul_lo_u32 v42, s51, v23                                 // 000000003604: d72c002a 02022e33
	v_mul_lo_u32 v47, v32, s0                                  // 00000000360c: d72c002f 02000120
	s_wait_alu depctr_va_vcc(0)                                // 000000003614: bf88ff9d
	v_cndmask_b32_e32 v44, 0, v7, vcc_lo                       // 000000003618: 02580e80
	v_cndmask_b32_e64 v46, 0, s53, vcc_lo                      // 00000000361c: d501002e 01a86a80
	v_mul_lo_u32 v48, v36, s0                                  // 000000003624: d72c0030 02000124
	v_mul_lo_u32 v49, v39, s2                                  // 00000000362c: d72c0031 02000527
	v_add3_u32 v27, v27, v43, v29                              // 000000003634: d655001b 0476571b
	v_mad_co_u64_u32 v[28:29], null, s50, v23, v[19:20]        // 00000000363c: d6fe7c1c 044e2e32
	v_mul_lo_u32 v19, s50, v45                                 // 000000003644: d72c0013 02025a32
	v_mul_lo_u32 v43, v30, s2                                  // 00000000364c: d72c002b 0200051e
	v_mul_lo_u32 v45, v2, s0                                   // 000000003654: d72c002d 02000102
	v_mad_co_u64_u32 v[30:31], null, v2, s2, 0                 // 00000000365c: d6fe7c1e 02000502
	v_mul_lo_u32 v2, v33, s2                                   // 000000003664: d72c0002 02000521
	v_mad_co_u64_u32 v[32:33], null, v32, s2, 0                // 00000000366c: d6fe7c20 02000520
	v_mul_lo_u32 v50, v38, s0                                  // 000000003674: d72c0032 02000126
	v_mad_co_u64_u32 v[38:39], null, v38, s2, 0                // 00000000367c: d6fe7c26 02000526
	v_add3_u32 v19, v42, v29, v19                              // 000000003684: d6550013 044e3b2a
	v_mul_lo_u32 v51, v41, s2                                  // 00000000368c: d72c0033 02000529
	v_mul_lo_u32 v52, v40, s0                                  // 000000003694: d72c0034 02000128
	v_add3_u32 v31, v31, v45, v43                              // 00000000369c: d655001f 04ae5b1f
	v_mul_lo_u32 v45, v34, s0                                  // 0000000036a4: d72c002d 02000122
	v_add3_u32 v33, v33, v47, v2                               // 0000000036ac: d6550021 040a5f21
	v_mul_lo_u32 v2, v35, s2                                   // 0000000036b4: d72c0002 02000523
	v_mad_co_u64_u32 v[34:35], null, v34, s2, 0                // 0000000036bc: d6fe7c22 02000522
	v_mul_lo_u32 v47, v37, s2                                  // 0000000036c4: d72c002f 02000525
	v_mad_co_u64_u32 v[36:37], null, v36, s2, 0                // 0000000036cc: d6fe7c24 02000524
	v_mad_co_u64_u32 v[40:41], null, v40, s2, 0                // 0000000036d4: d6fe7c28 02000528
	v_mul_lo_u32 v46, v46, s2                                  // 0000000036dc: d72c002e 0200052e
	v_mul_lo_u32 v53, v44, s0                                  // 0000000036e4: d72c0035 0200012c
	v_mad_co_u64_u32 v[42:43], null, v44, s2, 0                // 0000000036ec: d6fe7c2a 0200052c
	v_add3_u32 v39, v39, v50, v49                              // 0000000036f4: d6550027 04c66527
	v_add3_u32 v35, v35, v45, v2                               // 0000000036fc: d6550023 040a5b23
	v_add_co_u32 v23, vcc_lo, s48, v24                         // 000000003704: d7006a17 02023030
	v_add3_u32 v37, v37, v48, v47                              // 00000000370c: d6550025 04be6125
	v_add3_u32 v41, v41, v52, v51                              // 000000003714: d6550029 04ce6929
	s_wait_alu depctr_va_vcc(0)                                // 00000000371c: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s49, v25, vcc_lo            // 000000003720: d5207c18 01aa3231
	v_add3_u32 v43, v43, v53, v46                              // 000000003728: d655002b 04ba6b2b
	v_lshlrev_b64_e32 v[25:26], 2, v[26:27]                    // 000000003730: 3e323482
	v_add_co_u32 v27, vcc_lo, s46, v28                         // 000000003734: d7006a1b 0202382e
	v_lshlrev_b64_e32 v[29:30], 2, v[30:31]                    // 00000000373c: 3e3a3c82
	v_lshlrev_b64_e32 v[31:32], 2, v[32:33]                    // 000000003740: 3e3e4082
	v_lshlrev_b64_e32 v[33:34], 2, v[34:35]                    // 000000003744: 3e424482
	v_lshlrev_b64_e32 v[35:36], 2, v[36:37]                    // 000000003748: 3e464882
	v_lshlrev_b64_e32 v[37:38], 2, v[38:39]                    // 00000000374c: 3e4a4c82
	v_lshlrev_b64_e32 v[39:40], 2, v[40:41]                    // 000000003750: 3e4e5082
	v_lshlrev_b64_e32 v[41:42], 2, v[42:43]                    // 000000003754: 3e525482
	s_wait_alu depctr_va_vcc(0)                                // 000000003758: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s47, v19, vcc_lo            // 00000000375c: d5207c1c 01aa262f
	v_dual_mov_b32 v47, v20 :: v_dual_mov_b32 v46, v20         // 000000003764: ca100114 2f2e0114
	v_dual_mov_b32 v45, v20 :: v_dual_mov_b32 v44, v20         // 00000000376c: ca100114 2d2c0114
	v_dual_mov_b32 v43, v20 :: v_dual_mov_b32 v2, v20          // 000000003774: ca100114 2b020114
	v_mov_b32_e32 v19, v20                                     // 00000000377c: 7e260314
	s_lshl_b64 s[0:1], s[38:39], 2                             // 000000003780: 84808226
	s_mov_b64 s[2:3], 0                                        // 000000003784: be820180
	v_add_co_u32 v48, vcc_lo, s42, v29                         // 000000003788: d7006a30 02023a2a
	s_wait_alu depctr_va_vcc(0)                                // 000000003790: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s43, v30, vcc_lo            // 000000003794: d5207c31 01aa3c2b
	v_add_co_u32 v50, vcc_lo, s42, v31                         // 00000000379c: d7006a32 02023e2a
	s_wait_alu depctr_va_vcc(0)                                // 0000000037a4: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s43, v32, vcc_lo            // 0000000037a8: d5207c33 01aa402b
	v_add_co_u32 v52, vcc_lo, s42, v33                         // 0000000037b0: d7006a34 0202422a
	s_wait_alu depctr_va_vcc(0)                                // 0000000037b8: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s43, v34, vcc_lo            // 0000000037bc: d5207c35 01aa442b
	v_add_co_u32 v54, vcc_lo, s42, v35                         // 0000000037c4: d7006a36 0202462a
	s_wait_alu depctr_va_vcc(0)                                // 0000000037cc: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s43, v36, vcc_lo            // 0000000037d0: d5207c37 01aa482b
	v_add_co_u32 v64, vcc_lo, s42, v37                         // 0000000037d8: d7006a40 02024a2a
	s_wait_alu depctr_va_vcc(0)                                // 0000000037e0: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s43, v38, vcc_lo            // 0000000037e4: d5207c41 01aa4c2b
	v_add_co_u32 v66, vcc_lo, s42, v39                         // 0000000037ec: d7006a42 02024e2a
	s_clause 0x1                                               // 0000000037f4: bf850001
	global_load_b64 v[56:57], v[27:28], off                    // 0000000037f8: ee05407c 00000038 0000001b
	global_load_b64 v[58:59], v[27:28], off offset:16          // 000000003804: ee05407c 0000003a 0000101b
	s_clause 0x1                                               // 000000003810: bf850001
	global_load_b64 v[60:61], v[23:24], off                    // 000000003814: ee05407c 0000003c 00000017
	global_load_b64 v[62:63], v[23:24], off offset:16          // 000000003820: ee05407c 0000003e 00001017
	s_wait_alu depctr_va_vcc(0)                                // 00000000382c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s43, v40, vcc_lo            // 000000003830: d5207c43 01aa502b
	v_add_co_u32 v68, vcc_lo, s42, v41                         // 000000003838: d7006a44 0202522a
	s_wait_alu depctr_va_vcc(0)                                // 000000003840: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s43, v42, vcc_lo            // 000000003844: d5207c45 01aa542b
	v_add_co_u32 v70, vcc_lo, s42, v25                         // 00000000384c: d7006a46 0202322a
	global_load_b32 v72, v[21:22], off                         // 000000003854: ee05007c 00000048 00000015
	s_wait_alu depctr_va_vcc(0)                                // 000000003860: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s43, v26, vcc_lo            // 000000003864: d5207c47 01aa342b
	s_clause 0x7                                               // 00000000386c: bf850007
	global_load_b32 v73, v[48:49], off                         // 000000003870: ee05007c 00000049 00000030
	global_load_b32 v74, v[50:51], off                         // 00000000387c: ee05007c 0000004a 00000032
	global_load_b32 v75, v[52:53], off                         // 000000003888: ee05007c 0000004b 00000034
	global_load_b32 v76, v[54:55], off                         // 000000003894: ee05007c 0000004c 00000036
	global_load_b32 v64, v[64:65], off                         // 0000000038a0: ee05007c 00000040 00000040
	global_load_b32 v65, v[66:67], off                         // 0000000038ac: ee05007c 00000041 00000042
	global_load_b32 v66, v[68:69], off                         // 0000000038b8: ee05007c 00000042 00000044
	global_load_b32 v67, v[70:71], off                         // 0000000038c4: ee05007c 00000043 00000046
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038d0: bf88ff9e
	v_add_co_u32 v21, vcc_lo, v21, s0                          // 0000000038d4: d7006a15 02000115
	s_add_nc_u64 s[2:3], s[2:3], 32                            // 0000000038dc: a982a002
	s_wait_alu depctr_va_vcc(0)                                // 0000000038e0: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s1, v22, vcc_lo             // 0000000038e4: d5207c16 01aa2c01
	v_add_co_u32 v23, vcc_lo, v23, 32                          // 0000000038ec: d7006a17 02014117
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038f4: bf88ff9e
	v_cmp_lt_i64_e64 s4, s[2:3], s[44:45]                      // 0000000038f8: d4510004 02005802
	s_wait_alu depctr_va_vcc(0)                                // 000000003900: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, 0, v24, vcc_lo              // 000000003904: d5207c18 01aa3080
	v_add_co_u32 v27, vcc_lo, v27, 32                          // 00000000390c: d7006a1b 0201411b
	s_wait_alu depctr_va_vcc(0)                                // 000000003914: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, 0, v28, vcc_lo              // 000000003918: d5207c1c 01aa3880
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000003920: 8b6a047e
	s_add_nc_u64 s[42:43], s[42:43], 4                         // 000000003924: a9aa842a
	s_wait_loadcnt 0xa                                         // 000000003928: bfc0000a
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[56:57], v[60:61], 0// 00000000392c: cc464030 1a027938
	s_wait_loadcnt 0x9                                         // 000000003934: bfc00009
	s_delay_alu instid0(valu_dep_1)                            // 000000003938: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[58:59], v[62:63], v[48:55]// 00000000393c: cc464030 1cc27d3a
	s_wait_loadcnt 0x6                                         // 000000003944: bfc00006
	v_dual_mul_f32 v56, v73, v72 :: v_dual_mul_f32 v57, v72, v74// 000000003948: c8c69149 38389548
	s_wait_loadcnt 0x4                                         // 000000003950: bfc00004
	v_dual_mul_f32 v58, v72, v75 :: v_dual_mul_f32 v59, v72, v76// 000000003954: c8c69748 3a3a9948
	s_wait_loadcnt 0x2                                         // 00000000395c: bfc00002
	v_dual_mul_f32 v60, v72, v64 :: v_dual_mul_f32 v61, v72, v65// 000000003960: c8c68148 3c3c8348
	s_wait_loadcnt 0x0                                         // 000000003968: bfc00000
	v_dual_mul_f32 v62, v72, v66 :: v_dual_mul_f32 v63, v72, v67// 00000000396c: c8c68548 3e3e8748
	v_dual_mul_f32 v48, v48, v56 :: v_dual_mul_f32 v49, v49, v57// 000000003974: c8c67130 30307331
	v_dual_mul_f32 v50, v50, v58 :: v_dual_mul_f32 v51, v51, v59// 00000000397c: c8c67532 32327733
	v_dual_mul_f32 v52, v52, v60 :: v_dual_mul_f32 v53, v53, v61// 000000003984: c8c67934 34347b35
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000398c: bf870214
	v_dual_mul_f32 v54, v54, v62 :: v_dual_mul_f32 v55, v55, v63// 000000003990: c8c67d36 36367f37
	v_dual_add_f32 v20, v20, v48 :: v_dual_add_f32 v47, v47, v49// 000000003998: c9086114 142e632f
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000039a0: bf870214
	v_dual_add_f32 v46, v46, v50 :: v_dual_add_f32 v45, v45, v51// 0000000039a4: c908652e 2e2c672d
	v_dual_add_f32 v44, v44, v52 :: v_dual_add_f32 v43, v43, v53// 0000000039ac: c908692c 2c2a6b2b
	s_delay_alu instid0(valu_dep_4)                            // 0000000039b4: bf870004
	v_dual_add_f32 v2, v2, v54 :: v_dual_add_f32 v19, v19, v55 // 0000000039b8: c9086d02 02126f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c0: bf88ff9e
	s_cbranch_vccnz 65392                                      // 0000000039c4: bfa4ff70 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x1c88>
	v_mul_lo_u32 v21, s39, v15                                 // 0000000039c8: d72c0015 02021e27
	v_mul_lo_u32 v22, s38, v16                                 // 0000000039d0: d72c0016 02022026
	v_mad_co_u64_u32 v[15:16], null, s38, v15, 0               // 0000000039d8: d6fe7c0f 02021e26
	v_mul_lo_u32 v23, s39, v17                                 // 0000000039e0: d72c0017 02022227
	v_mul_lo_u32 v24, s38, v18                                 // 0000000039e8: d72c0018 02022426
	v_mad_co_u64_u32 v[17:18], null, s38, v17, 0               // 0000000039f0: d6fe7c11 02022226
	v_bfe_u32 v25, v20, 16, 1                                  // 0000000039f8: d6100019 02052114
	v_or_b32_e32 v27, 0x400000, v20                            // 000000003a00: 383628ff 00400000
	v_bfe_u32 v26, v47, 16, 1                                  // 000000003a08: d610001a 0205212f
	s_mov_b32 s0, -1                                           // 000000003a10: be8000c1
	v_add3_u32 v16, v16, v22, v21                              // 000000003a14: d6550010 04562d10
	v_lshlrev_b64_e32 v[21:22], 1, v[3:4]                      // 000000003a1c: 3e2a0681
	v_add3_u32 v25, v25, v20, 0x7fff                           // 000000003a20: d6550019 03fe2919 00007fff
	v_add3_u32 v18, v18, v24, v23                              // 000000003a2c: d6550012 045e3112
	v_mul_lo_u32 v24, s39, v13                                 // 000000003a34: d72c0018 02021a27
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 000000003a3c: 3e1e1e81
	v_add3_u32 v26, v26, v47, 0x7fff                           // 000000003a40: d655001a 03fe5f1a 00007fff
	v_or_b32_e32 v23, 0x400000, v47                            // 000000003a4c: 382e5eff 00400000
	v_lshlrev_b64_e32 v[17:18], 1, v[17:18]                    // 000000003a54: 3e222281
	s_delay_alu instid0(valu_dep_4)                            // 000000003a58: bf870004
	v_add_co_u32 v15, vcc_lo, s40, v15                         // 000000003a5c: d7006a0f 02021e28
	s_wait_alu depctr_va_vcc(0)                                // 000000003a64: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s41, v16, vcc_lo            // 000000003a68: d5207c10 01aa2029
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000003a70: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000003a74: bf88ff9d
	v_cndmask_b32_e32 v20, v25, v27, vcc_lo                    // 000000003a78: 02283719
	v_add_co_u32 v15, vcc_lo, v15, v21                         // 000000003a7c: d7006a0f 02022b0f
	v_mul_lo_u32 v25, s38, v14                                 // 000000003a84: d72c0019 02021c26
	v_mad_co_u64_u32 v[13:14], null, s38, v13, 0               // 000000003a8c: d6fe7c0d 02021a26
	s_wait_alu depctr_va_vcc(0)                                // 000000003a94: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, v16, v22, vcc_lo            // 000000003a98: d5207c10 01aa2d10
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000003aa0: 7c305f2f
	global_store_d16_hi_b16 v[15:16], v20, off                 // 000000003aa4: ee09407c 0a000000 0000000f
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab0: bf88ff9d
	v_cndmask_b32_e32 v20, v26, v23, vcc_lo                    // 000000003ab4: 02282f1a
	v_add_co_u32 v15, vcc_lo, s40, v17                         // 000000003ab8: d7006a0f 02022228
	v_add3_u32 v14, v14, v25, v24                              // 000000003ac0: d655000e 0462330e
	s_wait_alu depctr_va_vcc(0)                                // 000000003ac8: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s41, v18, vcc_lo            // 000000003acc: d5207c10 01aa2429
	v_bfe_u32 v17, v46, 16, 1                                  // 000000003ad4: d6100011 0205212e
	v_add_co_u32 v15, vcc_lo, v15, v21                         // 000000003adc: d7006a0f 02022b0f
	v_mul_lo_u32 v23, s39, v11                                 // 000000003ae4: d72c0017 02021627
	v_mul_lo_u32 v24, s38, v12                                 // 000000003aec: d72c0018 02021826
	v_mad_co_u64_u32 v[11:12], null, s38, v11, 0               // 000000003af4: d6fe7c0b 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000003afc: 3e1a1a81
	s_wait_alu depctr_va_vcc(0)                                // 000000003b00: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, v16, v22, vcc_lo            // 000000003b04: d5207c10 01aa2d10
	v_add3_u32 v17, v17, v46, 0x7fff                           // 000000003b0c: d6550011 03fe5d11 00007fff
	v_or_b32_e32 v18, 0x400000, v46                            // 000000003b18: 38245cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000003b20: 7c305d2e
	global_store_d16_hi_b16 v[15:16], v20, off                 // 000000003b24: ee09407c 0a000000 0000000f
	v_add3_u32 v12, v12, v24, v23                              // 000000003b30: d655000c 045e310c
	v_bfe_u32 v16, v45, 16, 1                                  // 000000003b38: d6100010 0205212d
	v_mul_lo_u32 v20, s38, v10                                 // 000000003b40: d72c0014 02021426
	s_wait_alu depctr_va_vcc(0)                                // 000000003b48: bf88ff9d
	v_cndmask_b32_e32 v15, v17, v18, vcc_lo                    // 000000003b4c: 021e2511
	v_add_co_u32 v13, vcc_lo, s40, v13                         // 000000003b50: d7006a0d 02021a28
	s_wait_alu depctr_va_vcc(0)                                // 000000003b58: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s41, v14, vcc_lo            // 000000003b5c: d5207c0e 01aa1c29
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000003b64: 3e161681
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b68: bf8701a3
	v_add_co_u32 v13, vcc_lo, v13, v21                         // 000000003b6c: d7006a0d 02022b0d
	s_wait_alu depctr_va_vcc(0)                                // 000000003b74: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, v14, v22, vcc_lo            // 000000003b78: d5207c0e 01aa2d0e
	v_add3_u32 v16, v16, v45, 0x7fff                           // 000000003b80: d6550010 03fe5b10 00007fff
	v_or_b32_e32 v17, 0x400000, v45                            // 000000003b8c: 38225aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000003b94: 7c305b2d
	v_mul_lo_u32 v18, s39, v9                                  // 000000003b98: d72c0012 02021227
	v_mad_co_u64_u32 v[9:10], null, s38, v9, 0                 // 000000003ba0: d6fe7c09 02021226
	global_store_d16_hi_b16 v[13:14], v15, off                 // 000000003ba8: ee09407c 07800000 0000000d
	v_bfe_u32 v14, v44, 16, 1                                  // 000000003bb4: d610000e 0205212c
	s_wait_alu depctr_va_vcc(0)                                // 000000003bbc: bf88ff9d
	v_cndmask_b32_e32 v13, v16, v17, vcc_lo                    // 000000003bc0: 021a2310
	v_add_co_u32 v11, vcc_lo, s40, v11                         // 000000003bc4: d7006a0b 02021628
	s_wait_alu depctr_va_vcc(0)                                // 000000003bcc: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s41, v12, vcc_lo            // 000000003bd0: d5207c0c 01aa1829
	v_add3_u32 v10, v10, v20, v18                              // 000000003bd8: d655000a 044a290a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003be0: bf8701a3
	v_add_co_u32 v11, vcc_lo, v11, v21                         // 000000003be4: d7006a0b 02022b0b
	s_wait_alu depctr_va_vcc(0)                                // 000000003bec: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, v12, v22, vcc_lo            // 000000003bf0: d5207c0c 01aa2d0c
	v_mul_lo_u32 v16, s39, v5                                  // 000000003bf8: d72c0010 02020a27
	v_mul_lo_u32 v17, s38, v6                                  // 000000003c00: d72c0011 02020c26
	v_mad_co_u64_u32 v[5:6], null, s38, v5, 0                  // 000000003c08: d6fe7c05 02020a26
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000003c10: 3e121281
	v_add3_u32 v14, v14, v44, 0x7fff                           // 000000003c14: d655000e 03fe590e 00007fff
	v_or_b32_e32 v15, 0x400000, v44                            // 000000003c20: 381e58ff 00400000
	global_store_d16_hi_b16 v[11:12], v13, off                 // 000000003c28: ee09407c 06800000 0000000b
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000003c34: 7c30592c
	v_bfe_u32 v12, v43, 16, 1                                  // 000000003c38: d610000c 0205212b
	v_or_b32_e32 v13, 0x400000, v43                            // 000000003c40: 381a56ff 00400000
	v_add3_u32 v6, v6, v17, v16                                // 000000003c48: d6550006 04422306
	v_mul_lo_u32 v18, s38, v1                                  // 000000003c50: d72c0012 02020226
	s_delay_alu instid0(valu_dep_4)                            // 000000003c58: bf870004
	v_add3_u32 v12, v12, v43, 0x7fff                           // 000000003c5c: d655000c 03fe570c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003c68: bf88ff9d
	v_cndmask_b32_e32 v11, v14, v15, vcc_lo                    // 000000003c6c: 02161f0e
	v_add_co_u32 v9, vcc_lo, s40, v9                           // 000000003c70: d7006a09 02021228
	s_wait_alu depctr_va_vcc(0)                                // 000000003c78: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s41, v10, vcc_lo            // 000000003c7c: d5207c0a 01aa1429
	v_mul_lo_u32 v14, s39, v7                                  // 000000003c84: d72c000e 02020e27
	v_mul_lo_u32 v15, s38, v8                                  // 000000003c8c: d72c000f 02021026
	v_mad_co_u64_u32 v[7:8], null, s38, v7, 0                  // 000000003c94: d6fe7c07 02020e26
	v_add_co_u32 v9, vcc_lo, v9, v21                           // 000000003c9c: d7006a09 02022b09
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]                        // 000000003ca4: 3e0a0a81
	s_wait_alu depctr_va_vcc(0)                                // 000000003ca8: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v10, v22, vcc_lo            // 000000003cac: d5207c0a 01aa2d0a
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 000000003cb4: 7c30572b
	v_add3_u32 v8, v8, v15, v14                                // 000000003cb8: d6550008 043a1f08
	v_mul_lo_u32 v15, s39, v0                                  // 000000003cc0: d72c000f 02020027
	v_mad_co_u64_u32 v[0:1], null, s38, v0, 0                  // 000000003cc8: d6fe7c00 02020026
	s_wait_alu depctr_va_vcc(0)                                // 000000003cd0: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo                    // 000000003cd4: 02181b0c
	v_add_co_u32 v16, vcc_lo, s40, v5                          // 000000003cd8: d7006a10 02020a28
	v_bfe_u32 v13, v2, 16, 1                                   // 000000003ce0: d610000d 02052102
	s_wait_alu depctr_va_vcc(0)                                // 000000003ce8: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s41, v6, vcc_lo             // 000000003cec: d5207c11 01aa0c29
	v_lshlrev_b64_e32 v[5:6], 1, v[7:8]                        // 000000003cf4: 3e0a0e81
	v_add_co_u32 v7, vcc_lo, v16, v21                          // 000000003cf8: d7006a07 02022b10
	v_add3_u32 v13, v13, v2, 0x7fff                            // 000000003d00: d655000d 03fe050d 00007fff
	v_or_b32_e32 v14, 0x400000, v2                             // 000000003d0c: 381c04ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d14: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v17, v22, vcc_lo             // 000000003d18: d5207c08 01aa2d11
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000003d20: 7c300502
	v_add3_u32 v1, v1, v18, v15                                // 000000003d24: d6550001 043e2501
	s_clause 0x1                                               // 000000003d2c: bf850001
	global_store_d16_hi_b16 v[9:10], v11, off                  // 000000003d30: ee09407c 05800000 00000009
	global_store_d16_hi_b16 v[7:8], v12, off                   // 000000003d3c: ee09407c 06000000 00000007
	s_wait_alu depctr_va_vcc(0)                                // 000000003d48: bf88ff9d
	v_cndmask_b32_e32 v2, v13, v14, vcc_lo                     // 000000003d4c: 02041d0d
	v_add_co_u32 v5, vcc_lo, s40, v5                           // 000000003d50: d7006a05 02020a28
	s_wait_alu depctr_va_vcc(0)                                // 000000003d58: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s41, v6, vcc_lo              // 000000003d5c: d5207c06 01aa0c29
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003d64: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d68: bf8701a3
	v_add_co_u32 v5, vcc_lo, v5, v21                           // 000000003d6c: d7006a05 02022b05
	s_wait_alu depctr_va_vcc(0)                                // 000000003d74: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, v6, v22, vcc_lo              // 000000003d78: d5207c06 01aa2d06
	s_delay_alu instid0(valu_dep_3)                            // 000000003d80: bf870003
	v_add_co_u32 v0, vcc_lo, s40, v0                           // 000000003d84: d7006a00 02020028
	s_wait_alu depctr_va_vcc(0)                                // 000000003d8c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v1, vcc_lo              // 000000003d90: d5207c01 01aa0229
	global_store_d16_hi_b16 v[5:6], v2, off                    // 000000003d98: ee09407c 01000000 00000005
	s_wait_alu depctr_sa_sdst(0)                               // 000000003da4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003da8: be812000
	s_cbranch_execnz 1                                         // 000000003dac: bfa60001 <tessera_rocm_scaled_matmul_c6867cc7267a885c+0x22b4>
	s_endpgm                                                   // 000000003db0: bfb00000
	v_bfe_u32 v2, v19, 16, 1                                   // 000000003db4: d6100002 02052113
	v_or_b32_e32 v5, 0x400000, v19                             // 000000003dbc: 380a26ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 000000003dc4: 7c302713
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 000000003dc8: bf870133
	v_add3_u32 v6, v2, v19, 0x7fff                             // 000000003dcc: d6550006 03fe2702 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[3:4]                        // 000000003dd8: 3e040681
	s_wait_alu depctr_va_vcc(0)                                // 000000003ddc: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v5, vcc_lo                       // 000000003de0: 02080b06
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003de4: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000003de8: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000003df0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000003df4: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 000000003dfc: ee09407c 02000000 00000000
	s_endpgm                                                   // 000000003e08: bfb00000
	s_code_end                                                 // 000000003e0c: bf9f0000
	s_code_end                                                 // 000000003e10: bf9f0000
	s_code_end                                                 // 000000003e14: bf9f0000
	s_code_end                                                 // 000000003e18: bf9f0000
	s_code_end                                                 // 000000003e1c: bf9f0000
	s_code_end                                                 // 000000003e20: bf9f0000
	s_code_end                                                 // 000000003e24: bf9f0000
	s_code_end                                                 // 000000003e28: bf9f0000
	s_code_end                                                 // 000000003e2c: bf9f0000
	s_code_end                                                 // 000000003e30: bf9f0000
	s_code_end                                                 // 000000003e34: bf9f0000
	s_code_end                                                 // 000000003e38: bf9f0000
	s_code_end                                                 // 000000003e3c: bf9f0000
	s_code_end                                                 // 000000003e40: bf9f0000
	s_code_end                                                 // 000000003e44: bf9f0000
	s_code_end                                                 // 000000003e48: bf9f0000
	s_code_end                                                 // 000000003e4c: bf9f0000
	s_code_end                                                 // 000000003e50: bf9f0000
	s_code_end                                                 // 000000003e54: bf9f0000
	s_code_end                                                 // 000000003e58: bf9f0000
	s_code_end                                                 // 000000003e5c: bf9f0000
	s_code_end                                                 // 000000003e60: bf9f0000
	s_code_end                                                 // 000000003e64: bf9f0000
	s_code_end                                                 // 000000003e68: bf9f0000
	s_code_end                                                 // 000000003e6c: bf9f0000
	s_code_end                                                 // 000000003e70: bf9f0000
	s_code_end                                                 // 000000003e74: bf9f0000
	s_code_end                                                 // 000000003e78: bf9f0000
	s_code_end                                                 // 000000003e7c: bf9f0000
	s_code_end                                                 // 000000003e80: bf9f0000
	s_code_end                                                 // 000000003e84: bf9f0000
	s_code_end                                                 // 000000003e88: bf9f0000
	s_code_end                                                 // 000000003e8c: bf9f0000
	s_code_end                                                 // 000000003e90: bf9f0000
	s_code_end                                                 // 000000003e94: bf9f0000
	s_code_end                                                 // 000000003e98: bf9f0000
	s_code_end                                                 // 000000003e9c: bf9f0000
	s_code_end                                                 // 000000003ea0: bf9f0000
	s_code_end                                                 // 000000003ea4: bf9f0000
	s_code_end                                                 // 000000003ea8: bf9f0000
	s_code_end                                                 // 000000003eac: bf9f0000
	s_code_end                                                 // 000000003eb0: bf9f0000
	s_code_end                                                 // 000000003eb4: bf9f0000
	s_code_end                                                 // 000000003eb8: bf9f0000
	s_code_end                                                 // 000000003ebc: bf9f0000
	s_code_end                                                 // 000000003ec0: bf9f0000
	s_code_end                                                 // 000000003ec4: bf9f0000
	s_code_end                                                 // 000000003ec8: bf9f0000
	s_code_end                                                 // 000000003ecc: bf9f0000
	s_code_end                                                 // 000000003ed0: bf9f0000
	s_code_end                                                 // 000000003ed4: bf9f0000
	s_code_end                                                 // 000000003ed8: bf9f0000
	s_code_end                                                 // 000000003edc: bf9f0000
	s_code_end                                                 // 000000003ee0: bf9f0000
	s_code_end                                                 // 000000003ee4: bf9f0000
	s_code_end                                                 // 000000003ee8: bf9f0000
	s_code_end                                                 // 000000003eec: bf9f0000
	s_code_end                                                 // 000000003ef0: bf9f0000
	s_code_end                                                 // 000000003ef4: bf9f0000
	s_code_end                                                 // 000000003ef8: bf9f0000
	s_code_end                                                 // 000000003efc: bf9f0000
	s_code_end                                                 // 000000003f00: bf9f0000
	s_code_end                                                 // 000000003f04: bf9f0000
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
