
/tmp/tmp5xz68ueq.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_b867c3d0aa38586b>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[36:39], s[0:1], 0xc8                         // 000000001b04: f4004900 f80000c8
	s_load_b64 s[40:41], s[0:1], 0xa8                          // 000000001b0c: f4002a00 f80000a8
	s_load_b64 s[46:47], s[0:1], 0xd8                          // 000000001b14: f4002b80 f80000d8
	s_load_b64 s[48:49], s[0:1], 0x8                           // 000000001b1c: f4002c00 f8000008
	s_load_b64 s[50:51], s[0:1], 0x30                          // 000000001b24: f4002c80 f8000030
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
	v_dual_mov_b32 v4, s55 :: v_dual_and_b32 v39, 15, v0       // 000000001b5c: ca240037 0426008f
	v_bfe_u32 v29, v0, 4, 1                                    // 000000001b64: d610001d 02050900
	s_delay_alu instid0(valu_dep_2)                            // 000000001b6c: bf870002
	v_or_b32_e32 v3, s54, v39                                  // 000000001b70: 38064e36
	s_wait_kmcnt 0x0                                           // 000000001b74: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[36:37]                      // 000000001b78: d4540000 02004800
	v_cmp_gt_i64_e64 s1, s[2:3], s[38:39]                      // 000000001b80: d4540001 02004c02
	v_cmp_lt_i64_e64 s35, s[46:47], 32                         // 000000001b88: d4510023 0201402e
	s_and_b32 s44, s46, 0xffffffe0                             // 000000001b90: 8b2cff2e ffffffe0
	s_mov_b32 s45, s47                                         // 000000001b98: bead002f
	s_or_b32 s0, s0, s1                                        // 000000001b9c: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ba0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001ba4: 8b6a007e
	s_cbranch_vccz 10                                          // 000000001ba8: bfa3000a <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0xd4>
	s_and_b32 s0, s35, exec_lo                                 // 000000001bac: 8b007e23
	s_cselect_b32 s0, 1, 0                                     // 000000001bb0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bb8: bf078100
	s_cbranch_scc1 8                                           // 000000001bbc: bfa20008 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0xe0>
	v_lshl_or_b32 v5, v29, 3, s52                              // 000000001bc0: d6560005 00d1071d
	v_mov_b32_e32 v6, s53                                      // 000000001bc8: 7e0c0235
	s_mov_b32 s0, 0                                            // 000000001bcc: be800080
	s_branch 4                                                 // 000000001bd0: bfa00004 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0xe4>
	s_mov_b32 s0, 0                                            // 000000001bd4: be800080
	s_cbranch_execnz 1569                                      // 000000001bd8: bfa60621 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1960>
	s_branch 2260                                              // 000000001bdc: bfa008d4 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2430>
	s_mov_b32 s0, -1                                           // 000000001be0: be8000c1
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v2, 0              // 000000001be4: ca100080 2b020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bec: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001bf0: 8b007e00
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0             // 000000001bf4: ca100080 1e1e0080
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0             // 000000001bfc: ca100080 20200080
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v1, 0              // 000000001c04: ca100080 22000080
	s_cselect_b32 s0, 1, 0                                     // 000000001c0c: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c10: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c14: bf078100
	s_cbranch_scc1 1224                                        // 000000001c18: bfa204c8 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x143c>
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v0, 3, v29      // 000000001c1c: ca220080 01003a83
	v_mov_b32_e32 v6, s53                                      // 000000001c24: 7e0c0235
	v_or_b32_e32 v9, s52, v39                                  // 000000001c28: 38124e34
	s_lshr_b64 s[4:5], s[46:47], 5                             // 000000001c2c: 8584852e
	s_delay_alu instid0(valu_dep_3)                            // 000000001c30: bf870003
	v_or_b32_e32 v5, s52, v0                                   // 000000001c34: 380a0034
	s_mul_i32 s1, s46, s53                                     // 000000001c38: 9601352e
	s_lshr_b32 s3, s47, 5                                      // 000000001c3c: 8503852f
	v_mul_lo_u32 v11, s47, v9                                  // 000000001c40: d72c000b 0202122f
	v_mad_co_u64_u32 v[7:8], null, s46, v9, v[0:1]             // 000000001c48: d6fe7c07 0402122e
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[5:6]                  // 000000001c50: 7ca80a24
	v_mov_b32_e32 v10, s53                                     // 000000001c54: 7e140235
	v_or_b32_e32 v13, 1, v5                                    // 000000001c58: 381a0a81
	v_mul_lo_u32 v2, s46, v4                                   // 000000001c5c: d72c0002 0202082e
	v_mul_lo_u32 v16, s47, v3                                  // 000000001c64: d72c0010 0202062f
	v_or_b32_e32 v19, 4, v5                                    // 000000001c6c: 38260a84
	v_cndmask_b32_e32 v12, 0, v5, vcc_lo                       // 000000001c70: 02180a80
	v_cndmask_b32_e64 v15, 0, s53, vcc_lo                      // 000000001c74: d501000f 01a86a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c7c: bf88ff9e
	v_add3_u32 v8, v11, v8, s1                                 // 000000001c80: d6550008 0006110b
	v_mov_b32_e32 v20, s53                                     // 000000001c88: 7e280235
	v_mov_b32_e32 v22, s53                                     // 000000001c8c: 7e2c0235
	v_mul_lo_u32 v18, s3, v12                                  // 000000001c90: d72c0012 02021803
	v_mul_lo_u32 v17, s4, v15                                  // 000000001c98: d72c0011 02021e04
	v_mad_co_u64_u32 v[11:12], null, s4, v12, s[42:43]         // 000000001ca0: d6fe7c0b 00aa1804
	v_mov_b32_e32 v15, s53                                     // 000000001ca8: 7e1e0235
	v_cmp_gt_i64_e64 s0, s[36:37], v[9:10]                     // 000000001cac: d4540000 02021224
	v_mad_co_u64_u32 v[9:10], null, s46, v3, v[0:1]            // 000000001cb4: d6fe7c09 0402062e
	v_cmp_gt_i64_e64 s2, s[36:37], v[19:20]                    // 000000001cbc: d4540002 02022624
	v_or_b32_e32 v23, 7, v5                                    // 000000001cc4: 382e0a87
	v_mov_b32_e32 v24, s53                                     // 000000001cc8: 7e300235
	v_cmp_gt_i64_e64 s1, s[38:39], v[3:4]                      // 000000001ccc: d4540001 02020626
	v_add3_u32 v12, v18, v12, v17                              // 000000001cd4: d655000c 04461912
	v_or_b32_e32 v17, 3, v5                                    // 000000001cdc: 38220a83
	v_mov_b32_e32 v14, s53                                     // 000000001ce0: 7e1c0235
	v_add3_u32 v10, v16, v10, v2                               // 000000001ce4: d655000a 040a1510
	v_mov_b32_e32 v18, s53                                     // 000000001cec: 7e240235
	s_wait_alu depctr_va_sdst(0)                               // 000000001cf0: bf88f19f
	v_cndmask_b32_e64 v25, 0, v19, s2                          // 000000001cf4: d5010019 000a2680
	v_or_b32_e32 v19, 5, v5                                    // 000000001cfc: 38260a85
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[13:14]                // 000000001d00: 7ca81a24
	v_or_b32_e32 v14, 2, v5                                    // 000000001d04: 381c0a82
	v_cndmask_b32_e64 v21, 0, s53, s2                          // 000000001d08: d5010015 00086a80
	v_cmp_gt_i64_e64 s2, s[36:37], v[23:24]                    // 000000001d10: d4540002 02022e24
	v_mul_lo_u32 v37, s3, v25                                  // 000000001d18: d72c0025 02023203
	v_cndmask_b32_e64 v27, 0, v3, s1                           // 000000001d20: d501001b 00060680
	s_wait_alu depctr_va_vcc(0)                                // 000000001d28: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v13, vcc_lo                       // 000000001d2c: 02041a80
	v_cndmask_b32_e64 v13, 0, s53, vcc_lo                      // 000000001d30: d501000d 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[14:15]                // 000000001d38: 7ca81c24
	v_mul_lo_u32 v35, s4, v21                                  // 000000001d3c: d72c0023 02022a04
	v_or_b32_e32 v21, 6, v5                                    // 000000001d44: 382a0a86
	s_wait_alu depctr_va_sdst(0)                               // 000000001d48: bf88f19f
	v_cndmask_b32_e64 v40, 0, v23, s2                          // 000000001d4c: d5010028 000a2e80
	v_cndmask_b32_e64 v41, 0, s53, s2                          // 000000001d54: d5010029 00086a80
	v_mul_lo_u32 v30, s4, v13                                  // 000000001d5c: d72c001e 02021a04
	s_wait_alu depctr_va_vcc(0)                                // 000000001d64: bf88ff9d
	v_cndmask_b32_e32 v15, 0, v14, vcc_lo                      // 000000001d68: 021e1c80
	v_cndmask_b32_e64 v16, 0, s53, vcc_lo                      // 000000001d6c: d5010010 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[17:18]                // 000000001d74: 7ca82224
	v_mul_lo_u32 v31, s3, v2                                   // 000000001d78: d72c001f 02020403
	v_mad_co_u64_u32 v[13:14], null, s4, v2, s[42:43]          // 000000001d80: d6fe7c0d 00aa0404
	v_mul_lo_u32 v32, s3, v15                                  // 000000001d88: d72c0020 02021e03
	v_mul_lo_u32 v2, s4, v16                                   // 000000001d90: d72c0002 02022004
	v_mad_co_u64_u32 v[15:16], null, s4, v15, s[42:43]         // 000000001d98: d6fe7c0f 00aa1e04
	s_wait_alu depctr_va_vcc(0)                                // 000000001da0: bf88ff9d
	v_cndmask_b32_e32 v17, 0, v17, vcc_lo                      // 000000001da4: 02222280
	v_cndmask_b32_e64 v18, 0, s53, vcc_lo                      // 000000001da8: d5010012 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[19:20]                // 000000001db0: 7ca82624
	v_mul_lo_u32 v41, s4, v41                                  // 000000001db4: d72c0029 02025204
	v_mul_lo_u32 v44, s3, v40                                  // 000000001dbc: d72c002c 02025003
	v_mul_lo_u32 v34, s3, v17                                  // 000000001dc4: d72c0022 02022203
	v_mul_lo_u32 v33, s4, v18                                  // 000000001dcc: d72c0021 02022404
	v_mad_co_u64_u32 v[17:18], null, s4, v17, s[42:43]         // 000000001dd4: d6fe7c11 00aa2204
	s_wait_alu depctr_va_vcc(0)                                // 000000001ddc: bf88ff9d
	v_cndmask_b32_e32 v26, 0, v19, vcc_lo                      // 000000001de0: 02342680
	v_cndmask_b32_e64 v36, 0, s53, vcc_lo                      // 000000001de4: d5010024 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[21:22]                // 000000001dec: 7ca82a24
	v_mad_co_u64_u32 v[19:20], null, s4, v25, s[42:43]         // 000000001df0: d6fe7c13 00aa3204
	v_cndmask_b32_e64 v28, 0, v4, s1                           // 000000001df8: d501001c 00060880
	v_mul_lo_u32 v38, s3, v26                                  // 000000001e00: d72c0026 02023403
	v_mul_lo_u32 v36, s4, v36                                  // 000000001e08: d72c0024 02024804
	v_add3_u32 v14, v31, v14, v30                              // 000000001e10: d655000e 047a1d1f
	s_wait_alu depctr_va_vcc(0)                                // 000000001e18: bf88ff9d
	v_cndmask_b32_e32 v24, 0, v21, vcc_lo                      // 000000001e1c: 02302a80
	v_cndmask_b32_e64 v25, 0, s53, vcc_lo                      // 000000001e20: d5010019 01a86a80
	v_mad_co_u64_u32 v[21:22], null, s4, v26, s[42:43]         // 000000001e28: d6fe7c15 00aa3404
	v_add_co_u32 v27, vcc_lo, s56, v27                         // 000000001e30: d7006a1b 02023638
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001e38: bf870214
	v_mul_lo_u32 v43, s3, v24                                  // 000000001e3c: d72c002b 02023003
	v_mul_lo_u32 v42, s4, v25                                  // 000000001e44: d72c002a 02023204
	v_mad_co_u64_u32 v[23:24], null, s4, v24, s[42:43]         // 000000001e4c: d6fe7c17 00aa3004
	v_mad_co_u64_u32 v[25:26], null, s4, v40, s[42:43]         // 000000001e54: d6fe7c19 00aa5004
	s_wait_alu depctr_va_vcc(0)                                // 000000001e5c: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s57, v28, vcc_lo            // 000000001e60: d5207c1c 01aa3839
	v_add3_u32 v16, v32, v16, v2                               // 000000001e68: d6550010 040a2120
	v_add3_u32 v18, v34, v18, v33                              // 000000001e70: d6550012 04862522
	v_add3_u32 v20, v37, v20, v35                              // 000000001e78: d6550014 048e2925
	v_add3_u32 v22, v38, v22, v36                              // 000000001e80: d6550016 04922d26
	v_add3_u32 v24, v43, v24, v42                              // 000000001e88: d6550018 04aa312b
	v_add3_u32 v26, v44, v26, v41                              // 000000001e90: d655001a 04a6352c
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v33, v1           // 000000001e98: ca100101 22200101
	v_dual_mov_b32 v32, v1 :: v_dual_mov_b32 v31, v1           // 000000001ea0: ca100101 201e0101
	v_mov_b32_e32 v30, v1                                      // 000000001ea8: 7e3c0301
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v43, v1            // 000000001eac: ca100101 022a0101
	s_mov_b64 s[58:59], 0                                      // 000000001eb4: beba0180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_4)// 000000001eb8: bf870249
	v_or_b32_e32 v35, s58, v0                                  // 000000001ebc: 3846003a
	v_add_co_u32 v42, vcc_lo, v7, s58                          // 000000001ec0: d7006a2a 02007507
	v_mov_b32_e32 v36, s59                                     // 000000001ec8: 7e48023b
	v_mov_b32_e32 v38, s59                                     // 000000001ecc: 7e4c023b
	v_or_b32_e32 v37, 1, v35                                   // 000000001ed0: 384a4681
	s_wait_alu depctr_va_vcc(0)                                // 000000001ed4: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, s59, v8, vcc_lo             // 000000001ed8: d5207c36 01aa103b
	v_cmp_gt_i64_e64 s9, s[46:47], v[35:36]                    // 000000001ee0: d4540009 0202462e
	v_or_b32_e32 v46, 3, v35                                   // 000000001ee8: 385c4683
	v_cmp_gt_i64_e64 s10, s[46:47], v[37:38]                   // 000000001eec: d454000a 02024a2e
	v_add_co_u32 v37, vcc_lo, v42, 1                           // 000000001ef4: d7006a25 0201032a
	s_wait_alu depctr_va_vcc(0)                                // 000000001efc: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, 0, v54, vcc_lo              // 000000001f00: d5207c26 01aa6c80
	s_and_b32 vcc_lo, s0, s9                                   // 000000001f08: 8b6a0900
	s_and_b32 s2, s0, s10                                      // 000000001f0c: 8b020a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f10: bf88ff9e
	v_cndmask_b32_e32 v41, 0, v42, vcc_lo                      // 000000001f14: 02525480
	v_dual_cndmask_b32 v40, 0, v54 :: v_dual_mov_b32 v47, s59  // 000000001f18: ca506c80 282e003b
	v_cndmask_b32_e64 v45, 0, v37, s2                          // 000000001f20: d501002d 000a4a80
	v_cndmask_b32_e64 v44, 0, v38, s2                          // 000000001f28: d501002c 000a4c80
	s_delay_alu instid0(valu_dep_4)                            // 000000001f30: bf870004
	v_add_co_u32 v37, s3, s48, v41                             // 000000001f34: d7000325 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000001f3c: bf88f19f
	v_add_co_ci_u32_e64 v38, null, s49, v40, s3                // 000000001f40: d5207c26 000e5031
	v_add_co_u32 v40, s3, s48, v45                             // 000000001f48: d7000328 02025a30
	s_wait_alu depctr_va_sdst(0)                               // 000000001f50: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s49, v44, s3                // 000000001f54: d5207c29 000e5831
	v_or_b32_e32 v44, 2, v35                                   // 000000001f5c: 38584682
	v_mov_b32_e32 v45, s59                                     // 000000001f60: 7e5a023b
	v_cmp_gt_i64_e64 s13, s[46:47], v[46:47]                   // 000000001f64: d454000d 02025c2e
	v_add_co_u32 v48, s3, v42, 2                               // 000000001f6c: d7000330 0201052a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f74: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v54, s3                  // 000000001f78: d5207c31 000e6c80
	v_cmp_gt_i64_e64 s11, s[46:47], v[44:45]                   // 000000001f80: d454000b 0202582e
	v_add_co_u32 v44, s3, v42, 3                               // 000000001f88: d700032c 0201072a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f90: bf88f19f
	v_add_co_ci_u32_e64 v45, null, 0, v54, s3                  // 000000001f94: d5207c2d 000e6c80
	s_and_b32 s3, s0, s13                                      // 000000001f9c: 8b030d00
	s_and_b32 s4, s0, s11                                      // 000000001fa0: 8b040b00
	v_or_b32_e32 v50, 5, v35                                   // 000000001fa4: 38644685
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa8: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v48, s4                          // 000000001fac: d501002f 00126080
	v_cndmask_b32_e64 v46, 0, v49, s4                          // 000000001fb4: d501002e 00126280
	v_cndmask_b32_e64 v49, 0, v44, s3                          // 000000001fbc: d5010031 000e5880
	v_cndmask_b32_e64 v48, 0, v45, s3                          // 000000001fc4: d5010030 000e5a80
	v_mov_b32_e32 v51, s59                                     // 000000001fcc: 7e66023b
	v_add_co_u32 v44, s5, s48, v47                             // 000000001fd0: d700052c 02025e30
	s_wait_alu depctr_va_sdst(0)                               // 000000001fd8: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s49, v46, s5                // 000000001fdc: d5207c2d 00165c31
	v_add_co_u32 v46, s5, s48, v49                             // 000000001fe4: d700052e 02026230
	s_wait_alu depctr_va_sdst(0)                               // 000000001fec: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s49, v48, s5                // 000000001ff0: d5207c2f 00166031
	v_or_b32_e32 v48, 4, v35                                   // 000000001ff8: 38604684
	v_mov_b32_e32 v49, s59                                     // 000000001ffc: 7e62023b
	v_add_co_u32 v52, s5, v42, 4                               // 000000002000: d7000534 0201092a
	v_cmp_gt_i64_e64 s15, s[46:47], v[50:51]                   // 000000002008: d454000f 0202642e
	s_wait_alu depctr_va_sdst(0)                               // 000000002010: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v54, s5                  // 000000002014: d5207c35 00166c80
	v_cmp_gt_i64_e64 s14, s[46:47], v[48:49]                   // 00000000201c: d454000e 0202602e
	v_add_co_u32 v48, s5, v42, 5                               // 000000002024: d7000530 02010b2a
	s_wait_alu depctr_va_sdst(0)                               // 00000000202c: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v54, s5                  // 000000002030: d5207c31 00166c80
	s_and_b32 s6, s0, s15                                      // 000000002038: 8b060f00
	s_and_b32 s5, s0, s14                                      // 00000000203c: 8b050e00
	s_and_b32 s9, s1, s9                                       // 000000002040: 8b090901
	s_wait_alu depctr_sa_sdst(0)                               // 000000002044: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v52, s5                          // 000000002048: d5010033 00166880
	v_cndmask_b32_e64 v50, 0, v53, s5                          // 000000002050: d5010032 00166a80
	v_cndmask_b32_e64 v53, 0, v48, s6                          // 000000002058: d5010035 001a6080
	v_cndmask_b32_e64 v52, 0, v49, s6                          // 000000002060: d5010034 001a6280
	s_and_b32 s10, s1, s10                                     // 000000002068: 8b0a0a01
	v_add_co_u32 v48, s7, s48, v51                             // 00000000206c: d7000730 02026630
	s_wait_alu depctr_va_sdst(0)                               // 000000002074: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s49, v50, s7                // 000000002078: d5207c31 001e6431
	v_add_co_u32 v50, s7, s48, v53                             // 000000002080: d7000732 02026a30
	s_wait_alu depctr_va_sdst(0)                               // 000000002088: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s49, v52, s7                // 00000000208c: d5207c33 001e6831
	v_or_b32_e32 v52, 6, v35                                   // 000000002094: 38684686
	v_mov_b32_e32 v53, s59                                     // 000000002098: 7e6a023b
	v_or_b32_e32 v35, 7, v35                                   // 00000000209c: 38464687
	v_add_co_u32 v55, s7, v42, 6                               // 0000000020a0: d7000737 02010d2a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a8: bf88f19f
	v_add_co_ci_u32_e64 v56, null, 0, v54, s7                  // 0000000020ac: d5207c38 001e6c80
	v_cmp_gt_i64_e64 s16, s[46:47], v[52:53]                   // 0000000020b4: d4540010 0202682e
	v_cmp_gt_i64_e64 s17, s[46:47], v[35:36]                   // 0000000020bc: d4540011 0202462e
	v_add_co_u32 v35, s7, v42, 7                               // 0000000020c4: d7000723 02010f2a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020cc: bf88f19f
	v_add_co_ci_u32_e64 v36, null, 0, v54, s7                  // 0000000020d0: d5207c24 001e6c80
	s_and_b32 s7, s0, s16                                      // 0000000020d8: 8b071000
	s_and_b32 s8, s0, s17                                      // 0000000020dc: 8b081100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020e0: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v55, s7                          // 0000000020e4: d5010034 001e6e80
	v_cndmask_b32_e64 v42, 0, v56, s7                          // 0000000020ec: d501002a 001e7080
	v_cndmask_b32_e64 v35, 0, v35, s8                          // 0000000020f4: d5010023 00224680
	v_cndmask_b32_e64 v36, 0, v36, s8                          // 0000000020fc: d5010024 00224880
	s_or_b32 s60, s58, 16                                      // 000000002104: 8c3c903a
	v_add_co_u32 v52, s12, s48, v52                            // 000000002108: d7000c34 02026830
	s_wait_alu depctr_va_sdst(0)                               // 000000002110: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s49, v42, s12               // 000000002114: d5207c35 00325431
	v_add_co_u32 v54, s12, s48, v35                            // 00000000211c: d7000c36 02024630
	s_wait_alu depctr_va_sdst(0)                               // 000000002124: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s49, v36, s12               // 000000002128: d5207c37 00324831
	v_add_co_u32 v42, s12, v9, s58                             // 000000002130: d7000c2a 02007509
	s_wait_alu depctr_va_sdst(0)                               // 000000002138: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s59, v10, s12               // 00000000213c: d5207c38 0032143b
	s_clause 0x7                                               // 000000002144: bf850007
	global_load_d16_u8 v35, v[37:38], off                      // 000000002148: ee07807c 00000023 00000025
	global_load_d16_hi_u8 v35, v[40:41], off                   // 000000002154: ee08407c 00000023 00000028
	global_load_d16_u8 v36, v[44:45], off                      // 000000002160: ee07807c 00000024 0000002c
	global_load_d16_hi_u8 v36, v[46:47], off                   // 00000000216c: ee08407c 00000024 0000002e
	global_load_d16_u8 v37, v[48:49], off                      // 000000002178: ee07807c 00000025 00000030
	global_load_d16_hi_u8 v37, v[50:51], off                   // 000000002184: ee08407c 00000025 00000032
	global_load_d16_u8 v38, v[52:53], off                      // 000000002190: ee07807c 00000026 00000034
	global_load_d16_hi_u8 v38, v[54:55], off                   // 00000000219c: ee08407c 00000026 00000036
	v_add_co_u32 v40, s12, v42, 1                              // 0000000021a8: d7000c28 0201032a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b0: bf88f19f
	v_add_co_ci_u32_e64 v41, null, 0, v56, s12                 // 0000000021b4: d5207c29 00327080
	v_cndmask_b32_e64 v45, 0, v42, s9                          // 0000000021bc: d501002d 00265480
	v_cndmask_b32_e64 v44, 0, v56, s9                          // 0000000021c4: d501002c 00267080
	v_cndmask_b32_e64 v47, 0, v40, s10                         // 0000000021cc: d501002f 002a5080
	s_delay_alu instid0(valu_dep_4)                            // 0000000021d4: bf870004
	v_cndmask_b32_e64 v46, 0, v41, s10                         // 0000000021d8: d501002e 002a5280
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021e0: bf88ff9e
	v_or_b32_e32 v58, s60, v0                                  // 0000000021e4: 3874003c
	v_add_co_u32 v40, s12, s50, v45                            // 0000000021e8: d7000c28 02025a32
	s_wait_alu depctr_va_sdst(0)                               // 0000000021f0: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s51, v44, s12               // 0000000021f4: d5207c29 00325833
	v_add_co_u32 v44, s12, s50, v47                            // 0000000021fc: d7000c2c 02025e32
	s_wait_alu depctr_va_sdst(0)                               // 000000002204: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s51, v46, s12               // 000000002208: d5207c2d 00325c33
	v_add_co_u32 v46, s12, v42, 2                              // 000000002210: d7000c2e 0201052a
	s_wait_alu depctr_va_sdst(0)                               // 000000002218: bf88f19f
	v_add_co_ci_u32_e64 v47, null, 0, v56, s12                 // 00000000221c: d5207c2f 00327080
	v_add_co_u32 v48, s12, v42, 3                              // 000000002224: d7000c30 0201072a
	s_wait_alu depctr_va_sdst(0)                               // 00000000222c: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v56, s12                 // 000000002230: d5207c31 00327080
	s_and_b32 s12, s1, s11                                     // 000000002238: 8b0c0b01
	s_and_b32 s11, s1, s13                                     // 00000000223c: 8b0b0d01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002240: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v46, s12                         // 000000002244: d501002e 00325c80
	v_cndmask_b32_e64 v47, 0, v47, s12                         // 00000000224c: d501002f 00325e80
	v_cndmask_b32_e64 v48, 0, v48, s11                         // 000000002254: d5010030 002e6080
	v_cndmask_b32_e64 v49, 0, v49, s11                         // 00000000225c: d5010031 002e6280
	v_mov_b32_e32 v59, s59                                     // 000000002264: 7e76023b
	v_add_co_u32 v46, s13, s50, v46                            // 000000002268: d7000d2e 02025c32
	s_wait_alu depctr_va_sdst(0)                               // 000000002270: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s51, v47, s13               // 000000002274: d5207c2f 00365e33
	v_add_co_u32 v48, s13, s50, v48                            // 00000000227c: d7000d30 02026032
	s_wait_alu depctr_va_sdst(0)                               // 000000002284: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s51, v49, s13               // 000000002288: d5207c31 00366233
	v_add_co_u32 v50, s13, v42, 4                              // 000000002290: d7000d32 0201092a
	s_wait_alu depctr_va_sdst(0)                               // 000000002298: bf88f19f
	v_add_co_ci_u32_e64 v51, null, 0, v56, s13                 // 00000000229c: d5207c33 00367080
	v_add_co_u32 v52, s13, v42, 5                              // 0000000022a4: d7000d34 02010b2a
	s_wait_alu depctr_va_sdst(0)                               // 0000000022ac: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v56, s13                 // 0000000022b0: d5207c35 00367080
	s_and_b32 s13, s1, s14                                     // 0000000022b8: 8b0d0e01
	s_and_b32 s14, s1, s15                                     // 0000000022bc: 8b0e0f01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022c0: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v50, s13                         // 0000000022c4: d5010032 00366480
	v_cndmask_b32_e64 v51, 0, v51, s13                         // 0000000022cc: d5010033 00366680
	v_cndmask_b32_e64 v52, 0, v52, s14                         // 0000000022d4: d5010034 003a6880
	v_cndmask_b32_e64 v53, 0, v53, s14                         // 0000000022dc: d5010035 003a6a80
	v_cmp_gt_i64_e64 s25, s[46:47], v[58:59]                   // 0000000022e4: d4540019 0202742e
	v_add_co_u32 v50, s15, s50, v50                            // 0000000022ec: d7000f32 02026432
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f4: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s51, v51, s15               // 0000000022f8: d5207c33 003e6633
	v_add_co_u32 v52, s15, s50, v52                            // 000000002300: d7000f34 02026832
	s_wait_alu depctr_va_sdst(0)                               // 000000002308: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s51, v53, s15               // 00000000230c: d5207c35 003e6a33
	v_add_co_u32 v54, s15, v42, 6                              // 000000002314: d7000f36 02010d2a
	s_wait_alu depctr_va_sdst(0)                               // 00000000231c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v56, s15                 // 000000002320: d5207c37 003e7080
	v_add_co_u32 v42, s15, v42, 7                              // 000000002328: d7000f2a 02010f2a
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v56, null, 0, v56, s15                 // 000000002334: d5207c38 003e7080
	s_and_b32 s15, s1, s16                                     // 00000000233c: 8b0f1001
	s_and_b32 s16, s1, s17                                     // 000000002340: 8b101101
	s_wait_alu depctr_sa_sdst(0)                               // 000000002344: bf88ff9e
	v_cndmask_b32_e64 v54, 0, v54, s15                         // 000000002348: d5010036 003e6c80
	v_cndmask_b32_e64 v55, 0, v55, s15                         // 000000002350: d5010037 003e6e80
	v_cndmask_b32_e64 v42, 0, v42, s16                         // 000000002358: d501002a 00425480
	v_cndmask_b32_e64 v57, 0, v56, s16                         // 000000002360: d5010039 00427080
	s_delay_alu instid0(valu_dep_4)                            // 000000002368: bf870004
	v_add_co_u32 v54, s17, s50, v54                            // 00000000236c: d7001136 02026c32
	s_wait_alu depctr_va_sdst(0)                               // 000000002374: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s51, v55, s17               // 000000002378: d5207c37 00466e33
	v_add_co_u32 v56, s17, s50, v42                            // 000000002380: d7001138 02025432
	s_wait_alu depctr_va_sdst(0)                               // 000000002388: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s51, v57, s17               // 00000000238c: d5207c39 00467233
	s_clause 0x7                                               // 000000002394: bf850007
	global_load_d16_u8 v40, v[40:41], off                      // 000000002398: ee07807c 00000028 00000028
	global_load_d16_hi_u8 v40, v[44:45], off                   // 0000000023a4: ee08407c 00000028 0000002c
	global_load_d16_u8 v41, v[46:47], off                      // 0000000023b0: ee07807c 00000029 0000002e
	global_load_d16_hi_u8 v41, v[48:49], off                   // 0000000023bc: ee08407c 00000029 00000030
	global_load_d16_u8 v42, v[50:51], off                      // 0000000023c8: ee07807c 0000002a 00000032
	global_load_d16_hi_u8 v42, v[52:53], off                   // 0000000023d4: ee08407c 0000002a 00000034
	global_load_d16_u8 v44, v[54:55], off                      // 0000000023e0: ee07807c 0000002c 00000036
	global_load_d16_hi_u8 v44, v[56:57], off                   // 0000000023ec: ee08407c 0000002c 00000038
	v_or_b32_e32 v45, 1, v58                                   // 0000000023f8: 385a7481
	v_mov_b32_e32 v46, s59                                     // 0000000023fc: 7e5c023b
	v_add_co_u32 v53, s17, v7, s60                             // 000000002400: d7001135 02007907
	s_wait_alu depctr_va_sdst(0)                               // 000000002408: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s59, v8, s17                // 00000000240c: d5207c3e 0046103b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002414: bf870193
	v_cmp_gt_i64_e64 s26, s[46:47], v[45:46]                   // 000000002418: d454001a 02025a2e
	v_add_co_u32 v45, s17, v53, 1                              // 000000002420: d700112d 02010335
	s_wait_alu depctr_va_sdst(0)                               // 000000002428: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000242c: bf870003
	v_add_co_ci_u32_e64 v46, null, 0, v62, s17                 // 000000002430: d5207c2e 00467c80
	s_and_b32 s17, s0, s25                                     // 000000002438: 8b111900
	s_and_b32 s18, s0, s26                                     // 00000000243c: 8b121a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002440: bf88ff9e
	v_cndmask_b32_e64 v48, 0, v53, s17                         // 000000002444: d5010030 00466a80
	v_cndmask_b32_e64 v47, 0, v62, s17                         // 00000000244c: d501002f 00467c80
	v_cndmask_b32_e64 v50, 0, v45, s18                         // 000000002454: d5010032 004a5a80
	v_cndmask_b32_e64 v49, 0, v46, s18                         // 00000000245c: d5010031 004a5c80
	v_or_b32_e32 v51, 3, v58                                   // 000000002464: 38667483
	v_add_co_u32 v45, s19, s48, v48                            // 000000002468: d700132d 02026030
	s_wait_alu depctr_va_sdst(0)                               // 000000002470: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s49, v47, s19               // 000000002474: d5207c2e 004e5e31
	v_add_co_u32 v47, s19, s48, v50                            // 00000000247c: d700132f 02026430
	s_wait_alu depctr_va_sdst(0)                               // 000000002484: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s49, v49, s19               // 000000002488: d5207c30 004e6231
	v_or_b32_e32 v49, 2, v58                                   // 000000002490: 38627482
	v_mov_b32_e32 v50, s59                                     // 000000002494: 7e64023b
	v_mov_b32_e32 v52, s59                                     // 000000002498: 7e68023b
	v_add_co_u32 v54, s19, v53, 2                              // 00000000249c: d7001336 02010535
	s_wait_alu depctr_va_sdst(0)                               // 0000000024a4: bf88f19f
	v_add_co_ci_u32_e64 v55, null, 0, v62, s19                 // 0000000024a8: d5207c37 004e7c80
	v_cmp_gt_i64_e64 s27, s[46:47], v[49:50]                   // 0000000024b0: d454001b 0202622e
	v_cmp_gt_i64_e64 s29, s[46:47], v[51:52]                   // 0000000024b8: d454001d 0202662e
	v_add_co_u32 v49, s19, v53, 3                              // 0000000024c0: d7001331 02010735
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c8: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v62, s19                 // 0000000024cc: d5207c32 004e7c80
	s_and_b32 s20, s0, s27                                     // 0000000024d4: 8b141b00
	s_and_b32 s19, s0, s29                                     // 0000000024d8: 8b131d00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024dc: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v54, s20                         // 0000000024e0: d5010034 00526c80
	v_cndmask_b32_e64 v51, 0, v55, s20                         // 0000000024e8: d5010033 00526e80
	v_cndmask_b32_e64 v54, 0, v49, s19                         // 0000000024f0: d5010036 004e6280
	v_cndmask_b32_e64 v55, 0, v50, s19                         // 0000000024f8: d5010037 004e6480
	v_or_b32_e32 v56, 5, v58                                   // 000000002500: 38707485
	v_add_co_u32 v49, s21, s48, v52                            // 000000002504: d7001531 02026830
	s_wait_alu depctr_va_sdst(0)                               // 00000000250c: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s49, v51, s21               // 000000002510: d5207c32 00566631
	v_or_b32_e32 v51, 4, v58                                   // 000000002518: 38667484
	v_dual_mov_b32 v52, s59 :: v_dual_mov_b32 v57, s59         // 00000000251c: ca10003b 3438003b
	v_add_co_u32 v54, s21, s48, v54                            // 000000002524: d7001536 02026c30
	s_wait_alu depctr_va_sdst(0)                               // 00000000252c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s49, v55, s21               // 000000002530: d5207c37 00566e31
	s_delay_alu instid0(valu_dep_3)                            // 000000002538: bf870003
	v_cmp_gt_i64_e64 s30, s[46:47], v[51:52]                   // 00000000253c: d454001e 0202662e
	v_add_co_u32 v60, s21, v53, 4                              // 000000002544: d700153c 02010935
	v_cmp_gt_i64_e64 s31, s[46:47], v[56:57]                   // 00000000254c: d454001f 0202702e
	s_wait_alu depctr_va_sdst(0)                               // 000000002554: bf88f19f
	v_add_co_ci_u32_e64 v61, null, 0, v62, s21                 // 000000002558: d5207c3d 00567c80
	v_add_co_u32 v51, s21, v53, 5                              // 000000002560: d7001533 02010b35
	s_wait_alu depctr_va_sdst(0)                               // 000000002568: bf88f19f
	v_add_co_ci_u32_e64 v52, null, 0, v62, s21                 // 00000000256c: d5207c34 00567c80
	s_and_b32 s21, s0, s30                                     // 000000002574: 8b151e00
	s_and_b32 s22, s0, s31                                     // 000000002578: 8b161f00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000257c: bf88ff9e
	v_cndmask_b32_e64 v56, 0, v60, s21                         // 000000002580: d5010038 00567880
	v_cndmask_b32_e64 v57, 0, v61, s21                         // 000000002588: d5010039 00567a80
	v_cndmask_b32_e64 v51, 0, v51, s22                         // 000000002590: d5010033 005a6680
	v_cndmask_b32_e64 v52, 0, v52, s22                         // 000000002598: d5010034 005a6880
	s_and_b32 s25, s1, s25                                     // 0000000025a0: 8b191901
	v_add_co_u32 v56, s23, s48, v56                            // 0000000025a4: d7001738 02027030
	s_wait_alu depctr_va_sdst(0)                               // 0000000025ac: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s49, v57, s23               // 0000000025b0: d5207c39 005e7231
	v_add_co_u32 v60, s23, s48, v51                            // 0000000025b8: d700173c 02026630
	s_wait_alu depctr_va_sdst(0)                               // 0000000025c0: bf88f19f
	v_add_co_ci_u32_e64 v61, null, s49, v52, s23               // 0000000025c4: d5207c3d 005e6831
	v_or_b32_e32 v51, 6, v58                                   // 0000000025cc: 38667486
	v_mov_b32_e32 v52, s59                                     // 0000000025d0: 7e68023b
	v_or_b32_e32 v58, 7, v58                                   // 0000000025d4: 38747487
	v_add_co_u32 v63, s23, v53, 6                              // 0000000025d8: d700173f 02010d35
	s_wait_alu depctr_va_sdst(0)                               // 0000000025e0: bf88f19f
	v_add_co_ci_u32_e64 v64, null, 0, v62, s23                 // 0000000025e4: d5207c40 005e7c80
	v_cmp_gt_i64_e64 s33, s[46:47], v[51:52]                   // 0000000025ec: d4540021 0202662e
	v_cmp_gt_i64_e64 s34, s[46:47], v[58:59]                   // 0000000025f4: d4540022 0202742e
	v_add_co_u32 v51, s23, v53, 7                              // 0000000025fc: d7001733 02010f35
	s_wait_alu depctr_va_sdst(0)                               // 000000002604: bf88f19f
	v_add_co_ci_u32_e64 v52, null, 0, v62, s23                 // 000000002608: d5207c34 005e7c80
	s_and_b32 s23, s0, s33                                     // 000000002610: 8b172100
	s_and_b32 s24, s0, s34                                     // 000000002614: 8b182200
	s_wait_alu depctr_sa_sdst(0)                               // 000000002618: bf88ff9e
	v_cndmask_b32_e64 v58, 0, v63, s23                         // 00000000261c: d501003a 005e7e80
	v_cndmask_b32_e64 v53, 0, v64, s23                         // 000000002624: d5010035 005e8080
	v_cndmask_b32_e64 v51, 0, v51, s24                         // 00000000262c: d5010033 00626680
	v_cndmask_b32_e64 v52, 0, v52, s24                         // 000000002634: d5010034 00626880
	s_and_b32 s26, s1, s26                                     // 00000000263c: 8b1a1a01
	v_add_co_u32 v58, s28, s48, v58                            // 000000002640: d7001c3a 02027430
	s_wait_alu depctr_va_sdst(0)                               // 000000002648: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s49, v53, s28               // 00000000264c: d5207c3b 00726a31
	v_add_co_u32 v62, s28, s48, v51                            // 000000002654: d7001c3e 02026630
	s_wait_alu depctr_va_sdst(0)                               // 00000000265c: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s49, v52, s28               // 000000002660: d5207c3f 00726831
	v_add_co_u32 v51, s28, v9, s60                             // 000000002668: d7001c33 02007909
	s_wait_alu depctr_va_sdst(0)                               // 000000002670: bf88f19f
	v_add_co_ci_u32_e64 v65, null, s59, v10, s28               // 000000002674: d5207c41 0072143b
	s_clause 0x7                                               // 00000000267c: bf850007
	global_load_d16_u8 v52, v[45:46], off                      // 000000002680: ee07807c 00000034 0000002d
	global_load_d16_hi_u8 v52, v[47:48], off                   // 00000000268c: ee08407c 00000034 0000002f
	global_load_d16_u8 v53, v[49:50], off                      // 000000002698: ee07807c 00000035 00000031
	global_load_d16_hi_u8 v53, v[54:55], off                   // 0000000026a4: ee08407c 00000035 00000036
	global_load_d16_u8 v54, v[56:57], off                      // 0000000026b0: ee07807c 00000036 00000038
	global_load_d16_hi_u8 v54, v[60:61], off                   // 0000000026bc: ee08407c 00000036 0000003c
	global_load_d16_u8 v55, v[58:59], off                      // 0000000026c8: ee07807c 00000037 0000003a
	global_load_d16_hi_u8 v55, v[62:63], off                   // 0000000026d4: ee08407c 00000037 0000003e
	v_add_co_u32 v45, s28, v51, 1                              // 0000000026e0: d7001c2d 02010333
	s_wait_alu depctr_va_sdst(0)                               // 0000000026e8: bf88f19f
	v_add_co_ci_u32_e64 v46, null, 0, v65, s28                 // 0000000026ec: d5207c2e 00728280
	v_cndmask_b32_e64 v48, 0, v51, s25                         // 0000000026f4: d5010030 00666680
	v_cndmask_b32_e64 v47, 0, v65, s25                         // 0000000026fc: d501002f 00668280
	s_wait_alu depctr_sa_sdst(0)                               // 000000002704: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v45, s26                         // 000000002708: d5010032 006a5a80
	v_cndmask_b32_e64 v49, 0, v46, s26                         // 000000002710: d5010031 006a5c80
	s_lshr_b64 s[60:61], s[58:59], 5                           // 000000002718: 85bc853a
	v_add_co_u32 v45, s28, s50, v48                            // 00000000271c: d7001c2d 02026032
	s_wait_alu depctr_va_sdst(0)                               // 000000002724: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s51, v47, s28               // 000000002728: d5207c2e 00725e33
	v_add_co_u32 v47, s28, s50, v50                            // 000000002730: d7001c2f 02026432
	s_wait_alu depctr_va_sdst(0)                               // 000000002738: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s51, v49, s28               // 00000000273c: d5207c30 00726233
	v_add_co_u32 v49, s28, v51, 2                              // 000000002744: d7001c31 02010533
	s_wait_alu depctr_va_sdst(0)                               // 00000000274c: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v65, s28                 // 000000002750: d5207c32 00728280
	v_add_co_u32 v56, s28, v51, 3                              // 000000002758: d7001c38 02010733
	s_wait_alu depctr_va_sdst(0)                               // 000000002760: bf88f19f
	v_add_co_ci_u32_e64 v57, null, 0, v65, s28                 // 000000002764: d5207c39 00728280
	s_and_b32 s28, s1, s27                                     // 00000000276c: 8b1c1b01
	s_and_b32 s27, s1, s29                                     // 000000002770: 8b1b1d01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002774: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v49, s28                         // 000000002778: d5010031 00726280
	v_cndmask_b32_e64 v50, 0, v50, s28                         // 000000002780: d5010032 00726480
	v_cndmask_b32_e64 v56, 0, v56, s27                         // 000000002788: d5010038 006e7080
	v_cndmask_b32_e64 v57, 0, v57, s27                         // 000000002790: d5010039 006e7280
	v_mad_co_u64_u32 v[69:70], null, s60, s38, v[27:28]        // 000000002798: d6fe7c45 046c4c3c
	v_add_co_u32 v49, s29, s50, v49                            // 0000000027a0: d7001d31 02026232
	s_wait_alu depctr_va_sdst(0)                               // 0000000027a8: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s51, v50, s29               // 0000000027ac: d5207c32 00766433
	v_add_co_u32 v59, s29, s50, v56                            // 0000000027b4: d7001d3b 02027032
	s_wait_alu depctr_va_sdst(0)                               // 0000000027bc: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s51, v57, s29               // 0000000027c0: d5207c3c 00767233
	v_add_co_u32 v56, s29, v51, 4                              // 0000000027c8: d7001d38 02010933
	s_wait_alu depctr_va_sdst(0)                               // 0000000027d0: bf88f19f
	v_add_co_ci_u32_e64 v57, null, 0, v65, s29                 // 0000000027d4: d5207c39 00768280
	v_add_co_u32 v58, s29, v51, 5                              // 0000000027dc: d7001d3a 02010b33
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e4: bf88f19f
	v_add_co_ci_u32_e64 v61, null, 0, v65, s29                 // 0000000027e8: d5207c3d 00768280
	s_and_b32 s29, s1, s30                                     // 0000000027f0: 8b1d1e01
	s_and_b32 s30, s1, s31                                     // 0000000027f4: 8b1e1f01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027f8: bf88ff9e
	v_cndmask_b32_e64 v56, 0, v56, s29                         // 0000000027fc: d5010038 00767080
	v_cndmask_b32_e64 v57, 0, v57, s29                         // 000000002804: d5010039 00767280
	v_cndmask_b32_e64 v58, 0, v58, s30                         // 00000000280c: d501003a 007a7480
	v_cndmask_b32_e64 v64, 0, v61, s30                         // 000000002814: d5010040 007a7a80
	s_mul_i32 s63, s60, s39                                    // 00000000281c: 963f273c
	v_add_co_u32 v61, s31, s50, v56                            // 000000002820: d7001f3d 02027032
	s_wait_alu depctr_va_sdst(0)                               // 000000002828: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s51, v57, s31               // 00000000282c: d5207c3e 007e7233
	v_add_co_u32 v63, s31, s50, v58                            // 000000002834: d7001f3f 02027432
	s_wait_alu depctr_va_sdst(0)                               // 00000000283c: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s51, v64, s31               // 000000002840: d5207c40 007e8033
	v_add_co_u32 v56, s31, v51, 6                              // 000000002848: d7001f38 02010d33
	s_wait_alu depctr_va_sdst(0)                               // 000000002850: bf88f19f
	v_add_co_ci_u32_e64 v57, null, 0, v65, s31                 // 000000002854: d5207c39 007e8280
	v_add_co_u32 v51, s31, v51, 7                              // 00000000285c: d7001f33 02010f33
	s_wait_alu depctr_va_sdst(0)                               // 000000002864: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v65, s31                 // 000000002868: d5207c3a 007e8280
	s_and_b32 s31, s1, s33                                     // 000000002870: 8b1f2101
	s_and_b32 s33, s1, s34                                     // 000000002874: 8b212201
	s_wait_alu depctr_sa_sdst(0)                               // 000000002878: bf88ff9e
	v_cndmask_b32_e64 v56, 0, v56, s31                         // 00000000287c: d5010038 007e7080
	v_cndmask_b32_e64 v57, 0, v57, s31                         // 000000002884: d5010039 007e7280
	v_cndmask_b32_e64 v51, 0, v51, s33                         // 00000000288c: d5010033 00866680
	v_cndmask_b32_e64 v58, 0, v58, s33                         // 000000002894: d501003a 00867480
	s_delay_alu instid0(valu_dep_4)                            // 00000000289c: bf870004
	v_add_co_u32 v65, s34, s50, v56                            // 0000000028a0: d7002241 02027032
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a8: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s51, v57, s34               // 0000000028ac: d5207c42 008a7233
	v_add_co_u32 v67, s34, s50, v51                            // 0000000028b4: d7002243 02026632
	s_wait_alu depctr_va_sdst(0)                               // 0000000028bc: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s51, v58, s34               // 0000000028c0: d5207c44 008a7433
	s_clause 0x7                                               // 0000000028c8: bf850007
	global_load_d16_u8 v56, v[45:46], off                      // 0000000028cc: ee07807c 00000038 0000002d
	global_load_d16_hi_u8 v56, v[47:48], off                   // 0000000028d8: ee08407c 00000038 0000002f
	global_load_d16_u8 v57, v[49:50], off                      // 0000000028e4: ee07807c 00000039 00000031
	global_load_d16_hi_u8 v57, v[63:64], off                   // 0000000028f0: ee08407c 00000039 0000003f
	global_load_d16_u8 v58, v[67:68], off                      // 0000000028fc: ee07807c 0000003a 00000043
	global_load_d16_hi_u8 v58, v[65:66], off                   // 000000002908: ee08407c 0000003a 00000041
	global_load_d16_u8 v59, v[59:60], off                      // 000000002914: ee07807c 0000003b 0000003b
	global_load_d16_hi_u8 v59, v[61:62], off                   // 000000002920: ee08407c 0000003b 0000003d
	s_lshr_b32 s34, s59, 5                                     // 00000000292c: 8522853b
	s_add_nc_u64 s[58:59], s[58:59], 32                        // 000000002930: a9baa03a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002934: bf88ff9e
	s_mul_i32 s62, s34, s38                                    // 000000002938: 963e2622
	v_add_co_u32 v45, s34, v11, s60                            // 00000000293c: d700222d 0200790b
	s_wait_alu depctr_va_sdst(0)                               // 000000002944: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s61, v12, s34               // 000000002948: d5207c2e 008a183d
	v_add_co_u32 v47, s34, v13, s60                            // 000000002950: d700222f 0200790d
	s_wait_alu depctr_sa_sdst(0)                               // 000000002958: bf88ff9e
	v_add3_u32 v70, s63, s62, v70                              // 00000000295c: d6550046 05187c3f
	s_wait_alu depctr_va_sdst(0)                               // 000000002964: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s61, v14, s34               // 000000002968: d5207c30 008a1c3d
	global_load_u8 v62, v[45:46], off                          // 000000002970: ee04007c 0000003e 0000002d
	global_load_u8 v63, v[69:70], off                          // 00000000297c: ee04007c 0000003f 00000045
	global_load_u8 v64, v[47:48], off                          // 000000002988: ee04007c 00000040 0000002f
	v_add_co_u32 v45, s34, v15, s60                            // 000000002994: d700222d 0200790f
	s_wait_alu depctr_va_sdst(0)                               // 00000000299c: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s61, v16, s34               // 0000000029a0: d5207c2e 008a203d
	v_add_co_u32 v47, s34, v17, s60                            // 0000000029a8: d700222f 02007911
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b0: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s61, v18, s34               // 0000000029b4: d5207c30 008a243d
	s_clause 0x1                                               // 0000000029bc: bf850001
	global_load_u8 v65, v[45:46], off                          // 0000000029c0: ee04007c 00000041 0000002d
	global_load_u8 v66, v[47:48], off                          // 0000000029cc: ee04007c 00000042 0000002f
	v_add_co_u32 v45, s34, v19, s60                            // 0000000029d8: d700222d 02007913
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e0: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s61, v20, s34               // 0000000029e4: d5207c2e 008a283d
	v_add_co_u32 v47, s34, v21, s60                            // 0000000029ec: d700222f 02007915
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f4: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s61, v22, s34               // 0000000029f8: d5207c30 008a2c3d
	s_clause 0x1                                               // 000000002a00: bf850001
	global_load_u8 v67, v[45:46], off                          // 000000002a04: ee04007c 00000043 0000002d
	global_load_u8 v68, v[47:48], off                          // 000000002a10: ee04007c 00000044 0000002f
	v_add_co_u32 v45, s34, v23, s60                            // 000000002a1c: d700222d 02007917
	s_wait_alu depctr_va_sdst(0)                               // 000000002a24: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s61, v24, s34               // 000000002a28: d5207c2e 008a303d
	v_add_co_u32 v47, s34, v25, s60                            // 000000002a30: d700222f 02007919
	s_wait_alu depctr_va_sdst(0)                               // 000000002a38: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s61, v26, s34               // 000000002a3c: d5207c30 008a343d
	s_clause 0x1                                               // 000000002a44: bf850001
	global_load_u8 v69, v[45:46], off                          // 000000002a48: ee04007c 00000045 0000002d
	global_load_u8 v70, v[47:48], off                          // 000000002a54: ee04007c 00000046 0000002f
	s_wait_loadcnt 0x25                                        // 000000002a60: bfc00025
	v_cndmask_b16 v36.l, 0, v36.l, s4                          // 000000002a64: d65d0024 00124880
	s_wait_loadcnt 0x21                                        // 000000002a6c: bfc00021
	v_cndmask_b16 v38.h, 0, v38.h, s8                          // 000000002a70: d65d5026 00224c80
	v_cndmask_b16 v38.l, 0, v38.l, s7                          // 000000002a78: d65d0026 001e4c80
	v_cndmask_b16 v37.h, 0, v37.h, s6                          // 000000002a80: d65d5025 001a4a80
	v_cndmask_b16 v37.l, 0, v37.l, s5                          // 000000002a88: d65d0025 00164a80
	v_cndmask_b16 v36.h, 0, v36.h, s3                          // 000000002a90: d65d5024 000e4880
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002a98: d7385026 02024c88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002aa0: d7620026 02024cff 000000ff
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002aac: d7385025 02024a88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002ab4: d7620025 02024aff 000000ff
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 000000002ac0: d7385024 02024888
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002ac8: d7620024 020248ff 000000ff
	v_cndmask_b16 v35.h, 0, v35.h, s2                          // 000000002ad4: d65d5023 000a4680
	v_cndmask_b16 v35.l, 0, v35.l, vcc_lo                      // 000000002adc: d65d0023 01aa4680
	v_or_b16 v61.h, v38.l, v38.h op_sel:[0,1,1]                // 000000002ae4: d763503d 02024d26
	v_or_b16 v61.l, v37.l, v37.h op_sel:[0,1,0]                // 000000002aec: d763103d 02024b25
	v_or_b16 v60.h, v36.l, v36.h op_sel:[0,1,1]                // 000000002af4: d763503c 02024924
	v_lshlrev_b16 v35.h, 8, v35.h op_sel:[0,1,1]               // 000000002afc: d7385023 02024688
	v_and_b16 v35.l, 0xff, v35.l                               // 000000002b04: d7620023 020246ff 000000ff
	s_delay_alu instid0(valu_dep_1)                            // 000000002b10: bf870001
	v_or_b16 v60.l, v35.l, v35.h op_sel:[0,1,0]                // 000000002b14: d763103c 02024723
	s_wait_loadcnt 0x1f                                        // 000000002b1c: bfc0001f
	v_cndmask_b16 v36.l, 0, v40.l, s9                          // 000000002b20: d65d0024 00265080
	v_cndmask_b16 v36.h, 0, v40.h, s10                         // 000000002b28: d65d5024 002a5080
	s_wait_loadcnt 0x1d                                        // 000000002b30: bfc0001d
	v_cndmask_b16 v37.l, 0, v41.l, s12                         // 000000002b34: d65d0025 00325280
	v_cndmask_b16 v40.h, 0, v41.h, s11                         // 000000002b3c: d65d5028 002e5280
	s_wait_loadcnt 0x1b                                        // 000000002b44: bfc0001b
	v_cndmask_b16 v40.l, 0, v42.l, s13                         // 000000002b48: d65d0028 00365480
	v_cndmask_b16 v38.h, 0, v42.h, s14                         // 000000002b50: d65d5026 003a5480
	s_wait_loadcnt 0x19                                        // 000000002b58: bfc00019
	v_cndmask_b16 v38.l, 0, v44.l, s15                         // 000000002b5c: d65d0026 003e5880
	v_cndmask_b16 v37.h, 0, v44.h, s16                         // 000000002b64: d65d5025 00425880
	v_lshlrev_b16 v40.h, 8, v40.h op_sel:[0,1,1]               // 000000002b6c: d7385028 02025088
	v_and_b16 v40.l, 0xff, v40.l                               // 000000002b74: d7620028 020250ff 000000ff
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002b80: d7385026 02024c88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002b88: d7620026 02024cff 000000ff
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002b94: d7385025 02024a88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002b9c: d7620025 02024aff 000000ff
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 000000002ba8: d7385024 02024888
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002bb0: d7620024 020248ff 000000ff
	v_or_b16 v41.l, v40.l, v38.h op_sel:[0,1,0]                // 000000002bbc: d7631029 02024d28
	v_or_b16 v41.h, v38.l, v37.h op_sel:[0,1,1]                // 000000002bc4: d7635029 02024b26
	v_or_b16 v40.h, v37.l, v40.h op_sel:[0,1,1]                // 000000002bcc: d7635028 02025125
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000002bd4: bf870094
	v_or_b16 v40.l, v36.l, v36.h op_sel:[0,1,0]                // 000000002bd8: d7631028 02024924
	v_wmma_f32_16x16x16_fp8_fp8 v[44:51], v[60:61], v[40:41], 0// 000000002be0: cc46402c 1a02513c
	s_wait_loadcnt 0x17                                        // 000000002be8: bfc00017
	v_cndmask_b16 v38.h, 0, v52.l, s17                         // 000000002bec: d65d4026 00466880
	v_cndmask_b16 v38.l, 0, v52.h, s18                         // 000000002bf4: d65d1026 004a6880
	s_wait_loadcnt 0x15                                        // 000000002bfc: bfc00015
	v_cndmask_b16 v35.l, 0, v53.l, s20                         // 000000002c00: d65d0023 00526a80
	v_cndmask_b16 v37.h, 0, v53.h, s19                         // 000000002c08: d65d5025 004e6a80
	s_wait_loadcnt 0x13                                        // 000000002c10: bfc00013
	v_cndmask_b16 v37.l, 0, v54.l, s21                         // 000000002c14: d65d0025 00566c80
	v_cndmask_b16 v36.h, 0, v54.h, s22                         // 000000002c1c: d65d5024 005a6c80
	s_wait_loadcnt 0x11                                        // 000000002c24: bfc00011
	v_cndmask_b16 v36.l, 0, v55.l, s23                         // 000000002c28: d65d0024 005e6e80
	v_cndmask_b16 v35.h, 0, v55.h, s24                         // 000000002c30: d65d5023 00626e80
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002c38: d7385025 02024a88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002c40: d7620025 02024aff 000000ff
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 000000002c4c: d7385024 02024888
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002c54: d7620024 020248ff 000000ff
	v_lshlrev_b16 v35.h, 8, v35.h op_sel:[0,1,1]               // 000000002c60: d7385023 02024688
	v_and_b16 v35.l, 0xff, v35.l                               // 000000002c68: d7620023 020246ff 000000ff
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002c74: bf870194
	v_or_b16 v42.l, v37.l, v36.h op_sel:[0,1,0]                // 000000002c78: d763102a 02024925
	v_or_b16 v42.h, v36.l, v35.h op_sel:[0,1,1]                // 000000002c80: d763502a 02024724
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_1)// 000000002c88: bf8700b3
	v_or_b16 v41.h, v35.l, v37.h op_sel:[0,1,1]                // 000000002c8c: d7635029 02024b23
	v_lshlrev_b16 v35.l, 8, v38.l                              // 000000002c94: d7380023 02024c88
	v_and_b16 v35.h, 0xff, v38.h op_sel:[0,1,1]                // 000000002c9c: d7625023 02024cff 000000ff
	v_or_b16 v41.l, v35.h, v35.l op_sel:[1,0,0]                // 000000002ca8: d7630829 02024723
	s_wait_loadcnt 0xf                                         // 000000002cb0: bfc0000f
	v_cndmask_b16 v36.l, 0, v56.l, s25                         // 000000002cb4: d65d0024 00667080
	v_cndmask_b16 v36.h, 0, v56.h, s26                         // 000000002cbc: d65d5024 006a7080
	s_wait_loadcnt 0xd                                         // 000000002cc4: bfc0000d
	v_cndmask_b16 v37.l, 0, v57.l, s28                         // 000000002cc8: d65d0025 00727280
	v_cndmask_b16 v38.h, 0, v57.h, s30                         // 000000002cd0: d65d5026 007a7280
	s_wait_loadcnt 0xb                                         // 000000002cd8: bfc0000b
	v_cndmask_b16 v37.h, 0, v58.l, s33                         // 000000002cdc: d65d4025 00867480
	v_cndmask_b16 v38.l, 0, v58.h, s31                         // 000000002ce4: d65d1026 007e7480
	s_wait_loadcnt 0x9                                         // 000000002cec: bfc00009
	v_cndmask_b16 v40.h, 0, v59.l, s27                         // 000000002cf0: d65d4028 006e7680
	v_cndmask_b16 v40.l, 0, v59.h, s29                         // 000000002cf8: d65d1028 00767680
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002d00: d7385026 02024c88
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002d08: d7385025 02024a88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002d10: d7620026 02024cff 000000ff
	v_lshlrev_b16 v40.h, 8, v40.h op_sel:[0,1,1]               // 000000002d1c: d7385028 02025088
	v_and_b16 v40.l, 0xff, v40.l                               // 000000002d24: d7620028 020250ff 000000ff
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002d30: d7620025 02024aff 000000ff
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 000000002d3c: d7385024 02024888
	v_and_b16 v36.l, 0xff, v36.l                               // 000000002d44: d7620024 020248ff 000000ff
	v_or_b16 v53.h, v38.l, v37.h op_sel:[0,1,1]                // 000000002d50: d7635035 02024b26
	v_or_b16 v53.l, v40.l, v38.h op_sel:[0,1,0]                // 000000002d58: d7631035 02024d28
	v_or_b16 v52.h, v37.l, v40.h op_sel:[0,1,1]                // 000000002d60: d7635034 02025125
	s_wait_loadcnt 0x8                                         // 000000002d68: bfc00008
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v62                         // 000000002d6c: 7c947cff 000000ff
	v_or_b16 v52.l, v36.l, v36.h op_sel:[0,1,0]                // 000000002d74: d7631034 02024924
	s_wait_loadcnt 0x7                                         // 000000002d7c: bfc00007
	v_add_nc_u32_e32 v35, 0xffffff02, v63                      // 000000002d80: 4a467eff ffffff02
	v_cmp_eq_u32_e64 s2, 0xff, v63                             // 000000002d88: d44a0002 02027eff 000000ff
	s_wait_loadcnt 0x6                                         // 000000002d94: bfc00006
	v_cmp_eq_u32_e64 s3, 0xff, v64                             // 000000002d98: d44a0003 020280ff 000000ff
	v_wmma_f32_16x16x16_fp8_fp8 v[44:51], v[41:42], v[52:53], v[44:51]// 000000002da4: cc46402c 1cb26929
	v_add_nc_u32_e32 v37, v35, v64                             // 000000002dac: 4a4a8123
	s_or_b32 s4, vcc_lo, s2                                    // 000000002db0: 8c04026a
	s_or_b32 s3, s2, s3                                        // 000000002db4: 8c030302
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 000000002db8: bf870141
	v_ldexp_f32 v37, v45, v37                                  // 000000002dbc: d71c0025 02024b2d
	s_wait_loadcnt 0x5                                         // 000000002dc4: bfc00005
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v65                         // 000000002dc8: 7c9482ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dd0: bf88ff9e
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s3                 // 000000002dd4: d5010025 000dff25 7fc00000
	s_or_b32 s3, s2, vcc_lo                                    // 000000002de0: 8c036a02
	s_wait_loadcnt 0x4                                         // 000000002de4: bfc00004
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v66                         // 000000002de8: 7c9484ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_1)// 000000002df0: bf8700a2
	v_add_f32_e32 v34, v34, v37                                // 000000002df4: 06444b22
	v_add_nc_u32_e32 v36, v35, v62                             // 000000002df8: 4a487d23
	v_ldexp_f32 v36, v44, v36                                  // 000000002dfc: d71c0024 0202492c
	s_delay_alu instid0(valu_dep_1)                            // 000000002e04: bf870001
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s4                 // 000000002e08: d5010024 0011ff24 7fc00000
	v_add_nc_u32_e32 v38, v35, v65                             // 000000002e14: 4a4c8323
	s_or_b32 s4, s2, vcc_lo                                    // 000000002e18: 8c046a02
	s_wait_loadcnt 0x1                                         // 000000002e1c: bfc00001
	v_add_nc_u32_e32 v40, v35, v69                             // 000000002e20: 4a508b23
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v68                         // 000000002e24: 7c9488ff 000000ff
	v_dual_add_f32 v1, v1, v36 :: v_dual_add_nc_u32 v36, v35, v66// 000000002e2c: c9204901 01248523
	v_ldexp_f32 v38, v46, v38                                  // 000000002e34: d71c0026 02024d2e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002e3c: bf870194
	v_ldexp_f32 v40, v50, v40                                  // 000000002e40: d71c0028 02025132
	v_ldexp_f32 v36, v47, v36                                  // 000000002e48: d71c0024 0202492f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e50: bf88ff9e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e54: bf8701a3
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s3                 // 000000002e58: d5010026 000dff26 7fc00000
	v_cmp_eq_u32_e64 s3, 0xff, v67                             // 000000002e64: d44a0003 020286ff 000000ff
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s4                 // 000000002e70: d5010024 0011ff24 7fc00000
	v_add_nc_u32_e32 v37, v35, v67                             // 000000002e7c: 4a4a8723
	s_or_b32 s3, s2, s3                                        // 000000002e80: 8c030302
	v_dual_add_f32 v33, v33, v38 :: v_dual_add_nc_u32 v38, v35, v68// 000000002e84: c9204d21 21268923
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002e8c: bf870193
	v_add_f32_e32 v32, v32, v36                                // 000000002e90: 06404920
	v_ldexp_f32 v37, v48, v37                                  // 000000002e94: d71c0025 02024b30
	s_or_b32 s4, s2, vcc_lo                                    // 000000002e9c: 8c046a02
	s_wait_loadcnt 0x0                                         // 000000002ea0: bfc00000
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v70                         // 000000002ea4: 7c948cff 000000ff
	v_ldexp_f32 v38, v49, v38                                  // 000000002eac: d71c0026 02024d31
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eb4: bf88ff9e
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s3                 // 000000002eb8: d5010025 000dff25 7fc00000
	v_cmp_eq_u32_e64 s3, 0xff, v69                             // 000000002ec4: d44a0003 02028aff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ed0: bf870193
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s4                 // 000000002ed4: d5010026 0011ff26 7fc00000
	v_add_f32_e32 v31, v31, v37                                // 000000002ee0: 063e4b1f
	s_or_b32 s3, s2, s3                                        // 000000002ee4: 8c030302
	s_or_b32 s2, s2, vcc_lo                                    // 000000002ee8: 8c026a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eec: bf88ff9e
	v_cndmask_b32_e64 v40, v40, 0x7fc00000, s3                 // 000000002ef0: d5010028 000dff28 7fc00000
	v_add_nc_u32_e32 v35, v35, v70                             // 000000002efc: 4a468d23
	v_add_f32_e32 v30, v30, v38                                // 000000002f00: 063c4d1e
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002f04: bf870193
	v_add_f32_e32 v2, v2, v40                                  // 000000002f08: 06045102
	v_ldexp_f32 v35, v51, v35                                  // 000000002f0c: d71c0023 02024733
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f14: bf870121
	v_cndmask_b32_e64 v35, v35, 0x7fc00000, s2                 // 000000002f18: d5010023 0009ff23 7fc00000
	v_cmp_lt_i64_e64 s2, s[58:59], s[44:45]                    // 000000002f24: d4510002 0200583a
	v_add_f32_e32 v43, v43, v35                                // 000000002f2c: 0656472b
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002f30: 8b6a027e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f34: bf88ff9e
	s_cbranch_vccnz 64479                                      // 000000002f38: bfa4fbdf <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x3b8>
	v_mul_lo_u32 v0, s39, v5                                   // 000000002f3c: d72c0000 02020a27
	v_mul_lo_u32 v11, s38, v6                                  // 000000002f44: d72c000b 02020c26
	v_mad_co_u64_u32 v[9:10], null, s38, v5, 0                 // 000000002f4c: d6fe7c09 02020a26
	v_sub_co_u32 v7, vcc_lo, s36, v5                           // 000000002f54: d7016a07 02020a24
	s_wait_alu depctr_va_vcc(0)                                // 000000002f5c: bf88ff9d
	v_sub_co_ci_u32_e64 v8, null, s37, v6, vcc_lo              // 000000002f60: d5217c08 01aa0c25
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[3:4]                  // 000000002f68: 7ca80626
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002f6c: bf870194
	v_add3_u32 v10, v10, v11, v0                               // 000000002f70: d655000a 0402170a
	v_cmp_lt_i64_e64 s0, 0, v[7:8]                             // 000000002f78: d4510000 02020e80
	s_delay_alu instid0(valu_dep_2)                            // 000000002f80: bf870002
	v_lshlrev_b64_e32 v[5:6], 1, v[9:10]                       // 000000002f84: 3e0a1281
	s_and_b32 s0, s0, vcc_lo                                   // 000000002f88: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f8c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f90: be812000
	s_cbranch_execz 28                                         // 000000002f94: bfa5001c <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1508>
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002f98: 3e120681
	v_add_co_u32 v11, s0, s40, v5                              // 000000002f9c: d700000b 02020a28
	v_bfe_u32 v0, v1, 16, 1                                    // 000000002fa4: d6100000 02052101
	s_wait_alu depctr_va_sdst(0)                               // 000000002fac: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s41, v6, s0                 // 000000002fb0: d5207c0c 00020c29
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002fb8: bf870193
	v_add_co_u32 v9, s0, v11, v9                               // 000000002fbc: d7000009 0202130b
	v_add3_u32 v0, v0, v1, 0x7fff                              // 000000002fc4: d6550000 03fe0300 00007fff
	v_or_b32_e32 v13, 0x400000, v1                             // 000000002fd0: 381a02ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd8: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v12, v10, s0                // 000000002fdc: d5207c0a 0002150c
	v_cmp_u_f32_e64 s0, v1, v1                                 // 000000002fe4: d4180000 02020301
	s_wait_alu depctr_va_sdst(0)                               // 000000002fec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ff0: bf870001
	v_cndmask_b32_e64 v0, v0, v13, s0                          // 000000002ff4: d5010000 00021b00
	global_store_d16_hi_b16 v[9:10], v0, off                   // 000000002ffc: ee09407c 00000000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000003008: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000300c: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[7:8]                             // 000000003010: d4510000 02020e81
	s_and_b32 s0, s0, vcc_lo                                   // 000000003018: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003020: be812000
	s_cbranch_execz 35                                         // 000000003024: bfa50023 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x15b4>
	v_add_co_u32 v9, s0, s40, v5                               // 000000003028: d7000009 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003030: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s41, v6, s0                 // 000000003034: d5207c0a 00020c29
	s_lshl_b64 s[2:3], s[38:39], 1                             // 00000000303c: 84828126
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 000000003040: 3e000681
	s_wait_alu depctr_sa_sdst(0)                               // 000000003044: bf88ff9e
	v_add_co_u32 v9, s0, v9, s2                                // 000000003048: d7000009 02000509
	v_bfe_u32 v11, v34, 16, 1                                  // 000000003050: d610000b 02052122
	s_wait_alu depctr_va_sdst(0)                               // 000000003058: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s3, v10, s0                 // 00000000305c: d5207c0a 00021403
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003064: bf870193
	v_add_co_u32 v0, s0, v9, v0                                // 000000003068: d7000000 02020109
	v_add3_u32 v11, v11, v34, 0x7fff                           // 000000003070: d655000b 03fe450b 00007fff
	v_or_b32_e32 v12, 0x400000, v34                            // 00000000307c: 381844ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003084: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v10, v1, s0                  // 000000003088: d5207c01 0002030a
	v_cmp_u_f32_e64 s0, v34, v34                               // 000000003090: d4180000 02024522
	s_wait_alu depctr_va_sdst(0)                               // 000000003098: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000309c: bf870001
	v_cndmask_b32_e64 v9, v11, v12, s0                         // 0000000030a0: d5010009 0002190b
	global_store_d16_hi_b16 v[0:1], v9, off                    // 0000000030a8: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030b8: 8c7e017e
	v_cmp_lt_i64_e64 s0, 2, v[7:8]                             // 0000000030bc: d4510000 02020e82
	s_and_b32 s0, s0, vcc_lo                                   // 0000000030c4: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030c8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030cc: be812000
	s_cbranch_execz 34                                         // 0000000030d0: bfa50022 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x165c>
	v_add_co_u32 v11, s0, s40, v5                              // 0000000030d4: d700000b 02020a28
	v_bfe_u32 v0, v33, 16, 1                                   // 0000000030dc: d6100000 02052121
	s_wait_alu depctr_va_sdst(0)                               // 0000000030e4: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s41, v6, s0                 // 0000000030e8: d5207c0c 00020c29
	s_lshl_b64 s[2:3], s[38:39], 2                             // 0000000030f0: 84828226
	v_or_b32_e32 v9, 0x400000, v33                             // 0000000030f4: 381242ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030fc: bf88ff9e
	v_add_co_u32 v11, s0, v11, s2                              // 000000003100: d700000b 0200050b
	v_add3_u32 v10, v0, v33, 0x7fff                            // 000000003108: d655000a 03fe4300 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 000000003114: 3e000681
	s_wait_alu depctr_va_sdst(0)                               // 000000003118: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s3, v12, s0                 // 00000000311c: d5207c0c 00021803
	v_cmp_u_f32_e64 s0, v33, v33                               // 000000003124: d4180000 02024321
	s_wait_alu depctr_va_sdst(0)                               // 00000000312c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003130: bf870001
	v_cndmask_b32_e64 v9, v10, v9, s0                          // 000000003134: d5010009 0002130a
	v_add_co_u32 v0, s0, v11, v0                               // 00000000313c: d7000000 0202010b
	s_wait_alu depctr_va_sdst(0)                               // 000000003144: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v12, v1, s0                  // 000000003148: d5207c01 0002030c
	global_store_d16_hi_b16 v[0:1], v9, off                    // 000000003150: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000315c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003160: 8c7e017e
	v_cmp_lt_i64_e64 s0, 3, v[7:8]                             // 000000003164: d4510000 02020e83
	s_and_b32 s0, s0, vcc_lo                                   // 00000000316c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003170: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003174: be812000
	s_cbranch_execz 33                                         // 000000003178: bfa50021 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1700>
	v_add_co_u32 v0, s0, s40, v5                               // 00000000317c: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 000000003188: d5207c01 00020c29
	v_bfe_u32 v9, v32, 16, 1                                   // 000000003190: d6100009 02052120
	v_or_b32_e32 v12, 0x400000, v32                            // 000000003198: 381840ff 00400000
	v_cmp_u_f32_e64 s0, v32, v32                               // 0000000031a0: d4180000 02024120
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000031a8: bf870214
	v_mad_co_u64_u32 v[0:1], null, s38, 6, v[0:1]              // 0000000031ac: d6fe7c00 04010c26
	v_add3_u32 v13, v9, v32, 0x7fff                            // 0000000031b4: d655000d 03fe4109 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 0000000031c0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 0000000031c4: bf870191
	v_cndmask_b32_e64 v12, v13, v12, s0                        // 0000000031c8: d501000c 0002190d
	v_mad_co_u64_u32 v[9:10], null, s39, 6, v[1:2]             // 0000000031d0: d6fe7c09 04050c27
	v_lshlrev_b64_e32 v[10:11], 1, v[3:4]                      // 0000000031d8: 3e140681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031dc: bf870121
	v_add_co_u32 v0, s0, v0, v10                               // 0000000031e0: d7000000 02021500
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v11, s0                  // 0000000031ec: d5207c01 00021709
	global_store_d16_hi_b16 v[0:1], v12, off                   // 0000000031f4: ee09407c 06000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003200: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003204: 8c7e017e
	v_cmp_lt_i64_e64 s0, 4, v[7:8]                             // 000000003208: d4510000 02020e84
	s_and_b32 s0, s0, vcc_lo                                   // 000000003210: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003218: be812000
	s_cbranch_execz 34                                         // 00000000321c: bfa50022 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x17a8>
	v_add_co_u32 v11, s0, s40, v5                              // 000000003220: d700000b 02020a28
	v_bfe_u32 v0, v31, 16, 1                                   // 000000003228: d6100000 0205211f
	s_wait_alu depctr_va_sdst(0)                               // 000000003230: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s41, v6, s0                 // 000000003234: d5207c0c 00020c29
	s_lshl_b64 s[2:3], s[38:39], 3                             // 00000000323c: 84828326
	v_or_b32_e32 v9, 0x400000, v31                             // 000000003240: 38123eff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003248: bf88ff9e
	v_add_co_u32 v11, s0, v11, s2                              // 00000000324c: d700000b 0200050b
	v_add3_u32 v10, v0, v31, 0x7fff                            // 000000003254: d655000a 03fe3f00 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[3:4]                        // 000000003260: 3e000681
	s_wait_alu depctr_va_sdst(0)                               // 000000003264: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s3, v12, s0                 // 000000003268: d5207c0c 00021803
	v_cmp_u_f32_e64 s0, v31, v31                               // 000000003270: d4180000 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000003278: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000327c: bf870001
	v_cndmask_b32_e64 v9, v10, v9, s0                          // 000000003280: d5010009 0002130a
	v_add_co_u32 v0, s0, v11, v0                               // 000000003288: d7000000 0202010b
	s_wait_alu depctr_va_sdst(0)                               // 000000003290: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v12, v1, s0                  // 000000003294: d5207c01 0002030c
	global_store_d16_hi_b16 v[0:1], v9, off                    // 00000000329c: ee09407c 04800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032ac: 8c7e017e
	v_cmp_lt_i64_e64 s0, 5, v[7:8]                             // 0000000032b0: d4510000 02020e85
	s_and_b32 s0, s0, vcc_lo                                   // 0000000032b8: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032c0: be812000
	s_cbranch_execz 33                                         // 0000000032c4: bfa50021 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x184c>
	v_add_co_u32 v0, s0, s40, v5                               // 0000000032c8: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 0000000032d0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 0000000032d4: d5207c01 00020c29
	v_bfe_u32 v9, v30, 16, 1                                   // 0000000032dc: d6100009 0205211e
	v_or_b32_e32 v12, 0x400000, v30                            // 0000000032e4: 38183cff 00400000
	v_cmp_u_f32_e64 s0, v30, v30                               // 0000000032ec: d4180000 02023d1e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000032f4: bf870214
	v_mad_co_u64_u32 v[0:1], null, s38, 10, v[0:1]             // 0000000032f8: d6fe7c00 04011426
	v_add3_u32 v13, v9, v30, 0x7fff                            // 000000003300: d655000d 03fe3d09 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000330c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003310: bf870191
	v_cndmask_b32_e64 v12, v13, v12, s0                        // 000000003314: d501000c 0002190d
	v_mad_co_u64_u32 v[9:10], null, s39, 10, v[1:2]            // 00000000331c: d6fe7c09 04051427
	v_lshlrev_b64_e32 v[10:11], 1, v[3:4]                      // 000000003324: 3e140681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003328: bf870121
	v_add_co_u32 v0, s0, v0, v10                               // 00000000332c: d7000000 02021500
	s_wait_alu depctr_va_sdst(0)                               // 000000003334: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v11, s0                  // 000000003338: d5207c01 00021709
	global_store_d16_hi_b16 v[0:1], v12, off                   // 000000003340: ee09407c 06000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000334c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003350: 8c7e017e
	v_cmp_lt_i64_e64 s0, 6, v[7:8]                             // 000000003354: d4510000 02020e86
	s_and_b32 s0, s0, vcc_lo                                   // 00000000335c: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003360: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003364: be812000
	s_cbranch_execz 33                                         // 000000003368: bfa50021 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x18f0>
	v_add_co_u32 v0, s0, s40, v5                               // 00000000336c: d7000000 02020a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003374: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s41, v6, s0                  // 000000003378: d5207c01 00020c29
	v_bfe_u32 v9, v2, 16, 1                                    // 000000003380: d6100009 02052102
	v_or_b32_e32 v12, 0x400000, v2                             // 000000003388: 381804ff 00400000
	v_cmp_u_f32_e64 s0, v2, v2                                 // 000000003390: d4180000 02020502
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003398: bf870214
	v_mad_co_u64_u32 v[0:1], null, s38, 12, v[0:1]             // 00000000339c: d6fe7c00 04011826
	v_add3_u32 v13, v9, v2, 0x7fff                             // 0000000033a4: d655000d 03fe0509 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 0000000033b0: bf8701b2
	v_mad_co_u64_u32 v[9:10], null, s39, 12, v[1:2]            // 0000000033b4: d6fe7c09 04051827
	v_lshlrev_b64_e32 v[10:11], 1, v[3:4]                      // 0000000033bc: 3e140681
	s_wait_alu depctr_va_sdst(0)                               // 0000000033c0: bf88f19f
	v_cndmask_b32_e64 v2, v13, v12, s0                         // 0000000033c4: d5010002 0002190d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000033cc: bf8701a2
	v_add_co_u32 v0, s0, v0, v10                               // 0000000033d0: d7000000 02021500
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v9, v11, s0                  // 0000000033dc: d5207c01 00021709
	global_store_d16_hi_b16 v[0:1], v2, off                    // 0000000033e4: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000033f4: 8c7e017e
	v_cmp_lt_i64_e64 s0, 7, v[7:8]                             // 0000000033f8: d4510000 02020e87
	s_mov_b32 s1, 0                                            // 000000003400: be810080
	s_and_b32 s2, s0, vcc_lo                                   // 000000003404: 8b026a00
	s_mov_b32 s0, 0                                            // 000000003408: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000340c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003410: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003414: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 000000003418: 8d02037e
	v_add_co_u32 v0, vcc_lo, s40, v5                           // 00000000341c: d7006a00 02020a28
	s_wait_alu depctr_va_vcc(0)                                // 000000003424: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v6, vcc_lo              // 000000003428: d5207c01 01aa0c29
	s_mov_b32 s0, exec_lo                                      // 000000003430: be80007e
	v_mad_co_u64_u32 v[0:1], null, s38, 14, v[0:1]             // 000000003434: d6fe7c00 04011c26
	s_delay_alu instid0(valu_dep_1)                            // 00000000343c: bf870001
	v_mad_co_u64_u32 v[1:2], null, s39, 14, v[1:2]             // 000000003440: d6fe7c01 04051c27
	s_wait_alu depctr_sa_sdst(0)                               // 000000003448: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 00000000344c: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 000000003450: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000003454: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003458: bf88ff9e
	s_cbranch_vccz 692                                         // 00000000345c: bfa302b4 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2430>
	s_and_b32 s0, s35, exec_lo                                 // 000000003460: 8b007e23
	s_cselect_b32 s0, 1, 0                                     // 000000003464: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003468: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 00000000346c: bf078100
	s_cbranch_scc1 20                                          // 000000003470: bfa20014 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x19c4>
	v_lshl_or_b32 v11, v29, 3, s52                             // 000000003474: d656000b 00d1071d
	v_dual_mov_b32 v12, s53 :: v_dual_mov_b32 v1, s53          // 00000000347c: ca100035 0c000035
	v_mov_b32_e32 v18, s53                                     // 000000003484: 7e240235
	v_mov_b32_e32 v8, s53                                      // 000000003488: 7e100235
	s_delay_alu instid0(valu_dep_4)                            // 00000000348c: bf870004
	v_or_b32_e32 v17, 1, v11                                   // 000000003490: 38221681
	v_or_b32_e32 v7, 2, v11                                    // 000000003494: 380e1682
	v_or_b32_e32 v0, 3, v11                                    // 000000003498: 38001683
	v_or_b32_e32 v13, 4, v11                                   // 00000000349c: 381a1684
	v_mov_b32_e32 v14, s53                                     // 0000000034a0: 7e1c0235
	v_or_b32_e32 v15, 5, v11                                   // 0000000034a4: 381e1685
	v_mov_b32_e32 v16, s53                                     // 0000000034a8: 7e200235
	v_or_b32_e32 v9, 6, v11                                    // 0000000034ac: 38121686
	v_mov_b32_e32 v10, s53                                     // 0000000034b0: 7e140235
	v_or_b32_e32 v5, 7, v11                                    // 0000000034b4: 380a1687
	v_mov_b32_e32 v6, s53                                      // 0000000034b8: 7e0c0235
	s_mov_b32 s0, 0                                            // 0000000034bc: be800080
	s_branch 1                                                 // 0000000034c0: bfa00001 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x19c8>
	s_mov_b32 s0, -1                                           // 0000000034c4: be8000c1
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v2, 0              // 0000000034c8: ca100080 2b020080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034d0: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 0000000034d4: 8b007e00
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v44, 0             // 0000000034d8: ca100080 132c0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0             // 0000000034e0: ca100080 2d2e0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v20, 0             // 0000000034e8: ca100080 2f140080
	s_cselect_b32 s0, 1, 0                                     // 0000000034f0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000034f8: bf078100
	s_cbranch_scc1 405                                         // 0000000034fc: bfa20195 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2054>
	v_dual_mov_b32 v20, 0 :: v_dual_lshlrev_b32 v19, 3, v29    // 000000003500: ca220080 14123a83
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[3:4]                  // 000000003508: 7ca80626
	v_dual_mov_b32 v12, s53 :: v_dual_mov_b32 v1, s53          // 00000000350c: ca100035 0c000035
	s_delay_alu instid0(valu_dep_3)                            // 000000003514: bf870003
	v_or_b32_e32 v11, s52, v19                                 // 000000003518: 38162634
	s_lshr_b64 s[2:3], s[46:47], 5                             // 00000000351c: 8582852e
	s_lshr_b32 s1, s47, 5                                      // 000000003520: 8501852f
	s_wait_alu depctr_va_vcc(0)                                // 000000003524: bf88ff9d
	v_dual_cndmask_b32 v2, 0, v4 :: v_dual_cndmask_b32 v5, 0, v3// 000000003528: ca520880 02040680
	v_or_b32_e32 v17, 1, v11                                   // 000000003530: 38221681
	v_or_b32_e32 v15, 5, v11                                   // 000000003534: 381e1685
	v_mov_b32_e32 v18, s53                                     // 000000003538: 7e240235
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[11:12]                // 00000000353c: 7ca81624
	v_or_b32_e32 v7, 2, v11                                    // 000000003540: 380e1682
	v_or_b32_e32 v0, 3, v11                                    // 000000003544: 38001683
	v_mov_b32_e32 v16, s53                                     // 000000003548: 7e200235
	v_cmp_gt_i64_e64 s0, s[36:37], v[17:18]                    // 00000000354c: d4540000 02022224
	s_mov_b64 s[10:11], 0                                      // 000000003554: be8a0180
	s_wait_alu depctr_va_vcc(0)                                // 000000003558: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v11, vcc_lo                       // 00000000355c: 020c1680
	v_cndmask_b32_e64 v13, 0, s53, vcc_lo                      // 000000003560: d501000d 01a86a80
	v_add_co_u32 v25, vcc_lo, s56, v5                          // 000000003568: d7006a19 02020a38
	s_wait_alu depctr_va_sdst(0)                               // 000000003570: bf88f19f
	v_cndmask_b32_e64 v9, 0, v17, s0                           // 000000003574: d5010009 00022280
	s_wait_alu depctr_sa_sdst(0)                               // 00000000357c: bf88ff9e
	v_mul_lo_u32 v10, s1, v6                                   // 000000003580: d72c000a 02020c01
	v_mad_co_u64_u32 v[21:22], null, s2, v6, 0                 // 000000003588: d6fe7c15 02020c02
	v_cndmask_b32_e64 v14, 0, s53, s0                          // 000000003590: d501000e 00006a80
	s_wait_alu depctr_va_vcc(0)                                // 000000003598: bf88ff9d
	v_add_co_ci_u32_e64 v26, null, s57, v2, vcc_lo             // 00000000359c: d5207c1a 01aa0439
	v_mul_lo_u32 v6, s1, v9                                    // 0000000035a4: d72c0006 02021201
	v_mad_co_u64_u32 v[23:24], null, s2, v9, 0                 // 0000000035ac: d6fe7c17 02021202
	v_mul_lo_u32 v9, s2, v13                                   // 0000000035b4: d72c0009 02021a02
	v_mul_lo_u32 v13, s2, v14                                  // 0000000035bc: d72c000d 02021c02
	v_mov_b32_e32 v14, s53                                     // 0000000035c4: 7e1c0235
	s_delay_alu instid0(valu_dep_3)                            // 0000000035c8: bf870003
	v_add3_u32 v22, v22, v9, v10                               // 0000000035cc: d6550016 042a1316
	v_or_b32_e32 v9, 6, v11                                    // 0000000035d4: 38121686
	v_mov_b32_e32 v8, s53                                      // 0000000035d8: 7e100235
	v_add3_u32 v24, v24, v13, v6                               // 0000000035dc: d6550018 041a1b18
	v_or_b32_e32 v13, 4, v11                                   // 0000000035e4: 381a1684
	v_mov_b32_e32 v10, s53                                     // 0000000035e8: 7e140235
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 0000000035ec: bf870194
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[7:8]                  // 0000000035f0: 7ca80e24
	v_cmp_gt_i64_e64 s0, s[36:37], v[13:14]                    // 0000000035f4: d4540000 02021a24
	s_wait_alu depctr_va_vcc(0)                                // 0000000035fc: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v7, vcc_lo                        // 000000003600: 02040e80
	v_cndmask_b32_e64 v5, 0, s53, vcc_lo                       // 000000003604: d5010005 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[0:1]                  // 00000000360c: 7ca80024
	s_wait_alu depctr_va_sdst(0)                               // 000000003610: bf88f19f
	v_cndmask_b32_e64 v31, 0, v13, s0                          // 000000003614: d501001f 00021a80
	v_mul_lo_u32 v43, s1, v2                                   // 00000000361c: d72c002b 02020401
	v_mul_lo_u32 v44, s2, v5                                   // 000000003624: d72c002c 02020a02
	v_mad_co_u64_u32 v[27:28], null, s2, v2, 0                 // 00000000362c: d6fe7c1b 02020402
	s_wait_alu depctr_va_vcc(0)                                // 000000003634: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v0, vcc_lo                        // 000000003638: 020c0080
	v_cndmask_b32_e64 v5, 0, s53, s0                           // 00000000363c: d5010005 00006a80
	v_cndmask_b32_e64 v2, 0, s53, vcc_lo                       // 000000003644: d5010002 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[15:16]                // 00000000364c: 7ca81e24
	v_mul_lo_u32 v46, s1, v31                                  // 000000003650: d72c002e 02023e01
	v_mad_co_u64_u32 v[31:32], null, s2, v31, 0                // 000000003658: d6fe7c1f 02023e02
	v_mul_lo_u32 v47, s2, v5                                   // 000000003660: d72c002f 02020a02
	v_or_b32_e32 v5, 7, v11                                    // 000000003668: 380a1687
	v_add3_u32 v28, v28, v44, v43                              // 00000000366c: d655001c 04ae591c
	v_mov_b32_e32 v43, v20                                     // 000000003674: 7e560314
	v_mul_lo_u32 v45, s1, v6                                   // 000000003678: d72c002d 02020c01
	v_mad_co_u64_u32 v[29:30], null, s2, v6, 0                 // 000000003680: d6fe7c1d 02020c02
	s_wait_alu depctr_va_vcc(0)                                // 000000003688: bf88ff9d
	v_dual_mov_b32 v6, s53 :: v_dual_cndmask_b32 v33, 0, v15   // 00000000368c: ca120035 06201e80
	v_cndmask_b32_e64 v34, 0, s53, vcc_lo                      // 000000003694: d5010022 01a86a80
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[9:10]                 // 00000000369c: 7ca81224
	v_mul_lo_u32 v2, s2, v2                                    // 0000000036a0: d72c0002 02020402
	s_delay_alu instid0(valu_dep_4)                            // 0000000036a8: bf870004
	v_cmp_gt_i64_e64 s0, s[36:37], v[5:6]                      // 0000000036ac: d4540000 02020a24
	v_mul_lo_u32 v48, s1, v33                                  // 0000000036b4: d72c0030 02024201
	v_mul_lo_u32 v49, s2, v34                                  // 0000000036bc: d72c0031 02024402
	v_mad_co_u64_u32 v[33:34], null, s2, v33, 0                // 0000000036c4: d6fe7c21 02024202
	s_wait_alu depctr_va_vcc(0)                                // 0000000036cc: bf88ff9d
	v_cndmask_b32_e32 v35, 0, v9, vcc_lo                       // 0000000036d0: 02461280
	v_cndmask_b32_e64 v36, 0, s53, vcc_lo                      // 0000000036d4: d5010024 01a86a80
	s_wait_alu depctr_va_sdst(0)                               // 0000000036dc: bf88f19f
	v_cndmask_b32_e64 v37, 0, v5, s0                           // 0000000036e0: d5010025 00020a80
	v_cndmask_b32_e64 v38, 0, s53, s0                          // 0000000036e8: d5010026 00006a80
	v_add_co_u32 v41, s0, s54, v39                             // 0000000036f0: d7000029 02024e36
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f8: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s55, 0, s0                  // 0000000036fc: d5207c2a 00010037
	v_add_co_u32 v54, s0, s52, v39                             // 000000003704: d7000036 02024e34
	s_wait_alu depctr_va_sdst(0)                               // 00000000370c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s53, 0, s0                  // 000000003710: d5207c37 00010035
	v_mad_co_u64_u32 v[39:40], null, s46, v41, v[19:20]        // 000000003718: d6fe7c27 044e522e
	v_mul_lo_u32 v56, s46, v42                                 // 000000003720: d72c0038 0202542e
	v_mul_lo_u32 v57, s47, v41                                 // 000000003728: d72c0039 0202522f
	v_mad_co_u64_u32 v[41:42], null, s46, v54, v[19:20]        // 000000003730: d6fe7c29 044e6c2e
	v_mul_lo_u32 v19, s46, v55                                 // 000000003738: d72c0013 02026e2e
	v_mul_lo_u32 v54, s47, v54                                 // 000000003740: d72c0036 02026c2f
	v_mul_lo_u32 v50, s1, v35                                  // 000000003748: d72c0032 02024601
	v_mul_lo_u32 v51, s2, v36                                  // 000000003750: d72c0033 02024802
	v_mad_co_u64_u32 v[35:36], null, s2, v35, 0                // 000000003758: d6fe7c23 02024602
	v_mul_lo_u32 v52, s1, v37                                  // 000000003760: d72c0034 02024a01
	v_mul_lo_u32 v53, s2, v38                                  // 000000003768: d72c0035 02024c02
	v_mad_co_u64_u32 v[37:38], null, s2, v37, 0                // 000000003770: d6fe7c25 02024a02
	v_add3_u32 v30, v30, v2, v45                               // 000000003778: d655001e 04b6051e
	v_add3_u32 v2, v57, v40, v56                               // 000000003780: d6550002 04e25139
	v_add3_u32 v19, v54, v42, v19                              // 000000003788: d6550013 044e5536
	v_add_co_u32 v39, vcc_lo, s50, v39                         // 000000003790: d7006a27 02024e32
	v_add3_u32 v32, v32, v47, v46                              // 000000003798: d6550020 04ba5f20
	s_wait_alu depctr_va_vcc(0)                                // 0000000037a0: bf88ff9d
	v_add_co_ci_u32_e64 v40, null, s51, v2, vcc_lo             // 0000000037a4: d5207c28 01aa0433
	v_add_co_u32 v41, vcc_lo, s48, v41                         // 0000000037ac: d7006a29 02025230
	v_add3_u32 v34, v34, v49, v48                              // 0000000037b4: d6550022 04c26322
	v_add3_u32 v36, v36, v51, v50                              // 0000000037bc: d6550024 04ca6724
	v_add3_u32 v38, v38, v53, v52                              // 0000000037c4: d6550026 04d26b26
	s_wait_alu depctr_va_vcc(0)                                // 0000000037cc: bf88ff9d
	v_add_co_ci_u32_e64 v42, null, s49, v19, vcc_lo            // 0000000037d0: d5207c2a 01aa2631
	v_dual_mov_b32 v47, v20 :: v_dual_mov_b32 v46, v20         // 0000000037d8: ca100114 2f2e0114
	v_dual_mov_b32 v45, v20 :: v_dual_mov_b32 v44, v20         // 0000000037e0: ca100114 2d2c0114
	v_dual_mov_b32 v19, v20 :: v_dual_mov_b32 v2, v20          // 0000000037e8: ca100114 13020114
	v_add_co_u32 v48, vcc_lo, s42, v21                         // 0000000037f0: d7006a30 02022a2a
	s_wait_alu depctr_va_vcc(0)                                // 0000000037f8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s43, v22, vcc_lo            // 0000000037fc: d5207c31 01aa2c2b
	v_add_co_u32 v50, vcc_lo, s42, v23                         // 000000003804: d7006a32 02022e2a
	s_wait_alu depctr_va_vcc(0)                                // 00000000380c: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s43, v24, vcc_lo            // 000000003810: d5207c33 01aa302b
	v_add_co_u32 v52, vcc_lo, s42, v27                         // 000000003818: d7006a34 0202362a
	global_load_u8 v72, v[25:26], off                          // 000000003820: ee04007c 00000048 00000019
	s_clause 0x1                                               // 00000000382c: bf850001
	global_load_b64 v[56:57], v[41:42], off                    // 000000003830: ee05407c 00000038 00000029
	global_load_b64 v[58:59], v[41:42], off offset:16          // 00000000383c: ee05407c 0000003a 00001029
	s_clause 0x1                                               // 000000003848: bf850001
	global_load_b64 v[60:61], v[39:40], off                    // 00000000384c: ee05407c 0000003c 00000027
	global_load_b64 v[62:63], v[39:40], off offset:16          // 000000003858: ee05407c 0000003e 00001027
	s_wait_alu depctr_va_vcc(0)                                // 000000003864: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s43, v28, vcc_lo            // 000000003868: d5207c35 01aa382b
	v_add_co_u32 v54, vcc_lo, s42, v29                         // 000000003870: d7006a36 02023a2a
	s_wait_alu depctr_va_vcc(0)                                // 000000003878: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s43, v30, vcc_lo            // 00000000387c: d5207c37 01aa3c2b
	v_add_co_u32 v64, vcc_lo, s42, v31                         // 000000003884: d7006a40 02023e2a
	s_wait_alu depctr_va_vcc(0)                                // 00000000388c: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s43, v32, vcc_lo            // 000000003890: d5207c41 01aa402b
	v_add_co_u32 v66, vcc_lo, s42, v33                         // 000000003898: d7006a42 0202422a
	s_wait_alu depctr_va_vcc(0)                                // 0000000038a0: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s43, v34, vcc_lo            // 0000000038a4: d5207c43 01aa442b
	v_add_co_u32 v68, vcc_lo, s42, v35                         // 0000000038ac: d7006a44 0202462a
	s_wait_alu depctr_va_vcc(0)                                // 0000000038b4: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s43, v36, vcc_lo            // 0000000038b8: d5207c45 01aa482b
	v_add_co_u32 v70, vcc_lo, s42, v37                         // 0000000038c0: d7006a46 02024a2a
	s_wait_alu depctr_va_vcc(0)                                // 0000000038c8: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s43, v38, vcc_lo            // 0000000038cc: d5207c47 01aa4c2b
	s_clause 0x7                                               // 0000000038d4: bf850007
	global_load_u8 v73, v[48:49], off                          // 0000000038d8: ee04007c 00000049 00000030
	global_load_u8 v74, v[50:51], off                          // 0000000038e4: ee04007c 0000004a 00000032
	global_load_u8 v75, v[52:53], off                          // 0000000038f0: ee04007c 0000004b 00000034
	global_load_u8 v76, v[54:55], off                          // 0000000038fc: ee04007c 0000004c 00000036
	global_load_u8 v64, v[64:65], off                          // 000000003908: ee04007c 00000040 00000040
	global_load_u8 v65, v[66:67], off                          // 000000003914: ee04007c 00000041 00000042
	global_load_u8 v66, v[68:69], off                          // 000000003920: ee04007c 00000042 00000044
	global_load_u8 v67, v[70:71], off                          // 00000000392c: ee04007c 00000043 00000046
	s_add_nc_u64 s[10:11], s[10:11], 32                        // 000000003938: a98aa00a
	v_add_co_u32 v25, vcc_lo, v25, s38                         // 00000000393c: d7006a19 02004d19
	s_wait_alu depctr_sa_sdst(0)                               // 000000003944: bf88ff9e
	v_cmp_lt_i64_e64 s0, s[10:11], s[44:45]                    // 000000003948: d4510000 0200580a
	s_wait_alu depctr_va_vcc(0)                                // 000000003950: bf88ff9d
	v_add_co_ci_u32_e64 v26, null, s39, v26, vcc_lo            // 000000003954: d5207c1a 01aa3427
	v_add_co_u32 v39, vcc_lo, v39, 32                          // 00000000395c: d7006a27 02014127
	s_wait_alu depctr_va_vcc(0)                                // 000000003964: bf88ff9d
	v_add_co_ci_u32_e64 v40, null, 0, v40, vcc_lo              // 000000003968: d5207c28 01aa5080
	v_add_co_u32 v41, vcc_lo, v41, 32                          // 000000003970: d7006a29 02014129
	s_wait_alu depctr_va_vcc(0)                                // 000000003978: bf88ff9d
	v_add_co_ci_u32_e64 v42, null, 0, v42, vcc_lo              // 00000000397c: d5207c2a 01aa5480
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000003984: 8b6a007e
	s_add_nc_u64 s[42:43], s[42:43], 1                         // 000000003988: a9aa812a
	s_wait_loadcnt 0x9                                         // 00000000398c: bfc00009
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[56:57], v[60:61], 0// 000000003990: cc464030 1a027938
	v_add_nc_u32_e32 v56, 0xffffff02, v72                      // 000000003998: 4a7090ff ffffff02
	v_cmp_eq_u32_e64 s0, 0xff, v72                             // 0000000039a0: d44a0000 020290ff 000000ff
	s_wait_loadcnt 0x8                                         // 0000000039ac: bfc00008
	s_delay_alu instid0(valu_dep_3)                            // 0000000039b0: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[58:59], v[62:63], v[48:55]// 0000000039b4: cc464030 1cc27d3a
	s_wait_loadcnt 0x7                                         // 0000000039bc: bfc00007
	v_cmp_eq_u32_e64 s1, 0xff, v73                             // 0000000039c0: d44a0001 020292ff 000000ff
	s_wait_loadcnt 0x6                                         // 0000000039cc: bfc00006
	v_add_nc_u32_e32 v58, v56, v74                             // 0000000039d0: 4a749538
	v_cmp_eq_u32_e64 s2, 0xff, v74                             // 0000000039d4: d44a0002 020294ff 000000ff
	s_wait_loadcnt 0x5                                         // 0000000039e0: bfc00005
	v_cmp_eq_u32_e64 s3, 0xff, v75                             // 0000000039e4: d44a0003 020296ff 000000ff
	s_wait_loadcnt 0x4                                         // 0000000039f0: bfc00004
	v_cmp_eq_u32_e64 s4, 0xff, v76                             // 0000000039f4: d44a0004 020298ff 000000ff
	s_wait_loadcnt 0x2                                         // 000000003a00: bfc00002
	v_cmp_eq_u32_e64 s6, 0xff, v65                             // 000000003a04: d44a0006 020282ff 000000ff
	v_ldexp_f32 v49, v49, v58                                  // 000000003a10: d71c0031 02027531
	s_or_b32 s2, s0, s2                                        // 000000003a18: 8c020200
	s_wait_loadcnt 0x1                                         // 000000003a1c: bfc00001
	v_cmp_eq_u32_e64 s7, 0xff, v66                             // 000000003a20: d44a0007 020284ff 000000ff
	s_or_b32 s3, s0, s3                                        // 000000003a2c: 8c030300
	s_or_b32 s6, s0, s6                                        // 000000003a30: 8c060600
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a34: bf88ff9e
	v_cndmask_b32_e64 v49, v49, 0x7fc00000, s2                 // 000000003a38: d5010031 0009ff31 7fc00000
	v_cmp_eq_u32_e64 s5, 0xff, v64                             // 000000003a44: d44a0005 020280ff 000000ff
	s_or_b32 s7, s0, s7                                        // 000000003a50: 8c070700
	s_wait_loadcnt 0x0                                         // 000000003a54: bfc00000
	v_cmp_eq_u32_e64 s8, 0xff, v67                             // 000000003a58: d44a0008 020286ff 000000ff
	s_or_b32 s4, s0, s4                                        // 000000003a64: 8c040400
	v_add_f32_e32 v47, v47, v49                                // 000000003a68: 065e632f
	v_add_nc_u32_e32 v63, v56, v66                             // 000000003a6c: 4a7e8538
	v_add_nc_u32_e32 v62, v56, v65                             // 000000003a70: 4a7c8338
	v_add_nc_u32_e32 v61, v56, v64                             // 000000003a74: 4a7a8138
	s_or_b32 s5, s0, s5                                        // 000000003a78: 8c050500
	s_or_b32 s8, s0, s8                                        // 000000003a7c: 8c080800
	v_ldexp_f32 v54, v54, v63                                  // 000000003a80: d71c0036 02027f36
	v_ldexp_f32 v53, v53, v62                                  // 000000003a88: d71c0035 02027d35
	v_ldexp_f32 v52, v52, v61                                  // 000000003a90: d71c0034 02027b34
	s_or_b32 s0, s1, s0                                        // 000000003a98: 8c000001
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a9c: bf88ff9e
	v_cndmask_b32_e64 v54, v54, 0x7fc00000, s7                 // 000000003aa0: d5010036 001dff36 7fc00000
	v_add_nc_u32_e32 v59, v56, v75                             // 000000003aac: 4a769738
	v_cndmask_b32_e64 v53, v53, 0x7fc00000, s6                 // 000000003ab0: d5010035 0019ff35 7fc00000
	v_add_nc_u32_e32 v60, v56, v76                             // 000000003abc: 4a789938
	v_cndmask_b32_e64 v52, v52, 0x7fc00000, s5                 // 000000003ac0: d5010034 0015ff34 7fc00000
	v_add_f32_e32 v2, v2, v54                                  // 000000003acc: 06046d02
	v_ldexp_f32 v50, v50, v59                                  // 000000003ad0: d71c0032 02027732
	v_add_f32_e32 v19, v19, v53                                // 000000003ad8: 06266b13
	v_ldexp_f32 v51, v51, v60                                  // 000000003adc: d71c0033 02027933
	v_add_f32_e32 v44, v44, v52                                // 000000003ae4: 0658692c
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 000000003ae8: bf870244
	v_cndmask_b32_e64 v50, v50, 0x7fc00000, s3                 // 000000003aec: d5010032 000dff32 7fc00000
	v_add_nc_u32_e32 v57, v56, v73                             // 000000003af8: 4a729338
	v_add_nc_u32_e32 v56, v56, v67                             // 000000003afc: 4a708738
	v_cndmask_b32_e64 v51, v51, 0x7fc00000, s4                 // 000000003b00: d5010033 0011ff33 7fc00000
	v_add_f32_e32 v46, v46, v50                                // 000000003b0c: 065c652e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003b10: bf870214
	v_ldexp_f32 v48, v48, v57                                  // 000000003b14: d71c0030 02027330
	v_ldexp_f32 v55, v55, v56                                  // 000000003b1c: d71c0037 02027137
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000003b24: bf870194
	v_add_f32_e32 v45, v45, v51                                // 000000003b28: 065a672d
	v_cndmask_b32_e64 v48, v48, 0x7fc00000, s0                 // 000000003b2c: d5010030 0001ff30 7fc00000
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003b38: bf870093
	v_cndmask_b32_e64 v55, v55, 0x7fc00000, s8                 // 000000003b3c: d5010037 0021ff37 7fc00000
	v_dual_add_f32 v20, v20, v48 :: v_dual_add_f32 v43, v43, v55// 000000003b48: c9086114 142a6f2b
	s_cbranch_vccnz 65319                                      // 000000003b50: bfa4ff27 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1cf0>
	v_mul_lo_u32 v21, s39, v11                                 // 000000003b54: d72c0015 02021627
	v_mul_lo_u32 v22, s38, v12                                 // 000000003b5c: d72c0016 02021826
	v_mad_co_u64_u32 v[11:12], null, s38, v11, 0               // 000000003b64: d6fe7c0b 02021626
	v_mul_lo_u32 v23, s39, v17                                 // 000000003b6c: d72c0017 02022227
	v_mul_lo_u32 v24, s38, v18                                 // 000000003b74: d72c0018 02022426
	v_mad_co_u64_u32 v[17:18], null, s38, v17, 0               // 000000003b7c: d6fe7c11 02022226
	v_bfe_u32 v25, v20, 16, 1                                  // 000000003b84: d6100019 02052114
	v_or_b32_e32 v27, 0x400000, v20                            // 000000003b8c: 383628ff 00400000
	v_bfe_u32 v26, v47, 16, 1                                  // 000000003b94: d610001a 0205212f
	v_mul_lo_u32 v14, s38, v14                                 // 000000003b9c: d72c000e 02021c26
	v_add3_u32 v12, v12, v22, v21                              // 000000003ba4: d655000c 04562d0c
	v_lshlrev_b64_e32 v[21:22], 1, v[3:4]                      // 000000003bac: 3e2a0681
	v_add3_u32 v25, v25, v20, 0x7fff                           // 000000003bb0: d6550019 03fe2919 00007fff
	v_add3_u32 v18, v18, v24, v23                              // 000000003bbc: d6550012 045e3112
	v_mul_lo_u32 v24, s39, v7                                  // 000000003bc4: d72c0018 02020e27
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000003bcc: 3e161681
	v_add3_u32 v26, v26, v47, 0x7fff                           // 000000003bd0: d655001a 03fe5f1a 00007fff
	v_or_b32_e32 v23, 0x400000, v47                            // 000000003bdc: 382e5eff 00400000
	v_lshlrev_b64_e32 v[17:18], 1, v[17:18]                    // 000000003be4: 3e222281
	v_mul_lo_u32 v16, s38, v16                                 // 000000003be8: d72c0010 02022026
	s_mov_b32 s0, -1                                           // 000000003bf0: be8000c1
	v_add_co_u32 v11, vcc_lo, s40, v11                         // 000000003bf4: d7006a0b 02021628
	s_wait_alu depctr_va_vcc(0)                                // 000000003bfc: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s41, v12, vcc_lo            // 000000003c00: d5207c0c 01aa1829
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000003c08: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000003c0c: bf88ff9d
	v_cndmask_b32_e32 v20, v25, v27, vcc_lo                    // 000000003c10: 02283719
	v_add_co_u32 v11, vcc_lo, v11, v21                         // 000000003c14: d7006a0b 02022b0b
	v_mul_lo_u32 v25, s38, v8                                  // 000000003c1c: d72c0019 02021026
	v_mad_co_u64_u32 v[7:8], null, s38, v7, 0                  // 000000003c24: d6fe7c07 02020e26
	s_wait_alu depctr_va_vcc(0)                                // 000000003c2c: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, v12, v22, vcc_lo            // 000000003c30: d5207c0c 01aa2d0c
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000003c38: 7c305f2f
	global_store_d16_hi_b16 v[11:12], v20, off                 // 000000003c3c: ee09407c 0a000000 0000000b
	s_wait_alu depctr_va_vcc(0)                                // 000000003c48: bf88ff9d
	v_cndmask_b32_e32 v20, v26, v23, vcc_lo                    // 000000003c4c: 02282f1a
	v_add_co_u32 v11, vcc_lo, s40, v17                         // 000000003c50: d7006a0b 02022228
	v_add3_u32 v8, v8, v25, v24                                // 000000003c58: d6550008 04623308
	s_wait_alu depctr_va_vcc(0)                                // 000000003c60: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s41, v18, vcc_lo            // 000000003c64: d5207c0c 01aa2429
	v_bfe_u32 v17, v46, 16, 1                                  // 000000003c6c: d6100011 0205212e
	v_add_co_u32 v11, vcc_lo, v11, v21                         // 000000003c74: d7006a0b 02022b0b
	v_mul_lo_u32 v23, s39, v0                                  // 000000003c7c: d72c0017 02020027
	v_mul_lo_u32 v24, s38, v1                                  // 000000003c84: d72c0018 02020226
	v_mad_co_u64_u32 v[0:1], null, s38, v0, 0                  // 000000003c8c: d6fe7c00 02020026
	v_lshlrev_b64_e32 v[7:8], 1, v[7:8]                        // 000000003c94: 3e0e0e81
	s_wait_alu depctr_va_vcc(0)                                // 000000003c98: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, v12, v22, vcc_lo            // 000000003c9c: d5207c0c 01aa2d0c
	v_add3_u32 v17, v17, v46, 0x7fff                           // 000000003ca4: d6550011 03fe5d11 00007fff
	v_or_b32_e32 v18, 0x400000, v46                            // 000000003cb0: 38245cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000003cb8: 7c305d2e
	v_add3_u32 v1, v1, v24, v23                                // 000000003cbc: d6550001 045e3101
	v_mul_lo_u32 v23, s39, v13                                 // 000000003cc4: d72c0017 02021a27
	s_wait_alu depctr_va_vcc(0)                                // 000000003ccc: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo                    // 000000003cd0: 02222511
	global_store_d16_hi_b16 v[11:12], v20, off                 // 000000003cd4: ee09407c 0a000000 0000000b
	v_add_co_u32 v7, vcc_lo, s40, v7                           // 000000003ce0: d7006a07 02020e28
	v_bfe_u32 v11, v45, 16, 1                                  // 000000003ce8: d610000b 0205212d
	s_wait_alu depctr_va_vcc(0)                                // 000000003cf0: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s41, v8, vcc_lo              // 000000003cf4: d5207c08 01aa1029
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003cfc: bf870193
	v_add_co_u32 v7, vcc_lo, v7, v21                           // 000000003d00: d7006a07 02022b07
	v_add3_u32 v18, v11, v45, 0x7fff                           // 000000003d08: d6550012 03fe5b0b 00007fff
	v_mad_co_u64_u32 v[11:12], null, s38, v13, 0               // 000000003d14: d6fe7c0b 02021a26
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003d1c: 3e000081
	s_wait_alu depctr_va_vcc(0)                                // 000000003d20: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v8, v22, vcc_lo              // 000000003d24: d5207c08 01aa2d08
	v_or_b32_e32 v20, 0x400000, v45                            // 000000003d2c: 38285aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000003d34: 7c305b2d
	global_store_d16_hi_b16 v[7:8], v17, off                   // 000000003d38: ee09407c 08800000 00000007
	v_add3_u32 v12, v12, v14, v23                              // 000000003d44: d655000c 045e1d0c
	v_bfe_u32 v7, v44, 16, 1                                   // 000000003d4c: d6100007 0205212c
	s_wait_alu depctr_va_vcc(0)                                // 000000003d54: bf88ff9d
	v_cndmask_b32_e32 v13, v18, v20, vcc_lo                    // 000000003d58: 021a2912
	v_add_co_u32 v0, vcc_lo, s40, v0                           // 000000003d5c: d7006a00 02020028
	s_wait_alu depctr_va_vcc(0)                                // 000000003d64: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v1, vcc_lo              // 000000003d68: d5207c01 01aa0229
	v_add3_u32 v14, v7, v44, 0x7fff                            // 000000003d70: d655000e 03fe5907 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003d7c: bf870003
	v_add_co_u32 v0, vcc_lo, v0, v21                           // 000000003d80: d7006a00 02022b00
	v_lshlrev_b64_e32 v[7:8], 1, v[11:12]                      // 000000003d88: 3e0e1681
	v_mul_lo_u32 v18, s39, v15                                 // 000000003d8c: d72c0012 02021e27
	v_mad_co_u64_u32 v[11:12], null, s38, v15, 0               // 000000003d94: d6fe7c0b 02021e26
	s_wait_alu depctr_va_vcc(0)                                // 000000003d9c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v22, vcc_lo              // 000000003da0: d5207c01 01aa2d01
	v_or_b32_e32 v17, 0x400000, v44                            // 000000003da8: 382258ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000003db0: 7c30592c
	v_or_b32_e32 v15, 0x400000, v19                            // 000000003db4: 381e26ff 00400000
	global_store_d16_hi_b16 v[0:1], v13, off                   // 000000003dbc: ee09407c 06800000 00000000
	v_add3_u32 v12, v12, v16, v18                              // 000000003dc8: d655000c 044a210c
	s_wait_alu depctr_va_vcc(0)                                // 000000003dd0: bf88ff9d
	v_cndmask_b32_e32 v13, v14, v17, vcc_lo                    // 000000003dd4: 021a230e
	v_add_co_u32 v0, vcc_lo, s40, v7                           // 000000003dd8: d7006a00 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 000000003de0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v8, vcc_lo              // 000000003de4: d5207c01 01aa1029
	v_bfe_u32 v14, v19, 16, 1                                  // 000000003dec: d610000e 02052113
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003df4: bf8701a3
	v_add_co_u32 v7, vcc_lo, v0, v21                           // 000000003df8: d7006a07 02022b00
	s_wait_alu depctr_va_vcc(0)                                // 000000003e00: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v1, v22, vcc_lo              // 000000003e04: d5207c08 01aa2d01
	v_lshlrev_b64_e32 v[0:1], 1, v[11:12]                      // 000000003e0c: 3e001681
	v_mul_lo_u32 v11, s39, v9                                  // 000000003e10: d72c000b 02021227
	v_mul_lo_u32 v12, s38, v10                                 // 000000003e18: d72c000c 02021426
	v_mad_co_u64_u32 v[9:10], null, s38, v9, 0                 // 000000003e20: d6fe7c09 02021226
	v_add3_u32 v14, v14, v19, 0x7fff                           // 000000003e28: d655000e 03fe270e 00007fff
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 000000003e34: 7c302713
	v_mul_lo_u32 v18, s38, v6                                  // 000000003e38: d72c0012 02020c26
	s_wait_alu depctr_va_vcc(0)                                // 000000003e40: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003e44: bf870003
	v_cndmask_b32_e32 v14, v14, v15, vcc_lo                    // 000000003e48: 021c1f0e
	v_bfe_u32 v15, v2, 16, 1                                   // 000000003e4c: d610000f 02052102
	v_add_co_u32 v16, vcc_lo, s40, v0                          // 000000003e54: d7006a10 02020028
	v_add3_u32 v10, v10, v12, v11                              // 000000003e5c: d655000a 042e190a
	s_wait_alu depctr_va_vcc(0)                                // 000000003e64: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s41, v1, vcc_lo             // 000000003e68: d5207c11 01aa0229
	v_add3_u32 v11, v15, v2, 0x7fff                            // 000000003e70: d655000b 03fe050f 00007fff
	v_mul_lo_u32 v15, s39, v5                                  // 000000003e7c: d72c000f 02020a27
	v_mad_co_u64_u32 v[0:1], null, s38, v5, 0                  // 000000003e84: d6fe7c00 02020a26
	v_lshlrev_b64_e32 v[5:6], 1, v[9:10]                       // 000000003e8c: 3e0a1281
	v_add_co_u32 v9, vcc_lo, v16, v21                          // 000000003e90: d7006a09 02022b10
	v_or_b32_e32 v12, 0x400000, v2                             // 000000003e98: 381804ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ea0: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v17, v22, vcc_lo            // 000000003ea4: d5207c0a 01aa2d11
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000003eac: 7c300502
	v_add3_u32 v1, v1, v18, v15                                // 000000003eb0: d6550001 043e2501
	s_clause 0x1                                               // 000000003eb8: bf850001
	global_store_d16_hi_b16 v[7:8], v13, off                   // 000000003ebc: ee09407c 06800000 00000007
	global_store_d16_hi_b16 v[9:10], v14, off                  // 000000003ec8: ee09407c 07000000 00000009
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed4: bf88ff9d
	v_cndmask_b32_e32 v2, v11, v12, vcc_lo                     // 000000003ed8: 0204190b
	v_add_co_u32 v5, vcc_lo, s40, v5                           // 000000003edc: d7006a05 02020a28
	s_wait_alu depctr_va_vcc(0)                                // 000000003ee4: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s41, v6, vcc_lo              // 000000003ee8: d5207c06 01aa0c29
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003ef0: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ef4: bf8701a3
	v_add_co_u32 v5, vcc_lo, v5, v21                           // 000000003ef8: d7006a05 02022b05
	s_wait_alu depctr_va_vcc(0)                                // 000000003f00: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, v6, v22, vcc_lo              // 000000003f04: d5207c06 01aa2d06
	s_delay_alu instid0(valu_dep_3)                            // 000000003f0c: bf870003
	v_add_co_u32 v0, vcc_lo, s40, v0                           // 000000003f10: d7006a00 02020028
	s_wait_alu depctr_va_vcc(0)                                // 000000003f18: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v1, vcc_lo              // 000000003f1c: d5207c01 01aa0229
	global_store_d16_hi_b16 v[5:6], v2, off                    // 000000003f24: ee09407c 01000000 00000005
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f30: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003f34: be812000
	s_cbranch_execnz 1                                         // 000000003f38: bfa60001 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2440>
	s_endpgm                                                   // 000000003f3c: bfb00000
	v_bfe_u32 v2, v43, 16, 1                                   // 000000003f40: d6100002 0205212b
	v_or_b32_e32 v5, 0x400000, v43                             // 000000003f48: 380a56ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 000000003f50: 7c30572b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 000000003f54: bf870133
	v_add3_u32 v6, v2, v43, 0x7fff                             // 000000003f58: d6550006 03fe5702 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[3:4]                        // 000000003f64: 3e040681
	s_wait_alu depctr_va_vcc(0)                                // 000000003f68: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v5, vcc_lo                       // 000000003f6c: 02080b06
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f70: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000003f74: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000003f7c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000003f80: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 000000003f88: ee09407c 02000000 00000000
	s_endpgm                                                   // 000000003f94: bfb00000
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
	s_code_end                                                 // 000000004100: bf9f0000
	s_code_end                                                 // 000000004104: bf9f0000
	s_code_end                                                 // 000000004108: bf9f0000
	s_code_end                                                 // 00000000410c: bf9f0000
	s_code_end                                                 // 000000004110: bf9f0000
	s_code_end                                                 // 000000004114: bf9f0000
	s_code_end                                                 // 000000004118: bf9f0000
	s_code_end                                                 // 00000000411c: bf9f0000
	s_code_end                                                 // 000000004120: bf9f0000
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
