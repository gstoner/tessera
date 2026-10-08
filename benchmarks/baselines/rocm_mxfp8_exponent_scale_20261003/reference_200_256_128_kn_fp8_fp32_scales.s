
/tmp/tmpsyqc35r2.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[52:55], s[0:1], 0xc8                         // 000000001b04: f4004d00 f80000c8
	s_load_b64 s[64:65], s[0:1], 0xd8                          // 000000001b0c: f4003000 f80000d8
	s_load_b64 s[56:57], s[0:1], 0xa8                          // 000000001b14: f4002e00 f80000a8
	s_load_b64 s[68:69], s[0:1], 0x8                           // 000000001b1c: f4003100 f8000008
	s_load_b64 s[66:67], s[0:1], 0x30                          // 000000001b24: f4003080 f8000030
	s_load_b64 s[58:59], s[0:1], 0x58                          // 000000001b2c: f4002e80 f8000058
	s_load_b64 s[62:63], s[0:1], 0x80                          // 000000001b34: f4002f80 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b40: 86039f75
	s_mov_b32 s4, ttmp7                                        // 000000001b44: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[70:71], s[2:3], 5                             // 000000001b4c: 84c68502
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000001b50: bf8700c9
	v_dual_mov_b32 v11, s71 :: v_dual_and_b32 v52, 15, v0      // 000000001b54: ca240047 0b34008f
	s_lshl_b64 s[72:73], s[4:5], 4                             // 000000001b5c: 84c88404
	s_add_nc_u64 s[2:3], s[70:71], 32                          // 000000001b60: a982a046
	s_add_nc_u64 s[0:1], s[72:73], 16                          // 000000001b64: a9809048
	v_or_b32_e32 v10, s70, v52                                 // 000000001b68: 38146846
	v_mov_b32_e32 v1, s71                                      // 000000001b6c: 7e020247
	v_bfe_u32 v28, v0, 4, 1                                    // 000000001b70: d610001c 02050900
	s_delay_alu instid0(valu_dep_3)                            // 000000001b78: bf870003
	v_or_b32_e32 v0, 16, v10                                   // 000000001b7c: 38001490
	s_wait_kmcnt 0x0                                           // 000000001b80: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[52:53]                      // 000000001b84: d4540000 02006800
	v_cmp_gt_i64_e64 s1, s[2:3], s[54:55]                      // 000000001b8c: d4540001 02006c02
	v_cmp_lt_i64_e64 s92, s[64:65], 32                         // 000000001b94: d451005c 02014040
	s_and_b32 s60, s64, 0xffffffe0                             // 000000001b9c: 8b3cff40 ffffffe0
	s_mov_b32 s61, s65                                         // 000000001ba4: bebd0041
	s_or_b32 s0, s0, s1                                        // 000000001ba8: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bac: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb0: 8b6a007e
	s_mov_b32 s0, -1                                           // 000000001bb4: be8000c1
	s_cbranch_vccz 2342                                        // 000000001bb8: bfa30926 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2554>
	s_and_b32 s0, s92, exec_lo                                 // 000000001bbc: 8b007e5c
	s_cselect_b32 s0, 1, 0                                     // 000000001bc0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bc8: bf078100
	s_cbranch_scc1 5                                           // 000000001bcc: bfa20005 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0xe4>
	v_lshl_or_b32 v4, v28, 3, s72                              // 000000001bd0: d6560004 0121071c
	v_mov_b32_e32 v5, s73                                      // 000000001bd8: 7e0a0249
	s_mov_b32 s0, 0                                            // 000000001bdc: be800080
	s_branch 1                                                 // 000000001be0: bfa00001 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0xe8>
	s_mov_b32 s0, -1                                           // 000000001be4: be8000c1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001be8: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001bec: 8b007e00
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0             // 000000001bf0: ca100080 1a1a0080
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0             // 000000001bf8: ca100080 1d1e0080
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v32, 0             // 000000001c00: ca100080 1f200080
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v18, 0             // 000000001c08: ca100080 23120080
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v20, 0              // 000000001c10: ca100080 03140080
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v22, 0             // 000000001c18: ca100080 13160080
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v24, 0             // 000000001c20: ca100080 15180080
	v_mov_b32_e32 v23, 0                                       // 000000001c28: 7e2e0280
	v_mov_b32_e32 v25, 0                                       // 000000001c2c: 7e320280
	s_cselect_b32 s0, 1, 0                                     // 000000001c30: 98008081
	s_mul_u64 s[80:81], s[54:55], 3                            // 000000001c34: aad08336
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c38: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c3c: bf078100
	s_mul_u64 s[78:79], s[54:55], 5                            // 000000001c40: aace8536
	s_mul_u64 s[76:77], s[54:55], 6                            // 000000001c44: aacc8636
	s_mul_u64 s[74:75], s[54:55], 7                            // 000000001c48: aaca8736
	s_cbranch_scc1 1650                                        // 000000001c4c: bfa20672 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1b18>
	v_or_b32_e32 v12, s72, v52                                 // 000000001c50: 38186848
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 3, v28      // 000000001c54: ca220080 03023883
	s_mul_i32 s1, s64, s73                                     // 000000001c5c: 96014940
	s_lshr_b64 s[4:5], s[64:65], 5                             // 000000001c60: 85848540
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000001c64: bf870112
	v_mul_lo_u32 v14, s65, v12                                 // 000000001c68: d72c000e 02021841
	v_or_b32_e32 v4, s72, v2                                   // 000000001c70: 38080448
	v_mad_co_u64_u32 v[6:7], null, s64, v12, v[2:3]            // 000000001c74: d6fe7c06 040a1840
	v_mad_co_u64_u32 v[8:9], null, s54, v2, v[10:11]           // 000000001c7c: d6fe7c08 042a0436
	v_mul_lo_u32 v17, s55, v2                                  // 000000001c84: d72c0011 02020437
	s_lshr_b32 s5, s65, 5                                      // 000000001c8c: 85058541
	v_cmp_gt_i64_e64 s2, s[54:55], v[0:1]                      // 000000001c90: d4540002 02020036
	v_dual_mov_b32 v32, v3 :: v_dual_mov_b32 v31, v3           // 000000001c98: ca100103 201e0103
	s_lshl_b64 s[82:83], s[54:55], 1                           // 000000001ca0: 84d28136
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ca4: bf88ff9e
	v_add3_u32 v7, v14, v7, s1                                 // 000000001ca8: d6550007 00060f0e
	v_or_b32_e32 v14, 1, v4                                    // 000000001cb0: 381c0881
	v_mov_b32_e32 v5, s73                                      // 000000001cb4: 7e0a0249
	v_add_nc_u32_e32 v9, v17, v9                               // 000000001cb8: 4a121311
	v_cmp_gt_i64_e64 s1, s[54:55], v[10:11]                    // 000000001cbc: d4540001 02021436
	s_lshl_b64 s[84:85], s[54:55], 2                           // 000000001cc4: 84d48236
	s_lshl_b64 s[86:87], s[54:55], 4                           // 000000001cc8: 84d68436
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[4:5]                  // 000000001ccc: 7ca80834
	v_mov_b32_e32 v13, s73                                     // 000000001cd0: 7e1a0249
	s_mov_b64 s[88:89], 0                                      // 000000001cd4: bed80180
	v_dual_mov_b32 v35, v3 :: v_dual_cndmask_b32 v16, 0, v4    // 000000001cd8: ca120103 23100880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_4)// 000000001ce0: bf870232
	v_cmp_gt_i64_e64 s0, s[52:53], v[12:13]                    // 000000001ce4: d4540000 02021834
	v_mad_co_u64_u32 v[12:13], null, s54, v2, v[0:1]           // 000000001cec: d6fe7c0c 04020436
	v_cndmask_b32_e64 v18, 0, s73, vcc_lo                      // 000000001cf4: d5010012 01a89280
	v_mul_lo_u32 v19, s5, v16                                  // 000000001cfc: d72c0013 02022005
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_4)// 000000001d04: bf870212
	v_mul_lo_u32 v20, s4, v18                                  // 000000001d08: d72c0014 02022404
	v_dual_mov_b32 v18, s73 :: v_dual_add_nc_u32 v13, v17, v13 // 000000001d10: ca200049 120c1b11
	v_or_b32_e32 v17, 2, v4                                    // 000000001d18: 38220882
	v_mov_b32_e32 v15, s73                                     // 000000001d1c: 7e1e0249
	s_delay_alu instid0(valu_dep_1)                            // 000000001d20: bf870001
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[14:15]                // 000000001d24: 7ca81c34
	v_mad_co_u64_u32 v[15:16], null, s4, v16, 0                // 000000001d28: d6fe7c0f 02022004
	s_wait_alu depctr_va_vcc(0)                                // 000000001d30: bf88ff9d
	v_cndmask_b32_e32 v14, 0, v14, vcc_lo                      // 000000001d34: 021c1c80
	v_cndmask_b32_e64 v21, 0, s73, vcc_lo                      // 000000001d38: d5010015 01a89280
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[17:18]                // 000000001d40: 7ca82234
	s_delay_alu instid0(valu_dep_4)                            // 000000001d44: bf870004
	v_add3_u32 v16, v16, v20, v19                              // 000000001d48: d6550010 044e2910
	v_cndmask_b32_e64 v20, 0, v10, s1                          // 000000001d50: d5010014 00061480
	v_mul_lo_u32 v22, s5, v14                                  // 000000001d58: d72c0016 02021c05
	v_mul_lo_u32 v23, s4, v21                                  // 000000001d60: d72c0017 02022a04
	v_mad_co_u64_u32 v[18:19], null, s4, v14, 0                // 000000001d68: d6fe7c12 02021c04
	v_lshlrev_b64_e32 v[14:15], 2, v[15:16]                    // 000000001d70: 3e1c1e82
	s_wait_alu depctr_va_vcc(0)                                // 000000001d74: bf88ff9d
	v_cndmask_b32_e32 v24, 0, v17, vcc_lo                      // 000000001d78: 02302280
	v_or_b32_e32 v16, 3, v4                                    // 000000001d7c: 38200883
	v_mov_b32_e32 v17, s73                                     // 000000001d80: 7e220249
	v_cndmask_b32_e64 v25, 0, s73, vcc_lo                      // 000000001d84: d5010019 01a89280
	v_cndmask_b32_e64 v21, 0, v11, s1                          // 000000001d8c: d5010015 00061680
	v_mul_lo_u32 v26, s5, v24                                  // 000000001d94: d72c001a 02023005
	v_add3_u32 v19, v19, v23, v22                              // 000000001d9c: d6550013 045a2f13
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[16:17]                // 000000001da4: 7ca82034
	v_mul_lo_u32 v25, s4, v25                                  // 000000001da8: d72c0019 02023204
	v_mad_co_u64_u32 v[22:23], null, s4, v24, 0                // 000000001db0: d6fe7c16 02023004
	v_add_co_u32 v33, s3, s58, v14                             // 000000001db8: d7000321 02021c3a
	s_wait_alu depctr_va_sdst(0)                               // 000000001dc0: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s59, v15, s3                // 000000001dc4: d5207c22 000e1e3b
	v_lshlrev_b64_e32 v[14:15], 2, v[18:19]                    // 000000001dcc: 3e1c2482
	s_wait_alu depctr_va_vcc(0)                                // 000000001dd0: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v16, vcc_lo                      // 000000001dd4: 02242080
	v_or_b32_e32 v16, 4, v4                                    // 000000001dd8: 38200884
	v_cndmask_b32_e64 v19, 0, s73, vcc_lo                      // 000000001ddc: d5010013 01a89280
	v_add3_u32 v23, v23, v25, v26                              // 000000001de4: d6550017 046a3317
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001dec: bf870214
	v_mul_lo_u32 v24, s5, v18                                  // 000000001df0: d72c0018 02022405
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[16:17]                // 000000001df8: 7ca82034
	s_delay_alu instid0(valu_dep_4)                            // 000000001dfc: bf870004
	v_mul_lo_u32 v25, s4, v19                                  // 000000001e00: d72c0019 02022604
	v_mad_co_u64_u32 v[18:19], null, s4, v18, 0                // 000000001e08: d6fe7c12 02022404
	v_add_co_u32 v36, s3, s58, v14                             // 000000001e10: d7000324 02021c3a
	s_wait_alu depctr_va_sdst(0)                               // 000000001e18: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s59, v15, s3                // 000000001e1c: d5207c25 000e1e3b
	v_lshlrev_b64_e32 v[14:15], 2, v[22:23]                    // 000000001e24: 3e1c2c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001e28: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v16, vcc_lo                      // 000000001e2c: 022c2080
	v_or_b32_e32 v16, 5, v4                                    // 000000001e30: 38200885
	v_cndmask_b32_e64 v23, 0, s73, vcc_lo                      // 000000001e34: d5010017 01a89280
	v_add3_u32 v19, v19, v25, v24                              // 000000001e3c: d6550013 04623313
	v_or_b32_e32 v24, 6, v4                                    // 000000001e44: 38300886
	v_add_co_u32 v38, s3, s58, v14                             // 000000001e48: d7000326 02021c3a
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[16:17]                // 000000001e50: 7ca82034
	v_mov_b32_e32 v25, s73                                     // 000000001e54: 7e320249
	v_mul_lo_u32 v26, s5, v22                                  // 000000001e58: d72c001a 02022c05
	v_mul_lo_u32 v27, s4, v23                                  // 000000001e60: d72c001b 02022e04
	v_mad_co_u64_u32 v[22:23], null, s4, v22, 0                // 000000001e68: d6fe7c16 02022c04
	s_wait_alu depctr_va_sdst(0)                               // 000000001e70: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s59, v15, s3                // 000000001e74: d5207c27 000e1e3b
	v_lshlrev_b64_e32 v[14:15], 2, v[18:19]                    // 000000001e7c: 3e1c2482
	s_wait_alu depctr_va_vcc(0)                                // 000000001e80: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v16, vcc_lo                      // 000000001e84: 02242080
	v_or_b32_e32 v16, 7, v4                                    // 000000001e88: 38200887
	v_cmp_gt_i64_e64 s3, s[52:53], v[24:25]                    // 000000001e8c: d4540003 02023034
	v_cndmask_b32_e64 v19, 0, s73, vcc_lo                      // 000000001e94: d5010013 01a89280
	v_add3_u32 v23, v23, v27, v26                              // 000000001e9c: d6550017 046a3717
	v_mul_lo_u32 v26, s5, v18                                  // 000000001ea4: d72c001a 02022405
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[16:17]                // 000000001eac: 7ca82034
	v_mad_co_u64_u32 v[17:18], null, s4, v18, 0                // 000000001eb0: d6fe7c11 02022404
	v_mul_lo_u32 v19, s4, v19                                  // 000000001eb8: d72c0013 02022604
	s_wait_alu depctr_va_sdst(0)                               // 000000001ec0: bf88f19f
	v_cndmask_b32_e64 v24, 0, v24, s3                          // 000000001ec4: d5010018 000e3080
	v_cndmask_b32_e64 v25, 0, s73, s3                          // 000000001ecc: d5010019 000c9280
	s_wait_alu depctr_va_vcc(0)                                // 000000001ed4: bf88ff9d
	v_cndmask_b32_e32 v16, 0, v16, vcc_lo                      // 000000001ed8: 02202080
	v_cndmask_b32_e64 v30, 0, s73, vcc_lo                      // 000000001edc: d501001e 01a89280
	v_mul_lo_u32 v27, s5, v24                                  // 000000001ee4: d72c001b 02023005
	v_mul_lo_u32 v29, s4, v25                                  // 000000001eec: d72c001d 02023204
	v_mad_co_u64_u32 v[24:25], null, s4, v24, 0                // 000000001ef4: d6fe7c18 02023004
	v_add_co_u32 v40, vcc_lo, s58, v14                         // 000000001efc: d7006a28 02021c3a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f04: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s59, v15, vcc_lo            // 000000001f08: d5207c29 01aa1e3b
	v_add3_u32 v18, v18, v19, v26                              // 000000001f10: d6550012 046a2712
	v_lshlrev_b64_e32 v[14:15], 2, v[22:23]                    // 000000001f18: 3e1c2c82
	v_mul_lo_u32 v19, s5, v16                                  // 000000001f1c: d72c0013 02022005
	v_mul_lo_u32 v26, s4, v30                                  // 000000001f24: d72c001a 02023c04
	v_mad_co_u64_u32 v[22:23], null, s4, v16, 0                // 000000001f2c: d6fe7c16 02022004
	v_lshlrev_b64_e32 v[16:17], 2, v[17:18]                    // 000000001f34: 3e202282
	v_add3_u32 v25, v25, v29, v27                              // 000000001f38: d6550019 046e3b19
	v_add_co_u32 v42, vcc_lo, s58, v14                         // 000000001f40: d7006a2a 02021c3a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f48: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s59, v15, vcc_lo            // 000000001f4c: d5207c2b 01aa1e3b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000001f54: bf8701d3
	v_lshlrev_b64_e32 v[14:15], 2, v[24:25]                    // 000000001f58: 3e1c3082
	v_add3_u32 v23, v23, v26, v19                              // 000000001f5c: d6550017 044e3517
	v_add_co_u32 v44, vcc_lo, s58, v16                         // 000000001f64: d7006a2c 0202203a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f6c: bf88ff9d
	v_add_co_ci_u32_e64 v45, null, s59, v17, vcc_lo            // 000000001f70: d5207c2d 01aa223b
	v_lshlrev_b64_e32 v[16:17], 2, v[22:23]                    // 000000001f78: 3e202c82
	v_cndmask_b32_e64 v19, 0, v1, s2                           // 000000001f7c: d5010013 000a0280
	v_cndmask_b32_e64 v18, 0, v0, s2                           // 000000001f84: d5010012 000a0080
	v_add_co_u32 v46, vcc_lo, s58, v14                         // 000000001f8c: d7006a2e 02021c3a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f94: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s59, v15, vcc_lo            // 000000001f98: d5207c2f 01aa1e3b
	v_add_co_u32 v48, vcc_lo, s58, v16                         // 000000001fa0: d7006a30 0202203a
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s59, v17, vcc_lo            // 000000001fac: d5207c31 01aa223b
	v_lshlrev_b64_e32 v[14:15], 2, v[20:21]                    // 000000001fb4: 3e1c2882
	v_lshlrev_b64_e32 v[16:17], 2, v[18:19]                    // 000000001fb8: 3e202482
	v_dual_mov_b32 v30, v3 :: v_dual_mov_b32 v29, v3           // 000000001fbc: ca100103 1e1c0103
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v3           // 000000001fc4: ca100103 1b1a0103
	v_dual_mov_b32 v25, v3 :: v_dual_mov_b32 v24, v3           // 000000001fcc: ca100103 19180103
	v_dual_mov_b32 v23, v3 :: v_dual_mov_b32 v22, v3           // 000000001fd4: ca100103 17160103
	v_dual_mov_b32 v21, v3 :: v_dual_mov_b32 v20, v3           // 000000001fdc: ca100103 15140103
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v3           // 000000001fe4: ca100103 13120103
	v_or_b32_e32 v50, s88, v2                                  // 000000001fec: 38640458
	v_dual_mov_b32 v51, s89 :: v_dual_mov_b32 v54, s89         // 000000001ff0: ca100059 33360059
	v_add_co_u32 v67, vcc_lo, v6, s88                          // 000000001ff8: d7006a43 0200b106
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002000: bf870193
	v_or_b32_e32 v53, 1, v50                                   // 000000002004: 386a6481
	v_cmp_gt_i64_e64 s18, s[64:65], v[50:51]                   // 000000002008: d4540012 02026440
	s_wait_alu depctr_va_vcc(0)                                // 000000002010: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, s89, v7, vcc_lo             // 000000002014: d5207c44 01aa0e59
	v_or_b32_e32 v59, 3, v50                                   // 00000000201c: 38766483
	v_cmp_gt_i64_e64 s19, s[64:65], v[53:54]                   // 000000002020: d4540013 02026a40
	v_add_co_u32 v53, vcc_lo, v67, 1                           // 000000002028: d7006a35 02010343
	s_wait_alu depctr_va_vcc(0)                                // 000000002030: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, 0, v68, vcc_lo              // 000000002034: d5207c36 01aa8880
	s_and_b32 vcc_lo, s0, s18                                  // 00000000203c: 8b6a1200
	s_and_b32 s3, s0, s19                                      // 000000002040: 8b031300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002044: bf88ff9e
	v_dual_cndmask_b32 v55, 0, v68 :: v_dual_cndmask_b32 v56, 0, v67// 000000002048: ca528880 37388680
	v_cndmask_b32_e64 v58, 0, v53, s3                          // 000000002050: d501003a 000e6a80
	v_cndmask_b32_e64 v57, 0, v54, s3                          // 000000002058: d5010039 000e6c80
	v_mov_b32_e32 v60, s89                                     // 000000002060: 7e780259
	v_or_b32_e32 v63, 5, v50                                   // 000000002064: 387e6485
	v_add_co_u32 v53, s4, s68, v56                             // 000000002068: d7000435 02027044
	s_wait_alu depctr_va_sdst(0)                               // 000000002070: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s69, v55, s4                // 000000002074: d5207c36 00126e45
	v_add_co_u32 v55, s4, s68, v58                             // 00000000207c: d7000437 02027444
	s_wait_alu depctr_va_sdst(0)                               // 000000002084: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s69, v57, s4                // 000000002088: d5207c38 00127245
	v_or_b32_e32 v57, 2, v50                                   // 000000002090: 38726482
	v_mov_b32_e32 v58, s89                                     // 000000002094: 7e740259
	v_cmp_gt_i64_e64 s20, s[64:65], v[59:60]                   // 000000002098: d4540014 02027640
	v_add_co_u32 v61, s4, v67, 2                               // 0000000020a0: d700043d 02010543
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a8: bf88f19f
	v_add_co_ci_u32_e64 v62, null, 0, v68, s4                  // 0000000020ac: d5207c3e 00128880
	v_cmp_gt_i64_e64 s21, s[64:65], v[57:58]                   // 0000000020b4: d4540015 02027240
	v_add_co_u32 v57, s4, v67, 3                               // 0000000020bc: d7000439 02010743
	s_wait_alu depctr_va_sdst(0)                               // 0000000020c4: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v68, s4                  // 0000000020c8: d5207c3a 00128880
	s_and_b32 s4, s0, s20                                      // 0000000020d0: 8b041400
	s_and_b32 s5, s0, s21                                      // 0000000020d4: 8b051500
	v_mov_b32_e32 v64, s89                                     // 0000000020d8: 7e800259
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020dc: bf88ff9e
	v_cndmask_b32_e64 v60, 0, v61, s5                          // 0000000020e0: d501003c 00167a80
	v_cndmask_b32_e64 v59, 0, v62, s5                          // 0000000020e8: d501003b 00167c80
	v_cndmask_b32_e64 v62, 0, v57, s4                          // 0000000020f0: d501003e 00127280
	v_cndmask_b32_e64 v61, 0, v58, s4                          // 0000000020f8: d501003d 00127480
	v_cmp_gt_i64_e64 s23, s[64:65], v[63:64]                   // 000000002100: d4540017 02027e40
	v_add_co_u32 v57, s6, s68, v60                             // 000000002108: d7000639 02027844
	s_wait_alu depctr_va_sdst(0)                               // 000000002110: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s69, v59, s6                // 000000002114: d5207c3a 001a7645
	v_add_co_u32 v59, s6, s68, v62                             // 00000000211c: d700063b 02027c44
	s_wait_alu depctr_va_sdst(0)                               // 000000002124: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s69, v61, s6                // 000000002128: d5207c3c 001a7a45
	v_or_b32_e32 v61, 4, v50                                   // 000000002130: 387a6484
	v_mov_b32_e32 v62, s89                                     // 000000002134: 7e7c0259
	v_add_co_u32 v65, s6, v67, 4                               // 000000002138: d7000641 02010943
	s_wait_alu depctr_va_sdst(0)                               // 000000002140: bf88f19f
	v_add_co_ci_u32_e64 v66, null, 0, v68, s6                  // 000000002144: d5207c42 001a8880
	s_delay_alu instid0(valu_dep_3)                            // 00000000214c: bf870003
	v_cmp_gt_i64_e64 s22, s[64:65], v[61:62]                   // 000000002150: d4540016 02027a40
	v_add_co_u32 v61, s6, v67, 5                               // 000000002158: d700063d 02010b43
	s_wait_alu depctr_va_sdst(0)                               // 000000002160: bf88f19f
	v_add_co_ci_u32_e64 v62, null, 0, v68, s6                  // 000000002164: d5207c3e 001a8880
	s_and_b32 s7, s0, s23                                      // 00000000216c: 8b071700
	s_and_b32 s6, s0, s22                                      // 000000002170: 8b061600
	s_mul_u64 s[36:37], s[88:89], s[54:55]                     // 000000002174: aaa43658
	s_wait_alu depctr_sa_sdst(0)                               // 000000002178: bf88ff9e
	v_cndmask_b32_e64 v64, 0, v65, s6                          // 00000000217c: d5010040 001a8280
	v_cndmask_b32_e64 v63, 0, v66, s6                          // 000000002184: d501003f 001a8480
	v_cndmask_b32_e64 v66, 0, v61, s7                          // 00000000218c: d5010042 001e7a80
	v_cndmask_b32_e64 v65, 0, v62, s7                          // 000000002194: d5010041 001e7c80
	s_and_b32 s11, s1, s19                                     // 00000000219c: 8b0b1301
	v_add_co_u32 v61, s8, s68, v64                             // 0000000021a0: d700083d 02028044
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a8: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s69, v63, s8                // 0000000021ac: d5207c3e 00227e45
	v_add_co_u32 v63, s8, s68, v66                             // 0000000021b4: d700083f 02028444
	s_wait_alu depctr_va_sdst(0)                               // 0000000021bc: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s69, v65, s8                // 0000000021c0: d5207c40 00228245
	v_or_b32_e32 v65, 6, v50                                   // 0000000021c8: 38826486
	v_mov_b32_e32 v66, s89                                     // 0000000021cc: 7e840259
	v_or_b32_e32 v50, 7, v50                                   // 0000000021d0: 38646487
	v_add_co_u32 v69, s8, v67, 6                               // 0000000021d4: d7000845 02010d43
	s_wait_alu depctr_va_sdst(0)                               // 0000000021dc: bf88f19f
	v_add_co_ci_u32_e64 v70, null, 0, v68, s8                  // 0000000021e0: d5207c46 00228880
	v_cmp_gt_i64_e64 s24, s[64:65], v[65:66]                   // 0000000021e8: d4540018 02028240
	v_cmp_gt_i64_e64 s25, s[64:65], v[50:51]                   // 0000000021f0: d4540019 02026440
	v_add_co_u32 v50, s8, v67, 7                               // 0000000021f8: d7000832 02010f43
	s_wait_alu depctr_va_sdst(0)                               // 000000002200: bf88f19f
	v_add_co_ci_u32_e64 v51, null, 0, v68, s8                  // 000000002204: d5207c33 00228880
	s_and_b32 s8, s0, s24                                      // 00000000220c: 8b081800
	s_and_b32 s9, s0, s25                                      // 000000002210: 8b091900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002214: bf88ff9e
	v_cndmask_b32_e64 v65, 0, v69, s8                          // 000000002218: d5010041 00228a80
	v_cndmask_b32_e64 v66, 0, v70, s8                          // 000000002220: d5010042 00228c80
	v_cndmask_b32_e64 v50, 0, v50, s9                          // 000000002228: d5010032 00266480
	v_cndmask_b32_e64 v51, 0, v51, s9                          // 000000002230: d5010033 00266680
	s_and_b32 s13, s1, s21                                     // 000000002238: 8b0d1501
	v_add_co_u32 v65, s10, s68, v65                            // 00000000223c: d7000a41 02028244
	s_wait_alu depctr_va_sdst(0)                               // 000000002244: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s69, v66, s10               // 000000002248: d5207c42 002a8445
	v_add_co_u32 v67, s10, s68, v50                            // 000000002250: d7000a43 02026444
	s_wait_alu depctr_va_sdst(0)                               // 000000002258: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s69, v51, s10               // 00000000225c: d5207c44 002a6645
	v_add_co_u32 v69, s10, v8, s36                             // 000000002264: d7000a45 02004908
	s_wait_alu depctr_va_sdst(0)                               // 00000000226c: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s37, v9, s10                // 000000002270: d5207c46 002a1225
	s_clause 0x7                                               // 000000002278: bf850007
	global_load_d16_u8 v50, v[53:54], off                      // 00000000227c: ee07807c 00000032 00000035
	global_load_d16_hi_u8 v50, v[55:56], off                   // 000000002288: ee08407c 00000032 00000037
	global_load_d16_u8 v51, v[57:58], off                      // 000000002294: ee07807c 00000033 00000039
	global_load_d16_hi_u8 v51, v[59:60], off                   // 0000000022a0: ee08407c 00000033 0000003b
	global_load_d16_u8 v53, v[61:62], off                      // 0000000022ac: ee07807c 00000035 0000003d
	global_load_d16_hi_u8 v53, v[63:64], off                   // 0000000022b8: ee08407c 00000035 0000003f
	global_load_d16_u8 v54, v[65:66], off                      // 0000000022c4: ee07807c 00000036 00000041
	global_load_d16_hi_u8 v54, v[67:68], off                   // 0000000022d0: ee08407c 00000036 00000043
	v_add_co_u32 v55, s10, v69, s54                            // 0000000022dc: d7000a37 02006d45
	s_wait_alu depctr_va_sdst(0)                               // 0000000022e4: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s55, v70, s10               // 0000000022e8: d5207c38 002a8c37
	s_and_b32 s10, s1, s18                                     // 0000000022f0: 8b0a1201
	v_cndmask_b32_e64 v60, 0, v55, s11                         // 0000000022f4: d501003c 002e6e80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022fc: bf88ff9e
	v_cndmask_b32_e64 v58, 0, v69, s10                         // 000000002300: d501003a 002a8a80
	v_cndmask_b32_e64 v57, 0, v70, s10                         // 000000002308: d5010039 002a8c80
	v_cndmask_b32_e64 v59, 0, v56, s11                         // 000000002310: d501003b 002e7080
	s_and_b32 s15, s1, s23                                     // 000000002318: 8b0f1701
	s_and_b32 s17, s1, s25                                     // 00000000231c: 8b111901
	v_add_co_u32 v55, s12, s66, v58                            // 000000002320: d7000c37 02027442
	s_wait_alu depctr_va_sdst(0)                               // 000000002328: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s67, v57, s12               // 00000000232c: d5207c38 00327243
	v_add_co_u32 v57, s12, s66, v60                            // 000000002334: d7000c39 02027842
	s_wait_alu depctr_va_sdst(0)                               // 00000000233c: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s67, v59, s12               // 000000002340: d5207c3a 00327643
	v_add_co_u32 v59, s12, v69, s82                            // 000000002348: d7000c3b 0200a545
	s_wait_alu depctr_va_sdst(0)                               // 000000002350: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s83, v70, s12               // 000000002354: d5207c3c 00328c53
	v_add_co_u32 v61, s12, v69, s80                            // 00000000235c: d7000c3d 0200a145
	s_wait_alu depctr_va_sdst(0)                               // 000000002364: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s81, v70, s12               // 000000002368: d5207c3e 00328c51
	v_cndmask_b32_e64 v59, 0, v59, s13                         // 000000002370: d501003b 00367680
	s_and_b32 s12, s1, s20                                     // 000000002378: 8b0c1401
	v_cndmask_b32_e64 v60, 0, v60, s13                         // 00000000237c: d501003c 00367880
	s_wait_alu depctr_sa_sdst(0)                               // 000000002384: bf88ff9e
	v_cndmask_b32_e64 v61, 0, v61, s12                         // 000000002388: d501003d 00327a80
	v_cndmask_b32_e64 v62, 0, v62, s12                         // 000000002390: d501003e 00327c80
	v_add_co_u32 v59, s14, s66, v59                            // 000000002398: d7000e3b 02027642
	s_wait_alu depctr_va_sdst(0)                               // 0000000023a0: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s67, v60, s14               // 0000000023a4: d5207c3c 003a7843
	v_add_co_u32 v61, s14, s66, v61                            // 0000000023ac: d7000e3d 02027a42
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b4: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s67, v62, s14               // 0000000023b8: d5207c3e 003a7c43
	v_add_co_u32 v63, s14, v69, s84                            // 0000000023c0: d7000e3f 0200a945
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c8: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s85, v70, s14               // 0000000023cc: d5207c40 003a8c55
	v_add_co_u32 v65, s14, v69, s78                            // 0000000023d4: d7000e41 02009d45
	s_wait_alu depctr_va_sdst(0)                               // 0000000023dc: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s79, v70, s14               // 0000000023e0: d5207c42 003a8c4f
	s_and_b32 s14, s1, s22                                     // 0000000023e8: 8b0e1601
	v_cndmask_b32_e64 v65, 0, v65, s15                         // 0000000023ec: d5010041 003e8280
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023f4: bf88ff9e
	v_cndmask_b32_e64 v63, 0, v63, s14                         // 0000000023f8: d501003f 003a7e80
	v_cndmask_b32_e64 v64, 0, v64, s14                         // 000000002400: d5010040 003a8080
	v_cndmask_b32_e64 v66, 0, v66, s15                         // 000000002408: d5010042 003e8480
	s_and_b32 s18, s2, s18                                     // 000000002410: 8b121202
	s_and_b32 s19, s2, s19                                     // 000000002414: 8b131302
	v_add_co_u32 v63, s16, s66, v63                            // 000000002418: d700103f 02027e42
	s_wait_alu depctr_va_sdst(0)                               // 000000002420: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s67, v64, s16               // 000000002424: d5207c40 00428043
	v_add_co_u32 v65, s16, s66, v65                            // 00000000242c: d7001041 02028242
	s_wait_alu depctr_va_sdst(0)                               // 000000002434: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s67, v66, s16               // 000000002438: d5207c42 00428443
	v_add_co_u32 v67, s16, v69, s76                            // 000000002440: d7001043 02009945
	s_wait_alu depctr_va_sdst(0)                               // 000000002448: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s77, v70, s16               // 00000000244c: d5207c44 00428c4d
	v_add_co_u32 v69, s16, v69, s74                            // 000000002454: d7001045 02009545
	s_wait_alu depctr_va_sdst(0)                               // 00000000245c: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s75, v70, s16               // 000000002460: d5207c46 00428c4b
	s_and_b32 s16, s1, s24                                     // 000000002468: 8b101801
	v_cndmask_b32_e64 v69, 0, v69, s17                         // 00000000246c: d5010045 00468a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002474: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v67, s16                         // 000000002478: d5010043 00428680
	v_cndmask_b32_e64 v68, 0, v68, s16                         // 000000002480: d5010044 00428880
	v_cndmask_b32_e64 v70, 0, v70, s17                         // 000000002488: d5010046 00468c80
	s_and_b32 s21, s2, s21                                     // 000000002490: 8b151502
	s_and_b32 s20, s2, s20                                     // 000000002494: 8b141402
	v_add_co_u32 v67, s26, s66, v67                            // 000000002498: d7001a43 02028642
	s_wait_alu depctr_va_sdst(0)                               // 0000000024a0: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s67, v68, s26               // 0000000024a4: d5207c44 006a8843
	v_add_co_u32 v69, s26, s66, v69                            // 0000000024ac: d7001a45 02028a42
	s_wait_alu depctr_va_sdst(0)                               // 0000000024b4: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s67, v70, s26               // 0000000024b8: d5207c46 006a8c43
	v_add_co_u32 v71, s26, v12, s36                            // 0000000024c0: d7001a47 0200490c
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s37, v13, s26               // 0000000024cc: d5207c48 006a1a25
	s_clause 0x7                                               // 0000000024d4: bf850007
	global_load_d16_u8 v55, v[55:56], off                      // 0000000024d8: ee07807c 00000037 00000037
	global_load_d16_hi_u8 v55, v[57:58], off                   // 0000000024e4: ee08407c 00000037 00000039
	global_load_d16_u8 v56, v[59:60], off                      // 0000000024f0: ee07807c 00000038 0000003b
	global_load_d16_hi_u8 v56, v[61:62], off                   // 0000000024fc: ee08407c 00000038 0000003d
	global_load_d16_u8 v57, v[63:64], off                      // 000000002508: ee07807c 00000039 0000003f
	global_load_d16_hi_u8 v57, v[65:66], off                   // 000000002514: ee08407c 00000039 00000041
	global_load_d16_u8 v58, v[67:68], off                      // 000000002520: ee07807c 0000003a 00000043
	global_load_d16_hi_u8 v58, v[69:70], off                   // 00000000252c: ee08407c 0000003a 00000045
	v_add_co_u32 v59, s26, v71, s54                            // 000000002538: d7001a3b 02006d47
	s_wait_alu depctr_va_sdst(0)                               // 000000002540: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s55, v72, s26               // 000000002544: d5207c3c 006a9037
	v_cndmask_b32_e64 v62, 0, v71, s18                         // 00000000254c: d501003e 004a8e80
	v_cndmask_b32_e64 v61, 0, v72, s18                         // 000000002554: d501003d 004a9080
	v_cndmask_b32_e64 v64, 0, v59, s19                         // 00000000255c: d5010040 004e7680
	s_delay_alu instid0(valu_dep_4)                            // 000000002564: bf870004
	v_cndmask_b32_e64 v63, 0, v60, s19                         // 000000002568: d501003f 004e7880
	s_and_b32 s22, s2, s22                                     // 000000002570: 8b161602
	v_add_co_u32 v59, s26, s66, v62                            // 000000002574: d7001a3b 02027c42
	s_wait_alu depctr_va_sdst(0)                               // 00000000257c: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s67, v61, s26               // 000000002580: d5207c3c 006a7a43
	v_add_co_u32 v61, s26, s66, v64                            // 000000002588: d7001a3d 02028042
	s_wait_alu depctr_va_sdst(0)                               // 000000002590: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s67, v63, s26               // 000000002594: d5207c3e 006a7e43
	v_add_co_u32 v63, s26, v71, s82                            // 00000000259c: d7001a3f 0200a547
	s_wait_alu depctr_va_sdst(0)                               // 0000000025a4: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s83, v72, s26               // 0000000025a8: d5207c40 006a9053
	v_add_co_u32 v65, s26, v71, s80                            // 0000000025b0: d7001a41 0200a147
	s_wait_alu depctr_va_sdst(0)                               // 0000000025b8: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s81, v72, s26               // 0000000025bc: d5207c42 006a9051
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025c4: bf88ff9e
	v_cndmask_b32_e64 v63, 0, v63, s21                         // 0000000025c8: d501003f 00567e80
	v_cndmask_b32_e64 v64, 0, v64, s21                         // 0000000025d0: d5010040 00568080
	v_cndmask_b32_e64 v65, 0, v65, s20                         // 0000000025d8: d5010041 00528280
	v_cndmask_b32_e64 v66, 0, v66, s20                         // 0000000025e0: d5010042 00528480
	s_and_b32 s23, s2, s23                                     // 0000000025e8: 8b171702
	v_add_co_u32 v63, s26, s66, v63                            // 0000000025ec: d7001a3f 02027e42
	s_wait_alu depctr_va_sdst(0)                               // 0000000025f4: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s67, v64, s26               // 0000000025f8: d5207c40 006a8043
	v_add_co_u32 v65, s26, s66, v65                            // 000000002600: d7001a41 02028242
	s_wait_alu depctr_va_sdst(0)                               // 000000002608: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s67, v66, s26               // 00000000260c: d5207c42 006a8443
	v_add_co_u32 v67, s26, v71, s84                            // 000000002614: d7001a43 0200a947
	s_wait_alu depctr_va_sdst(0)                               // 00000000261c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s85, v72, s26               // 000000002620: d5207c44 006a9055
	v_add_co_u32 v69, s26, v71, s78                            // 000000002628: d7001a45 02009d47
	s_wait_alu depctr_va_sdst(0)                               // 000000002630: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s79, v72, s26               // 000000002634: d5207c46 006a904f
	v_cndmask_b32_e64 v67, 0, v67, s22                         // 00000000263c: d5010043 005a8680
	v_cndmask_b32_e64 v68, 0, v68, s22                         // 000000002644: d5010044 005a8880
	s_wait_alu depctr_sa_sdst(0)                               // 00000000264c: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s23                         // 000000002650: d5010045 005e8a80
	v_cndmask_b32_e64 v70, 0, v70, s23                         // 000000002658: d5010046 005e8c80
	s_and_b32 s24, s2, s24                                     // 000000002660: 8b181802
	v_add_co_u32 v67, s26, s66, v67                            // 000000002664: d7001a43 02028642
	s_wait_alu depctr_va_sdst(0)                               // 00000000266c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s67, v68, s26               // 000000002670: d5207c44 006a8843
	v_add_co_u32 v69, s26, s66, v69                            // 000000002678: d7001a45 02028a42
	s_wait_alu depctr_va_sdst(0)                               // 000000002680: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s67, v70, s26               // 000000002684: d5207c46 006a8c43
	v_add_co_u32 v73, s26, v71, s76                            // 00000000268c: d7001a49 02009947
	s_wait_alu depctr_va_sdst(0)                               // 000000002694: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s77, v72, s26               // 000000002698: d5207c4a 006a904d
	v_add_co_u32 v71, s26, v71, s74                            // 0000000026a0: d7001a47 02009547
	s_wait_alu depctr_va_sdst(0)                               // 0000000026a8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s75, v72, s26               // 0000000026ac: d5207c48 006a904b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026b4: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v73, s24                         // 0000000026b8: d5010049 00629280
	s_and_b32 s25, s2, s25                                     // 0000000026c0: 8b191902
	v_cndmask_b32_e64 v74, 0, v74, s24                         // 0000000026c4: d501004a 00629480
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026cc: bf88ff9e
	v_cndmask_b32_e64 v76, 0, v71, s25                         // 0000000026d0: d501004c 00668e80
	v_cndmask_b32_e64 v75, 0, v72, s25                         // 0000000026d8: d501004b 00669080
	v_add_co_u32 v71, s26, s66, v73                            // 0000000026e0: d7001a47 02029242
	s_wait_alu depctr_va_sdst(0)                               // 0000000026e8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s67, v74, s26               // 0000000026ec: d5207c48 006a9443
	v_add_co_u32 v73, s26, s66, v76                            // 0000000026f4: d7001a49 02029842
	s_wait_alu depctr_va_sdst(0)                               // 0000000026fc: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s67, v75, s26               // 000000002700: d5207c4a 006a9643
	s_or_b32 s26, s88, 16                                      // 000000002708: 8c1a9058
	v_mov_b32_e32 v76, s89                                     // 00000000270c: 7e980259
	s_wait_alu depctr_sa_sdst(0)                               // 000000002710: bf88ff9e
	v_or_b32_e32 v75, s26, v2                                  // 000000002714: 3896041a
	s_clause 0x7                                               // 000000002718: bf850007
	global_load_d16_u8 v59, v[59:60], off                      // 00000000271c: ee07807c 0000003b 0000003b
	global_load_d16_hi_u8 v59, v[61:62], off                   // 000000002728: ee08407c 0000003b 0000003d
	global_load_d16_u8 v60, v[63:64], off                      // 000000002734: ee07807c 0000003c 0000003f
	global_load_d16_hi_u8 v60, v[65:66], off                   // 000000002740: ee08407c 0000003c 00000041
	global_load_d16_u8 v61, v[67:68], off                      // 00000000274c: ee07807c 0000003d 00000043
	global_load_d16_hi_u8 v61, v[69:70], off                   // 000000002758: ee08407c 0000003d 00000045
	global_load_d16_u8 v62, v[71:72], off                      // 000000002764: ee07807c 0000003e 00000047
	global_load_d16_hi_u8 v62, v[73:74], off                   // 000000002770: ee08407c 0000003e 00000049
	v_mov_b32_e32 v64, s89                                     // 00000000277c: 7e800259
	v_add_co_u32 v79, s26, v6, s26                             // 000000002780: d7001a4f 02003506
	v_or_b32_e32 v63, 1, v75                                   // 000000002788: 387e9681
	v_cmp_gt_i64_e64 s43, s[64:65], v[75:76]                   // 00000000278c: d454002b 02029640
	s_wait_alu depctr_va_sdst(0)                               // 000000002794: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s89, v7, s26                // 000000002798: d5207c50 006a0e59
	v_or_b32_e32 v69, 3, v75                                   // 0000000027a0: 388a9683
	v_cmp_gt_i64_e64 s44, s[64:65], v[63:64]                   // 0000000027a4: d454002c 02027e40
	v_add_co_u32 v63, s26, v79, 1                              // 0000000027ac: d7001a3f 0201034f
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b4: bf88f19f
	v_add_co_ci_u32_e64 v64, null, 0, v80, s26                 // 0000000027b8: d5207c40 006aa080
	s_and_b32 s26, s0, s43                                     // 0000000027c0: 8b1a2b00
	s_and_b32 s27, s0, s44                                     // 0000000027c4: 8b1b2c00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027c8: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v79, s26                         // 0000000027cc: d5010042 006a9e80
	v_cndmask_b32_e64 v65, 0, v80, s26                         // 0000000027d4: d5010041 006aa080
	v_cndmask_b32_e64 v68, 0, v63, s27                         // 0000000027dc: d5010044 006e7e80
	v_cndmask_b32_e64 v67, 0, v64, s27                         // 0000000027e4: d5010043 006e8080
	v_mov_b32_e32 v70, s89                                     // 0000000027ec: 7e8c0259
	v_add_co_u32 v63, s28, s68, v66                            // 0000000027f0: d7001c3f 02028444
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f8: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s69, v65, s28               // 0000000027fc: d5207c40 00728245
	v_add_co_u32 v65, s28, s68, v68                            // 000000002804: d7001c41 02028844
	s_wait_alu depctr_va_sdst(0)                               // 00000000280c: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s69, v67, s28               // 000000002810: d5207c42 00728645
	v_or_b32_e32 v67, 2, v75                                   // 000000002818: 38869682
	v_mov_b32_e32 v68, s89                                     // 00000000281c: 7e880259
	v_cmp_gt_i64_e64 s45, s[64:65], v[69:70]                   // 000000002820: d454002d 02028a40
	v_add_co_u32 v71, s28, v79, 2                              // 000000002828: d7001c47 0201054f
	s_wait_alu depctr_va_sdst(0)                               // 000000002830: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v80, s28                 // 000000002834: d5207c48 0072a080
	v_cmp_gt_i64_e64 s46, s[64:65], v[67:68]                   // 00000000283c: d454002e 02028640
	v_add_co_u32 v67, s28, v79, 3                              // 000000002844: d7001c43 0201074f
	s_wait_alu depctr_va_sdst(0)                               // 00000000284c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, 0, v80, s28                 // 000000002850: d5207c44 0072a080
	s_and_b32 s28, s0, s45                                     // 000000002858: 8b1c2d00
	s_and_b32 s29, s0, s46                                     // 00000000285c: 8b1d2e00
	v_or_b32_e32 v73, 5, v75                                   // 000000002860: 38929685
	s_wait_alu depctr_sa_sdst(0)                               // 000000002864: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v71, s29                         // 000000002868: d5010046 00768e80
	v_cndmask_b32_e64 v69, 0, v72, s29                         // 000000002870: d5010045 00769080
	v_cndmask_b32_e64 v72, 0, v67, s28                         // 000000002878: d5010048 00728680
	v_cndmask_b32_e64 v71, 0, v68, s28                         // 000000002880: d5010047 00728880
	v_mov_b32_e32 v74, s89                                     // 000000002888: 7e940259
	v_add_co_u32 v67, s30, s68, v70                            // 00000000288c: d7001e43 02028c44
	s_wait_alu depctr_va_sdst(0)                               // 000000002894: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s69, v69, s30               // 000000002898: d5207c44 007a8a45
	v_add_co_u32 v69, s30, s68, v72                            // 0000000028a0: d7001e45 02029044
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a8: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s69, v71, s30               // 0000000028ac: d5207c46 007a8e45
	v_or_b32_e32 v71, 4, v75                                   // 0000000028b4: 388e9684
	v_mov_b32_e32 v72, s89                                     // 0000000028b8: 7e900259
	v_add_co_u32 v77, s30, v79, 4                              // 0000000028bc: d7001e4d 0201094f
	v_cmp_gt_i64_e64 s47, s[64:65], v[73:74]                   // 0000000028c4: d454002f 02029240
	s_wait_alu depctr_va_sdst(0)                               // 0000000028cc: bf88f19f
	v_add_co_ci_u32_e64 v78, null, 0, v80, s30                 // 0000000028d0: d5207c4e 007aa080
	v_cmp_gt_i64_e64 s48, s[64:65], v[71:72]                   // 0000000028d8: d4540030 02028e40
	v_add_co_u32 v71, s30, v79, 5                              // 0000000028e0: d7001e47 02010b4f
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v80, s30                 // 0000000028ec: d5207c48 007aa080
	s_and_b32 s31, s0, s47                                     // 0000000028f4: 8b1f2f00
	s_and_b32 s30, s0, s48                                     // 0000000028f8: 8b1e3000
	s_add_nc_u64 s[90:91], s[36:37], s[86:87]                  // 0000000028fc: a9da5624
	s_wait_alu depctr_sa_sdst(0)                               // 000000002900: bf88ff9e
	v_cndmask_b32_e64 v74, 0, v77, s30                         // 000000002904: d501004a 007a9a80
	v_cndmask_b32_e64 v73, 0, v78, s30                         // 00000000290c: d5010049 007a9c80
	v_cndmask_b32_e64 v78, 0, v71, s31                         // 000000002914: d501004e 007e8e80
	v_cndmask_b32_e64 v77, 0, v72, s31                         // 00000000291c: d501004d 007e9080
	s_and_b32 s36, s1, s44                                     // 000000002924: 8b242c01
	v_add_co_u32 v71, s33, s68, v74                            // 000000002928: d7002147 02029444
	s_wait_alu depctr_va_sdst(0)                               // 000000002930: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s69, v73, s33               // 000000002934: d5207c48 00869245
	v_add_co_u32 v73, s33, s68, v78                            // 00000000293c: d7002149 02029c44
	s_wait_alu depctr_va_sdst(0)                               // 000000002944: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s69, v77, s33               // 000000002948: d5207c4a 00869a45
	v_or_b32_e32 v77, 6, v75                                   // 000000002950: 389a9686
	v_mov_b32_e32 v78, s89                                     // 000000002954: 7e9c0259
	v_or_b32_e32 v75, 7, v75                                   // 000000002958: 38969687
	v_add_co_u32 v81, s33, v79, 6                              // 00000000295c: d7002151 02010d4f
	s_wait_alu depctr_va_sdst(0)                               // 000000002964: bf88f19f
	v_add_co_ci_u32_e64 v82, null, 0, v80, s33                 // 000000002968: d5207c52 0086a080
	v_cmp_gt_i64_e64 s49, s[64:65], v[77:78]                   // 000000002970: d4540031 02029a40
	v_cmp_gt_i64_e64 s50, s[64:65], v[75:76]                   // 000000002978: d4540032 02029640
	v_add_co_u32 v75, s33, v79, 7                              // 000000002980: d700214b 02010f4f
	s_wait_alu depctr_va_sdst(0)                               // 000000002988: bf88f19f
	v_add_co_ci_u32_e64 v76, null, 0, v80, s33                 // 00000000298c: d5207c4c 0086a080
	s_and_b32 s33, s0, s49                                     // 000000002994: 8b213100
	s_and_b32 s34, s0, s50                                     // 000000002998: 8b223200
	s_wait_alu depctr_sa_sdst(0)                               // 00000000299c: bf88ff9e
	v_cndmask_b32_e64 v78, 0, v81, s33                         // 0000000029a0: d501004e 0086a280
	v_cndmask_b32_e64 v77, 0, v82, s33                         // 0000000029a8: d501004d 0086a480
	v_cndmask_b32_e64 v80, 0, v75, s34                         // 0000000029b0: d5010050 008a9680
	v_cndmask_b32_e64 v79, 0, v76, s34                         // 0000000029b8: d501004f 008a9880
	s_and_b32 s38, s1, s46                                     // 0000000029c0: 8b262e01
	v_add_co_u32 v75, s35, s68, v78                            // 0000000029c4: d700234b 02029c44
	s_wait_alu depctr_va_sdst(0)                               // 0000000029cc: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s69, v77, s35               // 0000000029d0: d5207c4c 008e9a45
	v_add_co_u32 v77, s35, s68, v80                            // 0000000029d8: d700234d 0202a044
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e0: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s69, v79, s35               // 0000000029e4: d5207c4e 008e9e45
	v_add_co_u32 v80, s35, v8, s90                             // 0000000029ec: d7002350 0200b508
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f4: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s91, v9, s35                // 0000000029f8: d5207c51 008e125b
	s_clause 0x7                                               // 000000002a00: bf850007
	global_load_d16_u8 v63, v[63:64], off                      // 000000002a04: ee07807c 0000003f 0000003f
	global_load_d16_hi_u8 v63, v[65:66], off                   // 000000002a10: ee08407c 0000003f 00000041
	global_load_d16_u8 v64, v[67:68], off                      // 000000002a1c: ee07807c 00000040 00000043
	global_load_d16_hi_u8 v64, v[69:70], off                   // 000000002a28: ee08407c 00000040 00000045
	global_load_d16_u8 v65, v[71:72], off                      // 000000002a34: ee07807c 00000041 00000047
	global_load_d16_hi_u8 v65, v[73:74], off                   // 000000002a40: ee08407c 00000041 00000049
	global_load_d16_u8 v66, v[75:76], off                      // 000000002a4c: ee07807c 00000042 0000004b
	global_load_d16_hi_u8 v66, v[77:78], off                   // 000000002a58: ee08407c 00000042 0000004d
	v_add_co_u32 v67, s35, v80, s54                            // 000000002a64: d7002343 02006d50
	s_wait_alu depctr_va_sdst(0)                               // 000000002a6c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s55, v81, s35               // 000000002a70: d5207c44 008ea237
	s_and_b32 s35, s1, s43                                     // 000000002a78: 8b232b01
	v_cndmask_b32_e64 v72, 0, v67, s36                         // 000000002a7c: d5010048 00928680
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a84: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v80, s35                         // 000000002a88: d5010046 008ea080
	v_cndmask_b32_e64 v69, 0, v81, s35                         // 000000002a90: d5010045 008ea280
	v_cndmask_b32_e64 v71, 0, v68, s36                         // 000000002a98: d5010047 00928880
	s_and_b32 s40, s1, s47                                     // 000000002aa0: 8b282f01
	s_and_b32 s42, s1, s50                                     // 000000002aa4: 8b2a3201
	v_add_co_u32 v67, s37, s66, v70                            // 000000002aa8: d7002543 02028c42
	s_wait_alu depctr_va_sdst(0)                               // 000000002ab0: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s67, v69, s37               // 000000002ab4: d5207c44 00968a43
	v_add_co_u32 v70, s37, s66, v72                            // 000000002abc: d7002546 02029042
	s_wait_alu depctr_va_sdst(0)                               // 000000002ac4: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s67, v71, s37               // 000000002ac8: d5207c47 00968e43
	v_add_co_u32 v69, s37, v80, s82                            // 000000002ad0: d7002545 0200a550
	s_wait_alu depctr_va_sdst(0)                               // 000000002ad8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s83, v81, s37               // 000000002adc: d5207c48 0096a253
	v_add_co_u32 v73, s37, v80, s80                            // 000000002ae4: d7002549 0200a150
	s_wait_alu depctr_va_sdst(0)                               // 000000002aec: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s81, v81, s37               // 000000002af0: d5207c4a 0096a251
	v_cndmask_b32_e64 v69, 0, v69, s38                         // 000000002af8: d5010045 009a8a80
	s_and_b32 s37, s1, s45                                     // 000000002b00: 8b252d01
	v_cndmask_b32_e64 v75, 0, v72, s38                         // 000000002b04: d501004b 009a9080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b0c: bf88ff9e
	v_cndmask_b32_e64 v76, 0, v74, s37                         // 000000002b10: d501004c 00969480
	v_cndmask_b32_e64 v74, 0, v73, s37                         // 000000002b18: d501004a 00969280
	v_add_co_u32 v72, s39, s66, v69                            // 000000002b20: d7002748 02028a42
	s_wait_alu depctr_va_sdst(0)                               // 000000002b28: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s67, v75, s39               // 000000002b2c: d5207c49 009e9643
	s_delay_alu instid0(valu_dep_3)                            // 000000002b34: bf870003
	v_add_co_u32 v74, s39, s66, v74                            // 000000002b38: d700274a 02029442
	s_wait_alu depctr_va_sdst(0)                               // 000000002b40: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s67, v76, s39               // 000000002b44: d5207c4b 009e9843
	v_add_co_u32 v69, s39, v80, s84                            // 000000002b4c: d7002745 0200a950
	s_wait_alu depctr_va_sdst(0)                               // 000000002b54: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s85, v81, s39               // 000000002b58: d5207c4c 009ea255
	v_add_co_u32 v77, s39, v80, s78                            // 000000002b60: d700274d 02009d50
	s_wait_alu depctr_va_sdst(0)                               // 000000002b68: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s79, v81, s39               // 000000002b6c: d5207c4e 009ea24f
	s_and_b32 s39, s1, s48                                     // 000000002b74: 8b273001
	s_and_b32 s43, s2, s43                                     // 000000002b78: 8b2b2b02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b7c: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s39                         // 000000002b80: d5010045 009e8a80
	v_cndmask_b32_e64 v79, 0, v76, s39                         // 000000002b88: d501004f 009e9880
	v_cndmask_b32_e64 v82, 0, v78, s40                         // 000000002b90: d5010052 00a29c80
	v_cndmask_b32_e64 v78, 0, v77, s40                         // 000000002b98: d501004e 00a29a80
	s_and_b32 s44, s2, s44                                     // 000000002ba0: 8b2c2c02
	v_add_co_u32 v76, s41, s66, v69                            // 000000002ba4: d700294c 02028a42
	s_wait_alu depctr_va_sdst(0)                               // 000000002bac: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s67, v79, s41               // 000000002bb0: d5207c4d 00a69e43
	v_add_co_u32 v78, s41, s66, v78                            // 000000002bb8: d700294e 02029c42
	s_wait_alu depctr_va_sdst(0)                               // 000000002bc0: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s67, v82, s41               // 000000002bc4: d5207c4f 00a6a443
	v_add_co_u32 v69, s41, v80, s76                            // 000000002bcc: d7002945 02009950
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd4: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s77, v81, s41               // 000000002bd8: d5207c52 00a6a24d
	v_add_co_u32 v80, s41, v80, s74                            // 000000002be0: d7002950 02009550
	s_wait_alu depctr_va_sdst(0)                               // 000000002be8: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s75, v81, s41               // 000000002bec: d5207c51 00a6a24b
	s_and_b32 s41, s1, s49                                     // 000000002bf4: 8b293101
	v_cndmask_b32_e64 v84, 0, v80, s42                         // 000000002bf8: d5010054 00aaa080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c00: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s41                         // 000000002c04: d5010045 00a68a80
	v_cndmask_b32_e64 v82, 0, v82, s41                         // 000000002c0c: d5010052 00a6a480
	v_cndmask_b32_e64 v83, 0, v81, s42                         // 000000002c14: d5010053 00aaa280
	s_and_b32 s46, s2, s46                                     // 000000002c1c: 8b2e2e02
	s_and_b32 s45, s2, s45                                     // 000000002c20: 8b2d2d02
	v_add_co_u32 v80, s51, s66, v69                            // 000000002c24: d7003350 02028a42
	s_wait_alu depctr_va_sdst(0)                               // 000000002c2c: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s67, v82, s51               // 000000002c30: d5207c51 00cea443
	v_add_co_u32 v82, s51, s66, v84                            // 000000002c38: d7003352 0202a842
	s_wait_alu depctr_va_sdst(0)                               // 000000002c40: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s67, v83, s51               // 000000002c44: d5207c53 00cea643
	v_add_co_u32 v84, s51, v12, s90                            // 000000002c4c: d7003354 0200b50c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c54: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s91, v13, s51               // 000000002c58: d5207c55 00ce1a5b
	s_clause 0x7                                               // 000000002c60: bf850007
	global_load_d16_u8 v69, v[67:68], off                      // 000000002c64: ee07807c 00000045 00000043
	global_load_d16_hi_u8 v69, v[70:71], off                   // 000000002c70: ee08407c 00000045 00000046
	global_load_d16_u8 v70, v[72:73], off                      // 000000002c7c: ee07807c 00000046 00000048
	global_load_d16_hi_u8 v70, v[74:75], off                   // 000000002c88: ee08407c 00000046 0000004a
	global_load_d16_u8 v71, v[76:77], off                      // 000000002c94: ee07807c 00000047 0000004c
	global_load_d16_hi_u8 v71, v[78:79], off                   // 000000002ca0: ee08407c 00000047 0000004e
	global_load_d16_u8 v72, v[80:81], off                      // 000000002cac: ee07807c 00000048 00000050
	global_load_d16_hi_u8 v72, v[82:83], off                   // 000000002cb8: ee08407c 00000048 00000052
	v_add_co_u32 v67, s51, v84, s54                            // 000000002cc4: d7003343 02006d54
	s_wait_alu depctr_va_sdst(0)                               // 000000002ccc: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s55, v85, s51               // 000000002cd0: d5207c44 00ceaa37
	v_cndmask_b32_e64 v74, 0, v84, s43                         // 000000002cd8: d501004a 00aea880
	v_cndmask_b32_e64 v73, 0, v85, s43                         // 000000002ce0: d5010049 00aeaa80
	v_cndmask_b32_e64 v76, 0, v67, s44                         // 000000002ce8: d501004c 00b28680
	s_delay_alu instid0(valu_dep_4)                            // 000000002cf0: bf870004
	v_cndmask_b32_e64 v75, 0, v68, s44                         // 000000002cf4: d501004b 00b28880
	s_and_b32 s48, s2, s48                                     // 000000002cfc: 8b303002
	v_add_co_u32 v67, s51, s66, v74                            // 000000002d00: d7003343 02029442
	s_wait_alu depctr_va_sdst(0)                               // 000000002d08: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s67, v73, s51               // 000000002d0c: d5207c44 00ce9243
	v_add_co_u32 v74, s51, s66, v76                            // 000000002d14: d700334a 02029842
	s_wait_alu depctr_va_sdst(0)                               // 000000002d1c: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s67, v75, s51               // 000000002d20: d5207c4b 00ce9643
	v_add_co_u32 v73, s51, v84, s82                            // 000000002d28: d7003349 0200a554
	s_wait_alu depctr_va_sdst(0)                               // 000000002d30: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s83, v85, s51               // 000000002d34: d5207c4c 00ceaa53
	v_add_co_u32 v77, s51, v84, s80                            // 000000002d3c: d700334d 0200a154
	s_wait_alu depctr_va_sdst(0)                               // 000000002d44: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s81, v85, s51               // 000000002d48: d5207c4e 00ceaa51
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d50: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v73, s46                         // 000000002d54: d5010049 00ba9280
	v_cndmask_b32_e64 v79, 0, v76, s46                         // 000000002d5c: d501004f 00ba9880
	s_and_b32 s47, s2, s47                                     // 000000002d64: 8b2f2f02
	v_cndmask_b32_e64 v80, 0, v78, s45                         // 000000002d68: d5010050 00b69c80
	v_cndmask_b32_e64 v78, 0, v77, s45                         // 000000002d70: d501004e 00b69a80
	v_add_co_u32 v76, s51, s66, v73                            // 000000002d78: d700334c 02029242
	s_wait_alu depctr_va_sdst(0)                               // 000000002d80: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s67, v79, s51               // 000000002d84: d5207c4d 00ce9e43
	s_delay_alu instid0(valu_dep_3)                            // 000000002d8c: bf870003
	v_add_co_u32 v78, s51, s66, v78                            // 000000002d90: d700334e 02029c42
	s_wait_alu depctr_va_sdst(0)                               // 000000002d98: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s67, v80, s51               // 000000002d9c: d5207c4f 00cea043
	v_add_co_u32 v73, s51, v84, s84                            // 000000002da4: d7003349 0200a954
	s_wait_alu depctr_va_sdst(0)                               // 000000002dac: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s85, v85, s51               // 000000002db0: d5207c50 00ceaa55
	v_add_co_u32 v81, s51, v84, s78                            // 000000002db8: d7003351 02009d54
	s_wait_alu depctr_va_sdst(0)                               // 000000002dc0: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s79, v85, s51               // 000000002dc4: d5207c52 00ceaa4f
	v_cndmask_b32_e64 v73, 0, v73, s48                         // 000000002dcc: d5010049 00c29280
	v_cndmask_b32_e64 v83, 0, v80, s48                         // 000000002dd4: d5010053 00c2a080
	s_and_b32 s49, s2, s49                                     // 000000002ddc: 8b313102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002de0: bf88ff9e
	v_cndmask_b32_e64 v86, 0, v82, s47                         // 000000002de4: d5010056 00bea480
	v_cndmask_b32_e64 v82, 0, v81, s47                         // 000000002dec: d5010052 00bea280
	v_add_co_u32 v80, s51, s66, v73                            // 000000002df4: d7003350 02029242
	s_wait_alu depctr_va_sdst(0)                               // 000000002dfc: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s67, v83, s51               // 000000002e00: d5207c51 00cea643
	s_delay_alu instid0(valu_dep_3)                            // 000000002e08: bf870003
	v_add_co_u32 v82, s51, s66, v82                            // 000000002e0c: d7003352 0202a442
	s_wait_alu depctr_va_sdst(0)                               // 000000002e14: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s67, v86, s51               // 000000002e18: d5207c53 00ceac43
	v_add_co_u32 v73, s51, v84, s76                            // 000000002e20: d7003349 02009954
	s_wait_alu depctr_va_sdst(0)                               // 000000002e28: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s77, v85, s51               // 000000002e2c: d5207c56 00ceaa4d
	v_add_co_u32 v84, s51, v84, s74                            // 000000002e34: d7003354 02009554
	s_wait_alu depctr_va_sdst(0)                               // 000000002e3c: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s75, v85, s51               // 000000002e40: d5207c55 00ceaa4b
	v_cndmask_b32_e64 v73, 0, v73, s49                         // 000000002e48: d5010049 00c69280
	s_and_b32 s50, s2, s50                                     // 000000002e50: 8b323202
	v_cndmask_b32_e64 v86, 0, v86, s49                         // 000000002e54: d5010056 00c6ac80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e5c: bf88ff9e
	v_cndmask_b32_e64 v88, 0, v84, s50                         // 000000002e60: d5010058 00caa880
	s_lshr_b64 s[90:91], s[88:89], 5                           // 000000002e68: 85da8558
	v_cndmask_b32_e64 v87, 0, v85, s50                         // 000000002e6c: d5010057 00caaa80
	v_add_co_u32 v84, s51, s66, v73                            // 000000002e74: d7003354 02029242
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e7c: bf88ff9e
	s_mul_u64 s[90:91], s[90:91], s[54:55]                     // 000000002e80: aada365a
	v_add_co_ci_u32_e64 v85, null, s67, v86, s51               // 000000002e84: d5207c55 00ceac43
	v_add_co_u32 v86, s51, s66, v88                            // 000000002e8c: d7003356 0202b042
	s_lshr_b64 s[94:95], s[88:89], 3                           // 000000002e94: 85de8358
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e98: bf88ff9e
	s_lshl_b64 s[90:91], s[90:91], 2                           // 000000002e9c: 84da825a
	v_add_co_ci_u32_e64 v87, null, s67, v87, s51               // 000000002ea0: d5207c57 00ceae43
	v_add_co_u32 v88, s51, v33, s94                            // 000000002ea8: d7003358 0200bd21
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eb0: bf88ff9e
	s_add_nc_u64 s[90:91], s[62:63], s[90:91]                  // 000000002eb4: a9da5a3e
	v_add_co_ci_u32_e64 v89, null, s95, v34, s51               // 000000002eb8: d5207c59 00ce445f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ec0: bf88ff9e
	v_add_co_u32 v90, s51, s90, v14                            // 000000002ec4: d700335a 02021c5a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ecc: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s91, v15, s51               // 000000002ed0: d5207c5b 00ce1e5b
	s_clause 0x7                                               // 000000002ed8: bf850007
	global_load_d16_u8 v73, v[67:68], off                      // 000000002edc: ee07807c 00000049 00000043
	global_load_d16_hi_u8 v73, v[74:75], off                   // 000000002ee8: ee08407c 00000049 0000004a
	global_load_d16_u8 v74, v[76:77], off                      // 000000002ef4: ee07807c 0000004a 0000004c
	global_load_d16_hi_u8 v74, v[78:79], off                   // 000000002f00: ee08407c 0000004a 0000004e
	global_load_d16_u8 v75, v[84:85], off                      // 000000002f0c: ee07807c 0000004b 00000054
	global_load_d16_hi_u8 v75, v[86:87], off                   // 000000002f18: ee08407c 0000004b 00000056
	global_load_d16_u8 v76, v[80:81], off                      // 000000002f24: ee07807c 0000004c 00000050
	global_load_d16_hi_u8 v76, v[82:83], off                   // 000000002f30: ee08407c 0000004c 00000052
	global_load_b32 v85, v[88:89], off                         // 000000002f3c: ee05007c 00000055 00000058
	global_load_b32 v86, v[90:91], off                         // 000000002f48: ee05007c 00000056 0000005a
	v_add_co_u32 v67, s51, v36, s94                            // 000000002f54: d7003343 0200bd24
	s_wait_alu depctr_va_sdst(0)                               // 000000002f5c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s95, v37, s51               // 000000002f60: d5207c44 00ce4a5f
	v_add_co_u32 v77, s51, v38, s94                            // 000000002f68: d700334d 0200bd26
	s_wait_alu depctr_va_sdst(0)                               // 000000002f70: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s95, v39, s51               // 000000002f74: d5207c4e 00ce4e5f
	v_add_co_u32 v79, s51, v40, s94                            // 000000002f7c: d700334f 0200bd28
	s_wait_alu depctr_va_sdst(0)                               // 000000002f84: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s95, v41, s51               // 000000002f88: d5207c50 00ce525f
	v_add_co_u32 v81, s51, v42, s94                            // 000000002f90: d7003351 0200bd2a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f98: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s95, v43, s51               // 000000002f9c: d5207c52 00ce565f
	v_add_co_u32 v83, s51, v44, s94                            // 000000002fa4: d7003353 0200bd2c
	s_wait_alu depctr_va_sdst(0)                               // 000000002fac: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s95, v45, s51               // 000000002fb0: d5207c54 00ce5a5f
	s_clause 0x4                                               // 000000002fb8: bf850004
	global_load_b32 v87, v[67:68], off                         // 000000002fbc: ee05007c 00000057 00000043
	global_load_b32 v88, v[77:78], off                         // 000000002fc8: ee05007c 00000058 0000004d
	global_load_b32 v89, v[79:80], off                         // 000000002fd4: ee05007c 00000059 0000004f
	global_load_b32 v90, v[81:82], off                         // 000000002fe0: ee05007c 0000005a 00000051
	global_load_b32 v83, v[83:84], off                         // 000000002fec: ee05007c 00000053 00000053
	v_add_co_u32 v67, s51, v46, s94                            // 000000002ff8: d7003343 0200bd2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003000: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s95, v47, s51               // 000000003004: d5207c44 00ce5e5f
	v_add_co_u32 v77, s51, v48, s94                            // 00000000300c: d700334d 0200bd30
	s_wait_alu depctr_va_sdst(0)                               // 000000003014: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s95, v49, s51               // 000000003018: d5207c4e 00ce625f
	v_add_co_u32 v79, s51, s90, v16                            // 000000003020: d700334f 0202205a
	s_wait_alu depctr_va_sdst(0)                               // 000000003028: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s91, v17, s51               // 00000000302c: d5207c50 00ce225b
	s_clause 0x1                                               // 000000003034: bf850001
	global_load_b32 v84, v[67:68], off                         // 000000003038: ee05007c 00000054 00000043
	global_load_b32 v91, v[77:78], off                         // 000000003044: ee05007c 0000005b 0000004d
	global_load_b32 v92, v[79:80], off                         // 000000003050: ee05007c 0000005c 0000004f
	s_wait_loadcnt 0x38                                        // 00000000305c: bfc00038
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 000000003060: d65d0032 01aa6480
	v_cndmask_b16 v50.h, 0, v50.h, s3                          // 000000003068: d65d5032 000e6480
	s_wait_loadcnt 0x36                                        // 000000003070: bfc00036
	v_cndmask_b16 v51.l, 0, v51.l, s5                          // 000000003074: d65d0033 00166680
	s_wait_loadcnt 0x34                                        // 00000000307c: bfc00034
	v_cndmask_b16 v53.h, 0, v53.h, s7                          // 000000003080: d65d5035 001e6a80
	v_cndmask_b16 v53.l, 0, v53.l, s6                          // 000000003088: d65d0035 001a6a80
	v_cndmask_b16 v51.h, 0, v51.h, s4                          // 000000003090: d65d5033 00126680
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000003098: d7385032 02026488
	v_and_b16 v51.l, 0xff, v51.l                               // 0000000030a0: d7620033 020266ff 000000ff
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 0000000030ac: d7385035 02026a88
	v_and_b16 v53.l, 0xff, v53.l                               // 0000000030b4: d7620035 02026aff 000000ff
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 0000000030c0: d7385033 02026688
	v_and_b16 v50.l, 0xff, v50.l                               // 0000000030c8: d7620032 020264ff 000000ff
	s_wait_loadcnt 0x32                                        // 0000000030d4: bfc00032
	v_cndmask_b16 v54.h, 0, v54.h, s9                          // 0000000030d8: d65d5036 00266c80
	v_cndmask_b16 v54.l, 0, v54.l, s8                          // 0000000030e0: d65d0036 00226c80
	v_or_b16 v80.l, v53.l, v53.h op_sel:[0,1,0]                // 0000000030e8: d7631050 02026b35
	v_or_b16 v79.h, v51.l, v51.h op_sel:[0,1,1]                // 0000000030f0: d763504f 02026733
	v_or_b16 v79.l, v50.l, v50.h op_sel:[0,1,0]                // 0000000030f8: d763104f 02026532
	s_wait_loadcnt 0x2a                                        // 000000003100: bfc0002a
	v_cndmask_b16 v50.h, 0, v58.h, s17                         // 000000003104: d65d5032 00467480
	v_cndmask_b16 v51.l, 0, v58.l, s16                         // 00000000310c: d65d0033 00427480
	v_cndmask_b16 v51.h, 0, v57.h, s15                         // 000000003114: d65d5033 003e7280
	v_cndmask_b16 v53.l, 0, v57.l, s14                         // 00000000311c: d65d0035 003a7280
	v_lshlrev_b16 v54.h, 8, v54.h op_sel:[0,1,1]               // 000000003124: d7385036 02026c88
	v_and_b16 v54.l, 0xff, v54.l                               // 00000000312c: d7620036 02026cff 000000ff
	v_cndmask_b16 v50.l, 0, v56.l, s13                         // 000000003138: d65d0032 00367080
	v_cndmask_b16 v53.h, 0, v56.h, s12                         // 000000003140: d65d5035 00327080
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000003148: d7385032 02026488
	v_and_b16 v51.l, 0xff, v51.l                               // 000000003150: d7620033 020266ff 000000ff
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 00000000315c: d7385033 02026688
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003164: d7620035 02026aff 000000ff
	v_or_b16 v80.h, v54.l, v54.h op_sel:[0,1,1]                // 000000003170: d7635050 02026d36
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 000000003178: d7385035 02026a88
	v_and_b16 v50.l, 0xff, v50.l                               // 000000003180: d7620032 020264ff 000000ff
	v_cndmask_b16 v54.l, 0, v55.h, s11                         // 00000000318c: d65d1036 002e6e80
	v_cndmask_b16 v54.h, 0, v55.l, s10                         // 000000003194: d65d4036 002a6e80
	v_or_b16 v68.h, v51.l, v50.h op_sel:[0,1,1]                // 00000000319c: d7635044 02026533
	v_or_b16 v68.l, v53.l, v51.h op_sel:[0,1,0]                // 0000000031a4: d7631044 02026735
	s_wait_loadcnt 0x28                                        // 0000000031ac: bfc00028
	v_cndmask_b16 v51.l, 0, v59.l, s18                         // 0000000031b0: d65d0033 004a7680
	v_cndmask_b16 v51.h, 0, v59.h, s19                         // 0000000031b8: d65d5033 004e7680
	v_or_b16 v67.h, v50.l, v53.h op_sel:[0,1,1]                // 0000000031c0: d7635043 02026b32
	v_lshlrev_b16 v50.l, 8, v54.l                              // 0000000031c8: d7380032 02026c88
	v_and_b16 v50.h, 0xff, v54.h op_sel:[0,1,1]                // 0000000031d0: d7625032 02026cff 000000ff
	s_wait_loadcnt 0x26                                        // 0000000031dc: bfc00026
	v_cndmask_b16 v53.l, 0, v60.l, s21                         // 0000000031e0: d65d0035 00567880
	s_wait_loadcnt 0x22                                        // 0000000031e8: bfc00022
	v_cndmask_b16 v53.h, 0, v62.h, s25                         // 0000000031ec: d65d5035 00667c80
	v_cndmask_b16 v54.l, 0, v62.l, s24                         // 0000000031f4: d65d0036 00627c80
	v_cndmask_b16 v54.h, 0, v61.h, s23                         // 0000000031fc: d65d5036 005e7a80
	v_cndmask_b16 v55.l, 0, v61.l, s22                         // 000000003204: d65d0037 005a7a80
	v_cndmask_b16 v55.h, 0, v60.h, s20                         // 00000000320c: d65d5037 00527880
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000003214: d7385033 02026688
	v_and_b16 v51.l, 0xff, v51.l                               // 00000000321c: d7620033 020266ff 000000ff
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 000000003228: d7385035 02026a88
	v_and_b16 v54.l, 0xff, v54.l                               // 000000003230: d7620036 02026cff 000000ff
	v_lshlrev_b16 v54.h, 8, v54.h op_sel:[0,1,1]               // 00000000323c: d7385036 02026c88
	v_and_b16 v55.l, 0xff, v55.l                               // 000000003244: d7620037 02026eff 000000ff
	v_lshlrev_b16 v55.h, 8, v55.h op_sel:[0,1,1]               // 000000003250: d7385037 02026e88
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003258: d7620035 02026aff 000000ff
	v_or_b16 v67.l, v50.h, v50.l op_sel:[1,0,0]                // 000000003264: d7630843 02026532
	v_or_b16 v81.l, v51.l, v51.h op_sel:[0,1,0]                // 00000000326c: d7631051 02026733
	v_or_b16 v82.h, v54.l, v53.h op_sel:[0,1,1]                // 000000003274: d7635052 02026b36
	v_or_b16 v82.l, v55.l, v54.h op_sel:[0,1,0]                // 00000000327c: d7631052 02026d37
	v_or_b16 v81.h, v53.l, v55.h op_sel:[0,1,1]                // 000000003284: d7635051 02026f35
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[79:80], v[67:68], 0// 00000000328c: cc464035 1a02874f
	s_add_nc_u64 s[88:89], s[88:89], 32                        // 000000003294: a9d8a058
	s_wait_alu depctr_sa_sdst(0)                               // 000000003298: bf88ff9e
	v_cmp_lt_i64_e64 s3, s[88:89], s[60:61]                    // 00000000329c: d4510003 02007858
	s_and_b32 vcc_lo, exec_lo, s3                              // 0000000032a4: 8b6a037e
	s_wait_loadcnt 0x20                                        // 0000000032a8: bfc00020
	v_cndmask_b16 v50.l, 0, v63.l, s26                         // 0000000032ac: d65d0032 006a7e80
	v_cndmask_b16 v50.h, 0, v63.h, s27                         // 0000000032b4: d65d5032 006e7e80
	s_wait_loadcnt 0x1e                                        // 0000000032bc: bfc0001e
	v_cndmask_b16 v51.l, 0, v64.l, s29                         // 0000000032c0: d65d0033 00768080
	v_cndmask_b16 v62.h, 0, v64.h, s28                         // 0000000032c8: d65d503e 00728080
	s_wait_loadcnt 0x1c                                        // 0000000032d0: bfc0001c
	v_cndmask_b16 v62.l, 0, v65.l, s30                         // 0000000032d4: d65d003e 007a8280
	v_cndmask_b16 v61.h, 0, v65.h, s31                         // 0000000032dc: d65d503d 007e8280
	s_wait_loadcnt 0x1a                                        // 0000000032e4: bfc0001a
	v_cndmask_b16 v61.l, 0, v66.l, s33                         // 0000000032e8: d65d003d 00868480
	v_cndmask_b16 v51.h, 0, v66.h, s34                         // 0000000032f0: d65d5033 008a8480
	v_lshlrev_b16 v78.h, 8, v62.h op_sel:[0,1,1]               // 0000000032f8: d738504e 02027c88
	v_and_b16 v78.l, 0xff, v62.l                               // 000000003300: d762004e 02027cff 000000ff
	v_lshlrev_b16 v77.h, 8, v61.h op_sel:[0,1,1]               // 00000000330c: d738504d 02027a88
	v_and_b16 v77.l, 0xff, v61.l                               // 000000003314: d762004d 02027aff 000000ff
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 000000003320: d7385033 02026688
	v_and_b16 v51.l, 0xff, v51.l                               // 000000003328: d7620033 020266ff 000000ff
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 000000003334: d7385032 02026488
	v_and_b16 v50.l, 0xff, v50.l                               // 00000000333c: d7620032 020264ff 000000ff
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[79:80], v[81:82], 0// 000000003348: cc46403d 1a02a34f
	v_or_b16 v79.h, v77.l, v51.h op_sel:[0,1,1]                // 000000003350: d763504f 0202674d
	v_or_b16 v79.l, v78.l, v77.h op_sel:[0,1,0]                // 000000003358: d763104f 02029b4e
	v_or_b16 v78.h, v51.l, v78.h op_sel:[0,1,1]                // 000000003360: d763504e 02029d33
	v_or_b16 v78.l, v50.l, v50.h op_sel:[0,1,0]                // 000000003368: d763104e 02026532
	s_wait_loadcnt 0x18                                        // 000000003370: bfc00018
	v_cndmask_b16 v50.l, 0, v69.l, s35                         // 000000003374: d65d0032 008e8a80
	v_cndmask_b16 v50.h, 0, v69.h, s36                         // 00000000337c: d65d5032 00928a80
	s_wait_loadcnt 0x16                                        // 000000003384: bfc00016
	v_cndmask_b16 v51.l, 0, v70.l, s38                         // 000000003388: d65d0033 009a8c80
	v_cndmask_b16 v70.h, 0, v70.h, s37                         // 000000003390: d65d5046 00968c80
	s_wait_loadcnt 0x14                                        // 000000003398: bfc00014
	v_cndmask_b16 v70.l, 0, v71.l, s39                         // 00000000339c: d65d0046 009e8e80
	v_cndmask_b16 v69.h, 0, v71.h, s40                         // 0000000033a4: d65d5045 00a28e80
	s_wait_loadcnt 0x12                                        // 0000000033ac: bfc00012
	v_cndmask_b16 v69.l, 0, v72.l, s41                         // 0000000033b0: d65d0045 00a69080
	v_cndmask_b16 v51.h, 0, v72.h, s42                         // 0000000033b8: d65d5033 00aa9080
	v_lshlrev_b16 v70.h, 8, v70.h op_sel:[0,1,1]               // 0000000033c0: d7385046 02028c88
	v_and_b16 v51.l, 0xff, v51.l                               // 0000000033c8: d7620033 020266ff 000000ff
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 0000000033d4: d7385045 02028a88
	v_and_b16 v69.l, 0xff, v69.l                               // 0000000033dc: d7620045 02028aff 000000ff
	v_lshlrev_b16 v51.h, 8, v51.h op_sel:[0,1,1]               // 0000000033e8: d7385033 02026688
	v_and_b16 v70.l, 0xff, v70.l                               // 0000000033f0: d7620046 02028cff 000000ff
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 0000000033fc: d7385032 02026488
	v_and_b16 v50.l, 0xff, v50.l                               // 000000003404: d7620032 020264ff 000000ff
	v_or_b16 v71.h, v51.l, v70.h op_sel:[0,1,1]                // 000000003410: d7635047 02028d33
	v_or_b16 v72.h, v69.l, v51.h op_sel:[0,1,1]                // 000000003418: d7635048 02026745
	v_or_b16 v72.l, v70.l, v69.h op_sel:[0,1,0]                // 000000003420: d7631048 02028b46
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000003428: bf870094
	v_or_b16 v71.l, v50.l, v50.h op_sel:[0,1,0]                // 00000000342c: d7631047 02026532
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[78:79], v[71:72], v[53:60]// 000000003434: cc464035 1cd68f4e
	s_wait_loadcnt 0x10                                        // 00000000343c: bfc00010
	v_cndmask_b16 v50.l, 0, v73.l, s43                         // 000000003440: d65d0032 00ae9280
	v_cndmask_b16 v50.h, 0, v73.h, s44                         // 000000003448: d65d5032 00b29280
	s_wait_loadcnt 0xe                                         // 000000003450: bfc0000e
	v_cndmask_b16 v70.h, 0, v74.l, s46                         // 000000003454: d65d4046 00ba9480
	v_cndmask_b16 v70.l, 0, v74.h, s45                         // 00000000345c: d65d1046 00b69480
	s_wait_loadcnt 0xc                                         // 000000003464: bfc0000c
	v_cndmask_b16 v51.h, 0, v75.l, s49                         // 000000003468: d65d4033 00c69680
	v_cndmask_b16 v51.l, 0, v75.h, s50                         // 000000003470: d65d1033 00ca9680
	s_wait_loadcnt 0xa                                         // 000000003478: bfc0000a
	v_cndmask_b16 v69.h, 0, v76.l, s48                         // 00000000347c: d65d4045 00c29880
	v_cndmask_b16 v69.l, 0, v76.h, s47                         // 000000003484: d65d1045 00be9880
	v_lshlrev_b16 v50.h, 8, v50.h op_sel:[0,1,1]               // 00000000348c: d7385032 02026488
	v_and_b16 v51.h, 0xff, v51.h op_sel:[0,1,1]                // 000000003494: d7625033 020266ff 000000ff
	v_lshlrev_b16 v51.l, 8, v51.l                              // 0000000034a0: d7380033 02026688
	v_and_b16 v50.l, 0xff, v50.l                               // 0000000034a8: d7620032 020264ff 000000ff
	s_delay_alu instid0(valu_dep_2)                            // 0000000034b4: bf870002
	v_or_b16 v73.h, v51.h, v51.l op_sel:[1,0,1]                // 0000000034b8: d7634849 02026733
	v_lshlrev_b16 v51.l, 8, v69.l                              // 0000000034c0: d7380033 02028a88
	v_and_b16 v51.h, 0xff, v69.h op_sel:[0,1,1]                // 0000000034c8: d7625033 02028aff 000000ff
	v_lshlrev_b16 v69.l, 8, v70.l                              // 0000000034d4: d7380045 02028c88
	v_and_b16 v69.h, 0xff, v70.h op_sel:[0,1,1]                // 0000000034dc: d7625045 02028cff 000000ff
	s_wait_loadcnt 0x8                                         // 0000000034e8: bfc00008
	v_mul_f32_e32 v70, v85, v86                                // 0000000034ec: 108cad55
	v_or_b16 v72.l, v50.l, v50.h op_sel:[0,1,0]                // 0000000034f0: d7631048 02026532
	v_or_b16 v73.l, v51.h, v51.l op_sel:[1,0,0]                // 0000000034f8: d7630849 02026733
	v_or_b16 v72.h, v69.h, v69.l op_sel:[1,0,1]                // 000000003500: d7634848 02028b45
	s_wait_loadcnt 0x7                                         // 000000003508: bfc00007
	v_dual_mul_f32 v51, v86, v87 :: v_dual_mul_f32 v50, v53, v70// 00000000350c: c8c6af56 33328d35
	s_wait_loadcnt 0x5                                         // 000000003514: bfc00005
	v_mul_f32_e32 v53, v86, v89                                // 000000003518: 106ab356
	s_wait_loadcnt 0x4                                         // 00000000351c: bfc00004
	v_mul_f32_e32 v69, v86, v90                                // 000000003520: 108ab556
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[78:79], v[72:73], v[61:68]// 000000003524: cc46403d 1cf6914e
	v_dual_add_f32 v3, v3, v50 :: v_dual_mul_f32 v50, v86, v88 // 00000000352c: c9066503 0332b156
	v_mul_f32_e32 v51, v54, v51                                // 000000003534: 10666736
	s_wait_loadcnt 0x3                                         // 000000003538: bfc00003
	v_mul_f32_e32 v54, v86, v83                                // 00000000353c: 106ca756
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003540: bf870193
	v_dual_mul_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v57, v69// 000000003544: c8c66537 32368b39
	v_add_f32_e32 v35, v35, v51                                // 00000000354c: 06466723
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003550: bf870193
	v_mul_f32_e32 v51, v58, v54                                // 000000003554: 10666d3a
	v_add_f32_e32 v32, v32, v50                                // 000000003558: 06406520
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_2)// 00000000355c: bf870144
	v_dual_mul_f32 v53, v56, v53 :: v_dual_add_f32 v30, v30, v55// 000000003560: c8c86b38 351e6f1e
	s_wait_loadcnt 0x0                                         // 000000003568: bfc00000
	v_mul_f32_e32 v54, v87, v92                                // 00000000356c: 106cb957
	v_dual_mul_f32 v50, v86, v84 :: v_dual_add_f32 v29, v29, v51// 000000003570: c8c8a956 321c671d
	v_dual_add_f32 v31, v31, v53 :: v_dual_mul_f32 v54, v62, v54// 000000003578: c9066b1f 1f366d3e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003580: bf870112
	v_dual_mul_f32 v51, v86, v91 :: v_dual_mul_f32 v50, v59, v50// 000000003584: c8c6b756 3332653b
	v_add_f32_e32 v24, v24, v54                                // 00000000358c: 06306d18
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_4)// 000000003590: bf870222
	v_mul_f32_e32 v51, v60, v51                                // 000000003594: 1066673c
	v_mul_f32_e32 v55, v88, v92                                // 000000003598: 106eb958
	v_dual_add_f32 v27, v27, v50 :: v_dual_mul_f32 v54, v84, v92// 00000000359c: c906651b 1b36b954
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000035a4: bf870193
	v_dual_add_f32 v26, v26, v51 :: v_dual_mul_f32 v53, v85, v92// 0000000035a8: c906671a 1a34b955
	v_mul_f32_e32 v50, v63, v55                                // 0000000035b0: 10646f3f
	v_mul_f32_e32 v55, v91, v92                                // 0000000035b4: 106eb95b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000035b8: bf870113
	v_dual_mul_f32 v54, v67, v54 :: v_dual_mul_f32 v53, v61, v53// 0000000035bc: c8c66d43 36346b3d
	v_mul_f32_e32 v55, v68, v55                                // 0000000035c4: 106e6f44
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_3)// 0000000035c8: bf870192
	v_add_f32_e32 v19, v19, v54                                // 0000000035cc: 06266d13
	v_add_f32_e32 v25, v25, v53                                // 0000000035d0: 06326b19
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000035d4: bf870093
	v_dual_mul_f32 v53, v83, v92 :: v_dual_add_f32 v18, v18, v55// 0000000035d8: c8c8b953 35126f12
	v_mul_f32_e32 v53, v66, v53                                // 0000000035e0: 106a6b42
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035e4: bf870091
	v_dual_mul_f32 v51, v89, v92 :: v_dual_add_f32 v20, v20, v53// 0000000035e8: c8c8b959 33146b14
	v_mul_f32_e32 v51, v64, v51                                // 0000000035f0: 10666740
	v_dual_add_f32 v23, v23, v50 :: v_dual_mul_f32 v50, v90, v92// 0000000035f4: c9066517 1732b95a
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000035fc: bf870112
	v_add_f32_e32 v22, v22, v51                                // 000000003600: 062c6716
	v_mul_f32_e32 v50, v65, v50                                // 000000003604: 10646541
	s_delay_alu instid0(valu_dep_1)                            // 000000003608: bf870001
	v_add_f32_e32 v21, v21, v50                                // 00000000360c: 062a6515
	s_wait_alu depctr_sa_sdst(0)                               // 000000003610: bf88ff9e
	s_cbranch_vccnz 64117                                      // 000000003614: bfa4fa75 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x4ec>
	v_mul_lo_u32 v2, s55, v4                                   // 000000003618: d72c0002 02020837
	v_mul_lo_u32 v12, s54, v5                                  // 000000003620: d72c000c 02020a36
	v_mad_co_u64_u32 v[8:9], null, s54, v4, 0                  // 000000003628: d6fe7c08 02020836
	v_sub_co_u32 v6, vcc_lo, s52, v4                           // 000000003630: d7016a06 02020834
	s_wait_alu depctr_va_vcc(0)                                // 000000003638: bf88ff9d
	v_sub_co_ci_u32_e64 v7, null, s53, v5, vcc_lo              // 00000000363c: d5217c07 01aa0a35
	v_cmp_gt_i64_e64 s7, s[54:55], v[10:11]                    // 000000003644: d4540007 02021436
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 00000000364c: bf870194
	v_add3_u32 v9, v9, v12, v2                                 // 000000003650: d6550009 040a1909
	v_cmp_lt_i64_e32 vcc_lo, 0, v[6:7]                         // 000000003658: 7ca20c80
	s_delay_alu instid0(valu_dep_2)                            // 00000000365c: bf870002
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 000000003660: 3e081081
	s_and_b32 s0, vcc_lo, s7                                   // 000000003664: 8b00076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003668: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000366c: be812000
	s_cbranch_execz 28                                         // 000000003670: bfa5001c <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1be4>
	v_lshlrev_b64_e32 v[8:9], 1, v[10:11]                      // 000000003674: 3e101481
	v_add_co_u32 v12, s0, s56, v4                              // 000000003678: d700000c 02020838
	v_bfe_u32 v2, v3, 16, 1                                    // 000000003680: d6100002 02052103
	s_wait_alu depctr_va_sdst(0)                               // 000000003688: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s57, v5, s0                 // 00000000368c: d5207c0d 00020a39
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003694: bf870193
	v_add_co_u32 v8, s0, v12, v8                               // 000000003698: d7000008 0202110c
	v_add3_u32 v2, v2, v3, 0x7fff                              // 0000000036a0: d6550002 03fe0702 00007fff
	v_or_b32_e32 v14, 0x400000, v3                             // 0000000036ac: 381c06ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v13, v9, s0                  // 0000000036b8: d5207c09 0002130d
	v_cmp_u_f32_e64 s0, v3, v3                                 // 0000000036c0: d4180000 02020703
	s_wait_alu depctr_va_sdst(0)                               // 0000000036c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036cc: bf870001
	v_cndmask_b32_e64 v2, v2, v14, s0                          // 0000000036d0: d5010002 00021d02
	global_store_d16_hi_b16 v[8:9], v2, off                    // 0000000036d8: ee09407c 01000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000036e8: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[6:7]                             // 0000000036ec: d4510000 02020c81
	s_and_b32 s1, s0, s7                                       // 0000000036f4: 8b010700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036f8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000036fc: be822001
	s_cbranch_execz 35                                         // 000000003700: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1c90>
	v_add_co_u32 v8, s1, s56, v4                               // 000000003704: d7000108 02020838
	s_wait_alu depctr_va_sdst(0)                               // 00000000370c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s1                  // 000000003710: d5207c09 00060a39
	s_lshl_b64 s[4:5], s[54:55], 1                             // 000000003718: 84848136
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 00000000371c: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003720: bf88ff9e
	v_add_co_u32 v8, s1, v8, s4                                // 000000003724: d7000108 02000908
	v_bfe_u32 v12, v35, 16, 1                                  // 00000000372c: d610000c 02052123
	s_wait_alu depctr_va_sdst(0)                               // 000000003734: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s5, v9, s1                   // 000000003738: d5207c09 00061205
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003740: bf870193
	v_add_co_u32 v2, s1, v8, v2                                // 000000003744: d7000102 02020508
	v_add3_u32 v12, v12, v35, 0x7fff                           // 00000000374c: d655000c 03fe470c 00007fff
	v_or_b32_e32 v13, 0x400000, v35                            // 000000003758: 381a46ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003760: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s1                   // 000000003764: d5207c03 00060709
	v_cmp_u_f32_e64 s1, v35, v35                               // 00000000376c: d4180001 02024723
	s_wait_alu depctr_va_sdst(0)                               // 000000003774: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003778: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s1                         // 00000000377c: d5010008 00061b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000003784: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003790: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003794: 8c7e027e
	v_cmp_lt_i64_e64 s1, 2, v[6:7]                             // 000000003798: d4510001 02020c82
	s_lshl_b64 s[8:9], s[54:55], 1                             // 0000000037a0: 84888136
	s_and_b32 s2, s1, s7                                       // 0000000037a4: 8b020701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037a8: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000037ac: be832002
	s_cbranch_execz 35                                         // 0000000037b0: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1d40>
	v_add_co_u32 v8, s2, s56, v4                               // 0000000037b4: d7000208 02020838
	s_wait_alu depctr_va_sdst(0)                               // 0000000037bc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s2                  // 0000000037c0: d5207c09 000a0a39
	s_lshl_b64 s[4:5], s[8:9], 1                               // 0000000037c8: 84848108
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 0000000037cc: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037d0: bf88ff9e
	v_add_co_u32 v8, s2, v8, s4                                // 0000000037d4: d7000208 02000908
	v_bfe_u32 v12, v32, 16, 1                                  // 0000000037dc: d610000c 02052120
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s5, v9, s2                   // 0000000037e8: d5207c09 000a1205
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037f0: bf870193
	v_add_co_u32 v2, s2, v8, v2                                // 0000000037f4: d7000202 02020508
	v_add3_u32 v12, v12, v32, 0x7fff                           // 0000000037fc: d655000c 03fe410c 00007fff
	v_or_b32_e32 v13, 0x400000, v32                            // 000000003808: 381a40ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003810: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s2                   // 000000003814: d5207c03 000a0709
	v_cmp_u_f32_e64 s2, v32, v32                               // 00000000381c: d4180002 02024120
	s_wait_alu depctr_va_sdst(0)                               // 000000003824: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003828: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s2                         // 00000000382c: d5010008 000a1b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000003834: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003840: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003844: 8c7e037e
	v_cmp_lt_i64_e64 s2, 3, v[6:7]                             // 000000003848: d4510002 02020c83
	s_and_b32 s3, s2, s7                                       // 000000003850: 8b030702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003854: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003858: be842003
	s_cbranch_execz 35                                         // 00000000385c: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1dec>
	v_add_co_u32 v8, s3, s56, v4                               // 000000003860: d7000308 02020838
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s3                  // 00000000386c: d5207c09 000e0a39
	s_lshl_b64 s[10:11], s[80:81], 1                           // 000000003874: 848a8150
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003878: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 00000000387c: bf88ff9e
	v_add_co_u32 v8, s3, v8, s10                               // 000000003880: d7000308 02001508
	v_bfe_u32 v12, v31, 16, 1                                  // 000000003888: d610000c 0205211f
	s_wait_alu depctr_va_sdst(0)                               // 000000003890: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s11, v9, s3                  // 000000003894: d5207c09 000e120b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000389c: bf870193
	v_add_co_u32 v2, s3, v8, v2                                // 0000000038a0: d7000302 02020508
	v_add3_u32 v12, v12, v31, 0x7fff                           // 0000000038a8: d655000c 03fe3f0c 00007fff
	v_or_b32_e32 v13, 0x400000, v31                            // 0000000038b4: 381a3eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000038bc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s3                   // 0000000038c0: d5207c03 000e0709
	v_cmp_u_f32_e64 s3, v31, v31                               // 0000000038c8: d4180003 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 0000000038d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038d4: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s3                         // 0000000038d8: d5010008 000e1b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 0000000038e0: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000038f0: 8c7e047e
	v_cmp_lt_i64_e64 s3, 4, v[6:7]                             // 0000000038f4: d4510003 02020c84
	s_lshl_b64 s[10:11], s[54:55], 2                           // 0000000038fc: 848a8236
	s_and_b32 s4, s3, s7                                       // 000000003900: 8b040703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003904: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000003908: be852004
	s_cbranch_execz 35                                         // 00000000390c: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1e9c>
	v_add_co_u32 v8, s4, s56, v4                               // 000000003910: d7000408 02020838
	s_wait_alu depctr_va_sdst(0)                               // 000000003918: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s4                  // 00000000391c: d5207c09 00120a39
	s_lshl_b64 s[12:13], s[10:11], 1                           // 000000003924: 848c810a
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003928: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 00000000392c: bf88ff9e
	v_add_co_u32 v8, s4, v8, s12                               // 000000003930: d7000408 02001908
	v_bfe_u32 v12, v30, 16, 1                                  // 000000003938: d610000c 0205211e
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s13, v9, s4                  // 000000003944: d5207c09 0012120d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000394c: bf870193
	v_add_co_u32 v2, s4, v8, v2                                // 000000003950: d7000402 02020508
	v_add3_u32 v12, v12, v30, 0x7fff                           // 000000003958: d655000c 03fe3d0c 00007fff
	v_or_b32_e32 v13, 0x400000, v30                            // 000000003964: 381a3cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000396c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s4                   // 000000003970: d5207c03 00120709
	v_cmp_u_f32_e64 s4, v30, v30                               // 000000003978: d4180004 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003980: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003984: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s4                         // 000000003988: d5010008 00121b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000003990: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000399c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000039a0: 8c7e057e
	v_cmp_lt_i64_e64 s4, 5, v[6:7]                             // 0000000039a4: d4510004 02020c85
	s_and_b32 s5, s4, s7                                       // 0000000039ac: 8b050704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039b0: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 0000000039b4: be862005
	s_cbranch_execz 35                                         // 0000000039b8: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1f48>
	v_add_co_u32 v8, s5, s56, v4                               // 0000000039bc: d7000508 02020838
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s5                  // 0000000039c8: d5207c09 00160a39
	s_lshl_b64 s[12:13], s[78:79], 1                           // 0000000039d0: 848c814e
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 0000000039d4: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039d8: bf88ff9e
	v_add_co_u32 v8, s5, v8, s12                               // 0000000039dc: d7000508 02001908
	v_bfe_u32 v12, v29, 16, 1                                  // 0000000039e4: d610000c 0205211d
	s_wait_alu depctr_va_sdst(0)                               // 0000000039ec: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s13, v9, s5                  // 0000000039f0: d5207c09 0016120d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000039f8: bf870193
	v_add_co_u32 v2, s5, v8, v2                                // 0000000039fc: d7000502 02020508
	v_add3_u32 v12, v12, v29, 0x7fff                           // 000000003a04: d655000c 03fe3b0c 00007fff
	v_or_b32_e32 v13, 0x400000, v29                            // 000000003a10: 381a3aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s5                   // 000000003a1c: d5207c03 00160709
	v_cmp_u_f32_e64 s5, v29, v29                               // 000000003a24: d4180005 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000003a2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a30: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s5                         // 000000003a34: d5010008 00161b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000003a3c: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003a4c: 8c7e067e
	v_cmp_lt_i64_e64 s5, 6, v[6:7]                             // 000000003a50: d4510005 02020c86
	s_and_b32 s6, s5, s7                                       // 000000003a58: 8b060705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a5c: bf88ff9e
	s_and_saveexec_b32 s12, s6                                 // 000000003a60: be8c2006
	s_cbranch_execz 35                                         // 000000003a64: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x1ff4>
	v_add_co_u32 v8, s6, s56, v4                               // 000000003a68: d7000608 02020838
	s_wait_alu depctr_va_sdst(0)                               // 000000003a70: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s57, v5, s6                  // 000000003a74: d5207c09 001a0a39
	s_lshl_b64 s[14:15], s[76:77], 1                           // 000000003a7c: 848e814c
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003a80: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a84: bf88ff9e
	v_add_co_u32 v8, s6, v8, s14                               // 000000003a88: d7000608 02001d08
	v_bfe_u32 v12, v27, 16, 1                                  // 000000003a90: d610000c 0205211b
	s_wait_alu depctr_va_sdst(0)                               // 000000003a98: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s15, v9, s6                  // 000000003a9c: d5207c09 001a120f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003aa4: bf870193
	v_add_co_u32 v2, s6, v8, v2                                // 000000003aa8: d7000602 02020508
	v_add3_u32 v12, v12, v27, 0x7fff                           // 000000003ab0: d655000c 03fe370c 00007fff
	v_or_b32_e32 v13, 0x400000, v27                            // 000000003abc: 381a36ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v9, v3, s6                   // 000000003ac8: d5207c03 001a0709
	v_cmp_u_f32_e64 s6, v27, v27                               // 000000003ad0: d4180006 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003adc: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s6                         // 000000003ae0: d5010008 001a1b0c
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000003ae8: ee09407c 04000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003af8: 8c7e0c7e
	v_cmp_lt_i64_e64 s6, 7, v[6:7]                             // 000000003afc: d4510006 02020c87
	s_and_b32 s7, s6, s7                                       // 000000003b04: 8b070706
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b08: bf88ff9e
	s_and_saveexec_b32 s12, s7                                 // 000000003b0c: be8c2007
	s_cbranch_execz 35                                         // 000000003b10: bfa50023 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x20a0>
	v_add_co_u32 v6, s7, s56, v4                               // 000000003b14: d7000706 02020838
	s_wait_alu depctr_va_sdst(0)                               // 000000003b1c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s57, v5, s7                  // 000000003b20: d5207c07 001e0a39
	s_lshl_b64 s[14:15], s[74:75], 1                           // 000000003b28: 848e814a
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003b2c: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b30: bf88ff9e
	v_add_co_u32 v6, s7, v6, s14                               // 000000003b34: d7000706 02001d06
	v_bfe_u32 v8, v26, 16, 1                                   // 000000003b3c: d6100008 0205211a
	s_wait_alu depctr_va_sdst(0)                               // 000000003b44: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s15, v7, s7                  // 000000003b48: d5207c07 001e0e0f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003b50: bf870193
	v_add_co_u32 v2, s7, v6, v2                                // 000000003b54: d7000702 02020506
	v_add3_u32 v8, v8, v26, 0x7fff                             // 000000003b5c: d6550008 03fe3508 00007fff
	v_or_b32_e32 v9, 0x400000, v26                             // 000000003b68: 381234ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003b70: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v7, v3, s7                   // 000000003b74: d5207c03 001e0707
	v_cmp_u_f32_e64 s7, v26, v26                               // 000000003b7c: d4180007 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000003b84: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b88: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s7                           // 000000003b8c: d5010006 001e1308
	global_store_d16_hi_b16 v[2:3], v6, off                    // 000000003b94: ee09407c 03000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ba0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003ba4: 8c7e0c7e
	v_cmp_gt_i64_e64 s7, s[54:55], v[0:1]                      // 000000003ba8: d4540007 02020036
	s_and_b32 s13, vcc_lo, s7                                  // 000000003bb0: 8b0d076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb4: bf88ff9e
	s_and_saveexec_b32 s12, s13                                // 000000003bb8: be8c200d
	s_cbranch_execz 25                                         // 000000003bbc: bfa50019 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2124>
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003bc0: 3e041481
	v_add_co_u32 v7, vcc_lo, s56, v4                           // 000000003bc4: d7006a07 02020838
	v_bfe_u32 v6, v25, 16, 1                                   // 000000003bcc: d6100006 02052119
	s_wait_alu depctr_va_vcc(0)                                // 000000003bd4: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s57, v5, vcc_lo              // 000000003bd8: d5207c08 01aa0a39
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003be0: bf870193
	v_add_co_u32 v2, vcc_lo, v7, v2                            // 000000003be4: d7006a02 02020507
	v_add3_u32 v6, v6, v25, 0x7fff                             // 000000003bec: d6550006 03fe3306 00007fff
	v_or_b32_e32 v9, 0x400000, v25                             // 000000003bf8: 381232ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003c00: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v8, v3, vcc_lo               // 000000003c04: d5207c03 01aa0708
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 000000003c0c: 7c303319
	s_wait_alu depctr_va_vcc(0)                                // 000000003c10: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v9, vcc_lo                       // 000000003c14: 020c1306
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003c18: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003c28: 8c7e0c7e
	s_and_b32 s12, s0, s7                                      // 000000003c2c: 8b0c0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c30: bf88ff9e
	s_and_saveexec_b32 s0, s12                                 // 000000003c34: be80200c
	s_cbranch_execz 31                                         // 000000003c38: bfa5001f <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x21b8>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003c3c: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003c44: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003c48: d5207c07 01aa0a39
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003c50: 3e041481
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003c54: bf8701c3
	v_add_co_u32 v6, vcc_lo, v6, s8                            // 000000003c58: d7006a06 02001106
	v_bfe_u32 v8, v24, 16, 1                                   // 000000003c60: d6100008 02052118
	s_wait_alu depctr_va_vcc(0)                                // 000000003c68: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s9, v7, vcc_lo               // 000000003c6c: d5207c07 01aa0e09
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003c74: d7006a02 02020506
	s_delay_alu instid0(valu_dep_3)                            // 000000003c7c: bf870003
	v_add3_u32 v8, v8, v24, 0x7fff                             // 000000003c80: d6550008 03fe3108 00007fff
	v_or_b32_e32 v9, 0x400000, v24                             // 000000003c8c: 381230ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003c94: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003c98: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000003ca0: 7c303118
	s_wait_alu depctr_va_vcc(0)                                // 000000003ca4: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003ca8: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003cac: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003cbc: 8c7e007e
	s_and_b32 s1, s1, s7                                       // 000000003cc0: 8b010701
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cc4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003cc8: be802001
	s_cbranch_execz 32                                         // 000000003ccc: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2250>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003cd0: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003cd8: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003cdc: d5207c07 01aa0a39
	s_lshl_b64 s[8:9], s[8:9], 1                               // 000000003ce4: 84888108
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003ce8: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cec: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v6, s8                            // 000000003cf0: d7006a06 02001106
	v_bfe_u32 v8, v23, 16, 1                                   // 000000003cf8: d6100008 02052117
	s_wait_alu depctr_va_vcc(0)                                // 000000003d00: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s9, v7, vcc_lo               // 000000003d04: d5207c07 01aa0e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d0c: bf870193
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003d10: d7006a02 02020506
	v_add3_u32 v8, v8, v23, 0x7fff                             // 000000003d18: d6550008 03fe2f08 00007fff
	v_or_b32_e32 v9, 0x400000, v23                             // 000000003d24: 38122eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d2c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003d30: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v23, v23                           // 000000003d38: 7c302f17
	s_wait_alu depctr_va_vcc(0)                                // 000000003d3c: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003d40: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003d44: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003d54: 8c7e007e
	s_and_b32 s1, s2, s7                                       // 000000003d58: 8b010702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d5c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003d60: be802001
	s_cbranch_execz 32                                         // 000000003d64: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x22e8>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003d68: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003d70: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003d74: d5207c07 01aa0a39
	s_lshl_b64 s[8:9], s[80:81], 1                             // 000000003d7c: 84888150
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003d80: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d84: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v6, s8                            // 000000003d88: d7006a06 02001106
	v_bfe_u32 v8, v22, 16, 1                                   // 000000003d90: d6100008 02052116
	s_wait_alu depctr_va_vcc(0)                                // 000000003d98: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s9, v7, vcc_lo               // 000000003d9c: d5207c07 01aa0e09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003da4: bf870193
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003da8: d7006a02 02020506
	v_add3_u32 v8, v8, v22, 0x7fff                             // 000000003db0: d6550008 03fe2d08 00007fff
	v_or_b32_e32 v9, 0x400000, v22                             // 000000003dbc: 38122cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003dc4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003dc8: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 000000003dd0: 7c302d16
	s_wait_alu depctr_va_vcc(0)                                // 000000003dd4: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003dd8: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003ddc: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003dec: 8c7e007e
	s_and_b32 s1, s3, s7                                       // 000000003df0: 8b010703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003df4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003df8: be802001
	s_cbranch_execz 32                                         // 000000003dfc: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2380>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003e00: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003e08: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003e0c: d5207c07 01aa0a39
	s_lshl_b64 s[2:3], s[10:11], 1                             // 000000003e14: 8482810a
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003e18: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e1c: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v6, s2                            // 000000003e20: d7006a06 02000506
	v_bfe_u32 v8, v21, 16, 1                                   // 000000003e28: d6100008 02052115
	s_wait_alu depctr_va_vcc(0)                                // 000000003e30: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo               // 000000003e34: d5207c07 01aa0e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003e3c: bf870193
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003e40: d7006a02 02020506
	v_add3_u32 v8, v8, v21, 0x7fff                             // 000000003e48: d6550008 03fe2b08 00007fff
	v_or_b32_e32 v9, 0x400000, v21                             // 000000003e54: 38122aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e5c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003e60: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v21, v21                           // 000000003e68: 7c302b15
	s_wait_alu depctr_va_vcc(0)                                // 000000003e6c: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003e70: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003e74: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e84: 8c7e007e
	s_and_b32 s1, s4, s7                                       // 000000003e88: 8b010704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e8c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003e90: be802001
	s_cbranch_execz 32                                         // 000000003e94: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2418>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003e98: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003ea0: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003ea4: d5207c07 01aa0a39
	s_lshl_b64 s[2:3], s[78:79], 1                             // 000000003eac: 8482814e
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003eb0: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eb4: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v6, s2                            // 000000003eb8: d7006a06 02000506
	v_bfe_u32 v8, v20, 16, 1                                   // 000000003ec0: d6100008 02052114
	s_wait_alu depctr_va_vcc(0)                                // 000000003ec8: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo               // 000000003ecc: d5207c07 01aa0e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003ed4: bf870193
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003ed8: d7006a02 02020506
	v_add3_u32 v8, v8, v20, 0x7fff                             // 000000003ee0: d6550008 03fe2908 00007fff
	v_or_b32_e32 v9, 0x400000, v20                             // 000000003eec: 381228ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ef4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003ef8: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000003f00: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000003f04: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003f08: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003f0c: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003f1c: 8c7e007e
	s_and_b32 s1, s5, s7                                       // 000000003f20: 8b010705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f24: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003f28: be802001
	s_cbranch_execz 32                                         // 000000003f2c: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x24b0>
	v_add_co_u32 v6, vcc_lo, s56, v4                           // 000000003f30: d7006a06 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003f38: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s57, v5, vcc_lo              // 000000003f3c: d5207c07 01aa0a39
	s_lshl_b64 s[2:3], s[76:77], 1                             // 000000003f44: 8482814c
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003f48: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f4c: bf88ff9e
	v_add_co_u32 v6, vcc_lo, v6, s2                            // 000000003f50: d7006a06 02000506
	v_bfe_u32 v8, v19, 16, 1                                   // 000000003f58: d6100008 02052113
	s_wait_alu depctr_va_vcc(0)                                // 000000003f60: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo               // 000000003f64: d5207c07 01aa0e03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003f6c: bf870193
	v_add_co_u32 v2, vcc_lo, v6, v2                            // 000000003f70: d7006a02 02020506
	v_add3_u32 v8, v8, v19, 0x7fff                             // 000000003f78: d6550008 03fe2708 00007fff
	v_or_b32_e32 v9, 0x400000, v19                             // 000000003f84: 381226ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003f8c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v3, vcc_lo               // 000000003f90: d5207c03 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 000000003f98: 7c302713
	s_wait_alu depctr_va_vcc(0)                                // 000000003f9c: bf88ff9d
	v_cndmask_b32_e32 v6, v8, v9, vcc_lo                       // 000000003fa0: 020c1308
	global_store_d16_hi_b16 v[2:3], v6, off offset:32          // 000000003fa4: ee09407c 03000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fb0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003fb4: 8c7e007e
	s_and_b32 s1, s6, s7                                       // 000000003fb8: 8b010706
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fbc: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003fc0: be802001
	s_cbranch_execz 32                                         // 000000003fc4: bfa50020 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2548>
	v_add_co_u32 v4, vcc_lo, s56, v4                           // 000000003fc8: d7006a04 02020838
	s_wait_alu depctr_va_vcc(0)                                // 000000003fd0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s57, v5, vcc_lo              // 000000003fd4: d5207c05 01aa0a39
	s_lshl_b64 s[2:3], s[74:75], 1                             // 000000003fdc: 8482814a
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000003fe0: 3e041481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fe4: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000003fe8: d7006a04 02000504
	v_bfe_u32 v6, v18, 16, 1                                   // 000000003ff0: d6100006 02052112
	s_wait_alu depctr_va_vcc(0)                                // 000000003ff8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000003ffc: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004004: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000004008: d7006a02 02020504
	v_add3_u32 v6, v6, v18, 0x7fff                             // 000000004010: d6550006 03fe2506 00007fff
	v_or_b32_e32 v7, 0x400000, v18                             // 00000000401c: 380e24ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004024: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000004028: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v18, v18                           // 000000004030: 7c302512
	s_wait_alu depctr_va_vcc(0)                                // 000000004034: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000004038: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 00000000403c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004048: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000404c: 8c7e007e
	s_mov_b32 s0, 0                                            // 000000004050: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004054: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000004058: 8b6a007e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000405c: bf88ff9e
	s_cbranch_vccz 24                                          // 000000004060: bfa30018 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x25c4>
	s_and_b32 s0, s92, exec_lo                                 // 000000004064: 8b007e5c
	s_cselect_b32 s0, 1, 0                                     // 000000004068: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000406c: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000004070: bf078100
	s_cbranch_scc1 22                                          // 000000004074: bfa20016 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x25d0>
	v_lshl_or_b32 v24, v28, 3, s72                             // 000000004078: d6560018 0121071c
	v_mov_b32_e32 v25, s73                                     // 000000004080: 7e320249
	v_mov_b32_e32 v27, s73                                     // 000000004084: 7e360249
	v_mov_b32_e32 v23, s73                                     // 000000004088: 7e2e0249
	v_mov_b32_e32 v21, s73                                     // 00000000408c: 7e2a0249
	v_or_b32_e32 v26, 1, v24                                   // 000000004090: 38343081
	v_or_b32_e32 v22, 2, v24                                   // 000000004094: 382c3082
	v_or_b32_e32 v20, 3, v24                                   // 000000004098: 38283083
	v_or_b32_e32 v18, 4, v24                                   // 00000000409c: 38243084
	v_mov_b32_e32 v19, s73                                     // 0000000040a0: 7e260249
	v_or_b32_e32 v16, 5, v24                                   // 0000000040a4: 38203085
	v_mov_b32_e32 v17, s73                                     // 0000000040a8: 7e220249
	v_or_b32_e32 v14, 6, v24                                   // 0000000040ac: 381c3086
	v_mov_b32_e32 v15, s73                                     // 0000000040b0: 7e1e0249
	v_or_b32_e32 v12, 7, v24                                   // 0000000040b4: 38183087
	v_mov_b32_e32 v13, s73                                     // 0000000040b8: 7e1a0249
	s_mov_b32 s0, 0                                            // 0000000040bc: be800080
	s_branch 4                                                 // 0000000040c0: bfa00004 <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x25d4>
	s_nop 0                                                    // 0000000040c4: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000040c8: bfb60003
	s_endpgm                                                   // 0000000040cc: bfb00000
	s_mov_b32 s0, -1                                           // 0000000040d0: be8000c1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v61, 0             // 0000000040d4: ca100080 3a3c0080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040dc: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 0000000040e0: 8b007e00
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v67, 0             // 0000000040e4: ca100080 3c420080
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v75, 0             // 0000000040ec: ca100080 404a0080
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v29, 0             // 0000000040f4: ca100080 461c0080
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v9, 0               // 0000000040fc: ca100080 06080080
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0             // 000000004104: ca100080 35360080
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0             // 00000000410c: ca100080 37380080
	v_mov_b32_e32 v57, 0                                       // 000000004114: 7e720280
	v_mov_b32_e32 v59, 0                                       // 000000004118: 7e760280
	s_cselect_b32 s0, 1, 0                                     // 00000000411c: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004120: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000004124: bf078100
	s_cbranch_scc1 828                                         // 000000004128: bfa2033c <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x331c>
	v_dual_mov_b32 v29, 0 :: v_dual_lshlrev_b32 v28, 3, v28    // 00000000412c: ca220080 1d1c3883
	v_cmp_gt_i64_e32 vcc_lo, s[54:55], v[10:11]                // 000000004134: 7ca81436
	v_mov_b32_e32 v25, s73                                     // 000000004138: 7e320249
	v_mov_b32_e32 v21, s73                                     // 00000000413c: 7e2a0249
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_3)// 000000004140: bf8701c4
	v_or_b32_e32 v24, s72, v28                                 // 000000004144: 38303848
	v_dual_mov_b32 v70, v29 :: v_dual_mov_b32 v17, s73         // 000000004148: ca10011d 46100049
	s_wait_alu depctr_va_vcc(0)                                // 000000004150: bf88ff9d
	v_dual_cndmask_b32 v2, 0, v10 :: v_dual_mov_b32 v19, s73   // 000000004154: ca501480 02120049
	v_or_b32_e32 v20, 3, v24                                   // 00000000415c: 38283083
	v_cndmask_b32_e32 v3, 0, v11, vcc_lo                       // 000000004160: 02061680
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[24:25]                // 000000004164: 7ca83034
	v_or_b32_e32 v16, 5, v24                                   // 000000004168: 38203085
	v_or_b32_e32 v26, 1, v24                                   // 00000000416c: 38343081
	v_or_b32_e32 v18, 4, v24                                   // 000000004170: 38243084
	v_or_b32_e32 v14, 6, v24                                   // 000000004174: 381c3086
	v_mov_b32_e32 v15, s73                                     // 000000004178: 7e1e0249
	s_wait_alu depctr_va_vcc(0)                                // 00000000417c: bf88ff9d
	v_cndmask_b32_e32 v9, 0, v24, vcc_lo                       // 000000004180: 02123080
	v_cndmask_b32_e64 v50, 0, s73, vcc_lo                      // 000000004184: d5010032 01a89280
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[20:21]                // 00000000418c: 7ca82834
	v_or_b32_e32 v12, 7, v24                                   // 000000004190: 38183087
	v_mov_b32_e32 v13, s73                                     // 000000004194: 7e1a0249
	v_or_b32_e32 v22, 2, v24                                   // 000000004198: 382c3082
	s_lshr_b64 s[2:3], s[64:65], 5                             // 00000000419c: 85828540
	s_lshr_b32 s3, s65, 5                                      // 0000000041a0: 85038541
	s_wait_alu depctr_va_vcc(0)                                // 0000000041a4: bf88ff9d
	v_cndmask_b32_e32 v42, 0, v20, vcc_lo                      // 0000000041a8: 02542880
	v_cndmask_b32_e64 v43, 0, s73, vcc_lo                      // 0000000041ac: d501002b 01a89280
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[16:17]                // 0000000041b4: 7ca82034
	v_mov_b32_e32 v27, s73                                     // 0000000041b8: 7e360249
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 0000000041bc: 3e040482
	v_or_b32_e32 v41, 18, v28                                  // 0000000041c0: 38523892
	v_or_b32_e32 v51, 19, v28                                  // 0000000041c4: 38663893
	v_or_b32_e32 v53, 22, v28                                  // 0000000041c8: 386a3896
	s_wait_alu depctr_va_vcc(0)                                // 0000000041cc: bf88ff9d
	v_cndmask_b32_e32 v38, 0, v16, vcc_lo                      // 0000000041d0: 024c2080
	v_cmp_gt_i64_e64 s0, s[52:53], v[26:27]                    // 0000000041d4: d4540000 02023434
	v_cndmask_b32_e64 v39, 0, s73, vcc_lo                      // 0000000041dc: d5010027 01a89280
	v_cmp_gt_i64_e32 vcc_lo, s[52:53], v[12:13]                // 0000000041e4: 7ca81834
	v_mov_b32_e32 v23, s73                                     // 0000000041e8: 7e2e0249
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041ec: bf88ff9e
	v_mul_lo_u32 v54, v50, s2                                  // 0000000041f0: d72c0036 02000532
	v_mul_lo_u32 v55, v9, s3                                   // 0000000041f8: d72c0037 02000709
	s_wait_alu depctr_va_sdst(0)                               // 000000004200: bf88f19f
	v_cndmask_b32_e64 v46, 0, v26, s0                          // 000000004204: d501002e 00023480
	v_cndmask_b32_e64 v47, 0, s73, s0                          // 00000000420c: d501002f 00009280
	v_cmp_gt_i64_e64 s0, s[52:53], v[18:19]                    // 000000004214: d4540000 02022434
	s_wait_alu depctr_va_vcc(0)                                // 00000000421c: bf88ff9d
	v_cndmask_b32_e32 v32, 0, v12, vcc_lo                      // 000000004220: 02401880
	v_cndmask_b32_e64 v35, 0, s73, vcc_lo                      // 000000004224: d5010023 01a89280
	v_cmp_gt_i64_e64 s1, s[52:53], v[22:23]                    // 00000000422c: d4540001 02022c34
	v_or_b32_e32 v59, 4, v28                                   // 000000004234: 38763884
	v_or_b32_e32 v64, 5, v28                                   // 000000004238: 38803885
	s_wait_alu depctr_va_sdst(0)                               // 00000000423c: bf88f19f
	v_cndmask_b32_e64 v8, 0, v18, s0                           // 000000004240: d5010008 00022480
	v_cndmask_b32_e64 v40, 0, s73, s0                          // 000000004248: d5010028 00009280
	v_cmp_gt_i64_e64 s0, s[52:53], v[14:15]                    // 000000004250: d4540000 02021c34
	v_cndmask_b32_e64 v44, 0, v22, s1                          // 000000004258: d501002c 00062c80
	v_cndmask_b32_e64 v45, 0, s73, s1                          // 000000004260: d501002d 00049280
	v_or_b32_e32 v61, 7, v28                                   // 000000004268: 387a3887
	v_or_b32_e32 v67, 6, v28                                   // 00000000426c: 38863886
	v_mov_b32_e32 v75, v29                                     // 000000004270: 7e96031d
	s_wait_alu depctr_va_sdst(0)                               // 000000004274: bf88f19f
	v_cndmask_b32_e64 v6, 0, v14, s0                           // 000000004278: d5010006 00021c80
	v_cndmask_b32_e64 v7, 0, s73, s0                           // 000000004280: d5010007 00009280
	v_add_co_u32 v30, s0, s72, v52                             // 000000004288: d700001e 02026848
	s_wait_alu depctr_va_sdst(0)                               // 000000004290: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s73, 0, s0                  // 000000004294: d5207c1f 00010049
	v_cmp_gt_i64_e64 s0, s[54:55], v[0:1]                      // 00000000429c: d4540000 02020036
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000042a4: bf870223
	v_mad_co_u64_u32 v[4:5], null, s64, v30, v[28:29]          // 0000000042a8: d6fe7c04 04723c40
	v_mul_lo_u32 v34, s65, v30                                 // 0000000042b0: d72c0022 02023c41
	v_mul_lo_u32 v33, s64, v31                                 // 0000000042b8: d72c0021 02023e40
	v_add_co_u32 v30, vcc_lo, s62, v2                          // 0000000042c0: d7006a1e 0202043e
	s_wait_alu depctr_va_sdst(0)                               // 0000000042c8: bf88f19f
	v_cndmask_b32_e64 v1, 0, v1, s0                            // 0000000042cc: d5010001 00020280
	v_cndmask_b32_e64 v0, 0, v0, s0                            // 0000000042d4: d5010000 00020080
	s_wait_alu depctr_va_vcc(0)                                // 0000000042dc: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, s63, v3, vcc_lo             // 0000000042e0: d5207c1f 01aa063f
	v_mad_co_u64_u32 v[2:3], null, v32, s2, 0                  // 0000000042e8: d6fe7c02 02000520
	v_add3_u32 v5, v34, v5, v33                                // 0000000042f0: d6550005 04860b22
	v_mul_lo_u32 v34, v35, s2                                  // 0000000042f8: d72c0022 02000523
	v_mul_lo_u32 v35, v32, s3                                  // 000000004300: d72c0023 02000720
	v_add_co_u32 v32, vcc_lo, s68, v4                          // 000000004308: d7006a20 02020844
	s_add_nc_u64 s[0:1], s[66:67], s[70:71]                    // 000000004310: a9804642
	s_wait_alu depctr_va_vcc(0)                                // 000000004314: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s69, v5, vcc_lo             // 000000004318: d5207c21 01aa0a45
	v_lshlrev_b64_e32 v[4:5], 2, v[0:1]                        // 000000004320: 3e080082
	s_wait_alu depctr_sa_sdst(0)                               // 000000004324: bf88ff9e
	v_mad_co_u64_u32 v[0:1], null, s54, v41, s[0:1]            // 000000004328: d6fe7c00 00025236
	v_mul_lo_u32 v48, v7, s2                                   // 000000004330: d72c0030 02000507
	v_mul_lo_u32 v49, v6, s3                                   // 000000004338: d72c0031 02000706
	v_mad_co_u64_u32 v[6:7], null, v6, s2, 0                   // 000000004340: d6fe7c06 02000506
	v_add3_u32 v3, v3, v35, v34                                // 000000004348: d6550003 048a4703
	v_add_co_u32 v34, vcc_lo, s62, v4                          // 000000004350: d7006a22 0202083e
	s_wait_alu depctr_va_vcc(0)                                // 000000004358: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s63, v5, vcc_lo             // 00000000435c: d5207c23 01aa0a3f
	s_delay_alu instid0(valu_dep_3)                            // 000000004364: bf870003
	v_lshlrev_b64_e32 v[36:37], 2, v[2:3]                      // 000000004368: 3e480482
	v_mad_co_u64_u32 v[3:4], null, s55, v41, v[1:2]            // 00000000436c: d6fe7c03 04065237
	v_mad_co_u64_u32 v[1:2], null, s54, v51, s[0:1]            // 000000004374: d6fe7c01 00026636
	v_add3_u32 v7, v7, v49, v48                                // 00000000437c: d6550007 04c26307
	v_mul_lo_u32 v41, v39, s2                                  // 000000004384: d72c0029 02000527
	v_mul_lo_u32 v48, v38, s3                                  // 00000000438c: d72c0030 02000726
	v_mad_co_u64_u32 v[4:5], null, v38, s2, 0                  // 000000004394: d6fe7c04 02000526
	v_or_b32_e32 v49, 20, v28                                  // 00000000439c: 38623894
	v_add_co_u32 v62, vcc_lo, v0, 16                           // 0000000043a0: d7006a3e 02012100
	v_lshlrev_b64_e32 v[38:39], 2, v[6:7]                      // 0000000043a8: 3e4c0c82
	v_mad_co_u64_u32 v[6:7], null, s55, v51, v[2:3]            // 0000000043ac: d6fe7c06 040a6637
	s_wait_alu depctr_va_vcc(0)                                // 0000000043b4: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, 0, v3, vcc_lo               // 0000000043b8: d5207c3f 01aa0680
	v_add3_u32 v5, v5, v48, v41                                // 0000000043c0: d6550005 04a66105
	v_mad_co_u64_u32 v[2:3], null, s54, v49, s[0:1]            // 0000000043c8: d6fe7c02 00026236
	v_mul_lo_u32 v0, v40, s2                                   // 0000000043d0: d72c0000 02000528
	v_mul_lo_u32 v48, v8, s3                                   // 0000000043d8: d72c0030 02000708
	v_mad_co_u64_u32 v[7:8], null, v8, s2, 0                   // 0000000043e0: d6fe7c07 02000508
	v_or_b32_e32 v51, 21, v28                                  // 0000000043e8: 38663895
	v_add_co_u32 v65, vcc_lo, v1, 16                           // 0000000043ec: d7006a41 02012101
	v_lshlrev_b64_e32 v[40:41], 2, v[4:5]                      // 0000000043f4: 3e500882
	v_mad_co_u64_u32 v[3:4], null, s55, v49, v[3:4]            // 0000000043f8: d6fe7c03 040e6237
	v_mul_lo_u32 v49, v42, s3                                  // 000000004400: d72c0031 0200072a
	v_mad_co_u64_u32 v[4:5], null, v42, s2, 0                  // 000000004408: d6fe7c04 0200052a
	v_add3_u32 v8, v8, v48, v0                                 // 000000004410: d6550008 04026108
	v_mad_co_u64_u32 v[0:1], null, s54, v51, s[0:1]            // 000000004418: d6fe7c00 00026636
	v_mul_lo_u32 v48, v43, s2                                  // 000000004420: d72c0030 0200052b
	s_wait_alu depctr_va_vcc(0)                                // 000000004428: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, 0, v6, vcc_lo               // 00000000442c: d5207c42 01aa0c80
	v_add_co_u32 v68, vcc_lo, v2, 16                           // 000000004434: d7006a44 02012102
	v_lshlrev_b64_e32 v[42:43], 2, v[7:8]                      // 00000000443c: 3e540e82
	s_wait_alu depctr_va_vcc(0)                                // 000000004440: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v3, vcc_lo               // 000000004444: d5207c45 01aa0680
	v_mad_co_u64_u32 v[6:7], null, s55, v51, v[1:2]            // 00000000444c: d6fe7c06 04066637
	v_add3_u32 v5, v5, v49, v48                                // 000000004454: d6550005 04c26305
	v_mad_co_u64_u32 v[1:2], null, s54, v53, s[0:1]            // 00000000445c: d6fe7c01 00026a36
	v_mul_lo_u32 v3, v45, s2                                   // 000000004464: d72c0003 0200052d
	v_mul_lo_u32 v48, v44, s3                                  // 00000000446c: d72c0030 0200072c
	v_mad_co_u64_u32 v[7:8], null, v44, s2, 0                  // 000000004474: d6fe7c07 0200052c
	v_or_b32_e32 v51, 23, v28                                  // 00000000447c: 38663897
	v_lshlrev_b64_e32 v[44:45], 2, v[4:5]                      // 000000004480: 3e580882
	v_add_co_u32 v71, vcc_lo, v0, 16                           // 000000004484: d7006a47 02012100
	v_mul_lo_u32 v0, v47, s2                                   // 00000000448c: d72c0000 0200052f
	v_mad_co_u64_u32 v[4:5], null, s55, v53, v[2:3]            // 000000004494: d6fe7c04 040a6a37
	v_mul_lo_u32 v5, v46, s3                                   // 00000000449c: d72c0005 0200072e
	v_add3_u32 v8, v8, v48, v3                                 // 0000000044a4: d6550008 040e6108
	v_mad_co_u64_u32 v[2:3], null, s54, v51, s[0:1]            // 0000000044ac: d6fe7c02 00026636
	v_mad_co_u64_u32 v[48:49], null, v46, s2, 0                // 0000000044b4: d6fe7c30 0200052e
	s_wait_alu depctr_va_vcc(0)                                // 0000000044bc: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, 0, v6, vcc_lo               // 0000000044c0: d5207c48 01aa0c80
	v_or_b32_e32 v6, 17, v28                                   // 0000000044c8: 380c3891
	v_or_b32_e32 v53, 16, v28                                  // 0000000044cc: 386a3890
	v_add_co_u32 v73, vcc_lo, v1, 16                           // 0000000044d0: d7006a49 02012101
	s_wait_alu depctr_va_vcc(0)                                // 0000000044d8: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, 0, v4, vcc_lo               // 0000000044dc: d5207c4a 01aa0880
	v_mad_co_u64_u32 v[3:4], null, s55, v51, v[3:4]            // 0000000044e4: d6fe7c03 040e6637
	v_add3_u32 v49, v49, v5, v0                                // 0000000044ec: d6550031 04020b31
	v_mad_co_u64_u32 v[0:1], null, s54, v6, s[0:1]             // 0000000044f4: d6fe7c00 00020c36
	v_mad_co_u64_u32 v[4:5], null, s54, v53, s[0:1]            // 0000000044fc: d6fe7c04 00026a36
	v_lshlrev_b64_e32 v[46:47], 2, v[7:8]                      // 000000004504: 3e5c0e82
	v_mad_co_u64_u32 v[7:8], null, v9, s2, 0                   // 000000004508: d6fe7c07 02000509
	v_or_b32_e32 v9, 2, v28                                    // 000000004510: 38123882
	v_add_co_u32 v76, vcc_lo, v2, 16                           // 000000004514: d7006a4c 02012102
	s_wait_alu depctr_va_vcc(0)                                // 00000000451c: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, 0, v3, vcc_lo               // 000000004520: d5207c4d 01aa0680
	v_mad_co_u64_u32 v[1:2], null, s55, v6, v[1:2]             // 000000004528: d6fe7c01 04060c37
	v_mad_co_u64_u32 v[50:51], null, s55, v53, v[5:6]          // 000000004530: d6fe7c32 04166a37
	v_mad_co_u64_u32 v[5:6], null, s54, v9, s[0:1]             // 000000004538: d6fe7c05 00021236
	v_add3_u32 v8, v8, v55, v54                                // 000000004540: d6550008 04da6f08
	v_or_b32_e32 v53, 1, v28                                   // 000000004548: 386a3881
	v_add_co_u32 v78, vcc_lo, v4, 16                           // 00000000454c: d7006a4e 02012104
	v_mad_co_u64_u32 v[2:3], null, s54, v28, s[0:1]            // 000000004554: d6fe7c02 00023836
	v_mad_co_u64_u32 v[55:56], null, s54, v59, s[0:1]          // 00000000455c: d6fe7c37 00027636
	s_wait_alu depctr_va_vcc(0)                                // 000000004564: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, 0, v50, vcc_lo              // 000000004568: d5207c4f 01aa6480
	v_lshlrev_b64_e32 v[50:51], 2, v[7:8]                      // 000000004570: 3e640e82
	v_mad_co_u64_u32 v[8:9], null, s55, v9, v[6:7]             // 000000004574: d6fe7c08 041a1237
	v_mad_co_u64_u32 v[6:7], null, s54, v53, s[0:1]            // 00000000457c: d6fe7c06 00026a36
	v_or_b32_e32 v9, 3, v28                                    // 000000004584: 38123883
	v_mad_co_u64_u32 v[57:58], null, s54, v64, s[0:1]          // 000000004588: d6fe7c39 00028036
	v_mad_co_u64_u32 v[3:4], null, s55, v28, v[3:4]            // 000000004590: d6fe7c03 040e3837
	v_add_co_u32 v80, vcc_lo, v5, 16                           // 000000004598: d7006a50 02012105
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000045a0: 3e606082
	s_wait_alu depctr_va_vcc(0)                                // 0000000045a4: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v8, vcc_lo               // 0000000045a8: d5207c51 01aa1080
	v_mad_co_u64_u32 v[4:5], null, s55, v53, v[7:8]            // 0000000045b0: d6fe7c04 041e6a37
	v_mad_co_u64_u32 v[53:54], null, s54, v9, s[0:1]           // 0000000045b8: d6fe7c35 00021236
	v_mad_co_u64_u32 v[59:60], null, s55, v59, v[56:57]        // 0000000045c0: d6fe7c3b 04e27637
	v_add_co_u32 v82, vcc_lo, v6, 16                           // 0000000045c8: d7006a52 02012106
	v_mov_b32_e32 v60, v29                                     // 0000000045d0: 7e78031d
	v_mov_b32_e32 v56, v29                                     // 0000000045d4: 7e70031d
	s_lshl_b64 s[18:19], s[54:55], 2                           // 0000000045d8: 84928236
	s_wait_alu depctr_va_vcc(0)                                // 0000000045dc: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, 0, v4, vcc_lo               // 0000000045e0: d5207c53 01aa0880
	v_mad_co_u64_u32 v[6:7], null, s55, v9, v[54:55]           // 0000000045e8: d6fe7c06 04da1237
	v_mad_co_u64_u32 v[4:5], null, s54, v61, s[0:1]            // 0000000045f0: d6fe7c04 00027a36
	v_mad_co_u64_u32 v[7:8], null, s54, v67, s[0:1]            // 0000000045f8: d6fe7c07 00028636
	v_add_co_u32 v28, vcc_lo, v53, 16                          // 000000004600: d7006a1c 02012135
	v_mad_co_u64_u32 v[53:54], null, s55, v64, v[58:59]        // 000000004608: d6fe7c35 04ea8037
	v_mov_b32_e32 v64, v29                                     // 000000004610: 7e80031d
	v_mov_b32_e32 v58, v29                                     // 000000004614: 7e74031d
	s_wait_alu depctr_va_vcc(0)                                // 000000004618: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, 0, v6, vcc_lo               // 00000000461c: d5207c54 01aa0c80
	v_add_co_u32 v85, vcc_lo, v55, 16                          // 000000004624: d7006a55 02012137
	v_mad_co_u64_u32 v[5:6], null, s55, v61, v[5:6]            // 00000000462c: d6fe7c05 04167a37
	v_mad_co_u64_u32 v[8:9], null, s55, v67, v[8:9]            // 000000004634: d6fe7c08 04228637
	s_wait_alu depctr_va_vcc(0)                                // 00000000463c: bf88ff9d
	v_add_co_ci_u32_e64 v86, null, 0, v59, vcc_lo              // 000000004640: d5207c56 01aa7680
	v_add_co_u32 v87, vcc_lo, v57, 16                          // 000000004648: d7006a57 02012139
	s_wait_alu depctr_va_vcc(0)                                // 000000004650: bf88ff9d
	v_add_co_ci_u32_e64 v88, null, 0, v53, vcc_lo              // 000000004654: d5207c58 01aa6a80
	v_mov_b32_e32 v67, v29                                     // 00000000465c: 7e86031d
	v_mov_b32_e32 v61, v29                                     // 000000004660: 7e7a031d
	v_mov_b32_e32 v59, v29                                     // 000000004664: 7e76031d
	v_mov_b32_e32 v57, v29                                     // 000000004668: 7e72031d
	v_dual_mov_b32 v55, v29 :: v_dual_mov_b32 v54, v29         // 00000000466c: ca10011d 3736011d
	v_mov_b32_e32 v53, v29                                     // 000000004674: 7e6a031d
	v_dual_mov_b32 v9, v29 :: v_dual_mov_b32 v6, v29           // 000000004678: ca10011d 0906011d
	s_lshl_b64 s[20:21], s[54:55], 5                           // 000000004680: 84948536
	s_mov_b64 s[22:23], 0                                      // 000000004684: be960180
	v_add_co_u32 v97, s3, v87, v52                             // 000000004688: d7000361 02026957
	v_add_co_u32 v95, s2, v85, v52                             // 000000004690: d700025f 02026955
	s_wait_alu depctr_va_sdst(0)                               // 000000004698: bf88f19f
	v_add_co_ci_u32_e64 v98, null, 0, v88, s3                  // 00000000469c: d5207c62 000eb080
	v_add_co_u32 v91, s0, v82, v52                             // 0000000046a4: d700005b 02026952
	v_add_co_u32 v93, s1, v28, v52                             // 0000000046ac: d700015d 0202691c
	v_add_co_u32 v99, s4, v7, v52                              // 0000000046b4: d7000463 02026907
	v_add_co_u32 v101, s5, v4, v52                             // 0000000046bc: d7000565 02026904
	v_add_co_u32 v103, s6, v80, v52                            // 0000000046c4: d7000667 02026950
	v_add_co_u32 v105, s7, v71, v52                            // 0000000046cc: d7000769 02026947
	v_add_co_u32 v107, s8, v68, v52                            // 0000000046d4: d700086b 02026944
	v_add_co_u32 v109, s9, v76, v52                            // 0000000046dc: d700096d 0202694c
	v_add_co_u32 v111, s10, v73, v52                           // 0000000046e4: d7000a6f 02026949
	v_add_co_u32 v113, s11, v0, v52                            // 0000000046ec: d7000b71 02026900
	v_add_co_u32 v115, s12, v78, v52                           // 0000000046f4: d7000c73 0202694e
	v_add_co_ci_u32_e64 v96, null, 0, v86, s2                  // 0000000046fc: d5207c60 000aac80
	v_add_co_u32 v89, vcc_lo, v2, v52                          // 000000004704: d7006a59 02026902
	v_add_co_u32 v117, s13, v65, v52                           // 00000000470c: d7000d75 02026941
	v_add_co_u32 v119, s14, v62, v52                           // 000000004714: d7000e77 0202693e
	s_wait_alu depctr_va_sdst(0)                               // 00000000471c: bf88f19f
	v_add_co_ci_u32_e64 v92, null, 0, v83, s0                  // 000000004720: d5207c5c 0002a680
	v_add_co_ci_u32_e64 v94, null, 0, v84, s1                  // 000000004728: d5207c5e 0006a880
	v_add_co_ci_u32_e64 v100, null, 0, v8, s4                  // 000000004730: d5207c64 00121080
	v_add_co_ci_u32_e64 v102, null, 0, v5, s5                  // 000000004738: d5207c66 00160a80
	v_add_co_ci_u32_e64 v104, null, 0, v81, s6                 // 000000004740: d5207c68 001aa280
	v_add_co_ci_u32_e64 v106, null, 0, v72, s7                 // 000000004748: d5207c6a 001e9080
	v_add_co_ci_u32_e64 v108, null, 0, v69, s8                 // 000000004750: d5207c6c 00228a80
	v_add_co_ci_u32_e64 v110, null, 0, v77, s9                 // 000000004758: d5207c6e 00269a80
	v_add_co_ci_u32_e64 v112, null, 0, v74, s10                // 000000004760: d5207c70 002a9480
	v_add_co_ci_u32_e64 v114, null, 0, v1, s11                 // 000000004768: d5207c72 002e0280
	v_add_co_ci_u32_e64 v116, null, 0, v79, s12                // 000000004770: d5207c74 00329e80
	s_wait_alu depctr_va_vcc(0)                                // 000000004778: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, 0, v3, vcc_lo               // 00000000477c: d5207c5a 01aa0680
	v_add_co_ci_u32_e64 v118, null, 0, v66, s13                // 000000004784: d5207c76 00368480
	v_add_co_ci_u32_e64 v120, null, 0, v63, s14                // 00000000478c: d5207c78 003a7e80
	s_clause 0x1                                               // 000000004794: bf850001
	global_load_u8 v121, v[97:98], off offset:-16              // 000000004798: ee04007c 00000079 fffff061
	global_load_u8 v97, v[97:98], off                          // 0000000047a4: ee04007c 00000061 00000061
	s_clause 0x1                                               // 0000000047b0: bf850001
	global_load_u8 v98, v[95:96], off offset:-16               // 0000000047b4: ee04007c 00000062 fffff05f
	global_load_u8 v95, v[95:96], off                          // 0000000047c0: ee04007c 0000005f 0000005f
	s_clause 0x1                                               // 0000000047cc: bf850001
	global_load_u8 v96, v[101:102], off                        // 0000000047d0: ee04007c 00000060 00000065
	global_load_u8 v101, v[101:102], off offset:16             // 0000000047dc: ee04007c 00000065 00001065
	s_clause 0x1                                               // 0000000047e8: bf850001
	global_load_u8 v102, v[99:100], off                        // 0000000047ec: ee04007c 00000066 00000063
	global_load_u8 v99, v[99:100], off offset:16               // 0000000047f8: ee04007c 00000063 00001063
	s_clause 0x1                                               // 000000004804: bf850001
	global_load_u8 v100, v[91:92], off offset:-16              // 000000004808: ee04007c 00000064 fffff05b
	global_load_u8 v91, v[91:92], off                          // 000000004814: ee04007c 0000005b 0000005b
	s_clause 0x1                                               // 000000004820: bf850001
	global_load_u8 v92, v[89:90], off                          // 000000004824: ee04007c 0000005c 00000059
	global_load_u8 v122, v[89:90], off offset:16               // 000000004830: ee04007c 0000007a 00001059
	s_clause 0x1                                               // 00000000483c: bf850001
	global_load_u8 v123, v[93:94], off offset:-16              // 000000004840: ee04007c 0000007b fffff05d
	global_load_u8 v93, v[93:94], off                          // 00000000484c: ee04007c 0000005d 0000005d
	s_clause 0x1                                               // 000000004858: bf850001
	global_load_u8 v94, v[103:104], off offset:-16             // 00000000485c: ee04007c 0000005e fffff067
	global_load_u8 v103, v[103:104], off                       // 000000004868: ee04007c 00000067 00000067
	s_clause 0x1                                               // 000000004874: bf850001
	global_load_u8 v104, v[105:106], off offset:-16            // 000000004878: ee04007c 00000068 fffff069
	global_load_u8 v124, v[105:106], off                       // 000000004884: ee04007c 0000007c 00000069
	s_clause 0x1                                               // 000000004890: bf850001
	global_load_u8 v105, v[107:108], off offset:-16            // 000000004894: ee04007c 00000069 fffff06b
	global_load_u8 v125, v[107:108], off                       // 0000000048a0: ee04007c 0000007d 0000006b
	s_clause 0x1                                               // 0000000048ac: bf850001
	global_load_u8 v106, v[109:110], off offset:-16            // 0000000048b0: ee04007c 0000006a fffff06d
	global_load_u8 v109, v[109:110], off                       // 0000000048bc: ee04007c 0000006d 0000006d
	s_clause 0x1                                               // 0000000048c8: bf850001
	global_load_u8 v107, v[111:112], off offset:-16            // 0000000048cc: ee04007c 0000006b fffff06f
	global_load_u8 v110, v[111:112], off                       // 0000000048d8: ee04007c 0000006e 0000006f
	s_clause 0x1                                               // 0000000048e4: bf850001
	global_load_u8 v108, v[113:114], off                       // 0000000048e8: ee04007c 0000006c 00000071
	global_load_u8 v111, v[113:114], off offset:16             // 0000000048f4: ee04007c 0000006f 00001071
	s_clause 0x1                                               // 000000004900: bf850001
	global_load_u8 v112, v[115:116], off offset:-16            // 000000004904: ee04007c 00000070 fffff073
	global_load_u8 v113, v[115:116], off                       // 000000004910: ee04007c 00000071 00000073
	s_clause 0x1                                               // 00000000491c: bf850001
	global_load_u8 v114, v[117:118], off offset:-16            // 000000004920: ee04007c 00000072 fffff075
	global_load_u8 v115, v[117:118], off                       // 00000000492c: ee04007c 00000073 00000075
	s_clause 0x1                                               // 000000004938: bf850001
	global_load_u8 v116, v[119:120], off offset:-16            // 00000000493c: ee04007c 00000074 fffff077
	global_load_u8 v117, v[119:120], off                       // 000000004948: ee04007c 00000075 00000077
	v_add_co_u32 v89, vcc_lo, s58, v50                         // 000000004954: d7006a59 0202643a
	s_wait_alu depctr_va_vcc(0)                                // 00000000495c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v51, vcc_lo            // 000000004960: d5207c5a 01aa663b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004968: bf88ff9e
	s_add_nc_u64 s[22:23], s[22:23], 32                        // 00000000496c: a996a016
	v_add_co_u32 v62, s2, v62, s20                             // 000000004970: d700023e 0200293e
	global_load_b32 v118, v[89:90], off                        // 000000004978: ee05007c 00000076 00000059
	v_add_co_u32 v89, vcc_lo, s58, v48                         // 000000004984: d7006a59 0202603a
	s_wait_alu depctr_va_vcc(0)                                // 00000000498c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v49, vcc_lo            // 000000004990: d5207c5a 01aa623b
	v_add_co_u32 v65, s3, v65, s20                             // 000000004998: d7000341 02002941
	v_add_co_u32 v68, s4, v68, s20                             // 0000000049a0: d7000444 02002944
	global_load_b32 v119, v[89:90], off                        // 0000000049a8: ee05007c 00000077 00000059
	v_add_co_u32 v89, vcc_lo, s58, v46                         // 0000000049b4: d7006a59 02025c3a
	s_wait_alu depctr_va_vcc(0)                                // 0000000049bc: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v47, vcc_lo            // 0000000049c0: d5207c5a 01aa5e3b
	v_add_co_u32 v71, s5, v71, s20                             // 0000000049c8: d7000547 02002947
	v_add_co_u32 v73, s6, v73, s20                             // 0000000049d0: d7000649 02002949
	global_load_b32 v120, v[89:90], off                        // 0000000049d8: ee05007c 00000078 00000059
	v_add_co_u32 v89, vcc_lo, s58, v44                         // 0000000049e4: d7006a59 0202583a
	s_wait_alu depctr_va_vcc(0)                                // 0000000049ec: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v45, vcc_lo            // 0000000049f0: d5207c5a 01aa5a3b
	v_add_co_u32 v76, s7, v76, s20                             // 0000000049f8: d700074c 0200294c
	v_add_co_u32 v0, s8, v0, s20                               // 000000004a00: d7000800 02002900
	global_load_b32 v126, v[89:90], off                        // 000000004a08: ee05007c 0000007e 00000059
	v_add_co_u32 v89, vcc_lo, s58, v42                         // 000000004a14: d7006a59 0202543a
	s_wait_alu depctr_va_vcc(0)                                // 000000004a1c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v43, vcc_lo            // 000000004a20: d5207c5a 01aa563b
	v_add_co_u32 v78, s9, v78, s20                             // 000000004a28: d700094e 0200294e
	v_add_co_u32 v2, s10, v2, s20                              // 000000004a30: d7000a02 02002902
	global_load_b32 v127, v[89:90], off                        // 000000004a38: ee05007c 0000007f 00000059
	v_add_co_u32 v89, vcc_lo, s58, v40                         // 000000004a44: d7006a59 0202503a
	s_wait_alu depctr_va_vcc(0)                                // 000000004a4c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v41, vcc_lo            // 000000004a50: d5207c5a 01aa523b
	v_add_co_u32 v80, s11, v80, s20                            // 000000004a58: d7000b50 02002950
	v_add_co_u32 v82, s12, v82, s20                            // 000000004a60: d7000c52 02002952
	global_load_b32 v128, v[89:90], off                        // 000000004a68: ee05007c 00000080 00000059
	v_add_co_u32 v89, vcc_lo, s58, v38                         // 000000004a74: d7006a59 02024c3a
	s_wait_alu depctr_va_vcc(0)                                // 000000004a7c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v39, vcc_lo            // 000000004a80: d5207c5a 01aa4e3b
	v_add_co_u32 v28, s13, v28, s20                            // 000000004a88: d7000d1c 0200291c
	v_add_co_u32 v4, s14, v4, s20                              // 000000004a90: d7000e04 02002904
	global_load_b32 v129, v[89:90], off                        // 000000004a98: ee05007c 00000081 00000059
	v_add_co_u32 v89, vcc_lo, s58, v36                         // 000000004aa4: d7006a59 0202483a
	s_wait_alu depctr_va_vcc(0)                                // 000000004aac: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s59, v37, vcc_lo            // 000000004ab0: d5207c5a 01aa4a3b
	v_add_co_u32 v85, s15, v85, s20                            // 000000004ab8: d7000f55 02002955
	v_add_co_u32 v7, s16, v7, s20                              // 000000004ac0: d7001007 02002907
	global_load_b32 v130, v[89:90], off                        // 000000004ac8: ee05007c 00000082 00000059
	v_add_co_u32 v87, s17, v87, s20                            // 000000004ad4: d7001157 02002957
	s_wait_alu depctr_va_sdst(0)                               // 000000004adc: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s21, v63, s2                // 000000004ae0: d5207c3f 000a7e15
	v_add_co_ci_u32_e64 v66, null, s21, v66, s3                // 000000004ae8: d5207c42 000e8415
	v_add_co_ci_u32_e64 v69, null, s21, v69, s4                // 000000004af0: d5207c45 00128a15
	v_add_co_ci_u32_e64 v72, null, s21, v72, s5                // 000000004af8: d5207c48 00169015
	v_add_co_ci_u32_e64 v74, null, s21, v74, s6                // 000000004b00: d5207c4a 001a9415
	v_add_co_ci_u32_e64 v77, null, s21, v77, s7                // 000000004b08: d5207c4d 001e9a15
	v_add_co_ci_u32_e64 v1, null, s21, v1, s8                  // 000000004b10: d5207c01 00220215
	v_add_co_ci_u32_e64 v79, null, s21, v79, s9                // 000000004b18: d5207c4f 00269e15
	v_add_co_ci_u32_e64 v3, null, s21, v3, s10                 // 000000004b20: d5207c03 002a0615
	v_add_co_ci_u32_e64 v81, null, s21, v81, s11               // 000000004b28: d5207c51 002ea215
	v_add_co_ci_u32_e64 v83, null, s21, v83, s12               // 000000004b30: d5207c53 0032a615
	v_add_co_ci_u32_e64 v84, null, s21, v84, s13               // 000000004b38: d5207c54 0036a815
	v_add_co_ci_u32_e64 v5, null, s21, v5, s14                 // 000000004b40: d5207c05 003a0a15
	v_add_co_ci_u32_e64 v86, null, s21, v86, s15               // 000000004b48: d5207c56 003eac15
	v_add_co_ci_u32_e64 v8, null, s21, v8, s16                 // 000000004b50: d5207c08 00421015
	v_add_co_ci_u32_e64 v88, null, s21, v88, s17               // 000000004b58: d5207c58 0046b015
	s_add_nc_u64 s[58:59], s[58:59], 4                         // 000000004b60: a9ba843a
	s_wait_loadcnt 0x25                                        // 000000004b64: bfc00025
	v_perm_b32 v89, v98, v121, 0xc0c0004                       // 000000004b68: d6440059 03fef362 0c0c0004
	s_wait_loadcnt 0x24                                        // 000000004b74: bfc00024
	v_perm_b32 v95, v95, v97, 0xc0c0004                        // 000000004b78: d644005f 03fec35f 0c0c0004
	s_wait_loadcnt 0x21                                        // 000000004b84: bfc00021
	v_perm_b32 v90, v102, v96, 0xc0c0004                       // 000000004b88: d644005a 03fec166 0c0c0004
	s_wait_loadcnt 0x20                                        // 000000004b94: bfc00020
	v_perm_b32 v101, v99, v101, 0xc0c0004                      // 000000004b98: d6440065 03fecb63 0c0c0004
	s_wait_loadcnt 0x1d                                        // 000000004ba4: bfc0001d
	v_perm_b32 v92, v92, v100, 0xc0c0004                       // 000000004ba8: d644005c 03fec95c 0c0c0004
	s_wait_loadcnt 0x1c                                        // 000000004bb4: bfc0001c
	v_perm_b32 v91, v122, v91, 0xc0c0004                       // 000000004bb8: d644005b 03feb77a 0c0c0004
	v_lshl_or_b32 v98, v90, 16, v89                            // 000000004bc4: d6560062 0565215a
	s_wait_loadcnt 0x19                                        // 000000004bcc: bfc00019
	v_perm_b32 v94, v94, v123, 0xc0c0004                       // 000000004bd0: d644005e 03fef75e 0c0c0004
	s_wait_loadcnt 0x18                                        // 000000004bdc: bfc00018
	v_perm_b32 v93, v103, v93, 0xc0c0004                       // 000000004be0: d644005d 03febb67 0c0c0004
	s_wait_loadcnt 0x15                                        // 000000004bec: bfc00015
	v_perm_b32 v96, v105, v104, 0xc0c0004                      // 000000004bf0: d6440060 03fed169 0c0c0004
	v_lshl_or_b32 v97, v94, 16, v92                            // 000000004bfc: d6560061 0571215e
	s_wait_loadcnt 0x14                                        // 000000004c04: bfc00014
	v_perm_b32 v103, v125, v124, 0xc0c0004                     // 000000004c08: d6440067 03fef97d 0c0c0004
	s_wait_loadcnt 0x11                                        // 000000004c14: bfc00011
	v_perm_b32 v100, v107, v106, 0xc0c0004                     // 000000004c18: d6440064 03fed56b 0c0c0004
	global_load_b64 v[105:106], v[32:33], off                  // 000000004c24: ee05407c 00000069 00000020
	s_wait_loadcnt 0xe                                         // 000000004c30: bfc0000e
	v_perm_b32 v102, v112, v108, 0xc0c0004                     // 000000004c34: d6440066 03fed970 0c0c0004
	global_load_b64 v[107:108], v[32:33], off offset:16        // 000000004c40: ee05407c 0000006b 00001020
	v_perm_b32 v112, v110, v109, 0xc0c0004                     // 000000004c4c: d6440070 03fedb6e 0c0c0004
	s_wait_loadcnt 0xe                                         // 000000004c58: bfc0000e
	v_perm_b32 v111, v113, v111, 0xc0c0004                     // 000000004c5c: d644006f 03fedf71 0c0c0004
	s_wait_loadcnt 0xb                                         // 000000004c68: bfc0000b
	v_perm_b32 v104, v116, v114, 0xc0c0004                     // 000000004c6c: d6440068 03fee574 0c0c0004
	global_load_b32 v114, v[30:31], off                        // 000000004c78: ee05007c 00000072 0000001e
	global_load_b32 v116, v[34:35], off                        // 000000004c84: ee05007c 00000074 00000022
	s_wait_loadcnt 0xc                                         // 000000004c90: bfc0000c
	v_perm_b32 v115, v117, v115, 0xc0c0004                     // 000000004c94: d6440073 03fee775 0c0c0004
	v_lshl_or_b32 v100, v100, 16, v96                          // 000000004ca0: d6560064 05812164
	v_lshl_or_b32 v99, v104, 16, v102                          // 000000004ca8: d6560063 05992168
	v_lshl_or_b32 v110, v101, 16, v95                          // 000000004cb0: d656006e 057d2165
	v_lshl_or_b32 v109, v93, 16, v91                           // 000000004cb8: d656006d 056d215d
	v_lshl_or_b32 v112, v112, 16, v103                         // 000000004cc0: d6560070 059d2170
	v_lshl_or_b32 v111, v115, 16, v111                         // 000000004cc8: d656006f 05bd2173
	v_add_co_u32 v32, s0, v32, 32                              // 000000004cd0: d7000020 02014120
	s_wait_alu depctr_va_sdst(0)                               // 000000004cd8: bf88f19f
	v_add_co_ci_u32_e64 v33, null, 0, v33, s0                  // 000000004cdc: d5207c21 00024280
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ce4: bf88ff9e
	v_cmp_lt_i64_e64 s0, s[22:23], s[60:61]                    // 000000004ce8: d4510000 02007816
	v_add_co_u32 v30, vcc_lo, v30, s18                         // 000000004cf0: d7006a1e 0200251e
	v_add_co_u32 v34, s1, v34, s18                             // 000000004cf8: d7000122 02002522
	s_wait_alu depctr_va_vcc(0)                                // 000000004d00: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, s19, v31, vcc_lo            // 000000004d04: d5207c1f 01aa3e13
	s_wait_alu depctr_va_sdst(0)                               // 000000004d0c: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s19, v35, s1                // 000000004d10: d5207c23 00064613
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000004d18: 8b6a007e
	s_wait_loadcnt 0x3                                         // 000000004d1c: bfc00003
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[105:106], v[97:98], 0// 000000004d20: cc464059 1a02c369
	s_wait_loadcnt 0x2                                         // 000000004d28: bfc00002
	s_delay_alu instid0(valu_dep_1)                            // 000000004d2c: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[107:108], v[99:100], v[89:96]// 000000004d30: cc464059 1d66c76b
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[105:106], v[109:110], 0// 000000004d38: cc464061 1a02db69
	s_wait_loadcnt 0x0                                         // 000000004d40: bfc00000
	v_dual_mul_f32 v121, v118, v114 :: v_dual_mul_f32 v110, v118, v116// 000000004d44: c8c6e576 796ee976
	v_mul_f32_e32 v113, v114, v119                             // 000000004d4c: 10e2ef72
	v_dual_mul_f32 v117, v114, v120 :: v_dual_mul_f32 v106, v114, v127// 000000004d50: c8c6f172 756aff72
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[107:108], v[111:112], v[97:104]// 000000004d58: cc464061 1d86df6b
	v_dual_mul_f32 v105, v114, v126 :: v_dual_mul_f32 v112, v120, v116// 000000004d60: c8c6fd72 6970e978
	v_dual_mul_f32 v107, v114, v128 :: v_dual_mul_f32 v108, v114, v129// 000000004d68: c8c70172 6b6d0372
	v_dual_mul_f32 v109, v114, v130 :: v_dual_mul_f32 v118, v128, v116// 000000004d70: c8c70572 6d76e980
	v_dual_mul_f32 v111, v119, v116 :: v_dual_mul_f32 v114, v126, v116// 000000004d78: c8c6e977 6f72e97e
	v_mul_f32_e32 v115, v127, v116                             // 000000004d80: 10e6e97f
	v_dual_mul_f32 v119, v129, v116 :: v_dual_mul_f32 v90, v90, v113// 000000004d84: c8c6e981 775ae35a
	v_dual_mul_f32 v116, v130, v116 :: v_dual_mul_f32 v91, v91, v117// 000000004d8c: c8c6e982 745aeb5b
	s_delay_alu instid0(valu_dep_4)                            // 000000004d94: bf870004
	v_mul_f32_e32 v98, v98, v111                               // 000000004d98: 10c4df62
	v_mul_f32_e32 v92, v92, v105                               // 000000004d9c: 10b8d35c
	v_dual_mul_f32 v89, v89, v121 :: v_dual_mul_f32 v94, v94, v107// 000000004da0: c8c6f359 595ed75e
	v_dual_mul_f32 v93, v93, v106 :: v_dual_mul_f32 v96, v96, v109// 000000004da8: c8c6d55d 5d60db60
	v_dual_mul_f32 v95, v95, v108 :: v_dual_mul_f32 v100, v100, v114// 000000004db0: c8c6d95f 5f64e564
	v_dual_mul_f32 v97, v97, v110 :: v_dual_mul_f32 v104, v104, v116// 000000004db8: c8c6dd61 6168e968
	v_dual_mul_f32 v99, v99, v112 :: v_dual_mul_f32 v102, v102, v118// 000000004dc0: c8c6e163 6366ed66
	s_delay_alu instid0(valu_dep_4)                            // 000000004dc8: bf870004
	v_dual_mul_f32 v101, v101, v115 :: v_dual_add_f32 v64, v64, v93// 000000004dcc: c8c8e765 6540bb40
	v_dual_mul_f32 v103, v103, v119 :: v_dual_add_f32 v58, v58, v96// 000000004dd4: c8c8ef67 673ac13a
	v_dual_add_f32 v29, v29, v89 :: v_dual_add_f32 v70, v70, v91// 000000004ddc: c908b31d 1d46b746
	v_dual_add_f32 v75, v75, v90 :: v_dual_add_f32 v60, v60, v95// 000000004de4: c908b54b 4b3cbf3c
	v_dual_add_f32 v67, v67, v92 :: v_dual_add_f32 v56, v56, v99// 000000004dec: c908b943 4338c738
	v_dual_add_f32 v61, v61, v94 :: v_dual_add_f32 v54, v54, v101// 000000004df4: c908bd3d 3d36cb36
	v_dual_add_f32 v59, v59, v97 :: v_dual_add_f32 v6, v6, v104// 000000004dfc: c908c33b 3b06d106
	v_add_f32_e32 v57, v57, v98                                // 000000004e04: 0672c539
	v_add_f32_e32 v55, v55, v100                               // 000000004e08: 066ec937
	v_add_f32_e32 v53, v53, v102                               // 000000004e0c: 066acd35
	v_add_f32_e32 v9, v9, v103                                 // 000000004e10: 0612cf09
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e14: bf88ff9e
	s_cbranch_vccnz 65051                                      // 000000004e18: bfa4fe1b <tessera_rocm_scaled_matmul_82ffe065c77e5ad0+0x2b88>
	v_mul_lo_u32 v2, s55, v24                                  // 000000004e1c: d72c0002 02023037
	v_mul_lo_u32 v3, s54, v25                                  // 000000004e24: d72c0003 02023236
	v_mad_co_u64_u32 v[0:1], null, s54, v24, 0                 // 000000004e2c: d6fe7c00 02023036
	v_mul_lo_u32 v7, s55, v26                                  // 000000004e34: d72c0007 02023437
	v_mul_lo_u32 v8, s54, v27                                  // 000000004e3c: d72c0008 02023636
	v_bfe_u32 v24, v29, 16, 1                                  // 000000004e44: d6100018 0205211d
	v_lshlrev_b64_e32 v[4:5], 1, v[10:11]                      // 000000004e4c: 3e081481
	v_or_b32_e32 v25, 0x400000, v29                            // 000000004e50: 38323aff 00400000
	v_mul_lo_u32 v11, s55, v22                                 // 000000004e58: d72c000b 02022c37
	v_mul_lo_u32 v23, s54, v23                                 // 000000004e60: d72c0017 02022e36
	v_add3_u32 v1, v1, v3, v2                                  // 000000004e68: d6550001 040a0701
	v_mad_co_u64_u32 v[2:3], null, s54, v26, 0                 // 000000004e70: d6fe7c02 02023436
	v_add3_u32 v24, v24, v29, 0x7fff                           // 000000004e78: d6550018 03fe3b18 00007fff
	v_bfe_u32 v10, v75, 16, 1                                  // 000000004e84: d610000a 0205214b
	v_mul_lo_u32 v21, s54, v21                                 // 000000004e8c: d72c0015 02022a36
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004e94: 3e000081
	v_mul_lo_u32 v26, s54, v17                                 // 000000004e98: d72c001a 02022236
	s_delay_alu instid0(valu_dep_4)                            // 000000004ea0: bf870004
	v_add3_u32 v10, v10, v75, 0x7fff                           // 000000004ea4: d655000a 03fe970a 00007fff
	v_add3_u32 v3, v3, v8, v7                                  // 000000004eb0: d6550003 041e1103
	v_mad_co_u64_u32 v[7:8], null, s54, v22, 0                 // 000000004eb8: d6fe7c07 02022c36
	v_add_co_u32 v0, vcc_lo, s56, v0                           // 000000004ec0: d7006a00 02020038
	s_wait_alu depctr_va_vcc(0)                                // 000000004ec8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s57, v1, vcc_lo              // 000000004ecc: d5207c01 01aa0239
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000004ed4: 7c303b1d
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004ed8: 3e040481
	v_or_b32_e32 v22, 0x400000, v75                            // 000000004edc: 382c96ff 00400000
	v_add3_u32 v8, v8, v23, v11                                // 000000004ee4: d6550008 042e2f08
	v_bfe_u32 v23, v70, 16, 1                                  // 000000004eec: d6100017 02052146
	s_wait_alu depctr_va_vcc(0)                                // 000000004ef4: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo                    // 000000004ef8: 02303318
	v_add_co_u32 v0, vcc_lo, v0, v4                            // 000000004efc: d7006a00 02020900
	s_wait_alu depctr_va_vcc(0)                                // 000000004f04: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v5, vcc_lo               // 000000004f08: d5207c01 01aa0b01
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 000000004f10: 7c30974b
	v_mul_lo_u32 v25, s55, v20                                 // 000000004f14: d72c0019 02022837
	global_store_d16_hi_b16 v[0:1], v24, off                   // 000000004f1c: ee09407c 0c000000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f28: bf88ff9d
	v_cndmask_b32_e32 v22, v10, v22, vcc_lo                    // 000000004f2c: 022c2d0a
	v_add_co_u32 v10, vcc_lo, s56, v2                          // 000000004f30: d7006a0a 02020438
	s_wait_alu depctr_va_vcc(0)                                // 000000004f38: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s57, v3, vcc_lo             // 000000004f3c: d5207c0b 01aa0639
	v_lshlrev_b64_e32 v[2:3], 1, v[7:8]                        // 000000004f44: 3e040e81
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f48: bf8701a3
	v_add_co_u32 v7, vcc_lo, v10, v4                           // 000000004f4c: d7006a07 0202090a
	s_wait_alu depctr_va_vcc(0)                                // 000000004f54: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v11, v5, vcc_lo              // 000000004f58: d5207c08 01aa0b0b
	v_add3_u32 v10, v23, v70, 0x7fff                           // 000000004f60: d655000a 03fe8d17 00007fff
	s_delay_alu instid0(valu_dep_4)                            // 000000004f6c: bf870004
	v_add_co_u32 v23, vcc_lo, s56, v2                          // 000000004f70: d7006a17 02020438
	s_wait_alu depctr_va_vcc(0)                                // 000000004f78: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s57, v3, vcc_lo             // 000000004f7c: d5207c18 01aa0639
	v_mad_co_u64_u32 v[2:3], null, s54, v20, 0                 // 000000004f84: d6fe7c02 02022836
	v_or_b32_e32 v11, 0x400000, v70                            // 000000004f8c: 38168cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000004f94: 7c308d46
	global_store_d16_hi_b16 v[7:8], v22, off                   // 000000004f98: ee09407c 0b000000 00000007
	s_wait_alu depctr_va_vcc(0)                                // 000000004fa4: bf88ff9d
	v_cndmask_b32_e32 v20, v10, v11, vcc_lo                    // 000000004fa8: 0228170a
	v_add_co_u32 v10, vcc_lo, v23, v4                          // 000000004fac: d7006a0a 02020917
	s_wait_alu depctr_va_vcc(0)                                // 000000004fb4: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v24, v5, vcc_lo             // 000000004fb8: d5207c0b 01aa0b18
	v_add3_u32 v3, v3, v21, v25                                // 000000004fc0: d6550003 04662b03
	v_mul_lo_u32 v21, s55, v18                                 // 000000004fc8: d72c0015 02022437
	v_mul_lo_u32 v24, s54, v19                                 // 000000004fd0: d72c0018 02022636
	v_mad_co_u64_u32 v[18:19], null, s54, v18, 0               // 000000004fd8: d6fe7c12 02022436
	v_bfe_u32 v23, v67, 16, 1                                  // 000000004fe0: d6100017 02052143
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004fe8: 3e040481
	v_or_b32_e32 v25, 0x400000, v67                            // 000000004fec: 383286ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 000000004ff4: 7c308743
	global_store_d16_hi_b16 v[10:11], v20, off                 // 000000004ff8: ee09407c 0a000000 0000000a
	v_add3_u32 v23, v23, v67, 0x7fff                           // 000000005004: d6550017 03fe8717 00007fff
	v_add3_u32 v19, v19, v24, v21                              // 000000005010: d6550013 04563113
	s_wait_alu depctr_va_vcc(0)                                // 000000005018: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 00000000501c: bf870002
	v_cndmask_b32_e32 v20, v23, v25, vcc_lo                    // 000000005020: 02283317
	v_add_co_u32 v21, vcc_lo, s56, v2                          // 000000005024: d7006a15 02020438
	s_wait_alu depctr_va_vcc(0)                                // 00000000502c: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s57, v3, vcc_lo             // 000000005030: d5207c16 01aa0639
	v_lshlrev_b64_e32 v[2:3], 1, v[18:19]                      // 000000005038: 3e042481
	v_bfe_u32 v23, v64, 16, 1                                  // 00000000503c: d6100017 02052140
	v_add_co_u32 v18, vcc_lo, v21, v4                          // 000000005044: d7006a12 02020915
	s_wait_alu depctr_va_vcc(0)                                // 00000000504c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v22, v5, vcc_lo             // 000000005050: d5207c13 01aa0b16
	s_delay_alu instid0(valu_dep_3)                            // 000000005058: bf870003
	v_add3_u32 v21, v23, v64, 0x7fff                           // 00000000505c: d6550015 03fe8117 00007fff
	v_add_co_u32 v23, vcc_lo, s56, v2                          // 000000005068: d7006a17 02020438
	s_wait_alu depctr_va_vcc(0)                                // 000000005070: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s57, v3, vcc_lo             // 000000005074: d5207c18 01aa0639
	v_mul_lo_u32 v25, s55, v16                                 // 00000000507c: d72c0019 02022037
	v_mad_co_u64_u32 v[2:3], null, s54, v16, 0                 // 000000005084: d6fe7c02 02022036
	v_or_b32_e32 v22, 0x400000, v64                            // 00000000508c: 382c80ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 000000005094: 7c308140
	s_wait_alu depctr_va_vcc(0)                                // 000000005098: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_4)// 00000000509c: bf870212
	v_cndmask_b32_e32 v21, v21, v22, vcc_lo                    // 0000000050a0: 022a2d15
	v_add3_u32 v3, v3, v26, v25                                // 0000000050a4: d6550003 04663503
	v_add_co_u32 v16, vcc_lo, v23, v4                          // 0000000050ac: d7006a10 02020917
	v_bfe_u32 v22, v61, 16, 1                                  // 0000000050b4: d6100016 0205213d
	s_wait_alu depctr_va_vcc(0)                                // 0000000050bc: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v24, v5, vcc_lo             // 0000000050c0: d5207c11 01aa0b18
	v_mul_lo_u32 v23, s55, v14                                 // 0000000050c8: d72c0017 02021c37
	v_mul_lo_u32 v24, s54, v15                                 // 0000000050d0: d72c0018 02021e36
	v_mad_co_u64_u32 v[14:15], null, s54, v14, 0               // 0000000050d8: d6fe7c0e 02021c36
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000050e0: 3e040481
	v_add3_u32 v22, v22, v61, 0x7fff                           // 0000000050e4: d6550016 03fe7b16 00007fff
	v_or_b32_e32 v25, 0x400000, v61                            // 0000000050f0: 38327aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 0000000050f8: 7c307b3d
	s_clause 0x1                                               // 0000000050fc: bf850001
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000005100: ee09407c 0a000000 00000012
	global_store_d16_hi_b16 v[16:17], v21, off                 // 00000000510c: ee09407c 0a800000 00000010
	v_bfe_u32 v21, v60, 16, 1                                  // 000000005118: d6100015 0205213c
	v_mul_lo_u32 v26, s54, v13                                 // 000000005120: d72c001a 02021a36
	v_add3_u32 v15, v15, v24, v23                              // 000000005128: d655000f 045e310f
	s_wait_alu depctr_va_vcc(0)                                // 000000005130: bf88ff9d
	v_cndmask_b32_e32 v20, v22, v25, vcc_lo                    // 000000005134: 02283316
	v_add_co_u32 v22, vcc_lo, s56, v2                          // 000000005138: d7006a16 02020438
	s_wait_alu depctr_va_vcc(0)                                // 000000005140: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s57, v3, vcc_lo             // 000000005144: d5207c17 01aa0639
	v_lshlrev_b64_e32 v[2:3], 1, v[14:15]                      // 00000000514c: 3e041c81
	v_mul_lo_u32 v25, s55, v12                                 // 000000005150: d72c0019 02021837
	v_mad_co_u64_u32 v[12:13], null, s54, v12, 0               // 000000005158: d6fe7c0c 02021836
	v_add_co_u32 v14, vcc_lo, v22, v4                          // 000000005160: d7006a0e 02020916
	v_add3_u32 v21, v21, v60, 0x7fff                           // 000000005168: d6550015 03fe7915 00007fff
	v_or_b32_e32 v24, 0x400000, v60                            // 000000005174: 383078ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000517c: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v23, v5, vcc_lo             // 000000005180: d5207c0f 01aa0b17
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000005188: 7c30793c
	v_add3_u32 v13, v13, v26, v25                              // 00000000518c: d655000d 0466350d
	v_bfe_u32 v22, v58, 16, 1                                  // 000000005194: d6100016 0205213a
	v_or_b32_e32 v23, 0x400000, v58                            // 00000000519c: 382e74ff 00400000
	global_store_d16_hi_b16 v[14:15], v20, off                 // 0000000051a4: ee09407c 0a000000 0000000e
	s_wait_alu depctr_va_vcc(0)                                // 0000000051b0: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v24, vcc_lo                    // 0000000051b4: 022a3115
	v_add_co_u32 v2, vcc_lo, s56, v2                           // 0000000051b8: d7006a02 02020438
	s_wait_alu depctr_va_vcc(0)                                // 0000000051c0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s57, v3, vcc_lo              // 0000000051c4: d5207c03 01aa0639
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 0000000051cc: 3e181881
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051d0: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v4                            // 0000000051d4: d7006a02 02020902
	s_wait_alu depctr_va_vcc(0)                                // 0000000051dc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v5, vcc_lo               // 0000000051e0: d5207c03 01aa0b03
	v_add3_u32 v22, v22, v58, 0x7fff                           // 0000000051e8: d6550016 03fe7516 00007fff
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 0000000051f4: 7c30753a
	global_store_d16_hi_b16 v[2:3], v21, off                   // 0000000051f8: ee09407c 0a800000 00000002
	v_bfe_u32 v21, v59, 16, 1                                  // 000000005204: d6100015 0205213b
	s_wait_alu depctr_va_vcc(0)                                // 00000000520c: bf88ff9d
	v_cndmask_b32_e32 v20, v22, v23, vcc_lo                    // 000000005210: 02282f16
	v_add_co_u32 v12, vcc_lo, s56, v12                         // 000000005214: d7006a0c 02021838
	s_wait_alu depctr_va_vcc(0)                                // 00000000521c: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s57, v13, vcc_lo            // 000000005220: d5207c0d 01aa1a39
	v_add3_u32 v21, v21, v59, 0x7fff                           // 000000005228: d6550015 03fe7715 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005234: bf870003
	v_add_co_u32 v4, vcc_lo, v12, v4                           // 000000005238: d7006a04 0202090c
	v_or_b32_e32 v22, 0x400000, v59                            // 000000005240: 382c76ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005248: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v13, v5, vcc_lo              // 00000000524c: d5207c05 01aa0b0d
	v_bfe_u32 v12, v57, 16, 1                                  // 000000005254: d610000c 02052139
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 00000000525c: 7c30773b
	global_store_d16_hi_b16 v[4:5], v20, off                   // 000000005260: ee09407c 0a000000 00000004
	v_or_b32_e32 v20, 0x400000, v57                            // 00000000526c: 382872ff 00400000
	v_add3_u32 v12, v12, v57, 0x7fff                           // 000000005274: d655000c 03fe730c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005280: bf88ff9d
	v_cndmask_b32_e32 v13, v21, v22, vcc_lo                    // 000000005284: 021a2d15
	v_bfe_u32 v21, v56, 16, 1                                  // 000000005288: d6100015 02052138
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000005290: 7c307339
	global_store_d16_hi_b16 v[0:1], v13, off offset:32         // 000000005294: ee09407c 06800000 00002000
	v_add3_u32 v0, v21, v56, 0x7fff                            // 0000000052a0: d6550000 03fe7115 00007fff
	v_or_b32_e32 v1, 0x400000, v56                             // 0000000052ac: 380270ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000052b4: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v20, vcc_lo                    // 0000000052b8: 0218290c
	v_bfe_u32 v13, v55, 16, 1                                  // 0000000052bc: d610000d 02052137
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 0000000052c4: 7c307138
	global_store_d16_hi_b16 v[7:8], v12, off offset:32         // 0000000052c8: ee09407c 06000000 00002007
	v_add3_u32 v7, v13, v55, 0x7fff                            // 0000000052d4: d6550007 03fe6f0d 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000052e0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000052e4: 02000300
	v_bfe_u32 v1, v54, 16, 1                                   // 0000000052e8: d6100001 02052136
	v_or_b32_e32 v8, 0x400000, v55                             // 0000000052f0: 38106eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 0000000052f8: 7c306f37
	v_or_b32_e32 v12, 0x400000, v9                             // 0000000052fc: 381812ff 00400000
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 000000005304: ee09407c 00000000 0000200a
	v_add3_u32 v0, v1, v54, 0x7fff                             // 000000005310: d6550000 03fe6d01 00007fff
	v_or_b32_e32 v1, 0x400000, v54                             // 00000000531c: 38026cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005324: bf88ff9d
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo                       // 000000005328: 020e1107
	v_bfe_u32 v8, v53, 16, 1                                   // 00000000532c: d6100008 02052135
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 000000005334: 7c306d36
	v_bfe_u32 v10, v9, 16, 1                                   // 000000005338: d610000a 02052109
	v_or_b32_e32 v11, 0x400000, v53                            // 000000005340: 38166aff 00400000
	v_or_b32_e32 v13, 0x400000, v6                             // 000000005348: 381a0cff 00400000
	v_add3_u32 v8, v8, v53, 0x7fff                             // 000000005350: d6550008 03fe6b08 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000535c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000005360: 02000300
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 000000005364: 7c306b35
	v_bfe_u32 v1, v6, 16, 1                                    // 000000005368: d6100001 02052106
	v_add3_u32 v10, v10, v9, 0x7fff                            // 000000005370: d655000a 03fe130a 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000537c: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v11, vcc_lo                      // 000000005380: 02101708
	v_cmp_u_f32_e32 vcc_lo, v9, v9                             // 000000005384: 7c301309
	v_add3_u32 v1, v1, v6, 0x7fff                              // 000000005388: d6550001 03fe0d01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005394: bf88ff9d
	v_cndmask_b32_e32 v9, v10, v12, vcc_lo                     // 000000005398: 0212190a
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 00000000539c: 7c300d06
	s_wait_alu depctr_va_vcc(0)                                // 0000000053a0: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v13, vcc_lo                      // 0000000053a4: 02021b01
	s_clause 0x4                                               // 0000000053a8: bf850004
	global_store_d16_hi_b16 v[18:19], v7, off offset:32        // 0000000053ac: ee09407c 03800000 00002012
	global_store_d16_hi_b16 v[16:17], v0, off offset:32        // 0000000053b8: ee09407c 00000000 00002010
	global_store_d16_hi_b16 v[14:15], v8, off offset:32        // 0000000053c4: ee09407c 04000000 0000200e
	global_store_d16_hi_b16 v[2:3], v9, off offset:32          // 0000000053d0: ee09407c 04800000 00002002
	global_store_d16_hi_b16 v[4:5], v1, off offset:32          // 0000000053dc: ee09407c 00800000 00002004
	s_nop 0                                                    // 0000000053e8: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000053ec: bfb60003
	s_endpgm                                                   // 0000000053f0: bfb00000
	s_code_end                                                 // 0000000053f4: bf9f0000
	s_code_end                                                 // 0000000053f8: bf9f0000
	s_code_end                                                 // 0000000053fc: bf9f0000
	s_code_end                                                 // 000000005400: bf9f0000
	s_code_end                                                 // 000000005404: bf9f0000
	s_code_end                                                 // 000000005408: bf9f0000
	s_code_end                                                 // 00000000540c: bf9f0000
	s_code_end                                                 // 000000005410: bf9f0000
	s_code_end                                                 // 000000005414: bf9f0000
	s_code_end                                                 // 000000005418: bf9f0000
	s_code_end                                                 // 00000000541c: bf9f0000
	s_code_end                                                 // 000000005420: bf9f0000
	s_code_end                                                 // 000000005424: bf9f0000
	s_code_end                                                 // 000000005428: bf9f0000
	s_code_end                                                 // 00000000542c: bf9f0000
	s_code_end                                                 // 000000005430: bf9f0000
	s_code_end                                                 // 000000005434: bf9f0000
	s_code_end                                                 // 000000005438: bf9f0000
	s_code_end                                                 // 00000000543c: bf9f0000
	s_code_end                                                 // 000000005440: bf9f0000
	s_code_end                                                 // 000000005444: bf9f0000
	s_code_end                                                 // 000000005448: bf9f0000
	s_code_end                                                 // 00000000544c: bf9f0000
	s_code_end                                                 // 000000005450: bf9f0000
	s_code_end                                                 // 000000005454: bf9f0000
	s_code_end                                                 // 000000005458: bf9f0000
	s_code_end                                                 // 00000000545c: bf9f0000
	s_code_end                                                 // 000000005460: bf9f0000
	s_code_end                                                 // 000000005464: bf9f0000
	s_code_end                                                 // 000000005468: bf9f0000
	s_code_end                                                 // 00000000546c: bf9f0000
	s_code_end                                                 // 000000005470: bf9f0000
	s_code_end                                                 // 000000005474: bf9f0000
	s_code_end                                                 // 000000005478: bf9f0000
	s_code_end                                                 // 00000000547c: bf9f0000
	s_code_end                                                 // 000000005480: bf9f0000
	s_code_end                                                 // 000000005484: bf9f0000
	s_code_end                                                 // 000000005488: bf9f0000
	s_code_end                                                 // 00000000548c: bf9f0000
	s_code_end                                                 // 000000005490: bf9f0000
	s_code_end                                                 // 000000005494: bf9f0000
	s_code_end                                                 // 000000005498: bf9f0000
	s_code_end                                                 // 00000000549c: bf9f0000
	s_code_end                                                 // 0000000054a0: bf9f0000
	s_code_end                                                 // 0000000054a4: bf9f0000
	s_code_end                                                 // 0000000054a8: bf9f0000
	s_code_end                                                 // 0000000054ac: bf9f0000
	s_code_end                                                 // 0000000054b0: bf9f0000
	s_code_end                                                 // 0000000054b4: bf9f0000
	s_code_end                                                 // 0000000054b8: bf9f0000
	s_code_end                                                 // 0000000054bc: bf9f0000
	s_code_end                                                 // 0000000054c0: bf9f0000
	s_code_end                                                 // 0000000054c4: bf9f0000
	s_code_end                                                 // 0000000054c8: bf9f0000
	s_code_end                                                 // 0000000054cc: bf9f0000
	s_code_end                                                 // 0000000054d0: bf9f0000
	s_code_end                                                 // 0000000054d4: bf9f0000
	s_code_end                                                 // 0000000054d8: bf9f0000
	s_code_end                                                 // 0000000054dc: bf9f0000
	s_code_end                                                 // 0000000054e0: bf9f0000
	s_code_end                                                 // 0000000054e4: bf9f0000
	s_code_end                                                 // 0000000054e8: bf9f0000
	s_code_end                                                 // 0000000054ec: bf9f0000
	s_code_end                                                 // 0000000054f0: bf9f0000
	s_code_end                                                 // 0000000054f4: bf9f0000
	s_code_end                                                 // 0000000054f8: bf9f0000
	s_code_end                                                 // 0000000054fc: bf9f0000
	s_code_end                                                 // 000000005500: bf9f0000
	s_code_end                                                 // 000000005504: bf9f0000
	s_code_end                                                 // 000000005508: bf9f0000
	s_code_end                                                 // 00000000550c: bf9f0000
	s_code_end                                                 // 000000005510: bf9f0000
	s_code_end                                                 // 000000005514: bf9f0000
	s_code_end                                                 // 000000005518: bf9f0000
	s_code_end                                                 // 00000000551c: bf9f0000
	s_code_end                                                 // 000000005520: bf9f0000
	s_code_end                                                 // 000000005524: bf9f0000
	s_code_end                                                 // 000000005528: bf9f0000
	s_code_end                                                 // 00000000552c: bf9f0000
	s_code_end                                                 // 000000005530: bf9f0000
	s_code_end                                                 // 000000005534: bf9f0000
	s_code_end                                                 // 000000005538: bf9f0000
	s_code_end                                                 // 00000000553c: bf9f0000
	s_code_end                                                 // 000000005540: bf9f0000
	s_code_end                                                 // 000000005544: bf9f0000
	s_code_end                                                 // 000000005548: bf9f0000
	s_code_end                                                 // 00000000554c: bf9f0000
	s_code_end                                                 // 000000005550: bf9f0000
	s_code_end                                                 // 000000005554: bf9f0000
	s_code_end                                                 // 000000005558: bf9f0000
	s_code_end                                                 // 00000000555c: bf9f0000
	s_code_end                                                 // 000000005560: bf9f0000
	s_code_end                                                 // 000000005564: bf9f0000
	s_code_end                                                 // 000000005568: bf9f0000
	s_code_end                                                 // 00000000556c: bf9f0000
	s_code_end                                                 // 000000005570: bf9f0000
	s_code_end                                                 // 000000005574: bf9f0000
	s_code_end                                                 // 000000005578: bf9f0000
	s_code_end                                                 // 00000000557c: bf9f0000
