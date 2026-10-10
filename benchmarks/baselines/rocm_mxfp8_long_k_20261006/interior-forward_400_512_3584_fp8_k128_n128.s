
/tmp/tmpv58d9bhl.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_9160add00a4b6aae>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[12:15], s[0:1], 0xc8                         // 000000001b04: f4004300 f80000c8
	s_load_b64 s[24:25], s[0:1], 0xd8                          // 000000001b0c: f4002600 f80000d8
	s_load_b64 s[16:17], s[0:1], 0xa8                          // 000000001b14: f4002400 f80000a8
	s_load_b64 s[28:29], s[0:1], 0x8                           // 000000001b1c: f4002700 f8000008
	s_load_b64 s[26:27], s[0:1], 0x30                          // 000000001b24: f4002680 f8000030
	s_load_b64 s[18:19], s[0:1], 0x58                          // 000000001b2c: f4002480 f8000058
	s_load_b64 s[20:21], s[0:1], 0x80                          // 000000001b34: f4002500 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b40: 86039f75
	s_mov_b32 s4, ttmp7                                        // 000000001b44: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[30:31], s[2:3], 5                             // 000000001b4c: 849e8502
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000001b50: bf8700c9
	v_dual_mov_b32 v17, s31 :: v_dual_and_b32 v32, 15, v0      // 000000001b54: ca24001f 1120008f
	s_lshl_b64 s[34:35], s[4:5], 4                             // 000000001b5c: 84a28404
	s_add_nc_u64 s[2:3], s[30:31], 32                          // 000000001b60: a982a01e
	s_add_nc_u64 s[0:1], s[34:35], 16                          // 000000001b64: a9809022
	v_or_b32_e32 v16, s30, v32                                 // 000000001b68: 3820401e
	v_mov_b32_e32 v19, s31                                     // 000000001b6c: 7e26021f
	v_bfe_u32 v33, v0, 4, 1                                    // 000000001b70: d6100021 02050900
	s_delay_alu instid0(valu_dep_3)                            // 000000001b78: bf870003
	v_or_b32_e32 v18, 16, v16                                  // 000000001b7c: 38242090
	s_wait_kmcnt 0x0                                           // 000000001b80: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[12:13]                      // 000000001b84: d4540000 02001800
	v_cmp_gt_i64_e64 s1, s[2:3], s[14:15]                      // 000000001b8c: d4540001 02001c02
	v_cmp_gt_i64_e64 s33, 0x80, s[24:25]                       // 000000001b94: d4540021 020030ff 00000080
	s_and_b32 s22, s24, 0xffffff80                             // 000000001ba0: 8b16ff18 ffffff80
	s_mov_b32 s23, s25                                         // 000000001ba8: be970019
	s_or_b32 s0, s0, s1                                        // 000000001bac: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb4: 8b6a007e
	s_mov_b32 s0, -1                                           // 000000001bb8: be8000c1
	s_cbranch_vccz 2496                                        // 000000001bbc: bfa309c0 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x27c0>
	s_and_b32 s0, s33, exec_lo                                 // 000000001bc0: 8b007e21
	s_cselect_b32 s0, 1, 0                                     // 000000001bc4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bcc: bf078100
	s_cbranch_scc1 5                                           // 000000001bd0: bfa20005 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0xe8>
	v_lshl_or_b32 v20, v33, 3, s34                             // 000000001bd4: d6560014 00890721
	v_mov_b32_e32 v21, s35                                     // 000000001bdc: 7e2a0223
	s_mov_b32 s0, 0                                            // 000000001be0: be800080
	s_branch 1                                                 // 000000001be4: bfa00001 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0xec>
	s_mov_b32 s0, -1                                           // 000000001be8: be8000c1
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0             // 000000001bec: ca100080 2a2a0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bf4: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001bf8: 8b007e00
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0             // 000000001bfc: ca100080 2c2c0080
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0             // 000000001c04: ca100080 2e2e0080
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v55, 0             // 000000001c0c: ca100080 34360080
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0             // 000000001c14: ca100080 22220080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0             // 000000001c1c: ca100080 24240080
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0             // 000000001c24: ca100080 26260080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0             // 000000001c2c: ca100080 28280080
	s_cselect_b32 s0, 1, 0                                     // 000000001c34: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c38: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c3c: bf078100
	s_cbranch_scc1 1808                                        // 000000001c40: bfa20710 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1d84>
	v_dual_mov_b32 v21, s35 :: v_dual_lshlrev_b32 v22, 3, v33  // 000000001c44: ca220023 15164283
	s_add_nc_u64 s[0:1], s[14:15], 0x7f                        // 000000001c4c: a980ff0e 0000007f
	s_lshr_b64 s[6:7], s[24:25], 7                             // 000000001c54: 85868718
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c58: bf88ff9e
	s_lshr_b64 s[36:37], s[0:1], 7                             // 000000001c5c: 85a48700
	v_or_b32_e32 v20, s34, v22                                 // 000000001c60: 38282c22
	s_lshr_b64 s[0:1], s[30:31], 7                             // 000000001c64: 8580871e
	s_add_nc_u64 s[2:3], s[36:37], -1                          // 000000001c68: a982c124
	v_dual_mov_b32 v1, s35 :: v_dual_mov_b32 v2, s35           // 000000001c6c: ca100023 01020023
	s_delay_alu instid0(valu_dep_2)                            // 000000001c74: bf870002
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[20:21]                // 000000001c78: 7ca8280c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c7c: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[0:1], s[2:3]                        // 000000001c80: d4590004 02000400
	v_or_b32_e32 v0, 1, v20                                    // 000000001c88: 38002881
	v_or_b32_e32 v5, 2, v20                                    // 000000001c8c: 380a2882
	s_mov_b64 s[42:43], 0                                      // 000000001c90: beaa0180
	v_dual_mov_b32 v6, s35 :: v_dual_cndmask_b32 v3, 0, v20    // 000000001c94: ca120023 06022880
	v_cndmask_b32_e64 v4, 0, s35, vcc_lo                       // 000000001c9c: d5010004 01a84680
	s_and_b32 s4, s4, exec_lo                                  // 000000001ca4: 8b047e04
	s_cselect_b32 s5, s1, s3                                   // 000000001ca8: 98050301
	s_cselect_b32 s4, s0, s2                                   // 000000001cac: 98040200
	s_lshr_b32 s7, s25, 7                                      // 000000001cb0: 85078719
	v_mul_lo_u32 v8, s6, v4                                    // 000000001cb4: d72c0008 02020806
	v_mul_lo_u32 v7, s7, v3                                    // 000000001cbc: d72c0007 02020607
	v_mad_co_u64_u32 v[3:4], null, s6, v3, 0                   // 000000001cc4: d6fe7c03 02020606
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[0:1]                  // 000000001ccc: 7ca8000c
	v_mov_b32_e32 v23, 0                                       // 000000001cd0: 7e2e0280
	v_or_b32_e32 v1, s34, v32                                  // 000000001cd4: 38024022
	v_cmp_gt_i64_e64 s1, s[14:15], v[16:17]                    // 000000001cd8: d4540001 0202200e
	v_cmp_gt_i64_e64 s2, s[14:15], v[18:19]                    // 000000001ce0: d4540002 0202240e
	s_lshl_b64 s[38:39], s[4:5], 2                             // 000000001ce8: 84a68204
	s_wait_alu depctr_va_vcc(0)                                // 000000001cec: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo                        // 000000001cf0: 02000080
	v_cndmask_b32_e64 v9, 0, s35, vcc_lo                       // 000000001cf4: d5010009 01a84680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001cfc: 7ca80a0c
	v_add3_u32 v4, v4, v8, v7                                  // 000000001d00: d6550004 041e1104
	v_cmp_gt_i64_e64 s0, s[12:13], v[1:2]                      // 000000001d08: d4540000 0202020c
	v_mul_lo_u32 v7, s7, v0                                    // 000000001d10: d72c0007 02020007
	v_mul_lo_u32 v8, s6, v9                                    // 000000001d18: d72c0008 02021206
	v_mad_co_u64_u32 v[0:1], null, s6, v0, 0                   // 000000001d20: d6fe7c00 02020006
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]                        // 000000001d28: 3e040682
	s_wait_alu depctr_va_vcc(0)                                // 000000001d2c: bf88ff9d
	v_dual_cndmask_b32 v6, 0, v5 :: v_dual_mov_b32 v5, s35     // 000000001d30: ca500a80 06040023
	v_or_b32_e32 v4, 3, v20                                    // 000000001d38: 38082883
	v_cndmask_b32_e64 v9, 0, s35, vcc_lo                       // 000000001d3c: d5010009 01a84680
	v_dual_mov_b32 v52, v23 :: v_dual_mov_b32 v45, v23         // 000000001d44: ca100117 342c0117
	v_add3_u32 v1, v1, v8, v7                                  // 000000001d4c: d6550001 041e1101
	s_delay_alu instid0(valu_dep_4)                            // 000000001d54: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[4:5]                  // 000000001d58: 7ca8080c
	v_mul_lo_u32 v8, s7, v6                                    // 000000001d5c: d72c0008 02020c07
	v_mul_lo_u32 v9, s6, v9                                    // 000000001d64: d72c0009 02021206
	v_mad_co_u64_u32 v[6:7], null, s6, v6, 0                   // 000000001d6c: d6fe7c06 02020c06
	v_add_co_u32 v48, s3, s18, v2                              // 000000001d74: d7000330 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000001d7c: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s19, v3, s3                 // 000000001d80: d5207c31 000e0613
	s_wait_alu depctr_va_vcc(0)                                // 000000001d88: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_mov_b32 v3, s35     // 000000001d8c: ca500880 04020023
	v_or_b32_e32 v2, 4, v20                                    // 000000001d94: 38042884
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000001d98: 3e000082
	v_cndmask_b32_e64 v5, 0, s35, vcc_lo                       // 000000001d9c: d5010005 01a84680
	v_add3_u32 v7, v7, v9, v8                                  // 000000001da4: d6550007 04221307
	v_mul_lo_u32 v8, s7, v4                                    // 000000001dac: d72c0008 02020807
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001db4: 7ca8040c
	v_dual_mov_b32 v46, v23 :: v_dual_mov_b32 v43, v23         // 000000001db8: ca100117 2e2a0117
	v_mul_lo_u32 v9, s6, v5                                    // 000000001dc0: d72c0009 02020a06
	v_mad_co_u64_u32 v[4:5], null, s6, v4, 0                   // 000000001dc8: d6fe7c04 02020806
	v_add_co_u32 v50, s3, s18, v0                              // 000000001dd0: d7000332 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000001dd8: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s19, v1, s3                 // 000000001ddc: d5207c33 000e0213
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001de4: 3e000c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001de8: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v2, vcc_lo                        // 000000001dec: 020c0480
	v_cndmask_b32_e64 v7, 0, s35, vcc_lo                       // 000000001df0: d5010007 01a84680
	v_or_b32_e32 v2, 5, v20                                    // 000000001df8: 38042885
	v_add3_u32 v5, v5, v9, v8                                  // 000000001dfc: d6550005 04221305
	v_mov_b32_e32 v55, v23                                     // 000000001e04: 7e6e0317
	v_mul_lo_u32 v8, s7, v6                                    // 000000001e08: d72c0008 02020c07
	v_mul_lo_u32 v9, s6, v7                                    // 000000001e10: d72c0009 02020e06
	v_mad_co_u64_u32 v[6:7], null, s6, v6, 0                   // 000000001e18: d6fe7c06 02020c06
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e20: 7ca8040c
	v_add_co_u32 v53, s3, s18, v0                              // 000000001e24: d7000335 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000001e2c: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s19, v1, s3                 // 000000001e30: d5207c36 000e0213
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001e38: 3e000882
	s_wait_alu depctr_va_vcc(0)                                // 000000001e3c: bf88ff9d
	v_cndmask_b32_e32 v4, 0, v2, vcc_lo                        // 000000001e40: 02080480
	v_or_b32_e32 v2, 6, v20                                    // 000000001e44: 38042886
	v_add3_u32 v7, v7, v9, v8                                  // 000000001e48: d6550007 04221307
	v_or_b32_e32 v8, 7, v20                                    // 000000001e50: 38102887
	v_mov_b32_e32 v9, s35                                      // 000000001e54: 7e120223
	v_cndmask_b32_e64 v5, 0, s35, vcc_lo                       // 000000001e58: d5010005 01a84680
	v_add_co_u32 v56, s3, s18, v0                              // 000000001e60: d7000338 02020012
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000001e68: 7ca8040c
	s_wait_alu depctr_va_sdst(0)                               // 000000001e6c: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s19, v1, s3                 // 000000001e70: d5207c39 000e0213
	v_cmp_gt_i64_e64 s3, s[12:13], v[8:9]                      // 000000001e78: d4540003 0202100c
	v_mul_lo_u32 v10, s7, v4                                   // 000000001e80: d72c000a 02020807
	v_mul_lo_u32 v11, s6, v5                                   // 000000001e88: d72c000b 02020a06
	v_mad_co_u64_u32 v[4:5], null, s6, v4, 0                   // 000000001e90: d6fe7c04 02020806
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e98: 3e000c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001e9c: bf88ff9d
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_mov_b32 v47, v23    // 000000001ea0: ca500480 022e0117
	v_cndmask_b32_e64 v3, 0, s35, vcc_lo                       // 000000001ea8: d5010003 01a84680
	s_wait_alu depctr_va_sdst(0)                               // 000000001eb0: bf88f19f
	v_cndmask_b32_e64 v6, 0, v8, s3                            // 000000001eb4: d5010006 000e1080
	v_cndmask_b32_e64 v7, 0, s35, s3                           // 000000001ebc: d5010007 000c4680
	v_mul_lo_u32 v8, s7, v2                                    // 000000001ec4: d72c0008 02020407
	v_add3_u32 v5, v5, v11, v10                                // 000000001ecc: d6550005 042a1705
	v_mul_lo_u32 v9, s6, v3                                    // 000000001ed4: d72c0009 02020606
	v_mad_co_u64_u32 v[2:3], null, s6, v2, 0                   // 000000001edc: d6fe7c02 02020406
	v_mul_lo_u32 v10, s7, v6                                   // 000000001ee4: d72c000a 02020c07
	v_mul_lo_u32 v11, s6, v7                                   // 000000001eec: d72c000b 02020e06
	v_mad_co_u64_u32 v[6:7], null, s6, v6, 0                   // 000000001ef4: d6fe7c06 02020c06
	v_add_co_u32 v58, vcc_lo, s18, v0                          // 000000001efc: d7006a3a 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000001f04: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s19, v1, vcc_lo             // 000000001f08: d5207c3b 01aa0213
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001f10: 3e000882
	v_add_co_u32 v4, s3, s30, v32                              // 000000001f14: d7000304 0202401e
	s_wait_alu depctr_va_sdst(0)                               // 000000001f1c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, 0, s3                   // 000000001f20: d5207c05 000d001f
	v_add3_u32 v3, v3, v9, v8                                  // 000000001f28: d6550003 04221303
	v_add3_u32 v7, v7, v11, v10                                // 000000001f30: d6550007 042a1707
	v_add_co_u32 v8, vcc_lo, v4, 16                            // 000000001f38: d7006a08 02012104
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v5, vcc_lo                // 000000001f44: d5207c09 01aa0a80
	v_add_co_u32 v60, vcc_lo, s18, v0                          // 000000001f4c: d7006a3c 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000001f54: bf88ff9d
	v_add_co_ci_u32_e64 v61, null, s19, v1, vcc_lo             // 000000001f58: d5207c3d 01aa0213
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001f60: 3e000482
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001f64: 3e040c82
	v_mad_co_u64_u32 v[24:25], null, s24, v8, v[22:23]         // 000000001f68: d6fe7c18 045a1018
	v_mul_lo_u32 v7, s25, v8                                   // 000000001f70: d72c0007 02021019
	v_add_co_u32 v8, s3, s34, v32                              // 000000001f78: d7000308 02024022
	v_mul_lo_u32 v6, s24, v9                                   // 000000001f80: d72c0006 02021218
	s_wait_alu depctr_va_sdst(0)                               // 000000001f88: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s35, 0, s3                   // 000000001f8c: d5207c09 000d0023
	v_mad_co_u64_u32 v[26:27], null, s24, v4, v[22:23]         // 000000001f94: d6fe7c1a 045a0818
	v_mul_lo_u32 v5, s24, v5                                   // 000000001f9c: d72c0005 02020a18
	v_mul_lo_u32 v4, s25, v4                                   // 000000001fa4: d72c0004 02020819
	v_mad_co_u64_u32 v[28:29], null, s24, v8, v[22:23]         // 000000001fac: d6fe7c1c 045a1018
	v_mul_lo_u32 v9, s24, v9                                   // 000000001fb4: d72c0009 02021218
	v_mul_lo_u32 v8, s25, v8                                   // 000000001fbc: d72c0008 02021019
	v_add_co_u32 v62, vcc_lo, s18, v0                          // 000000001fc4: d7006a3e 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000001fcc: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, s19, v1, vcc_lo             // 000000001fd0: d5207c3f 01aa0213
	v_add_co_u32 v64, vcc_lo, s18, v2                          // 000000001fd8: d7006a40 02020412
	s_wait_alu depctr_va_vcc(0)                                // 000000001fe0: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s19, v3, vcc_lo             // 000000001fe4: d5207c41 01aa0613
	v_add3_u32 v25, v7, v25, v6                                // 000000001fec: d6550019 041a3307
	v_add3_u32 v27, v4, v27, v5                                // 000000001ff4: d655001b 04163704
	v_add3_u32 v29, v8, v29, v9                                // 000000001ffc: d655001d 04263b08
	v_dual_mov_b32 v44, v23 :: v_dual_mov_b32 v41, v23         // 000000002004: ca100117 2c280117
	v_dual_mov_b32 v42, v23 :: v_dual_mov_b32 v39, v23         // 00000000200c: ca100117 2a260117
	v_dual_mov_b32 v40, v23 :: v_dual_mov_b32 v37, v23         // 000000002014: ca100117 28240117
	v_dual_mov_b32 v38, v23 :: v_dual_mov_b32 v35, v23         // 00000000201c: ca100117 26220117
	v_mov_b32_e32 v36, v23                                     // 000000002024: 7e480317
	v_mov_b32_e32 v34, v23                                     // 000000002028: 7e440317
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v9, v23             // 00000000202c: ca100080 08080117
	v_dual_mov_b32 v10, v23 :: v_dual_mov_b32 v11, v23         // 000000002034: ca100117 0a0a0117
	v_dual_mov_b32 v12, v23 :: v_dual_mov_b32 v13, v23         // 00000000203c: ca100117 0c0c0117
	v_dual_mov_b32 v14, v23 :: v_dual_mov_b32 v15, v23         // 000000002044: ca100117 0e0e0117
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, v23             // 00000000204c: ca100080 00000117
	v_dual_mov_b32 v2, v23 :: v_dual_mov_b32 v3, v23           // 000000002054: ca100117 02020117
	v_dual_mov_b32 v4, v23 :: v_dual_mov_b32 v5, v23           // 00000000205c: ca100117 04040117
	v_dual_mov_b32 v6, v23 :: v_dual_mov_b32 v7, v23           // 000000002064: ca100117 06060117
	s_add_nc_u64 s[40:41], s[42:43], 0x80                      // 00000000206c: a9a8ff2a 00000080
	s_mov_b64 s[44:45], s[42:43]                               // 000000002074: beac012a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002078: bf88ff9e
	v_add_co_u32 v30, s3, v22, s44                             // 00000000207c: d700031e 02005916
	s_wait_alu depctr_va_sdst(0)                               // 000000002084: bf88f19f
	v_add_co_ci_u32_e64 v31, null, 0, s45, s3                  // 000000002088: d5207c1f 000c5a80
	v_add_co_u32 v70, vcc_lo, v28, s44                         // 000000002090: d7006a46 0200591c
	s_wait_alu depctr_va_vcc(0)                                // 000000002098: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s45, v29, vcc_lo            // 00000000209c: d5207c47 01aa3a2d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 0000000020a4: bf8700c3
	v_cmp_gt_i64_e64 s9, s[24:25], v[30:31]                    // 0000000020a8: d4540009 02023c18
	s_and_b32 vcc_lo, s0, s9                                   // 0000000020b0: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020b4: bf88ff9e
	v_dual_cndmask_b32 v67, 0, v71 :: v_dual_cndmask_b32 v66, 0, v70// 0000000020b8: ca528e80 43428c80
	v_add_co_u32 v66, s3, s28, v66                             // 0000000020c0: d7000342 0202841c
	s_wait_alu depctr_va_sdst(0)                               // 0000000020c8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000020cc: bf870002
	v_add_co_ci_u32_e64 v67, null, s29, v67, s3                // 0000000020d0: d5207c43 000e861d
	global_load_d16_u8 v66, v[66:67], off                      // 0000000020d8: ee07807c 00000042 00000042
	s_wait_loadcnt 0x0                                         // 0000000020e4: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, vcc_lo                      // 0000000020e8: d65d0042 01aa8480
	v_add_co_u32 v69, vcc_lo, v70, 1                           // 0000000020f0: d7006a45 02010346
	s_wait_alu depctr_va_vcc(0)                                // 0000000020f8: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, 0, v71, vcc_lo              // 0000000020fc: d5207c48 01aa8e80
	v_add_co_u32 v67, vcc_lo, v30, 1                           // 000000002104: d7006a43 0201031e
	s_wait_alu depctr_va_vcc(0)                                // 00000000210c: bf88ff9d
	v_add_co_ci_u32_e64 v68, null, 0, v31, vcc_lo              // 000000002110: d5207c44 01aa3e80
	v_and_b16 v66.l, 0xff, v66.l                               // 000000002118: d7620042 020284ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002124: bf870152
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[67:68]                // 000000002128: 7ca88618
	s_and_b32 s3, s0, vcc_lo                                   // 00000000212c: 8b036a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002130: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v69, s3                          // 000000002134: d5010043 000e8a80
	v_cndmask_b32_e64 v68, 0, v72, s3                          // 00000000213c: d5010044 000e9080
	v_add_co_u32 v67, s4, s28, v67                             // 000000002144: d7000443 0202861c
	s_wait_alu depctr_va_sdst(0)                               // 00000000214c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002150: bf870002
	v_add_co_ci_u32_e64 v68, null, s29, v68, s4                // 000000002154: d5207c44 0012881d
	global_load_d16_hi_u8 v66, v[67:68], off                   // 00000000215c: ee08407c 00000042 00000043
	s_wait_loadcnt 0x0                                         // 000000002168: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, s3                          // 00000000216c: d65d5042 000e8480
	v_add_co_u32 v69, s3, v70, 2                               // 000000002174: d7000345 02010546
	s_wait_alu depctr_va_sdst(0)                               // 00000000217c: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v71, s3                  // 000000002180: d5207c48 000e8e80
	v_add_co_u32 v67, s3, v30, 2                               // 000000002188: d7000343 0201051e
	s_wait_alu depctr_va_sdst(0)                               // 000000002190: bf88f19f
	v_add_co_ci_u32_e64 v68, null, 0, v31, s3                  // 000000002194: d5207c44 000e3e80
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 00000000219c: d7385042 02028488
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000021a4: bf870112
	v_cmp_gt_i64_e64 s3, s[24:25], v[67:68]                    // 0000000021a8: d4540003 02028618
	v_or_b16 v76.l, v66.l, v66.h op_sel:[0,1,0]                // 0000000021b0: d763104c 02028542
	s_and_b32 s4, s0, s3                                       // 0000000021b8: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021bc: bf88ff9e
	v_cndmask_b32_e64 v67, 0, v69, s4                          // 0000000021c0: d5010043 00128a80
	v_cndmask_b32_e64 v68, 0, v72, s4                          // 0000000021c8: d5010044 00129080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000021d0: bf870122
	v_add_co_u32 v67, s5, s28, v67                             // 0000000021d4: d7000543 0202861c
	s_wait_alu depctr_va_sdst(0)                               // 0000000021dc: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s29, v68, s5                // 0000000021e0: d5207c44 0016881d
	global_load_d16_u8 v67, v[67:68], off                      // 0000000021e8: ee07807c 00000043 00000043
	s_wait_loadcnt 0x0                                         // 0000000021f4: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, s4                          // 0000000021f8: d65d0043 00128680
	v_add_co_u32 v72, s4, v70, 3                               // 000000002200: d7000448 02010746
	s_wait_alu depctr_va_sdst(0)                               // 000000002208: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v71, s4                  // 00000000220c: d5207c49 00128e80
	v_add_co_u32 v68, s4, v30, 3                               // 000000002214: d7000444 0201071e
	s_wait_alu depctr_va_sdst(0)                               // 00000000221c: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v31, s4                  // 000000002220: d5207c45 00123e80
	v_and_b16 v67.l, 0xff, v67.l                               // 000000002228: d7620043 020286ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002234: bf870152
	v_cmp_gt_i64_e64 s4, s[24:25], v[68:69]                    // 000000002238: d4540004 02028818
	s_and_b32 s5, s0, s4                                       // 000000002240: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002244: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v72, s5                          // 000000002248: d5010044 00169080
	v_cndmask_b32_e64 v69, 0, v73, s5                          // 000000002250: d5010045 00169280
	v_add_co_u32 v68, s6, s28, v68                             // 000000002258: d7000644 0202881c
	s_wait_alu depctr_va_sdst(0)                               // 000000002260: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002264: bf870002
	v_add_co_ci_u32_e64 v69, null, s29, v69, s6                // 000000002268: d5207c45 001a8a1d
	global_load_d16_hi_u8 v67, v[68:69], off                   // 000000002270: ee08407c 00000043 00000044
	s_wait_loadcnt 0x0                                         // 00000000227c: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, s5                          // 000000002280: d65d5043 00168680
	v_add_co_u32 v72, s5, v70, 4                               // 000000002288: d7000548 02010946
	s_wait_alu depctr_va_sdst(0)                               // 000000002290: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v71, s5                  // 000000002294: d5207c49 00168e80
	v_add_co_u32 v68, s5, v30, 4                               // 00000000229c: d7000544 0201091e
	s_wait_alu depctr_va_sdst(0)                               // 0000000022a4: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v31, s5                  // 0000000022a8: d5207c45 00163e80
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 0000000022b0: d7385043 02028688
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000022b8: bf870112
	v_cmp_gt_i64_e64 s5, s[24:25], v[68:69]                    // 0000000022bc: d4540005 02028818
	v_or_b16 v76.h, v67.l, v67.h op_sel:[0,1,1]                // 0000000022c4: d763504c 02028743
	s_and_b32 s6, s0, s5                                       // 0000000022cc: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v72, s6                          // 0000000022d4: d5010044 001a9080
	v_cndmask_b32_e64 v69, 0, v73, s6                          // 0000000022dc: d5010045 001a9280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000022e4: bf870122
	v_add_co_u32 v68, s7, s28, v68                             // 0000000022e8: d7000744 0202881c
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f0: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s29, v69, s7                // 0000000022f4: d5207c45 001e8a1d
	global_load_d16_u8 v68, v[68:69], off                      // 0000000022fc: ee07807c 00000044 00000044
	s_wait_loadcnt 0x0                                         // 000000002308: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s6                          // 00000000230c: d65d0044 001a8880
	v_add_co_u32 v69, s6, v70, 5                               // 000000002314: d7000645 02010b46
	s_wait_alu depctr_va_sdst(0)                               // 00000000231c: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v71, s6                  // 000000002320: d5207c4a 001a8e80
	v_add_co_u32 v72, s6, v30, 5                               // 000000002328: d7000648 02010b1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v31, s6                  // 000000002334: d5207c49 001a3e80
	v_and_b16 v68.l, 0xff, v68.l                               // 00000000233c: d7620044 020288ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002348: bf870152
	v_cmp_gt_i64_e64 s6, s[24:25], v[72:73]                    // 00000000234c: d4540006 02029018
	s_and_b32 s7, s0, s6                                       // 000000002354: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002358: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s7                          // 00000000235c: d5010045 001e8a80
	v_cndmask_b32_e64 v73, 0, v74, s7                          // 000000002364: d5010049 001e9480
	v_add_co_u32 v72, s8, s28, v69                             // 00000000236c: d7000848 02028a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002374: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002378: bf870002
	v_add_co_ci_u32_e64 v73, null, s29, v73, s8                // 00000000237c: d5207c49 0022921d
	global_load_d16_hi_u8 v68, v[72:73], off                   // 000000002384: ee08407c 00000044 00000048
	s_wait_loadcnt 0x0                                         // 000000002390: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s7                          // 000000002394: d65d5044 001e8880
	v_add_co_u32 v69, s7, v70, 6                               // 00000000239c: d7000745 02010d46
	s_wait_alu depctr_va_sdst(0)                               // 0000000023a4: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v71, s7                  // 0000000023a8: d5207c4a 001e8e80
	v_add_co_u32 v72, s7, v30, 6                               // 0000000023b0: d7000748 02010d1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b8: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v31, s7                  // 0000000023bc: d5207c49 001e3e80
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 0000000023c4: d7385044 02028888
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000023cc: bf870112
	v_cmp_gt_i64_e64 s7, s[24:25], v[72:73]                    // 0000000023d0: d4540007 02029018
	v_or_b16 v77.l, v68.l, v68.h op_sel:[0,1,0]                // 0000000023d8: d763104d 02028944
	s_and_b32 s8, s0, s7                                       // 0000000023e0: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023e4: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s8                          // 0000000023e8: d5010045 00228a80
	v_cndmask_b32_e64 v73, 0, v74, s8                          // 0000000023f0: d5010049 00229480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000023f8: bf870122
	v_add_co_u32 v72, s10, s28, v69                            // 0000000023fc: d7000a48 02028a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002404: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s29, v73, s10               // 000000002408: d5207c49 002a921d
	global_load_d16_u8 v69, v[72:73], off                      // 000000002410: ee07807c 00000045 00000048
	s_wait_loadcnt 0x0                                         // 00000000241c: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s8                          // 000000002420: d65d0045 00228a80
	v_add_co_u32 v74, s8, v70, 7                               // 000000002428: d700084a 02010f46
	s_wait_alu depctr_va_sdst(0)                               // 000000002430: bf88f19f
	v_add_co_ci_u32_e64 v75, null, 0, v71, s8                  // 000000002434: d5207c4b 00228e80
	v_add_co_u32 v72, s8, v30, 7                               // 00000000243c: d7000848 02010f1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002444: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v31, s8                  // 000000002448: d5207c49 00223e80
	v_and_b16 v69.l, 0xff, v69.l                               // 000000002450: d7620045 02028aff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 00000000245c: bf870152
	v_cmp_gt_i64_e64 s8, s[24:25], v[72:73]                    // 000000002460: d4540008 02029018
	s_and_b32 s10, s0, s8                                      // 000000002468: 8b0a0800
	s_wait_alu depctr_sa_sdst(0)                               // 00000000246c: bf88ff9e
	v_cndmask_b32_e64 v72, 0, v74, s10                         // 000000002470: d5010048 002a9480
	v_cndmask_b32_e64 v73, 0, v75, s10                         // 000000002478: d5010049 002a9680
	v_add_co_u32 v72, s11, s28, v72                            // 000000002480: d7000b48 0202901c
	s_wait_alu depctr_va_sdst(0)                               // 000000002488: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000248c: bf870002
	v_add_co_ci_u32_e64 v73, null, s29, v73, s11               // 000000002490: d5207c49 002e921d
	global_load_d16_hi_u8 v69, v[72:73], off                   // 000000002498: ee08407c 00000045 00000048
	s_wait_loadcnt 0x0                                         // 0000000024a4: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s10                         // 0000000024a8: d65d5045 002a8a80
	v_add_co_u32 v66, s10, v26, s44                            // 0000000024b0: d7000a42 0200591a
	s_wait_alu depctr_va_sdst(0)                               // 0000000024b8: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s45, v27, s10               // 0000000024bc: d5207c43 002a362d
	s_delay_alu instid0(valu_dep_3)                            // 0000000024c4: bf870003
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 0000000024c8: d7385045 02028a88
	s_and_b32 s10, s1, s9                                      // 0000000024d0: 8b0a0901
	s_and_b32 s9, s2, s9                                       // 0000000024d4: 8b090902
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024d8: bf88ff9e
	v_cndmask_b32_e64 v68, 0, v66, s10                         // 0000000024dc: d5010044 002a8480
	v_or_b16 v77.h, v69.l, v69.h op_sel:[0,1,1]                // 0000000024e4: d763504d 02028b45
	v_cndmask_b32_e64 v69, 0, v67, s10                         // 0000000024ec: d5010045 002a8680
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024f4: bf870123
	v_add_co_u32 v68, s11, s26, v68                            // 0000000024f8: d7000b44 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 000000002500: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s27, v69, s11               // 000000002504: d5207c45 002e8a1b
	global_load_d16_u8 v68, v[68:69], off                      // 00000000250c: ee07807c 00000044 00000044
	s_wait_loadcnt 0x0                                         // 000000002518: bfc00000
	v_cndmask_b16 v68.l, 0, v68.l, s10                         // 00000000251c: d65d0044 002a8880
	v_add_co_u32 v69, s10, v66, 1                              // 000000002524: d7000a45 02010342
	s_wait_alu depctr_va_sdst(0)                               // 00000000252c: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v67, s10                 // 000000002530: d5207c48 002a8680
	s_and_b32 s10, s1, vcc_lo                                  // 000000002538: 8b0a6a01
	v_and_b16 v68.l, 0xff, v68.l                               // 00000000253c: d7620044 020288ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002548: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 00000000254c: d5010045 002a8a80
	v_cndmask_b32_e64 v73, 0, v72, s10                         // 000000002554: d5010049 002a9080
	s_and_b32 vcc_lo, s2, vcc_lo                               // 00000000255c: 8b6a6a02
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002560: bf870122
	v_add_co_u32 v72, s11, s26, v69                            // 000000002564: d7000b48 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 00000000256c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 000000002570: d5207c49 002e921b
	global_load_d16_hi_u8 v68, v[72:73], off                   // 000000002578: ee08407c 00000044 00000048
	s_wait_loadcnt 0x0                                         // 000000002584: bfc00000
	v_cndmask_b16 v68.h, 0, v68.h, s10                         // 000000002588: d65d5044 002a8880
	v_add_co_u32 v69, s10, v66, 2                              // 000000002590: d7000a45 02010542
	s_wait_alu depctr_va_sdst(0)                               // 000000002598: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v67, s10                 // 00000000259c: d5207c48 002a8680
	s_and_b32 s10, s1, s3                                      // 0000000025a4: 8b0a0301
	v_lshlrev_b16 v68.h, 8, v68.h op_sel:[0,1,1]               // 0000000025a8: d7385044 02028888
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025b0: bf88ff9e
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 0000000025b4: d5010045 002a8a80
	v_cndmask_b32_e64 v73, 0, v72, s10                         // 0000000025bc: d5010049 002a9080
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000025c4: bf870193
	v_or_b16 v78.l, v68.l, v68.h op_sel:[0,1,0]                // 0000000025c8: d763104e 02028944
	v_add_co_u32 v72, s11, s26, v69                            // 0000000025d0: d7000b48 02028a1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000025d8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000025dc: bf870003
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 0000000025e0: d5207c49 002e921b
	global_load_d16_u8 v69, v[72:73], off                      // 0000000025e8: ee07807c 00000045 00000048
	s_wait_loadcnt 0x0                                         // 0000000025f4: bfc00000
	v_cndmask_b16 v69.l, 0, v69.l, s10                         // 0000000025f8: d65d0045 002a8a80
	v_add_co_u32 v72, s10, v66, 3                              // 000000002600: d7000a48 02010742
	s_wait_alu depctr_va_sdst(0)                               // 000000002608: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v67, s10                 // 00000000260c: d5207c49 002a8680
	s_and_b32 s10, s1, s4                                      // 000000002614: 8b0a0401
	v_and_b16 v69.l, 0xff, v69.l                               // 000000002618: d7620045 02028aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002624: bf88ff9e
	v_cndmask_b32_e64 v72, 0, v72, s10                         // 000000002628: d5010048 002a9080
	v_cndmask_b32_e64 v73, 0, v73, s10                         // 000000002630: d5010049 002a9280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002638: bf870122
	v_add_co_u32 v72, s11, s26, v72                            // 00000000263c: d7000b48 0202901a
	s_wait_alu depctr_va_sdst(0)                               // 000000002644: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 000000002648: d5207c49 002e921b
	global_load_d16_hi_u8 v69, v[72:73], off                   // 000000002650: ee08407c 00000045 00000048
	s_wait_loadcnt 0x0                                         // 00000000265c: bfc00000
	v_cndmask_b16 v69.h, 0, v69.h, s10                         // 000000002660: d65d5045 002a8a80
	v_add_co_u32 v72, s10, v66, 4                              // 000000002668: d7000a48 02010942
	s_wait_alu depctr_va_sdst(0)                               // 000000002670: bf88f19f
	v_add_co_ci_u32_e64 v73, null, 0, v67, s10                 // 000000002674: d5207c49 002a8680
	s_and_b32 s10, s1, s5                                      // 00000000267c: 8b0a0501
	v_lshlrev_b16 v69.h, 8, v69.h op_sel:[0,1,1]               // 000000002680: d7385045 02028a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000002688: bf88ff9e
	v_cndmask_b32_e64 v72, 0, v72, s10                         // 00000000268c: d5010048 002a9080
	v_cndmask_b32_e64 v73, 0, v73, s10                         // 000000002694: d5010049 002a9280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000269c: bf870193
	v_or_b16 v78.h, v69.l, v69.h op_sel:[0,1,1]                // 0000000026a0: d763504e 02028b45
	v_add_co_u32 v72, s11, s26, v72                            // 0000000026a8: d7000b48 0202901a
	s_wait_alu depctr_va_sdst(0)                               // 0000000026b0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000026b4: bf870003
	v_add_co_ci_u32_e64 v73, null, s27, v73, s11               // 0000000026b8: d5207c49 002e921b
	global_load_d16_u8 v72, v[72:73], off                      // 0000000026c0: ee07807c 00000048 00000048
	s_wait_loadcnt 0x0                                         // 0000000026cc: bfc00000
	v_cndmask_b16 v72.l, 0, v72.l, s10                         // 0000000026d0: d65d0048 002a9080
	v_add_co_u32 v73, s10, v66, 5                              // 0000000026d8: d7000a49 02010b42
	s_wait_alu depctr_va_sdst(0)                               // 0000000026e0: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v67, s10                 // 0000000026e4: d5207c4a 002a8680
	s_and_b32 s10, s1, s6                                      // 0000000026ec: 8b0a0601
	v_and_b16 v72.l, 0xff, v72.l                               // 0000000026f0: d7620048 020290ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026fc: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v73, s10                         // 000000002700: d5010049 002a9280
	v_cndmask_b32_e64 v74, 0, v74, s10                         // 000000002708: d501004a 002a9480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002710: bf870122
	v_add_co_u32 v73, s11, s26, v73                            // 000000002714: d7000b49 0202921a
	s_wait_alu depctr_va_sdst(0)                               // 00000000271c: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s27, v74, s11               // 000000002720: d5207c4a 002e941b
	global_load_d16_hi_u8 v72, v[73:74], off                   // 000000002728: ee08407c 00000048 00000049
	s_wait_loadcnt 0x0                                         // 000000002734: bfc00000
	v_cndmask_b16 v72.h, 0, v72.h, s10                         // 000000002738: d65d5048 002a9080
	v_add_co_u32 v73, s10, v66, 6                              // 000000002740: d7000a49 02010d42
	s_wait_alu depctr_va_sdst(0)                               // 000000002748: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v67, s10                 // 00000000274c: d5207c4a 002a8680
	s_and_b32 s10, s1, s7                                      // 000000002754: 8b0a0701
	v_lshlrev_b16 v72.h, 8, v72.h op_sel:[0,1,1]               // 000000002758: d7385048 02029088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002760: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v73, s10                         // 000000002764: d5010049 002a9280
	v_cndmask_b32_e64 v74, 0, v74, s10                         // 00000000276c: d501004a 002a9480
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002774: bf870193
	v_or_b16 v79.l, v72.l, v72.h op_sel:[0,1,0]                // 000000002778: d763104f 02029148
	v_add_co_u32 v73, s11, s26, v73                            // 000000002780: d7000b49 0202921a
	s_wait_alu depctr_va_sdst(0)                               // 000000002788: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000278c: bf870003
	v_add_co_ci_u32_e64 v74, null, s27, v74, s11               // 000000002790: d5207c4a 002e941b
	global_load_d16_u8 v73, v[73:74], off                      // 000000002798: ee07807c 00000049 00000049
	s_wait_loadcnt 0x0                                         // 0000000027a4: bfc00000
	v_cndmask_b16 v73.l, 0, v73.l, s10                         // 0000000027a8: d65d0049 002a9280
	v_add_co_u32 v74, s10, v66, 7                              // 0000000027b0: d7000a4a 02010f42
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b8: bf88f19f
	v_add_co_ci_u32_e64 v75, null, 0, v67, s10                 // 0000000027bc: d5207c4b 002a8680
	s_and_b32 s10, s1, s8                                      // 0000000027c4: 8b0a0801
	v_and_b16 v73.l, 0xff, v73.l                               // 0000000027c8: d7620049 020292ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027d4: bf88ff9e
	v_cndmask_b32_e64 v74, 0, v74, s10                         // 0000000027d8: d501004a 002a9480
	v_cndmask_b32_e64 v75, 0, v75, s10                         // 0000000027e0: d501004b 002a9680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000027e8: bf870122
	v_add_co_u32 v74, s11, s26, v74                            // 0000000027ec: d7000b4a 0202941a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f4: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s27, v75, s11               // 0000000027f8: d5207c4b 002e961b
	global_load_d16_hi_u8 v73, v[74:75], off                   // 000000002800: ee08407c 00000049 0000004a
	s_wait_loadcnt 0x0                                         // 00000000280c: bfc00000
	v_cndmask_b16 v73.h, 0, v73.h, s10                         // 000000002810: d65d5049 002a9280
	v_add_co_u32 v68, s10, v24, s44                            // 000000002818: d7000a44 02005918
	s_wait_alu depctr_va_sdst(0)                               // 000000002820: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s45, v25, s10               // 000000002824: d5207c45 002a322d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000282c: bf870193
	v_lshlrev_b16 v73.h, 8, v73.h op_sel:[0,1,1]               // 000000002830: d7385049 02029288
	v_cndmask_b32_e64 v72, 0, v68, s9                          // 000000002838: d5010048 00268880
	s_add_nc_u64 s[44:45], s[44:45], 32                        // 000000002840: a9aca02c
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002844: bf8701a2
	v_or_b16 v79.h, v73.l, v73.h op_sel:[0,1,1]                // 000000002848: d763504f 02029349
	v_cndmask_b32_e64 v73, 0, v69, s9                          // 000000002850: d5010049 00268a80
	v_add_co_u32 v72, s10, s26, v72                            // 000000002858: d7000a48 0202901a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002860: bf8701a3
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[76:77], v[78:79], v[8:15]// 000000002864: cc464008 1c229d4c
	s_wait_alu depctr_va_sdst(0)                               // 00000000286c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s10               // 000000002870: d5207c49 002a921b
	global_load_d16_u8 v72, v[72:73], off                      // 000000002878: ee07807c 00000048 00000048
	s_wait_loadcnt 0x0                                         // 000000002884: bfc00000
	v_cndmask_b16 v72.l, 0, v72.l, s9                          // 000000002888: d65d0048 00269080
	v_add_co_u32 v73, s9, v68, 1                               // 000000002890: d7000949 02010344
	s_wait_alu depctr_va_sdst(0)                               // 000000002898: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v69, s9                  // 00000000289c: d5207c4a 00268a80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000028a4: bf870113
	v_and_b16 v72.l, 0xff, v72.l                               // 0000000028a8: d7620048 020290ff 000000ff
	v_dual_cndmask_b32 v73, 0, v73 :: v_dual_cndmask_b32 v74, 0, v74// 0000000028b4: ca529280 494a9480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000028bc: bf870121
	v_add_co_u32 v73, s9, s26, v73                             // 0000000028c0: d7000949 0202921a
	s_wait_alu depctr_va_sdst(0)                               // 0000000028c8: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s27, v74, s9                // 0000000028cc: d5207c4a 0026941b
	global_load_d16_hi_u8 v72, v[73:74], off                   // 0000000028d4: ee08407c 00000048 00000049
	s_wait_loadcnt 0x0                                         // 0000000028e0: bfc00000
	v_cndmask_b16 v72.h, 0, v72.h, vcc_lo                      // 0000000028e4: d65d5048 01aa9080
	v_add_co_u32 v73, vcc_lo, v68, 2                           // 0000000028ec: d7006a49 02010544
	s_wait_alu depctr_va_vcc(0)                                // 0000000028f4: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, 0, v69, vcc_lo              // 0000000028f8: d5207c4a 01aa8a80
	s_and_b32 vcc_lo, s2, s3                                   // 000000002900: 8b6a0302
	v_lshlrev_b16 v72.h, 8, v72.h op_sel:[0,1,1]               // 000000002904: d7385048 02029088
	s_wait_alu depctr_sa_sdst(0)                               // 00000000290c: bf88ff9e
	v_dual_cndmask_b32 v73, 0, v73 :: v_dual_cndmask_b32 v74, 0, v74// 000000002910: ca529280 494a9480
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002918: bf870121
	v_add_co_u32 v73, s3, s26, v73                             // 00000000291c: d7000349 0202921a
	s_wait_alu depctr_va_sdst(0)                               // 000000002924: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s27, v74, s3                // 000000002928: d5207c4a 000e941b
	global_load_d16_u8 v73, v[73:74], off                      // 000000002930: ee07807c 00000049 00000049
	s_wait_loadcnt 0x0                                         // 00000000293c: bfc00000
	v_cndmask_b16 v73.l, 0, v73.l, vcc_lo                      // 000000002940: d65d0049 01aa9280
	v_add_co_u32 v74, vcc_lo, v68, 3                           // 000000002948: d7006a4a 02010744
	s_wait_alu depctr_va_vcc(0)                                // 000000002950: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v69, vcc_lo              // 000000002954: d5207c4b 01aa8a80
	s_and_b32 vcc_lo, s2, s4                                   // 00000000295c: 8b6a0402
	v_and_b16 v73.l, 0xff, v73.l                               // 000000002960: d7620049 020292ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000296c: bf88ff9e
	v_dual_cndmask_b32 v74, 0, v74 :: v_dual_cndmask_b32 v75, 0, v75// 000000002970: ca529480 4a4a9680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002978: bf870121
	v_add_co_u32 v74, s3, s26, v74                             // 00000000297c: d700034a 0202941a
	s_wait_alu depctr_va_sdst(0)                               // 000000002984: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s27, v75, s3                // 000000002988: d5207c4b 000e961b
	global_load_d16_hi_u8 v73, v[74:75], off                   // 000000002990: ee08407c 00000049 0000004a
	s_wait_loadcnt 0x0                                         // 00000000299c: bfc00000
	v_cndmask_b16 v73.h, 0, v73.h, vcc_lo                      // 0000000029a0: d65d5049 01aa9280
	v_add_co_u32 v74, vcc_lo, v68, 4                           // 0000000029a8: d7006a4a 02010944
	s_wait_alu depctr_va_vcc(0)                                // 0000000029b0: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v69, vcc_lo              // 0000000029b4: d5207c4b 01aa8a80
	s_and_b32 vcc_lo, s2, s5                                   // 0000000029bc: 8b6a0502
	v_lshlrev_b16 v73.h, 8, v73.h op_sel:[0,1,1]               // 0000000029c0: d7385049 02029288
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029c8: bf88ff9e
	v_dual_cndmask_b32 v74, 0, v74 :: v_dual_cndmask_b32 v75, 0, v75// 0000000029cc: ca529480 4a4a9680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000029d4: bf870121
	v_add_co_u32 v74, s3, s26, v74                             // 0000000029d8: d700034a 0202941a
	s_wait_alu depctr_va_sdst(0)                               // 0000000029e0: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s27, v75, s3                // 0000000029e4: d5207c4b 000e961b
	global_load_d16_u8 v74, v[74:75], off                      // 0000000029ec: ee07807c 0000004a 0000004a
	s_wait_loadcnt 0x0                                         // 0000000029f8: bfc00000
	v_cndmask_b16 v74.l, 0, v74.l, vcc_lo                      // 0000000029fc: d65d004a 01aa9480
	v_add_co_u32 v75, vcc_lo, v68, 5                           // 000000002a04: d7006a4b 02010b44
	s_wait_alu depctr_va_vcc(0)                                // 000000002a0c: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, 0, v69, vcc_lo              // 000000002a10: d5207c50 01aa8a80
	s_and_b32 vcc_lo, s2, s6                                   // 000000002a18: 8b6a0602
	v_and_b16 v74.l, 0xff, v74.l                               // 000000002a1c: d762004a 020294ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a28: bf88ff9e
	v_cndmask_b32_e32 v75, 0, v75, vcc_lo                      // 000000002a2c: 02969680
	v_cndmask_b32_e32 v81, 0, v80, vcc_lo                      // 000000002a30: 02a2a080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a34: bf870122
	v_add_co_u32 v80, s3, s26, v75                             // 000000002a38: d7000350 0202961a
	s_wait_alu depctr_va_sdst(0)                               // 000000002a40: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s27, v81, s3                // 000000002a44: d5207c51 000ea21b
	global_load_d16_hi_u8 v74, v[80:81], off                   // 000000002a4c: ee08407c 0000004a 00000050
	s_wait_loadcnt 0x0                                         // 000000002a58: bfc00000
	v_cndmask_b16 v74.h, 0, v74.h, vcc_lo                      // 000000002a5c: d65d504a 01aa9480
	v_add_co_u32 v75, vcc_lo, v68, 6                           // 000000002a64: d7006a4b 02010d44
	s_wait_alu depctr_va_vcc(0)                                // 000000002a6c: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, 0, v69, vcc_lo              // 000000002a70: d5207c50 01aa8a80
	s_and_b32 vcc_lo, s2, s7                                   // 000000002a78: 8b6a0702
	v_lshlrev_b16 v74.h, 8, v74.h op_sel:[0,1,1]               // 000000002a7c: d738504a 02029488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a84: bf88ff9e
	v_cndmask_b32_e32 v75, 0, v75, vcc_lo                      // 000000002a88: 02969680
	v_cndmask_b32_e32 v81, 0, v80, vcc_lo                      // 000000002a8c: 02a2a080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a90: bf870122
	v_add_co_u32 v80, s3, s26, v75                             // 000000002a94: d7000350 0202961a
	s_wait_alu depctr_va_sdst(0)                               // 000000002a9c: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s27, v81, s3                // 000000002aa0: d5207c51 000ea21b
	global_load_d16_u8 v75, v[80:81], off                      // 000000002aa8: ee07807c 0000004b 00000050
	s_wait_loadcnt 0x0                                         // 000000002ab4: bfc00000
	v_cndmask_b16 v75.l, 0, v75.l, vcc_lo                      // 000000002ab8: d65d004b 01aa9680
	v_add_co_u32 v80, vcc_lo, v68, 7                           // 000000002ac0: d7006a50 02010f44
	s_wait_alu depctr_va_vcc(0)                                // 000000002ac8: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v69, vcc_lo              // 000000002acc: d5207c51 01aa8a80
	s_and_b32 vcc_lo, s2, s8                                   // 000000002ad4: 8b6a0802
	v_and_b16 v75.l, 0xff, v75.l                               // 000000002ad8: d762004b 020296ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ae4: bf88ff9e
	v_dual_cndmask_b32 v80, 0, v80 :: v_dual_cndmask_b32 v81, 0, v81// 000000002ae8: ca52a080 5050a280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002af0: bf870121
	v_add_co_u32 v80, s3, s26, v80                             // 000000002af4: d7000350 0202a01a
	s_wait_alu depctr_va_sdst(0)                               // 000000002afc: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s27, v81, s3                // 000000002b00: d5207c51 000ea21b
	global_load_d16_hi_u8 v75, v[80:81], off                   // 000000002b08: ee08407c 0000004b 00000050
	s_wait_loadcnt 0x0                                         // 000000002b14: bfc00000
	v_cndmask_b16 v75.h, 0, v75.h, vcc_lo                      // 000000002b18: d65d504b 01aa9680
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002b20: bf870091
	v_lshlrev_b16 v75.h, 8, v75.h op_sel:[0,1,1]               // 000000002b24: d738504b 02029688
	v_or_b16 v75.h, v75.l, v75.h op_sel:[0,1,1]                // 000000002b2c: d763504b 0202974b
	v_or_b16 v75.l, v74.l, v74.h op_sel:[0,1,0]                // 000000002b34: d763104b 0202954a
	v_or_b16 v74.h, v73.l, v73.h op_sel:[0,1,1]                // 000000002b3c: d763504a 02029349
	v_or_b16 v74.l, v72.l, v72.h op_sel:[0,1,0]                // 000000002b44: d763104a 02029148
	v_add_co_u32 v72, vcc_lo, v30, 16                          // 000000002b4c: d7006a48 0201211e
	s_wait_alu depctr_va_vcc(0)                                // 000000002b54: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, 0, v31, vcc_lo              // 000000002b58: d5207c49 01aa3e80
	s_delay_alu instid0(valu_dep_3)                            // 000000002b60: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[76:77], v[74:75], v[0:7]// 000000002b64: cc464000 1c02954c
	v_add_co_u32 v74, vcc_lo, v70, 16                          // 000000002b6c: d7006a4a 02012146
	s_wait_alu depctr_va_vcc(0)                                // 000000002b74: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v71, vcc_lo              // 000000002b78: d5207c4b 01aa8e80
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[72:73]                // 000000002b80: 7ca89018
	s_and_b32 s3, s0, vcc_lo                                   // 000000002b84: 8b036a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b88: bf88ff9e
	v_cndmask_b32_e64 v72, 0, v74, s3                          // 000000002b8c: d5010048 000e9480
	v_cndmask_b32_e64 v73, 0, v75, s3                          // 000000002b94: d5010049 000e9680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b9c: bf870122
	v_add_co_u32 v72, s4, s28, v72                             // 000000002ba0: d7000448 0202901c
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba8: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s29, v73, s4                // 000000002bac: d5207c49 0012921d
	global_load_d16_u8 v72, v[72:73], off                      // 000000002bb4: ee07807c 00000048 00000048
	s_wait_loadcnt 0x0                                         // 000000002bc0: bfc00000
	v_cndmask_b16 v72.l, 0, v72.l, s3                          // 000000002bc4: d65d0048 000e9080
	v_add_co_u32 v75, s3, v70, 17                              // 000000002bcc: d700034b 02012346
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd4: bf88f19f
	v_add_co_ci_u32_e64 v76, null, 0, v71, s3                  // 000000002bd8: d5207c4c 000e8e80
	v_add_co_u32 v73, s3, v30, 17                              // 000000002be0: d7000349 0201231e
	s_wait_alu depctr_va_sdst(0)                               // 000000002be8: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v31, s3                  // 000000002bec: d5207c4a 000e3e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002bf4: bf870151
	v_cmp_gt_i64_e64 s3, s[24:25], v[73:74]                    // 000000002bf8: d4540003 02029218
	s_and_b32 s4, s0, s3                                       // 000000002c00: 8b040300
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c04: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v75, s4                          // 000000002c08: d5010049 00129680
	v_cndmask_b32_e64 v74, 0, v76, s4                          // 000000002c10: d501004a 00129880
	v_add_co_u32 v73, s5, s28, v73                             // 000000002c18: d7000549 0202921c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c20: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002c24: bf870002
	v_add_co_ci_u32_e64 v74, null, s29, v74, s5                // 000000002c28: d5207c4a 0016941d
	global_load_d16_hi_u8 v72, v[73:74], off                   // 000000002c30: ee08407c 00000048 00000049
	s_wait_loadcnt 0x0                                         // 000000002c3c: bfc00000
	v_cndmask_b16 v72.h, 0, v72.h, s4                          // 000000002c40: d65d5048 00129080
	v_add_co_u32 v75, s4, v70, 18                              // 000000002c48: d700044b 02012546
	s_wait_alu depctr_va_sdst(0)                               // 000000002c50: bf88f19f
	v_add_co_ci_u32_e64 v76, null, 0, v71, s4                  // 000000002c54: d5207c4c 00128e80
	v_add_co_u32 v73, s4, v30, 18                              // 000000002c5c: d7000449 0201251e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c64: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v31, s4                  // 000000002c68: d5207c4a 00123e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002c70: bf870151
	v_cmp_gt_i64_e64 s4, s[24:25], v[73:74]                    // 000000002c74: d4540004 02029218
	s_and_b32 s5, s0, s4                                       // 000000002c7c: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c80: bf88ff9e
	v_cndmask_b32_e64 v73, 0, v75, s5                          // 000000002c84: d5010049 00169680
	v_cndmask_b32_e64 v74, 0, v76, s5                          // 000000002c8c: d501004a 00169880
	v_add_co_u32 v73, s6, s28, v73                             // 000000002c94: d7000649 0202921c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c9c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002ca0: bf870002
	v_add_co_ci_u32_e64 v74, null, s29, v74, s6                // 000000002ca4: d5207c4a 001a941d
	global_load_d16_u8 v73, v[73:74], off                      // 000000002cac: ee07807c 00000049 00000049
	s_wait_loadcnt 0x0                                         // 000000002cb8: bfc00000
	v_cndmask_b16 v73.l, 0, v73.l, s5                          // 000000002cbc: d65d0049 00169280
	v_add_co_u32 v76, s5, v70, 19                              // 000000002cc4: d700054c 02012746
	s_wait_alu depctr_va_sdst(0)                               // 000000002ccc: bf88f19f
	v_add_co_ci_u32_e64 v77, null, 0, v71, s5                  // 000000002cd0: d5207c4d 00168e80
	v_add_co_u32 v74, s5, v30, 19                              // 000000002cd8: d700054a 0201271e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ce0: bf88f19f
	v_add_co_ci_u32_e64 v75, null, 0, v31, s5                  // 000000002ce4: d5207c4b 00163e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002cec: bf870151
	v_cmp_gt_i64_e64 s5, s[24:25], v[74:75]                    // 000000002cf0: d4540005 02029418
	s_and_b32 s6, s0, s5                                       // 000000002cf8: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cfc: bf88ff9e
	v_cndmask_b32_e64 v74, 0, v76, s6                          // 000000002d00: d501004a 001a9880
	v_cndmask_b32_e64 v75, 0, v77, s6                          // 000000002d08: d501004b 001a9a80
	v_add_co_u32 v74, s7, s28, v74                             // 000000002d10: d700074a 0202941c
	s_wait_alu depctr_va_sdst(0)                               // 000000002d18: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002d1c: bf870002
	v_add_co_ci_u32_e64 v75, null, s29, v75, s7                // 000000002d20: d5207c4b 001e961d
	global_load_d16_hi_u8 v73, v[74:75], off                   // 000000002d28: ee08407c 00000049 0000004a
	s_wait_loadcnt 0x0                                         // 000000002d34: bfc00000
	v_cndmask_b16 v73.h, 0, v73.h, s6                          // 000000002d38: d65d5049 001a9280
	v_add_co_u32 v76, s6, v70, 20                              // 000000002d40: d700064c 02012946
	s_wait_alu depctr_va_sdst(0)                               // 000000002d48: bf88f19f
	v_add_co_ci_u32_e64 v77, null, 0, v71, s6                  // 000000002d4c: d5207c4d 001a8e80
	v_add_co_u32 v74, s6, v30, 20                              // 000000002d54: d700064a 0201291e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d5c: bf88f19f
	v_add_co_ci_u32_e64 v75, null, 0, v31, s6                  // 000000002d60: d5207c4b 001a3e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002d68: bf870151
	v_cmp_gt_i64_e64 s6, s[24:25], v[74:75]                    // 000000002d6c: d4540006 02029418
	s_and_b32 s7, s0, s6                                       // 000000002d74: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d78: bf88ff9e
	v_cndmask_b32_e64 v74, 0, v76, s7                          // 000000002d7c: d501004a 001e9880
	v_cndmask_b32_e64 v75, 0, v77, s7                          // 000000002d84: d501004b 001e9a80
	v_add_co_u32 v74, s8, s28, v74                             // 000000002d8c: d700084a 0202941c
	s_wait_alu depctr_va_sdst(0)                               // 000000002d94: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002d98: bf870002
	v_add_co_ci_u32_e64 v75, null, s29, v75, s8                // 000000002d9c: d5207c4b 0022961d
	global_load_d16_u8 v74, v[74:75], off                      // 000000002da4: ee07807c 0000004a 0000004a
	s_wait_loadcnt 0x0                                         // 000000002db0: bfc00000
	v_cndmask_b16 v74.l, 0, v74.l, s7                          // 000000002db4: d65d004a 001e9480
	v_add_co_u32 v77, s7, v70, 21                              // 000000002dbc: d700074d 02012b46
	s_wait_alu depctr_va_sdst(0)                               // 000000002dc4: bf88f19f
	v_add_co_ci_u32_e64 v78, null, 0, v71, s7                  // 000000002dc8: d5207c4e 001e8e80
	v_add_co_u32 v75, s7, v30, 21                              // 000000002dd0: d700074b 02012b1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002dd8: bf88f19f
	v_add_co_ci_u32_e64 v76, null, 0, v31, s7                  // 000000002ddc: d5207c4c 001e3e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002de4: bf870151
	v_cmp_gt_i64_e64 s7, s[24:25], v[75:76]                    // 000000002de8: d4540007 02029618
	s_and_b32 s8, s0, s7                                       // 000000002df0: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002df4: bf88ff9e
	v_cndmask_b32_e64 v75, 0, v77, s8                          // 000000002df8: d501004b 00229a80
	v_cndmask_b32_e64 v76, 0, v78, s8                          // 000000002e00: d501004c 00229c80
	v_add_co_u32 v75, s9, s28, v75                             // 000000002e08: d700094b 0202961c
	s_wait_alu depctr_va_sdst(0)                               // 000000002e10: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002e14: bf870002
	v_add_co_ci_u32_e64 v76, null, s29, v76, s9                // 000000002e18: d5207c4c 0026981d
	global_load_d16_hi_u8 v74, v[75:76], off                   // 000000002e20: ee08407c 0000004a 0000004b
	s_wait_loadcnt 0x0                                         // 000000002e2c: bfc00000
	v_cndmask_b16 v74.h, 0, v74.h, s8                          // 000000002e30: d65d504a 00229480
	v_add_co_u32 v77, s8, v70, 22                              // 000000002e38: d700084d 02012d46
	s_wait_alu depctr_va_sdst(0)                               // 000000002e40: bf88f19f
	v_add_co_ci_u32_e64 v78, null, 0, v71, s8                  // 000000002e44: d5207c4e 00228e80
	v_add_co_u32 v75, s8, v30, 22                              // 000000002e4c: d700084b 02012d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e54: bf88f19f
	v_add_co_ci_u32_e64 v76, null, 0, v31, s8                  // 000000002e58: d5207c4c 00223e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002e60: bf870151
	v_cmp_gt_i64_e64 s8, s[24:25], v[75:76]                    // 000000002e64: d4540008 02029618
	s_and_b32 s9, s0, s8                                       // 000000002e6c: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e70: bf88ff9e
	v_cndmask_b32_e64 v75, 0, v77, s9                          // 000000002e74: d501004b 00269a80
	v_cndmask_b32_e64 v76, 0, v78, s9                          // 000000002e7c: d501004c 00269c80
	v_add_co_u32 v75, s10, s28, v75                            // 000000002e84: d7000a4b 0202961c
	s_wait_alu depctr_va_sdst(0)                               // 000000002e8c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002e90: bf870002
	v_add_co_ci_u32_e64 v76, null, s29, v76, s10               // 000000002e94: d5207c4c 002a981d
	global_load_d16_u8 v75, v[75:76], off                      // 000000002e9c: ee07807c 0000004b 0000004b
	s_wait_loadcnt 0x0                                         // 000000002ea8: bfc00000
	v_cndmask_b16 v75.l, 0, v75.l, s9                          // 000000002eac: d65d004b 00269680
	v_add_co_u32 v70, s9, v70, 23                              // 000000002eb4: d7000946 02012f46
	s_wait_alu depctr_va_sdst(0)                               // 000000002ebc: bf88f19f
	v_add_co_ci_u32_e64 v71, null, 0, v71, s9                  // 000000002ec0: d5207c47 00268e80
	v_add_co_u32 v30, s9, v30, 23                              // 000000002ec8: d700091e 02012f1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed0: bf88f19f
	v_add_co_ci_u32_e64 v31, null, 0, v31, s9                  // 000000002ed4: d5207c1f 00263e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002edc: bf870151
	v_cmp_gt_i64_e64 s9, s[24:25], v[30:31]                    // 000000002ee0: d4540009 02023c18
	s_and_b32 s10, s0, s9                                      // 000000002ee8: 8b0a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eec: bf88ff9e
	v_cndmask_b32_e64 v30, 0, v70, s10                         // 000000002ef0: d501001e 002a8c80
	v_cndmask_b32_e64 v31, 0, v71, s10                         // 000000002ef8: d501001f 002a8e80
	v_add_co_u32 v30, s11, s28, v30                            // 000000002f00: d7000b1e 02023c1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f08: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_1)// 000000002f0c: bf8700d2
	v_add_co_ci_u32_e64 v31, null, s29, v31, s11               // 000000002f10: d5207c1f 002e3e1d
	global_load_d16_u8 v30, v[30:31], off                      // 000000002f18: ee07807c 0000001e 0000001e
	s_wait_loadcnt 0x0                                         // 000000002f24: bfc00000
	v_and_b16 v30.h, 0xff, v75.l op_sel:[0,0,1]                // 000000002f28: d762401e 020296ff 000000ff
	v_cndmask_b16 v30.l, 0, v30.l, s10                         // 000000002f34: d65d001e 002a3c80
	v_lshlrev_b16 v30.l, 8, v30.l                              // 000000002f3c: d738001e 02023c88
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000002f44: bf8700b1
	v_or_b16 v75.h, v30.h, v30.l op_sel:[1,0,1]                // 000000002f48: d763484b 02023d1e
	v_lshlrev_b16 v30.l, 8, v74.h op_sel:[0,1,0]               // 000000002f50: d738101e 02029488
	v_and_b16 v30.h, 0xff, v74.l op_sel:[0,0,1]                // 000000002f58: d762401e 020294ff 000000ff
	v_or_b16 v75.l, v30.h, v30.l op_sel:[1,0,0]                // 000000002f64: d763084b 02023d1e
	v_lshlrev_b16 v30.l, 8, v73.h op_sel:[0,1,0]               // 000000002f6c: d738101e 02029288
	v_and_b16 v30.h, 0xff, v73.l op_sel:[0,0,1]                // 000000002f74: d762401e 020292ff 000000ff
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000002f80: bf8700b1
	v_or_b16 v74.h, v30.h, v30.l op_sel:[1,0,1]                // 000000002f84: d763484a 02023d1e
	v_lshlrev_b16 v30.l, 8, v72.h op_sel:[0,1,0]               // 000000002f8c: d738101e 02029088
	v_and_b16 v30.h, 0xff, v72.l op_sel:[0,0,1]                // 000000002f94: d762401e 020290ff 000000ff
	v_or_b16 v74.l, v30.h, v30.l op_sel:[1,0,0]                // 000000002fa0: d763084a 02023d1e
	v_add_co_u32 v30, s10, v66, 16                             // 000000002fa8: d7000a1e 02012142
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb0: bf88f19f
	v_add_co_ci_u32_e64 v31, null, 0, v67, s10                 // 000000002fb4: d5207c1f 002a8680
	s_and_b32 s10, s1, vcc_lo                                  // 000000002fbc: 8b0a6a01
	s_and_b32 vcc_lo, s2, vcc_lo                               // 000000002fc0: 8b6a6a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fc4: bf88ff9e
	v_cndmask_b32_e64 v30, 0, v30, s10                         // 000000002fc8: d501001e 002a3c80
	v_cndmask_b32_e64 v31, 0, v31, s10                         // 000000002fd0: d501001f 002a3e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002fd8: bf870122
	v_add_co_u32 v30, s11, s26, v30                            // 000000002fdc: d7000b1e 02023c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002fe4: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s27, v31, s11               // 000000002fe8: d5207c1f 002e3e1b
	global_load_d16_u8 v30, v[30:31], off                      // 000000002ff0: ee07807c 0000001e 0000001e
	s_wait_loadcnt 0x0                                         // 000000002ffc: bfc00000
	v_cndmask_b16 v30.l, 0, v30.l, s10                         // 000000003000: d65d001e 002a3c80
	v_add_co_u32 v31, s10, v66, 17                             // 000000003008: d7000a1f 02012342
	s_wait_alu depctr_va_sdst(0)                               // 000000003010: bf88f19f
	v_add_co_ci_u32_e64 v70, null, 0, v67, s10                 // 000000003014: d5207c46 002a8680
	s_and_b32 s10, s1, s3                                      // 00000000301c: 8b0a0301
	v_and_b16 v30.l, 0xff, v30.l                               // 000000003020: d762001e 02023cff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000302c: bf88ff9e
	v_cndmask_b32_e64 v31, 0, v31, s10                         // 000000003030: d501001f 002a3e80
	v_cndmask_b32_e64 v71, 0, v70, s10                         // 000000003038: d5010047 002a8c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003040: bf870122
	v_add_co_u32 v70, s11, s26, v31                            // 000000003044: d7000b46 02023e1a
	s_wait_alu depctr_va_sdst(0)                               // 00000000304c: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s27, v71, s11               // 000000003050: d5207c47 002e8e1b
	global_load_d16_hi_u8 v30, v[70:71], off                   // 000000003058: ee08407c 0000001e 00000046
	s_wait_loadcnt 0x0                                         // 000000003064: bfc00000
	v_cndmask_b16 v30.h, 0, v30.h, s10                         // 000000003068: d65d501e 002a3c80
	v_add_co_u32 v31, s10, v66, 18                             // 000000003070: d7000a1f 02012542
	s_wait_alu depctr_va_sdst(0)                               // 000000003078: bf88f19f
	v_add_co_ci_u32_e64 v70, null, 0, v67, s10                 // 00000000307c: d5207c46 002a8680
	s_and_b32 s10, s1, s4                                      // 000000003084: 8b0a0401
	v_lshlrev_b16 v30.h, 8, v30.h op_sel:[0,1,1]               // 000000003088: d738501e 02023c88
	s_wait_alu depctr_sa_sdst(0)                               // 000000003090: bf88ff9e
	v_cndmask_b32_e64 v31, 0, v31, s10                         // 000000003094: d501001f 002a3e80
	v_cndmask_b32_e64 v71, 0, v70, s10                         // 00000000309c: d5010047 002a8c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000030a4: bf870122
	v_add_co_u32 v70, s11, s26, v31                            // 0000000030a8: d7000b46 02023e1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000030b0: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s27, v71, s11               // 0000000030b4: d5207c47 002e8e1b
	global_load_d16_u8 v31, v[70:71], off                      // 0000000030bc: ee07807c 0000001f 00000046
	s_wait_loadcnt 0x0                                         // 0000000030c8: bfc00000
	v_cndmask_b16 v31.l, 0, v31.l, s10                         // 0000000030cc: d65d001f 002a3e80
	v_add_co_u32 v70, s10, v66, 19                             // 0000000030d4: d7000a46 02012742
	s_wait_alu depctr_va_sdst(0)                               // 0000000030dc: bf88f19f
	v_add_co_ci_u32_e64 v71, null, 0, v67, s10                 // 0000000030e0: d5207c47 002a8680
	s_and_b32 s10, s1, s5                                      // 0000000030e8: 8b0a0501
	v_and_b16 v31.l, 0xff, v31.l                               // 0000000030ec: d762001f 02023eff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030f8: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 0000000030fc: d5010046 002a8c80
	v_cndmask_b32_e64 v71, 0, v71, s10                         // 000000003104: d5010047 002a8e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000310c: bf870122
	v_add_co_u32 v70, s11, s26, v70                            // 000000003110: d7000b46 02028c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000003118: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s27, v71, s11               // 00000000311c: d5207c47 002e8e1b
	global_load_d16_hi_u8 v31, v[70:71], off                   // 000000003124: ee08407c 0000001f 00000046
	s_wait_loadcnt 0x0                                         // 000000003130: bfc00000
	v_cndmask_b16 v31.h, 0, v31.h, s10                         // 000000003134: d65d501f 002a3e80
	v_add_co_u32 v70, s10, v66, 20                             // 00000000313c: d7000a46 02012942
	s_wait_alu depctr_va_sdst(0)                               // 000000003144: bf88f19f
	v_add_co_ci_u32_e64 v71, null, 0, v67, s10                 // 000000003148: d5207c47 002a8680
	s_and_b32 s10, s1, s6                                      // 000000003150: 8b0a0601
	v_lshlrev_b16 v31.h, 8, v31.h op_sel:[0,1,1]               // 000000003154: d738501f 02023e88
	s_wait_alu depctr_sa_sdst(0)                               // 00000000315c: bf88ff9e
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000003160: d5010046 002a8c80
	v_cndmask_b32_e64 v71, 0, v71, s10                         // 000000003168: d5010047 002a8e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003170: bf870122
	v_add_co_u32 v70, s11, s26, v70                            // 000000003174: d7000b46 02028c1a
	s_wait_alu depctr_va_sdst(0)                               // 00000000317c: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s27, v71, s11               // 000000003180: d5207c47 002e8e1b
	global_load_d16_u8 v70, v[70:71], off                      // 000000003188: ee07807c 00000046 00000046
	s_wait_loadcnt 0x0                                         // 000000003194: bfc00000
	v_cndmask_b16 v70.l, 0, v70.l, s10                         // 000000003198: d65d0046 002a8c80
	v_add_co_u32 v71, s10, v66, 21                             // 0000000031a0: d7000a47 02012b42
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v67, s10                 // 0000000031ac: d5207c48 002a8680
	s_and_b32 s10, s1, s7                                      // 0000000031b4: 8b0a0701
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031b8: bf88ff9e
	v_cndmask_b32_e64 v71, 0, v71, s10                         // 0000000031bc: d5010047 002a8e80
	v_cndmask_b32_e64 v72, 0, v72, s10                         // 0000000031c4: d5010048 002a9080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031cc: bf870122
	v_add_co_u32 v71, s11, s26, v71                            // 0000000031d0: d7000b47 02028e1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000031d8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s27, v72, s11               // 0000000031dc: d5207c48 002e901b
	global_load_d16_hi_u8 v70, v[71:72], off                   // 0000000031e4: ee08407c 00000046 00000047
	s_wait_loadcnt 0x0                                         // 0000000031f0: bfc00000
	v_cndmask_b16 v70.h, 0, v70.h, s10                         // 0000000031f4: d65d5046 002a8c80
	v_add_co_u32 v71, s10, v66, 22                             // 0000000031fc: d7000a47 02012d42
	s_wait_alu depctr_va_sdst(0)                               // 000000003204: bf88f19f
	v_add_co_ci_u32_e64 v72, null, 0, v67, s10                 // 000000003208: d5207c48 002a8680
	s_and_b32 s10, s1, s8                                      // 000000003210: 8b0a0801
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	v_cndmask_b32_e64 v71, 0, v71, s10                         // 000000003218: d5010047 002a8e80
	v_cndmask_b32_e64 v72, 0, v72, s10                         // 000000003220: d5010048 002a9080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003228: bf870122
	v_add_co_u32 v71, s11, s26, v71                            // 00000000322c: d7000b47 02028e1a
	s_wait_alu depctr_va_sdst(0)                               // 000000003234: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s27, v72, s11               // 000000003238: d5207c48 002e901b
	global_load_d16_u8 v71, v[71:72], off                      // 000000003240: ee07807c 00000047 00000047
	s_wait_loadcnt 0x0                                         // 00000000324c: bfc00000
	v_cndmask_b16 v71.l, 0, v71.l, s10                         // 000000003250: d65d0047 002a8e80
	v_add_co_u32 v66, s10, v66, 23                             // 000000003258: d7000a42 02012f42
	s_wait_alu depctr_va_sdst(0)                               // 000000003260: bf88f19f
	v_add_co_ci_u32_e64 v67, null, 0, v67, s10                 // 000000003264: d5207c43 002a8680
	s_and_b32 s10, s1, s9                                      // 00000000326c: 8b0a0901
	s_wait_alu depctr_sa_sdst(0)                               // 000000003270: bf88ff9e
	v_cndmask_b32_e64 v66, 0, v66, s10                         // 000000003274: d5010042 002a8480
	v_cndmask_b32_e64 v67, 0, v67, s10                         // 00000000327c: d5010043 002a8680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003284: bf870122
	v_add_co_u32 v66, s11, s26, v66                            // 000000003288: d7000b42 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 000000003290: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s11               // 000000003294: d5207c43 002e861b
	global_load_d16_u8 v66, v[66:67], off                      // 00000000329c: ee07807c 00000042 00000042
	s_wait_loadcnt 0x0                                         // 0000000032a8: bfc00000
	v_and_b16 v66.h, 0xff, v71.l op_sel:[0,0,1]                // 0000000032ac: d7624042 02028eff 000000ff
	v_cndmask_b16 v66.l, 0, v66.l, s10                         // 0000000032b8: d65d0042 002a8480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000032c0: bf870091
	v_lshlrev_b16 v66.l, 8, v66.l                              // 0000000032c4: d7380042 02028488
	v_or_b16 v71.h, v66.h, v66.l op_sel:[1,0,1]                // 0000000032cc: d7634847 02028542
	v_and_b16 v66.h, 0xff, v70.l op_sel:[0,0,1]                // 0000000032d4: d7624042 02028cff 000000ff
	v_or_b16 v70.l, v30.l, v30.h op_sel:[0,1,0]                // 0000000032e0: d7631046 02023d1e
	v_add_co_u32 v30, s10, v68, 16                             // 0000000032e8: d7000a1e 02012144
	v_lshlrev_b16 v66.l, 8, v70.h op_sel:[0,1,0]               // 0000000032f0: d7381042 02028c88
	v_or_b16 v70.h, v31.l, v31.h op_sel:[0,1,1]                // 0000000032f8: d7635046 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000003300: bf88f19f
	v_add_co_ci_u32_e64 v31, null, 0, v69, s10                 // 000000003304: d5207c1f 002a8a80
	v_cndmask_b32_e32 v30, 0, v30, vcc_lo                      // 00000000330c: 023c3c80
	v_or_b16 v71.l, v66.h, v66.l op_sel:[1,0,0]                // 000000003310: d7630847 02028542
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003318: bf870193
	v_cndmask_b32_e32 v31, 0, v31, vcc_lo                      // 00000000331c: 023e3e80
	v_add_co_u32 v30, s10, s26, v30                            // 000000003320: d7000a1e 02023c1a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003328: bf8701a3
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[74:75], v[70:71], v[8:15]// 00000000332c: cc464008 1c228d4a
	s_wait_alu depctr_va_sdst(0)                               // 000000003334: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s27, v31, s10               // 000000003338: d5207c1f 002a3e1b
	global_load_d16_u8 v30, v[30:31], off                      // 000000003340: ee07807c 0000001e 0000001e
	s_wait_loadcnt 0x0                                         // 00000000334c: bfc00000
	v_cndmask_b16 v30.l, 0, v30.l, vcc_lo                      // 000000003350: d65d001e 01aa3c80
	v_add_co_u32 v31, vcc_lo, v68, 17                          // 000000003358: d7006a1f 02012344
	s_wait_alu depctr_va_vcc(0)                                // 000000003360: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, 0, v69, vcc_lo              // 000000003364: d5207c42 01aa8a80
	s_and_b32 vcc_lo, s2, s3                                   // 00000000336c: 8b6a0302
	v_and_b16 v30.l, 0xff, v30.l                               // 000000003370: d762001e 02023cff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000337c: bf88ff9e
	v_cndmask_b32_e32 v31, 0, v31, vcc_lo                      // 000000003380: 023e3e80
	v_cndmask_b32_e32 v67, 0, v66, vcc_lo                      // 000000003384: 02868480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003388: bf870122
	v_add_co_u32 v66, s3, s26, v31                             // 00000000338c: d7000342 02023e1a
	s_wait_alu depctr_va_sdst(0)                               // 000000003394: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s3                // 000000003398: d5207c43 000e861b
	global_load_d16_hi_u8 v30, v[66:67], off                   // 0000000033a0: ee08407c 0000001e 00000042
	s_wait_loadcnt 0x0                                         // 0000000033ac: bfc00000
	v_cndmask_b16 v30.h, 0, v30.h, vcc_lo                      // 0000000033b0: d65d501e 01aa3c80
	v_add_co_u32 v31, vcc_lo, v68, 18                          // 0000000033b8: d7006a1f 02012544
	s_wait_alu depctr_va_vcc(0)                                // 0000000033c0: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, 0, v69, vcc_lo              // 0000000033c4: d5207c42 01aa8a80
	s_and_b32 vcc_lo, s2, s4                                   // 0000000033cc: 8b6a0402
	v_lshlrev_b16 v30.h, 8, v30.h op_sel:[0,1,1]               // 0000000033d0: d738501e 02023c88
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d8: bf88ff9e
	v_cndmask_b32_e32 v31, 0, v31, vcc_lo                      // 0000000033dc: 023e3e80
	v_cndmask_b32_e32 v67, 0, v66, vcc_lo                      // 0000000033e0: 02868480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000033e4: bf870122
	v_add_co_u32 v66, s3, s26, v31                             // 0000000033e8: d7000342 02023e1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f0: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s3                // 0000000033f4: d5207c43 000e861b
	global_load_d16_u8 v31, v[66:67], off                      // 0000000033fc: ee07807c 0000001f 00000042
	s_wait_loadcnt 0x0                                         // 000000003408: bfc00000
	v_cndmask_b16 v31.l, 0, v31.l, vcc_lo                      // 00000000340c: d65d001f 01aa3e80
	v_add_co_u32 v66, vcc_lo, v68, 19                          // 000000003414: d7006a42 02012744
	s_wait_alu depctr_va_vcc(0)                                // 00000000341c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, 0, v69, vcc_lo              // 000000003420: d5207c43 01aa8a80
	s_and_b32 vcc_lo, s2, s5                                   // 000000003428: 8b6a0502
	v_and_b16 v31.l, 0xff, v31.l                               // 00000000342c: d762001f 02023eff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003438: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v66 :: v_dual_cndmask_b32 v67, 0, v67// 00000000343c: ca528480 42428680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003444: bf870121
	v_add_co_u32 v66, s3, s26, v66                             // 000000003448: d7000342 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 000000003450: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s3                // 000000003454: d5207c43 000e861b
	global_load_d16_hi_u8 v31, v[66:67], off                   // 00000000345c: ee08407c 0000001f 00000042
	s_wait_loadcnt 0x0                                         // 000000003468: bfc00000
	v_cndmask_b16 v31.h, 0, v31.h, vcc_lo                      // 00000000346c: d65d501f 01aa3e80
	v_add_co_u32 v66, vcc_lo, v68, 20                          // 000000003474: d7006a42 02012944
	s_wait_alu depctr_va_vcc(0)                                // 00000000347c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, 0, v69, vcc_lo              // 000000003480: d5207c43 01aa8a80
	s_and_b32 vcc_lo, s2, s6                                   // 000000003488: 8b6a0602
	v_lshlrev_b16 v31.h, 8, v31.h op_sel:[0,1,1]               // 00000000348c: d738501f 02023e88
	s_wait_alu depctr_sa_sdst(0)                               // 000000003494: bf88ff9e
	v_dual_cndmask_b32 v66, 0, v66 :: v_dual_cndmask_b32 v67, 0, v67// 000000003498: ca528480 42428680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000034a0: bf870121
	v_add_co_u32 v66, s3, s26, v66                             // 0000000034a4: d7000342 0202841a
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ac: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s27, v67, s3                // 0000000034b0: d5207c43 000e861b
	global_load_d16_u8 v66, v[66:67], off                      // 0000000034b8: ee07807c 00000042 00000042
	s_wait_loadcnt 0x0                                         // 0000000034c4: bfc00000
	v_cndmask_b16 v66.l, 0, v66.l, vcc_lo                      // 0000000034c8: d65d0042 01aa8480
	v_add_co_u32 v67, vcc_lo, v68, 21                          // 0000000034d0: d7006a43 02012b44
	s_wait_alu depctr_va_vcc(0)                                // 0000000034d8: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, 0, v69, vcc_lo              // 0000000034dc: d5207c48 01aa8a80
	s_and_b32 vcc_lo, s2, s7                                   // 0000000034e4: 8b6a0702
	v_and_b16 v66.l, 0xff, v66.l                               // 0000000034e8: d7620042 020284ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f4: bf88ff9e
	v_cndmask_b32_e32 v67, 0, v67, vcc_lo                      // 0000000034f8: 02868680
	v_cndmask_b32_e32 v73, 0, v72, vcc_lo                      // 0000000034fc: 02929080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003500: bf870122
	v_add_co_u32 v72, s3, s26, v67                             // 000000003504: d7000348 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 00000000350c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s3                // 000000003510: d5207c49 000e921b
	global_load_d16_hi_u8 v66, v[72:73], off                   // 000000003518: ee08407c 00000042 00000048
	s_wait_loadcnt 0x0                                         // 000000003524: bfc00000
	v_cndmask_b16 v66.h, 0, v66.h, vcc_lo                      // 000000003528: d65d5042 01aa8480
	v_add_co_u32 v67, vcc_lo, v68, 22                          // 000000003530: d7006a43 02012d44
	s_wait_alu depctr_va_vcc(0)                                // 000000003538: bf88ff9d
	v_add_co_ci_u32_e64 v72, null, 0, v69, vcc_lo              // 00000000353c: d5207c48 01aa8a80
	s_and_b32 vcc_lo, s2, s8                                   // 000000003544: 8b6a0802
	v_lshlrev_b16 v66.h, 8, v66.h op_sel:[0,1,1]               // 000000003548: d7385042 02028488
	s_wait_alu depctr_sa_sdst(0)                               // 000000003550: bf88ff9e
	v_cndmask_b32_e32 v67, 0, v67, vcc_lo                      // 000000003554: 02868680
	v_cndmask_b32_e32 v73, 0, v72, vcc_lo                      // 000000003558: 02929080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000355c: bf870122
	v_add_co_u32 v72, s3, s26, v67                             // 000000003560: d7000348 0202861a
	s_wait_alu depctr_va_sdst(0)                               // 000000003568: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s27, v73, s3                // 00000000356c: d5207c49 000e921b
	global_load_d16_u8 v67, v[72:73], off                      // 000000003574: ee07807c 00000043 00000048
	s_wait_loadcnt 0x0                                         // 000000003580: bfc00000
	v_cndmask_b16 v67.l, 0, v67.l, vcc_lo                      // 000000003584: d65d0043 01aa8680
	v_add_co_u32 v68, vcc_lo, v68, 23                          // 00000000358c: d7006a44 02012f44
	s_wait_alu depctr_va_vcc(0)                                // 000000003594: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v69, vcc_lo              // 000000003598: d5207c45 01aa8a80
	s_and_b32 vcc_lo, s2, s9                                   // 0000000035a0: 8b6a0902
	v_and_b16 v67.l, 0xff, v67.l                               // 0000000035a4: d7620043 020286ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035b0: bf88ff9e
	v_dual_cndmask_b32 v68, 0, v68 :: v_dual_cndmask_b32 v69, 0, v69// 0000000035b4: ca528880 44448a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000035bc: bf870121
	v_add_co_u32 v68, s3, s26, v68                             // 0000000035c0: d7000344 0202881a
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	v_add_co_ci_u32_e64 v69, null, s27, v69, s3                // 0000000035cc: d5207c45 000e8a1b
	v_cmp_lt_u64_e64 s3, s[44:45], s[40:41]                    // 0000000035d4: d4590003 0200502c
	global_load_d16_hi_u8 v67, v[68:69], off                   // 0000000035dc: ee08407c 00000043 00000044
	s_wait_loadcnt 0x0                                         // 0000000035e8: bfc00000
	v_cndmask_b16 v67.h, 0, v67.h, vcc_lo                      // 0000000035ec: d65d5043 01aa8680
	s_and_b32 vcc_lo, exec_lo, s3                              // 0000000035f4: 8b6a037e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035f8: bf870091
	v_lshlrev_b16 v67.h, 8, v67.h op_sel:[0,1,1]               // 0000000035fc: d7385043 02028688
	v_or_b16 v67.h, v67.l, v67.h op_sel:[0,1,1]                // 000000003604: d7635043 02028743
	v_or_b16 v67.l, v66.l, v66.h op_sel:[0,1,0]                // 00000000360c: d7631043 02028542
	v_or_b16 v66.h, v31.l, v31.h op_sel:[0,1,1]                // 000000003614: d7635042 02023f1f
	v_or_b16 v66.l, v30.l, v30.h op_sel:[0,1,0]                // 00000000361c: d7631042 02023d1e
	s_delay_alu instid0(valu_dep_1)                            // 000000003624: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[74:75], v[66:67], v[0:7]// 000000003628: cc464000 1c02854a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003630: bf88ff9e
	s_cbranch_vccnz 64144                                      // 000000003634: bfa4fa90 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x578>
	s_lshr_b64 s[4:5], s[42:43], 5                             // 000000003638: 8584852a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000363c: bf88ff9e
	v_add_co_u32 v30, vcc_lo, v48, s4                          // 000000003640: d7006a1e 02000930
	s_wait_alu depctr_va_vcc(0)                                // 000000003648: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, s5, v49, vcc_lo             // 00000000364c: d5207c1f 01aa6205
	v_add_co_u32 v66, vcc_lo, v50, s4                          // 000000003654: d7006a42 02000932
	s_wait_alu depctr_va_vcc(0)                                // 00000000365c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s5, v51, vcc_lo             // 000000003660: d5207c43 01aa6605
	v_add_co_u32 v68, vcc_lo, v53, s4                          // 000000003668: d7006a44 02000935
	s_wait_alu depctr_va_vcc(0)                                // 000000003670: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s5, v54, vcc_lo             // 000000003674: d5207c45 01aa6c05
	v_add_co_u32 v70, vcc_lo, v56, s4                          // 00000000367c: d7006a46 02000938
	s_wait_alu depctr_va_vcc(0)                                // 000000003684: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, s5, v57, vcc_lo             // 000000003688: d5207c47 01aa7205
	v_add_co_u32 v72, vcc_lo, v58, s4                          // 000000003690: d7006a48 0200093a
	s_wait_alu depctr_va_vcc(0)                                // 000000003698: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, s5, v59, vcc_lo             // 00000000369c: d5207c49 01aa7605
	v_add_co_u32 v74, vcc_lo, v60, s4                          // 0000000036a4: d7006a4a 0200093c
	s_wait_alu depctr_va_vcc(0)                                // 0000000036ac: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, s5, v61, vcc_lo             // 0000000036b0: d5207c4b 01aa7a05
	v_add_co_u32 v76, vcc_lo, v62, s4                          // 0000000036b8: d7006a4c 0200093e
	s_wait_alu depctr_va_vcc(0)                                // 0000000036c0: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s5, v63, vcc_lo             // 0000000036c4: d5207c4d 01aa7e05
	v_add_co_u32 v78, vcc_lo, v64, s4                          // 0000000036cc: d7006a4e 02000940
	s_wait_alu depctr_va_vcc(0)                                // 0000000036d4: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, s5, v65, vcc_lo             // 0000000036d8: d5207c4f 01aa8205
	s_clause 0x7                                               // 0000000036e0: bf850007
	global_load_b32 v30, v[30:31], off                         // 0000000036e4: ee05007c 0000001e 0000001e
	global_load_b32 v31, v[66:67], off                         // 0000000036f0: ee05007c 0000001f 00000042
	global_load_b32 v66, v[68:69], off                         // 0000000036fc: ee05007c 00000042 00000044
	global_load_b32 v67, v[70:71], off                         // 000000003708: ee05007c 00000043 00000046
	global_load_b32 v68, v[72:73], off                         // 000000003714: ee05007c 00000044 00000048
	global_load_b32 v69, v[74:75], off                         // 000000003720: ee05007c 00000045 0000004a
	global_load_b32 v70, v[76:77], off                         // 00000000372c: ee05007c 00000046 0000004c
	global_load_b32 v71, v[78:79], off                         // 000000003738: ee05007c 00000047 0000004e
	s_lshr_b64 s[4:5], s[42:43], 7                             // 000000003744: 8584872a
	s_mov_b64 s[42:43], s[40:41]                               // 000000003748: beaa0128
	s_wait_alu depctr_sa_sdst(0)                               // 00000000374c: bf88ff9e
	s_mul_u64 s[4:5], s[4:5], s[36:37]                         // 000000003750: aa842404
	s_wait_alu depctr_sa_sdst(0)                               // 000000003754: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000003758: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000375c: bf88ff9e
	s_add_nc_u64 s[4:5], s[20:21], s[4:5]                      // 000000003760: a9840414
	s_wait_alu depctr_sa_sdst(0)                               // 000000003764: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[38:39]                      // 000000003768: a9862604
	s_clause 0x1                                               // 00000000376c: bf850001
	s_load_b32 s3, s[6:7], 0x0                                 // 000000003770: f40000c3 f8000000
	s_load_b32 s4, s[4:5], 0x0                                 // 000000003778: f4000102 f8000000
	s_wait_kmcnt 0x0                                           // 000000003780: bfc70000
	v_mov_b32_e32 v72, s3                                      // 000000003784: 7e900203
	v_cmp_lt_i64_e64 s3, s[40:41], s[22:23]                    // 000000003788: d4510003 02002c28
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 000000003790: bf8700b2
	v_cndmask_b32_e64 v73, s4, v72, s1                         // 000000003794: d5010049 00069004
	s_and_b32 vcc_lo, exec_lo, s3                              // 00000000379c: 8b6a037e
	s_wait_loadcnt 0x6                                         // 0000000037a0: bfc00006
	v_dual_mul_f32 v74, v30, v73 :: v_dual_mul_f32 v75, v73, v31// 0000000037a4: c8c6931e 4a4a3f49
	v_cndmask_b32_e64 v72, s4, v72, s2                         // 0000000037ac: d5010048 000a9004
	s_wait_loadcnt 0x4                                         // 0000000037b4: bfc00004
	v_dual_mul_f32 v76, v73, v66 :: v_dual_mul_f32 v77, v73, v67// 0000000037b8: c8c68549 4c4c8749
	s_wait_loadcnt 0x2                                         // 0000000037c0: bfc00002
	v_dual_mul_f32 v78, v73, v68 :: v_dual_mul_f32 v79, v73, v69// 0000000037c4: c8c68949 4e4e8b49
	s_wait_loadcnt 0x1                                         // 0000000037cc: bfc00001
	v_dual_mul_f32 v80, v73, v70 :: v_dual_mul_f32 v31, v72, v31// 0000000037d0: c8c68d49 501e3f48
	s_wait_loadcnt 0x0                                         // 0000000037d8: bfc00000
	v_dual_mul_f32 v73, v73, v71 :: v_dual_mul_f32 v30, v30, v72// 0000000037dc: c8c68f49 491e911e
	v_dual_mul_f32 v67, v72, v67 :: v_dual_mul_f32 v66, v72, v66// 0000000037e4: c8c68748 43428548
	v_dual_mul_f32 v69, v72, v69 :: v_dual_mul_f32 v68, v72, v68// 0000000037ec: c8c68b48 45448948
	v_dual_mul_f32 v71, v72, v71 :: v_dual_mul_f32 v70, v72, v70// 0000000037f4: c8c68f48 47468d48
	v_dual_mul_f32 v9, v9, v75 :: v_dual_mul_f32 v8, v8, v74   // 0000000037fc: c8c69709 09089508
	v_dual_mul_f32 v11, v11, v77 :: v_dual_mul_f32 v10, v10, v76// 000000003804: c8c69b0b 0b0a990a
	v_dual_mul_f32 v13, v13, v79 :: v_dual_mul_f32 v12, v12, v78// 00000000380c: c8c69f0d 0d0c9d0c
	v_dual_mul_f32 v15, v15, v73 :: v_dual_mul_f32 v14, v14, v80// 000000003814: c8c6930f 0f0ea10e
	v_dual_mul_f32 v1, v1, v31 :: v_dual_mul_f32 v0, v0, v30   // 00000000381c: c8c63f01 01003d00
	v_dual_mul_f32 v3, v3, v67 :: v_dual_mul_f32 v2, v2, v66   // 000000003824: c8c68703 03028502
	v_dual_mul_f32 v5, v5, v69 :: v_dual_mul_f32 v4, v4, v68   // 00000000382c: c8c68b05 05048904
	v_dual_mul_f32 v7, v7, v71 :: v_dual_mul_f32 v6, v6, v70   // 000000003834: c8c68f07 07068d06
	v_dual_add_f32 v55, v55, v8 :: v_dual_add_f32 v52, v52, v9 // 00000000383c: c9081137 37341334
	v_dual_add_f32 v47, v47, v10 :: v_dual_add_f32 v46, v46, v11// 000000003844: c908152f 2f2e172e
	v_dual_add_f32 v45, v45, v12 :: v_dual_add_f32 v44, v44, v13// 00000000384c: c908192d 2d2c1b2c
	v_dual_add_f32 v43, v43, v14 :: v_dual_add_f32 v42, v42, v15// 000000003854: c9081d2b 2b2a1f2a
	v_dual_add_f32 v41, v41, v0 :: v_dual_add_f32 v40, v40, v1 // 00000000385c: c9080129 29280328
	v_dual_add_f32 v39, v39, v2 :: v_dual_add_f32 v38, v38, v3 // 000000003864: c9080527 27260726
	v_dual_add_f32 v37, v37, v4 :: v_dual_add_f32 v36, v36, v5 // 00000000386c: c9080925 25240b24
	v_dual_add_f32 v35, v35, v6 :: v_dual_add_f32 v34, v34, v7 // 000000003874: c9080d23 23220f22
	s_wait_alu depctr_sa_sdst(0)                               // 00000000387c: bf88ff9e
	s_cbranch_vccnz 63978                                      // 000000003880: bfa4f9ea <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x52c>
	v_mul_lo_u32 v4, s15, v20                                  // 000000003884: d72c0004 0202280f
	v_mul_lo_u32 v5, s14, v21                                  // 00000000388c: d72c0005 02022a0e
	v_mad_co_u64_u32 v[0:1], null, s14, v20, 0                 // 000000003894: d6fe7c00 0202280e
	v_sub_co_u32 v2, vcc_lo, s12, v20                          // 00000000389c: d7016a02 0202280c
	s_wait_alu depctr_va_vcc(0)                                // 0000000038a4: bf88ff9d
	v_sub_co_ci_u32_e64 v3, null, s13, v21, vcc_lo             // 0000000038a8: d5217c03 01aa2a0d
	v_cmp_gt_i64_e64 s7, s[14:15], v[16:17]                    // 0000000038b0: d4540007 0202200e
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 0000000038b8: bf870194
	v_add3_u32 v1, v1, v5, v4                                  // 0000000038bc: d6550001 04120b01
	v_cmp_lt_i64_e32 vcc_lo, 0, v[2:3]                         // 0000000038c4: 7ca20480
	s_delay_alu instid0(valu_dep_2)                            // 0000000038c8: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000038cc: 3e000081
	s_and_b32 s0, vcc_lo, s7                                   // 0000000038d0: 8b00076a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038d4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000038d8: be812000
	s_cbranch_execz 28                                         // 0000000038dc: bfa5001c <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1e50>
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 0000000038e0: 3e082081
	v_add_co_u32 v7, s0, s16, v0                               // 0000000038e4: d7000007 02020010
	v_bfe_u32 v6, v55, 16, 1                                   // 0000000038ec: d6100006 02052137
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v1, s0                  // 0000000038f8: d5207c08 00020211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003900: bf870193
	v_add_co_u32 v4, s0, v7, v4                                // 000000003904: d7000004 02020907
	v_add3_u32 v6, v6, v55, 0x7fff                             // 00000000390c: d6550006 03fe6f06 00007fff
	v_or_b32_e32 v9, 0x400000, v55                             // 000000003918: 38126eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003920: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s0                   // 000000003924: d5207c05 00020b08
	v_cmp_u_f32_e64 s0, v55, v55                               // 00000000392c: d4180000 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000003934: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003938: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s0                           // 00000000393c: d5010006 00021306
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003944: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003950: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003954: 8c7e017e
	v_cmp_lt_i64_e64 s0, 1, v[2:3]                             // 000000003958: d4510000 02020481
	s_and_b32 s1, s0, s7                                       // 000000003960: 8b010700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003964: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000003968: be822001
	s_cbranch_execz 35                                         // 00000000396c: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1efc>
	v_add_co_u32 v6, s1, s16, v0                               // 000000003970: d7000106 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003978: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s1                  // 00000000397c: d5207c07 00060211
	s_lshl_b64 s[4:5], s[14:15], 1                             // 000000003984: 8484810e
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003988: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000398c: bf88ff9e
	v_add_co_u32 v6, s1, v6, s4                                // 000000003990: d7000106 02000906
	v_bfe_u32 v8, v52, 16, 1                                   // 000000003998: d6100008 02052134
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s5, v7, s1                   // 0000000039a4: d5207c07 00060e05
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000039ac: bf870193
	v_add_co_u32 v4, s1, v6, v4                                // 0000000039b0: d7000104 02020906
	v_add3_u32 v8, v8, v52, 0x7fff                             // 0000000039b8: d6550008 03fe6908 00007fff
	v_or_b32_e32 v9, 0x400000, v52                             // 0000000039c4: 381268ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000039cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s1                   // 0000000039d0: d5207c05 00060b07
	v_cmp_u_f32_e64 s1, v52, v52                               // 0000000039d8: d4180001 02026934
	s_wait_alu depctr_va_sdst(0)                               // 0000000039e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039e4: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s1                           // 0000000039e8: d5010006 00061308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000039f0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003a00: 8c7e027e
	v_cmp_lt_i64_e64 s1, 2, v[2:3]                             // 000000003a04: d4510001 02020482
	s_lshl_b64 s[8:9], s[14:15], 1                             // 000000003a0c: 8488810e
	s_and_b32 s2, s1, s7                                       // 000000003a10: 8b020701
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a14: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003a18: be832002
	s_cbranch_execz 35                                         // 000000003a1c: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x1fac>
	v_add_co_u32 v6, s2, s16, v0                               // 000000003a20: d7000206 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003a28: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s2                  // 000000003a2c: d5207c07 000a0211
	s_lshl_b64 s[4:5], s[8:9], 1                               // 000000003a34: 84848108
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003a38: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a3c: bf88ff9e
	v_add_co_u32 v6, s2, v6, s4                                // 000000003a40: d7000206 02000906
	v_bfe_u32 v8, v47, 16, 1                                   // 000000003a48: d6100008 0205212f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a50: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s5, v7, s2                   // 000000003a54: d5207c07 000a0e05
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003a5c: bf870193
	v_add_co_u32 v4, s2, v6, v4                                // 000000003a60: d7000204 02020906
	v_add3_u32 v8, v8, v47, 0x7fff                             // 000000003a68: d6550008 03fe5f08 00007fff
	v_or_b32_e32 v9, 0x400000, v47                             // 000000003a74: 38125eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a7c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s2                   // 000000003a80: d5207c05 000a0b07
	v_cmp_u_f32_e64 s2, v47, v47                               // 000000003a88: d4180002 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 000000003a90: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a94: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s2                           // 000000003a98: d5010006 000a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003aa0: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003aac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000003ab0: 8c7e037e
	v_cmp_lt_i64_e64 s2, 3, v[2:3]                             // 000000003ab4: d4510002 02020483
	s_mul_u64 s[10:11], s[14:15], 3                            // 000000003abc: aa8a830e
	s_and_b32 s3, s2, s7                                       // 000000003ac0: 8b030702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ac4: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003ac8: be842003
	s_cbranch_execz 34                                         // 000000003acc: bfa50022 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2058>
	v_add_co_u32 v6, s3, s16, v0                               // 000000003ad0: d7000306 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s3                  // 000000003adc: d5207c07 000e0211
	s_lshl_b64 s[36:37], s[10:11], 1                           // 000000003ae4: 84a4810a
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003ae8: 3e082081
	v_add_co_u32 v6, s3, v6, s36                               // 000000003aec: d7000306 02004906
	v_bfe_u32 v8, v46, 16, 1                                   // 000000003af4: d6100008 0205212e
	s_wait_alu depctr_va_sdst(0)                               // 000000003afc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s37, v7, s3                  // 000000003b00: d5207c07 000e0e25
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003b08: bf870193
	v_add_co_u32 v4, s3, v6, v4                                // 000000003b0c: d7000304 02020906
	v_add3_u32 v8, v8, v46, 0x7fff                             // 000000003b14: d6550008 03fe5d08 00007fff
	v_or_b32_e32 v9, 0x400000, v46                             // 000000003b20: 38125cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003b28: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s3                   // 000000003b2c: d5207c05 000e0b07
	v_cmp_u_f32_e64 s3, v46, v46                               // 000000003b34: d4180003 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b40: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s3                           // 000000003b44: d5010006 000e1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003b4c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b5c: 8c7e047e
	v_cmp_lt_i64_e64 s3, 4, v[2:3]                             // 000000003b60: d4510003 02020484
	s_lshl_b64 s[36:37], s[14:15], 2                           // 000000003b68: 84a4820e
	s_and_b32 s4, s3, s7                                       // 000000003b6c: 8b040703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b70: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000003b74: be852004
	s_cbranch_execz 34                                         // 000000003b78: bfa50022 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2104>
	v_add_co_u32 v6, s4, s16, v0                               // 000000003b7c: d7000406 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003b84: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s4                  // 000000003b88: d5207c07 00120211
	s_lshl_b64 s[38:39], s[36:37], 1                           // 000000003b90: 84a68124
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003b94: 3e082081
	v_add_co_u32 v6, s4, v6, s38                               // 000000003b98: d7000406 02004d06
	v_bfe_u32 v8, v45, 16, 1                                   // 000000003ba0: d6100008 0205212d
	s_wait_alu depctr_va_sdst(0)                               // 000000003ba8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s39, v7, s4                  // 000000003bac: d5207c07 00120e27
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003bb4: bf870193
	v_add_co_u32 v4, s4, v6, v4                                // 000000003bb8: d7000404 02020906
	v_add3_u32 v8, v8, v45, 0x7fff                             // 000000003bc0: d6550008 03fe5b08 00007fff
	v_or_b32_e32 v9, 0x400000, v45                             // 000000003bcc: 38125aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s4                   // 000000003bd8: d5207c05 00120b07
	v_cmp_u_f32_e64 s4, v45, v45                               // 000000003be0: d4180004 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 000000003be8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003bec: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s4                           // 000000003bf0: d5010006 00121308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003bf8: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003c08: 8c7e057e
	v_cmp_lt_i64_e64 s4, 5, v[2:3]                             // 000000003c0c: d4510004 02020485
	s_mul_u64 s[38:39], s[14:15], 5                            // 000000003c14: aaa6850e
	s_and_b32 s5, s4, s7                                       // 000000003c18: 8b050704
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c1c: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000003c20: be862005
	s_cbranch_execz 35                                         // 000000003c24: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x21b4>
	v_add_co_u32 v6, s5, s16, v0                               // 000000003c28: d7000506 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003c30: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s5                  // 000000003c34: d5207c07 00160211
	s_lshl_b64 s[40:41], s[38:39], 1                           // 000000003c3c: 84a88126
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003c40: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c44: bf88ff9e
	v_add_co_u32 v6, s5, v6, s40                               // 000000003c48: d7000506 02005106
	v_bfe_u32 v8, v44, 16, 1                                   // 000000003c50: d6100008 0205212c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c58: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s41, v7, s5                  // 000000003c5c: d5207c07 00160e29
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003c64: bf870193
	v_add_co_u32 v4, s5, v6, v4                                // 000000003c68: d7000504 02020906
	v_add3_u32 v8, v8, v44, 0x7fff                             // 000000003c70: d6550008 03fe5908 00007fff
	v_or_b32_e32 v9, 0x400000, v44                             // 000000003c7c: 381258ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003c84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s5                   // 000000003c88: d5207c05 00160b07
	v_cmp_u_f32_e64 s5, v44, v44                               // 000000003c90: d4180005 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c98: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c9c: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s5                           // 000000003ca0: d5010006 00161308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003ca8: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003cb8: 8c7e067e
	v_cmp_lt_i64_e64 s5, 6, v[2:3]                             // 000000003cbc: d4510005 02020486
	s_mul_u64 s[40:41], s[14:15], 6                            // 000000003cc4: aaa8860e
	s_and_b32 s6, s5, s7                                       // 000000003cc8: 8b060705
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ccc: bf88ff9e
	s_and_saveexec_b32 s42, s6                                 // 000000003cd0: beaa2006
	s_cbranch_execz 35                                         // 000000003cd4: bfa50023 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2264>
	v_add_co_u32 v6, s6, s16, v0                               // 000000003cd8: d7000606 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s17, v1, s6                  // 000000003ce4: d5207c07 001a0211
	s_lshl_b64 s[44:45], s[40:41], 1                           // 000000003cec: 84ac8128
	v_lshlrev_b64_e32 v[4:5], 1, v[16:17]                      // 000000003cf0: 3e082081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cf4: bf88ff9e
	v_add_co_u32 v6, s6, v6, s44                               // 000000003cf8: d7000606 02005906
	v_bfe_u32 v8, v43, 16, 1                                   // 000000003d00: d6100008 0205212b
	s_wait_alu depctr_va_sdst(0)                               // 000000003d08: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s45, v7, s6                  // 000000003d0c: d5207c07 001a0e2d
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d14: bf870193
	v_add_co_u32 v4, s6, v6, v4                                // 000000003d18: d7000604 02020906
	v_add3_u32 v8, v8, v43, 0x7fff                             // 000000003d20: d6550008 03fe5708 00007fff
	v_or_b32_e32 v9, 0x400000, v43                             // 000000003d2c: 381256ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003d34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v7, v5, s6                   // 000000003d38: d5207c05 001a0b07
	v_cmp_u_f32_e64 s6, v43, v43                               // 000000003d40: d4180006 0202572b
	s_wait_alu depctr_va_sdst(0)                               // 000000003d48: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d4c: bf870001
	v_cndmask_b32_e64 v6, v8, v9, s6                           // 000000003d50: d5010006 001a1308
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003d58: ee09407c 03000000 00000004
	s_or_b32 exec_lo, exec_lo, s42                             // 000000003d64: 8c7e2a7e
	v_cmp_lt_i64_e64 s6, 7, v[2:3]                             // 000000003d68: d4510006 02020487
	s_mul_u64 s[42:43], s[14:15], 7                            // 000000003d70: aaaa870e
	s_and_b32 s7, s6, s7                                       // 000000003d74: 8b070706
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d78: bf88ff9e
	s_and_saveexec_b32 s44, s7                                 // 000000003d7c: beac2007
	s_cbranch_execz 34                                         // 000000003d80: bfa50022 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x230c>
	v_add_co_u32 v4, s7, s16, v0                               // 000000003d84: d7000704 02020010
	s_wait_alu depctr_va_sdst(0)                               // 000000003d8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s17, v1, s7                  // 000000003d90: d5207c05 001e0211
	s_lshl_b64 s[46:47], s[42:43], 1                           // 000000003d98: 84ae812a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003d9c: 3e042081
	v_add_co_u32 v4, s7, v4, s46                               // 000000003da0: d7000704 02005d04
	v_bfe_u32 v6, v42, 16, 1                                   // 000000003da8: d6100006 0205212a
	s_wait_alu depctr_va_sdst(0)                               // 000000003db0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s47, v5, s7                  // 000000003db4: d5207c05 001e0a2f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003dbc: bf870193
	v_add_co_u32 v2, s7, v4, v2                                // 000000003dc0: d7000702 02020504
	v_add3_u32 v6, v6, v42, 0x7fff                             // 000000003dc8: d6550006 03fe5506 00007fff
	v_or_b32_e32 v7, 0x400000, v42                             // 000000003dd4: 380e54ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ddc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v5, v3, s7                   // 000000003de0: d5207c03 001e0705
	v_cmp_u_f32_e64 s7, v42, v42                               // 000000003de8: d4180007 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 000000003df0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003df4: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s7                           // 000000003df8: d5010004 001e0f06
	global_store_d16_hi_b16 v[2:3], v4, off                    // 000000003e00: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s44                             // 000000003e10: 8c7e2c7e
	v_cmp_gt_i64_e64 s7, s[14:15], v[18:19]                    // 000000003e14: d4540007 0202240e
	s_and_b32 s45, vcc_lo, s7                                  // 000000003e1c: 8b2d076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e20: bf88ff9e
	s_and_saveexec_b32 s44, s45                                // 000000003e24: beac202d
	s_cbranch_execz 25                                         // 000000003e28: bfa50019 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2390>
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003e2c: 3e042081
	v_add_co_u32 v5, vcc_lo, s16, v0                           // 000000003e30: d7006a05 02020010
	v_bfe_u32 v4, v41, 16, 1                                   // 000000003e38: d6100004 02052129
	s_wait_alu depctr_va_vcc(0)                                // 000000003e40: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s17, v1, vcc_lo              // 000000003e44: d5207c06 01aa0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003e4c: bf870193
	v_add_co_u32 v2, vcc_lo, v5, v2                            // 000000003e50: d7006a02 02020505
	v_add3_u32 v4, v4, v41, 0x7fff                             // 000000003e58: d6550004 03fe5304 00007fff
	v_or_b32_e32 v7, 0x400000, v41                             // 000000003e64: 380e52ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e6c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v6, v3, vcc_lo               // 000000003e70: d5207c03 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000003e78: 7c305329
	s_wait_alu depctr_va_vcc(0)                                // 000000003e7c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 000000003e80: 02080f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003e84: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e90: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s44                             // 000000003e94: 8c7e2c7e
	s_and_b32 s44, s0, s7                                      // 000000003e98: 8b2c0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e9c: bf88ff9e
	s_and_saveexec_b32 s0, s44                                 // 000000003ea0: be80202c
	s_cbranch_execz 31                                         // 000000003ea4: bfa5001f <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2424>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003ea8: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003eb0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003eb4: d5207c05 01aa0211
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003ebc: 3e042081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000003ec0: bf8701c3
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003ec4: d7006a04 02001104
	v_bfe_u32 v6, v40, 16, 1                                   // 000000003ecc: d6100006 02052128
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000003ed8: d5207c05 01aa0a09
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003ee0: d7006a02 02020504
	s_delay_alu instid0(valu_dep_3)                            // 000000003ee8: bf870003
	v_add3_u32 v6, v6, v40, 0x7fff                             // 000000003eec: d6550006 03fe5106 00007fff
	v_or_b32_e32 v7, 0x400000, v40                             // 000000003ef8: 380e50ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003f00: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003f04: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000003f0c: 7c305128
	s_wait_alu depctr_va_vcc(0)                                // 000000003f10: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003f14: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003f18: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003f28: 8c7e007e
	s_and_b32 s1, s1, s7                                       // 000000003f2c: 8b010701
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f30: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003f34: be802001
	s_cbranch_execz 32                                         // 000000003f38: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x24bc>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003f3c: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003f44: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003f48: d5207c05 01aa0211
	s_lshl_b64 s[8:9], s[8:9], 1                               // 000000003f50: 84888108
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003f54: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f58: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003f5c: d7006a04 02001104
	v_bfe_u32 v6, v39, 16, 1                                   // 000000003f64: d6100006 02052127
	s_wait_alu depctr_va_vcc(0)                                // 000000003f6c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000003f70: d5207c05 01aa0a09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003f78: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000003f7c: d7006a02 02020504
	v_add3_u32 v6, v6, v39, 0x7fff                             // 000000003f84: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v7, 0x400000, v39                             // 000000003f90: 380e4eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003f98: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000003f9c: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 000000003fa4: 7c304f27
	s_wait_alu depctr_va_vcc(0)                                // 000000003fa8: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000003fac: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003fb0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fbc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003fc0: 8c7e007e
	s_and_b32 s1, s2, s7                                       // 000000003fc4: 8b010702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fc8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003fcc: be802001
	s_cbranch_execz 32                                         // 000000003fd0: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2554>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000003fd4: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000003fdc: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000003fe0: d5207c05 01aa0211
	s_lshl_b64 s[8:9], s[10:11], 1                             // 000000003fe8: 8488810a
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000003fec: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ff0: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s8                            // 000000003ff4: d7006a04 02001104
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003ffc: d6100006 02052126
	s_wait_alu depctr_va_vcc(0)                                // 000000004004: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s9, v5, vcc_lo               // 000000004008: d5207c05 01aa0a09
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004010: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000004014: d7006a02 02020504
	v_add3_u32 v6, v6, v38, 0x7fff                             // 00000000401c: d6550006 03fe4d06 00007fff
	v_or_b32_e32 v7, 0x400000, v38                             // 000000004028: 380e4cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004030: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000004034: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 00000000403c: 7c304d26
	s_wait_alu depctr_va_vcc(0)                                // 000000004040: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000004044: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004048: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004054: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004058: 8c7e007e
	s_and_b32 s1, s3, s7                                       // 00000000405c: 8b010703
	s_wait_alu depctr_sa_sdst(0)                               // 000000004060: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004064: be802001
	s_cbranch_execz 32                                         // 000000004068: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x25ec>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 00000000406c: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004074: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000004078: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[36:37], 1                             // 000000004080: 84828124
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 000000004084: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004088: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 00000000408c: d7006a04 02000504
	v_bfe_u32 v6, v37, 16, 1                                   // 000000004094: d6100006 02052125
	s_wait_alu depctr_va_vcc(0)                                // 00000000409c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 0000000040a0: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000040a8: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 0000000040ac: d7006a02 02020504
	v_add3_u32 v6, v6, v37, 0x7fff                             // 0000000040b4: d6550006 03fe4b06 00007fff
	v_or_b32_e32 v7, 0x400000, v37                             // 0000000040c0: 380e4aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000040c8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 0000000040cc: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 0000000040d4: 7c304b25
	s_wait_alu depctr_va_vcc(0)                                // 0000000040d8: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 0000000040dc: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 0000000040e0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000040f0: 8c7e007e
	s_and_b32 s1, s4, s7                                       // 0000000040f4: 8b010704
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040f8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000040fc: be802001
	s_cbranch_execz 32                                         // 000000004100: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2684>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 000000004104: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 00000000410c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 000000004110: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000004118: 84828126
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 00000000411c: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004120: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 000000004124: d7006a04 02000504
	v_bfe_u32 v6, v36, 16, 1                                   // 00000000412c: d6100006 02052124
	s_wait_alu depctr_va_vcc(0)                                // 000000004134: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000004138: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004140: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 000000004144: d7006a02 02020504
	v_add3_u32 v6, v6, v36, 0x7fff                             // 00000000414c: d6550006 03fe4906 00007fff
	v_or_b32_e32 v7, 0x400000, v36                             // 000000004158: 380e48ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004160: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 000000004164: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 00000000416c: 7c304924
	s_wait_alu depctr_va_vcc(0)                                // 000000004170: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 000000004174: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004178: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004184: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004188: 8c7e007e
	s_and_b32 s1, s5, s7                                       // 00000000418c: 8b010705
	s_wait_alu depctr_sa_sdst(0)                               // 000000004190: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004194: be802001
	s_cbranch_execz 32                                         // 000000004198: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x271c>
	v_add_co_u32 v4, vcc_lo, s16, v0                           // 00000000419c: d7006a04 02020010
	s_wait_alu depctr_va_vcc(0)                                // 0000000041a4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo              // 0000000041a8: d5207c05 01aa0211
	s_lshl_b64 s[2:3], s[40:41], 1                             // 0000000041b0: 84828128
	v_lshlrev_b64_e32 v[2:3], 1, v[16:17]                      // 0000000041b4: 3e042081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041b8: bf88ff9e
	v_add_co_u32 v4, vcc_lo, v4, s2                            // 0000000041bc: d7006a04 02000504
	v_bfe_u32 v6, v35, 16, 1                                   // 0000000041c4: d6100006 02052123
	s_wait_alu depctr_va_vcc(0)                                // 0000000041cc: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 0000000041d0: d5207c05 01aa0a03
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000041d8: bf870193
	v_add_co_u32 v2, vcc_lo, v4, v2                            // 0000000041dc: d7006a02 02020504
	v_add3_u32 v6, v6, v35, 0x7fff                             // 0000000041e4: d6550006 03fe4706 00007fff
	v_or_b32_e32 v7, 0x400000, v35                             // 0000000041f0: 380e46ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000041f8: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v5, v3, vcc_lo               // 0000000041fc: d5207c03 01aa0705
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000004204: 7c304723
	s_wait_alu depctr_va_vcc(0)                                // 000000004208: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 00000000420c: 02080f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004210: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000421c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004220: 8c7e007e
	s_and_b32 s1, s6, s7                                       // 000000004224: 8b010706
	s_wait_alu depctr_sa_sdst(0)                               // 000000004228: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000422c: be802001
	s_cbranch_execz 32                                         // 000000004230: bfa50020 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x27b4>
	v_add_co_u32 v2, vcc_lo, s16, v0                           // 000000004234: d7006a02 02020010
	s_wait_alu depctr_va_vcc(0)                                // 00000000423c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v1, vcc_lo              // 000000004240: d5207c03 01aa0211
	s_lshl_b64 s[2:3], s[42:43], 1                             // 000000004248: 8482812a
	v_lshlrev_b64_e32 v[0:1], 1, v[16:17]                      // 00000000424c: 3e002081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004250: bf88ff9e
	v_add_co_u32 v2, vcc_lo, v2, s2                            // 000000004254: d7006a02 02000502
	v_bfe_u32 v4, v34, 16, 1                                   // 00000000425c: d6100004 02052122
	s_wait_alu depctr_va_vcc(0)                                // 000000004264: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo               // 000000004268: d5207c03 01aa0603
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004270: bf870193
	v_add_co_u32 v0, vcc_lo, v2, v0                            // 000000004274: d7006a00 02020102
	v_add3_u32 v4, v4, v34, 0x7fff                             // 00000000427c: d6550004 03fe4504 00007fff
	v_or_b32_e32 v5, 0x400000, v34                             // 000000004288: 380a44ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004290: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo               // 000000004294: d5207c01 01aa0303
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 00000000429c: 7c304522
	s_wait_alu depctr_va_vcc(0)                                // 0000000042a0: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 0000000042a4: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000042a8: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000042b8: 8c7e007e
	s_mov_b32 s0, 0                                            // 0000000042bc: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042c0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000042c4: 8b6a007e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042c8: bf88ff9e
	s_cbranch_vccz 24                                          // 0000000042cc: bfa30018 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2830>
	s_and_b32 s0, s33, exec_lo                                 // 0000000042d0: 8b007e21
	s_cselect_b32 s0, 1, 0                                     // 0000000042d4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042d8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000042dc: bf078100
	s_cbranch_scc1 20                                          // 0000000042e0: bfa20014 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2834>
	v_lshl_or_b32 v12, v33, 3, s34                             // 0000000042e4: d656000c 00890721
	v_mov_b32_e32 v13, s35                                     // 0000000042ec: 7e1a0223
	v_mov_b32_e32 v15, s35                                     // 0000000042f0: 7e1e0223
	v_mov_b32_e32 v11, s35                                     // 0000000042f4: 7e160223
	v_mov_b32_e32 v7, s35                                      // 0000000042f8: 7e0e0223
	v_or_b32_e32 v14, 1, v12                                   // 0000000042fc: 381c1881
	v_or_b32_e32 v10, 2, v12                                   // 000000004300: 38141882
	v_or_b32_e32 v6, 3, v12                                    // 000000004304: 380c1883
	v_or_b32_e32 v8, 4, v12                                    // 000000004308: 38101884
	v_mov_b32_e32 v9, s35                                      // 00000000430c: 7e120223
	v_or_b32_e32 v4, 5, v12                                    // 000000004310: 38081885
	v_mov_b32_e32 v5, s35                                      // 000000004314: 7e0a0223
	v_or_b32_e32 v2, 6, v12                                    // 000000004318: 38041886
	v_mov_b32_e32 v3, s35                                      // 00000000431c: 7e060223
	v_or_b32_e32 v0, 7, v12                                    // 000000004320: 38001887
	v_mov_b32_e32 v1, s35                                      // 000000004324: 7e020223
	s_mov_b32 s0, 0                                            // 000000004328: be800080
	s_branch 2                                                 // 00000000432c: bfa00002 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2838>
	s_endpgm                                                   // 000000004330: bfb00000
	s_mov_b32 s0, -1                                           // 000000004334: be8000c1
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v45, 0             // 000000004338: ca100080 2a2c0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004340: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000004344: 8b007e00
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v47, 0             // 000000004348: ca100080 2c2e0080
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v49, 0             // 000000004350: ca100080 2e300080
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v37, 0             // 000000004358: ca100080 30240080
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v39, 0             // 000000004360: ca100080 32260080
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v41, 0             // 000000004368: ca100080 14280080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v43, 0             // 000000004370: ca100080 242a0080
	v_mov_b32_e32 v38, 0                                       // 000000004378: 7e4c0280
	v_mov_b32_e32 v40, 0                                       // 00000000437c: 7e500280
	s_cselect_b32 s0, 1, 0                                     // 000000004380: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004384: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000004388: bf078100
	s_cbranch_scc1 534                                         // 00000000438c: bfa20216 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x30e8>
	v_dual_mov_b32 v21, 0 :: v_dual_lshlrev_b32 v20, 3, v33    // 000000004390: ca220080 15144283
	s_add_nc_u64 s[0:1], s[14:15], 0x7f                        // 000000004398: a980ff0e 0000007f
	v_mov_b32_e32 v1, s35                                      // 0000000043a0: 7e020223
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043a4: bf88ff9e
	s_lshr_b64 s[6:7], s[0:1], 7                               // 0000000043a8: 85868700
	v_or_b32_e32 v12, s34, v20                                 // 0000000043ac: 38182822
	s_lshr_b64 s[0:1], s[30:31], 7                             // 0000000043b0: 8580871e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043b4: bf88ff9e
	s_add_nc_u64 s[2:3], s[6:7], -1                            // 0000000043b8: a982c106
	v_mov_b32_e32 v3, s35                                      // 0000000043bc: 7e060223
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043c0: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[0:1], s[2:3]                        // 0000000043c4: d4590004 02000400
	v_or_b32_e32 v0, 7, v12                                    // 0000000043cc: 38001887
	v_or_b32_e32 v2, 6, v12                                    // 0000000043d0: 38041886
	s_lshr_b64 s[10:11], s[24:25], 7                           // 0000000043d4: 858a8718
	v_mov_b32_e32 v15, s35                                     // 0000000043d8: 7e1e0223
	v_or_b32_e32 v14, 1, v12                                   // 0000000043dc: 381c1881
	s_and_b32 s4, s4, exec_lo                                  // 0000000043e0: 8b047e04
	s_cselect_b32 s9, s1, s3                                   // 0000000043e4: 98090301
	s_cselect_b32 s8, s0, s2                                   // 0000000043e8: 98080200
	v_cmp_gt_i64_e64 s2, s[12:13], v[0:1]                      // 0000000043ec: d4540002 0202000c
	v_cmp_gt_i64_e64 s3, s[12:13], v[2:3]                      // 0000000043f4: d4540003 0202040c
	v_or_b32_e32 v8, 4, v12                                    // 0000000043fc: 38101884
	v_mov_b32_e32 v9, s35                                      // 000000004400: 7e120223
	v_or_b32_e32 v4, 5, v12                                    // 000000004404: 38081885
	v_mov_b32_e32 v5, s35                                      // 000000004408: 7e0a0223
	v_cmp_gt_i64_e64 s1, s[12:13], v[14:15]                    // 00000000440c: d4540001 02021c0c
	v_or_b32_e32 v6, 3, v12                                    // 000000004414: 380c1883
	v_mov_b32_e32 v7, s35                                      // 000000004418: 7e0e0223
	s_wait_alu depctr_va_sdst(0)                               // 00000000441c: bf88f19f
	v_cndmask_b32_e64 v22, 0, v0, s2                           // 000000004420: d5010016 000a0080
	v_cndmask_b32_e64 v23, 0, s35, s2                          // 000000004428: d5010017 00084680
	v_cndmask_b32_e64 v24, 0, v2, s3                           // 000000004430: d5010018 000e0480
	v_cndmask_b32_e64 v25, 0, s35, s3                          // 000000004438: d5010019 000c4680
	v_cmp_gt_i64_e64 s4, s[12:13], v[8:9]                      // 000000004440: d4540004 0202100c
	v_cmp_gt_i64_e64 s2, s[12:13], v[4:5]                      // 000000004448: d4540002 0202080c
	s_lshr_b32 s3, s25, 7                                      // 000000004450: 85038719
	v_cndmask_b32_e64 v35, 0, v14, s1                          // 000000004454: d5010023 00061c80
	v_cndmask_b32_e64 v36, 0, s35, s1                          // 00000000445c: d5010024 00044680
	v_cmp_gt_i64_e64 s1, s[12:13], v[6:7]                      // 000000004464: d4540001 02020c0c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000446c: bf88ff9e
	v_mul_lo_u32 v26, v23, s10                                 // 000000004470: d72c001a 02001517
	v_mul_lo_u32 v27, v22, s3                                  // 000000004478: d72c001b 02000716
	v_mad_co_u64_u32 v[22:23], null, v22, s10, 0               // 000000004480: d6fe7c16 02001516
	v_mul_lo_u32 v28, v25, s10                                 // 000000004488: d72c001c 02001519
	v_mul_lo_u32 v29, v24, s3                                  // 000000004490: d72c001d 02000718
	v_mad_co_u64_u32 v[24:25], null, v24, s10, 0               // 000000004498: d6fe7c18 02001518
	v_dual_mov_b32 v13, s35 :: v_dual_mov_b32 v50, 0           // 0000000044a0: ca100023 0d320080
	v_cndmask_b32_e64 v33, 0, v4, s2                           // 0000000044a8: d5010021 000a0880
	v_cndmask_b32_e64 v39, 0, v8, s4                           // 0000000044b0: d5010027 00121080
	v_cndmask_b32_e64 v40, 0, s35, s4                          // 0000000044b8: d5010028 00104680
	s_wait_alu depctr_va_sdst(0)                               // 0000000044c0: bf88f19f
	v_cndmask_b32_e64 v30, 0, v6, s1                           // 0000000044c4: d501001e 00060c80
	v_cndmask_b32_e64 v31, 0, s35, s1                          // 0000000044cc: d501001f 00044680
	v_cndmask_b32_e64 v34, 0, s35, s2                          // 0000000044d4: d5010022 00084680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[12:13]                // 0000000044dc: 7ca8180c
	v_or_b32_e32 v10, 2, v12                                   // 0000000044e0: 38141882
	v_mov_b32_e32 v11, s35                                     // 0000000044e4: 7e160223
	v_add3_u32 v23, v23, v27, v26                              // 0000000044e8: d6550017 046a3717
	v_add3_u32 v25, v25, v29, v28                              // 0000000044f0: d6550019 04723b19
	v_mul_lo_u32 v41, v33, s3                                  // 0000000044f8: d72c0029 02000721
	v_mad_co_u64_u32 v[26:27], null, v33, s10, 0               // 000000004500: d6fe7c1a 02001521
	v_mul_lo_u32 v33, v40, s10                                 // 000000004508: d72c0021 02001528
	v_mul_lo_u32 v40, v39, s3                                  // 000000004510: d72c0028 02000727
	v_mad_co_u64_u32 v[28:29], null, v39, s10, 0               // 000000004518: d6fe7c1c 02001527
	v_mul_lo_u32 v34, v34, s10                                 // 000000004520: d72c0022 02001522
	v_mul_lo_u32 v39, v31, s10                                 // 000000004528: d72c0027 0200151f
	v_mul_lo_u32 v42, v30, s3                                  // 000000004530: d72c002a 0200071e
	v_mad_co_u64_u32 v[30:31], null, v30, s10, 0               // 000000004538: d6fe7c1e 0200151e
	s_wait_alu depctr_va_vcc(0)                                // 000000004540: bf88ff9d
	v_cndmask_b32_e32 v37, 0, v12, vcc_lo                      // 000000004544: 024a1880
	v_cndmask_b32_e64 v38, 0, s35, vcc_lo                      // 000000004548: d5010026 01a84680
	v_cmp_gt_i64_e32 vcc_lo, s[12:13], v[10:11]                // 000000004550: 7ca8140c
	v_add3_u32 v29, v29, v40, v33                              // 000000004554: d655001d 0486511d
	v_add_co_u32 v33, s2, s34, v32                             // 00000000455c: d7000221 02024022
	v_add3_u32 v27, v27, v41, v34                              // 000000004564: d655001b 048a531b
	v_add3_u32 v31, v31, v42, v39                              // 00000000456c: d655001f 049e551f
	s_wait_alu depctr_va_vcc(0)                                // 000000004574: bf88ff9d
	v_cndmask_b32_e32 v43, 0, v10, vcc_lo                      // 000000004578: 02561480
	v_cndmask_b32_e64 v44, 0, s35, vcc_lo                      // 00000000457c: d501002c 01a84680
	s_wait_alu depctr_va_sdst(0)                               // 000000004584: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s35, 0, s2                  // 000000004588: d5207c22 00090023
	v_add_co_u32 v32, s2, s30, v32                             // 000000004590: d7000220 0202401e
	s_wait_alu depctr_va_sdst(0)                               // 000000004598: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s31, 0, s2                  // 00000000459c: d5207c2f 0009001f
	v_cmp_gt_i64_e64 s1, s[14:15], v[18:19]                    // 0000000045a4: d4540001 0202240e
	v_lshlrev_b64_e32 v[18:19], 2, v[22:23]                    // 0000000045ac: 3e242c82
	v_lshlrev_b64_e32 v[22:23], 2, v[24:25]                    // 0000000045b0: 3e2c3082
	v_lshlrev_b64_e32 v[24:25], 2, v[26:27]                    // 0000000045b4: 3e303482
	v_lshlrev_b64_e32 v[26:27], 2, v[28:29]                    // 0000000045b8: 3e343882
	v_lshlrev_b64_e32 v[28:29], 2, v[30:31]                    // 0000000045bc: 3e383c82
	v_mad_co_u64_u32 v[30:31], null, s24, v33, v[20:21]        // 0000000045c0: d6fe7c1e 04524218
	v_mul_lo_u32 v39, s24, v34                                 // 0000000045c8: d72c0027 02024418
	v_mul_lo_u32 v40, s25, v33                                 // 0000000045d0: d72c0028 02024219
	v_mul_lo_u32 v41, v44, s10                                 // 0000000045d8: d72c0029 0200152c
	v_mul_lo_u32 v42, v43, s3                                  // 0000000045e0: d72c002a 0200072b
	v_mad_co_u64_u32 v[33:34], null, v43, s10, 0               // 0000000045e8: d6fe7c21 0200152b
	v_mul_lo_u32 v43, v36, s10                                 // 0000000045f0: d72c002b 02001524
	v_mul_lo_u32 v44, v35, s3                                  // 0000000045f8: d72c002c 02000723
	v_mad_co_u64_u32 v[35:36], null, v35, s10, 0               // 000000004600: d6fe7c23 02001523
	v_add_co_u32 v48, vcc_lo, v32, 16                          // 000000004608: d7006a30 02012120
	v_mul_lo_u32 v45, v38, s10                                 // 000000004610: d72c002d 02001526
	v_mul_lo_u32 v46, v37, s3                                  // 000000004618: d72c002e 02000725
	v_mad_co_u64_u32 v[37:38], null, v37, s10, 0               // 000000004620: d6fe7c25 02001525
	s_wait_alu depctr_va_vcc(0)                                // 000000004628: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, 0, v47, vcc_lo              // 00000000462c: d5207c31 01aa5e80
	v_add3_u32 v31, v40, v31, v39                              // 000000004634: d655001f 049e3f28
	v_add3_u32 v36, v36, v44, v43                              // 00000000463c: d6550024 04ae5924
	v_mad_co_u64_u32 v[39:40], null, s24, v48, v[20:21]        // 000000004644: d6fe7c27 04526018
	s_delay_alu instid0(valu_dep_4)                            // 00000000464c: bf870004
	v_mul_lo_u32 v43, s24, v49                                 // 000000004650: d72c002b 02026218
	v_mul_lo_u32 v44, s25, v48                                 // 000000004658: d72c002c 02026019
	v_add3_u32 v34, v34, v42, v41                              // 000000004660: d6550022 04a65522
	v_add3_u32 v38, v38, v46, v45                              // 000000004668: d6550026 04b65d26
	v_mad_co_u64_u32 v[41:42], null, s24, v32, v[20:21]        // 000000004670: d6fe7c29 04524018
	v_mul_lo_u32 v20, s24, v47                                 // 000000004678: d72c0014 02025e18
	v_mul_lo_u32 v45, s25, v32                                 // 000000004680: d72c002d 02024019
	v_add_co_u32 v51, vcc_lo, s28, v30                         // 000000004688: d7006a33 02023c1c
	s_wait_alu depctr_va_vcc(0)                                // 000000004690: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, s29, v31, vcc_lo            // 000000004694: d5207c34 01aa3e1d
	v_lshlrev_b64_e32 v[30:31], 2, v[33:34]                    // 00000000469c: 3e3c4282
	v_lshlrev_b64_e32 v[32:33], 2, v[35:36]                    // 0000000046a0: 3e404682
	v_add3_u32 v36, v44, v40, v43                              // 0000000046a4: d6550024 04ae512c
	v_add3_u32 v20, v45, v42, v20                              // 0000000046ac: d6550014 0452552d
	v_add_co_u32 v53, vcc_lo, s26, v39                         // 0000000046b4: d7006a35 02024e1a
	v_cmp_gt_i64_e64 s0, s[14:15], v[16:17]                    // 0000000046bc: d4540000 0202200e
	s_wait_alu depctr_va_vcc(0)                                // 0000000046c4: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, s27, v36, vcc_lo            // 0000000046c8: d5207c36 01aa481b
	v_add_co_u32 v55, vcc_lo, s26, v41                         // 0000000046d0: d7006a37 0202521a
	v_lshlrev_b64_e32 v[34:35], 2, v[37:38]                    // 0000000046d8: 3e444a82
	s_wait_alu depctr_va_vcc(0)                                // 0000000046dc: bf88ff9d
	v_add_co_ci_u32_e64 v56, null, s27, v20, vcc_lo            // 0000000046e0: d5207c38 01aa281b
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0             // 0000000046e8: ca100080 31300080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0             // 0000000046f0: ca100080 2f2e0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0             // 0000000046f8: ca100080 2d2c0080
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0             // 000000004700: ca100080 2a2a0080
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0             // 000000004708: ca100080 29280080
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0             // 000000004710: ca100080 27260080
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0             // 000000004718: ca100080 25240080
	v_mov_b32_e32 v20, 0                                       // 000000004720: 7e280280
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000004724: 84828208
	s_lshl_b64 s[4:5], s[6:7], 2                               // 000000004728: 84848206
	s_mov_b64 s[8:9], 0                                        // 00000000472c: be880180
	s_wait_alu depctr_sa_sdst(0)                               // 000000004730: bf88ff9e
	v_add_co_u32 v73, vcc_lo, v51, s8                          // 000000004734: d7006a49 02001133
	s_wait_alu depctr_va_vcc(0)                                // 00000000473c: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, s9, v52, vcc_lo             // 000000004740: d5207c4a 01aa6809
	v_add_co_u32 v77, vcc_lo, v55, s8                          // 000000004748: d7006a4d 02001137
	s_wait_alu depctr_va_vcc(0)                                // 000000004750: bf88ff9d
	v_add_co_ci_u32_e64 v78, null, s9, v56, vcc_lo             // 000000004754: d5207c4e 01aa7009
	v_add_co_u32 v79, vcc_lo, v53, s8                          // 00000000475c: d7006a4f 02001135
	s_wait_alu depctr_va_vcc(0)                                // 000000004764: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, s9, v54, vcc_lo             // 000000004768: d5207c50 01aa6c09
	global_load_b64 v[75:76], v[73:74], off                    // 000000004770: ee05407c 0000004b 00000049
	global_load_b64 v[65:66], v[77:78], off                    // 00000000477c: ee05407c 00000041 0000004d
	s_add_nc_u64 s[6:7], s[8:9], 0x80                          // 000000004788: a986ff08 00000080
	global_load_b64 v[81:82], v[79:80], off                    // 000000004790: ee05407c 00000051 0000004f
	s_add_nc_u64 s[8:9], s[20:21], s[2:3]                      // 00000000479c: a9880214
	s_wait_loadcnt 0x1                                         // 0000000047a0: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[65:66], 0// 0000000047a4: cc464039 1a02834b
	s_wait_loadcnt 0x0                                         // 0000000047ac: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[81:82], 0// 0000000047b0: cc464041 1a02a34b
	global_load_b64 v[75:76], v[73:74], off offset:16          // 0000000047b8: ee05407c 0000004b 00001049
	s_clause 0x1                                               // 0000000047c4: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:16          // 0000000047c8: ee05407c 00000051 0000104d
	global_load_b64 v[83:84], v[79:80], off offset:16          // 0000000047d4: ee05407c 00000053 0000104f
	s_wait_loadcnt 0x1                                         // 0000000047e0: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 0000000047e4: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 0000000047ec: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 0000000047f0: cc464041 1d06a74b
	global_load_b64 v[75:76], v[73:74], off offset:32          // 0000000047f8: ee05407c 0000004b 00002049
	s_clause 0x1                                               // 000000004804: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:32          // 000000004808: ee05407c 00000051 0000204d
	global_load_b64 v[83:84], v[79:80], off offset:32          // 000000004814: ee05407c 00000053 0000204f
	s_wait_loadcnt 0x1                                         // 000000004820: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 000000004824: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 00000000482c: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 000000004830: cc464041 1d06a74b
	global_load_b64 v[75:76], v[73:74], off offset:48          // 000000004838: ee05407c 0000004b 00003049
	s_clause 0x1                                               // 000000004844: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:48          // 000000004848: ee05407c 00000051 0000304d
	global_load_b64 v[83:84], v[79:80], off offset:48          // 000000004854: ee05407c 00000053 0000304f
	s_wait_loadcnt 0x1                                         // 000000004860: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 000000004864: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 00000000486c: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 000000004870: cc464041 1d06a74b
	global_load_b64 v[75:76], v[73:74], off offset:64          // 000000004878: ee05407c 0000004b 00004049
	s_clause 0x1                                               // 000000004884: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:64          // 000000004888: ee05407c 00000051 0000404d
	global_load_b64 v[83:84], v[79:80], off offset:64          // 000000004894: ee05407c 00000053 0000404f
	s_wait_loadcnt 0x1                                         // 0000000048a0: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 0000000048a4: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 0000000048ac: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 0000000048b0: cc464041 1d06a74b
	global_load_b64 v[75:76], v[73:74], off offset:80          // 0000000048b8: ee05407c 0000004b 00005049
	s_clause 0x1                                               // 0000000048c4: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:80          // 0000000048c8: ee05407c 00000051 0000504d
	global_load_b64 v[83:84], v[79:80], off offset:80          // 0000000048d4: ee05407c 00000053 0000504f
	s_wait_loadcnt 0x1                                         // 0000000048e0: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 0000000048e4: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 0000000048ec: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 0000000048f0: cc464041 1d06a74b
	global_load_b64 v[75:76], v[73:74], off offset:96          // 0000000048f8: ee05407c 0000004b 00006049
	s_clause 0x1                                               // 000000004904: bf850001
	global_load_b64 v[81:82], v[77:78], off offset:96          // 000000004908: ee05407c 00000051 0000604d
	global_load_b64 v[83:84], v[79:80], off offset:96          // 000000004914: ee05407c 00000053 0000604f
	s_wait_loadcnt 0x1                                         // 000000004920: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[75:76], v[81:82], v[57:64]// 000000004924: cc464039 1ce6a34b
	s_wait_loadcnt 0x0                                         // 00000000492c: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[75:76], v[83:84], v[65:72]// 000000004930: cc464041 1d06a74b
	global_load_b64 v[73:74], v[73:74], off offset:112         // 000000004938: ee05407c 00000049 00007049
	s_clause 0x1                                               // 000000004944: bf850001
	global_load_b64 v[75:76], v[77:78], off offset:112         // 000000004948: ee05407c 0000004b 0000704d
	global_load_b64 v[77:78], v[79:80], off offset:112         // 000000004954: ee05407c 0000004d 0000704f
	s_wait_loadcnt 0x1                                         // 000000004960: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[73:74], v[75:76], v[57:64]// 000000004964: cc464039 1ce69749
	s_wait_loadcnt 0x0                                         // 00000000496c: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[73:74], v[77:78], v[65:72]// 000000004970: cc464041 1d069b49
	v_add_co_u32 v73, vcc_lo, s18, v34                         // 000000004978: d7006a49 02024412
	global_load_b32 v75, v21, s[8:9]                           // 000000004980: ee050008 0000004b 00000015
	s_wait_alu depctr_va_vcc(0)                                // 00000000498c: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, s19, v35, vcc_lo            // 000000004990: d5207c4a 01aa4613
	s_load_b32 s8, s[20:21], 0x0                               // 000000004998: f400020a f8000000
	s_add_nc_u64 s[20:21], s[20:21], s[4:5]                    // 0000000049a0: a9940414
	global_load_b32 v76, v[73:74], off                         // 0000000049a4: ee05007c 0000004c 00000049
	s_wait_loadcnt 0x1                                         // 0000000049b0: bfc00001
	s_wait_kmcnt 0x0                                           // 0000000049b4: bfc70000
	v_cndmask_b32_e64 v77, s8, v75, s0                         // 0000000049b8: d501004d 00029608
	s_wait_loadcnt 0x0                                         // 0000000049c0: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000049c4: bf870091
	v_mul_f32_e32 v73, v76, v77                                // 0000000049c8: 10929b4c
	v_mul_f32_e32 v57, v57, v73                                // 0000000049cc: 10729339
	v_add_co_u32 v73, vcc_lo, s18, v32                         // 0000000049d0: d7006a49 02024012
	s_wait_alu depctr_va_vcc(0)                                // 0000000049d8: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, s19, v33, vcc_lo            // 0000000049dc: d5207c4a 01aa4213
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 0000000049e4: bf8700c3
	v_add_f32_e32 v50, v50, v57                                // 0000000049e8: 06647332
	global_load_b32 v73, v[73:74], off                         // 0000000049ec: ee05007c 00000049 00000049
	s_wait_loadcnt 0x0                                         // 0000000049f8: bfc00000
	v_mul_f32_e32 v57, v77, v73                                // 0000000049fc: 1072934d
	v_mul_f32_e32 v57, v58, v57                                // 000000004a00: 1072733a
	s_delay_alu instid0(valu_dep_1)                            // 000000004a04: bf870001
	v_add_f32_e32 v49, v49, v57                                // 000000004a08: 06627331
	v_add_co_u32 v57, vcc_lo, s18, v30                         // 000000004a0c: d7006a39 02023c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004a14: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v31, vcc_lo            // 000000004a18: d5207c3a 01aa3e13
	global_load_b32 v74, v[57:58], off                         // 000000004a20: ee05007c 0000004a 00000039
	s_wait_loadcnt 0x0                                         // 000000004a2c: bfc00000
	v_mul_f32_e32 v57, v77, v74                                // 000000004a30: 1072954d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004a34: bf870091
	v_mul_f32_e32 v57, v59, v57                                // 000000004a38: 1072733b
	v_add_f32_e32 v48, v48, v57                                // 000000004a3c: 06607330
	v_add_co_u32 v57, vcc_lo, s18, v28                         // 000000004a40: d7006a39 02023812
	s_wait_alu depctr_va_vcc(0)                                // 000000004a48: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v29, vcc_lo            // 000000004a4c: d5207c3a 01aa3a13
	global_load_b32 v59, v[57:58], off                         // 000000004a54: ee05007c 0000003b 00000039
	s_wait_loadcnt 0x0                                         // 000000004a60: bfc00000
	v_mul_f32_e32 v57, v77, v59                                // 000000004a64: 1072774d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004a68: bf870091
	v_mul_f32_e32 v57, v60, v57                                // 000000004a6c: 1072733c
	v_add_f32_e32 v47, v47, v57                                // 000000004a70: 065e732f
	v_add_co_u32 v57, vcc_lo, s18, v26                         // 000000004a74: d7006a39 02023412
	s_wait_alu depctr_va_vcc(0)                                // 000000004a7c: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v27, vcc_lo            // 000000004a80: d5207c3a 01aa3613
	global_load_b32 v60, v[57:58], off                         // 000000004a88: ee05007c 0000003c 00000039
	s_wait_loadcnt 0x0                                         // 000000004a94: bfc00000
	v_mul_f32_e32 v57, v77, v60                                // 000000004a98: 1072794d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004a9c: bf870091
	v_mul_f32_e32 v57, v61, v57                                // 000000004aa0: 1072733d
	v_add_f32_e32 v46, v46, v57                                // 000000004aa4: 065c732e
	v_add_co_u32 v57, vcc_lo, s18, v24                         // 000000004aa8: d7006a39 02023012
	s_wait_alu depctr_va_vcc(0)                                // 000000004ab0: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v25, vcc_lo            // 000000004ab4: d5207c3a 01aa3213
	global_load_b32 v61, v[57:58], off                         // 000000004abc: ee05007c 0000003d 00000039
	s_wait_loadcnt 0x0                                         // 000000004ac8: bfc00000
	v_mul_f32_e32 v57, v77, v61                                // 000000004acc: 10727b4d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004ad0: bf870091
	v_mul_f32_e32 v57, v62, v57                                // 000000004ad4: 1072733e
	v_add_f32_e32 v45, v45, v57                                // 000000004ad8: 065a732d
	v_add_co_u32 v57, vcc_lo, s18, v22                         // 000000004adc: d7006a39 02022c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004ae4: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v23, vcc_lo            // 000000004ae8: d5207c3a 01aa2e13
	global_load_b32 v62, v[57:58], off                         // 000000004af0: ee05007c 0000003e 00000039
	s_wait_loadcnt 0x0                                         // 000000004afc: bfc00000
	v_mul_f32_e32 v57, v77, v62                                // 000000004b00: 10727d4d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004b04: bf870091
	v_mul_f32_e32 v57, v63, v57                                // 000000004b08: 1072733f
	v_add_f32_e32 v44, v44, v57                                // 000000004b0c: 0658732c
	v_add_co_u32 v57, vcc_lo, s18, v18                         // 000000004b10: d7006a39 02022412
	s_wait_alu depctr_va_vcc(0)                                // 000000004b18: bf88ff9d
	v_add_co_ci_u32_e64 v58, null, s19, v19, vcc_lo            // 000000004b1c: d5207c3a 01aa2613
	s_add_nc_u64 s[18:19], s[18:19], 4                         // 000000004b24: a9928412
	global_load_b32 v57, v[57:58], off                         // 000000004b28: ee05007c 00000039 00000039
	s_wait_loadcnt 0x0                                         // 000000004b34: bfc00000
	v_mul_f32_e32 v58, v77, v57                                // 000000004b38: 1074734d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004b3c: bf870091
	v_mul_f32_e32 v58, v64, v58                                // 000000004b40: 10747540
	v_add_f32_e32 v42, v42, v58                                // 000000004b44: 0654752a
	v_cndmask_b32_e64 v58, s8, v75, s1                         // 000000004b48: d501003a 00069608
	v_cmp_lt_i64_e64 s8, s[6:7], s[22:23]                      // 000000004b50: d4510008 02002c06
	s_delay_alu instid0(valu_dep_2)                            // 000000004b58: bf870002
	v_mul_f32_e32 v59, v58, v59                                // 000000004b5c: 1076773a
	v_mul_f32_e32 v63, v76, v58                                // 000000004b60: 107e754c
	v_mul_f32_e32 v57, v58, v57                                // 000000004b64: 1072733a
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000004b68: 8b6a087e
	s_mov_b64 s[8:9], s[6:7]                                   // 000000004b6c: be880106
	v_mul_f32_e32 v59, v68, v59                                // 000000004b70: 10767744
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b74: bf8701a2
	v_mul_f32_e32 v57, v72, v57                                // 000000004b78: 10727348
	v_mul_f32_e32 v63, v65, v63                                // 000000004b7c: 107e7f41
	v_add_f32_e32 v39, v39, v59                                // 000000004b80: 064e7727
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b84: bf870193
	v_dual_mul_f32 v59, v58, v60 :: v_dual_add_f32 v20, v20, v57// 000000004b88: c8c8793a 3b147314
	v_add_f32_e32 v43, v43, v63                                // 000000004b90: 06567f2b
	v_mul_f32_e32 v63, v58, v73                                // 000000004b94: 107e933a
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000004b98: bf870113
	v_mul_f32_e32 v59, v69, v59                                // 000000004b9c: 10767745
	v_mul_f32_e32 v63, v66, v63                                // 000000004ba0: 107e7f42
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ba4: bf8701a2
	v_add_f32_e32 v38, v38, v59                                // 000000004ba8: 064c7726
	v_mul_f32_e32 v59, v58, v61                                // 000000004bac: 10767b3a
	v_add_f32_e32 v41, v41, v63                                // 000000004bb0: 06527f29
	v_mul_f32_e32 v63, v58, v74                                // 000000004bb4: 107e953a
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000004bb8: bf870113
	v_mul_f32_e32 v59, v70, v59                                // 000000004bbc: 10767746
	v_mul_f32_e32 v63, v67, v63                                // 000000004bc0: 107e7f43
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000004bc4: bf870112
	v_add_f32_e32 v37, v37, v59                                // 000000004bc8: 064a7725
	v_dual_mul_f32 v59, v58, v62 :: v_dual_add_f32 v40, v40, v63// 000000004bcc: c8c87d3a 3b287f28
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004bd4: bf870091
	v_mul_f32_e32 v59, v71, v59                                // 000000004bd8: 10767747
	v_add_f32_e32 v36, v36, v59                                // 000000004bdc: 06487724
	s_wait_alu depctr_sa_sdst(0)                               // 000000004be0: bf88ff9e
	s_cbranch_vccnz 65234                                      // 000000004be4: bfa4fed2 <tessera_rocm_scaled_matmul_9160add00a4b6aae+0x2c30>
	v_mul_lo_u32 v18, s15, v12                                 // 000000004be8: d72c0012 0202180f
	v_mul_lo_u32 v19, s14, v13                                 // 000000004bf0: d72c0013 02021a0e
	v_mad_co_u64_u32 v[12:13], null, s14, v12, 0               // 000000004bf8: d6fe7c0c 0202180e
	v_mul_lo_u32 v21, s15, v14                                 // 000000004c00: d72c0015 02021c0f
	v_mul_lo_u32 v22, s14, v15                                 // 000000004c08: d72c0016 02021e0e
	v_mad_co_u64_u32 v[14:15], null, s14, v14, 0               // 000000004c10: d6fe7c0e 02021c0e
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000004c18: 3e202081
	v_mul_lo_u32 v24, s15, v10                                 // 000000004c1c: d72c0018 0202140f
	v_bfe_u32 v23, v49, 16, 1                                  // 000000004c24: d6100017 02052131
	v_add3_u32 v13, v13, v19, v18                              // 000000004c2c: d655000d 044a270d
	v_bfe_u32 v18, v50, 16, 1                                  // 000000004c34: d6100012 02052132
	v_or_b32_e32 v19, 0x400000, v50                            // 000000004c3c: 382664ff 00400000
	v_add3_u32 v15, v15, v22, v21                              // 000000004c44: d655000f 04562d0f
	v_add3_u32 v21, v23, v49, 0x7fff                           // 000000004c4c: d6550015 03fe6317 00007fff
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000004c58: 3e181881
	v_add3_u32 v18, v18, v50, 0x7fff                           // 000000004c5c: d6550012 03fe6512 00007fff
	v_or_b32_e32 v22, 0x400000, v49                            // 000000004c68: 382c62ff 00400000
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000004c70: 3e1c1c81
	v_mul_lo_u32 v23, s14, v7                                  // 000000004c74: d72c0017 02020e0e
	v_add_co_u32 v12, vcc_lo, s16, v12                         // 000000004c7c: d7006a0c 02021810
	s_wait_alu depctr_va_vcc(0)                                // 000000004c84: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo            // 000000004c88: d5207c0d 01aa1a11
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000004c90: 7c306532
	s_wait_alu depctr_va_vcc(0)                                // 000000004c94: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 000000004c98: 02242712
	v_mul_lo_u32 v19, s14, v11                                 // 000000004c9c: d72c0013 0202160e
	v_mad_co_u64_u32 v[10:11], null, s14, v10, 0               // 000000004ca4: d6fe7c0a 0202140e
	v_add_co_u32 v12, vcc_lo, v12, v16                         // 000000004cac: d7006a0c 0202210c
	s_wait_alu depctr_va_vcc(0)                                // 000000004cb4: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v13, v17, vcc_lo            // 000000004cb8: d5207c0d 01aa230d
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 000000004cc0: 7c306331
	s_delay_alu instid0(valu_dep_4)                            // 000000004cc4: bf870004
	v_add3_u32 v11, v11, v19, v24                              // 000000004cc8: d655000b 0462270b
	global_store_d16_hi_b16 v[12:13], v18, off                 // 000000004cd0: ee09407c 09000000 0000000c
	s_wait_alu depctr_va_vcc(0)                                // 000000004cdc: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v22, vcc_lo                    // 000000004ce0: 02242d15
	v_add_co_u32 v14, vcc_lo, s16, v14                         // 000000004ce4: d7006a0e 02021c10
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000004cec: 3e141481
	s_wait_alu depctr_va_vcc(0)                                // 000000004cf0: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, s17, v15, vcc_lo            // 000000004cf4: d5207c0f 01aa1e11
	v_bfe_u32 v19, v48, 16, 1                                  // 000000004cfc: d6100013 02052130
	v_add_co_u32 v14, vcc_lo, v14, v16                         // 000000004d04: d7006a0e 0202210e
	v_mul_lo_u32 v22, s15, v6                                  // 000000004d0c: d72c0016 02020c0f
	v_mad_co_u64_u32 v[6:7], null, s14, v6, 0                  // 000000004d14: d6fe7c06 02020c0e
	s_wait_alu depctr_va_vcc(0)                                // 000000004d1c: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v15, v17, vcc_lo            // 000000004d20: d5207c0f 01aa230f
	v_add_co_u32 v10, vcc_lo, s16, v10                         // 000000004d28: d7006a0a 02021410
	v_add3_u32 v19, v19, v48, 0x7fff                           // 000000004d30: d6550013 03fe6113 00007fff
	v_or_b32_e32 v21, 0x400000, v48                            // 000000004d3c: 382a60ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004d44: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc_lo            // 000000004d48: d5207c0b 01aa1611
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000004d50: 7c306130
	v_add3_u32 v7, v7, v23, v22                                // 000000004d54: d6550007 045a2f07
	v_mul_lo_u32 v22, s15, v8                                  // 000000004d5c: d72c0016 0202100f
	v_mul_lo_u32 v23, s14, v9                                  // 000000004d64: d72c0017 0202120e
	v_mad_co_u64_u32 v[8:9], null, s14, v8, 0                  // 000000004d6c: d6fe7c08 0202100e
	s_wait_alu depctr_va_vcc(0)                                // 000000004d74: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004d78: 02262b13
	v_bfe_u32 v21, v47, 16, 1                                  // 000000004d7c: d6100015 0205212f
	v_add_co_u32 v10, vcc_lo, v10, v16                         // 000000004d84: d7006a0a 0202210a
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000004d8c: 3e0c0c81
	s_wait_alu depctr_va_vcc(0)                                // 000000004d90: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v11, v17, vcc_lo            // 000000004d94: d5207c0b 01aa230b
	v_add3_u32 v21, v21, v47, 0x7fff                           // 000000004d9c: d6550015 03fe5f15 00007fff
	v_or_b32_e32 v24, 0x400000, v47                            // 000000004da8: 38305eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000004db0: 7c305f2f
	v_add3_u32 v9, v9, v23, v22                                // 000000004db4: d6550009 045a2f09
	s_clause 0x1                                               // 000000004dbc: bf850001
	global_store_d16_hi_b16 v[14:15], v18, off                 // 000000004dc0: ee09407c 09000000 0000000e
	global_store_d16_hi_b16 v[10:11], v19, off                 // 000000004dcc: ee09407c 09800000 0000000a
	v_bfe_u32 v22, v46, 16, 1                                  // 000000004dd8: d6100016 0205212e
	s_wait_alu depctr_va_vcc(0)                                // 000000004de0: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v24, vcc_lo                    // 000000004de4: 02243115
	v_add_co_u32 v19, vcc_lo, s16, v6                          // 000000004de8: d7006a13 02020c10
	s_wait_alu depctr_va_vcc(0)                                // 000000004df0: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, s17, v7, vcc_lo             // 000000004df4: d5207c15 01aa0e11
	v_lshlrev_b64_e32 v[6:7], 1, v[8:9]                        // 000000004dfc: 3e0c1081
	s_delay_alu instid0(valu_dep_3)                            // 000000004e00: bf870003
	v_add_co_u32 v8, vcc_lo, v19, v16                          // 000000004e04: d7006a08 02022113
	v_add3_u32 v19, v22, v46, 0x7fff                           // 000000004e0c: d6550013 03fe5d16 00007fff
	v_mul_lo_u32 v22, s15, v4                                  // 000000004e18: d72c0016 0202080f
	v_mul_lo_u32 v23, s14, v5                                  // 000000004e20: d72c0017 02020a0e
	v_mad_co_u64_u32 v[4:5], null, s14, v4, 0                  // 000000004e28: d6fe7c04 0202080e
	s_wait_alu depctr_va_vcc(0)                                // 000000004e30: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v21, v17, vcc_lo             // 000000004e34: d5207c09 01aa2315
	v_add_co_u32 v6, vcc_lo, s16, v6                           // 000000004e3c: d7006a06 02020c10
	v_or_b32_e32 v21, 0x400000, v46                            // 000000004e44: 382a5cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e4c: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo              // 000000004e50: d5207c07 01aa0e11
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000004e58: 7c305d2e
	v_add3_u32 v5, v5, v23, v22                                // 000000004e5c: d6550005 045a2f05
	v_mul_lo_u32 v22, s15, v2                                  // 000000004e64: d72c0016 0202040f
	v_mul_lo_u32 v23, s14, v3                                  // 000000004e6c: d72c0017 0202060e
	v_mad_co_u64_u32 v[2:3], null, s14, v2, 0                  // 000000004e74: d6fe7c02 0202040e
	s_wait_alu depctr_va_vcc(0)                                // 000000004e7c: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004e80: 02262b13
	v_bfe_u32 v21, v45, 16, 1                                  // 000000004e84: d6100015 0205212d
	v_add_co_u32 v6, vcc_lo, v6, v16                           // 000000004e8c: d7006a06 02022106
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004e94: 3e080881
	s_wait_alu depctr_va_vcc(0)                                // 000000004e98: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v17, vcc_lo              // 000000004e9c: d5207c07 01aa2307
	v_add3_u32 v21, v21, v45, 0x7fff                           // 000000004ea4: d6550015 03fe5b15 00007fff
	v_or_b32_e32 v24, 0x400000, v45                            // 000000004eb0: 38305aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000004eb8: 7c305b2d
	s_clause 0x1                                               // 000000004ebc: bf850001
	global_store_d16_hi_b16 v[8:9], v18, off                   // 000000004ec0: ee09407c 09000000 00000008
	global_store_d16_hi_b16 v[6:7], v19, off                   // 000000004ecc: ee09407c 09800000 00000006
	v_add3_u32 v3, v3, v23, v22                                // 000000004ed8: d6550003 045a2f03
	v_bfe_u32 v19, v44, 16, 1                                  // 000000004ee0: d6100013 0205212c
	v_mul_lo_u32 v22, s15, v0                                  // 000000004ee8: d72c0016 0202000f
	s_wait_alu depctr_va_vcc(0)                                // 000000004ef0: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v24, vcc_lo                    // 000000004ef4: 02243115
	v_add_co_u32 v4, vcc_lo, s16, v4                           // 000000004ef8: d7006a04 02020810
	s_wait_alu depctr_va_vcc(0)                                // 000000004f00: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo              // 000000004f04: d5207c05 01aa0a11
	v_mul_lo_u32 v23, s14, v1                                  // 000000004f0c: d72c0017 0202020e
	v_mad_co_u64_u32 v[0:1], null, s14, v0, 0                  // 000000004f14: d6fe7c00 0202000e
	v_add_co_u32 v4, vcc_lo, v4, v16                           // 000000004f1c: d7006a04 02022104
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004f24: 3e040481
	v_add3_u32 v19, v19, v44, 0x7fff                           // 000000004f28: d6550013 03fe5913 00007fff
	v_or_b32_e32 v21, 0x400000, v44                            // 000000004f34: 382a58ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f3c: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v5, v17, vcc_lo              // 000000004f40: d5207c05 01aa2305
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000004f48: 7c30592c
	v_add3_u32 v1, v1, v23, v22                                // 000000004f4c: d6550001 045a2f01
	v_or_b32_e32 v22, 0x400000, v42                            // 000000004f54: 382c54ff 00400000
	global_store_d16_hi_b16 v[4:5], v18, off                   // 000000004f5c: ee09407c 09000000 00000004
	s_wait_alu depctr_va_vcc(0)                                // 000000004f68: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v21, vcc_lo                    // 000000004f6c: 02262b13
	v_add_co_u32 v2, vcc_lo, s16, v2                           // 000000004f70: d7006a02 02020410
	s_wait_alu depctr_va_vcc(0)                                // 000000004f78: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo              // 000000004f7c: d5207c03 01aa0611
	v_bfe_u32 v21, v42, 16, 1                                  // 000000004f84: d6100015 0205212a
	s_delay_alu instid0(valu_dep_3)                            // 000000004f8c: bf870003
	v_add_co_u32 v2, vcc_lo, v2, v16                           // 000000004f90: d7006a02 02022102
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004f98: 3e000081
	s_wait_alu depctr_va_vcc(0)                                // 000000004f9c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v17, vcc_lo              // 000000004fa0: d5207c03 01aa2303
	v_add3_u32 v21, v21, v42, 0x7fff                           // 000000004fa8: d6550015 03fe5515 00007fff
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000004fb4: 7c30552a
	global_store_d16_hi_b16 v[2:3], v19, off                   // 000000004fb8: ee09407c 09800000 00000002
	v_bfe_u32 v19, v43, 16, 1                                  // 000000004fc4: d6100013 0205212b
	s_wait_alu depctr_va_vcc(0)                                // 000000004fcc: bf88ff9d
	v_cndmask_b32_e32 v18, v21, v22, vcc_lo                    // 000000004fd0: 02242d15
	v_add_co_u32 v0, vcc_lo, s16, v0                           // 000000004fd4: d7006a00 02020010
	s_wait_alu depctr_va_vcc(0)                                // 000000004fdc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo              // 000000004fe0: d5207c01 01aa0211
	v_add3_u32 v19, v19, v43, 0x7fff                           // 000000004fe8: d6550013 03fe5713 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004ff4: bf870003
	v_add_co_u32 v0, vcc_lo, v0, v16                           // 000000004ff8: d7006a00 02022100
	v_or_b32_e32 v21, 0x400000, v43                            // 000000005000: 382a56ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005008: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v17, vcc_lo              // 00000000500c: d5207c01 01aa2301
	v_bfe_u32 v16, v41, 16, 1                                  // 000000005014: d6100010 02052129
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 00000000501c: 7c30572b
	global_store_d16_hi_b16 v[0:1], v18, off                   // 000000005020: ee09407c 09000000 00000000
	v_or_b32_e32 v18, 0x400000, v41                            // 00000000502c: 382452ff 00400000
	v_add3_u32 v16, v16, v41, 0x7fff                           // 000000005034: d6550010 03fe5310 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005040: bf88ff9d
	v_cndmask_b32_e32 v17, v19, v21, vcc_lo                    // 000000005044: 02222b13
	v_bfe_u32 v19, v40, 16, 1                                  // 000000005048: d6100013 02052128
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000005050: 7c305329
	global_store_d16_hi_b16 v[12:13], v17, off offset:32       // 000000005054: ee09407c 08800000 0000200c
	v_add3_u32 v12, v19, v40, 0x7fff                           // 000000005060: d655000c 03fe5113 00007fff
	v_or_b32_e32 v13, 0x400000, v40                            // 00000000506c: 381a50ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005074: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v18, vcc_lo                    // 000000005078: 02202510
	v_bfe_u32 v17, v39, 16, 1                                  // 00000000507c: d6100011 02052127
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000005084: 7c305128
	global_store_d16_hi_b16 v[14:15], v16, off offset:32       // 000000005088: ee09407c 08000000 0000200e
	v_add3_u32 v14, v17, v39, 0x7fff                           // 000000005094: d655000e 03fe4f11 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000050a0: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo                    // 0000000050a4: 02181b0c
	v_bfe_u32 v13, v38, 16, 1                                  // 0000000050a8: d610000d 02052126
	v_or_b32_e32 v15, 0x400000, v39                            // 0000000050b0: 381e4eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 0000000050b8: 7c304f27
	v_or_b32_e32 v16, 0x400000, v36                            // 0000000050bc: 382048ff 00400000
	global_store_d16_hi_b16 v[10:11], v12, off offset:32       // 0000000050c4: ee09407c 06000000 0000200a
	v_add3_u32 v10, v13, v38, 0x7fff                           // 0000000050d0: d655000a 03fe4d0d 00007fff
	v_or_b32_e32 v11, 0x400000, v38                            // 0000000050dc: 38164cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000050e4: bf88ff9d
	v_cndmask_b32_e32 v12, v14, v15, vcc_lo                    // 0000000050e8: 02181f0e
	v_bfe_u32 v13, v37, 16, 1                                  // 0000000050ec: d610000d 02052125
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 0000000050f4: 7c304d26
	v_bfe_u32 v14, v36, 16, 1                                  // 0000000050f8: d610000e 02052124
	v_or_b32_e32 v15, 0x400000, v37                            // 000000005100: 381e4aff 00400000
	v_or_b32_e32 v17, 0x400000, v20                            // 000000005108: 382228ff 00400000
	v_add3_u32 v13, v13, v37, 0x7fff                           // 000000005110: d655000d 03fe4b0d 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000511c: bf88ff9d
	v_cndmask_b32_e32 v10, v10, v11, vcc_lo                    // 000000005120: 0214170a
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000005124: 7c304b25
	v_bfe_u32 v11, v20, 16, 1                                  // 000000005128: d610000b 02052114
	v_add3_u32 v14, v14, v36, 0x7fff                           // 000000005130: d655000e 03fe490e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000513c: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v15, vcc_lo                    // 000000005140: 021a1f0d
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000005144: 7c304924
	v_add3_u32 v11, v11, v20, 0x7fff                           // 000000005148: d655000b 03fe290b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005154: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo                    // 000000005158: 021c210e
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 00000000515c: 7c302914
	s_wait_alu depctr_va_vcc(0)                                // 000000005160: bf88ff9d
	v_cndmask_b32_e32 v11, v11, v17, vcc_lo                    // 000000005164: 0216230b
	s_clause 0x4                                               // 000000005168: bf850004
	global_store_d16_hi_b16 v[8:9], v12, off offset:32         // 00000000516c: ee09407c 06000000 00002008
	global_store_d16_hi_b16 v[6:7], v10, off offset:32         // 000000005178: ee09407c 05000000 00002006
	global_store_d16_hi_b16 v[4:5], v13, off offset:32         // 000000005184: ee09407c 06800000 00002004
	global_store_d16_hi_b16 v[2:3], v14, off offset:32         // 000000005190: ee09407c 07000000 00002002
	global_store_d16_hi_b16 v[0:1], v11, off offset:32         // 00000000519c: ee09407c 05800000 00002000
	s_endpgm                                                   // 0000000051a8: bfb00000
	s_code_end                                                 // 0000000051ac: bf9f0000
	s_code_end                                                 // 0000000051b0: bf9f0000
	s_code_end                                                 // 0000000051b4: bf9f0000
	s_code_end                                                 // 0000000051b8: bf9f0000
	s_code_end                                                 // 0000000051bc: bf9f0000
	s_code_end                                                 // 0000000051c0: bf9f0000
	s_code_end                                                 // 0000000051c4: bf9f0000
	s_code_end                                                 // 0000000051c8: bf9f0000
	s_code_end                                                 // 0000000051cc: bf9f0000
	s_code_end                                                 // 0000000051d0: bf9f0000
	s_code_end                                                 // 0000000051d4: bf9f0000
	s_code_end                                                 // 0000000051d8: bf9f0000
	s_code_end                                                 // 0000000051dc: bf9f0000
	s_code_end                                                 // 0000000051e0: bf9f0000
	s_code_end                                                 // 0000000051e4: bf9f0000
	s_code_end                                                 // 0000000051e8: bf9f0000
	s_code_end                                                 // 0000000051ec: bf9f0000
	s_code_end                                                 // 0000000051f0: bf9f0000
	s_code_end                                                 // 0000000051f4: bf9f0000
	s_code_end                                                 // 0000000051f8: bf9f0000
	s_code_end                                                 // 0000000051fc: bf9f0000
	s_code_end                                                 // 000000005200: bf9f0000
	s_code_end                                                 // 000000005204: bf9f0000
	s_code_end                                                 // 000000005208: bf9f0000
	s_code_end                                                 // 00000000520c: bf9f0000
	s_code_end                                                 // 000000005210: bf9f0000
	s_code_end                                                 // 000000005214: bf9f0000
	s_code_end                                                 // 000000005218: bf9f0000
	s_code_end                                                 // 00000000521c: bf9f0000
	s_code_end                                                 // 000000005220: bf9f0000
	s_code_end                                                 // 000000005224: bf9f0000
	s_code_end                                                 // 000000005228: bf9f0000
	s_code_end                                                 // 00000000522c: bf9f0000
	s_code_end                                                 // 000000005230: bf9f0000
	s_code_end                                                 // 000000005234: bf9f0000
	s_code_end                                                 // 000000005238: bf9f0000
	s_code_end                                                 // 00000000523c: bf9f0000
	s_code_end                                                 // 000000005240: bf9f0000
	s_code_end                                                 // 000000005244: bf9f0000
	s_code_end                                                 // 000000005248: bf9f0000
	s_code_end                                                 // 00000000524c: bf9f0000
	s_code_end                                                 // 000000005250: bf9f0000
	s_code_end                                                 // 000000005254: bf9f0000
	s_code_end                                                 // 000000005258: bf9f0000
	s_code_end                                                 // 00000000525c: bf9f0000
	s_code_end                                                 // 000000005260: bf9f0000
	s_code_end                                                 // 000000005264: bf9f0000
	s_code_end                                                 // 000000005268: bf9f0000
	s_code_end                                                 // 00000000526c: bf9f0000
	s_code_end                                                 // 000000005270: bf9f0000
	s_code_end                                                 // 000000005274: bf9f0000
	s_code_end                                                 // 000000005278: bf9f0000
	s_code_end                                                 // 00000000527c: bf9f0000
	s_code_end                                                 // 000000005280: bf9f0000
	s_code_end                                                 // 000000005284: bf9f0000
	s_code_end                                                 // 000000005288: bf9f0000
	s_code_end                                                 // 00000000528c: bf9f0000
	s_code_end                                                 // 000000005290: bf9f0000
	s_code_end                                                 // 000000005294: bf9f0000
	s_code_end                                                 // 000000005298: bf9f0000
	s_code_end                                                 // 00000000529c: bf9f0000
	s_code_end                                                 // 0000000052a0: bf9f0000
	s_code_end                                                 // 0000000052a4: bf9f0000
	s_code_end                                                 // 0000000052a8: bf9f0000
	s_code_end                                                 // 0000000052ac: bf9f0000
	s_code_end                                                 // 0000000052b0: bf9f0000
	s_code_end                                                 // 0000000052b4: bf9f0000
	s_code_end                                                 // 0000000052b8: bf9f0000
	s_code_end                                                 // 0000000052bc: bf9f0000
	s_code_end                                                 // 0000000052c0: bf9f0000
	s_code_end                                                 // 0000000052c4: bf9f0000
	s_code_end                                                 // 0000000052c8: bf9f0000
	s_code_end                                                 // 0000000052cc: bf9f0000
	s_code_end                                                 // 0000000052d0: bf9f0000
	s_code_end                                                 // 0000000052d4: bf9f0000
	s_code_end                                                 // 0000000052d8: bf9f0000
	s_code_end                                                 // 0000000052dc: bf9f0000
	s_code_end                                                 // 0000000052e0: bf9f0000
	s_code_end                                                 // 0000000052e4: bf9f0000
	s_code_end                                                 // 0000000052e8: bf9f0000
	s_code_end                                                 // 0000000052ec: bf9f0000
	s_code_end                                                 // 0000000052f0: bf9f0000
	s_code_end                                                 // 0000000052f4: bf9f0000
	s_code_end                                                 // 0000000052f8: bf9f0000
	s_code_end                                                 // 0000000052fc: bf9f0000
	s_code_end                                                 // 000000005300: bf9f0000
	s_code_end                                                 // 000000005304: bf9f0000
	s_code_end                                                 // 000000005308: bf9f0000
	s_code_end                                                 // 00000000530c: bf9f0000
	s_code_end                                                 // 000000005310: bf9f0000
	s_code_end                                                 // 000000005314: bf9f0000
	s_code_end                                                 // 000000005318: bf9f0000
	s_code_end                                                 // 00000000531c: bf9f0000
	s_code_end                                                 // 000000005320: bf9f0000
	s_code_end                                                 // 000000005324: bf9f0000
	s_code_end                                                 // 000000005328: bf9f0000
	s_code_end                                                 // 00000000532c: bf9f0000
	s_code_end                                                 // 000000005330: bf9f0000
	s_code_end                                                 // 000000005334: bf9f0000
	s_code_end                                                 // 000000005338: bf9f0000
	s_code_end                                                 // 00000000533c: bf9f0000
	s_code_end                                                 // 000000005340: bf9f0000
	s_code_end                                                 // 000000005344: bf9f0000
	s_code_end                                                 // 000000005348: bf9f0000
	s_code_end                                                 // 00000000534c: bf9f0000
	s_code_end                                                 // 000000005350: bf9f0000
	s_code_end                                                 // 000000005354: bf9f0000
	s_code_end                                                 // 000000005358: bf9f0000
	s_code_end                                                 // 00000000535c: bf9f0000
	s_code_end                                                 // 000000005360: bf9f0000
	s_code_end                                                 // 000000005364: bf9f0000
	s_code_end                                                 // 000000005368: bf9f0000
	s_code_end                                                 // 00000000536c: bf9f0000
	s_code_end                                                 // 000000005370: bf9f0000
	s_code_end                                                 // 000000005374: bf9f0000
	s_code_end                                                 // 000000005378: bf9f0000
	s_code_end                                                 // 00000000537c: bf9f0000
