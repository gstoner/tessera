
/tmp/tmpwd8gwqdp.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b04: f4004400 f80000c8
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000001b0c: f4002500 f80000a8
	s_load_b64 s[28:29], s[0:1], 0xd8                          // 000000001b14: f4002700 f80000d8
	s_load_b64 s[26:27], s[0:1], 0x8                           // 000000001b1c: f4002680 f8000008
	s_load_b64 s[30:31], s[0:1], 0x30                          // 000000001b24: f4002780 f8000030
	s_load_b64 s[22:23], s[0:1], 0x58                          // 000000001b2c: f4002580 f8000058
	s_load_b64 s[34:35], s[0:1], 0x80                          // 000000001b34: f4002880 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b40: 86039f75
	s_mov_b32 s4, ttmp7                                        // 000000001b44: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[38:39], s[2:3], 5                             // 000000001b4c: 84a68502
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000001b50: bf8700c9
	v_dual_mov_b32 v33, s39 :: v_dual_and_b32 v54, 15, v0      // 000000001b54: ca240027 2136008f
	s_lshl_b64 s[36:37], s[4:5], 5                             // 000000001b5c: 84a48504
	s_add_nc_u64 s[2:3], s[38:39], 32                          // 000000001b60: a982a026
	s_add_nc_u64 s[0:1], s[36:37], 32                          // 000000001b64: a980a024
	v_or_b32_e32 v32, s38, v54                                 // 000000001b68: 38406c26
	v_mov_b32_e32 v35, s39                                     // 000000001b6c: 7e460227
	v_bfe_u32 v55, v0, 4, 1                                    // 000000001b70: d6100037 02050900
	s_or_b32 s33, s36, 16                                      // 000000001b78: 8c219024
	s_delay_alu instid0(valu_dep_3)                            // 000000001b7c: bf870003
	v_or_b32_e32 v34, 16, v32                                  // 000000001b80: 38444090
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[16:17]                      // 000000001b88: d4540000 02002000
	v_cmp_gt_i64_e64 s1, s[2:3], s[18:19]                      // 000000001b90: d4540001 02002402
	v_cmp_lt_i64_e64 s40, s[28:29], 32                         // 000000001b98: d4510028 0201401c
	s_and_b32 s24, s28, 0xffffffe0                             // 000000001ba0: 8b18ff1c ffffffe0
	s_mov_b32 s25, s29                                         // 000000001ba8: be99001d
	s_or_b32 s0, s0, s1                                        // 000000001bac: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb4: 8b6a007e
	s_mov_b32 s0, -1                                           // 000000001bb8: be8000c1
	s_cbranch_vccz 3663                                        // 000000001bbc: bfa30e4f <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x39fc>
	s_and_b32 s0, s40, exec_lo                                 // 000000001bc0: 8b007e28
	s_cselect_b32 s0, 1, 0                                     // 000000001bc4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bcc: bf078100
	s_cbranch_scc1 7                                           // 000000001bd0: bfa20007 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0xf0>
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v36, 3, v55    // 000000001bd4: ca220080 25246e83
	v_mov_b32_e32 v39, s37                                     // 000000001bdc: 7e4e0225
	s_mov_b32 s0, 0                                            // 000000001be0: be800080
	s_delay_alu instid0(valu_dep_2)                            // 000000001be4: bf870002
	v_or_b32_e32 v38, s36, v36                                 // 000000001be8: 384c4824
	s_branch 1                                                 // 000000001bec: bfa00001 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0xf4>
	s_mov_b32 s0, -1                                           // 000000001bf0: be8000c1
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v81, 0             // 000000001bf4: ca100080 50500080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bfc: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001c00: 8b007e00
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v87, 0             // 000000001c04: ca100080 54560080
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v95, 0             // 000000001c0c: ca100080 5a5e0080
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v103, 0           // 000000001c14: ca100080 64660080
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v66, 0             // 000000001c1c: ca100080 3f420080
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v68, 0             // 000000001c24: ca100080 41440080
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v70, 0             // 000000001c2c: ca100080 43460080
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v72, 0             // 000000001c34: ca100080 45480080
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v74, 0             // 000000001c3c: ca100080 474a0080
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v76, 0             // 000000001c44: ca100080 494c0080
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v78, 0             // 000000001c4c: ca100080 4b4e0080
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v56, 0             // 000000001c54: ca100080 4d380080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v58, 0             // 000000001c5c: ca100080 4f3a0080
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v60, 0             // 000000001c64: ca100080 393c0080
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v62, 0             // 000000001c6c: ca100080 3b3e0080
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v64, 0             // 000000001c74: ca100080 3d400080
	s_cselect_b32 s0, 1, 0                                     // 000000001c7c: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c80: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c84: bf078100
	s_cbranch_scc1 2483                                        // 000000001c88: bfa209b3 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2858>
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v36, 3, v55    // 000000001c8c: ca220080 25246e83
	v_or_b32_e32 v0, s36, v54                                  // 000000001c94: 38006c24
	v_or_b32_e32 v2, s33, v54                                  // 000000001c98: 38046c21
	v_mov_b32_e32 v39, s37                                     // 000000001c9c: 7e4e0225
	s_delay_alu instid0(valu_dep_4)                            // 000000001ca0: bf870004
	v_or_b32_e32 v38, s36, v36                                 // 000000001ca4: 384c4824
	v_or_b32_e32 v10, 1, v36                                   // 000000001ca8: 38144881
	v_mul_lo_u32 v4, s29, v0                                   // 000000001cac: d72c0004 0202001d
	v_mad_co_u64_u32 v[40:41], null, s28, v0, v[36:37]         // 000000001cb4: d6fe7c28 0492001c
	v_mul_lo_u32 v6, s29, v2                                   // 000000001cbc: d72c0006 0202041d
	v_mad_co_u64_u32 v[42:43], null, s28, v2, v[36:37]         // 000000001cc4: d6fe7c2a 0492041c
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[38:39]                // 000000001ccc: 7ca84c10
	v_mov_b32_e32 v1, s37                                      // 000000001cd0: 7e020225
	v_mov_b32_e32 v3, s37                                      // 000000001cd4: 7e060225
	s_mul_i32 s1, s28, s37                                     // 000000001cd8: 9601251c
	v_mul_lo_u32 v7, s28, v35                                  // 000000001cdc: d72c0007 0202461c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ce4: bf88ff9e
	v_add3_u32 v41, v4, v41, s1                                // 000000001ce8: d6550029 00065304
	v_mul_lo_u32 v8, s29, v34                                  // 000000001cf0: d72c0008 0202441d
	v_add3_u32 v43, v6, v43, s1                                // 000000001cf8: d655002b 00065706
	v_cmp_gt_i64_e64 s1, s[16:17], v[2:3]                      // 000000001d00: d4540001 02020410
	v_cndmask_b32_e32 v2, 0, v38, vcc_lo                       // 000000001d08: 02044c80
	v_cmp_gt_i64_e64 s0, s[16:17], v[0:1]                      // 000000001d0c: d4540000 02020010
	v_or_b32_e32 v0, s36, v10                                  // 000000001d14: 38001424
	v_mad_co_u64_u32 v[46:47], null, s28, v34, v[36:37]        // 000000001d18: d6fe7c2e 0492441c
	v_mul_lo_u32 v5, s28, v33                                  // 000000001d20: d72c0005 0202421c
	v_mul_lo_u32 v4, s29, v32                                  // 000000001d28: d72c0004 0202401d
	v_mad_co_u64_u32 v[44:45], null, s28, v32, v[36:37]        // 000000001d30: d6fe7c2c 0492401c
	v_cndmask_b32_e64 v3, 0, s37, vcc_lo                       // 000000001d38: d5010003 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001d40: 7ca80010
	v_or_b32_e32 v11, 2, v36                                   // 000000001d44: 38164882
	s_lshr_b64 s[6:7], s[28:29], 5                             // 000000001d48: 8586851c
	v_add3_u32 v47, v8, v47, v7                                // 000000001d4c: d655002f 041e5f08
	s_lshr_b32 s5, s29, 5                                      // 000000001d54: 8505851d
	v_mov_b32_e32 v9, s37                                      // 000000001d58: 7e120225
	s_wait_alu depctr_va_vcc(0)                                // 000000001d5c: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v0, vcc_lo                        // 000000001d60: 020e0080
	v_or_b32_e32 v0, s36, v11                                  // 000000001d64: 38001624
	v_add3_u32 v45, v4, v45, v5                                // 000000001d68: d655002d 04165b04
	v_mul_lo_u32 v4, s5, v2                                    // 000000001d70: d72c0004 02020405
	v_mul_lo_u32 v8, s6, v3                                    // 000000001d78: d72c0008 02020606
	v_mad_co_u64_u32 v[2:3], null, s6, v2, 0                   // 000000001d80: d6fe7c02 02020406
	v_cndmask_b32_e32 v6, 0, v1, vcc_lo                        // 000000001d88: 020c0280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001d8c: 7ca80010
	v_or_b32_e32 v14, 3, v36                                   // 000000001d90: 381c4883
	v_mul_lo_u32 v12, s5, v7                                   // 000000001d94: d72c000c 02020e05
	v_cmp_gt_i64_e64 s2, s[18:19], v[34:35]                    // 000000001d9c: d4540002 02024412
	v_mul_lo_u32 v13, s6, v6                                   // 000000001da4: d72c000d 02020c06
	v_mad_co_u64_u32 v[6:7], null, s6, v7, 0                   // 000000001dac: d6fe7c06 02020e06
	v_add3_u32 v3, v3, v8, v4                                  // 000000001db4: d6550003 04121103
	v_or_b32_e32 v8, s36, v14                                  // 000000001dbc: 38101c24
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc0: bf88ff9d
	v_dual_cndmask_b32 v15, 0, v1 :: v_dual_cndmask_b32 v16, 0, v0// 000000001dc4: ca520280 0f100080
	v_cmp_gt_i64_e64 s3, s[18:19], v[32:33]                    // 000000001dcc: d4540003 02024012
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001dd4: 3e000482
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001dd8: 7ca81010
	v_add3_u32 v7, v7, v13, v12                                // 000000001ddc: d6550007 04321b07
	v_mul_lo_u32 v13, s6, v15                                  // 000000001de4: d72c000d 02021e06
	v_or_b32_e32 v15, 4, v36                                   // 000000001dec: 381e4884
	v_mul_lo_u32 v12, s5, v16                                  // 000000001df0: d72c000c 02022005
	v_mad_co_u64_u32 v[2:3], null, s6, v16, 0                  // 000000001df8: d6fe7c02 02022006
	s_wait_alu depctr_va_vcc(0)                                // 000000001e00: bf88ff9d
	v_dual_cndmask_b32 v17, 0, v8 :: v_dual_cndmask_b32 v16, 0, v9// 000000001e04: ca521080 11101280
	v_or_b32_e32 v8, s36, v15                                  // 000000001e0c: 38101e24
	v_add_co_u32 v82, vcc_lo, s22, v0                          // 000000001e10: d7006a52 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001e18: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s23, v1, vcc_lo             // 000000001e1c: d5207c53 01aa0217
	s_delay_alu instid0(valu_dep_3)                            // 000000001e24: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001e28: 7ca81010
	v_add3_u32 v3, v3, v13, v12                                // 000000001e2c: d6550003 04321b03
	v_mul_lo_u32 v13, s6, v16                                  // 000000001e34: d72c000d 02022006
	v_or_b32_e32 v16, 5, v36                                   // 000000001e3c: 38204885
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e40: 3e000c82
	v_mul_lo_u32 v12, s5, v17                                  // 000000001e44: d72c000c 02022205
	v_mad_co_u64_u32 v[6:7], null, s6, v17, 0                  // 000000001e4c: d6fe7c06 02022206
	s_wait_alu depctr_va_vcc(0)                                // 000000001e54: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v8, vcc_lo                       // 000000001e58: 02241080
	v_or_b32_e32 v8, s36, v16                                  // 000000001e5c: 38102024
	v_cndmask_b32_e32 v17, 0, v9, vcc_lo                       // 000000001e60: 02221280
	v_add_co_u32 v85, vcc_lo, s22, v0                          // 000000001e64: d7006a55 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001e6c: bf88ff9d
	v_add_co_ci_u32_e64 v86, null, s23, v1, vcc_lo             // 000000001e70: d5207c56 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001e78: 7ca81010
	v_add3_u32 v7, v7, v13, v12                                // 000000001e7c: d6550007 04321b07
	v_mul_lo_u32 v13, s6, v17                                  // 000000001e84: d72c000d 02022206
	v_or_b32_e32 v17, 6, v36                                   // 000000001e8c: 38224886
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001e90: 3e000482
	v_mul_lo_u32 v12, s5, v18                                  // 000000001e94: d72c000c 02022405
	v_mad_co_u64_u32 v[2:3], null, s6, v18, 0                  // 000000001e9c: d6fe7c02 02022406
	s_wait_alu depctr_va_vcc(0)                                // 000000001ea4: bf88ff9d
	v_cndmask_b32_e32 v19, 0, v8, vcc_lo                       // 000000001ea8: 02261080
	v_or_b32_e32 v8, s36, v17                                  // 000000001eac: 38102224
	v_cndmask_b32_e32 v18, 0, v9, vcc_lo                       // 000000001eb0: 02241280
	v_add_co_u32 v88, vcc_lo, s22, v0                          // 000000001eb4: d7006a58 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001ebc: bf88ff9d
	v_add_co_ci_u32_e64 v89, null, s23, v1, vcc_lo             // 000000001ec0: d5207c59 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001ec8: 7ca81010
	v_add3_u32 v3, v3, v13, v12                                // 000000001ecc: d6550003 04321b03
	v_mul_lo_u32 v13, s6, v18                                  // 000000001ed4: d72c000d 02022406
	v_or_b32_e32 v18, 7, v36                                   // 000000001edc: 38244887
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001ee0: 3e000c82
	v_mul_lo_u32 v12, s5, v19                                  // 000000001ee4: d72c000c 02022605
	v_mad_co_u64_u32 v[6:7], null, s6, v19, 0                  // 000000001eec: d6fe7c06 02022606
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef4: bf88ff9d
	v_cndmask_b32_e32 v20, 0, v8, vcc_lo                       // 000000001ef8: 02281080
	v_or_b32_e32 v8, s36, v18                                  // 000000001efc: 38102424
	v_cndmask_b32_e32 v19, 0, v9, vcc_lo                       // 000000001f00: 02261280
	v_add_co_u32 v91, vcc_lo, s22, v0                          // 000000001f04: d7006a5b 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001f0c: bf88ff9d
	v_add_co_ci_u32_e64 v92, null, s23, v1, vcc_lo             // 000000001f10: d5207c5c 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001f18: 3e000482
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001f1c: 7ca81010
	v_add3_u32 v7, v7, v13, v12                                // 000000001f20: d6550007 04321b07
	v_mul_lo_u32 v12, s5, v20                                  // 000000001f28: d72c000c 02022805
	v_mul_lo_u32 v13, s6, v19                                  // 000000001f30: d72c000d 02022606
	v_mad_co_u64_u32 v[2:3], null, s6, v20, 0                  // 000000001f38: d6fe7c02 02022806
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_dual_mov_b32 v100, v37 :: v_dual_cndmask_b32 v9, 0, v9   // 000000001f44: ca120125 64081280
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo                        // 000000001f4c: 02101080
	v_add_co_u32 v93, vcc_lo, s22, v0                          // 000000001f50: d7006a5d 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v94, null, s23, v1, vcc_lo             // 000000001f5c: d5207c5e 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001f64: 3e000c82
	v_or_b32_e32 v6, s33, v36                                  // 000000001f68: 380c4821
	v_mov_b32_e32 v7, s37                                      // 000000001f6c: 7e0e0225
	v_add3_u32 v3, v3, v13, v12                                // 000000001f70: d6550003 04321b03
	v_mul_lo_u32 v19, s5, v8                                   // 000000001f78: d72c0013 02021005
	v_mul_lo_u32 v20, s6, v9                                   // 000000001f80: d72c0014 02021206
	v_mad_co_u64_u32 v[8:9], null, s6, v8, 0                   // 000000001f88: d6fe7c08 02021006
	v_add_co_u32 v96, s4, s22, v0                              // 000000001f90: d7000460 02020016
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000001f98: 7ca80c10
	s_wait_alu depctr_va_sdst(0)                               // 000000001f9c: bf88f19f
	v_add_co_ci_u32_e64 v97, null, s23, v1, s4                 // 000000001fa0: d5207c61 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001fa8: 3e000482
	v_mov_b32_e32 v3, s37                                      // 000000001fac: 7e060225
	v_or_b32_e32 v2, s33, v10                                  // 000000001fb0: 38041421
	v_add3_u32 v9, v9, v20, v19                                // 000000001fb4: d6550009 044e2909
	s_wait_alu depctr_va_vcc(0)                                // 000000001fbc: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo                        // 000000001fc0: 020c0c80
	v_cndmask_b32_e64 v7, 0, s37, vcc_lo                       // 000000001fc4: d5010007 01a84a80
	v_add_co_u32 v98, vcc_lo, s22, v0                          // 000000001fcc: d7006a62 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001fd4: bf88ff9d
	v_add_co_ci_u32_e64 v99, null, s23, v1, vcc_lo             // 000000001fd8: d5207c63 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001fe0: 7ca80410
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]                        // 000000001fe4: 3e001082
	v_mul_lo_u32 v8, s5, v6                                    // 000000001fe8: d72c0008 02020c05
	v_mul_lo_u32 v9, s6, v7                                    // 000000001ff0: d72c0009 02020e06
	v_mad_co_u64_u32 v[6:7], null, s6, v6, 0                   // 000000001ff8: d6fe7c06 02020c06
	v_mov_b32_e32 v84, v37                                     // 000000002000: 7ea80325
	s_wait_alu depctr_va_vcc(0)                                // 000000002004: bf88ff9d
	v_cndmask_b32_e32 v12, 0, v2, vcc_lo                       // 000000002008: 02180480
	v_or_b32_e32 v2, s33, v11                                  // 00000000200c: 38041621
	v_cndmask_b32_e64 v10, 0, s37, vcc_lo                      // 000000002010: d501000a 01a84a80
	v_add_co_u32 v101, vcc_lo, s22, v0                         // 000000002018: d7006a65 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002020: bf88ff9d
	v_add_co_ci_u32_e64 v102, null, s23, v1, vcc_lo            // 000000002024: d5207c66 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 00000000202c: 7ca80410
	v_add3_u32 v7, v7, v9, v8                                  // 000000002030: d6550007 04221307
	v_mul_lo_u32 v11, s5, v12                                  // 000000002038: d72c000b 02021805
	v_mul_lo_u32 v10, s6, v10                                  // 000000002040: d72c000a 02021406
	v_mad_co_u64_u32 v[0:1], null, s6, v12, 0                  // 000000002048: d6fe7c00 02021806
	s_wait_alu depctr_va_vcc(0)                                // 000000002050: bf88ff9d
	v_dual_mov_b32 v90, v37 :: v_dual_cndmask_b32 v13, 0, v2   // 000000002054: ca120125 5a0c0480
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 00000000205c: 3e040c82
	v_mov_b32_e32 v7, s37                                      // 000000002060: 7e0e0225
	v_or_b32_e32 v6, s33, v14                                  // 000000002064: 380c1c21
	v_cndmask_b32_e64 v12, 0, s37, vcc_lo                      // 000000002068: d501000c 01a84a80
	v_mul_lo_u32 v14, s5, v13                                  // 000000002070: d72c000e 02021a05
	v_add3_u32 v1, v1, v10, v11                                // 000000002078: d6550001 042e1501
	v_add_co_u32 v104, s4, s22, v2                             // 000000002080: d7000468 02020416
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000002088: 7ca80c10
	s_wait_alu depctr_va_sdst(0)                               // 00000000208c: bf88f19f
	v_add_co_ci_u32_e64 v105, null, s23, v3, s4                // 000000002090: d5207c69 00120617
	v_dual_mov_b32 v3, s37 :: v_dual_mov_b32 v80, v37          // 000000002098: ca100025 03500125
	v_or_b32_e32 v2, s33, v15                                  // 0000000020a0: 38041e21
	v_mul_lo_u32 v12, s6, v12                                  // 0000000020a4: d72c000c 02021806
	v_mad_co_u64_u32 v[10:11], null, s6, v13, 0                // 0000000020ac: d6fe7c0a 02021a06
	s_wait_alu depctr_va_vcc(0)                                // 0000000020b4: bf88ff9d
	v_cndmask_b32_e64 v7, 0, s37, vcc_lo                       // 0000000020b8: d5010007 01a84a80
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo                        // 0000000020c0: 020c0c80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 0000000020c4: 7ca80410
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 0000000020c8: 3e000082
	v_mov_b32_e32 v70, v37                                     // 0000000020cc: 7e8c0325
	v_mul_lo_u32 v13, s6, v7                                   // 0000000020d0: d72c000d 02020e06
	v_mov_b32_e32 v68, v37                                     // 0000000020d8: 7e880325
	v_add3_u32 v11, v11, v12, v14                              // 0000000020dc: d655000b 043a190b
	v_mul_lo_u32 v12, s5, v6                                   // 0000000020e4: d72c000c 02020c05
	v_mad_co_u64_u32 v[6:7], null, s6, v6, 0                   // 0000000020ec: d6fe7c06 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 0000000020f4: bf88ff9d
	v_cndmask_b32_e32 v15, 0, v2, vcc_lo                       // 0000000020f8: 021e0480
	v_or_b32_e32 v2, s33, v16                                  // 0000000020fc: 38042021
	v_add_co_u32 v106, s4, s22, v0                             // 000000002100: d700046a 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000002108: bf88f19f
	v_add_co_ci_u32_e64 v107, null, s23, v1, s4                // 00000000210c: d5207c6b 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]                      // 000000002114: 3e001482
	v_cndmask_b32_e64 v14, 0, s37, vcc_lo                      // 000000002118: d501000e 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000002120: 7ca80410
	v_add3_u32 v7, v7, v13, v12                                // 000000002124: d6550007 04321b07
	v_mov_b32_e32 v13, s37                                     // 00000000212c: 7e1a0225
	v_or_b32_e32 v12, s33, v17                                 // 000000002130: 38182221
	v_add_co_u32 v108, s4, s22, v0                             // 000000002134: d700046c 02020016
	s_wait_alu depctr_va_sdst(0)                               // 00000000213c: bf88f19f
	v_add_co_ci_u32_e64 v109, null, s23, v1, s4                // 000000002140: d5207c6d 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000002148: 3e000c82
	s_wait_alu depctr_va_vcc(0)                                // 00000000214c: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v2 :: v_dual_mov_b32 v66, v37    // 000000002150: ca500480 07420125
	v_or_b32_e32 v2, s33, v18                                  // 000000002158: 38042421
	v_mul_lo_u32 v16, s5, v15                                  // 00000000215c: d72c0010 02021e05
	v_mul_lo_u32 v14, s6, v14                                  // 000000002164: d72c000e 02021c06
	v_mad_co_u64_u32 v[10:11], null, s6, v15, 0                // 00000000216c: d6fe7c0a 02021e06
	v_cmp_gt_i64_e64 s4, s[16:17], v[12:13]                    // 000000002174: d4540004 02021810
	v_cndmask_b32_e64 v6, 0, s37, vcc_lo                       // 00000000217c: d5010006 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000002184: 7ca80410
	v_cndmask_b32_e64 v5, 0, v33, s3                           // 000000002188: d5010005 000e4280
	v_cndmask_b32_e64 v4, 0, v32, s3                           // 000000002190: d5010004 000e4080
	v_cndmask_b32_e64 v9, 0, v35, s2                           // 000000002198: d5010009 000a4680
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a0: bf88f19f
	v_cndmask_b32_e64 v13, 0, s37, s4                          // 0000000021a4: d501000d 00104a80
	v_cndmask_b32_e64 v12, 0, v12, s4                          // 0000000021ac: d501000c 00121880
	v_add3_u32 v11, v11, v14, v16                              // 0000000021b4: d655000b 04421d0b
	s_wait_alu depctr_va_vcc(0)                                // 0000000021bc: bf88ff9d
	v_cndmask_b32_e64 v3, 0, s37, vcc_lo                       // 0000000021c0: d5010003 01a84a80
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 0000000021c8: 02040480
	v_mul_lo_u32 v14, s5, v7                                   // 0000000021cc: d72c000e 02020e05
	v_mul_lo_u32 v15, s6, v6                                   // 0000000021d4: d72c000f 02020c06
	v_mad_co_u64_u32 v[6:7], null, s6, v7, 0                   // 0000000021dc: d6fe7c06 02020e06
	v_mul_lo_u32 v16, s5, v12                                  // 0000000021e4: d72c0010 02021805
	v_mul_lo_u32 v17, s6, v13                                  // 0000000021ec: d72c0011 02021a06
	v_mad_co_u64_u32 v[12:13], null, s6, v12, 0                // 0000000021f4: d6fe7c0c 02021806
	v_add_co_u32 v110, vcc_lo, s22, v0                         // 0000000021fc: d7006a6e 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002204: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, s23, v1, vcc_lo            // 000000002208: d5207c6f 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[10:11]                      // 000000002210: 3e001482
	v_mul_lo_u32 v10, s5, v2                                   // 000000002214: d72c000a 02020405
	v_mul_lo_u32 v11, s6, v3                                   // 00000000221c: d72c000b 02020606
	v_mad_co_u64_u32 v[2:3], null, s6, v2, 0                   // 000000002224: d6fe7c02 02020406
	v_add3_u32 v7, v7, v15, v14                                // 00000000222c: d6550007 043a1f07
	v_add3_u32 v13, v13, v17, v16                              // 000000002234: d655000d 0442230d
	v_add_co_u32 v112, vcc_lo, s22, v0                         // 00000000223c: d7006a70 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002244: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, s23, v1, vcc_lo            // 000000002248: d5207c71 01aa0217
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 000000002250: 3e0c0c82
	v_add3_u32 v3, v3, v11, v10                                // 000000002254: d6550003 042a1703
	v_lshlrev_b64_e32 v[0:1], 2, v[12:13]                      // 00000000225c: 3e001882
	v_cndmask_b32_e64 v8, 0, v34, s2                           // 000000002260: d5010008 000a4480
	v_lshlrev_b64_e32 v[48:49], 2, v[4:5]                      // 000000002268: 3e600882
	v_dual_mov_b32 v103, v37 :: v_dual_mov_b32 v78, v37        // 00000000226c: ca100125 674e0125
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000002274: 3e040482
	v_add_co_u32 v114, vcc_lo, s22, v6                         // 000000002278: d7006a72 02020c16
	s_wait_alu depctr_va_vcc(0)                                // 000000002280: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, s23, v7, vcc_lo            // 000000002284: d5207c73 01aa0e17
	v_add_co_u32 v116, vcc_lo, s22, v0                         // 00000000228c: d7006a74 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002294: bf88ff9d
	v_add_co_ci_u32_e64 v117, null, s23, v1, vcc_lo            // 000000002298: d5207c75 01aa0217
	v_add_co_u32 v118, vcc_lo, s22, v2                         // 0000000022a0: d7006a76 02020416
	v_lshlrev_b64_e32 v[50:51], 2, v[8:9]                      // 0000000022a8: 3e641082
	s_wait_alu depctr_va_vcc(0)                                // 0000000022ac: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s23, v3, vcc_lo            // 0000000022b0: d5207c77 01aa0617
	v_dual_mov_b32 v95, v37 :: v_dual_mov_b32 v76, v37         // 0000000022b8: ca100125 5f4c0125
	v_dual_mov_b32 v87, v37 :: v_dual_mov_b32 v74, v37         // 0000000022c0: ca100125 574a0125
	v_dual_mov_b32 v81, v37 :: v_dual_mov_b32 v72, v37         // 0000000022c8: ca100125 51480125
	v_dual_mov_b32 v71, v37 :: v_dual_mov_b32 v64, v37         // 0000000022d0: ca100125 47400125
	v_dual_mov_b32 v69, v37 :: v_dual_mov_b32 v62, v37         // 0000000022d8: ca100125 453e0125
	v_dual_mov_b32 v67, v37 :: v_dual_mov_b32 v60, v37         // 0000000022e0: ca100125 433c0125
	v_dual_mov_b32 v65, v37 :: v_dual_mov_b32 v58, v37         // 0000000022e8: ca100125 413a0125
	v_dual_mov_b32 v63, v37 :: v_dual_mov_b32 v56, v37         // 0000000022f0: ca100125 3f380125
	v_mov_b32_e32 v79, v37                                     // 0000000022f8: 7e9e0325
	v_mov_b32_e32 v77, v37                                     // 0000000022fc: 7e9a0325
	v_mov_b32_e32 v75, v37                                     // 000000002300: 7e960325
	v_mov_b32_e32 v73, v37                                     // 000000002304: 7e920325
	v_mov_b32_e32 v61, v37                                     // 000000002308: 7e7a0325
	v_mov_b32_e32 v59, v37                                     // 00000000230c: 7e760325
	v_mov_b32_e32 v57, v37                                     // 000000002310: 7e720325
	s_mov_b64 s[14:15], 0                                      // 000000002314: be8e0180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000002318: bf8701d9
	v_dual_mov_b32 v5, s15 :: v_dual_mov_b32 v2, s15           // 00000000231c: ca10000f 0502000f
	v_or_b32_e32 v4, s14, v36                                  // 000000002324: 3808480e
	v_add_co_u32 v8, vcc_lo, v40, s14                          // 000000002328: d7006a08 02001d28
	s_wait_alu depctr_va_vcc(0)                                // 000000002330: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s15, v41, vcc_lo             // 000000002334: d5207c09 01aa520f
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[4:5]                  // 00000000233c: 7ca8081c
	v_mov_b32_e32 v7, s15                                      // 000000002340: 7e0e020f
	s_or_b32 s13, s14, 16                                      // 000000002344: 8c0d900e
	v_mov_b32_e32 v123, s15                                    // 000000002348: 7ef6020f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000234c: bf88ff9e
	v_or_b32_e32 v122, s13, v36                                // 000000002350: 38f4480d
	s_and_b32 s4, s0, vcc_lo                                   // 000000002354: 8b046a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002358: bf88ff9e
	v_cndmask_b32_e64 v0, 0, v8, s4                            // 00000000235c: d5010000 00121080
	v_cndmask_b32_e64 v1, 0, v9, s4                            // 000000002364: d5010001 00121280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000236c: bf870122
	v_add_co_u32 v0, s5, s26, v0                               // 000000002370: d7000500 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 000000002378: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s27, v1, s5                  // 00000000237c: d5207c01 0016021b
	global_load_d16_u8 v0, v[0:1], off                         // 000000002384: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v4                                     // 000000002390: 38020881
	s_wait_loadcnt 0x0                                         // 000000002394: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s4                            // 000000002398: d65d0000 00120080
	v_add_co_u32 v3, s4, v8, 1                                 // 0000000023a0: d7000403 02010308
	s_wait_alu depctr_va_sdst(0)                               // 0000000023a8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v9, s4                    // 0000000023ac: d5207c06 00121280
	v_cmp_gt_i64_e64 s4, s[28:29], v[1:2]                      // 0000000023b4: d4540004 0202021c
	v_and_b16 v0.l, 0xff, v0.l                                 // 0000000023bc: d7620000 020200ff 000000ff
	s_and_b32 s5, s0, s4                                       // 0000000023c8: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023cc: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v3, s5                            // 0000000023d0: d5010001 00160680
	v_cndmask_b32_e64 v2, 0, v6, s5                            // 0000000023d8: d5010002 00160c80
	v_or_b32_e32 v6, 2, v4                                     // 0000000023e0: 380c0882
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000023e4: bf8701a3
	v_add_co_u32 v1, s6, s26, v1                               // 0000000023e8: d7000601 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023f0: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s27, v2, s6                  // 0000000023f4: d5207c02 001a041b
	global_load_d16_hi_u8 v0, v[1:2], off                      // 0000000023fc: ee08407c 00000000 00000001
	s_wait_loadcnt 0x0                                         // 000000002408: bfc00000
	v_cndmask_b16 v2.l, 0, v0.h, s5                            // 00000000240c: d65d1002 00160080
	v_add_co_u32 v1, s5, v8, 2                                 // 000000002414: d7000501 02010508
	s_wait_alu depctr_va_sdst(0)                               // 00000000241c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v9, s5                    // 000000002420: d5207c03 00161280
	v_cmp_gt_i64_e64 s5, s[28:29], v[6:7]                      // 000000002428: d4540005 02020c1c
	v_lshlrev_b16 v2.l, 8, v2.l                                // 000000002430: d7380002 02020488
	s_and_b32 s6, s0, s5                                       // 000000002438: 8b060500
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 00000000243c: bf870141
	v_or_b16 v0.l, v0.l, v2.l                                  // 000000002440: d7630000 02020500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002448: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s6                            // 00000000244c: d5010001 001a0280
	v_cndmask_b32_e64 v3, 0, v3, s6                            // 000000002454: d5010003 001a0680
	v_add_co_u32 v6, s7, s26, v1                               // 00000000245c: d7000706 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 000000002464: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002468: bf870002
	v_add_co_ci_u32_e64 v7, null, s27, v3, s7                  // 00000000246c: d5207c07 001e061b
	global_load_d16_hi_u8 v0, v[6:7], off                      // 000000002474: ee08407c 00000000 00000006
	v_or_b32_e32 v6, 3, v4                                     // 000000002480: 380c0883
	v_mov_b32_e32 v7, s15                                      // 000000002484: 7e0e020f
	s_wait_loadcnt 0x0                                         // 000000002488: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s6                            // 00000000248c: d65d5000 001a0080
	v_add_co_u32 v1, s6, v8, 3                                 // 000000002494: d7000601 02010708
	s_wait_alu depctr_va_sdst(0)                               // 00000000249c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v9, s6                    // 0000000024a0: d5207c03 001a1280
	v_cmp_gt_i64_e64 s6, s[28:29], v[6:7]                      // 0000000024a8: d4540006 02020c1c
	v_and_b16 v0.h, 0xff, v0.h op_sel:[0,1,1]                  // 0000000024b0: d7625000 020200ff 000000ff
	s_and_b32 s7, s0, s6                                       // 0000000024bc: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024c0: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s7                            // 0000000024c4: d5010001 001e0280
	v_cndmask_b32_e64 v3, 0, v3, s7                            // 0000000024cc: d5010003 001e0680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024d4: bf870122
	v_add_co_u32 v6, s8, s26, v1                               // 0000000024d8: d7000806 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 0000000024e0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v3, s8                  // 0000000024e4: d5207c07 0022061b
	global_load_d16_u8 v1, v[6:7], off                         // 0000000024ec: ee07807c 00000001 00000006
	v_or_b32_e32 v6, 4, v4                                     // 0000000024f8: 380c0884
	v_mov_b32_e32 v7, s15                                      // 0000000024fc: 7e0e020f
	s_wait_loadcnt 0x0                                         // 000000002500: bfc00000
	v_cndmask_b16 v2.h, 0, v1.l, s7                            // 000000002504: d65d4002 001e0280
	v_add_co_u32 v1, s7, v8, 4                                 // 00000000250c: d7000701 02010908
	s_wait_alu depctr_va_sdst(0)                               // 000000002514: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v9, s7                    // 000000002518: d5207c03 001e1280
	v_cmp_gt_i64_e64 s7, s[28:29], v[6:7]                      // 000000002520: d4540007 02020c1c
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002528: d7385002 02020488
	s_and_b32 s8, s0, s7                                       // 000000002530: 8b080700
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 000000002534: bf870141
	v_or_b16 v0.h, v0.h, v2.h op_sel:[1,1,1]                   // 000000002538: d7635800 02020500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002540: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s8                            // 000000002544: d5010001 00220280
	v_cndmask_b32_e64 v3, 0, v3, s8                            // 00000000254c: d5010003 00220680
	v_add_co_u32 v6, s9, s26, v1                               // 000000002554: d7000906 0202021a
	s_wait_alu depctr_va_sdst(0)                               // 00000000255c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002560: bf870002
	v_add_co_ci_u32_e64 v7, null, s27, v3, s9                  // 000000002564: d5207c07 0026061b
	global_load_d16_u8 v1, v[6:7], off                         // 00000000256c: ee07807c 00000001 00000006
	v_or_b32_e32 v6, 5, v4                                     // 000000002578: 380c0885
	v_mov_b32_e32 v7, s15                                      // 00000000257c: 7e0e020f
	s_wait_loadcnt 0x0                                         // 000000002580: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s8                            // 000000002584: d65d0001 00220280
	v_add_co_u32 v3, s8, v8, 5                                 // 00000000258c: d7000803 02010b08
	s_wait_alu depctr_va_sdst(0)                               // 000000002594: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v9, s8                   // 000000002598: d5207c0a 00221280
	v_cmp_gt_i64_e64 s8, s[28:29], v[6:7]                      // 0000000025a0: d4540008 02020c1c
	v_and_b16 v1.l, 0xff, v1.l                                 // 0000000025a8: d7620001 020202ff 000000ff
	s_and_b32 s9, s0, s8                                       // 0000000025b4: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025b8: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s9                            // 0000000025bc: d5010003 00260680
	v_cndmask_b32_e64 v7, 0, v10, s9                           // 0000000025c4: d5010007 00261480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000025cc: bf870122
	v_add_co_u32 v6, s10, s26, v3                              // 0000000025d0: d7000a06 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000025d8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v7, s10                 // 0000000025dc: d5207c07 002a0e1b
	global_load_d16_hi_u8 v1, v[6:7], off                      // 0000000025e4: ee08407c 00000001 00000006
	v_or_b32_e32 v6, 6, v4                                     // 0000000025f0: 380c0886
	v_mov_b32_e32 v7, s15                                      // 0000000025f4: 7e0e020f
	v_or_b32_e32 v4, 7, v4                                     // 0000000025f8: 38080887
	s_wait_loadcnt 0x0                                         // 0000000025fc: bfc00000
	v_cndmask_b16 v3.l, 0, v1.h, s9                            // 000000002600: d65d1003 00260280
	v_add_co_u32 v10, s9, v8, 6                                // 000000002608: d700090a 02010d08
	s_wait_alu depctr_va_sdst(0)                               // 000000002610: bf88f19f
	v_add_co_ci_u32_e64 v11, null, 0, v9, s9                   // 000000002614: d5207c0b 00261280
	v_cmp_gt_i64_e64 s9, s[28:29], v[6:7]                      // 00000000261c: d4540009 02020c1c
	v_lshlrev_b16 v3.l, 8, v3.l                                // 000000002624: d7380003 02020688
	s_and_b32 s10, s0, s9                                      // 00000000262c: 8b0a0900
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 000000002630: bf870141
	v_or_b16 v1.l, v1.l, v3.l                                  // 000000002634: d7630001 02020701
	s_wait_alu depctr_sa_sdst(0)                               // 00000000263c: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v10, s10                          // 000000002640: d5010006 002a1480
	v_cndmask_b32_e64 v7, 0, v11, s10                          // 000000002648: d5010007 002a1680
	v_add_co_u32 v6, s11, s26, v6                              // 000000002650: d7000b06 02020c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002658: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000265c: bf870002
	v_add_co_ci_u32_e64 v7, null, s27, v7, s11                 // 000000002660: d5207c07 002e0e1b
	global_load_d16_hi_u8 v1, v[6:7], off                      // 000000002668: ee08407c 00000001 00000006
	s_wait_loadcnt 0x0                                         // 000000002674: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s10                           // 000000002678: d65d5001 002a0280
	v_add_co_u32 v6, s10, v8, 7                                // 000000002680: d7000a06 02010f08
	s_wait_alu depctr_va_sdst(0)                               // 000000002688: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v9, s10                   // 00000000268c: d5207c07 002a1280
	v_cmp_gt_i64_e64 s10, s[28:29], v[4:5]                     // 000000002694: d454000a 0202081c
	v_and_b16 v1.h, 0xff, v1.h op_sel:[0,1,1]                  // 00000000269c: d7625001 020202ff 000000ff
	s_and_b32 s11, s0, s10                                     // 0000000026a8: 8b0b0a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026ac: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s11                           // 0000000026b0: d5010004 002e0c80
	v_cndmask_b32_e64 v5, 0, v7, s11                           // 0000000026b8: d5010005 002e0e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000026c0: bf870122
	v_add_co_u32 v4, s12, s26, v4                              // 0000000026c4: d7000c04 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 0000000026cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s12                 // 0000000026d0: d5207c05 00320a1b
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000026d8: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000026e4: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 0000000026e8: d65d5003 002e0680
	v_add_co_u32 v7, s11, v42, s14                             // 0000000026f0: d7000b07 02001d2a
	s_wait_alu depctr_va_sdst(0)                               // 0000000026f8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s15, v43, s11                // 0000000026fc: d5207c08 002e560f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000002704: bf870143
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002708: d7385003 02020688
	s_and_b32 s11, s1, vcc_lo                                  // 000000002710: 8b0b6a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002714: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v7, s11                           // 000000002718: d5010002 002e0e80
	v_or_b16 v1.h, v1.h, v3.h op_sel:[1,1,1]                   // 000000002720: d7635801 02020701
	v_cndmask_b32_e64 v3, 0, v8, s11                           // 000000002728: d5010003 002e1080
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002730: bf870123
	v_add_co_u32 v2, s12, s26, v2                              // 000000002734: d7000c02 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 00000000273c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s12                 // 000000002740: d5207c03 0032061b
	global_load_d16_u8 v2, v[2:3], off                         // 000000002748: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000002754: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s11                           // 000000002758: d65d0002 002e0480
	v_add_co_u32 v3, s11, v7, 1                                // 000000002760: d7000b03 02010307
	s_wait_alu depctr_va_sdst(0)                               // 000000002768: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v8, s11                   // 00000000276c: d5207c04 002e1080
	s_and_b32 s11, s1, s4                                      // 000000002774: 8b0b0401
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002778: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002784: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 000000002788: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002790: d5010004 002e0880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002798: bf870122
	v_add_co_u32 v3, s12, s26, v3                              // 00000000279c: d7000c03 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000027a4: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s27, v4, s12                 // 0000000027a8: d5207c04 0032081b
	global_load_d16_hi_u8 v2, v[3:4], off                      // 0000000027b0: ee08407c 00000002 00000003
	s_wait_loadcnt 0x0                                         // 0000000027bc: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s11                           // 0000000027c0: d65d5002 002e0480
	v_add_co_u32 v3, s11, v7, 2                                // 0000000027c8: d7000b03 02010507
	s_wait_alu depctr_va_sdst(0)                               // 0000000027d0: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v8, s11                   // 0000000027d4: d5207c04 002e1080
	s_and_b32 s11, s1, s5                                      // 0000000027dc: 8b0b0501
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000027e0: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e8: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 0000000027ec: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 0000000027f4: d5010004 002e0880
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000027fc: bf870193
	v_or_b16 v52.l, v2.l, v2.h op_sel:[0,1,0]                  // 000000002800: d7631034 02020502
	v_add_co_u32 v3, s12, s26, v3                              // 000000002808: d7000c03 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 000000002810: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002814: bf870003
	v_add_co_ci_u32_e64 v4, null, s27, v4, s12                 // 000000002818: d5207c04 0032081b
	global_load_d16_u8 v3, v[3:4], off                         // 000000002820: ee07807c 00000003 00000003
	s_wait_loadcnt 0x0                                         // 00000000282c: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s11                           // 000000002830: d65d0003 002e0680
	v_add_co_u32 v4, s11, v7, 3                                // 000000002838: d7000b04 02010707
	s_wait_alu depctr_va_sdst(0)                               // 000000002840: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v8, s11                   // 000000002844: d5207c05 002e1080
	s_and_b32 s11, s1, s6                                      // 00000000284c: 8b0b0601
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002850: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000285c: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002860: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002868: d5010005 002e0a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002870: bf870122
	v_add_co_u32 v4, s12, s26, v4                              // 000000002874: d7000c04 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 00000000287c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s12                 // 000000002880: d5207c05 00320a1b
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002888: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 000000002894: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 000000002898: d65d5003 002e0680
	v_add_co_u32 v4, s11, v7, 4                                // 0000000028a0: d7000b04 02010907
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v8, s11                   // 0000000028ac: d5207c05 002e1080
	s_and_b32 s11, s1, s7                                      // 0000000028b4: 8b0b0701
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000028b8: d7385003 02020688
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028c0: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 0000000028c4: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 0000000028cc: d5010005 002e0a80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000028d4: bf870193
	v_or_b16 v52.h, v3.l, v3.h op_sel:[0,1,1]                  // 0000000028d8: d7635034 02020703
	v_add_co_u32 v4, s12, s26, v4                              // 0000000028e0: d7000c04 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000028ec: bf870003
	v_add_co_ci_u32_e64 v5, null, s27, v5, s12                 // 0000000028f0: d5207c05 00320a1b
	global_load_d16_u8 v4, v[4:5], off                         // 0000000028f8: ee07807c 00000004 00000004
	s_wait_loadcnt 0x0                                         // 000000002904: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, s11                           // 000000002908: d65d0004 002e0880
	v_add_co_u32 v5, s11, v7, 5                                // 000000002910: d7000b05 02010b07
	s_wait_alu depctr_va_sdst(0)                               // 000000002918: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v8, s11                   // 00000000291c: d5207c06 002e1080
	s_and_b32 s11, s1, s8                                      // 000000002924: 8b0b0801
	v_and_b16 v4.l, 0xff, v4.l                                 // 000000002928: d7620004 020208ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002934: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002938: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002940: d5010006 002e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002948: bf870122
	v_add_co_u32 v5, s12, s26, v5                              // 00000000294c: d7000c05 02020a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002954: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s27, v6, s12                 // 000000002958: d5207c06 00320c1b
	global_load_d16_hi_u8 v4, v[5:6], off                      // 000000002960: ee08407c 00000004 00000005
	s_wait_loadcnt 0x0                                         // 00000000296c: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, s11                           // 000000002970: d65d5004 002e0880
	v_add_co_u32 v5, s11, v7, 6                                // 000000002978: d7000b05 02010d07
	s_wait_alu depctr_va_sdst(0)                               // 000000002980: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v8, s11                   // 000000002984: d5207c06 002e1080
	s_and_b32 s11, s1, s9                                      // 00000000298c: 8b0b0901
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000002990: d7385004 02020888
	s_wait_alu depctr_sa_sdst(0)                               // 000000002998: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 00000000299c: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 0000000029a4: d5010006 002e0c80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000029ac: bf870193
	v_or_b16 v53.l, v4.l, v4.h op_sel:[0,1,0]                  // 0000000029b0: d7631035 02020904
	v_add_co_u32 v5, s12, s26, v5                              // 0000000029b8: d7000c05 02020a1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000029c0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000029c4: bf870003
	v_add_co_ci_u32_e64 v6, null, s27, v6, s12                 // 0000000029c8: d5207c06 00320c1b
	global_load_d16_u8 v5, v[5:6], off                         // 0000000029d0: ee07807c 00000005 00000005
	s_wait_loadcnt 0x0                                         // 0000000029dc: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, s11                           // 0000000029e0: d65d0005 002e0a80
	v_add_co_u32 v6, s11, v7, 7                                // 0000000029e8: d7000b06 02010f07
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v8, s11                   // 0000000029f4: d5207c07 002e1080
	s_and_b32 s11, s1, s10                                     // 0000000029fc: 8b0b0a01
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000002a00: d7620005 02020aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a0c: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002a10: d5010006 002e0c80
	v_cndmask_b32_e64 v7, 0, v7, s11                           // 000000002a18: d5010007 002e0e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a20: bf870122
	v_add_co_u32 v6, s12, s26, v6                              // 000000002a24: d7000c06 02020c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000002a2c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v7, s12                 // 000000002a30: d5207c07 00320e1b
	global_load_d16_hi_u8 v5, v[6:7], off                      // 000000002a38: ee08407c 00000005 00000006
	s_wait_loadcnt 0x0                                         // 000000002a44: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, s11                           // 000000002a48: d65d5005 002e0a80
	v_add_co_u32 v7, s11, v44, s14                             // 000000002a50: d7000b07 02001d2c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a58: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s15, v45, s11                // 000000002a5c: d5207c08 002e5a0f
	s_and_b32 s11, s3, vcc_lo                                  // 000000002a64: 8b0b6a03
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 000000002a68: d7385005 02020a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a70: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v7, s11                           // 000000002a74: d5010002 002e0e80
	v_cndmask_b32_e64 v3, 0, v8, s11                           // 000000002a7c: d5010003 002e1080
	s_and_b32 vcc_lo, s2, vcc_lo                               // 000000002a84: 8b6a6a02
	v_or_b16 v53.h, v5.l, v5.h op_sel:[0,1,1]                  // 000000002a88: d7635035 02020b05
	s_delay_alu instid0(valu_dep_3)                            // 000000002a90: bf870003
	v_add_co_u32 v2, s12, s30, v2                              // 000000002a94: d7000c02 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002a9c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s12                 // 000000002aa0: d5207c03 0032061f
	global_load_d16_u8 v2, v[2:3], off                         // 000000002aa8: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000002ab4: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s11                           // 000000002ab8: d65d0002 002e0480
	v_add_co_u32 v3, s11, v7, 1                                // 000000002ac0: d7000b03 02010307
	s_wait_alu depctr_va_sdst(0)                               // 000000002ac8: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v8, s11                   // 000000002acc: d5207c04 002e1080
	s_and_b32 s11, s3, s4                                      // 000000002ad4: 8b0b0403
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002ad8: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ae4: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 000000002ae8: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002af0: d5010004 002e0880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002af8: bf870122
	v_add_co_u32 v3, s12, s30, v3                              // 000000002afc: d7000c03 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b04: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s31, v4, s12                 // 000000002b08: d5207c04 0032081f
	global_load_d16_hi_u8 v2, v[3:4], off                      // 000000002b10: ee08407c 00000002 00000003
	s_wait_loadcnt 0x0                                         // 000000002b1c: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s11                           // 000000002b20: d65d5002 002e0480
	v_add_co_u32 v3, s11, v7, 2                                // 000000002b28: d7000b03 02010507
	s_wait_alu depctr_va_sdst(0)                               // 000000002b30: bf88f19f
	v_add_co_ci_u32_e64 v4, null, 0, v8, s11                   // 000000002b34: d5207c04 002e1080
	s_and_b32 s11, s3, s5                                      // 000000002b3c: 8b0b0503
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002b40: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b48: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 000000002b4c: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002b54: d5010004 002e0880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b5c: bf870122
	v_add_co_u32 v3, s12, s30, v3                              // 000000002b60: d7000c03 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002b68: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s31, v4, s12                 // 000000002b6c: d5207c04 0032081f
	global_load_d16_u8 v3, v[3:4], off                         // 000000002b74: ee07807c 00000003 00000003
	s_wait_loadcnt 0x0                                         // 000000002b80: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s11                           // 000000002b84: d65d0003 002e0680
	v_add_co_u32 v4, s11, v7, 3                                // 000000002b8c: d7000b04 02010707
	s_wait_alu depctr_va_sdst(0)                               // 000000002b94: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v8, s11                   // 000000002b98: d5207c05 002e1080
	s_and_b32 s11, s3, s6                                      // 000000002ba0: 8b0b0603
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002ba4: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bb0: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002bb4: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002bbc: d5010005 002e0a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bc4: bf870122
	v_add_co_u32 v4, s12, s30, v4                              // 000000002bc8: d7000c04 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s12                 // 000000002bd4: d5207c05 00320a1f
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002bdc: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 000000002be8: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 000000002bec: d65d5003 002e0680
	v_add_co_u32 v4, s11, v7, 4                                // 000000002bf4: d7000b04 02010907
	s_wait_alu depctr_va_sdst(0)                               // 000000002bfc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v8, s11                   // 000000002c00: d5207c05 002e1080
	s_and_b32 s11, s3, s7                                      // 000000002c08: 8b0b0703
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002c0c: d7385003 02020688
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c14: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002c18: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002c20: d5010005 002e0a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c28: bf870122
	v_add_co_u32 v4, s12, s30, v4                              // 000000002c2c: d7000c04 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s12                 // 000000002c38: d5207c05 00320a1f
	global_load_d16_u8 v4, v[4:5], off                         // 000000002c40: ee07807c 00000004 00000004
	s_wait_loadcnt 0x0                                         // 000000002c4c: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, s11                           // 000000002c50: d65d0004 002e0880
	v_add_co_u32 v5, s11, v7, 5                                // 000000002c58: d7000b05 02010b07
	s_wait_alu depctr_va_sdst(0)                               // 000000002c60: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v8, s11                   // 000000002c64: d5207c06 002e1080
	s_and_b32 s11, s3, s8                                      // 000000002c6c: 8b0b0803
	v_and_b16 v4.l, 0xff, v4.l                                 // 000000002c70: d7620004 020208ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c7c: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002c80: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002c88: d5010006 002e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c90: bf870122
	v_add_co_u32 v5, s12, s30, v5                              // 000000002c94: d7000c05 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c9c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s12                 // 000000002ca0: d5207c06 00320c1f
	global_load_d16_hi_u8 v4, v[5:6], off                      // 000000002ca8: ee08407c 00000004 00000005
	s_wait_loadcnt 0x0                                         // 000000002cb4: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, s11                           // 000000002cb8: d65d5004 002e0880
	v_add_co_u32 v5, s11, v7, 6                                // 000000002cc0: d7000b05 02010d07
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, 0, v8, s11                   // 000000002ccc: d5207c06 002e1080
	s_and_b32 s11, s3, s9                                      // 000000002cd4: 8b0b0903
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000002cd8: d7385004 02020888
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ce0: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002ce4: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002cec: d5010006 002e0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cf4: bf870122
	v_add_co_u32 v5, s12, s30, v5                              // 000000002cf8: d7000c05 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d00: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s12                 // 000000002d04: d5207c06 00320c1f
	global_load_d16_u8 v5, v[5:6], off                         // 000000002d0c: ee07807c 00000005 00000005
	s_wait_loadcnt 0x0                                         // 000000002d18: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, s11                           // 000000002d1c: d65d0005 002e0a80
	v_add_co_u32 v6, s11, v7, 7                                // 000000002d24: d7000b06 02010f07
	s_wait_alu depctr_va_sdst(0)                               // 000000002d2c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, 0, v8, s11                   // 000000002d30: d5207c07 002e1080
	s_and_b32 s11, s3, s10                                     // 000000002d38: 8b0b0a03
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000002d3c: d7620005 02020aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d48: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002d4c: d5010006 002e0c80
	v_cndmask_b32_e64 v7, 0, v7, s11                           // 000000002d54: d5010007 002e0e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002d5c: bf870122
	v_add_co_u32 v6, s12, s30, v6                              // 000000002d60: d7000c06 02020c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d68: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v7, s12                 // 000000002d6c: d5207c07 00320e1f
	global_load_d16_hi_u8 v5, v[6:7], off                      // 000000002d74: ee08407c 00000005 00000006
	v_or_b16 v6.l, v2.l, v2.h op_sel:[0,1,0]                   // 000000002d80: d7631006 02020502
	v_or_b16 v6.h, v3.l, v3.h op_sel:[0,1,1]                   // 000000002d88: d7635006 02020703
	v_or_b16 v7.l, v4.l, v4.h op_sel:[0,1,0]                   // 000000002d90: d7631007 02020904
	s_wait_loadcnt 0x0                                         // 000000002d98: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, s11                           // 000000002d9c: d65d5005 002e0a80
	v_add_co_u32 v10, s11, v46, s14                            // 000000002da4: d7000b0a 02001d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002dac: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s15, v47, s11               // 000000002db0: d5207c0b 002e5e0f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002db8: bf870113
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 000000002dbc: d7385005 02020a88
	v_dual_cndmask_b32 v2, 0, v10 :: v_dual_cndmask_b32 v3, 0, v11// 000000002dc4: ca521480 02021680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002dcc: bf870112
	v_or_b16 v7.h, v5.l, v5.h op_sel:[0,1,1]                   // 000000002dd0: d7635007 02020b05
	v_add_co_u32 v2, s11, s30, v2                              // 000000002dd8: d7000b02 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002de0: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002de4: bf870193
	v_add_co_ci_u32_e64 v3, null, s31, v3, s11                 // 000000002de8: d5207c03 002e061f
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[0:1], v[6:7], 0    // 000000002df0: cc464018 1a020d00
	global_load_d16_u8 v2, v[2:3], off                         // 000000002df8: ee07807c 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000002e04: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 000000002e08: d65d0002 01aa0480
	v_add_co_u32 v3, vcc_lo, v10, 1                            // 000000002e10: d7006a03 0201030a
	s_wait_alu depctr_va_vcc(0)                                // 000000002e18: bf88ff9d
	v_add_co_ci_u32_e64 v4, null, 0, v11, vcc_lo               // 000000002e1c: d5207c04 01aa1680
	s_and_b32 vcc_lo, s2, s4                                   // 000000002e24: 8b6a0402
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002e28: d7620002 020204ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e34: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v4, 0, v4// 000000002e38: ca520680 03040880
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002e40: bf870121
	v_add_co_u32 v3, s4, s30, v3                               // 000000002e44: d7000403 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e4c: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s31, v4, s4                  // 000000002e50: d5207c04 0012081f
	global_load_d16_hi_u8 v2, v[3:4], off                      // 000000002e58: ee08407c 00000002 00000003
	s_wait_loadcnt 0x0                                         // 000000002e64: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 000000002e68: d65d5002 01aa0480
	v_add_co_u32 v3, vcc_lo, v10, 2                            // 000000002e70: d7006a03 0201050a
	s_wait_alu depctr_va_vcc(0)                                // 000000002e78: bf88ff9d
	v_add_co_ci_u32_e64 v4, null, 0, v11, vcc_lo               // 000000002e7c: d5207c04 01aa1680
	s_and_b32 vcc_lo, s2, s5                                   // 000000002e84: 8b6a0502
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002e88: d7385002 02020488
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e90: bf88ff9e
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v4, 0, v4// 000000002e94: ca520680 03040880
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002e9c: bf870112
	v_or_b16 v120.l, v2.l, v2.h op_sel:[0,1,0]                 // 000000002ea0: d7631078 02020502
	v_add_co_u32 v3, s4, s30, v3                               // 000000002ea8: d7000403 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002eb4: bf870003
	v_add_co_ci_u32_e64 v4, null, s31, v4, s4                  // 000000002eb8: d5207c04 0012081f
	global_load_d16_u8 v3, v[3:4], off                         // 000000002ec0: ee07807c 00000003 00000003
	s_wait_loadcnt 0x0                                         // 000000002ecc: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 000000002ed0: d65d0003 01aa0680
	v_add_co_u32 v4, vcc_lo, v10, 3                            // 000000002ed8: d7006a04 0201070a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ee0: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, 0, v11, vcc_lo               // 000000002ee4: d5207c05 01aa1680
	s_and_b32 vcc_lo, s2, s6                                   // 000000002eec: 8b6a0602
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002ef0: d7620003 020206ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002efc: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v5// 000000002f00: ca520880 04040a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f08: bf870121
	v_add_co_u32 v4, s4, s30, v4                               // 000000002f0c: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002f14: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000002f18: d5207c05 00120a1f
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002f20: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 000000002f2c: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 000000002f30: d65d5003 01aa0680
	v_add_co_u32 v4, vcc_lo, v10, 4                            // 000000002f38: d7006a04 0201090a
	s_wait_alu depctr_va_vcc(0)                                // 000000002f40: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, 0, v11, vcc_lo               // 000000002f44: d5207c05 01aa1680
	s_and_b32 vcc_lo, s2, s7                                   // 000000002f4c: 8b6a0702
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002f50: d7385003 02020688
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f58: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v5// 000000002f5c: ca520880 04040a80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002f64: bf870112
	v_or_b16 v120.h, v3.l, v3.h op_sel:[0,1,1]                 // 000000002f68: d7635078 02020703
	v_add_co_u32 v4, s4, s30, v4                               // 000000002f70: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002f78: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002f7c: bf870003
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000002f80: d5207c05 00120a1f
	global_load_d16_u8 v4, v[4:5], off                         // 000000002f88: ee07807c 00000004 00000004
	s_wait_loadcnt 0x0                                         // 000000002f94: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, vcc_lo                        // 000000002f98: d65d0004 01aa0880
	v_add_co_u32 v5, vcc_lo, v10, 5                            // 000000002fa0: d7006a05 02010b0a
	s_wait_alu depctr_va_vcc(0)                                // 000000002fa8: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, 0, v11, vcc_lo               // 000000002fac: d5207c08 01aa1680
	s_and_b32 vcc_lo, s2, s8                                   // 000000002fb4: 8b6a0802
	v_and_b16 v4.l, 0xff, v4.l                                 // 000000002fb8: d7620004 020208ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fc4: bf88ff9e
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000002fc8: 020a0a80
	v_cndmask_b32_e32 v9, 0, v8, vcc_lo                        // 000000002fcc: 02121080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002fd0: bf870122
	v_add_co_u32 v8, s4, s30, v5                               // 000000002fd4: d7000408 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002fdc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s31, v9, s4                  // 000000002fe0: d5207c09 0012121f
	global_load_d16_hi_u8 v4, v[8:9], off                      // 000000002fe8: ee08407c 00000004 00000008
	s_wait_loadcnt 0x0                                         // 000000002ff4: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, vcc_lo                        // 000000002ff8: d65d5004 01aa0880
	v_add_co_u32 v5, vcc_lo, v10, 6                            // 000000003000: d7006a05 02010d0a
	s_wait_alu depctr_va_vcc(0)                                // 000000003008: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, 0, v11, vcc_lo               // 00000000300c: d5207c08 01aa1680
	s_and_b32 vcc_lo, s2, s9                                   // 000000003014: 8b6a0902
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000003018: d7385004 02020888
	s_wait_alu depctr_sa_sdst(0)                               // 000000003020: bf88ff9e
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000003024: 020a0a80
	v_cndmask_b32_e32 v9, 0, v8, vcc_lo                        // 000000003028: 02121080
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000302c: bf870193
	v_or_b16 v121.l, v4.l, v4.h op_sel:[0,1,0]                 // 000000003030: d7631079 02020904
	v_add_co_u32 v8, s4, s30, v5                               // 000000003038: d7000408 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003040: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003044: bf870003
	v_add_co_ci_u32_e64 v9, null, s31, v9, s4                  // 000000003048: d5207c09 0012121f
	global_load_d16_u8 v5, v[8:9], off                         // 000000003050: ee07807c 00000005 00000008
	s_wait_loadcnt 0x0                                         // 00000000305c: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, vcc_lo                        // 000000003060: d65d0005 01aa0a80
	v_add_co_u32 v8, vcc_lo, v10, 7                            // 000000003068: d7006a08 02010f0a
	s_wait_alu depctr_va_vcc(0)                                // 000000003070: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v11, vcc_lo               // 000000003074: d5207c09 01aa1680
	s_and_b32 vcc_lo, s2, s10                                  // 00000000307c: 8b6a0a02
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000003080: d7620005 02020aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000308c: bf88ff9e
	v_dual_cndmask_b32 v8, 0, v8 :: v_dual_cndmask_b32 v9, 0, v9// 000000003090: ca521080 08081280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003098: bf870121
	v_add_co_u32 v8, s4, s30, v8                               // 00000000309c: d7000408 0202101e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030a4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s31, v9, s4                  // 0000000030a8: d5207c09 0012121f
	global_load_d16_hi_u8 v5, v[8:9], off                      // 0000000030b0: ee08407c 00000005 00000008
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[52:53], v[6:7], 0   // 0000000030bc: cc464008 1a020d34
	s_wait_loadcnt 0x0                                         // 0000000030c4: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, vcc_lo                        // 0000000030c8: d65d5005 01aa0a80
	v_add_co_u32 v126, vcc_lo, v40, s13                        // 0000000030d0: d7006a7e 02001b28
	s_wait_alu depctr_va_vcc(0)                                // 0000000030d8: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s15, v41, vcc_lo           // 0000000030dc: d5207c7f 01aa520f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 0000000030e4: bf870123
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 0000000030e8: d7385005 02020a88
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[122:123]              // 0000000030f0: 7ca8f41c
	v_or_b16 v121.h, v5.l, v5.h op_sel:[0,1,1]                 // 0000000030f4: d7635079 02020b05
	s_and_b32 s4, s0, vcc_lo                                   // 0000000030fc: 8b046a00
	s_delay_alu instid0(valu_dep_1)                            // 000000003100: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[0:1], v[120:121], 0// 000000003104: cc464010 1a02f100
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[52:53], v[120:121], 0// 00000000310c: cc464000 1a02f134
	s_wait_alu depctr_sa_sdst(0)                               // 000000003114: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v126, s4                         // 000000003118: d5010034 0012fc80
	v_cndmask_b32_e64 v53, 0, v127, s4                         // 000000003120: d5010035 0012fe80
	v_or_b32_e32 v120, 1, v122                                 // 000000003128: 38f0f481
	v_mov_b32_e32 v121, s15                                    // 00000000312c: 7ef2020f
	s_delay_alu instid0(valu_dep_4)                            // 000000003130: bf870004
	v_add_co_u32 v52, s5, s26, v52                             // 000000003134: d7000534 0202681a
	s_wait_alu depctr_va_sdst(0)                               // 00000000313c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s27, v53, s5                // 000000003140: d5207c35 00166a1b
	global_load_d16_u8 v52, v[52:53], off                      // 000000003148: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 000000003154: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, s4                          // 000000003158: d65d0034 00126880
	v_add_co_u32 v53, s4, v126, 1                              // 000000003160: d7000435 0201037e
	s_wait_alu depctr_va_sdst(0)                               // 000000003168: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v127, s4                // 00000000316c: d5207c7c 0012fe80
	v_cmp_gt_i64_e64 s4, s[28:29], v[120:121]                  // 000000003174: d4540004 0202f01c
	v_and_b16 v52.l, 0xff, v52.l                               // 00000000317c: d7620034 020268ff 000000ff
	s_and_b32 s5, s0, s4                                       // 000000003188: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 00000000318c: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s5                          // 000000003190: d5010035 00166a80
	v_cndmask_b32_e64 v121, 0, v124, s5                        // 000000003198: d5010079 0016f880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031a0: bf870122
	v_add_co_u32 v120, s6, s26, v53                            // 0000000031a4: d7000678 02026a1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000031ac: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s6              // 0000000031b0: d5207c79 001af21b
	global_load_d16_hi_u8 v52, v[120:121], off                 // 0000000031b8: ee08407c 00000034 00000078
	v_or_b32_e32 v120, 2, v122                                 // 0000000031c4: 38f0f482
	v_mov_b32_e32 v121, s15                                    // 0000000031c8: 7ef2020f
	s_wait_loadcnt 0x0                                         // 0000000031cc: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s5                          // 0000000031d0: d65d5034 00166880
	v_add_co_u32 v53, s5, v126, 2                              // 0000000031d8: d7000535 0201057e
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e0: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v127, s5                // 0000000031e4: d5207c7c 0016fe80
	v_cmp_gt_i64_e64 s5, s[28:29], v[120:121]                  // 0000000031ec: d4540005 0202f01c
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 0000000031f4: d7385034 02026888
	s_and_b32 s6, s0, s5                                       // 0000000031fc: 8b060500
	s_wait_alu depctr_sa_sdst(0)                               // 000000003200: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s6                          // 000000003204: d5010035 001a6a80
	v_cndmask_b32_e64 v121, 0, v124, s6                        // 00000000320c: d5010079 001af880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003214: bf870122
	v_add_co_u32 v120, s7, s26, v53                            // 000000003218: d7000778 02026a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000003220: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s7              // 000000003224: d5207c79 001ef21b
	global_load_d16_u8 v53, v[120:121], off                    // 00000000322c: ee07807c 00000035 00000078
	v_or_b32_e32 v120, 3, v122                                 // 000000003238: 38f0f483
	v_mov_b32_e32 v121, s15                                    // 00000000323c: 7ef2020f
	s_wait_loadcnt 0x0                                         // 000000003240: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s6                          // 000000003244: d65d0035 001a6a80
	v_add_co_u32 v124, s6, v126, 3                             // 00000000324c: d700067c 0201077e
	s_wait_alu depctr_va_sdst(0)                               // 000000003254: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s6                // 000000003258: d5207c7d 001afe80
	v_cmp_gt_i64_e64 s6, s[28:29], v[120:121]                  // 000000003260: d4540006 0202f01c
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003268: d7620035 02026aff 000000ff
	s_and_b32 s7, s0, s6                                       // 000000003274: 8b070600
	s_wait_alu depctr_sa_sdst(0)                               // 000000003278: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v124, s7                        // 00000000327c: d5010078 001ef880
	v_cndmask_b32_e64 v121, 0, v125, s7                        // 000000003284: d5010079 001efa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000328c: bf870122
	v_add_co_u32 v120, s8, s26, v120                           // 000000003290: d7000878 0202f01a
	s_wait_alu depctr_va_sdst(0)                               // 000000003298: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s8              // 00000000329c: d5207c79 0022f21b
	global_load_d16_hi_u8 v53, v[120:121], off                 // 0000000032a4: ee08407c 00000035 00000078
	v_or_b32_e32 v120, 4, v122                                 // 0000000032b0: 38f0f484
	v_mov_b32_e32 v121, s15                                    // 0000000032b4: 7ef2020f
	s_wait_loadcnt 0x0                                         // 0000000032b8: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s7                          // 0000000032bc: d65d5035 001e6a80
	v_add_co_u32 v124, s7, v126, 4                             // 0000000032c4: d700077c 0201097e
	s_wait_alu depctr_va_sdst(0)                               // 0000000032cc: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s7                // 0000000032d0: d5207c7d 001efe80
	v_cmp_gt_i64_e64 s7, s[28:29], v[120:121]                  // 0000000032d8: d4540007 0202f01c
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 0000000032e0: d7385035 02026a88
	s_and_b32 s8, s0, s7                                       // 0000000032e8: 8b080700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032ec: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v124, s8                        // 0000000032f0: d5010078 0022f880
	v_cndmask_b32_e64 v121, 0, v125, s8                        // 0000000032f8: d5010079 0022fa80
	v_or_b32_e32 v124, 5, v122                                 // 000000003300: 38f8f485
	v_mov_b32_e32 v125, s15                                    // 000000003304: 7efa020f
	s_delay_alu instid0(valu_dep_4)                            // 000000003308: bf870004
	v_add_co_u32 v120, s9, s26, v120                           // 00000000330c: d7000978 0202f01a
	s_wait_alu depctr_va_sdst(0)                               // 000000003314: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s9              // 000000003318: d5207c79 0026f21b
	global_load_d16_u8 v120, v[120:121], off                   // 000000003320: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 00000000332c: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s8                        // 000000003330: d65d0078 0022f080
	v_add_co_u32 v121, s8, v126, 5                             // 000000003338: d7000879 02010b7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003340: bf88f19f
	v_add_co_ci_u32_e64 v128, null, 0, v127, s8                // 000000003344: d5207c80 0022fe80
	v_cmp_gt_i64_e64 s8, s[28:29], v[124:125]                  // 00000000334c: d4540008 0202f81c
	v_and_b16 v120.l, 0xff, v120.l                             // 000000003354: d7620078 0202f0ff 000000ff
	s_and_b32 s9, s0, s8                                       // 000000003360: 8b090800
	s_wait_alu depctr_sa_sdst(0)                               // 000000003364: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s9                        // 000000003368: d5010079 0026f280
	v_cndmask_b32_e64 v125, 0, v128, s9                        // 000000003370: d501007d 00270080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003378: bf870122
	v_add_co_u32 v124, s10, s26, v121                          // 00000000337c: d7000a7c 0202f21a
	s_wait_alu depctr_va_sdst(0)                               // 000000003384: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s27, v125, s10             // 000000003388: d5207c7d 002afa1b
	global_load_d16_hi_u8 v120, v[124:125], off                // 000000003390: ee08407c 00000078 0000007c
	v_or_b32_e32 v124, 6, v122                                 // 00000000339c: 38f8f486
	v_mov_b32_e32 v125, s15                                    // 0000000033a0: 7efa020f
	v_or_b32_e32 v122, 7, v122                                 // 0000000033a4: 38f4f487
	s_wait_loadcnt 0x0                                         // 0000000033a8: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s9                        // 0000000033ac: d65d5078 0026f080
	v_add_co_u32 v121, s9, v126, 6                             // 0000000033b4: d7000979 02010d7e
	s_wait_alu depctr_va_sdst(0)                               // 0000000033bc: bf88f19f
	v_add_co_ci_u32_e64 v128, null, 0, v127, s9                // 0000000033c0: d5207c80 0026fe80
	v_cmp_gt_i64_e64 s9, s[28:29], v[124:125]                  // 0000000033c8: d4540009 0202f81c
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 0000000033d0: d7385078 0202f088
	s_and_b32 s10, s0, s9                                      // 0000000033d8: 8b0a0900
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033dc: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s10                       // 0000000033e0: d5010079 002af280
	v_cndmask_b32_e64 v125, 0, v128, s10                       // 0000000033e8: d501007d 002b0080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000033f0: bf870122
	v_add_co_u32 v124, s11, s26, v121                          // 0000000033f4: d7000b7c 0202f21a
	s_wait_alu depctr_va_sdst(0)                               // 0000000033fc: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s27, v125, s11             // 000000003400: d5207c7d 002efa1b
	global_load_d16_u8 v121, v[124:125], off                   // 000000003408: ee07807c 00000079 0000007c
	s_wait_loadcnt 0x0                                         // 000000003414: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s10                       // 000000003418: d65d0079 002af280
	v_add_co_u32 v124, s10, v126, 7                            // 000000003420: d7000a7c 02010f7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003428: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s10               // 00000000342c: d5207c7d 002afe80
	v_cmp_gt_i64_e64 s10, s[28:29], v[122:123]                 // 000000003434: d454000a 0202f41c
	v_and_b16 v121.l, 0xff, v121.l                             // 00000000343c: d7620079 0202f2ff 000000ff
	s_and_b32 s11, s0, s10                                     // 000000003448: 8b0b0a00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000344c: bf88ff9e
	v_cndmask_b32_e64 v122, 0, v124, s11                       // 000000003450: d501007a 002ef880
	v_cndmask_b32_e64 v123, 0, v125, s11                       // 000000003458: d501007b 002efa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003460: bf870122
	v_add_co_u32 v122, s12, s26, v122                          // 000000003464: d7000c7a 0202f41a
	s_wait_alu depctr_va_sdst(0)                               // 00000000346c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s27, v123, s12             // 000000003470: d5207c7b 0032f61b
	global_load_d16_hi_u8 v121, v[122:123], off                // 000000003478: ee08407c 00000079 0000007a
	v_or_b16 v122.l, v52.l, v52.h op_sel:[0,1,0]               // 000000003484: d763107a 02026934
	v_or_b16 v122.h, v53.l, v53.h op_sel:[0,1,1]               // 00000000348c: d763507a 02026b35
	v_or_b16 v123.l, v120.l, v120.h op_sel:[0,1,0]             // 000000003494: d763107b 0202f178
	s_wait_loadcnt 0x0                                         // 00000000349c: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 0000000034a0: d65d5079 002ef280
	v_add_co_u32 v126, s11, v42, s13                           // 0000000034a8: d7000b7e 02001b2a
	s_wait_alu depctr_va_sdst(0)                               // 0000000034b0: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s15, v43, s11              // 0000000034b4: d5207c7f 002e560f
	s_and_b32 s11, s1, vcc_lo                                  // 0000000034bc: 8b0b6a01
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 0000000034c0: d7385079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034c8: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v126, s11                        // 0000000034cc: d5010034 002efc80
	v_cndmask_b32_e64 v53, 0, v127, s11                        // 0000000034d4: d5010035 002efe80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000034dc: bf870193
	v_or_b16 v123.h, v121.l, v121.h op_sel:[0,1,1]             // 0000000034e0: d763507b 0202f379
	v_add_co_u32 v52, s12, s26, v52                            // 0000000034e8: d7000c34 0202681a
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000034f4: bf870003
	v_add_co_ci_u32_e64 v53, null, s27, v53, s12               // 0000000034f8: d5207c35 00326a1b
	global_load_d16_u8 v52, v[52:53], off                      // 000000003500: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 00000000350c: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, s11                         // 000000003510: d65d0034 002e6880
	v_add_co_u32 v53, s11, v126, 1                             // 000000003518: d7000b35 0201037e
	s_wait_alu depctr_va_sdst(0)                               // 000000003520: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v127, s11               // 000000003524: d5207c78 002efe80
	s_and_b32 s11, s1, s4                                      // 00000000352c: 8b0b0401
	v_and_b16 v52.l, 0xff, v52.l                               // 000000003530: d7620034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000353c: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003540: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v120, s11                       // 000000003548: d5010079 002ef080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003550: bf870122
	v_add_co_u32 v120, s12, s26, v53                           // 000000003554: d7000c78 02026a1a
	s_wait_alu depctr_va_sdst(0)                               // 00000000355c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s12             // 000000003560: d5207c79 0032f21b
	global_load_d16_hi_u8 v52, v[120:121], off                 // 000000003568: ee08407c 00000034 00000078
	s_wait_loadcnt 0x0                                         // 000000003574: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s11                         // 000000003578: d65d5034 002e6880
	v_add_co_u32 v53, s11, v126, 2                             // 000000003580: d7000b35 0201057e
	s_wait_alu depctr_va_sdst(0)                               // 000000003588: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v127, s11               // 00000000358c: d5207c78 002efe80
	s_and_b32 s11, s1, s5                                      // 000000003594: 8b0b0501
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 000000003598: d7385034 02026888
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035a0: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000035a4: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v120, s11                       // 0000000035ac: d5010079 002ef080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000035b4: bf870122
	v_add_co_u32 v120, s12, s26, v53                           // 0000000035b8: d7000c78 02026a1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c0: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s12             // 0000000035c4: d5207c79 0032f21b
	global_load_d16_u8 v53, v[120:121], off                    // 0000000035cc: ee07807c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 0000000035d8: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s11                         // 0000000035dc: d65d0035 002e6a80
	v_add_co_u32 v120, s11, v126, 3                            // 0000000035e4: d7000b78 0201077e
	s_wait_alu depctr_va_sdst(0)                               // 0000000035ec: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v127, s11               // 0000000035f0: d5207c79 002efe80
	s_and_b32 s11, s1, s6                                      // 0000000035f8: 8b0b0601
	v_and_b16 v53.l, 0xff, v53.l                               // 0000000035fc: d7620035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003608: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 00000000360c: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003614: d5010079 002ef280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000361c: bf870122
	v_add_co_u32 v120, s12, s26, v120                          // 000000003620: d7000c78 0202f01a
	s_wait_alu depctr_va_sdst(0)                               // 000000003628: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s12             // 00000000362c: d5207c79 0032f21b
	global_load_d16_hi_u8 v53, v[120:121], off                 // 000000003634: ee08407c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000003640: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 000000003644: d65d5035 002e6a80
	v_add_co_u32 v120, s11, v126, 4                            // 00000000364c: d7000b78 0201097e
	s_wait_alu depctr_va_sdst(0)                               // 000000003654: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v127, s11               // 000000003658: d5207c79 002efe80
	s_and_b32 s11, s1, s7                                      // 000000003660: 8b0b0701
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 000000003664: d7385035 02026a88
	s_wait_alu depctr_sa_sdst(0)                               // 00000000366c: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 000000003670: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003678: d5010079 002ef280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003680: bf870122
	v_add_co_u32 v120, s12, s26, v120                          // 000000003684: d7000c78 0202f01a
	s_wait_alu depctr_va_sdst(0)                               // 00000000368c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s27, v121, s12             // 000000003690: d5207c79 0032f21b
	global_load_d16_u8 v120, v[120:121], off                   // 000000003698: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 0000000036a4: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s11                       // 0000000036a8: d65d0078 002ef080
	v_add_co_u32 v121, s11, v126, 5                            // 0000000036b0: d7000b79 02010b7e
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b8: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v127, s11               // 0000000036bc: d5207c7c 002efe80
	s_and_b32 s11, s1, s8                                      // 0000000036c4: 8b0b0801
	v_and_b16 v120.l, 0xff, v120.l                             // 0000000036c8: d7620078 0202f0ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036d4: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000036d8: d5010079 002ef280
	v_cndmask_b32_e64 v125, 0, v124, s11                       // 0000000036e0: d501007d 002ef880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000036e8: bf870122
	v_add_co_u32 v124, s12, s26, v121                          // 0000000036ec: d7000c7c 0202f21a
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f4: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s27, v125, s12             // 0000000036f8: d5207c7d 0032fa1b
	global_load_d16_hi_u8 v120, v[124:125], off                // 000000003700: ee08407c 00000078 0000007c
	s_wait_loadcnt 0x0                                         // 00000000370c: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s11                       // 000000003710: d65d5078 002ef080
	v_add_co_u32 v121, s11, v126, 6                            // 000000003718: d7000b79 02010d7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003720: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v127, s11               // 000000003724: d5207c7c 002efe80
	s_and_b32 s11, s1, s9                                      // 00000000372c: 8b0b0901
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000003730: d7385078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 000000003738: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 00000000373c: d5010079 002ef280
	v_cndmask_b32_e64 v125, 0, v124, s11                       // 000000003744: d501007d 002ef880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000374c: bf870122
	v_add_co_u32 v124, s12, s26, v121                          // 000000003750: d7000c7c 0202f21a
	s_wait_alu depctr_va_sdst(0)                               // 000000003758: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s27, v125, s12             // 00000000375c: d5207c7d 0032fa1b
	global_load_d16_u8 v121, v[124:125], off                   // 000000003764: ee07807c 00000079 0000007c
	s_wait_loadcnt 0x0                                         // 000000003770: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s11                       // 000000003774: d65d0079 002ef280
	v_add_co_u32 v124, s11, v126, 7                            // 00000000377c: d7000b7c 02010f7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003784: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s11               // 000000003788: d5207c7d 002efe80
	s_and_b32 s11, s1, s10                                     // 000000003790: 8b0b0a01
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003794: d7620079 0202f2ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037a0: bf88ff9e
	v_cndmask_b32_e64 v124, 0, v124, s11                       // 0000000037a4: d501007c 002ef880
	v_cndmask_b32_e64 v125, 0, v125, s11                       // 0000000037ac: d501007d 002efa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000037b4: bf870122
	v_add_co_u32 v124, s12, s26, v124                          // 0000000037b8: d7000c7c 0202f81a
	s_wait_alu depctr_va_sdst(0)                               // 0000000037c0: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s27, v125, s12             // 0000000037c4: d5207c7d 0032fa1b
	global_load_d16_hi_u8 v121, v[124:125], off                // 0000000037cc: ee08407c 00000079 0000007c
	v_or_b16 v124.l, v52.l, v52.h op_sel:[0,1,0]               // 0000000037d8: d763107c 02026934
	v_or_b16 v124.h, v53.l, v53.h op_sel:[0,1,1]               // 0000000037e0: d763507c 02026b35
	v_or_b16 v125.l, v120.l, v120.h op_sel:[0,1,0]             // 0000000037e8: d763107d 0202f178
	s_wait_loadcnt 0x0                                         // 0000000037f0: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 0000000037f4: d65d5079 002ef280
	v_add_co_u32 v128, s11, s13, v44                           // 0000000037fc: d7000b80 0202580d
	s_wait_alu depctr_va_sdst(0)                               // 000000003804: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s15, v45, s11              // 000000003808: d5207c81 002e5a0f
	s_and_b32 s11, s3, vcc_lo                                  // 000000003810: 8b0b6a03
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000003814: d7385079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 00000000381c: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v128, s11                        // 000000003820: d5010034 002f0080
	v_cndmask_b32_e64 v53, 0, v129, s11                        // 000000003828: d5010035 002f0280
	s_and_b32 vcc_lo, s2, vcc_lo                               // 000000003830: 8b6a6a02
	v_or_b16 v125.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003834: d763507d 0202f379
	s_delay_alu instid0(valu_dep_3)                            // 00000000383c: bf870003
	v_add_co_u32 v52, s12, s30, v52                            // 000000003840: d7000c34 0202681e
	s_wait_alu depctr_va_sdst(0)                               // 000000003848: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s31, v53, s12               // 00000000384c: d5207c35 00326a1f
	global_load_d16_u8 v52, v[52:53], off                      // 000000003854: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 000000003860: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, s11                         // 000000003864: d65d0034 002e6880
	v_add_co_u32 v53, s11, v128, 1                             // 00000000386c: d7000b35 02010380
	s_wait_alu depctr_va_sdst(0)                               // 000000003874: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v129, s11               // 000000003878: d5207c78 002f0280
	s_and_b32 s11, s3, s4                                      // 000000003880: 8b0b0403
	v_and_b16 v52.l, 0xff, v52.l                               // 000000003884: d7620034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003890: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003894: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v120, s11                       // 00000000389c: d5010079 002ef080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000038a4: bf870122
	v_add_co_u32 v120, s12, s30, v53                           // 0000000038a8: d7000c78 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b0: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 0000000038b4: d5207c79 0032f21f
	global_load_d16_hi_u8 v52, v[120:121], off                 // 0000000038bc: ee08407c 00000034 00000078
	s_wait_loadcnt 0x0                                         // 0000000038c8: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s11                         // 0000000038cc: d65d5034 002e6880
	v_add_co_u32 v53, s11, v128, 2                             // 0000000038d4: d7000b35 02010580
	s_wait_alu depctr_va_sdst(0)                               // 0000000038dc: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v129, s11               // 0000000038e0: d5207c78 002f0280
	s_and_b32 s11, s3, s5                                      // 0000000038e8: 8b0b0503
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 0000000038ec: d7385034 02026888
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038f4: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000038f8: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v120, s11                       // 000000003900: d5010079 002ef080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003908: bf870122
	v_add_co_u32 v120, s12, s30, v53                           // 00000000390c: d7000c78 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003914: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000003918: d5207c79 0032f21f
	global_load_d16_u8 v53, v[120:121], off                    // 000000003920: ee07807c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 00000000392c: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s11                         // 000000003930: d65d0035 002e6a80
	v_add_co_u32 v120, s11, v128, 3                            // 000000003938: d7000b78 02010780
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v129, s11               // 000000003944: d5207c79 002f0280
	s_and_b32 s11, s3, s6                                      // 00000000394c: 8b0b0603
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003950: d7620035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000395c: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 000000003960: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003968: d5010079 002ef280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003970: bf870122
	v_add_co_u32 v120, s12, s30, v120                          // 000000003974: d7000c78 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000003980: d5207c79 0032f21f
	global_load_d16_hi_u8 v53, v[120:121], off                 // 000000003988: ee08407c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000003994: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 000000003998: d65d5035 002e6a80
	v_add_co_u32 v120, s11, v128, 4                            // 0000000039a0: d7000b78 02010980
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a8: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v129, s11               // 0000000039ac: d5207c79 002f0280
	s_and_b32 s11, s3, s7                                      // 0000000039b4: 8b0b0703
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 0000000039b8: d7385035 02026a88
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c0: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 0000000039c4: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000039cc: d5010079 002ef280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000039d4: bf870122
	v_add_co_u32 v120, s12, s30, v120                          // 0000000039d8: d7000c78 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 0000000039e0: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 0000000039e4: d5207c79 0032f21f
	global_load_d16_u8 v120, v[120:121], off                   // 0000000039ec: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 0000000039f8: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s11                       // 0000000039fc: d65d0078 002ef080
	v_add_co_u32 v121, s11, v128, 5                            // 000000003a04: d7000b79 02010b80
	s_wait_alu depctr_va_sdst(0)                               // 000000003a0c: bf88f19f
	v_add_co_ci_u32_e64 v126, null, 0, v129, s11               // 000000003a10: d5207c7e 002f0280
	s_and_b32 s11, s3, s8                                      // 000000003a18: 8b0b0803
	v_and_b16 v120.l, 0xff, v120.l                             // 000000003a1c: d7620078 0202f0ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a28: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003a2c: d5010079 002ef280
	v_cndmask_b32_e64 v127, 0, v126, s11                       // 000000003a34: d501007f 002efc80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003a3c: bf870122
	v_add_co_u32 v126, s12, s30, v121                          // 000000003a40: d7000c7e 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a48: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s31, v127, s12             // 000000003a4c: d5207c7f 0032fe1f
	global_load_d16_hi_u8 v120, v[126:127], off                // 000000003a54: ee08407c 00000078 0000007e
	s_wait_loadcnt 0x0                                         // 000000003a60: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s11                       // 000000003a64: d65d5078 002ef080
	v_add_co_u32 v121, s11, v128, 6                            // 000000003a6c: d7000b79 02010d80
	s_wait_alu depctr_va_sdst(0)                               // 000000003a74: bf88f19f
	v_add_co_ci_u32_e64 v126, null, 0, v129, s11               // 000000003a78: d5207c7e 002f0280
	s_and_b32 s11, s3, s9                                      // 000000003a80: 8b0b0903
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000003a84: d7385078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a8c: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003a90: d5010079 002ef280
	v_cndmask_b32_e64 v127, 0, v126, s11                       // 000000003a98: d501007f 002efc80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003aa0: bf870122
	v_add_co_u32 v126, s12, s30, v121                          // 000000003aa4: d7000c7e 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000003aac: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s31, v127, s12             // 000000003ab0: d5207c7f 0032fe1f
	global_load_d16_u8 v121, v[126:127], off                   // 000000003ab8: ee07807c 00000079 0000007e
	s_wait_loadcnt 0x0                                         // 000000003ac4: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s11                       // 000000003ac8: d65d0079 002ef280
	v_add_co_u32 v126, s11, v128, 7                            // 000000003ad0: d7000b7e 02010f80
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad8: bf88f19f
	v_add_co_ci_u32_e64 v127, null, 0, v129, s11               // 000000003adc: d5207c7f 002f0280
	s_and_b32 s11, s3, s10                                     // 000000003ae4: 8b0b0a03
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003ae8: d7620079 0202f2ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af4: bf88ff9e
	v_cndmask_b32_e64 v126, 0, v126, s11                       // 000000003af8: d501007e 002efc80
	v_cndmask_b32_e64 v127, 0, v127, s11                       // 000000003b00: d501007f 002efe80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003b08: bf870122
	v_add_co_u32 v126, s12, s30, v126                          // 000000003b0c: d7000c7e 0202fc1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b14: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s31, v127, s12             // 000000003b18: d5207c7f 0032fe1f
	global_load_d16_hi_u8 v121, v[126:127], off                // 000000003b20: ee08407c 00000079 0000007e
	v_or_b16 v126.l, v52.l, v52.h op_sel:[0,1,0]               // 000000003b2c: d763107e 02026934
	v_or_b16 v126.h, v53.l, v53.h op_sel:[0,1,1]               // 000000003b34: d763507e 02026b35
	v_or_b16 v127.l, v120.l, v120.h op_sel:[0,1,0]             // 000000003b3c: d763107f 0202f178
	s_wait_loadcnt 0x0                                         // 000000003b44: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 000000003b48: d65d5079 002ef280
	v_add_co_u32 v130, s11, s13, v46                           // 000000003b50: d7000b82 02025c0d
	s_wait_alu depctr_va_sdst(0)                               // 000000003b58: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s15, v47, s11              // 000000003b5c: d5207c83 002e5e0f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003b64: bf870113
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000003b68: d7385079 0202f288
	v_dual_cndmask_b32 v52, 0, v130 :: v_dual_cndmask_b32 v53, 0, v131// 000000003b70: ca530480 34350680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003b78: bf870112
	v_or_b16 v127.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003b7c: d763507f 0202f379
	v_add_co_u32 v52, s11, s30, v52                            // 000000003b84: d7000b34 0202681e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b8c: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003b90: bf870193
	v_add_co_ci_u32_e64 v53, null, s31, v53, s11               // 000000003b94: d5207c35 002e6a1f
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[122:123], v[126:127], v[24:31]// 000000003b9c: cc464018 1c62fd7a
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[124:125], v[126:127], v[8:15]// 000000003ba4: cc464008 1c22fd7c
	global_load_d16_u8 v52, v[52:53], off                      // 000000003bac: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 000000003bb8: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, vcc_lo                      // 000000003bbc: d65d0034 01aa6880
	v_add_co_u32 v53, vcc_lo, v130, 1                          // 000000003bc4: d7006a35 02010382
	s_wait_alu depctr_va_vcc(0)                                // 000000003bcc: bf88ff9d
	v_add_co_ci_u32_e64 v120, null, 0, v131, vcc_lo            // 000000003bd0: d5207c78 01ab0680
	s_and_b32 vcc_lo, s2, s4                                   // 000000003bd8: 8b6a0402
	v_and_b16 v52.l, 0xff, v52.l                               // 000000003bdc: d7620034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003be8: bf88ff9e
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 000000003bec: 026a6a80
	v_cndmask_b32_e32 v121, 0, v120, vcc_lo                    // 000000003bf0: 02f2f080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003bf4: bf870122
	v_add_co_u32 v120, s4, s30, v53                            // 000000003bf8: d7000478 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c00: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s4              // 000000003c04: d5207c79 0012f21f
	global_load_d16_hi_u8 v52, v[120:121], off                 // 000000003c0c: ee08407c 00000034 00000078
	s_wait_loadcnt 0x0                                         // 000000003c18: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, vcc_lo                      // 000000003c1c: d65d5034 01aa6880
	v_add_co_u32 v53, vcc_lo, v130, 2                          // 000000003c24: d7006a35 02010582
	s_wait_alu depctr_va_vcc(0)                                // 000000003c2c: bf88ff9d
	v_add_co_ci_u32_e64 v120, null, 0, v131, vcc_lo            // 000000003c30: d5207c78 01ab0680
	s_and_b32 vcc_lo, s2, s5                                   // 000000003c38: 8b6a0502
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 000000003c3c: d7385034 02026888
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c44: bf88ff9e
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 000000003c48: 026a6a80
	v_cndmask_b32_e32 v121, 0, v120, vcc_lo                    // 000000003c4c: 02f2f080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003c50: bf870122
	v_add_co_u32 v120, s4, s30, v53                            // 000000003c54: d7000478 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c5c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s4              // 000000003c60: d5207c79 0012f21f
	global_load_d16_u8 v53, v[120:121], off                    // 000000003c68: ee07807c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000003c74: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, vcc_lo                      // 000000003c78: d65d0035 01aa6a80
	v_add_co_u32 v120, vcc_lo, v130, 3                         // 000000003c80: d7006a78 02010782
	s_wait_alu depctr_va_vcc(0)                                // 000000003c88: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v131, vcc_lo            // 000000003c8c: d5207c79 01ab0680
	s_and_b32 vcc_lo, s2, s6                                   // 000000003c94: 8b6a0602
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003c98: d7620035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ca4: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v120 :: v_dual_cndmask_b32 v121, 0, v121// 000000003ca8: ca52f080 7878f280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003cb0: bf870121
	v_add_co_u32 v120, s4, s30, v120                           // 000000003cb4: d7000478 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cbc: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s4              // 000000003cc0: d5207c79 0012f21f
	global_load_d16_hi_u8 v53, v[120:121], off                 // 000000003cc8: ee08407c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000003cd4: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, vcc_lo                      // 000000003cd8: d65d5035 01aa6a80
	v_add_co_u32 v120, vcc_lo, v130, 4                         // 000000003ce0: d7006a78 02010982
	s_wait_alu depctr_va_vcc(0)                                // 000000003ce8: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v131, vcc_lo            // 000000003cec: d5207c79 01ab0680
	s_and_b32 vcc_lo, s2, s7                                   // 000000003cf4: 8b6a0702
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 000000003cf8: d7385035 02026a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d00: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v120 :: v_dual_cndmask_b32 v121, 0, v121// 000000003d04: ca52f080 7878f280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003d0c: bf870121
	v_add_co_u32 v120, s4, s30, v120                           // 000000003d10: d7000478 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000003d18: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s4              // 000000003d1c: d5207c79 0012f21f
	global_load_d16_u8 v120, v[120:121], off                   // 000000003d24: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 000000003d30: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, vcc_lo                    // 000000003d34: d65d0078 01aaf080
	v_add_co_u32 v121, vcc_lo, v130, 5                         // 000000003d3c: d7006a79 02010b82
	s_wait_alu depctr_va_vcc(0)                                // 000000003d44: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, 0, v131, vcc_lo            // 000000003d48: d5207c80 01ab0680
	s_and_b32 vcc_lo, s2, s8                                   // 000000003d50: 8b6a0802
	v_and_b16 v120.l, 0xff, v120.l                             // 000000003d54: d7620078 0202f0ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d60: bf88ff9e
	v_cndmask_b32_e32 v121, 0, v121, vcc_lo                    // 000000003d64: 02f2f280
	v_cndmask_b32_e32 v129, 0, v128, vcc_lo                    // 000000003d68: 03030080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003d6c: bf870122
	v_add_co_u32 v128, s4, s30, v121                           // 000000003d70: d7000480 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000003d78: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s4              // 000000003d7c: d5207c81 0013021f
	global_load_d16_hi_u8 v120, v[128:129], off                // 000000003d84: ee08407c 00000078 00000080
	s_wait_loadcnt 0x0                                         // 000000003d90: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, vcc_lo                    // 000000003d94: d65d5078 01aaf080
	v_add_co_u32 v121, vcc_lo, v130, 6                         // 000000003d9c: d7006a79 02010d82
	s_wait_alu depctr_va_vcc(0)                                // 000000003da4: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, 0, v131, vcc_lo            // 000000003da8: d5207c80 01ab0680
	s_and_b32 vcc_lo, s2, s9                                   // 000000003db0: 8b6a0902
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000003db4: d7385078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dbc: bf88ff9e
	v_cndmask_b32_e32 v121, 0, v121, vcc_lo                    // 000000003dc0: 02f2f280
	v_cndmask_b32_e32 v129, 0, v128, vcc_lo                    // 000000003dc4: 03030080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003dc8: bf870122
	v_add_co_u32 v128, s4, s30, v121                           // 000000003dcc: d7000480 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd4: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s4              // 000000003dd8: d5207c81 0013021f
	global_load_d16_u8 v121, v[128:129], off                   // 000000003de0: ee07807c 00000079 00000080
	s_wait_loadcnt 0x0                                         // 000000003dec: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, vcc_lo                    // 000000003df0: d65d0079 01aaf280
	v_add_co_u32 v128, vcc_lo, v130, 7                         // 000000003df8: d7006a80 02010f82
	s_wait_alu depctr_va_vcc(0)                                // 000000003e00: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, 0, v131, vcc_lo            // 000000003e04: d5207c81 01ab0680
	s_and_b32 vcc_lo, s2, s10                                  // 000000003e0c: 8b6a0a02
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003e10: d7620079 0202f2ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e1c: bf88ff9e
	v_dual_cndmask_b32 v128, 0, v128 :: v_dual_cndmask_b32 v129, 0, v129// 000000003e20: ca530080 80810280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003e28: bf870121
	v_add_co_u32 v128, s4, s30, v128                           // 000000003e2c: d7000480 0203001e
	s_wait_alu depctr_va_sdst(0)                               // 000000003e34: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s4              // 000000003e38: d5207c81 0013021f
	s_lshr_b64 s[4:5], s[14:15], 5                             // 000000003e40: 8584850e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e44: bf88ff9e
	s_mul_u64 s[6:7], s[4:5], s[18:19]                         // 000000003e48: aa861204
	global_load_d16_hi_u8 v121, v[128:129], off                // 000000003e4c: ee08407c 00000079 00000080
	s_lshr_b64 s[4:5], s[14:15], 3                             // 000000003e58: 8584830e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e5c: bf88ff9e
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000003e60: 84868206
	s_add_nc_u64 s[14:15], s[14:15], 32                        // 000000003e64: a98ea00e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e68: bf88ff9e
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 000000003e6c: a9860622
	s_wait_loadcnt 0x0                                         // 000000003e70: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, vcc_lo                    // 000000003e74: d65d5079 01aaf280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e7c: bf870091
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000003e80: d7385079 0202f288
	v_or_b16 v121.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003e88: d7635079 0202f379
	v_or_b16 v121.l, v120.l, v120.h op_sel:[0,1,0]             // 000000003e90: d7631079 0202f178
	v_or_b16 v120.l, v52.l, v52.h op_sel:[0,1,0]               // 000000003e98: d7631078 02026934
	v_add_co_u32 v52, vcc_lo, v82, s4                          // 000000003ea0: d7006a34 02000952
	v_or_b16 v120.h, v53.l, v53.h op_sel:[0,1,1]               // 000000003ea8: d7635078 02026b35
	s_wait_alu depctr_va_vcc(0)                                // 000000003eb0: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s5, v83, vcc_lo             // 000000003eb4: d5207c35 01aaa605
	s_delay_alu instid0(valu_dep_2)                            // 000000003ebc: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[122:123], v[120:121], v[16:23]// 000000003ec0: cc464010 1c42f17a
	global_load_b32 v122, v[52:53], off                        // 000000003ec8: ee05007c 0000007a 00000034
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ed4: bf88ff9e
	v_add_co_u32 v52, vcc_lo, s6, v48                          // 000000003ed8: d7006a34 02026006
	s_wait_alu depctr_va_vcc(0)                                // 000000003ee0: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s7, v49, vcc_lo             // 000000003ee4: d5207c35 01aa6207
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[124:125], v[120:121], v[0:7]// 000000003eec: cc464000 1c02f17c
	v_add_co_u32 v120, vcc_lo, v85, s4                         // 000000003ef4: d7006a78 02000955
	global_load_b32 v52, v[52:53], off                         // 000000003efc: ee05007c 00000034 00000034
	s_wait_alu depctr_va_vcc(0)                                // 000000003f08: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s5, v86, vcc_lo            // 000000003f0c: d5207c79 01aaac05
	s_wait_loadcnt 0x0                                         // 000000003f14: bfc00000
	v_mul_f32_e32 v53, v122, v52                               // 000000003f18: 106a697a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000003f1c: bf8700c1
	v_mul_f32_e32 v24, v24, v53                                // 000000003f20: 10306b18
	global_load_b32 v53, v[120:121], off                       // 000000003f24: ee05007c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000003f30: bfc00000
	v_dual_add_f32 v103, v103, v24 :: v_dual_mul_f32 v24, v52, v53// 000000003f34: c9063167 67186b34
	v_mul_f32_e32 v24, v25, v24                                // 000000003f3c: 10303119
	s_delay_alu instid0(valu_dep_1)                            // 000000003f40: bf870001
	v_add_f32_e32 v100, v100, v24                              // 000000003f44: 06c83164
	v_add_co_u32 v24, vcc_lo, v88, s4                          // 000000003f48: d7006a18 02000958
	s_wait_alu depctr_va_vcc(0)                                // 000000003f50: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v89, vcc_lo             // 000000003f54: d5207c19 01aab205
	global_load_b32 v120, v[24:25], off                        // 000000003f5c: ee05007c 00000078 00000018
	s_wait_loadcnt 0x0                                         // 000000003f68: bfc00000
	v_mul_f32_e32 v24, v52, v120                               // 000000003f6c: 1030f134
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f70: bf870091
	v_mul_f32_e32 v24, v26, v24                                // 000000003f74: 1030311a
	v_add_f32_e32 v95, v95, v24                                // 000000003f78: 06be315f
	v_add_co_u32 v24, vcc_lo, v91, s4                          // 000000003f7c: d7006a18 0200095b
	s_wait_alu depctr_va_vcc(0)                                // 000000003f84: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v92, vcc_lo             // 000000003f88: d5207c19 01aab805
	global_load_b32 v26, v[24:25], off                         // 000000003f90: ee05007c 0000001a 00000018
	s_wait_loadcnt 0x0                                         // 000000003f9c: bfc00000
	v_mul_f32_e32 v24, v52, v26                                // 000000003fa0: 10303534
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003fa4: bf870091
	v_mul_f32_e32 v24, v27, v24                                // 000000003fa8: 1030311b
	v_add_f32_e32 v90, v90, v24                                // 000000003fac: 06b4315a
	v_add_co_u32 v24, vcc_lo, v93, s4                          // 000000003fb0: d7006a18 0200095d
	s_wait_alu depctr_va_vcc(0)                                // 000000003fb8: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v94, vcc_lo             // 000000003fbc: d5207c19 01aabc05
	global_load_b32 v27, v[24:25], off                         // 000000003fc4: ee05007c 0000001b 00000018
	s_wait_loadcnt 0x0                                         // 000000003fd0: bfc00000
	v_mul_f32_e32 v24, v52, v27                                // 000000003fd4: 10303734
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003fd8: bf870091
	v_mul_f32_e32 v24, v28, v24                                // 000000003fdc: 1030311c
	v_add_f32_e32 v87, v87, v24                                // 000000003fe0: 06ae3157
	v_add_co_u32 v24, vcc_lo, v96, s4                          // 000000003fe4: d7006a18 02000960
	s_wait_alu depctr_va_vcc(0)                                // 000000003fec: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v97, vcc_lo             // 000000003ff0: d5207c19 01aac205
	global_load_b32 v28, v[24:25], off                         // 000000003ff8: ee05007c 0000001c 00000018
	s_wait_loadcnt 0x0                                         // 000000004004: bfc00000
	v_mul_f32_e32 v24, v52, v28                                // 000000004008: 10303934
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000400c: bf870091
	v_mul_f32_e32 v24, v29, v24                                // 000000004010: 1030311d
	v_add_f32_e32 v84, v84, v24                                // 000000004014: 06a83154
	v_add_co_u32 v24, vcc_lo, v98, s4                          // 000000004018: d7006a18 02000962
	s_wait_alu depctr_va_vcc(0)                                // 000000004020: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v99, vcc_lo             // 000000004024: d5207c19 01aac605
	global_load_b32 v29, v[24:25], off                         // 00000000402c: ee05007c 0000001d 00000018
	s_wait_loadcnt 0x0                                         // 000000004038: bfc00000
	v_mul_f32_e32 v24, v52, v29                                // 00000000403c: 10303b34
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004040: bf870091
	v_mul_f32_e32 v24, v30, v24                                // 000000004044: 1030311e
	v_add_f32_e32 v81, v81, v24                                // 000000004048: 06a23151
	v_add_co_u32 v24, vcc_lo, v101, s4                         // 00000000404c: d7006a18 02000965
	s_wait_alu depctr_va_vcc(0)                                // 000000004054: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v102, vcc_lo            // 000000004058: d5207c19 01aacc05
	global_load_b32 v30, v[24:25], off                         // 000000004060: ee05007c 0000001e 00000018
	s_wait_loadcnt 0x0                                         // 00000000406c: bfc00000
	v_mul_f32_e32 v24, v52, v30                                // 000000004070: 10303d34
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004074: bf870091
	v_mul_f32_e32 v24, v31, v24                                // 000000004078: 1030311f
	v_add_f32_e32 v80, v80, v24                                // 00000000407c: 06a03150
	v_add_co_u32 v24, vcc_lo, s6, v50                          // 000000004080: d7006a18 02026406
	s_wait_alu depctr_va_vcc(0)                                // 000000004088: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s7, v51, vcc_lo             // 00000000408c: d5207c19 01aa6607
	global_load_b32 v24, v[24:25], off                         // 000000004094: ee05007c 00000018 00000018
	s_wait_loadcnt 0x0                                         // 0000000040a0: bfc00000
	v_mul_f32_e32 v25, v122, v24                               // 0000000040a4: 1032317a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040a8: bf870091
	v_mul_f32_e32 v16, v16, v25                                // 0000000040ac: 10203310
	v_add_f32_e32 v71, v71, v16                                // 0000000040b0: 068e2147
	v_mul_f32_e32 v16, v53, v24                                // 0000000040b4: 10203135
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040b8: bf870091
	v_mul_f32_e32 v16, v17, v16                                // 0000000040bc: 10202111
	v_add_f32_e32 v70, v70, v16                                // 0000000040c0: 068c2146
	v_mul_f32_e32 v16, v120, v24                               // 0000000040c4: 10203178
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040c8: bf870091
	v_mul_f32_e32 v16, v18, v16                                // 0000000040cc: 10202112
	v_add_f32_e32 v69, v69, v16                                // 0000000040d0: 068a2145
	v_mul_f32_e32 v16, v26, v24                                // 0000000040d4: 1020311a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040d8: bf870091
	v_mul_f32_e32 v16, v19, v16                                // 0000000040dc: 10202113
	v_add_f32_e32 v68, v68, v16                                // 0000000040e0: 06882144
	v_mul_f32_e32 v16, v27, v24                                // 0000000040e4: 1020311b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040e8: bf870091
	v_mul_f32_e32 v16, v20, v16                                // 0000000040ec: 10202114
	v_add_f32_e32 v67, v67, v16                                // 0000000040f0: 06862143
	v_mul_f32_e32 v16, v28, v24                                // 0000000040f4: 1020311c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040f8: bf870091
	v_mul_f32_e32 v16, v21, v16                                // 0000000040fc: 10202115
	v_add_f32_e32 v66, v66, v16                                // 000000004100: 06842142
	v_mul_f32_e32 v16, v29, v24                                // 000000004104: 1020311d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004108: bf870091
	v_mul_f32_e32 v16, v22, v16                                // 00000000410c: 10202116
	v_add_f32_e32 v65, v65, v16                                // 000000004110: 06822141
	v_mul_f32_e32 v16, v30, v24                                // 000000004114: 1020311e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004118: bf870091
	v_mul_f32_e32 v16, v23, v16                                // 00000000411c: 10202117
	v_add_f32_e32 v63, v63, v16                                // 000000004120: 067e213f
	v_add_co_u32 v16, vcc_lo, v104, s4                         // 000000004124: d7006a10 02000968
	s_wait_alu depctr_va_vcc(0)                                // 00000000412c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s5, v105, vcc_lo            // 000000004130: d5207c11 01aad205
	global_load_b32 v16, v[16:17], off                         // 000000004138: ee05007c 00000010 00000010
	s_wait_loadcnt 0x0                                         // 000000004144: bfc00000
	v_mul_f32_e32 v17, v52, v16                                // 000000004148: 10222134
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 00000000414c: bf8701c1
	v_mul_f32_e32 v8, v8, v17                                  // 000000004150: 10102308
	v_add_co_u32 v17, vcc_lo, v106, s4                         // 000000004154: d7006a11 0200096a
	s_wait_alu depctr_va_vcc(0)                                // 00000000415c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v107, vcc_lo            // 000000004160: d5207c12 01aad605
	v_add_f32_e32 v79, v79, v8                                 // 000000004168: 069e114f
	global_load_b32 v8, v[17:18], off                          // 00000000416c: ee05007c 00000008 00000011
	s_wait_loadcnt 0x0                                         // 000000004178: bfc00000
	v_mul_f32_e32 v17, v52, v8                                 // 00000000417c: 10221134
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000004180: bf8701c1
	v_mul_f32_e32 v9, v9, v17                                  // 000000004184: 10122309
	v_add_co_u32 v17, vcc_lo, v108, s4                         // 000000004188: d7006a11 0200096c
	s_wait_alu depctr_va_vcc(0)                                // 000000004190: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v109, vcc_lo            // 000000004194: d5207c12 01aada05
	v_add_f32_e32 v78, v78, v9                                 // 00000000419c: 069c134e
	global_load_b32 v9, v[17:18], off                          // 0000000041a0: ee05007c 00000009 00000011
	s_wait_loadcnt 0x0                                         // 0000000041ac: bfc00000
	v_mul_f32_e32 v17, v52, v9                                 // 0000000041b0: 10221334
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 0000000041b4: bf8701c1
	v_mul_f32_e32 v10, v10, v17                                // 0000000041b8: 1014230a
	v_add_co_u32 v17, vcc_lo, v110, s4                         // 0000000041bc: d7006a11 0200096e
	s_wait_alu depctr_va_vcc(0)                                // 0000000041c4: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v111, vcc_lo            // 0000000041c8: d5207c12 01aade05
	v_add_f32_e32 v77, v77, v10                                // 0000000041d0: 069a154d
	global_load_b32 v10, v[17:18], off                         // 0000000041d4: ee05007c 0000000a 00000011
	s_wait_loadcnt 0x0                                         // 0000000041e0: bfc00000
	v_mul_f32_e32 v17, v52, v10                                // 0000000041e4: 10221534
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 0000000041e8: bf8701c1
	v_mul_f32_e32 v11, v11, v17                                // 0000000041ec: 1016230b
	v_add_co_u32 v17, vcc_lo, v112, s4                         // 0000000041f0: d7006a11 02000970
	s_wait_alu depctr_va_vcc(0)                                // 0000000041f8: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v113, vcc_lo            // 0000000041fc: d5207c12 01aae205
	v_add_f32_e32 v76, v76, v11                                // 000000004204: 0698174c
	global_load_b32 v11, v[17:18], off                         // 000000004208: ee05007c 0000000b 00000011
	s_wait_loadcnt 0x0                                         // 000000004214: bfc00000
	v_mul_f32_e32 v17, v52, v11                                // 000000004218: 10221734
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 00000000421c: bf8701c1
	v_mul_f32_e32 v12, v12, v17                                // 000000004220: 1018230c
	v_add_co_u32 v17, vcc_lo, v114, s4                         // 000000004224: d7006a11 02000972
	s_wait_alu depctr_va_vcc(0)                                // 00000000422c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v115, vcc_lo            // 000000004230: d5207c12 01aae605
	v_add_f32_e32 v75, v75, v12                                // 000000004238: 0696194b
	global_load_b32 v12, v[17:18], off                         // 00000000423c: ee05007c 0000000c 00000011
	s_wait_loadcnt 0x0                                         // 000000004248: bfc00000
	v_mul_f32_e32 v17, v52, v12                                // 00000000424c: 10221934
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000004250: bf8701c1
	v_mul_f32_e32 v13, v13, v17                                // 000000004254: 101a230d
	v_add_co_u32 v17, vcc_lo, v116, s4                         // 000000004258: d7006a11 02000974
	s_wait_alu depctr_va_vcc(0)                                // 000000004260: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v117, vcc_lo            // 000000004264: d5207c12 01aaea05
	v_add_f32_e32 v74, v74, v13                                // 00000000426c: 06941b4a
	global_load_b32 v13, v[17:18], off                         // 000000004270: ee05007c 0000000d 00000011
	s_wait_loadcnt 0x0                                         // 00000000427c: bfc00000
	v_mul_f32_e32 v17, v52, v13                                // 000000004280: 10221b34
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000004284: bf8701c1
	v_mul_f32_e32 v14, v14, v17                                // 000000004288: 101c230e
	v_add_co_u32 v17, vcc_lo, v118, s4                         // 00000000428c: d7006a11 02000976
	s_wait_alu depctr_va_vcc(0)                                // 000000004294: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v119, vcc_lo            // 000000004298: d5207c12 01aaee05
	v_add_f32_e32 v73, v73, v14                                // 0000000042a0: 06921d49
	v_cmp_lt_i64_e64 s4, s[14:15], s[24:25]                    // 0000000042a4: d4510004 0200300e
	global_load_b32 v14, v[17:18], off                         // 0000000042ac: ee05007c 0000000e 00000011
	s_and_b32 vcc_lo, exec_lo, s4                              // 0000000042b8: 8b6a047e
	s_wait_loadcnt 0x0                                         // 0000000042bc: bfc00000
	v_mul_f32_e32 v17, v52, v14                                // 0000000042c0: 10221d34
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000042c4: bf870091
	v_mul_f32_e32 v15, v15, v17                                // 0000000042c8: 101e230f
	v_add_f32_e32 v72, v72, v15                                // 0000000042cc: 06901f48
	v_mul_f32_e32 v15, v24, v16                                // 0000000042d0: 101e2118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000042d4: bf870091
	v_mul_f32_e32 v0, v0, v15                                  // 0000000042d8: 10001f00
	v_add_f32_e32 v64, v64, v0                                 // 0000000042dc: 06800140
	v_mul_f32_e32 v0, v24, v8                                  // 0000000042e0: 10001118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000042e4: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 0000000042e8: 10000101
	v_add_f32_e32 v62, v62, v0                                 // 0000000042ec: 067c013e
	v_mul_f32_e32 v0, v24, v9                                  // 0000000042f0: 10001318
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000042f4: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 0000000042f8: 10000102
	v_dual_add_f32 v61, v61, v0 :: v_dual_mul_f32 v0, v24, v10 // 0000000042fc: c906013d 3d001518
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004304: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 000000004308: 10000103
	v_add_f32_e32 v60, v60, v0                                 // 00000000430c: 0678013c
	v_mul_f32_e32 v0, v24, v11                                 // 000000004310: 10001718
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004314: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 000000004318: 10000104
	v_add_f32_e32 v59, v59, v0                                 // 00000000431c: 0676013b
	v_mul_f32_e32 v0, v24, v12                                 // 000000004320: 10001918
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004324: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 000000004328: 10000105
	v_add_f32_e32 v58, v58, v0                                 // 00000000432c: 0674013a
	v_mul_f32_e32 v0, v24, v13                                 // 000000004330: 10001b18
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004334: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 000000004338: 10000106
	v_dual_add_f32 v57, v57, v0 :: v_dual_mul_f32 v0, v24, v14 // 00000000433c: c9060139 39001d18
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004344: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000004348: 10000107
	v_add_f32_e32 v56, v56, v0                                 // 00000000434c: 06700138
	s_wait_alu depctr_sa_sdst(0)                               // 000000004350: bf88ff9e
	s_cbranch_vccnz 63472                                      // 000000004354: bfa4f7f0 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x818>
	v_mul_lo_u32 v2, s19, v38                                  // 000000004358: d72c0002 02024c13
	v_mul_lo_u32 v3, s18, v39                                  // 000000004360: d72c0003 02024e12
	v_mad_co_u64_u32 v[0:1], null, s18, v38, 0                 // 000000004368: d6fe7c00 02024c12
	v_sub_co_u32 v14, vcc_lo, s16, v38                         // 000000004370: d7016a0e 02024c10
	s_wait_alu depctr_va_vcc(0)                                // 000000004378: bf88ff9d
	v_sub_co_ci_u32_e64 v15, null, s17, v39, vcc_lo            // 00000000437c: d5217c0f 01aa4e11
	v_cmp_gt_i64_e64 s3, s[18:19], v[32:33]                    // 000000004384: d4540003 02024012
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 00000000438c: bf870194
	v_add3_u32 v1, v1, v3, v2                                  // 000000004390: d6550001 040a0701
	v_cmp_lt_i64_e32 vcc_lo, 0, v[14:15]                       // 000000004398: 7ca21c80
	s_delay_alu instid0(valu_dep_2)                            // 00000000439c: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 0000000043a0: 3e0c0081
	s_and_b32 s0, vcc_lo, s3                                   // 0000000043a4: 8b00036a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043a8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000043ac: be812000
	s_cbranch_execz 28                                         // 0000000043b0: bfa5001c <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2924>
	v_lshlrev_b64_e32 v[0:1], 1, v[32:33]                      // 0000000043b4: 3e004081
	v_add_co_u32 v3, s0, s20, v6                               // 0000000043b8: d7000003 02020c14
	v_bfe_u32 v2, v103, 16, 1                                  // 0000000043c0: d6100002 02052167
	s_wait_alu depctr_va_sdst(0)                               // 0000000043c8: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s21, v7, s0                  // 0000000043cc: d5207c04 00020e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000043d4: bf870193
	v_add_co_u32 v0, s0, v3, v0                                // 0000000043d8: d7000000 02020103
	v_add3_u32 v2, v2, v103, 0x7fff                            // 0000000043e0: d6550002 03fecf02 00007fff
	v_or_b32_e32 v5, 0x400000, v103                            // 0000000043ec: 380aceff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000043f4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v4, v1, s0                   // 0000000043f8: d5207c01 00020304
	v_cmp_u_f32_e64 s0, v103, v103                             // 000000004400: d4180000 0202cf67
	s_wait_alu depctr_va_sdst(0)                               // 000000004408: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000440c: bf870001
	v_cndmask_b32_e64 v2, v2, v5, s0                           // 000000004410: d5010002 00020b02
	global_store_d16_hi_b16 v[0:1], v2, off                    // 000000004418: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004424: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004428: 8c7e017e
	v_add_co_u32 v0, s0, s18, v32                              // 00000000442c: d7000000 02024012
	s_wait_alu depctr_va_sdst(0)                               // 000000004434: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s19, v33, s0                 // 000000004438: d5207c01 00024213
	v_cmp_lt_i64_e64 s0, 1, v[14:15]                           // 000000004440: d4510000 02021c81
	s_delay_alu instid0(valu_dep_2)                            // 000000004448: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 00000000444c: 3e000081
	s_and_b32 s1, s0, s3                                       // 000000004450: 8b010300
	s_wait_alu depctr_sa_sdst(0)                               // 000000004454: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000004458: be822001
	s_cbranch_execz 27                                         // 00000000445c: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x29cc>
	v_bfe_u32 v2, v100, 16, 1                                  // 000000004460: d6100002 02052164
	v_add_co_u32 v3, s1, s20, v6                               // 000000004468: d7000103 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004470: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s21, v7, s1                  // 000000004474: d5207c04 00060e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000447c: bf870193
	v_add3_u32 v5, v2, v100, 0x7fff                            // 000000004480: d6550005 03fec902 00007fff
	v_add_co_u32 v2, s1, v3, v0                                // 00000000448c: d7000102 02020103
	v_or_b32_e32 v8, 0x400000, v100                            // 000000004494: 3810c8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000449c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v4, v1, s1                   // 0000000044a0: d5207c03 00060304
	v_cmp_u_f32_e64 s1, v100, v100                             // 0000000044a8: d4180001 0202c964
	s_wait_alu depctr_va_sdst(0)                               // 0000000044b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000044b4: bf870001
	v_cndmask_b32_e64 v4, v5, v8, s1                           // 0000000044b8: d5010004 00061105
	global_store_d16_hi_b16 v[2:3], v4, off                    // 0000000044c0: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000044d0: 8c7e027e
	s_lshl_b64 s[4:5], s[18:19], 1                             // 0000000044d4: 84848112
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044d8: bf88ff9e
	v_add_co_u32 v2, s1, s4, v32                               // 0000000044dc: d7000102 02024004
	s_wait_alu depctr_va_sdst(0)                               // 0000000044e4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s5, v33, s1                  // 0000000044e8: d5207c03 00064205
	v_cmp_lt_i64_e64 s1, 2, v[14:15]                           // 0000000044f0: d4510001 02021c82
	s_delay_alu instid0(valu_dep_2)                            // 0000000044f8: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000044fc: 3e040481
	s_and_b32 s2, s1, s3                                       // 000000004500: 8b020301
	s_wait_alu depctr_sa_sdst(0)                               // 000000004504: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000004508: be842002
	s_cbranch_execz 27                                         // 00000000450c: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2a7c>
	v_bfe_u32 v4, v95, 16, 1                                   // 000000004510: d6100004 0205215f
	v_add_co_u32 v5, s2, s20, v6                               // 000000004518: d7000205 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004520: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v7, s2                  // 000000004524: d5207c08 000a0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000452c: bf870193
	v_add3_u32 v9, v4, v95, 0x7fff                             // 000000004530: d6550009 03febf04 00007fff
	v_add_co_u32 v4, s2, v5, v2                                // 00000000453c: d7000204 02020505
	v_or_b32_e32 v10, 0x400000, v95                            // 000000004544: 3814beff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000454c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v3, s2                   // 000000004550: d5207c05 000a0708
	v_cmp_u_f32_e64 s2, v95, v95                               // 000000004558: d4180002 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 000000004560: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004564: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s2                          // 000000004568: d5010008 000a1509
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000004570: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000457c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004580: 8c7e047e
	v_mad_co_u64_u32 v[8:9], null, s18, 3, v[32:33]            // 000000004584: d6fe7c08 04810612
	v_cmp_lt_i64_e64 s2, 3, v[14:15]                           // 00000000458c: d4510002 02021c83
	s_and_b32 s4, s2, s3                                       // 000000004594: 8b040302
	v_mad_co_u64_u32 v[9:10], null, s19, 3, v[9:10]            // 000000004598: d6fe7c09 04250613
	s_delay_alu instid0(valu_dep_1)                            // 0000000045a0: bf870001
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 0000000045a4: 3e081081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045a8: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 0000000045ac: be852004
	s_cbranch_execz 27                                         // 0000000045b0: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2b20>
	v_bfe_u32 v8, v90, 16, 1                                   // 0000000045b4: d6100008 0205215a
	v_add_co_u32 v9, s4, s20, v6                               // 0000000045bc: d7000409 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000045c4: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s21, v7, s4                 // 0000000045c8: d5207c0a 00120e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000045d0: bf870193
	v_add3_u32 v11, v8, v90, 0x7fff                            // 0000000045d4: d655000b 03feb508 00007fff
	v_add_co_u32 v8, s4, v9, v4                                // 0000000045e0: d7000408 02020909
	v_or_b32_e32 v12, 0x400000, v90                            // 0000000045e8: 3818b4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v5, s4                  // 0000000045f4: d5207c09 00120b0a
	v_cmp_u_f32_e64 s4, v90, v90                               // 0000000045fc: d4180004 0202b55a
	s_wait_alu depctr_va_sdst(0)                               // 000000004604: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004608: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s4                        // 00000000460c: d501000a 0012190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 000000004614: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000004620: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000004624: 8c7e057e
	s_lshl_b64 s[4:5], s[18:19], 2                             // 000000004628: 84848212
	s_wait_alu depctr_sa_sdst(0)                               // 00000000462c: bf88ff9e
	v_add_co_u32 v8, s4, s4, v32                               // 000000004630: d7000408 02024004
	s_wait_alu depctr_va_sdst(0)                               // 000000004638: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s5, v33, s4                  // 00000000463c: d5207c09 00124205
	v_cmp_lt_i64_e64 s4, 4, v[14:15]                           // 000000004644: d4510004 02021c84
	s_delay_alu instid0(valu_dep_2)                            // 00000000464c: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000004650: 3e101081
	s_and_b32 s5, s4, s3                                       // 000000004654: 8b050304
	s_wait_alu depctr_sa_sdst(0)                               // 000000004658: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 00000000465c: be862005
	s_cbranch_execz 27                                         // 000000004660: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2bd0>
	v_bfe_u32 v10, v87, 16, 1                                  // 000000004664: d610000a 02052157
	v_add_co_u32 v11, s5, s20, v6                              // 00000000466c: d700050b 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004674: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s21, v7, s5                 // 000000004678: d5207c0c 00160e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004680: bf870193
	v_add3_u32 v13, v10, v87, 0x7fff                           // 000000004684: d655000d 03feaf0a 00007fff
	v_add_co_u32 v10, s5, v11, v8                              // 000000004690: d700050a 0202110b
	v_or_b32_e32 v16, 0x400000, v87                            // 000000004698: 3820aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a0: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s5                 // 0000000046a4: d5207c0b 0016130c
	v_cmp_u_f32_e64 s5, v87, v87                               // 0000000046ac: d4180005 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 0000000046b4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000046b8: bf870001
	v_cndmask_b32_e64 v12, v13, v16, s5                        // 0000000046bc: d501000c 0016210d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 0000000046c4: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000046d4: 8c7e067e
	v_mad_co_u64_u32 v[10:11], null, s18, 5, v[32:33]          // 0000000046d8: d6fe7c0a 04810a12
	v_cmp_lt_i64_e64 s5, 5, v[14:15]                           // 0000000046e0: d4510005 02021c85
	s_and_b32 s6, s5, s3                                       // 0000000046e8: 8b060305
	v_mad_co_u64_u32 v[11:12], null, s19, 5, v[11:12]          // 0000000046ec: d6fe7c0b 042d0a13
	s_delay_alu instid0(valu_dep_1)                            // 0000000046f4: bf870001
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 0000000046f8: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046fc: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000004700: be872006
	s_cbranch_execz 27                                         // 000000004704: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2c74>
	v_bfe_u32 v12, v84, 16, 1                                  // 000000004708: d610000c 02052154
	v_add_co_u32 v13, s6, s20, v6                              // 000000004710: d700060d 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004718: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s21, v7, s6                 // 00000000471c: d5207c10 001a0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004724: bf870193
	v_add3_u32 v17, v12, v84, 0x7fff                           // 000000004728: d6550011 03fea90c 00007fff
	v_add_co_u32 v12, s6, v13, v10                             // 000000004734: d700060c 0202150d
	v_or_b32_e32 v18, 0x400000, v84                            // 00000000473c: 3824a8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004744: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v16, v11, s6                // 000000004748: d5207c0d 001a1710
	v_cmp_u_f32_e64 s6, v84, v84                               // 000000004750: d4180006 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 000000004758: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000475c: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s6                        // 000000004760: d5010010 001a2511
	global_store_d16_hi_b16 v[12:13], v16, off                 // 000000004768: ee09407c 08000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004774: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000004778: 8c7e077e
	v_mad_co_u64_u32 v[16:17], null, s18, 6, v[32:33]          // 00000000477c: d6fe7c10 04810c12
	v_cmp_lt_i64_e64 s6, 6, v[14:15]                           // 000000004784: d4510006 02021c86
	s_and_b32 s7, s6, s3                                       // 00000000478c: 8b070306
	v_mad_co_u64_u32 v[17:18], null, s19, 6, v[17:18]          // 000000004790: d6fe7c11 04450c13
	s_delay_alu instid0(valu_dep_1)                            // 000000004798: bf870001
	v_lshlrev_b64_e32 v[12:13], 1, v[16:17]                    // 00000000479c: 3e182081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047a0: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 0000000047a4: be882007
	s_cbranch_execz 27                                         // 0000000047a8: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2d18>
	v_bfe_u32 v16, v81, 16, 1                                  // 0000000047ac: d6100010 02052151
	v_add_co_u32 v17, s7, s20, v6                              // 0000000047b4: d7000711 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000047bc: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v7, s7                 // 0000000047c0: d5207c12 001e0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000047c8: bf870193
	v_add3_u32 v19, v16, v81, 0x7fff                           // 0000000047cc: d6550013 03fea310 00007fff
	v_add_co_u32 v16, s7, v17, v12                             // 0000000047d8: d7000710 02021911
	v_or_b32_e32 v20, 0x400000, v81                            // 0000000047e0: 3828a2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000047e8: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v13, s7                // 0000000047ec: d5207c11 001e1b12
	v_cmp_u_f32_e64 s7, v81, v81                               // 0000000047f4: d4180007 0202a351
	s_wait_alu depctr_va_sdst(0)                               // 0000000047fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004800: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s7                        // 000000004804: d5010012 001e2913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 00000000480c: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000004818: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000481c: 8c7e087e
	v_mad_co_u64_u32 v[16:17], null, s18, 7, v[32:33]          // 000000004820: d6fe7c10 04810e12
	v_cmp_lt_i64_e64 s7, 7, v[14:15]                           // 000000004828: d4510007 02021c87
	s_and_b32 s8, s7, s3                                       // 000000004830: 8b080307
	v_mad_co_u64_u32 v[17:18], null, s19, 7, v[17:18]          // 000000004834: d6fe7c11 04450e13
	s_delay_alu instid0(valu_dep_1)                            // 00000000483c: bf870001
	v_lshlrev_b64_e32 v[14:15], 1, v[16:17]                    // 000000004840: 3e1c2081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004844: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000004848: be892008
	s_cbranch_execz 27                                         // 00000000484c: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2dbc>
	v_bfe_u32 v16, v80, 16, 1                                  // 000000004850: d6100010 02052150
	v_add_co_u32 v17, s8, s20, v6                              // 000000004858: d7000811 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004860: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v7, s8                 // 000000004864: d5207c12 00220e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000486c: bf870193
	v_add3_u32 v19, v16, v80, 0x7fff                           // 000000004870: d6550013 03fea110 00007fff
	v_add_co_u32 v16, s8, v17, v14                             // 00000000487c: d7000810 02021d11
	v_or_b32_e32 v20, 0x400000, v80                            // 000000004884: 3828a0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000488c: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s8                // 000000004890: d5207c11 00221f12
	v_cmp_u_f32_e64 s8, v80, v80                               // 000000004898: d4180008 0202a150
	s_wait_alu depctr_va_sdst(0)                               // 0000000048a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048a4: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s8                        // 0000000048a8: d5010012 00222913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 0000000048b0: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000048c0: 8c7e097e
	v_or_b32_e32 v18, s33, v36                                 // 0000000048c4: 38244821
	v_or_b32_e32 v19, s37, v37                                 // 0000000048c8: 38264a25
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000048cc: bf870112
	v_mul_lo_u32 v20, s19, v18                                 // 0000000048d0: d72c0014 02022413
	v_mul_lo_u32 v21, s18, v19                                 // 0000000048d8: d72c0015 02022612
	v_mad_co_u64_u32 v[16:17], null, s18, v18, 0               // 0000000048e0: d6fe7c10 02022412
	v_sub_co_u32 v18, s8, s16, v18                             // 0000000048e8: d7010812 02022410
	s_wait_alu depctr_va_sdst(0)                               // 0000000048f0: bf88f19f
	v_sub_co_ci_u32_e64 v19, null, s17, v19, s8                // 0000000048f4: d5217c13 00222611
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000048fc: bf870211
	v_cmp_lt_i64_e64 s8, 0, v[18:19]                           // 000000004900: d4510008 02022480
	v_add3_u32 v17, v17, v21, v20                              // 000000004908: d6550011 04522b11
	s_delay_alu instid0(valu_dep_1)                            // 000000004910: bf870001
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000004914: 3e202081
	s_and_b32 s9, s8, s3                                       // 000000004918: 8b090308
	s_wait_alu depctr_sa_sdst(0)                               // 00000000491c: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000004920: be8a2009
	s_cbranch_execz 28                                         // 000000004924: bfa5001c <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2e98>
	v_lshlrev_b64_e32 v[20:21], 1, v[32:33]                    // 000000004928: 3e284081
	v_add_co_u32 v23, s9, s20, v16                             // 00000000492c: d7000917 02022014
	v_bfe_u32 v22, v79, 16, 1                                  // 000000004934: d6100016 0205214f
	s_wait_alu depctr_va_sdst(0)                               // 00000000493c: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s21, v17, s9                // 000000004940: d5207c18 00262215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004948: bf870193
	v_add_co_u32 v20, s9, v23, v20                             // 00000000494c: d7000914 02022917
	v_add3_u32 v22, v22, v79, 0x7fff                           // 000000004954: d6550016 03fe9f16 00007fff
	v_or_b32_e32 v25, 0x400000, v79                            // 000000004960: 38329eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004968: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v24, v21, s9                // 00000000496c: d5207c15 00262b18
	v_cmp_u_f32_e64 s9, v79, v79                               // 000000004974: d4180009 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 00000000497c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004980: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s9                        // 000000004984: d5010016 00263316
	global_store_d16_hi_b16 v[20:21], v22, off                 // 00000000498c: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004998: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 00000000499c: 8c7e0a7e
	v_cmp_lt_i64_e64 s9, 1, v[18:19]                           // 0000000049a0: d4510009 02022481
	s_and_b32 s10, s9, s3                                      // 0000000049a8: 8b0a0309
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049ac: bf88ff9e
	s_and_saveexec_b32 s11, s10                                // 0000000049b0: be8b200a
	s_cbranch_execz 27                                         // 0000000049b4: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2f24>
	v_bfe_u32 v20, v78, 16, 1                                  // 0000000049b8: d6100014 0205214e
	v_add_co_u32 v21, s10, s20, v16                            // 0000000049c0: d7000a15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 0000000049c8: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s10               // 0000000049cc: d5207c16 002a2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000049d4: bf870193
	v_add3_u32 v23, v20, v78, 0x7fff                           // 0000000049d8: d6550017 03fe9d14 00007fff
	v_add_co_u32 v20, s10, v21, v0                             // 0000000049e4: d7000a14 02020115
	v_or_b32_e32 v24, 0x400000, v78                            // 0000000049ec: 38309cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000049f4: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v1, s10                // 0000000049f8: d5207c15 002a0316
	v_cmp_u_f32_e64 s10, v78, v78                              // 000000004a00: d418000a 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000004a08: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a0c: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s10                       // 000000004a10: d5010016 002a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004a18: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 000000004a28: 8c7e0b7e
	v_cmp_lt_i64_e64 s10, 2, v[18:19]                          // 000000004a2c: d451000a 02022482
	s_and_b32 s11, s10, s3                                     // 000000004a34: 8b0b030a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a38: bf88ff9e
	s_and_saveexec_b32 s12, s11                                // 000000004a3c: be8c200b
	s_cbranch_execz 27                                         // 000000004a40: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2fb0>
	v_bfe_u32 v20, v77, 16, 1                                  // 000000004a44: d6100014 0205214d
	v_add_co_u32 v21, s11, s20, v16                            // 000000004a4c: d7000b15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004a54: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s11               // 000000004a58: d5207c16 002e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a60: bf870193
	v_add3_u32 v23, v20, v77, 0x7fff                           // 000000004a64: d6550017 03fe9b14 00007fff
	v_add_co_u32 v20, s11, v21, v2                             // 000000004a70: d7000b14 02020515
	v_or_b32_e32 v24, 0x400000, v77                            // 000000004a78: 38309aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004a80: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v3, s11                // 000000004a84: d5207c15 002e0716
	v_cmp_u_f32_e64 s11, v77, v77                              // 000000004a8c: d418000b 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000004a94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a98: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s11                       // 000000004a9c: d5010016 002e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004aa4: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ab0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004ab4: 8c7e0c7e
	v_cmp_lt_i64_e64 s11, 3, v[18:19]                          // 000000004ab8: d451000b 02022483
	s_and_b32 s12, s11, s3                                     // 000000004ac0: 8b0c030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ac4: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000004ac8: be8d200c
	s_cbranch_execz 27                                         // 000000004acc: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x303c>
	v_bfe_u32 v20, v76, 16, 1                                  // 000000004ad0: d6100014 0205214c
	v_add_co_u32 v21, s12, s20, v16                            // 000000004ad8: d7000c15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004ae0: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s12               // 000000004ae4: d5207c16 00322215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004aec: bf870193
	v_add3_u32 v23, v20, v76, 0x7fff                           // 000000004af0: d6550017 03fe9914 00007fff
	v_add_co_u32 v20, s12, v21, v4                             // 000000004afc: d7000c14 02020915
	v_or_b32_e32 v24, 0x400000, v76                            // 000000004b04: 383098ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b0c: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v5, s12                // 000000004b10: d5207c15 00320b16
	v_cmp_u_f32_e64 s12, v76, v76                              // 000000004b18: d418000c 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 000000004b20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b24: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s12                       // 000000004b28: d5010016 00323117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004b30: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000004b40: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 4, v[18:19]                          // 000000004b44: d451000c 02022484
	s_and_b32 s13, s12, s3                                     // 000000004b4c: 8b0d030c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b50: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 000000004b54: be8e200d
	s_cbranch_execz 27                                         // 000000004b58: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x30c8>
	v_bfe_u32 v20, v75, 16, 1                                  // 000000004b5c: d6100014 0205214b
	v_add_co_u32 v21, s13, s20, v16                            // 000000004b64: d7000d15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004b6c: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s13               // 000000004b70: d5207c16 00362215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b78: bf870193
	v_add3_u32 v23, v20, v75, 0x7fff                           // 000000004b7c: d6550017 03fe9714 00007fff
	v_add_co_u32 v20, s13, v21, v8                             // 000000004b88: d7000d14 02021115
	v_or_b32_e32 v24, 0x400000, v75                            // 000000004b90: 383096ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b98: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v9, s13                // 000000004b9c: d5207c15 00361316
	v_cmp_u_f32_e64 s13, v75, v75                              // 000000004ba4: d418000d 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 000000004bac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004bb0: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s13                       // 000000004bb4: d5010016 00363117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004bbc: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 000000004bcc: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 5, v[18:19]                          // 000000004bd0: d451000d 02022485
	s_and_b32 s14, s13, s3                                     // 000000004bd8: 8b0e030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bdc: bf88ff9e
	s_and_saveexec_b32 s15, s14                                // 000000004be0: be8f200e
	s_cbranch_execz 27                                         // 000000004be4: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3154>
	v_bfe_u32 v20, v74, 16, 1                                  // 000000004be8: d6100014 0205214a
	v_add_co_u32 v21, s14, s20, v16                            // 000000004bf0: d7000e15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004bf8: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s14               // 000000004bfc: d5207c16 003a2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004c04: bf870193
	v_add3_u32 v23, v20, v74, 0x7fff                           // 000000004c08: d6550017 03fe9514 00007fff
	v_add_co_u32 v20, s14, v21, v10                            // 000000004c14: d7000e14 02021515
	v_or_b32_e32 v24, 0x400000, v74                            // 000000004c1c: 383094ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004c24: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v11, s14               // 000000004c28: d5207c15 003a1716
	v_cmp_u_f32_e64 s14, v74, v74                              // 000000004c30: d418000e 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 000000004c38: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c3c: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s14                       // 000000004c40: d5010016 003a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004c48: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000004c58: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 6, v[18:19]                          // 000000004c5c: d451000e 02022486
	s_and_b32 s15, s14, s3                                     // 000000004c64: 8b0f030e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c68: bf88ff9e
	s_and_saveexec_b32 s41, s15                                // 000000004c6c: bea9200f
	s_cbranch_execz 27                                         // 000000004c70: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x31e0>
	v_bfe_u32 v20, v73, 16, 1                                  // 000000004c74: d6100014 02052149
	v_add_co_u32 v21, s15, s20, v16                            // 000000004c7c: d7000f15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004c84: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s15               // 000000004c88: d5207c16 003e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004c90: bf870193
	v_add3_u32 v23, v20, v73, 0x7fff                           // 000000004c94: d6550017 03fe9314 00007fff
	v_add_co_u32 v20, s15, v21, v12                            // 000000004ca0: d7000f14 02021915
	v_or_b32_e32 v24, 0x400000, v73                            // 000000004ca8: 383092ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004cb0: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v13, s15               // 000000004cb4: d5207c15 003e1b16
	v_cmp_u_f32_e64 s15, v73, v73                              // 000000004cbc: d418000f 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000004cc4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004cc8: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s15                       // 000000004ccc: d5010016 003e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004cd4: ee09407c 0b000000 00000014
	s_or_b32 exec_lo, exec_lo, s41                             // 000000004ce0: 8c7e297e
	v_cmp_lt_i64_e64 s15, 7, v[18:19]                          // 000000004ce4: d451000f 02022487
	s_and_b32 s3, s15, s3                                      // 000000004cec: 8b03030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cf0: bf88ff9e
	s_and_saveexec_b32 s41, s3                                 // 000000004cf4: bea92003
	s_cbranch_execz 27                                         // 000000004cf8: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3268>
	v_bfe_u32 v18, v72, 16, 1                                  // 000000004cfc: d6100012 02052148
	v_add_co_u32 v19, s3, s20, v16                             // 000000004d04: d7000313 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004d0c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s21, v17, s3                // 000000004d10: d5207c14 000e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004d18: bf870193
	v_add3_u32 v21, v18, v72, 0x7fff                           // 000000004d1c: d6550015 03fe9112 00007fff
	v_add_co_u32 v18, s3, v19, v14                             // 000000004d28: d7000312 02021d13
	v_or_b32_e32 v22, 0x400000, v72                            // 000000004d30: 382c90ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004d38: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v15, s3                // 000000004d3c: d5207c13 000e1f14
	v_cmp_u_f32_e64 s3, v72, v72                               // 000000004d44: d4180003 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000004d4c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d50: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s3                        // 000000004d54: d5010014 000e2d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000004d5c: ee09407c 0a000000 00000012
	s_or_b32 exec_lo, exec_lo, s41                             // 000000004d68: 8c7e297e
	v_cmp_gt_i64_e64 s3, s[18:19], v[34:35]                    // 000000004d6c: d4540003 02024412
	s_and_b32 s42, vcc_lo, s3                                  // 000000004d74: 8b2a036a
	s_delay_alu instid0(salu_cycle_1)                          // 000000004d78: bf870009
	s_and_saveexec_b32 s41, s42                                // 000000004d7c: bea9202a
	s_cbranch_execz 25                                         // 000000004d80: bfa50019 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x32e8>
	v_lshlrev_b64_e32 v[18:19], 1, v[32:33]                    // 000000004d84: 3e244081
	v_add_co_u32 v21, vcc_lo, s20, v6                          // 000000004d88: d7006a15 02020c14
	v_bfe_u32 v20, v71, 16, 1                                  // 000000004d90: d6100014 02052147
	s_wait_alu depctr_va_vcc(0)                                // 000000004d98: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s21, v7, vcc_lo             // 000000004d9c: d5207c16 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004da4: bf870193
	v_add_co_u32 v18, vcc_lo, v21, v18                         // 000000004da8: d7006a12 02022515
	v_add3_u32 v20, v20, v71, 0x7fff                           // 000000004db0: d6550014 03fe8f14 00007fff
	v_or_b32_e32 v23, 0x400000, v71                            // 000000004dbc: 382e8eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004dc4: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v22, v19, vcc_lo            // 000000004dc8: d5207c13 01aa2716
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000004dd0: 7c308f47
	s_wait_alu depctr_va_vcc(0)                                // 000000004dd4: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v23, vcc_lo                    // 000000004dd8: 02282f14
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004ddc: ee09407c 0a000000 00002012
	s_or_b32 exec_lo, exec_lo, s41                             // 000000004de8: 8c7e297e
	s_and_b32 s41, s0, s3                                      // 000000004dec: 8b290300
	s_delay_alu instid0(salu_cycle_1)                          // 000000004df0: bf870009
	s_and_saveexec_b32 s0, s41                                 // 000000004df4: be802029
	s_cbranch_execz 24                                         // 000000004df8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x335c>
	v_bfe_u32 v18, v70, 16, 1                                  // 000000004dfc: d6100012 02052146
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004e04: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004e0c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004e10: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e18: bf870193
	v_add3_u32 v21, v18, v70, 0x7fff                           // 000000004e1c: d6550015 03fe8d12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v0                          // 000000004e28: d7006a12 02020113
	v_or_b32_e32 v22, 0x400000, v70                            // 000000004e30: 382c8cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e38: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v1, vcc_lo             // 000000004e3c: d5207c13 01aa0314
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000004e44: 7c308d46
	s_wait_alu depctr_va_vcc(0)                                // 000000004e48: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004e4c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004e50: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004e60: 8c7e007e
	s_and_b32 s1, s1, s3                                       // 000000004e64: 8b010301
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e68: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004e6c: be802001
	s_cbranch_execz 24                                         // 000000004e70: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x33d4>
	v_bfe_u32 v18, v69, 16, 1                                  // 000000004e74: d6100012 02052145
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004e7c: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004e84: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004e88: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e90: bf870193
	v_add3_u32 v21, v18, v69, 0x7fff                           // 000000004e94: d6550015 03fe8b12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000004ea0: d7006a12 02020513
	v_or_b32_e32 v22, 0x400000, v69                            // 000000004ea8: 382c8aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004eb0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v3, vcc_lo             // 000000004eb4: d5207c13 01aa0714
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000004ebc: 7c308b45
	s_wait_alu depctr_va_vcc(0)                                // 000000004ec0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004ec4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004ec8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ed4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004ed8: 8c7e007e
	s_and_b32 s1, s2, s3                                       // 000000004edc: 8b010302
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ee0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004ee4: be802001
	s_cbranch_execz 24                                         // 000000004ee8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x344c>
	v_bfe_u32 v18, v68, 16, 1                                  // 000000004eec: d6100012 02052144
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004ef4: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004efc: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004f00: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f08: bf870193
	v_add3_u32 v21, v18, v68, 0x7fff                           // 000000004f0c: d6550015 03fe8912 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v4                          // 000000004f18: d7006a12 02020913
	v_or_b32_e32 v22, 0x400000, v68                            // 000000004f20: 382c88ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f28: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v5, vcc_lo             // 000000004f2c: d5207c13 01aa0b14
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 000000004f34: 7c308944
	s_wait_alu depctr_va_vcc(0)                                // 000000004f38: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004f3c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004f40: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004f50: 8c7e007e
	s_and_b32 s1, s4, s3                                       // 000000004f54: 8b010304
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f58: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004f5c: be802001
	s_cbranch_execz 24                                         // 000000004f60: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x34c4>
	v_bfe_u32 v18, v67, 16, 1                                  // 000000004f64: d6100012 02052143
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004f6c: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004f74: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004f78: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f80: bf870193
	v_add3_u32 v21, v18, v67, 0x7fff                           // 000000004f84: d6550015 03fe8712 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v8                          // 000000004f90: d7006a12 02021113
	v_or_b32_e32 v22, 0x400000, v67                            // 000000004f98: 382c86ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004fa0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v9, vcc_lo             // 000000004fa4: d5207c13 01aa1314
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 000000004fac: 7c308743
	s_wait_alu depctr_va_vcc(0)                                // 000000004fb0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004fb4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004fb8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fc4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004fc8: 8c7e007e
	s_and_b32 s1, s5, s3                                       // 000000004fcc: 8b010305
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fd0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004fd4: be802001
	s_cbranch_execz 24                                         // 000000004fd8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x353c>
	v_bfe_u32 v18, v66, 16, 1                                  // 000000004fdc: d6100012 02052142
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004fe4: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004fec: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004ff0: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ff8: bf870193
	v_add3_u32 v21, v18, v66, 0x7fff                           // 000000004ffc: d6550015 03fe8512 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v10                         // 000000005008: d7006a12 02021513
	v_or_b32_e32 v22, 0x400000, v66                            // 000000005010: 382c84ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005018: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v11, vcc_lo            // 00000000501c: d5207c13 01aa1714
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 000000005024: 7c308542
	s_wait_alu depctr_va_vcc(0)                                // 000000005028: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 00000000502c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000005030: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 00000000503c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005040: 8c7e007e
	s_and_b32 s1, s6, s3                                       // 000000005044: 8b010306
	s_wait_alu depctr_sa_sdst(0)                               // 000000005048: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000504c: be802001
	s_cbranch_execz 24                                         // 000000005050: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x35b4>
	v_bfe_u32 v18, v65, 16, 1                                  // 000000005054: d6100012 02052141
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 00000000505c: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000005064: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000005068: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005070: bf870193
	v_add3_u32 v21, v18, v65, 0x7fff                           // 000000005074: d6550015 03fe8312 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v12                         // 000000005080: d7006a12 02021913
	v_or_b32_e32 v22, 0x400000, v65                            // 000000005088: 382c82ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005090: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v13, vcc_lo            // 000000005094: d5207c13 01aa1b14
	v_cmp_u_f32_e32 vcc_lo, v65, v65                           // 00000000509c: 7c308341
	s_wait_alu depctr_va_vcc(0)                                // 0000000050a0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 0000000050a4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 0000000050a8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050b8: 8c7e007e
	s_and_b32 s1, s7, s3                                       // 0000000050bc: 8b010307
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050c0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000050c4: be802001
	s_cbranch_execz 24                                         // 0000000050c8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x362c>
	v_add_co_u32 v6, vcc_lo, s20, v6                           // 0000000050cc: d7006a06 02020c14
	v_bfe_u32 v18, v63, 16, 1                                  // 0000000050d4: d6100012 0205213f
	s_wait_alu depctr_va_vcc(0)                                // 0000000050dc: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo              // 0000000050e0: d5207c07 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000050e8: bf870193
	v_add_co_u32 v6, vcc_lo, v6, v14                           // 0000000050ec: d7006a06 02021d06
	v_add3_u32 v18, v18, v63, 0x7fff                           // 0000000050f4: d6550012 03fe7f12 00007fff
	v_or_b32_e32 v19, 0x400000, v63                            // 000000005100: 38267eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005108: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo              // 00000000510c: d5207c07 01aa1f07
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000005114: 7c307f3f
	s_wait_alu depctr_va_vcc(0)                                // 000000005118: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 00000000511c: 02242712
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 000000005120: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000512c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005130: 8c7e007e
	s_and_b32 s1, s8, s3                                       // 000000005134: 8b010308
	s_wait_alu depctr_sa_sdst(0)                               // 000000005138: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000513c: be802001
	s_cbranch_execz 25                                         // 000000005140: bfa50019 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x36a8>
	v_lshlrev_b64_e32 v[6:7], 1, v[32:33]                      // 000000005144: 3e0c4081
	v_add_co_u32 v19, vcc_lo, s20, v16                         // 000000005148: d7006a13 02022014
	v_bfe_u32 v18, v64, 16, 1                                  // 000000005150: d6100012 02052140
	s_wait_alu depctr_va_vcc(0)                                // 000000005158: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v17, vcc_lo            // 00000000515c: d5207c14 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005164: bf870193
	v_add_co_u32 v6, vcc_lo, v19, v6                           // 000000005168: d7006a06 02020d13
	v_add3_u32 v18, v18, v64, 0x7fff                           // 000000005170: d6550012 03fe8112 00007fff
	v_or_b32_e32 v21, 0x400000, v64                            // 00000000517c: 382a80ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005184: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v20, v7, vcc_lo              // 000000005188: d5207c07 01aa0f14
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 000000005190: 7c308140
	s_wait_alu depctr_va_vcc(0)                                // 000000005194: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo                    // 000000005198: 02242b12
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 00000000519c: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000051ac: 8c7e007e
	s_and_b32 s1, s9, s3                                       // 0000000051b0: 8b010309
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051b4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000051b8: be802001
	s_cbranch_execz 24                                         // 0000000051bc: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3720>
	v_add_co_u32 v7, vcc_lo, s20, v16                          // 0000000051c0: d7006a07 02022014
	v_bfe_u32 v6, v62, 16, 1                                   // 0000000051c8: d6100006 0205213e
	s_wait_alu depctr_va_vcc(0)                                // 0000000051d0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s21, v17, vcc_lo            // 0000000051d4: d5207c12 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000051dc: bf870193
	v_add_co_u32 v0, vcc_lo, v7, v0                            // 0000000051e0: d7006a00 02020107
	v_add3_u32 v6, v6, v62, 0x7fff                             // 0000000051e8: d6550006 03fe7d06 00007fff
	v_or_b32_e32 v19, 0x400000, v62                            // 0000000051f4: 38267cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000051fc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v18, v1, vcc_lo              // 000000005200: d5207c01 01aa0312
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000005208: 7c307d3e
	s_wait_alu depctr_va_vcc(0)                                // 00000000520c: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v19, vcc_lo                      // 000000005210: 020c2706
	global_store_d16_hi_b16 v[0:1], v6, off offset:32          // 000000005214: ee09407c 03000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005220: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005224: 8c7e007e
	s_and_b32 s1, s10, s3                                      // 000000005228: 8b01030a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000522c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005230: be802001
	s_cbranch_execz 24                                         // 000000005234: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3798>
	v_bfe_u32 v0, v61, 16, 1                                   // 000000005238: d6100000 0205213d
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005240: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005248: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s21, v17, vcc_lo             // 00000000524c: d5207c06 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005254: bf870193
	v_add3_u32 v7, v0, v61, 0x7fff                             // 000000005258: d6550007 03fe7b00 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v2                            // 000000005264: d7006a00 02020501
	v_or_b32_e32 v18, 0x400000, v61                            // 00000000526c: 38247aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005274: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v6, v3, vcc_lo               // 000000005278: d5207c01 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000005280: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 000000005284: bf88ff9d
	v_cndmask_b32_e32 v2, v7, v18, vcc_lo                      // 000000005288: 02042507
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000528c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005298: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000529c: 8c7e007e
	s_and_b32 s1, s11, s3                                      // 0000000052a0: 8b01030b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052a4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000052a8: be802001
	s_cbranch_execz 24                                         // 0000000052ac: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3810>
	v_bfe_u32 v0, v60, 16, 1                                   // 0000000052b0: d6100000 0205213c
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 0000000052b8: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000052c0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 0000000052c4: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000052cc: bf870193
	v_add3_u32 v3, v0, v60, 0x7fff                             // 0000000052d0: d6550003 03fe7900 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v4                            // 0000000052dc: d7006a00 02020901
	v_or_b32_e32 v6, 0x400000, v60                             // 0000000052e4: 380c78ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000052ec: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v5, vcc_lo               // 0000000052f0: d5207c01 01aa0b02
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 0000000052f8: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 0000000052fc: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v6, vcc_lo                       // 000000005300: 02040d03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005304: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005310: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005314: 8c7e007e
	s_and_b32 s1, s12, s3                                      // 000000005318: 8b01030c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000531c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005320: be802001
	s_cbranch_execz 24                                         // 000000005324: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3888>
	v_bfe_u32 v0, v59, 16, 1                                   // 000000005328: d6100000 0205213b
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005330: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005338: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 00000000533c: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005344: bf870193
	v_add3_u32 v3, v0, v59, 0x7fff                             // 000000005348: d6550003 03fe7700 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v8                            // 000000005354: d7006a00 02021101
	v_or_b32_e32 v4, 0x400000, v59                             // 00000000535c: 380876ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005364: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v9, vcc_lo               // 000000005368: d5207c01 01aa1302
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000005370: 7c30773b
	s_wait_alu depctr_va_vcc(0)                                // 000000005374: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005378: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000537c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005388: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000538c: 8c7e007e
	s_and_b32 s1, s13, s3                                      // 000000005390: 8b01030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005394: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005398: be802001
	s_cbranch_execz 24                                         // 00000000539c: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3900>
	v_bfe_u32 v0, v58, 16, 1                                   // 0000000053a0: d6100000 0205213a
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 0000000053a8: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000053b0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 0000000053b4: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000053bc: bf870193
	v_add3_u32 v3, v0, v58, 0x7fff                             // 0000000053c0: d6550003 03fe7500 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v10                           // 0000000053cc: d7006a00 02021501
	v_or_b32_e32 v4, 0x400000, v58                             // 0000000053d4: 380874ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000053dc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v11, vcc_lo              // 0000000053e0: d5207c01 01aa1702
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 0000000053e8: 7c30753a
	s_wait_alu depctr_va_vcc(0)                                // 0000000053ec: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 0000000053f0: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000053f4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005400: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005404: 8c7e007e
	s_and_b32 s1, s14, s3                                      // 000000005408: 8b01030e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000540c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005410: be802001
	s_cbranch_execz 24                                         // 000000005414: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3978>
	v_bfe_u32 v0, v57, 16, 1                                   // 000000005418: d6100000 02052139
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005420: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005428: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 00000000542c: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005434: bf870193
	v_add3_u32 v3, v0, v57, 0x7fff                             // 000000005438: d6550003 03fe7300 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v12                           // 000000005444: d7006a00 02021901
	v_or_b32_e32 v4, 0x400000, v57                             // 00000000544c: 380872ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005454: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v13, vcc_lo              // 000000005458: d5207c01 01aa1b02
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000005460: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 000000005464: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005468: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000546c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005478: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000547c: 8c7e007e
	s_and_b32 s1, s15, s3                                      // 000000005480: 8b01030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005484: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005488: be802001
	s_cbranch_execz 24                                         // 00000000548c: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x39f0>
	v_bfe_u32 v0, v56, 16, 1                                   // 000000005490: d6100000 02052138
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005498: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000054a0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 0000000054a4: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000054ac: bf870193
	v_add3_u32 v3, v0, v56, 0x7fff                             // 0000000054b0: d6550003 03fe7100 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v14                           // 0000000054bc: d7006a00 02021d01
	v_or_b32_e32 v4, 0x400000, v56                             // 0000000054c4: 380870ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000054cc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v15, vcc_lo              // 0000000054d0: d5207c01 01aa1f02
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 0000000054d8: 7c307138
	s_wait_alu depctr_va_vcc(0)                                // 0000000054dc: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 0000000054e0: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000054e4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000054f4: 8c7e007e
	s_mov_b32 s0, 0                                            // 0000000054f8: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054fc: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000005500: 8b6a007e
	s_wait_alu depctr_sa_sdst(0)                               // 000000005504: bf88ff9e
	s_cbranch_vccz 48                                          // 000000005508: bfa30030 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3acc>
	s_and_b32 s0, s40, exec_lo                                 // 00000000550c: 8b007e28
	s_cselect_b32 s0, 1, 0                                     // 000000005510: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000005514: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000005518: bf078100
	s_cbranch_scc1 46                                          // 00000000551c: bfa2002e <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3ad8>
	v_dual_mov_b32 v31, s37 :: v_dual_lshlrev_b32 v0, 3, v55   // 000000005520: ca220025 1f006e83
	v_mov_b32_e32 v29, s37                                     // 000000005528: 7e3a0225
	v_mov_b32_e32 v25, s37                                     // 00000000552c: 7e320225
	v_mov_b32_e32 v27, s37                                     // 000000005530: 7e360225
	s_delay_alu instid0(valu_dep_4)                            // 000000005534: bf870004
	v_or_b32_e32 v1, 1, v0                                     // 000000005538: 38020081
	v_or_b32_e32 v2, 2, v0                                     // 00000000553c: 38040082
	v_or_b32_e32 v3, 3, v0                                     // 000000005540: 38060083
	v_or_b32_e32 v4, 4, v0                                     // 000000005544: 38080084
	v_or_b32_e32 v5, 5, v0                                     // 000000005548: 380a0085
	v_or_b32_e32 v36, 6, v0                                    // 00000000554c: 38480086
	v_or_b32_e32 v37, 7, v0                                    // 000000005550: 384a0087
	v_or_b32_e32 v30, s36, v0                                  // 000000005554: 383c0024
	v_or_b32_e32 v28, s36, v1                                  // 000000005558: 38380224
	v_or_b32_e32 v24, s36, v2                                  // 00000000555c: 38300424
	v_or_b32_e32 v26, s36, v3                                  // 000000005560: 38340624
	v_mov_b32_e32 v21, s37                                     // 000000005564: 7e2a0225
	v_or_b32_e32 v20, s36, v4                                  // 000000005568: 38280824
	v_mov_b32_e32 v23, s37                                     // 00000000556c: 7e2e0225
	v_or_b32_e32 v22, s36, v5                                  // 000000005570: 382c0a24
	v_mov_b32_e32 v19, s37                                     // 000000005574: 7e260225
	v_or_b32_e32 v18, s36, v36                                 // 000000005578: 38244824
	v_mov_b32_e32 v17, s37                                     // 00000000557c: 7e220225
	v_or_b32_e32 v16, s36, v37                                 // 000000005580: 38204a24
	v_or_b32_e32 v14, s33, v0                                  // 000000005584: 381c0021
	v_mov_b32_e32 v15, s37                                     // 000000005588: 7e1e0225
	v_mov_b32_e32 v13, s37                                     // 00000000558c: 7e1a0225
	v_or_b32_e32 v12, s33, v1                                  // 000000005590: 38180221
	v_mov_b32_e32 v11, s37                                     // 000000005594: 7e160225
	v_or_b32_e32 v10, s33, v2                                  // 000000005598: 38140421
	v_mov_b32_e32 v9, s37                                      // 00000000559c: 7e120225
	v_or_b32_e32 v8, s33, v3                                   // 0000000055a0: 38100621
	v_mov_b32_e32 v7, s37                                      // 0000000055a4: 7e0e0225
	v_or_b32_e32 v6, s33, v4                                   // 0000000055a8: 380c0821
	v_mov_b32_e32 v1, s37                                      // 0000000055ac: 7e020225
	v_or_b32_e32 v0, s33, v5                                   // 0000000055b0: 38000a21
	v_mov_b32_e32 v3, s37                                      // 0000000055b4: 7e060225
	v_or_b32_e32 v2, s33, v36                                  // 0000000055b8: 38044821
	v_mov_b32_e32 v5, s37                                      // 0000000055bc: 7e0a0225
	v_or_b32_e32 v4, s33, v37                                  // 0000000055c0: 38084a21
	s_mov_b32 s0, 0                                            // 0000000055c4: be800080
	s_branch 4                                                 // 0000000055c8: bfa00004 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3adc>
	s_nop 0                                                    // 0000000055cc: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000055d0: bfb60003
	s_endpgm                                                   // 0000000055d4: bfb00000
	s_mov_b32 s0, -1                                           // 0000000055d8: be8000c1
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v97, 0             // 0000000055dc: ca100080 5e600080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055e4: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 0000000055e8: 8b007e00
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v99, 0             // 0000000055ec: ca100080 60620080
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v101, 0            // 0000000055f4: ca100080 62640080
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v37, 0            // 0000000055fc: ca100080 64240080
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v79, 0             // 000000005604: ca100080 4c4e0080
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v83, 0             // 00000000560c: ca100080 50520080
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v85, 0             // 000000005614: ca100080 52540080
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v87, 0             // 00000000561c: ca100080 54560080
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v89, 0             // 000000005624: ca100080 56580080
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v91, 0             // 00000000562c: ca100080 585a0080
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v93, 0             // 000000005634: ca100080 5a5c0080
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v95, 0             // 00000000563c: ca100080 5c5e0080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v73, 0             // 000000005644: ca100080 24480080
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v75, 0             // 00000000564c: ca100080 484a0080
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v77, 0             // 000000005654: ca100080 4a4c0080
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v81, 0             // 00000000565c: ca100080 4e500080
	s_cselect_b32 s0, 1, 0                                     // 000000005664: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000005668: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 00000000566c: bf078100
	s_cbranch_scc1 746                                         // 000000005670: bfa202ea <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x471c>
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v36, 3, v55    // 000000005674: ca220080 25246e83
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[32:33]                // 00000000567c: 7ca84012
	v_mov_b32_e32 v31, s37                                     // 000000005680: 7e3e0225
	v_mov_b32_e32 v19, s37                                     // 000000005684: 7e260225
	s_delay_alu instid0(valu_dep_4)                            // 000000005688: bf870004
	v_or_b32_e32 v1, 2, v36                                    // 00000000568c: 38024882
	v_or_b32_e32 v5, 6, v36                                    // 000000005690: 380a4886
	v_or_b32_e32 v3, 4, v36                                    // 000000005694: 38064884
	s_wait_alu depctr_va_vcc(0)                                // 000000005698: bf88ff9d
	v_cndmask_b32_e32 v38, 0, v32, vcc_lo                      // 00000000569c: 024c4080
	v_or_b32_e32 v14, s33, v36                                 // 0000000056a0: 381c4821
	v_or_b32_e32 v24, s36, v1                                  // 0000000056a4: 38300224
	v_or_b32_e32 v18, s36, v5                                  // 0000000056a8: 38240a24
	v_mov_b32_e32 v25, s37                                     // 0000000056ac: 7e320225
	v_or_b32_e32 v20, s36, v3                                  // 0000000056b0: 38280624
	v_cndmask_b32_e32 v39, 0, v33, vcc_lo                      // 0000000056b4: 024e4280
	v_or_b32_e32 v10, s33, v1                                  // 0000000056b8: 38140221
	v_mov_b32_e32 v21, s37                                     // 0000000056bc: 7e2a0225
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[24:25]                // 0000000056c0: 7ca83010
	v_mov_b32_e32 v29, s37                                     // 0000000056c4: 7e3a0225
	v_or_b32_e32 v42, 7, v36                                   // 0000000056c8: 38544887
	v_dual_mov_b32 v100, v37 :: v_dual_mov_b32 v17, s37        // 0000000056cc: ca100125 64100025
	s_wait_alu depctr_va_vcc(0)                                // 0000000056d4: bf88ff9d
	v_dual_mov_b32 v96, v37 :: v_dual_cndmask_b32 v51, 0, v24  // 0000000056d8: ca120125 60323080
	v_cndmask_b32_e32 v52, 0, v25, vcc_lo                      // 0000000056e0: 02683280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[20:21]                // 0000000056e4: 7ca82810
	v_or_b32_e32 v16, s36, v42                                 // 0000000056e8: 38205424
	v_or_b32_e32 v30, s36, v36                                 // 0000000056ec: 383c4824
	v_dual_mov_b32 v15, s37 :: v_dual_mov_b32 v86, v37         // 0000000056f0: ca100025 0f560125
	v_or_b32_e32 v0, 1, v36                                    // 0000000056f8: 38004881
	s_wait_alu depctr_va_vcc(0)                                // 0000000056fc: bf88ff9d
	v_dual_cndmask_b32 v56, 0, v20 :: v_dual_cndmask_b32 v57, 0, v21// 000000005700: ca522880 38382a80
	v_mov_b32_e32 v98, v37                                     // 000000005708: 7ec40325
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[18:19]                // 00000000570c: 7ca82410
	v_cmp_gt_i64_e64 s0, s[16:17], v[30:31]                    // 000000005710: d4540000 02023c10
	v_or_b32_e32 v28, s36, v0                                  // 000000005718: 38380024
	v_mov_b32_e32 v11, s37                                     // 00000000571c: 7e160225
	v_or_b32_e32 v2, 3, v36                                    // 000000005720: 38044883
	v_or_b32_e32 v4, 5, v36                                    // 000000005724: 38084885
	s_wait_alu depctr_va_vcc(0)                                // 000000005728: bf88ff9d
	v_dual_cndmask_b32 v60, 0, v18 :: v_dual_cndmask_b32 v61, 0, v19// 00000000572c: ca522480 3c3c2680
	v_mov_b32_e32 v94, v37                                     // 000000005734: 7ebc0325
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[16:17]                // 000000005738: 7ca82010
	v_mov_b32_e32 v27, s37                                     // 00000000573c: 7e360225
	s_wait_alu depctr_va_sdst(0)                               // 000000005740: bf88f19f
	v_cndmask_b32_e64 v47, 0, v30, s0                          // 000000005744: d501002f 00023c80
	v_cndmask_b32_e64 v48, 0, s37, s0                          // 00000000574c: d5010030 00004a80
	v_cmp_gt_i64_e64 s0, s[16:17], v[28:29]                    // 000000005754: d4540000 02023810
	v_or_b32_e32 v26, s36, v2                                  // 00000000575c: 38340424
	s_wait_alu depctr_va_vcc(0)                                // 000000005760: bf88ff9d
	v_dual_cndmask_b32 v62, 0, v16 :: v_dual_cndmask_b32 v63, 0, v17// 000000005764: ca522080 3e3e2280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[14:15]                // 00000000576c: 7ca81c10
	v_or_b32_e32 v12, s33, v0                                  // 000000005770: 38180021
	v_dual_mov_b32 v1, s37 :: v_dual_mov_b32 v90, v37          // 000000005774: ca100025 015a0125
	v_or_b32_e32 v0, s33, v4                                   // 00000000577c: 38000821
	s_wait_alu depctr_va_sdst(0)                               // 000000005780: bf88f19f
	v_cndmask_b32_e64 v49, 0, v28, s0                          // 000000005784: d5010031 00023880
	s_wait_alu depctr_va_vcc(0)                                // 00000000578c: bf88ff9d
	v_cndmask_b32_e32 v64, 0, v14, vcc_lo                      // 000000005790: 02801c80
	v_cndmask_b32_e64 v65, 0, s37, vcc_lo                      // 000000005794: d5010041 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[10:11]                // 00000000579c: 7ca81410
	v_cndmask_b32_e64 v50, 0, v29, s0                          // 0000000057a0: d5010032 00023a80
	v_cmp_gt_i64_e64 s0, s[16:17], v[26:27]                    // 0000000057a8: d4540000 02023410
	v_or_b32_e32 v22, s36, v4                                  // 0000000057b0: 382c0824
	v_dual_mov_b32 v13, s37 :: v_dual_mov_b32 v84, v37         // 0000000057b4: ca100025 0d540125
	s_wait_alu depctr_va_vcc(0)                                // 0000000057bc: bf88ff9d
	v_cndmask_b32_e32 v68, 0, v10, vcc_lo                      // 0000000057c0: 02881480
	v_cndmask_b32_e64 v69, 0, s37, vcc_lo                      // 0000000057c4: d5010045 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 0000000057cc: 7ca80010
	v_mov_b32_e32 v23, s37                                     // 0000000057d0: 7e2e0225
	s_wait_alu depctr_va_sdst(0)                               // 0000000057d4: bf88f19f
	v_cndmask_b32_e64 v53, 0, v26, s0                          // 0000000057d8: d5010035 00023480
	v_cndmask_b32_e64 v55, 0, v27, s0                          // 0000000057e0: d5010037 00023680
	v_mov_b32_e32 v9, s37                                      // 0000000057e8: 7e120225
	v_or_b32_e32 v8, s33, v2                                   // 0000000057ec: 38100421
	s_wait_alu depctr_va_vcc(0)                                // 0000000057f0: bf88ff9d
	v_cndmask_b32_e32 v74, 0, v0, vcc_lo                       // 0000000057f4: 02940080
	v_cmp_gt_i64_e64 s0, s[16:17], v[22:23]                    // 0000000057f8: d4540000 02022c10
	v_dual_mov_b32 v7, s37 :: v_dual_mov_b32 v92, v37          // 000000005800: ca100025 075c0125
	v_or_b32_e32 v6, s33, v3                                   // 000000005808: 380c0621
	v_or_b32_e32 v2, s33, v5                                   // 00000000580c: 38040a21
	v_mov_b32_e32 v5, s37                                      // 000000005810: 7e0a0225
	s_wait_alu depctr_va_sdst(0)                               // 000000005814: bf88f19f
	v_cndmask_b32_e64 v58, 0, v22, s0                          // 000000005818: d501003a 00022c80
	v_cndmask_b32_e64 v59, 0, s37, s0                          // 000000005820: d501003b 00004a80
	v_cmp_gt_i64_e64 s0, s[18:19], v[34:35]                    // 000000005828: d4540000 02024412
	v_cmp_gt_i64_e64 s1, s[16:17], v[6:7]                      // 000000005830: d4540001 02020c10
	v_or_b32_e32 v4, s33, v42                                  // 000000005838: 38085421
	v_dual_mov_b32 v3, s37 :: v_dual_mov_b32 v88, v37          // 00000000583c: ca100025 03580125
	s_lshr_b64 s[2:3], s[28:29], 5                             // 000000005844: 8582851c
	v_cndmask_b32_e64 v41, 0, v35, s0                          // 000000005848: d5010029 00024680
	v_cndmask_b32_e64 v40, 0, v34, s0                          // 000000005850: d5010028 00024480
	v_cmp_gt_i64_e64 s0, s[16:17], v[12:13]                    // 000000005858: d4540000 02021810
	v_cndmask_b32_e64 v72, 0, v6, s1                           // 000000005860: d5010048 00060c80
	v_cndmask_b32_e64 v73, 0, s37, s1                          // 000000005868: d5010049 00044a80
	v_cmp_gt_i64_e64 s1, s[16:17], v[4:5]                      // 000000005870: d4540001 02020810
	v_cndmask_b32_e64 v76, 0, s37, vcc_lo                      // 000000005878: d501004c 01a84a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000005880: bf88ff9e
	v_mul_lo_u32 v59, v59, s2                                  // 000000005884: d72c003b 0200053b
	s_wait_alu depctr_va_sdst(0)                               // 00000000588c: bf88f19f
	v_cndmask_b32_e64 v66, 0, v12, s0                          // 000000005890: d5010042 00021880
	v_cndmask_b32_e64 v67, 0, s37, s0                          // 000000005898: d5010043 00004a80
	v_cmp_gt_i64_e64 s0, s[16:17], v[8:9]                      // 0000000058a0: d4540000 02021010
	v_cndmask_b32_e64 v80, 0, s37, s1                          // 0000000058a8: d5010050 00044a80
	v_cndmask_b32_e64 v79, 0, v4, s1                           // 0000000058b0: d501004f 00060880
	v_mul_lo_u32 v65, v65, s2                                  // 0000000058b8: d72c0041 02000541
	v_mul_lo_u32 v76, v76, s2                                  // 0000000058c0: d72c004c 0200054c
	v_mov_b32_e32 v101, v37                                    // 0000000058c8: 7eca0325
	s_wait_alu depctr_va_sdst(0)                               // 0000000058cc: bf88f19f
	v_cndmask_b32_e64 v70, 0, v8, s0                           // 0000000058d0: d5010046 00021080
	v_cndmask_b32_e64 v71, 0, s37, s0                          // 0000000058d8: d5010047 00004a80
	v_add_co_u32 v44, s0, s38, v54                             // 0000000058e0: d700002c 02026c26
	s_wait_alu depctr_va_sdst(0)                               // 0000000058e8: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s39, 0, s0                  // 0000000058ec: d5207c2d 00010027
	v_add_co_u32 v54, s1, s36, v54                             // 0000000058f4: d7000136 02026c24
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000058fc: bf8701a3
	v_add_co_u32 v34, s0, v44, 16                              // 000000005900: d7000022 0201212c
	s_wait_alu depctr_va_sdst(0)                               // 000000005908: bf88f19f
	v_add_co_ci_u32_e64 v35, null, 0, v45, s0                  // 00000000590c: d5207c23 00025a80
	v_cmp_gt_i64_e64 s0, s[16:17], v[2:3]                      // 000000005914: d4540000 02020410
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 00000000591c: bf870223
	v_mad_co_u64_u32 v[42:43], null, s28, v34, v[36:37]        // 000000005920: d6fe7c2a 0492441c
	v_mul_lo_u32 v75, s29, v34                                 // 000000005928: d72c004b 0202441d
	v_mul_lo_u32 v46, s28, v35                                 // 000000005930: d72c002e 0202461c
	v_lshlrev_b64_e32 v[34:35], 2, v[38:39]                    // 000000005938: 3e444c82
	v_mul_lo_u32 v45, s28, v45                                 // 00000000593c: d72c002d 02025a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000005944: bf88f19f
	v_cndmask_b32_e64 v77, 0, v2, s0                           // 000000005948: d501004d 00020480
	v_cndmask_b32_e64 v78, 0, s37, s0                          // 000000005950: d501004e 00004a80
	s_lshr_b32 s0, s29, 5                                      // 000000005958: 8500851d
	v_mov_b32_e32 v99, v37                                     // 00000000595c: 7ec60325
	v_add_co_u32 v34, vcc_lo, s34, v34                         // 000000005960: d7006a22 02024422
	v_add3_u32 v38, v75, v43, v46                              // 000000005968: d6550026 04ba574b
	s_wait_alu depctr_va_vcc(0)                                // 000000005970: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s35, v35, vcc_lo            // 000000005974: d5207c23 01aa4623
	v_add_co_u32 v102, vcc_lo, s30, v42                        // 00000000597c: d7006a66 0202541e
	v_mul_lo_u32 v46, v80, s2                                  // 000000005984: d72c002e 02000550
	v_add_co_ci_u32_e64 v80, null, s37, 0, s1                  // 00000000598c: d5207c50 00050025
	s_wait_alu depctr_va_vcc(0)                                // 000000005994: bf88ff9d
	v_add_co_ci_u32_e64 v103, null, s31, v38, vcc_lo           // 000000005998: d5207c67 01aa4c1f
	v_lshlrev_b64_e32 v[38:39], 2, v[40:41]                    // 0000000059a0: 3e4c5082
	v_mad_co_u64_u32 v[40:41], null, s28, v44, v[36:37]        // 0000000059a4: d6fe7c28 0492581c
	v_mul_lo_u32 v44, s29, v44                                 // 0000000059ac: d72c002c 0202581d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059b4: bf88ff9e
	v_mul_lo_u32 v75, v79, s0                                  // 0000000059b8: d72c004b 0200014f
	v_mad_co_u64_u32 v[42:43], null, v79, s2, 0                // 0000000059c0: d6fe7c2a 0200054f
	v_add_co_u32 v79, vcc_lo, v54, 16                          // 0000000059c8: d7006a4f 02012136
	s_wait_alu depctr_va_vcc(0)                                // 0000000059d0: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v80, vcc_lo              // 0000000059d4: d5207c51 01aaa080
	v_add_co_u32 v38, vcc_lo, s34, v38                         // 0000000059dc: d7006a26 02024c22
	v_add3_u32 v41, v44, v41, v45                              // 0000000059e4: d6550029 04b6532c
	v_add3_u32 v43, v43, v75, v46                              // 0000000059ec: d655002b 04ba972b
	v_mad_co_u64_u32 v[44:45], null, s28, v79, v[36:37]        // 0000000059f4: d6fe7c2c 04929e1c
	v_mul_lo_u32 v46, s28, v81                                 // 0000000059fc: d72c002e 0202a21c
	v_mul_lo_u32 v75, s29, v79                                 // 000000005a04: d72c004b 02029e1d
	s_wait_alu depctr_va_vcc(0)                                // 000000005a0c: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s35, v39, vcc_lo            // 000000005a10: d5207c27 01aa4e23
	v_add_co_u32 v104, vcc_lo, s30, v40                        // 000000005a18: d7006a68 0202501e
	s_wait_alu depctr_va_vcc(0)                                // 000000005a20: bf88ff9d
	v_add_co_ci_u32_e64 v105, null, s31, v41, vcc_lo           // 000000005a24: d5207c69 01aa521f
	v_lshlrev_b64_e32 v[40:41], 2, v[42:43]                    // 000000005a2c: 3e505482
	v_mul_lo_u32 v78, v78, s2                                  // 000000005a30: d72c004e 0200054e
	v_mul_lo_u32 v79, v77, s0                                  // 000000005a38: d72c004f 0200014d
	v_mad_co_u64_u32 v[42:43], null, v77, s2, 0                // 000000005a40: d6fe7c2a 0200054d
	v_add3_u32 v75, v75, v45, v46                              // 000000005a48: d655004b 04ba5b4b
	v_mad_co_u64_u32 v[45:46], null, s28, v54, v[36:37]        // 000000005a50: d6fe7c2d 04926c1c
	v_mul_lo_u32 v36, s28, v80                                 // 000000005a58: d72c0024 0202a01c
	v_mul_lo_u32 v54, s29, v54                                 // 000000005a60: d72c0036 02026c1d
	v_mul_lo_u32 v77, v48, s2                                  // 000000005a68: d72c004d 02000530
	v_mul_lo_u32 v80, v47, s0                                  // 000000005a70: d72c0050 0200012f
	v_mad_co_u64_u32 v[47:48], null, v47, s2, 0                // 000000005a78: d6fe7c2f 0200052f
	v_add_co_u32 v106, vcc_lo, s26, v44                        // 000000005a80: d7006a6a 0202581a
	v_add3_u32 v43, v43, v79, v78                              // 000000005a88: d655002b 053a9f2b
	s_wait_alu depctr_va_vcc(0)                                // 000000005a90: bf88ff9d
	v_add_co_ci_u32_e64 v107, null, s27, v75, vcc_lo           // 000000005a94: d5207c6b 01aa961b
	v_add3_u32 v36, v54, v46, v36                              // 000000005a9c: d6550024 04925d36
	v_mul_lo_u32 v46, v50, s2                                  // 000000005aa4: d72c002e 02000532
	v_add3_u32 v48, v48, v80, v77                              // 000000005aac: d6550030 0536a130
	v_mul_lo_u32 v75, v49, s0                                  // 000000005ab4: d72c004b 02000131
	v_mad_co_u64_u32 v[49:50], null, v49, s2, 0                // 000000005abc: d6fe7c31 02000531
	v_mul_lo_u32 v77, v52, s2                                  // 000000005ac4: d72c004d 02000534
	v_mul_lo_u32 v78, v51, s0                                  // 000000005acc: d72c004e 02000133
	v_mad_co_u64_u32 v[51:52], null, v51, s2, 0                // 000000005ad4: d6fe7c33 02000533
	v_mul_lo_u32 v79, v55, s2                                  // 000000005adc: d72c004f 02000537
	v_mul_lo_u32 v80, v53, s0                                  // 000000005ae4: d72c0050 02000135
	v_mad_co_u64_u32 v[53:54], null, v53, s2, 0                // 000000005aec: d6fe7c35 02000535
	v_mul_lo_u32 v81, v57, s2                                  // 000000005af4: d72c0051 02000539
	v_mul_lo_u32 v83, v58, s0                                  // 000000005afc: d72c0053 0200013a
	v_mad_co_u64_u32 v[57:58], null, v58, s2, 0                // 000000005b04: d6fe7c39 0200053a
	v_add_co_u32 v108, vcc_lo, s26, v45                        // 000000005b0c: d7006a6c 02025a1a
	v_mul_lo_u32 v82, v56, s0                                  // 000000005b14: d72c0052 02000138
	v_mad_co_u64_u32 v[55:56], null, v56, s2, 0                // 000000005b1c: d6fe7c37 02000538
	s_wait_alu depctr_va_vcc(0)                                // 000000005b24: bf88ff9d
	v_add_co_ci_u32_e64 v109, null, s27, v36, vcc_lo           // 000000005b28: d5207c6d 01aa481b
	v_add3_u32 v50, v50, v75, v46                              // 000000005b30: d6550032 04ba9732
	v_add3_u32 v52, v52, v78, v77                              // 000000005b38: d6550034 05369d34
	v_add3_u32 v54, v54, v80, v79                              // 000000005b40: d6550036 053ea136
	v_add3_u32 v58, v58, v83, v59                              // 000000005b48: d655003a 04eea73a
	v_mul_lo_u32 v36, v61, s2                                  // 000000005b50: d72c0024 0200053d
	v_mul_lo_u32 v75, v60, s0                                  // 000000005b58: d72c004b 0200013c
	v_mad_co_u64_u32 v[59:60], null, v60, s2, 0                // 000000005b60: d6fe7c3b 0200053c
	v_mul_lo_u32 v77, v63, s2                                  // 000000005b68: d72c004d 0200053f
	v_mul_lo_u32 v78, v62, s0                                  // 000000005b70: d72c004e 0200013e
	v_mad_co_u64_u32 v[61:62], null, v62, s2, 0                // 000000005b78: d6fe7c3d 0200053e
	v_mul_lo_u32 v79, v64, s0                                  // 000000005b80: d72c004f 02000140
	v_mad_co_u64_u32 v[63:64], null, v64, s2, 0                // 000000005b88: d6fe7c3f 02000540
	v_add3_u32 v56, v56, v82, v81                              // 000000005b90: d6550038 0546a538
	v_mul_lo_u32 v80, v70, s0                                  // 000000005b98: d72c0050 02000146
	v_add3_u32 v60, v60, v75, v36                              // 000000005ba0: d655003c 0492973c
	v_mul_lo_u32 v36, v67, s2                                  // 000000005ba8: d72c0024 02000543
	v_mul_lo_u32 v75, v66, s0                                  // 000000005bb0: d72c004b 02000142
	v_add3_u32 v62, v62, v78, v77                              // 000000005bb8: d655003e 05369d3e
	v_mul_lo_u32 v77, v69, s2                                  // 000000005bc0: d72c004d 02000545
	v_add3_u32 v64, v64, v79, v65                              // 000000005bc8: d6550040 05069f40
	v_mad_co_u64_u32 v[65:66], null, v66, s2, 0                // 000000005bd0: d6fe7c41 02000542
	v_mul_lo_u32 v78, v68, s0                                  // 000000005bd8: d72c004e 02000144
	v_mad_co_u64_u32 v[67:68], null, v68, s2, 0                // 000000005be0: d6fe7c43 02000544
	v_mul_lo_u32 v79, v71, s2                                  // 000000005be8: d72c004f 02000547
	v_mad_co_u64_u32 v[69:70], null, v70, s2, 0                // 000000005bf0: d6fe7c45 02000546
	v_mul_lo_u32 v81, v73, s2                                  // 000000005bf8: d72c0051 02000549
	v_mul_lo_u32 v82, v72, s0                                  // 000000005c00: d72c0052 02000148
	v_mad_co_u64_u32 v[71:72], null, v72, s2, 0                // 000000005c08: d6fe7c47 02000548
	v_mul_lo_u32 v83, v74, s0                                  // 000000005c10: d72c0053 0200014a
	v_mad_co_u64_u32 v[73:74], null, v74, s2, 0                // 000000005c18: d6fe7c49 0200054a
	v_add3_u32 v66, v66, v75, v36                              // 000000005c20: d6550042 04929742
	v_add3_u32 v68, v68, v78, v77                              // 000000005c28: d6550044 05369d44
	v_add3_u32 v70, v70, v80, v79                              // 000000005c30: d6550046 053ea146
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]                    // 000000005c38: 3e545482
	v_lshlrev_b64_e32 v[44:45], 2, v[47:48]                    // 000000005c3c: 3e585e82
	v_add3_u32 v72, v72, v82, v81                              // 000000005c40: d6550048 0546a548
	v_lshlrev_b64_e32 v[46:47], 2, v[49:50]                    // 000000005c48: 3e5c6282
	v_add3_u32 v74, v74, v83, v76                              // 000000005c4c: d655004a 0532a74a
	v_lshlrev_b64_e32 v[48:49], 2, v[51:52]                    // 000000005c54: 3e606682
	v_lshlrev_b64_e32 v[50:51], 2, v[53:54]                    // 000000005c58: 3e646a82
	v_lshlrev_b64_e32 v[52:53], 2, v[55:56]                    // 000000005c5c: 3e686e82
	v_lshlrev_b64_e32 v[54:55], 2, v[57:58]                    // 000000005c60: 3e6c7282
	v_lshlrev_b64_e32 v[56:57], 2, v[59:60]                    // 000000005c64: 3e707682
	v_lshlrev_b64_e32 v[58:59], 2, v[61:62]                    // 000000005c68: 3e747a82
	v_lshlrev_b64_e32 v[60:61], 2, v[63:64]                    // 000000005c6c: 3e787e82
	v_lshlrev_b64_e32 v[62:63], 2, v[65:66]                    // 000000005c70: 3e7c8282
	v_lshlrev_b64_e32 v[64:65], 2, v[67:68]                    // 000000005c74: 3e808682
	v_lshlrev_b64_e32 v[66:67], 2, v[69:70]                    // 000000005c78: 3e848a82
	v_lshlrev_b64_e32 v[68:69], 2, v[71:72]                    // 000000005c7c: 3e888e82
	v_lshlrev_b64_e32 v[70:71], 2, v[73:74]                    // 000000005c80: 3e8c9282
	v_dual_mov_b32 v97, v37 :: v_dual_mov_b32 v36, v37         // 000000005c84: ca100125 61240125
	v_mov_b32_e32 v85, v37                                     // 000000005c8c: 7eaa0325
	v_dual_mov_b32 v83, v37 :: v_dual_mov_b32 v82, v37         // 000000005c90: ca100125 53520125
	v_dual_mov_b32 v80, v37 :: v_dual_mov_b32 v79, v37         // 000000005c98: ca100125 504e0125
	v_dual_mov_b32 v76, v37 :: v_dual_mov_b32 v95, v37         // 000000005ca0: ca100125 4c5e0125
	v_mov_b32_e32 v93, v37                                     // 000000005ca8: 7eba0325
	v_mov_b32_e32 v91, v37                                     // 000000005cac: 7eb60325
	v_mov_b32_e32 v89, v37                                     // 000000005cb0: 7eb20325
	v_mov_b32_e32 v87, v37                                     // 000000005cb4: 7eae0325
	v_dual_mov_b32 v81, v37 :: v_dual_mov_b32 v78, v37         // 000000005cb8: ca100125 514e0125
	v_mov_b32_e32 v77, v37                                     // 000000005cc0: 7e9a0325
	v_dual_mov_b32 v75, v37 :: v_dual_mov_b32 v74, v37         // 000000005cc4: ca100125 4b4a0125
	v_dual_mov_b32 v73, v37 :: v_dual_mov_b32 v72, v37         // 000000005ccc: ca100125 49480125
	s_lshl_b64 s[0:1], s[18:19], 2                             // 000000005cd4: 84808212
	s_mov_b64 s[2:3], 0                                        // 000000005cd8: be820180
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cdc: bf88ff9e
	v_add_co_u32 v110, vcc_lo, v108, s2                        // 000000005ce0: d7006a6e 0200056c
	s_wait_alu depctr_va_vcc(0)                                // 000000005ce8: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, s3, v109, vcc_lo           // 000000005cec: d5207c6f 01aada03
	v_add_co_u32 v126, vcc_lo, v106, s2                        // 000000005cf4: d7006a7e 0200056a
	s_wait_alu depctr_va_vcc(0)                                // 000000005cfc: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s3, v107, vcc_lo           // 000000005d00: d5207c7f 01aad603
	v_add_co_u32 v112, vcc_lo, v104, s2                        // 000000005d08: d7006a70 02000568
	s_wait_alu depctr_va_vcc(0)                                // 000000005d10: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, s3, v105, vcc_lo           // 000000005d14: d5207c71 01aad203
	v_add_co_u32 v118, vcc_lo, v102, s2                        // 000000005d1c: d7006a76 02000566
	s_wait_alu depctr_va_vcc(0)                                // 000000005d24: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s3, v103, vcc_lo           // 000000005d28: d5207c77 01aace03
	v_add_co_u32 v114, vcc_lo, s22, v44                        // 000000005d30: d7006a72 02025816
	s_wait_alu depctr_va_vcc(0)                                // 000000005d38: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, s23, v45, vcc_lo           // 000000005d3c: d5207c73 01aa5a17
	v_add_co_u32 v116, vcc_lo, s22, v46                        // 000000005d44: d7006a74 02025c16
	s_wait_alu depctr_va_vcc(0)                                // 000000005d4c: bf88ff9d
	v_add_co_ci_u32_e64 v117, null, s23, v47, vcc_lo           // 000000005d50: d5207c75 01aa5e17
	v_add_co_u32 v120, vcc_lo, s22, v48                        // 000000005d58: d7006a78 02026016
	s_wait_alu depctr_va_vcc(0)                                // 000000005d60: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s23, v49, vcc_lo           // 000000005d64: d5207c79 01aa6217
	v_add_co_u32 v122, vcc_lo, s22, v50                        // 000000005d6c: d7006a7a 02026416
	s_wait_alu depctr_va_vcc(0)                                // 000000005d74: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s23, v51, vcc_lo           // 000000005d78: d5207c7b 01aa6617
	v_add_co_u32 v124, vcc_lo, s22, v52                        // 000000005d80: d7006a7c 02026816
	s_wait_alu depctr_va_vcc(0)                                // 000000005d88: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v53, vcc_lo           // 000000005d8c: d5207c7d 01aa6a17
	v_add_co_u32 v128, vcc_lo, s22, v54                        // 000000005d94: d7006a80 02026c16
	s_wait_alu depctr_va_vcc(0)                                // 000000005d9c: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, s23, v55, vcc_lo           // 000000005da0: d5207c81 01aa6e17
	v_add_co_u32 v130, vcc_lo, s22, v56                        // 000000005da8: d7006a82 02027016
	global_load_b32 v144, v[34:35], off                        // 000000005db0: ee05007c 00000090 00000022
	s_wait_alu depctr_va_vcc(0)                                // 000000005dbc: bf88ff9d
	v_add_co_ci_u32_e64 v131, null, s23, v57, vcc_lo           // 000000005dc0: d5207c83 01aa7217
	v_add_co_u32 v132, vcc_lo, s22, v58                        // 000000005dc8: d7006a84 02027416
	s_wait_alu depctr_va_vcc(0)                                // 000000005dd0: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v59, vcc_lo           // 000000005dd4: d5207c85 01aa7617
	global_load_b64 v[134:135], v[112:113], off                // 000000005ddc: ee05407c 00000086 00000070
	s_clause 0x2                                               // 000000005de8: bf850002
	global_load_b64 v[136:137], v[110:111], off                // 000000005dec: ee05407c 00000088 0000006e
	global_load_b64 v[138:139], v[126:127], off offset:16      // 000000005df8: ee05407c 0000008a 0000107e
	global_load_b64 v[140:141], v[110:111], off offset:16      // 000000005e04: ee05407c 0000008c 0000106e
	s_clause 0x5                                               // 000000005e10: bf850005
	global_load_b32 v145, v[114:115], off                      // 000000005e14: ee05007c 00000091 00000072
	global_load_b32 v146, v[116:117], off                      // 000000005e20: ee05007c 00000092 00000074
	global_load_b32 v147, v[120:121], off                      // 000000005e2c: ee05007c 00000093 00000078
	global_load_b32 v148, v[122:123], off                      // 000000005e38: ee05007c 00000094 0000007a
	global_load_b32 v149, v[124:125], off                      // 000000005e44: ee05007c 00000095 0000007c
	global_load_b32 v150, v[128:129], off                      // 000000005e50: ee05007c 00000096 00000080
	s_clause 0x1                                               // 000000005e5c: bf850001
	global_load_b64 v[128:129], v[118:119], off offset:16      // 000000005e60: ee05407c 00000080 00001076
	global_load_b64 v[142:143], v[112:113], off offset:16      // 000000005e6c: ee05407c 0000008e 00001070
	s_clause 0x1                                               // 000000005e78: bf850001
	global_load_b32 v151, v[130:131], off                      // 000000005e7c: ee05007c 00000097 00000082
	global_load_b32 v132, v[132:133], off                      // 000000005e88: ee05007c 00000084 00000084
	global_load_b64 v[130:131], v[118:119], off                // 000000005e94: ee05407c 00000082 00000076
	s_add_nc_u64 s[2:3], s[2:3], 32                            // 000000005ea0: a982a002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ea4: bf88ff9e
	v_cmp_lt_i64_e64 s4, s[2:3], s[24:25]                      // 000000005ea8: d4510004 02003002
	s_wait_loadcnt 0xd                                         // 000000005eb0: bfc0000d
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[136:137], v[134:135], 0// 000000005eb4: cc46406e 1a030d88
	s_wait_loadcnt 0x9                                         // 000000005ebc: bfc00009
	v_dual_mul_f32 v120, v145, v144 :: v_dual_mul_f32 v121, v144, v146// 000000005ec0: c8c72191 78792590
	s_wait_loadcnt 0x7                                         // 000000005ec8: bfc00007
	v_dual_mul_f32 v122, v144, v147 :: v_dual_mul_f32 v123, v144, v148// 000000005ecc: c8c72790 7a7b2990
	s_wait_loadcnt 0x5                                         // 000000005ed4: bfc00005
	v_dual_mul_f32 v124, v144, v149 :: v_dual_mul_f32 v125, v144, v150// 000000005ed8: c8c72b90 7c7d2d90
	s_wait_loadcnt 0x3                                         // 000000005ee0: bfc00003
	v_wmma_f32_16x16x16_fp8_fp8 v[110:117], v[140:141], v[142:143], v[110:117]// 000000005ee4: cc46406e 1dbb1d8c
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000005eec: bf8701b1
	v_dual_mul_f32 v110, v110, v120 :: v_dual_mul_f32 v111, v111, v121// 000000005ef0: c8c6f16e 6e6ef36f
	s_wait_loadcnt 0x2                                         // 000000005ef8: bfc00002
	v_mul_f32_e32 v120, v144, v151                             // 000000005efc: 10f12f90
	v_dual_mul_f32 v112, v112, v122 :: v_dual_mul_f32 v113, v113, v123// 000000005f00: c8c6f570 7070f771
	v_dual_mul_f32 v114, v114, v124 :: v_dual_mul_f32 v115, v115, v125// 000000005f08: c8c6f972 7272fb73
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000005f10: bf870143
	v_mul_f32_e32 v116, v116, v120                             // 000000005f14: 10e8f174
	s_wait_loadcnt 0x1                                         // 000000005f18: bfc00001
	v_dual_mul_f32 v120, v144, v132 :: v_dual_add_f32 v37, v37, v110// 000000005f1c: c8c90990 7824dd25
	v_dual_add_f32 v100, v100, v112 :: v_dual_add_f32 v101, v101, v111// 000000005f24: c908e164 6464df65
	v_dual_add_f32 v98, v98, v114 :: v_dual_mul_f32 v117, v117, v120// 000000005f2c: c906e562 6274f175
	s_wait_loadcnt 0x0                                         // 000000005f34: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[136:137], v[130:131], 0// 000000005f38: cc464076 1a030588
	global_load_b32 v136, v[38:39], off                        // 000000005f40: ee05007c 00000088 00000026
	v_dual_add_f32 v99, v99, v113 :: v_dual_add_f32 v96, v96, v116// 000000005f4c: c908e363 6360e960
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[140:141], v[128:129], v[118:125]// 000000005f54: cc464076 1ddb018c
	v_dual_add_f32 v97, v97, v115 :: v_dual_add_f32 v94, v94, v117// 000000005f5c: c908e761 615eeb5e
	s_wait_loadcnt 0x0                                         // 000000005f64: bfc00000
	v_mul_f32_e32 v133, v145, v136                             // 000000005f68: 110b1191
	v_dual_mul_f32 v137, v146, v136 :: v_dual_mul_f32 v140, v147, v136// 000000005f6c: c8c71192 898d1193
	v_dual_mul_f32 v141, v148, v136 :: v_dual_mul_f32 v146, v150, v136// 000000005f74: c8c71194 8d931196
	v_dual_mul_f32 v145, v149, v136 :: v_dual_mul_f32 v132, v132, v136// 000000005f7c: c8c71195 91851184
	s_delay_alu instid0(valu_dep_4)                            // 000000005f84: bf870004
	v_dual_mul_f32 v147, v151, v136 :: v_dual_mul_f32 v148, v118, v133// 000000005f88: c8c71197 93950b76
	v_add_co_u32 v118, vcc_lo, s22, v60                        // 000000005f90: d7006a76 02027816
	v_dual_mul_f32 v137, v119, v137 :: v_dual_mul_f32 v140, v120, v140// 000000005f98: c8c71377 898d1978
	v_mul_f32_e32 v141, v121, v141                             // 000000005fa0: 111b1b79
	s_wait_alu depctr_va_vcc(0)                                // 000000005fa4: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s23, v61, vcc_lo           // 000000005fa8: d5207c77 01aa7a17
	v_add_co_u32 v120, vcc_lo, s22, v62                        // 000000005fb0: d7006a78 02027c16
	v_dual_mul_f32 v145, v122, v145 :: v_dual_mul_f32 v146, v123, v146// 000000005fb8: c8c7237a 9193257b
	s_wait_alu depctr_va_vcc(0)                                // 000000005fc0: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s23, v63, vcc_lo           // 000000005fc4: d5207c79 01aa7e17
	v_add_co_u32 v122, vcc_lo, s22, v64                        // 000000005fcc: d7006a7a 02028016
	s_wait_alu depctr_va_vcc(0)                                // 000000005fd4: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s23, v65, vcc_lo           // 000000005fd8: d5207c7b 01aa8217
	global_load_b64 v[126:127], v[126:127], off                // 000000005fe0: ee05407c 0000007e 0000007e
	s_clause 0x2                                               // 000000005fec: bf850002
	global_load_b32 v150, v[118:119], off                      // 000000005ff0: ee05007c 00000096 00000076
	global_load_b32 v151, v[120:121], off                      // 000000005ffc: ee05007c 00000097 00000078
	global_load_b32 v152, v[122:123], off                      // 000000006008: ee05007c 00000098 0000007a
	v_mul_f32_e32 v149, v125, v132                             // 000000006014: 112b097d
	v_add_co_u32 v132, vcc_lo, s22, v66                        // 000000006018: d7006a84 02028416
	s_wait_alu depctr_va_vcc(0)                                // 000000006020: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v67, vcc_lo           // 000000006024: d5207c85 01aa8617
	v_dual_mul_f32 v147, v124, v147 :: v_dual_add_f32 v86, v86, v148// 00000000602c: c8c9277c 93572956
	v_dual_add_f32 v85, v85, v137 :: v_dual_add_f32 v84, v84, v140// 000000006034: c9091355 55551954
	v_add_f32_e32 v83, v83, v141                               // 00000000603c: 06a71b53
	s_wait_loadcnt 0x3                                         // 000000006040: bfc00003
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[126:127], v[134:135], 0// 000000006044: cc464076 1a030d7e
	global_load_b32 v134, v[132:133], off                      // 00000000604c: ee05007c 00000086 00000084
	v_add_co_u32 v132, vcc_lo, s22, v68                        // 000000006058: d7006a84 02028816
	s_wait_alu depctr_va_vcc(0)                                // 000000006060: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v69, vcc_lo           // 000000006064: d5207c85 01aa8a17
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[138:139], v[142:143], v[118:125]// 00000000606c: cc464076 1ddb1d8a
	s_wait_loadcnt 0x1                                         // 000000006074: bfc00001
	v_mul_f32_e32 v154, v144, v152                             // 000000006078: 11353190
	global_load_b32 v135, v[132:133], off                      // 00000000607c: ee05007c 00000087 00000084
	v_add_co_u32 v132, vcc_lo, s22, v70                        // 000000006088: d7006a84 02028c16
	s_wait_alu depctr_va_vcc(0)                                // 000000006090: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v71, vcc_lo           // 000000006094: d5207c85 01aa8e17
	v_mul_f32_e32 v154, v120, v154                             // 00000000609c: 11353578
	global_load_b32 v142, v[132:133], off                      // 0000000060a0: ee05007c 0000008e 00000084
	v_add_co_u32 v132, vcc_lo, s22, v42                        // 0000000060ac: d7006a84 02025416
	s_wait_alu depctr_va_vcc(0)                                // 0000000060b4: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v43, vcc_lo           // 0000000060b8: d5207c85 01aa5617
	v_add_f32_e32 v92, v92, v154                               // 0000000060c0: 06b9355c
	global_load_b32 v143, v[132:133], off                      // 0000000060c4: ee05007c 0000008f 00000084
	v_add_co_u32 v132, vcc_lo, s22, v40                        // 0000000060d0: d7006a84 02025016
	s_wait_alu depctr_va_vcc(0)                                // 0000000060d8: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s23, v41, vcc_lo           // 0000000060dc: d5207c85 01aa5217
	v_add_co_u32 v34, vcc_lo, v34, s0                          // 0000000060e4: d7006a22 02000122
	s_wait_alu depctr_va_vcc(0)                                // 0000000060ec: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s1, v35, vcc_lo             // 0000000060f0: d5207c23 01aa4601
	global_load_b32 v132, v[132:133], off                      // 0000000060f8: ee05007c 00000084 00000084
	v_mul_f32_e32 v133, v144, v150                             // 000000006104: 110b2d90
	v_mul_f32_e32 v153, v144, v151                             // 000000006108: 11332f90
	v_add_co_u32 v38, vcc_lo, v38, s0                          // 00000000610c: d7006a26 02000126
	s_wait_alu depctr_va_vcc(0)                                // 000000006114: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s1, v39, vcc_lo             // 000000006118: d5207c27 01aa4e01
	v_dual_mul_f32 v133, v118, v133 :: v_dual_add_f32 v80, v80, v146// 000000006120: c8c90b76 85512550
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000006128: 8b6a047e
	s_add_nc_u64 s[22:23], s[22:23], 4                         // 00000000612c: a9968416
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000006130: bf8700b1
	v_add_f32_e32 v95, v95, v133                               // 000000006134: 06bf0b5f
	s_wait_loadcnt 0x3                                         // 000000006138: bfc00003
	v_dual_mul_f32 v155, v144, v134 :: v_dual_mul_f32 v156, v144, v135// 00000000613c: c8c70d90 9b9d0f90
	v_mul_f32_e32 v156, v122, v156                             // 000000006144: 1139397a
	v_dual_add_f32 v82, v82, v145 :: v_dual_add_f32 v79, v79, v147// 000000006148: c9092352 524f274f
	s_wait_loadcnt 0x1                                         // 000000006150: bfc00001
	v_dual_mul_f32 v157, v144, v142 :: v_dual_mul_f32 v158, v144, v143// 000000006154: c8c71d90 9d9f1f90
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 00000000615c: bf870141
	v_dual_mul_f32 v155, v121, v155 :: v_dual_mul_f32 v158, v124, v158// 000000006160: c8c73779 9b9f3d7c
	s_wait_loadcnt 0x0                                         // 000000006168: bfc00000
	v_dual_mul_f32 v144, v144, v132 :: v_dual_mul_f32 v153, v119, v153// 00000000616c: c8c70990 90993377
	v_mul_f32_e32 v132, v136, v132                             // 000000006174: 11090988
	v_dual_mul_f32 v144, v125, v144 :: v_dual_mul_f32 v157, v123, v157// 000000006178: c8c7217d 909d3b7b
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[126:127], v[130:131], 0// 000000006180: cc464076 1a03057e
	v_dual_mul_f32 v126, v136, v150 :: v_dual_mul_f32 v127, v136, v151// 000000006188: c8c72d88 7e7f2f88
	v_mul_f32_e32 v130, v136, v135                             // 000000006190: 11050f88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000006194: bf8701b3
	v_wmma_f32_16x16x16_fp8_fp8 v[118:125], v[138:139], v[128:129], v[118:125]// 000000006198: cc464076 1ddb018a
	v_dual_mul_f32 v128, v136, v152 :: v_dual_mul_f32 v129, v136, v134// 0000000061a0: c8c73188 80810d88
	v_dual_mul_f32 v131, v136, v142 :: v_dual_mul_f32 v134, v136, v143// 0000000061a8: c8c71d88 83871f88
	v_dual_mul_f32 v118, v118, v126 :: v_dual_mul_f32 v119, v119, v127// 0000000061b0: c8c6fd76 7676ff77
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000061b8: bf870193
	v_dual_mul_f32 v120, v120, v128 :: v_dual_mul_f32 v121, v121, v129// 0000000061bc: c8c70178 78790379
	v_dual_mul_f32 v122, v122, v130 :: v_dual_mul_f32 v123, v123, v131// 0000000061c4: c8c7057a 7a7b077b
	s_delay_alu instid0(valu_dep_4)                            // 0000000061cc: bf870004
	v_dual_mul_f32 v124, v124, v134 :: v_dual_mul_f32 v125, v125, v132// 0000000061d0: c8c70d7c 7c7d097d
	v_dual_add_f32 v76, v76, v149 :: v_dual_add_f32 v91, v91, v155// 0000000061d8: c9092b4c 4c5b375b
	v_dual_add_f32 v93, v93, v153 :: v_dual_add_f32 v90, v90, v156// 0000000061e0: c909335d 5d5b395a
	v_dual_add_f32 v89, v89, v157 :: v_dual_add_f32 v88, v88, v158// 0000000061e8: c9093b59 59593d58
	v_add_f32_e32 v87, v87, v144                               // 0000000061f0: 06af2157
	v_dual_add_f32 v81, v81, v118 :: v_dual_add_f32 v78, v78, v119// 0000000061f4: c908ed51 514eef4e
	v_dual_add_f32 v77, v77, v120 :: v_dual_add_f32 v74, v74, v122// 0000000061fc: c908f14d 4d4af54a
	v_dual_add_f32 v75, v75, v121 :: v_dual_add_f32 v72, v72, v124// 000000006204: c908f34b 4b48f948
	v_dual_add_f32 v73, v73, v123 :: v_dual_add_f32 v36, v36, v125// 00000000620c: c908f749 4924fb24
	s_wait_alu depctr_sa_sdst(0)                               // 000000006214: bf88ff9e
	s_cbranch_vccnz 65200                                      // 000000006218: bfa4feb0 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x41dc>
	v_mul_lo_u32 v34, s19, v30                                 // 00000000621c: d72c0022 02023c13
	v_mul_lo_u32 v35, s18, v31                                 // 000000006224: d72c0023 02023e12
	v_mad_co_u64_u32 v[30:31], null, s18, v30, 0               // 00000000622c: d6fe7c1e 02023c12
	v_mul_lo_u32 v38, s19, v28                                 // 000000006234: d72c0026 02023813
	v_mul_lo_u32 v39, s18, v29                                 // 00000000623c: d72c0027 02023a12
	v_mad_co_u64_u32 v[28:29], null, s18, v28, 0               // 000000006244: d6fe7c1c 02023812
	v_lshlrev_b64_e32 v[32:33], 1, v[32:33]                    // 00000000624c: 3e404081
	v_mul_lo_u32 v41, s19, v24                                 // 000000006250: d72c0029 02023013
	v_bfe_u32 v40, v101, 16, 1                                 // 000000006258: d6100028 02052165
	v_add3_u32 v31, v31, v35, v34                              // 000000006260: d655001f 048a471f
	v_bfe_u32 v34, v37, 16, 1                                  // 000000006268: d6100022 02052125
	v_or_b32_e32 v35, 0x400000, v37                            // 000000006270: 38464aff 00400000
	v_add3_u32 v29, v29, v39, v38                              // 000000006278: d655001d 049a4f1d
	v_or_b32_e32 v38, 0x400000, v101                           // 000000006280: 384ccaff 00400000
	v_lshlrev_b64_e32 v[30:31], 1, v[30:31]                    // 000000006288: 3e3c3c81
	v_add3_u32 v34, v34, v37, 0x7fff                           // 00000000628c: d6550022 03fe4b22 00007fff
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000006298: bf870194
	v_lshlrev_b64_e32 v[28:29], 1, v[28:29]                    // 00000000629c: 3e383881
	v_add_co_u32 v30, vcc_lo, s20, v30                         // 0000000062a0: d7006a1e 02023c14
	s_wait_alu depctr_va_vcc(0)                                // 0000000062a8: bf88ff9d
	s_delay_alu instid0(valu_dep_4)                            // 0000000062ac: bf870004
	v_add_co_ci_u32_e64 v31, null, s21, v31, vcc_lo            // 0000000062b0: d5207c1f 01aa3e15
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 0000000062b8: 7c304b25
	v_add3_u32 v37, v40, v101, 0x7fff                          // 0000000062bc: d6550025 03fecb28 00007fff
	v_mul_lo_u32 v40, s19, v26                                 // 0000000062c8: d72c0028 02023413
	s_wait_alu depctr_va_vcc(0)                                // 0000000062d0: bf88ff9d
	v_cndmask_b32_e32 v34, v34, v35, vcc_lo                    // 0000000062d4: 02444722
	v_mul_lo_u32 v35, s18, v25                                 // 0000000062d8: d72c0023 02023212
	v_mad_co_u64_u32 v[24:25], null, s18, v24, 0               // 0000000062e0: d6fe7c18 02023012
	v_add_co_u32 v30, vcc_lo, v30, v32                         // 0000000062e8: d7006a1e 0202411e
	s_wait_alu depctr_va_vcc(0)                                // 0000000062f0: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, v31, v33, vcc_lo            // 0000000062f4: d5207c1f 01aa431f
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 0000000062fc: 7c30cb65
	s_delay_alu instid0(valu_dep_4)                            // 000000006300: bf870004
	v_add3_u32 v25, v25, v35, v41                              // 000000006304: d6550019 04a64719
	global_store_d16_hi_b16 v[30:31], v34, off                 // 00000000630c: ee09407c 11000000 0000001e
	s_wait_alu depctr_va_vcc(0)                                // 000000006318: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v38, vcc_lo                    // 00000000631c: 02444d25
	v_add_co_u32 v28, vcc_lo, s20, v28                         // 000000006320: d7006a1c 02023814
	v_lshlrev_b64_e32 v[24:25], 1, v[24:25]                    // 000000006328: 3e303081
	s_wait_alu depctr_va_vcc(0)                                // 00000000632c: bf88ff9d
	v_add_co_ci_u32_e64 v29, null, s21, v29, vcc_lo            // 000000006330: d5207c1d 01aa3a15
	v_bfe_u32 v35, v100, 16, 1                                 // 000000006338: d6100023 02052164
	v_add_co_u32 v28, vcc_lo, v28, v32                         // 000000006340: d7006a1c 0202411c
	s_wait_alu depctr_va_vcc(0)                                // 000000006348: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000634c: bf870003
	v_add_co_ci_u32_e64 v29, null, v29, v33, vcc_lo            // 000000006350: d5207c1d 01aa431d
	v_add_co_u32 v38, vcc_lo, s20, v24                         // 000000006358: d7006a26 02023014
	v_add3_u32 v35, v35, v100, 0x7fff                          // 000000006360: d6550023 03fec923 00007fff
	v_or_b32_e32 v37, 0x400000, v100                           // 00000000636c: 384ac8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006374: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v25, vcc_lo            // 000000006378: d5207c27 01aa3215
	v_mul_lo_u32 v41, s18, v27                                 // 000000006380: d72c0029 02023612
	v_mad_co_u64_u32 v[24:25], null, s18, v26, 0               // 000000006388: d6fe7c18 02023412
	v_cmp_u_f32_e32 vcc_lo, v100, v100                         // 000000006390: 7c30c964
	global_store_d16_hi_b16 v[28:29], v34, off                 // 000000006394: ee09407c 11000000 0000001c
	s_wait_alu depctr_va_vcc(0)                                // 0000000063a0: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000063a4: 02464b23
	v_add_co_u32 v26, vcc_lo, v38, v32                         // 0000000063a8: d7006a1a 02024126
	s_wait_alu depctr_va_vcc(0)                                // 0000000063b0: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v39, v33, vcc_lo            // 0000000063b4: d5207c1b 01aa4327
	v_mul_lo_u32 v38, s19, v20                                 // 0000000063bc: d72c0026 02022813
	v_mul_lo_u32 v39, s18, v21                                 // 0000000063c4: d72c0027 02022a12
	v_mad_co_u64_u32 v[20:21], null, s18, v20, 0               // 0000000063cc: d6fe7c14 02022812
	v_add3_u32 v25, v25, v41, v40                              // 0000000063d4: d6550019 04a25319
	v_bfe_u32 v37, v99, 16, 1                                  // 0000000063dc: d6100025 02052163
	v_or_b32_e32 v40, 0x400000, v99                            // 0000000063e4: 3850c6ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v99, v99                           // 0000000063ec: 7c30c763
	global_store_d16_hi_b16 v[26:27], v35, off                 // 0000000063f0: ee09407c 11800000 0000001a
	v_lshlrev_b64_e32 v[24:25], 1, v[24:25]                    // 0000000063fc: 3e303081
	v_add3_u32 v37, v37, v99, 0x7fff                           // 000000006400: d6550025 03fec725 00007fff
	v_add3_u32 v21, v21, v39, v38                              // 00000000640c: d6550015 049a4f15
	v_bfe_u32 v35, v98, 16, 1                                  // 000000006414: d6100023 02052162
	v_mul_lo_u32 v41, s18, v23                                 // 00000000641c: d72c0029 02022e12
	s_wait_alu depctr_va_vcc(0)                                // 000000006424: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006428: 02445125
	v_add_co_u32 v24, vcc_lo, s20, v24                         // 00000000642c: d7006a18 02023014
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000006434: 3e282881
	s_wait_alu depctr_va_vcc(0)                                // 000000006438: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s21, v25, vcc_lo            // 00000000643c: d5207c19 01aa3215
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000006444: bf8701b3
	v_add_co_u32 v24, vcc_lo, v24, v32                         // 000000006448: d7006a18 02024118
	v_add3_u32 v35, v35, v98, 0x7fff                           // 000000006450: d6550023 03fec523 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000645c: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v25, v33, vcc_lo            // 000000006460: d5207c19 01aa4319
	v_add_co_u32 v38, vcc_lo, s20, v20                         // 000000006468: d7006a26 02022814
	v_or_b32_e32 v37, 0x400000, v98                            // 000000006470: 384ac4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006478: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v21, vcc_lo            // 00000000647c: d5207c27 01aa2a15
	v_mul_lo_u32 v40, s19, v22                                 // 000000006484: d72c0028 02022c13
	v_mad_co_u64_u32 v[20:21], null, s18, v22, 0               // 00000000648c: d6fe7c14 02022c12
	v_cmp_u_f32_e32 vcc_lo, v98, v98                           // 000000006494: 7c30c562
	global_store_d16_hi_b16 v[24:25], v34, off                 // 000000006498: ee09407c 11000000 00000018
	s_wait_alu depctr_va_vcc(0)                                // 0000000064a4: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000064a8: 02464b23
	v_add_co_u32 v22, vcc_lo, v38, v32                         // 0000000064ac: d7006a16 02024126
	s_wait_alu depctr_va_vcc(0)                                // 0000000064b4: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, v39, v33, vcc_lo            // 0000000064b8: d5207c17 01aa4327
	v_mul_lo_u32 v38, s19, v18                                 // 0000000064c0: d72c0026 02022413
	v_mul_lo_u32 v39, s18, v19                                 // 0000000064c8: d72c0027 02022612
	v_mad_co_u64_u32 v[18:19], null, s18, v18, 0               // 0000000064d0: d6fe7c12 02022412
	v_add3_u32 v21, v21, v41, v40                              // 0000000064d8: d6550015 04a25315
	v_bfe_u32 v37, v97, 16, 1                                  // 0000000064e0: d6100025 02052161
	v_or_b32_e32 v40, 0x400000, v97                            // 0000000064e8: 3850c2ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v97, v97                           // 0000000064f0: 7c30c361
	global_store_d16_hi_b16 v[22:23], v35, off                 // 0000000064f4: ee09407c 11800000 00000016
	v_lshlrev_b64_e32 v[20:21], 1, v[20:21]                    // 000000006500: 3e282881
	v_add3_u32 v37, v37, v97, 0x7fff                           // 000000006504: d6550025 03fec325 00007fff
	v_add3_u32 v19, v19, v39, v38                              // 000000006510: d6550013 049a4f13
	v_bfe_u32 v35, v96, 16, 1                                  // 000000006518: d6100023 02052160
	v_mul_lo_u32 v38, s19, v16                                 // 000000006520: d72c0026 02022013
	v_mul_lo_u32 v39, s18, v17                                 // 000000006528: d72c0027 02022212
	s_wait_alu depctr_va_vcc(0)                                // 000000006530: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006534: 02445125
	v_add_co_u32 v20, vcc_lo, s20, v20                         // 000000006538: d7006a14 02022814
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000006540: 3e242481
	s_wait_alu depctr_va_vcc(0)                                // 000000006544: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, s21, v21, vcc_lo            // 000000006548: d5207c15 01aa2a15
	v_mad_co_u64_u32 v[16:17], null, s18, v16, 0               // 000000006550: d6fe7c10 02022012
	v_add_co_u32 v20, vcc_lo, v20, v32                         // 000000006558: d7006a14 02024114
	s_wait_alu depctr_va_vcc(0)                                // 000000006560: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000006564: bf870003
	v_add_co_ci_u32_e64 v21, null, v21, v33, vcc_lo            // 000000006568: d5207c15 01aa4315
	v_add_co_u32 v18, vcc_lo, s20, v18                         // 000000006570: d7006a12 02022414
	v_add3_u32 v35, v35, v96, 0x7fff                           // 000000006578: d6550023 03fec123 00007fff
	v_or_b32_e32 v37, 0x400000, v96                            // 000000006584: 384ac0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000658c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s21, v19, vcc_lo            // 000000006590: d5207c13 01aa2615
	v_cmp_u_f32_e32 vcc_lo, v96, v96                           // 000000006598: 7c30c160
	v_add3_u32 v17, v17, v39, v38                              // 00000000659c: d6550011 049a4f11
	v_mul_lo_u32 v38, s19, v14                                 // 0000000065a4: d72c0026 02021c13
	v_mul_lo_u32 v39, s18, v15                                 // 0000000065ac: d72c0027 02021e12
	v_mad_co_u64_u32 v[14:15], null, s18, v14, 0               // 0000000065b4: d6fe7c0e 02021c12
	s_wait_alu depctr_va_vcc(0)                                // 0000000065bc: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000065c0: 02464b23
	v_bfe_u32 v37, v94, 16, 1                                  // 0000000065c4: d6100025 0205215e
	v_add_co_u32 v18, vcc_lo, v18, v32                         // 0000000065cc: d7006a12 02024112
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 0000000065d4: 3e202081
	s_wait_alu depctr_va_vcc(0)                                // 0000000065d8: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v19, v33, vcc_lo            // 0000000065dc: d5207c13 01aa4313
	v_add3_u32 v37, v37, v94, 0x7fff                           // 0000000065e4: d6550025 03febd25 00007fff
	v_or_b32_e32 v40, 0x400000, v94                            // 0000000065f0: 3850bcff 00400000
	v_cmp_u_f32_e32 vcc_lo, v94, v94                           // 0000000065f8: 7c30bd5e
	v_add3_u32 v15, v15, v39, v38                              // 0000000065fc: d655000f 049a4f0f
	global_store_d16_hi_b16 v[20:21], v34, off                 // 000000006604: ee09407c 11000000 00000014
	v_mul_lo_u32 v38, s19, v12                                 // 000000006610: d72c0026 02021813
	v_mul_lo_u32 v39, s18, v13                                 // 000000006618: d72c0027 02021a12
	s_wait_alu depctr_va_vcc(0)                                // 000000006620: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006624: 02445125
	v_add_co_u32 v16, vcc_lo, s20, v16                         // 000000006628: d7006a10 02022014
	v_mad_co_u64_u32 v[12:13], null, s18, v12, 0               // 000000006630: d6fe7c0c 02021812
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000006638: 3e1c1c81
	s_wait_alu depctr_va_vcc(0)                                // 00000000663c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo            // 000000006640: d5207c11 01aa2215
	global_store_d16_hi_b16 v[18:19], v35, off                 // 000000006648: ee09407c 11800000 00000012
	v_bfe_u32 v35, v95, 16, 1                                  // 000000006654: d6100023 0205215f
	v_add_co_u32 v16, vcc_lo, v16, v32                         // 00000000665c: d7006a10 02024110
	s_wait_alu depctr_va_vcc(0)                                // 000000006664: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v17, v33, vcc_lo            // 000000006668: d5207c11 01aa4311
	v_add_co_u32 v14, vcc_lo, s20, v14                         // 000000006670: d7006a0e 02021c14
	v_add3_u32 v13, v13, v39, v38                              // 000000006678: d655000d 049a4f0d
	v_mul_lo_u32 v38, s19, v10                                 // 000000006680: d72c0026 02021413
	v_mul_lo_u32 v39, s18, v11                                 // 000000006688: d72c0027 02021612
	v_mad_co_u64_u32 v[10:11], null, s18, v10, 0               // 000000006690: d6fe7c0a 02021412
	v_add3_u32 v35, v35, v95, 0x7fff                           // 000000006698: d6550023 03febf23 00007fff
	v_or_b32_e32 v37, 0x400000, v95                            // 0000000066a4: 384abeff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000066ac: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, s21, v15, vcc_lo            // 0000000066b0: d5207c0f 01aa1e15
	v_cmp_u_f32_e32 vcc_lo, v95, v95                           // 0000000066b8: 7c30bf5f
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 0000000066bc: 3e181881
	v_or_b32_e32 v40, 0x400000, v93                            // 0000000066c0: 3850baff 00400000
	v_add3_u32 v11, v11, v39, v38                              // 0000000066c8: d655000b 049a4f0b
	v_mul_lo_u32 v38, s19, v8                                  // 0000000066d0: d72c0026 02021013
	s_wait_alu depctr_va_vcc(0)                                // 0000000066d8: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000066dc: 02464b23
	v_bfe_u32 v37, v93, 16, 1                                  // 0000000066e0: d6100025 0205215d
	v_mul_lo_u32 v39, s18, v9                                  // 0000000066e8: d72c0027 02021212
	v_mad_co_u64_u32 v[8:9], null, s18, v8, 0                  // 0000000066f0: d6fe7c08 02021012
	v_add_co_u32 v14, vcc_lo, v14, v32                         // 0000000066f8: d7006a0e 0202410e
	s_wait_alu depctr_va_vcc(0)                                // 000000006700: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v15, v33, vcc_lo            // 000000006704: d5207c0f 01aa430f
	v_add3_u32 v37, v37, v93, 0x7fff                           // 00000000670c: d6550025 03febb25 00007fff
	v_cmp_u_f32_e32 vcc_lo, v93, v93                           // 000000006718: 7c30bb5d
	global_store_d16_hi_b16 v[16:17], v34, off                 // 00000000671c: ee09407c 11000000 00000010
	v_add3_u32 v9, v9, v39, v38                                // 000000006728: d6550009 049a4f09
	v_mul_lo_u32 v38, s19, v6                                  // 000000006730: d72c0026 02020c13
	v_mul_lo_u32 v39, s18, v7                                  // 000000006738: d72c0027 02020e12
	s_wait_alu depctr_va_vcc(0)                                // 000000006740: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006744: 02445125
	v_add_co_u32 v12, vcc_lo, s20, v12                         // 000000006748: d7006a0c 02021814
	v_mad_co_u64_u32 v[6:7], null, s18, v6, 0                  // 000000006750: d6fe7c06 02020c12
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000006758: 3e141481
	s_wait_alu depctr_va_vcc(0)                                // 00000000675c: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s21, v13, vcc_lo            // 000000006760: d5207c0d 01aa1a15
	global_store_d16_hi_b16 v[14:15], v35, off                 // 000000006768: ee09407c 11800000 0000000e
	v_bfe_u32 v35, v92, 16, 1                                  // 000000006774: d6100023 0205215c
	v_add_co_u32 v12, vcc_lo, v12, v32                         // 00000000677c: d7006a0c 0202410c
	s_wait_alu depctr_va_vcc(0)                                // 000000006784: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v13, v33, vcc_lo            // 000000006788: d5207c0d 01aa430d
	v_add_co_u32 v10, vcc_lo, s20, v10                         // 000000006790: d7006a0a 02021414
	v_add3_u32 v7, v7, v39, v38                                // 000000006798: d6550007 049a4f07
	v_mul_lo_u32 v38, s19, v0                                  // 0000000067a0: d72c0026 02020013
	v_mul_lo_u32 v39, s18, v1                                  // 0000000067a8: d72c0027 02020212
	v_mad_co_u64_u32 v[0:1], null, s18, v0, 0                  // 0000000067b0: d6fe7c00 02020012
	v_add3_u32 v35, v35, v92, 0x7fff                           // 0000000067b8: d6550023 03feb923 00007fff
	v_or_b32_e32 v37, 0x400000, v92                            // 0000000067c4: 384ab8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000067cc: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s21, v11, vcc_lo            // 0000000067d0: d5207c0b 01aa1615
	v_cmp_u_f32_e32 vcc_lo, v92, v92                           // 0000000067d8: 7c30b95c
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 0000000067dc: 3e101081
	v_or_b32_e32 v40, 0x400000, v91                            // 0000000067e0: 3850b6ff 00400000
	v_add3_u32 v1, v1, v39, v38                                // 0000000067e8: d6550001 049a4f01
	v_mul_lo_u32 v38, s19, v2                                  // 0000000067f0: d72c0026 02020413
	s_wait_alu depctr_va_vcc(0)                                // 0000000067f8: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000067fc: 02464b23
	v_bfe_u32 v37, v91, 16, 1                                  // 000000006800: d6100025 0205215b
	v_mul_lo_u32 v39, s18, v3                                  // 000000006808: d72c0027 02020612
	v_mad_co_u64_u32 v[2:3], null, s18, v2, 0                  // 000000006810: d6fe7c02 02020412
	v_add_co_u32 v10, vcc_lo, v10, v32                         // 000000006818: d7006a0a 0202410a
	s_wait_alu depctr_va_vcc(0)                                // 000000006820: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v11, v33, vcc_lo            // 000000006824: d5207c0b 01aa430b
	v_add3_u32 v37, v37, v91, 0x7fff                           // 00000000682c: d6550025 03feb725 00007fff
	v_cmp_u_f32_e32 vcc_lo, v91, v91                           // 000000006838: 7c30b75b
	global_store_d16_hi_b16 v[12:13], v34, off                 // 00000000683c: ee09407c 11000000 0000000c
	v_add3_u32 v3, v3, v39, v38                                // 000000006848: d6550003 049a4f03
	v_or_b32_e32 v39, 0x400000, v88                            // 000000006850: 384eb0ff 00400000
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000006858: 3e0c0c81
	s_wait_alu depctr_va_vcc(0)                                // 00000000685c: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006860: 02445125
	v_add_co_u32 v8, vcc_lo, s20, v8                           // 000000006864: d7006a08 02021014
	s_wait_alu depctr_va_vcc(0)                                // 00000000686c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo              // 000000006870: d5207c09 01aa1215
	global_store_d16_hi_b16 v[10:11], v35, off                 // 000000006878: ee09407c 11800000 0000000a
	v_bfe_u32 v35, v90, 16, 1                                  // 000000006884: d6100023 0205215a
	v_add_co_u32 v8, vcc_lo, v8, v32                           // 00000000688c: d7006a08 02024108
	s_wait_alu depctr_va_vcc(0)                                // 000000006894: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v9, v33, vcc_lo              // 000000006898: d5207c09 01aa4309
	v_add_co_u32 v6, vcc_lo, s20, v6                           // 0000000068a0: d7006a06 02020c14
	v_add3_u32 v35, v35, v90, 0x7fff                           // 0000000068a8: d6550023 03feb523 00007fff
	v_or_b32_e32 v37, 0x400000, v90                            // 0000000068b4: 384ab4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000068bc: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo              // 0000000068c0: d5207c07 01aa0e15
	v_cmp_u_f32_e32 vcc_lo, v90, v90                           // 0000000068c8: 7c30b55a
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000068cc: 3e000081
	v_or_b32_e32 v40, 0x400000, v89                            // 0000000068d0: 3850b2ff 00400000
	v_mul_lo_u32 v41, s18, v5                                  // 0000000068d8: d72c0029 02020a12
	s_wait_alu depctr_va_vcc(0)                                // 0000000068e0: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v37, vcc_lo                    // 0000000068e4: 02464b23
	v_bfe_u32 v37, v89, 16, 1                                  // 0000000068e8: d6100025 02052159
	v_add_co_u32 v6, vcc_lo, v6, v32                           // 0000000068f0: d7006a06 02024106
	s_wait_alu depctr_va_vcc(0)                                // 0000000068f8: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v33, vcc_lo              // 0000000068fc: d5207c07 01aa4307
	s_delay_alu instid0(valu_dep_3)                            // 000000006904: bf870003
	v_add3_u32 v37, v37, v89, 0x7fff                           // 000000006908: d6550025 03feb325 00007fff
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 000000006914: 7c30b359
	s_clause 0x1                                               // 000000006918: bf850001
	global_store_d16_hi_b16 v[8:9], v34, off                   // 00000000691c: ee09407c 11000000 00000008
	global_store_d16_hi_b16 v[6:7], v35, off                   // 000000006928: ee09407c 11800000 00000006
	v_bfe_u32 v35, v88, 16, 1                                  // 000000006934: d6100023 02052158
	s_wait_alu depctr_va_vcc(0)                                // 00000000693c: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v40, vcc_lo                    // 000000006940: 02445125
	v_add_co_u32 v37, vcc_lo, s20, v0                          // 000000006944: d7006a25 02020014
	s_wait_alu depctr_va_vcc(0)                                // 00000000694c: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, s21, v1, vcc_lo             // 000000006950: d5207c26 01aa0215
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]                        // 000000006958: 3e000481
	v_mul_lo_u32 v40, s19, v4                                  // 00000000695c: d72c0028 02020813
	v_mad_co_u64_u32 v[2:3], null, s18, v4, 0                  // 000000006964: d6fe7c02 02020812
	v_add_co_u32 v4, vcc_lo, v37, v32                          // 00000000696c: d7006a04 02024125
	v_add3_u32 v35, v35, v88, 0x7fff                           // 000000006974: d6550023 03feb123 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006980: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v38, v33, vcc_lo             // 000000006984: d5207c05 01aa4326
	v_cmp_u_f32_e32 vcc_lo, v88, v88                           // 00000000698c: 7c30b158
	v_bfe_u32 v37, v87, 16, 1                                  // 000000006990: d6100025 02052157
	v_add3_u32 v3, v3, v41, v40                                // 000000006998: d6550003 04a25303
	v_or_b32_e32 v38, 0x400000, v87                            // 0000000069a0: 384caeff 00400000
	global_store_d16_hi_b16 v[4:5], v34, off                   // 0000000069a8: ee09407c 11000000 00000004
	s_wait_alu depctr_va_vcc(0)                                // 0000000069b4: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v39, vcc_lo                    // 0000000069b8: 02464f23
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 0000000069bc: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 0000000069c4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 0000000069c8: d5207c01 01aa0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000069d0: 3e040481
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000069d4: bf8701a3
	v_add_co_u32 v0, vcc_lo, v0, v32                           // 0000000069d8: d7006a00 02024100
	s_wait_alu depctr_va_vcc(0)                                // 0000000069e0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v33, vcc_lo              // 0000000069e4: d5207c01 01aa4301
	v_add3_u32 v37, v37, v87, 0x7fff                           // 0000000069ec: d6550025 03feaf25 00007fff
	v_cmp_u_f32_e32 vcc_lo, v87, v87                           // 0000000069f8: 7c30af57
	global_store_d16_hi_b16 v[0:1], v35, off                   // 0000000069fc: ee09407c 11800000 00000000
	v_bfe_u32 v35, v86, 16, 1                                  // 000000006a08: d6100023 02052156
	s_wait_alu depctr_va_vcc(0)                                // 000000006a10: bf88ff9d
	v_cndmask_b32_e32 v34, v37, v38, vcc_lo                    // 000000006a14: 02444d25
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 000000006a18: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006a20: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 000000006a24: d5207c03 01aa0615
	v_add3_u32 v35, v35, v86, 0x7fff                           // 000000006a2c: d6550023 03fead23 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006a38: bf870003
	v_add_co_u32 v2, vcc_lo, v2, v32                           // 000000006a3c: d7006a02 02024102
	v_or_b32_e32 v37, 0x400000, v86                            // 000000006a44: 384aacff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006a4c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v33, vcc_lo              // 000000006a50: d5207c03 01aa4303
	v_bfe_u32 v32, v85, 16, 1                                  // 000000006a58: d6100020 02052155
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 000000006a60: 7c30ad56
	global_store_d16_hi_b16 v[2:3], v34, off                   // 000000006a64: ee09407c 11000000 00000002
	v_or_b32_e32 v34, 0x400000, v85                            // 000000006a70: 3844aaff 00400000
	v_add3_u32 v32, v32, v85, 0x7fff                           // 000000006a78: d6550020 03feab20 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006a84: bf88ff9d
	v_cndmask_b32_e32 v33, v35, v37, vcc_lo                    // 000000006a88: 02424b23
	v_bfe_u32 v35, v84, 16, 1                                  // 000000006a8c: d6100023 02052154
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 000000006a94: 7c30ab55
	global_store_d16_hi_b16 v[30:31], v33, off offset:32       // 000000006a98: ee09407c 10800000 0000201e
	v_add3_u32 v30, v35, v84, 0x7fff                           // 000000006aa4: d655001e 03fea923 00007fff
	v_or_b32_e32 v31, 0x400000, v84                            // 000000006ab0: 383ea8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006ab8: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000006abc: 02404520
	v_bfe_u32 v33, v83, 16, 1                                  // 000000006ac0: d6100021 02052153
	v_cmp_u_f32_e32 vcc_lo, v84, v84                           // 000000006ac8: 7c30a954
	global_store_d16_hi_b16 v[28:29], v32, off offset:32       // 000000006acc: ee09407c 10000000 0000201c
	v_add3_u32 v28, v33, v83, 0x7fff                           // 000000006ad8: d655001c 03fea721 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006ae4: bf88ff9d
	v_cndmask_b32_e32 v30, v30, v31, vcc_lo                    // 000000006ae8: 023c3f1e
	v_bfe_u32 v31, v82, 16, 1                                  // 000000006aec: d610001f 02052152
	v_or_b32_e32 v29, 0x400000, v83                            // 000000006af4: 383aa6ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v83, v83                           // 000000006afc: 7c30a753
	global_store_d16_hi_b16 v[26:27], v30, off offset:32       // 000000006b00: ee09407c 0f000000 0000201a
	v_add3_u32 v26, v31, v82, 0x7fff                           // 000000006b0c: d655001a 03fea51f 00007fff
	v_or_b32_e32 v27, 0x400000, v82                            // 000000006b18: 3836a4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006b20: bf88ff9d
	v_cndmask_b32_e32 v28, v28, v29, vcc_lo                    // 000000006b24: 02383b1c
	v_bfe_u32 v29, v80, 16, 1                                  // 000000006b28: d610001d 02052150
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000006b30: 7c30a552
	global_store_d16_hi_b16 v[24:25], v28, off offset:32       // 000000006b34: ee09407c 0e000000 00002018
	v_add3_u32 v24, v29, v80, 0x7fff                           // 000000006b40: d6550018 03fea11d 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006b4c: bf88ff9d
	v_cndmask_b32_e32 v26, v26, v27, vcc_lo                    // 000000006b50: 0234371a
	v_bfe_u32 v27, v79, 16, 1                                  // 000000006b54: d610001b 0205214f
	v_or_b32_e32 v25, 0x400000, v80                            // 000000006b5c: 3832a0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v80, v80                           // 000000006b64: 7c30a150
	global_store_d16_hi_b16 v[22:23], v26, off offset:32       // 000000006b68: ee09407c 0d000000 00002016
	v_add3_u32 v22, v27, v79, 0x7fff                           // 000000006b74: d6550016 03fe9f1b 00007fff
	v_or_b32_e32 v23, 0x400000, v79                            // 000000006b80: 382e9eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006b88: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo                    // 000000006b8c: 02303318
	v_bfe_u32 v25, v76, 16, 1                                  // 000000006b90: d6100019 0205214c
	v_cmp_u_f32_e32 vcc_lo, v79, v79                           // 000000006b98: 7c309f4f
	global_store_d16_hi_b16 v[20:21], v24, off offset:32       // 000000006b9c: ee09407c 0c000000 00002014
	v_add3_u32 v20, v25, v76, 0x7fff                           // 000000006ba8: d6550014 03fe9919 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006bb4: bf88ff9d
	v_cndmask_b32_e32 v22, v22, v23, vcc_lo                    // 000000006bb8: 022c2f16
	v_bfe_u32 v23, v81, 16, 1                                  // 000000006bbc: d6100017 02052151
	v_or_b32_e32 v21, 0x400000, v76                            // 000000006bc4: 382a98ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v76, v76                           // 000000006bcc: 7c30994c
	global_store_d16_hi_b16 v[18:19], v22, off offset:32       // 000000006bd0: ee09407c 0b000000 00002012
	v_add3_u32 v18, v23, v81, 0x7fff                           // 000000006bdc: d6550012 03fea317 00007fff
	v_or_b32_e32 v19, 0x400000, v81                            // 000000006be8: 3826a2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006bf0: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v21, vcc_lo                    // 000000006bf4: 02282b14
	v_bfe_u32 v21, v78, 16, 1                                  // 000000006bf8: d6100015 0205214e
	v_cmp_u_f32_e32 vcc_lo, v81, v81                           // 000000006c00: 7c30a351
	global_store_d16_hi_b16 v[16:17], v20, off offset:32       // 000000006c04: ee09407c 0a000000 00002010
	v_add3_u32 v16, v21, v78, 0x7fff                           // 000000006c10: d6550010 03fe9d15 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006c1c: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 000000006c20: 02242712
	v_bfe_u32 v19, v77, 16, 1                                  // 000000006c24: d6100013 0205214d
	v_or_b32_e32 v17, 0x400000, v78                            // 000000006c2c: 38229cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v78, v78                           // 000000006c34: 7c309d4e
	global_store_d16_hi_b16 v[14:15], v18, off offset:32       // 000000006c38: ee09407c 09000000 0000200e
	v_add3_u32 v14, v19, v77, 0x7fff                           // 000000006c44: d655000e 03fe9b13 00007fff
	v_or_b32_e32 v15, 0x400000, v77                            // 000000006c50: 381e9aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006c58: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v17, vcc_lo                    // 000000006c5c: 02202310
	v_bfe_u32 v17, v75, 16, 1                                  // 000000006c60: d6100011 0205214b
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000006c68: 7c309b4d
	global_store_d16_hi_b16 v[12:13], v16, off offset:32       // 000000006c6c: ee09407c 08000000 0000200c
	v_add3_u32 v12, v17, v75, 0x7fff                           // 000000006c78: d655000c 03fe9711 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006c84: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v15, vcc_lo                    // 000000006c88: 021c1f0e
	v_bfe_u32 v15, v74, 16, 1                                  // 000000006c8c: d610000f 0205214a
	v_or_b32_e32 v13, 0x400000, v75                            // 000000006c94: 381a96ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 000000006c9c: 7c30974b
	v_or_b32_e32 v16, 0x400000, v72                            // 000000006ca0: 382090ff 00400000
	global_store_d16_hi_b16 v[10:11], v14, off offset:32       // 000000006ca8: ee09407c 07000000 0000200a
	v_add3_u32 v10, v15, v74, 0x7fff                           // 000000006cb4: d655000a 03fe950f 00007fff
	v_or_b32_e32 v11, 0x400000, v74                            // 000000006cc0: 381694ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006cc8: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo                    // 000000006ccc: 02181b0c
	v_bfe_u32 v13, v73, 16, 1                                  // 000000006cd0: d610000d 02052149
	v_cmp_u_f32_e32 vcc_lo, v74, v74                           // 000000006cd8: 7c30954a
	v_bfe_u32 v14, v72, 16, 1                                  // 000000006cdc: d610000e 02052148
	v_or_b32_e32 v15, 0x400000, v73                            // 000000006ce4: 381e92ff 00400000
	v_or_b32_e32 v17, 0x400000, v36                            // 000000006cec: 382248ff 00400000
	v_add3_u32 v13, v13, v73, 0x7fff                           // 000000006cf4: d655000d 03fe930d 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006d00: bf88ff9d
	v_cndmask_b32_e32 v10, v10, v11, vcc_lo                    // 000000006d04: 0214170a
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 000000006d08: 7c309349
	v_bfe_u32 v11, v36, 16, 1                                  // 000000006d0c: d610000b 02052124
	v_add3_u32 v14, v14, v72, 0x7fff                           // 000000006d14: d655000e 03fe910e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006d20: bf88ff9d
	v_cndmask_b32_e32 v13, v13, v15, vcc_lo                    // 000000006d24: 021a1f0d
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 000000006d28: 7c309148
	v_add3_u32 v11, v11, v36, 0x7fff                           // 000000006d2c: d655000b 03fe490b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006d38: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo                    // 000000006d3c: 021c210e
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000006d40: 7c304924
	s_wait_alu depctr_va_vcc(0)                                // 000000006d44: bf88ff9d
	v_cndmask_b32_e32 v11, v11, v17, vcc_lo                    // 000000006d48: 0216230b
	s_clause 0x4                                               // 000000006d4c: bf850004
	global_store_d16_hi_b16 v[8:9], v12, off offset:32         // 000000006d50: ee09407c 06000000 00002008
	global_store_d16_hi_b16 v[6:7], v10, off offset:32         // 000000006d5c: ee09407c 05000000 00002006
	global_store_d16_hi_b16 v[4:5], v13, off offset:32         // 000000006d68: ee09407c 06800000 00002004
	global_store_d16_hi_b16 v[0:1], v14, off offset:32         // 000000006d74: ee09407c 07000000 00002000
	global_store_d16_hi_b16 v[2:3], v11, off offset:32         // 000000006d80: ee09407c 05800000 00002002
	s_nop 0                                                    // 000000006d8c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000006d90: bfb60003
	s_endpgm                                                   // 000000006d94: bfb00000
	s_code_end                                                 // 000000006d98: bf9f0000
	s_code_end                                                 // 000000006d9c: bf9f0000
	s_code_end                                                 // 000000006da0: bf9f0000
	s_code_end                                                 // 000000006da4: bf9f0000
	s_code_end                                                 // 000000006da8: bf9f0000
	s_code_end                                                 // 000000006dac: bf9f0000
	s_code_end                                                 // 000000006db0: bf9f0000
	s_code_end                                                 // 000000006db4: bf9f0000
	s_code_end                                                 // 000000006db8: bf9f0000
	s_code_end                                                 // 000000006dbc: bf9f0000
	s_code_end                                                 // 000000006dc0: bf9f0000
	s_code_end                                                 // 000000006dc4: bf9f0000
	s_code_end                                                 // 000000006dc8: bf9f0000
	s_code_end                                                 // 000000006dcc: bf9f0000
	s_code_end                                                 // 000000006dd0: bf9f0000
	s_code_end                                                 // 000000006dd4: bf9f0000
	s_code_end                                                 // 000000006dd8: bf9f0000
	s_code_end                                                 // 000000006ddc: bf9f0000
	s_code_end                                                 // 000000006de0: bf9f0000
	s_code_end                                                 // 000000006de4: bf9f0000
	s_code_end                                                 // 000000006de8: bf9f0000
	s_code_end                                                 // 000000006dec: bf9f0000
	s_code_end                                                 // 000000006df0: bf9f0000
	s_code_end                                                 // 000000006df4: bf9f0000
	s_code_end                                                 // 000000006df8: bf9f0000
	s_code_end                                                 // 000000006dfc: bf9f0000
	s_code_end                                                 // 000000006e00: bf9f0000
	s_code_end                                                 // 000000006e04: bf9f0000
	s_code_end                                                 // 000000006e08: bf9f0000
	s_code_end                                                 // 000000006e0c: bf9f0000
	s_code_end                                                 // 000000006e10: bf9f0000
	s_code_end                                                 // 000000006e14: bf9f0000
	s_code_end                                                 // 000000006e18: bf9f0000
	s_code_end                                                 // 000000006e1c: bf9f0000
	s_code_end                                                 // 000000006e20: bf9f0000
	s_code_end                                                 // 000000006e24: bf9f0000
	s_code_end                                                 // 000000006e28: bf9f0000
	s_code_end                                                 // 000000006e2c: bf9f0000
	s_code_end                                                 // 000000006e30: bf9f0000
	s_code_end                                                 // 000000006e34: bf9f0000
	s_code_end                                                 // 000000006e38: bf9f0000
	s_code_end                                                 // 000000006e3c: bf9f0000
	s_code_end                                                 // 000000006e40: bf9f0000
	s_code_end                                                 // 000000006e44: bf9f0000
	s_code_end                                                 // 000000006e48: bf9f0000
	s_code_end                                                 // 000000006e4c: bf9f0000
	s_code_end                                                 // 000000006e50: bf9f0000
	s_code_end                                                 // 000000006e54: bf9f0000
	s_code_end                                                 // 000000006e58: bf9f0000
	s_code_end                                                 // 000000006e5c: bf9f0000
	s_code_end                                                 // 000000006e60: bf9f0000
	s_code_end                                                 // 000000006e64: bf9f0000
	s_code_end                                                 // 000000006e68: bf9f0000
	s_code_end                                                 // 000000006e6c: bf9f0000
	s_code_end                                                 // 000000006e70: bf9f0000
	s_code_end                                                 // 000000006e74: bf9f0000
	s_code_end                                                 // 000000006e78: bf9f0000
	s_code_end                                                 // 000000006e7c: bf9f0000
	s_code_end                                                 // 000000006e80: bf9f0000
	s_code_end                                                 // 000000006e84: bf9f0000
	s_code_end                                                 // 000000006e88: bf9f0000
	s_code_end                                                 // 000000006e8c: bf9f0000
	s_code_end                                                 // 000000006e90: bf9f0000
	s_code_end                                                 // 000000006e94: bf9f0000
	s_code_end                                                 // 000000006e98: bf9f0000
	s_code_end                                                 // 000000006e9c: bf9f0000
	s_code_end                                                 // 000000006ea0: bf9f0000
	s_code_end                                                 // 000000006ea4: bf9f0000
	s_code_end                                                 // 000000006ea8: bf9f0000
	s_code_end                                                 // 000000006eac: bf9f0000
	s_code_end                                                 // 000000006eb0: bf9f0000
	s_code_end                                                 // 000000006eb4: bf9f0000
	s_code_end                                                 // 000000006eb8: bf9f0000
	s_code_end                                                 // 000000006ebc: bf9f0000
	s_code_end                                                 // 000000006ec0: bf9f0000
	s_code_end                                                 // 000000006ec4: bf9f0000
	s_code_end                                                 // 000000006ec8: bf9f0000
	s_code_end                                                 // 000000006ecc: bf9f0000
	s_code_end                                                 // 000000006ed0: bf9f0000
	s_code_end                                                 // 000000006ed4: bf9f0000
	s_code_end                                                 // 000000006ed8: bf9f0000
	s_code_end                                                 // 000000006edc: bf9f0000
	s_code_end                                                 // 000000006ee0: bf9f0000
	s_code_end                                                 // 000000006ee4: bf9f0000
	s_code_end                                                 // 000000006ee8: bf9f0000
	s_code_end                                                 // 000000006eec: bf9f0000
	s_code_end                                                 // 000000006ef0: bf9f0000
	s_code_end                                                 // 000000006ef4: bf9f0000
	s_code_end                                                 // 000000006ef8: bf9f0000
	s_code_end                                                 // 000000006efc: bf9f0000
	s_code_end                                                 // 000000006f00: bf9f0000
	s_code_end                                                 // 000000006f04: bf9f0000
	s_code_end                                                 // 000000006f08: bf9f0000
	s_code_end                                                 // 000000006f0c: bf9f0000
	s_code_end                                                 // 000000006f10: bf9f0000
	s_code_end                                                 // 000000006f14: bf9f0000
	s_code_end                                                 // 000000006f18: bf9f0000
	s_code_end                                                 // 000000006f1c: bf9f0000
	s_code_end                                                 // 000000006f20: bf9f0000
	s_code_end                                                 // 000000006f24: bf9f0000
	s_code_end                                                 // 000000006f28: bf9f0000
	s_code_end                                                 // 000000006f2c: bf9f0000
	s_code_end                                                 // 000000006f30: bf9f0000
	s_code_end                                                 // 000000006f34: bf9f0000
	s_code_end                                                 // 000000006f38: bf9f0000
	s_code_end                                                 // 000000006f3c: bf9f0000
	s_code_end                                                 // 000000006f40: bf9f0000
	s_code_end                                                 // 000000006f44: bf9f0000
	s_code_end                                                 // 000000006f48: bf9f0000
	s_code_end                                                 // 000000006f4c: bf9f0000
	s_code_end                                                 // 000000006f50: bf9f0000
	s_code_end                                                 // 000000006f54: bf9f0000
	s_code_end                                                 // 000000006f58: bf9f0000
	s_code_end                                                 // 000000006f5c: bf9f0000
	s_code_end                                                 // 000000006f60: bf9f0000
	s_code_end                                                 // 000000006f64: bf9f0000
	s_code_end                                                 // 000000006f68: bf9f0000
	s_code_end                                                 // 000000006f6c: bf9f0000
	s_code_end                                                 // 000000006f70: bf9f0000
	s_code_end                                                 // 000000006f74: bf9f0000
	s_code_end                                                 // 000000006f78: bf9f0000
	s_code_end                                                 // 000000006f7c: bf9f0000
